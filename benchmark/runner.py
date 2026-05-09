"""Runner: query Claude with and without pamiec recall, capture answers + token cost.

Two arms per question:
  - baseline: Claude alone, no tools.
  - with_pamiec: Claude with a `recall_context` tool that hits pamiec's recall
    function on the test DB.

The "test DB" is whichever DB pamiec is configured to use via the PAMIEC_DB
env var. Run capture+consolidate over the narrative first to populate it.

Requires:
  - ANTHROPIC_API_KEY in env
  - anthropic SDK installed (pip install anthropic)
  - PAMIEC_DB pointing at the populated test DB
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Late import so we can give a friendly error if missing
try:
    import anthropic
except ImportError:  # pragma: no cover
    print(
        "ERROR: 'anthropic' SDK not installed. Run:\n"
        "    cd ~/pamiec && uv pip install anthropic\n"
        "Or add it to pyproject.toml dependencies.",
        file=sys.stderr,
    )
    sys.exit(1)

MODEL = os.environ.get("BENCH_MODEL", "claude-haiku-4-5-20251001")

# Two system prompts let us isolate the contribution of pamiec from the contribution
# of prompt-engineered calibration. The 2x2 over (prompt × recall) is below.
SYSTEM_PROMPT_CALIBRATED = """You are answering questions about a software engineering team's recent work.

Rules:
- Answer concisely, in one or two sentences.
- If the answer is not supported by available context, say "no information" or "not discussed". Do NOT guess.
- Do not invent specific names, numbers, or facts that aren't grounded.
"""

SYSTEM_PROMPT_NAIVE = """You are a helpful assistant. Answer the user's question concisely."""

# Arm specification: (prompt_to_use, expose_recall_tool)
ARMS = {
    "baseline":          (SYSTEM_PROMPT_CALIBRATED, False),  # calibrated, no recall
    "with_pamiec":       (SYSTEM_PROMPT_CALIBRATED, True),   # calibrated, with recall
    "naive_baseline":    (SYSTEM_PROMPT_NAIVE,      False),  # naive, no recall — likely hallucinates
    "naive_with_pamiec": (SYSTEM_PROMPT_NAIVE,      True),   # naive, with recall — does pamiec save it?
}


# ── Tools ────────────────────────────────────────────────────────────────────

RECALL_TOOL = {
    "name": "recall_context",
    "description": (
        "Query the long-term memory graph for context relevant to the user's question. "
        "Returns relevant entity facts and past episode summaries.\n\n"
        "ALWAYS call this BEFORE answering any question that references a specific named "
        "person, project, company, decision, grant, tool, or organization — even if the "
        "name appears unfamiliar to you. The graph may contain it. Do not refuse with "
        "'no information' until recall_context has been tried at least once.\n\n"
        "When you do refuse, refuse only AFTER an empty recall result, and explicitly note "
        "that you searched the graph."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language description of what you're looking for. Include the named entity from the question verbatim."}
        },
        "required": ["query"],
    },
}


def _do_recall(query: str) -> str:
    """Invoke pamiec's recall on the configured DB."""
    from pamiec.retrieval import format_context, recall
    return format_context(recall(query))


# ── Runner ───────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    question_id: str
    category: str
    arm: str
    model: str
    answer: str
    tool_calls: list  # [{"query": "...", "result_chars": N}, ...]
    input_tokens: int
    output_tokens: int
    latency_ms: int
    error: str | None = None


def run_one(client: anthropic.Anthropic, question: dict, arm: str) -> RunResult:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm '{arm}'. Known: {list(ARMS.keys())}")
    system_prompt, has_recall = ARMS[arm]

    qid = question["id"]
    category = question["category"]
    user_msg = question["question"]

    tools = [RECALL_TOOL] if has_recall else []
    messages = [{"role": "user", "content": user_msg}]
    tool_calls = []
    input_tokens = 0
    output_tokens = 0

    t0 = time.time()
    try:
        # Loop until model returns end_turn (handle multi-step tool use)
        while True:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=600,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )
            input_tokens += resp.usage.input_tokens
            output_tokens += resp.usage.output_tokens

            if resp.stop_reason == "tool_use":
                # Append assistant turn
                messages.append({"role": "assistant", "content": resp.content})
                # Execute tools, build tool_result blocks
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use" and block.name == "recall_context":
                        q = block.input.get("query", "")
                        result = _do_recall(q)
                        tool_calls.append({"query": q, "result_chars": len(result)})
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})
                continue

            # end_turn — collect text answer
            answer_parts = [b.text for b in resp.content if b.type == "text"]
            answer = "\n".join(answer_parts).strip()
            break

    except Exception as e:
        return RunResult(
            question_id=qid, category=category, arm=arm, model=MODEL,
            answer="", tool_calls=tool_calls,
            input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=int((time.time() - t0) * 1000),
            error=f"{type(e).__name__}: {e}",
        )

    return RunResult(
        question_id=qid, category=category, arm=arm, model=MODEL,
        answer=answer, tool_calls=tool_calls,
        input_tokens=input_tokens, output_tokens=output_tokens,
        latency_ms=int((time.time() - t0) * 1000),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="questions/b2b_v1.json")
    parser.add_argument("--out", default="results/b2b_v1.jsonl")
    parser.add_argument(
        "--arm",
        action="append",
        choices=list(ARMS.keys()) + ["all"],
        help="arm to run (repeat flag for multiple); 'all' runs every arm. Default: all.",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in env.", file=sys.stderr)
        sys.exit(1)

    bench_dir = Path(__file__).parent
    qs = json.loads((bench_dir / args.questions).read_text())["questions"]
    out_path = bench_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    if not args.arm or "all" in args.arm:
        arms = list(ARMS.keys())
    else:
        arms = args.arm

    with out_path.open("w") as f:
        for q in qs:
            for arm in arms:
                r = run_one(client, q, arm)
                f.write(json.dumps(asdict(r)) + "\n")
                f.flush()
                marker = "ERR" if r.error else "OK "
                tools = f" tools={len(r.tool_calls)}" if r.tool_calls else ""
                print(
                    f"  {marker} {q['id']} [{q['category']:14s}] {arm:18s} "
                    f"{r.input_tokens:>5}+{r.output_tokens:<4} tok  "
                    f"{r.latency_ms:>5}ms{tools}  {(r.answer or r.error)[:75]}"
                )

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
