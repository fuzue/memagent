"""Score a benchmark run against expected answers.

Usage:
  python score.py --questions questions/b2b_v1.json --results results/b2b_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Phrases that indicate the model is NOT confirming the proposition — either
# admitting ignorance or grounding a denial. Both are valid answers to a
# negative-probe question.
NO_INFO_PHRASES = [
    # admissions of ignorance
    "no information", "not discussed", "no evidence", "not mentioned",
    "i don't have", "no record", "no context", "isn't mentioned",
    "no details", "not in", "couldn't find", "no confirmation",
    # grounded denials
    "no mention", "no, ", "no.", "is not ", "isn't ", "wasn't ",
    "didn't ", "did not ", "does not ", "doesn't ",
]

# Strong hallucination signals — confident confirmations of made-up facts.
HALLUCINATION_PHRASES = [
    "yes,", "yes ", "yes.",
    "indeed", "was considered", "has been integrated",
]


def _has_any(text: str, phrases: list[str]) -> bool:
    text_l = text.lower()
    return any(p in text_l for p in phrases)


def _has_all_keywords(text: str, keywords: list[str]) -> bool:
    text_l = text.lower()
    return all(kw.lower() in text_l for kw in keywords)


def _matches_any_group(text: str, groups: list[list[str]]) -> bool:
    """True if ANY group's keywords are ALL present (OR over groups, AND within)."""
    return any(_has_all_keywords(text, g) for g in groups)


def score_question(question: dict, answer: str) -> dict:
    """Return a per-question score dict."""
    cat = question["category"]
    answer = answer or ""

    if cat == "negative_probe":
        admits = _has_any(answer, NO_INFO_PHRASES)
        hallucinates = _has_any(answer, HALLUCINATION_PHRASES) and not admits
        correct = admits and not hallucinates
        return {
            "correct": correct,
            "admits_no_info": admits,
            "hallucinates": hallucinates,
        }

    # single_hop / multi_hop / temporal:
    #   - expected_keywords (if present): ALL must appear
    #   - expected_keywords_any_of (if present): at least ONE group must be fully present
    # If both are present, both must be satisfied.
    # CRITICAL: if the answer also admits no-info, it can't be 'correct' regardless of
    # keyword presence — the model didn't actually answer the question. Common pattern
    # is "I don't have information about Postgres→ClickHouse but here are general
    # reasons..." which would otherwise score positive by parroting question terms.
    kws = question.get("expected_keywords", [])
    any_of = question.get("expected_keywords_any_of")

    keyword_pass = _has_all_keywords(answer, kws) if kws else True
    any_of_pass = _matches_any_group(answer, any_of) if any_of else True
    admits = _has_any(answer, NO_INFO_PHRASES)
    correct = keyword_pass and any_of_pass and not admits

    return {
        "correct": correct,
        "admits_no_info": admits,
        "hallucinates": False,
    }


def aggregate(results_path: Path, questions: list[dict]) -> dict:
    qs_by_id = {q["id"]: q for q in questions}
    by_arm_cat: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "correct": 0, "no_info": 0, "halluc": 0,
                 "in_tok": 0, "out_tok": 0, "latency_ms": 0}
    )

    rows = []
    with results_path.open() as f:
        for line in f:
            r = json.loads(line)
            q = qs_by_id[r["question_id"]]
            s = score_question(q, r["answer"])
            key = (r["arm"], q["category"])
            agg = by_arm_cat[key]
            agg["n"] += 1
            agg["correct"] += int(s["correct"])
            agg["no_info"] += int(s["admits_no_info"])
            agg["halluc"] += int(s["hallucinates"])
            agg["in_tok"] += r["input_tokens"]
            agg["out_tok"] += r["output_tokens"]
            agg["latency_ms"] += r["latency_ms"]
            rows.append({**r, "score": s, "category": q["category"]})

    return {"by_arm_cat": by_arm_cat, "rows": rows}


def print_table(by_arm_cat: dict, arms: list[str], cats: list[str]) -> None:
    print()
    print(f"{'Category':<16} | {'Arm':<20} | {'Acc':>6}  {'NoInfo':>6}  {'Halluc':>6}  {'InTok':>7}  {'OutTok':>7}  {'Lat(ms)':>7}")
    print("-" * 96)
    for cat in cats:
        for arm in arms:
            agg = by_arm_cat.get((arm, cat))
            if not agg or agg["n"] == 0:
                continue
            n = agg["n"]
            acc = agg["correct"] / n
            ni  = agg["no_info"] / n
            ha  = agg["halluc"] / n
            it  = agg["in_tok"] / n
            ot  = agg["out_tok"] / n
            lat = agg["latency_ms"] / n
            print(f"{cat:<16} | {arm:<20} | {acc:>5.0%}  {ni:>5.0%}   {ha:>5.0%}   {it:>7.0f}  {ot:>7.0f}  {lat:>7.0f}")
        print()


def print_summary(by_arm_cat: dict, arms: list[str]) -> None:
    print(f"\n{'OVERALL':<16}")
    print("-" * 96)
    for arm in arms:
        n = sum(a["n"]      for k, a in by_arm_cat.items() if k[0] == arm)
        c = sum(a["correct"]for k, a in by_arm_cat.items() if k[0] == arm)
        h = sum(a["halluc"] for k, a in by_arm_cat.items() if k[0] == arm)
        it= sum(a["in_tok"] for k, a in by_arm_cat.items() if k[0] == arm)
        ot= sum(a["out_tok"]for k, a in by_arm_cat.items() if k[0] == arm)
        lt= sum(a["latency_ms"] for k, a in by_arm_cat.items() if k[0] == arm)
        print(f"{arm:<20} | accuracy {c}/{n} = {c/n:>4.0%}  | hallucinations {h}/{n}  "
              f"| avg {it/n:>5.0f}+{ot/n:<3.0f} tok  | avg {lt/n:>5.0f} ms")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="questions/b2b_v1.json")
    parser.add_argument("--results", default="results/b2b_v1.jsonl")
    args = parser.parse_args()

    bench_dir = Path(__file__).parent
    qs = json.loads((bench_dir / args.questions).read_text())["questions"]
    res_path = bench_dir / args.results
    out = aggregate(res_path, qs)

    # Order arms by the 2x2 design: naive first, calibrated second; within each, no-recall before recall.
    arm_order = ["naive_baseline", "naive_with_pamiec", "baseline", "with_pamiec"]
    arms_present = sorted(
        {k[0] for k in out["by_arm_cat"]},
        key=lambda a: arm_order.index(a) if a in arm_order else 999,
    )
    cats = ["single_hop", "multi_hop", "temporal", "negative_probe"]
    print_table(out["by_arm_cat"], arms_present, cats)
    print_summary(out["by_arm_cat"], arms_present)

    # Save aggregated JSON for downstream analysis
    summary = {
        "by_arm_cat": {f"{k[0]}|{k[1]}": v for k, v in out["by_arm_cat"].items()}
    }
    (bench_dir / "results" / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
