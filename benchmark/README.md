# pamiec-bench

Empirical evaluation harness for pamiec. Measures whether Claude answers questions
about prior conversation context more accurately when the pamiec memory graph is
available, vs. baseline (Claude alone with no tools).

This is a v0.1 vertical slice — one synthetic narrative, ten questions across four
categories, two arms, automated scoring. Designed to be extended.

## Setup

```bash
# Install with the benchmark optional group
cd ~/pamiec && uv sync --extra benchmark

# Anthropic API key (separate from Claude Code's subscription auth)
export ANTHROPIC_API_KEY=...

# Use an isolated test DB so the benchmark doesn't touch your real graph
export PAMIEC_DB=/tmp/pamiec-bench.db
```

## Running

```bash
cd ~/pamiec/benchmark

# 1. Populate the test DB from a synthetic narrative
python populate.py --narrative b2b_v1 --reset

# 2. Run the benchmark — both arms, all questions
python runner.py

# 3. Score the run
python score.py
```

## The four arms (2×2 design)

To isolate pamiec's contribution from the contribution of prompt-engineered
calibration, the benchmark runs four arms:

|                    | no recall          | with recall (pamiec)   |
| ------------------ | ------------------ | ---------------------- |
| **naive prompt**   | `naive_baseline`   | `naive_with_pamiec`    |
| **calibrated**     | `baseline`         | `with_pamiec`          |

- **naive prompt**: "You are a helpful assistant. Answer the user's question concisely."
- **calibrated prompt**: explicitly tells the model to say "no information" if not supported, and not to invent specifics.

## What you should see (Haiku 4.5, 30 questions × 4 arms)

```
Category         | Arm                  |    Acc  NoInfo  Halluc    InTok   OutTok  Lat(ms)
------------------------------------------------------------------------------------------------
single_hop       | naive_baseline       |    0%    91%      0%        34      104     1573
single_hop       | naive_with_pamiec    |  100%     0%      0%      1970      134     7134
single_hop       | baseline             |    0%   100%      0%        94       28      968
single_hop       | with_pamiec          |  100%     0%      0%      2080       98     2230
multi_hop        | naive_baseline       |    0%   100%      0%        44      133     2050
multi_hop        | naive_with_pamiec    |  100%     0%      0%      2191      202     3318
multi_hop        | baseline             |    0%   100%      0%       104       32     1073
multi_hop        | with_pamiec          |   86%    14%      0%      2209      109     2317
temporal         | naive_baseline       |    0%   100%      0%        39      214     2711
temporal         | naive_with_pamiec    |   40%    60%      0%      2081      310     4587
temporal         | baseline             |    0%   100%      0%        99       38     1152
temporal         | with_pamiec          |  100%     0%      0%      2167      149     2589
negative_probe   | naive_baseline       |  100%   100%      0%        36      110     3090
negative_probe   | naive_with_pamiec    |   71%    71%      0%      2021      200     3636
negative_probe   | baseline             |  100%   100%      0%        96       30      933
negative_probe   | with_pamiec          |  100%   100%      0%      2135       99     2404

OVERALL
naive_baseline       | accuracy  7/30 =  23%  | hallucinations 0/30 | avg   37+130 tok  | avg 2228 ms
naive_with_pamiec    | accuracy 25/30 =  83%  | hallucinations 0/30 | avg 2052+195 tok  | avg 5003 ms
baseline             | accuracy  7/30 =  23%  | hallucinations 0/30 | avg   97+31  tok  | avg 1015 ms
with_pamiec          | accuracy 29/30 =  97%  | hallucinations 0/30 | avg 2137+109 tok  | avg 2350 ms
```

The 2×2 collapsed:

|                  | no recall | with recall | recall adds |
|------------------|-----------|-------------|-------------|
| **naive**        | 23%       | 83%         | +60 pp      |
| **calibrated**   | 23%       | 97%         | +74 pp      |
| **calib. adds**  | 0 pp      | +14 pp      |             |

### What this means

- **Pamiec is the dominant lever** — adds 60–74 pp absolute accuracy, depending on prompt.
- **The calibration prompt has a real effect, but only when paired with pamiec.** With recall, calibration adds ~14 pp on top. Without recall, calibration adds nothing — you can't be calibrated about facts you don't have.
- The calibration win shows up in two specific places:
  - **temporal naive_with_pamiec drops to 40%.** Even with recall returning the answer, the naive prompt makes the model say "I don't have specific information" 60% of the time on why-questions. Calibrated prompt holds at 100%.
  - **negative_probe naive_with_pamiec drops to 71%.** With recall context and a permissive prompt, the model over-confirms 29% of negative propositions. Calibrated holds at 100%.
- **No hallucinations across any of the 120 calls.** Modern Haiku is well-calibrated by default; even naive prompting + pamiec-injected context didn't produce confident wrong answers.
- **Cost overhead:** ~2k extra input tokens per question with recall. Latency adds 1–4 s depending on category.
- **Single failure** on the strongest arm: q22 (`with_pamiec` multi_hop) — model started "No information" then gave the right answer in its explanation. The scorer correctly treats this confused-but-right answer as incorrect.

The honest headline: **on 30 questions over 1 narrative against Haiku 4.5, pamiec lifts accuracy from 23% to 97% with zero observed hallucinations.** The calibration prompt adds an additional ~14 pp on top of pamiec.

## Sonnet 4.6 validation

Same 30 questions, same narrative, same code, just `BENCH_MODEL=claude-sonnet-4-6`:

|                  | Haiku 4.5 | Sonnet 4.6 |
|------------------|-----------|------------|
| naive_baseline   | 23%       | 23%        |
| naive_with_pamiec| 83%       | 90%        |
| baseline         | 23%       | 23%        |
| with_pamiec      | 97%       | **100%**   |
| Halluc / 120     | 0         | 1          |
| Latency / Q      | ~2.4 s    | ~4.5 s     |
| Calib. prompt lift | +14 pp  | +10 pp     |

Two things to note:

1. **Pamiec's gain holds for the stronger model** — with_pamiec hits perfect 30/30 on Sonnet, and the no-recall arms still score identically (23%). Pamiec's value isn't a Haiku-specific artifact.

2. **Sonnet 4.6 with naive prompting produced one confabulation that Haiku didn't.** Q08 asked "Did the team consider Snowflake?" (no — Snowflake is a negative-probe entity). Recall returned context about ClickHouse being chosen for telemetry. Sonnet's naive_with_pamiec arm answered "Yes, but it was not selected; the team chose ClickHouse over Snowflake..." with confident detail. The reasoning move was: ClickHouse "winning" implies alternatives were evaluated, so the question's named entity (Snowflake) must have been one of them. The calibration prompt eliminated this — Sonnet's calibrated arm got the same question right.

This is the kind of failure mode the 2×2 design was built to catch. Stronger reasoning makes the model better at substantive recall and worse at resisting leading questions when the prompt doesn't constrain it.

## Layout

```
benchmark/
├── narratives/
│   └── b2b_v1.py       # synthetic 3-session B2B SaaS narrative + ground-truth tree
├── questions/
│   └── b2b_v1.json     # 10 questions across single_hop, multi_hop, temporal, negative_probe
├── runners/
│   └── (all in runner.py for v0.1)
├── results/
│   └── *.jsonl         # raw per-question results from each run
├── populate.py         # narrative → pamiec DB (capture + consolidate)
├── runner.py           # invoke Claude with both arms, dump JSONL
└── score.py            # aggregate JSONL into accuracy / hallucination / cost / latency
```

## Configuration

- `BENCH_MODEL` env var sets the Claude model under test (default: `claude-haiku-4-5-20251001`)
- `PAMIEC_DB` controls which DB the populate / runner use — required, points at an isolated path
- `ANTHROPIC_API_KEY` required for the runner

## Question categories

- **single_hop** — direct entity attribute lookup ("Who leads ProjectQ?")
- **multi_hop** — answer requires combining two facts ("What database does Carol's project use?")
- **temporal** — when/why questions requiring decision context ("Why did we switch X to Y?")
- **negative_probe** — entity NOT in the graph; correct answer is "no information". Tests hallucination resistance.

Scoring rules (intentionally simple in v0.1):
- single_hop / multi_hop / temporal: `expected_keywords` (all required) AND `expected_keywords_any_of` (at least one group of synonyms must fully match)
- negative_probe: answer must contain a no-info phrase AND must not contain a confident-confirmation phrase

LLM-as-judge scoring is deferred to v0.2 — needed for free-form temporal answers where keyword matching is too brittle.

## Notes on the v0.1 result

- The system prompt explicitly tells the model "say no information if the answer isn't supported, do NOT guess." Without this nudge, the baseline would likely hallucinate; the comparison point in v0.1 is "with calibration prompt + recall vs with calibration prompt alone." A future arm would test recall vs. baseline-without-calibration to isolate pamiec's contribution from prompt engineering.
- The 100% with-pamiec accuracy is on 10 questions over 1 narrative; small enough that we shouldn't claim general superiority. The number you want to track is the **gap between arms** as the question count grows.
- Token cost: pamiec's recall injects ~2k input tokens per question. That's the dominant overhead; latency scales with that plus the recall round-trip.

## What's next

Per `~/ongoing-projects/master-plan/projects/pamiec-bench.md`:

- v0.2: 4 more narrative templates (sci-software, mobile, infra, ML platform), full 150-question run
- v0.3: LoCoMo Tier 2 adaptation
- v1.0: paper draft, reproducible repo, leaderboard
