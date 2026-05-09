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

## What you should see (Haiku 4.5)

```
Category         | Arm                  |    Acc  NoInfo  Halluc    InTok   OutTok  Lat(ms)
------------------------------------------------------------------------------------------------
single_hop       | naive_baseline       |    0%   100%      0%        30      114     1644
single_hop       | naive_with_pamiec    |  100%     0%      0%      1963      135     2337
single_hop       | baseline             |    0%   100%      0%        90       30      904
single_hop       | with_pamiec          |  100%     0%      0%      2079      106     2665
multi_hop        | naive_baseline       |    0%   100%      0%        42      126     3444
multi_hop        | naive_with_pamiec    |  100%     0%      0%      2053      182     4134
multi_hop        | baseline             |    0%   100%      0%       102       28      868
multi_hop        | with_pamiec          |  100%     0%      0%      2171      110     2064
temporal         | naive_baseline       |    0%   100%      0%        38      212     3462
temporal         | naive_with_pamiec    |  100%     0%      0%      2018      277     4436
temporal         | baseline             |    0%   100%      0%        98       32     1090
temporal         | with_pamiec          |  100%     0%      0%      2127      162     2715
negative_probe   | naive_baseline       |  100%   100%      0%        35      116     1746
negative_probe   | naive_with_pamiec    |  100%   100%      0%      2025      193     2971
negative_probe   | baseline             |  100%   100%      0%        95       25      955
negative_probe   | with_pamiec          |  100%   100%      0%      2123      103     2961

OVERALL
naive_baseline       | accuracy 3/10 =  30%  | hallucinations 0/10  | avg    35+137 tok  | avg 2398 ms
naive_with_pamiec    | accuracy 10/10 = 100% | hallucinations 0/10  | avg  2010+190 tok  | avg 3306 ms
baseline             | accuracy 3/10 =  30%  | hallucinations 0/10  | avg    95+28  tok  | avg  950 ms
with_pamiec          | accuracy 10/10 = 100% | hallucinations 0/10  | avg  2120+117 tok  | avg 2644 ms
```

### What this means

- **Pamiec accounts for the entire accuracy gain.** Both with-recall arms score 100%; both no-recall arms score 30% (which equals the 3/10 negative probes that are correctly answered "no" without recall).
- **The calibration prompt makes no measurable difference for Haiku 4.5.** Naive and calibrated arms have identical scores. The model is well-calibrated by default — when it doesn't know, it says so. This is a finding about Haiku 4.5, not about pamiec; with older / weaker models the calibration prompt may matter more.
- **No hallucinations in any arm.** The 0% hallucination rate across all four conditions is itself a notable result: even with naive prompting and pamiec-injected context, Haiku 4.5 doesn't confabulate confident wrong answers.
- **Cost overhead:** pamiec's recall injects ~2k input tokens per question. Output tokens stay similar; latency adds ~1–2 s per question.

The honest headline: **on this benchmark, pamiec doubles+ Claude's recall accuracy with no observable hallucination cost.** The 70 pp absolute improvement should not be claimed as a general result until the question count scales beyond 10 and the model count beyond Haiku 4.5.

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
