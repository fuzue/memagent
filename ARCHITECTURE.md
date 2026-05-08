# memagent — Architecture

memagent is a hierarchical graph memory system for Claude Code sessions. It is
inspired by [GAM (Graph Attention Memory, arxiv 2604.12285)](https://arxiv.org/abs/2604.12285),
adapted for cross-session entity-level memory rather than within-session episodic
memory.

The core design principle is **strict write isolation between three tiers**:
fast real-time capture must never contaminate the stable long-term entity graph.

---

## Three-tier data model

```
                ┌──────────────────────────────────────┐
                │  Tier 3 — Entity graph (TAN)         │
                │  topic_nodes + topic_edges           │
                │  Long-term, deduplicated, compacted  │
                │  Updated only on consolidation       │
                └────────────────▲─────────────────────┘
                                 │ promote
                                 │ entity_episode_links
                ┌────────────────┴─────────────────────┐
                │  Tier 2 — Archive (S_arch)           │
                │  episodes + episode_turns            │
                │  Frozen snapshots of past sessions   │
                │  Each has summary + transcript       │
                └────────────────▲─────────────────────┘
                                 │ promote at semantic boundaries
                                 │
                ┌────────────────┴─────────────────────┐
                │  Tier 1 — Live EPG                   │
                │  epg_turns                           │
                │  Real-time, near-zero cost           │
                │  Append-only, drained by consolidate │
                └────────────────▲─────────────────────┘
                                 │ capture
                                 │
                       Claude Code session JSONL
                       (~/.claude/projects/.../*.jsonl)
```

### Tier 1 — Live EPG (`epg_turns`)

The real-time buffer. Captures each user / assistant turn as it appears in the
Claude Code session JSONL.

| Field | Meaning |
|-------|---------|
| `id` | UUID |
| `session_file` | absolute path to source JSONL |
| `role` | `user` or `assistant` |
| `text` | turn text (capped to 2 KB) |
| `timestamp` | unix epoch from JSONL ISO timestamp |
| `iso_ts` | original ISO string (for checkpointing) |
| `embedding` | 384-dim float32 vector (BAAI/bge-small-en-v1.5) |
| `captured_at` | unix epoch when captured |

**Lifecycle:** populated by `memagent capture` (cron every 2 min). Drained by
`memagent consolidate-session` after promotion. No LLM call ever touches Tier 1.

**Why a separate table:** the live buffer has a different lifecycle (deleted
after promotion) than archived turns. Keeping them separate avoids ambiguity.

### Tier 2 — Archive (`episodes`, `episode_turns`)

Frozen snapshots of consolidated sessions.

`episodes`:

| Field | Meaning |
|-------|---------|
| `id` | UUID |
| `session_file` | source JSONL path |
| `started_at`, `ended_at` | unix epochs |
| `transcript` | concatenated `User: …\n\nClaude: …` text |
| `summary` | 1-2 sentence Haiku-generated description |
| `embedding` | embedding of the summary (for direct semantic search) |

`episode_turns`:
Permanent storage of individual turns once an episode is promoted. Mirror of
the EPG turn record but with `episode_id` foreign key.

**Lifecycle:** created by consolidation, never modified after creation.

### Tier 3 — Entity graph (`topic_nodes`, `topic_edges`)

The stable long-term memory.

`topic_nodes`:

| Field | Meaning |
|-------|---------|
| `id` | UUID |
| `csum` | one-line summary like `"Alice: founder of Acme; based in Berlin"` |
| `craw` | bullet-list facts under a `# Name` header |
| `entity_type` | `person`, `project`, `company`, `tool`, `grant`, `paper`, … (open vocabulary) |
| `embedding` | embedding of the `csum` |
| `created_at`, `updated_at` | unix epochs |

`topic_edges`: typed directed edges between nodes. `edge_type` is open
vocabulary (`FOUNDED`, `OWNS`, `WORKS_ON`, `BLOCKS`, `FUNDS`, …).

**Cross-layer linkage:** `entity_episode_links` connects every entity touched by
an episode back to that episode. This is GAM's `E_cross` — enables drill-down
from a stable entity to the conversations that produced it.

---

## Data flow

### Capture (cron `*/2 * * * *`)

```
read_turns_since(session_file, captured_checkpoint)
  → read_turns_since walks the JSONL, filters to user/assistant turns
embed_batch([t.text for t in turns])
add_epg_turn(...) for each turn
update captured checkpoint to last turn's iso_ts
```

Sub-second runtime, no LLM, single fastembed call. Idempotent: rerunning before
the next turn is a no-op.

### Consolidation (cron `*/30 * * * *`)

```
get_epg_turns()                       # drain everything in the live buffer
group by session_file
for each session:
    segments = split_at_boundaries(turns)   # semantic event-triggered partitioning
    for each segment:
        consolidate_turns(segment)          # one call per segment
delete_epg_turns(promoted_ids)
```

Each segment becomes one episode. The boundary detector ensures topically-
distinct portions of one cron window become separate episodes rather than one
fat noisy summary.

### Boundary detection (`boundaries.split_at_boundaries`)

```
embeddings = batch_embed([t.text for t in turns])
for each candidate position i in [window, len(turns) - window]:
    prev_centroid = mean(embeddings[i-window : i])
    next_centroid = mean(embeddings[i : i+window])
    sim = cosine(prev_centroid, next_centroid)
    record (i, sim)

for each (i, sim):
    if sim < THRESHOLD                       # 0.72
       and sim is a strict local minimum
       and (i - last_boundary) >= MIN_SEGMENT:  # 4
        mark i as a boundary
```

Threshold 0.72 is calibrated against bge-small-en-v1.5: same-topic ~0.78,
related-but-different ~0.51, unrelated ~0.38. The local-minimum constraint
prevents false positives in long monotone discussions.

### Segment consolidation (`consolidation.consolidate_turns`)

For one segment (1 episode):

```
transcript = turns_to_transcript(turns)
extracted = _extract_chunked(transcript)         # Haiku, chunked at 8k chars
filter entities/edges by confidence >= 0.7

create episodes row with summary + transcript embedding
for each turn: insert episode_turns row

for each entity:
    matched = find_matching_node(name, embed(csum))    # name first, then >0.82 cosine
    if matched:
        new_facts = dedupe_facts(facts, existing_craw)  # exact + cosine >= 0.90
        merged_craw = existing + new_facts
        if len > COMPACT_THRESHOLD: compact via Haiku
        update_topic_node(matched.id, csum, merged_craw, embed)
    else:
        add_topic_node(...)
    add_entity_episode_link(node.id, episode.id)

for each edge in extracted:
    resolve from/to to node ids
    add_topic_edge(src, tgt, edge_type)
```

### Haiku extraction (`consolidation._extract`)

Single prompt, JSON output, `confidence` field on every entity and edge.
Uses `claude --model claude-haiku-4-5-20251001 -p`.

The prompt explicitly enumerates **what NOT to extract** — concepts being
discussed, design critiques, problems being solved, generic technical topics —
to suppress hallucinations like turning "memory gap" or "design flaw" into
entity nodes.

Confidence threshold (`CONFIDENCE_THRESHOLD = 0.7`) drops anything Haiku
flagged as uncertain. This is GAM's two-step coarse-to-fine, condensed into
the same call (Haiku scores its own outputs).

### Compaction (`consolidation._compact_craw`)

When a node's `craw` exceeds `COMPACT_THRESHOLD_LINES = 25`, a single Haiku
call merges redundant facts and strips session-narrative noise (e.g.
"recently fixed", "design flaw identified"), reordering the remaining facts
logically (identity → purpose → components → status → risks).

Falls back to the original craw on any failure.

---

## Retrieval

`retrieval.recall(query, top_n=8)` returns a ranked mix of results from all
three tiers.

```
query_vec = embed(query)

# Tier 3 — entity nodes
score topic nodes by 0.8 * cosine + 0.2 * recency (14-day half-life)
take top 4 above 0.15 → return craw
expand one hop via topic_edges, keep neighbors with cosine > 0.4 → return csum

# Tier 2 — archived episodes
for each episode with embedding:
    if cosine(query, episode.embedding) >= 0.4:
        score = (0.7*sim + 0.3*recency) * 0.85    # discount vs entities
        return "[episode YYYY-MM-DD] {summary}"

# Tier 1 — live EPG turns
for each EPG turn with embedding:
    if cosine(query, turn.embedding) >= 0.5:
        recency uses 0.04-day (~1h) half-life
        score = (0.6*sim + 0.4*recency) * 0.75
        return "[live HH:MM] {role}: {snippet}"

deduplicate by node_id (highest score wins)
return top_n results
```

`format_context(results)` separates results into three sections:
**Entities** (stable), **Past episodes** (archived), **Live (current session)**.

The MCP tool `recall_context(query)` exposes this to Claude Code sessions —
the model calls it whenever the conversation touches an entity, project,
person, decision, or anything with prior history.

---

## Embedding model

[`fastembed`](https://github.com/qdrant/fastembed) running `BAAI/bge-small-en-v1.5`.

- 384 dimensions, float32
- Local inference, no API calls
- Single batch call for many texts is the only fast path; loading the model
  per-call is what caused the original CPU spike

The model is loaded lazily on first call and cached as a module-level singleton
(`embedder._model`).

---

## Visualization

`graph_export.export_html(out_path)` writes a self-contained HTML file with
inlined D3.js (cached at `~/.memagent/d3.v7.min.js`).

Colors are deterministic:

```python
def _color_for(label, lightness=55, saturation=65):
    h = int(hashlib.md5(label.lower().encode()).hexdigest()[:6], 16) % 360
    return f"hsl({h}, {saturation}%, {lightness}%)"
```

Any new entity type or edge type the extractor invents gets a stable color
without code changes. The legend is built from types actually present in the
data, not from a hardcoded list.

Episodes render as small grey squares with dashed cross-link edges to the
entities they mention. Entities render as colored circles. Click any node to
see its detail and outgoing/incoming relationships.

Served over a one-shot Python `http.server` on a free port (avoids Firefox's
file:// restrictions on inline scripts).

---

## Module layout

```
src/memagent/
├── db.py             SQLite schema + connection helper
├── models.py         Dataclasses (Event, TopicNode, Session)
├── store.py          Storage layer for all three tiers
├── embedder.py       fastembed wrapper, cosine, batched embedding
├── session_reader.py Read Claude Code JSONL into structured Turn objects
├── boundaries.py     Semantic boundary detection (Tier 1 → 2 split)
├── consolidation.py  Tier 1 → 2 → 3 promotion: extract, dedupe, compact
├── retrieval.py      Multi-tier search + scoring + dedup
├── graph_export.py   Self-contained D3 HTML export
├── mcp_server.py     FastMCP wrapper exposing recall_context + remember
└── cli.py            Command surface: capture, consolidate-session, recall, …
```

---

## Configuration constants

| Constant | Value | Where | Purpose |
|----------|-------|-------|---------|
| `CONFIDENCE_THRESHOLD` | 0.7 | `consolidation.py` | Drop low-confidence entities/edges from Haiku |
| `MERGE_THRESHOLD` | 0.82 | `consolidation.py` | Cosine threshold for merging two entities into one |
| `RELATED_THRESHOLD` | 0.60 | `consolidation.py` | Cosine threshold for one-hop expansion in retrieval |
| `FACT_DEDUPE_SIM` | 0.90 | `consolidation.py` | Cosine threshold for dropping a fact as a duplicate |
| `COMPACT_THRESHOLD_LINES` | 25 | `consolidation.py` | Line count that triggers craw compaction |
| `DEFAULT_THRESHOLD` (boundary) | 0.72 | `boundaries.py` | Below this → topic boundary candidate |
| `DEFAULT_WINDOW` | 3 | `boundaries.py` | Centroid window size on each side of split point |
| `MIN_SEGMENT_TURNS` | 4 | `boundaries.py` | Minimum turns between consecutive boundaries |

---

## Comparison with GAM

| Aspect | GAM | memagent |
|--------|-----|----------|
| Tier 1 EPG | Real-time per-turn append, temporal edges between turns | Real-time per-turn append in `epg_turns`, no temporal edges |
| Tier 2 Archive | Frozen `G_event` snapshots, cross-linked from topic nodes | `episodes` + `episode_turns`, with `entity_episode_links` |
| Tier 3 TAN | LLM-scored typed edges, dual-granularity nodes (`c_sum`, `c_raw`) | Same: `csum`/`craw`, typed edges with confidence gating |
| Boundary detection | LLM verdict on the EPG buffer (semantic event trigger) | Embedding-only sliding-window centroid, local-minima detection |
| Retrieval | TAN anchor → cross-encoder rerank with role/time/conf factors | Multi-tier (TAN + Archive + Live), recency-weighted, no rerank |
| Optimisation target | Long-conversation episodic memory | Cross-session entity-level memory |

memagent simplifies GAM's mechanics in two places (boundary detection without
LLM, retrieval without cross-encoder) and adds one capability (a third
queryable layer in Tier 1 — the live buffer is searchable directly, not just
via Tier 2/3).

The biggest deliberate divergence is the unit of memory. GAM optimises for
"what happened in this long conversation". memagent optimises for "what do I
know about Alice / Acme / ProjectX across all sessions". The architectural
similarity is real but the use case is different.

---

## File locations

```
~/.memagent/
├── memory.db          SQLite — all three tiers
├── checkpoint.json    {"captured": {session_path: last_iso_ts}}
├── d3.v7.min.js       Cached D3 for offline graph rendering
├── graph.html         Last rendered visualization
└── cron.log           Output from capture + consolidate-session crons
```

---

## Operating crons

```
*/2  * * * * memagent capture              # Tier 1 ingestion
*/30 * * * * memagent consolidate-session  # Tier 1 → 2 → 3 promotion
```

Capture is cheap (sub-second, no LLM). Consolidation cost depends on how many
turns accumulated in the EPG since the last run; expect one Haiku call per
~8 KB of transcript chunk per detected segment, plus zero or more compaction
calls when entity craws cross the 25-line threshold.
