# memagent

Persistent, queryable memory for Claude Code. Builds a knowledge graph of the
people, projects, companies, and decisions you discuss across sessions, so
Claude doesn't forget who you are or what you're working on the next time you
open a terminal.

Inspired by [GAM (Graph Attention Memory, arxiv 2604.12285)](https://arxiv.org/abs/2604.12285).

---

## What you get

When you start a Claude Code session and ask anything that touches an entity
already in your memory — a project, a person, a grant, a decision — Claude
calls `recall_context` and gets back the relevant graph slice automatically.
You don't tell Claude what to remember, and you don't tell it when to recall.

Three layers feed the recall:

- **Entities** — stable long-term facts. "Alice founded Acme, based in Berlin
  Italy." "ProjectX is an AI reasoning partner for wet labs."
- **Past episodes** — date-stamped summaries of past conversations, queryable
  directly. "What did we discuss about grants last Tuesday?"
- **Live (current session)** — turn-by-turn capture of the in-progress
  session, available within ~2 minutes. "What did the user just say about
  pricing?"

A web visualization (`memagent graph`) shows the whole graph as a
force-directed network — entities as colored circles, episodes as grey
squares, edges as typed arrows.

---

## Setup

### 1. Install

memagent is a Python project with a CLI and an MCP server.

```bash
cd ~/memagent
uv sync                    # or pip install -e .
memagent init              # creates the SQLite DB and pre-loads the embedding model
```

The first `init` downloads `BAAI/bge-small-en-v1.5` (~130 MB) for local
embeddings. After that, no network calls except to Claude Haiku for
extraction.

### 2. Register the MCP server

```bash
claude mcp add --scope user memagent /home/$USER/memagent/.venv/bin/memagent-mcp
```

This makes `recall_context` and `remember` available as tools in every Claude
Code session. Verify:

```bash
claude mcp list
# memagent: /home/.../memagent-mcp  - ✓ Connected
```

### 3. Add the cron jobs

```bash
crontab -e
```

Append:

```
*/2  * * * * /home/USER/memagent/.venv/bin/memagent capture              >> ~/.memagent/cron.log 2>&1
*/30 * * * * /home/USER/memagent/.venv/bin/memagent consolidate-session  >> ~/.memagent/cron.log 2>&1
```

That's it. The system runs in the background.

---

## What runs when

| Job | Cadence | Cost | What it does |
|-----|---------|------|--------------|
| `memagent capture` | every 2 min | ~0.5 s, no LLM | Embeds new conversation turns and writes them to a live buffer (Tier 1). |
| `memagent consolidate-session` | every 30 min | one Haiku call per ~8 KB of transcript per segment | Drains the live buffer, detects topic boundaries, promotes each segment to a stored episode + extracts entity facts. |

You can also run either manually any time.

---

## How Claude uses it (no manual action)

In a Claude Code session, when you mention any project, person, decision, or
ask a question that has prior history, Claude will call `recall_context` and
prepend the result. You'll see something like:

```
## Entities

# Alice
- founder of Acme
- based in Berlin
- speaks Portuguese (native), English, Italian, Spanish, German — does not
  speak Polish

# ProjectX
- AI reasoning partner for wet labs
- knowledge graph with full provenance
- every hypothesis traceable to source data and literature

## Past episodes

[episode 2026-05-07] Designed the three-tier memory architecture for memagent…

## Live (current session)

[live 13:15] User: should we apply for the EU-Grant?
```

Claude only calls `recall` when it's relevant — you don't trigger it. If a
conversation is about something the graph has nothing on, recall returns
nothing and Claude proceeds normally.

---

## Manual operations

### Force a recall (for inspection)

```bash
memagent recall "grants and funding options"
```

Returns exactly what Claude would see for that query.

### Add a fact mid-session

```bash
memagent remember "Alice's friend Carol runs a wet lab in Munich, possible second pilot"
```

The text is treated as a one-turn micro-conversation and processed through
the same extraction pipeline (Haiku call + entity merging). Use this only for
things that won't appear naturally in the conversation transcript.

### Inspect episodes

```bash
memagent episodes                     # list all
memagent episodes dfcef718            # detail of one episode (id prefix)
```

Detail view shows the summary, every linked entity, and every turn with
timestamps.

### Visualize the graph

```bash
memagent graph
```

Opens a browser tab with the force-directed graph. Drag nodes to rearrange,
scroll to zoom, click any node for full detail in a side panel. Episodes
render as grey squares connected by dashed lines to the entities they
mention. Hover an edge to see its relationship type.

### Compact bloated entities

```bash
memagent compact
```

When a single entity has accumulated too many redundant facts across many
sessions, compaction runs a single Haiku call to merge overlapping facts and
remove session-narrative noise. Triggered automatically during consolidation,
but you can run it manually too.

### Status

```bash
memagent status
```

Counts and last-run timestamps.

---

## What gets remembered, and what doesn't

memagent extracts:

- Specific named **people** (Alice, Bob)
- Named **projects, products, codebases** (ProjectX, projectx-app, CoreLib.jl)
- **Companies and institutions** (Acme, CentralLab)
- **Published works** (papers by title or DOI)
- Named **grants or programs** (EU-Grant)
- Concrete **tools or technologies** discussed as entities

It deliberately ignores:

- Concepts being discussed or designed ("memory gap", "consolidation flow")
- Generic technical topics ("graph visualization", "knowledge graph")
- Problems being solved or critiques ("hallucinated edges", "noisy nodes")
- Architecture decisions or approaches ("three-tier architecture")
- Anything that wouldn't exist in the world outside the conversation

Every extracted entity and edge has a confidence score. Anything below 0.7 is
dropped before it touches the graph.

---

## Trust but verify

The graph is your memory, not a source of truth. If something is wrong:

```bash
sqlite3 ~/.memagent/memory.db
> UPDATE topic_nodes SET craw='...' WHERE csum LIKE 'Alice%';
> DELETE FROM topic_nodes WHERE id='...';
```

Run `memagent graph` again to see the updated state. Wrong facts in entities
have been the most common issue — usually from over-aggressive Haiku
extraction. The confidence gate has reduced this dramatically but isn't
perfect.

---

## File locations

```
~/.memagent/
├── memory.db        SQLite — all data
├── checkpoint.json  Per-session capture progress
├── d3.v7.min.js     Cached D3 for offline graph rendering
├── graph.html       Last rendered visualization
└── cron.log         Output from background jobs
```

To wipe everything and start fresh:

```bash
rm -rf ~/.memagent/memory.db ~/.memagent/checkpoint.json
memagent init
```

---

## Troubleshooting

**`memagent recall` returns nothing.**
Check whether anything is in the graph yet: `memagent status`. If counts are
zero, run `memagent capture && memagent consolidate-session` manually to
process the active session.

**`recall_context` doesn't appear in Claude Code.**
Verify: `claude mcp list`. If memagent isn't there, re-register with the
command in the setup section. Restart Claude Code after registering.

**Cron isn't running.**
Check `~/.memagent/cron.log` for output from the last runs. If it's empty,
verify `crontab -l` shows the entries with the correct binary path.

**Bloated entity descriptions.**
Run `memagent compact`. Adjust `COMPACT_THRESHOLD_LINES` in `consolidation.py`
if 25 lines is wrong for your usage.

**Hallucinated entities (concepts treated as entities).**
The extraction prompt has explicit negative examples but isn't perfect. Edit
or delete bad nodes directly via SQLite. If a class of hallucination keeps
recurring, add it to the negative examples in the prompt.

---

## Architecture in depth

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data model, schemas,
algorithms, and design choices.
