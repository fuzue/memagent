"""Three-tier consolidation: EPG turns → Archive episode → Entity graph (TAN).

Inspired by GAM (arxiv 2604.12285): write isolation between real-time event
buffer and stable entity graph, with cross-links so episodes remain queryable.
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from typing import Optional

from .embedder import cosine_similarity, embed_one, from_bytes, to_bytes
from .models import TopicNode
from .session_reader import Turn, turns_to_transcript
from .store import (
    add_entity_episode_link, add_episode, add_episode_turn, add_topic_edge,
    add_topic_node, get_all_topic_nodes, update_topic_node,
)

MERGE_THRESHOLD = 0.82
RELATED_THRESHOLD = 0.60
CONFIDENCE_THRESHOLD = 0.7
FACT_DEDUPE_SIM = 0.90        # facts with cosine similarity above this are duplicates
COMPACT_THRESHOLD_LINES = 25  # craw line count that triggers compaction


def consolidate_turns(turns: list[Turn], session_file: str = "") -> dict:
    """Process a batch of turns into the three tiers.

    Returns: {episode_id, nodes_created, edges_created}
    """
    if not turns:
        return {"episode_id": None, "nodes_created": 0, "edges_created": 0}

    transcript = turns_to_transcript(turns)
    started_at = turns[0].timestamp or time.time()
    ended_at = turns[-1].timestamp or time.time()

    # Tier 2: extract entities + episode summary, chunked if transcript is long
    extracted = _extract_chunked(transcript)
    summary = extracted.get("summary", f"Conversation with {len(turns)} turns")

    # Filter by confidence — drop low-confidence entities and edges
    entities = [
        e for e in extracted.get("entities", [])
        if float(e.get("confidence", 0.0)) >= CONFIDENCE_THRESHOLD
    ]
    raw_edges = [
        e for e in extracted.get("edges", [])
        if float(e.get("confidence", 0.0)) >= CONFIDENCE_THRESHOLD
    ]
    dropped_entities = len(extracted.get("entities", [])) - len(entities)
    dropped_edges = len(extracted.get("edges", [])) - len(raw_edges)

    # Archive episode
    episode_id = str(uuid.uuid4())
    add_episode(
        episode_id=episode_id,
        session_file=session_file,
        started_at=started_at,
        ended_at=ended_at,
        transcript=transcript,
        summary=summary,
        embedding=to_bytes(embed_one(summary)),
    )

    # Archive individual turns (Tier 1 frozen)
    for t in turns:
        add_episode_turn(
            turn_id=str(uuid.uuid4()),
            episode_id=episode_id,
            role=t.role,
            text=t.text,
            timestamp=t.timestamp,
            embedding=None,  # turns embedded lazily on first query
        )

    # Tier 3: update entity graph
    existing = get_all_topic_nodes()
    nodes_created = 0
    touched_node_ids: set[str] = set()

    for ent in entities:
        name: str = ent.get("name", "").strip()
        etype: str = ent.get("type", "fact")
        facts: list[str] = ent.get("facts", [])
        if not name or not facts:
            continue

        craw = f"# {name}\n" + "\n".join(f"- {f}" for f in facts)
        csum = f"{name}: " + "; ".join(facts[:3])
        new_vec = embed_one(csum)

        matched = _find_matching_node(name, new_vec, existing)
        if matched:
            new_facts = _dedupe_facts(facts, matched.craw)
            if new_facts:
                merged_craw = matched.craw + "\n" + "\n".join(f"- {f}" for f in new_facts)
                # Compact if too many lines
                if merged_craw.count("\n") + 1 >= COMPACT_THRESHOLD_LINES:
                    merged_craw = _compact_craw(name, merged_craw)
            else:
                merged_craw = matched.craw  # nothing new
            merged_embedding = to_bytes(embed_one(matched.csum))
            update_topic_node(matched.id, matched.csum, merged_craw, merged_embedding)
            touched_node_ids.add(matched.id)
            for n in existing:
                if n.id == matched.id:
                    n.craw = merged_craw
                    n.embedding = merged_embedding
        else:
            node = TopicNode.new(csum=csum, craw=craw, entity_type=etype)
            node.embedding = to_bytes(new_vec)
            add_topic_node(node)
            existing.append(node)
            touched_node_ids.add(node.id)
            nodes_created += 1

    # Cross-links: every entity touched by this episode is linked to it
    for nid in touched_node_ids:
        add_entity_episode_link(nid, episode_id, score=1.0)

    # Typed edges from extraction
    name_to_id = {n.csum.split(":")[0].strip().lower(): n.id for n in existing}
    edges_created = 0
    for e in raw_edges:
        src = name_to_id.get(e.get("from", "").strip().lower())
        tgt = name_to_id.get(e.get("to", "").strip().lower())
        etype = e.get("type", "RELATED_TO")
        if src and tgt and src != tgt:
            add_topic_edge(src, tgt, etype, weight=1.0)
            edges_created += 1

    return {
        "episode_id": episode_id,
        "nodes_created": nodes_created,
        "edges_created": edges_created,
        "entities_touched": len(touched_node_ids),
        "dropped_entities": dropped_entities,
        "dropped_edges": dropped_edges,
    }


def _extract_chunked(transcript: str, chunk_size: int = 8000) -> dict:
    """Extract entities/edges/summary across chunks and merge results."""
    if len(transcript) <= chunk_size:
        return _extract(transcript)

    summaries: list[str] = []
    entities_by_name: dict[str, dict] = {}
    edges_seen: set[tuple] = set()
    edges: list[dict] = []

    for i in range(0, len(transcript), chunk_size):
        chunk = transcript[i:i + chunk_size]
        result = _extract(chunk)
        if not result:
            continue
        if result.get("summary"):
            summaries.append(result["summary"])
        for ent in result.get("entities", []):
            name = ent.get("name", "").strip()
            if not name:
                continue
            key = name.lower()
            conf = float(ent.get("confidence", 0.0))
            if key in entities_by_name:
                existing = entities_by_name[key]
                existing_facts = set(existing.get("facts", []))
                for f in ent.get("facts", []):
                    if f not in existing_facts:
                        existing.setdefault("facts", []).append(f)
                        existing_facts.add(f)
                # Keep highest confidence seen across chunks
                existing["confidence"] = max(existing.get("confidence", 0.0), conf)
            else:
                entities_by_name[key] = {
                    "name": name,
                    "type": ent.get("type", "fact"),
                    "facts": list(ent.get("facts", [])),
                    "confidence": conf,
                }
        for e in result.get("edges", []):
            sig = (e.get("from", "").lower(), e.get("to", "").lower(), e.get("type", ""))
            if sig not in edges_seen and all(sig[:2]):
                edges_seen.add(sig)
                edges.append(e)

    return {
        "summary": " | ".join(summaries[:3])[:300] if summaries else "",
        "entities": list(entities_by_name.values()),
        "edges": edges,
    }


def _extract(transcript: str) -> dict:
    """Single Haiku call: returns episode summary + entities + edges with confidence."""
    prompt = f"""Analyze this conversation and extract long-term memory.

WHAT TO EXTRACT (real entities that exist in the world):
- Specific named people (Alice, Bob)
- Named projects, products, codebases (ProjectX, internal-cli)
- Companies and institutions (Acme, the central university)
- Published works (specific papers by title or DOI)
- Named grants or programs (an EU funding call by name)
- Concrete tools or technologies if discussed as an entity (PostgreSQL, React)

WHAT NOT TO EXTRACT:
- Concepts being discussed or designed ("memory gap", "consolidation flow", "design flaw")
- Generic technical topics ("graph visualization", "knowledge graph implementation")
- Problems being solved or critiques ("hallucinated edges", "noisy nodes")
- Architecture decisions or approaches ("three-tier architecture", "cron strategy")
- Anything you would describe with a definite article like "the X" — those are usually concepts, not entities

A good test: if removing this entity doesn't lose information about a real-world thing that exists outside this conversation, don't extract it.

Each entity and edge MUST include a confidence score from 0.0 to 1.0:
- 0.9+ : explicitly named multiple times with specific facts
- 0.7-0.9: named once with clear factual context
- 0.5-0.7: mentioned but ambiguous or could be a concept
- below 0.5: skip it

Conversation:
{transcript[:8000]}

Return ONLY valid JSON:
{{
  "summary": "1-2 sentence description of what was discussed",
  "entities": [
    {{"name": "Alice", "type": "person", "facts": ["founder of Acme"], "confidence": 0.95}},
    {{"name": "ProjectX", "type": "project", "facts": ["internal customer data ingestion tool"], "confidence": 0.9}}
  ],
  "edges": [
    {{"from": "Alice", "to": "Acme", "type": "FOUNDED", "confidence": 0.95}}
  ]
}}

Entity types: person, project, company, tool, grant, paper (snake_case, lowercase).
Edge types: SCREAMING_SNAKE_CASE. Common: FOUNDED, OWNS, WORKS_ON, COLLABORATES_WITH, MEMBER_OF, PART_OF, BLOCKS, FUNDS. Invent new types when they're more precise."""

    try:
        result = subprocess.run(
            ["claude", "--model", "claude-haiku-4-5-20251001", "-p", prompt],
            capture_output=True, text=True, timeout=60,
        )
        raw = result.stdout.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception:
        return {}


def _dedupe_facts(new_facts: list[str], existing_craw: str) -> list[str]:
    """Filter new_facts: drop any that exactly or semantically match an existing fact.

    Embedding similarity above FACT_DEDUPE_SIM is treated as a duplicate.
    """
    existing_facts = [
        line.lstrip("- ").strip()
        for line in existing_craw.splitlines()
        if line.startswith("- ")
    ]
    if not existing_facts:
        return [f for f in new_facts if f.strip()]

    existing_lower = {f.lower() for f in existing_facts}
    existing_vecs = [embed_one(f) for f in existing_facts]

    kept = []
    for fact in new_facts:
        f = fact.strip()
        if not f or f.lower() in existing_lower:
            continue
        v = embed_one(f)
        if any(cosine_similarity(v, ev) >= FACT_DEDUPE_SIM for ev in existing_vecs):
            continue
        kept.append(f)
        existing_facts.append(f)
        existing_lower.add(f.lower())
        existing_vecs.append(v)
    return kept


def _compact_craw(name: str, craw: str) -> str:
    """Use Haiku to aggressively merge a long craw into a clean entity description."""
    prompt = f"""Rewrite this entity description as a clean, non-redundant set of facts.

The current description has accumulated facts across many sessions, with overlap.
Be aggressive about consolidation:
- Merge facts that describe the same thing in different words (keep the most informative version)
- Drop generic restatements ('memory system for X' when 'dogfood prototype of X' is already there)
- Drop session-narrative noise ('design flaw identified', 'recently fixed', 'needs dynamic types') — these belong in episodes, not in the entity's stable description
- Keep concrete facts: identity, status, components, dependencies, risks
- Reorder logically: identity → purpose → key components → status → risks
- Aim for half the original length when possible

Format:
- Start with '# {name}'
- Bullet points only ('- fact')
- No commentary, no explanations, return only the cleaned description

Entity: {name}

Current description:
{craw}"""

    try:
        result = subprocess.run(
            ["claude", "--model", "claude-haiku-4-5-20251001", "-p", prompt],
            capture_output=True, text=True, timeout=30,
        )
        out = result.stdout.strip()
        if out.startswith("```"):
            out = out.split("```")[1]
            if out.startswith("markdown") or out.startswith("md"):
                out = out.split("\n", 1)[1]
        # Sanity check: must still start with the entity header
        if out.startswith(f"# {name}") or out.startswith(f"#{name}"):
            return out
    except Exception:
        pass
    return craw  # fall back to original if anything goes wrong


def _find_matching_node(name: str, new_vec, existing: list[TopicNode]) -> Optional[TopicNode]:
    name_lower = name.lower()
    for node in existing:
        if node.csum.lower().startswith(name_lower + ":") or node.csum.lower() == name_lower:
            return node

    best: Optional[TopicNode] = None
    best_sim = MERGE_THRESHOLD
    for node in existing:
        if node.embedding is None:
            continue
        sim = cosine_similarity(new_vec, from_bytes(node.embedding))
        if sim > best_sim:
            best_sim = sim
            best = node
    return best
