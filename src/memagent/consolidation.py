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
    entities = extracted.get("entities", [])
    raw_edges = extracted.get("edges", [])

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
            merged_craw = matched.craw + "\n" + "\n".join(f"- {f}" for f in facts)
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
            if key in entities_by_name:
                # Merge facts, dedupe
                existing_facts = set(entities_by_name[key].get("facts", []))
                for f in ent.get("facts", []):
                    if f not in existing_facts:
                        entities_by_name[key].setdefault("facts", []).append(f)
                        existing_facts.add(f)
            else:
                entities_by_name[key] = {
                    "name": name,
                    "type": ent.get("type", "fact"),
                    "facts": list(ent.get("facts", [])),
                }
        for e in result.get("edges", []):
            sig = (e.get("from", "").lower(), e.get("to", "").lower(), e.get("type", ""))
            if sig not in edges_seen and all(sig):
                edges_seen.add(sig)
                edges.append(e)

    return {
        "summary": " | ".join(summaries[:3])[:300] if summaries else "",
        "entities": list(entities_by_name.values()),
        "edges": edges,
    }


def _extract(transcript: str) -> dict:
    """Single Haiku call: returns episode summary + entities + edges."""
    prompt = f"""Analyze this conversation and extract structured memory.

Rules:
- Only extract facts explicitly stated. No inferences, no meta-references.
- Focus on long-term memory: people, projects, companies, tools, decisions, constraints.
- Ignore ephemeral actions (running commands, reading files, tool outputs).

Conversation:
{transcript[:8000]}

Return ONLY valid JSON:
{{
  "summary": "1-2 sentence description of what was discussed in this episode",
  "entities": [
    {{"name": "Alice", "type": "person", "facts": ["founder of Acme", "based in Berlin"]}},
    {{"name": "ProjectX", "type": "project", "facts": ["AI reasoning partner for wet labs"]}}
  ],
  "edges": [
    {{"from": "Alice", "to": "Acme", "type": "FOUNDED"}},
    {{"from": "Acme", "to": "ProjectX", "type": "OWNS"}}
  ]
}}

Use snake_case for entity types and SCREAMING_SNAKE for edge types. Common entity types: person, project, company, tool, decision, constraint, grant. Common edge types: FOUNDED, OWNS, WORKS_ON, COLLABORATES_WITH, MEMBER_OF, PART_OF, BLOCKS, FUNDS. You may invent new types when they describe the relationship more precisely than any of these."""

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
