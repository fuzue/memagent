from __future__ import annotations

import json
import os
from typing import List, Optional

import anthropic

from .embedder import embed_one, to_bytes
from .models import Event, TopicNode
from .store import (
    add_cross_link, add_topic_edge, add_topic_node,
    get_all_topic_nodes, get_unconsolidated_events,
    mark_events_consolidated, update_topic_node,
)

_client: Optional[anthropic.Anthropic] = None


def _llm() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


MODEL = "claude-haiku-4-5-20251001"


def consolidate(session_id: str) -> int:
    events = get_unconsolidated_events(session_id)
    if not events:
        return 0

    segments = _segment_events(events)
    nodes_created = 0

    for seg in segments:
        csum = seg["csum"]
        craw = "\n".join(seg["texts"])
        entity_type = seg.get("entity_type", "fact")

        embedding = to_bytes(embed_one(csum))

        existing = get_all_topic_nodes()
        matched_id = _find_matching_topic(csum, existing) if existing else None

        if matched_id:
            existing_node = next(n for n in existing if n.id == matched_id)
            merged_craw = existing_node.craw + "\n\n---\n\n" + craw
            merged_csum = _merge_summaries(existing_node.csum, csum)
            merged_embedding = to_bytes(embed_one(merged_csum))
            update_topic_node(matched_id, merged_csum, merged_craw, merged_embedding)
            add_cross_link(matched_id, session_id)
        else:
            node = TopicNode.new(csum=csum, craw=craw, entity_type=entity_type)
            node.embedding = embedding
            add_topic_node(node)
            add_cross_link(node.id, session_id)
            nodes_created += 1

            # Link to semantically related existing nodes
            _link_to_related(node, existing)

    mark_events_consolidated(session_id)
    return nodes_created


def _segment_events(events: List[Event]) -> List[dict]:
    event_text = "\n".join(
        f"[{i+1}] ({e.tool_name or 'note'}) {e.text[:300]}"
        for i, e in enumerate(events)
    )

    prompt = f"""You are analyzing work session events and must segment them into coherent topic chunks.

Events:
{event_text}

Instructions:
- Group consecutive related events into 2-6 segments
- Each segment should cover a single coherent topic or task
- For each segment provide: a 1-2 sentence summary (csum), the entity type, and which event numbers belong to it
- Entity types: fact, decision, work, problem, solution, person, project

Return ONLY valid JSON in this format:
{{
  "segments": [
    {{
      "event_indices": [1, 2, 3],
      "csum": "concise summary of what happened",
      "entity_type": "work"
    }}
  ]
}}"""

    response = _llm().messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except Exception:
        # Fallback: treat all events as one segment
        return [{
            "texts": [e.text for e in events],
            "csum": f"Session with {len(events)} events covering: {events[0].text[:100]}",
            "entity_type": "work",
        }]

    result = []
    for seg in data.get("segments", []):
        indices = [i - 1 for i in seg.get("event_indices", []) if 1 <= i <= len(events)]
        texts = [events[i].text for i in indices if i < len(events)]
        if texts:
            result.append({
                "texts": texts,
                "csum": seg.get("csum", ""),
                "entity_type": seg.get("entity_type", "fact"),
            })

    return result if result else [{"texts": [e.text for e in events], "csum": "Session events", "entity_type": "work"}]


def _find_matching_topic(new_csum: str, existing: List[TopicNode]) -> Optional[str]:
    if not existing:
        return None

    # Coarse filter: top 5 by embedding similarity
    from .embedder import cosine_similarity, embed_one, from_bytes
    new_vec = embed_one(new_csum)
    scored = []
    for node in existing:
        if node.embedding is None:
            continue
        sim = cosine_similarity(new_vec, from_bytes(node.embedding))
        scored.append((node, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    candidates = scored[:5]

    if not candidates or candidates[0][1] < 0.5:
        return None

    # Fine filter: LLM decides
    candidate_text = "\n".join(
        f"{i+1}. [{n.entity_type}] {n.csum}"
        for i, (n, _) in enumerate(candidates)
    )

    prompt = f"""New topic summary: "{new_csum}"

Existing topics:
{candidate_text}

Is the new topic the same as any existing topic (same subject, should be merged)?
Return ONLY JSON: {{"match": 1}} if it matches topic 1, or {{"match": null}} if no match."""

    response = _llm().messages.create(
        model=MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        data = json.loads(response.content[0].text.strip())
        match_idx = data.get("match")
        if match_idx and 1 <= match_idx <= len(candidates):
            return candidates[match_idx - 1][0].id
    except Exception:
        pass
    return None


def _merge_summaries(old: str, new: str) -> str:
    prompt = f"""Merge these two summaries of the same topic into one concise 1-2 sentence summary:

Old: {old}
New: {new}

Return only the merged summary, no explanation."""

    response = _llm().messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _link_to_related(new_node: TopicNode, existing: List[TopicNode]) -> None:
    from .embedder import cosine_similarity, from_bytes
    if new_node.embedding is None:
        return
    new_vec = from_bytes(new_node.embedding)

    for node in existing:
        if node.embedding is None:
            continue
        sim = cosine_similarity(new_vec, from_bytes(node.embedding))
        if sim > 0.6:
            add_topic_edge(new_node.id, node.id, "RELATED_TO", weight=sim)
