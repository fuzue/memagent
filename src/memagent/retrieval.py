from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .embedder import cosine_similarity, embed_one, from_bytes
from .models import Event, TopicNode
from .store import get_all_topic_nodes, get_current_session_events, get_topic_neighbors


@dataclass
class Result:
    text: str
    score: float
    source: str  # "topic" | "event"
    entity_type: Optional[str] = None
    timestamp: Optional[float] = None


def recall(query: str, session_id: Optional[str] = None, top_n: int = 8) -> List[Result]:
    query_vec = embed_one(query)
    results: List[Result] = []

    # ── Search topic graph ────────────────────────────────────────────────────
    topic_nodes = get_all_topic_nodes()
    topic_scores: List[tuple[TopicNode, float]] = []

    for node in topic_nodes:
        if node.embedding is None:
            continue
        vec = from_bytes(node.embedding)
        sim = cosine_similarity(query_vec, vec)
        recency = _recency_boost(node.updated_at)
        score = sim * 0.8 + recency * 0.2
        topic_scores.append((node, score))

    topic_scores.sort(key=lambda x: x[1], reverse=True)
    anchor_nodes = topic_scores[:4]

    seen_ids = set()
    for node, score in anchor_nodes:
        if score < 0.15:
            continue
        results.append(Result(
            text=node.csum,
            score=score,
            source="topic",
            entity_type=node.entity_type,
            timestamp=node.updated_at,
        ))
        seen_ids.add(node.id)

        # One-hop expansion via graph edges
        for neighbor in get_topic_neighbors(node.id):
            if neighbor.id in seen_ids or neighbor.embedding is None:
                continue
            n_vec = from_bytes(neighbor.embedding)
            n_sim = cosine_similarity(query_vec, n_vec)
            if n_sim > 0.1:
                results.append(Result(
                    text=neighbor.csum,
                    score=n_sim * 0.7,  # discount for being indirect
                    source="topic",
                    entity_type=neighbor.entity_type,
                    timestamp=neighbor.updated_at,
                ))
                seen_ids.add(neighbor.id)

    # ── Search current session event graph ────────────────────────────────────
    if session_id:
        events = get_current_session_events(session_id, limit=150)
        event_scores: List[tuple[Event, float]] = []

        for event in events:
            if event.embedding is not None:
                vec = from_bytes(event.embedding)
                sim = cosine_similarity(query_vec, vec)
            else:
                # Fall back to keyword overlap when embedding not yet ready
                sim = _keyword_score(query, event.text)
            recency = _recency_boost(event.timestamp)
            score = sim * 0.75 + recency * 0.25
            event_scores.append((event, score))

        event_scores.sort(key=lambda x: x[1], reverse=True)
        for event, score in event_scores[:5]:
            if score < 0.1:
                continue
            results.append(Result(
                text=event.text,
                score=score,
                source="event",
                timestamp=event.timestamp,
            ))

    # Deduplicate and rank
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]


def format_context(results: List[Result]) -> str:
    if not results:
        return "No relevant context found."

    lines = []
    topic_results = [r for r in results if r.source == "topic"]
    event_results = [r for r in results if r.source == "event"]

    if topic_results:
        lines.append("## From memory")
        for r in topic_results:
            tag = f"[{r.entity_type}] " if r.entity_type else ""
            lines.append(f"- {tag}{r.text}")

    if event_results:
        lines.append("\n## From this session")
        for r in event_results:
            lines.append(f"- {r.text}")

    return "\n".join(lines)


def _recency_boost(timestamp: float, half_life_days: float = 14.0) -> float:
    age_seconds = time.time() - timestamp
    age_days = age_seconds / 86400.0
    return float(np.exp(-age_days / half_life_days))


def _keyword_score(query: str, text: str) -> float:
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not query_words:
        return 0.0
    overlap = query_words & text_words
    return len(overlap) / len(query_words)
