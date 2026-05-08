from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .embedder import cosine_similarity, embed_one, from_bytes
from .models import Event, TopicNode
from .store import (
    get_all_episodes, get_all_topic_nodes, get_current_session_events,
    get_epg_turns, get_episodes_for_entity, get_topic_neighbors,
)


@dataclass
class Result:
    text: str
    score: float
    source: str  # "topic" | "event"
    node_id: Optional[str] = None
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
            text=node.craw,
            score=score,
            source="topic",
            node_id=node.id,
            entity_type=node.entity_type,
            timestamp=node.updated_at,
        ))
        seen_ids.add(node.id)

        # One-hop expansion — only pull neighbors that are also relevant to the query
        for neighbor in get_topic_neighbors(node.id):
            if neighbor.id in seen_ids or neighbor.embedding is None:
                continue
            n_vec = from_bytes(neighbor.embedding)
            n_sim = cosine_similarity(query_vec, n_vec)
            if n_sim > 0.4:
                results.append(Result(
                    text=neighbor.csum,
                    score=n_sim * 0.7,
                    source="topic",
                    node_id=neighbor.id,
                    entity_type=neighbor.entity_type,
                    timestamp=neighbor.updated_at,
                ))
                seen_ids.add(neighbor.id)

    # ── Search episodes (Tier 2 archive) ──────────────────────────────────────
    for ep in get_all_episodes():
        if not ep.get("embedding"):
            continue
        sim = cosine_similarity(query_vec, from_bytes(ep["embedding"]))
        if sim < 0.4:
            continue
        recency = _recency_boost(ep["started_at"])
        score = sim * 0.7 + recency * 0.3
        results.append(Result(
            text=f"[episode {time.strftime('%Y-%m-%d', time.localtime(ep['started_at']))}] {ep['summary']}",
            score=score * 0.85,
            source="episode",
            node_id=ep["id"],
            timestamp=ep["started_at"],
        ))

    # ── Search live EPG (Tier 1 — pre-consolidation, real-time) ───────────────
    for r in get_epg_turns():
        if not r.get("embedding"):
            continue
        sim = cosine_similarity(query_vec, from_bytes(r["embedding"]))
        if sim < 0.5:
            continue
        recency = _recency_boost(r["timestamp"], half_life_days=0.04)  # ~1h half-life
        score = sim * 0.6 + recency * 0.4
        ts_label = time.strftime("%H:%M", time.localtime(r["timestamp"]))
        role_label = "User" if r["role"] == "user" else "Claude"
        snippet = r["text"][:200].replace("\n", " ")
        results.append(Result(
            text=f"[live {ts_label}] {role_label}: {snippet}",
            score=score * 0.75,  # discount vs stable layers
            source="epg",
            node_id=r["id"],
            timestamp=r["timestamp"],
        ))

    # Deduplicate by node ID, keep highest score
    seen_keys: set[str] = set()
    deduped = []
    for r in sorted(results, key=lambda r: r.score, reverse=True):
        key = r.node_id or r.text[:80]
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(r)
    return deduped[:top_n]


def format_context(results: List[Result]) -> str:
    if not results:
        return "No relevant context found."

    topic_lines = [r.text for r in results if r.source == "topic"]
    episode_lines = [r.text for r in results if r.source == "episode"]
    epg_lines = [r.text for r in results if r.source == "epg"]

    parts = []
    if topic_lines:
        parts.append("## Entities\n\n" + "\n\n".join(topic_lines))
    if episode_lines:
        parts.append("## Past episodes\n\n" + "\n".join(episode_lines))
    if epg_lines:
        parts.append("## Live (current session)\n\n" + "\n".join(epg_lines))
    return "\n\n".join(parts)


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
