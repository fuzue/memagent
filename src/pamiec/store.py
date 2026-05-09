from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from .db import get_conn, init_db
from .models import Event, Session, TopicNode

SESSION_FILE = Path.home() / ".pamiec" / "session.json"


# ── Session management ────────────────────────────────────────────────────────

def get_or_create_session(cwd: str = None) -> Session:
    if SESSION_FILE.exists():
        data = json.loads(SESSION_FILE.read_text())
        return Session(**data)
    session = Session.new(cwd=cwd)
    _save_session_file(session)
    init_db()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, started_at, cwd) VALUES (?, ?, ?)",
            (session.id, session.started_at, session.cwd),
        )
    return session


def end_session(session_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at=? WHERE id=?",
            (time.time(), session_id),
        )
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


def _save_session_file(session: Session) -> None:
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(json.dumps({
        "id": session.id,
        "started_at": session.started_at,
        "cwd": session.cwd,
    }))


# ── Events ────────────────────────────────────────────────────────────────────

def add_event(event: Event) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO events
               (id, session_id, text, embedding, timestamp, tool_name, file_path, consolidated)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (event.id, event.session_id, event.text, event.embedding,
             event.timestamp, event.tool_name, event.file_path),
        )


def update_event_embedding(event_id: str, embedding: bytes) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE events SET embedding=? WHERE id=?",
            (embedding, event_id),
        )


def get_unconsolidated_events(session_id: str) -> List[Event]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM events WHERE session_id=? AND consolidated=0
               ORDER BY timestamp ASC""",
            (session_id,),
        ).fetchall()
    return [_row_to_event(r) for r in rows]


def get_current_session_events(session_id: str, limit: int = 200) -> List[Event]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM events WHERE session_id=?
               ORDER BY timestamp DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
    return [_row_to_event(r) for r in reversed(rows)]


def mark_events_consolidated(session_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE events SET consolidated=1 WHERE session_id=?",
            (session_id,),
        )


def _row_to_event(r) -> Event:
    return Event(
        id=r["id"], session_id=r["session_id"], text=r["text"],
        timestamp=r["timestamp"], tool_name=r["tool_name"],
        file_path=r["file_path"], embedding=r["embedding"],
        consolidated=bool(r["consolidated"]),
    )


# ── Topic nodes ───────────────────────────────────────────────────────────────

def add_topic_node(node: TopicNode) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO topic_nodes
               (id, csum, craw, embedding, entity_type, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (node.id, node.csum, node.craw, node.embedding,
             node.entity_type, node.created_at, node.updated_at),
        )


def update_topic_node(node_id: str, csum: str, craw: str, embedding: bytes) -> None:
    with get_conn() as conn:
        conn.execute(
            """UPDATE topic_nodes SET csum=?, craw=?, embedding=?, updated_at=?
               WHERE id=?""",
            (csum, craw, embedding, time.time(), node_id),
        )


def get_all_topic_nodes() -> List[TopicNode]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM topic_nodes ORDER BY updated_at DESC"
        ).fetchall()
    return [_row_to_topic(r) for r in rows]


def add_topic_edge(source_id: str, target_id: str, edge_type: str, weight: float = 1.0) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO topic_edges
               (source_id, target_id, edge_type, weight, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (source_id, target_id, edge_type, weight, time.time()),
        )


def get_topic_neighbors(node_id: str) -> List[TopicNode]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT tn.* FROM topic_nodes tn
               JOIN topic_edges te ON (te.target_id=tn.id OR te.source_id=tn.id)
               WHERE (te.source_id=? OR te.target_id=?) AND tn.id != ?""",
            (node_id, node_id, node_id),
        ).fetchall()
    return [_row_to_topic(r) for r in rows]


def add_cross_link(topic_id: str, session_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO cross_links (topic_id, session_id) VALUES (?, ?)",
            (topic_id, session_id),
        )


def _row_to_topic(r) -> TopicNode:
    return TopicNode(
        id=r["id"], csum=r["csum"], craw=r["craw"],
        entity_type=r["entity_type"], created_at=r["created_at"],
        updated_at=r["updated_at"], embedding=r["embedding"],
    )


# ── Episodes (Tier 2 — archive of consolidated EPGs) ──────────────────────────

def add_episode(
    episode_id: str, session_file: str, started_at: float, ended_at: float,
    transcript: str, summary: str, embedding: bytes
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO episodes (id, session_file, started_at, ended_at, transcript, summary, embedding)
               VALUES (?,?,?,?,?,?,?)""",
            (episode_id, session_file, started_at, ended_at, transcript, summary, embedding),
        )


def add_episode_turn(
    turn_id: str, episode_id: str, role: str, text: str,
    timestamp: float, embedding: bytes = None
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO episode_turns (id, episode_id, role, text, timestamp, embedding)
               VALUES (?,?,?,?,?,?)""",
            (turn_id, episode_id, role, text, timestamp, embedding),
        )


def get_all_episode_turns() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, episode_id, role, text, timestamp, embedding FROM episode_turns"
        ).fetchall()
    return [dict(r) for r in rows]


def update_episode_turn_embedding(turn_id: str, embedding: bytes) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE episode_turns SET embedding=? WHERE id=?",
            (embedding, turn_id),
        )


def add_entity_episode_link(entity_node_id: str, episode_id: str, score: float = 1.0) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO entity_episode_links
               (entity_node_id, episode_id, relevance_score, created_at)
               VALUES (?,?,?,?)""",
            (entity_node_id, episode_id, score, time.time()),
        )


def get_episodes_for_entity(entity_node_id: str, limit: int = 5) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT e.id, e.summary, e.transcript, e.started_at, e.ended_at, l.relevance_score
               FROM entity_episode_links l
               JOIN episodes e ON e.id = l.episode_id
               WHERE l.entity_node_id = ?
               ORDER BY l.relevance_score DESC, e.started_at DESC
               LIMIT ?""",
            (entity_node_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_episodes() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, summary, transcript, started_at, ended_at, embedding
               FROM episodes
               ORDER BY started_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


# ── Live EPG buffer (Tier 1 — real-time, pre-consolidation) ───────────────────

def add_epg_turn(
    turn_id: str, session_file: str, role: str, text: str,
    timestamp: float, iso_ts: str, embedding: bytes
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO epg_turns
               (id, session_file, role, text, timestamp, iso_ts, embedding, captured_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (turn_id, session_file, role, text, timestamp, iso_ts, embedding, time.time()),
        )


def get_epg_turns(session_file: str = None) -> list:
    with get_conn() as conn:
        if session_file:
            rows = conn.execute(
                """SELECT * FROM epg_turns WHERE session_file = ?
                   ORDER BY timestamp ASC""",
                (session_file,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM epg_turns ORDER BY timestamp ASC"
            ).fetchall()
    return [dict(r) for r in rows]


def delete_epg_turns(turn_ids: list[str]) -> None:
    if not turn_ids:
        return
    placeholders = ",".join("?" for _ in turn_ids)
    with get_conn() as conn:
        conn.execute(f"DELETE FROM epg_turns WHERE id IN ({placeholders})", turn_ids)


def epg_turn_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM epg_turns").fetchone()[0]
