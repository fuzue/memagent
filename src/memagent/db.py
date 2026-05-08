from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path.home() / ".memagent" / "memory.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                started_at  REAL NOT NULL,
                ended_at    REAL,
                cwd         TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id           TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL,
                text         TEXT NOT NULL,
                embedding    BLOB,
                timestamp    REAL NOT NULL,
                tool_name    TEXT,
                file_path    TEXT,
                consolidated INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS topic_nodes (
                id          TEXT PRIMARY KEY,
                csum        TEXT NOT NULL,
                craw        TEXT NOT NULL,
                embedding   BLOB,
                entity_type TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS topic_edges (
                source_id   TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                edge_type   TEXT NOT NULL,
                weight      REAL DEFAULT 1.0,
                created_at  REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_type),
                FOREIGN KEY (source_id) REFERENCES topic_nodes(id),
                FOREIGN KEY (target_id) REFERENCES topic_nodes(id)
            );

            CREATE TABLE IF NOT EXISTS cross_links (
                topic_id    TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                PRIMARY KEY (topic_id, session_id),
                FOREIGN KEY (topic_id) REFERENCES topic_nodes(id)
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id           TEXT PRIMARY KEY,
                session_file TEXT,
                started_at   REAL NOT NULL,
                ended_at     REAL NOT NULL,
                transcript   TEXT NOT NULL,
                summary      TEXT,
                embedding    BLOB
            );

            CREATE TABLE IF NOT EXISTS episode_turns (
                id          TEXT PRIMARY KEY,
                episode_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                text        TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                embedding   BLOB,
                FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS entity_episode_links (
                entity_node_id  TEXT NOT NULL,
                episode_id      TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                created_at      REAL NOT NULL,
                PRIMARY KEY (entity_node_id, episode_id),
                FOREIGN KEY (entity_node_id) REFERENCES topic_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (episode_id)     REFERENCES episodes(id)    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_events_session
                ON events(session_id, consolidated);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_topic_updated
                ON topic_nodes(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_started
                ON episodes(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_episode_turns_episode
                ON episode_turns(episode_id);
            CREATE INDEX IF NOT EXISTS idx_entity_episode_entity
                ON entity_episode_links(entity_node_id);
        """)
