"""memagent CLI.

Commands:
  memagent consolidate-session   read active Claude Code session, extract entities (run via cron)
  memagent remember <text>       explicitly store a fact mid-session
  memagent recall <query>        test retrieval from the command line
  memagent graph                 export knowledge graph as HTML and open in browser
  memagent episodes [<id>]       list episodes or show one in detail
  memagent compact               merge redundant facts in oversized entity descriptions
  memagent status                show graph stats and last consolidation time
  memagent init                  create DB and download embedding model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CHECKPOINT_FILE = Path.home() / ".memagent" / "checkpoint.json"


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]

    if cmd == "consolidate-session":
        _cmd_consolidate_session()
    elif cmd == "graph":
        _cmd_graph()
    elif cmd == "episodes":
        if len(args) > 1:
            _cmd_episode_detail(args[1])
        else:
            _cmd_episodes_list()
    elif cmd == "compact":
        _cmd_compact()
    elif cmd == "remember":
        text = " ".join(args[1:])
        if not text:
            print("Usage: memagent remember <text>", file=sys.stderr)
            sys.exit(1)
        _cmd_remember(text)
    elif cmd == "recall":
        query = " ".join(args[1:]) if len(args) > 1 else "current project context"
        _cmd_recall(query)
    elif cmd == "status":
        _cmd_status()
    elif cmd == "init":
        _cmd_init()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


# ── Commands ──────────────────────────────────────────────────────────────────

def _cmd_consolidate_session():
    from .boundaries import split_at_boundaries
    from .consolidation import consolidate_turns
    from .db import init_db
    from .session_reader import find_active_session_file, read_turns_since

    init_db()

    session_file = find_active_session_file()
    if not session_file:
        print("[memagent] No Claude Code session found.", file=sys.stderr)
        return

    checkpoint = _load_checkpoint()
    file_key = str(session_file)
    since_iso = checkpoint.get("sessions", {}).get(file_key)

    turns = read_turns_since(session_file, since_iso)
    if not turns:
        print("[memagent] No new turns since last consolidation.", file=sys.stderr)
        return

    # Detect semantic boundaries — split into one episode per coherent segment
    segments = split_at_boundaries(turns)

    print(
        f"[memagent] {len(turns)} turns split into {len(segments)} segment(s)",
        file=sys.stderr,
    )

    total_new = total_touched = total_edges = total_dropped_e = total_dropped_x = 0
    for i, segment in enumerate(segments, 1):
        result = consolidate_turns(segment, session_file=str(session_file))
        total_new += result["nodes_created"]
        total_touched += result["entities_touched"]
        total_edges += result["edges_created"]
        total_dropped_e += result.get("dropped_entities", 0)
        total_dropped_x += result.get("dropped_edges", 0)
        print(
            f"  [{i}/{len(segments)}] episode {result['episode_id'][:8] if result['episode_id'] else '-'} "
            f"| {len(segment)} turns | +{result['nodes_created']} entities "
            f"| {result['entities_touched']} touched "
            f"| {result['edges_created']} edges",
            file=sys.stderr,
        )

    if turns[-1].iso_ts:
        checkpoint.setdefault("sessions", {})[file_key] = turns[-1].iso_ts
        _save_checkpoint(checkpoint)

    print(
        f"[memagent] total: +{total_new} entities, {total_touched} touched, "
        f"{total_edges} edges, dropped {total_dropped_e} entities, "
        f"{total_dropped_x} edges (low confidence)",
        file=sys.stderr,
    )


def _cmd_graph():
    import http.server
    import socket
    import subprocess
    import threading
    from .db import init_db
    from .graph_export import export_html

    init_db()
    output = Path.home() / ".memagent" / "graph.html"
    export_html(output)

    # Find a free port
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    serve_dir = str(output.parent)
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("", port), lambda *a, **kw: handler(*a, directory=serve_dir, **kw))

    url = f"http://localhost:{port}/graph.html"
    print(f"Serving at {url}  (Ctrl+C to stop)", file=sys.stderr)

    threading.Thread(target=lambda: subprocess.call(["xdg-open", url]), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


def _cmd_episodes_list():
    import time as _time
    from .db import get_conn, init_db
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT e.id, e.summary, e.started_at, e.ended_at,
                      (SELECT COUNT(*) FROM episode_turns WHERE episode_id=e.id) AS n_turns,
                      (SELECT COUNT(*) FROM entity_episode_links WHERE episode_id=e.id) AS n_entities
               FROM episodes e
               ORDER BY e.started_at DESC"""
        ).fetchall()

    if not rows:
        print("No episodes archived yet.")
        return

    for r in rows:
        date = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(r["started_at"]))
        dur_min = (r["ended_at"] - r["started_at"]) / 60
        print(f"{r['id'][:8]} | {date} | {dur_min:5.0f}m | {r['n_turns']:3} turns | {r['n_entities']} entities")
        print(f"         {r['summary'][:200]}\n")


def _cmd_episode_detail(episode_id_prefix: str):
    import time as _time
    from .db import get_conn, init_db
    init_db()
    with get_conn() as conn:
        ep = conn.execute(
            "SELECT * FROM episodes WHERE id LIKE ?",
            (episode_id_prefix + "%",)
        ).fetchone()
        if not ep:
            print(f"Episode {episode_id_prefix} not found.")
            return
        turns = conn.execute(
            "SELECT role, text, timestamp FROM episode_turns WHERE episode_id=? ORDER BY timestamp",
            (ep["id"],)
        ).fetchall()
        entities = conn.execute(
            """SELECT t.csum FROM entity_episode_links l
               JOIN topic_nodes t ON t.id = l.entity_node_id
               WHERE l.episode_id=?""",
            (ep["id"],)
        ).fetchall()

    print(f"Episode: {ep['id']}")
    print(f"Period:  {_time.strftime('%Y-%m-%d %H:%M', _time.localtime(ep['started_at']))} → "
          f"{_time.strftime('%H:%M', _time.localtime(ep['ended_at']))}")
    print(f"Summary: {ep['summary']}\n")

    print(f"Linked entities ({len(entities)}):")
    for e in entities:
        print(f"  • {e['csum'][:120]}")

    print(f"\nTurns ({len(turns)}):")
    for t in turns:
        ts = _time.strftime("%H:%M:%S", _time.localtime(t["timestamp"]))
        role = "User " if t["role"] == "user" else "Claude"
        preview = t["text"][:200].replace("\n", " ")
        print(f"  [{ts}] {role}: {preview}")


def _cmd_compact():
    """Compact every entity node whose craw exceeds the threshold."""
    from .consolidation import COMPACT_THRESHOLD_LINES, _compact_craw
    from .db import init_db
    from .embedder import embed_one, to_bytes
    from .store import get_all_topic_nodes, update_topic_node

    init_db()
    nodes = get_all_topic_nodes()
    compacted = 0

    for node in nodes:
        line_count = node.craw.count("\n") + 1
        if line_count < COMPACT_THRESHOLD_LINES:
            continue
        name = node.csum.split(":")[0].strip()
        before = line_count
        new_craw = _compact_craw(name, node.craw)
        after = new_craw.count("\n")
        if new_craw != node.craw:
            update_topic_node(
                node.id, node.csum, new_craw, to_bytes(embed_one(node.csum))
            )
            compacted += 1
            print(f"  {name}: {before} → {after} lines", file=sys.stderr)

    print(f"Compacted {compacted} entities.", file=sys.stderr)


def _cmd_remember(text: str):
    import time as _time
    from .consolidation import consolidate_turns
    from .db import init_db
    from .session_reader import Turn

    init_db()
    now = _time.time()
    turn = Turn(role="user", text=f"Remember this: {text}", timestamp=now, iso_ts="")
    result = consolidate_turns([turn], session_file="manual:remember")
    print(
        f"Stored. {result['nodes_created']} new entities, "
        f"{result['entities_touched']} touched.",
        file=sys.stderr,
    )


def _cmd_recall(query: str):
    from .retrieval import format_context, recall
    from .db import init_db

    init_db()
    results = recall(query)
    print(format_context(results))


def _cmd_status():
    from .db import get_conn, init_db

    init_db()
    with get_conn() as conn:
        n_topics = conn.execute("SELECT COUNT(*) FROM topic_nodes").fetchone()[0]
        n_edges = conn.execute("SELECT COUNT(*) FROM topic_edges").fetchone()[0]

    checkpoint = _load_checkpoint()
    sessions = checkpoint.get("sessions", {})
    last = max(sessions.values()) if sessions else "never"

    print(f"Topic nodes : {n_topics}")
    print(f"Topic edges : {n_edges}")
    print(f"Last run    : {last}")


def _cmd_init():
    from .db import init_db

    init_db()
    print("Database initialised.", file=sys.stderr)
    print("Downloading embedding model (first time only)...", file=sys.stderr)
    from .embedder import embed_one
    embed_one("test")
    print("Ready.")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_checkpoint(data: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))
