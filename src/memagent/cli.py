"""memagent CLI.

Commands:
  memagent hook post-tool-use   called by Claude Code PostToolUse hook (reads stdin)
  memagent hook stop            called by Claude Code Stop hook
  memagent consolidate          manually run consolidation for current session
  memagent recall <query>       test retrieval from the command line
  memagent status               show current session and graph stats
  memagent init                 create DB and download embedding model
"""
from __future__ import annotations

import json
import os
import sys
import time


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]

    if cmd == "hook" and len(args) > 1:
        subcmd = args[1]
        if subcmd == "post-tool-use":
            _hook_post_tool_use()
        elif subcmd == "stop":
            _hook_stop()
    elif cmd == "consolidate":
        _cmd_consolidate()
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


# ── Hooks ─────────────────────────────────────────────────────────────────────

TRACKED_TOOLS = {"Write", "Edit", "Bash"}


def _hook_post_tool_use():
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
    except Exception:
        return

    tool_name = data.get("tool_name", "")
    if tool_name not in TRACKED_TOOLS:
        return

    text = _extract_event_text(tool_name, data)
    if not text:
        return

    file_path = _extract_file_path(tool_name, data)

    from .db import init_db
    from .models import Event
    from .store import add_event, get_or_create_session

    init_db()
    session = get_or_create_session(cwd=os.getcwd())
    event = Event.new(
        session_id=session.id,
        text=text,
        tool_name=tool_name,
        file_path=file_path,
    )
    add_event(event)

    # Background embedding — non-blocking
    import threading
    def _embed():
        try:
            from .embedder import embed_one, to_bytes
            from .store import update_event_embedding
            update_event_embedding(event.id, to_bytes(embed_one(text)))
        except Exception:
            pass
    threading.Thread(target=_embed, daemon=True).start()


def _hook_stop():
    try:
        raw = sys.stdin.read()
    except Exception:
        raw = "{}"

    from pathlib import Path
    session_file = Path.home() / ".memagent" / "session.json"
    if not session_file.exists():
        return

    try:
        session_data = json.loads(session_file.read_text())
        session_id = session_data["id"]
    except Exception:
        return

    try:
        from .consolidation import consolidate
        from .store import end_session
        n = consolidate(session_id)
        end_session(session_id)
        print(f"[memagent] Session consolidated: {n} new topic nodes", file=sys.stderr)
    except Exception as e:
        print(f"[memagent] Consolidation error: {e}", file=sys.stderr)


def _extract_event_text(tool_name: str, data: dict) -> str:
    inp = data.get("tool_input", {})

    if tool_name == "Write":
        path = inp.get("file_path", "")
        content = inp.get("content", "")
        preview = content[:400] + ("..." if len(content) > 400 else "")
        return f"Wrote {path}:\n{preview}"

    if tool_name == "Edit":
        path = inp.get("file_path", "")
        old = inp.get("old_string", "")[:150]
        new = inp.get("new_string", "")[:150]
        return f"Edited {path}:\n- {old}\n+ {new}"

    if tool_name == "Bash":
        cmd = inp.get("command", "")[:200]
        response = data.get("tool_response", "")
        if isinstance(response, dict):
            response = response.get("text") or response.get("output") or str(response)
        output = str(response)[:400]
        return f"Ran: {cmd}\n{output}"

    return ""


def _extract_file_path(tool_name: str, data: dict) -> str | None:
    inp = data.get("tool_input", {})
    if tool_name in ("Write", "Edit"):
        return inp.get("file_path")
    return None


# ── Commands ──────────────────────────────────────────────────────────────────

def _cmd_consolidate():
    from pathlib import Path
    session_file = Path.home() / ".memagent" / "session.json"
    if not session_file.exists():
        print("No active session.", file=sys.stderr)
        return
    session_data = json.loads(session_file.read_text())
    session_id = session_data["id"]

    from .consolidation import consolidate
    from .store import end_session
    print(f"Consolidating session {session_id}...", file=sys.stderr)
    n = consolidate(session_id)
    end_session(session_id)
    print(f"Done. {n} new topic nodes created.")


def _cmd_recall(query: str):
    from .retrieval import format_context, recall
    from pathlib import Path

    session_id = None
    session_file = Path.home() / ".memagent" / "session.json"
    if session_file.exists():
        session_id = json.loads(session_file.read_text()).get("id")

    results = recall(query, session_id=session_id)
    print(format_context(results))


def _cmd_status():
    from .db import get_conn, init_db
    init_db()
    with get_conn() as conn:
        n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        n_topics = conn.execute("SELECT COUNT(*) FROM topic_nodes").fetchone()[0]
        n_edges = conn.execute("SELECT COUNT(*) FROM topic_edges").fetchone()[0]
        n_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    from pathlib import Path
    session_file = Path.home() / ".memagent" / "session.json"
    active = "yes" if session_file.exists() else "no"

    print(f"Active session: {active}")
    print(f"Sessions: {n_sessions}")
    print(f"Events: {n_events}")
    print(f"Topic nodes: {n_topics}")
    print(f"Topic edges: {n_edges}")


def _cmd_init():
    from .db import init_db
    init_db()
    print("Database initialised.", file=sys.stderr)
    print("Downloading embedding model (first time only)...", file=sys.stderr)
    from .embedder import embed_one
    embed_one("test")
    print("Ready.")
