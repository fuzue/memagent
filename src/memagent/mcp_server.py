"""memagent MCP server.

Exposes memory tools to Claude so it can recall context autonomously.

Run via:
  memagent-mcp       (stdio, for Claude Code)
"""
from __future__ import annotations

import os
from mcp.server.fastmcp import FastMCP

from .retrieval import format_context, recall
from .store import get_or_create_session

mcp = FastMCP("memagent")


@mcp.tool()
def recall_context(query: str) -> str:
    """Retrieve relevant context from memory for a given query.

    Call this autonomously when you need background context about the user,
    their projects, past decisions, or ongoing work. The query should be a
    natural language description of what you are trying to understand.

    Examples:
      recall_context("Acme company background and location")
      recall_context("current sprint goals for ProjectX")
      recall_context("why was confidence scoring removed")
    """
    session = get_or_create_session(cwd=os.getcwd())
    results = recall(query, session_id=session.id)
    return format_context(results)


@mcp.tool()
def remember(text: str, entity_type: str = "fact") -> str:
    """Explicitly store an important fact, decision, or observation in memory.

    Use this for things that are important enough to remember across sessions:
    user preferences, key decisions, constraints, facts about the project.
    Do NOT use for every action — only for things that would otherwise be lost.

    entity_type: fact | decision | work | problem | solution | person | project
    """
    import time, uuid
    from .models import Event
    from .store import add_event

    session = get_or_create_session(cwd=os.getcwd())
    event = Event(
        id=str(uuid.uuid4()),
        session_id=session.id,
        text=text,
        timestamp=time.time(),
        tool_name="remember",
    )
    add_event(event)
    _queue_embedding(event.id, text)
    return f"Stored: {text[:80]}{'...' if len(text) > 80 else ''}"


def _queue_embedding(event_id: str, text: str) -> None:
    import threading
    def _run():
        try:
            from .embedder import embed_one, to_bytes
            from .store import update_event_embedding
            vec = embed_one(text)
            update_event_embedding(event_id, to_bytes(vec))
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def main():
    import asyncio
    mcp.run()


if __name__ == "__main__":
    main()
