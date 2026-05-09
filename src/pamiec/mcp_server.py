"""pamiec MCP server.

Exposes memory tools to Claude so it can recall context autonomously.

Run via:
  pamiec-mcp       (stdio, for Claude Code)
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import Tool as MCPTool


class CompatFastMCP(FastMCP):
    """FastMCP with outputSchema stripped — Claude Code's client doesn't support it."""

    async def list_tools(self) -> list[MCPTool]:
        tools = await super().list_tools()
        return [
            MCPTool(name=t.name, description=t.description, inputSchema=t.inputSchema)
            for t in tools
        ]


mcp = CompatFastMCP("pamiec")


@mcp.tool()
def recall_context(query: str) -> str:
    """Query the knowledge graph for context relevant to the current conversation.

    Use this whenever the conversation touches on people, projects, companies,
    decisions, constraints, or anything that might have prior history. Returns
    only the most relevant nodes — not the full graph.

    The query should reflect what is being discussed right now, not a generic
    description. Bad: "user context". Good: "deployment options for ProjectX".
    """
    from .retrieval import format_context, recall
    return format_context(recall(query))


@mcp.tool()
def remember(text: str, entity_type: str = "fact") -> str:
    """Explicitly store an important fact mid-session.

    Use this only for things that won't appear naturally in the conversation
    transcript and so would be missed by the cron's auto-extraction —
    e.g. a decision made verbally, a constraint, a key contact someone mentioned.
    For everything that's discussed in conversation, do nothing: the cron will
    pick it up.

    The text is treated as a one-turn micro-conversation and processed through
    the same Haiku entity-extraction pipeline as a regular cron consolidation,
    so the same confidence threshold applies.

    entity_type is a hint for the extractor; common values: fact, decision,
    person, project, company, tool, constraint.
    """
    import time
    from .consolidation import consolidate_turns
    from .db import init_db
    from .session_reader import Turn

    init_db()
    turn = Turn(
        role="user",
        text=f"Remember this {entity_type}: {text}",
        timestamp=time.time(),
        iso_ts="",
    )
    result = consolidate_turns([turn], session_file="mcp:remember")
    if result.get("skipped_no_entities"):
        return (
            f"Stored as raw note but no entities extracted "
            f"(text below confidence threshold). "
            f"Try rephrasing with specific named entities."
        )
    return (
        f"Stored. {result['nodes_created']} new entities, "
        f"{result['entities_touched']} touched, "
        f"{result['edges_created']} edges."
    )


@mcp.tool()
def consolidate() -> str:
    """Trigger consolidation now — drain the live EPG buffer, detect topic
    boundaries, promote each segment to an archived episode, and extract
    entities into the long-term graph.

    Use this when freshly-discussed entities should be available for
    recall_context within the same session, instead of waiting up to 30 min
    for the cron's next run. Common cases: end of a working block before
    handing off; right before asking a recall_context question that depends
    on something just discussed; or when the user asks you to flush.

    Side effects: makes Haiku API calls (one per segment). Cheap but not free.
    """
    from .boundaries import split_at_boundaries
    from .consolidation import consolidate_turns
    from .db import init_db
    from .session_reader import Turn
    from .store import delete_epg_turns, get_epg_turns

    init_db()
    rows = get_epg_turns()
    if not rows:
        return "EPG buffer is empty — nothing to consolidate."

    by_session: dict[str, list] = {}
    for r in rows:
        by_session.setdefault(r["session_file"], []).append(r)

    total = {"new": 0, "touched": 0, "edges": 0, "skipped": 0, "episodes": 0}
    for session_file, srows in by_session.items():
        turns = [
            Turn(role=r["role"], text=r["text"], timestamp=r["timestamp"], iso_ts=r["iso_ts"])
            for r in srows
        ]
        segments = split_at_boundaries(turns)
        for segment in segments:
            result = consolidate_turns(segment, session_file=session_file)
            if result.get("skipped_no_entities"):
                total["skipped"] += 1
            else:
                total["episodes"] += 1
            total["new"] += result["nodes_created"]
            total["touched"] += result["entities_touched"]
            total["edges"] += result["edges_created"]
        delete_epg_turns([r["id"] for r in srows])

    return (
        f"Consolidated {len(rows)} EPG turns across {len(by_session)} session(s): "
        f"{total['episodes']} episodes created ({total['skipped']} segments skipped, no entities). "
        f"+{total['new']} new entities, {total['touched']} entities touched, {total['edges']} edges."
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
