"""Read Claude Code session JSONL files into structured turn lists."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Turn:
    role: str          # "user" or "assistant"
    text: str
    timestamp: float   # unix epoch
    iso_ts: str        # original ISO timestamp string


def find_active_session_file() -> Optional[Path]:
    """Return the most recently modified Claude Code session JSONL across all projects."""
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        return None
    candidates = list(claude_dir.rglob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_turns_since(session_file: Path, since_iso: Optional[str]) -> list[Turn]:
    """Read user+assistant turns after since_iso. Returns structured Turn objects."""
    turns: list[Turn] = []

    with session_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            iso_ts = entry.get("timestamp", "")
            if since_iso and iso_ts and iso_ts <= since_iso:
                continue

            role = entry.get("type")
            if role not in ("user", "assistant"):
                continue

            msg = entry.get("message", {})
            content = msg.get("content", "")

            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts = [
                    block.get("text", "").strip()
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                text = "\n".join(p for p in parts if p)
            else:
                continue

            if not text:
                continue

            try:
                ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = 0.0

            turns.append(Turn(role=role, text=text[:2000], timestamp=ts, iso_ts=iso_ts))

    return turns


def turns_to_transcript(turns: list[Turn]) -> str:
    return "\n\n".join(
        f"{'User' if t.role == 'user' else 'Claude'}: {t.text}"
        for t in turns
    )
