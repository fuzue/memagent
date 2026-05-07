from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time
import uuid


@dataclass
class Event:
    id: str
    session_id: str
    text: str
    timestamp: float
    tool_name: Optional[str] = None
    file_path: Optional[str] = None
    embedding: Optional[bytes] = None
    consolidated: bool = False

    @staticmethod
    def new(session_id: str, text: str, tool_name: str = None, file_path: str = None) -> "Event":
        return Event(
            id=str(uuid.uuid4()),
            session_id=session_id,
            text=text,
            timestamp=time.time(),
            tool_name=tool_name,
            file_path=file_path,
        )


@dataclass
class TopicNode:
    id: str
    csum: str
    craw: str
    entity_type: str
    created_at: float
    updated_at: float
    embedding: Optional[bytes] = None

    @staticmethod
    def new(csum: str, craw: str, entity_type: str = "fact") -> "TopicNode":
        now = time.time()
        return TopicNode(
            id=str(uuid.uuid4()),
            csum=csum,
            craw=craw,
            entity_type=entity_type,
            created_at=now,
            updated_at=now,
        )


@dataclass
class Session:
    id: str
    started_at: float
    cwd: Optional[str] = None
    ended_at: Optional[float] = None

    @staticmethod
    def new(cwd: str = None) -> "Session":
        return Session(
            id=str(uuid.uuid4()),
            started_at=time.time(),
            cwd=cwd,
        )
