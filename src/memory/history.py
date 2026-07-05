"""Persisted conversation history.

Requirement: remember the context of previous messages within a session and
*persist* the conversation history, keyed by a session id, keeping the last N
messages (N configurable). This uses SQLite (stdlib, zero extra services) so
history survives restarts and is queryable by the analytics module.

- ``get_window(session_id, n)`` returns the last N messages for prompt context.
- Assistant messages carry metadata (retrieval score, confidence, sources,
  latency, out-of-scope flag, model) that the analytics feature aggregates.
- Feedback (thumbs up/down) is stored against a message id.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    session_id: str = ""
    id: Optional[int] = None
    created_at: str = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationStore:
    """SQLite-backed store for conversation history and feedback."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + a lock: FastAPI may touch this from threads.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id INTEGER,
                    rating INTEGER NOT NULL,       -- 1 = up, -1 = down
                    created_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # --- writes ------------------------------------------------------------
    def add_message(
        self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, _now(), json.dumps(metadata or {}, ensure_ascii=False)),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_feedback(self, session_id: str, rating: int, message_id: Optional[int] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO feedback (session_id, message_id, rating, created_at) VALUES (?, ?, ?, ?)",
                (session_id, message_id, 1 if rating >= 0 else -1, _now()),
            )
            self._conn.commit()

    # --- reads -------------------------------------------------------------
    def get_window(self, session_id: str, n: int) -> List[Message]:
        """Return the last N messages for a session in chronological order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (session_id, n),
            ).fetchall()
        return [self._to_message(r) for r in rows]

    def get_session(self, session_id: str) -> List[Message]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
        return [self._to_message(r) for r in rows]

    def list_sessions(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT session_id FROM messages ORDER BY session_id"
            ).fetchall()
        return [r["session_id"] for r in rows]

    def all_messages(self) -> List[Message]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM messages ORDER BY id ASC").fetchall()
        return [self._to_message(r) for r in rows]

    def all_feedback(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM feedback ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _to_message(row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )
