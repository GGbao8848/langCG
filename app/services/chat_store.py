from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = ROOT_DIR / "data" / "langcg.sqlite"


def _database_path() -> Path:
    return Path(os.getenv("LANGCG_DB_PATH", str(DEFAULT_DATABASE_PATH))).expanduser()


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def init_chat_store() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                text TEXT,
                tool_calls TEXT NOT NULL DEFAULT '[]',
                position INTEGER NOT NULL,
                UNIQUE(session_id, position)
            );

            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_position
                ON chat_messages(session_id, position);
            """
        )


def load_chat_state() -> dict[str, Any]:
    init_chat_store()

    with _connect() as connection:
        current_session_id = (
            connection.execute("SELECT value FROM app_state WHERE key = ?", ("current_session_id",)).fetchone()
            or {"value": ""}
        )["value"]
        session_rows = connection.execute(
            "SELECT id, name, updated_at FROM chat_sessions ORDER BY updated_at DESC"
        ).fetchall()

        sessions = []
        for session in session_rows:
            message_rows = connection.execute(
                """
                SELECT id, role, text, tool_calls
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY position ASC
                """,
                (session["id"],),
            ).fetchall()
            messages = []
            for message in message_rows:
                tool_calls = json.loads(message["tool_calls"] or "[]")
                item: dict[str, Any] = {
                    "id": message["id"],
                    "role": message["role"],
                    "text": message["text"] or "",
                }
                if tool_calls:
                    item["toolCalls"] = tool_calls
                messages.append(item)

            sessions.append(
                {
                    "id": session["id"],
                    "name": session["name"],
                    "updatedAt": session["updated_at"],
                    "messages": messages,
                }
            )

    return {
        "sessions": sessions,
        "currentSessionId": current_session_id,
        "savedAt": 0,
    }


def save_chat_state(state: dict[str, Any]) -> None:
    init_chat_store()

    sessions = state.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("sessions must be a list")

    with _connect() as connection:
        connection.execute("BEGIN")
        connection.execute("DELETE FROM chat_messages")
        connection.execute("DELETE FROM chat_sessions")

        for session in sessions:
            session_id = str(session["id"])
            connection.execute(
                "INSERT INTO chat_sessions (id, name, updated_at) VALUES (?, ?, ?)",
                (session_id, str(session["name"]), int(session["updatedAt"])),
            )

            messages = session.get("messages") or []
            for position, message in enumerate(messages):
                connection.execute(
                    """
                    INSERT INTO chat_messages (id, session_id, role, text, tool_calls, position)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(message["id"]),
                        session_id,
                        str(message["role"]),
                        message.get("text") or "",
                        json.dumps(message.get("toolCalls") or [], ensure_ascii=False),
                        position,
                    ),
                )

        connection.execute(
            """
            INSERT INTO app_state (key, value)
            VALUES ('current_session_id', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(state.get("currentSessionId") or ""),),
        )
        connection.commit()


def load_app_state_value(key: str) -> dict[str, Any]:
    init_chat_store()

    with _connect() as connection:
        row = connection.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    if row is None:
        return {}

    try:
        value = json.loads(row["value"])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def save_app_state_value(key: str, value: dict[str, Any]) -> None:
    init_chat_store()

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO app_state (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, json.dumps(value, ensure_ascii=False)),
        )
