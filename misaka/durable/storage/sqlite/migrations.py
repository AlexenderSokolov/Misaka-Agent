"""The ordered schema history of the durable SQLite storage.

Translated from pi's ``packages/durable/src/storage/sqlite/migrations.ts``. A migration is ``{"version": int,
"statements": [sql, ...]}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from .database import SqliteDatabase, SqliteExecutor


class SqliteMigration(TypedDict):
    version: int
    statements: Sequence[str]


# next_id is TEXT because node:sqlite rejects INTEGER results outside JavaScript's safe integer range. The column
# stays TEXT here so the file keeps the upstream layout.
INITIAL_SCHEMA: Sequence[str] = [
    """CREATE TABLE durable_metadata (
		singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
		next_id TEXT NOT NULL,
		next_seq INTEGER NOT NULL
	) STRICT""",
    "INSERT INTO durable_metadata (singleton, next_id, next_seq) VALUES (1, '2', 1)",
    """CREATE TABLE record_ids (
		id INTEGER PRIMARY KEY,
		record_type TEXT NOT NULL CHECK (record_type IN ('conversation', 'entry', 'task', 'submission', 'document'))
	) STRICT""",
    """CREATE TABLE conversations (
		id INTEGER PRIMARY KEY,
		owner_conversation_id INTEGER,
		owner_task_id INTEGER,
		record TEXT NOT NULL CHECK (json_valid(record))
	) STRICT""",
    "CREATE INDEX conversations_by_owner_conversation ON conversations (owner_conversation_id, id)",
    "CREATE INDEX conversations_by_owner_task ON conversations (owner_task_id, id)",
    """CREATE TABLE entries (
		id INTEGER PRIMARY KEY,
		conversation_id INTEGER NOT NULL,
		head INTEGER,
		commit_seq INTEGER NOT NULL,
		record TEXT NOT NULL CHECK (json_valid(record))
	) STRICT""",
    "CREATE INDEX entries_by_conversation ON entries (conversation_id, id DESC)",
    "CREATE INDEX entry_heads_by_conversation ON entries (conversation_id, id DESC) WHERE head IS NOT NULL",
    """CREATE TABLE tasks (
		id INTEGER PRIMARY KEY,
		conversation_id INTEGER NOT NULL,
		kind TEXT NOT NULL,
		status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'waiting', 'completing', 'terminal')),
		abort_requested INTEGER NOT NULL CHECK (abort_requested IN (0, 1)),
		background INTEGER NOT NULL CHECK (background IN (0, 1)),
		record TEXT NOT NULL CHECK (json_valid(record))
	) STRICT""",
    "CREATE INDEX tasks_by_status ON tasks (status, id)",
    "CREATE INDEX tasks_by_conversation ON tasks (conversation_id, id)",
    "CREATE INDEX tasks_by_kind ON tasks (kind, id)",
    "CREATE INDEX tasks_by_abort_requested ON tasks (abort_requested, id)",
    "CREATE INDEX tasks_by_background ON tasks (background, id)",
    """CREATE TABLE submissions (
		id INTEGER PRIMARY KEY,
		conversation_id INTEGER NOT NULL,
		request_id TEXT,
		status TEXT NOT NULL CHECK (status IN ('queued', 'placed', 'done', 'unanswered')),
		record TEXT NOT NULL CHECK (json_valid(record))
	) STRICT""",
    "CREATE INDEX submissions_by_request ON submissions (conversation_id, request_id)",
    "CREATE INDEX submissions_by_conversation ON submissions (conversation_id, id)",
    "CREATE INDEX submissions_by_status ON submissions (status, id)",
    """CREATE TABLE documents (
		id INTEGER PRIMARY KEY,
		kind TEXT NOT NULL,
		family INTEGER NOT NULL CHECK (family IN (0, 1)),
		key_value TEXT NOT NULL,
		scope_kind TEXT NOT NULL CHECK (scope_kind IN ('session', 'conversation', 'task')),
		owner_id INTEGER NOT NULL,
		created_at INTEGER NOT NULL,
		retired_at INTEGER,
		record TEXT NOT NULL CHECK (json_valid(record))
	) STRICT""",
    """CREATE INDEX documents_by_address
		ON documents (kind, scope_kind, owner_id, family, key_value, created_at DESC, retired_at)""",
    "CREATE INDEX documents_by_scope ON documents (scope_kind, owner_id, id)",
    "CREATE INDEX documents_by_scope_kind ON documents (scope_kind, owner_id, kind, id)",
    """CREATE TABLE document_revisions (
		document_id INTEGER NOT NULL,
		seq INTEGER NOT NULL,
		kind TEXT NOT NULL CHECK (kind IN ('base', 'delta')),
		version INTEGER NOT NULL,
		content TEXT NOT NULL CHECK (json_valid(content)),
		PRIMARY KEY (document_id, seq)
	) STRICT""",
    "CREATE INDEX document_revisions_by_kind ON document_revisions (document_id, kind, seq DESC)",
]

SQLITE_MIGRATIONS: Sequence[SqliteMigration] = [{"version": 1, "statements": INITIAL_SCHEMA}]
"""Immutable, ordered schema history. Append new migrations after the initial schema ships."""

CURRENT_SQLITE_SCHEMA_VERSION: int = SQLITE_MIGRATIONS[-1]["version"] if SQLITE_MIGRATIONS else 0


async def applySqliteMigrations(database: SqliteDatabase, migrations: Sequence[SqliteMigration] | None = None) -> None:
    """Apply all pending schema migrations atomically."""
    if migrations is None:
        migrations = SQLITE_MIGRATIONS
    for index, migration in enumerate(migrations):
        if migration["version"] != index + 1:
            raise RuntimeError("Durable SQLite migrations must have contiguous versions starting at 1")

    async def migrate(transaction: SqliteExecutor) -> None:
        await transaction.exec("""CREATE TABLE IF NOT EXISTS durable_schema (
			singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
			version INTEGER NOT NULL CHECK (version >= 0)
		) STRICT""")
        await transaction.run("INSERT OR IGNORE INTO durable_schema (singleton, version) VALUES (1, 0)")
        row = await transaction.get("SELECT version FROM durable_schema WHERE singleton = 1")
        if row is None:
            raise RuntimeError("Durable SQLite schema metadata is missing")
        currentVersion = migrations[-1]["version"] if migrations else 0
        if row["version"] > currentVersion:
            raise RuntimeError(
                f"Durable SQLite schema version {row['version']} is newer than supported version {currentVersion}"
            )
        for migration in migrations:
            if migration["version"] <= row["version"]:
                continue
            for statement in migration["statements"]:
                await transaction.exec(statement)
            await transaction.run("UPDATE durable_schema SET version = ? WHERE singleton = 1", migration["version"])

    await database.transaction(migrate)


__all__ = [
    "CURRENT_SQLITE_SCHEMA_VERSION",
    "INITIAL_SCHEMA",
    "SQLITE_MIGRATIONS",
    "SqliteMigration",
    "applySqliteMigrations",
]
