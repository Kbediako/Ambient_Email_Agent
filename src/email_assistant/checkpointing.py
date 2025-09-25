"""Helpers for LangGraph checkpoint + store configuration.

Centralising checkpointer creation keeps all agents aligned with the
LangGraph 0.6 guidance around Sqlite connection lifecycle management.
"""

from __future__ import annotations

import atexit
import logging
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

# Default filenames when env vars are not provided. Stored under the user's
# home directory so CLI runs and tests share a predictable location while
# remaining easy to clean up.
_DEFAULT_CHECKPOINT_FILENAME = "email_assistant_checkpoints.sqlite"
_DEFAULT_STORE_FILENAME = "email_assistant_store.sqlite"
_DEFAULT_SQLITE_TIMEOUT_SECONDS = 30.0


logger = logging.getLogger(__name__)


def _resolve_path(env_value: Optional[str], fallback_filename: str) -> Path:
    """Return an absolute path for SQLite artefacts, creating folders as needed."""

    if env_value:
        path = Path(env_value).expanduser()
    else:
        path = Path.home() / ".langgraph" / fallback_filename

    if not path.is_absolute():
        path = Path.cwd() / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_timeout_seconds() -> float:
    """Parse the SQLite timeout env var, falling back when invalid."""

    raw_value = os.getenv("EMAIL_ASSISTANT_SQLITE_TIMEOUT")
    if raw_value is None:
        return _DEFAULT_SQLITE_TIMEOUT_SECONDS

    try:
        timeout = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid EMAIL_ASSISTANT_SQLITE_TIMEOUT=%r; falling back to %.1fs",
            raw_value,
            _DEFAULT_SQLITE_TIMEOUT_SECONDS,
        )
        return _DEFAULT_SQLITE_TIMEOUT_SECONDS

    if timeout <= 0:
        logger.warning(
            "Non-positive EMAIL_ASSISTANT_SQLITE_TIMEOUT=%r; falling back to %.1fs",
            raw_value,
            _DEFAULT_SQLITE_TIMEOUT_SECONDS,
        )
        return _DEFAULT_SQLITE_TIMEOUT_SECONDS

    return timeout


@lru_cache(maxsize=4)
def get_sqlite_checkpointer(path: Optional[str] = None) -> SqliteSaver:
    """Return a cached SqliteSaver configured for the requested path."""

    resolved_path = _resolve_path(path or os.getenv("EMAIL_ASSISTANT_CHECKPOINT_PATH"), _DEFAULT_CHECKPOINT_FILENAME)
    timeout_seconds = _resolve_timeout_seconds()
    conn = sqlite3.connect(
        str(resolved_path),
        check_same_thread=False,
        timeout=timeout_seconds,
    )
    # Improve concurrency for parallel pytest runs / LangSmith judges.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout = %d" % int(timeout_seconds * 1000))
    atexit.register(conn.close)
    return SqliteSaver(conn)


@lru_cache(maxsize=4)
def get_sqlite_store(path: Optional[str] = None) -> SqliteStore:
    """Return a cached SqliteStore (with schema ensured) for the requested path."""

    resolved_path = _resolve_path(path or os.getenv("EMAIL_ASSISTANT_STORE_PATH"), _DEFAULT_STORE_FILENAME)
    timeout_seconds = _resolve_timeout_seconds()
    conn = sqlite3.connect(
        str(resolved_path),
        check_same_thread=False,
        timeout=timeout_seconds,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout = %d" % int(timeout_seconds * 1000))
    atexit.register(conn.close)
    store = SqliteStore(conn)
    store.setup()
    return store


def new_memory_checkpointer() -> MemorySaver:
    """Lightweight helper for tests that need an in-memory checkpointer."""

    return MemorySaver()


def new_memory_store() -> InMemoryStore:
    """Return an isolated in-memory store instance."""

    return InMemoryStore()
