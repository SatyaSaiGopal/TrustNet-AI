"""SQLite connections and schema migration."""

import sqlite3
from pathlib import Path

from trustnet.storage.schema import MIGRATIONS

IN_MEMORY = ":memory:"


def connect(path: Path | str, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced and rows addressable by column name.

    Pass ``check_same_thread=False`` only where one connection is reused by a host that
    runs each request on a different thread, one at a time (the Streamlit dashboard).
    """
    if str(path) != IN_MEMORY:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize(connection: sqlite3.Connection) -> int:
    """Apply any migrations not yet applied. Returns the resulting schema version."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, "
        "applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))"
    )
    connection.commit()
    current = connection.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
    ).fetchone()[0]

    for version, script in enumerate(MIGRATIONS, start=1):
        if version <= current:
            continue
        try:
            connection.executescript(
                f"BEGIN;\n{script}\n"
                f"INSERT INTO schema_migrations (version) VALUES ({version});\n"
                "COMMIT;"
            )
        except sqlite3.Error:
            connection.rollback()
            raise
    return len(MIGRATIONS)
