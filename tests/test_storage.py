import sqlite3
from pathlib import Path

import pytest

from trustnet.storage import connect, initialize
from trustnet.storage.schema import MIGRATIONS

EXPECTED_TABLES = {
    "schema_migrations",
    "agents",
    "tasks",
    "agent_responses",
    "evidence_items",
    "outcomes",
    "decisions",
    "evidence_assessments",
    "reliability_metrics",
}


def seed_response(db: sqlite3.Connection) -> None:
    db.execute(
        "INSERT INTO agents (agent_id, display_name, provider, model, is_synthetic) "
        "VALUES ('a1', 'Synthetic Agent', 'fake', 'deterministic-v0', 1)"
    )
    db.execute("INSERT INTO tasks (task_id, question, is_synthetic) VALUES ('t1', 'q?', 1)")
    db.execute(
        "INSERT INTO agent_responses (response_id, task_id, agent_id, stance, answer) "
        "VALUES ('r1', 't1', 'a1', 'answer', '42')"
    )


def test_initialize_creates_all_tables(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert {row["name"] for row in rows} == EXPECTED_TABLES


def test_initialize_is_idempotent(db: sqlite3.Connection) -> None:
    assert initialize(db) == len(MIGRATIONS)
    assert db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == len(MIGRATIONS)


def test_failed_migration_leaves_no_partial_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = "CREATE TABLE half_done (id INTEGER); CREATE TABLE half_done (id INTEGER);"
    monkeypatch.setattr("trustnet.storage.database.MIGRATIONS", (broken,))
    connection = connect(":memory:")
    with pytest.raises(sqlite3.OperationalError):
        initialize(connection)
    tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master")}
    assert tables == {"schema_migrations"}
    assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 0
    connection.close()


def test_connect_creates_parent_directory_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "trustnet.db"
    first = connect(path)
    initialize(first)
    seed_response(first)
    first.commit()
    first.close()

    second = connect(path)
    assert initialize(second) == len(MIGRATIONS)
    assert second.execute("SELECT answer FROM agent_responses").fetchone()["answer"] == "42"
    second.close()


def test_foreign_keys_are_enforced(db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO agent_responses (response_id, task_id, agent_id, stance, answer) "
            "VALUES ('r1', 'missing-task', 'missing-agent', 'answer', '42')"
        )


def test_model_assertion_cannot_be_stored_as_verified(db: sqlite3.Connection) -> None:
    seed_response(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO evidence_items "
            "(evidence_id, response_id, source_type, excerpt, reference, verification_status) "
            "VALUES ('e1', 'r1', 'model_assertion', 'trust me', 'synthetic://x', 'verified')"
        )


def test_verified_evidence_requires_a_reference_in_storage(db: sqlite3.Connection) -> None:
    seed_response(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO evidence_items "
            "(evidence_id, response_id, source_type, excerpt, verification_status) "
            "VALUES ('e1', 'r1', 'document', 'a claim', 'verified')"
        )
    db.execute(
        "INSERT INTO evidence_items "
        "(evidence_id, response_id, source_type, excerpt, reference, verification_status) "
        "VALUES ('e2', 'r1', 'document', 'a claim', 'synthetic://doc', 'verified')"
    )


def test_confidence_out_of_range_is_rejected(db: sqlite3.Connection) -> None:
    seed_response(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO agent_responses "
            "(response_id, task_id, agent_id, stance, answer, self_reported_confidence) "
            "VALUES ('r2', 't1', 'a1', 'answer', '42', 1.5)"
        )


def test_abstention_cannot_be_stored_with_an_answer(db: sqlite3.Connection) -> None:
    seed_response(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO agent_responses (response_id, task_id, agent_id, stance, answer) "
            "VALUES ('r2', 't1', 'a1', 'abstain', '42')"
        )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO decisions (decision_id, task_id, outcome, answer) "
            "VALUES ('d1', 't1', 'abstained', '42')"
        )
