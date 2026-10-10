import sqlite3
from pathlib import Path

import pytest

from trustnet.decision import decide_all
from trustnet.schemas import (
    AgentProfile,
    AgentResponse,
    Evidence,
    EvidenceAssessment,
    EvidenceSourceType,
    EvidenceStatus,
    Outcome,
    Stance,
    Task,
)
from trustnet.storage import DuplicateOutcomeError, Repository, connect, initialize
from trustnet.storage.schema import MIGRATIONS
from trustnet.trust import TrustEngine

AGENT = AgentProfile(
    agent_id="a1", display_name="[MOCK] a1", provider="mock", model="test", is_synthetic=True
)


def make_task(task_id: str, category: str = "general") -> Task:
    return Task(
        task_id=task_id,
        question="MOCK: which option?",
        is_synthetic=True,
        category=category,
        equivalence_group="g1",
        options=("yes", "no"),
    )


def make_response(task_id: str, recommendation: str | None, run: int = 1) -> AgentResponse:
    response_id = f"{task_id}:a1:{run}"
    return AgentResponse(
        response_id=response_id,
        task_id=task_id,
        agent_id="a1",
        stance=Stance.ABSTAIN if recommendation is None else Stance.ANSWER,
        recommendation=recommendation,
        rationale="MOCK rationale",
        self_reported_confidence=0.7,
        evidence=(
            Evidence(
                evidence_id=f"{response_id}:e1",
                source_type=EvidenceSourceType.DOCUMENT,
                excerpt="MOCK claim",
                reference="mock-source",
            ),
        ),
        uncertainties=("MOCK unknown",),
        model_id="mock:test",
        is_mock=True,
        prompt_version="mock-prompt-v1",
    )


@pytest.fixture
def repo(db: sqlite3.Connection) -> Repository:
    repository = Repository(db)
    repository.save_agent(AGENT)
    return repository


def seed(repo: Repository, count: int, correct: int) -> None:
    """``count`` answered tasks, the first ``correct`` of them verified as right."""
    for index in range(count):
        task = make_task(f"t{index}")
        repo.save_task(task)
        repo.save_responses([make_response(task.task_id, "yes")])
        repo.record_outcome(
            Outcome(
                task_id=task.task_id,
                correct_answer="yes" if index < correct else "no",
                resolution_source="mock-truth",
            )
        )


def test_everything_round_trips_through_sqlite(repo: Repository) -> None:
    task = make_task("t1", category="maths")
    response = make_response("t1", "yes")
    assessment = EvidenceAssessment(
        evidence_id=response.evidence[0].evidence_id,
        status=EvidenceStatus.VERIFIED,
        source_available=True,
        relevance=0.9,
        note="MOCK note",
    )
    outcome = Outcome(task_id="t1", correct_answer="yes", resolution_source="mock-truth")

    repo.save_task(task)
    repo.save_responses([response], [assessment])
    repo.record_outcome(outcome)

    assert repo.agent_ids() == ["a1"]
    assert repo.tasks() == {"t1": task}
    assert repo.responses() == [response]
    assert repo.responses(task_id="t1") == [response]
    assert repo.responses(task_id="other") == []
    assert repo.assessments() == {assessment.evidence_id: assessment}
    assert repo.outcomes() == {"t1": outcome}


def test_abstentions_and_unanswerable_outcomes_round_trip(repo: Repository) -> None:
    repo.save_task(make_task("t1"))
    abstention = make_response("t1", None)
    unanswerable = Outcome(task_id="t1", correct_answer=None, resolution_source="mock-truth")
    repo.save_responses([abstention])
    repo.record_outcome(unanswerable)

    assert repo.responses() == [abstention]
    assert repo.outcomes()["t1"] == unanswerable
    assert not repo.outcomes()["t1"].answerable


def test_a_second_outcome_for_the_same_task_is_refused(repo: Repository) -> None:
    repo.save_task(make_task("t1"))
    first = Outcome(task_id="t1", correct_answer="yes", resolution_source="mock-truth")
    repo.record_outcome(first)

    for duplicate in (
        first,
        Outcome(task_id="t1", correct_answer="no", resolution_source="mock-other"),
        Outcome(task_id="t1", correct_answer=None, resolution_source="mock-other"),
    ):
        with pytest.raises(DuplicateOutcomeError, match="already has a verified outcome"):
            repo.record_outcome(duplicate)
    assert repo.outcomes() == {"t1": first}


def test_an_outcome_for_an_unknown_task_is_refused(repo: Repository) -> None:
    with pytest.raises(ValueError, match="unknown task"):
        repo.record_outcome(
            Outcome(task_id="missing", correct_answer="yes", resolution_source="mock-truth")
        )


def test_timestamps_and_model_and_prompt_versions_are_stored(
    repo: Repository, db: sqlite3.Connection
) -> None:
    repo.save_task(make_task("t1"))
    repo.save_responses([make_response("t1", "yes")])
    repo.record_outcome(Outcome(task_id="t1", correct_answer="yes", resolution_source="truth"))

    row = db.execute("SELECT model_id, prompt_version, created_at FROM agent_responses").fetchone()
    assert (row["model_id"], row["prompt_version"]) == ("mock:test", "mock-prompt-v1")
    assert row["created_at"].endswith("Z")
    assert db.execute("SELECT resolved_at FROM outcomes").fetchone()["resolved_at"].endswith("Z")


def test_reliability_is_computed_and_stored_from_verified_outcomes(repo: Repository) -> None:
    seed(repo, count=10, correct=7)
    profiles = repo.refresh_reliability(TrustEngine(min_samples=5))

    overall = profiles["a1"][None]
    assert overall.historical_accuracy == pytest.approx(0.7)
    assert overall.resolved_sample_size == 10
    assert overall.trust_score is not None
    assert repo.metrics("a1") == profiles["a1"]
    assert set(repo.metrics("a1")) == {None, "general"}


def test_reliability_does_not_move_without_a_verified_outcome(repo: Repository) -> None:
    seed(repo, count=6, correct=6)
    before = repo.refresh_reliability(TrustEngine())["a1"][None]

    for index in range(5):
        repo.save_task(make_task(f"open{index}"))
        repo.save_responses([make_response(f"open{index}", "no")])
    after = repo.refresh_reliability(TrustEngine())["a1"][None]

    assert after.historical_accuracy == before.historical_accuracy == 1.0
    assert after.resolved_sample_size == before.resolved_sample_size == 6
    assert after.calibration == before.calibration


def test_recording_an_outcome_updates_reliability(repo: Repository) -> None:
    seed(repo, count=5, correct=5)
    repo.save_task(make_task("late"))
    repo.save_responses([make_response("late", "yes")])
    before = repo.refresh_reliability(TrustEngine())["a1"][None]

    repo.record_outcome(Outcome(task_id="late", correct_answer="no", resolution_source="truth"))
    after = repo.refresh_reliability(TrustEngine())["a1"][None]

    assert (before.resolved_sample_size, after.resolved_sample_size) == (5, 6)
    assert after.historical_accuracy == pytest.approx(5 / 6)


def test_repeated_predictions_are_kept_and_feed_consistency(repo: Repository) -> None:
    repo.save_task(make_task("t1"))
    repo.save_responses([make_response("t1", "yes", run=1)])
    repo.save_responses([make_response("t1", "no", run=2)])

    assert len(repo.responses()) == 2
    assert repo.refresh_reliability(TrustEngine())["a1"][None].consistency == 0.0


def test_decisions_round_trip_with_their_explanations(repo: Repository) -> None:
    seed(repo, count=6, correct=6)
    task = make_task("new")
    response = make_response("new", "yes")
    repo.save_task(task)
    repo.save_responses([response])
    trust = {"a1": repo.refresh_reliability(TrustEngine())["a1"][None]}

    decisions = decide_all(task, [response], trust)
    repo.save_decisions(decisions)
    repo.save_decisions(decisions)  # saving again replaces rather than duplicates

    stored = repo.decisions("new")
    assert stored == list(decisions)
    assert stored[0].contributions[0].trust_score == trust["a1"].trust_score


def test_a_failed_write_leaves_nothing_behind(repo: Repository) -> None:
    repo.save_task(make_task("t1"))
    good, orphan = make_response("t1", "yes"), make_response("no-such-task", "yes")
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_responses([good, orphan])
    assert repo.responses() == []


def test_level_one_database_upgrades_without_losing_outcomes(tmp_path: Path) -> None:
    connection = connect(tmp_path / "old.db")
    connection.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, "
        "applied_at TEXT NOT NULL DEFAULT 'then')"
    )
    connection.executescript(MIGRATIONS[0])
    connection.execute("INSERT INTO schema_migrations (version) VALUES (1)")
    connection.execute("INSERT INTO tasks (task_id, question) VALUES ('old', 'MOCK?')")
    connection.execute(
        "INSERT INTO outcomes (task_id, correct_answer, resolution_source) "
        "VALUES ('old', 'yes', 'mock-truth')"
    )
    connection.commit()

    assert initialize(connection) == len(MIGRATIONS)
    repository = Repository(connection)
    assert repository.outcomes()["old"].correct_answer == "yes"
    assert repository.tasks()["old"].category == "general"
    connection.close()
