import sqlite3

import pytest

from trustnet.agents.mock import MockAgent
from trustnet.config import Settings
from trustnet.decision import STRATEGIES
from trustnet.evaluation.demo import seed_demo
from trustnet.pipeline import (
    build_engine,
    build_evaluator,
    build_policy,
    evaluate_task,
    record_outcome,
)
from trustnet.providers import DeterministicProvider, ProviderUnavailableError
from trustnet.schemas import DecisionOutcome, Outcome, Task
from trustnet.storage import DuplicateOutcomeError, Repository
from trustnet.trust import TrustEngine


@pytest.fixture
def repo(db: sqlite3.Connection) -> Repository:
    return Repository(db)


def test_agents_with_no_history_lead_to_abstention_not_a_guess(repo: Repository, task: Task) -> None:
    agents = [MockAgent(name, recommendation="go") for name in ("analyst", "skeptic", "researcher")]
    evaluation = evaluate_task(task, agents, repo)

    assert evaluation.result.is_complete
    assert tuple(decision.strategy for decision in evaluation.decisions) == STRATEGIES
    assert evaluation.decision.outcome is DecisionOutcome.ABSTAINED
    assert "enough verified history" in evaluation.decision.abstention_reason
    assert all(not profile.has_sufficient_history for profile in evaluation.trust.values())
    # The baselines, which ignore reliability, still answer.
    assert [decision.answer for decision in evaluation.decisions[1:]] == ["[MOCK] go"] * 3


def test_a_run_is_persisted(repo: Repository, task: Task) -> None:
    agents = [MockAgent("analyst"), MockAgent("skeptic")]
    evaluation = evaluate_task(task, agents, repo)

    assert repo.agent_ids() == ["analyst", "skeptic"]
    assert repo.tasks() == {task.task_id: task}
    assert repo.responses(task.task_id) == list(evaluation.result.responses)
    assert repo.decisions(task.task_id) == list(evaluation.decisions)
    assert set(repo.metrics("analyst")) == {None, task.category}


def test_conflicting_answers_are_reported(repo: Repository, task: Task) -> None:
    agents = [MockAgent("a", "go"), MockAgent("b", "stop"), MockAgent("c", "go"), MockAgent("d", None)]
    evaluation = evaluate_task(task, agents, repo)
    assert evaluation.conflicting_answers == {"[MOCK] go": ["a", "c"], "[MOCK] stop": ["b"]}


def test_failed_agents_are_reported_and_nothing_is_invented(repo: Repository, task: Task) -> None:
    down = ProviderUnavailableError("Cannot reach Ollama")
    evaluation = evaluate_task(task, [MockAgent("analyst", error=down)], repo)

    assert evaluation.result.responses == ()
    assert evaluation.result.failures[0].message == "Cannot reach Ollama"
    assert all(d.outcome is DecisionOutcome.ABSTAINED for d in evaluation.decisions)
    assert evaluation.decision.abstention_reason == "no agent responded"
    assert repo.responses() == []


def test_history_gives_agents_measured_trust_and_explained_decisions(repo: Repository) -> None:
    demo = seed_demo(repo, history_size=60, unseen_size=5)
    case = demo.cases[0]
    evaluation = evaluate_task(case.task, demo.agents, repo, evaluator=demo.evaluator)

    assert len(evaluation.trust) == 5
    assert all(profile.trust_score is not None for profile in evaluation.trust.values())
    assert all(response.is_mock for response in evaluation.result.responses)
    assert len(evaluation.decision.contributions) == 5
    assert all(item.weight is not None for item in evaluation.decision.contributions)
    # Outcomes of unseen tasks are not in the database until someone records them.
    assert case.task.task_id not in repo.outcomes()


def test_recording_an_outcome_updates_reliability_once(repo: Repository) -> None:
    demo = seed_demo(repo, history_size=60, unseen_size=5)
    case = next(case for case in demo.cases if case.outcome.answerable)
    before = evaluate_task(case.task, demo.agents, repo, evaluator=demo.evaluator).trust["steady"]

    profiles = record_outcome(repo, case.outcome)
    after = TrustEngine.select(profiles["steady"], case.task.category)

    assert after.resolved_sample_size == before.resolved_sample_size + 1
    with pytest.raises(DuplicateOutcomeError):
        record_outcome(repo, case.outcome)
    with pytest.raises(DuplicateOutcomeError):
        record_outcome(
            repo, Outcome(task_id=case.task.task_id, correct_answer="x", resolution_source="y")
        )


def test_engine_and_policy_are_built_from_settings(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("TRUSTNET_WEIGHT_CALIBRATION", "0")
    clean_env.setenv("TRUSTNET_TRUST_MIN_SAMPLES", "12")
    clean_env.setenv("TRUSTNET_DECISION_MIN_SUPPORT", "0.75")
    settings = Settings(_env_file=None)

    engine, policy = build_engine(settings), build_policy(settings)
    assert engine.min_samples == 12
    assert engine.weights.calibration == 0
    assert engine.weights.historical_accuracy == 0.40
    assert policy.config.min_support == 0.75
    assert policy.config.min_trust == 0.5


def test_llm_evidence_judge_is_opt_in(clean_env: pytest.MonkeyPatch) -> None:
    provider = DeterministicProvider()
    assert build_evaluator(Settings(_env_file=None), provider)._judge is None
    clean_env.setenv("TRUSTNET_EVIDENCE_LLM_JUDGE", "true")
    assert build_evaluator(Settings(_env_file=None), provider)._judge is not None
    assert build_evaluator(Settings(_env_file=None), None)._judge is None
