import pytest

from trustnet.agents import Agent, AgentOutputError
from trustnet.decision import STRATEGIES
from trustnet.evaluation import SYNTHETIC_NOTICE, format_report, run_benchmark
from trustnet.evaluation.synthetic import AGENT_SPECS, build_agents, make_cases
from trustnet.schemas import Task

SIZES = {"development_size": 60, "test_size": 60}


@pytest.fixture(scope="module")
def report():
    return run_benchmark(seed=3, **SIZES)


def test_benchmark_is_reproducible(report) -> None:
    assert run_benchmark(seed=3, **SIZES) == report
    assert run_benchmark(seed=4, **SIZES) != report


def test_all_four_strategies_are_scored_on_the_same_cases(report) -> None:
    assert tuple(row.strategy for row in report.strategies) == STRATEGIES
    assert {row.cases for row in report.strategies} == {report.test_cases}


def test_reported_numbers_are_internally_consistent(report) -> None:
    for row in report.strategies:
        assert row.answered + row.abstained == row.cases
        assert row.accuracy == pytest.approx(row.correct / row.cases)
        assert row.abstention_rate == pytest.approx(row.abstained / row.cases)
        assert row.accuracy_interval[0] <= row.accuracy <= row.accuracy_interval[1]
        assert row.answered_correct <= row.correct
        if row.answered:
            assert row.answered_accuracy == pytest.approx(row.answered_correct / row.answered)
            assert row.answered_interval[0] <= row.answered_accuracy <= row.answered_interval[1]
        else:
            assert row.answered_accuracy is None and row.answered_interval is None


def test_development_and_test_splits_do_not_overlap() -> None:
    development = {case.task.task_id for case in make_cases(60, 3, "dev")}
    test = {case.task.task_id for case in make_cases(60, 3, "test")}
    assert development and test and not development & test


def test_agent_profiles_depend_only_on_the_development_split(report) -> None:
    other_test_split = run_benchmark(seed=3, development_size=60, test_size=20)
    assert other_test_split.agents == report.agents
    assert other_test_split.strategies != report.strategies


def test_fixtures_are_labelled_synthetic(report) -> None:
    cases = make_cases(20, 3, "dev")
    agents = build_agents({case.task.task_id: case.outcome for case in cases}, seed=3)
    responses = [agent.respond(case.task) for case in cases for agent in agents]

    assert all(case.task.is_synthetic and "[SYNTHETIC]" in case.task.question for case in cases)
    assert all(response.is_mock and "[SYNTHETIC]" in response.rationale for response in responses)
    assert all(agent.profile.is_synthetic and isinstance(agent, Agent) for agent in agents)
    assert report.notice == SYNTHETIC_NOTICE
    assert format_report(report).startswith("SYNTHETIC DATA")


def test_fixtures_include_unanswerable_and_reworded_tasks() -> None:
    cases = make_cases(60, 3, "dev")
    assert any(not case.outcome.answerable and case.source is None for case in cases)
    groups = [case.task.equivalence_group for case in cases]
    assert len(groups) > len(set(groups))


def test_agents_on_the_same_model_answer_identically() -> None:
    cases = make_cases(40, 3, "dev")
    agents = {
        agent.spec.agent_id: agent
        for agent in build_agents({case.task.task_id: case.outcome for case in cases}, seed=3)
    }
    assert sum(spec.model == "synthetic-clone" for spec in AGENT_SPECS) == 2
    for case in cases:
        first, second = agents["clone-1"].respond(case.task), agents["clone-2"].respond(case.task)
        assert first.recommendation == second.recommendation
        assert first.model_id == second.model_id


def test_simulated_agents_refuse_tasks_outside_the_benchmark() -> None:
    agent = build_agents({}, seed=3)[0]
    with pytest.raises(AgentOutputError, match="synthetic benchmark tasks"):
        agent.respond(Task(task_id="real", question="A real question?"))


def test_report_shows_sample_counts_and_missing_values(report) -> None:
    text = format_report(report)
    assert f"held-out test cases={report.test_cases}" in text
    assert "95% CI" in text
    for row in report.strategies:
        assert row.strategy in text
