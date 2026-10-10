"""A reproducible SYNTHETIC benchmark comparing the four decision strategies.

Reliability profiles are built only from the development split. The strategies are
then scored on a held-out test split they have never seen, using identical agent
responses for all four. Thresholds are the library defaults; nothing is tuned on the
test split.

A case counts as correct when the strategy decides on the verified answer, or abstains
on a task verified as unanswerable.
"""

from collections.abc import Sequence

from trustnet.decision.strategies import STRATEGIES, TrustWeightedPolicy, decide_all
from trustnet.evaluation.synthetic import (
    AGENT_SPECS,
    SYNTHETIC_NOTICE,
    Case,
    build_agents,
    make_cases,
)
from trustnet.evidence.evaluator import EvidenceEvaluator, InMemorySourceStore
from trustnet.schemas import (
    AgentResponse,
    DecisionOutcome,
    EvidenceAssessment,
    TrustNetModel,
    TrustSignals,
)
from trustnet.trust.engine import TrustEngine
from trustnet.trust.metrics import brier_score, normalize_answer, wilson_interval

Interval = tuple[float, float]


class StrategyResult(TrustNetModel):
    strategy: str
    cases: int
    correct: int
    accuracy: float
    accuracy_interval: Interval
    abstained: int
    abstention_rate: float
    answered: int
    answered_correct: int
    # None when the strategy answered nothing.
    answered_accuracy: float | None
    answered_interval: Interval | None
    # Brier score of the strategy's support share on the cases it answered.
    brier_score: float | None


class BenchmarkReport(TrustNetModel):
    notice: str
    seed: int
    development_cases: int
    test_cases: int
    test_unanswerable: int
    # Overall profile of each simulated agent, measured on the development split only.
    agents: tuple[TrustSignals, ...]
    strategies: tuple[StrategyResult, ...]


def _simulate(
    cases: Sequence[Case], seed: int, evaluator: EvidenceEvaluator
) -> tuple[list[AgentResponse], dict[str, EvidenceAssessment]]:
    agents = build_agents({case.task.task_id: case.outcome for case in cases}, seed)
    responses = [agent.respond(case.task) for case in cases for agent in agents]
    assessments = {
        item.evidence_id: item
        for response in responses
        for item in evaluator.assess_response(response)
    }
    return responses, assessments


def run_benchmark(
    seed: int = 7,
    development_size: int = 300,
    test_size: int = 300,
    engine: TrustEngine | None = None,
    policy: TrustWeightedPolicy | None = None,
) -> BenchmarkReport:
    engine = engine or TrustEngine()
    development = make_cases(development_size, seed, "dev")
    test = make_cases(test_size, seed, "test")
    evaluator = EvidenceEvaluator(
        InMemorySourceStore(dict(case.source for case in development + test if case.source))
    )

    # Development split: the only data reliability is learned from.
    history, history_assessments = _simulate(development, seed, evaluator)
    tasks = {case.task.task_id: case.task for case in development}
    outcomes = {case.task.task_id: case.outcome for case in development}
    profiles = {
        spec.agent_id: engine.profiles(spec.agent_id, history, outcomes, tasks, history_assessments)
        for spec in AGENT_SPECS
    }

    # Held-out test split: outcomes are used only to score the decisions.
    responses, assessments = _simulate(test, seed, evaluator)
    by_task: dict[str, list[AgentResponse]] = {}
    for response in responses:
        by_task.setdefault(response.task_id, []).append(response)

    tallies = {name: {"correct": 0, "abstained": 0, "answered_correct": 0} for name in STRATEGIES}
    forecasts: dict[str, list[tuple[float, bool]]] = {name: [] for name in STRATEGIES}
    for case in test:
        trust = {
            agent_id: profile
            for agent_id, by_category in profiles.items()
            if (profile := engine.select(by_category, case.task.category)) is not None
        }
        decisions = decide_all(case.task, by_task[case.task.task_id], trust, assessments, policy)
        for decision in decisions:
            tally = tallies[decision.strategy]
            if decision.outcome is DecisionOutcome.ABSTAINED:
                tally["abstained"] += 1
                tally["correct"] += not case.outcome.answerable
                continue
            right = case.outcome.answerable and normalize_answer(
                decision.answer
            ) == normalize_answer(case.outcome.correct_answer)
            tally["correct"] += right
            tally["answered_correct"] += right
            if decision.support is not None:
                forecasts[decision.strategy].append((decision.support, right))

    total = len(test)
    results = []
    for name in STRATEGIES:
        tally = tallies[name]
        answered = total - tally["abstained"]
        results.append(
            StrategyResult(
                strategy=name,
                cases=total,
                correct=tally["correct"],
                accuracy=tally["correct"] / total,
                accuracy_interval=wilson_interval(tally["correct"], total) or (0.0, 1.0),
                abstained=tally["abstained"],
                abstention_rate=tally["abstained"] / total,
                answered=answered,
                answered_correct=tally["answered_correct"],
                answered_accuracy=tally["answered_correct"] / answered if answered else None,
                answered_interval=wilson_interval(tally["answered_correct"], answered),
                brier_score=brier_score(forecasts[name]),
            )
        )
    return BenchmarkReport(
        notice=SYNTHETIC_NOTICE,
        seed=seed,
        development_cases=len(development),
        test_cases=total,
        test_unanswerable=sum(not case.outcome.answerable for case in test),
        agents=tuple(by_category[None] for by_category in profiles.values()),
        strategies=tuple(results),
    )


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_report(report: BenchmarkReport) -> str:
    lines = [
        report.notice,
        f"seed={report.seed}  development cases={report.development_cases}  "
        f"held-out test cases={report.test_cases} "
        f"({report.test_unanswerable} unanswerable)",
        "",
        "Agent profiles (development split only)",
        f"{'agent':<14}{'n':>5}{'accuracy':>10}{'lower95':>9}{'calib':>8}{'evidence':>10}"
        f"{'consist':>9}{'abstain':>9}{'trust':>8}",
    ]
    for agent in report.agents:
        lines.append(
            f"{agent.agent_id:<14}{agent.resolved_sample_size:>5}"
            f"{_number(agent.historical_accuracy):>10}{_number(agent.accuracy_lower_bound):>9}"
            f"{_number(agent.calibration):>8}{_number(agent.evidence_quality):>10}"
            f"{_number(agent.consistency):>9}"
            f"{_number(agent.insufficient_information_behavior):>9}{_number(agent.trust_score):>8}"
        )
    lines += [
        "",
        "Strategies on the held-out test split",
        f"{'strategy':<20}{'n':>5}{'accuracy':>10}{'95% CI':>16}{'abstain':>9}"
        f"{'answered':>10}{'ans.acc':>9}{'95% CI':>16}{'brier':>8}",
    ]
    for row in report.strategies:
        interval = f"[{row.accuracy_interval[0]:.3f},{row.accuracy_interval[1]:.3f}]"
        answered = (
            f"[{row.answered_interval[0]:.3f},{row.answered_interval[1]:.3f}]"
            if row.answered_interval
            else "n/a"
        )
        lines.append(
            f"{row.strategy:<20}{row.cases:>5}{row.accuracy:>10.3f}{interval:>16}"
            f"{row.abstention_rate:>9.3f}{row.answered:>10}{_number(row.answered_accuracy):>9}"
            f"{answered:>16}{_number(row.brier_score):>8}"
        )
    return "\n".join(lines)
