from typing import Any

import pytest
from pydantic import ValidationError

from trustnet.schemas import (
    AgentResponse,
    Evidence,
    EvidenceAssessment,
    EvidenceSourceType,
    EvidenceStatus,
    Outcome,
    Stance,
    Task,
    TrustSignals,
)
from trustnet.trust import (
    TrustEngine,
    TrustEvaluator,
    TrustWeights,
    brier_score,
    calibration_summary,
    normalize_answer,
    pairwise_agreement,
    wilson_interval,
)


def answer(
    task_id: str,
    recommendation: str | None,
    confidence: float = 0.5,
    agent_id: str = "a1",
    run: int = 1,
    evidence: tuple[Evidence, ...] = (),
) -> AgentResponse:
    return AgentResponse(
        response_id=f"{task_id}:{agent_id}:{run}",
        task_id=task_id,
        agent_id=agent_id,
        stance=Stance.ABSTAIN if recommendation is None else Stance.ANSWER,
        recommendation=recommendation,
        rationale="MOCK",
        self_reported_confidence=confidence,
        evidence=evidence,
        model_id="mock:test",
        is_mock=True,
    )


def outcome(task_id: str, correct: str | None) -> Outcome:
    return Outcome(task_id=task_id, correct_answer=correct, resolution_source="mock-truth")


def history(correct: int, wrong: int, confidence: float = 0.5) -> tuple[list[AgentResponse], dict]:
    """An agent that answered ``correct`` tasks rightly and ``wrong`` tasks wrongly."""
    responses, outcomes = [], {}
    for index in range(correct + wrong):
        task_id = f"t{index}"
        responses.append(answer(task_id, "yes" if index < correct else "no", confidence))
        outcomes[task_id] = outcome(task_id, "yes")
    return responses, outcomes


# --- metrics ----------------------------------------------------------------------------


def test_answers_are_compared_ignoring_case_spacing_and_edge_punctuation() -> None:
    assert normalize_answer("  The  Answer. ") == normalize_answer("the answer")
    assert normalize_answer("4.5") != normalize_answer("45")
    assert normalize_answer(None) is None
    assert normalize_answer(" . ") is None


def test_brier_score_matches_hand_calculation() -> None:
    assert brier_score([(1.0, True), (0.0, False)]) == 0.0
    assert brier_score([(1.0, False)]) == 1.0
    assert brier_score([(0.8, True), (0.6, False)]) == pytest.approx((0.04 + 0.36) / 2)
    assert brier_score([]) is None


def test_calibration_summary_needs_enough_samples() -> None:
    assert calibration_summary([(0.9, True)] * 9, min_samples=10) is None


def test_calibration_summary_separates_calibrated_from_overconfident() -> None:
    calibrated = calibration_summary([(0.8, True)] * 8 + [(0.8, False)] * 2)
    overconfident = calibration_summary([(0.95, True)] * 5 + [(0.95, False)] * 5)

    assert calibrated is not None and overconfident is not None
    assert calibrated.sample_size == 10
    assert calibrated.expected_calibration_error == pytest.approx(0.0)
    assert overconfident.expected_calibration_error == pytest.approx(0.45)
    assert overconfident.brier_score > calibrated.brier_score
    (band,) = overconfident.bins
    assert (band.lower, band.upper, band.count, band.accuracy) == (0.8, 1.0, 10, 0.5)


def test_wilson_interval_is_wide_for_small_samples() -> None:
    small, large = wilson_interval(3, 3), wilson_interval(300, 300)
    assert small is not None and large is not None
    assert small[0] < 0.5 < large[0]
    assert wilson_interval(0, 0) is None


def test_pairwise_agreement() -> None:
    assert pairwise_agreement(["a", "a", "a"]) == 1.0
    assert pairwise_agreement(["a", "b"]) == 0.0
    assert pairwise_agreement(["a", "a", "b"]) == pytest.approx(1 / 3)
    assert pairwise_agreement([None, None]) == 1.0
    assert pairwise_agreement(["a"]) is None


# --- historical accuracy ----------------------------------------------------------------


def test_accuracy_comes_from_verified_outcomes_not_from_confidence() -> None:
    responses, outcomes = history(correct=3, wrong=7, confidence=0.99)
    signals = TrustEngine().evaluate("a1", responses, outcomes)

    assert signals.historical_accuracy == pytest.approx(0.3)
    assert signals.resolved_sample_size == 10
    assert signals.accuracy_lower_bound < 0.3


def test_stated_confidence_alone_never_changes_accuracy() -> None:
    low = TrustEngine().evaluate("a1", *history(6, 4, confidence=0.1))
    high = TrustEngine().evaluate("a1", *history(6, 4, confidence=0.99))
    assert low.historical_accuracy == high.historical_accuracy == pytest.approx(0.6)


def test_responses_without_a_verified_outcome_do_not_count() -> None:
    responses, outcomes = history(correct=5, wrong=0)
    unresolved = [answer(f"open{index}", "no") for index in range(20)]
    signals = TrustEngine().evaluate("a1", responses + unresolved, outcomes)

    assert signals.resolved_sample_size == 5
    assert signals.historical_accuracy == 1.0


def test_other_agents_history_is_ignored() -> None:
    responses, outcomes = history(correct=5, wrong=0)
    others = [answer(f"t{index}", "no", agent_id="a2") for index in range(5)]
    assert TrustEngine().evaluate("a1", responses + others, outcomes).historical_accuracy == 1.0


# --- missing measurements and sample counts ---------------------------------------------


def test_an_agent_with_no_history_has_no_scores_rather_than_zero() -> None:
    signals = TrustEngine().evaluate("newcomer", [], {})

    for name in (
        "historical_accuracy",
        "calibration",
        "evidence_quality",
        "consistency",
        "insufficient_information_behavior",
        "accuracy_lower_bound",
        "brier_score",
        "trust_score",
    ):
        assert getattr(signals, name) is None
    assert signals.resolved_sample_size == 0
    assert not signals.has_sufficient_history


def test_a_genuine_zero_stays_zero_and_distinct_from_missing() -> None:
    signals = TrustEngine().evaluate("a1", *history(correct=0, wrong=6))
    assert signals.historical_accuracy == 0.0
    assert signals.historical_accuracy is not None


def test_trust_score_is_withheld_until_there_is_enough_history() -> None:
    engine = TrustEngine(min_samples=5)
    short = engine.evaluate("a1", *history(correct=4, wrong=0))
    enough = engine.evaluate("a1", *history(correct=5, wrong=0))

    assert short.historical_accuracy == 1.0
    assert not short.has_sufficient_history
    assert short.trust_score is None
    assert enough.has_sufficient_history
    assert enough.trust_score is not None


def test_a_short_perfect_record_scores_below_a_long_one() -> None:
    engine = TrustEngine(min_samples=5)
    short = engine.evaluate("a1", *history(correct=5, wrong=0, confidence=1.0))
    long = engine.evaluate("a1", *history(correct=200, wrong=0, confidence=1.0))
    assert short.historical_accuracy == long.historical_accuracy == 1.0
    assert short.trust_score < long.trust_score


def test_trust_score_cannot_be_set_without_sufficient_history() -> None:
    with pytest.raises(ValidationError, match="sufficient evaluation history"):
        TrustSignals(agent_id="a1", trust_score=0.9)


# --- weights ----------------------------------------------------------------------------


def test_trust_score_is_the_weighted_mean_of_the_measured_signals() -> None:
    responses, outcomes = history(correct=8, wrong=2, confidence=0.8)
    signals = TrustEngine(min_samples=5).evaluate("a1", responses, outcomes)

    # Measured here: accuracy (lower bound), calibration, and evidence (no evidence cited = 0).
    weights = TrustWeights()
    used = weights.historical_accuracy + weights.calibration + weights.evidence_quality
    expected = (
        weights.historical_accuracy * signals.accuracy_lower_bound
        + weights.calibration * signals.calibration
        + weights.evidence_quality * 0.0
    ) / used
    assert signals.consistency is None
    assert signals.insufficient_information_behavior is None
    assert signals.trust_score == pytest.approx(expected)


def test_weights_are_configurable() -> None:
    responses, outcomes = history(correct=8, wrong=2, confidence=0.8)
    accuracy_only = TrustEngine(
        TrustWeights(
            calibration=0, evidence_quality=0, consistency=0, insufficient_information_behavior=0
        )
    ).evaluate("a1", responses, outcomes)
    assert accuracy_only.trust_score == pytest.approx(accuracy_only.accuracy_lower_bound)


@pytest.mark.parametrize("weights", [{"historical_accuracy": 0}, {"calibration": -1}])
def test_invalid_weights_are_rejected(weights: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TrustWeights(**weights)


# --- calibration ------------------------------------------------------------------------


def test_overconfident_agent_is_scored_as_worse_calibrated() -> None:
    engine = TrustEngine()
    honest = engine.evaluate("a1", *history(5, 5, confidence=0.5))
    overconfident = engine.evaluate("a1", *history(5, 5, confidence=0.95))

    assert honest.brier_score == pytest.approx(0.25)
    assert overconfident.brier_score == pytest.approx((5 * 0.05**2 + 5 * 0.95**2) / 10)
    assert honest.calibration > overconfident.calibration
    assert honest.trust_score > overconfident.trust_score


def test_engine_gives_a_calibration_summary_only_with_enough_data() -> None:
    engine = TrustEngine()
    assert engine.calibration("a1", *history(3, 2)) is None
    summary = engine.calibration("a1", *history(8, 4, confidence=0.9))
    assert summary is not None and summary.sample_size == 12


# --- evidence quality -------------------------------------------------------------------


def cited(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.DOCUMENT,
        excerpt="MOCK claim",
        reference="mock-source",
    )


def assessed(evidence_id: str, status: EvidenceStatus) -> EvidenceAssessment:
    return EvidenceAssessment(
        evidence_id=evidence_id,
        status=status,
        source_available=True if status is not EvidenceStatus.UNVERIFIED else None,
    )


def test_evidence_quality_is_the_share_of_checked_evidence_that_was_verified() -> None:
    responses = [
        answer("t1", "yes", evidence=(cited("e1"), cited("e2"))),
        answer("t2", "yes", evidence=(cited("e3"),)),
    ]
    assessments = {
        "e1": assessed("e1", EvidenceStatus.VERIFIED),
        "e2": assessed("e2", EvidenceStatus.CONTRADICTORY),
        "e3": assessed("e3", EvidenceStatus.VERIFIED),
    }
    signals = TrustEngine().evaluate("a1", responses, {}, assessments=assessments)

    assert signals.evidence_quality == pytest.approx((0.5 + 1.0) / 2)
    assert signals.evidence_sample_size == 2


def test_an_answer_citing_nothing_scores_zero_evidence_quality() -> None:
    signals = TrustEngine().evaluate("a1", [answer("t1", "yes")], {})
    assert signals.evidence_quality == 0.0
    assert signals.evidence_sample_size == 1


def test_evidence_that_could_not_be_checked_is_unmeasured_not_zero() -> None:
    responses = [answer("t1", "yes", evidence=(cited("e1"),))]
    unverified = {"e1": assessed("e1", EvidenceStatus.UNVERIFIED)}

    for assessments in ({}, unverified):
        signals = TrustEngine().evaluate("a1", responses, {}, assessments=assessments)
        assert signals.evidence_quality is None
        assert signals.evidence_sample_size == 0


# --- consistency ------------------------------------------------------------------------


def test_consistency_across_repeated_predictions_on_one_task() -> None:
    same = [answer("t1", "yes", run=1), answer("t1", "Yes.", run=2)]
    changed = [answer("t1", "yes", run=1), answer("t1", "no", run=2)]

    assert TrustEngine().evaluate("a1", same, {}).consistency == 1.0
    assert TrustEngine().evaluate("a1", changed, {}).consistency == 0.0


def test_consistency_across_equivalent_prompts() -> None:
    tasks = {
        "t1": Task(task_id="t1", question="MOCK wording one?", equivalence_group="g1"),
        "t2": Task(task_id="t2", question="MOCK wording two?", equivalence_group="g1"),
        "t3": Task(task_id="t3", question="MOCK unrelated?"),
    }
    responses = [answer("t1", "yes"), answer("t2", "no"), answer("t3", "maybe")]
    signals = TrustEngine().evaluate("a1", responses, {}, tasks=tasks)

    assert signals.consistency == 0.0
    assert signals.consistency_sample_size == 1


def test_consistency_is_unmeasured_without_repeats() -> None:
    signals = TrustEngine().evaluate("a1", [answer("t1", "yes"), answer("t2", "no")], {})
    assert signals.consistency is None
    assert signals.consistency_sample_size == 0


# --- behaviour when information is insufficient -----------------------------------------


def test_abstaining_on_unanswerable_tasks_is_rewarded() -> None:
    outcomes = {"u1": outcome("u1", None), "u2": outcome("u2", None), "t1": outcome("t1", "yes")}
    careful = [answer("u1", None), answer("u2", None), answer("t1", "yes")]
    reckless = [answer("u1", "guess"), answer("u2", "guess"), answer("t1", "yes")]
    timid = [answer("u1", None), answer("u2", None), answer("t1", None)]

    def score(responses: list[AgentResponse]) -> float | None:
        return TrustEngine().evaluate("a1", responses, outcomes).insufficient_information_behavior

    assert score(careful) == 1.0
    assert score(reckless) == 0.5
    assert score(timid) == 0.5
    assert TrustEngine().evaluate("a1", careful, outcomes).abstention_sample_size == 2


def test_abstention_behaviour_is_unmeasured_without_unanswerable_tasks() -> None:
    signals = TrustEngine().evaluate("a1", *history(5, 5))
    assert signals.insufficient_information_behavior is None


def test_answering_an_unanswerable_task_does_not_count_as_an_accuracy_sample() -> None:
    outcomes = {"u1": outcome("u1", None)}
    signals = TrustEngine().evaluate("a1", [answer("u1", "guess")], outcomes)
    assert signals.resolved_sample_size == 0
    assert signals.historical_accuracy is None


# --- task-specific profiles -------------------------------------------------------------


def category_history() -> tuple[list[AgentResponse], dict[str, Outcome], dict[str, Task]]:
    responses, outcomes, tasks = [], {}, {}
    for category, right in (("maths", True), ("history", False)):
        for index in range(6):
            task_id = f"{category}{index}"
            tasks[task_id] = Task(task_id=task_id, question="MOCK?", category=category)
            outcomes[task_id] = outcome(task_id, "yes")
            responses.append(answer(task_id, "yes" if right else "no"))
    return responses, outcomes, tasks


def test_profiles_are_kept_per_task_category() -> None:
    responses, outcomes, tasks = category_history()
    profiles = TrustEngine().profiles("a1", responses, outcomes, tasks)

    assert set(profiles) == {None, "maths", "history"}
    assert profiles["maths"].historical_accuracy == 1.0
    assert profiles["history"].historical_accuracy == 0.0
    assert profiles[None].historical_accuracy == 0.5
    assert profiles["maths"].category == "maths"
    assert profiles["maths"].trust_score > profiles["history"].trust_score


def test_select_prefers_the_category_profile_and_falls_back_to_overall() -> None:
    responses, outcomes, tasks = category_history()
    engine = TrustEngine(min_samples=5)
    profiles = engine.profiles("a1", responses, outcomes, tasks)

    assert engine.select(profiles, "maths") is profiles["maths"]
    assert engine.select(profiles, "never-seen") is profiles[None]
    assert engine.select({}, "maths") is None

    thin = TrustEngine(min_samples=10).profiles("a1", responses, outcomes, tasks)
    assert TrustEngine.select(thin, "maths") is thin[None]


def test_engine_satisfies_the_trust_evaluator_interface() -> None:
    assert isinstance(TrustEngine(), TrustEvaluator)
