import pytest

from trustnet.decision import (
    STRATEGIES,
    DecisionConfig,
    DecisionPolicy,
    TrustWeightedPolicy,
    answer_groups,
    decide_all,
    equal_weight_vote,
    highest_confidence,
    majority_vote,
)
from trustnet.schemas import (
    AgentResponse,
    DecisionOutcome,
    Evidence,
    EvidenceAssessment,
    EvidenceSourceType,
    EvidenceStatus,
    Stance,
    Task,
    TrustSignals,
)

TASK = Task(task_id="t1", question="MOCK: which option?", is_synthetic=True)


def response(
    agent_id: str,
    recommendation: str | None,
    confidence: float = 0.5,
    model: str | None = None,
    evidence_id: str | None = None,
) -> AgentResponse:
    return AgentResponse(
        response_id=f"r-{agent_id}",
        task_id="t1",
        agent_id=agent_id,
        stance=Stance.ABSTAIN if recommendation is None else Stance.ANSWER,
        recommendation=recommendation,
        rationale="MOCK",
        self_reported_confidence=confidence,
        evidence=(
            Evidence(
                evidence_id=evidence_id,
                source_type=EvidenceSourceType.DOCUMENT,
                excerpt="MOCK claim",
                reference="mock-source",
            ),
        )
        if evidence_id
        else (),
        model_id=f"mock:{model or agent_id}",
        is_mock=True,
    )


def trusted(agent_id: str, score: float | None, samples: int = 50) -> TrustSignals:
    return TrustSignals(
        agent_id=agent_id,
        historical_accuracy=score,
        resolved_sample_size=samples if score is not None else 0,
        has_sufficient_history=score is not None,
        trust_score=score,
    )


def trust(**scores: float | None) -> dict[str, TrustSignals]:
    return {agent_id: trusted(agent_id, score) for agent_id, score in scores.items()}


def decide(responses: list[AgentResponse], scores: dict[str, TrustSignals], **config: object):
    return TrustWeightedPolicy(DecisionConfig(**config)).decide(TASK, responses, scores)


# --- trust-weighted: conflicting agents -------------------------------------------------


def test_reliable_agent_outweighs_two_unreliable_ones() -> None:
    responses = [response("a", "X"), response("b", "Y"), response("c", "Y")]
    decision = decide(responses, trust(a=0.9, b=0.2, c=0.2))

    assert decision.outcome is DecisionOutcome.DECIDED
    assert decision.answer == "X"
    assert decision.support == pytest.approx(0.9 / 1.3)
    assert majority_vote(TASK, responses).answer == "Y"


def test_highest_trust_agent_is_not_followed_blindly() -> None:
    responses = [response("top", "X"), response("b", "Y"), response("c", "Y")]
    decision = decide(responses, trust(top=0.8, b=0.7, c=0.7))

    assert decision.answer == "Y"
    assert decision.support == pytest.approx(1.4 / 2.2)


def test_unresolved_disagreement_leads_to_abstention() -> None:
    responses = [response("a", "X"), response("b", "Y")]
    decision = decide(responses, trust(a=0.8, b=0.7))

    assert decision.outcome is DecisionOutcome.ABSTAINED
    assert decision.answer is None
    assert "disagreement unresolved" in decision.abstention_reason
    assert decision.support == pytest.approx(0.8 / 1.5)


def test_an_exact_tie_leads_to_abstention() -> None:
    decision = decide([response("a", "X"), response("b", "Y")], trust(a=0.8, b=0.8))
    assert decision.outcome is DecisionOutcome.ABSTAINED


def test_answers_that_differ_only_in_formatting_are_the_same_answer() -> None:
    responses = [response("a", "Option X."), response("b", "  option x")]
    assert len(answer_groups(responses)) == 1
    assert decide(responses, trust(a=0.8, b=0.7)).support == 1.0


# --- trust-weighted: missing metrics and unreliable agents ------------------------------


def test_agents_without_validated_reliability_are_not_counted() -> None:
    responses = [response("known", "X"), response("new1", "Y"), response("new2", "Y")]
    decision = decide(responses, trust(known=0.8, new1=None))

    assert decision.answer == "X"
    by_agent = {item.agent_id: item for item in decision.contributions}
    assert by_agent["known"].weight == pytest.approx(0.8)
    for newcomer in ("new1", "new2"):
        assert by_agent[newcomer].weight is None
        assert by_agent[newcomer].trust_score is None
        assert "not counted" in by_agent[newcomer].note


def test_abstains_when_no_agent_has_validated_reliability() -> None:
    responses = [response("a", "X"), response("b", "X"), response("c", "X")]
    decision = decide(responses, trust(a=None))

    assert decision.outcome is DecisionOutcome.ABSTAINED
    assert "enough verified history" in decision.abstention_reason
    assert all(item.weight is None for item in decision.contributions)
    assert majority_vote(TASK, responses).answer == "X"


def test_abstains_when_every_supporter_of_the_leading_answer_is_unreliable() -> None:
    responses = [response("a", "X"), response("b", "X")]
    decision = decide(responses, trust(a=0.3, b=0.2))

    assert decision.outcome is DecisionOutcome.ABSTAINED
    assert "reliable enough" in decision.abstention_reason
    assert decide(responses, trust(a=0.3, b=0.2), min_trust=0.25).answer == "X"


def test_abstains_when_no_agent_responded() -> None:
    assert decide([], {}).abstention_reason == "no agent responded"


# --- trust-weighted: abstaining agents --------------------------------------------------


def test_reliable_agents_abstaining_stop_a_decision() -> None:
    responses = [response("a", None), response("b", None), response("c", "X")]
    decision = decide(responses, trust(a=0.9, b=0.8, c=0.6))

    assert decision.outcome is DecisionOutcome.ABSTAINED
    assert "insufficient" in decision.abstention_reason


def test_an_unreliable_abstainer_does_not_block_reliable_agreement() -> None:
    responses = [response("a", "X"), response("b", "X"), response("c", None)]
    decision = decide(responses, trust(a=0.9, b=0.8, c=0.2))

    assert decision.answer == "X"
    assert decision.support == pytest.approx(1.7 / 1.9)


# --- trust-weighted: correlated agents --------------------------------------------------


def test_agents_on_the_same_model_share_one_weight() -> None:
    clones = [response(f"clone{i}", "Y", model="shared") for i in range(3)]
    responses = [response("independent", "X"), *clones]
    scores = trust(independent=0.8, clone0=0.7, clone1=0.7, clone2=0.7)

    decision = decide(responses, scores, min_support=0.51)

    # Three clones count as one 0.7 vote for Y, against 0.8 for X.
    assert decision.answer == "X"
    assert decision.support == pytest.approx(0.8 / 1.5)
    by_agent = {item.agent_id: item for item in decision.contributions}
    assert by_agent["clone0"].weight == pytest.approx(0.7 / 3)
    assert "same model" in by_agent["clone0"].note
    assert equal_weight_vote(TASK, responses).answer == "Y"


def test_correlated_agents_cannot_manufacture_a_consensus() -> None:
    clones = [response(f"clone{i}", "Y", model="shared") for i in range(3)]
    responses = [response("independent", "X"), *clones]
    scores = trust(independent=0.8, clone0=0.7, clone1=0.7, clone2=0.7)

    # Majority vote sees 3 of 4 agents agreeing; the trust engine sees an open question.
    assert majority_vote(TASK, responses).answer == "Y"
    assert decide(responses, scores).outcome is DecisionOutcome.ABSTAINED


def test_same_model_agents_that_disagree_keep_their_own_weight() -> None:
    responses = [response("a", "X", model="shared"), response("b", "Y", model="shared")]
    decision = decide(responses, trust(a=0.9, b=0.3))
    assert {item.agent_id: item.weight for item in decision.contributions} == {"a": 0.9, "b": 0.3}


# --- trust-weighted: evidence -----------------------------------------------------------


def assessment(evidence_id: str, status: EvidenceStatus) -> EvidenceAssessment:
    return EvidenceAssessment(evidence_id=evidence_id, status=status, source_available=True)


def test_a_response_contradicted_by_its_own_source_loses_weight() -> None:
    responses = [response("a", "X", evidence_id="ea"), response("b", "Y", evidence_id="eb")]
    assessments = {
        "ea": assessment("ea", EvidenceStatus.CONTRADICTORY),
        "eb": assessment("eb", EvidenceStatus.VERIFIED),
    }
    decision = TrustWeightedPolicy().decide(TASK, responses, trust(a=0.9, b=0.6), assessments)

    assert decision.answer == "Y"
    by_agent = {item.agent_id: item for item in decision.contributions}
    assert by_agent["a"].weight == pytest.approx(0.9 * 0.25)
    assert by_agent["a"].evidence_status == "contradicted"
    assert by_agent["b"].evidence_status == "verified"


def test_can_require_verified_evidence_before_deciding() -> None:
    responses = [response("a", "X", evidence_id="ea"), response("b", "X")]
    scores = trust(a=0.9, b=0.8)
    policy = TrustWeightedPolicy(DecisionConfig(require_verified_evidence=True))
    unsupported = {"ea": assessment("ea", EvidenceStatus.UNSUPPORTED)}
    verified = {"ea": assessment("ea", EvidenceStatus.VERIFIED)}

    refused = policy.decide(TASK, responses, scores, unsupported)
    assert refused.outcome is DecisionOutcome.ABSTAINED
    assert "no verified evidence" in refused.abstention_reason
    assert policy.decide(TASK, responses, scores, verified).answer == "X"
    assert TrustWeightedPolicy().decide(TASK, responses, scores, unsupported).answer == "X"


def test_every_agents_contribution_is_explained() -> None:
    responses = [response("a", "X", evidence_id="ea"), response("b", None), response("c", "Y")]
    decision = TrustWeightedPolicy().decide(
        TASK, responses, trust(a=0.9, b=0.1), {"ea": assessment("ea", EvidenceStatus.VERIFIED)}
    )

    assert [item.agent_id for item in decision.contributions] == ["a", "b", "c"]
    a, b, c = decision.contributions
    assert (a.recommendation, a.trust_score, a.evidence_status) == ("X", 0.9, "verified")
    assert (b.recommendation, b.evidence_status) == (None, "none cited")
    assert c.weight is None
    assert "a" in decision.rationale and "%" in decision.rationale


# --- baselines --------------------------------------------------------------------------


def test_majority_vote_needs_more_than_half_of_all_agents() -> None:
    split = [response("a", "X"), response("b", "X"), response("c", "Y"), response("d", None)]
    clear = [response("a", "X"), response("b", "X"), response("c", "Y")]

    assert majority_vote(TASK, split).outcome is DecisionOutcome.ABSTAINED
    assert majority_vote(TASK, clear).answer == "X"
    assert majority_vote(TASK, clear).support == pytest.approx(2 / 3)


def test_equal_weight_vote_takes_the_plurality_of_answers_and_abstains_on_a_tie() -> None:
    plurality = [response("a", "X"), response("b", "X"), response("c", "Y"), response("d", None)]
    tie = [response("a", "X"), response("b", "Y")]

    assert equal_weight_vote(TASK, plurality).answer == "X"
    assert equal_weight_vote(TASK, tie).outcome is DecisionOutcome.ABSTAINED


def test_highest_confidence_follows_the_most_confident_agent() -> None:
    responses = [response("a", "X", 0.6), response("b", "Y", 0.99), response("c", "X", 0.7)]
    decision = highest_confidence(TASK, responses)
    assert decision.answer == "Y"
    assert "b" in decision.rationale


def test_highest_confidence_abstains_when_different_answers_tie() -> None:
    tie = [response("a", "X", 0.9), response("b", "Y", 0.9)]
    agree = [response("a", "X", 0.9), response("b", "X", 0.9)]
    assert highest_confidence(TASK, tie).outcome is DecisionOutcome.ABSTAINED
    assert highest_confidence(TASK, agree).answer == "X"


@pytest.mark.parametrize("strategy", [majority_vote, equal_weight_vote, highest_confidence])
def test_baselines_abstain_when_nobody_answers(strategy) -> None:
    for responses in ([], [response("a", None), response("b", None)]):
        decision = strategy(TASK, responses)
        assert decision.outcome is DecisionOutcome.ABSTAINED
        assert decision.abstention_reason == "no agent gave an answer"


def test_decide_all_runs_the_four_strategies_on_identical_inputs() -> None:
    responses = [response("a", "X", 0.4), response("b", "Y", 0.9), response("c", "Y", 0.5)]
    decisions = decide_all(TASK, responses, trust(a=0.9, b=0.2, c=0.2))

    assert tuple(decision.strategy for decision in decisions) == STRATEGIES
    assert [decision.answer for decision in decisions] == ["X", "Y", "Y", "Y"]
    assert len({decision.decision_id for decision in decisions}) == 4


def test_policy_satisfies_the_decision_policy_interface() -> None:
    assert isinstance(TrustWeightedPolicy(), DecisionPolicy)
