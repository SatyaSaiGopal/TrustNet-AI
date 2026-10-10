import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from trustnet.schemas import (
    MAX_QUESTION_LENGTH,
    AgentFailure,
    AgentProfile,
    AgentResponse,
    Decision,
    DecisionOutcome,
    Evidence,
    EvidenceSourceType,
    Outcome,
    Stance,
    Task,
    TrustSignals,
    VerificationStatus,
    WorkflowResult,
)

SCENARIO = (
    Path(__file__).resolve().parent.parent / "examples" / "synthetic" / "SYNTHETIC_scenario.json"
)


def make_response(**overrides: Any) -> AgentResponse:
    fields: dict[str, Any] = {
        "response_id": "r1",
        "task_id": "t1",
        "agent_id": "a1",
        "stance": Stance.ANSWER,
        "recommendation": "42",
        "rationale": "MOCK rationale",
        "self_reported_confidence": 0.5,
        "model_id": "mock:deterministic-v0",
        "is_mock": True,
    }
    fields.update(overrides)
    return AgentResponse(**fields)


def test_model_assertion_cannot_be_verified() -> None:
    with pytest.raises(ValidationError, match="model assertion"):
        Evidence(
            evidence_id="e1",
            source_type=EvidenceSourceType.MODEL_ASSERTION,
            excerpt="trust me",
            reference="https://example.invalid/anything",
            verification_status=VerificationStatus.VERIFIED,
        )


def test_verified_evidence_requires_a_reference() -> None:
    with pytest.raises(ValidationError, match="reference"):
        Evidence(
            evidence_id="e1",
            source_type=EvidenceSourceType.DOCUMENT,
            excerpt="a claim",
            verification_status=VerificationStatus.VERIFIED,
        )


def test_evidence_is_unverified_by_default() -> None:
    evidence = Evidence(
        evidence_id="e1",
        source_type=EvidenceSourceType.DOCUMENT,
        excerpt="a claim",
        reference="synthetic://doc",
    )
    assert evidence.verification_status is VerificationStatus.UNVERIFIED
    assert not evidence.is_verified


def test_agent_response_carries_every_required_field() -> None:
    evidence = Evidence(
        evidence_id="e1", source_type=EvidenceSourceType.MODEL_ASSERTION, excerpt="a claim"
    )
    response = make_response(evidence=(evidence,), uncertainties=("unknown thing",))
    assert response.agent_id == "a1"
    assert response.task_id == "t1"
    assert response.recommendation == "42"
    assert response.rationale == "MOCK rationale"
    assert response.self_reported_confidence == 0.5
    assert response.evidence == (evidence,)
    assert response.uncertainties == ("unknown thing",)
    assert response.model_id == "mock:deterministic-v0"


@pytest.mark.parametrize(
    "missing",
    ["agent_id", "task_id", "rationale", "self_reported_confidence", "model_id"],
)
def test_agent_response_requires_its_fields(missing: str) -> None:
    fields = make_response().model_dump()
    del fields[missing]
    with pytest.raises(ValidationError, match=missing):
        AgentResponse(**fields)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf"), "high", None])
def test_invalid_confidence_is_rejected(confidence: object) -> None:
    with pytest.raises(ValidationError, match="self_reported_confidence"):
        make_response(self_reported_confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_boundary_confidence_is_accepted(confidence: float) -> None:
    assert make_response(self_reported_confidence=confidence).self_reported_confidence == confidence


def test_abstaining_response_cannot_carry_a_recommendation() -> None:
    with pytest.raises(ValidationError, match="abstaining"):
        make_response(stance=Stance.ABSTAIN)
    assert make_response(stance=Stance.ABSTAIN, recommendation=None).recommendation is None


@pytest.mark.parametrize("recommendation", [None, "", "   "])
def test_answering_response_requires_a_recommendation(recommendation: str | None) -> None:
    with pytest.raises(ValidationError, match="answering"):
        make_response(recommendation=recommendation)


def test_mock_output_cannot_pass_as_real_and_real_cannot_be_called_mock() -> None:
    with pytest.raises(ValidationError, match="is_mock"):
        make_response(model_id="mock:deterministic-v0", is_mock=False)
    with pytest.raises(ValidationError, match="is_mock"):
        make_response(model_id="ollama:some-model", is_mock=True)
    assert not make_response(model_id="ollama:some-model", is_mock=False).is_mock


@pytest.mark.parametrize("question", ["", "   ", "\n\t", "x" * (MAX_QUESTION_LENGTH + 1)])
def test_task_rejects_blank_or_oversized_questions(question: str) -> None:
    with pytest.raises(ValidationError):
        Task(task_id="t1", question=question)


def test_workflow_result_reports_missing_agents() -> None:
    task = Task(task_id="t1", question="q?")
    failure = AgentFailure(agent_id="a2", task_id="t1", error_type="ProviderError", message="down")
    result = WorkflowResult(
        task=task,
        expected_agent_ids=("a1", "a2"),
        responses=(make_response(),),
        failures=(failure,),
    )
    assert result.missing_agent_ids == ("a2",)
    assert not result.is_complete
    assert result.model_dump()["missing_agent_ids"] == ("a2",)


@pytest.mark.parametrize(
    "responses, reason",
    [
        ((make_response(task_id="other"),), "different task"),
        ((make_response(agent_id="stranger"),), "unexpected agent"),
        ((make_response(), make_response(response_id="r2")), "more than one"),
    ],
)
def test_workflow_result_rejects_responses_that_do_not_belong(
    responses: tuple[AgentResponse, ...], reason: str
) -> None:
    with pytest.raises(ValidationError, match=reason):
        WorkflowResult(
            task=Task(task_id="t1", question="q?"),
            expected_agent_ids=("a1", "a2"),
            responses=responses,
        )


@pytest.mark.parametrize("signal", ["historical_accuracy", "calibration"])
def test_outcome_based_signals_require_resolved_outcomes(signal: str) -> None:
    with pytest.raises(ValidationError, match="resolved outcome"):
        TrustSignals(agent_id="a1", **{signal: 0.9})
    assert getattr(TrustSignals(agent_id="a1", resolved_sample_size=3, **{signal: 0.9}), signal) == 0.9


def test_trust_signals_default_to_unknown_not_zero() -> None:
    signals = TrustSignals(agent_id="a1")
    assert signals.historical_accuracy is None
    assert signals.evidence_quality is None
    assert signals.consistency is None
    assert signals.calibration is None
    assert signals.insufficient_information_behavior is None


def test_abstained_decision_cannot_carry_an_answer() -> None:
    with pytest.raises(ValidationError, match="abstained"):
        Decision(decision_id="d1", task_id="t1", outcome=DecisionOutcome.ABSTAINED, answer="42")
    abstained = Decision(decision_id="d1", task_id="t1", outcome=DecisionOutcome.ABSTAINED)
    assert abstained.answer is None


def test_models_are_immutable_and_reject_unknown_fields() -> None:
    task = Task(task_id="t1", question="q?")
    with pytest.raises(ValidationError):
        task.question = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Task(task_id="t1", question="q?", historical_accuracy=1.0)  # type: ignore[call-arg]


def test_synthetic_scenario_is_valid_and_marked_synthetic() -> None:
    scenario = json.loads(SCENARIO.read_text(encoding="utf-8"))
    assert scenario["synthetic"] is True
    assert "SYNTHETIC" in scenario["_notice"]

    agents = [AgentProfile.model_validate(item) for item in scenario["agents"]]
    tasks = [Task.model_validate(item) for item in scenario["tasks"]]
    responses = [AgentResponse.model_validate(item) for item in scenario["responses"]]
    outcomes = [Outcome.model_validate(item) for item in scenario["outcomes"]]

    assert agents and all(agent.is_synthetic for agent in agents)
    assert tasks and all(task.is_synthetic for task in tasks)
    assert responses and all(response.is_mock for response in responses)
    assert {response.agent_id for response in responses} <= {agent.agent_id for agent in agents}
    assert {outcome.task_id for outcome in outcomes} <= {task.task_id for task in tasks}
    assert not any(item.is_verified for response in responses for item in response.evidence)
