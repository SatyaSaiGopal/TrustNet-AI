"""Shared domain schemas.

These models are the contract between the agent, trust, decision, storage, API and
UI modules. Three project rules are enforced here rather than left to convention:

* An agent's self-reported confidence is a claim made by the agent. It lives on
  ``AgentResponse`` and is never a measure of accuracy. Accuracy lives on
  ``TrustSignals`` and can only exist when resolved outcomes back it.
* A model's own assertion is never verified evidence. ``Evidence`` rejects that
  combination, and anything marked verified must cite a reference.
* Mock output is always labelled. A response is flagged ``is_mock`` exactly when
  its ``model_id`` names the mock provider.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
NonEmptyStr = Annotated[str, Field(min_length=1)]

MAX_QUESTION_LENGTH = 10_000
MOCK_PROVIDER_NAME = "mock"


def is_mock_model_id(model_id: str) -> bool:
    return model_id.startswith(f"{MOCK_PROVIDER_NAME}:")


class TrustNetModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Stance(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


class EvidenceSourceType(StrEnum):
    DOCUMENT = "document"
    DATASET = "dataset"
    TOOL_OUTPUT = "tool_output"
    # The agent's own unsupported statement. Recorded, but can never be verified.
    MODEL_ASSERTION = "model_assertion"


class VerificationStatus(StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REFUTED = "refuted"


class EvidenceStatus(StrEnum):
    """The result of checking one evidence item against its source."""

    VERIFIED = "verified"  # the source was read and supports the claim
    UNSUPPORTED = "unsupported"  # no reference, or the source does not support the claim
    CONTRADICTORY = "contradictory"  # the source was read and contradicts the claim
    UNVERIFIED = "unverified"  # a reference was given but could not be checked


class DecisionOutcome(StrEnum):
    DECIDED = "decided"
    ABSTAINED = "abstained"


class AgentProfile(TrustNetModel):
    agent_id: NonEmptyStr
    display_name: NonEmptyStr
    provider: NonEmptyStr
    model: NonEmptyStr
    is_synthetic: bool = False


class Task(TrustNetModel):
    task_id: NonEmptyStr
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_LENGTH)]
    is_synthetic: bool = False
    # Reliability is tracked per category as well as overall.
    category: NonEmptyStr = "general"
    # Tasks asking the same thing in different words share a group (consistency checks).
    equivalence_group: str | None = None
    # When given, agents are asked to answer with exactly one of these.
    options: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _question_is_not_blank(self) -> Self:
        if not self.question.strip():
            raise ValueError("question must contain text, not only whitespace")
        return self


class Evidence(TrustNetModel):
    evidence_id: NonEmptyStr
    source_type: EvidenceSourceType
    excerpt: NonEmptyStr
    reference: str | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    @model_validator(mode="after")
    def _verified_evidence_must_be_checkable(self) -> Self:
        if self.verification_status is VerificationStatus.VERIFIED:
            if self.source_type is EvidenceSourceType.MODEL_ASSERTION:
                raise ValueError("a model assertion cannot be marked as verified evidence")
            if not self.reference:
                raise ValueError("verified evidence must cite a reference")
        return self

    @property
    def is_verified(self) -> bool:
        return self.verification_status is VerificationStatus.VERIFIED


class AgentResponse(TrustNetModel):
    response_id: NonEmptyStr
    task_id: NonEmptyStr
    agent_id: NonEmptyStr
    stance: Stance
    recommendation: str | None = None
    rationale: NonEmptyStr
    # The agent's own claim about itself; not evidence of accuracy.
    self_reported_confidence: UnitInterval
    # Evidence references the agent cited. Unverified unless something else checks them.
    evidence: tuple[Evidence, ...] = ()
    uncertainties: tuple[NonEmptyStr, ...] = ()
    # "<provider>:<model>" that produced this response.
    model_id: NonEmptyStr
    is_mock: bool = False
    prompt_version: str | None = None

    @model_validator(mode="after")
    def _recommendation_matches_stance(self) -> Self:
        has_recommendation = bool(self.recommendation and self.recommendation.strip())
        if self.stance is Stance.ABSTAIN and self.recommendation is not None:
            raise ValueError("an abstaining response must not carry a recommendation")
        if self.stance is Stance.ANSWER and not has_recommendation:
            raise ValueError("an answering response must carry a recommendation")
        return self

    @model_validator(mode="after")
    def _mock_output_is_labelled(self) -> Self:
        if self.is_mock != is_mock_model_id(self.model_id):
            raise ValueError(
                f"is_mock must be true exactly when model_id starts with '{MOCK_PROVIDER_NAME}:'"
            )
        return self


class AgentFailure(TrustNetModel):
    """An agent that produced no usable response, and why. Never replaced by a stand-in."""

    agent_id: NonEmptyStr
    task_id: NonEmptyStr
    error_type: NonEmptyStr
    message: NonEmptyStr


class WorkflowResult(TrustNetModel):
    """The collected, uncombined outputs of one workflow run."""

    task: Task
    expected_agent_ids: tuple[NonEmptyStr, ...]
    responses: tuple[AgentResponse, ...] = ()
    failures: tuple[AgentFailure, ...] = ()

    @model_validator(mode="after")
    def _responses_belong_to_this_run(self) -> Self:
        seen: set[str] = set()
        for response in self.responses:
            if response.task_id != self.task.task_id:
                raise ValueError(f"response {response.response_id} is for a different task")
            if response.agent_id not in self.expected_agent_ids:
                raise ValueError(f"response from unexpected agent '{response.agent_id}'")
            if response.agent_id in seen:
                raise ValueError(f"more than one response from agent '{response.agent_id}'")
            seen.add(response.agent_id)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def missing_agent_ids(self) -> tuple[str, ...]:
        responded = {response.agent_id for response in self.responses}
        return tuple(agent_id for agent_id in self.expected_agent_ids if agent_id not in responded)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_complete(self) -> bool:
        return not self.missing_agent_ids


class EvidenceAssessment(TrustNetModel):
    evidence_id: NonEmptyStr
    status: EvidenceStatus
    # None means availability was not checked (for example, a remote URL).
    source_available: bool | None = None
    # Share of the claim's terms found in the best-matching part of the source.
    relevance: UnitInterval | None = None
    method: Literal["deterministic", "llm"] = "deterministic"
    note: str = ""

    @model_validator(mode="after")
    def _definite_verdicts_need_a_source(self) -> Self:
        definite = (EvidenceStatus.VERIFIED, EvidenceStatus.CONTRADICTORY)
        if self.status in definite and self.source_available is not True:
            raise ValueError(f"'{self.status}' requires a source that was actually read")
        return self


class Outcome(TrustNetModel):
    """The verified ground truth for a task, from a source outside the agents."""

    task_id: NonEmptyStr
    # None records that the task could not be answered from the information available,
    # so abstaining was the right response.
    correct_answer: NonEmptyStr | None
    resolution_source: NonEmptyStr

    @property
    def answerable(self) -> bool:
        return self.correct_answer is not None


class CalibrationBin(TrustNetModel):
    lower: UnitInterval
    upper: UnitInterval
    count: int = Field(ge=1)
    mean_confidence: UnitInterval
    accuracy: UnitInterval


class CalibrationSummary(TrustNetModel):
    sample_size: int = Field(ge=1)
    brier_score: UnitInterval
    expected_calibration_error: UnitInterval
    mean_confidence: UnitInterval
    accuracy: UnitInterval
    bins: tuple[CalibrationBin, ...]


class TrustSignals(TrustNetModel):
    """The five trust signals for one agent. ``None`` means not enough data to say."""

    agent_id: NonEmptyStr
    # None is the profile across all task categories.
    category: str | None = None
    historical_accuracy: UnitInterval | None = None
    evidence_quality: UnitInterval | None = None
    consistency: UnitInterval | None = None
    calibration: UnitInterval | None = None
    insufficient_information_behavior: UnitInterval | None = None
    # Answers given to tasks that have a verified, answerable Outcome.
    resolved_sample_size: int = Field(default=0, ge=0)
    # Wilson 95% lower bound of historical_accuracy; shrinks with small samples.
    accuracy_lower_bound: UnitInterval | None = None
    brier_score: UnitInterval | None = None
    evidence_sample_size: int = Field(default=0, ge=0)
    consistency_sample_size: int = Field(default=0, ge=0)
    abstention_sample_size: int = Field(default=0, ge=0)
    has_sufficient_history: bool = False
    # Only set when has_sufficient_history; never a default or placeholder value.
    trust_score: UnitInterval | None = None

    @model_validator(mode="after")
    def _outcome_based_signals_need_outcomes(self) -> Self:
        if self.resolved_sample_size == 0:
            for name in ("historical_accuracy", "calibration", "brier_score"):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} requires at least one resolved outcome")
        if self.trust_score is not None and not self.has_sufficient_history:
            raise ValueError("trust_score requires sufficient evaluation history")
        return self


class AgentContribution(TrustNetModel):
    """How one agent's response counted towards a decision."""

    agent_id: NonEmptyStr
    recommendation: str | None
    # None means the agent had no validated reliability and was not counted.
    weight: float | None = Field(default=None, ge=0)
    trust_score: UnitInterval | None = None
    evidence_status: str = ""
    note: str = ""


class Decision(TrustNetModel):
    decision_id: NonEmptyStr
    task_id: NonEmptyStr
    outcome: DecisionOutcome
    answer: str | None = None
    rationale: str = ""
    strategy: NonEmptyStr = "unspecified"
    abstention_reason: str | None = None
    # Share of the counted weight (or votes) behind the leading answer.
    support: UnitInterval | None = None
    contributions: tuple[AgentContribution, ...] = ()

    @model_validator(mode="after")
    def _answer_matches_outcome(self) -> Self:
        if self.outcome is DecisionOutcome.ABSTAINED and self.answer is not None:
            raise ValueError("an abstained decision must not carry an answer")
        if self.outcome is DecisionOutcome.DECIDED and not self.answer:
            raise ValueError("a decided decision must carry an answer")
        return self
