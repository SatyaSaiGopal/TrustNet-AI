"""Turns an agent's recorded history into trust signals and a trust score.

How each signal is measured:

* historical_accuracy: share of the agent's answers that match a verified outcome.
  Self-reported confidence plays no part.
* evidence_quality: per answer, the share of checked evidence items found ``verified``.
  An answer citing nothing scores 0; an answer whose items could not be checked is
  left out rather than scored.
* consistency: agreement between the agent's answers to the same task or to tasks in
  the same equivalence group (share of matching answer pairs, averaged over groups).
* calibration: 1 minus the Brier score of stated confidence against verified outcomes.
* insufficient_information_behavior: mean of the abstention rate on tasks verified as
  unanswerable and the answer rate on answerable ones. Needs an unanswerable task.

The trust score is the weighted mean of the signals that could be measured, using the
Wilson lower bound in place of raw accuracy so that short histories count for less.
It is ``None`` until the agent has ``min_samples`` verified answers.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trustnet.schemas import (
    AgentResponse,
    CalibrationSummary,
    EvidenceAssessment,
    EvidenceStatus,
    Outcome,
    Stance,
    Task,
    TrustSignals,
)
from trustnet.trust.metrics import (
    brier_score,
    calibration_summary,
    is_correct,
    normalize_answer,
    pairwise_agreement,
    wilson_interval,
)

_DEFINITE = (EvidenceStatus.VERIFIED, EvidenceStatus.UNSUPPORTED, EvidenceStatus.CONTRADICTORY)


class TrustWeights(BaseModel):
    """Relative importance of each signal. Normalised over the signals that exist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    historical_accuracy: float = Field(default=0.40, ge=0)
    calibration: float = Field(default=0.20, ge=0)
    evidence_quality: float = Field(default=0.15, ge=0)
    consistency: float = Field(default=0.15, ge=0)
    insufficient_information_behavior: float = Field(default=0.10, ge=0)

    @model_validator(mode="after")
    def _accuracy_must_count(self) -> "TrustWeights":
        if self.historical_accuracy <= 0:
            raise ValueError("historical_accuracy must have a positive weight")
        return self


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


class TrustEngine:
    def __init__(self, weights: TrustWeights | None = None, min_samples: int = 5) -> None:
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        self.weights = weights or TrustWeights()
        self.min_samples = min_samples

    def evaluate(
        self,
        agent_id: str,
        responses: Sequence[AgentResponse],
        outcomes: Mapping[str, Outcome],
        tasks: Mapping[str, Task] | None = None,
        assessments: Mapping[str, EvidenceAssessment] | None = None,
        category: str | None = None,
    ) -> TrustSignals:
        tasks = tasks or {}
        assessments = assessments or {}
        mine = [
            response
            for response in responses
            if response.agent_id == agent_id
            and (category is None or self._category(response, tasks) == category)
        ]
        predictions = self._predictions(mine, outcomes)
        correct = sum(was_correct for _, was_correct in predictions)
        interval = wilson_interval(correct, len(predictions))
        brier = brier_score(predictions)
        abstention, abstention_samples = self._abstention(mine, outcomes)
        evidence_scores = self._evidence_scores(mine, assessments)
        consistency_scores = self._consistency_scores(mine, tasks)

        signals = {
            "historical_accuracy": correct / len(predictions) if predictions else None,
            "calibration": None if brier is None else 1 - brier,
            "evidence_quality": _mean(evidence_scores),
            "consistency": _mean(consistency_scores),
            "insufficient_information_behavior": abstention,
        }
        sufficient = len(predictions) >= self.min_samples
        lower_bound = interval[0] if interval else None
        return TrustSignals(
            agent_id=agent_id,
            category=category,
            **signals,
            resolved_sample_size=len(predictions),
            accuracy_lower_bound=lower_bound,
            brier_score=brier,
            evidence_sample_size=len(evidence_scores),
            consistency_sample_size=len(consistency_scores),
            abstention_sample_size=abstention_samples,
            has_sufficient_history=sufficient,
            trust_score=self._score({**signals, "historical_accuracy": lower_bound})
            if sufficient
            else None,
        )

    def profiles(
        self,
        agent_id: str,
        responses: Sequence[AgentResponse],
        outcomes: Mapping[str, Outcome],
        tasks: Mapping[str, Task],
        assessments: Mapping[str, EvidenceAssessment] | None = None,
    ) -> dict[str | None, TrustSignals]:
        """The overall profile (key ``None``) and one per task category the agent has seen."""
        categories = {
            tasks[response.task_id].category
            for response in responses
            if response.agent_id == agent_id and response.task_id in tasks
        }
        return {
            category: self.evaluate(agent_id, responses, outcomes, tasks, assessments, category)
            for category in (None, *sorted(categories))
        }

    def calibration(
        self,
        agent_id: str,
        responses: Sequence[AgentResponse],
        outcomes: Mapping[str, Outcome],
        min_samples: int = 10,
    ) -> CalibrationSummary | None:
        mine = [response for response in responses if response.agent_id == agent_id]
        return calibration_summary(self._predictions(mine, outcomes), min_samples=min_samples)

    @staticmethod
    def select(profiles: Mapping[str | None, TrustSignals], category: str) -> TrustSignals | None:
        """The category profile when it has enough history, otherwise the overall one."""
        specific = profiles.get(category)
        if specific is not None and specific.has_sufficient_history:
            return specific
        return profiles.get(None)

    @staticmethod
    def _category(response: AgentResponse, tasks: Mapping[str, Task]) -> str | None:
        task = tasks.get(response.task_id)
        return task.category if task else None

    @staticmethod
    def _predictions(
        responses: Sequence[AgentResponse], outcomes: Mapping[str, Outcome]
    ) -> list[tuple[float, bool]]:
        """(stated confidence, was correct) for each answer to a verified answerable task."""
        return [
            (response.self_reported_confidence, is_correct(response, outcomes[response.task_id]))
            for response in responses
            if response.stance is Stance.ANSWER
            and response.task_id in outcomes
            and outcomes[response.task_id].answerable
        ]

    @staticmethod
    def _abstention(
        responses: Sequence[AgentResponse], outcomes: Mapping[str, Outcome]
    ) -> tuple[float | None, int]:
        resolved = [response for response in responses if response.task_id in outcomes]
        abstained_when_unanswerable = [
            float(response.stance is Stance.ABSTAIN)
            for response in resolved
            if not outcomes[response.task_id].answerable
        ]
        if not abstained_when_unanswerable:
            return None, 0
        answered_when_answerable = [
            float(response.stance is Stance.ANSWER)
            for response in resolved
            if outcomes[response.task_id].answerable
        ]
        rates = [_mean(abstained_when_unanswerable), _mean(answered_when_answerable)]
        return _mean([rate for rate in rates if rate is not None]), len(abstained_when_unanswerable)

    @staticmethod
    def _evidence_scores(
        responses: Sequence[AgentResponse], assessments: Mapping[str, EvidenceAssessment]
    ) -> list[float]:
        scores = []
        for response in responses:
            if response.stance is not Stance.ANSWER:
                continue
            if not response.evidence:
                scores.append(0.0)
                continue
            statuses = [
                assessments[item.evidence_id].status
                for item in response.evidence
                if item.evidence_id in assessments
                and assessments[item.evidence_id].status in _DEFINITE
            ]
            if statuses:
                verified = sum(status is EvidenceStatus.VERIFIED for status in statuses)
                scores.append(verified / len(statuses))
        return scores

    @staticmethod
    def _consistency_scores(
        responses: Sequence[AgentResponse], tasks: Mapping[str, Task]
    ) -> list[float]:
        groups: dict[str, list[str | None]] = defaultdict(list)
        for response in responses:
            task = tasks.get(response.task_id)
            key = (task.equivalence_group if task else None) or response.task_id
            groups[key].append(normalize_answer(response.recommendation))
        scores = (pairwise_agreement(answers) for answers in groups.values())
        return [score for score in scores if score is not None]

    def _score(self, signals: Mapping[str, float | None]) -> float | None:
        weighted = [
            (getattr(self.weights, name), value)
            for name, value in signals.items()
            if value is not None and getattr(self.weights, name) > 0
        ]
        total = sum(weight for weight, _ in weighted)
        return sum(weight * value for weight, value in weighted) / total if total else None
