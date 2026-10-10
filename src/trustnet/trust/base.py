"""The interface for computing an agent's trust signals."""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from trustnet.schemas import AgentResponse, EvidenceAssessment, Outcome, Task, TrustSignals


@runtime_checkable
class TrustEvaluator(Protocol):
    def evaluate(
        self,
        agent_id: str,
        responses: Sequence[AgentResponse],
        outcomes: Mapping[str, Outcome],
        tasks: Mapping[str, Task] | None = None,
        assessments: Mapping[str, EvidenceAssessment] | None = None,
        category: str | None = None,
    ) -> TrustSignals:
        """Score one agent from its past responses.

        ``outcomes`` maps task_id to verified ground truth. Accuracy and calibration
        must be computed against it, never from ``self_reported_confidence`` alone.
        """
        ...
