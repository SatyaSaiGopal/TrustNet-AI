"""The interface for turning agent responses plus trust signals into a decision."""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from trustnet.schemas import AgentResponse, Decision, Task, TrustSignals


@runtime_checkable
class DecisionPolicy(Protocol):
    def decide(
        self,
        task: Task,
        responses: Sequence[AgentResponse],
        trust: Mapping[str, TrustSignals],
    ) -> Decision:
        """Combine the responses, keyed to ``trust`` by agent_id.

        Must return an abstained decision when the evidence is insufficient.
        """
        ...
