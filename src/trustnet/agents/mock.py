"""Deterministic mock agents FOR UNIT TESTS ONLY.

Nothing in the application builds a ``MockAgent``: ``build_agents`` and
``create_provider`` only ever return real, model-backed objects. Every response a
mock produces is labelled three ways: ``is_mock`` is true, ``model_id`` starts with
``mock:``, and the text fields start with ``[MOCK]``.
"""

from trustnet.schemas import MOCK_PROVIDER_NAME, AgentProfile, AgentResponse, Stance, Task

MOCK_LABEL = "[MOCK]"
MOCK_MODEL = "deterministic-agent-v0"


class MockAgent:
    def __init__(
        self,
        agent_id: str,
        recommendation: str | None = "mock recommendation",
        confidence: float = 0.5,
        error: Exception | None = None,
    ) -> None:
        """``recommendation=None`` makes the mock abstain; ``error`` makes it fail."""
        self.profile = AgentProfile(
            agent_id=agent_id,
            display_name=f"{MOCK_LABEL} {agent_id}",
            provider=MOCK_PROVIDER_NAME,
            model=MOCK_MODEL,
            is_synthetic=True,
        )
        self._recommendation = recommendation
        self._confidence = confidence
        self._error = error
        self.received: list[Task] = []

    def respond(self, task: Task) -> AgentResponse:
        self.received.append(task)
        if self._error is not None:
            raise self._error
        agent_id = self.profile.agent_id
        abstains = self._recommendation is None
        return AgentResponse(
            response_id=f"mock-{task.task_id}-{agent_id}",
            task_id=task.task_id,
            agent_id=agent_id,
            stance=Stance.ABSTAIN if abstains else Stance.ANSWER,
            recommendation=None if abstains else f"{MOCK_LABEL} {self._recommendation}",
            rationale=f"{MOCK_LABEL} scripted rationale from {agent_id}",
            self_reported_confidence=self._confidence,
            uncertainties=(f"{MOCK_LABEL} scripted uncertainty",),
            model_id=f"{MOCK_PROVIDER_NAME}:{MOCK_MODEL}",
            is_mock=True,
        )
