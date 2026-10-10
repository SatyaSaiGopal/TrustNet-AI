"""The interface an agent exposes to the rest of the system."""

from typing import Protocol, runtime_checkable

from trustnet.schemas import AgentProfile, AgentResponse, Task


class AgentOutputError(RuntimeError):
    """An agent's output could not be turned into a valid ``AgentResponse``."""


@runtime_checkable
class Agent(Protocol):
    profile: AgentProfile

    def respond(self, task: Task) -> AgentResponse:
        """Answer the task, or abstain. Must not score or rank itself or other agents.

        The task is the only input: an agent never sees another agent's response.
        Raises ``ProviderError`` or ``AgentOutputError`` rather than inventing output.
        """
        ...
