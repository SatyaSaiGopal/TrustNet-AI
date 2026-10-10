import threading
from collections.abc import Callable
from typing import Any

import httpx
import ollama
import pytest

from trustnet.agents import (
    ANALYST,
    DEFAULT_ROLES,
    RESEARCHER,
    SKEPTIC,
    LLMAgent,
    WorkflowInputError,
    build_agents,
    build_workflow,
    run_workflow,
)
from trustnet.agents.mock import MockAgent
from trustnet.agents.workflow import COLLECT_NODE, AgentInput
from trustnet.providers import DeterministicProvider, OllamaProvider, ProviderUnavailableError
from trustnet.schemas import AgentResponse, Task, WorkflowResult

Draft = Callable[..., str]
AGENT_IDS = ("analyst", "skeptic", "researcher")


def mock_agents() -> list[MockAgent]:
    return [MockAgent(agent_id, recommendation=f"view of {agent_id}") for agent_id in AGENT_IDS]


# --- Successful graph execution ---------------------------------------------------------


def test_graph_runs_all_three_agents_and_collects_their_outputs(task: Task) -> None:
    result = run_workflow(task, mock_agents())

    assert isinstance(result, WorkflowResult)
    assert result.task == task
    assert result.is_complete
    assert result.missing_agent_ids == ()
    assert result.failures == ()
    assert tuple(response.agent_id for response in result.responses) == AGENT_IDS
    assert all(response.task_id == task.task_id for response in result.responses)
    assert all(response.is_mock for response in result.responses)


def test_graph_fans_out_from_start_and_joins_at_the_collector(task: Task) -> None:
    drawn = build_workflow(mock_agents()).get_graph()
    edges = {(edge.source, edge.target) for edge in drawn.edges}

    assert edges == {
        *(("__start__", agent_id) for agent_id in AGENT_IDS),
        *((agent_id, COLLECT_NODE) for agent_id in AGENT_IDS),
        (COLLECT_NODE, "__end__"),
    }


def test_graph_runs_llm_agents_end_to_end_through_a_scripted_provider(
    task: Task, draft_json: Draft
) -> None:
    result = run_workflow(task, build_agents(DeterministicProvider(default=draft_json())))

    assert result.is_complete
    assert tuple(response.agent_id for response in result.responses) == AGENT_IDS
    assert len({response.response_id for response in result.responses}) == 3
    assert all(response.is_mock for response in result.responses)


def test_collected_outputs_follow_agent_order_not_completion_order(task: Task) -> None:
    release_analyst = threading.Event()

    class SlowAnalyst(MockAgent):
        def respond(self, task: Task) -> AgentResponse:
            release_analyst.wait(timeout=10)
            return super().respond(task)

    class Releasing(MockAgent):
        def respond(self, task: Task) -> AgentResponse:
            response = super().respond(task)
            release_analyst.set()
            return response

    agents = [SlowAnalyst("analyst"), MockAgent("skeptic"), Releasing("researcher")]
    result = run_workflow(task, agents)
    assert tuple(response.agent_id for response in result.responses) == AGENT_IDS


# --- Independent agent execution --------------------------------------------------------


def test_each_agent_is_called_once_with_only_the_task(task: Task) -> None:
    agents = mock_agents()
    run_workflow(task, agents)
    for agent in agents:
        assert agent.received == [task]


def test_agent_nodes_are_given_the_task_and_not_the_collected_outputs() -> None:
    builder = build_workflow(mock_agents()).builder
    for agent_id in AGENT_IDS:
        assert builder.nodes[agent_id].input_schema is AgentInput
    assert set(AgentInput.__annotations__) == {"task"}


def test_agents_run_in_the_same_step(task: Task) -> None:
    # Each agent blocks until all three have started. Agents run one after another
    # could never satisfy the barrier, so this passes only if none waits on another.
    barrier = threading.Barrier(len(AGENT_IDS))

    class Waiting(MockAgent):
        def respond(self, task: Task) -> AgentResponse:
            barrier.wait(timeout=10)
            return super().respond(task)

    result = run_workflow(task, [Waiting(agent_id) for agent_id in AGENT_IDS])
    assert result.is_complete


def test_no_agent_prompt_contains_another_agents_output(task: Task, draft_json: Draft) -> None:
    secrets = {role.agent_id: f"MOCK-SECRET-{role.agent_id.upper()}" for role in DEFAULT_ROLES}
    providers = {
        role.agent_id: DeterministicProvider(
            default=draft_json(recommendation=secrets[role.agent_id])
        )
        for role in DEFAULT_ROLES
    }
    agents = [LLMAgent(role, providers[role.agent_id]) for role in DEFAULT_ROLES]

    result = run_workflow(task, agents)

    assert {response.recommendation for response in result.responses} == set(secrets.values())
    for provider in providers.values():
        assert len(provider.calls) == 1
        sent = f"{provider.calls[0].system}\n{provider.calls[0].prompt}"
        assert task.question in sent
        assert not any(secret in sent for secret in secrets.values())


def test_an_agents_output_does_not_depend_on_which_other_agents_ran(task: Task) -> None:
    def analyst() -> MockAgent:
        return MockAgent("analyst", recommendation="hold", confidence=0.3)

    alone = run_workflow(task, [analyst()])
    with_agreement = run_workflow(
        task, [analyst(), MockAgent("skeptic", "hold", 0.9), MockAgent("researcher", "hold", 0.9)]
    )
    with_dissent_reordered = run_workflow(
        task, [MockAgent("researcher", "sell", 1.0), MockAgent("skeptic", None), analyst()]
    )

    expected = alone.responses[0]
    assert with_agreement.responses[0] == expected
    assert with_dissent_reordered.responses[-1] == expected


# --- Missing agent responses ------------------------------------------------------------


def test_a_failed_agent_is_reported_as_missing_and_not_replaced(task: Task) -> None:
    agents = [
        MockAgent("analyst"),
        MockAgent("skeptic", error=ProviderUnavailableError("Cannot reach Ollama at http://x")),
        MockAgent("researcher"),
    ]
    result = run_workflow(task, agents)

    assert not result.is_complete
    assert result.missing_agent_ids == ("skeptic",)
    assert tuple(response.agent_id for response in result.responses) == ("analyst", "researcher")
    (failure,) = result.failures
    assert failure.agent_id == "skeptic"
    assert failure.task_id == task.task_id
    assert failure.error_type == "ProviderUnavailableError"
    assert failure.message == "Cannot reach Ollama at http://x"


def test_invalid_model_output_makes_that_agent_missing(task: Task, draft_json: Draft) -> None:
    agents = [
        LLMAgent(ANALYST, DeterministicProvider(default=draft_json())),
        LLMAgent(SKEPTIC, DeterministicProvider(default=draft_json(confidence=7))),
        LLMAgent(RESEARCHER, DeterministicProvider(default="not json")),
    ]
    result = run_workflow(task, agents)

    assert result.missing_agent_ids == ("skeptic", "researcher")
    assert [failure.error_type for failure in result.failures] == ["AgentOutputError"] * 2
    assert "confidence" in result.failures[0].message
    assert "Invalid JSON" in result.failures[1].message


@pytest.mark.parametrize(
    "mislabel", [{"agent_id": "analyst"}, {"task_id": "some-other-task"}], ids=["agent", "task"]
)
def test_a_response_labelled_for_someone_else_is_rejected(
    task: Task, mislabel: dict[str, str]
) -> None:
    class Mislabelling(MockAgent):
        def respond(self, task: Task) -> AgentResponse:
            return super().respond(task).model_copy(update=mislabel)

    result = run_workflow(task, [MockAgent("analyst"), Mislabelling("skeptic")])

    assert result.missing_agent_ids == ("skeptic",)
    assert tuple(response.agent_id for response in result.responses) == ("analyst",)
    assert result.failures[0].error_type == "AgentOutputError"


def test_an_agent_that_returns_nothing_is_reported(task: Task) -> None:
    class Silent(MockAgent):
        def respond(self, task: Task) -> Any:
            return None

    result = run_workflow(task, [Silent("analyst")])
    assert result.missing_agent_ids == ("analyst",)
    assert "NoneType" in result.failures[0].message


def test_a_programming_error_in_an_agent_is_not_swallowed(task: Task) -> None:
    with pytest.raises(RuntimeError, match="bug"):
        run_workflow(task, [MockAgent("analyst", error=RuntimeError("bug"))])


# --- Model connection failures ----------------------------------------------------------


def agents_on_ollama_failing_with(fake_chat: Any, error: Exception) -> tuple[LLMAgent, ...]:
    provider = OllamaProvider(
        model="test-model",
        base_url="http://host:1234",
        timeout_seconds=5,
        chat_model_factory=fake_chat(error=error),
    )
    return build_agents(provider)


@pytest.mark.parametrize(
    "error, error_type, expected_text",
    [
        (
            httpx.ConnectError("actively refused"),
            "ProviderUnavailableError",
            "Cannot reach Ollama at http://host:1234",
        ),
        (
            ollama.ResponseError("model 'test-model' not found", 404),
            "ProviderUnavailableError",
            "ollama pull test-model",
        ),
        (httpx.ReadTimeout("timed out"), "ProviderTimeoutError", "within 5 seconds"),
    ],
    ids=["server-down", "model-missing", "timeout"],
)
def test_model_failure_is_reported_for_every_agent_with_no_fabricated_output(
    task: Task, fake_chat: Any, error: Exception, error_type: str, expected_text: str
) -> None:
    result = run_workflow(task, agents_on_ollama_failing_with(fake_chat, error))

    assert result.responses == ()
    assert not result.is_complete
    assert result.missing_agent_ids == AGENT_IDS
    assert tuple(failure.agent_id for failure in result.failures) == AGENT_IDS
    for failure in result.failures:
        assert failure.error_type == error_type
        assert expected_text in failure.message


# --- Input validation -------------------------------------------------------------------


@pytest.mark.parametrize("not_a_task", ["a question", None, {"task_id": "t", "question": "q"}])
def test_workflow_rejects_anything_that_is_not_a_task(not_a_task: Any) -> None:
    with pytest.raises(WorkflowInputError, match="must be a trustnet.schemas.Task"):
        run_workflow(not_a_task, mock_agents())


def test_workflow_rejects_an_empty_agent_list(task: Task) -> None:
    with pytest.raises(WorkflowInputError, match="at least one agent"):
        run_workflow(task, [])


def test_workflow_rejects_duplicate_agent_ids(task: Task) -> None:
    with pytest.raises(WorkflowInputError, match="unique"):
        run_workflow(task, [MockAgent("analyst"), MockAgent("analyst")])


def test_workflow_rejects_the_reserved_node_name(task: Task) -> None:
    with pytest.raises(WorkflowInputError, match="reserved"):
        run_workflow(task, [MockAgent(COLLECT_NODE)])
