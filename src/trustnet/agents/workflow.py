"""The multi-agent execution workflow, as a LangGraph ``StateGraph``.

    START --> analyst ----+
          --> skeptic ----+--> collect_outputs --> END
          --> researcher -+

Every agent node starts from START in the same step and is given only the task, so no
agent can see another's output. ``collect_outputs`` gathers the results; it does not
score, rank or merge them.
"""

import operator
from collections.abc import Callable, Sequence
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from trustnet.agents.base import Agent, AgentOutputError
from trustnet.providers.base import ProviderError
from trustnet.schemas import AgentFailure, AgentResponse, Task, WorkflowResult

COLLECT_NODE = "collect_outputs"


class WorkflowInputError(ValueError):
    """The workflow was given a task or a set of agents it cannot run."""


class WorkflowState(TypedDict, total=False):
    """Shared graph state: the task and the outputs collected from the agents."""

    task: Task
    responses: Annotated[list[AgentResponse], operator.add]
    failures: Annotated[list[AgentFailure], operator.add]
    result: WorkflowResult


class AgentInput(TypedDict):
    """All an agent node is given. Deliberately excludes the collected outputs."""

    task: Task


def _agent_node(agent: Agent) -> Callable[[AgentInput], dict[str, Any]]:
    agent_id = agent.profile.agent_id

    def node(state: AgentInput) -> dict[str, Any]:
        task = state["task"]
        try:
            response = agent.respond(task)
            if not isinstance(response, AgentResponse):
                raise AgentOutputError(
                    f"Agent '{agent_id}' returned {type(response).__name__}, not an AgentResponse"
                )
            if response.agent_id != agent_id or response.task_id != task.task_id:
                raise AgentOutputError(
                    f"Agent '{agent_id}' returned a response labelled for agent "
                    f"'{response.agent_id}' and task '{response.task_id}'"
                )
        except (ProviderError, AgentOutputError) as exc:
            failure = AgentFailure(
                agent_id=agent_id,
                task_id=task.task_id,
                error_type=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
            )
            return {"failures": [failure]}
        return {"responses": [response]}

    return node


def _collect_node(agent_ids: tuple[str, ...]) -> Callable[[WorkflowState], dict[str, Any]]:
    position = {agent_id: index for index, agent_id in enumerate(agent_ids)}

    def node(state: WorkflowState) -> dict[str, Any]:
        result = WorkflowResult(
            task=state["task"],
            expected_agent_ids=agent_ids,
            responses=tuple(
                sorted(state.get("responses", []), key=lambda item: position[item.agent_id])
            ),
            failures=tuple(
                sorted(state.get("failures", []), key=lambda item: position[item.agent_id])
            ),
        )
        return {"result": result}

    return node


def build_workflow(agents: Sequence[Agent]) -> CompiledStateGraph:
    agent_ids = tuple(agent.profile.agent_id for agent in agents)
    if not agent_ids:
        raise WorkflowInputError("the workflow needs at least one agent")
    if len(set(agent_ids)) != len(agent_ids):
        raise WorkflowInputError(f"agent ids must be unique, got {list(agent_ids)}")
    if COLLECT_NODE in agent_ids:
        raise WorkflowInputError(f"'{COLLECT_NODE}' is reserved and cannot be an agent id")

    graph = StateGraph(WorkflowState)
    for agent in agents:
        graph.add_node(agent.profile.agent_id, _agent_node(agent))
        graph.add_edge(START, agent.profile.agent_id)
    graph.add_node(COLLECT_NODE, _collect_node(agent_ids))
    graph.add_edge(list(agent_ids), COLLECT_NODE)
    graph.add_edge(COLLECT_NODE, END)
    return graph.compile()


def run_workflow(task: Task, agents: Sequence[Agent]) -> WorkflowResult:
    """Run every agent independently on the task and collect what each returned.

    An agent that fails is reported in ``WorkflowResult.failures`` and listed in
    ``missing_agent_ids``; its output is never substituted.
    """
    if not isinstance(task, Task):
        raise WorkflowInputError(
            f"task must be a trustnet.schemas.Task, got {type(task).__name__}"
        )
    final_state = build_workflow(agents).invoke({"task": task, "responses": [], "failures": []})
    return final_state["result"]
