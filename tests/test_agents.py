from collections.abc import Callable
from typing import Any

import pytest

from trustnet.agents import (
    ANALYST,
    RESEARCHER,
    SKEPTIC,
    Agent,
    AgentOutputError,
    LLMAgent,
    build_agents,
)
from trustnet.agents.mock import MOCK_LABEL, MockAgent
from trustnet.providers import (
    DeterministicProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderUnavailableError,
)
from trustnet.schemas import EvidenceSourceType, Stance, Task, VerificationStatus

Draft = Callable[..., str]


def analyst_replying(text: str) -> LLMAgent:
    return LLMAgent(ANALYST, DeterministicProvider(default=text))


def test_llm_agent_returns_a_validated_response(task: Task, draft_json: Draft) -> None:
    response = analyst_replying(draft_json()).respond(task)

    assert response.agent_id == "analyst"
    assert response.task_id == task.task_id
    assert response.stance is Stance.ANSWER
    assert response.recommendation == "MOCK do X"
    assert response.rationale == "MOCK because of Y"
    assert response.self_reported_confidence == 0.6
    assert [item.excerpt for item in response.evidence] == ["MOCK claim"]
    assert response.uncertainties == ("MOCK unknown",)
    assert response.model_id == "mock:deterministic-v0"
    assert response.is_mock


def test_cited_evidence_is_recorded_as_an_unverified_model_assertion(
    task: Task, draft_json: Draft
) -> None:
    references = [
        {"claim": "MOCK claim with source", "reference": "MOCK-source-1"},
        {"claim": "MOCK claim without source", "reference": "  "},
    ]
    response = analyst_replying(draft_json(evidence_references=references)).respond(task)

    assert [item.reference for item in response.evidence] == ["MOCK-source-1", None]
    for item in response.evidence:
        assert item.source_type is EvidenceSourceType.MODEL_ASSERTION
        assert item.verification_status is VerificationStatus.UNVERIFIED


@pytest.mark.parametrize("confidence", [-0.1, 1.5, 85, "high", None])
def test_invalid_confidence_from_the_model_is_an_error(
    task: Task, draft_json: Draft, confidence: Any
) -> None:
    with pytest.raises(AgentOutputError, match="confidence") as raised:
        analyst_replying(draft_json(confidence=confidence)).respond(task)
    assert "analyst" in str(raised.value)


@pytest.mark.parametrize("field", ["recommendation", "rationale", "confidence", "uncertainties"])
def test_output_missing_a_field_is_an_error(task: Task, draft_json: Draft, field: str) -> None:
    with pytest.raises(AgentOutputError, match=field):
        analyst_replying(draft_json(**{field: ...})).respond(task)


@pytest.mark.parametrize("text", ["MOCK RESPONSE", "Sure! Here is my answer.", "[1, 2]"])
def test_output_that_is_not_a_json_object_is_an_error(task: Task, text: str) -> None:
    with pytest.raises(AgentOutputError, match="not a valid response"):
        analyst_replying(text).respond(task)


def test_a_null_recommendation_is_an_abstention(task: Task, draft_json: Draft) -> None:
    response = analyst_replying(draft_json(recommendation=None)).respond(task)
    assert response.stance is Stance.ABSTAIN
    assert response.recommendation is None


@pytest.mark.parametrize("recommendation", ["", "   "])
def test_a_blank_recommendation_is_an_error_not_an_abstention(
    task: Task, draft_json: Draft, recommendation: str
) -> None:
    with pytest.raises(AgentOutputError, match="must carry a recommendation"):
        analyst_replying(draft_json(recommendation=recommendation)).respond(task)


def test_provider_failure_propagates_instead_of_producing_output(task: Task) -> None:
    class DownProvider:
        name = "ollama"
        model = "some-model"

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            raise ProviderUnavailableError("Cannot reach Ollama")

    with pytest.raises(ProviderUnavailableError, match="Cannot reach Ollama"):
        LLMAgent(ANALYST, DownProvider()).respond(task)


def test_request_contains_the_role_the_problem_and_the_output_schema(task: Task) -> None:
    request = LLMAgent(SKEPTIC, DeterministicProvider()).build_request(task)

    assert SKEPTIC.instructions in (request.system or "")
    assert task.question in request.prompt
    assert request.temperature == 0.0
    assert set(request.response_schema["required"]) == {
        "recommendation",
        "rationale",
        "confidence",
        "evidence_references",
        "uncertainties",
    }


def test_build_agents_returns_the_three_roles_backed_by_the_given_provider() -> None:
    provider = DeterministicProvider()
    agents = build_agents(provider)

    assert [agent.profile.agent_id for agent in agents] == ["analyst", "skeptic", "researcher"]
    assert all(isinstance(agent, LLMAgent) and isinstance(agent, Agent) for agent in agents)
    assert all(agent.profile.model == provider.model for agent in agents)
    assert len({ANALYST.instructions, SKEPTIC.instructions, RESEARCHER.instructions}) == 3


def test_mock_agent_output_is_deterministic_and_labelled(task: Task) -> None:
    agent = MockAgent("analyst", recommendation="do X", confidence=0.4)
    first, second = agent.respond(task), agent.respond(task)

    assert first == second
    assert first.is_mock
    assert first.model_id.startswith("mock:")
    assert first.recommendation == f"{MOCK_LABEL} do X"
    assert first.rationale.startswith(MOCK_LABEL)
    assert all(item.startswith(MOCK_LABEL) for item in first.uncertainties)
    assert agent.profile.is_synthetic
    assert agent.profile.display_name.startswith(MOCK_LABEL)
    assert isinstance(agent, Agent)


def test_mock_agent_can_abstain(task: Task) -> None:
    response = MockAgent("skeptic", recommendation=None).respond(task)
    assert response.stance is Stance.ABSTAIN
    assert response.recommendation is None
    assert response.is_mock
