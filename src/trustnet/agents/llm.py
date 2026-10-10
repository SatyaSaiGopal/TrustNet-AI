"""An agent backed by an LLM provider."""

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trustnet.agents.base import AgentOutputError
from trustnet.agents.roles import DEFAULT_ROLES, OUTPUT_INSTRUCTIONS, PROMPT_VERSION, AgentRole
from trustnet.providers.base import GenerationRequest, LLMProvider
from trustnet.schemas import (
    AgentProfile,
    AgentResponse,
    Evidence,
    EvidenceSourceType,
    Stance,
    Task,
    is_mock_model_id,
)


class _EvidenceDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim: str = Field(min_length=1)
    reference: str | None = None


class _Draft(BaseModel):
    """The fields the model writes. Identity fields are filled in by the agent."""

    model_config = ConfigDict(extra="ignore")

    # null is how the model abstains, so stance and recommendation cannot disagree.
    recommendation: str | None
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_references: list[_EvidenceDraft]
    uncertainties: list[str]


def _describe(error: ValidationError) -> str:
    problems = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "response"
        problems.append(f"{location}: {item['msg']}")
    return "; ".join(problems)


class LLMAgent:
    def __init__(self, role: AgentRole, provider: LLMProvider) -> None:
        self.role = role
        self._provider = provider
        self.profile = AgentProfile(
            agent_id=role.agent_id,
            display_name=role.display_name,
            provider=provider.name,
            model=provider.model,
        )

    def build_request(self, task: Task) -> GenerationRequest:
        prompt = f"Problem:\n{task.question}"
        if task.options:
            choices = "\n".join(f"- {option}" for option in task.options)
            prompt += (
                "\n\nYour recommendation must be exactly one of these options, copied as "
                f"written, or null:\n{choices}"
            )
        return GenerationRequest(
            system=f"{self.role.instructions}\n\n{OUTPUT_INSTRUCTIONS}",
            prompt=prompt,
            temperature=0.0,
            response_schema=_Draft.model_json_schema(),
        )

    def respond(self, task: Task) -> AgentResponse:
        reply = self._provider.generate(self.build_request(task))
        model_id = f"{reply.provider}:{reply.model}"
        response_id = uuid4().hex
        try:
            draft = _Draft.model_validate_json(reply.text)
            return AgentResponse(
                response_id=response_id,
                task_id=task.task_id,
                agent_id=self.role.agent_id,
                stance=Stance.ABSTAIN if draft.recommendation is None else Stance.ANSWER,
                recommendation=draft.recommendation,
                rationale=draft.rationale,
                self_reported_confidence=draft.confidence,
                # Nothing was retrieved or checked, so every citation is the model's
                # own assertion and stays unverified.
                evidence=tuple(
                    Evidence(
                        evidence_id=f"{response_id}:e{index}",
                        source_type=EvidenceSourceType.MODEL_ASSERTION,
                        excerpt=item.claim,
                        reference=(item.reference or "").strip() or None,
                    )
                    for index, item in enumerate(draft.evidence_references, start=1)
                ),
                uncertainties=tuple(item for item in draft.uncertainties if item.strip()),
                model_id=model_id,
                is_mock=is_mock_model_id(model_id),
                prompt_version=PROMPT_VERSION,
            )
        except ValidationError as exc:
            raise AgentOutputError(
                f"Agent '{self.role.agent_id}' ({model_id}) returned output that is not a "
                f"valid response: {_describe(exc)}"
            ) from exc


def build_agents(provider: LLMProvider) -> tuple[LLMAgent, ...]:
    """The Analyst, Skeptic and Researcher, all backed by the given provider."""
    return tuple(LLMAgent(role, provider) for role in DEFAULT_ROLES)
