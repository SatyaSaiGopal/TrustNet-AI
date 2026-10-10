"""The full path for one task: run agents, check evidence, weigh by trust, decide, store.

The API and UI call this module; it holds no scoring or decision logic of its own.
"""

from collections.abc import Sequence

from trustnet.agents.base import Agent
from trustnet.agents.workflow import run_workflow
from trustnet.config import Settings
from trustnet.decision.strategies import (
    DecisionConfig,
    TrustWeightedPolicy,
    answer_groups,
    decide_all,
)
from trustnet.evidence.evaluator import DirectorySourceStore, EvidenceEvaluator, LLMEvidenceJudge
from trustnet.providers.base import LLMProvider
from trustnet.schemas import (
    Decision,
    EvidenceAssessment,
    Outcome,
    Task,
    TrustNetModel,
    TrustSignals,
    WorkflowResult,
)
from trustnet.storage.repository import Repository
from trustnet.trust.engine import TrustEngine, TrustWeights


class TaskEvaluation(TrustNetModel):
    result: WorkflowResult
    assessments: tuple[EvidenceAssessment, ...]
    # The reliability profile applied to each agent for this task's category.
    trust: dict[str, TrustSignals]
    # One per strategy; the trust-weighted decision is first.
    decisions: tuple[Decision, ...]

    @property
    def decision(self) -> Decision:
        return self.decisions[0]

    @property
    def conflicting_answers(self) -> dict[str, list[str]]:
        """Each distinct answer and the agents that gave it; more than one is a conflict."""
        return {
            members[0].recommendation or "": [member.agent_id for member in members]
            for members in answer_groups(self.result.responses).values()
        }


def _set(**values: object) -> dict[str, object]:
    return {name: value for name, value in values.items() if value is not None}


def build_engine(settings: Settings) -> TrustEngine:
    weights = TrustWeights(
        **_set(
            historical_accuracy=settings.weight_historical_accuracy,
            calibration=settings.weight_calibration,
            evidence_quality=settings.weight_evidence_quality,
            consistency=settings.weight_consistency,
            insufficient_information_behavior=settings.weight_insufficient_information_behavior,
        )
    )
    return TrustEngine(weights, min_samples=settings.trust_min_samples)


def build_policy(settings: Settings) -> TrustWeightedPolicy:
    return TrustWeightedPolicy(
        DecisionConfig(
            **_set(
                min_support=settings.decision_min_support,
                min_trust=settings.decision_min_trust,
                require_verified_evidence=settings.decision_require_verified_evidence,
            )
        )
    )


def build_evaluator(settings: Settings, provider: LLMProvider | None = None) -> EvidenceEvaluator:
    sources = DirectorySourceStore(settings.evidence_dir) if settings.evidence_dir else None
    judge = LLMEvidenceJudge(provider) if settings.evidence_llm_judge and provider else None
    return EvidenceEvaluator(sources, judge)


def evaluate_task(
    task: Task,
    agents: Sequence[Agent],
    repository: Repository,
    engine: TrustEngine | None = None,
    evaluator: EvidenceEvaluator | None = None,
    policy: TrustWeightedPolicy | None = None,
) -> TaskEvaluation:
    engine = engine or TrustEngine()
    evaluator = evaluator or EvidenceEvaluator()

    result = run_workflow(task, agents)
    assessments = tuple(
        item for response in result.responses for item in evaluator.assess_response(response)
    )
    for agent in agents:
        repository.save_agent(agent.profile)
    repository.save_task(task)
    repository.save_responses(result.responses, assessments)

    profiles = repository.refresh_reliability(engine)
    trust = {
        agent_id: profile
        for agent_id in result.expected_agent_ids
        if (profile := engine.select(profiles.get(agent_id, {}), task.category)) is not None
    }
    decisions = decide_all(
        task, result.responses, trust, {item.evidence_id: item for item in assessments}, policy
    )
    repository.save_decisions(decisions)
    return TaskEvaluation(result=result, assessments=assessments, trust=trust, decisions=decisions)


def record_outcome(
    repository: Repository, outcome: Outcome, engine: TrustEngine | None = None
) -> dict[str, dict[str | None, TrustSignals]]:
    """Store a verified outcome and recompute reliability. Raises on a duplicate."""
    repository.record_outcome(outcome)
    return repository.refresh_reliability(engine or TrustEngine())
