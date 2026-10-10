"""Four ways of turning the agents' responses into one decision.

``TrustWeightedPolicy`` is the trust-aware engine. The other three are baselines that
ignore reliability, kept for comparison: majority vote, equal-weight vote and
highest self-reported confidence.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from trustnet.schemas import (
    AgentContribution,
    AgentResponse,
    Decision,
    DecisionOutcome,
    EvidenceAssessment,
    EvidenceStatus,
    Stance,
    Task,
    TrustSignals,
)
from trustnet.trust.metrics import normalize_answer

TRUST_WEIGHTED = "trust_weighted"
MAJORITY = "majority_vote"
EQUAL_WEIGHT = "equal_weight_vote"
HIGHEST_CONFIDENCE = "highest_confidence"
STRATEGIES = (TRUST_WEIGHTED, MAJORITY, EQUAL_WEIGHT, HIGHEST_CONFIDENCE)


def answer_groups(responses: Sequence[AgentResponse]) -> dict[str, list[AgentResponse]]:
    """Answering responses grouped by normalised recommendation. More than one group is a conflict."""
    groups: dict[str, list[AgentResponse]] = defaultdict(list)
    for response in responses:
        key = normalize_answer(response.recommendation)
        if response.stance is Stance.ANSWER and key is not None:
            groups[key].append(response)
    return dict(groups)


def evidence_summary(
    response: AgentResponse, assessments: Mapping[str, EvidenceAssessment] | None
) -> str:
    if not response.evidence:
        return "none cited"
    statuses = {
        assessments[item.evidence_id].status
        for item in response.evidence
        if assessments and item.evidence_id in assessments
    }
    for status, label in (
        (EvidenceStatus.CONTRADICTORY, "contradicted"),
        (EvidenceStatus.VERIFIED, "verified"),
        (EvidenceStatus.UNSUPPORTED, "unsupported"),
    ):
        if status in statuses:
            return label
    return "unverified"


def _decide(
    task: Task,
    strategy: str,
    answer: str | None,
    explanation: str,
    contributions: Sequence[AgentContribution],
    support: float | None = None,
) -> Decision:
    decided = answer is not None
    return Decision(
        decision_id=f"{task.task_id}:{strategy}",
        task_id=task.task_id,
        outcome=DecisionOutcome.DECIDED if decided else DecisionOutcome.ABSTAINED,
        answer=answer,
        rationale=explanation,
        strategy=strategy,
        abstention_reason=None if decided else explanation,
        support=support,
        contributions=tuple(contributions),
    )


def _votes(responses: Sequence[AgentResponse], counted: bool = True) -> list[AgentContribution]:
    return [
        AgentContribution(
            agent_id=response.agent_id,
            recommendation=response.recommendation,
            weight=1.0 if counted or response.stance is Stance.ANSWER else 0.0,
            note="one vote" if response.stance is Stance.ANSWER else "abstained",
        )
        for response in responses
    ]


def _leaders(sizes: Mapping[str, float]) -> list[str]:
    top = max(sizes.values())
    return [key for key, size in sizes.items() if size == top]


def majority_vote(task: Task, responses: Sequence[AgentResponse]) -> Decision:
    """Decide only if more than half of all responding agents give the same answer."""
    groups = answer_groups(responses)
    votes = _votes(responses)
    if not groups:
        return _decide(task, MAJORITY, None, "no agent gave an answer", votes)
    key = _leaders({key: len(members) for key, members in groups.items()})[0]
    share = len(groups[key]) / len(responses)
    if share <= 0.5:
        reason = f"no answer was given by more than half of the {len(responses)} agents"
        return _decide(task, MAJORITY, None, reason, votes, share)
    answer = groups[key][0].recommendation
    reason = f"{len(groups[key])} of {len(responses)} agents gave this answer"
    return _decide(task, MAJORITY, answer, reason, votes, share)


def equal_weight_vote(task: Task, responses: Sequence[AgentResponse]) -> Decision:
    """The most common answer among the agents that answered; a tie means no decision."""
    groups = answer_groups(responses)
    votes = _votes(responses, counted=False)
    if not groups:
        return _decide(task, EQUAL_WEIGHT, None, "no agent gave an answer", votes)
    answering = sum(len(members) for members in groups.values())
    leaders = _leaders({key: len(members) for key, members in groups.items()})
    share = len(groups[leaders[0]]) / answering
    if len(leaders) > 1:
        reason = f"{len(leaders)} different answers tied for the most votes"
        return _decide(task, EQUAL_WEIGHT, None, reason, votes, share)
    answer = groups[leaders[0]][0].recommendation
    reason = f"{len(groups[leaders[0]])} of {answering} answering agents gave this answer"
    return _decide(task, EQUAL_WEIGHT, answer, reason, votes, share)


def highest_confidence(task: Task, responses: Sequence[AgentResponse]) -> Decision:
    """The answer of whichever agent states the highest confidence in itself."""
    contributions = [
        AgentContribution(
            agent_id=response.agent_id,
            recommendation=response.recommendation,
            weight=response.self_reported_confidence if response.stance is Stance.ANSWER else 0.0,
            note="weight is the agent's self-reported confidence",
        )
        for response in responses
    ]
    answering = [response for response in responses if response.stance is Stance.ANSWER]
    if not answering:
        return _decide(task, HIGHEST_CONFIDENCE, None, "no agent gave an answer", contributions)
    top = max(response.self_reported_confidence for response in answering)
    best = [response for response in answering if response.self_reported_confidence == top]
    if len(answer_groups(best)) > 1:
        reason = f"agents with different answers tied at confidence {top:.2f}"
        return _decide(task, HIGHEST_CONFIDENCE, None, reason, contributions, top)
    reason = f"'{best[0].agent_id}' stated the highest confidence ({top:.2f})"
    return _decide(task, HIGHEST_CONFIDENCE, best[0].recommendation, reason, contributions, top)


class DecisionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Share of all counted reliability weight the leading answer needs.
    min_support: float = Field(default=0.6, gt=0.5, le=1.0)
    # The leading answer needs at least one supporter with this trust score.
    min_trust: float = Field(default=0.5, ge=0.0, le=1.0)
    # Weight multiplier for a response whose own cited source contradicts it.
    contradicted_evidence_factor: float = Field(default=0.25, ge=0.0, le=1.0)
    # When true, decide only if a supporter of the leading answer has verified evidence.
    require_verified_evidence: bool = False


class TrustWeightedPolicy:
    """Weighs each response by the agent's measured trust score.

    * An agent without enough verified history has no trust score and is not counted;
      no default weight is assumed for it.
    * Agents on the same model giving the same response are treated as correlated and
      share one weight between them instead of counting separately.
    * Abstaining agents count against deciding: their weight is part of the total.
    * It abstains when nothing can be weighted, when the leading answer lacks
      ``min_support`` of the weight, or when none of its supporters is trusted enough.
    """

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()

    def decide(
        self,
        task: Task,
        responses: Sequence[AgentResponse],
        trust: Mapping[str, TrustSignals],
        assessments: Mapping[str, EvidenceAssessment] | None = None,
    ) -> Decision:
        config = self.config
        if not responses:
            return _decide(task, TRUST_WEIGHTED, None, "no agent responded", [])

        scores = {
            response.agent_id: profile.trust_score
            for response in responses
            if (profile := trust.get(response.agent_id)) is not None
        }
        evidence = {
            response.agent_id: evidence_summary(response, assessments) for response in responses
        }
        # Same model + same response = one correlated block.
        blocks: dict[tuple[str | None, str], list[AgentResponse]] = defaultdict(list)
        for response in responses:
            if scores.get(response.agent_id) is not None:
                key = normalize_answer(response.recommendation)
                blocks[(key, response.model_id)].append(response)

        weights: dict[str, float] = {}
        notes: dict[str, str] = {}
        for members in blocks.values():
            for response in members:
                weight = scores[response.agent_id] or 0.0
                parts = []
                if evidence[response.agent_id] == "contradicted":
                    weight *= config.contradicted_evidence_factor
                    parts.append("reduced: its own cited source contradicts it")
                if len(members) > 1:
                    weight /= len(members)
                    parts.append(f"shared with {len(members) - 1} other agent(s) on the same model")
                weights[response.agent_id] = weight
                notes[response.agent_id] = "; ".join(parts) or "counted in full"

        contributions = []
        for response in responses:
            profile = trust.get(response.agent_id)
            note = notes.get(response.agent_id)
            if note is None:
                seen = profile.resolved_sample_size if profile else 0
                note = f"not counted: no validated reliability ({seen} verified answers on record)"
            contributions.append(
                AgentContribution(
                    agent_id=response.agent_id,
                    recommendation=response.recommendation,
                    weight=weights.get(response.agent_id),
                    trust_score=scores.get(response.agent_id),
                    evidence_status=evidence[response.agent_id],
                    note=note,
                )
            )

        def abstain(reason: str, support: float | None = None) -> Decision:
            return _decide(task, TRUST_WEIGHTED, None, reason, contributions, support)

        total = sum(weights.values())
        if total <= 0:
            return abstain("no responding agent has enough verified history to be weighted")

        groups = answer_groups([response for response in responses if response.agent_id in weights])
        totals = {
            key: sum(weights[member.agent_id] for member in members)
            for key, members in groups.items()
        }
        abstaining = total - sum(totals.values())
        if not totals or abstaining >= max(totals.values()):
            return abstain(
                "the agents holding the most reliability weight judged the information "
                "insufficient to answer",
                abstaining / total,
            )
        leaders = _leaders(totals)
        share = totals[leaders[0]] / total
        answer = groups[leaders[0]][0].recommendation
        if len(leaders) > 1 or share < config.min_support:
            return abstain(
                f"disagreement unresolved: the leading answer '{answer}' holds {share:.0%} of "
                f"the reliability weight, below the {config.min_support:.0%} required",
                share,
            )
        supporters = groups[leaders[0]]
        best = max(scores[member.agent_id] or 0.0 for member in supporters)
        if best < config.min_trust:
            return abstain(
                f"no agent behind the leading answer '{answer}' is reliable enough (best "
                f"trust score {best:.2f}, {config.min_trust:.2f} required)",
                share,
            )
        if config.require_verified_evidence and not any(
            evidence[member.agent_id] == "verified" for member in supporters
        ):
            return abstain(f"no verified evidence supports the leading answer '{answer}'", share)

        names = ", ".join(member.agent_id for member in supporters)
        reason = f"backed by {names} with {share:.0%} of the reliability weight"
        return _decide(task, TRUST_WEIGHTED, answer, reason, contributions, share)


def decide_all(
    task: Task,
    responses: Sequence[AgentResponse],
    trust: Mapping[str, TrustSignals],
    assessments: Mapping[str, EvidenceAssessment] | None = None,
    policy: TrustWeightedPolicy | None = None,
) -> tuple[Decision, ...]:
    """One decision per strategy, in ``STRATEGIES`` order, from identical inputs."""
    return (
        (policy or TrustWeightedPolicy()).decide(task, responses, trust, assessments),
        majority_vote(task, responses),
        equal_weight_vote(task, responses),
        highest_confidence(task, responses),
    )
