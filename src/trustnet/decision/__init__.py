"""Decision logic: combining conflicting agent responses into a decision or an abstention."""

from trustnet.decision.base import DecisionPolicy
from trustnet.decision.strategies import (
    EQUAL_WEIGHT,
    HIGHEST_CONFIDENCE,
    MAJORITY,
    STRATEGIES,
    TRUST_WEIGHTED,
    DecisionConfig,
    TrustWeightedPolicy,
    answer_groups,
    decide_all,
    equal_weight_vote,
    evidence_summary,
    highest_confidence,
    majority_vote,
)

__all__ = [
    "EQUAL_WEIGHT",
    "HIGHEST_CONFIDENCE",
    "MAJORITY",
    "STRATEGIES",
    "TRUST_WEIGHTED",
    "DecisionConfig",
    "DecisionPolicy",
    "TrustWeightedPolicy",
    "answer_groups",
    "decide_all",
    "equal_weight_vote",
    "evidence_summary",
    "highest_confidence",
    "majority_vote",
]
