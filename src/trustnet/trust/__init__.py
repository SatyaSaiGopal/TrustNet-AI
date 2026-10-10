"""Trust evaluation: deriving trust signals for an agent from its recorded history."""

from trustnet.trust.base import TrustEvaluator
from trustnet.trust.engine import TrustEngine, TrustWeights
from trustnet.trust.metrics import (
    brier_score,
    calibration_summary,
    is_correct,
    normalize_answer,
    pairwise_agreement,
    wilson_interval,
)

__all__ = [
    "TrustEngine",
    "TrustEvaluator",
    "TrustWeights",
    "brier_score",
    "calibration_summary",
    "is_correct",
    "normalize_answer",
    "pairwise_agreement",
    "wilson_interval",
]
