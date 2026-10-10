"""Pure scoring functions. Every function returns ``None`` when it has nothing to measure."""

import math
import re
from collections.abc import Sequence
from itertools import combinations

from trustnet.schemas import AgentResponse, CalibrationBin, CalibrationSummary, Outcome

_EDGE_PUNCTUATION = " \t\n\r.,;:!?\"'()[]"


def normalize_answer(text: str | None) -> str | None:
    """Case, spacing and surrounding punctuation are ignored when comparing answers."""
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).strip(_EDGE_PUNCTUATION).casefold() or None


def is_correct(response: AgentResponse, outcome: Outcome) -> bool:
    """Whether an answer matches a verified outcome. Abstentions are never 'correct' here."""
    return (
        outcome.answerable
        and response.recommendation is not None
        and normalize_answer(response.recommendation) == normalize_answer(outcome.correct_answer)
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% confidence interval for a proportion; wide when the sample is small."""
    if total <= 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def brier_score(predictions: Sequence[tuple[float, bool]]) -> float | None:
    """Mean squared gap between stated confidence and what happened. 0 is perfect."""
    if not predictions:
        return None
    return sum((confidence - float(correct)) ** 2 for confidence, correct in predictions) / len(
        predictions
    )


def calibration_summary(
    predictions: Sequence[tuple[float, bool]], bins: int = 5, min_samples: int = 10
) -> CalibrationSummary | None:
    """Confidence versus accuracy by confidence band, or ``None`` below ``min_samples``."""
    total = len(predictions)
    if total < max(min_samples, 1):
        return None
    grouped: dict[int, list[tuple[float, bool]]] = {}
    for confidence, correct in predictions:
        grouped.setdefault(min(int(confidence * bins), bins - 1), []).append((confidence, correct))
    rows = []
    error = 0.0
    for index in sorted(grouped):
        members = grouped[index]
        mean_confidence = sum(confidence for confidence, _ in members) / len(members)
        accuracy = sum(correct for _, correct in members) / len(members)
        error += len(members) / total * abs(mean_confidence - accuracy)
        rows.append(
            CalibrationBin(
                lower=index / bins,
                upper=(index + 1) / bins,
                count=len(members),
                mean_confidence=mean_confidence,
                accuracy=accuracy,
            )
        )
    return CalibrationSummary(
        sample_size=total,
        brier_score=brier_score(predictions) or 0.0,
        expected_calibration_error=error,
        mean_confidence=sum(confidence for confidence, _ in predictions) / total,
        accuracy=sum(correct for _, correct in predictions) / total,
        bins=tuple(rows),
    )


def pairwise_agreement(answers: Sequence[str | None]) -> float | None:
    """Share of answer pairs that match. An abstention (``None``) only matches another."""
    pairs = list(combinations(answers, 2))
    if not pairs:
        return None
    return sum(first == second for first, second in pairs) / len(pairs)
