"""Evaluation: the synthetic benchmark that compares the decision strategies."""

from trustnet.evaluation.benchmark import (
    BenchmarkReport,
    StrategyResult,
    format_report,
    run_benchmark,
)
from trustnet.evaluation.synthetic import SYNTHETIC_NOTICE

__all__ = ["SYNTHETIC_NOTICE", "BenchmarkReport", "StrategyResult", "format_report", "run_benchmark"]
