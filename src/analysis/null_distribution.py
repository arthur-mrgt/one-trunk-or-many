"""Public API for null-distribution computation and adaptive stop logic."""

from src.analysis.null_runner import (
    NullRunResult,
    compute_null_distribution,
    compute_null_distribution_adaptive,
)
from src.analysis.null_stop import evaluate_adaptive_stop

__all__ = [
    "NullRunResult",
    "compute_null_distribution",
    "compute_null_distribution_adaptive",
    "evaluate_adaptive_stop",
]
