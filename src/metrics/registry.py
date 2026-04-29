"""Metric factory for benchmark scoring functions."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.metrics.cka import linear_cka

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def build_metric(metric_name: str, cka_cfg: dict) -> MetricFn:
    """Build and return the metric function by name."""
    if metric_name == "cka":
        center_gram = bool(cka_cfg.get("center_gram", True))
        return lambda x, y: linear_cka(x, y, center_gram=center_gram)
    raise ValueError(f"Unknown metric: {metric_name}")
