"""Metric factory for benchmark scoring functions."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.metrics.cka import linear_cka
from src.metrics.pwcca import pwcca
from src.metrics.knn_overlap import knn_overlap

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def build_metric(metric_name: str, cka_cfg: dict, metrics_cfg: dict | None = None) -> MetricFn:
    """Build and return a metric function by name.

    Parameters
    ----------
    metric_name:
        One of ``cka``, ``pwcca``, ``knn_overlap``.
    cka_cfg:
        Config dict for CKA (kept for backwards compatibility).
    metrics_cfg:
        Full metrics config dict.  Used to extract per-metric sub-configs
        (e.g. ``metrics_cfg["pwcca"]``, ``metrics_cfg["knn_overlap"]``).
    """
    if metrics_cfg is None:
        metrics_cfg = {}

    if metric_name == "cka":
        center_gram = bool(cka_cfg.get("center_gram", True))
        return lambda x, y: linear_cka(x, y, center_gram=center_gram)

    if metric_name == "pwcca":
        cfg = metrics_cfg.get("pwcca", {})
        return lambda x, y: pwcca(x, y, cfg=cfg)

    if metric_name == "knn_overlap":
        cfg = metrics_cfg.get("knn_overlap", {})
        return lambda x, y: knn_overlap(x, y, cfg=cfg)

    raise ValueError(
        f"Unknown metric: '{metric_name}'. "
        f"Supported: cka, pwcca, knn_overlap."
    )
