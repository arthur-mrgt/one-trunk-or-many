"""Pure statistical helpers for null-hypothesis significance testing."""

from __future__ import annotations

import numpy as np


def compute_p_value(
    observed: float,
    null_values: np.ndarray,
    side: str = "greater",
) -> float:
    """Compute an empirical p-value against a null distribution.

    Uses the conservative +1 correction to avoid p=0:

        p = (1 + count(null >= observed)) / (N + 1)    [side="greater"]

    Parameters
    ----------
    observed:
        The observed metric value (e.g. CKA on matched pairs).
    null_values:
        1-D array of metric values from the null distribution.
    side:
        ``"greater"`` (default) — test whether observed > null (one-sided).
        ``"less"``    — test whether observed < null.
        ``"two"``     — two-sided test.

    Returns
    -------
    float in (0, 1].
    """
    null = np.asarray(null_values, dtype=float)
    n = len(null)
    if n == 0:
        return float("nan")

    if side == "greater":
        count = int(np.sum(null >= observed))
    elif side == "less":
        count = int(np.sum(null <= observed))
    elif side == "two":
        median = float(np.median(null))
        deviation = abs(observed - median)
        count = int(np.sum(np.abs(null - median) >= deviation))
    else:
        raise ValueError(f"Unknown side='{side}'. Expected 'greater', 'less', or 'two'.")

    return (1 + count) / (n + 1)


def bh_fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Parameters
    ----------
    p_values:
        1-D array of raw p-values.
    alpha:
        Target false-discovery rate (default 0.05).

    Returns
    -------
    rejected : bool array — True where the null is rejected after correction.
    p_adjusted : float array — BH-adjusted p-values (same order as input).
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p)
    ranks = np.arange(1, n + 1)

    # Adjusted p-values in sorted order
    p_sorted = p[order]
    p_adj_sorted = np.minimum(1.0, p_sorted * n / ranks)
    # Enforce monotonicity from the right
    p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]

    # Map back to original order
    p_adjusted = np.empty(n)
    p_adjusted[order] = p_adj_sorted

    rejected = p_adjusted <= alpha
    return rejected, p_adjusted
