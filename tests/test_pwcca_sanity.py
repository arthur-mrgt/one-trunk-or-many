"""Sanity tests for PWCCA numerical behavior."""

from __future__ import annotations

import numpy as np

from src.metrics.pwcca import pwcca


def test_pwcca_identity_is_high() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 64))
    value = pwcca(x, x, cfg={"pca": {"enabled": False}})
    assert 0.98 <= value <= 1.0


def test_pwcca_independent_is_lower_than_identity() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((128, 64))
    y = rng.standard_normal((128, 64))
    indep = pwcca(x, y, cfg={"pca": {"enabled": False}})
    ident = pwcca(x, x, cfg={"pca": {"enabled": False}})
    assert 0.0 <= indep <= 1.0
    assert indep < ident


def test_pwcca_rank_mismatch_stays_finite() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((128, 64))
    z = rng.standard_normal((128, 5))
    b = rng.standard_normal((5, 64))
    y = z @ b  # low-rank representation

    value = pwcca(x, y, cfg={"pca": {"enabled": False}})
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0
