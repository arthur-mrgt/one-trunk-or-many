"""PWCCA — Projection-Weighted CCA similarity metric.

Reference
---------
Morcos et al., "Insights on representational similarity in neural networks
with canonical correlation", NeurIPS 2018.
https://arxiv.org/abs/1806.05759

Mathematical summary
--------------------
Given activation matrices X (N × d1) and Y (N × d2):

1. Optionally reduce X and Y via PCA (recommended when d >> N).
2. Compute CCA: find directions u_i, v_i that maximise corr(Xu_i, Yv_i).
   This yields canonical correlations rho_1 >= rho_2 >= ... >= rho_k.
3. Compute projection weights w_i = |X @ u_i| (L1 norm of the projection
   of X onto canonical direction u_i), which measures how much of the
   variance in X each canonical component captures.
4. PWCCA = sum(w_i * rho_i) / sum(w_i)

The result is a scalar in [0, 1] where 1 means perfect alignment.
"""

from __future__ import annotations

import numpy as np

from src.metrics.reduction import maybe_reduce


def _cca(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CCA in a numerically stable way using thin SVD factors.

    Returns
    -------
    correlations : (k,) canonical correlation values rho_i in [0, 1]
    a            : (d1, k) left canonical directions (in x-space)
    b            : (d2, k) right canonical directions (in y-space)
    """
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)

    # X = Ux Sx Vx^T, Y = Uy Sy Vy^T
    ux, sx, vtx = np.linalg.svd(x, full_matrices=False)
    uy, sy, vty = np.linalg.svd(y, full_matrices=False)

    # Keep only well-conditioned singular directions.
    mask_x = sx > eps
    mask_y = sy > eps
    if not np.any(mask_x) or not np.any(mask_y):
        return (
            np.array([], dtype=float),
            np.empty((x.shape[1], 0)),
            np.empty((y.shape[1], 0)),
        )

    ux = ux[:, mask_x]
    sx = sx[mask_x]
    vx = vtx[mask_x, :].T

    uy = uy[:, mask_y]
    sy = sy[mask_y]
    vy = vty[mask_y, :].T

    # Canonical correlations are cosines of principal angles between subspaces.
    m = ux.T @ uy
    p, rho, qt = np.linalg.svd(m, full_matrices=False)
    rho = np.clip(rho, 0.0, 1.0)

    # Map canonical directions back to feature spaces.
    a = vx @ (p / sx[:, None])
    b = vy @ (qt.T / sy[:, None])
    return rho, a, b


def _pwcca_one_side(
    centered: np.ndarray,
    dirs: np.ndarray,
    rho: np.ndarray,
    eps: float,
) -> float:
    """Compute one-sided PWCCA score from canonical variates.

    Uses projection-weight idea:
      H = X A
      solve H C ~= X
      w_i = ||row_i(C)||_1
    """
    if len(rho) == 0 or dirs.shape[1] == 0:
        return 0.0

    h = centered @ dirs  # (N, k)
    if h.size == 0:
        return 0.0

    try:
        coeff = np.linalg.lstsq(h, centered, rcond=None)[0]  # (k, d)
        weights = np.sum(np.abs(coeff), axis=1)
    except np.linalg.LinAlgError:
        # Conservative fallback if lstsq is ill-conditioned.
        weights = np.sum(np.abs(h), axis=0)

    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight < eps:
        return float(np.mean(rho))

    weights = np.asarray(weights, dtype=float) / total_weight
    score = float(np.sum(weights * rho))
    if not np.isfinite(score):
        return float(np.mean(rho))
    return score


def pwcca(
    x: np.ndarray,
    y: np.ndarray,
    cfg: dict | None = None,
) -> float:
    """Compute PWCCA similarity between activation matrices x and y.

    Parameters
    ----------
    x, y:
        Activation matrices of shape (N, d).  Must have the same N.
    cfg:
        Metric config dict.  Recognised keys:
          pca.enabled        (bool,  default True)
          pca.n_components   (int,   default 64)
          pca.seed           (int,   default 0)
          eps                (float, default 1e-10)

    Returns
    -------
    float in [0, 1].
    """
    if cfg is None:
        cfg = {}

    if x.shape[0] != y.shape[0]:
        raise ValueError("PWCCA requires the same number of samples in x and y.")

    eps: float = float(cfg.get("eps", 1e-10))

    # PCA reduction (enabled by default for PWCCA)
    pca_cfg = {"pca": {"enabled": True, "n_components": 64, "seed": 0}}
    pca_cfg["pca"].update(cfg.get("pca", {}))
    x_r, y_r = maybe_reduce(x, y, pca_cfg)

    if x_r.shape[1] == 0 or y_r.shape[1] == 0:
        return 0.0

    rho, a, b = _cca(x_r, y_r, eps=eps)

    if len(rho) == 0:
        return 0.0

    x_c = x_r - x_r.mean(axis=0)
    y_c = y_r - y_r.mean(axis=0)

    # Symmetric PWCCA by averaging both directions.
    left = _pwcca_one_side(x_c, a, rho, eps=eps)
    right = _pwcca_one_side(y_c, b, rho, eps=eps)
    score = 0.5 * (left + right)

    if not np.isfinite(score):
        return float(np.mean(rho))
    return float(np.clip(score, 0.0, 1.0))
