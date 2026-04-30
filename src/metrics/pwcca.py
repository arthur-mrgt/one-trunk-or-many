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
    """Compute CCA between x and y via SVD of the cross-covariance.

    Returns
    -------
    correlations : (k,) canonical correlation values rho_i in [0, 1]
    u            : (d1, k) left canonical directions (in x-space)
    v            : (d2, k) right canonical directions (in y-space)
    """
    n = x.shape[0]
    # Centre
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)

    # Whitening: X_w = X @ Sx^{-1/2}
    def _whiten(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return whitened matrix and the whitening matrix W s.t. m @ W = m_w."""
        cov = (m.T @ m) / (n - 1) + eps * np.eye(m.shape[1])
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, eps)
        W = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        return m @ W, W

    x_w, wx = _whiten(x)
    y_w, wy = _whiten(y)

    # SVD of cross-covariance of whitened matrices
    cross = (x_w.T @ y_w) / (n - 1)
    u_w, rho, vt_w = np.linalg.svd(cross, full_matrices=False)

    # Map back to original (non-whitened) space
    u = wx @ u_w          # (d1, k)
    v = wy @ vt_w.T       # (d2, k)
    return rho, u, v


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

    rho, u, _ = _cca(x_r, y_r, eps=eps)

    if len(rho) == 0:
        return 0.0

    # Projection weights: how much variance each canonical dir captures in x
    weights = np.abs(x_r @ u).sum(axis=0)   # (k,)
    total_weight = weights.sum()
    if total_weight < eps:
        return float(rho.mean())

    return float((weights * rho).sum() / total_weight)
