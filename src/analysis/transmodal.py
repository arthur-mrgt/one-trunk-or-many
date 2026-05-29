"""Transmodal triangulation — residual projection and fragmentation scores.

Mathematical foundation
-----------------------
For an ordered pair of activation matrices (A, B) of shape (N, D_A) and
(N, D_B) respectively, the **linear residual** of B with respect to A is:

    R_{A,B} = B_c - A_c @ W

where B_c and A_c are column-centred versions of B and A, and W is the
ridge-regularised least-squares solution:

    W = argmin_{W} ||A_c @ W - B_c||² + λ ||W||²_F
      = (A_c.T @ A_c + λI)^{-1} A_c.T @ B_c

R_{A,B} lives in the **space of B** (shape (N, D_B)) and represents the
component of B that is linearly orthogonal to A.

Fragmentation scores
--------------------
* **frag_global(A, B)** = ||R_{A,B}||²_F / ||B_c||²_F
    Fraction of B's variance unexplained by A.  Scalar in [0, 1].

* **frag_cka(A, B | C)** = CKA(R_{A,B}, C) / CKA(B, C)
    Fraction of the B–C shared structure that survives in the residual.
    A ratio close to 0 means A absorbed nearly all of the B–C structure;
    close to 1 means none was absorbed.  This ratio has a clean
    interpretation because CKA is based on normalised HSIC.

* **sim_raw(R, C, metric)** = metric(R_{A,B}, C)
    Raw (unnormalised) similarity between the residual and a witness
    modality.  Used for PWCCA and kNN triangulation, where the ratio
    interpretation does not hold.

Known limitations
-----------------
* All projections are **linear**.  Shared structure living on non-linear
  sub-manifolds of A is invisible to OLS and will inflate frag scores.
  The scores therefore constitute an **upper bound** on fragmentation
  under linear probing, not a global measure.
* The centred-column means are invariant to row permutations, so the
  null permutation does not require re-centring.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from src.metrics.cka import linear_cka
from src.metrics.knn_overlap import knn_overlap
from src.metrics.pwcca import pwcca


# ---------------------------------------------------------------------------
# Ridge helpers
# ---------------------------------------------------------------------------

def _auto_lambda(G: np.ndarray, ridge: float) -> float:
    """Return an auto-scaled ridge penalty based on the trace of the Gram.

    Scaling by ``trace(G) / D`` makes the penalty dimensionless and
    invariant to the overall magnitude of the activations.

    Parameters
    ----------
    G:
        Gram matrix A_c.T @ A_c of shape (D, D).
    ridge:
        Relative ridge coefficient (default 1e-4).
    """
    d = G.shape[0]
    if d == 0:
        return 0.0
    return float(ridge * np.trace(G) / d)


# ---------------------------------------------------------------------------
# Core residual computation
# ---------------------------------------------------------------------------

def compute_residual(
    A: np.ndarray,
    B: np.ndarray,
    ridge: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project B onto the column span of A and return the residual.

    Parameters
    ----------
    A:
        Left activation matrix, shape (N, D_A).
    B:
        Right activation matrix, shape (N, D_B).
    ridge:
        Relative ridge coefficient for regularisation.  The actual penalty
        is auto-scaled as ``ridge * trace(A_c.T @ A_c) / D_A``.

    Returns
    -------
    R:
        Residual matrix of shape (N, D_B), in the space of B.
    A_c:
        Column-centred A, shape (N, D_A).
    B_c:
        Column-centred B, shape (N, D_B).
    """
    A64 = A.astype(np.float64, copy=False)
    B64 = B.astype(np.float64, copy=False)

    A_c = A64 - A64.mean(axis=0, keepdims=True)
    B_c = B64 - B64.mean(axis=0, keepdims=True)

    G = A_c.T @ A_c  # (D_A, D_A)
    lam = _auto_lambda(G, ridge)
    G_reg = G + lam * np.eye(G.shape[0], dtype=np.float64)

    rhs = A_c.T @ B_c  # (D_A, D_B)
    W = scipy.linalg.solve(G_reg, rhs, assume_a="pos")  # (D_A, D_B)

    R = B_c - A_c @ W  # (N, D_B)
    return R, A_c, B_c


def precompute_gram_inverse(
    A: np.ndarray,
    ridge: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the centred A and the regularised inverse Gram matrix.

    This enables the fast-path residual computation used in the null
    distribution loop.  The key identity is:

        (A_perm.T @ A_perm) = (A_c.T @ A_c)   for any row permutation

    so the inverse only needs to be computed once per (A, layer).

    Parameters
    ----------
    A:
        Activation matrix, shape (N, D).
    ridge:
        Relative ridge coefficient.

    Returns
    -------
    A_c:
        Column-centred A, shape (N, D).
    M:
        Regularised inverse Gram (A_c.T @ A_c + λI)^{-1}, shape (D, D).
    """
    A64 = A.astype(np.float64, copy=False)
    A_c = A64 - A64.mean(axis=0, keepdims=True)
    G = A_c.T @ A_c
    lam = _auto_lambda(G, ridge)
    G_reg = G + lam * np.eye(G.shape[0], dtype=np.float64)
    M = scipy.linalg.inv(G_reg)
    return A_c, M


def compute_residual_fast(
    A_perm: np.ndarray,
    B_c: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Compute the residual using the precomputed inverse Gram.

    Only the cross-covariance term A_perm.T @ B_c changes per null draw;
    the regularised inverse M is identical for all row permutations of A.

    Parameters
    ----------
    A_perm:
        Row-permuted (and centred) A, shape (N, D_A).
    B_c:
        Column-centred B, shape (N, D_B).  Must not be permuted.
    M:
        Precomputed (A_c.T @ A_c + λI)^{-1}, shape (D_A, D_A).

    Returns
    -------
    R:
        Residual of shape (N, D_B).
    """
    W = M @ (A_perm.T @ B_c)  # (D_A, D_B)
    return B_c - A_perm @ W   # (N, D_B)


# ---------------------------------------------------------------------------
# Fragmentation scores
# ---------------------------------------------------------------------------

def frag_global(
    A: np.ndarray,
    B: np.ndarray,
    ridge: float = 1e-4,
) -> float:
    """Fraction of B's variance linearly unexplained by A.

    Parameters
    ----------
    A:
        Left activation matrix, shape (N, D_A).
    B:
        Right activation matrix, shape (N, D_B).
    ridge:
        Relative ridge coefficient.

    Returns
    -------
    float in [0, 1].  Values close to 0 indicate strong linear alignment;
    values close to 1 indicate near-zero alignment.
    """
    R, _, B_c = compute_residual(A, B, ridge=ridge)
    norm_b = float(np.sum(B_c ** 2))
    if norm_b < 1e-12:
        return float("nan")
    return float(np.sum(R ** 2) / norm_b)


def frag_cka(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    ridge: float = 1e-4,
    denom_threshold: float | None = None,
) -> dict:
    """Fraction of B–C shared structure remaining in the residual of B w.r.t. A.

    This is the primary RQ2 fragmentation metric.  Only CKA is used for the
    normalised ratio because CKA decomposes as normalised HSIC, making the
    ratio interpretable as a fraction of shared structure.

    Parameters
    ----------
    A:
        Left activation matrix, shape (N, D_A).
    B:
        Right activation matrix, shape (N, D_B).
    C:
        Witness activation matrix, shape (N, D_C).  May differ in D from B.
    ridge:
        Relative ridge coefficient.
    denom_threshold:
        Minimum value of CKA(B, C) required to consider the ratio valid.
        If None, no threshold is applied (ratio computed regardless).

    Returns
    -------
    dict with keys:
        ``frag_cka``  — float (NaN if invalid)
        ``denom_cka`` — float, CKA(B, C)
        ``num_cka``   — float, CKA(R, C)
        ``valid``     — bool
    """
    R, _, _ = compute_residual(A, B, ridge=ridge)
    denom = float(linear_cka(B, C, center_gram=True, backend="numpy"))
    num = float(linear_cka(R, C, center_gram=True, backend="numpy"))

    valid = True
    if denom_threshold is not None and denom < denom_threshold:
        valid = False

    frag = float("nan") if not valid else (num / denom if denom > 1e-12 else float("nan"))
    return {"frag_cka": frag, "denom_cka": denom, "num_cka": num, "valid": valid}


def sim_raw(
    R: np.ndarray,
    C: np.ndarray,
    metric: str,
    cfg: dict | None = None,
) -> float:
    """Raw (unnormalised) similarity between a residual matrix and a witness.

    Used for PWCCA and kNN triangulation.  For these metrics the ratio
    interpretation does not hold, so raw values are reported separately
    alongside sim(B, C) for directional comparison.

    Parameters
    ----------
    R:
        Residual or activation matrix, shape (N, D_R).
    C:
        Witness activation matrix, shape (N, D_C).
    metric:
        One of ``"cka"``, ``"pwcca"``, ``"knn"``.
    cfg:
        Optional metric config dict forwarded to the underlying metric
        function.  Pass ``{"pca": {"enabled": False}}`` when matrices
        have already been PCA-reduced to avoid re-running SVD.

    Returns
    -------
    float scalar.
    """
    cfg = cfg or {}
    metric = str(metric).lower()
    if metric == "cka":
        return float(linear_cka(R, C, center_gram=True, backend="numpy"))
    if metric == "pwcca":
        return float(pwcca(R, C, cfg=cfg))
    if metric == "knn":
        return float(knn_overlap(R, C, cfg=cfg))
    raise ValueError(f"Unknown metric '{metric}'. Expected 'cka', 'pwcca', or 'knn'.")
