"""PCA dimensionality reduction utility for metric preprocessing.

Both PWCCA and k-NN overlap benefit from projecting high-dimensional
activations to a lower-dimensional space before computing distances or
canonical correlations.  This module provides a single reusable helper
that fits PCA on the union of both activation matrices and transforms each.
"""

from __future__ import annotations

import numpy as np


def pca_reduce(
    x: np.ndarray,
    y: np.ndarray,
    n_components: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project x and y to at most n_components dimensions via PCA.

    PCA is fitted on the row-wise concatenation [x; y] so that both
    matrices share the same projection basis.  This avoids information
    leakage that would occur if fitting on x or y alone.

    Parameters
    ----------
    x, y:
        Activation matrices of shape (N, d).  Must have the same number
        of columns (d) but may differ in number of rows.
    n_components:
        Target number of principal components.  Automatically clamped to
        ``min(n_components, N_total - 1, d)`` to satisfy PCA constraints,
        where ``N_total = len(x) + len(y)``.
    seed:
        Random seed passed to the SVD solver for reproducibility.

    Returns
    -------
    x_reduced, y_reduced:
        Projected matrices of shape (N, k) where k <= n_components.
    """
    n_total = x.shape[0] + y.shape[0]
    d = x.shape[1]
    k = min(n_components, n_total - 1, d)
    if k <= 0:
        return x, y

    # Stack, centre, compute economy SVD
    stacked = np.concatenate([x, y], axis=0).astype(np.float64)
    mean = stacked.mean(axis=0)
    stacked -= mean

    rng = np.random.default_rng(seed)
    # Use a deterministic random state for the SVD solver
    _ = rng  # reserved for future randomised SVD

    _, _, vt = np.linalg.svd(stacked, full_matrices=False)
    components = vt[:k]          # (k, d)

    x_proj = (x.astype(np.float64) - mean) @ components.T
    y_proj = (y.astype(np.float64) - mean) @ components.T
    return x_proj, y_proj


def maybe_reduce(
    x: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply PCA reduction if enabled in cfg, otherwise return x and y as-is.

    Expected cfg keys:
        pca.enabled   (bool, default False)
        pca.n_components (int, default 64)
        pca.seed      (int, default 0)
    """
    pca_cfg = cfg.get("pca", {})
    if not pca_cfg.get("enabled", False):
        return x, y
    n_components = int(pca_cfg.get("n_components", 64))
    seed = int(pca_cfg.get("seed", 0))
    return pca_reduce(x, y, n_components=n_components, seed=seed)
