"""PCA dimensionality reduction utilities for metric preprocessing."""

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


def pca_reduce_independent(
    x: np.ndarray,
    y: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project x and y with separate PCA bases.

    This is generally preferable for representation-comparison metrics
    (PWCCA / kNN-overlap) because it preserves each space's geometry
    independently instead of forcing a shared basis.
    """
    return _pca_single(x, n_components=n_components), _pca_single(y, n_components=n_components)


def pca_reduce_independent_torch(
    x: np.ndarray,
    y: np.ndarray,
    n_components: int,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Project x and y with separate PCA bases using torch."""
    return (
        _pca_single_torch(x, n_components=n_components, device=device),
        _pca_single_torch(y, n_components=n_components, device=device),
    )


def _pca_single(a: np.ndarray, n_components: int) -> np.ndarray:
    """Project one matrix to at most n_components principal directions."""
    n, d = a.shape
    k = min(int(n_components), n - 1, d)
    if k <= 0:
        return a
    centered = a.astype(np.float64) - a.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:k].T


def _pca_single_torch(a: np.ndarray, n_components: int, device: str = "auto") -> np.ndarray:
    """Project one matrix with torch SVD and return numpy output."""
    import torch

    n, d = a.shape
    k = min(int(n_components), n - 1, d)
    if k <= 0:
        return a
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = str(device)
    t = torch.as_tensor(a, dtype=torch.float32, device=resolved)
    centered = t - t.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:k].T
    return proj.detach().cpu().numpy()


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
    shared_basis = bool(pca_cfg.get("shared_basis", False))
    backend = str(pca_cfg.get("backend", "numpy")).lower()
    device = str(pca_cfg.get("device", "auto")).lower()
    if backend not in {"numpy", "torch"}:
        raise ValueError("pca.backend must be one of: numpy, torch")
    if shared_basis:
        if backend == "torch":
            # Shared-basis torch backend falls back to numpy implementation.
            # For independent-basis metrics this path is generally not used.
            return pca_reduce(x, y, n_components=n_components, seed=seed)
        return pca_reduce(x, y, n_components=n_components, seed=seed)
    if backend == "torch":
        return pca_reduce_independent_torch(x, y, n_components=n_components, device=device)
    return pca_reduce_independent(x, y, n_components=n_components)
