"""k-NN overlap metric.

Measures local topology preservation between two representation spaces.

Mathematical summary
--------------------
Given activation matrices X (N × d1) and Y (N × d2):

1. Optionally reduce X and Y via PCA.
2. For each sample i, find its k nearest neighbours in X → set N_X(i).
3. For each sample i, find its k nearest neighbours in Y → set N_Y(i).
4. kNN overlap = (1/N) * sum_i  |N_X(i) ∩ N_Y(i)| / k

The result is a scalar in [0, 1] where:
  - 1.0 means every sample has the exact same k neighbours in both spaces
    (perfect local topology alignment)
  - 1/N approximates the random baseline (chance overlap)
"""

from __future__ import annotations

import warnings

import numpy as np

from src.metrics.reduction import maybe_reduce


def _knn_indices(x: np.ndarray, k: int) -> np.ndarray:
    """Return (N, k) array of k nearest neighbour indices for each row of x.

    Uses exact brute-force search (Euclidean distance).  Efficient enough
    for the activation matrix sizes used in this project (N < 10k, d < 512).
    """
    # Pairwise squared distances via expansion: ||a-b||^2 = ||a||^2 - 2a·b + ||b||^2
    sq_norms = (x ** 2).sum(axis=1)
    dists = sq_norms[:, None] - 2 * (x @ x.T) + sq_norms[None, :]
    np.fill_diagonal(dists, np.inf)   # exclude self
    return np.argpartition(dists, k, axis=1)[:, :k]


def _knn_indices_faiss(x: np.ndarray, k: int, cfg: dict) -> np.ndarray:
    """Return kNN indices with FAISS backend (exact or approximate)."""
    import faiss

    x32 = np.ascontiguousarray(x.astype(np.float32, copy=False))
    n, d = x32.shape
    mode = str(cfg.get("mode", "exact")).lower()
    use_gpu = bool(cfg.get("use_gpu", True))

    if mode == "exact":
        index = faiss.IndexFlatL2(d)
    elif mode == "approx":
        nlist = int(cfg.get("nlist", 128))
        nprobe = int(cfg.get("nprobe", 16))
        m = int(cfg.get("m", 16))
        nbits = int(cfg.get("nbits", 8))
        quantizer = faiss.IndexFlatL2(d)
        if m > 0 and d % m == 0:
            index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        else:
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        train_size = min(int(cfg.get("train_size", 20000)), n)
        train_idx = np.random.default_rng(int(cfg.get("seed", 0))).choice(n, size=train_size, replace=False)
        index.train(x32[train_idx])
        index.nprobe = nprobe
    else:
        raise ValueError("knn_overlap.faiss.mode must be one of: exact, approx")

    if use_gpu and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        gpu_device = int(cfg.get("gpu_device", 0))
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_device, index)

    index.add(x32)
    search_k = min(k + 1, n)
    _, neigh = index.search(x32, search_k)

    out = np.empty((n, k), dtype=int)
    for i in range(n):
        row = [int(j) for j in neigh[i].tolist() if int(j) >= 0 and int(j) != i]
        if len(row) < k:
            fallback = [j for j in range(n) if j != i and j not in row]
            row.extend(fallback[: max(0, k - len(row))])
        out[i] = np.array(row[:k], dtype=int)
    return out


def knn_overlap(
    x: np.ndarray,
    y: np.ndarray,
    cfg: dict | None = None,
) -> float:
    """Compute k-NN overlap between activation matrices x and y.

    Parameters
    ----------
    x, y:
        Activation matrices of shape (N, d).  Must have the same N.
    cfg:
        Metric config dict.  Recognised keys:
          k                  (int,   default 10)
          pca.enabled        (bool,  default True)
          pca.n_components   (int,   default 64)
          pca.seed           (int,   default 0)
          pca.shared_basis   (bool,  default False)

    Returns
    -------
    float in [0, 1].
    """
    if cfg is None:
        cfg = {}

    if x.shape[0] != y.shape[0]:
        raise ValueError("k-NN overlap requires the same number of samples in x and y.")

    n = x.shape[0]
    k: int = min(int(cfg.get("k", 10)), n - 1)
    if k <= 0:
        return 0.0

    # PCA reduction (enabled by default)
    pca_cfg = {
        "pca": {
            "enabled": True,
            "n_components": 64,
            "seed": 0,
            "shared_basis": False,
        }
    }
    pca_cfg["pca"].update(cfg.get("pca", {}))
    x_r, y_r = maybe_reduce(x, y, pca_cfg)
    backend = str(cfg.get("backend", "numpy")).lower()
    if backend == "faiss":
        faiss_cfg = dict(cfg.get("faiss", {}))
        try:
            nn_x = _knn_indices_faiss(x_r, k, faiss_cfg)  # (N, k)
            nn_y = _knn_indices_faiss(y_r, k, faiss_cfg)  # (N, k)
        except Exception as exc:
            warnings.warn(
                f"FAISS backend unavailable or failed ({exc}); falling back to numpy kNN.",
                RuntimeWarning,
                stacklevel=2,
            )
            nn_x = _knn_indices(x_r, k)
            nn_y = _knn_indices(y_r, k)
    elif backend == "numpy":
        nn_x = _knn_indices(x_r, k)
        nn_y = _knn_indices(y_r, k)
    else:
        raise ValueError("knn_overlap backend must be one of: numpy, faiss")

    # For each sample, count neighbours that appear in both sets
    overlaps = np.array([
        len(np.intersect1d(nn_x[i], nn_y[i]))
        for i in range(n)
    ])
    return float(overlaps.mean() / k)
