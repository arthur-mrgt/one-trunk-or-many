"""Linear CKA metric implementation."""

from __future__ import annotations

import numpy as np


def _center_gram(gram: np.ndarray) -> np.ndarray:
    """Center a Gram matrix in feature space."""
    n = gram.shape[0]
    unit = np.ones((n, n), dtype=gram.dtype) / n
    return gram - unit @ gram - gram @ unit + unit @ gram @ unit


def linear_cka(x: np.ndarray, y: np.ndarray, center_gram: bool = True) -> float:
    """Compute linear CKA between two activation matrices."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires same number of samples in x and y.")

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    gram_x = x @ x.T
    gram_y = y @ y.T

    if center_gram:
        gram_x = _center_gram(gram_x)
        gram_y = _center_gram(gram_y)

    hsic = float(np.sum(gram_x * gram_y))
    norm_x = float(np.sqrt(np.sum(gram_x * gram_x)))
    norm_y = float(np.sqrt(np.sum(gram_y * gram_y)))
    denom = max(norm_x * norm_y, 1e-12)
    return hsic / denom
