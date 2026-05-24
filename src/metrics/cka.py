"""Linear CKA metric implementation."""

from __future__ import annotations

import numpy as np


def _linear_cka_numpy(x: np.ndarray, y: np.ndarray, center_gram: bool = True) -> float:
    """Compute linear CKA on CPU with feature-space formulation."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires same number of samples in x and y.")
    if x.shape[0] < 2:
        return 0.0

    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    if center_gram:
        x64 = x64 - x64.mean(axis=0, keepdims=True)
        y64 = y64 - y64.mean(axis=0, keepdims=True)

    xty = x64.T @ y64
    xtx = x64.T @ x64
    yty = y64.T @ y64

    hsic = float(np.sum(xty * xty))
    norm_x = float(np.sqrt(np.sum(xtx * xtx)))
    norm_y = float(np.sqrt(np.sum(yty * yty)))
    denom = max(norm_x * norm_y, 1e-12)
    return hsic / denom


def _linear_cka_torch(
    x: np.ndarray,
    y: np.ndarray,
    center_gram: bool = True,
    device: str = "cuda",
) -> float:
    """Compute linear CKA on GPU with PyTorch when available."""
    import torch

    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires same number of samples in x and y.")
    if x.shape[0] < 2:
        return 0.0

    tx = torch.as_tensor(x, device=device, dtype=torch.float32)
    ty = torch.as_tensor(y, device=device, dtype=torch.float32)
    if center_gram:
        tx = tx - tx.mean(dim=0, keepdim=True)
        ty = ty - ty.mean(dim=0, keepdim=True)

    xty = tx.T @ ty
    xtx = tx.T @ tx
    yty = ty.T @ ty

    hsic = torch.sum(xty * xty)
    norm_x = torch.sqrt(torch.sum(xtx * xtx))
    norm_y = torch.sqrt(torch.sum(yty * yty))
    denom = torch.clamp(norm_x * norm_y, min=1e-12)
    return float((hsic / denom).detach().cpu().item())


def linear_cka(
    x: np.ndarray,
    y: np.ndarray,
    center_gram: bool = True,
    backend: str = "auto",
    device: str = "auto",
) -> float:
    """Compute linear CKA between two activation matrices.

    Parameters
    ----------
    backend:
        ``"auto"`` (default), ``"numpy"``, or ``"torch"``.
    device:
        Device for torch backend: ``"auto"``, ``"cuda"``, or ``"cpu"``.
    """
    backend_norm = str(backend).lower()
    if backend_norm not in {"auto", "numpy", "torch"}:
        raise ValueError("CKA backend must be one of: auto, numpy, torch.")

    if backend_norm in {"auto", "torch"}:
        try:
            import torch

            if str(device).lower() == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                resolved_device = str(device).lower()
            if backend_norm == "torch" or resolved_device == "cuda":
                return _linear_cka_torch(
                    x=x,
                    y=y,
                    center_gram=center_gram,
                    device=resolved_device,
                )
        except Exception:
            if backend_norm == "torch":
                raise

    return _linear_cka_numpy(x=x, y=y, center_gram=center_gram)
