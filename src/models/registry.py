"""Model factory for encoder backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.model_fourm import FourMEncoder, FourMMockEncoder


def build_model(model_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> Any:
    """Instantiate the configured model backend."""
    backend = model_cfg["backend"]
    layers = model_cfg["layers"]
    if layers == "all" and backend == "fourm_mock":
        layers = [f"layer_{idx:02d}" for idx in range(12)]

    if backend == "fourm_mock":
        return FourMMockEncoder(
            layers=layers,
            embedding_dim=int(model_cfg["embedding_dim"]),
        )
    if backend == "fourm":
        return FourMEncoder(
            model_dir=Path(model_cfg["local_dir"]),
            hf_repo=str(model_cfg["hf_repo"]),
            layers=layers,
            runtime_cfg=runtime_cfg,
            model_cfg=model_cfg,
        )
    raise ValueError(f"Unknown model backend: {backend}")
