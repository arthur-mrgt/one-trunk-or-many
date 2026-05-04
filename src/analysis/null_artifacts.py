"""I/O and metadata helpers for null-distribution workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import write_optional_parquet, write_table

log = logging.getLogger(__name__)


def null_draw_counts(null_df: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    """Count existing null draws per `(pair, layer, metric)` key."""
    if null_df.empty:
        return {}
    grouped = null_df.groupby(["pair", "layer", "metric"]).size()
    return {(str(k[0]), str(k[1]), str(k[2])): int(v) for k, v in grouped.items()}


def write_null_artifacts(
    csv_path: Path,
    state_path: Path,
    null_df: pd.DataFrame,
    save_parquet: bool,
    stop_reason: str,
    batches_completed: int,
) -> None:
    """Persist null table and run-state checkpoint to disk."""
    write_table(csv_path, null_df)
    write_optional_parquet(
        csv_path.with_suffix(".parquet"),
        null_df,
        enabled=save_parquet,
    )
    payload = {
        "artifact_path": str(csv_path),
        "n_rows": int(len(null_df)),
        "stop_reason": str(stop_reason),
        "batches_completed": int(batches_completed),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scene_type_map(null_cfg: dict[str, Any]) -> dict[str, str]:
    """Load `scene_id -> scene_type` mapping from Hypersim metadata CSV."""
    metadata_path = null_cfg.get("metadata_path")
    if not metadata_path:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        log.warning("Scene type metadata not found at '%s'.", path)
        return {}

    try:
        df = pd.read_csv(path)
        if "Animation" not in df.columns or "Scene type" not in df.columns:
            return {}
        mapping: dict[str, str] = {}
        for _, row in df.iterrows():
            anim = str(row["Animation"])
            scene_id = "_".join(anim.split("_")[:3])
            scene_type = str(row["Scene type"]).strip()
            if scene_id and scene_type and scene_type.lower() not in {"nan", ""}:
                mapping[scene_id] = scene_type
        return mapping
    except Exception as exc:
        log.warning("Could not parse scene-type metadata from '%s': %s", path, exc)
        return {}


def load_vectors(paths: list[str]) -> np.ndarray | None:
    """Load activation vectors from `.npy` paths and stack them."""
    try:
        return np.stack([np.load(Path(p)) for p in paths], axis=0)
    except Exception as exc:
        log.debug("Failed to load activation vectors: %s", exc)
        return None
