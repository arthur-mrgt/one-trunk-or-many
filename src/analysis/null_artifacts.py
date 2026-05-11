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


def load_scene_type_map(
    null_cfg: dict[str, Any],
    activation_index: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Return a `scene_id -> scene_type` mapping for null sampling.

    Two sources, tried in order:
      1. ``metadata_path`` → Hypersim-style CSV with ``Animation`` and
         ``Scene type`` columns.
      2. ``scene_type_source: scene_id_parent`` → derive the type as the
         parent path of each ``scene_id`` in ``activation_index``
         (DIODE: ``<env>/<scene>/<scan>`` ↦ ``<env>/<scene>``). Flat
         scene IDs (no ``/``) are skipped, so Hypersim is unaffected.
    Returns an empty map if neither applies; the caller degrades
    ``cross_scene_type_random`` → ``cross_scene_random``.
    """
    metadata_path = null_cfg.get("metadata_path")
    if metadata_path and Path(metadata_path).exists():
        df = pd.read_csv(metadata_path)
        if "Animation" not in df.columns or "Scene type" not in df.columns:
            return {}
        return {
            scene_id: scene_type
            for anim, raw_type in zip(df["Animation"], df["Scene type"])
            if (scene_id := "_".join(str(anim).split("_")[:3]))
            and (scene_type := str(raw_type).strip())
            and scene_type.lower() != "nan"
        }

    if null_cfg.get("scene_type_source") == "scene_id_parent" and activation_index is not None:
        return {
            sid: sid.rsplit("/", 1)[0]
            for sid in (str(s) for s in activation_index["scene_id"].dropna().unique())
            if "/" in sid
        }

    return {}


def load_vectors(paths: list[str]) -> np.ndarray | None:
    """Load activation vectors from `.npy` paths and stack them."""
    try:
        return np.stack([np.load(Path(p)) for p in paths], axis=0)
    except Exception as exc:
        log.debug("Failed to load activation vectors: %s", exc)
        return None
