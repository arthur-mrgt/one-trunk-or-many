"""Null distribution computation via cross-scene mismatched sampling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.metrics.registry import build_metric
from src.utils.io import write_optional_parquet, write_table

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_null_distribution(
    activation_index: pd.DataFrame,
    null_cfg: dict[str, Any],
    metrics_cfg: dict[str, Any],
    run_id: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Compute null CKA distributions via cross-scene mismatched sampling.

    For each (pair, layer, metric), draws ``n_draws`` mismatched samples
    where left and right activations come from *different* scenes.
    When ``sampling=cross_scene_type_random`` and a metadata file is
    provided, left and right scenes are additionally required to have
    *different scene types* (e.g. Bathroom vs Office).

    Parameters
    ----------
    activation_index:
        The activation index produced by the extraction stage.
    null_cfg:
        The ``analysis.null_distribution`` config dict.
    metrics_cfg:
        The ``metrics`` config dict.
    run_id:
        Run identifier written into every output row.
    out_dir:
        Directory where ``null_distribution.csv`` (and optional
        ``null_distribution.parquet``) are written.
    """
    n_draws: int = int(null_cfg.get("n_draws", 1000))
    seed: int = int(null_cfg.get("seed", 42))
    replace: bool = bool(null_cfg.get("replace", True))
    min_scenes: int = int(null_cfg.get("min_scenes", 2))
    sampling: str = str(null_cfg.get("sampling", "cross_scene_type_random"))
    save_parquet: bool = bool(metrics_cfg.get("output", {}).get("save_parquet", False))

    scene_type_map = _load_scene_type_map(null_cfg)
    use_type_constraint = (
        sampling == "cross_scene_type_random" and bool(scene_type_map)
    )
    if use_type_constraint:
        log.info("Cross-scene-type sampling enabled (%d scenes with type info).",
                 len(scene_type_map))
    elif sampling == "cross_scene_type_random":
        log.warning(
            "sampling=cross_scene_type_random requested but no metadata loaded. "
            "Falling back to cross_scene_random."
        )

    rng = np.random.default_rng(seed)
    all_rows: list[dict[str, Any]] = []

    pairs = activation_index["pair"].unique().tolist()
    layers = sorted(activation_index["layer"].unique().tolist())
    metric_names: list[str] = list(metrics_cfg.get("enabled", ["cka"]))
    cka_cfg: dict = metrics_cfg.get("cka", {})

    for pair in pairs:
        left_mod, right_mod = tuple(pair.split("-", 1))
        pair_idx = activation_index[activation_index["pair"] == pair]

        scenes = sorted(pair_idx["scene_id"].unique().tolist())
        if len(scenes) < min_scenes:
            log.warning(
                "Pair '%s' has only %d scene(s) — need at least %d for "
                "cross-scene mismatching. Skipping.",
                pair, len(scenes), min_scenes,
            )
            continue

        log.info(
            "Null distribution: pair=%s | %d scenes | %d layers | %d draws each.",
            pair, len(scenes), len(layers), n_draws,
        )

        layer_iter = tqdm(layers, desc=f"Null[{pair}]", unit="layer")
        for layer in layer_iter:
            left_pool = pair_idx[
                (pair_idx["layer"] == layer) & (pair_idx["modality"] == left_mod)
            ].reset_index(drop=True)
            right_pool = pair_idx[
                (pair_idx["layer"] == layer) & (pair_idx["modality"] == right_mod)
            ].reset_index(drop=True)

            if left_pool.empty or right_pool.empty:
                continue

            for metric_name in metric_names:
                metric_fn = build_metric(metric_name=metric_name, cka_cfg=cka_cfg)
                null_values = _draw_mismatched_cka(
                    left_pool=left_pool,
                    right_pool=right_pool,
                    metric_fn=metric_fn,
                    n_draws=n_draws,
                    replace=replace,
                    rng=rng,
                    scene_type_map=scene_type_map if use_type_constraint else {},
                )
                for draw_id, val in enumerate(null_values):
                    all_rows.append({
                        "run_id": run_id,
                        "pair": pair,
                        "layer": layer,
                        "metric": metric_name,
                        "draw_id": draw_id,
                        "null_value": float(val),
                        "n_samples": len(null_values),
                        "sampling": sampling,
                        "seed": seed,
                    })

    null_df = pd.DataFrame.from_records(all_rows) if all_rows else pd.DataFrame()
    csv_path = out_dir / "null_distribution.csv"
    write_table(csv_path, null_df)
    write_optional_parquet(out_dir / "null_distribution.parquet", null_df,
                           enabled=save_parquet)
    log.info("Null distribution saved: %d rows → %s", len(null_df), csv_path)
    return null_df


# ---------------------------------------------------------------------------
# Scene-type metadata
# ---------------------------------------------------------------------------


def _load_scene_type_map(null_cfg: dict[str, Any]) -> dict[str, str]:
    """Load scene_id → scene_type mapping from the Hypersim metadata CSV.

    The CSV has an ``Animation`` column with values like ``ai_001_001_cam_00``
    and a ``Scene type`` column with values like ``Bathroom``.
    Scene ID is extracted as the first three ``_``-separated parts.
    """
    metadata_path = null_cfg.get("metadata_path")
    if not metadata_path:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        log.warning("Scene type metadata not found at '%s'. Type constraint disabled.", path)
        return {}

    try:
        df = pd.read_csv(path)
        if "Animation" not in df.columns or "Scene type" not in df.columns:
            log.warning("Expected 'Animation' and 'Scene type' columns in '%s'.", path)
            return {}

        mapping: dict[str, str] = {}
        for _, row in df.iterrows():
            anim = str(row["Animation"])                    # ai_001_001_cam_00
            scene_id = "_".join(anim.split("_")[:3])       # ai_001_001
            scene_type = str(row["Scene type"]).strip()
            if scene_id and scene_type and scene_type.lower() not in ("nan", ""):
                mapping[scene_id] = scene_type

        log.info("Loaded scene types for %d scenes from '%s'.", len(mapping), path)
        return mapping
    except Exception as exc:
        log.warning("Could not load scene type metadata: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _draw_mismatched_cka(
    left_pool: pd.DataFrame,
    right_pool: pd.DataFrame,
    metric_fn: Any,
    n_draws: int,
    replace: bool,
    rng: np.random.Generator,
    scene_type_map: dict[str, str],
) -> list[float]:
    """Draw n_draws mismatched CKA values.

    Each draw picks a left scene and a right scene that:
      1. Have different scene IDs (always enforced).
      2. Have different scene types (enforced when scene_type_map is provided).

    Falls back to cross-scene-only mismatching if no cross-type pair exists.
    """
    null_values: list[float] = []
    left_scenes = np.array(sorted(left_pool["scene_id"].unique()))
    right_scenes = np.array(sorted(right_pool["scene_id"].unique()))
    attempts = 0
    max_attempts = n_draws * 10

    while len(null_values) < n_draws and attempts < max_attempts:
        attempts += 1

        left_scene = rng.choice(left_scenes)
        left_type = scene_type_map.get(left_scene)

        # Candidates: different scene ID
        candidates = right_scenes[right_scenes != left_scene]

        # Tighten to different scene type when possible
        if left_type and scene_type_map:
            type_filtered = np.array([
                s for s in candidates
                if scene_type_map.get(s, left_type) != left_type
            ])
            if len(type_filtered) > 0:
                candidates = type_filtered
            else:
                log.debug(
                    "No cross-type candidate found for scene '%s' (type=%s). "
                    "Falling back to cross-scene-only.",
                    left_scene, left_type,
                )

        if len(candidates) == 0:
            break

        right_scene = rng.choice(candidates)

        left_rows = left_pool[left_pool["scene_id"] == left_scene]
        right_rows = right_pool[right_pool["scene_id"] == right_scene]
        if left_rows.empty or right_rows.empty:
            continue

        x = _load_vectors(left_rows["activation_path"].tolist())
        y = _load_vectors(right_rows["activation_path"].tolist())
        if x is None or y is None:
            continue

        # CKA requires equal sample counts; truncate to the shorter scene.
        if x.shape[0] != y.shape[0]:
            n = min(x.shape[0], y.shape[0])
            rng.shuffle(x)
            rng.shuffle(y)
            x, y = x[:n], y[:n]

        try:
            null_values.append(float(metric_fn(x, y)))
        except Exception as exc:
            log.debug("Metric failed for a null draw: %s", exc)

    if len(null_values) < n_draws:
        log.warning(
            "Only obtained %d / %d null draws after %d attempts.",
            len(null_values), n_draws, attempts,
        )
    return null_values


def _load_vectors(paths: list[str]) -> np.ndarray | None:
    """Stack activation .npy files into an (N, d) matrix."""
    try:
        return np.stack([np.load(Path(p)) for p in paths], axis=0)
    except Exception as exc:
        log.debug("Failed to load activation vectors: %s", exc)
        return None
