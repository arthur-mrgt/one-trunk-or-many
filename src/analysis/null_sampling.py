"""Sampling and caching primitives for null-distribution draws."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.analysis.null_artifacts import load_vectors
from src.metrics.registry import build_metric

log = logging.getLogger(__name__)


@dataclass
class HypothesisCache:
    """Preloaded tensors and constraints for one `(pair, layer, metric)` key."""

    pair: str
    layer: str
    metric: str
    observed_value: float
    observed_n_samples: int
    target_n_samples: int
    left_vectors: np.ndarray
    right_vectors: np.ndarray
    left_scene_ids: np.ndarray
    right_scene_ids: np.ndarray
    right_candidates_by_left_index: dict[int, np.ndarray]
    metric_fn: Any


def prepare_hypothesis_caches(
    activation_index: pd.DataFrame,
    observed_metrics: pd.DataFrame,
    metrics_cfg: dict[str, Any],
    null_metrics_enabled: list[str] | None,
    sample_size_mode: str,
    sample_size_value: Any,
    min_scenes: int,
    sampling_mode: str,
    scene_type_map: dict[str, str],
) -> dict[tuple[str, str, str], HypothesisCache]:
    """Build per-hypothesis caches from activation index and observed rows."""
    caches: dict[tuple[str, str, str], HypothesisCache] = {}
    cka_cfg = metrics_cfg.get("cka", {})
    allowed_metrics = (
        {str(m) for m in null_metrics_enabled}
        if null_metrics_enabled
        else {str(m) for m in metrics_cfg.get("enabled", ["cka"])}
    )

    obs_iter = tqdm(
        list(observed_metrics.iterrows()),
        desc="NullPrep[caches]",
        unit="hyp",
        total=len(observed_metrics),
    )
    for _, obs in obs_iter:
        pair = str(obs["pair"])
        layer = str(obs["layer"])
        metric = str(obs["metric"])
        if metric not in allowed_metrics:
            continue
        observed_value = float(obs["value"])
        observed_n = int(obs["n_samples"])
        left_mod, right_mod = tuple(pair.split("-", 1))

        pair_df = activation_index[activation_index["pair"] == pair]
        left_df = pair_df[
            (pair_df["layer"].astype(str) == layer)
            & (pair_df["modality"] == left_mod)
        ].reset_index(drop=True)
        right_df = pair_df[
            (pair_df["layer"].astype(str) == layer)
            & (pair_df["modality"] == right_mod)
        ].reset_index(drop=True)
        if left_df.empty or right_df.empty:
            continue
        # The min_scenes guard only makes sense for cross-scene modes; the
        # within-scene mode explicitly needs multiple frames *per* scene, not
        # multiple scenes.
        if sampling_mode != "within_scene_random":
            if left_df["scene_id"].nunique() < min_scenes or right_df["scene_id"].nunique() < min_scenes:
                continue

        left_vectors = load_vectors(left_df["activation_path"].tolist())
        right_vectors = load_vectors(right_df["activation_path"].tolist())
        if left_vectors is None or right_vectors is None:
            continue

        metric_cfg = dict(metrics_cfg.get(metric, {}))
        pca_cfg = dict(metric_cfg.get("pca", {}))
        if bool(pca_cfg.get("enabled", False)):
            left_vectors, right_vectors = project_pca(
                left_vectors=left_vectors,
                right_vectors=right_vectors,
                n_components=int(pca_cfg.get("n_components", 64)),
                shared_basis=bool(pca_cfg.get("shared_basis", False)),
                backend=str(pca_cfg.get("backend", "numpy")),
                device=str(pca_cfg.get("device", "auto")),
            )

        metrics_cfg_for_null = dict(metrics_cfg)
        metric_entry = dict(metrics_cfg_for_null.get(metric, {}))
        metric_entry["pca"] = {"enabled": False}
        metrics_cfg_for_null[metric] = metric_entry
        metric_fn = build_metric(metric_name=metric, cka_cfg=cka_cfg, metrics_cfg=metrics_cfg_for_null)

        left_scene_ids = left_df["scene_id"].astype(str).to_numpy()
        right_scene_ids = right_df["scene_id"].astype(str).to_numpy()
        left_sample_keys = left_df["sample_key"].astype(str).to_numpy()
        right_sample_keys = right_df["sample_key"].astype(str).to_numpy()
        candidates = build_right_candidates_by_left_index(
            left_scene_ids=left_scene_ids,
            right_scene_ids=right_scene_ids,
            sampling_mode=sampling_mode,
            scene_type_map=scene_type_map,
            left_sample_keys=left_sample_keys,
            right_sample_keys=right_sample_keys,
        )
        if not candidates:
            continue

        target_n_samples = resolve_target_sample_size(
            mode=sample_size_mode,
            value=sample_size_value,
            observed_n_samples=observed_n,
        )
        key = (pair, layer, metric)
        caches[key] = HypothesisCache(
            pair=pair,
            layer=layer,
            metric=metric,
            observed_value=observed_value,
            observed_n_samples=observed_n,
            target_n_samples=target_n_samples,
            left_vectors=left_vectors,
            right_vectors=right_vectors,
            left_scene_ids=left_scene_ids,
            right_scene_ids=right_scene_ids,
            right_candidates_by_left_index=candidates,
            metric_fn=metric_fn,
        )
    return caches


SUPPORTED_SAMPLING_MODES: frozenset[str] = frozenset({
    "cross_scene_type_random",     # different scene_id AND different scene_type  (Null A — easiest)
    "within_scene_type_random",    # different scene_id AND same scene_type       (Null B — medium)
    "within_scene_random",         # same scene_id AND different sample_key       (Null C — hardest)
})

MODES_NEEDING_TYPE_MAP: frozenset[str] = frozenset({
    "cross_scene_type_random",
    "within_scene_type_random",
})


def build_right_candidates_by_left_index(
    left_scene_ids: np.ndarray,
    right_scene_ids: np.ndarray,
    sampling_mode: str,
    scene_type_map: dict[str, str],
    left_sample_keys: np.ndarray | None = None,
    right_sample_keys: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Precompute valid right indices for each left index, mode-aware.

    Modes:
      * ``cross_scene_type_random``— right scene_id ≠ left scene_id AND
        right type ≠ left type. If a particular LEFT has no different-type
        partner (e.g. its type is missing from the map), it degrades to any
        different-scene partner for that LEFT only — never silently across
        the whole run.
      * ``within_scene_type_random``— right scene_id ≠ left scene_id AND
        right type == left type. No fallback (the type match is the point).
      * ``within_scene_random``     — right scene_id == left scene_id AND
        right sample_key ≠ left sample_key. Requires ``*_sample_keys``.
    """
    if sampling_mode not in SUPPORTED_SAMPLING_MODES:
        raise ValueError(
            f"Unknown sampling_mode '{sampling_mode}'. "
            f"Supported: {sorted(SUPPORTED_SAMPLING_MODES)}."
        )
    if sampling_mode == "within_scene_random" and (
        left_sample_keys is None or right_sample_keys is None
    ):
        raise ValueError(
            "sampling_mode=within_scene_random requires left/right sample_keys."
        )

    out: dict[int, np.ndarray] = {}
    right_all_idx = np.arange(len(right_scene_ids), dtype=int)
    different_scene_mask_cache: dict[str, np.ndarray] = {}

    for left_idx, left_scene in enumerate(left_scene_ids.tolist()):
        left_scene_str = str(left_scene)

        if sampling_mode == "within_scene_random":
            same_scene = right_scene_ids == left_scene
            different_key = right_sample_keys != left_sample_keys[left_idx]
            valid = right_all_idx[same_scene & different_key]
            if len(valid):
                out[left_idx] = valid
            continue

        # cross-scene base mask (cached per scene to avoid rebuilds)
        mask = different_scene_mask_cache.get(left_scene_str)
        if mask is None:
            mask = right_scene_ids != left_scene
            different_scene_mask_cache[left_scene_str] = mask
        valid = right_all_idx[mask]
        if len(valid) == 0:
            continue

        # cross_scene_type_random or within_scene_type_random
        left_type = scene_type_map.get(left_scene_str)
        if not left_type:
            if sampling_mode == "cross_scene_type_random":
                out[left_idx] = valid  # per-LEFT fallback when this scene's type is unknown
            continue

        if sampling_mode == "cross_scene_type_random":
            typed = [
                ridx for ridx in valid.tolist()
                if scene_type_map.get(str(right_scene_ids[ridx]), left_type) != left_type
            ]
            valid = np.array(typed, dtype=int) if typed else valid  # per-LEFT fallback
        else:  # within_scene_type_random
            typed = [
                ridx for ridx in valid.tolist()
                if scene_type_map.get(str(right_scene_ids[ridx])) == left_type
            ]
            valid = np.array(typed, dtype=int) if typed else np.array([], dtype=int)

        if len(valid):
            out[left_idx] = valid
    return out


def draw_mismatched_image_level(
    cache: HypothesisCache,
    n_draws: int,
    replace: bool,
    sampling_mode: str,
    scene_type_map: dict[str, str],
    rng: np.random.Generator,
    start_draw_id: int,
    run_id: str,
    seed: int,
    draw_batch_id: int,
) -> list[dict[str, Any]]:
    """Generate mismatched image-level null draws for one hypothesis cache."""
    rows: list[dict[str, Any]] = []
    left_count = len(cache.left_vectors)
    right_count = len(cache.right_vectors)
    if left_count == 0 or right_count == 0:
        return rows

    left_all_idx = np.arange(left_count, dtype=int)
    for draw_offset in range(n_draws):
        n_target = int(cache.target_n_samples)
        if not replace:
            n_target = min(n_target, left_count, right_count)
        if n_target <= 0:
            continue

        selected_left: list[int] = []
        selected_right: list[int] = []
        used_left: set[int] = set()
        used_right: set[int] = set()
        attempts = 0
        max_attempts = max(1000, n_target * 20)

        while len(selected_left) < n_target and attempts < max_attempts:
            attempts += 1
            left_idx = int(rng.choice(left_all_idx))
            if not replace and left_idx in used_left:
                continue
            candidates = cache.right_candidates_by_left_index.get(left_idx)
            if candidates is None or len(candidates) == 0:
                continue
            if not replace:
                candidates = np.array(
                    [c for c in candidates.tolist() if c not in used_right],
                    dtype=int,
                )
                if len(candidates) == 0:
                    continue

            right_idx = int(rng.choice(candidates))
            selected_left.append(left_idx)
            selected_right.append(right_idx)
            if not replace:
                used_left.add(left_idx)
                used_right.add(right_idx)

        if len(selected_left) < 2:
            continue

        x = cache.left_vectors[np.array(selected_left, dtype=int)]
        y = cache.right_vectors[np.array(selected_right, dtype=int)]
        try:
            value = float(cache.metric_fn(x, y))
        except Exception as exc:
            log.debug("Null metric failed for (%s, %s, %s): %s", cache.pair, cache.layer, cache.metric, exc)
            continue

        left_scene = str(cache.left_scene_ids[selected_left[0]])
        right_scene = str(cache.right_scene_ids[selected_right[0]])
        left_type = scene_type_map.get(left_scene, "unknown")
        right_type = scene_type_map.get(right_scene, "unknown")
        rows.append(
            {
                "run_id": run_id,
                "pair": cache.pair,
                "layer": cache.layer,
                "metric": cache.metric,
                "draw_id": int(start_draw_id + draw_offset),
                "draw_batch_id": int(draw_batch_id),
                "null_value": value,
                "sampling": sampling_mode,
                "seed": int(seed),
                "n_samples": int(len(selected_left)),
                "left_scene_id": left_scene,
                "right_scene_id": right_scene,
                "left_scene_type": left_type,
                "right_scene_type": right_type,
                "is_cross_scene": bool(left_scene != right_scene),
                "is_cross_type": bool(
                    left_type != "unknown"
                    and right_type != "unknown"
                    and left_type != right_type
                ),
            }
        )
    return rows


def resolve_target_sample_size(
    mode: str,
    value: Any,
    observed_n_samples: int,
) -> int:
    """Resolve target sample size for each null draw."""
    mode_norm = str(mode).lower()
    if mode_norm == "observed_n_samples":
        return int(observed_n_samples)
    if mode_norm == "fixed":
        if value is None:
            raise ValueError(
                "analysis.null_distribution.sample_size_value is required when "
                "sample_size_mode=fixed."
            )
        return int(value)
    raise ValueError(
        f"Unknown sample_size_mode '{mode}'. "
        "Supported values: observed_n_samples, fixed."
    )


def project_pca(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    n_components: int,
    shared_basis: bool,
    backend: str = "numpy",
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Project vectors either with shared PCA or modality-specific PCA."""
    backend_norm = str(backend).lower()
    if backend_norm not in {"numpy", "torch"}:
        raise ValueError("PCA backend must be one of: numpy, torch.")
    if not shared_basis:
        return (
            project_single(left_vectors, n_components, backend=backend_norm, device=device),
            project_single(right_vectors, n_components, backend=backend_norm, device=device),
        )

    stacked = np.concatenate([left_vectors, right_vectors], axis=0).astype(np.float64)
    n_total, dim = stacked.shape
    k = min(int(n_components), n_total - 1, dim)
    if k <= 0:
        return left_vectors, right_vectors

    mean = stacked.mean(axis=0)
    _, _, vt = np.linalg.svd(stacked - mean, full_matrices=False)
    comps = vt[:k]
    left_proj = (left_vectors.astype(np.float64) - mean) @ comps.T
    right_proj = (right_vectors.astype(np.float64) - mean) @ comps.T
    return left_proj, right_proj


def project_single(
    vectors: np.ndarray,
    n_components: int,
    backend: str = "numpy",
    device: str = "auto",
) -> np.ndarray:
    """Project one matrix to at most `n_components` principal directions."""
    n, dim = vectors.shape
    k = min(int(n_components), n - 1, dim)
    if k <= 0:
        return vectors
    if backend == "torch":
        import torch

        resolved = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else str(device)
        t = torch.as_tensor(vectors, dtype=torch.float32, device=resolved)
        centered_t = t - t.mean(dim=0, keepdim=True)
        _, _, vt = torch.linalg.svd(centered_t, full_matrices=False)
        return (centered_t @ vt[:k].T).detach().cpu().numpy()
    centered = vectors.astype(np.float64) - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:k].T
