"""Adaptive null-distribution runner with checkpointing and resume."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.analysis.null_artifacts import (
    load_scene_type_map,
    null_draw_counts,
    write_null_artifacts,
)
from src.analysis.null_sampling import draw_mismatched_image_level, prepare_hypothesis_caches
from src.analysis.null_stop import (
    build_observed_stub,
    evaluate_adaptive_stop,
    normalize_observed_metrics,
)

log = logging.getLogger(__name__)


@dataclass
class NullRunResult:
    """Result object returned by adaptive null execution."""

    null_df: pd.DataFrame
    artifact_path: Path
    state_path: Path
    stop_reason: str
    is_partial: bool
    batches_completed: int


def compute_null_distribution(
    activation_index: pd.DataFrame,
    null_cfg: dict[str, Any],
    metrics_cfg: dict[str, Any],
    run_id: str,
    out_dir: Path,
    out_filename: str = "null_distribution.csv",
) -> pd.DataFrame:
    """Backward-compatible wrapper for fixed-budget null draws."""
    observed_stub = build_observed_stub(activation_index=activation_index, metrics_cfg=metrics_cfg)
    result = compute_null_distribution_adaptive(
        activation_index=activation_index,
        observed_metrics=observed_stub,
        null_cfg={**null_cfg, "adaptive_stop": {"enabled": False}},
        metrics_cfg=metrics_cfg,
        run_id=run_id,
        out_dir=out_dir,
        out_filename=out_filename,
        existing_null_df=None,
    )
    return result.null_df


def compute_null_distribution_adaptive(
    activation_index: pd.DataFrame,
    observed_metrics: pd.DataFrame,
    null_cfg: dict[str, Any],
    metrics_cfg: dict[str, Any],
    run_id: str,
    out_dir: Path,
    out_filename: str = "null_distribution.csv",
    existing_null_df: pd.DataFrame | None = None,
) -> NullRunResult:
    """Compute or resume adaptive null draws until stop criterion is satisfied."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / out_filename
    state_path = out_dir / str(null_cfg.get("state_filename", "null_distribution_state.json"))

    save_parquet = bool(metrics_cfg.get("output", {}).get("save_parquet", False))
    replace = bool(null_cfg.get("replace", True))
    seed = int(null_cfg.get("seed", 42))
    sampling_mode = str(null_cfg.get("sampling_mode", null_cfg.get("sampling", "cross_scene_type_random")))
    min_scenes = int(null_cfg.get("min_scenes", 2))
    draws_per_batch = int(null_cfg.get("draws_per_batch", 200))
    checkpoint_every_batches = int(null_cfg.get("checkpoint_every_batches", 1))
    legacy_n_draws = int(null_cfg.get("n_draws", 1000))
    min_total_draws = int(null_cfg.get("min_total_draws", max(1, draws_per_batch)))
    max_total_draws = int(null_cfg.get("max_total_draws", legacy_n_draws))
    max_batches = int(null_cfg.get("max_batches", 0))

    adaptive_cfg = dict(null_cfg.get("adaptive_stop", {}))
    adaptive_enabled = bool(adaptive_cfg.get("enabled", True))
    alpha = float(adaptive_cfg.get("alpha", 0.05))
    correction = str(adaptive_cfg.get("correction", "bh_fdr"))
    require_all = bool(adaptive_cfg.get("require_all_hypotheses", True))
    on_keyboard_interrupt = str(null_cfg.get("on_keyboard_interrupt", "save_and_continue"))

    if draws_per_batch <= 0:
        raise ValueError("analysis.null_distribution.draws_per_batch must be > 0.")
    if checkpoint_every_batches <= 0:
        raise ValueError("analysis.null_distribution.checkpoint_every_batches must be > 0.")
    if max_total_draws <= 0:
        raise ValueError("analysis.null_distribution.max_total_draws must be > 0.")

    scene_type_map = load_scene_type_map(null_cfg)
    use_type_constraint = sampling_mode == "cross_scene_type_random" and bool(scene_type_map)
    if sampling_mode == "cross_scene_type_random" and not use_type_constraint:
        log.warning(
            "sampling_mode=cross_scene_type_random requested but metadata is unavailable; "
            "falling back to cross_scene_random."
        )

    observed = normalize_observed_metrics(observed_metrics, metrics_cfg)
    if observed.empty:
        write_null_artifacts(
            csv_path=csv_path,
            state_path=state_path,
            null_df=pd.DataFrame(),
            save_parquet=save_parquet,
            stop_reason="no_hypotheses",
            batches_completed=0,
        )
        return NullRunResult(
            null_df=pd.DataFrame(),
            artifact_path=csv_path,
            state_path=state_path,
            stop_reason="no_hypotheses",
            is_partial=False,
            batches_completed=0,
        )

    null_df = existing_null_df.copy() if existing_null_df is not None else pd.DataFrame()
    existing_counts = null_draw_counts(null_df)
    caches = prepare_hypothesis_caches(
        activation_index=activation_index,
        observed_metrics=observed,
        metrics_cfg=metrics_cfg,
        sample_size_mode=str(null_cfg.get("sample_size_mode", "observed_n_samples")),
        sample_size_value=null_cfg.get("sample_size_value"),
        min_scenes=min_scenes,
        use_type_constraint=use_type_constraint,
        scene_type_map=scene_type_map,
    )
    if not caches:
        write_null_artifacts(
            csv_path=csv_path,
            state_path=state_path,
            null_df=null_df,
            save_parquet=save_parquet,
            stop_reason="no_valid_hypotheses",
            batches_completed=0,
        )
        return NullRunResult(
            null_df=null_df,
            artifact_path=csv_path,
            state_path=state_path,
            stop_reason="no_valid_hypotheses",
            is_partial=False,
            batches_completed=0,
        )

    hypotheses = sorted(caches.keys())
    rng = np.random.default_rng(seed)
    batches_completed = 0
    stop_reason = "max_draws_reached"
    is_partial = False

    expected_batches = max(1, max_total_draws // max(1, draws_per_batch))
    if max_batches > 0:
        expected_batches = min(expected_batches, max_batches)
    batch_pbar = tqdm(
        total=expected_batches,
        desc="Null[batches]",
        unit="batch",
    )

    try:
        while True:
            if max_batches > 0 and batches_completed >= max_batches:
                stop_reason = "max_batches_reached"
                break

            can_draw_any = False
            batch_rows: list[dict[str, Any]] = []
            hyp_iter = tqdm(
                hypotheses,
                desc=f"Null[batch {batches_completed}]",
                unit="hyp",
                leave=False,
            )
            for hyp in hyp_iter:
                count = existing_counts.get(hyp, 0)
                remaining = max_total_draws - count
                if remaining <= 0:
                    continue
                n_to_draw = min(draws_per_batch, remaining)
                can_draw_any = True
                hyp_iter.set_postfix_str(f"{hyp[0]}|{hyp[1]}|{hyp[2]} draws={count}")

                rows = draw_mismatched_image_level(
                    cache=caches[hyp],
                    n_draws=n_to_draw,
                    replace=replace,
                    sampling_mode=sampling_mode,
                    scene_type_map=scene_type_map if use_type_constraint else {},
                    rng=rng,
                    start_draw_id=count,
                    run_id=run_id,
                    seed=seed,
                    draw_batch_id=batches_completed,
                )
                batch_rows.extend(rows)
                existing_counts[hyp] = count + len(rows)
            hyp_iter.close()

            if not can_draw_any:
                stop_reason = "max_draws_reached"
                break

            if batch_rows:
                batch_df = pd.DataFrame.from_records(batch_rows)
                null_df = batch_df if null_df.empty else pd.concat([null_df, batch_df], ignore_index=True)

            batches_completed += 1
            stop_reason = "running"
            batch_pbar.update(1)
            batch_pbar.set_postfix_str(f"rows={len(null_df)}")
            if batches_completed % checkpoint_every_batches == 0:
                write_null_artifacts(
                    csv_path=csv_path,
                    state_path=state_path,
                    null_df=null_df,
                    save_parquet=save_parquet,
                    stop_reason=stop_reason,
                    batches_completed=batches_completed,
                )

            if adaptive_enabled:
                should_stop, _ = evaluate_adaptive_stop(
                    observed_metrics=observed,
                    null_df=null_df,
                    alpha=alpha,
                    correction=correction,
                    require_all_hypotheses=require_all,
                    min_total_draws=min_total_draws,
                )
                if should_stop:
                    stop_reason = "adaptive_threshold_reached"
                    break

    except KeyboardInterrupt:
        if on_keyboard_interrupt != "save_and_continue":
            raise
        stop_reason = "manual_interrupt"
        is_partial = True
    finally:
        batch_pbar.close()

    write_null_artifacts(
        csv_path=csv_path,
        state_path=state_path,
        null_df=null_df,
        save_parquet=save_parquet,
        stop_reason=stop_reason,
        batches_completed=batches_completed,
    )

    if stop_reason in {"manual_interrupt", "max_draws_reached", "max_batches_reached"}:
        is_partial = True

    return NullRunResult(
        null_df=null_df,
        artifact_path=csv_path,
        state_path=state_path,
        stop_reason=stop_reason,
        is_partial=is_partial,
        batches_completed=batches_completed,
    )
