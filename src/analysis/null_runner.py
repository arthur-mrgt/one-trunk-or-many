"""Adaptive null-distribution runner with checkpointing and resume."""

from __future__ import annotations

import logging
import time
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
from src.analysis.null_sampling import (
    MODES_NEEDING_TYPE_MAP,
    SUPPORTED_SAMPLING_MODES,
    draw_mismatched_image_level,
    prepare_hypothesis_caches,
)
from src.analysis.null_stop import (
    build_observed_stub,
    evaluate_adaptive_stop,
    normalize_observed_metrics,
)
from src.utils.distributed import (
    all_gather_objects,
    any_rank_true,
    broadcast_object,
    get_dist_context,
    split_by_rank,
)

log = logging.getLogger(__name__)


def _emit(message: str) -> None:
    """Emit progress logs to stdout and logger."""
    dist_ctx = get_dist_context()
    if dist_ctx.enabled and not dist_ctx.is_main:
        return
    print(f"[INFO] {message}")


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
    dist_ctx = get_dist_context()
    csv_path = out_dir / out_filename
    state_path = out_dir / str(null_cfg.get("state_filename", "null_distribution_state.json"))

    save_parquet = bool(metrics_cfg.get("output", {}).get("save_parquet", False))
    replace = bool(null_cfg.get("replace", True))
    seed = int(null_cfg.get("seed", 42))
    sampling_mode = str(null_cfg.get("sampling_mode", null_cfg.get("sampling", "cross_scene_type_random")))
    min_scenes = int(null_cfg.get("min_scenes", 2))
    draws_per_batch = int(null_cfg.get("draws_per_batch", 200))
    checkpoint_every_batches = int(null_cfg.get("checkpoint_every_batches", 1))
    checkpoint_every_hypotheses = int(null_cfg.get("checkpoint_every_hypotheses", 1))
    progress_log_every_batches = int(null_cfg.get("progress_log_every_batches", 1))
    legacy_n_draws = int(null_cfg.get("n_draws", 1000))
    min_total_draws = int(null_cfg.get("min_total_draws", max(1, draws_per_batch)))
    max_total_draws = int(null_cfg.get("max_total_draws", legacy_n_draws))
    max_batches = int(null_cfg.get("max_batches", 0))
    draw_chunk_size = int(null_cfg.get("draw_chunk_size", min(50, draws_per_batch)))
    null_metrics_enabled = [str(m) for m in (null_cfg.get("metrics_enabled") or [])]

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
    if checkpoint_every_hypotheses <= 0:
        raise ValueError("analysis.null_distribution.checkpoint_every_hypotheses must be > 0.")
    if progress_log_every_batches <= 0:
        raise ValueError("analysis.null_distribution.progress_log_every_batches must be > 0.")
    if max_total_draws <= 0:
        raise ValueError("analysis.null_distribution.max_total_draws must be > 0.")
    if draw_chunk_size <= 0:
        raise ValueError("analysis.null_distribution.draw_chunk_size must be > 0.")

    scene_type_map = load_scene_type_map(null_cfg, activation_index=activation_index)
    needs_type_map = sampling_mode in MODES_NEEDING_TYPE_MAP
    if needs_type_map and not scene_type_map:
        raise ValueError(
            f"sampling_mode={sampling_mode} requires a scene-type map but none "
            f"was loaded. Set `analysis.null_distribution.metadata_path` to a "
            f"valid CSV, or set `analysis.null_distribution.scene_type_source` "
            f"(e.g. `scene_id_parent` for DIODE)."
        )

    observed = normalize_observed_metrics(observed_metrics, metrics_cfg, null_cfg=null_cfg)
    if observed.empty:
        if dist_ctx.is_main:
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
    existing_counts = null_draw_counts(null_df) if dist_ctx.is_main else {}
    existing_counts = broadcast_object(existing_counts, src=0, ctx=dist_ctx)
    caches = prepare_hypothesis_caches(
        activation_index=activation_index,
        observed_metrics=observed,
        metrics_cfg=metrics_cfg,
        null_metrics_enabled=null_metrics_enabled,
        sample_size_mode=str(null_cfg.get("sample_size_mode", "observed_n_samples")),
        sample_size_value=null_cfg.get("sample_size_value"),
        min_scenes=min_scenes,
        sampling_mode=sampling_mode,
        scene_type_map=scene_type_map,
    )
    if not caches:
        if dist_ctx.is_main:
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
    local_hypotheses = split_by_rank(hypotheses, ctx=dist_ctx)
    rng = np.random.default_rng(seed)
    batches_completed = 0
    stop_reason = "max_draws_reached"
    is_partial = False
    pending_rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()

    _emit(
        "Adaptive null configured: "
        f"metrics={sorted(set(m for _, _, m in hypotheses))} "
        f"hypotheses={len(hypotheses)} local_hypotheses={len(local_hypotheses)} "
        f"draws_per_batch={draws_per_batch} "
        f"draw_chunk_size={draw_chunk_size} max_total_draws={max_total_draws}"
    )
    total_target_draws = len(hypotheses) * max_total_draws
    current_draws = int(sum(existing_counts.get(h, 0) for h in hypotheses)) if dist_ctx.is_main else 0
    total_bar = None
    if dist_ctx.is_main:
        total_bar = tqdm(
            total=total_target_draws,
            initial=current_draws,
            desc="Null draws total",
            unit="draw",
            dynamic_ncols=True,
        )

    def flush_pending_rows() -> None:
        """Move pending draw rows into the in-memory null table."""
        nonlocal null_df, pending_rows
        local_rows = pending_rows
        pending_rows = []
        gathered = all_gather_objects(local_rows, ctx=dist_ctx)
        if not dist_ctx.is_main:
            return
        merged_rows: list[dict[str, Any]] = []
        for part in gathered:
            if part:
                merged_rows.extend(part)
        if not merged_rows:
            return
        batch_df = pd.DataFrame.from_records(merged_rows)
        null_df = batch_df if null_df.empty else pd.concat([null_df, batch_df], ignore_index=True)

    def checkpoint(reason: str) -> None:
        """Persist current null artifacts and state."""
        flush_pending_rows()
        if dist_ctx.is_main:
            write_null_artifacts(
                csv_path=csv_path,
                state_path=state_path,
                null_df=null_df,
                save_parquet=save_parquet,
                stop_reason=reason,
                batches_completed=batches_completed,
            )

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
            hypotheses_processed = 0
            batch_bar = tqdm(
                total=len(local_hypotheses),
                desc=f"Null batch {batches_completed + 1} [rank {dist_ctx.rank}]",
                unit="hyp",
                leave=False,
                dynamic_ncols=True,
                disable=True,
            )
            try:
                for hyp in local_hypotheses:
                    count = existing_counts.get(hyp, 0)
                    remaining = max_total_draws - count
                    if remaining <= 0:
                        batch_bar.update(1)
                        continue
                    n_to_draw = min(draws_per_batch, remaining)
                    can_draw_any = True
                    drawn_for_hyp = 0
                    pair_name, layer_name, metric_name = hyp
                    batch_bar.set_postfix_str(
                        f"{metric_name} {pair_name} {layer_name} {count}/{max_total_draws}"
                    )
                    while drawn_for_hyp < n_to_draw:
                        n_chunk = min(draw_chunk_size, n_to_draw - drawn_for_hyp)
                        rows = draw_mismatched_image_level(
                            cache=caches[hyp],
                            n_draws=n_chunk,
                            replace=replace,
                            sampling_mode=sampling_mode,
                            scene_type_map=scene_type_map if needs_type_map else {},
                            rng=rng,
                            start_draw_id=count + drawn_for_hyp,
                            run_id=run_id,
                            seed=seed,
                            draw_batch_id=batches_completed,
                        )
                        pending_rows.extend(rows)
                        drawn_for_hyp += len(rows)
                        if total_bar is not None:
                            total_bar.update(len(rows))
                        existing_counts[hyp] = count + drawn_for_hyp
                        batch_bar.set_postfix_str(
                            f"{metric_name} {pair_name} {layer_name} "
                            f"{existing_counts[hyp]}/{max_total_draws}"
                        )
                        if len(rows) == 0:
                            existing_counts[hyp] = max_total_draws
                            break
                    hypotheses_processed += 1
                    batch_bar.update(1)
                    if (not dist_ctx.enabled) and hypotheses_processed % checkpoint_every_hypotheses == 0:
                        checkpoint(reason="running")
            finally:
                batch_bar.close()

            can_draw_any = any_rank_true(can_draw_any, ctx=dist_ctx)
            if not can_draw_any:
                stop_reason = "max_draws_reached"
                break

            batches_completed += 1
            stop_reason = "running"
            batch_pbar.update(1)
            batch_pbar.set_postfix_str(f"rows={len(null_df)}")
            if batches_completed % checkpoint_every_batches == 0:
                checkpoint(reason=stop_reason)

            if adaptive_enabled:
                flush_pending_rows()
                should_stop, stop_eval = evaluate_adaptive_stop(
                    observed_metrics=observed,
                    null_df=null_df,
                    alpha=alpha,
                    correction=correction,
                    require_all_hypotheses=require_all,
                    min_total_draws=min_total_draws,
                )
                if (
                    dist_ctx.is_main
                    and batches_completed % progress_log_every_batches == 0
                    and not stop_eval.empty
                ):
                    valid = stop_eval.dropna(subset=["p_value_raw", "p_value_adjusted"])
                    if not valid.empty:
                        n_sig = int(valid["is_significant"].sum())
                        n_total = int(len(valid))
                        if total_bar is not None:
                            total_bar.set_postfix_str(
                                "batch="
                                f"{batches_completed} sig={n_sig}/{n_total} "
                                f"raw_p=[{valid['p_value_raw'].min():.3g},{valid['p_value_raw'].max():.3g}] "
                                f"adj_p=[{valid['p_value_adjusted'].min():.3g},{valid['p_value_adjusted'].max():.3g}]"
                            )
                    elif total_bar is not None:
                        total_bar.set_postfix_str(
                            f"batch={batches_completed} rows={len(null_df)} p-values=pending"
                        )
                if should_stop:
                    stop_reason = "adaptive_threshold_reached"
                should_stop = bool(broadcast_object(should_stop, src=0, ctx=dist_ctx))
                if should_stop:
                    break

    except KeyboardInterrupt:
        if on_keyboard_interrupt != "save_and_continue":
            raise
        stop_reason = "manual_interrupt"
        is_partial = True
        _emit("Manual interrupt detected during null stage. Saving intermediate artifacts...")
        checkpoint(reason=stop_reason)
    finally:
        if total_bar is not None:
            total_bar.close()

    checkpoint(reason=stop_reason)
    elapsed_s = time.perf_counter() - run_start
    _emit(
        "Adaptive null finished: "
        f"stop_reason={stop_reason} batches={batches_completed} rows={len(null_df)} "
        f"elapsed_s={elapsed_s:.1f}"
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
