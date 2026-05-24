"""Thin benchmark orchestrator for extraction, null, and metrics stages."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.analysis.null_distribution import (
    NullRunResult,
    compute_null_distribution_adaptive,
    evaluate_adaptive_stop,
)
from src.analysis.null_stop import normalize_observed_metrics
from src.data.registry import load_dataset_pairs
from src.models.registry import build_model
from src.pipeline.benchmark_config import (
    list_activation_index_files as _list_activation_index_files,
    read_null_state as _read_null_state,
    resolve_metric_pairs as _resolve_metric_pairs,
    resolve_null_artifact_path as _resolve_null_artifact_path,
    resolve_run_ctx_for_extraction as _resolve_run_ctx_for_extraction,
    resolve_run_ctx_for_metrics as _resolve_run_ctx_for_metrics,
)
from src.pipeline.benchmark_plots import log_metric_plots, log_significance_plots
from src.pipeline.benchmark_significance import enrich_with_significance
from src.pipeline.benchmark_tables import (
    compute_metric_table_from_indices as _compute_metric_table_from_indices,
    load_activation_indices as _load_activation_indices,
)
from src.pipeline.extraction import run_extraction
from src.utils.config import RunContext, cfg_to_container, make_run_context, make_run_context_from_id
from src.utils.distributed import (
    all_gather_objects,
    barrier,
    broadcast_object,
    concat_gathered_tables,
    get_dist_context,
    split_by_rank,
)
from src.utils.io import read_null_distribution, write_json, write_optional_parquet, write_table
from src.utils.tracking import NoopTracker, Tracker, build_tracker

log = logging.getLogger(__name__)


def _log(message: str) -> None:
    """Print a standardized info line for benchmark stages."""
    dist_ctx = get_dist_context()
    if dist_ctx.enabled and not dist_ctx.is_main:
        return
    print(f"[INFO] {message}")


def _load_pair_samples(cfg_dict: dict[str, Any], modalities: tuple[str, str]) -> list[Any]:
    """Resolve dataset/run config and load aligned samples for a pair."""
    frames_per_scene_raw = cfg_dict["data"].get("frames_per_scene")
    max_total_samples_raw = cfg_dict["data"].get("max_total_samples")
    environment_raw = cfg_dict["data"].get("environment", "indoors")
    environment = (
        list(environment_raw)
        if isinstance(environment_raw, (list, tuple))
        else str(environment_raw)
    )
    split_raw = cfg_dict["data"].get("split", "train")
    split = (
        list(split_raw) if isinstance(split_raw, (list, tuple)) else str(split_raw)
    )
    return load_dataset_pairs(
        dataset_name=cfg_dict["data"]["name"],
        root=Path(cfg_dict["data"]["root"]),
        modalities=modalities,
        n_scenes=int(cfg_dict["data"]["n_scenes"]),
        scene_stride=int(cfg_dict["data"]["scene_stride"]),
        exclude_scenes=list(cfg_dict["data"].get("exclude_scenes") or []),
        frames_per_scene=int(frames_per_scene_raw) if frames_per_scene_raw is not None else None,
        max_total_samples=int(max_total_samples_raw) if max_total_samples_raw is not None else None,
        seed=int(cfg_dict["data"].get("seed", 42)),
        split=split,
        environment=environment,
    )


def _run_joint_pass_for_pair(
    cfg_dict: dict[str, Any],
    model: Any,
    run_ctx: RunContext,
    left_mod: str,
    right_mod: str,
    samples: list[Any] | None,
    reuse_enabled: bool,
) -> None:
    """Run the joint forward pass for one pair and emit comparison indices.

    Three comparison index files are written next to the single-mod indices so
    the metrics engine picks them up automatically:

      * ``activation_index_<A>-<A>_joint_with_<B>.csv``
      * ``activation_index_<B>-<B>_joint_with_<A>.csv``
      * ``activation_index_<A>_joint_with_<B>-<B>_joint_with_<A>.csv``

    The raw joint extraction index is saved as ``joint_raw_index_<pair>.csv``
    so it stays available for inspection without being double-counted by the
    metrics glob (which only matches ``activation_index_*.csv``).
    """
    pair_name = f"{left_mod}-{right_mod}"
    single_idx_path = run_ctx.artifacts_dir / f"activation_index_{pair_name}.csv"
    joint_raw_path = run_ctx.artifacts_dir / f"joint_raw_index_{pair_name}.csv"
    cmp_paths = [
        run_ctx.artifacts_dir / f"activation_index_{left_mod}-{left_mod}_joint_with_{right_mod}.csv",
        run_ctx.artifacts_dir / f"activation_index_{right_mod}-{right_mod}_joint_with_{left_mod}.csv",
        run_ctx.artifacts_dir / f"activation_index_{left_mod}_joint_with_{right_mod}-{right_mod}_joint_with_{left_mod}.csv",
    ]

    if reuse_enabled and joint_raw_path.exists() and all(p.exists() for p in cmp_paths):
        _log(f"Skipping joint extraction for {pair_name}: existing artifacts found")
        return

    if samples is None:
        _log(f"Loading samples for joint pair {pair_name}")
        samples = _load_pair_samples(cfg_dict, modalities=(left_mod, right_mod))

    if not single_idx_path.exists():
        raise FileNotFoundError(
            f"Joint comparison indices require single-mod index at '{single_idx_path}'. "
            "Run single-modality extraction for this pair first."
        )
    single_mod_index = pd.read_csv(single_idx_path)

    joint_index = run_joint_extraction(
        model=model,
        samples=samples,
        pair_name=pair_name,
        modalities=(left_mod, right_mod),
        run_id=run_ctx.run_id,
        out_dir=run_ctx.activations_dir,
        show_progress=True,
    )
    write_table(joint_raw_path, joint_index)

    comparisons = build_joint_comparison_indices(
        single_mod_index=single_mod_index,
        joint_index=joint_index,
        left_modality=left_mod,
        right_modality=right_mod,
    )
    for comp in comparisons:
        out = run_ctx.artifacts_dir / f"activation_index_{comp.pair_name}.csv"
        write_table(out, comp.table)
        _log(
            f"Saved joint comparison index {comp.pair_name}: rows={len(comp.table)}"
        )


def run_extraction_stage(cfg: DictConfig, run_ctx_override: RunContext | None = None) -> RunContext:
    """Run extraction for all resolved modality pairs."""
    cfg_dict = cfg_to_container(cfg)
    dist_ctx = get_dist_context()
    resolved_pairs = _resolve_metric_pairs(cfg_dict["metrics"])
    cfg_dict["metrics"]["resolved_pairs"] = resolved_pairs
    if run_ctx_override is not None:
        run_ctx = run_ctx_override
    elif dist_ctx.enabled:
        run_id = None
        if dist_ctx.is_main:
            run_id = make_run_context(cfg).run_id
        run_id = broadcast_object(run_id, src=0, ctx=dist_ctx)
        run_ctx = make_run_context_from_id(Path(cfg_dict["paths"]["runs_root"]), str(run_id))
    else:
        run_ctx = _resolve_run_ctx_for_extraction(cfg, cfg_dict)
    _log(f"Starting extraction stage: run_id={run_ctx.run_id}")
    model = build_model(cfg_dict["model"], cfg_dict["runtime"])

    joint_cfg = dict(cfg_dict["metrics"].get("joint_pass", {}) or {})
    joint_enabled = bool(joint_cfg.get("enabled", False))
    reuse_cfg = dict(cfg_dict.get("runtime", {}).get("reuse", {}))
    reuse_enabled = bool(reuse_cfg.get("activations", reuse_cfg.get("extraction", False)))
    reuse_joint = bool(reuse_cfg.get("joint_activations", reuse_enabled))

    for pair in resolved_pairs:
        left_mod, right_mod = pair[0], pair[1]
        pair_name = f"{left_mod}-{right_mod}"
        out_idx = run_ctx.artifacts_dir / f"activation_index_{pair_name}.csv"

        if reuse_enabled and out_idx.exists():
            _log(f"Skipping extraction for {pair_name}: existing index found ({out_idx.name})")
            barrier()
            continue

        _log(f"Loading samples for pair {pair_name} (sampling-aware)")
        samples = _load_pair_samples(cfg_dict, modalities=(left_mod, right_mod))
        local_samples = split_by_rank(samples, ctx=dist_ctx)
        _log(
            f"Pair {pair_name}: loaded_samples={len(samples)} local_samples={len(local_samples)}"
        )
        if len(samples) == 0:
            raise RuntimeError(
                f"Pair {pair_name}: 0 samples available for modalities "
                f"({left_mod}, {right_mod}). The required modality files are "
                f"likely missing on disk. Re-run the dataset download with the "
                f"appropriate --include-* flags (e.g. --include-normals) and "
                f"ensure both modalities are present under "
                f"resources/datasets/{cfg_dict['data']['name']}."
            )
        activation_index = run_extraction(
            model=model,
            samples=local_samples,
            pair_name=pair_name,
            run_id=run_ctx.run_id,
            out_dir=run_ctx.activations_dir,
            show_progress=dist_ctx.is_main,
        )
        gathered = all_gather_objects(activation_index, ctx=dist_ctx)
        if dist_ctx.is_main:
            frames = [g for g in gathered if isinstance(g, pd.DataFrame) and not g.empty]
            merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            write_table(out_idx, merged)
            _log(f"Saved activation index for {pair_name}: rows={len(merged)}")
        barrier()

    if dist_ctx.is_main:
        write_json(run_ctx.run_dir / "resolved_config.json", cfg_dict)
        write_json(
            run_ctx.run_dir / "run_summary.json",
            {
                "run_id": run_ctx.run_id,
                "dataset": cfg_dict["data"]["name"],
                "pairs_mode": cfg_dict["metrics"].get("pairs_mode", "explicit"),
                "pairs": resolved_pairs,
                "stage": "extraction",
                "activation_indices": [str(p) for p in _list_activation_index_files(run_ctx)],
            },
        )
        _log("Extraction stage completed.")
    barrier()
    return run_ctx


def run_null_stage(
    cfg: DictConfig,
    run_ctx: RunContext,
    observed_metric_table: pd.DataFrame,
) -> NullRunResult | None:
    """Run or resume adaptive null-distribution stage."""
    cfg_dict = cfg_to_container(cfg)
    dist_ctx = get_dist_context()
    null_cfg = dict(cfg_dict.get("analysis", {}).get("null_distribution", {}))
    if not bool(null_cfg.get("enabled", False)):
        return None

    null_artifact_path = _resolve_null_artifact_path(cfg_dict, run_ctx=run_ctx)
    if null_artifact_path is None:
        return None

    existing_null = pd.DataFrame()
    reuse_if_exists = bool(null_cfg.get("reuse_if_exists", True))
    resume_if_partial = bool(null_cfg.get("resume_if_partial", True))
    state_path = null_artifact_path.parent / str(null_cfg.get("state_filename", "null_distribution_state.json"))
    reuse_null = bool(cfg_dict.get("runtime", {}).get("reuse", {}).get("null_distribution", False))
    if reuse_null and null_artifact_path.exists():
        existing_null = read_null_distribution(null_artifact_path)
        _log(f"Loaded existing null artifact: {null_artifact_path} ({len(existing_null)} rows)")
        state = _read_null_state(state_path)
        stop_reason = str(state.get("stop_reason", "unknown"))
        is_partial = stop_reason in {"manual_interrupt", "max_draws_reached", "max_batches_reached", "running"}
        if reuse_if_exists and (not is_partial or not resume_if_partial):
            _log("Reusing existing null artifact without additional draws.")
            return NullRunResult(
                null_df=existing_null,
                artifact_path=null_artifact_path,
                state_path=state_path,
                stop_reason="reused_existing",
                is_partial=is_partial,
                batches_completed=int(state.get("batches_completed", 0)),
            )

    activation_index = _load_activation_indices(run_ctx)
    if activation_index.empty:
        _log("No activation index found for null stage. Skipping null computation.")
        return None

    _log(
        "Starting adaptive null-distribution stage... "
        f"metrics={null_cfg.get('metrics_enabled', cfg_dict.get('metrics', {}).get('enabled', []))} "
        f"draws_per_batch={null_cfg.get('draws_per_batch')} "
        f"max_total_draws={null_cfg.get('max_total_draws')}"
    )
    result = compute_null_distribution_adaptive(
        activation_index=activation_index,
        observed_metrics=observed_metric_table,
        null_cfg=null_cfg,
        metrics_cfg=cfg_dict["metrics"],
        run_id=run_ctx.run_id,
        out_dir=null_artifact_path.parent,
        out_filename=null_artifact_path.name,
        existing_null_df=existing_null,
    )
    barrier()
    if dist_ctx.enabled and not dist_ctx.is_main and null_artifact_path.exists():
        refreshed = read_null_distribution(null_artifact_path)
        result = NullRunResult(
            null_df=refreshed,
            artifact_path=result.artifact_path,
            state_path=result.state_path,
            stop_reason=result.stop_reason,
            is_partial=result.is_partial,
            batches_completed=result.batches_completed,
        )
    _log(
        f"Null stage completed: rows={len(result.null_df)} "
        f"stop_reason={result.stop_reason} partial={result.is_partial}"
    )
    return result


def run_metrics_stage(
    cfg: DictConfig,
    run_ctx_override: RunContext | None = None,
    precomputed_metric_table: pd.DataFrame | None = None,
    null_result: NullRunResult | None = None,
    tracker: Tracker | None = None,
    log_config_to_tracker: bool = True,
    finish_tracker: bool = True,
    stage_durations_s: dict[str, float] | None = None,
    run_start_perf_s: float | None = None,
) -> RunContext:
    """Run metrics, optional null enrichment, and tracker logging."""
    cfg_dict = cfg_to_container(cfg)
    try:
        cfg_dict["metrics"]["resolved_pairs"] = _resolve_metric_pairs(cfg_dict["metrics"])
    except Exception:
        pass

    dist_ctx = get_dist_context()
    run_ctx = run_ctx_override or _resolve_run_ctx_for_metrics(cfg_dict)
    _log(f"Starting metrics stage for run_id={run_ctx.run_id}")
    owned_tracker = False
    if tracker is None:
        if dist_ctx.enabled and not dist_ctx.is_main:
            cfg_dict["tracking"]["wandb"]["enabled"] = False
        tracker = build_tracker(cfg=cfg_dict, run_id=run_ctx.run_id)
        owned_tracker = True
    if log_config_to_tracker:
        tracker.log_config(cfg_dict)

    metric_table = (
        precomputed_metric_table.copy()
        if precomputed_metric_table is not None
        else _compute_metric_table_from_indices(
            cfg_dict,
            run_ctx,
            show_progress=dist_ctx.is_main,
            log_fn=_log,
        )
    )
    precomputed_is_global = precomputed_metric_table is not None and dist_ctx.enabled

    null_cfg = dict(cfg_dict.get("analysis", {}).get("null_distribution", {}))
    adaptive_cfg = dict(null_cfg.get("adaptive_stop", {}))
    correction = str(adaptive_cfg.get("correction", "bh_fdr"))
    alpha = float(adaptive_cfg.get("alpha", 0.05))
    null_missing_behavior = str(cfg_dict.get("runtime", {}).get("null_missing_behavior", "warn_and_skip"))

    has_significance = False
    null_stop_reason = "not_requested"
    null_is_partial = False
    null_draws_used = 0

    null_artifact_path = _resolve_null_artifact_path(cfg_dict, run_ctx=run_ctx)
    if null_result is not None:
        null_df = null_result.null_df
        null_stop_reason = null_result.stop_reason
        null_is_partial = bool(null_result.is_partial)
    elif null_artifact_path is not None:
        _log(f"Loading null distribution from: {null_artifact_path}")
        null_df = read_null_distribution(null_artifact_path)
        null_stop_reason = "loaded_from_artifact"
    else:
        null_df = pd.DataFrame()

    if not null_df.empty and not metric_table.empty:
        metric_table = enrich_with_significance(metric_table, null_df, correction=correction, alpha=alpha)
        null_draws_used = int(len(null_df))
        has_significance = True
    elif null_artifact_path is not None and null_df.empty:
        msg = f"Null artifact not found or empty at '{null_artifact_path}'. Significance stats will be omitted."
        if null_missing_behavior == "error":
            raise FileNotFoundError(msg)
        log.warning(msg)

    if not precomputed_is_global:
        metric_table = concat_gathered_tables(metric_table, ctx=dist_ctx)

    if dist_ctx.is_main:
        write_table(run_ctx.metrics_dir / "metrics.csv", metric_table)
        write_optional_parquet(
            path=run_ctx.metrics_dir / "metrics.parquet",
            table=metric_table,
            enabled=bool(cfg_dict["metrics"]["output"]["save_parquet"]),
        )

    run_summary = {
        "run_id": run_ctx.run_id,
        "dataset": cfg_dict["data"]["name"],
        "metrics": cfg_dict["metrics"]["enabled"],
        "pairs_mode": cfg_dict["metrics"].get("pairs_mode", "explicit"),
        "pairs": sorted(metric_table["pair"].unique().tolist()) if not metric_table.empty else [],
        "n_rows_metrics": int(len(metric_table)),
        "has_significance": has_significance,
        "null_draws_used": int(null_draws_used),
        "null_stop_reason": str(null_stop_reason),
        "null_is_partial": bool(null_is_partial),
        "stage": "metrics",
    }
    if dist_ctx.is_main:
        write_json(run_ctx.run_dir / "run_summary.json", run_summary)
        write_json(run_ctx.run_dir / "resolved_config.json", cfg_dict)

    if dist_ctx.is_main and not metric_table.empty:
        tracker.log_table("metrics_table", metric_table)
        log_metric_plots(tracker=tracker, metric_table=metric_table)
        if has_significance:
            log_significance_plots(tracker=tracker, metric_table=metric_table)
        summary: dict[str, Any] = {"metrics_rows": int(len(metric_table))}
        if has_significance:
            sig_rows = metric_table.dropna(subset=["p_value"])
            if not sig_rows.empty:
                summary["n_significant_p05"] = int((sig_rows["p_value"] <= 0.05).sum())
                if "p_value_adjusted" in sig_rows.columns:
                    summary["n_significant_adj_p05"] = int((sig_rows["p_value_adjusted"] <= 0.05).sum())
                summary["mean_p_value"] = float(sig_rows["p_value"].mean())
                for metric_name, grp in sig_rows.groupby("metric"):
                    prefix = str(metric_name)
                    summary[f"{prefix}/n_significant_p05"] = int((grp["p_value"] <= 0.05).sum())
                    if "p_value_adjusted" in grp.columns:
                        summary[f"{prefix}/n_significant_adj_p05"] = int((grp["p_value_adjusted"] <= 0.05).sum())
                    summary[f"{prefix}/mean_p_value"] = float(grp["p_value"].mean())
                    summary[f"{prefix}/mean_value"] = float(grp["value"].mean())
        summary["null_stop_reason"] = null_stop_reason
        summary["null_draws_used"] = int(null_draws_used)
        summary["null_is_partial"] = bool(null_is_partial)
        if stage_durations_s:
            for stage_name, elapsed_s in stage_durations_s.items():
                summary[f"runtime/{stage_name}_seconds"] = float(elapsed_s)
        if run_start_perf_s is not None:
            summary["runtime/run_elapsed_seconds"] = float(time.perf_counter() - run_start_perf_s)
        tracker.log_summary(summary)
    if finish_tracker and (owned_tracker or tracker is not None):
        tracker.finish()
    if dist_ctx.is_main:
        _log(f"Metrics stage completed. rows={len(metric_table)}")
    barrier()
    return run_ctx


def run_benchmark(cfg: DictConfig) -> RunContext:
    """Run extraction -> optional adaptive null -> metrics."""
    run_start_perf_s = time.perf_counter()
    cfg_dict = cfg_to_container(cfg)
    dist_ctx = get_dist_context()
    if dist_ctx.enabled:
        run_id = None
        if dist_ctx.is_main:
            run_id = make_run_context(cfg).run_id
        run_id = broadcast_object(run_id, src=0, ctx=dist_ctx)
        run_ctx = make_run_context_from_id(Path(cfg_dict["paths"]["runs_root"]), str(run_id))
    else:
        run_ctx = _resolve_run_ctx_for_extraction(cfg, cfg_dict)
    tracker_cfg = cfg_to_container(cfg)
    if dist_ctx.enabled and not dist_ctx.is_main:
        tracker_cfg["tracking"]["wandb"]["enabled"] = False
        tracker: Tracker = NoopTracker()
    else:
        tracker = build_tracker(cfg=tracker_cfg, run_id=run_ctx.run_id)
        tracker.log_config(tracker_cfg)

    stage_durations_s: dict[str, float] = {}

    extraction_start = time.perf_counter()
    run_ctx = run_extraction_stage(cfg, run_ctx_override=run_ctx)
    stage_durations_s["extraction"] = time.perf_counter() - extraction_start
    cfg_dict["runtime"]["metrics_input_run_id"] = run_ctx.run_id
    cfg_dict["runtime"]["activation_input_run_id"] = run_ctx.run_id
    cfg_for_next = OmegaConf.create(cfg_dict)

    observed_start = time.perf_counter()
    observed_metric_table = _compute_metric_table_from_indices(
        cfg_dict=cfg_dict,
        run_ctx=run_ctx,
        show_progress=dist_ctx.is_main,
        log_fn=_log,
    )
    observed_metric_table = concat_gathered_tables(observed_metric_table, ctx=get_dist_context())
    stage_durations_s["observed_metrics"] = time.perf_counter() - observed_start

    null_result = None
    if bool(cfg_dict.get("analysis", {}).get("null_distribution", {}).get("enabled", False)):
        null_start = time.perf_counter()
        null_result = run_null_stage(cfg=cfg_for_next, run_ctx=run_ctx, observed_metric_table=observed_metric_table)
        stage_durations_s["null_distribution"] = time.perf_counter() - null_start
        if null_result is not None and not null_result.null_df.empty:
            adaptive_cfg = dict(cfg_dict.get("analysis", {}).get("null_distribution", {}).get("adaptive_stop", {}))
            observed_for_stop = normalize_observed_metrics(
                observed_metric_table,
                metrics_cfg=cfg_dict.get("metrics", {}),
                null_cfg=cfg_dict.get("analysis", {}).get("null_distribution", {}),
            )
            should_stop, stop_eval = evaluate_adaptive_stop(
                observed_metrics=observed_for_stop,
                null_df=null_result.null_df,
                alpha=float(adaptive_cfg.get("alpha", 0.05)),
                correction=str(adaptive_cfg.get("correction", "bh_fdr")),
                require_all_hypotheses=bool(adaptive_cfg.get("require_all_hypotheses", True)),
                min_total_draws=int(cfg_dict.get("analysis", {}).get("null_distribution", {}).get("min_total_draws", 1)),
            )
            if not stop_eval.empty:
                stop_eval["stop_criterion_met"] = bool(should_stop)
                if get_dist_context().is_main:
                    write_table(run_ctx.metrics_dir / "null_stop_evaluation.csv", stop_eval)

    return run_metrics_stage(
        cfg=cfg_for_next,
        run_ctx_override=run_ctx,
        precomputed_metric_table=observed_metric_table,
        null_result=null_result,
        tracker=tracker,
        log_config_to_tracker=False,
        finish_tracker=True,
        stage_durations_s=stage_durations_s,
        run_start_perf_s=run_start_perf_s,
    )
