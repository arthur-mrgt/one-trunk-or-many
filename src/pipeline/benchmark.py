"""Thin benchmark orchestrator for extraction, null, and metrics stages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.analysis.null_distribution import (
    NullRunResult,
    compute_null_distribution_adaptive,
    evaluate_adaptive_stop,
)
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
from src.utils.config import RunContext, cfg_to_container
from src.utils.io import read_null_distribution, write_json, write_optional_parquet, write_table
from src.utils.tracking import build_tracker

log = logging.getLogger(__name__)


def _log(message: str) -> None:
    """Print a standardized info line for benchmark stages."""
    print(f"[INFO] {message}")


def run_extraction_stage(cfg: DictConfig, run_ctx_override: RunContext | None = None) -> RunContext:
    """Run extraction for all resolved modality pairs."""
    cfg_dict = cfg_to_container(cfg)
    resolved_pairs = _resolve_metric_pairs(cfg_dict["metrics"])
    cfg_dict["metrics"]["resolved_pairs"] = resolved_pairs
    run_ctx = run_ctx_override or _resolve_run_ctx_for_extraction(cfg, cfg_dict)
    _log(f"Starting extraction stage: run_id={run_ctx.run_id}")
    model = build_model(cfg_dict["model"], cfg_dict["runtime"])

    for pair in resolved_pairs:
        left_mod, right_mod = pair[0], pair[1]
        pair_name = f"{left_mod}-{right_mod}"
        out_idx = run_ctx.artifacts_dir / f"activation_index_{pair_name}.csv"
        reuse_cfg = dict(cfg_dict.get("runtime", {}).get("reuse", {}))
        reuse_enabled = bool(reuse_cfg.get("activations", reuse_cfg.get("extraction", False)))
        if reuse_enabled and out_idx.exists():
            _log(f"Skipping extraction for {pair_name}: existing index found ({out_idx.name})")
            continue

        _log(f"Loading samples for pair {pair_name}")
        frames_per_scene_raw = cfg_dict["data"].get("frames_per_scene")
        max_total_samples_raw = cfg_dict["data"].get("max_total_samples")
        samples = load_dataset_pairs(
            dataset_name=cfg_dict["data"]["name"],
            root=Path(cfg_dict["data"]["root"]),
            modalities=(left_mod, right_mod),
            n_scenes=int(cfg_dict["data"]["n_scenes"]),
            scene_stride=int(cfg_dict["data"]["scene_stride"]),
            exclude_scenes=list(cfg_dict["data"].get("exclude_scenes") or []),
            frames_per_scene=int(frames_per_scene_raw) if frames_per_scene_raw is not None else None,
            max_total_samples=int(max_total_samples_raw) if max_total_samples_raw is not None else None,
            seed=int(cfg_dict["data"].get("seed", 42)),
            split=str(cfg_dict["data"].get("split", "train")),
            environment=str(cfg_dict["data"].get("environment", "indoors")),
        )
        activation_index = run_extraction(
            model=model,
            samples=samples,
            pair_name=pair_name,
            run_id=run_ctx.run_id,
            out_dir=run_ctx.activations_dir,
            show_progress=True,
        )
        write_table(out_idx, activation_index)
        _log(f"Saved activation index for {pair_name}: rows={len(activation_index)}")

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
    return run_ctx


def run_null_stage(
    cfg: DictConfig,
    run_ctx: RunContext,
    observed_metric_table: pd.DataFrame,
) -> NullRunResult | None:
    """Run or resume adaptive null-distribution stage."""
    cfg_dict = cfg_to_container(cfg)
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

    _log("Starting adaptive null-distribution stage...")
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
) -> RunContext:
    """Run metrics, optional null enrichment, and tracker logging."""
    cfg_dict = cfg_to_container(cfg)
    try:
        cfg_dict["metrics"]["resolved_pairs"] = _resolve_metric_pairs(cfg_dict["metrics"])
    except Exception:
        pass

    run_ctx = run_ctx_override or _resolve_run_ctx_for_metrics(cfg_dict)
    _log(f"Starting metrics stage for run_id={run_ctx.run_id}")
    tracker = build_tracker(cfg=cfg_dict, run_id=run_ctx.run_id)
    tracker.log_config(cfg_dict)

    metric_table = (
        precomputed_metric_table.copy()
        if precomputed_metric_table is not None
        else _compute_metric_table_from_indices(cfg_dict, run_ctx, show_progress=True, log_fn=_log)
    )

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
    write_json(run_ctx.run_dir / "run_summary.json", run_summary)
    write_json(run_ctx.run_dir / "resolved_config.json", cfg_dict)

    if not metric_table.empty:
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
        tracker.log_summary(summary)
    tracker.finish()
    _log(f"Metrics stage completed. rows={len(metric_table)}")
    return run_ctx


def run_benchmark(cfg: DictConfig) -> RunContext:
    """Run extraction -> optional adaptive null -> metrics."""
    cfg_dict = cfg_to_container(cfg)
    run_ctx = run_extraction_stage(cfg)
    cfg_dict["runtime"]["metrics_input_run_id"] = run_ctx.run_id
    cfg_dict["runtime"]["activation_input_run_id"] = run_ctx.run_id
    cfg_for_next = OmegaConf.create(cfg_dict)

    observed_metric_table = _compute_metric_table_from_indices(
        cfg_dict=cfg_dict,
        run_ctx=run_ctx,
        show_progress=False,
        log_fn=_log,
    )

    null_result = None
    if bool(cfg_dict.get("analysis", {}).get("null_distribution", {}).get("enabled", False)):
        null_result = run_null_stage(cfg=cfg_for_next, run_ctx=run_ctx, observed_metric_table=observed_metric_table)
        if null_result is not None and not null_result.null_df.empty:
            adaptive_cfg = dict(cfg_dict.get("analysis", {}).get("null_distribution", {}).get("adaptive_stop", {}))
            should_stop, stop_eval = evaluate_adaptive_stop(
                observed_metrics=observed_metric_table,
                null_df=null_result.null_df,
                alpha=float(adaptive_cfg.get("alpha", 0.05)),
                correction=str(adaptive_cfg.get("correction", "bh_fdr")),
                require_all_hypotheses=bool(adaptive_cfg.get("require_all_hypotheses", True)),
                min_total_draws=int(cfg_dict.get("analysis", {}).get("null_distribution", {}).get("min_total_draws", 1)),
            )
            if not stop_eval.empty:
                stop_eval["stop_criterion_met"] = bool(should_stop)
                write_table(run_ctx.metrics_dir / "null_stop_evaluation.csv", stop_eval)

    return run_metrics_stage(
        cfg=cfg_for_next,
        run_ctx_override=run_ctx,
        precomputed_metric_table=observed_metric_table,
        null_result=null_result,
    )
