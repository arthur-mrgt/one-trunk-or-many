"""Orchestrate extraction and metric stages for benchmark runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.analysis.significance import compute_p_value
from src.data.registry import load_dataset_pairs
from src.models.registry import build_model
from src.pipeline.extraction import run_extraction
from src.pipeline.metrics import run_metrics
from src.utils.config import RunContext, cfg_to_container, ensure_dir, make_run_context
from src.utils.io import read_null_distribution, write_json, write_optional_parquet, write_table
from src.utils.tracking import build_tracker

log = logging.getLogger(__name__)


def _list_activation_index_files(run_ctx: RunContext) -> list[Path]:
    """List activation index CSV files for a run."""
    return sorted(run_ctx.artifacts_dir.glob("activation_index_*.csv"))


def _log(message: str) -> None:
    """Print a standardized info log line."""
    print(f"[INFO] {message}")


def _layer_sort_key(layer_name: str) -> tuple[int, str]:
    """Build a stable sort key from layer naming."""
    try:
        return (int(str(layer_name).split("_")[-1]), str(layer_name))
    except Exception:
        return (10**9, str(layer_name))


def _sorted_layer_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``layer_order`` and ``layer_idx`` columns, return sorted copy."""
    df = df.copy()
    df["layer"] = df["layer"].astype(str)
    df["layer_order"] = df["layer"].map(lambda x: _layer_sort_key(x)[0])
    df = df.sort_values(["layer_order", "layer"]).reset_index(drop=True)
    df["layer_idx"] = df["layer_order"].astype(int)
    return df


def _log_cka_plots(tracker, metric_table: pd.DataFrame) -> None:
    """Log CKA-vs-layer plots to the tracker."""
    cka_df = metric_table[metric_table["metric"] == "cka"].copy()
    if cka_df.empty:
        return
    for pair in sorted(cka_df["pair"].unique().tolist()):
        pair_df = _sorted_layer_df(cka_df[cka_df["pair"] == pair])
        if pair_df.empty:
            continue
        plot_df = pair_df[["layer_idx", "layer", "value"]].copy()
        tracker.log_line_plot(
            name=f"cka_vs_layer/{pair}",
            table=plot_df,
            x="layer_idx",
            y="value",
            title=f"CKA vs Layer ({pair})",
        )
        tracker.log_scatter_plot(
            name=f"cka_vs_layer_points/{pair}",
            table=plot_df,
            x="layer_idx",
            y="value",
            title=f"CKA vs Layer Points ({pair})",
        )


def _log_significance_plots(tracker, metric_table: pd.DataFrame) -> None:
    """Log p-value and delta-vs-null-mean charts to the tracker.

    Produces two line plots per modality pair:
    - ``pvalue_vs_layer/<pair>``:        empirical p-value per layer
    - ``delta_vs_null_mean/<pair>``:  observed − null_mean per layer
    """
    needed = {"p_value", "delta_vs_null_mean", "pair", "layer", "metric"}
    if not needed.issubset(metric_table.columns):
        return

    sig_df = metric_table.dropna(subset=["p_value"]).copy()
    if sig_df.empty:
        return

    for pair in sorted(sig_df["pair"].unique().tolist()):
        pair_df = _sorted_layer_df(sig_df[sig_df["pair"] == pair])
        if pair_df.empty:
            continue

        p_plot = pair_df[["layer_idx", "layer", "p_value"]].copy()
        tracker.log_line_plot(
            name=f"pvalue_vs_layer/{pair}",
            table=p_plot,
            x="layer_idx",
            y="p_value",
            title=f"p-value vs Layer ({pair})",
        )

        delta_plot = pair_df[["layer_idx", "layer", "delta_vs_null_mean"]].copy()
        tracker.log_line_plot(
            name=f"delta_vs_null_mean/{pair}",
            table=delta_plot,
            x="layer_idx",
            y="delta_vs_null_mean",
            title=f"CKA − null_mean vs Layer ({pair})",
        )


def _resolve_null_artifact_path(cfg_dict: dict[str, Any]) -> Path | None:
    """Return the path to a precomputed null distribution CSV, or None.

    Resolution order:
    1. ``runtime.null_input_filename`` (exact versioned name, e.g.
       ``null_distribution_10scenes_2000draws.csv``) inside the run subdirectory.
    2. The latest versioned file matching
       ``null_distribution_*scenes_*draws.csv`` (most draws wins).
    3. Legacy fallback: ``null_distribution.csv``.
    """
    null_run_id = cfg_dict.get("runtime", {}).get("null_input_run_id")
    if not null_run_id:
        return None
    artifact_dir = Path(
        cfg_dict.get("analysis", {})
        .get("null_distribution", {})
        .get("artifact_dir", "")
    )
    if not artifact_dir:
        return None

    run_dir = artifact_dir / str(null_run_id)

    # 1. Explicit filename override
    explicit = cfg_dict.get("runtime", {}).get("null_input_filename")
    if explicit:
        candidate = run_dir / str(explicit)
        if candidate.exists():
            return candidate

    # 2. Latest versioned file (prefer most draws for stability)
    versioned = sorted(run_dir.glob("null_distribution_*scenes_*draws.csv"))
    if versioned:
        # Sort by n_draws (last numeric token before "draws")
        def _draws(p: Path) -> int:
            try:
                return int(p.stem.split("draws")[0].split("_")[-1])
            except ValueError:
                return 0
        return max(versioned, key=_draws)

    # 3. Legacy fallback
    for candidate in [
        run_dir / "null_distribution.csv",
        artifact_dir / "null_distribution.csv",
    ]:
        if candidate.exists():
            return candidate

    return run_dir / "null_distribution.csv"


def _enrich_with_significance(
    metric_table: pd.DataFrame,
    null_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join observed metrics with null distribution and add significance columns.

    Added columns: ``p_value``, ``null_mean``, ``null_std``,
    ``delta_vs_null_mean``, ``z_score``.
    Rows without a matching null entry receive ``NaN`` for all new fields.
    """
    sig_cols = ["p_value", "null_mean", "null_std", "delta_vs_null_mean", "z_score"]
    for col in sig_cols:
        metric_table[col] = float("nan")

    if null_df.empty:
        return metric_table

    for idx, row in metric_table.iterrows():
        mask = (
            (null_df["pair"] == row["pair"])
            & (null_df["layer"].astype(str) == str(row["layer"]))
            & (null_df["metric"] == row["metric"])
        )
        null_values = null_df.loc[mask, "null_value"].to_numpy(dtype=float)
        if len(null_values) == 0:
            continue

        observed = float(row["value"])
        null_mean = float(np.mean(null_values))
        null_std = float(np.std(null_values))

        metric_table.at[idx, "p_value"] = compute_p_value(
            observed, null_values, side="greater"
        )
        metric_table.at[idx, "null_mean"] = null_mean
        metric_table.at[idx, "null_std"] = null_std
        metric_table.at[idx, "delta_vs_null_mean"] = observed - null_mean
        metric_table.at[idx, "z_score"] = (
            (observed - null_mean) / null_std if null_std > 0 else float("nan")
        )

    return metric_table


def _resolve_run_ctx_for_metrics(cfg_dict: dict[str, Any]) -> RunContext:
    """Resolve which run directory to use for metrics stage."""
    run_id = cfg_dict["runtime"].get("metrics_input_run_id")
    runs_root = Path(cfg_dict["paths"]["runs_root"])
    if run_id:
        run_dir = runs_root / run_id
    else:
        candidates = sorted([p for p in runs_root.glob("*") if p.is_dir()])
        if not candidates:
            raise FileNotFoundError(
                "No run directory found for metrics stage. "
                "Run extraction first or set runtime.metrics_input_run_id."
            )
        run_dir = candidates[-1]
    return RunContext(
        run_id=run_dir.name,
        run_dir=run_dir,
        activations_dir=run_dir / "activations",
        metrics_dir=ensure_dir(run_dir / "metrics"),
        artifacts_dir=ensure_dir(run_dir / "artifacts"),
    )


def run_extraction_stage(cfg: DictConfig) -> RunContext:
    """Run extraction for all configured modality pairs."""
    cfg_dict = cfg_to_container(cfg)
    run_ctx = make_run_context(cfg)
    _log(f"Starting extraction stage: run_id={run_ctx.run_id}")
    model = build_model(cfg_dict["model"], cfg_dict["runtime"])

    for pair in cfg_dict["metrics"]["pairs"]:
        left_mod, right_mod = pair[0], pair[1]
        pair_name = f"{left_mod}-{right_mod}"
        _log(f"Loading samples for pair {pair_name}")
        samples = load_dataset_pairs(
            dataset_name=cfg_dict["data"]["name"],
            root=Path(cfg_dict["data"]["root"]),
            modalities=(left_mod, right_mod),
            n_scenes=int(cfg_dict["data"]["n_scenes"]),
            scene_stride=int(cfg_dict["data"]["scene_stride"]),
        )

        activation_index = run_extraction(
            model=model,
            samples=samples,
            pair_name=pair_name,
            run_id=run_ctx.run_id,
            out_dir=run_ctx.activations_dir,
            show_progress=True,
        )
        write_table(run_ctx.artifacts_dir / f"activation_index_{pair_name}.csv", activation_index)
        _log(f"Saved activation index for {pair_name}: rows={len(activation_index)}")
    write_json(run_ctx.run_dir / "resolved_config.json", cfg_dict)
    write_json(
        run_ctx.run_dir / "run_summary.json",
        {
            "run_id": run_ctx.run_id,
            "dataset": cfg_dict["data"]["name"],
            "pairs": cfg_dict["metrics"]["pairs"],
            "stage": "extraction",
            "activation_indices": [str(p) for p in _list_activation_index_files(run_ctx)],
        },
    )
    _log("Extraction stage completed.")
    return run_ctx


def run_metrics_stage(cfg: DictConfig) -> RunContext:
    """Run metrics using saved activation indices."""
    cfg_dict = cfg_to_container(cfg)
    run_ctx = _resolve_run_ctx_for_metrics(cfg_dict)
    _log(f"Starting metrics stage for run_id={run_ctx.run_id}")
    tracker = build_tracker(cfg=cfg_dict, run_id=run_ctx.run_id)
    tracker.log_config(cfg_dict)

    all_metric_frames: list[pd.DataFrame] = []
    for idx_file in _list_activation_index_files(run_ctx):
        activation_index = pd.read_csv(idx_file)
        pair_name = idx_file.stem.replace("activation_index_", "")
        _log(f"Processing metrics for pair={pair_name}")
        try:
            left_mod, right_mod = tuple(pair_name.split("-", 1))
        except ValueError:
            continue
        for metric_name in cfg_dict["metrics"]["enabled"]:
            metric_df = run_metrics(
                activation_index=activation_index,
                metric_name=metric_name,
                cka_cfg=cfg_dict["metrics"]["cka"],
                pair_modalities=(left_mod, right_mod),
                show_progress=True,
            )
            all_metric_frames.append(metric_df)

    if all_metric_frames:
        metric_table = pd.concat(all_metric_frames, ignore_index=True)
    else:
        metric_table = pd.DataFrame()

    # ------------------------------------------------------------------
    # Optional: enrich with null-distribution significance stats
    # ------------------------------------------------------------------
    null_missing_behavior: str = str(
        cfg_dict.get("runtime", {}).get("null_missing_behavior", "warn_and_skip")
    )
    null_artifact_path = _resolve_null_artifact_path(cfg_dict)
    has_significance = False

    if null_artifact_path is not None:
        _log(f"Loading null distribution from: {null_artifact_path}")
        null_df = read_null_distribution(null_artifact_path)
        if null_df.empty:
            msg = (
                f"Null artifact not found or empty at '{null_artifact_path}'. "
                "Significance stats will be omitted."
            )
            if null_missing_behavior == "error":
                raise FileNotFoundError(msg)
            log.warning(msg)
        else:
            _log(f"Null distribution loaded: {len(null_df)} rows. Enriching metrics...")
            if not metric_table.empty:
                metric_table = _enrich_with_significance(metric_table, null_df)
                has_significance = True
                _log("Significance columns added: p_value, null_mean, null_std, "
                     "delta_vs_null_mean, z_score")

    csv_path = run_ctx.metrics_dir / "metrics.csv"
    write_table(csv_path, metric_table)
    write_optional_parquet(
        path=run_ctx.metrics_dir / "metrics.parquet",
        table=metric_table,
        enabled=bool(cfg_dict["metrics"]["output"]["save_parquet"]),
    )

    write_json(
        run_ctx.run_dir / "run_summary.json",
        {
            "run_id": run_ctx.run_id,
            "dataset": cfg_dict["data"]["name"],
            "metrics": cfg_dict["metrics"]["enabled"],
            "n_rows_metrics": int(len(metric_table)),
            "has_significance": has_significance,
            "stage": "metrics",
        },
    )
    write_json(run_ctx.run_dir / "resolved_config.json", cfg_dict)

    if not metric_table.empty:
        tracker.log_table("metrics_table", metric_table)
        _log_cka_plots(tracker=tracker, metric_table=metric_table)
        if has_significance:
            _log_significance_plots(tracker=tracker, metric_table=metric_table)
        summary: dict[str, Any] = {"metrics_rows": int(len(metric_table))}
        if has_significance:
            sig_rows = metric_table.dropna(subset=["p_value"])
            if not sig_rows.empty:
                summary["n_significant_p05"] = int(
                    (sig_rows["p_value"] <= 0.05).sum()
                )
                summary["mean_p_value"] = float(sig_rows["p_value"].mean())
        tracker.log_summary(summary)
    tracker.finish()
    _log(f"Metrics stage completed. rows={len(metric_table)}")
    return run_ctx


def run_benchmark(cfg: DictConfig) -> RunContext:
    """Run extraction then metrics as one benchmark workflow."""
    run_ctx = run_extraction_stage(cfg)
    cfg_dict = cfg_to_container(cfg)
    cfg_dict["runtime"]["metrics_input_run_id"] = run_ctx.run_id
    from omegaconf import OmegaConf

    cfg_for_metrics = OmegaConf.create(cfg_dict)
    return run_metrics_stage(cfg_for_metrics)
