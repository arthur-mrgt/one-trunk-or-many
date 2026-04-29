"""Orchestrate extraction and metric stages for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from src.data.registry import load_dataset_pairs
from src.models.registry import build_model
from src.pipeline.extraction import run_extraction
from src.pipeline.metrics import run_metrics
from src.utils.config import RunContext, cfg_to_container, ensure_dir, make_run_context
from src.utils.io import write_json, write_optional_parquet, write_table
from src.utils.tracking import build_tracker


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


def _log_cka_plots(tracker, metric_table: pd.DataFrame) -> None:
    """Log CKA-vs-layer plots to the tracker."""
    cka_df = metric_table[metric_table["metric"] == "cka"].copy()
    if cka_df.empty:
        return
    for pair in sorted(cka_df["pair"].unique().tolist()):
        pair_df = cka_df[cka_df["pair"] == pair].copy()
        if pair_df.empty:
            continue
        pair_df["layer"] = pair_df["layer"].astype(str)
        pair_df["layer_order"] = pair_df["layer"].map(lambda x: _layer_sort_key(x)[0])
        pair_df = pair_df.sort_values(["layer_order", "layer"]).reset_index(drop=True)
        pair_df["layer_idx"] = pair_df["layer_order"].astype(int)
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
            "stage": "metrics",
        },
    )
    write_json(run_ctx.run_dir / "resolved_config.json", cfg_dict)

    if not metric_table.empty:
        tracker.log_table("metrics_table", metric_table)
        _log_cka_plots(tracker=tracker, metric_table=metric_table)
        tracker.log_summary({"metrics_rows": int(len(metric_table))})
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
