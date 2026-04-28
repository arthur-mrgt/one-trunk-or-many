from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.data.registry import load_dataset_pairs
from src.models.registry import build_model
from src.pipeline.extraction import run_extraction
from src.pipeline.metrics import run_metrics
from src.utils.config import RunContext, cfg_to_container, make_run_context
from src.utils.io import write_json, write_optional_parquet, write_table
from src.utils.tracking import build_tracker


def run_benchmark(cfg: DictConfig) -> RunContext:
    cfg_dict = cfg_to_container(cfg)
    run_ctx = make_run_context(cfg)
    tracker = build_tracker(cfg=cfg_dict, run_id=run_ctx.run_id)
    tracker.log_config(cfg_dict)

    model = build_model(cfg_dict["model"])

    all_metric_frames = []
    for pair in cfg_dict["metrics"]["pairs"]:
        left_mod, right_mod = pair[0], pair[1]
        pair_name = f"{left_mod}-{right_mod}"
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
        )
        write_table(run_ctx.artifacts_dir / f"activation_index_{pair_name}.csv", activation_index)

        for metric_name in cfg_dict["metrics"]["enabled"]:
            metric_df = run_metrics(
                activation_index=activation_index,
                metric_name=metric_name,
                cka_cfg=cfg_dict["metrics"]["cka"],
                pair_modalities=(left_mod, right_mod),
            )
            all_metric_frames.append(metric_df)

    if all_metric_frames:
        import pandas as pd

        metric_table = pd.concat(all_metric_frames, ignore_index=True)
    else:
        import pandas as pd

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
            "pairs": cfg_dict["metrics"]["pairs"],
            "metrics": cfg_dict["metrics"]["enabled"],
            "n_rows_metrics": int(len(metric_table)),
        },
    )
    write_json(run_ctx.run_dir / "resolved_config.json", cfg_dict)

    if not metric_table.empty:
        tracker.log_table("metrics_table", metric_table)
        tracker.log_summary({"metrics_rows": int(len(metric_table))})
    tracker.finish()
    return run_ctx
