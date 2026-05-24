"""Hydra entrypoint: run only the adaptive null-distribution stage."""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig

from src.pipeline.benchmark import (
    _compute_metric_table_from_indices,
    _resolve_run_ctx_for_metrics,
    run_null_stage,
)
from src.utils.config import cfg_to_container
from src.utils.distributed import concat_gathered_tables, get_dist_context, init_distributed, shutdown_distributed
from src.utils.hydra_setup import register_hydra_resolvers

register_hydra_resolvers()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run null distribution from saved extraction artifacts."""
    cfg_dict = cfg_to_container(cfg)
    init_distributed(cfg_dict.get("runtime", {}))
    try:
        null_cfg: dict = cfg_dict.get("analysis", {}).get("null_distribution", {})

        if not null_cfg.get("enabled", False):
            print(
                "[WARN] analysis.null_distribution.enabled=false — nothing to do.\n"
                "       Re-run with analysis.null_distribution.enabled=true"
            )
            sys.exit(0)

        run_ctx = _resolve_run_ctx_for_metrics(cfg_dict)
        observed_metric_table = _compute_metric_table_from_indices(
            cfg_dict=cfg_dict,
            run_ctx=run_ctx,
            show_progress=False,
        )
        observed_metric_table = concat_gathered_tables(observed_metric_table, ctx=get_dist_context())
        result = run_null_stage(
            cfg=cfg,
            run_ctx=run_ctx,
            observed_metric_table=observed_metric_table,
        )
        if result is None:
            print("[WARN] Null stage is disabled or null artifact path is unresolved.")
            sys.exit(0)
        print(
            f"[DONE] Null stage completed: rows={len(result.null_df)} "
            f"stop_reason={result.stop_reason} partial={result.is_partial}"
        )
        print(f"[DONE] Artifact: {result.artifact_path}")
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
