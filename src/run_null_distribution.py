"""Hydra entrypoint: precompute null CKA distributions for significance testing.

Typical usage
-------------
# Precompute null for the latest extraction run (auto-detected):
    python -m src.run_null_distribution \\
        analysis.null_distribution.enabled=true

# Precompute null for a specific extraction run:
    python -m src.run_null_distribution \\
        analysis.null_distribution.enabled=true \\
        runtime.metrics_input_run_id=<run_id>

# Override number of draws:
    python -m src.run_null_distribution \\
        analysis.null_distribution.enabled=true \\
        analysis.null_distribution.n_draws=500

The null distribution is saved under:
    {analysis.null_distribution.artifact_dir}/{run_id}/null_distribution.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.pipeline.benchmark import _resolve_run_ctx_for_metrics
from src.analysis.null_distribution import compute_null_distribution
from src.utils.config import cfg_to_container, ensure_dir
from src.utils.hydra_setup import register_hydra_resolvers

register_hydra_resolvers()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Precompute null distribution from saved activation indices."""
    cfg_dict = cfg_to_container(cfg)
    null_cfg: dict = cfg_dict.get("analysis", {}).get("null_distribution", {})

    if not null_cfg.get("enabled", False):
        print(
            "[WARN] analysis.null_distribution.enabled=false — nothing to do.\n"
            "       Re-run with analysis.null_distribution.enabled=true"
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Resolve the source run directory (activation indices live there)
    # ------------------------------------------------------------------
    run_ctx = _resolve_run_ctx_for_metrics(cfg_dict)
    print(f"[INFO] Using activation indices from run: {run_ctx.run_id}")
    print(f"[INFO] Artifacts dir: {run_ctx.artifacts_dir}")

    index_files = sorted(run_ctx.artifacts_dir.glob("activation_index_*.csv"))
    if not index_files:
        print(
            f"[ERROR] No activation_index_*.csv files found in "
            f"'{run_ctx.artifacts_dir}'.\n"
            "        Run extraction first or set runtime.metrics_input_run_id."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Merge all pair indices into one frame (null engine filters by pair)
    # ------------------------------------------------------------------
    activation_index = pd.concat(
        [pd.read_csv(f) for f in index_files], ignore_index=True
    )
    print(
        f"[INFO] Loaded {len(activation_index)} index rows from "
        f"{len(index_files)} pair file(s)."
    )

    # ------------------------------------------------------------------
    # Output directory: {artifact_dir}/{run_id}/
    # ------------------------------------------------------------------
    artifact_dir = Path(null_cfg.get("artifact_dir", "results/runs/null_distributions"))
    out_dir = ensure_dir(artifact_dir / run_ctx.run_id)

    # Reuse guard
    csv_out = out_dir / "null_distribution.csv"
    if csv_out.exists() and null_cfg.get("reuse_if_exists", True):
        print(f"[INFO] Null artifact already exists at '{csv_out}'. Skipping.")
        print(f"[INFO] Set analysis.null_distribution.reuse_if_exists=false to recompute.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    metrics_cfg: dict = cfg_dict.get("metrics", {})
    null_df = compute_null_distribution(
        activation_index=activation_index,
        null_cfg=null_cfg,
        metrics_cfg=metrics_cfg,
        run_id=run_ctx.run_id,
        out_dir=out_dir,
    )

    print(f"[DONE] Null distribution: {len(null_df)} rows → {csv_out}")
    print(f"[DONE] Pass runtime.null_input_run_id={run_ctx.run_id} to include "
          f"p-values in the next benchmark run.")


if __name__ == "__main__":
    main()
