from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pipeline.benchmark import run_metrics_stage


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_ctx = run_metrics_stage(cfg)
    print(f"[DONE] Metrics completed for run: {run_ctx.run_id}")
    print(f"[DONE] Metrics dir: {run_ctx.metrics_dir}")


if __name__ == "__main__":
    main()
