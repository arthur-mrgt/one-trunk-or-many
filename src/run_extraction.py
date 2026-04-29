"""Hydra entrypoint for extraction-only execution."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pipeline.benchmark import run_extraction_stage
from src.utils.hydra_setup import register_hydra_resolvers

register_hydra_resolvers()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run only the extraction stage."""
    run_ctx = run_extraction_stage(cfg)
    print(f"[DONE] Extraction completed: {run_ctx.run_id}")
    print(f"[DONE] Artifacts: {run_ctx.artifacts_dir}")


if __name__ == "__main__":
    main()
