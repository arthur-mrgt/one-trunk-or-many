from __future__ import annotations

from typing import Any

import pandas as pd


class Tracker:
    def log_config(self, cfg: dict[str, Any]) -> None:
        return

    def log_table(self, name: str, table: pd.DataFrame) -> None:
        return

    def log_summary(self, payload: dict[str, Any]) -> None:
        return

    def finish(self) -> None:
        return


class NoopTracker(Tracker):
    pass


class WandbTracker(Tracker):
    def __init__(self, cfg: dict[str, Any], run_id: str) -> None:
        import wandb

        run_name = cfg["tracking"]["wandb"]["run_name"] or run_id
        self._wandb = wandb
        self._run = wandb.init(
            project=cfg["tracking"]["wandb"]["project"],
            entity=cfg["tracking"]["wandb"]["entity"],
            name=run_name,
            tags=cfg["tracking"]["wandb"]["tags"],
            config=cfg,
        )

    def log_table(self, name: str, table: pd.DataFrame) -> None:
        self._wandb.log({name: self._wandb.Table(dataframe=table)})

    def log_summary(self, payload: dict[str, Any]) -> None:
        self._wandb.log(payload)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


def build_tracker(cfg: dict[str, Any], run_id: str) -> Tracker:
    if cfg["tracking"]["wandb"]["enabled"]:
        return WandbTracker(cfg=cfg, run_id=run_id)
    return NoopTracker()
