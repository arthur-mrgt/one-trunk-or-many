"""Tracking abstractions for local and W&B logging."""

from __future__ import annotations

from typing import Any

import pandas as pd


class Tracker:
    """Base tracker interface used by pipeline stages."""

    def log_config(self, cfg: dict[str, Any]) -> None:
        """Log the resolved run configuration."""
        return

    def log_table(self, name: str, table: pd.DataFrame) -> None:
        """Log a dataframe under a tracker-specific table name."""
        return

    def log_summary(self, payload: dict[str, Any]) -> None:
        """Log scalar summary values."""
        return

    def log_line_plot(
        self,
        name: str,
        table: pd.DataFrame,
        x: str,
        y: str,
        title: str,
    ) -> None:
        """Log a line plot from a tabular dataset."""
        return

    def log_scatter_plot(
        self,
        name: str,
        table: pd.DataFrame,
        x: str,
        y: str,
        title: str,
    ) -> None:
        """Log a scatter plot from a tabular dataset."""
        return

    def finish(self) -> None:
        """Finalize the tracker run."""
        return


class NoopTracker(Tracker):
    """Tracker implementation that performs no logging."""

    pass


class WandbTracker(Tracker):
    """Tracker implementation backed by Weights and Biases."""

    def __init__(self, cfg: dict[str, Any], run_id: str) -> None:
        """Initialize a W&B run with config and metadata."""
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
        """Log a dataframe as a W&B table."""
        self._wandb.log({name: self._wandb.Table(dataframe=table)})

    def log_summary(self, payload: dict[str, Any]) -> None:
        """Log summary scalars to W&B."""
        self._wandb.log(payload)

    def log_line_plot(
        self,
        name: str,
        table: pd.DataFrame,
        x: str,
        y: str,
        title: str,
    ) -> None:
        """Log a W&B line plot from a dataframe."""
        wb_table = self._wandb.Table(dataframe=table)
        self._wandb.log(
            {
                name: self._wandb.plot.line(
                    wb_table,
                    x=x,
                    y=y,
                    title=title,
                )
            }
        )

    def log_scatter_plot(
        self,
        name: str,
        table: pd.DataFrame,
        x: str,
        y: str,
        title: str,
    ) -> None:
        """Log a W&B scatter plot from a dataframe."""
        wb_table = self._wandb.Table(dataframe=table)
        self._wandb.log(
            {
                name: self._wandb.plot.scatter(
                    wb_table,
                    x=x,
                    y=y,
                    title=title,
                )
            }
        )

    def finish(self) -> None:
        """Close the active W&B run."""
        if self._run is not None:
            self._run.finish()


def build_tracker(cfg: dict[str, Any], run_id: str) -> Tracker:
    """Build the tracker selected by config."""
    if cfg["tracking"]["wandb"]["enabled"]:
        return WandbTracker(cfg=cfg, run_id=run_id)
    return NoopTracker()
