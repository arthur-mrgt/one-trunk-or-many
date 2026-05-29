"""CLI entry point for the RQ2 transmodal triangulation analysis.

Runs purely as a post-processing step on an existing RQ1 run directory —
no new forward passes are required.

Usage
-----
python -m src.run_rq2_triangulation \\
    --run_id rq1_final_hypersim-20260525-000411 \\
    --runs_root results/runs \\
    [--n_null_draws 1000] \\
    [--n_boots 500] \\
    [--ridge 1e-4] \\
    [--seed 42] \\
    [--null_csv path/to/null_distribution.csv]

The results CSV is written to:
    <runs_root>/<run_id>/rq2/fragmentation_results.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RQ2 transmodal triangulation: fragmentation analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_id",
        required=True,
        help="ID of the RQ1 run to analyse (name of the run directory).",
    )
    parser.add_argument(
        "--runs_root",
        default="results/runs",
        help="Root directory containing run subdirectories.",
    )
    parser.add_argument(
        "--n_null_draws",
        type=int,
        default=1000,
        help="Number of row-permutation draws for the frag_cka null distribution.",
    )
    parser.add_argument(
        "--n_boots",
        type=int,
        default=200,
        help=(
            "Number of scene-level bootstrap iterations for confidence intervals. "
            "Uses a fast Gram-approximation OLS; PWCCA/kNN CIs are not bootstrapped."
        ),
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-4,
        help=(
            "Relative ridge coefficient for OLS regularisation. "
            "The actual penalty is auto-scaled as ridge * trace(A.T@A) / D."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for null draws and bootstrap.",
    )
    parser.add_argument(
        "--null_csv",
        default=None,
        help=(
            "Explicit path to the RQ1 null distribution CSV used to compute "
            "denom thresholds. If omitted, the canonical location relative to "
            "runs_root is used."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    runs_root = Path(args.runs_root)
    run_dir = runs_root / args.run_id

    if not run_dir.exists():
        log.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    null_csv = Path(args.null_csv) if args.null_csv else None

    log.info("Starting RQ2 triangulation for run: %s", args.run_id)
    log.info(
        "Config: n_null_draws=%d  n_boots=%d  ridge=%.2e  seed=%d",
        args.n_null_draws, args.n_boots, args.ridge, args.seed,
    )

    from src.pipeline.rq2_triangulation import run_rq2_triangulation

    results_df = run_rq2_triangulation(
        run_dir=run_dir,
        out_dir=run_dir,
        n_null_draws=args.n_null_draws,
        n_boots=args.n_boots,
        ridge=args.ridge,
        seed=args.seed,
        null_csv=null_csv,
    )

    out_path = run_dir / "rq2" / "fragmentation_results.csv"
    print(f"[DONE] RQ2 analysis complete.")
    print(f"[DONE] Results: {out_path}  ({len(results_df)} rows)")
    print()
    print("Summary (mean frag_cka by pair, symmetrised):")
    results_df["pair"] = results_df.apply(
        lambda r: "-".join(sorted([r["A"], r["B"]])), axis=1
    )
    summary = (
        results_df.groupby("pair")["frag_cka"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "frag_cka_mean", "std": "frag_cka_std"})
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
