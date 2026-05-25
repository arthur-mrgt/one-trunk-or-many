"""Rewrite absolute paths in ``activation_index_*.csv`` files.

Useful after copying a run from a cluster — where activations live under an
ephemeral local path like ``/tmp/$USER/trunk_acts/<run_id>/activations/`` —
back to a workstation where activations now live next to the run dir at
``<workspace>/results/runs/<run_id>/activations/``.

The script edits CSVs in place: it replaces every occurrence of the user-
supplied old prefix with the absolute path of the local results dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier (e.g. rq1_final_hypersim-20260525-000411).",
    )
    parser.add_argument(
        "--old-prefix",
        required=True,
        help=(
            "Old activation_path prefix to replace. Should match the cluster "
            "value exactly, e.g. /tmp/margeat/trunk_acts"
        ),
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Path to the local results dir (default: results).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended replacements without writing the CSVs.",
    )
    args = parser.parse_args()

    run_dir = Path(args.results_root).resolve() / "runs" / args.run_id
    artifacts = run_dir / "artifacts"
    if not artifacts.exists():
        raise SystemExit(f"Artifacts dir not found: {artifacts}")

    new_prefix = str((Path(args.results_root).resolve() / "runs")).replace("\\", "/")

    csvs = sorted(artifacts.glob("activation_index_*.csv"))
    if not csvs:
        raise SystemExit(f"No activation_index_*.csv under {artifacts}")

    n_changed = 0
    for csv in csvs:
        df = pd.read_csv(csv)
        if "activation_path" not in df.columns:
            print(f"[SKIP]    {csv.name} (no activation_path column)")
            continue

        sample_before = str(df["activation_path"].iloc[0])
        df["activation_path"] = df["activation_path"].str.replace(
            args.old_prefix, new_prefix, regex=False
        )
        sample_after = str(df["activation_path"].iloc[0])

        if sample_before == sample_after:
            print(f"[NOOP]    {csv.name} (prefix not found)")
            continue

        print(f"[REWRITE] {csv.name}")
        print(f"  before: {sample_before}")
        print(f"  after : {sample_after}")
        if not args.dry_run:
            df.to_csv(csv, index=False)
        n_changed += 1

    print(f"\n[OK] {n_changed} CSV(s) rewritten under {artifacts}")


if __name__ == "__main__":
    main()
