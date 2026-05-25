"""Recover a benchmark run from a remote SLURM cluster to a local workspace.

Wraps the manual ``scp`` + ``tar`` + path-rewrite flow into a single command,
so the on-disk layout under ``results/`` matches what a fully local run would
have produced. Designed to be re-run safely (idempotent ``mkdir``/overwrite).

What it pulls
-------------
1. ``results/runs/<run_id>/``  (configs, metrics, activation_index_*.csv)
2. ``results/runs/null_distributions/<run_id>/``  (null draws + state)
3. ``results/wandb/wandb/<offline-run-...>/``  (W&B local cache)
4. ``results/slurm/trunk-*_<job_id>.{out,err}``  (SLURM logs)
5. ``results/snapshots/job_<job_id>/snapshot_FINAL_*.tar``  (activations
   archive produced by ``submit_slurm.sh`` watchdog)

It then extracts the activations tar into
``results/runs/<run_id>/activations/`` and rewrites the absolute paths in
every ``activation_index_*.csv`` so they point to the local copies.

Usage
-----
Minimal:

    python scripts/data/recover_run_from_cluster.py \\
        --host margeat@izar.epfl.ch \\
        --run-id rq1_final_hypersim-20260525-000411

The W&B dir, SLURM job id, and old activation-path prefix are auto-detected
when possible; pass them explicitly with ``--wandb-dir``, ``--job-id`` and
``--old-prefix`` if the heuristics miss.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ─── Shell helpers ───────────────────────────────────────────────────────────


def run_cmd(
    cmd: list[str], *, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    """Print a command, run it, and surface failures."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def ssh_capture(host: str, remote_cmd: str) -> str:
    """Run a command on ``host`` and return its stdout (stripped)."""
    proc = run_cmd(["ssh", host, remote_cmd], capture=True, check=False)
    return (proc.stdout or "").strip()


def scp_recursive(host: str, remote_path: str, local_dest: Path) -> None:
    """``scp -r host:remote_path local_dest`` into an existing dir."""
    local_dest.mkdir(parents=True, exist_ok=True)
    run_cmd(["scp", "-r", f"{host}:{remote_path}", str(local_dest)])


def scp_file(host: str, remote_path: str, local_dest: Path) -> None:
    """``scp host:remote_path local_dest`` for a single file."""
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["scp", f"{host}:{remote_path}", str(local_dest)])


# ─── Remote auto-detection ──────────────────────────────────────────────────


def detect_wandb_dir(host: str, remote_root: str) -> str | None:
    """Return the most recent W&B offline-run directory on the remote."""
    out = ssh_capture(
        host,
        f"ls -td {remote_root}/results/wandb/wandb/offline-run-* 2>/dev/null | head -1",
    )
    return out or None


def detect_slurm_job_id(host: str, remote_root: str, run_id: str) -> str | None:
    """Find the SLURM job id by grep'ing run_id in the .out files."""
    out = ssh_capture(
        host,
        f"grep -l {run_id!r} {remote_root}/results/slurm/*.out 2>/dev/null | head -1",
    )
    if not out:
        return None
    stem = Path(out).stem  # e.g. trunk-final_2945929
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    return None


def detect_slurm_paths(host: str, remote_root: str, job_id: str) -> list[str]:
    """Return remote paths for the SLURM .out and .err of ``job_id``."""
    out = ssh_capture(
        host,
        f"ls {remote_root}/results/slurm/*_{job_id}.* 2>/dev/null",
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def detect_snapshot_tar(host: str, remote_root: str, job_id: str | None) -> str | None:
    """Find the latest FINAL snapshot tar for ``job_id`` (or any job)."""
    pattern = (
        f"{remote_root}/results/snapshots/job_{job_id}/snapshot_FINAL_*.tar"
        if job_id
        else f"{remote_root}/results/snapshots/job_*/snapshot_FINAL_*.tar"
    )
    out = ssh_capture(host, f"ls -t {pattern} 2>/dev/null | head -1")
    return out or None


# ─── CSV path rewriting ─────────────────────────────────────────────────────


def rewrite_activation_index_paths(
    *, run_id: str, results_root: Path, old_prefix: str
) -> int:
    """Rewrite absolute paths in every ``activation_index_*.csv``.

    Replaces ``old_prefix`` with the absolute local ``<results_root>/runs``
    path, then writes the CSV in-place. Returns the number of CSVs changed.
    """
    artifacts = results_root / "runs" / run_id / "artifacts"
    if not artifacts.exists():
        raise SystemExit(f"Artifacts dir not found: {artifacts}")

    new_prefix = str((results_root / "runs").resolve()).replace("\\", "/")
    n_changed = 0

    for csv in sorted(artifacts.glob("activation_index_*.csv")):
        df = pd.read_csv(csv)
        if "activation_path" not in df.columns:
            print(f"  [SKIP] {csv.name} (no activation_path column)")
            continue

        sample_before = str(df["activation_path"].iloc[0])
        df["activation_path"] = df["activation_path"].str.replace(
            old_prefix, new_prefix, regex=False
        )
        sample_after = str(df["activation_path"].iloc[0])

        if sample_before == sample_after:
            print(f"  [NOOP] {csv.name} (prefix {old_prefix!r} not found)")
            continue

        df.to_csv(csv, index=False)
        n_changed += 1
        print(f"  [REWRITE] {csv.name}")
        print(f"    before: {sample_before}")
        print(f"    after : {sample_after}")

    return n_changed


# ─── Main flow ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        required=True,
        help="SSH host or alias (e.g. margeat@izar.epfl.ch, izar1).",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier (e.g. rq1_final_hypersim-20260525-000411).",
    )
    parser.add_argument(
        "--remote-root",
        default="~/one-trunk-or-many",
        help="Project root on the remote host (default: ~/one-trunk-or-many).",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Local results directory (default: results).",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help=(
            "SLURM job id of the run. Auto-detected from SLURM .out files if "
            "omitted; needed to locate the snapshot tar."
        ),
    )
    parser.add_argument(
        "--wandb-dir",
        default=None,
        help=(
            "Specific W&B offline-run-* directory name. If omitted, the most "
            "recently modified one on the remote is selected."
        ),
    )
    parser.add_argument(
        "--old-prefix",
        default=None,
        help=(
            "Prefix to replace in activation_index_*.csv. Defaults to "
            "/tmp/<remote_user>/trunk_acts inferred from --host."
        ),
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Do not download the W&B offline cache.",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help=(
            "Do not download/extract the activations snapshot tar. Useful "
            "when only the metrics + null CSVs are needed (e.g. for plots)."
        ),
    )
    args = parser.parse_args()

    host: str = args.host
    run_id: str = args.run_id
    remote_root: str = args.remote_root
    results_root: Path = Path(args.results_root).resolve()
    remote_user = host.split("@", 1)[0] if "@" in host else "$USER"
    old_prefix: str = args.old_prefix or f"/tmp/{remote_user}/trunk_acts"

    print(f"[INFO] Recovering run {run_id!r}")
    print(f"[INFO] Remote   : {host}:{remote_root}")
    print(f"[INFO] Local    : {results_root}")
    print(f"[INFO] Old path : {old_prefix}")

    # Connectivity check.
    print("\n[CHECK] Testing SSH connectivity...")
    run_cmd(["ssh", host, "echo SSH OK on $(hostname)"])

    # Local layout skeleton.
    for sub in ("runs", "runs/null_distributions", "wandb/wandb", "slurm", "snapshots"):
        (results_root / sub).mkdir(parents=True, exist_ok=True)

    # 1) Run dir.
    print("\n[1/7] Downloading run dir...")
    scp_recursive(host, f"{remote_root}/results/runs/{run_id}", results_root / "runs")

    # 2) Null distribution dir.
    print("\n[2/7] Downloading null distribution...")
    scp_recursive(
        host,
        f"{remote_root}/results/runs/null_distributions/{run_id}",
        results_root / "runs" / "null_distributions",
    )

    # 3) W&B offline cache.
    if args.no_wandb:
        print("\n[3/7] Skipping W&B cache (--no-wandb).")
    else:
        print("\n[3/7] Resolving + downloading W&B offline cache...")
        wandb_dir = args.wandb_dir
        if wandb_dir is None:
            detected = detect_wandb_dir(host, remote_root)
            if detected is None:
                print("  [WARN] No W&B offline-run-* dir found on remote.")
            else:
                wandb_dir = Path(detected).name
                print(f"  Auto-detected W&B dir: {wandb_dir}")
        if wandb_dir:
            scp_recursive(
                host,
                f"{remote_root}/results/wandb/wandb/{wandb_dir}",
                results_root / "wandb" / "wandb",
            )

    # 4) SLURM logs.
    print("\n[4/7] Downloading SLURM logs...")
    job_id = args.job_id
    if job_id is None:
        job_id = detect_slurm_job_id(host, remote_root, run_id)
        if job_id:
            print(f"  Auto-detected job_id={job_id}")
        else:
            print("  [WARN] Could not auto-detect job_id from SLURM logs.")
    if job_id:
        for remote_path in detect_slurm_paths(host, remote_root, job_id):
            try:
                scp_file(host, remote_path, results_root / "slurm" / Path(remote_path).name)
            except subprocess.CalledProcessError:
                print(f"  [WARN] Failed to download {remote_path}")

    # 5) Snapshot tar.
    local_tar: Path | None = None
    if args.no_snapshot:
        print("\n[5/7] Skipping activations snapshot (--no-snapshot).")
    else:
        print("\n[5/7] Locating + downloading activation snapshot tar...")
        remote_tar = detect_snapshot_tar(host, remote_root, job_id)
        if remote_tar is None:
            print(
                "  [WARN] No snapshot tar found. Activations will be missing locally; "
                "metrics/plots that only need CSVs will still work."
            )
        else:
            print(f"  Remote tar: {remote_tar}")
            local_tar = results_root / "snapshots" / Path(remote_tar).name
            scp_file(host, remote_tar, local_tar)

    # 6) Extract tar into the run dir.
    if local_tar is not None and local_tar.exists():
        print("\n[6/7] Extracting activations into the run dir...")
        run_dir = results_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                "tar",
                "-xf",
                str(local_tar),
                "-C",
                str(run_dir),
                "--strip-components=2",
                f"trunk_acts/{run_id}/activations",
            ]
        )
    else:
        print("\n[6/7] No tar extracted.")

    # 7) Rewrite absolute paths in activation_index_*.csv.
    print("\n[7/7] Rewriting absolute paths in activation_index_*.csv...")
    n_changed = rewrite_activation_index_paths(
        run_id=run_id, results_root=results_root, old_prefix=old_prefix
    )
    print(f"  {n_changed} CSV(s) rewritten.")

    print(f"\n[DONE] Run {run_id} recovered at {results_root / 'runs' / run_id}")


if __name__ == "__main__":
    sys.exit(main())
