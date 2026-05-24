#!/usr/bin/env python
"""Download full or subset Hypersim data with Apple contrib downloader."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    """Run a subprocess command and fail on errors."""
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _scene_zip_url(scene_id: str) -> str:
    """Build the download URL for one scene archive."""
    return (
        "https://docs-assets.developer.apple.com/ml-research/datasets/"
        f"hypersim/v1/scenes/{scene_id}.zip"
    )


def _download_scene_zip(scene_id: str, downloads_dir: Path) -> Path:
    """Download a scene zip if missing and return its path."""
    downloads_dir.mkdir(parents=True, exist_ok=True)
    zip_path = downloads_dir / f"{scene_id}.zip"
    if zip_path.exists():
        # Validate cache: partially downloaded files can exist but be invalid zips.
        if zipfile.is_zipfile(zip_path):
            return zip_path
        print(f"[WARN] Cached archive is invalid, re-downloading: {zip_path}")
        zip_path.unlink(missing_ok=True)
    url = _scene_zip_url(scene_id)
    print(f"[INFO] Downloading scene archive: {url}")
    urllib.request.urlretrieve(url, zip_path)
    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink(missing_ok=True)
        raise zipfile.BadZipFile(
            f"Downloaded file is not a valid zip archive: {zip_path}"
        )
    return zip_path


def _should_extract(member: str, scene_id: str, args: argparse.Namespace) -> bool:
    """Return whether a zip member matches requested filters."""
    if not member.startswith(f"{scene_id}/"):
        return False
    if member.endswith("/"):
        return False

    # If scene is requested but no modality flag is set, extract everything for that scene.
    modality_flags = any(
        [
            args.include_rgb,
            args.include_depth,
            args.include_metadata,
            args.include_normals,
            args.include_semantic,
        ]
    )
    if not modality_flags:
        return True

    checks: list[bool] = []
    if args.include_rgb:
        checks.append("final_hdf5" in member and member.endswith(".color.hdf5"))
    if args.include_depth:
        checks.append("geometry_hdf5" in member and member.endswith(".depth_meters.hdf5"))
    if args.include_normals:
        checks.append("geometry_hdf5" in member and member.endswith(".normal_cam.hdf5"))
        checks.append("geometry_hdf5" in member and member.endswith(".normal_world.hdf5"))
    if args.include_semantic:
        checks.append("geometry_hdf5" in member and member.endswith(".semantic.hdf5"))
        checks.append("geometry_hdf5" in member and member.endswith(".semantic_instance.hdf5"))
    if args.include_metadata:
        checks.append(member.endswith("_detail/metadata_cameras.csv"))
        checks.append(member.endswith("_detail/metadata_scene.csv"))
    return any(checks)


def _extract_scene_subset(
    scene_id: str,
    zip_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Extract only selected files from one scene archive."""
    print(f"[INFO] Extracting selected files from {zip_path.name}")
    extracted = 0
    skipped_existing = 0
    t0 = time.perf_counter()
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if _should_extract(m, scene_id, args)]
        for member in members:
            target = out_dir / member
            if target.exists() and not args.force_extract:
                skipped_existing += 1
                continue
            zf.extract(member, path=out_dir)
            extracted += 1
    dt = time.perf_counter() - t0
    print(
        f"[INFO] Scene {scene_id}: extracted={extracted}, "
        f"skipped_existing={skipped_existing}, elapsed={dt:.1f}s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for subset download."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resources-root",
        default=os.environ.get("OTM_RESOURCES_ROOT", str(Path.cwd() / "resources")),
        help="Root folder containing models/ and datasets/ (default: ./resources).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional scene IDs to limit the download, e.g. ai_001_001 ai_001_002",
    )
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="Download RGB color files (*.color.hdf5).",
    )
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="Download depth files (*.depth_meters.hdf5).",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Download key per-scene metadata files.",
    )
    parser.add_argument(
        "--include-normals",
        action="store_true",
        help="Download normal maps (*.normal_cam.hdf5 and *.normal_world.hdf5).",
    )
    parser.add_argument(
        "--include-semantic",
        action="store_true",
        help="Download semantic masks (*.semantic.hdf5 and *.semantic_instance.hdf5).",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Pass --silent to contrib script.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract files even if they already exist in destination.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute download workflow and return process exit code."""
    args = parse_args()
    t_global = time.perf_counter()

    has_modality_filter = any(
        [
            args.include_rgb,
            args.include_depth,
            args.include_metadata,
            args.include_normals,
            args.include_semantic,
        ]
    )
    has_scene_filter = bool(args.scenes)

    resources_root = Path(args.resources_root).resolve()
    hypersim_dir = resources_root / "datasets" / "hypersim"
    tmp_repo = resources_root / "tmp" / "ml-hypersim"

    hypersim_dir.mkdir(parents=True, exist_ok=True)
    tmp_repo.mkdir(parents=True, exist_ok=True)

    if not (tmp_repo / ".git").exists():
        run(["git", "clone", "https://github.com/apple/ml-hypersim", str(tmp_repo)])

    downloader = tmp_repo / "contrib" / "99991" / "download.py"
    if not downloader.exists():
        raise FileNotFoundError(f"Missing downloader script: {downloader}")

    # Copy scene-type metadata from the cloned repo if not already present.
    meta_src = tmp_repo / "contrib" / "99991" / "metadata_camera_trajectories.csv"
    meta_dst = hypersim_dir / "metadata_camera_trajectories.csv"
    if not meta_dst.exists():
        if meta_src.exists():
            shutil.copy2(meta_src, meta_dst)
            print(f"[INFO] Copied metadata_camera_trajectories.csv → {meta_dst}")
        else:
            print(f"[WARN] metadata_camera_trajectories.csv not found in cloned repo at {meta_src}")

    # Default: no filters => full dataset.
    if not has_scene_filter and not has_modality_filter:
        cmd = [sys.executable, str(downloader), "--directory", str(hypersim_dir)]
        if args.silent:
            cmd.append("--silent")
        run(cmd)
        print(f"[DONE] Full Hypersim dataset downloaded in {hypersim_dir}")
        return 0

    # Fast path for subset downloads: scene ZIP + selective extraction.
    if has_scene_filter:
        downloads_dir = hypersim_dir / "downloads"
        downloaded_count = 0
        reused_zip_count = 0
        for scene in args.scenes:
            scene_t0 = time.perf_counter()
            scene_zip = downloads_dir / f"{scene}.zip"
            if scene_zip.exists():
                reused_zip_count += 1
                print(f"[INFO] Reusing cached scene archive: {scene_zip}")
            else:
                downloaded_count += 1
            zip_path = _download_scene_zip(scene_id=scene, downloads_dir=downloads_dir)
            _extract_scene_subset(
                scene_id=scene,
                zip_path=zip_path,
                out_dir=hypersim_dir,
                args=args,
            )
            print(f"[INFO] Scene {scene}: total elapsed={time.perf_counter() - scene_t0:.1f}s")
        print(
            f"[INFO] Scene archives: downloaded={downloaded_count}, "
            f"reused_cache={reused_zip_count}"
        )
        print(f"[INFO] Total elapsed={time.perf_counter() - t_global:.1f}s")
        print(f"[DONE] Hypersim subset downloaded in {hypersim_dir}")
        return 0

    modality_filters: list[list[str]] = []
    if args.include_rgb:
        modality_filters.append(["final_hdf5", ".color.hdf5"])
    if args.include_depth:
        modality_filters.append(["geometry_hdf5", ".depth_meters.hdf5"])
    if args.include_normals:
        modality_filters.append(["geometry_hdf5", ".normal_cam.hdf5"])
        modality_filters.append(["geometry_hdf5", ".normal_world.hdf5"])
    if args.include_semantic:
        modality_filters.append(["geometry_hdf5", ".semantic.hdf5"])
        modality_filters.append(["geometry_hdf5", ".semantic_instance.hdf5"])
    if args.include_metadata:
        modality_filters.append(["_detail/metadata_cameras.csv"])
        modality_filters.append(["_detail/metadata_scene.csv"])

    scene_filters = args.scenes if has_scene_filter else [None]

    # If only scene filters are provided, download all files for those scenes.
    if has_scene_filter and not has_modality_filter:
        modality_filters = [[]]

    for scene in scene_filters:
        for mod in modality_filters:
            contains: list[str] = []
            if scene is not None:
                contains.append(scene)
            contains.extend(mod)

            cmd = [sys.executable, str(downloader), "--directory", str(hypersim_dir)]
            if contains:
                cmd.extend(["--contains", *contains])
            if args.silent:
                cmd.append("--silent")
            run(cmd)

    print(f"[DONE] Hypersim subset downloaded in {hypersim_dir}")
    print(f"[INFO] Total elapsed={time.perf_counter() - t_global:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
