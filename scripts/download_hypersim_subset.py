#!/usr/bin/env python
"""
Download Hypersim data with Apple contrib/99991 downloader.

Default behavior downloads the full dataset.
Passing scene/modality filters enables subset download.

Reference:
https://github.com/apple/ml-hypersim/tree/main/contrib/99991
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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

    # Default: no filters => full dataset.
    if not has_scene_filter and not has_modality_filter:
        cmd = [sys.executable, str(downloader), "--directory", str(hypersim_dir)]
        if args.silent:
            cmd.append("--silent")
        run(cmd)
        print(f"[DONE] Full Hypersim dataset downloaded in {hypersim_dir}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
