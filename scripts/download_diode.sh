#!/usr/bin/env bash
# download_diode.sh — download the DIODE dataset into resources/datasets/diode/
#
# MODES
#   (default)            Download the val split only  (~3 GB, recommended)
#   --scenes N           Stream N indoor train scenes without saving the full archive
#   --train-sample [N]   Stream N random scans per scene from train/{indoors,outdoor}
#                        without saving the full archive (default N=1)
#   --full               Download the complete train + val splits  (~80 GB, not recommended)
#
# EXAMPLES
#   bash scripts/download_diode.sh                  # val split — change config: split: val
#   bash scripts/download_diode.sh --scenes 10      # 10 train/indoors scenes
#   bash scripts/download_diode.sh --train-sample   # 1 random scan/scene, train indoor+outdoor
#   bash scripts/download_diode.sh --train-sample 3 # 3 random scans/scene
#   bash scripts/download_diode.sh --full           # everything (slow, large)
#
# ENV
#   OTM_RESOURCES_ROOT   override the resources/ root (default: ./resources)
#   DIODE_SAMPLE_SEED    seed for random scan selection (default 42)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODE="val"
N_SCENES=10
N_SCANS_PER_SCENE=1
SAMPLE_SEED="${DIODE_SAMPLE_SEED:-42}"
RESOURCES_ROOT="${OTM_RESOURCES_ROOT:-$(pwd)/resources}"
DIODE_DIR="${RESOURCES_ROOT}/datasets/diode"
BASE_URL="http://diode-dataset.s3.amazonaws.com"
META_URL="https://raw.githubusercontent.com/diode-dataset/diode-devkit/master/diode_meta.json"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --val)
      MODE="val"
      ;;
    --scenes)
      MODE="scenes"
      N_SCENES="${2:?'--scenes requires a number, e.g. --scenes 10'}"
      shift
      ;;
    --train-sample)
      MODE="train-sample"
      # Optional numeric argument: --train-sample [N]
      if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
        N_SCANS_PER_SCENE="$2"
        shift
      fi
      ;;
    --full)
      MODE="full"
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo "Usage: $0 [--val | --scenes N | --train-sample [N] | --full]"
      exit 1
      ;;
  esac
  shift
done

mkdir -p "${DIODE_DIR}"

# ---------------------------------------------------------------------------
# Helper: download a tar.gz, extract it, then remove the archive
# ---------------------------------------------------------------------------
download_and_extract() {
  local url="$1"
  local fname; fname="$(basename "${url}")"
  local dest="${DIODE_DIR}/${fname}"

  if [[ -f "${dest}" ]]; then
    echo "[INFO] Already downloaded, skipping: ${fname}"
  else
    echo "[INFO] Downloading ${fname} ..."
    curl -L --progress-bar "${url}" -o "${dest}"
  fi

  echo "[INFO] Extracting ${fname} ..."
  tar -xzf "${dest}" -C "${DIODE_DIR}"
  rm "${dest}"
  echo "[INFO] Done: ${fname}"
}

# ---------------------------------------------------------------------------
# Helper: detect a tar binary that supports --wildcards (GNU tar)
#   macOS ships BSD tar; install with: brew install gnu-tar
# ---------------------------------------------------------------------------
find_gnu_tar() {
  if command -v gtar &>/dev/null; then
    echo "gtar"
  elif tar --version 2>&1 | grep -q "GNU"; then
    echo "tar"
  else
    echo ""
  fi
}

# ---------------------------------------------------------------------------
# MODE: val  (default)
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "val" ]]; then
  echo "[INFO] ── VAL split only (RGB + depth + normals) ──────────────────"
  echo "[INFO] Estimated size: ~3 GB"
  download_and_extract "${BASE_URL}/val.tar.gz"
  download_and_extract "${BASE_URL}/val_normals.tar.gz"
  echo ""
  echo "[DONE] DIODE val split ready in ${DIODE_DIR}/val/"
  echo ""
  echo "       Next step — update configs/data/diode.yaml:"
  echo "         split: val"

# ---------------------------------------------------------------------------
# MODE: scenes  — stream N indoor train scenes without saving the full archive
# ---------------------------------------------------------------------------
elif [[ "${MODE}" == "scenes" ]]; then
  GNU_TAR="$(find_gnu_tar)"
  if [[ -z "${GNU_TAR}" ]]; then
    echo "[ERROR] GNU tar is required for streaming extraction but was not found."
    echo "        On macOS:  brew install gnu-tar"
    echo "        On Linux:  sudo apt-get install tar  (usually already GNU)"
    exit 1
  fi

  echo "[INFO] ── Streaming ${N_SCENES} train/indoors scene(s) ──────────────"
  echo "[INFO] Only matched files are written to disk — no full archive saved."

  # Build a wildcard argument for each scene (scene_00001 … scene_0000N)
  WILDCARDS=()
  for i in $(seq 1 "${N_SCENES}"); do
    SCENE_ID=$(printf "scene_%05d" "${i}")
    WILDCARDS+=("--wildcards" "train/indoors/${SCENE_ID}/*")
  done

  echo "[INFO] Streaming RGB + depth from train.tar.gz ..."
  echo "[INFO] (progress bar = bytes pulled from S3; matched files print as they extract)"
  curl -L --progress-bar "${BASE_URL}/train.tar.gz" \
    | "${GNU_TAR}" -xzv -C "${DIODE_DIR}" "${WILDCARDS[@]}"

  echo "[INFO] Streaming normals from train_normals.tar.gz ..."
  curl -L --progress-bar "${BASE_URL}/train_normals.tar.gz" \
    | "${GNU_TAR}" -xzv -C "${DIODE_DIR}" "${WILDCARDS[@]}"

  echo ""
  echo "[DONE] ${N_SCENES} scene(s) ready in ${DIODE_DIR}/train/indoors/"
  echo ""
  echo "       Next step — update configs/data/diode.yaml:"
  echo "         split: train"
  echo "         environment: indoors"
  echo "         n_scenes: ${N_SCENES}"

# ---------------------------------------------------------------------------
# MODE: train-sample — N random scans per scene from train/{indoors,outdoor}
#                      via wildcard streaming (no full archive saved)
# ---------------------------------------------------------------------------
elif [[ "${MODE}" == "train-sample" ]]; then
  GNU_TAR="$(find_gnu_tar)"
  if [[ -z "${GNU_TAR}" ]]; then
    echo "[ERROR] GNU tar is required for streaming extraction but was not found."
    echo "        On macOS:  brew install gnu-tar"
    echo "        On Linux:  sudo apt-get install tar  (usually already GNU)"
    exit 1
  fi
  if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 is required to parse DIODE metadata."
    exit 1
  fi

  echo "[INFO] ── Streaming ${N_SCANS_PER_SCENE} random scan(s)/scene from train ──"
  echo "[INFO] Environments: indoors + outdoor"
  echo "[INFO] Random seed:  ${SAMPLE_SEED}"
  echo "[INFO] Only matched files are written to disk — no full archive saved."

  # Fetch DIODE metadata JSON (~800 KB) and pick scans deterministically.
  META_PATH="${DIODE_DIR}/diode_meta.json"
  if [[ ! -f "${META_PATH}" ]]; then
    echo "[INFO] Downloading DIODE metadata ..."
    curl -L --progress-bar "${META_URL}" -o "${META_PATH}"
  else
    echo "[INFO] Reusing cached metadata: ${META_PATH}"
  fi

  PATTERNS_FILE="$(mktemp)"
  trap 'rm -f "${PATTERNS_FILE}"' EXIT

  python3 - "${META_PATH}" "${N_SCANS_PER_SCENE}" "${SAMPLE_SEED}" >"${PATTERNS_FILE}" <<'PY'
import json
import random
import sys
from pathlib import Path

meta_path = Path(sys.argv[1])
n_per_scene = int(sys.argv[2])
seed = int(sys.argv[3])

with meta_path.open() as f:
    meta = json.load(f)

rng = random.Random(seed)
n_scenes = {"indoors": 0, "outdoor": 0}
n_scans = {"indoors": 0, "outdoor": 0}

for env in ("indoors", "outdoor"):
    scenes = meta.get("train", {}).get(env, {})
    n_scenes[env] = len(scenes)
    for scene, scans in scenes.items():
        scan_ids = list(scans.keys())
        if len(scan_ids) <= n_per_scene:
            chosen = scan_ids
        else:
            chosen = rng.sample(scan_ids, n_per_scene)
        n_scans[env] += len(chosen)
        for scan in chosen:
            print(f"train/{env}/{scene}/{scan}/*")

print(
    f"[INFO] Selected {n_scans['indoors']} scan(s) "
    f"across {n_scenes['indoors']} indoor scene(s) and "
    f"{n_scans['outdoor']} scan(s) across {n_scenes['outdoor']} outdoor scene(s).",
    file=sys.stderr,
)
PY

  N_PATTERNS=$(wc -l <"${PATTERNS_FILE}")
  echo "[INFO] Pattern count: ${N_PATTERNS}"
  echo "[INFO] Note: full archives (~81 GB + ~126 GB compressed) must stream past"
  echo "       in order for tar to find your selected scans, even though only the"
  echo "       matched files hit disk. Expect ~60-120 min total at typical SCITAS bandwidth."

  echo "[INFO] Streaming RGB + depth from train.tar.gz ..."
  echo "[INFO] (progress bar = bytes pulled from S3; matched files print as they extract)"
  curl -L --progress-bar "${BASE_URL}/train.tar.gz" \
    | "${GNU_TAR}" -xzv -C "${DIODE_DIR}" --wildcards --files-from="${PATTERNS_FILE}"

  echo "[INFO] Streaming normals from train_normals.tar.gz ..."
  curl -L --progress-bar "${BASE_URL}/train_normals.tar.gz" \
    | "${GNU_TAR}" -xzv -C "${DIODE_DIR}" --wildcards --files-from="${PATTERNS_FILE}"

  echo ""
  echo "[DONE] Train sample ready in ${DIODE_DIR}/train/"
  echo ""
  echo "       Next step — update configs/data/diode.yaml:"
  echo "         split: train"
  echo "         environment: [indoors, outdoor]"

# ---------------------------------------------------------------------------
# MODE: full  — complete train + val download (~80 GB, not recommended)
# ---------------------------------------------------------------------------
elif [[ "${MODE}" == "full" ]]; then
  echo "[WARN] ── Full download: ~80 GB ─────────────────────────────────────"
  echo "[WARN] Press Ctrl-C within 5 seconds to cancel."
  sleep 5
  download_and_extract "${BASE_URL}/train.tar.gz"
  download_and_extract "${BASE_URL}/train_normals.tar.gz"
  download_and_extract "${BASE_URL}/val.tar.gz"
  download_and_extract "${BASE_URL}/val_normals.tar.gz"
  echo ""
  echo "[DONE] Full DIODE dataset ready in ${DIODE_DIR}/"
fi
