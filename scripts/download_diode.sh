#!/usr/bin/env bash
# download_diode.sh — download the DIODE dataset into resources/datasets/diode/
#
# MODES
#   (default)            Download the val split only  (~3 GB, recommended)
#   --scenes N           Stream N indoor train scenes without saving the full archive
#   --full               Download the complete train + val splits  (~80 GB, not recommended)
#
# EXAMPLES
#   bash scripts/download_diode.sh                  # val split — change config: split: val
#   bash scripts/download_diode.sh --scenes 10      # 10 train/indoors scenes
#   bash scripts/download_diode.sh --full           # everything (slow, large)
#
# ENV
#   OTM_RESOURCES_ROOT   override the resources/ root (default: ./resources)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODE="val"
N_SCENES=10
RESOURCES_ROOT="${OTM_RESOURCES_ROOT:-$(pwd)/resources}"
DIODE_DIR="${RESOURCES_ROOT}/datasets/diode"
BASE_URL="http://diode-dataset.s3.amazonaws.com"

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
    --full)
      MODE="full"
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo "Usage: $0 [--val | --scenes N | --full]"
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
  curl -sL "${BASE_URL}/train.tar.gz" \
    | "${GNU_TAR}" -xz -C "${DIODE_DIR}" "${WILDCARDS[@]}"

  echo "[INFO] Streaming normals from train_normals.tar.gz ..."
  curl -sL "${BASE_URL}/train_normals.tar.gz" \
    | "${GNU_TAR}" -xz -C "${DIODE_DIR}" "${WILDCARDS[@]}"

  echo ""
  echo "[DONE] ${N_SCENES} scene(s) ready in ${DIODE_DIR}/train/indoors/"
  echo ""
  echo "       Next step — update configs/data/diode.yaml:"
  echo "         split: train"
  echo "         environment: indoors"
  echo "         n_scenes: ${N_SCENES}"

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
