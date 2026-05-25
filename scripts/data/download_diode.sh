#!/usr/bin/env bash
# download_diode.sh — download the DIODE dataset into resources/datasets/diode/
#
# MODES
#   (default)            Download the val split only  (~3 GB, recommended)
#   --scenes N           Stream N indoor train scenes (archives under
#                        _archives/; resumable curl; two-stage state file)
#   --train-sample [N]   N random scans per scene from train/{indoors,outdoor}
#                        (same: resumable archives + state — skips finished stages)
#   --full               Download the complete train + val splits  (~80 GB, not recommended)
#
# OPTIONAL FLAGS (can be combined with the modes above)
#   --with-val           Also download the full val split after the selected
#                        train mode finishes. Compatible with --scenes and
#                        --train-sample. Useful when running with a Hydra
#                        preset that uses split=train+val.
#
# EXAMPLES
#   bash scripts/download_diode.sh                              # val split — change config: split: val
#   bash scripts/download_diode.sh --scenes 10                  # 10 train/indoors scenes
#   bash scripts/download_diode.sh --train-sample               # 1 random scan/scene, train indoor+outdoor
#   bash scripts/download_diode.sh --train-sample 3             # 3 random scans/scene
#   bash scripts/download_diode.sh --train-sample 2 --with-val  # train sample + full val (~13 GB)
#   bash scripts/download_diode.sh --full                       # everything (slow, large)
#
# RESUME / COPY FROM ANOTHER MACHINE
#   Partial .tar.gz files under resources/datasets/diode/_archives/ are resumed
#   with curl -C -. After a successful gzip integrity check, archives are
#   extracted then deleted to save space.
#   --train-sample / --scenes also write a small state file so if RGB+depth
#   finished but normals crashed, the next run skips re-downloading train.tar.gz
#   (only resumes / completes train_normals.tar.gz).
#
#   Copy an already-downloaded tree from your Mac to izar (run on Mac; adjust paths/user/host):
#     rsync -avP --partial \
#       /path/on/Mac/to/one-trunk-or-many/resources/datasets/diode/ \
#       YOUR_USER@izar.epfl.ch:/scratch/izar/afares/one-trunk-or-many/resources/datasets/diode/
#   Then on izar re-run the same script mode — it will resume partial archives and/or
#   continue from the saved stage file.
#
# ENV
#   OTM_RESOURCES_ROOT   override the resources/ root (default: ./resources)
#   DIODE_SAMPLE_SEED    seed for random scan selection (default 42)
#   DIODE_RESET_STATE     set to 1 to ignore .diode_*_state files and start stages from scratch
#                         (does not delete _archives; remove that dir manually if you want a clean slate)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODE="val"
N_SCENES=10
N_SCANS_PER_SCENE=1
WITH_VAL=0
SAMPLE_SEED="${DIODE_SAMPLE_SEED:-42}"
RESOURCES_ROOT="${OTM_RESOURCES_ROOT:-$(pwd)/resources}"
DIODE_DIR="${RESOURCES_ROOT}/datasets/diode"
ARCHIVE_DIR="${DIODE_DIR}/_archives"
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
    --with-val)
      WITH_VAL=1
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo "Usage: $0 [--val | --scenes N | --train-sample [N] | --full] [--with-val]"
      exit 1
      ;;
  esac
  shift
done

# --with-val only makes sense with the train-only modes; warn otherwise.
if [[ "${WITH_VAL}" == "1" && ( "${MODE}" == "val" || "${MODE}" == "full" ) ]]; then
  echo "[WARN] --with-val is redundant with --${MODE}; the val split is already included. Ignoring --with-val."
  WITH_VAL=0
fi

mkdir -p "${DIODE_DIR}"
mkdir -p "${ARCHIVE_DIR}"

# ---------------------------------------------------------------------------
# patterns_sha256 PATH  — one line hex digest (portable)
# ---------------------------------------------------------------------------
patterns_sha256() {
  local f="$1"
  if command -v sha256sum &>/dev/null; then
    sha256sum "${f}" | awk '{print $1}'
  else
    shasum -a 256 "${f}" | awk '{print $1}'
  fi
}

# ---------------------------------------------------------------------------
# download_file_resume URL DEST
#   Resumes partial files with curl -C -. Verifies gzip integrity before return.
# ---------------------------------------------------------------------------
download_file_resume() {
  local url="$1"
  local dest="$2"
  local name
  name="$(basename "${dest}")"
  mkdir -p "$(dirname "${dest}")"

  # Legacy layout: archive sitting in dataset root → move into _archives once
  if [[ -f "${DIODE_DIR}/${name}" ]] && [[ ! -e "${dest}" ]]; then
    echo "[INFO] Moving legacy archive into ${ARCHIVE_DIR}/: ${name}"
    mv "${DIODE_DIR}/${name}" "${dest}"
  fi

  if [[ -f "${dest}" ]]; then
    if gzip -t "${dest}" 2>/dev/null; then
      echo "[INFO] Archive already complete on disk: ${name}"
      return 0
    fi
    echo "[INFO] Resuming partial download: ${name}"
    curl -fL --progress-bar -C - "${url}" -o "${dest}"
  else
    echo "[INFO] Downloading: ${name}"
    curl -fL --progress-bar "${url}" -o "${dest}"
  fi
  if ! gzip -t "${dest}" 2>/dev/null; then
    echo "[ERROR] ${name} failed gzip integrity check (truncated/corrupt). Delete and retry:"
    echo "        rm -f '${dest}'"
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Helper: download a tar.gz (resumable), extract it, then remove the archive
# ---------------------------------------------------------------------------
download_and_extract() {
  local url="$1"
  local fname; fname="$(basename "${url}")"
  local dest="${ARCHIVE_DIR}/${fname}"

  download_file_resume "${url}" "${dest}"

  echo "[INFO] Extracting ${fname} ..."
  tar -xzf "${dest}" -C "${DIODE_DIR}"
  rm -f "${dest}"
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
# Two-stage streaming modes (--scenes, --train-sample): state helpers
# ---------------------------------------------------------------------------
write_scene_state() {
  local file="$1"
  local stage="$2"
  shift 2
  umask 077
  {
    echo "version=1"
    printf '%s\n' "$@"
    echo "stage=${stage}"
  } >"${file}.tmp"
  mv -f "${file}.tmp" "${file}"
}

read_scene_state_stage() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "train_pending"
    return
  fi
  # shellcheck disable=SC1090
  grep '^stage=' "${file}" | tail -n1 | cut -d= -f2-
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
# MODE: scenes  — N indoor train scenes (resumable archives + two-stage state)
# ---------------------------------------------------------------------------
elif [[ "${MODE}" == "scenes" ]]; then
  GNU_TAR="$(find_gnu_tar)"
  if [[ -z "${GNU_TAR}" ]]; then
    echo "[ERROR] GNU tar is required for streaming extraction but was not found."
    echo "        On macOS:  brew install gnu-tar"
    echo "        On Linux:  sudo apt-get install tar  (usually already GNU)"
    exit 1
  fi

  STATE_FILE="${DIODE_DIR}/.diode_scenes_state"
  if [[ "${DIODE_RESET_STATE:-0}" == "1" ]]; then
    rm -f "${STATE_FILE}"
  fi

  echo "[INFO] ── Streaming ${N_SCENES} train/indoors scene(s) ──────────────"
  echo "[INFO] Archives + resume: ${ARCHIVE_DIR}/"
  echo "[INFO] State file: ${STATE_FILE}"

  # Build a wildcard argument for each scene (scene_00001 … scene_0000N)
  WILDCARDS=()
  for i in $(seq 1 "${N_SCENES}"); do
    SCENE_ID=$(printf "scene_%05d" "${i}")
    WILDCARDS+=("--wildcards" "train/indoors/${SCENE_ID}/*")
  done

  EXPECT_KEY="n_scenes=${N_SCENES}"
  if [[ -f "${STATE_FILE}" ]] && ! grep -q "^${EXPECT_KEY}\$" "${STATE_FILE}" 2>/dev/null; then
    echo "[WARN] State file does not match current --scenes ${N_SCENES}. Remove ${STATE_FILE} or set DIODE_RESET_STATE=1"
    exit 1
  fi

  STAGE="$(read_scene_state_stage "${STATE_FILE}")"
  if [[ "${STAGE}" == "complete" ]]; then
    echo "[INFO] Scenes download already marked complete for n_scenes=${N_SCENES}. Nothing to do."
    exit 0
  fi

  ARCH_TRAIN="${ARCHIVE_DIR}/train.tar.gz"
  ARCH_NORM="${ARCHIVE_DIR}/train_normals.tar.gz"

  if [[ "${STAGE}" == "train_pending" ]]; then
    echo "[INFO] Stage 1/2: RGB + depth from train.tar.gz ..."
    download_file_resume "${BASE_URL}/train.tar.gz" "${ARCH_TRAIN}"
    echo "[INFO] Extracting matched paths from train.tar.gz ..."
    "${GNU_TAR}" -xzf "${ARCH_TRAIN}" -C "${DIODE_DIR}" "${WILDCARDS[@]}"
    rm -f "${ARCH_TRAIN}"
    write_scene_state "${STATE_FILE}" "normals_pending" "${EXPECT_KEY}"
  else
    echo "[INFO] Stage 1/2 already done (state=${STAGE}); skipping train.tar.gz."
  fi

  echo "[INFO] Stage 2/2: normals from train_normals.tar.gz ..."
  download_file_resume "${BASE_URL}/train_normals.tar.gz" "${ARCH_NORM}"
  echo "[INFO] Extracting matched paths from train_normals.tar.gz ..."
  "${GNU_TAR}" -xzf "${ARCH_NORM}" -C "${DIODE_DIR}" "${WILDCARDS[@]}"
  rm -f "${ARCH_NORM}"
  write_scene_state "${STATE_FILE}" "complete" "${EXPECT_KEY}"

  echo ""
  echo "[DONE] ${N_SCENES} scene(s) ready in ${DIODE_DIR}/train/indoors/"
  echo ""
  echo "       Next step — update configs/data/diode.yaml:"
  echo "         split: train"
  echo "         environment: indoors"
  echo "         n_scenes: ${N_SCENES}"

# ---------------------------------------------------------------------------
# MODE: train-sample — resumable archives + two-stage state (skip finished train)
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

  STATE_FILE="${DIODE_DIR}/.diode_train_sample_state"
  if [[ "${DIODE_RESET_STATE:-0}" == "1" ]]; then
    rm -f "${STATE_FILE}"
  fi

  echo "[INFO] ── Streaming ${N_SCANS_PER_SCENE} random scan(s)/scene from train ──"
  echo "[INFO] Environments: indoors + outdoor"
  echo "[INFO] Random seed:  ${SAMPLE_SEED}"
  echo "[INFO] Archives + resume: ${ARCHIVE_DIR}/"
  echo "[INFO] State file: ${STATE_FILE}"

  # Fetch DIODE metadata JSON (~800 KB) and pick scans deterministically.
  META_PATH="${DIODE_DIR}/diode_meta.json"
  if [[ ! -f "${META_PATH}" ]]; then
    echo "[INFO] Downloading DIODE metadata ..."
    curl -fL --progress-bar "${META_URL}" -o "${META_PATH}"
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
  PAT_SHA="$(patterns_sha256 "${PATTERNS_FILE}")"
  EXPECT_KEY="n_scans=${N_SCANS_PER_SCENE}"
  EXPECT_KEY2="seed=${SAMPLE_SEED}"
  EXPECT_KEY3="patterns_sha256=${PAT_SHA}"

  if [[ -f "${STATE_FILE}" ]]; then
    if ! grep -q "^${EXPECT_KEY}\$" "${STATE_FILE}" || ! grep -q "^${EXPECT_KEY2}\$" "${STATE_FILE}" || ! grep -q "^${EXPECT_KEY3}\$" "${STATE_FILE}"; then
      echo "[WARN] Existing state file does not match current --train-sample / seed / metadata."
      echo "[WARN] Remove ${STATE_FILE} or set DIODE_RESET_STATE=1, then re-run."
      exit 1
    fi
  fi

  STAGE="$(read_scene_state_stage "${STATE_FILE}")"
  if [[ "${STAGE}" == "complete" ]]; then
    echo "[INFO] Train-sample already marked complete for this pattern set. Nothing to do."
    exit 0
  fi

  echo "[INFO] Pattern count: ${N_PATTERNS}"
  echo "[INFO] Pattern fingerprint: ${PAT_SHA}"
  echo "[INFO] Note: each stage reads the full train archive sequentially for tar;"
  echo "       only matched paths are written under ${DIODE_DIR}/train/."

  ARCH_TRAIN="${ARCHIVE_DIR}/train.tar.gz"
  ARCH_NORM="${ARCHIVE_DIR}/train_normals.tar.gz"

  if [[ "${STAGE}" == "train_pending" ]]; then
    echo "[INFO] Stage 1/2: RGB + depth from train.tar.gz ..."
    download_file_resume "${BASE_URL}/train.tar.gz" "${ARCH_TRAIN}"
    echo "[INFO] Extracting matched paths from train.tar.gz ..."
    "${GNU_TAR}" -xzf "${ARCH_TRAIN}" -C "${DIODE_DIR}" --wildcards --files-from="${PATTERNS_FILE}"
    rm -f "${ARCH_TRAIN}"
    write_scene_state "${STATE_FILE}" "normals_pending" "${EXPECT_KEY}" "${EXPECT_KEY2}" "${EXPECT_KEY3}"
  else
    echo "[INFO] Stage 1/2 already done (state=${STAGE}); skipping train.tar.gz re-download."
  fi

  echo "[INFO] Stage 2/2: normals from train_normals.tar.gz ..."
  download_file_resume "${BASE_URL}/train_normals.tar.gz" "${ARCH_NORM}"
  echo "[INFO] Extracting matched paths from train_normals.tar.gz ..."
  "${GNU_TAR}" -xzf "${ARCH_NORM}" -C "${DIODE_DIR}" --wildcards --files-from="${PATTERNS_FILE}"
  rm -f "${ARCH_NORM}"
  write_scene_state "${STATE_FILE}" "complete" "${EXPECT_KEY}" "${EXPECT_KEY2}" "${EXPECT_KEY3}"

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

# ---------------------------------------------------------------------------
# Optional: also pull the full val split when --with-val is set.
# Compatible with --scenes and --train-sample. Idempotent: download_and_extract
# uses the resumable archive directory and skips fully-extracted archives.
# ---------------------------------------------------------------------------
if [[ "${WITH_VAL}" == "1" ]]; then
  echo ""
  echo "[INFO] ── Also fetching VAL split (RGB + depth + normals) ──────────"
  echo "[INFO] Estimated size: ~3 GB"
  download_and_extract "${BASE_URL}/val.tar.gz"
  download_and_extract "${BASE_URL}/val_normals.tar.gz"
  echo ""
  echo "[DONE] VAL split ready in ${DIODE_DIR}/val/"
fi
