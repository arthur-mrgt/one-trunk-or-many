#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_diode.sh [--depth-only]

RESOURCES_ROOT="${OTM_RESOURCES_ROOT:-$(pwd)/resources}"
DIODE_DIR="${RESOURCES_ROOT}/datasets/diode"
DEPTH_ONLY=0

for arg in "$@"; do
  if [[ "$arg" == "--depth-only" ]]; then
    DEPTH_ONLY=1
  fi
done

mkdir -p "${DIODE_DIR}"

DIODE_URLS=(
  "http://diode-dataset.s3.amazonaws.com/train.tar.gz"
  "http://diode-dataset.s3.amazonaws.com/val.tar.gz"
  "https://diode-1254389886.cos.ap-hongkong.myqcloud.com/data_list.zip"
)

if [[ "${DEPTH_ONLY}" -eq 0 ]]; then
  DIODE_URLS+=(
    "http://diode-dataset.s3.amazonaws.com/train_normals.tar.gz"
    "http://diode-dataset.s3.amazonaws.com/val_normals.tar.gz"
  )
fi

for url in "${DIODE_URLS[@]}"; do
  fname="$(basename "${url}")"
  if [[ -f "${DIODE_DIR}/${fname}" ]]; then
    echo "[INFO] Skipping existing ${fname}"
    continue
  fi
  echo "[INFO] Downloading ${fname}"
  curl -L "${url}" -o "${DIODE_DIR}/${fname}"
done

echo "[INFO] Extracting DIODE archives..."
for f in "${DIODE_DIR}"/*.tar.gz; do
  [ -e "$f" ] || continue
  tar -xzf "$f" -C "${DIODE_DIR}"
done

if [[ -f "${DIODE_DIR}/data_list.zip" ]]; then
  unzip -o "${DIODE_DIR}/data_list.zip" -d "${DIODE_DIR}"
fi

echo "[DONE] DIODE ready in ${DIODE_DIR}"
