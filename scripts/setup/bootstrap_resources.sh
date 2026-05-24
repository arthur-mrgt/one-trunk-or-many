#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup/bootstrap_resources.sh [MODEL_REPO] [--download-diode] [--download-hypersim]
# Example:
#   bash scripts/setup/bootstrap_resources.sh EPFL-VILAB/4M-7_B_CC12M --download-diode

MODEL_REPO="${1:-EPFL-VILAB/4M-7_B_CC12M}"
RESOURCES_ROOT="${OTM_RESOURCES_ROOT:-$(pwd)/resources}"
DOWNLOAD_DIODE=0
DOWNLOAD_HYPERSIM=0

for arg in "$@"; do
  if [[ "$arg" == "--download-diode" ]]; then
    DOWNLOAD_DIODE=1
  elif [[ "$arg" == "--download-hypersim" ]]; then
    DOWNLOAD_HYPERSIM=1
  fi
done

echo "[INFO] Using OTM_RESOURCES_ROOT=${RESOURCES_ROOT}"
echo "[INFO] Model repo: ${MODEL_REPO}"

MODEL_DIR="${RESOURCES_ROOT}/models/4m/${MODEL_REPO}"
HYPERSIM_DIR="${RESOURCES_ROOT}/datasets/hypersim"
DIODE_DIR="${RESOURCES_ROOT}/datasets/diode"

mkdir -p "${MODEL_DIR}" "${HYPERSIM_DIR}" "${DIODE_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[INFO] Installing huggingface_hub CLI..."
  python -m pip install -U "huggingface_hub[cli]"
fi

echo "[INFO] Downloading model snapshot to ${MODEL_DIR}"
huggingface-cli download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"

if [[ "${DOWNLOAD_DIODE}" -eq 1 ]]; then
  echo "[INFO] Downloading DIODE archives..."
  DIODE_URLS=(
    "http://diode-dataset.s3.amazonaws.com/train.tar.gz"
    "http://diode-dataset.s3.amazonaws.com/val.tar.gz"
    "http://diode-dataset.s3.amazonaws.com/train_normals.tar.gz"
    "http://diode-dataset.s3.amazonaws.com/val_normals.tar.gz"
    "https://diode-1254389886.cos.ap-hongkong.myqcloud.com/data_list.zip"
  )
  for url in "${DIODE_URLS[@]}"; do
    fname="$(basename "${url}")"
    if [[ -f "${DIODE_DIR}/${fname}" ]]; then
      echo "[INFO] Skipping existing ${fname}"
      continue
    fi
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
fi

if [[ "${DOWNLOAD_HYPERSIM}" -eq 1 ]]; then
  echo "[INFO] Downloading Hypersim using official Apple script..."
  TMP_HYPERSIM_DIR="${RESOURCES_ROOT}/tmp/ml-hypersim"
  mkdir -p "${TMP_HYPERSIM_DIR}"
  if [[ ! -d "${TMP_HYPERSIM_DIR}/.git" ]]; then
    git clone https://github.com/apple/ml-hypersim "${TMP_HYPERSIM_DIR}"
  fi
  python "${TMP_HYPERSIM_DIR}/code/python/tools/dataset_download_images.py" \
    --downloads_dir "${HYPERSIM_DIR}/downloads" \
    --decompress_dir "${HYPERSIM_DIR}/scenes"
fi

cat <<EOF

[DONE] Model downloaded.

Next steps:
1) Hypersim directory:
   ${HYPERSIM_DIR}
   (auto-download if you passed --download-hypersim)
2) DIODE directory:
   ${DIODE_DIR}
   (auto-download if you passed --download-diode)
3) Export in your shell/profile:
   export OTM_RESOURCES_ROOT="${RESOURCES_ROOT}"

EOF
