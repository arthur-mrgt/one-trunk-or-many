#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_base.sh [MODEL_REPO]

MODEL_REPO="${1:-EPFL-VILAB/4M-7_B_CC12M}"
RESOURCES_ROOT="${OTM_RESOURCES_ROOT:-$(pwd)/resources}"

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

echo ""
echo "[DONE] Base setup complete."
echo "Model: ${MODEL_DIR}"
echo "Hypersim root: ${HYPERSIM_DIR}"
echo "DIODE root: ${DIODE_DIR}"
