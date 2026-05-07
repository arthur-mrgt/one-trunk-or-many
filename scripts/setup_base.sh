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

# Tokenizer repos (separate HuggingFace checkpoints)
TOK_DEPTH_REPO="EPFL-VILAB/4M_tokenizers_depth_8k_224-448"
TOK_NORMAL_REPO="EPFL-VILAB/4M_tokenizers_normal_8k_224-448"
# TOK_RGB_REPO="EPFL-VILAB/4M_tokenizers_rgb_16k_224-448"     # only needed for tokenized RGB mode
# TOK_SEMSEG_REPO="EPFL-VILAB/4M_tokenizers_semseg_4k_224-448" # semseg disabled

mkdir -p \
  "${MODEL_DIR}" \
  "${RESOURCES_ROOT}/models/4m/${TOK_DEPTH_REPO}" \
  "${RESOURCES_ROOT}/models/4m/${TOK_NORMAL_REPO}" \
  "${HYPERSIM_DIR}" \
  "${DIODE_DIR}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[INFO] Installing huggingface_hub CLI..."
  python -m pip install -U "huggingface_hub[cli]"
fi

echo "[INFO] Downloading main model to ${MODEL_DIR}"
hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"

echo "[INFO] Downloading depth tokenizer"
hf download "${TOK_DEPTH_REPO}" --local-dir "${RESOURCES_ROOT}/models/4m/${TOK_DEPTH_REPO}"

echo "[INFO] Downloading normal tokenizer"
hf download "${TOK_NORMAL_REPO}" --local-dir "${RESOURCES_ROOT}/models/4m/${TOK_NORMAL_REPO}"

echo ""
echo "[DONE] Base setup complete."
echo "Model:          ${MODEL_DIR}"
echo "Depth tok:      ${RESOURCES_ROOT}/models/4m/${TOK_DEPTH_REPO}"
echo "Normal tok:     ${RESOURCES_ROOT}/models/4m/${TOK_NORMAL_REPO}"
echo "Hypersim root:  ${HYPERSIM_DIR}"
echo "DIODE root:     ${DIODE_DIR}"
