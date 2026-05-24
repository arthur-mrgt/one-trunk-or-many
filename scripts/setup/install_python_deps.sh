#!/usr/bin/env bash
# Install all Python dependencies for the trunk environment in the right order.
#
# Two extra packages must be installed AFTER `pip install -r requirements.txt`:
#   - `fourm` (Apple ml-4m library, not on PyPI)
#   - `faiss-gpu-cu12` (its metadata pulls numpy>=2 which breaks fourm; we use --no-deps)
#
# Usage:
#   bash scripts/setup/install_python_deps.sh             # GPU FAISS (default)
#   bash scripts/setup/install_python_deps.sh --cpu       # CPU FAISS fallback

set -euo pipefail

FAISS_MODE="gpu"
if [[ "${1:-}" == "--cpu" ]]; then
  FAISS_MODE="cpu"
fi

echo "[INFO] Installing pinned requirements..."
python -m pip install -U pip
python -m pip install -r requirements.txt

echo "[INFO] Installing fourm (Apple ml-4m library)..."
python -m pip install git+https://github.com/apple/ml-4m.git

if [[ "$FAISS_MODE" == "gpu" ]]; then
  echo "[INFO] Installing faiss-gpu-cu12 (--no-deps to keep numpy 1.26.x)..."
  python -m pip install --no-deps faiss-gpu-cu12==1.9.0.post1
else
  echo "[INFO] Installing faiss-cpu fallback (--no-deps to keep numpy 1.26.x)..."
  python -m pip install --no-deps faiss-cpu==1.8.0
fi

echo ""
echo "[INFO] Verifying environment..."
python - <<'PY'
import numpy, torch
print(f"numpy : {numpy.__version__}")
print(f"torch : {torch.__version__} | CUDA: {torch.cuda.is_available()}")
try:
    import fourm
    print(f"fourm : {fourm.__file__}")
except Exception as e:
    print(f"fourm : FAILED ({e})")
try:
    import faiss
    has_gpu_api = hasattr(faiss, "StandardGpuResources")
    print(f"faiss : {faiss.__version__} | GPU API: {has_gpu_api}")
except Exception as e:
    print(f"faiss : FAILED ({e})")
PY

echo ""
echo "[DONE] Python dependencies installed."
