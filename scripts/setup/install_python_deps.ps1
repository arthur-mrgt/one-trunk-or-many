# PowerShell equivalent of install_python_deps.sh
# Installs all Python dependencies in the right order.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/setup/install_python_deps.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/setup/install_python_deps.ps1 -Cpu

param(
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"

Write-Host "[INFO] Installing pinned requirements..." -ForegroundColor Cyan
python -m pip install -U pip
python -m pip install -r requirements.txt

Write-Host "[INFO] Installing fourm (Apple ml-4m library)..." -ForegroundColor Cyan
python -m pip install "git+https://github.com/apple/ml-4m.git"

if ($Cpu) {
    Write-Host "[INFO] Installing faiss-cpu fallback (--no-deps to keep numpy 1.26.x)..." -ForegroundColor Cyan
    python -m pip install --no-deps faiss-cpu==1.8.0
}
else {
    Write-Host "[INFO] Installing faiss-gpu-cu12 (--no-deps to keep numpy 1.26.x)..." -ForegroundColor Cyan
    python -m pip install --no-deps faiss-gpu-cu12==1.9.0.post1
}

Write-Host ""
Write-Host "[INFO] Verifying environment..." -ForegroundColor Cyan
python -c @"
import numpy, torch
print(f'numpy : {numpy.__version__}')
print(f'torch : {torch.__version__} | CUDA: {torch.cuda.is_available()}')
try:
    import fourm
    print(f'fourm : {fourm.__file__}')
except Exception as e:
    print(f'fourm : FAILED ({e})')
try:
    import faiss
    has_gpu_api = hasattr(faiss, 'StandardGpuResources')
    print(f'faiss : {faiss.__version__} | GPU API: {has_gpu_api}')
except Exception as e:
    print(f'faiss : FAILED ({e})')
"@

Write-Host ""
Write-Host "[DONE] Python dependencies installed." -ForegroundColor Green
