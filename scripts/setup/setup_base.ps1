param(
  [string]$ModelRepo = "EPFL-VILAB/4M-7_B_CC12M"
)

$ErrorActionPreference = "Stop"

$resourcesRoot = $env:OTM_RESOURCES_ROOT
if ([string]::IsNullOrWhiteSpace($resourcesRoot)) {
  $resourcesRoot = Join-Path (Get-Location) "resources"
}

Write-Host "[INFO] Using OTM_RESOURCES_ROOT=$resourcesRoot"
Write-Host "[INFO] Model repo: $ModelRepo"

$modelDir = Join-Path $resourcesRoot ("models\4m\" + $ModelRepo.Replace("/", "\"))
$hypersimDir = Join-Path $resourcesRoot "datasets\hypersim"
$diodeDir = Join-Path $resourcesRoot "datasets\diode"

New-Item -ItemType Directory -Force -Path $modelDir, $hypersimDir, $diodeDir | Out-Null

if (-not (Get-Command huggingface-cli -ErrorAction SilentlyContinue)) {
  Write-Host "[INFO] Installing huggingface_hub CLI..."
  python -m pip install -U "huggingface_hub[cli]"
}

Write-Host "[INFO] Downloading model snapshot to $modelDir"
huggingface-cli download $ModelRepo --local-dir $modelDir

Write-Host ""
Write-Host "[DONE] Base setup complete."
Write-Host "Model: $modelDir"
Write-Host "Hypersim root: $hypersimDir"
Write-Host "DIODE root: $diodeDir"
