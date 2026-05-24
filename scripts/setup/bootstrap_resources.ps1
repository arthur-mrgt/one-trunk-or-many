param(
  [string]$ModelRepo = "EPFL-VILAB/4M-7_B_CC12M",
  [switch]$DownloadDiode
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

if ($DownloadDiode) {
  $diodeUrls = @(
    "http://diode-dataset.s3.amazonaws.com/train.tar.gz",
    "http://diode-dataset.s3.amazonaws.com/val.tar.gz",
    "http://diode-dataset.s3.amazonaws.com/train_normals.tar.gz",
    "http://diode-dataset.s3.amazonaws.com/val_normals.tar.gz",
    "https://diode-1254389886.cos.ap-hongkong.myqcloud.com/data_list.zip"
  )

  foreach ($url in $diodeUrls) {
    $fileName = [System.IO.Path]::GetFileName(($url -split "\?")[0])
    $outFile = Join-Path $diodeDir $fileName
    if (Test-Path $outFile) {
      Write-Host "[INFO] Skipping existing $fileName"
      continue
    }
    Write-Host "[INFO] Downloading $fileName"
    Invoke-WebRequest -Uri $url -OutFile $outFile
  }

  Write-Host "[INFO] Extracting DIODE archives..."
  Get-ChildItem $diodeDir -Filter "*.tar.gz" | ForEach-Object {
    tar -xzf $_.FullName -C $diodeDir
  }
  if (Test-Path (Join-Path $diodeDir "data_list.zip")) {
    Expand-Archive -Path (Join-Path $diodeDir "data_list.zip") -DestinationPath $diodeDir -Force
  }
}

Write-Host ""
Write-Host "[DONE] Model downloaded."
Write-Host ""
Write-Host "Next steps:"
Write-Host "1) Download Hypersim raw files into: $hypersimDir"
if ($DownloadDiode) {
  Write-Host "2) DIODE downloaded and extracted in: $diodeDir"
} else {
  Write-Host "2) (Optional) auto-download DIODE next time with -DownloadDiode"
}
Write-Host "3) Persist env var in your shell profile:"
Write-Host '   $env:OTM_RESOURCES_ROOT = "'$resourcesRoot'"'
