param(
  [switch]$DepthOnly
)

$ErrorActionPreference = "Stop"

$resourcesRoot = $env:OTM_RESOURCES_ROOT
if ([string]::IsNullOrWhiteSpace($resourcesRoot)) {
  $resourcesRoot = Join-Path (Get-Location) "resources"
}

$diodeDir = Join-Path $resourcesRoot "datasets\diode"
New-Item -ItemType Directory -Force -Path $diodeDir | Out-Null

$diodeUrls = @(
  "http://diode-dataset.s3.amazonaws.com/train.tar.gz",
  "http://diode-dataset.s3.amazonaws.com/val.tar.gz",
  "https://diode-1254389886.cos.ap-hongkong.myqcloud.com/data_list.zip"
)

if (-not $DepthOnly) {
  $diodeUrls += @(
    "http://diode-dataset.s3.amazonaws.com/train_normals.tar.gz",
    "http://diode-dataset.s3.amazonaws.com/val_normals.tar.gz"
  )
}

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

Write-Host "[DONE] DIODE ready in $diodeDir"
