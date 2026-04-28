param(
  [string[]]$Scenes,
  [switch]$IncludeRgb,
  [switch]$IncludeDepth,
  [switch]$IncludeMetadata,
  [switch]$IncludeNormals,
  [switch]$IncludeSemantic
)

$ErrorActionPreference = "Stop"

$cmd = @("python", "scripts/download_hypersim_subset.py")
if ($Scenes -and $Scenes.Count -gt 0) {
  $cmd += @("--scenes") + $Scenes
}
if ($IncludeRgb) { $cmd += "--include-rgb" }
if ($IncludeDepth) { $cmd += "--include-depth" }
if ($IncludeMetadata) { $cmd += "--include-metadata" }
if ($IncludeNormals) { $cmd += "--include-normals" }
if ($IncludeSemantic) { $cmd += "--include-semantic" }

Write-Host "[INFO] Running: $($cmd -join ' ')"
& $cmd[0] $cmd[1..($cmd.Length-1)]
