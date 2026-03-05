$ErrorActionPreference = "Stop"

$seeds = @(42, 7, 123)

$configs = @(
  "projects\qgnn-lite\configs\mutag_mlp.yaml",
  "projects\qgnn-lite\configs\mutag_vqc.yaml",
  "projects\qgnn-lite\configs\proteins_mlp.yaml",
  "projects\qgnn-lite\configs\proteins_vqc.yaml"
)

foreach ($seed in $seeds) {
  foreach ($cfg in $configs) {

    Write-Host "`n=== seed=$seed | cfg=$cfg ===" -ForegroundColor Cyan

    $tmp = New-TemporaryFile
    $text = Get-Content $cfg -Raw

    $patched = $text -replace "(?m)^\s*seed:\s*\d+\s*$", "  seed: $seed"

    Set-Content -Path $tmp -Value $patched -Encoding UTF8

    D:\Anaconda\envs\qgnn_env\python.exe projects\qgnn-lite\scripts\run_experiment.py --config $tmp

    Remove-Item $tmp -Force
  }
}