# Arranca MetricLearning con el .venv de este repo (no el Python de uv).
# Uso, desde la raíz AntarSeeds:
#   .\start.ps1

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$App = Join-Path $RepoRoot "MetricLearning\app.py"

if (-not (Test-Path $Python)) {
    Write-Error "No esta el venv: $Python"
    exit 1
}

Set-Location (Join-Path $RepoRoot "MetricLearning")
& $Python $App @args
