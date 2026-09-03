# Inicia MetricLearning con el Python del venv de AntarSeeds.
# Uso: .\start.ps1
#      .\start.ps1 --cli

$MetricRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $MetricRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$App = Join-Path $MetricRoot "app.py"

if (-not (Test-Path $Python)) {
    Write-Error "No se encontro el entorno virtual en: $Python"
    exit 1
}

Set-Location $MetricRoot
& $Python $App @args