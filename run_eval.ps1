# CI Gating runner script to execute RAG evaluations and block builds below 80% accuracy

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "[INFO] Starting CI RAG Pipeline Evaluation..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Locate the Python virtual environment
# The active virtual environment is located at the parent workspace directory (c:\Users\hp\Documents\Rag project\venv)
$ParentVenv = Join-Path (Split-Path -Parent $ProjectDir) "venv"
$LocalVenv = Join-Path $ProjectDir "venv"

if (Test-Path "$ParentVenv\Scripts\Activate.ps1") {
    Write-Host "Activating parent workspace virtual environment at: $ParentVenv" -ForegroundColor Gray
    & "$ParentVenv\Scripts\Activate.ps1"
} elseif (Test-Path "$LocalVenv\Scripts\Activate.ps1") {
    Write-Host "Activating local virtual environment at: $LocalVenv" -ForegroundColor Gray
    & "$LocalVenv\Scripts\Activate.ps1"
} else {
    Write-Host "Warning: No virtual environment found. Attempting to execute with system Python..." -ForegroundColor Yellow
}

$EvalScript = Join-Path $ProjectDir "backend\rag\eval.py"

Write-Host "Executing evaluation script: $EvalScript" -ForegroundColor Gray
python "$EvalScript"

$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "[CI GATE SUCCESS] Pipeline meets quality metrics!" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
} else {
    Write-Host "=========================================" -ForegroundColor Red
    Write-Host "[CI GATE FAILURE] Accuracy is below standard! Commit blocked." -ForegroundColor Red
    Write-Host "=========================================" -ForegroundColor Red
}

Exit $ExitCode
