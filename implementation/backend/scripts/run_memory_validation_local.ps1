param(
    [int]$Iterations = 20,
    [double]$Threshold = 0.95
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendRoot = Split-Path -Parent $scriptRoot
$workspaceRoot = Split-Path -Parent $backendRoot
$workspaceRoot = Split-Path -Parent $workspaceRoot

$pythonPath = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Error "Could not locate Python executable at '$pythonPath'. Ensure the repository virtual environment is provisioned."
    exit 1
}

$env:REDIS__URL = "redis://localhost:16379/0"
$env:POSTGRES__DSN = "postgresql://postgres:postgres@localhost:15432/neuraforge"
$env:QDRANT__URL = "http://localhost:16333"

$arguments = @(
    "scripts/run_memory_validation.py",
    "--iterations", $Iterations,
    "--threshold", $Threshold,
    "--redis-url", $env:REDIS__URL,
    "--postgres-dsn", $env:POSTGRES__DSN,
    "--qdrant-url", $env:QDRANT__URL
)

Write-Host "Running memory validation harness with overrides:" -ForegroundColor Cyan
Write-Host "  Iterations: $Iterations" -ForegroundColor Cyan
Write-Host "  Threshold : $Threshold" -ForegroundColor Cyan
Write-Host "  Redis     : $env:REDIS__URL" -ForegroundColor Cyan
Write-Host "  Postgres  : $env:POSTGRES__DSN" -ForegroundColor Cyan
Write-Host "  Qdrant    : $env:QDRANT__URL" -ForegroundColor Cyan

Push-Location $backendRoot
try {
    & $pythonPath @arguments
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
    Remove-Item Env:REDIS__URL -ErrorAction SilentlyContinue
    Remove-Item Env:POSTGRES__DSN -ErrorAction SilentlyContinue
    Remove-Item Env:QDRANT__URL -ErrorAction SilentlyContinue
}

exit $exitCode
