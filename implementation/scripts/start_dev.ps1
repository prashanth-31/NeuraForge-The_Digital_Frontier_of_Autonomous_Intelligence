Param(
    [switch]$IncludeObservability
)

$ErrorActionPreference = "Stop"

function Test-CommandExists {
    Param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string]$InstallHint = ""
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        if ([string]::IsNullOrWhiteSpace($InstallHint)) {
            throw "Required command '$Name' is not available in PATH."
        }
        throw "Required command '$Name' is not available in PATH. $InstallHint"
    }
}

Test-CommandExists -Name "docker" -InstallHint "Install Docker Desktop and ensure 'docker compose' is enabled."
Test-CommandExists -Name "poetry" -InstallHint "Install Poetry: https://python-poetry.org/docs/#installation"
Test-CommandExists -Name "bun" -InstallHint "Install Bun: https://bun.sh/docs/installation"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Write-Host "Project root: $projectRoot" -ForegroundColor Cyan

Push-Location $projectRoot
try {
    Write-Host "Starting core infrastructure (Postgres, Redis, Qdrant) via docker compose..." -ForegroundColor Cyan
    docker compose up -d postgres redis qdrant | Out-Host

    if ($IncludeObservability) {
        Write-Host "Starting observability stack (Prometheus, Grafana)..." -ForegroundColor Cyan
        docker compose up -d prometheus grafana | Out-Host
    }

    $backendDir = Join-Path $projectRoot "backend"
    $frontendDir = Join-Path $projectRoot "frontend"

    if (-not (Test-Path $backendDir)) {
        throw "Backend directory not found at '$backendDir'."
    }
    if (-not (Test-Path $frontendDir)) {
        throw "Frontend directory not found at '$frontendDir'."
    }

    $backendCommand = "Set-Location `"$backendDir`"; poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
    $frontendCommand = "Set-Location `"$frontendDir`"; bun dev --host"

    Write-Host "Launching backend development server..." -ForegroundColor Cyan
    Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", $backendCommand

    Write-Host "Launching frontend development server..." -ForegroundColor Cyan
    Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", $frontendCommand

    Write-Host "\nServices started:" -ForegroundColor Green
    Write-Host "- Backend: http://localhost:8000" -ForegroundColor Green
    Write-Host "- Frontend: http://localhost:5173" -ForegroundColor Green
    if ($IncludeObservability) {
        Write-Host "- Grafana: http://localhost:3000" -ForegroundColor Green
        Write-Host "- Prometheus: http://localhost:9090" -ForegroundColor Green
    }

    Write-Host "\nUse 'docker compose down' to stop infrastructure containers when finished." -ForegroundColor Yellow
}
finally {
    Pop-Location
}
