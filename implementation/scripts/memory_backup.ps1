param(
    [string]$OutputDirectory = "backups/memory",
    [switch]$SkipQdrant,
    [string]$QdrantCollection = "neura_tasks",
    [string]$QdrantUrl = "http://localhost:16333"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeRoot = Split-Path -Parent $scriptDir
$workspaceRoot = Split-Path -Parent $composeRoot

$resolvedOutput = Join-Path $workspaceRoot $OutputDirectory
if (-not (Test-Path $resolvedOutput)) {
    New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
}

$timestamp = (Get-Date).ToString("yyyyMMdd-HHmmss")
$postgresFile = Join-Path $resolvedOutput "postgres-$timestamp.sql"
$redisFile = Join-Path $resolvedOutput "redis-$timestamp.rdb"
$qdrantFile = Join-Path $resolvedOutput "qdrant-$timestamp.snapshot"

function Invoke-DockerCompose {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$Args = @()
    )

    Push-Location $composeRoot
    try {
        $result = & docker compose $Command @Args
        $exit = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    if ($exit -ne 0) {
        throw "docker compose $Command failed with exit code $exit"
    }

    return $result
}

Write-Host "[Phase5] Exporting Postgres episodic memory..." -ForegroundColor Cyan
$pgDump = Invoke-DockerCompose -Command "exec" -Args @("-T", "postgres", "pg_dump", "-U", "postgres", "neuraforge")
$pgDump | Out-File -FilePath $postgresFile -Encoding utf8

Write-Host "[Phase5] Exporting Redis working memory snapshot..." -ForegroundColor Cyan
$redisTemp = "/tmp/redis-$timestamp.rdb"
Invoke-DockerCompose -Command "exec" -Args @("redis", "redis-cli", "--rdb", $redisTemp)
Invoke-DockerCompose -Command "cp" -Args @("redis:$redisTemp", $redisFile)
Invoke-DockerCompose -Command "exec" -Args @("redis", "rm", "-f", $redisTemp)

if (-not $SkipQdrant) {
    Write-Host "[Phase5] Capturing Qdrant semantic snapshot..." -ForegroundColor Cyan
    $snapshotResponse = Invoke-RestMethod -Method Post -Uri "$QdrantUrl/collections/$QdrantCollection/snapshots"
    $snapshotName = $snapshotResponse.name
    if (-not $snapshotName) {
        throw "Failed to create Qdrant snapshot via $QdrantUrl"
    }
    $snapshotUri = "$QdrantUrl/collections/$QdrantCollection/snapshots/$snapshotName"
    Invoke-WebRequest -Uri $snapshotUri -OutFile $qdrantFile | Out-Null
}

Write-Host "[Phase5] Memory backups complete:" -ForegroundColor Green
Write-Host "  Postgres dump : $postgresFile" -ForegroundColor Green
Write-Host "  Redis dump    : $redisFile" -ForegroundColor Green
if (-not $SkipQdrant) {
    Write-Host "  Qdrant dump   : $qdrantFile" -ForegroundColor Green
}
