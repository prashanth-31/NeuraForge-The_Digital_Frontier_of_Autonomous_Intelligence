pipeline {
    agent any

    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '30', artifactNumToKeepStr: '30'))
        timestamps()
    }

    triggers {
        cron('H 2 * * *') // Nightly at ~02:00 with hash-based spread
    }

    environment {
        BACKUP_SCRIPT = "${env.WORKSPACE}/implementation/scripts/memory_backup.ps1"
        BACKUP_DIR    = "${env.WORKSPACE}/backups/memory"
        WEEKLY_DIR    = "${env.WORKSPACE}/backups/memory/weekly"
        SKIP_QDRANT   = '1'
        QDRANT_URL    = 'http://localhost:16333'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Run Memory Backup') {
            steps {
                powershell '''
                if (-not (Test-Path $env:BACKUP_SCRIPT)) {
                    throw "Memory backup script not found at $env:BACKUP_SCRIPT"
                }

                Write-Host "[Memory Snapshot] Running backup script" -ForegroundColor Cyan
                $skipQdrant = $false
                if ($env:SKIP_QDRANT -eq '1' -or $env:SKIP_QDRANT -eq 'true') {
                    $skipQdrant = $true
                }

                $invokeArgs = @()
                if ($skipQdrant) {
                    $invokeArgs += '-SkipQdrant'
                } elseif ($env:QDRANT_URL) {
                    $invokeArgs += @('-QdrantUrl', $env:QDRANT_URL)
                }

                $result = & $env:BACKUP_SCRIPT @invokeArgs
                $result | Write-Host
                if ($LASTEXITCODE -ne 0) {
                    throw "memory_backup.ps1 exited with code $LASTEXITCODE"
                }
                '''
            }
        }

        stage('Retention & Weekly Promotion') {
            steps {
                powershell '''
                $nowUtc = Get-Date -AsUtc
                $dayOfWeek = $nowUtc.DayOfWeek

                if (-not (Test-Path $env:BACKUP_DIR)) {
                    Write-Warning "Backup directory $env:BACKUP_DIR missing; retention stage skipped."
                    return
                }

                if (-not (Test-Path $env:WEEKLY_DIR)) {
                    New-Item -ItemType Directory -Path $env:WEEKLY_DIR -Force | Out-Null
                }

                $snapshots = Get-ChildItem -Path $env:BACKUP_DIR -File | Sort-Object LastWriteTime -Descending
                if ($snapshots.Count -gt 0) {
                    # Promote the most recent snapshot to the weekly folder every Sunday
                    if ($dayOfWeek -eq 'Sunday') {
                        $latest = $snapshots | Select-Object -First 1
                        Copy-Item -Path $latest.FullName -Destination $env:WEEKLY_DIR -Force
                    }

                    # Prune nightly snapshots beyond the most recent 30
                    $snapshots | Select-Object -Skip 30 | Remove-Item -Force
                }

                $weeklySnapshots = Get-ChildItem -Path $env:WEEKLY_DIR -File | Sort-Object LastWriteTime -Descending
                $weeklySnapshots | Select-Object -Skip 12 | Remove-Item -Force
                '''
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'backups/memory/**/*', fingerprint: true, allowEmptyArchive: false
            }
        }
    }
}
