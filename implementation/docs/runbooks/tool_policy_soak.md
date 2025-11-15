# Tool Policy Staging Soak Runbook

_Date:_ 2025-11-15 09:00 UTC (scheduled)

_Owner:_ Platform Safety Engineering

## Purpose

to satisfy Phase 4 (Capability Isolation) checkpoint by executing 50 randomized orchestration runs in staging and verifying zero unauthorized tool invocations.

## Pre-Checks

1. Ensure staging stack is healthy (`docker-compose ps`, Prometheus/Grafana green).
2. Confirm latest backend build is deployed with policy enforcement (`git rev-parse HEAD`).
3. Reset in-memory policy metrics:
   ```powershell
   Invoke-RestMethod -Method POST -Uri "https://staging.neuraforge.ai/admin/metrics/reset" -Headers @{"Authorization" = "Bearer $env:STAGING_TOKEN"}
   ```
   *(skip if endpoint unavailable; metrics delta can be derived manually).* 

## Execution Steps

1. Activate virtualenv and move into backend workspace:
   ```powershell
   & E:/NeuraForge/NeuraForge-The_Digital_Frontier_of_Autonomous_Intelligence/.venv/Scripts/Activate.ps1
   Set-Location E:/NeuraForge/NeuraForge-The_Digital_Frontier_of_Autonomous_Intelligence/implementation/backend
   ```
2. Run 50 policy enforcement scenarios (random seed rotated per iteration):
   ```powershell
   1..50 | ForEach-Object {
       Write-Host "Run $_ / 50"
      $env:TOOL_POLICY_RANDOM_SEED = (Get-Random)
      E:/NeuraForge/NeuraForge-The_Digital_Frontier_of_Autonomous_Intelligence/.venv/Scripts/python.exe -m pytest tests/test_tool_policy.py::test_tool_policy_randomized --maxfail=1 --disable-warnings
       Start-Sleep -Seconds 2
   }
   ```
   *`TOOL_POLICY_RANDOM_SEED` seeds the randomized scenario selection; remove if fixture semantics change.*
3. Capture metrics snapshot after the loop:
   ```powershell
   Invoke-RestMethod -Method GET -Uri "https://staging.neuraforge.ai/metrics" -OutFile staging-metrics-$(Get-Date -Format "yyyyMMdd-HHmm").txt
   ```
4. Tail staging logs to confirm absence of `tool_policy_violation` events:
   ```powershell
   kubectl logs deploy/backend -n staging --since=1h | Select-String "tool_policy_violation"
   ```

## Acceptance Criteria

- All 50 pytest iterations report SUCCESS.
- No `tool_policy_violation` events in logs apart from intentional negative tests inside the suite.
- Prometheus counter `neuraforge_agent_tool_policy_total{outcome="violation"}` unchanged across run.

## Post-Run

1. File summary in `docs/daily-updates/YYYY-MM-DD.md` with pass/fail, links to metrics, and log snippet.
2. Update `system_risk_assessment.md` Phase 4 section with results.
3. If any violations occur, open Sev2 incident and escalate to Tooling Safety DRI.
