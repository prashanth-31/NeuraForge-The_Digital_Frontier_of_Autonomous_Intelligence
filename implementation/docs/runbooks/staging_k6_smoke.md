# Staging k6 Smoke Procedure

_Last updated: 2025-10-17_

Use this runbook after syncing observability artifacts from CI to validate that the staging environment can sustain basic task submission traffic. The procedure exercises the `/api/v1/submit_task` path with the standard k6 script and verifies that metrics flow into Prometheus and Grafana.

## Prerequisites
- Staging environment is reachable (`https://staging-api.neuraforge.ai`).
- Valid reviewer JWT with `reviewer` scope for API access.
- `k6` CLI installed locally or available in the operator container.
- Latest artifacts pulled via `scripts/staging_sync_artifacts.py` (dashboards, alert rules, k6 script).

## Steps
1. **Sync artifacts**
   ```powershell
   cd implementation/scripts
   python ../backend/.venv/Scripts/python.exe staging_sync_artifacts.py --workflow phase5-observability
   ```
   - Confirms Grafana dashboards and the `k6-submit-task.js` script match the latest CI release.
   - If prompted, supply a GitHub token with `actions:read` scope.
2. **Set environment variables**
   ```powershell
   $env:NEURAFORGE_BASE_URL = "https://staging-api.neuraforge.ai"
   $env:NEURAFORGE_VUS = "5"
   $env:NEURAFORGE_DURATION = "2m"
   $env:NEURAFORGE_SLEEP = "1"
   $env:NEURAFORGE_JWT = "<paste reviewer JWT>"
   ```
   - Adjust `NEURAFORGE_VUS` or `NEURAFORGE_DURATION` if running outside the standard smoke window.
3. **Run the smoke test**
   ```powershell
   cd implementation/scripts/loadtesting
   k6 run --tag environment=staging --tag test=smoke --header "Authorization=Bearer $env:NEURAFORGE_JWT" k6-submit-task.js
   ```
   - Observe k6 output; the run should report `status is 200` checks passing and zero failures.
4. **Verify observability signals**
   - Prometheus: confirm `submit_task_latency` and `submit_task_failures` series appear with the `environment="staging"` label.
   - Grafana: open the Reviewer Workload dashboard and check the smoke event annotation (Workflow sync adds annotation support).
   - Alertmanager: ensure no critical alerts fire; a transient warning is acceptable if concurrency crosses thresholds.
5. **Record results**
   - Capture the k6 summary and key Grafana screenshots.
   - Update `docs/daily/<YYYY-MM-DD>.md` with run details, including any anomalies and follow-up actions.
   - If failures occurred, file an incident ticket and leave the staging environment in a stable state before rerunning.

## Cleanup
- Clear exported environment variables (`Remove-Item Env:NEURAFORGE_*`).
- Revoke temporary reviewer tokens if they were generated solely for the smoke test.

## Troubleshooting
- k6 fails with `401 Unauthorized`: confirm the reviewer JWT still exists and includes the `reviewer` scope; generate a fresh token if the smoke window expired.
- Metrics missing in Prometheus: check that the backend container restarted with the synced dashboards by tailing the staging API logs (`docker compose logs backend`) and verifying the `.env` values match the operator export.
- Alertmanager raises critical alerts: acknowledge the alert, reduce `NEURAFORGE_VUS` to baseline values, and double-check that staging alert thresholds mirror production to avoid noisy rehearsals.
- Grafana dashboards show stale data: ensure `scripts/staging_sync_artifacts.py` ran within the last 24 hours and clear cached browser assets before reloading the panels.

## References
- `implementation/scripts/staging_sync_artifacts.py`
- `implementation/scripts/loadtesting/k6-submit-task.js`
- `implementation/docs/runbooks/reviewer_operations.md`
