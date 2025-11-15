# Orchestrator Incident Response Runbook

_Date:_ 2025-11-14

## Scope

This guide covers unplanned degradations affecting orchestration availability, end-to-end task completion, or regression guardrails introduced in Phase 6.

## First Five Minutes

1. Acknowledge alert in `#ops-memory` or `#ops-orchestrator` and assign Incident Commander.
2. Capture incident start time and affected environment (staging vs production).
3. Run `python -m pytest tests/e2e -k regression_end_to_end` locally to reproduce if safe.
4. Query `/api/v1/health` and `/metrics` to confirm service reachability.
5. Review latest Jenkins `MEMORY-SNAPSHOT-NIGHTLY` run for backup recency.

## Diagnostic Checklist

- **Task Failures:**
  - `GET /api/v1/tasks/{task_id}` for recent incidents; inspect `last_error` and `metrics` payload.
  - Confirm queue depth via `/api/v1/diagnostics/task_queue` (if enabled) or review application logs for `task_execution_failed` entries.
- **Planner/Contract Issues:**
  - Execute `python -m pytest tests/test_planner_contract_fuzz.py` to rule out schema regressions.
  - Inspect Prometheus counter `planner_contract_failure_total` for spikes.
- **Tool/Guardrail Failures:**
  - Check Alertmanager notifications tagged `team="ops-memory"` for tool latency or guardrail faults.
  - Review `/api/v1/tools/diagnostics` response using service token to verify adapter health.
- **Regression Workflow:**
  - Verify GitHub Action `Phase 6 Regression Hardening` latest run; ensure lint, contract, and e2e stages succeeded on the offending commit.

## Mitigation Steps

- Roll back to last known good deployment if Phase 6 regression tests fail reproducibly on current commit.
- Disable planner by setting `PLANNING__ENABLED=false` in environment and redeploy when planner outputs violate contract.
- Switch to sequential scheduler (`SCHEDULING__BACKEND=sequential`) if async queue starvation observed.
- Engage human reviewer workflow (`/api/v1/reviews/assign`) for high severity escalations pending automation restore.

## Communication

- Post updates every 15 minutes in incident channel including current status, mitigation progress, and blockers.
- File post-incident summary in `docs/daily-updates/{date}-incident.md` once resolved.
- Update `system_risk_assessment.md` Phase 6 section with root cause and remediation actions.

## Exit Criteria

- End-to-end regression suite (`tests/e2e/test_regression_end_to_end.py`) green.
- Planner contract fuzz tests passing.
- Prometheus alert levels returned to baseline for task completion and tool latency.
- Incident summary documented and approved in weekly operations sync.
