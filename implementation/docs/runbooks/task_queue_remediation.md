# Task Queue Remediation Runbook

_Last updated: 2025-10-17_

This runbook describes how to diagnose and remediate stuck or failed tasks in the NeuraForge orchestration queue. It assumes on-call access to the staging or production environment with Grafana, Prometheus, and the FastAPI API surface.

## 1. Signals to Watch
- **Prometheus metrics:**
  - `neuraforge_orchestrator_runs_active` rising without matching completions.
  - `orchestrator:throughput_per_minute` dropping below expected baseline.
  - `neuraforge_orchestrator_success_rate` trending under 0.9.
  - Alert rules `TaskLatencyWarning`, `TaskLatencyCritical`, and `OrchestratorHighFailureRate` firing simultaneously.
- **Logs:** `task_execution_failed`, `task_submission_failed`, or repeated guardrail deny events in structlog output.
- **Queue depth:** For Redis-backed deployments check `LLEN neuraforge:queue:pending` (the namespace is configurable via `settings.redis.task_queue_db`). For in-memory queues, rely on the metrics above and the orchestrator trace stream.

## 2. Immediate Stabilisation
1. **Acknowledge alerts** in PagerDuty/Slack to avoid duplicate escalations.
2. **Confirm environment health**:
   - `docker compose ps` (staging) or kubectl equivalent (production) to ensure `backend`, `prometheus`, `alertmanager`, and any worker pods are healthy.
   - Check Redis availability if rate limiting or queue persistence is enabled.
3. **Throttle new intake** if backlog drains slowly:
   - Temporarily raise the reviewer rate-limit window via environment overrides, or pause the upstream submission service if possible.

## 3. Clearing the Backlog
1. **Inspect the offending tasks**:
   - Use `GET /api/v1/tasks/{task_id}` or the reviewer console to see negotiation state, guardrail decisions, and tool failures.
   - Review the orchestrator simulation logs (`task_lifecycle` annotations in Grafana) to understand where execution stalls.
2. **Retry or resubmit**:
   - If the task failed due to transient tooling, resubmit via `POST /api/v1/submit_task` using the original payload (available in Task History or Audit logs).
   - For queued but unprocessed tasks, restarting the worker pod or `backend` service drains the in-memory queue; pending items will be reloaded from Postgres on startup.
3. **Scale out**:
   - Increase the number of orchestrator workers (horizontal pod autoscale / additional compose service instances) when CPU-bound agents are the bottleneck.

## 4. Dead-Letter Handling
- Tasks that fail deterministically after three retries should be marked for manual follow-up:
   1. Record the task ID and failure reason in the incident log.
   2. Escalate via the reviewer console (`/reviews`) to ensure human oversight before re-queueing.
   3. Capture the payload and negotiation transcript in `orchestration_runs` / `task_lifecycle_events` for auditing (psql example below).
- There is no automated dead-letter queue today; on-call responders maintain a manual list in the incident doc. Once the root cause is fixed, re-submit the payload and annotate the reviewer ticket with remediation steps and new task ID.
```
-- Example: fetch the latest orchestration payload for a failed task
SELECT task_id, status, state
FROM orchestration_runs
WHERE task_id = '<task_id>'
ORDER BY updated_at DESC
LIMIT 1;
```

## 5. Post-Mortem and Follow-Up
- Capture key timestamps and remediation actions in `docs/daily/<YYYY-MM-DD>.md`.
- Update alert thresholds if false positives were triggered or if the remediation required new instrumentation.
- File issues for long-term fixes (e.g., tool integration hardening, SLA adjustments) before closing the incident.

## Appendix
- **Related metrics:** `neuraforge_task_latency_seconds`, `neuraforge_guardrail_decision_total`, `neuraforge_orchestrator_escalations_total`.
- **Reference docs:** `docs/observability/grafana_dashboards.md`, `docs/runbooks/reviewer_operations.md`, `scripts/staging_sync_artifacts.py`.
