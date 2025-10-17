# Reviewer Operations Runbook

_Last updated: 2025-10-17_

This runbook covers the signals and response steps for NeuraForge's reviewer queue. It aligns alert thresholds with the latest Prometheus metrics and Grafana dashboards.

## 1. Key Dashboards

### 1.1 Task Latency Overview
- **Dashboard:** `Observability / Task & Guardrail Health`
- **Panels:** `Task p95 latency by agent-count`, `Guardrail decisions per policy`
- **Data Source:** Prometheus (`neuraforge_task_latency_seconds`, `neuraforge_guardrail_decision_total`)
- **Export:** `observability/dashboards/task-latency.json`
- **Action Thresholds:**
  - p95 latency > 45s for 10 minutes → investigate agent bottlenecks.
  - Guardrail denies ≥1 per policy in 15 minutes → escalate to safety lead.

### 1.2 Reviewer Queue Summary
- **Dashboard:** `Review Operations`
- **Panels:** `Open tickets`, `In review`, `Unassigned`, `Resolution velocity`
- **Data Source:** Prometheus (`neuraforge_review_tickets{status}` gauges, `neuraforge_review_ticket_oldest_age_seconds`) + Loki annotations.
- **Export:** `observability/dashboards/review-queue.json`
- **Action Thresholds:**
  - Open tickets > 12 for 10 minutes → assign available reviewers, consider deferring new escalations.
  - Unassigned tickets ≥3 for 5 minutes → redistribute ownership immediately.
  - Oldest ticket age > 2 hours → page duty reviewer.

## 2. Alert Streams

| Alert | Condition | Source | Response |
| --- | --- | --- | --- |
| `TaskLatencyCritical` | p95 latency > 60s for 10m | Prometheus | Inspect agent logs; verify task queue backlog. |
| `GuardrailDenyCritical` | Guardrail deny decision ≥1 per policy over 15m | Prometheus | Loop in safety lead; review policy at fault. |
| `GuardrailEscalationSpike` | Escalations > 3 for same policy over 15m | Prometheus | Review policy definition; check reviewer console filters for context. |
| `ReviewQueueCritical` | Open tickets > 16 for 10m | Prometheus | Follow Section 3 playbook. |
| `ReviewUnassignedBacklog` | Unassigned tickets ≥3 for 5m | Prometheus | Assign to standby reviewer within 5m. |
| `ReviewOldestTicketStale` | Oldest ticket age > 2 hours | Prometheus | Page duty reviewer; unblock within 30m. |
| `rate_limit_exceeded` | >25 occurrences per identifier in 5m | Loki → Grafana Alert | Confirm legitimate surge vs. abuse; coordinate with infra if malicious. |

> Note: Alert definitions live in Grafana's unified alerting configuration. Templates are documented in `docs/observability/log_alerting_examples.md`.

## 3. Incident Playbooks

### 3.1 Open Queue Backlog
1. Verify the `Review Operations` dashboard to confirm backlog size and age.
2. Use the reviewer console filters (status/reviewer/age) to target >1h tickets.
3. Assign standby reviewers and communicate expected time-to-clear in #review-ops Slack.
4. If backlog persists beyond 30 minutes, notify product owner and consider raising guardrail thresholds temporarily (per safety policy).

### 3.2 Guardrail Escalation Spike
1. Inspect the `Guardrail decisions per policy` panel to identify culprit policy.
2. Cross-reference Loki logs (`rate_limit_exceeded` or `guardrail_triggered`) for context when reviewing decisions.
3. If policy misconfiguration is suspected, open a change request; do **not** disable guardrails without approval.
4. Document incident in Ops notebook and link to relevant task IDs.

### 3.3 Task Latency Regression
1. Check `Task p95 latency by agent-count` to determine whether heavy agent stacks (`agent_count=5_plus`) or all tasks are affected.
2. Inspect worker logs for slow tool invocations (`neuraforge_tool_latency_seconds` panels).
3. If queue length rising concurrently, trigger scaling for orchestrator workers per infrastructure playbook.
4. After remediation, record start/end time of the latency event in the ops log.

## 4. Communication Channels

- **Slack:** `#review-ops`, `#observability-alerts` (warning-level reviewer alerts are routed here via `SLACK_REVIEWERS_WEBHOOK_URL`).
- **PagerDuty:** `NeuraForge Reviewer Rotation`
- **Email:** `reviewops@neuraforge.example`

## 5. References

- `docs/observability/grafana_dashboards.md` – step-by-step instructions for updating dashboards.
- `docs/observability/log_alerting_examples.md` – Loki/Elastic alert pipelines.
- `scripts/staging_sync_artifacts.py` – pull the latest dashboards/k6 scripts from CI before staging validation.
- `docs/runbooks/task_queue_remediation.md` – procedures for clearing orchestration backlogs and manual dead-letter handling.
- Phase 7 Roadmap – observability tasks and milestones.
