# Grafana Dashboard Updates – October 2025

The new Prometheus metrics added on 2025-10-17 enable richer visibility into orchestrator throughput and guardrail health. This guide shows how to wire the metrics into Grafana dashboards.

## 1. Task Latency Overview Panel

**Metric:** `neuraforge_task_latency_seconds`

1. **Query (PromQL)**
   ```promql
   histogram_quantile(
     0.95,
     sum by (le, agent_count) (
       rate(neuraforge_task_latency_seconds_bucket{entry_point="api"}[5m])
     )
   )
   ```
   - Add a second series filtered by `agent_count="5_plus"` to quickly spot complex orchestrations.
2. **Panel Type:** Time series.
3. **Thresholds:**
   - Warning at `45s` (yellow).
  - Critical at `60s` (red).
4. **Field Overrides:**
   - Display `agent_count` in the legend (`Override by regex agent_count=.*`).
5. **Annotations:**
   - Reference the `Review Queue Alerts` annotation stream (see runbook) to correlate spikes with reviewer backlog.

## 2. Guardrail Decision Ratio Panel

**Metric:** `neuraforge_guardrail_decision_total`

1. **Queries**
   - Escalations per policy:
     ```promql
     sum by (policy_id) (
       increase(neuraforge_guardrail_decision_total{decision="escalate"}[15m])
     )
     ```
   - Denies per policy:
     ```promql
     sum by (policy_id) (
       increase(neuraforge_guardrail_decision_total{decision="deny"}[15m])
     )
     ```
2. **Panel Type:** Bar gauge or stacked bar chart.
3. **Thresholds:**
   - Warning when escalations for a policy exceed 3 in 15 minutes.
   - Critical when denies exceed 1 per policy in 15 minutes.
4. **Links:**
   - Add panel links to the reviewer console filtered by policy (`/reviews?policy=<policy_id>` once supported).

## 3. Reviewer Workload Summary Update

Enhance the existing Review dashboard with a row of stat panels fed from API metrics (via Prometheus exporters or synthetic metrics):

- `sum(neuraforge_review_tickets{status="open"})`
- `sum(neuraforge_review_tickets{status="in_review"})`
- `max(neuraforge_review_tickets_open)`
- `sum(neuraforge_review_tickets{status="unassigned"})`
- `neuraforge_review_ticket_oldest_age_seconds / 60`

Provide matching alert rules in Alertmanager or Grafana:

- **Open queue high:** `sum(neuraforge_review_tickets{status="open"}) > 12` for 10m.
- **Unassigned backlog:** Leverage the dashboard variable `unassigned_open` (from API metrics or Loki log aggregations) and alert when `>= 3`.

## 4. Import & Version Control

1. Export the updated dashboard JSON and commit it under `observability/dashboards/` (create the directory if missing).
  - Current exports: `observability/dashboards/task-latency.json`, `observability/dashboards/review-queue.json`.
2. Document the dashboard version in `docs/runbooks/reviewer_operations.md` (see below) so on-call responders know which panels to reference.
3. CI uploads the dashboard JSON alongside the k6 load script (`phase5-observability` workflow → `compose-validate` job).
4. Staging hosts can pull the latest exported dashboards with `python scripts/staging_sync_artifacts.py` (requires `GITHUB_TOKEN`). The script also mirrors the JSON under `observability/grafana/dashboards/` so Docker Compose picks them up.
5. Alert routing separates reviewer warnings (`SLACK_REVIEWERS_WEBHOOK_URL`) from critical escalations (`PAGER_WEBHOOK_URL`). Update these environment variables before deploying Alertmanager.

## 5. Follow-up Actions

- Add `grafana/dashboards/task-latency.json` and `grafana/dashboards/review-queue.json` to the repo once the dashboard JSON exports are finalized.
- Wire alerts described above into the shared Alertmanager configuration (TODO for Phase 7 CI/CD milestone).
- Sync thresholds with the reviewer runbook so operational guidance stays aligned.
