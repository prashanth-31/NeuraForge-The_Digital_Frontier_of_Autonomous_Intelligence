# Log-Based Alerting Examples

This guide shows two approaches for turning NeuraForge's structured logs into actionable alerts. Both examples assume the FastAPI backend is emitting JSON via `structlog` (already enabled by `AuditLoggingMiddleware`).

## 1. Loki + Promtail + Grafana

1. **Bring the logging stack online**
  - The repository now ships Loki and Promtail in `docker-compose.yml`.
  - From `implementation/` run `docker compose up -d loki promtail grafana` (Prometheus and Alertmanager start automatically as dependencies).
  - Promtail follows every container via the bundled `observability/promtail-config.yml`, promoting Docker labels such as `compose_service`, `container`, and `level`.
2. **Explore the logs**
  - Open Grafana at <http://localhost:3000> (credentials: `admin`/`admin`).
  - Navigate to *Explore → Loki* and execute `{compose_service="backend"}` to live-stream backend entries.
  - A pre-provisioned dashboard (`Backend Loki Logs`) under the **Phase 5** folder captures the same query, plus panels for error counts and guardrail warnings. Select another service with the dashboard variable if needed.
3. **Create an alert**
  - Example LogQL for 5xx bursts:  
    ```logql
    sum(count_over_time({compose_service="$service"} | json | status >= 500 [5m]))
    ```
  - Convert the panel to a Grafana alert (threshold > 0 for 2 evaluations) or persist it as a Loki ruler rule. Ruler storage is already wired to Alertmanager (`observability/loki/config.yml`).
4. **Reuse existing contact points**
  - Grafana routes alerts through the same Alertmanager contact points as the Prometheus stack. Choose a contact point (Slack, email, PagerDuty) when creating the rule and add runbook links for reviewer hand-off.

### Why it matters
- Audit logs land in Loki with hashed payloads and request metadata.
- Alerting on 401/403 spikes or rate-limit warnings (`rate_limit_exceeded`) becomes trivial.

## 2. Elastic Stack (Filebeat + Logstash + Kibana)

1. **Filebeat input**
   - Add a Filebeat input to tail the backend log file.
   - Enable the JSON decoder:
     ```yaml
     filebeat.inputs:
       - type: filestream
         id: neuraforge-backend
         paths:
           - /var/log/neuraforge/backend.json
         parsers:
           - ndjson:
               overwrite_keys: true
               message_key: message
     ```
2. **Logstash pipeline (optional)**
   - Add Grok/Mutate stages to normalize fields (e.g., rename `request_path`).
   - Output to Elasticsearch with index lifecycle policies.
3. **Kibana detection rule**
   - Build a query on the `structlog_event` field (e.g., `rate_limit_exceeded`).
   - Create a detection rule that triggers when more than 25 occurrences happen in 5 minutes grouped by `identifier`.
4. **Action connectors**
   - Configure Slack/email connectors so the alert delivers context (identity, retry-after header).

### Optional Enhancements
- Export alert payloads to PagerDuty for on-call rotations.
- Combine the histogram metric `neuraforge_task_latency_seconds` with log alerts (e.g., when latency p95 > 60s **and** guardrail escalations occur within the same window).
- Track the counter `neuraforge_finance_quote_fallback_total{provider="yfinance"}` to catch Yahoo Finance throttling or credential issues and wire it into the new `FinanceQuoteFallbackSpike` alert.

## Linking Alerts to Observability Dashboards
- Grafana panels can overlay alert annotations by referencing the same Loki/Prometheus queries.
- Kibana's dashboard annotation feature can mark alert boundaries, helping reviewers correlate spikes in the reviewer console filters with backend guardrail decisions.

## Next Steps
- Check `implementation/scripts/loadtesting/k6-submit-task.js` for synthetic load generation to validate alert thresholds.
- Update runbooks (`docs/runbooks/reviewer_operations.md`) with alert playbooks once finalized.
- If Loki retains old container timestamps, clear `implementation/observability/loki/data/` (after stopping the stack) so that Promtail and the ruler start fresh.
