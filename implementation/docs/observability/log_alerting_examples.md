# Log-Based Alerting Examples

This guide shows two approaches for turning NeuraForge's structured logs into actionable alerts. Both examples assume the FastAPI backend is emitting JSON via `structlog` (already enabled by `AuditLoggingMiddleware`).

## 1. Loki + Promtail + Grafana

1. **Ship logs with Promtail**
   - Install Promtail alongside the backend host.
   - Configure the scrape job to watch the backend log file (or stdout).
   - Example snippet:
     ```yaml
     scrape_configs:
       - job_name: neuraforge-backend
         static_configs:
           - targets: ["localhost"]
             labels:
               job: neuraforge
               __path__: /var/log/neuraforge/*.log
     ```
2. **Label enrichment**
   - Promtail automatically forwards the JSON payload; Loki stores labels.
   - Add pipeline stages to promote key fields (e.g., `route`, `identity`, `status_code`).
3. **Alert rule in Grafana**
   - Query example (count 5xx over 5 minutes):
     ```logql
     sum(rate({job="neuraforge"} | json | status_code >= 500 [5m]))
     ```
   - Create a Grafana alert: trigger when value > 0 for two consecutive evaluations.
4. **Notification channel**
   - Use Grafana's contact points (Slack, PagerDuty, email) to route alerts.

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

## Linking Alerts to Observability Dashboards
- Grafana panels can overlay alert annotations by referencing the same Loki/Prometheus queries.
- Kibana's dashboard annotation feature can mark alert boundaries, helping reviewers correlate spikes in the reviewer console filters with backend guardrail decisions.

## Next Steps
- Check `implementation/scripts/loadtesting/k6-submit-task.js` for synthetic load generation to validate alert thresholds.
- Update runbooks (`docs/runbooks/reviewer_operations.md`) with the alert playbooks once finalized.
