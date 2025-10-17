# Phase 7 Roadmap – FastAPI Integration & Observability Hardening

_Date: 2025-10-16_

## Vision
Phase 7 focuses on production-hardening the NeuraForge service surface. We will extend FastAPI coverage, deepen observability hooks, introduce role-aware security, and ensure deployment assets can support sustained operations. Success means stakeholders can instrument, secure, and scale the platform with confidence.

## Objectives
1. **Expand API Surfaces & Streaming UX**
   - Provide explicit REST endpoints for task status, reviewer analytics, and orchestration introspection.
   - Offer SSE/WebSocket improvements for traceability (agent lifecycle, tool invocations, guardrail decisions).
2. **Security & Role Management**
   - Enforce JWT auth on all sensitive routes with role/permission checks.
   - Introduce rate limiting and audit logging around reviewer operations and task submission.
3. **Observability Hardening**
   - Evolve Prometheus metrics into SLO-backed dashboards and tighten alert rules.
   - Wire Grafana panels for reviewer workload, negotiation health, and agent tool usage.
4. **Frontend ↔ Backend Integration & Operational Runbooks**
   - Expand GitHub Actions to lint/type-check frontend, execute contract tests, and validate Docker Compose.
   - Produce runbooks for alert triage, queue remediation, and long-running task investigation.
   - Deliver end-to-end integration between frontend review console and backend APIs (token management, streaming telemetry, health checks).

## Work Breakdown

### 1. API Surface Expansion
- [ ] Add `/api/v1/tasks/{task_id}` for real-time status (queue, in-progress, completed, failed).
- [ ] Implement `/api/v1/reviews/metrics` returning aggregated reviewer workload, aging tickets, assignment velocity.
- [ ] Expose `/api/v1/orchestrator/runs/{run_id}` to fetch negotiation snapshots and guardrail decisions.
- [ ] Upgrade SSE stream to broadcast structured telemetry (`agent_started`, `guardrail_triggered`, `tool_invoked`).

### 2. Security & Rate Limiting
- [x] Integrate FastAPI dependencies for role-based access; map "reviewer", "review_admin", "observer" scopes.
- [x] Require JWT auth for all review endpoints and task dossier downloads.
- [x] Apply rate limiting (Redis-backed) for task submission and reviewer actions.
- [x] Implement audit logging middleware capturing requester identity, route, payload hash, and response status.
- [x] Document environment variables and setup steps in `docs/security.md`.

### 3. Observability Enhancements
- [x] Add Prometheus histogram buckets for task latency broken down by agent involvement.
- [x] Track guardrail decision counts per policy and emit `neuraforge_guardrail_decision_total` metrics.
- [x] Extend reviewer dashboard with panel filters (status, reviewer, age) and alert annotations.
- [x] Introduce log-based alerting examples (e.g., structlog output piped to Loki or local Elastic stack).
- [x] Provide k6 or Locust scripts for load testing task submission, capturing metrics in Grafana.
- [x] Emit reviewer backlog metrics for unassigned counts and oldest ticket age.
- [x] Add `/metrics` regression tests to guard task latency, guardrail, and reviewer series serialization.
- [x] Wire Alertmanager routing so reviewer warnings go to Slack and critical incidents page on-call responders.

## Progress Notes

- **2025-10-16**: Security & rate limiting work is complete, including rate-limit Redis fallback to keep tests green offline. Next focus area is Observability Enhancements—start with Prometheus latency histograms and guardrail decision metrics, then extend the reviewer dashboard filters.

### 4. CI/CD Pipeline Upgrades
- [x] Expand `.github/workflows/phase5-observability.yml` to run `npm run lint`, `npx tsc --noEmit`, `npm run build`.
- [x] Publish Grafana dashboards and k6 load script as workflow artifacts.
- [x] Include contract tests covering `/submit_task/stream`, `/reviews/metrics`, and auth rejection cases.
- [x] Use Docker Compose in CI to sanity check Prometheus/Grafana provisioning and alert rule syntax.
- [x] Publish Prometheus/Grafana configs as workflow artifacts for staging environments.
- [x] Add end-to-end smoke test that boots the frontend against the backend API, ensuring reviewer console loads tickets and displays SSE updates.

### 5. Operational Runbooks & Docs
- [x] Draft `docs/runbooks/reviewer_operations.md` (alert meaning, dashboards, escalation paths).
- [x] Document task queue remediation, including retry policies and dead-letter handling.
- [x] Provide upgrade guidance for settings (environment variable matrix, secrets management).
- [x] Document staging k6 smoke procedure so operators can rehearse artifact rollout.

## Milestones & Timeline (estimates)
1. **Week 1**: API endpoint expansion + SSE instrumentation.
2. **Week 2**: Security & rate limiting; start runbook drafts.
3. **Week 3**: Observability dashboards/alerts; load test scripts.
4. **Week 4**: CI/CD workflow upgrades, compose validation, doc polish.

## Definition of Done
- All new endpoints documented in `docs/api.md` with request/response examples.
- JWT-secured routes enforce role checks; integration tests cover accept/deny paths.
- Prometheus scrapes new metrics without errors; Grafana dashboards imported successfully.
- CI pipeline runs backend + frontend lint/test/build workflows.
- Runbooks exist for reviewer alerts, queue issues, and task failures.

## Risks & Mitigations
- **Metric Cardinality**: Ensure new labels don’t explode Prometheus storage. Mitigate with aggregation and temporary metrics review process.
- **JWT/Token Drift**: Stage environment should mirror production tokens; use feature flags and config toggles for rollout.
- **Alert Fatigue**: Tune thresholds, add quiet hours, and document suppression/acknowledgment procedures.

## Acceptance
Phase 7 is complete when operations can monitor reviewer health, enforce access control, and deploy the stack via CI with predictable outcomes.
