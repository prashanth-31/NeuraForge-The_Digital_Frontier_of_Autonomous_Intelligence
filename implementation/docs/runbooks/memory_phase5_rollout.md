# Phase 5 Memory & Observability Rollout

_Date:_ 2025-11-13 (kickoff)

_Owner:_ Memory & Observability Working Group

## Objectives

1. Stand up tiered memory services (Redis, Postgres, Qdrant) with health monitoring.
2. Wire backend services to persist short-term, episodic, and semantic data flows.
3. Extend observability (Prometheus, Grafana, Loki) to include memory metrics and alerts.
4. Demonstrate >95% successful context recall during scripted scenarios.

## Workstreams

### A. Infrastructure Readiness
- [x] Validate docker-compose healthchecks for `postgres`, `redis`, `qdrant` (see `docs/reports/memory_phase5_infra.md`).
- [x] Provision persistent volumes in staging/production clusters (tracked in Terraform bucket `infra/memory`).
- [x] Configure backups & retention: Postgres (daily dump), Qdrant (snapshot), Redis (AOF weekly verify) via `scripts/memory_backup.ps1`.
- [x] Update security groups / firewall rules for memory services (documented in `infra/memory/security-groups.tf`).

### B. Backend Integration
- [x] Implement `ShortTermMemoryStore` backed by Redis (exposed via `RedisRepository` in `app/services/memory.py`).
- [x] Implement `EpisodicMemoryRepository` using Postgres (see `PostgresRepository` + migrations).
- [x] Implement `SemanticMemoryClient` for Qdrant (see `QdrantRepository` helpers).
- [x] Introduce unified `HybridMemoryService` orchestrating reads/writes across layers.
- [x] Add FastAPI dependencies to make memory services available to agents/planner (`get_hybrid_memory`).

### C. Observability Enhancements
- [x] Add Prometheus exporters / metrics for cache hits, store latency, and service health.
- [x] Provision Grafana dashboards (`observability/grafana/dashboards/memory.json`).
- [x] Configure Alertmanager rules for health degradation (`observability/rules/memory_rules.yml`).
- [x] Add Loki correlation queries (documented in Grafana dashboard annotations & runbook appendix).

### D. Testing & Validation
- [x] Create pytest fixtures/mocks for each memory tier (`tests/test_memory.py`).
- [x] Add integration tests covering write/read paths and failure recovery.
- [x] Develop scripted scenario harness (`scripts/run_memory_validation.py`) achieving >95% retrieval success.
- [x] Capture validation evidence once backing stores reachable (see `docs/reports/memory_validation_2025-11-13.txt`).
- [x] Automate host runner for validation harness (use `scripts/run_memory_validation_local.ps1`).
- [x] Document acceptance evidence (`docs/reports/memory_validation_2025-11-13.txt`, `docs/reports/memory_phase5_infra.md`, Grafana exports).

## Milestones

| Date | Milestone | Owner |
|------|-----------|-------|
| 2025-11-14 | Infra health check completed (Dev env) | Infra Ops |
| 2025-11-18 | Backend memory interfaces merged behind feature flag | Platform Eng |
| 2025-11-22 | Observability dashboards live in staging | Observability Guild |
| 2025-11-27 | Retrieval demo passes 95% threshold | Memory WG |

## Run Coordination

- Daily standup notes captured in `docs/daily-updates/`.
- Track Jira epics `MEM-401` (Memory Service) and `OBS-512` (Observability Expansion).
- Escalation path: DRI (Platform Eng Lead) → Director of Engineering.

## Operationalization

- Nightly snapshot automation runs via Jenkins job `MEMORY-SNAPSHOT-NIGHTLY` (see repository `Jenkinsfile`); artifacts retained for 30 days with weekly promotions capped at 12.
- Memory alerting routes to Slack `#ops-memory` through Alertmanager receiver `ops-memory` backed by `pager-proxy` webhook (`observability/alertmanager/config.yml`).
- Seven-day staging soak plan documented in `docs/reports/memory_phase5_staging_soak.md`; soak telemetry collected from Grafana dashboard `memory-phase5`.
- Backup execution evidence: latest dry-run `scripts/memory_backup.ps1 -SkipQdrant` (Postgres/Redis successful) archived under `backups/memory`.

## References

- Latest validation run (local dev, 2025-11-13): success rate 1.0, coverage 1.0 using remapped host ports (`Redis 16379`, `Postgres 15432`, `Qdrant 16333`). Run via `scripts/run_memory_validation_local.ps1` or raw CLI overrides.
- Infrastructure summary: `docs/reports/memory_phase5_infra.md`.
- Grafana dashboard: `observability/grafana/dashboards/memory.json` (import UID `memory-phase5`).
- Alerting rules: `observability/rules/memory_rules.yml`.
- Backup automation: `implementation/scripts/memory_backup.ps1`.
- Staging soak log: `docs/reports/memory_phase5_staging_soak.md`.
- `implementation/backend/docker-compose.yml` (service definitions)
- `implementation/backend/app/core/config.py` (Redis/Postgres settings)
- `implementation/backend/app/services/memory.py` (hybrid memory service)
- `implementation/backend/scripts/run_memory_validation.py` (validation harness)
- `implementation/observability/` (Prometheus/Loki configs)
- Sample Loki query: `{compose_service="backend"} |= "memory_operation_failed"`
