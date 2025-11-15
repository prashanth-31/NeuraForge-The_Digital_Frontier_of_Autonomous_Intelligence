# Phase 5 Infrastructure Validation (2025-11-13)

## Health Checks

- `docker compose ps` reports `healthy` for `redis`, `postgres`, and `qdrant` using host ports `16379`, `15432`, and `16333` respectively.
- Prometheus target `backend:8000/metrics` now emits `neuraforge_memory_service_health` and `neuraforge_memory_store_latency_seconds`.

## Persistence Guarantees

- Named volumes in `implementation/docker-compose.yml` (`redis_data`, `postgres_data`, `qdrant_data`) verified via `docker volume inspect`.
- Terraform staging manifests under `infra/memory` map these volumes to `gp3` storage classes with weekly retention.

## Backup & Retention

- Script `implementation/scripts/memory_backup.ps1` captures:
  - Postgres episodic memory (`pg_dump`).
  - Redis working memory (`redis-cli --rdb` + `docker compose cp`).
  - Qdrant semantic snapshot (disable with `-SkipQdrant`).
- Example run: `implementation/scripts/memory_backup.ps1`
- Retention policy: keep daily dumps 30 days, weekly snapshots 12 weeks.

## Network Controls

- Memory services bind to remapped localhost ports; firewall rules restrict ingress to orchestrator and observability subnets.
- Staging/production security groups tracked in `infra/memory/security-groups.tf` (CIDR allow-lists).

## Follow-up Actions

- Jenkins job `MEMORY-SNAPSHOT-NIGHTLY` codified in repository `Jenkinsfile` (nightly 02:00 UTC, 30-day retention, weekly promotions).
- Alertmanager now routes memory alerts (`team="ops-memory"`) to Slack `#ops-memory` via `pager-proxy` webhook.
- Next milestone: execute seven-day staging soak per `docs/reports/memory_phase5_staging_soak.md` and capture restore drill evidence.
