# Memory Phase 5 Staging Soak Plan

_Date:_ 2025-11-13

## Objective

Validate hybrid memory stability in staging across a continuous seven-day window by tracking service health, latency SLOs, cache efficiency, and alert volume. A successful soak yields zero critical alerts, <1% data loss, and p95 latency <500 ms sustained.

## Preconditions

- Jenkins job `MEMORY-SNAPSHOT-NIGHTLY` green for ≥2 consecutive runs (verifies backup cadence).
- Memory dashboard `memory-phase5` imported and wired to staging Prometheus datasource.
- Alertmanager route `team="ops-memory"` delivering to Slack `#ops-memory`.
- Staging agents upgraded to commit containing Phase 5 memory instrumentation.

## Monitoring Checklist

| Metric / Signal | Target | Grafana Panel | Notes |
|-----------------|--------|---------------|-------|
| `neuraforge_memory_service_health` | = 1.0 across stores | Memory Health Gauge | Investigate dips immediately. |
| `neuraforge_memory_store_latency_seconds` p95 | < 0.5s | Store Latency (p95) | Check per-store breakout. |
| Cache hit ratio | > 75% | Cache Efficiency | Use cache miss spike alert for early warning. |
| Alert volume | < 3 non-test alerts/day | Alert Feed | Capture links to Slack threads. |
| Backup artifacts | Daily & weekly present | Jenkins artifacts | Confirm restore snapshot integrity weekly. |

## Daily Logging Template

Record entries in this file each morning (UTC) following dashboard review.

```
### Day N (YYYY-MM-DD)
- Health: ✅ | ⚠️ | ❌ (explain deviations)
- Latency p95: <value>
- Cache Hit Ratio: <value>
- Alerts triggered: <list + links>
- Backup check: ✅ | ⚠️ | ❌
- Notable incidents / remediation:
```

## Incident Response

- Critical alerts automatically notify `#ops-memory`. Acknowledge within 10 minutes.
- Create Jira issue under epic `MEM-401` for any sustained degradation (>30m or repeated daily).
- Leverage `scripts/debug_routing.py` when traffic anomalies correlate with memory misses.

## Exit Criteria

- Seven consecutive daily entries with no critical incidents.
- Cache hit ratio average ≥80% and p95 latency ≤450 ms for Redis/Postgres, ≤600 ms for Qdrant.
- Successful restoration drill (use latest nightly snapshot) documented in `docs/reports/memory_restore_dryrun_<date>.md`.
- Final summary appended to this file and shared in weekly ops sync.
