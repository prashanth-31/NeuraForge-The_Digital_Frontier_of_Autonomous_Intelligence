# Rate Limit Operations Runbook

_Date:_ 2025-11-14

## Objective

Maintain healthy request throughput and respond to throttling incidents introduced by Phase 6 regression guardrails.

## Monitoring

- Alert: `RateLimitThrottleSpike` in Alertmanager (routes to `#ops-memory`). Trigger when `neuraforge_rate_limited_requests_total` increases > 300 per 5 minutes.
- Dashboard: Grafana panel `API Rate Compliance` (import UID `memory-phase5`). Tracks permit consumption vs configured capacity.
- Logs: Search for `rate_limit_exceeded` in Loki when debugging client errors.

## Runbook Steps

1. Confirm Redis availability for rate limit counters (`redis-cli --scan --pattern 'neuraforge:ratelimit:*' | head`).
2. Inspect rule configuration:
   ```python
   from app.core.config import get_settings
   settings = get_settings()
   print(settings.rate_limit.task_submission.capacity)
   ```
3. Validate enforcement using automated suite:
   ```bash
   python -m pytest tests/test_security_rate_limit.py
   python -m pytest tests/e2e/test_regression_end_to_end.py::test_multiple_tasks_do_not_clobber_state
   ```
4. If clients report 429s:
   - Increase `RATE_LIMIT__TASK_SUBMISSION__CAPACITY` gradually (increments of +5) and redeploy.
   - Communicate temporary limits to consumers; capture timeline in incident notes.

## Backoff Strategy

- Encourage clients to implement exponential backoff starting at 500 ms with jitter.
- Provide staged quota increases for trusted services via service token scopes (`reviews:write`, etc.).
- For emergency overrides, disable rate limits by setting `RATE_LIMIT__ENABLED=false` and redeploying (limit to <30 minutes, document thoroughly).

## Post-Incident Tasks

- Restore baseline configuration and confirm Alertmanager silence removed.
- Update `docs/reports/system_risk_assessment.md` rate limit section with summary of impact and remediation.
- File change request for permanent capacity adjustments if usage trend continues for three consecutive days.
