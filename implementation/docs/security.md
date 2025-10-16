# Security & Rate Limiting Guide

This document captures the operational guidance for the Phase 7 security hardening work. It explains how JWT scopes are evaluated, how Redis-backed throttling is configured, and how to observe audit trails emitted by the backend.

## JWT Roles & Scopes

The backend now expands token roles into normalized scopes before enforcing access control. Tokens may provide either a comma-separated `roles` claim or an explicit `scopes`/`scope` claim. The default role-to-scope mapping is:

| Role           | Implied Scopes                                  |
|----------------|--------------------------------------------------|
| `reviewer`     | `reviews:read`, `reviews:write`, `reports:read`  |
| `review_admin` | all reviewer scopes plus `reviews:admin`         |
| `observer`     | `reviews:read`, `reports:read`                   |

FastAPI dependencies use these scopes to guard endpoints:

- Reviewer analytics (`/api/v1/reviews/metrics`) require `reviews:read`.
- Reviewer ticket mutations (`assign`, `notes`, `unassign`, `resolve`) require `reviews:write`.
- Decision dossier downloads now require `reports:read`.

`AUTH__SUPERUSER_EMAIL` or `AUTH__SERVICE_TOKEN` values continue to bypass scope checks for automation users.

## Redis Rate Limiting

All task submission endpoints (`/api/v1/submit_task`, `/api/v1/submit_task/stream`) and reviewer APIs are protected by a Redis-backed sliding window limiter. Limits are evaluated per-subject (JWT `sub`) and fall back to the caller IP when no token is present.

Configuration lives under the `rate_limit` settings block:

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `RATE_LIMIT__ENABLED` | `true` | Globally toggle API rate limiting. |
| `RATE_LIMIT__NAMESPACE` | `neuraforge:ratelimit` | Redis key prefix used for counters. |
| `RATE_LIMIT__TASK_SUBMISSION__CAPACITY` | `6` | Requests allowed per task submission window (roughly one task every 10 seconds). |
| `RATE_LIMIT__TASK_SUBMISSION__WINDOW_SECONDS` | `60` | Sliding window length for task submission. |
| `RATE_LIMIT__REVIEW_ACTION__CAPACITY` | `90` | Requests allowed per reviewer window (supports sustained 1.5 rps bursts). |
| `RATE_LIMIT__REVIEW_ACTION__WINDOW_SECONDS` | `60` | Sliding window length for reviewer operations. |

> Redis connection details reuse `REDIS__URL`. Ensure the service has network access to the same Redis instance provisioned for memory and queue workloads.

When limits are exceeded FastAPI returns `429 Too Many Requests` and includes a `Retry-After` header populated from the remaining TTL of the Redis bucket.

## Audit Logging Middleware

`AuditLoggingMiddleware` is registered globally and emits structured events for all API prefixes. Each log entry includes:

- HTTP method and route path
- Response status code and optional `Retry-After`
- JWT subject (or `anonymous`), token roles, and scopes
- SHA-256 hash of the request body (for mutating methods)
- Client IP address and request latency (milliseconds)

Logs are produced through `structlog` with the logger name `audit`. Forward these events to your centralized log sink to maintain an immutable trail of reviewer activity and rate-limit violations.

## Deployment Checklist

1. Set strong values for `AUTH__JWT_SECRET_KEY` and related auth env vars in production.
2. Point `REDIS__URL` at the shared Redis instance; confirm connectivity from the backend container.
3. Tune `RATE_LIMIT__*` thresholds to match expected reviewer throughput.
4. Update token issuers so that reviewers carry the correct `roles` claims.
5. Verify `/metrics` exposes `neuraforge_guardrail_decisions_total` and new audit events appear in structured logs during smoke tests.
