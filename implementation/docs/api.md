# NeuraForge API Reference

_Last updated: 2025-10-19_

This document captures the Phase 7 additions to the NeuraForge FastAPI surface. Existing routes continue to behave as documented in earlier phases; the sections below focus on the expanded task telemetry, reviewer analytics, and orchestration introspection endpoints, along with the streaming schema emitted by `/api/v1/submit_task/stream`.

> **Auth & Scopes**
> - All REST examples assume a valid bearer token. Supply the token via `Authorization: Bearer <JWT>`.
> - Reviewer analytics endpoints (metrics, ticket CRUD) require the `reviews:read` or `reviews:write` scope as noted below.
>
> **Conventions**
> - Unless otherwise stated, responses are JSON encoded in UTF-8.
> - Timestamps use ISO 8601 with UTC (`YYYY-MM-DDTHH:MM:SSZ`).

## Task Status – `GET /api/v1/tasks/{task_id}`

Provides a real-time snapshot of a task, including latest outputs, orchestration metrics, and the bounded event transcript.

```http
GET /api/v1/tasks/6f0aa7f3-02b8-4d29-8325-0f4224af9bf1 HTTP/1.1
Authorization: Bearer <JWT>
```

```json
{
  "task_id": "6f0aa7f3-02b8-4d29-8325-0f4224af9bf1",
  "status": "completed",
  "run_id": "f5d7f5a6-8f45-4c80-a215-3da1e4af0016",
  "prompt": "Summarise the latest task latency trends",
  "metadata": {
    "priority": "normal",
    "submitted_by": "review-ops"
  },
  "outputs": [
    {
      "agent": "research",
      "summary": "Latency has normalised after cache warm-up.",
      "confidence": 0.82,
      "metadata": {
        "sources": ["grafana:task-latency"]
      }
    }
  ],
  "plan": {"status": "planned", "steps": 3},
  "negotiation": {"status": "completed", "rounds": 2},
  "guardrails": {
    "decisions": [
      {
        "policy": "llm-toxicity",
        "outcome": "allow",
        "timestamp": "2025-10-19T14:03:22Z"
      }
    ]
  },
  "metrics": {
    "agents_completed": 4,
    "agents_failed": 0,
    "guardrail_events": 1,
    "negotiation_rounds": 2
  },
  "last_error": null,
  "created_at": "2025-10-19T14:02:11Z",
  "updated_at": "2025-10-19T14:03:28Z",
  "events": [
    {
      "sequence": 1,
      "event_type": "task_started",
      "agent": null,
      "payload": {"prompt": "Summarise the latest task latency trends"},
      "created_at": "2025-10-19T14:02:11Z"
    },
    {
      "sequence": 12,
      "event_type": "agent_completed",
      "agent": "research",
      "payload": {
        "summary": "Latency has normalised after cache warm-up.",
        "confidence": 0.82
      },
      "created_at": "2025-10-19T14:03:24Z"
    }
  ]
}
```

**Notes**
- The response aggregates state from ephemeral memory and the orchestration state store. Events are bounded to the latest 250 entries.
- `metrics.negotiation_rounds` is populated when orchestration emits negotiation telemetry; if not available it defaults to `null`.
- A `404` indicates the task has never been observed or the ephemeral record has expired.

## Reviewer Metrics – `GET /api/v1/reviews/metrics`

Returns backlog, aging, and throughput insights for reviewer tickets. Requires the `reviews:read` scope and rate-limit token for reviewer actions.

```http
GET /api/v1/reviews/metrics HTTP/1.1
Authorization: Bearer <JWT with reviews:read>
```

```json
{
  "generated_at": "2025-10-19T14:15:03Z",
  "totals": {
    "open": 5,
    "in_review": 4,
    "resolved": 28,
    "dismissed": 3
  },
  "assignment": {
    "by_reviewer": {
      "reviewer-a": 2,
      "reviewer-b": 1,
      "reviewer-c": 3
    },
    "unassigned_open": 2
  },
  "aging": {
    "open_average_minutes": 136.5,
    "open_oldest_minutes": 410.0,
    "in_review_average_minutes": 92.3
  },
  "resolution": {
    "average_minutes": 57.2,
    "median_minutes": 45.0,
    "completed_last_24h": 9
  },
  "queue_health": {
    "backlog_pressure": 3.0,
    "sla_breaches": 1,
    "escalations_pending": 2
  },
  "trends": {
    "resolved_last_7d": 38,
    "dismissed_last_7d": 4,
    "median_resolution_minutes_7d": 42.0
  },
  "velocity": {
    "per_reviewer_last_7d": {
      "reviewer-a": 12,
      "reviewer-b": 9,
      "reviewer-c": 7
    },
    "median_resolution_minutes_last_7d": {
      "reviewer-a": 39.5,
      "reviewer-b": 44.0,
      "reviewer-c": 47.2
    },
    "average_daily_closed_last_7d": 5.43,
    "active_reviewers_last_7d": 3
  }
}
```

**Notes**
- `velocity` summarises the last 7 days of resolved or dismissed tickets, grouped by assignee. Values default to `0` when there is no historical data.
- `queue_health.backlog_pressure` scales backlog counts by active reviewers to highlight staffing pressure.
- The endpoint is cached in-memory; expect sub-second latency under normal load.

## Orchestrator Run Detail – `GET /api/v1/orchestrator/runs/{run_id}`

Fetches the persisted record for a specific orchestrator run alongside structured telemetry derived from its event log.

```http
GET /api/v1/orchestrator/runs/f5d7f5a6-8f45-4c80-a215-3da1e4af0016 HTTP/1.1
Authorization: Bearer <JWT>
```

```json
{
  "run_id": "f5d7f5a6-8f45-4c80-a215-3da1e4af0016",
  "task_id": "6f0aa7f3-02b8-4d29-8325-0f4224af9bf1",
  "status": "completed",
  "state": {
    "plan": {"status": "planned"},
    "outputs": [
      {
        "agent": "research",
        "summary": "Latency has normalised after cache warm-up.",
        "confidence": 0.82
      }
    ]
  },
  "created_at": "2025-10-19T14:02:11Z",
  "updated_at": "2025-10-19T14:03:28Z",
  "events": [
    {
      "sequence": 1,
      "event_type": "task_started",
      "agent": null,
      "payload": {"prompt": "Summarise the latest task latency trends"},
      "created_at": "2025-10-19T14:02:11Z"
    },
    {
      "sequence": 12,
      "event_type": "agent_completed",
      "agent": "research",
      "payload": {
        "summary": "Latency has normalised after cache warm-up.",
        "confidence": 0.82
      },
      "created_at": "2025-10-19T14:03:24Z"
    }
  ],
  "telemetry": {
    "agents_started": 4,
    "agents_completed": 4,
    "agents_failed": 0,
    "guardrail_events": 1,
    "tool_invocations": 2,
    "negotiation_events": 3,
    "last_event_at": "2025-10-19T14:03:28Z",
    "total_events": 18
  }
}
```

**Notes**
- Returns `503` if the orchestration state store is disabled, `404` if the run ID cannot be located.
- `telemetry.tool_invocations` counts both `tool_invoked` and legacy `tool_invocation` events.

## Streaming Task Submission – `POST /api/v1/submit_task/stream`

The task streaming endpoint now emits a versioned envelope for each SSE message. Clients should parse the top-level metadata and inspect the nested `payload` for structured data.

```http
POST /api/v1/submit_task/stream HTTP/1.1
Authorization: Bearer <JWT>
Content-Type: application/json

{"prompt": "Summarise the latest task latency trends"}
```

Example data channel output:

```
event: task_started
data: {
  "version": 1,
  "schema": "neuraforge.task-event.v1",
  "event": "task_started",
  "type": "task_started",
  "sequence": 1,
  "task_id": "6f0aa7f3-02b8-4d29-8325-0f4224af9bf1",
  "timestamp": "2025-10-19T14:02:11Z",
  "payload": {
    "prompt": "Summarise the latest task latency trends"
  }
}

```

Subsequent events (`agent_started`, `agent_completed`, `tool_invoked`, `mcp_status`, `task_failed`, `task_completed`, etc.) share the same envelope:

- `version`: schema version (currently `1`).
- `schema`: identifier for downstream routers (`neuraforge.task-event.v1`).
- `event`/`type`: repeated event name for backwards compatibility with legacy consumers.
- `sequence`: monotonically increasing per-stream sequence number.
- `task_id`: identifier to correlate across channels.
- `run_id`: present once known, enabling correlation with the run detail endpoint.
- `timestamp`: ISO timestamp; generated server-side if omitted by the orchestrator.
- `payload`: event-specific content (agent telemetry, tool diagnostics, error messages, etc.).

**Client Guidance**
- Treat unknown keys inside `payload` as forward-compatible extensions.
- A stream ends with the sentinel `task_completed` or `task_failed`, followed by the connection closing. Always read until the socket closes to receive the run ID.

## Error Codes & Rate Limiting

| Endpoint | Success | Auth Error | Not Found | Rate Limited |
| --- | --- | --- | --- | --- |
| `GET /api/v1/tasks/{task_id}` | `200` | `401` if JWT missing/invalid | `404` | n/a |
| `GET /api/v1/reviews/metrics` | `200` | `401` | `503` if review manager disabled | `429` when reviewer action quota exceeded |
| `GET /api/v1/orchestrator/runs/{run_id}` | `200` | `401` | `404` | `503` if state store offline |
| `POST /api/v1/submit_task/stream` | `200` streaming response | `401` | `503` when orchestration unavailable | `429` on task submission rate limits |

## Changelog

- **2025-10-19**: Documented Phase 7 endpoints and SSE schema revision. Added reviewer velocity metrics, orchestrator telemetry, and task status detail.
