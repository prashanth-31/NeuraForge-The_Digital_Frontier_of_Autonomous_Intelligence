# Backend ⇄ Frontend Integration Guide

_Date: 2025-10-16_

This guide outlines how the NeuraForge backend API connects to its data services (Redis, PostgreSQL, Qdrant), local LLM runtime (Ollama), authentication layer, observability stack, and MCP tool router, and how those integrations surface through the frontend reviewer console.

## 1. Environment Configuration

Populate `backend/.env` with connection details. Example values are already provided:

| Capability | Key Variables | Description |
| --- | --- | --- |
| Redis (task queue & working memory) | `REDIS__URL`, `REDIS__TASK_QUEUE_DB` | Points to the Redis instance used for queues and short-lived memory. |
| PostgreSQL (episodic storage) | `POSTGRES__DSN`, `POSTGRES__POOL_*` | Async DSN for review tickets, lifecycle logs, negotiation history. |
| Qdrant (semantic memory) | `QDRANT__URL`, `QDRANT__API_KEY` | Vector store for embeddings and contextual recall. |
| Ollama | `OLLAMA__HOST`, `OLLAMA__PORT`, `OLLAMA__MODEL` | HTTP endpoint and model identifier for local LLaMA serving. |
| Auth | `AUTH__JWT_SECRET_KEY`, `AUTH__ACCESS_TOKEN_EXPIRE_MINUTES` | JWT configuration for protected API routes. |
| Observability | `OBSERVABILITY__PROMETHEUS_ENABLED`, `OBSERVABILITY__LOG_LEVEL` | Toggles log level, Prometheus export, and Grafana bootstrap. |
| MCP Tool Router | `TOOLS__MCP__*` | Endpoint and signing information for optional MCP integrations. |
| Frontend URL | `BACKEND_BASE_URL` | Canonical base URL the frontend uses when constructing fetch calls. |

> **Tip:** When running Docker Compose, keep services on the same network. Frontend development (`npm run dev`) talks to the backend at `http://localhost:8000` by default. Update `VITE_API_BASE_URL` in `frontend/.env` if hosting elsewhere.

## 2. Backend Service Wiring

Within the backend codebase:

- `app/services/memory.py` loads Redis/PostgreSQL/Qdrant settings through Pydantic (`Settings.postgres`, `Settings.redis`, `Settings.qdrant`).
- `app/services/llm.py` (`OLLAMA__*`) instantiates the Ollama client for task orchestration and meta-agent summaries.
- `app/core/security.py` consumes `AUTH__*` variables to validate JWTs protecting reviewer APIs.
- `app/core/metrics.py` respects `OBSERVABILITY__*` toggles, emitting Prometheus metrics when enabled.
- `app/services/tools.py` activates MCP integrations if `TOOLS__MCP__ENABLED=true` and a reachable endpoint is provided.

## 3. Frontend Integration Points

- `frontend/src/lib/api.ts` reads `import.meta.env.VITE_API_BASE_URL` (defaults to `http://localhost:8000`) and attaches `Authorization` headers if a reviewer token is configured.
- Task streaming now relies on the expanded SSE surface exposed by `/api/v1/submit_task/stream`:
  - `frontend/src/contexts/TaskContext.tsx` listens for `agent_started`, `agent_completed`, `agent_failed`, `tool_invoked`, and `guardrail_triggered` events, persisting them in context for downstream components.
  - The session sidebar (`frontend/src/components/HistoryPanel.tsx`) renders these lifecycle updates inside a dedicated *Timeline* tab, alongside tool telemetry and MCP diagnostics.
  - After a stream completes, the context fetches `/api/v1/tasks/{task_id}` to populate run metrics, guardrail decisions, negotiation summaries, and orchestrator event history.
- The reviewer console (`frontend/src/pages/Reviews.tsx`) uses React Query to poll:
  - `/api/v1/reviews` and related mutations (`assign`, `unassign`, `notes`, `resolve`) for queue management.
  - `/api/v1/reviews/metrics` for aggregated workload analytics (open vs. in-review counts, unassigned tickets, resolution latency). These metrics are refreshed whenever reviewer actions succeed so the console stays in sync with backend totals.
- Any consumer that needs historical answers can call `/api/v1/history/{task_id}`; the same route feeds the History tab in the sidebar.

## 4. Observability Surface

- Prometheus scrapes the backend at `/metrics`; Grafana dashboards (Phase 6 assets) visualize reviewer queues, task throughput, agent latency, and guardrail outcomes. `/api/v1/tasks/{task_id}` surfaces the same counters (completed/failed agents, guardrail events, negotiation rounds) that power these dashboards.
- Frontend logs (via browser dev tools) can correlate SSE events with `neuraforge_*` metrics to diagnose orchestration issues. The new Timeline tab mirrors those metrics in real time, making it easier to jump from an agent failure toast to the corresponding Prometheus panel.

## 5. MCP & Tooling Flow

If `TOOLS__MCP__ENABLED=true`:
- Backend routes tool calls through the MCP router defined by `TOOLS__MCP__ENDPOINT`.
- Frontend exposes tool status to reviewers via toasts when the orchestration stream emits `mcp_status` or `tool_invoked` events, and the *Tools* tab in the sidebar lists the most recent invocations (latency, caching, composite routing, payload hints) captured from those SSE payloads.

## 6. Verification Checklist

Automation helper: run `python implementation/scripts/verify_stack.py --review-token <token>` once the stack is booted. The script exercises `/health`, reviewer APIs, `/submit_task/stream`, `/api/v1/tasks/{task_id}`, and the Prometheus/Grafana health endpoints, mirroring the manual steps below and printing a PASS/FAIL summary.

Manual walkthrough:

1. **Start services** via `docker compose up` (Redis, PostgreSQL, Qdrant, Prometheus, Grafana, backend).
2. **Launch Ollama** with the configured model (`ollama run llama3.1:8b`).
3. **Run backend** (`uvicorn app.main:app --reload`) ensuring `.env` is loaded.
4. **Start frontend** (`npm install && npm run dev`). Set `VITE_API_BASE_URL` if needed.
5. **Login** or configure reviewer token, then verify:
  - `/api/v1/reviews` populates the console.
  - Assign/unassign actions change state and emit notifications.
  - `/submit_task/stream` provides live updates when triggered from the task form.
6. **Check metrics** by visiting `http://localhost:9090` (Prometheus) and `http://localhost:3000` (Grafana) to confirm backend metrics are ingested.

With these settings in place, the backend API remains fully connected to Redis, Qdrant, PostgreSQL, Ollama, authentication, observability, and MCP tooling, while the frontend consumes its endpoints securely.
