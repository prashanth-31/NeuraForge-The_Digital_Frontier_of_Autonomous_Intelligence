# NeuraForge Backend

Foundational FastAPI + LangGraph service scaffold with hybrid memory, task queue, and observability hooks.

## Features

- **Config management** via Pydantic Settings with `.env` support.
- **FastAPI** REST interface with `/api/v1` namespace and health checks.
- **Task queue** abstraction (Redis-backed when available, in-memory fallback).
- **Hybrid memory** service stubs for Redis, PostgreSQL, and Qdrant.
- **Agent scaffolding** for research, finance, creative, and enterprise domains.
- **Benchmark harness** for agent evaluation metrics.
- **Security baseline** with JWT helpers.
- **Docker Compose** stack wiring all infrastructure dependencies locally.

## Getting Started

### 1. Install dependencies

```powershell
poetry install
```

Or with raw `pip` inside `.venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### 2. Copy environment template

```powershell
Copy-Item .env.example .env
```

Adjust the secrets and connection strings as needed for your local setup.

### 3. Run services locally

```powershell
poetry run alembic upgrade head
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Optional: bring up the full stack with Docker Compose from the repository root.

```powershell
cd ..
docker compose up --build
```

`docker compose` automatically runs migrations on container startup; rerun `poetry run alembic upgrade head` locally whenever models change.

### 4. Run tests & quality gates

```powershell
poetry run pytest
poetry run ruff check app
```

Coverage (optional):

```powershell
poetry run coverage run -m pytest
poetry run coverage report
```

## Architecture Notes

- `app/core` contains configuration, logging, and security utilities.
- `app/services` encapsulates memory providers with graceful degradation when backing services are offline.
- `app/queue` hosts the asynchronous task queue manager, which automatically uses Redis when configured.
- `app/orchestration` is prepared for LangGraph-based agent routing.
- `app/monitoring` includes a benchmarking helper for evaluating agent outputs.
- Tests cover API health and benchmark summarization as a baseline; expand with integration tests as features grow.

## Phase 1 Checklist

- [x] Poetry-based project scaffolding with pinned dependencies.
- [x] FastAPI service stub (`app/main.py`) and health endpoint.
- [x] Structured logging configured via `structlog`.
- [x] Task queue + hybrid memory abstractions with graceful fallbacks.
- [x] Docker Compose stack for Redis, PostgreSQL, Qdrant, Ollama, Prometheus, Grafana.
- [x] Baseline tests (`tests/test_health.py`) and lint command (`ruff`).

See `docs/architecture.md` for the full multi-phase roadmap.
