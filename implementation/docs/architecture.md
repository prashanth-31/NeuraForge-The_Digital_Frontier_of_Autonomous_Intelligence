# NeuraForge Architecture Overview

## 1. Vision & Scope
NeuraForge is a local-first, multi-agent intelligence platform designed to orchestrate specialized agents (Research, Finance, Creative, Enterprise) over a layered reasoning stack. The solution intentionally avoids any public cloud dependencies; all services run on developer-controlled infrastructure via Docker Compose or native processes.

### Core Goals
- **Agent Collaboration**: Coordinate heterogeneous agents through LangGraph workflows and negotiation logic.
- **Contextual Intelligence**: Blend short-term, episodic, and semantic memory for informed decision making.
- **Local LLM Serving**: Leverage Ollama-hosted LLaMA 3 for privacy-preserving inference.
- **Observability & Control**: Provide traceability, benchmarking, and structured logging without external telemetry services.

## 2. Layered System Architecture

```mermaid
%%{init: {"theme": "forest", "flowchart": {"curve": "basis"}} }%%
graph TD
    A[Frontend (Next.js 14 + Socket.io)] -->|Tasks / Streams| B[FastAPI Backend]
    B -->|Task Submission| C[Task Queue Manager]
    C -->|Jobs| D[LangGraph Orchestrator]
    D -->|Context Ops| E[Hybrid Memory Service]
    D -->|LLM Calls| F[LangChain + Ollama]
    D -->|Agent Requests| G[Specialized Agents]
    G -->|Insights| D
    E -->|Vector Search| H[Qdrant]
    E -->|Episodic Storage| I[PostgreSQL]
    E -->|Working Memory| J[Redis]
    F -->|Generations| D
    D -->|Resolution| K[Conflict Resolver + Meta-Agent]
    B -->|Metrics| L[Prometheus]
    L --> M[Grafana]
```

### Layer Breakdown
1. **Frontend Experience** (existing in `frontend/`): Next.js 14 + React, Tailwind UI, Socket.io client for streaming updates.
2. **API Gateway** (`backend/app/main.py`): FastAPI serves REST + WebSocket endpoints, performs auth and schema validation.
3. **Task Coordination** (`backend/app/queue/`): Async in-process queue with optional Redis backend to buffer long-running jobs.
4. **LangGraph Orchestration** (`backend/app/orchestration/`): Task routing, capability-based dispatch, and negotiation between agents.
5. **Agent Layer** (`backend/app/agents/`): Domain-specific logic for Research, Finance, Creative, Enterprise agents (extendable via LangChain tools).
6. **Hybrid Memory** (`backend/app/services/memory.py`): Unified interface to Redis (working memory), PostgreSQL (episodic logs), Qdrant (semantic vectors).
7. **LLM Layer** (`backend/app/utils/embeddings.py`, LangChain integrations): Ollama-hosted LLaMA 3 for completions; Sentence Transformers for embedding generation.
8. **Conflict Resolution** (`backend/app/orchestration/` roadmap): Confidence scoring, cross-validation, meta-agent synthesis using LangChain summarizers and NumPy weighting.
9. **Observability & Benchmarking** (`backend/app/core/logging.py`, `backend/app/monitoring/`): Structlog-based JSON logging, Prometheus metrics export (future), Grafana dashboards, agent benchmarking harness.

## 3. Component Interactions

> **Visual assets**: Editable Mermaid source files live in `docs/diagrams/`. Render them locally with the Mermaid CLI or VS Code Mermaid preview extension.
> ```powershell
> npm install -g @mermaid-js/mermaid-cli
> mmdc -i docs/diagrams/system-architecture.mmd -o docs/diagrams/system-architecture.png
> mmdc -i docs/diagrams/task-lifecycle.mmd -o docs/diagrams/task-lifecycle.png
> mmdc -i docs/diagrams/agent-negotiation-state.mmd -o docs/diagrams/agent-negotiation-state.png
> mmdc -i docs/diagrams/memory-consolidation-sequence.mmd -o docs/diagrams/memory-consolidation-sequence.png
> ```
> The generated PNG/SVG files can be dropped into slide decks or shared documentation as needed.

### Diagram Assets

| Diagram | PNG | SVG |
| --- | --- | --- |
| System Architecture | `docs/diagrams/system-architecture.png` | `docs/diagrams/system-architecture.svg` |
| Task Lifecycle | `docs/diagrams/task-lifecycle.png` | `docs/diagrams/task-lifecycle.svg` |
| Agent Negotiation State | `docs/diagrams/agent-negotiation-state.png` | `docs/diagrams/agent-negotiation-state.svg` |
| Memory Consolidation Sequence | `docs/diagrams/memory-consolidation-sequence.png` | `docs/diagrams/memory-consolidation-sequence.svg` |

### Request Lifecycle
1. **User submits a task** via the frontend Socket.io client or REST endpoint.
2. **FastAPI** validates the payload (`TaskRequest`) and enqueues a job through `TaskQueueManager`.
3. **Queue worker** invokes the LangGraph orchestrator with a normalized task object.
4. **Orchestrator** selects agents based on capabilities, hydrates context from `HybridMemoryService`, and calls out to the LLM layer when needed.
5. **Agents** perform domain-specific reasoning, enriching outputs with confidence scores and memory references.
6. **Conflict Resolver** evaluates agent outputs; if disagreements arise, it triggers comparative scoring or meta-synthesis.
7. **Results** are stored back into the hybrid memory (Redis for short-term replay, Postgres for audit history, Qdrant for semantic recall) and streamed to the frontend.
8. **Observability** captures metrics and structured logs for each task execution, enabling replay and benchmarking.

### Memory Strategy
- **Working Memory (Redis)**: Recent conversation turns, agent state, cache entries for quick lookups.
- **Episodic Memory (PostgreSQL)**: Task transcripts, negotiation logs, confidence trajectories for explainability.
- **Semantic Memory (Qdrant)**: Vectorized documents, agent outputs, and contextual summaries keyed by embeddings (`SentenceTransformer`).
- **Memory Consolidation**: Scheduled jobs aggregate Redis/Postgres entries into Qdrant and generate summaries for long-term use.

## 4. Local Deployment Footprint
- **Python Environment**: Managed via Poetry/uv (pending lock-in) with FastAPI, LangChain, LangGraph, asyncpg, redis-py, qdrant-client.
- **Ollama Runtime**: Hosts LLaMA 3 model locally; orchestrator communicates over HTTP.
- **Data Services**: Redis, PostgreSQL, and Qdrant provisioned through Docker Compose with persistent volumes.
- **Frontend**: Bun/Vite dev server for Next.js app; communicates with backend via REST + WebSocket endpoints.

## 5. Observability & Monitoring
- **Logging**: Structured JSON logs via `structlog`, enriched with correlation IDs and task metadata.
- **Metrics**: FastAPI routes expose Prometheus endpoints (to be implemented) for request latency, queue depth, agent success rates.
- **Dashboards**: Grafana visualizes metrics and benchmarks; local deployment avoids SaaS dependencies.
- **Benchmarking Harness**: `monitoring/benchmark.py` aggregates agent evaluation runs for continuous assessment.

## 6. Security & Configuration
- **Config Management**: Centralized in `core/config.py` using Pydantic settings with env-file support and nested namespaces.
- **Authentication**: JWT utilities in `core/security.py` for access tokens; extend FastAPI dependencies to protect endpoints.
- **Secrets Handling**: `.env` file (gitignored) feeds runtime secrets; no cloud secret managers required.

## 7. Phase-Wise Roadmap

### Phase 1 – Backend & Environment Setup
- Establish Python project scaffolding with `pyproject.toml`, dependency pins, and tooling (`ruff`, `pytest`).
- Author Docker Compose stack covering FastAPI, Redis, PostgreSQL, Qdrant, Ollama, Prometheus, Grafana.
- Implement baseline FastAPI app (`/health`, `/submit_task`, `/history`) and configure structured logging.
- Validate local development workflow (hot reload, lint/test automation).

### Phase 2 – Core LLM Integration Layer
- Configure LangChain + Ollama client, including retry/backoff policies and model selection.
- Abstract LLM access behind a service interface for easier testing and future swaps.
- Implement prompt templates and response normalization shared across agents.
- Add smoke tests exercising mock LLM responses.

### Phase 3 – Memory & RAG System
- Bootstrap Redis/PostgreSQL/Qdrant schemas and migrations.
- Implement `HybridMemoryService` CRUD and retrieval utilities with integration tests (using containers or mocks).
- Wire embedding generation via Sentence Transformers with caching and fallbacks.
- Create periodic consolidation job skeleton (Celery/async background task) to move short-term memory into Qdrant.

### Phase 4 – Agent Implementation (LangGraph Nodes)
- Define agent capability contracts and register Research, Finance, Creative, Enterprise nodes.
- Integrate domain-specific tools (e.g., DuckDuckGo, yfinance, transformer-based stylizer) with proper rate limiting.
- Implement agent confidence scoring and output schemas.
- Cover happy-path agent executions with fixtures/mocks.

### Phase 5 – Orchestrator & Negotiation Logic
- Design LangGraph DAG for capability-based routing and fallback strategies.
- Implement negotiation loop (confidence voting, reassignment) and persistence of negotiation logs.
- Introduce task queue workers consuming orchestrator jobs with graceful shutdown handling.
- Add integration tests validating routing decisions across multiple agent capabilities.

### Phase 6 – Conflict Resolution & Meta-Agent
- Build meta-agent leveraging LangChain summarizers for synthesis.
- Implement confidence weighting (NumPy/Scipy), cross-validation hooks, and dispute logging.
- Expose explainability artifacts via `/history` endpoint.
- Add regression tests covering conflicting agent outputs and synthesis results.

### Phase 7 – FastAPI Integration & Observability
- Extend REST + Socket.io endpoints to stream agent progress and final outcomes.
- Instrument Prometheus metrics (request latency, queue depth, agent success rate) and publish dashboards in Grafana.
- Harden security: JWT-protected routes, rate limiting, audit logging.
- Expand CI pipeline to run lint, type-check, unit, and integration suites; document local troubleshooting guide.

## 8. Directory Map (Backend)
```
backend/
 ├─ app/
 │   ├─ api/                 # FastAPI routers (REST/WebSocket)
 │   ├─ agents/              # Domain-specific agent implementations
 │   ├─ core/                # Config, logging, security primitives
 │   ├─ monitoring/          # Benchmarking and observability utilities
 │   ├─ orchestration/       # LangGraph workflows, conflict resolution
 │   ├─ queue/               # Task queue manager (Redis + in-memory)
 │   ├─ schemas/             # Pydantic request/response models
 │   ├─ services/            # Hybrid memory, LLM adapters, external services
 │   └─ utils/               # Shared helpers (embeddings, etc.)
 └─ tests/
     └─ test_health.py       # FastAPI health-check regression tests
```

## 9. Non-Cloud Deployment Checklist
- [ ] Install dependencies via Poetry/uv.
- [ ] Start Ollama with LLaMA 3 model locally.
- [ ] Launch Docker Compose stack for Redis, PostgreSQL, Qdrant, Prometheus, Grafana.
- [ ] Run FastAPI app with `uvicorn app.main:app --reload`.
- [ ] Start Next.js frontend dev server (`bun dev`).
- [ ] Execute test suite (`pytest`).

This document will evolve as the implementation matures; treat it as the living source of truth for system architecture decisions.
