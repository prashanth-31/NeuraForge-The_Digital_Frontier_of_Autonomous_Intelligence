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
    A[Frontend (React 18 + Vite SPA)] -->|Tasks / SSE Streams| B[FastAPI Backend]
    B -->|Task Submission| C[Task Queue Manager]
    C -->|Jobs| D[LangGraph Orchestrator]
    D -->|Plan Request| P[LLM Planner (llama3.2:1b)]
    P -->|Agent Plan| D
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
1. **Frontend Experience** (existing in `frontend/`): React 18 + Vite single-page app with Tailwind UI, TanStack Query caching, and a manual Server-Sent Events (SSE) reader layered on top of the `fetch` streaming API.
2. **API Gateway** (`backend/app/main.py`): FastAPI serves REST endpoints plus Server-Sent Event (SSE) streams, performs auth, and handles schema validation.
3. **Task Coordination** (`backend/app/queue/`): Async in-process queue with optional Redis backend to buffer long-running jobs.
4. **LangGraph Orchestration** (`backend/app/orchestration/`): The `LLMOrchestrationPlanner` (backed by `llama3.2:1b`) proposes ordered agent/tool plans which the LangGraph-based orchestrator enforces end-to-end with no heuristic router fallback.
5. **Agent Layer** (`backend/app/agents/`): Domain-specific logic for Research, Finance, Creative, Enterprise agents (extendable via LangChain tools).
6. **Hybrid Memory** (`backend/app/services/memory.py`): Unified interface to Redis (working memory), PostgreSQL (episodic logs), Qdrant (semantic vectors).
7. **LLM Layer** (`backend/app/utils/embeddings.py`, LangChain integrations): Ollama-hosted LLaMA 3 for completions; Sentence Transformers all-mpnet-base-v2 served from `backend/models/all-mpnet-base-v2/` for embedding generation.
8. **Conflict Resolution** (`backend/app/orchestration/` roadmap): Confidence scoring, cross-validation, meta-agent synthesis using LangChain summarizers and NumPy weighting.
9. **Observability & Benchmarking** (`backend/app/core/logging.py`, `backend/app/monitoring/`): Structlog-based JSON logging, Prometheus metrics export (tool/agent/memory), Grafana dashboards, agent benchmarking harness.
10. **Reviewer Operations** (`backend/app/orchestration/review.py`, `frontend/src/pages/Reviews.tsx`): Postgres-backed review store, JWT-secured reviewer console, async notifications, and observability assets dedicated to human-in-the-loop workflows.

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
| Orchestrator Observability Map | _generate from_ `docs/diagrams/orchestrator-observability.mmd` | - |

### Request Lifecycle
1. **User submits a task** via the frontend fetch client, which POSTs to `/submit_task/stream` and immediately begins consuming the streamed SSE response (or uses the plain JSON submit endpoint for fire-and-forget tasks).
2. **FastAPI** validates the payload (`TaskRequest`) and enqueues a job through `TaskQueueManager`.
3. **Queue worker** invokes the LangGraph orchestrator with a normalized task object.
4. **Orchestrator** captures a planner-directed sequence of agents and required tools, hydrates context from `HybridMemoryService`, and calls out to the LLM layer when needed.
5. **Agents** perform domain-specific reasoning, enriching outputs with confidence scores and memory references.
6. **Conflict Resolver** evaluates agent outputs; if disagreements arise, it triggers comparative scoring or meta-synthesis.
7. **Results** are stored back into the hybrid memory (Redis for short-term replay, Postgres for audit history, Qdrant for semantic recall) and streamed to the frontend over SSE events (task lifecycle, tool telemetry, completion payloads).
8. **Observability** captures metrics and structured logs for each task execution, enabling replay and benchmarking.

## Planner-Led Orchestration
- **Planner Entry Point**: `backend/app/orchestration/llm_planner.py` exports `LLMOrchestrationPlanner`, the sole orchestration selector invoked by `/submit_task` flows.
- **Capability Catalog**: Agent descriptors and dependency graphs live in `backend/app/orchestration/capabilities.json`, and we load them into the planner prompt so the LLM reasons over consistent metadata.
- **Plan Schema**: Planner outputs must conform to `PlannerPlanModel` (ordered agents, tool lists, rationale, handoff notes); validation failures raise `PlannerError` and surface a planner violation event back to the client stream.
- **Execution Flow**: LangGraph executes each planned step sequentially, binding `AgentContext` with planner-specified tool aliases and persisting intermediate results between agents.
- **Failure Handling**: Invalid planner output or plan enforcement errors emit a `planner_failed` stream event, stamp the task with `failure.type="planner_failed"`, and abort the run—there is no heuristic routing fallback.
- **Operational Checklist**: When shipping a new agent or tool, update the capability catalog, refresh the planner prompt fixtures, and add regression tests under `backend/tests/test_orchestrator_simulation.py` to cover the new plan branch.
### Planner Enforcement & Telemetry
- **Tool Contracts**: Planner-supplied primary/fallback tool lists are attached to each `AgentContext`, and `_ToolSession` verifies at least one of them succeeds before the agent can finish.
- **Plan Metadata**: Planner results are persisted alongside task metadata (`state["routing"]["planner"]`) and streamed to clients so downstream consumers can reference the intended path.
- **Metrics & Alerts**: Prometheus tracks `neuraforge_planner_plan_total`, adherence ratios, and violation counts, enabling Grafana to highlight deviations or repeated planner failures.
- **Plan Drift Audits**: Weekly notebooks replay canonical tasks and diff the generated plans against golden outputs to catch prompt drift before it affects production runs.

### Conversation Continuations & Metadata
- **Continuation IDs**: The frontend `TaskContext` threads `continuation_task_id` through follow-up prompts so the backend can stitch conversation chains for HybridMemory lookups.
- **Metadata Normalisation**: `_ensure_conversation_metadata` (in `backend/app/api/routes.py`) guarantees every task persists `root_task_id`, `latest_task_id`, `previous_task_id`, and optional `continuation_task_id` inside both the task state and shared context.
- **Hydrated Follow-ups**: When a continuation ID is provided, `_seed_conversation_state` hydrates outputs, metadata, and shared context from the prior task before the orchestrator runs, ensuring downstream agents receive the accumulated provenance.
- **Frontend State**: SSE responses update the client-side `TaskContext`, which exposes conversation-aware history panes and allows users to branch new tasks from any prior run while maintaining memory integrity.

### Tool-First Agent Loops
- **Mandatory Tool Evidence**: Each agent must log at least one successful MCP tool invocation before returning an answer when tooling is enabled. The orchestrator enforces this policy and will fail the run with a `ToolFirstPolicyViolation` if an agent attempts to respond without validated tool output.
- **Execution Wiring**: `Orchestrator` wraps the shared `ToolService` per agent invocation, records per-tool success/failure metrics, and propagates violation events to the task stream and state store for auditability.
- **Local MCP Router**: `ToolService` talks to the in-process MCP router (`/mcp/tools/...`) so adapters run inside the FastAPI app; aliases such as `finance.snapshot` resolve to `finance/yfinance` and benefit from shared caching, retries, and circuit-breaker safeguards.
- **Metadata Enrichment**: Agent outputs are automatically annotated with a `tools_used` list (alias, resolved identifier, cache status, latency) so downstream planners, reviewers, and telemetry dashboards can trace the provenance of every response.
- **Failure Handling**: If a tool call errors, the orchestrator surfaces the failure details and aborts the run rather than allowing an LLM-only fallback, keeping the system aligned with the tool-first contract.
- **Finance Snapshot Resilience**: The `finance/yfinance` adapter now falls back to on-demand historical bars when live quote APIs rate-limit, normalising metrics (price, change, volume, timestamps) before returning them to agents and ensuring follow-up analyses cite up-to-date figures.

### Memory Strategy
- **Working Memory (Redis)**: Recent conversation turns, agent state, cache entries for quick lookups.
- **Episodic Memory (PostgreSQL)**: Task transcripts, negotiation logs, confidence trajectories for explainability.
- **Semantic Memory (Qdrant)**: Vectorized documents, agent outputs, and contextual summaries keyed by embeddings (`SentenceTransformer`).
- **Memory Consolidation**: Scheduled jobs aggregate Redis/Postgres entries into Qdrant and generate summaries for long-term use.

## 4. Local Deployment Footprint
- **Python Environment**: Managed via Poetry/uv (pending lock-in) with FastAPI, LangChain, LangGraph, asyncpg, redis-py, qdrant-client.
- **Ollama Runtime**: Hosts LLaMA 3 model locally; orchestrator communicates over HTTP.
- **Data Services**: Redis, PostgreSQL, and Qdrant provisioned through Docker Compose with persistent volumes.
- **Frontend**: Vite dev server for the React SPA; communicates with the backend via REST and SSE streaming endpoints.

## 5. Observability & Monitoring
- **Logging**: Structured JSON logs via `structlog`, enriched with correlation IDs and task metadata.
- **Metrics**: Prometheus now captures orchestrator throughput, run latency, negotiation consensus, guardrail decisions, SLA adherence, and meta-agent synthesis telemetry (latency, dispute counts) via the `neuraforge_*` metric family in `app/core/metrics.py`. Phase 6 adds the `neuraforge_review_tickets` gauges that back reviewer workload dashboards and alerting.
- **Tool-First & Planner Telemetry**: Dedicated counters/Histograms (`neuraforge_agent_tool_usage_total`, `neuraforge_agent_tool_policy_total`, `neuraforge_planner_plan_total`, `neuraforge_planner_plan_steps`) track MCP compliance and plan generation health. Grafana dashboards can now highlight agents skipping tools, repeated tool failures, and plan depth trends across tasks.
- **Alerts & Recording Rules**: `observability/rules/orchestrator_rules.yml` defines throughput rollups and alert thresholds for failure rate, guardrail escalations, and negotiation consensus degradation. `observability/rules/review_rules.yml` layers in open-queue and stalled-review alerts. Alerts are forwarded to Alertmanager (`observability/alertmanager/config.yml`), which ships them to the Pager proxy or downstream paging system.
- **Grafana Provisioning**: Dashboards and datasources auto-provision from `observability/grafana/provisioning/`, and custom dashboard JSONs live under `observability/grafana/dashboards/`. The Phase 5 orchestrator dashboard and the new reviewer operations dashboard load automatically when Grafana starts inside Docker Compose.
- **Benchmarking Harness**: `monitoring/benchmark.py` aggregates agent evaluation runs, while `orchestration/simulation.py` replays synthetic scenarios against the orchestrator for load testing. The GitHub Action workflows now cover the simulation harness and the meta-agent benchmark CLI (`.github/workflows/benchmark-ci.yml`) to guard against regressions.
- **Developer Notebook**: `docs/notebooks/phase5_orchestration_demo.ipynb` provides an end-to-end walkthrough of a simulated negotiation run with captured metrics and summaries.
- **Real-time Telemetry**: `/submit_task/stream` streams `tool_invocation` events alongside agent lifecycle updates, giving dashboards and notebooks live visibility into orchestration.
- **Runbooks**: Meta-agent alerts are documented in `docs/observability/meta_agent_runbook.md`, covering dispute spikes, latency SLO breaches, and recommended remediation steps.

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
- Standardize request/response schemas via `app/schemas/agents.py` and maintain registry in `app/agents/contracts.py`.
- Integrate domain-specific tools (e.g., DuckDuckGo, yfinance, transformer-based stylizer) with proper rate limiting.
- Implement agent confidence scoring and output schemas.
- Cover happy-path agent executions with fixtures/mocks.

### Phase 5 – Orchestrator & Negotiation Logic
- Design LangGraph DAG for planner-enforced sequencing and failure-handling strategies.
- Ship the `LLMOrchestrationPlanner`, wire it into orchestration entrypoints, and document the plan schema, prompt controls, and validation path.
- Implement negotiation loop (confidence voting, reassignment) and persistence of negotiation logs.
- Introduce task queue workers consuming orchestrator jobs with graceful shutdown handling.
- Add integration tests validating routing decisions across multiple agent capabilities.

### Phase 6 – Conflict Resolution & Meta-Agent
- Build meta-agent leveraging LangChain summarizers for synthesis.
- Implement confidence weighting (NumPy/Scipy), cross-validation hooks, and dispute logging.
- Wire the meta-agent into the orchestrator lifecycle so every run records a synthesized resolution, escalation recommendation, and dispute metadata.
- Expose explainability artifacts via `/history` and `/reports/{task_id}/dossier.{json|md}` endpoints, including standard decision dossiers.
- Add regression tests covering conflicting agent outputs and synthesis results.

### Phase 7 – FastAPI Integration & Observability
- Extend REST + SSE streaming endpoints to expose agent progress and final outcomes.
- Instrument Prometheus metrics (request latency, queue depth, agent success rate) and publish dashboards in Grafana.
- Harden security: JWT-protected routes, rate limiting, audit logging.
- Expand CI pipeline to run lint, type-check, unit, and integration suites; document local troubleshooting guide.

## 8. Directory Map (Backend)
```
backend/
 ├─ app/
 │   ├─ api/                 # FastAPI routers (REST/SSE)
 │   ├─ agents/              # Domain-specific agent implementations
 │   ├─ core/                # Config, logging, security primitives
 │   ├─ monitoring/          # Benchmarking and observability utilities
 │   ├─ orchestration/       # LangGraph workflows, meta-agent synthesis, decision dossiers
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
- [ ] Launch Docker Compose stack for Redis, PostgreSQL, Qdrant, Prometheus, Grafana, and Alertmanager.
- [ ] Run FastAPI app with `uvicorn app.main:app --reload`.
- [ ] Start the Vite frontend dev server (`npm run dev`).
- [ ] Execute test suite (`pytest`).

## 10. CI/CD & Staging Integration
- **GitHub Actions Workflow**: `.github/workflows/phase5-observability.yml` runs on every push/PR, installing backend dependencies, executing the Phase 5 orchestration test suite, and performing a smoke run of the simulation harness.
- **Compose Validation**: The workflow validates `docker-compose.yml` to ensure Prometheus, Grafana (with provisioning), and Alertmanager definitions stay healthy and publishes the Grafana dashboard JSON as a build artifact for staging environments.
- **Staging Deployment**: Operations can `docker compose up prometheus grafana alertmanager` from `implementation/` to bring up the observability toolchain. Grafana auto-loads the Phase 5 dashboard, and Prometheus streams alerts to Alertmanager, which forwards notifications to the configured Pager/Webhook endpoint in `observability/alertmanager/config.yml` (set `PAGER_WEBHOOK_URL` in the environment or `.env` file to point at the real pager service).
- **Simulation Harness in Pipelines**: The smoke test leverages `SimulationHarness` with a synthetic orchestrator to catch regressions in negotiation, guardrail, and reporting flows before deployment.

This document will evolve as the implementation matures; treat it as the living source of truth for system architecture decisions.
