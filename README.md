# 🧠 NeuraForge - The Digital Frontier of Autonomous Intelligence

<div align="center">

![NeuraForge](https://img.shields.io/badge/NeuraForge-Autonomous%20Intelligence-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.119+-green?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-18.3+-61DAFB?style=for-the-badge&logo=react)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A local-first, multi-agent intelligence platform for orchestrating specialized AI agents with hybrid memory, LangGraph workflows, and comprehensive observability.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [API Reference](#-api-reference)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Agents](#-agents)
- [Memory System](#-memory-system)
- [Orchestration](#-orchestration)
- [Observability](#-observability)
- [Security](#-security)
- [Testing](#-testing)
- [Development](#-development)
- [Deployment](#-deployment)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

NeuraForge is a privacy-preserving, multi-agent AI orchestration platform designed to run entirely on local infrastructure. It coordinates specialized agents (Research, Finance, Creative, Enterprise) through sophisticated LangGraph workflows, leveraging hybrid memory systems and local LLM inference via Ollama.

### Core Philosophy

- **🔒 Privacy-First**: All services run locally—no cloud dependencies or external telemetry
- **🤖 Multi-Agent Collaboration**: Specialized agents coordinate through negotiation and consensus
- **🧠 Intelligent Memory**: Hybrid storage combining short-term, episodic, and semantic memory
- **📊 Full Observability**: Comprehensive metrics, logging, and dashboards for transparency
- **🎯 Tool-First Execution**: Agents must provide evidence from tool invocations before responding

---

## ✨ Features

### Agent System
- **Research Agent** - Web research, document analysis, and information synthesis
- **Finance Agent** - Market data, financial analysis, and portfolio insights
- **Creative Agent** - Content generation, brainstorming, and creative writing
- **Enterprise Agent** - Business intelligence, workflow automation, and data processing
- **General Agent** - Fallback agent for unclassified tasks

### Orchestration Engine
- **LLM-Powered Planner** - Intelligent task decomposition using `llama3.1:8b`
- **LangGraph Workflows** - DAG-based execution with state management
- **Negotiation Protocol** - Multi-round consensus building between agents
- **Conflict Resolution** - Meta-agent synthesis for disputed outcomes
- **Guardrails System** - Policy enforcement and content filtering

### Memory Architecture
- **Working Memory** (Redis) - Fast access for recent conversations and cache
- **Episodic Memory** (PostgreSQL) - Task transcripts and audit trails
- **Semantic Memory** (Qdrant) - Vector search for contextual retrieval
- **Memory Consolidation** - Automated aggregation and summarization jobs

### Frontend Experience
- **Modern React SPA** - Built with React 18, Vite, and TypeScript
- **Real-time Updates** - Server-Sent Events (SSE) for live task progress
- **Reviewer Console** - Human-in-the-loop review workflows
- **Knowledge Management** - Document and insight organization
- **Responsive UI** - Tailwind CSS with shadcn/ui components

### Observability Stack
- **Prometheus Metrics** - Custom `neuraforge_*` metric family
- **Grafana Dashboards** - Pre-configured orchestration and reviewer panels
- **Loki + Promtail** - Centralized log aggregation
- **Alertmanager** - Configurable alerting pipelines
- **Structured Logging** - JSON logs with correlation IDs via structlog

---

## 🏗 Architecture

```mermaid
graph TD
    A[Frontend - React 18 + Vite] -->|Tasks / SSE Streams| B[FastAPI Backend]
    B -->|Task Submission| C[Task Queue Manager]
    C -->|Jobs| D[LangGraph Orchestrator]
    D -->|Plan Request| P[LLM Planner - llama3.1:8b]
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

| Layer | Component | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 + Vite + TanStack Query | User interface and real-time updates |
| **API Gateway** | FastAPI + SSE | REST endpoints, streaming, authentication |
| **Task Queue** | Redis-backed async queue | Job scheduling and processing |
| **Orchestration** | LangGraph + LLM Planner | Workflow execution and agent coordination |
| **Agents** | Domain-specific modules | Specialized reasoning and tool usage |
| **Memory** | Redis + PostgreSQL + Qdrant | Hybrid storage with semantic search |
| **LLM Layer** | Ollama + LangChain | Local inference and embeddings |
| **Observability** | Prometheus + Grafana + Loki | Metrics, dashboards, and logging |

---

## 🛠 Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Core runtime |
| FastAPI | 0.119+ | REST API framework |
| LangChain | 0.3.x | LLM integration |
| LangGraph | 0.6.x | Workflow orchestration |
| Pydantic | 2.12+ | Data validation |
| SQLAlchemy | 2.0+ | ORM for PostgreSQL |
| Redis | 7.4+ | Caching and queues |
| Qdrant | 1.15+ | Vector database |
| PostgreSQL | 17+ | Relational database |
| Sentence Transformers | 5.1+ | Embedding generation |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.3+ | UI framework |
| TypeScript | 5.8+ | Type safety |
| Vite | 5.4+ | Build tooling |
| TanStack Query | 5.x | Data fetching |
| Tailwind CSS | 3.4+ | Styling |
| shadcn/ui | Latest | Component library |
| React Router | 6.x | Client routing |
| Zod | 3.x | Schema validation |

### Infrastructure
| Technology | Version | Purpose |
|------------|---------|---------|
| Docker Compose | Latest | Container orchestration |
| Ollama | Latest | Local LLM serving |
| Prometheus | 2.53+ | Metrics collection |
| Grafana | 11.1+ | Visualization |
| Loki | 3.0+ | Log aggregation |
| Alertmanager | 0.27+ | Alert routing |

---

## 📦 Prerequisites

- **Python 3.11+** with Poetry or pip
- **Node.js 18+** with npm or Bun
- **Docker Desktop** with Docker Compose
- **Ollama** with LLaMA 3 model
- **NVIDIA GPU** (optional, for CUDA acceleration)
- **Git** for version control

### GPU Acceleration (Optional)

For GPU-accelerated embeddings:
1. Install NVIDIA drivers
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
3. Verify with `nvidia-smi`

---

## 🚀 Quick Start

### 1. Clone the Repository

```powershell
git clone https://github.com/your-org/NeuraForge.git
cd NeuraForge-The_Digital_Frontier_of_Autonomous_Intelligence
```

### 2. Set Up Ollama

```powershell
# Install Ollama from https://ollama.ai
ollama pull llama3.1:8b
ollama pull llama3
```

### 3. Configure Environment

```powershell
# Backend configuration
cd implementation/backend
Copy-Item .env.example .env
# Edit .env with your settings (API keys, database credentials, etc.)
```

### 4. Start Infrastructure

```powershell
# From the implementation directory
cd implementation
docker compose up -d postgres redis qdrant prometheus grafana loki
```

### 5. Initialize Backend

```powershell
cd backend

# Using Poetry (recommended)
poetry install
poetry run alembic upgrade head
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using pip
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start Frontend

```powershell
cd ../frontend
npm install
npm run dev
```

### 7. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost:5173 | - |
| Backend API | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | - |
| Qdrant UI | http://localhost:16333/dashboard | - |

---

## 📁 Project Structure

```
NeuraForge/
├── implementation/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── agents/          # Domain-specific agent implementations
│   │   │   │   ├── base.py      # Base agent class
│   │   │   │   ├── research.py  # Research agent
│   │   │   │   ├── finance.py   # Finance agent
│   │   │   │   ├── creative.py  # Creative agent
│   │   │   │   ├── enterprise.py# Enterprise agent
│   │   │   │   └── general.py   # General fallback agent
│   │   │   ├── api/             # FastAPI routes (REST/SSE)
│   │   │   ├── core/            # Config, logging, security
│   │   │   ├── db/              # Database models and sessions
│   │   │   ├── mcp/             # Model Context Protocol adapters
│   │   │   ├── monitoring/      # Benchmarking utilities
│   │   │   ├── orchestration/   # LangGraph workflows
│   │   │   │   ├── llm_planner.py    # LLM-based task planner
│   │   │   │   ├── graph.py          # LangGraph DAG definition
│   │   │   │   ├── negotiation.py    # Multi-agent negotiation
│   │   │   │   ├── meta.py           # Meta-agent synthesis
│   │   │   │   ├── guardrails.py     # Policy enforcement
│   │   │   │   └── capabilities.json # Agent registry
│   │   │   ├── queue/           # Task queue manager
│   │   │   ├── schemas/         # Pydantic models
│   │   │   ├── services/        # Memory, LLM adapters
│   │   │   ├── tools/           # External tool integrations
│   │   │   └── utils/           # Shared helpers
│   │   ├── alembic/             # Database migrations
│   │   ├── tests/               # Test suites
│   │   ├── models/              # Pre-downloaded ML models
│   │   ├── pyproject.toml       # Python dependencies
│   │   └── Dockerfile           # Backend container
│   │
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── components/      # Reusable UI components
│   │   │   ├── pages/           # Route pages
│   │   │   │   ├── Dashboard.tsx
│   │   │   │   ├── Reviews.tsx
│   │   │   │   ├── Knowledge.tsx
│   │   │   │   └── Settings.tsx
│   │   │   ├── hooks/           # Custom React hooks
│   │   │   ├── contexts/        # React contexts
│   │   │   ├── lib/             # Utility functions
│   │   │   └── types/           # TypeScript definitions
│   │   ├── package.json
│   │   └── vite.config.ts
│   │
│   ├── observability/
│   │   ├── prometheus.yml       # Prometheus config
│   │   ├── alertmanager/        # Alert routing
│   │   ├── grafana/             # Dashboards and provisioning
│   │   ├── loki/                # Log aggregation config
│   │   └── rules/               # Recording and alerting rules
│   │
│   ├── docs/
│   │   ├── architecture.md      # System architecture
│   │   ├── api.md               # API reference
│   │   ├── security.md          # Security guidelines
│   │   └── runbooks/            # Operational runbooks
│   │
│   ├── scripts/                 # Utility scripts
│   └── docker-compose.yml       # Service orchestration
│
├── backups/                     # Database backups
├── Jenkinsfile                  # CI/CD pipeline
└── README.md                    # This file
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in `implementation/backend/` with:

```env
# Application
APP_ENV=development
DEBUG=true
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:15432/neuraforge
REDIS_URL=redis://localhost:16379/0

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=16333

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
PLANNER_LLM__MODEL=llama3.1:8b

# Finance Tools
ALPHAVANTAGE_API_KEY=demo
FINANCE_SNAPSHOT_PROVIDER=alpha_vantage,yfinance

# Observability
LOG_LEVEL=INFO
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc

# Consolidation
CONSOLIDATION_ENABLED=true
CONSOLIDATION_INTERVAL_SECONDS=300
```

### Key Configuration Files

| File | Purpose |
|------|---------|
| `backend/.env` | Runtime environment variables |
| `backend/app/core/config.py` | Pydantic settings with validation |
| `observability/prometheus.yml` | Metrics scraping configuration |
| `observability/alertmanager/config.yml` | Alert routing rules |
| `docker-compose.yml` | Service definitions and networking |

---

## 🖥 Running the Application

### Development Mode

```powershell
# Terminal 1: Infrastructure
cd implementation
docker compose up -d postgres redis qdrant prometheus grafana

# Terminal 2: Backend
cd implementation/backend
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Frontend
cd implementation/frontend
npm run dev
```

### Production Mode (Docker Compose)

```powershell
cd implementation
docker compose up --build -d
```

### Verify Stack Health

```powershell
# Check all services
docker compose ps

# Verify backend health
curl http://localhost:8000/health

# Verify GPU access (if available)
docker compose exec backend nvidia-smi
```

---

## 📡 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/api/v1/submit_task` | Submit a new task |
| `POST` | `/api/v1/submit_task/stream` | Submit task with SSE streaming |
| `GET` | `/api/v1/tasks/{task_id}` | Get task status and results |
| `GET` | `/api/v1/history` | List task history |

### Reviewer Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/reviews/metrics` | Reviewer workload metrics |
| `GET` | `/api/v1/reviews` | List review tickets |
| `POST` | `/api/v1/reviews/{id}/approve` | Approve a review |
| `POST` | `/api/v1/reviews/{id}/reject` | Reject a review |

### Orchestration Introspection

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/orchestrator/runs/{run_id}` | Get orchestration details |
| `GET` | `/api/v1/reports/{task_id}/dossier.json` | Decision dossier |

### SSE Event Types

```typescript
type SSEEvent =
  | { event: "task_started"; data: TaskStartPayload }
  | { event: "planner_result"; data: PlannerPlan }
  | { event: "agent_started"; data: AgentStartPayload }
  | { event: "agent_completed"; data: AgentResult }
  | { event: "tool_invocation"; data: ToolTelemetry }
  | { event: "guardrail_triggered"; data: GuardrailDecision }
  | { event: "task_completed"; data: TaskResult }
  | { event: "task_failed"; data: ErrorPayload };
```

### Example: Submit a Task

```bash
curl -X POST http://localhost:8000/api/v1/submit_task \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <JWT>" \
  -d '{
    "prompt": "Analyze the market trends for AAPL stock",
    "metadata": {
      "priority": "high",
      "domain": "finance"
    }
  }'
```

---

## 🤖 Agents

### Agent Capabilities

| Agent | Domain | Primary Tools | Use Cases |
|-------|--------|---------------|-----------|
| **Research** | Information gathering | Web search, document analysis | Market research, competitive analysis |
| **Finance** | Financial analysis | Alpha Vantage, yfinance | Stock quotes, portfolio analysis |
| **Creative** | Content generation | Text synthesis, brainstorming | Marketing copy, creative writing |
| **Enterprise** | Business operations | Data processing, automation | Reports, workflow automation |
| **General** | Fallback | Basic LLM inference | Unclassified queries |

### Tool-First Policy

Agents must provide evidence from MCP tool invocations before returning responses:

```python
# Each agent response includes tool provenance
{
  "agent": "finance",
  "summary": "AAPL is trading at $185.50...",
  "confidence": 0.92,
  "tools_used": [
    {
      "alias": "finance.snapshot",
      "resolved": "finance/alpha_vantage",
      "cache_status": "miss",
      "latency_ms": 342
    }
  ]
}
```

---

## 💾 Memory System

### Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Working Memory (Redis)                │
│  • Recent conversations    • Agent state cache          │
│  • Tool response cache     • Session data               │
├─────────────────────────────────────────────────────────┤
│                   Episodic Memory (PostgreSQL)          │
│  • Task transcripts        • Negotiation logs           │
│  • Audit trails            • Confidence trajectories    │
├─────────────────────────────────────────────────────────┤
│                   Semantic Memory (Qdrant)              │
│  • Vectorized documents    • Agent outputs              │
│  • Contextual summaries    • Knowledge embeddings       │
└─────────────────────────────────────────────────────────┘
```

### Memory Consolidation

Scheduled jobs automatically:
1. Aggregate Redis entries into PostgreSQL
2. Generate vector embeddings for Qdrant
3. Create summaries for long-term retrieval
4. Prune expired working memory

---

## 🎭 Orchestration

### Planner-Led Execution

1. **Task Submission** → User submits a prompt
2. **LLM Planning** → `llama3.1:8b` generates an execution plan
3. **Plan Validation** → Schema enforcement with fallback handling
4. **Agent Execution** → Sequential execution per plan steps
5. **Negotiation** → Multi-round consensus if agents disagree
6. **Meta-Synthesis** → Conflict resolution and final answer
7. **Response Streaming** → SSE events throughout execution

### Guardrails System

```python
# Guardrail policies
- llm-toxicity       # Content safety filtering
- pii-detection      # Personal data masking
- rate-limiting      # Abuse prevention
- confidence-threshold  # Quality gates
```

---

## 📊 Observability

### Metrics (Prometheus)

```promql
# Key metrics
neuraforge_task_total                    # Total tasks processed
neuraforge_task_latency_seconds          # Task execution latency
neuraforge_agent_tool_usage_total        # Tool invocation counts
neuraforge_planner_plan_total            # Plans generated
neuraforge_guardrail_decision_total      # Guardrail triggers
neuraforge_review_tickets                # Open review tickets
```

### Dashboards (Grafana)

| Dashboard | Purpose |
|-----------|---------|
| Orchestrator Overview | Task throughput, latency, agent health |
| Reviewer Operations | Ticket backlog, reviewer performance |
| Memory Health | Cache hit rates, vector search latency |
| System Resources | CPU, memory, GPU utilization |

### Alerts

Pre-configured alerts for:
- High task failure rate
- Guardrail escalation spikes
- Reviewer backlog threshold
- Memory consolidation failures
- Agent timeout violations

---

## 🔐 Security

### Authentication

- **JWT-based** authentication with role claims
- **Scopes**: `tasks:read`, `tasks:write`, `reviews:read`, `reviews:write`, `admin`
- **Token expiration** with refresh flow

### Rate Limiting

- Redis-backed rate limiter
- Per-endpoint and per-user limits
- Graceful degradation when Redis unavailable

### Audit Logging

All sensitive operations are logged with:
- Request timestamp
- User identity
- Route and method
- Payload hash
- Response status

---

## 🧪 Testing

### Running Tests

```powershell
# Backend tests
cd implementation/backend
poetry run pytest

# With coverage
poetry run coverage run -m pytest
poetry run coverage report

# Specific test file
poetry run pytest tests/test_orchestrator_simulation.py -v

# Frontend tests
cd implementation/frontend
npm run test:e2e
```

### Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `backend/tests/` | Individual component tests |
| Integration | `backend/tests/` | Service interaction tests |
| E2E | `frontend/tests/` | Full stack Playwright tests |
| Simulation | `backend/app/orchestration/simulation.py` | Load testing |

---

## 🔧 Development

### Code Quality

```powershell
# Linting
poetry run ruff check app

# Type checking
cd frontend && npx tsc --noEmit

# Format code
poetry run ruff format app
```

### Database Migrations

```powershell
# Create migration
poetry run alembic revision --autogenerate -m "description"

# Apply migrations
poetry run alembic upgrade head

# Rollback
poetry run alembic downgrade -1
```

### Adding a New Agent

1. Create agent class in `app/agents/`
2. Register in `app/orchestration/capabilities.json`
3. Add tools to `app/mcp/adapters/`
4. Update planner prompts
5. Add tests in `tests/`

---

## 🚢 Deployment

### Docker Compose (Recommended)

```powershell
cd implementation
docker compose up --build -d
```

### Manual Deployment

1. Set up PostgreSQL, Redis, Qdrant
2. Configure environment variables
3. Run database migrations
4. Start backend with Gunicorn/Uvicorn
5. Build and serve frontend
6. Configure reverse proxy (nginx)

### Health Checks

```powershell
# Verify all services
python implementation/scripts/verify_stack.py
```

---

## 🗺 Roadmap

### Completed Phases

- [x] **Phase 1**: Backend & Environment Setup
- [x] **Phase 2**: Core LLM Integration Layer
- [x] **Phase 3**: Memory & RAG System
- [x] **Phase 4**: Agent Implementation (LangGraph Nodes)
- [x] **Phase 5**: Orchestrator & Negotiation Logic
- [x] **Phase 6**: Conflict Resolution & Meta-Agent
- [x] **Phase 7**: FastAPI Integration & Observability

### Future Enhancements

- [ ] Multi-modal agent support (vision, audio)
- [ ] Distributed task queue (Celery/RQ)
- [ ] Kubernetes deployment manifests
- [ ] A/B testing for planner prompts
- [ ] Agent marketplace for custom agents
- [ ] Real-time collaboration features

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow existing code style
- Write tests for new features
- Update documentation
- Keep commits atomic and descriptive

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
- [Ollama](https://ollama.ai) - Local LLM serving
- [FastAPI](https://fastapi.tiangolo.com) - API framework
- [Qdrant](https://qdrant.tech) - Vector database
- [shadcn/ui](https://ui.shadcn.com) - UI components

---

<div align="center">

**Built with ❤️ by the NeuraForge Team**

[Report Bug](https://github.com/your-org/NeuraForge/issues) • [Request Feature](https://github.com/your-org/NeuraForge/issues)

</div>
