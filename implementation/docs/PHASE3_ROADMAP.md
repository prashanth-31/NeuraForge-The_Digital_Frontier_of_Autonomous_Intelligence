# Phase 3 Roadmap – Memory & Retrieval System

## Mission Statement
Deliver a production-ready hybrid memory subsystem that fuses Redis (working memory), PostgreSQL (episodic memory), and Qdrant (semantic memory) to power Retrieval-Augmented Generation (RAG) across all agents. Phase 3 culminates in an end-to-end data loop that captures task context, persists it durably, and serves high-quality retrievals to downstream orchestration.

## Success Criteria
- **Unified Memory APIs**: `HybridMemoryService` exposes consistent CRUD and search interfaces used by LangGraph flows and FastAPI endpoints.
- **Embeddings in Production**: Sentence Transformer-based embeddings generated, cached, and stored in Qdrant with versioned model metadata.
- **Consolidation Loop**: Automated background job migrates short-term memory (Redis/Postgres) into long-term semantic stores with summarization where appropriate.
- **Observability**: Metrics, logs, and alerts capture memory ingest, consolidation latency, cache hit rates, and retrieval relevance scores.
- **Test Coverage**: Integration tests validate memory CRUD, embedding, and retrieval pipelines using local containers or mocks.

## Key Workstreams & Tasks

### 1. Storage Provisioning & Schema Design
- Finalize Docker Compose services for Redis, PostgreSQL, and Qdrant (volumes, health checks, resource limits).
- Design PostgreSQL schemas/tables for episodic logs, negotiation transcripts, and consolidation status; create Alembic migrations.
- Define Redis key namespaces and TTL policies for conversation buffers and agent scratchpads.
- Establish Qdrant collection schema (distance metric, payload structure, shard/replica config) and bootstrap collections.

### 2. HybridMemoryService Implementation
- Implement repository layer abstractions for each store (RedisRepository, PostgresRepository, QdrantRepository).
- Compose repositories into `HybridMemoryService` with façade methods (store_episode, fetch_recent_context, upsert_vector, etc.).
- Add configurable read preferences (e.g., Redis-first with Postgres fallback) and batching for bulk writes.
- Provide FastAPI dependency wiring and service configuration using Pydantic settings.

### 3. Embedding & Caching Pipeline
- Choose Sentence Transformer model (e.g., `all-MiniLM-L6-v2`) and document hardware/runtime requirements.
- Implement embedding worker module with fallbacks to Ollama-hosted models if offline.
- Add caching strategy (Redis-based) to avoid recomputing embeddings for identical content.
- Track embedding metadata (model name, version, dimensionality) alongside vectors in Qdrant.

### 4. Retrieval & Context Assembly
- Implement semantic + episodic retrieval routines (hybrid search combining Qdrant vectors with Postgres/Redis filters).
- Build context assembly utilities that score and merge retrieved snippets, enforcing token budgets per agent request.
- Expose retrieval endpoints or internal APIs for LangGraph nodes; provide sample usage in orchestrator layer.

### 5. Consolidation & Maintenance Jobs
- Implement async background task (FastAPI startup event or Celery worker) that periodically:
  - Drains Redis scratchpads into Postgres episodic tables.
  - Summarizes episodic entries and pushes embeddings into Qdrant.
  - Marks processed items with idempotent checkpoints to avoid duplication.
- Add manual CLI/management commands to trigger consolidation for debugging.
- Include retention policies (prune expired Redis keys, archive old episodes).

### 6. Observability & QA
- Instrument Prometheus metrics: ingestion throughput, consolidation duration, retrieval latency, cache hit/miss counts.
- Extend structured logging to include memory operation metadata (store, operation type, latency, record ids).
- Create pytest integration suites using dockerized services or testcontainers; cover happy path and failure modes (store unavailable, embedding fallback, duplicate upserts).
- Document runbooks for troubleshooting (connection failures, data integrity issues).

### 7. Documentation & Developer Experience
- Update `README` and architecture docs with memory service usage guides and diagrams.
- Provide sample notebooks or scripts demonstrating memory ingest + retrieval.
- Add onboarding checklist covering environment variables, local service startup, and test execution.

## Timeline (Indicative, 3 Sprints)
- **Sprint 1**: Provision services, finalize schemas, scaffold repositories, baseline tests.
- **Sprint 2**: Embedding pipeline, hybrid retrieval APIs, consolidation job MVP.
- **Sprint 3**: Observability, QA hardening, documentation, developer tooling polish.

## Dependencies & Risks
- **Poetry Environment Stability**: Install/test failures must be resolved to run integration suites (carryover from Phase 2).
- **Model Availability**: Ensure embedding models are downloadable offline or mirrored locally.
- **Data Volume Growth**: Plan for Qdrant/Postgres scaling; validate backup/restore procedures.
- **Security**: Guard memory endpoints with proper auth scopes before exposing externally.

## Exit Checklist
- [ ] `HybridMemoryService` fully implemented with tests.
- [ ] Embedding worker deployed with caching & monitoring.
- [ ] Consolidation job running automatically with dashboards tracking throughput.
- [ ] Documentation updated and reviewed.
- [ ] End-to-end flow demo recorded (ingest ➝ retrieval powering agent response).
