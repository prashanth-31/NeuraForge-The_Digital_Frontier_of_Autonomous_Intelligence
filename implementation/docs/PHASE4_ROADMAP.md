# Phase 4 Roadmap – Agent Implementation (LangGraph Nodes)

## Mission Statement
Deliver production-ready LangGraph agent nodes for Research, Finance, Creative, and Enterprise personas, each leveraging domain-specific tools, unified confidence scoring, and structured outputs to support orchestration in later phases.

## Success Criteria
- **Capability Contracts**: Every agent publishes a documented interface (inputs, outputs, tool access, error semantics) consumable by orchestration flows.
- **Tool Integrations**: Required external data sources and utilities (DuckDuckGo search, yfinance market data, document stylizers, etc.) are wrapped with resilience features (timeouts, retries, rate limiting).
- **Confidence & Telemetry**: Agents emit standardized confidence scores, rationales, and telemetry hooks for observability and downstream negotiation logic.
- **Testing Coverage**: Deterministic fixtures and mocks validate happy-path executions and failure fallbacks for each agent.

## Key Workstreams & Tasks

### 1. Agent Contract Design
- Define Pydantic models describing agent input payloads, output schemas, and metadata (confidence, rationale, evidentiary links).
- Publish agent capability documentation in `docs/agents/README.md` including tool availability, expected latencies, and fallback behavior.
- Add validation utilities ensuring orchestrator requests align with declared contracts.

### 2. Agent Node Implementation
- Build LangGraph nodes for Research, Finance, Creative, and Enterprise agents in `app/agents/` with shared base class enhancements (telemetry hooks, context ingestion, timeout handling).
- Implement task-specific logic:
  - **ResearchAgent**: Leverage DuckDuckGo/Web search tool, summarization helpers, citation formatting.
  - **FinanceAgent**: Integrate yfinance or equivalent market data provider, portfolio math utilities, compliance guardrails.
  - **CreativeAgent**: Apply transformer-based tone stylizers, content filters, and persona prompts.
  - **EnterpriseAgent**: Aggregate strategic reasoning, policy alignment checks, and integration with knowledge base APIs.
- Ensure each node respects `ContextAssembler` outputs and memory write-back conventions.

### 3. Tooling & Rate Limiting Layer
- Integrate the emerging Model Context Protocol (MCP) interface so agents can invoke approved general-purpose tools (search, finance, document utilities) through standardized MCP endpoints.
- Create reusable connectors inside `app/services/tools/` that adapt MCP tools to agent-friendly abstractions while enforcing configurable timeouts, retries, and circuit breakers.
- Implement rate limiting middleware (token bucket or leaky bucket) to guard against upstream throttling; expose metrics for tool invocations and failures.
- Cache repeatable queries where appropriate (e.g., recent ticker prices) to reduce external calls.

### 4. Confidence Scoring & Output Normalization
- Define scoring rubric (0–1 scale) factoring in evidence quality, tool reliability signals, and agent self-assessment.
- Extend response schemas with confidence, rationale, and structured evidence references (URLs, document IDs).
- Instrument Prometheus metrics for agent confidence distribution and error counts; log structured events for negotiation debugging.

### 5. Testing & QA Automation
- Build pytest fixtures mocking external tools (DuckDuckGo, yfinance, stylizer) with deterministic payloads.
- Add unit tests covering happy-path agent executions, rate-limit handling, and confidence score calculations.
- Introduce integration tests simulating orchestrator requests through LangGraph nodes with stubbed memory/context services.
- Update CI to tag fast vs. slow tests, ensuring tool-heavy tests run in nightly builds if necessary.

### 6. MCP Tool Integration Rollout

#### 6.1 Payload Schema & Documentation
- Finalize a canonical MCP request/response schema (request metadata, auth envelope, tool input payload, standardized status fields).
- Capture schema in OpenAPI + JSON Schema artifacts stored in `docs/mcp/schemas/` with version tags and changelog.
- Update developer docs with sample requests per tool category and validation guidance for orchestrator contributors.

#### 6.2 Shared MCP Client Utility
- Build `app/services/mcp_client.py` exposing async helpers for auth token injection, request signing, and retryable POST/GET helpers.
- Centralize timeout configuration, exponential backoff, and circuit breaker logic with Prometheus counters for each outcome.
- Provide instrumentation hooks (structured logging, trace ids) consumable by the observability stack.

#### 6.3 Category Onboarding Plan
- Sprint 4A: enable Research tooling (Tavily, ArXiv, Wikipedia, document loader, Qdrant retriever, summarizer).
- Sprint 4B: enable Finance tooling (yfinance, Pandas analytics, plotting, NewsAPI, CSV analyzer, FinBERT).
- Post-Sprint 4B extension: enable Creative and Enterprise tool suites once core agents are stable.
- Maintain onboarding checklist per tool covering secrets, rate limits, and smoke tests before registration.

#### 6.4 Tool Wrapping & Registration
- `ToolService._invoke_enterprise_playbook` now composes the Notion connector with the policy checker, producing normalized `actions` payloads, caching responses, and emitting unified metrics.
- Default alias table aligns with the MCP catalog (`enterprise.playbook` resolved through the composite wrapper) and onboarding diagnostics reflect catalog coverage per category.
- Regression coverage in `tests/test_tool_service.py::test_enterprise_playbook_composite` validates mapping logic, cache behaviour, and structured output shape.

#### 6.5 Telemetry & Health Monitoring
- Prometheus counters/histograms (`neuraforge_tool_*`, `neuraforge_mcp_*`) are exercised on every invocation, including composite wrappers, capturing latency, cache hits, and error totals.
- `ToolService.instrument` feeds structured telemetry into the SSE stream (`tool_invocation` events) so dashboards can observe per-task tool usage and error causes alongside agent lifecycle updates.
- `/diagnostics/mcp` surface remains wired into the SSE bootstrap to broadcast the latest health snapshot before orchestration begins, enabling clients to grey-out unhealthy tools.

#### 6.6 Agent Playbook Updates
- `EnterpriseAgent` now consumes the composite playbook wrapper, yielding actionable steps from Notion search results with policy mitigation fallbacks when knowledge base data is unavailable.
- Fallback ordering is enforced by the wrapper (Notion → Policy Checker), guaranteeing degraded but actionable outputs even when the primary knowledge source is offline.
- Updated documentation in `docs/agents/README.md` calls out the available MCP tools per capability, reflecting the new enterprise playbook integration.

## Documentation & Developer Experience
- `docs/agents/README.md` now mirrors the live tool registry (Tavily search, MCP finance stack, composite enterprise playbook) so developers wire agents with the correct catalog identifiers.
- `docs/architecture.md` captures the updated telemetry flow (SSE tool events, Prometheus counters) and references the refreshed Mermaid diagrams for Phase 4.
- `docs/notebooks/phase4_playbook_demo.ipynb` provides a runnable multi-agent walkthrough using the new orchestration, giving developers a living sample they can adapt for demos or validation.

## Timeline (Indicative, 2 Sprints)
- **Sprint 4A**: Finalize contracts, implement Research & Finance agents with tool wrappers, establish confidence scoring scaffolds.
- **Sprint 4B**: Deliver Creative & Enterprise agents, expand testing, publish documentation, and integrate metrics.

## Dependencies & Risks
- **External API Limits**: DuckDuckGo and financial APIs may enforce throttling—ensure caching and rate limiting are in place.
- **Tool Availability**: Transformer-based stylizer requires model hosting or fine-tuned weights; secure deployment path early.
- **Security & Compliance**: Finance outputs must align with regulatory guidance; incorporate validation rules before launch.
- **MCP Compatibility**: MCP tool catalog must stay synchronized with agent expectations; coordinate with platform team to maintain schemas and auth scopes.
- **Orchestrator Integration**: Coordinate with Phase 5 planning to ensure agent contracts align with negotiation logic.

## Exit Checklist
- [ ] Four LangGraph agent nodes implemented with documented contracts.
- [ ] Tool wrappers hardened with retries, rate limits, and telemetry.
- [ ] Confidence scoring framework emitting metrics and structured outputs.
- [ ] Comprehensive test suite (unit + integration) covering agent happy paths and failure handling.
- [ ] Documentation and onboarding materials updated; demo notebook recorded.
