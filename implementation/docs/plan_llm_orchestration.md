# LLM-Orchestrated Execution Plan

## Goal
Enable the LLM to determine the complete orchestration plan (agents, order, and tool usage) using `llama3.2:1b` for planning while agents continue using `llama3.1:8b`.

---

## Status Snapshot (Oct 26, 2025)
- ✅ Planner settings, prompt contract, and validation models are live (`LLMOrchestrationPlanner`).
- ✅ Orchestrator now prefers planner output and falls back to legacy routing only on failure.
- ✅ Planner metadata is stored on each task for downstream consumption.
- ✅ Planner-directed tool usage is enforced with primary/fallback tracking.
- ⏳ Observability pipeline (metrics/dossiers) still needs richer coverage.
- ⏳ Broader regression tests and documentation updates remain outstanding.

---

## Phased To-Do

### 1. Assess Current Pipeline
- [x] Review existing orchestrator flow (DynamicAgentRouter + LangGraph) and identify replace points.
- [x] Confirm LLM service configuration split (planner `llama3.2:1b`, agents `llama3.1:8b`).
- [x] Enumerate available agents, capabilities, and tool aliases for planner prompt.

### 2. Design Planner Contract
- [x] Define JSON schema for planner output (agents, tool list, reasoning, fallbacks).
- [x] Draft planner prompt instructions covering capabilities, tool-first requirement, guardrails.
- [x] Create Pydantic models to validate planner responses.

### 3. Planner LLM Service
- [x] Extend `LLMService` to allow planner-specific model overrides (`llama3.2:1b`).
- [x] Add configuration knobs (`PLANNER_LLM__*`).
- [x] Ensure planner can access task context, metadata, prior exchanges.

### 4. Integrate Planner
- [x] Implement `LLMOrchestrationPlanner` in `app/orchestration/llm_planner.py`.
- [x] Hook planner into `Orchestrator.route_task` ahead of legacy routing.
- [x] Convert planner JSON into `PlannedAgentStep` sequence and annotate task state.
- [ ] Purge remaining DynamicAgentRouter dependencies (API docs, unused wiring, tests) once enforcement is complete.

### 5. Enforce Planned Execution (Next Focus)
- [x] Extend `AgentContext` to surface planned tool list/fallbacks per agent.
- [x] Modify `_ToolSession` to track adherence vs. plan and surface deviations.
- [x] Raise policy exceptions or trigger fallback tool chain when required tools fail.
- [ ] Capture planner handoff strategy (sequential vs other) in scheduling logic if expanded beyond sequential.

### 6. Observability & Guardrails
- [x] Record planner output in routing metadata on the task payload.
- [ ] Persist planner plan in decision dossier / snapshot stores for audits.
- [ ] Add metrics: planner latency, adherence rate, deviation events.
- [ ] Log plan validation failures with structured metadata for debugging.

### 7. Testing & Validation
- [ ] Shadow-run planner against historical tasks to measure differences vs. legacy routing.
- [x] Add unit tests for planner selection path (`test_orchestrator_prefers_llm_planner_when_available`).
- [x] Add enforcement-specific tests once tool tracking is implemented.
- [ ] Execute end-to-end flows to ensure tool-first compliance remains intact.

### 8. Documentation
- [ ] Document planner configuration and runtime behavior (README / architecture docs).
- [ ] Update architecture diagrams to reflect LLM-driven orchestration.
- [ ] Provide runbook for tuning planner prompt, temperature, and fallbacks.

---

## Implementation Notes
- Planner prompt describes available agents and tool aliases; constraints enforced (max 4 agents, at least one tool per agent).
- Planner response includes explicit `agents` array with `agent`, `tools`, `reason`, optional `fallback_tools`.
- Enforcement layer records planner-specified fallbacks (e.g., try `finance.snapshot`, then `finance.analytics`) and raises violations when neither primary nor fallback succeeds.
- Keep existing tool-first policy but augment metrics for planned vs. achieved tool invocations.
- Planner currently defaults to sequential handoff; revisit when parallelism or advanced strategies are needed.
- Provide fallback to legacy routing if planner output is invalid or planner call fails (already implemented).
