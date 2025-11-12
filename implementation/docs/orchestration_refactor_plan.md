# NeuraForge Orchestration Refactor Plan

## Phase 1 — Agent Metadata & Prompt Grounding

**Goal**
Add rich agent metadata (description, tool preferences, fallback agent, confidence bias) so the planner better understands each role. No planner schema or orchestrator changes yet.

**Scope**
- Base class: `backend/app/agents/base.py`
- Concrete agents: `backend/app/agents/{creative,research,finance,enterprise,general}.py`
- Planner prompt builder: `backend/app/orchestration/llm_planner.py`
- Do **not** touch planner schemas, orchestrator logic, or negotiation components.

**Tasks**
1. Extend `BaseAgent` with optional attributes:
   - `description: str = ""`
   - `tool_preference: list[str] = []`
   - `fallback_agent: str | None = None`
   - `confidence_bias: float = 1.0`
2. Populate these attributes for each concrete agent.
   - creative_agent → "Creates expressive, poetic, or stylistic content."; tools `["creative.tonecheck"]`; fallback `"general_agent"`; bias `0.8`
   - research_agent → "Finds factual, evidence-backed information."; tools `["research.search", "research.summarizer"]`; fallback `"enterprise_agent"`; bias `0.9`
   - finance_agent → "Performs financial analysis and forecasting."; tools `["finance.snapshot"]`; fallback `"enterprise_agent"`; bias `0.9`
   - enterprise_agent → "Creates workflows, strategies, and executive documentation."; tools `["enterprise.playbook"]`; fallback `"research_agent"`; bias `0.85`
   - general_agent → "Handles greetings, casual queries, or triage."; tools `[]`; fallback `None`; bias `0.6`
3. Add helper to `base.py`:
   ```python
   def get_agent_schema(agents: list[BaseAgent]) -> list[dict[str, Any]]:
       return [
           {
               "name": agent.name,
               "description": agent.description,
               "tools": agent.tool_preference,
               "fallback_agent": agent.fallback_agent,
               "confidence_bias": agent.confidence_bias,
           }
           for agent in agents
       ]
   ```
4. Update LLM planner prompt construction to append a section listing agent schema via `get_agent_schema()`.
5. Add `tests/test_agent_metadata.py` to assert:
   - Every agent exposes a non-empty description and the expected fallback agent.
   - The planner prompt includes all agent names.

---

## Phase 2 — Confidence Gating & Fallback Logic

**Goal**
Introduce planner confidence metadata and orchestrator fallback to `general_agent` when confidence is low. Maintain existing JSON contract.

**Tasks**
1. Extend the `PlannerPlan` dataclass so it always carries `metadata: dict[str, Any] = field(default_factory=dict)` and a top-level `confidence: float = 1.0`.
2. After parsing the LLM response, extract `confidence = parsed.get("confidence", 1.0)` and persist it on both `plan.metadata["confidence"]` and `plan.confidence`.
3. In `orchestrator._determine_agent_sequence()` inspect confidence before selecting agents:
   ```python
   confidence = planner_plan.metadata.get("confidence", 1.0)
   if confidence < 0.7:
       logger.warning("Low planner confidence, routing to general_agent")
       return [a for a in roster if a.name == "general_agent"], {}, roster
   ```
4. Enhance logging to capture confidence and fallback events.
5. Add unit tests covering low-confidence fallback (0.5) and high-confidence default routing (0.9).
6. Do not alter planner schemas.

---

## Phase 3 — Schema Migration (Dual Support)

**Goal**
Support both legacy `"agents"` payloads and new `{"steps": [...], "metadata": {...}}` plans without breaking existing flows.

**Tasks**
1. Introduce strict payload validators:
   ```python
   class PlannerStepPayload(BaseModel):
       agent: str = Field(..., min_length=1)
       reason: str = ""
       tools: list[str] = Field(default_factory=list)
       fallback_tools: list[str] = Field(default_factory=list)
       confidence: float = Field(default=1.0)

   class PlannerPlanPayload(BaseModel):
       steps: list[PlannerStepPayload] = Field(default_factory=list)
       metadata: dict[str, Any] = Field(default_factory=dict)
       confidence: float = Field(default=1.0)
   ```
2. Replace `_normalize_plan_payload()` with `_parse_steps_payload()` so the planner first attempts to validate the `steps` structure but, during the migration window, can still coerce legacy `{"agents": [...]}` payloads into `PlannerPlanPayload` instances.
3. When the LLM emits legacy JSON, build a single-step plan that preserves the legacy note as the reason field and stamps `metadata["schema_version"] = "legacy"` for observability.
4. Adjust planner prompt to strongly prefer `steps + metadata` while documenting (and testing) the transitional support path.
5. Mirror the parsed payload into the `PlannerPlan` dataclass—ensuring `raw_response` is preserved—and propagate schema metadata so the orchestrator can log usage of each contract version.
6. Add tests for new schema parsing and the temporary legacy compatibility path.

---

## Phase 4 — Orchestrator Integration

**Goal**
Execute multi-step plans directly, enforce per-step tool policies, and apply per-step confidence gating.

**Tasks**
1. In `_determine_agent_sequence()`:
   - Detect `plan.steps`.
   - Map roster agents by name.
   - Build the selection list from step agents and store `self._active_plan_steps` for tool enforcement.
   - If no valid steps survive parsing, request the planner fallback builder to synthesize a single-step `general_agent` plan rather than invoking the legacy router.
2. In sequential execution, iterate over `plan.steps` and apply per-step confidence (<0.7 → `general_agent`).
3. Update `_enforce_tool_first_policy()` to reference tools/fallbacks from `self._active_plan_steps`.
4. Ensure guardrails, scheduler, and review capture per-step metadata.
5. Add integration tests for multi-step execution, tool enforcement, confidence fallback, and fallback-plan behavior.
6. Log each executed step with agent, reason, and tools.

---

## Phase 5 — Cleanup & Deprecation

**Goal**
Remove legacy `"agents"` handling once the multi-step path is stable.

**Tasks**
1. Lock `_parse_steps_payload()` so it rejects payloads that omit `"steps"`, ending the migration bridge.
2. Remove the deprecated `PlannerExecutionPlan` model and any legacy prompt helpers.
3. Delete orchestrator conditionals (and associated telemetry branches) that looked for legacy schemas.
4. Update the planner system prompt and examples to demand steps-only output and explicitly forbid legacy `"agents"` arrays.
5. Run the full regression suites (backend pytest, targeted planner/orchestrator suites) to confirm stability.
6. Refresh documentation, comments, and versioning references to describe the steps-only planner contract.
7. Audit CI/CD scripts and deployment tooling to ensure they assume `steps`-based plans exclusively.

**Status**
Completed November 2025. Planner responses are validated exclusively via `PlannerPlanPayload`, the orchestrator executes steps-only `PlannerPlan` objects, and regression coverage plus CI/CD workflows now assume the v2 schema.
