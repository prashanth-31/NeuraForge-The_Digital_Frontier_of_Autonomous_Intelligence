from __future__ import annotations

from typing import Any

import pytest

from app.agents.base import AgentContext
from app.orchestration.graph import Orchestrator, ToolFirstPolicyViolation
from app.orchestration.llm_planner import PlannerPlan, PlannedAgentStep
from app.orchestration.routing import DynamicAgentRouter
from app.schemas.agents import AgentCapability, AgentOutput


class _StubLLM:
    async def generate(self, *args, **kwargs):  # pragma: no cover - simple stub
        return "ok"


class _StubMemory:
    async def store_working_memory(self, *args, **kwargs):  # pragma: no cover - simple stub
        return None


class _StubAgent:
    def __init__(self, name: str, capability: AgentCapability) -> None:
        self.name = name
        self.capability = capability
        self.calls = 0
        self.description = f"Stub agent for {name}"
        self.tool_preference: list[str] = []
        self.fallback_agent: str | None = None
        self.confidence_bias = 0.5

    async def handle(self, task, *, context):  # pragma: no cover - exercised in tests
        self.calls += 1
        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=f"{self.name} handled {task.prompt}",
            confidence=0.8,
            rationale="",
        )


class _PlannerStub:
    async def plan(self, *, task, prior_outputs, agents, tool_aliases=None):
        return PlannerPlan(
            steps=[
                PlannedAgentStep(
                    agent="creative_agent",
                    tools=["creative.write"],
                    fallback_tools=["search.web"],
                    reason="Creative agent best matches user request",
                )
            ],
            raw_response="{\"steps\": []}",
            metadata={"handoff_strategy": "sequential"},
        )


class _EmptyPlannerStub:
    async def plan(self, *, task, prior_outputs, agents, tool_aliases=None):
        return PlannerPlan(
            steps=[],
            raw_response="{\"steps\": []}",
            metadata={"handoff_strategy": "sequential"},
        )


class _ErrorPlannerStub:
    async def plan(self, *, task, prior_outputs, agents, tool_aliases=None):
        raise RuntimeError("planner unavailable")


class _SequentialPlannerStub:
    async def plan(self, *, task, prior_outputs, agents, tool_aliases=None):
        return PlannerPlan(
            steps=[
                PlannedAgentStep(
                    agent=agent.name,
                    tools=[],
                    fallback_tools=[],
                    reason="sequential fallback",
                )
                for agent in agents
            ],
            raw_response="{\"steps\": []}",
            metadata={"strategy": "sequential_fallback"},
        )


class _ConfidencePlannerStub:
    def __init__(self, confidence: float) -> None:
        self._confidence = confidence

    async def plan(self, *, task, prior_outputs, agents, tool_aliases=None):
        return PlannerPlan(
            steps=[
                PlannedAgentStep(
                    agent="creative_agent",
                    tools=["creative.write"],
                    fallback_tools=["search.web"],
                    reason="Creative agent best matches user request",
                )
            ],
            raw_response="{\"confidence\": %.2f}" % self._confidence,
            metadata={"confidence": self._confidence, "handoff_strategy": "sequential"},
        )


class _ToolResult:
    def __init__(self, tool: str) -> None:
        self.resolved_tool = tool
        self.cached = False
        self.latency = 0.0


class _ToolServiceStub:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(self, tool: str, payload: dict[str, Any]) -> _ToolResult:
        self.calls.append(tool)
        return _ToolResult(tool)


class _ToolUsingAgent:
    def __init__(self, name: str, capability: AgentCapability, tool_to_use: str) -> None:
        self.name = name
        self.capability = capability
        self._tool_to_use = tool_to_use
        self.description = f"Tool-using stub for {name}"
        self.tool_preference = [tool_to_use]
        self.fallback_agent = None
        self.confidence_bias = 0.6

    async def handle(self, task, *, context: AgentContext):
        await context.tools.invoke(self._tool_to_use, {"prompt": task.prompt})  # type: ignore[union-attr]
        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=f"{self.name} handled {task.prompt}",
            confidence=0.9,
            rationale="uses tool",
        )


@pytest.fixture
def dynamic_router(monkeypatch: pytest.MonkeyPatch) -> DynamicAgentRouter:
    monkeypatch.setattr(DynamicAgentRouter, "_ensure_model", lambda self: None)
    return DynamicAgentRouter()


@pytest.mark.asyncio
async def test_dynamic_router_prefers_creative_for_greeting(dynamic_router: DynamicAgentRouter) -> None:
    router = dynamic_router
    agents = [
        _StubAgent("general_agent", AgentCapability.GENERAL),
        _StubAgent("research_agent", AgentCapability.RESEARCH),
        _StubAgent("finance_agent", AgentCapability.FINANCE),
        _StubAgent("creative_agent", AgentCapability.CREATIVE),
    ]
    decision = await router.select(task={"prompt": "Hi"}, agents=agents)
    assert decision.names == ["creative_agent"]
    assert decision.reason == "dynamic.greeting"


@pytest.mark.asyncio
async def test_dynamic_router_scores_finance_keywords(dynamic_router: DynamicAgentRouter) -> None:
    router = dynamic_router
    agents = [
        _StubAgent("general_agent", AgentCapability.GENERAL),
        _StubAgent("research_agent", AgentCapability.RESEARCH),
        _StubAgent("finance_agent", AgentCapability.FINANCE),
        _StubAgent("enterprise_agent", AgentCapability.ENTERPRISE),
    ]
    prompt = "Create a revenue forecast and budget outlook"
    decision = await router.select(task={"prompt": prompt}, agents=agents)
    assert "finance_agent" in decision.names
    assert decision.reason == "dynamic.embedding_routing"


@pytest.mark.asyncio
async def test_orchestrator_runs_all_agents_when_planner_disabled() -> None:
    general = _StubAgent("general_agent", AgentCapability.GENERAL)
    research = _StubAgent("research_agent", AgentCapability.RESEARCH)
    creative = _StubAgent("creative_agent", AgentCapability.CREATIVE)
    planner = _SequentialPlannerStub()
    orchestrator = Orchestrator(agents=[general, research, creative], orchestration_planner=planner)
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM())
    task = {"id": "task-routing", "prompt": "Hello there", "metadata": {}}

    result = await orchestrator.route_task(task, context=context)

    assert general.calls == 1
    assert creative.calls == 1
    assert research.calls == 1
    assert result["routing"]["selected_agents"] == ["general_agent", "research_agent", "creative_agent"]
    assert result["routing"]["reason"] == "planner.llm_selected"
    planner_metadata = result["routing"]["metadata"]["planner"]
    assert planner_metadata["status"] == "planned"


@pytest.mark.asyncio
async def test_orchestrator_handles_empty_planner_response() -> None:
    general = _StubAgent("general_agent", AgentCapability.GENERAL)
    research = _StubAgent("research_agent", AgentCapability.RESEARCH)
    creative = _StubAgent("creative_agent", AgentCapability.CREATIVE)
    planner = _EmptyPlannerStub()
    orchestrator = Orchestrator(agents=[general, research, creative], orchestration_planner=planner)
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM())

    result = await orchestrator.route_task(
        {"id": "task-empty", "prompt": "Hello there", "metadata": {}},
        context=context,
    )

    assert general.calls == 0
    assert creative.calls == 0
    assert research.calls == 0
    assert result["status"] == "failed"
    assert result["routing"]["reason"] == "planner.empty_plan"
    planner_metadata = result["routing"]["metadata"]["planner"]
    assert planner_metadata["status"] == "failed"
    assert planner_metadata["raw_response"] == "{\"steps\": []}"


@pytest.mark.asyncio
async def test_orchestrator_records_planner_failure() -> None:
    general = _StubAgent("general_agent", AgentCapability.GENERAL)
    research = _StubAgent("research_agent", AgentCapability.RESEARCH)
    creative = _StubAgent("creative_agent", AgentCapability.CREATIVE)
    planner = _ErrorPlannerStub()
    orchestrator = Orchestrator(agents=[general, research, creative], orchestration_planner=planner)
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM())

    result = await orchestrator.route_task(
        {"id": "task-error", "prompt": "Investigate", "metadata": {}},
        context=context,
    )

    assert general.calls == 0
    assert creative.calls == 0
    assert research.calls == 0
    assert result["status"] == "failed"
    assert result["routing"]["reason"] == "planner.failed"
    planner_metadata = result["routing"]["metadata"]["planner"]
    assert planner_metadata["status"] == "failed"
    assert "error" in planner_metadata


@pytest.mark.asyncio
async def test_orchestrator_prefers_llm_planner_when_available() -> None:
    general = _StubAgent("general_agent", AgentCapability.GENERAL)
    research = _StubAgent("research_agent", AgentCapability.RESEARCH)
    creative = _StubAgent("creative_agent", AgentCapability.CREATIVE)
    planner = _PlannerStub()
    orchestrator = Orchestrator(
        agents=[general, research, creative],
        orchestration_planner=planner,
    )
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM())

    result = await orchestrator.route_task(
        {"id": "task-plan", "prompt": "Draft a launch story", "metadata": {}},
        context=context,
    )

    assert creative.calls == 1
    assert general.calls == 0
    assert research.calls == 0
    assert result["routing"]["reason"] == "planner.llm_selected"
    planner_metadata = result["routing"]["metadata"]["planner"]
    assert planner_metadata["strategy"] == "llm_orchestration"
    assert planner_metadata["steps"][0]["agent"] == "creative_agent"


@pytest.mark.asyncio
async def test_planner_tracks_primary_tool_usage() -> None:
    tool_service = _ToolServiceStub()
    agent = _ToolUsingAgent("creative_agent", AgentCapability.CREATIVE, tool_to_use="creative.write")
    planner = _PlannerStub()
    orchestrator = Orchestrator(agents=[agent], orchestration_planner=planner)
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM(), tools=tool_service)

    result = await orchestrator.route_task(
        {"id": "task-plan", "prompt": "Draft copy", "metadata": {}},
        context=context,
    )

    assert tool_service.calls == ["creative.write"]
    output_metadata = result["outputs"][-1]["metadata"]
    planner_tools = output_metadata["planner_tools"]
    assert planner_tools["classification"] == "primary"
    assert planner_tools["planned"] == ["creative.write"]


@pytest.mark.asyncio
async def test_planner_raises_when_planned_tools_unused() -> None:
    tool_service = _ToolServiceStub()
    agent = _ToolUsingAgent("creative_agent", AgentCapability.CREATIVE, tool_to_use="creative.image")
    planner = _PlannerStub()
    orchestrator = Orchestrator(agents=[agent], orchestration_planner=planner)
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM(), tools=tool_service)

    with pytest.raises(ToolFirstPolicyViolation):
        await orchestrator.route_task(
            {"id": "task-plan", "prompt": "Draft copy", "metadata": {}},
            context=context,
        )


@pytest.mark.asyncio
async def test_orchestrator_falls_back_on_low_confidence() -> None:
    general = _StubAgent("general_agent", AgentCapability.GENERAL)
    creative = _StubAgent("creative_agent", AgentCapability.CREATIVE)
    planner = _ConfidencePlannerStub(confidence=0.5)
    orchestrator = Orchestrator(agents=[general, creative], orchestration_planner=planner)
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM())

    result = await orchestrator.route_task(
        {"id": "task-low-confidence", "prompt": "Draft copy", "metadata": {}},
        context=context,
    )

    assert general.calls == 1
    assert creative.calls == 0
    routing = result["routing"]
    assert routing["selected_agents"] == ["general_agent"]
    assert routing["reason"] == "planner.low_confidence"
    planner_metadata = routing["metadata"]["planner"]
    assert planner_metadata["status"] == "fallback"
    assert planner_metadata["attributes"]["confidence"] == 0.5
    assert planner_metadata["attributes"]["fallback_reason"] == "low_confidence"


@pytest.mark.asyncio
async def test_orchestrator_respects_high_confidence_plan() -> None:
    general = _StubAgent("general_agent", AgentCapability.GENERAL)
    creative = _StubAgent("creative_agent", AgentCapability.CREATIVE)
    planner = _ConfidencePlannerStub(confidence=0.9)
    orchestrator = Orchestrator(agents=[general, creative], orchestration_planner=planner)
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM())

    result = await orchestrator.route_task(
        {"id": "task-high-confidence", "prompt": "Draft copy", "metadata": {}},
        context=context,
    )

    assert creative.calls == 1
    assert general.calls == 0
    routing = result["routing"]
    assert routing["selected_agents"] == ["creative_agent"]
    assert routing["reason"] == "planner.llm_selected"
    planner_metadata = routing["metadata"]["planner"]
    assert planner_metadata["status"] == "planned"
    assert planner_metadata["attributes"]["confidence"] == 0.9