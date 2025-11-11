from __future__ import annotations

from typing import Any

import pytest

from app.agents.base import AgentContext
from app.orchestration.graph import Orchestrator, ToolFirstPolicyViolation
from app.orchestration.llm_planner import PlannerExecutionPlan, PlannedAgentStep
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
        return PlannerExecutionPlan(
            steps=[
                PlannedAgentStep(
                    agent="creative_agent",
                    tools=["creative.write"],
                    fallback_tools=["search.web"],
                    reason="Creative agent best matches user request",
                )
            ],
            raw_response="{\"agents\": []}",
            metadata={"handoff_strategy": "sequential"},
        )


class _EmptyPlannerStub:
    async def plan(self, *, task, prior_outputs, agents, tool_aliases=None):
        return PlannerExecutionPlan(
            steps=[],
            raw_response="{}",
            metadata={"handoff_strategy": "sequential"},
        )


class _ErrorPlannerStub:
    async def plan(self, *, task, prior_outputs, agents, tool_aliases=None):
        raise RuntimeError("planner unavailable")


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
    orchestrator = Orchestrator(agents=[general, research, creative])
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM())
    task = {"id": "task-routing", "prompt": "Hello there", "metadata": {}}

    result = await orchestrator.route_task(task, context=context)

    assert general.calls == 1
    assert creative.calls == 1
    assert research.calls == 1
    assert result["routing"]["selected_agents"] == ["general_agent", "research_agent", "creative_agent"]
    assert result["routing"]["reason"] == "planner.fallback_all"
    planner_metadata = result["routing"]["metadata"]["planner"]
    assert planner_metadata["status"] == "disabled"


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

    assert general.calls == 1
    assert creative.calls == 1
    assert research.calls == 1
    assert result["routing"]["reason"] == "planner.fallback_all"
    planner_metadata = result["routing"]["metadata"]["planner"]
    assert planner_metadata["status"] == "empty"
    assert planner_metadata["raw_response"] == "{}"


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

    assert general.calls == 1
    assert creative.calls == 1
    assert research.calls == 1
    assert result["routing"]["reason"] == "planner.fallback_all"
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