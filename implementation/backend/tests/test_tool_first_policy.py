from __future__ import annotations

import pytest

from app.agents.base import AgentContext
from app.orchestration.graph import Orchestrator, ToolFirstPolicyViolation
from app.orchestration.llm_planner import PlannerPlan, PlannedAgentStep
from app.schemas.agents import AgentCapability, AgentOutput
from app.services.tools import ToolInvocationResult


class _StubLLM:
    async def generate(self, *args, **kwargs):  # pragma: no cover - simple stub
        return "summary"


class _StubMemory:
    async def store_working_memory(self, *args, **kwargs):  # pragma: no cover - simple stub
        return None


class _StubToolService:
    def __init__(self, *, succeed: bool = True) -> None:
        self._succeed = succeed
        self.invocations: list[tuple[str, dict[str, object]]] = []

    async def invoke(self, tool: str, payload: dict[str, object]) -> ToolInvocationResult:
        self.invocations.append((tool, payload))
        if not self._succeed:
            raise RuntimeError("tool failure")
        return ToolInvocationResult(
            tool=tool,
            payload=payload,
            response={"ok": True},
            cached=False,
            latency=0.05,
            resolved_tool=tool,
        )


class _StubAgent:
    def __init__(self, *, use_tool: bool, attach_metadata: bool = False) -> None:
        self.name = "stub_agent"
        self.capability = AgentCapability.RESEARCH
        self._use_tool = use_tool
        self._attach_metadata = attach_metadata
        self.description = "Stub research agent"
        self.tool_preference = ["research.search"] if use_tool else []
        self.fallback_agent = None
        self.confidence_bias = 0.6

    async def handle(self, task, *, context: AgentContext) -> AgentOutput:
        metadata: dict[str, object] = {}
        if self._use_tool and context.tools is not None:
            result = await context.tools.invoke("research.search", {"query": "example"})
            if self._attach_metadata:
                metadata["tool"] = {
                    "name": result.tool,
                    "resolved": result.resolved_tool,
                }
        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary="Handled",
            confidence=0.6,
            rationale="",
            metadata=metadata,
        )


class _PlannerStub:
    async def plan(self, *, task, prior_outputs, agents, tool_aliases=None):  # noqa: D401
        step = PlannedAgentStep(
            agent=agents[0].name,
            tools=["research.search"],
            fallback_tools=[],
            reason="tool policy test",
        )
        return PlannerPlan(steps=[step], raw_response="{}", metadata={})


@pytest.mark.asyncio
async def test_tool_first_policy_raises_for_missing_invocation() -> None:
    orchestrator = Orchestrator(agents=[_StubAgent(use_tool=False)], orchestration_planner=_PlannerStub())
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM(), tools=_StubToolService())
    task = {"id": "policy-missing", "prompt": "Collect data"}

    with pytest.raises(ToolFirstPolicyViolation):
        await orchestrator.route_task(task, context=context)


@pytest.mark.asyncio
async def test_tool_first_policy_allows_when_tools_disabled() -> None:
    orchestrator = Orchestrator(agents=[_StubAgent(use_tool=False)], orchestration_planner=_PlannerStub())
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM(), tools=None)
    task = {"id": "policy-disabled", "prompt": "Collect data"}

    result = await orchestrator.route_task(task, context=context)

    assert result["status"] == "completed"
    assert result["outputs"][0]["summary"] == "Handled"


@pytest.mark.asyncio
async def test_tool_first_policy_enriches_metadata() -> None:
    tool_service = _StubToolService()
    orchestrator = Orchestrator(agents=[_StubAgent(use_tool=True)], orchestration_planner=_PlannerStub())
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM(), tools=tool_service)
    task = {"id": "policy-metadata", "prompt": "Collect data"}

    result = await orchestrator.route_task(task, context=context)

    output = result["outputs"][0]
    metadata = output.get("metadata") or {}
    tools_used = metadata.get("tools_used")
    assert tools_used, "Expected orchestrator to record tools_used metadata"
    assert tools_used[0]["tool"] == "research.search"
    assert tools_used[0]["resolved"] == "research.search"
