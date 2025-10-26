from __future__ import annotations

import pytest

from app.agents.base import AgentContext
from app.orchestration.graph import Orchestrator
from app.orchestration.routing import DynamicAgentRouter, RoutingDecision
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


class _StaticRouter:
    def __init__(self, names: list[str]) -> None:
        self._names = {name.lower() for name in names}

    async def select(self, *, task, agents):
        selected = [agent for agent in agents if agent.name.lower() in self._names]
        scores = {agent.name: (1.0 if agent in selected else 0.0) for agent in agents}
        return RoutingDecision(agents=selected, scores=scores, reason="test.static")


class _NameRouter:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    async def select(self, *, task, agents):
        scores = {agent.name: (1.0 if agent.name in self._names else 0.0) for agent in agents}
        # Intentionally return agent names to ensure orchestrator normalizes properly.
        return RoutingDecision(agents=list(self._names), scores=scores, reason="test.names")


@pytest.fixture
def dynamic_router(monkeypatch: pytest.MonkeyPatch) -> DynamicAgentRouter:
    monkeypatch.setattr(DynamicAgentRouter, "_ensure_model", lambda self: None)
    return DynamicAgentRouter()


@pytest.mark.asyncio
async def test_dynamic_router_prefers_creative_for_greeting(dynamic_router: DynamicAgentRouter) -> None:
    router = dynamic_router
    agents = [
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
        _StubAgent("research_agent", AgentCapability.RESEARCH),
        _StubAgent("finance_agent", AgentCapability.FINANCE),
        _StubAgent("enterprise_agent", AgentCapability.ENTERPRISE),
    ]
    prompt = "Create a revenue forecast and budget outlook"
    decision = await router.select(task={"prompt": prompt}, agents=agents)
    assert "finance_agent" in decision.names
    assert decision.reason == "dynamic.embedding_routing"


@pytest.mark.asyncio
async def test_orchestrator_runs_only_selected_agents() -> None:
    research = _StubAgent("research_agent", AgentCapability.RESEARCH)
    creative = _StubAgent("creative_agent", AgentCapability.CREATIVE)
    orchestrator = Orchestrator(agents=[research, creative], router=_StaticRouter(["creative_agent"]))
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM())
    task = {"id": "task-routing", "prompt": "Hello there", "metadata": {}}

    result = await orchestrator.route_task(task, context=context)

    assert creative.calls == 1
    assert research.calls == 0
    assert result["routing"]["selected_agents"] == ["creative_agent"]
    assert result["routing"]["skipped_agents"] == ["research_agent"]


@pytest.mark.asyncio
async def test_orchestrator_accepts_router_name_selection() -> None:
    research = _StubAgent("research_agent", AgentCapability.RESEARCH)
    creative = _StubAgent("creative_agent", AgentCapability.CREATIVE)
    enterprise = _StubAgent("enterprise_agent", AgentCapability.ENTERPRISE)
    orchestrator = Orchestrator(
        agents=[research, creative, enterprise],
        router=_NameRouter(["creative_agent"]),
    )
    context = AgentContext(memory=_StubMemory(), llm=_StubLLM())

    result = await orchestrator.route_task({"id": "task-name", "prompt": "Hi"}, context=context)

    assert creative.calls == 1
    assert research.calls == 0
    assert enterprise.calls == 0
    assert result["routing"]["selected_agents"] == ["creative_agent"]
    assert set(result["routing"]["skipped_agents"]) == {"research_agent", "enterprise_agent"}