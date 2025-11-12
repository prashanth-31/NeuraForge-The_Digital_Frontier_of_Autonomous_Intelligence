from __future__ import annotations

import asyncio

import pytest

from app.agents.base import AgentContext
from app.orchestration.graph import Orchestrator
from app.orchestration.llm_planner import PlannerPlan, PlannedAgentStep
from app.orchestration.meta import MetaAgent
from app.schemas.agents import AgentCapability, AgentOutput
from app.core.config import Settings
from app.services.disputes import DisputeDetector, MetaConfidenceScorer
from app.services.llm import LLMService


class _FakeLLMClient:
    async def ainvoke(self, messages):  # pragma: no cover - simple stub
        await asyncio.sleep(0)
        return type("_Message", (), {"content": "Meta-agent synthesized summary."})()


class _DemoAgent:
    name = "demo"
    capability = AgentCapability.RESEARCH

    async def handle(self, task, *, context):  # pragma: no cover - simple stub
        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=f"Recommendation for {task.prompt}",
            confidence=0.82,
        )


class _PlannerStub:
    async def plan(self, *, task, prior_outputs, agents, tool_aliases=None):  # noqa: D401 - simple stub
        step = PlannedAgentStep(agent=agents[0].name, tools=[], fallback_tools=[], reason="meta test")
        return PlannerPlan(steps=[step], raw_response="{}", metadata={})


@pytest.mark.asyncio
async def test_orchestrator_produces_meta_resolution_and_dossier() -> None:
    settings = Settings(environment="test")
    llm_service = LLMService(settings=settings, model="stub-model", _client=_FakeLLMClient())
    meta_agent = MetaAgent(
        llm_service=llm_service,
        settings=settings.meta_agent,
        dispute_detector=DisputeDetector(
            consensus_delta_threshold=settings.meta_agent.consensus_delta_threshold,
            stddev_threshold=settings.meta_agent.stddev_threshold,
        ),
        confidence_scorer=MetaConfidenceScorer(),
    )
    orchestrator = Orchestrator(
        agents=[_DemoAgent()],
        meta_agent=meta_agent,
        orchestration_planner=_PlannerStub(),
    )

    context = AgentContext(memory=None, llm=llm_service)
    task = {"id": "task-meta", "prompt": "Evaluate revenue trends"}

    result = await orchestrator.route_task(task, context=context)

    assert result["status"] == "completed"
    assert "meta" in result
    assert result["meta"]["summary"].startswith("Meta-agent synthesized summary")
    assert "dossier" in result
    dossier = result["dossier"]
    assert dossier["status"] == "available"
    assert "Meta-agent synthesized summary" in dossier["markdown"]
    assert dossier["json"]["meta_resolution"]["summary"].startswith("Meta-agent synthesized summary")