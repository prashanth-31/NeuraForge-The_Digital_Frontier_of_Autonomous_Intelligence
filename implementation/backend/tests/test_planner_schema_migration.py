from __future__ import annotations

import json

import pytest

from app.agents.base import BaseAgent
from app.core.config import get_settings
from app.orchestration.llm_planner import LLMOrchestrationPlanner
from app.schemas.agents import AgentCapability


class _StubAgent(BaseAgent):
    def __init__(self, name: str, capability: AgentCapability) -> None:
        self.name = name
        self.capability = capability
        self.description = f"Stub agent for {name}"
        self.tool_preference = []
        self.fallback_agent = None
        self.confidence_bias = 0.8
        self.system_prompt = ""

    async def handle(self, task, *, context):  # pragma: no cover - planner-only stub
        raise NotImplementedError


class _V2SchemaLLM:
    async def generate(self, *args, **kwargs):  # pragma: no cover - simple stub
        payload = {
            "steps": [
                {
                    "agent": "general_agent",
                    "reason": "initial triage",
                    "tools": [],
                    "fallback_tools": [],
                    "confidence": 0.9,
                },
                {
                    "agent": "finance_agent",
                    "reason": "financial deep dive",
                    "tools": ["finance.snapshot"],
                    "fallback_tools": [],
                    "confidence": 0.75,
                },
            ],
            "metadata": {"handoff_strategy": "sequential", "notes": "Use finance follow-up"},
            "confidence": 0.72,
        }
        return json.dumps(payload)


class _LegacySchemaLLM:
    async def generate(self, *args, **kwargs):  # pragma: no cover - simple stub
        payload = {
            "agents": [
                {
                    "agent": "general_agent",
                    "tools": [],
                    "fallback_tools": [],
                    "reason": "legacy triage",
                }
            ],
            "handoff_strategy": "sequential",
            "notes": "Legacy schema",
        }
        return json.dumps(payload)


@pytest.mark.asyncio
async def test_planner_supports_steps_schema() -> None:
    settings = get_settings({"environment": "test"})
    planner = LLMOrchestrationPlanner(settings, llm_service=_V2SchemaLLM())
    agents = [
        _StubAgent("general_agent", AgentCapability.GENERAL),
        _StubAgent("finance_agent", AgentCapability.FINANCE),
    ]

    plan = await planner.plan(
        task={"prompt": "Provide an Amazon finance update", "metadata": {}},
        prior_outputs=[],
        agents=agents,
        tool_aliases=None,
    )

    assert [step.agent for step in plan.steps] == ["general_agent", "finance_agent"]
    assert pytest.approx(plan.steps[0].confidence, rel=1e-4) == 0.9
    assert pytest.approx(plan.steps[1].confidence, rel=1e-4) == 0.75
    assert plan.metadata["schema_version"] == "v2"
    assert pytest.approx(plan.metadata["confidence"], rel=1e-4) == 0.72
    assert pytest.approx(plan.confidence, rel=1e-4) == 0.72


@pytest.mark.asyncio
async def test_planner_falls_back_when_legacy_schema_returned() -> None:
    settings = get_settings({"environment": "test"})
    planner = LLMOrchestrationPlanner(settings, llm_service=_LegacySchemaLLM())
    agents = [_StubAgent("general_agent", AgentCapability.GENERAL)]

    plan = await planner.plan(
        task={"prompt": "Say hello", "metadata": {}},
        prior_outputs=[],
        agents=agents,
        tool_aliases=None,
    )

    assert [step.agent for step in plan.steps] == ["general_agent"]
    assert plan.metadata["schema_version"] == "v2"
    assert plan.metadata["fallback_reason"] == "Planner output could not be parsed"
    assert plan.metadata["confidence"] == pytest.approx(0.0)
    assert plan.confidence == pytest.approx(0.0)