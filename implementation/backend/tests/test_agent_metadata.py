from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent, get_agent_schema
from app.agents.creative import CreativeAgent
from app.agents.enterprise import EnterpriseAgent
from app.agents.finance import FinanceAgent
from app.agents.general import GeneralistAgent
from app.agents.research import ResearchAgent
from app.core.config import get_settings
from app.orchestration.llm_planner import LLMOrchestrationPlanner


class _StubLLM:
    async def generate(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - defensive stub
        raise AssertionError("Planner prompt should be built without invoking the LLM")


def _all_agents() -> list[BaseAgent]:
    return [
        GeneralistAgent(),
        ResearchAgent(),
        FinanceAgent(),
        CreativeAgent(),
        EnterpriseAgent(),
    ]


def test_agent_schema_contains_metadata() -> None:
    agents = _all_agents()
    schema = get_agent_schema(agents)

    assert {entry["name"] for entry in schema} == {agent.name for agent in agents}
    for entry in schema:
        assert isinstance(entry["description"], str) and entry["description"].strip()
        assert isinstance(entry["tools"], list)
        assert "confidence_bias" in entry
        assert "fallback_agent" in entry


def test_planner_prompt_includes_agent_metadata_section() -> None:
    settings = get_settings({"environment": "test"})
    planner = LLMOrchestrationPlanner(settings, llm_service=_StubLLM())
    agents = _all_agents()

    prompt = planner._build_prompt(
        task={"prompt": "Provide a financial update", "metadata": {"topic": "earnings"}},
        prior_outputs=[],
        agents=agents,
        tool_aliases=None,
    )

    assert "Agent Metadata" in prompt
    for agent in agents:
        assert agent.name in prompt
        assert agent.description in prompt
    assert "confidence_bias" in prompt
    assert "fallback" in prompt
    assert "\"confidence\"" in prompt
    assert "\"steps\"" in prompt
    assert "\"metadata\"" in prompt
