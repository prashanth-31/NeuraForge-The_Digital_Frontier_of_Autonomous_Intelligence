from __future__ import annotations

from typing import Any

import pytest

from app.core.config import get_settings
from app.orchestration.llm_planner import LLMOrchestrationPlanner, PlannedAgentStep
from app.tools.registry import tool_registry


class _StubLLM:
    async def generate(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - defensive stub
        raise AssertionError("Planner prompt should not invoke the LLM in tool whitelist tests")


@pytest.fixture()
def planner() -> LLMOrchestrationPlanner:
    settings = get_settings({"environment": "test"})
    return LLMOrchestrationPlanner(settings, llm_service=_StubLLM())


@pytest.fixture(autouse=True)
def _patch_cold_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force planner tool resolution to behave as if no tools were registered yet."""

    monkeypatch.setattr(tool_registry, "list", lambda: [])
    monkeypatch.setattr(tool_registry, "aliases", lambda: {})


def test_allowed_tool_names_include_defaults_when_registry_empty(planner: LLMOrchestrationPlanner) -> None:
    allowed = planner._allowed_tool_names(tool_aliases=None)

    assert "finance.snapshot" in allowed
    assert "finance/yfinance" in allowed
    assert "finance.snapshot.cached" in allowed
    assert "finance/cached_quotes" in allowed
    assert "research.search" in allowed
    assert "creative.tonecheck" in allowed
    assert "enterprise.playbook" in allowed


def test_post_process_retains_alias_tools_without_registry(planner: LLMOrchestrationPlanner) -> None:
    steps = [
        PlannedAgentStep(
            agent="finance_agent",
            tools=["finance.snapshot"],
            fallback_tools=["finance.snapshot.alpha", "finance.snapshot.cached", "finance.news"],
            reason="heuristic",
        ),
        PlannedAgentStep(
            agent="research_agent",
            tools=["research.search"],
            fallback_tools=["research.doc_loader"],
            reason="heuristic",
        ),
    ]

    processed, _ = planner._post_process_steps(
        "Need finance and research support",
        steps,
        tool_aliases=None,
    )

    finance_step = next(step for step in processed if step.agent == "finance_agent")
    research_step = next(step for step in processed if step.agent == "research_agent")

    assert "finance.snapshot" in finance_step.tools
    assert "finance.snapshot.alpha" in finance_step.fallback_tools
    assert "finance.snapshot.cached" in finance_step.fallback_tools
    assert "finance.news" in finance_step.fallback_tools
    assert "research.search" in research_step.tools
    assert "research.doc_loader" in research_step.fallback_tools
