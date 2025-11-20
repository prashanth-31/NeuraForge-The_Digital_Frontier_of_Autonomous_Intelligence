from __future__ import annotations

from typing import Any

import pytest

from app.core.config import get_settings
from app.orchestration.llm_planner import LLMOrchestrationPlanner, PlannedAgentStep
from app.tools.registry import tool_registry


class _StubLLM:
    async def generate(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - defensive stub
        raise AssertionError("Planner prompt should not invoke the LLM in tool whitelist tests")


class _AgentStub:
    def __init__(self, name: str) -> None:
        self.name = name


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


def test_finance_agent_inserted_first_when_needed(planner: LLMOrchestrationPlanner) -> None:
    steps = [
        PlannedAgentStep(
            agent="research_agent",
            tools=["research.search"],
            fallback_tools=[],
            reason="llm supplied",
        )
    ]

    processed, _ = planner._post_process_steps(
        "Need a revenue forecast for Q2 2025",
        steps,
        tool_aliases=None,
    )

    assert [step.agent for step in processed][:2] == ["finance_agent", "research_agent"]


def test_keyword_plan_prioritizes_finance_agent(planner: LLMOrchestrationPlanner) -> None:
    agents = [
        _AgentStub("finance_agent"),
        _AgentStub("research_agent"),
        _AgentStub("general_agent"),
    ]

    plan = planner._build_keyword_plan(
        prompt="Build a financial outlook and include supporting research",
        agents=agents,
        raw_response="{}",
        reason="short circuit",
        tool_aliases=None,
    )

    assert plan is not None
    assert [step.agent for step in plan.steps[:2]] == ["finance_agent", "research_agent"]


def test_enterprise_agent_inserted_before_general(planner: LLMOrchestrationPlanner) -> None:
    steps = [
        PlannedAgentStep(agent="general_agent", tools=[], fallback_tools=[], reason="intro")
    ]

    processed, _ = planner._post_process_steps(
        "Need an enterprise-level strategy plan for the board",
        steps,
        tool_aliases=None,
    )

    assert processed[0].agent == "enterprise_agent"


def test_enterprise_prompt_requires_enterprise_agent(planner: LLMOrchestrationPlanner) -> None:
    steps = [
        PlannedAgentStep(agent="general_agent", tools=[], fallback_tools=[], reason="triage"),
        PlannedAgentStep(
            agent="research_agent",
            tools=["research.search"],
            fallback_tools=["research.doc_loader"],
            reason="context",
        ),
    ]

    processed, adjustments = planner._post_process_steps(
        "Prepare an enterprise GTM roadmap for the board",
        steps,
        tool_aliases=None,
    )

    assert processed[0].agent == "enterprise_agent"
    assert processed[0].tools == ["enterprise.playbook"]
    assert processed[0].fallback_tools == ["enterprise.policy"]
    assert "enterprise_agent" in adjustments.get("added_agents", [])


def test_keyword_plan_handles_enterprise_prompts(planner: LLMOrchestrationPlanner) -> None:
    agents = [
        _AgentStub("enterprise_agent"),
        _AgentStub("general_agent"),
    ]

    plan = planner._build_keyword_plan(
        prompt="Develop an enterprise level expansion roadmap",
        agents=agents,
        raw_response="{}",
        reason="short circuit",
        tool_aliases=None,
    )

    assert plan is not None
    assert plan.steps[0].agent == "enterprise_agent"


def test_creative_agent_inserted_before_general(planner: LLMOrchestrationPlanner) -> None:
    steps = [
        PlannedAgentStep(agent="general_agent", tools=[], fallback_tools=[], reason="intro")
    ]

    processed, _ = planner._post_process_steps(
        "Please craft a Shakespeare styled poem about innovation",
        steps,
        tool_aliases=None,
    )

    assert processed[0].agent == "creative_agent"


def test_keyword_plan_handles_creative_prompts(planner: LLMOrchestrationPlanner) -> None:
    agents = [
        _AgentStub("creative_agent"),
        _AgentStub("general_agent"),
    ]

    plan = planner._build_keyword_plan(
        prompt="Compose a Shakespearean sonnet celebrating product launches",
        agents=agents,
        raw_response="{}",
        reason="short circuit",
        tool_aliases=None,
    )

    assert plan is not None
    assert plan.steps[0].agent == "creative_agent"


def test_creative_prompt_reorders_existing_steps(planner: LLMOrchestrationPlanner) -> None:
    steps = [
        PlannedAgentStep(agent="general_agent", tools=[], fallback_tools=[], reason="triage"),
        PlannedAgentStep(
            agent="creative_agent",
            tools=["creative.tonecheck"],
            fallback_tools=["creative.image"],
            reason="style",
        ),
    ]

    processed, adjustments = planner._post_process_steps(
        "Please draft a Shakespeare styled poem",
        steps,
        tool_aliases=None,
    )

    assert [step.agent for step in processed][:2] == ["creative_agent", "general_agent"]
    assert adjustments["reordered_agents"]["updated_order"][0] == "creative_agent"


def test_finance_prompt_reorders_existing_steps(planner: LLMOrchestrationPlanner) -> None:
    steps = [
        PlannedAgentStep(agent="general_agent", tools=[], fallback_tools=[], reason="triage"),
        PlannedAgentStep(
            agent="finance_agent",
            tools=["finance.snapshot"],
            fallback_tools=["finance.news"],
            reason="analysis",
        ),
        PlannedAgentStep(
            agent="research_agent",
            tools=["research.search"],
            fallback_tools=["research.doc_loader"],
            reason="context",
        ),
    ]

    processed, adjustments = planner._post_process_steps(
        "Need a financial forecast for FY2025",
        steps,
        tool_aliases=None,
    )

    assert processed[0].agent == "finance_agent"
    assert adjustments["reordered_agents"]["updated_order"][0] == "finance_agent"


def test_simple_greeting_does_not_trigger_finance(planner: LLMOrchestrationPlanner) -> None:
    steps = [
        PlannedAgentStep(agent="general_agent", tools=[], fallback_tools=[], reason="triage", confidence=0.9)
    ]

    processed, adjustments = planner._post_process_steps("hi", steps, tool_aliases=None)

    assert len(processed) == 1
    assert processed[0].agent == "general_agent"
    assert adjustments["reason"] == "simple_greeting"


def test_non_financial_quarter_prompt_keeps_general_agent(planner: LLMOrchestrationPlanner) -> None:
    steps = [
        PlannedAgentStep(agent="general_agent", tools=[], fallback_tools=[], reason="triage", confidence=0.9)
    ]

    processed, _ = planner._post_process_steps(
        "Summarize marketing OKRs for the quarter",
        steps,
        tool_aliases=None,
    )

    assert all(step.agent != "finance_agent" for step in processed)


def test_finance_agent_detected_for_stock_prompt(planner: LLMOrchestrationPlanner) -> None:
    steps = [
        PlannedAgentStep(agent="general_agent", tools=[], fallback_tools=[], reason="triage", confidence=0.9)
    ]

    processed, _ = planner._post_process_steps(
        "What is the latest stock outlook for NFLX?",
        steps,
        tool_aliases=None,
    )

    assert processed[0].agent == "finance_agent"
