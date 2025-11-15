import os
import random
import time

import pytest

from app.orchestration.graph import Orchestrator, ToolFirstPolicyViolation, _ToolSession
from app.orchestration.tool_policy import get_agent_tool_policy
from app.schemas.agents import AgentCapability
from app.tools.exceptions import ToolPolicyViolationError


class _StubAgent:
    def __init__(self, name: str, capability: AgentCapability) -> None:
        self.name = name
        self.capability = capability
        self.system_prompt = ""
        self.description = ""
        self.tool_preference: list[str] = []
        self.fallback_agent = None
        self.confidence_bias = 0.0

    async def handle(self, *args, **kwargs):  # pragma: no cover - not used in tests
        raise NotImplementedError


class _NullToolService:
    async def invoke(self, tool: str, payload: dict) -> dict:  # pragma: no cover - not used in tests
        return {"tool": tool, "payload": payload}


class _EchoToolService:
    async def invoke(self, tool: str, payload: dict) -> dict:
        return {"tool": tool, "payload": payload, "invoked": True}


@pytest.mark.asyncio
async def test_enforced_agent_requires_successful_tool() -> None:
    orchestrator = Orchestrator(agents=[])
    session = _ToolSession(agent_name="finance_agent", tool_service=_NullToolService())
    agent = _StubAgent("finance_agent", AgentCapability.FINANCE)

    with pytest.raises(ToolFirstPolicyViolation):
        await orchestrator._enforce_tool_first_policy(session, agent=agent, state={"outputs": []})


@pytest.mark.asyncio
async def test_optional_agent_can_skip_without_planned_tools() -> None:
    orchestrator = Orchestrator(agents=[])
    session = _ToolSession(agent_name="general_agent", tool_service=_NullToolService())
    agent = _StubAgent("general_agent", AgentCapability.GENERAL)

    await orchestrator._enforce_tool_first_policy(session, agent=agent, state={"outputs": []})


@pytest.mark.asyncio
async def test_optional_agent_can_skip_greeting_plan() -> None:
    orchestrator = Orchestrator(agents=[])
    session = _ToolSession(
        agent_name="general_agent",
        tool_service=_NullToolService(),
        planned_tools=("search",),
        planner_reason="greeting",
    )
    agent = _StubAgent("general_agent", AgentCapability.GENERAL)
    state = {"prompt": "hello", "outputs": []}

    await orchestrator._enforce_tool_first_policy(session, agent=agent, state=state)


@pytest.mark.asyncio
async def test_optional_agent_with_plan_raises_when_context_requires_tools() -> None:
    orchestrator = Orchestrator(agents=[])
    session = _ToolSession(
        agent_name="general_agent",
        tool_service=_NullToolService(),
        planned_tools=("search",),
        planner_reason="deep analysis",
    )
    agent = _StubAgent("general_agent", AgentCapability.GENERAL)
    state = {"prompt": "Provide detailed review", "outputs": []}

    with pytest.raises(ToolFirstPolicyViolation):
        await orchestrator._enforce_tool_first_policy(session, agent=agent, state=state)


@pytest.mark.asyncio
async def test_tool_policy_blocks_disallowed_invocation() -> None:
    policy = get_agent_tool_policy("finance_agent")
    session = _ToolSession(agent_name="finance_agent", tool_service=_NullToolService(), policy=policy)

    with pytest.raises(ToolPolicyViolationError):
        await session.proxy.invoke("creative.text.rewrite", {"prompt": "draft a slogan"})


@pytest.mark.asyncio
async def test_tool_policy_allows_permitted_invocation() -> None:
    policy = get_agent_tool_policy("finance_agent")
    session = _ToolSession(agent_name="finance_agent", tool_service=_NullToolService(), policy=policy)

    result = await session.proxy.invoke("finance.indicators.rsi", {"symbol": "NFLX"})
    assert result["tool"] == "finance.indicators.rsi"


@pytest.mark.asyncio
async def test_tool_policy_randomized() -> None:
    seed_env = os.environ.get("TOOL_POLICY_RANDOM_SEED")
    seed = int(seed_env) if seed_env and seed_env.isdigit() else time.time_ns()
    rng = random.Random(seed)

    scenarios = {
        "enterprise_agent": {
            "allowed": [
                "enterprise.summary.generate",
                "browser.open",
                "dataframe.describe",
                "memory.store",
            ],
            "disallowed": [
                "finance.snapshot",
                "creative.brainstorm",
                "research.vector_search",
            ],
        },
        "finance_agent": {
            "allowed": [
                "finance.snapshot",
                "finance.indicators.rsi",
                "dataframe.plot",
                "memory.retrieve",
            ],
            "disallowed": [
                "creative.image.generate",
                "terminal.execute",
                "research.vector_search",
            ],
        },
        "research_agent": {
            "allowed": [
                "research.search",
                "browser.extract_text",
                "pdf.extract_text",
                "text.summarize",
            ],
            "disallowed": [
                "finance.snapshot",
                "creative.compose",
                "terminal.execute",
            ],
        },
        "creative_agent": {
            "allowed": [
                "creative.storyboard",
                "creative.compose",
                "browser.open",
                "text.generate",
            ],
            "disallowed": [
                "finance.snapshot",
                "research.search",
                "code.execute",
            ],
        },
    }

    iterations = 10
    for _ in range(iterations):
        agent_name, scenario = rng.choice(list(scenarios.items()))
        policy = get_agent_tool_policy(agent_name)
        assert policy is not None

        session = _ToolSession(agent_name=agent_name, tool_service=_EchoToolService(), policy=policy)

        allowed_candidates = [tool for tool in scenario["allowed"] if policy.is_allowed(tool)]
        assert allowed_candidates, f"No allowed tools matched policy for {agent_name}"
        allowed_tool = rng.choice(allowed_candidates)
        allowed_payload = {"seed": seed, "tool": allowed_tool}
        result = await session.proxy.invoke(allowed_tool, allowed_payload)
        assert result["tool"] == allowed_tool

        disallowed_candidates = [tool for tool in scenario["disallowed"] if not policy.is_allowed(tool)]
        assert disallowed_candidates, f"No disallowed tools matched policy for {agent_name}"
        disallowed_tool = rng.choice(disallowed_candidates)
        with pytest.raises(ToolPolicyViolationError):
            await session.proxy.invoke(disallowed_tool, {"seed": seed, "tool": disallowed_tool})
