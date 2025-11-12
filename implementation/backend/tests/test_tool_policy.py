import pytest

from app.orchestration.graph import Orchestrator, ToolFirstPolicyViolation, _ToolSession
from app.schemas.agents import AgentCapability


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
