import pytest

from app.agents.finance import FinanceAgent
from app.services.tools import ToolInvocationResult


def _make_tool_result(tool: str) -> ToolInvocationResult:
    return ToolInvocationResult(
        tool=tool,
        payload={"symbols": ["MSFT"]},
        response={"metrics": []},
        cached=False,
        latency=0.01,
        resolved_tool=tool,
    )


@pytest.mark.asyncio
async def test_override_uses_planner_metadata_when_context_missing() -> None:
    agent = FinanceAgent()
    tool_trace = [
        {"tool": "finance.snapshot", "status": "error", "error": "rate limit"},
        {"tool": "finance.snapshot.cached", "status": "success"},
    ]
    metadata = {
        "planner_step": {
            "planned_tools": ["finance.snapshot", "finance/pandas"],
            "planned_fallback_tools": [],
        }
    }

    override = agent._tool_policy_override(  # type: ignore[attr-defined]
        tool_result=_make_tool_result("finance.snapshot.cached"),
        tool_trace=tool_trace,
        tools=object(),
        planned_tools=None,
        fallback_tools=None,
        task_metadata=metadata,
    )

    assert override is not None
    assert override["reason"] == "planner_tool_unsupported"
    assert override["requested_tools"] == ["finance.snapshot", "finance/pandas"]
    assert override["tool_used"] == "finance.snapshot.cached"
