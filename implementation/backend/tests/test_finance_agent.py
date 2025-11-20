import pytest

from app.agents.base import AgentContext
from app.agents.finance import FinanceAgent
from app.schemas.agents import AgentInput
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
    assert override["reason"] == "planner_tool_outage"
    assert override["requested_tools"] == ["finance.snapshot", "finance/pandas"]
    assert override["tool_used"] == "finance.snapshot.cached"


def _make_snapshot_with_fundamentals() -> ToolInvocationResult:
    fundamentals = {
        "quarterly": [
            {"period": "2024-06-30", "total_revenue": 120_000_000_000, "net_income": 25_000_000_000},
            {"period": "2024-03-31", "total_revenue": 110_000_000_000, "net_income": 23_000_000_000},
            {"period": "2023-12-31", "total_revenue": 105_000_000_000, "net_income": 21_000_000_000},
        ]
    }
    return ToolInvocationResult(
        tool="finance.snapshot",
        payload={"symbols": ["AAPL"]},
        response={
            "metrics": [
                {
                    "symbol": "AAPL",
                    "company_name": "Apple",
                    "fundamentals": fundamentals,
                }
            ]
        },
        cached=False,
        latency=0.01,
        resolved_tool="finance/yfinance",
    )


def test_build_plot_payload_uses_quarterly_fundamentals() -> None:
    agent = FinanceAgent()

    payload = agent._build_plot_payload(_make_snapshot_with_fundamentals())  # type: ignore[attr-defined]

    assert payload is not None
    assert payload["series"]
    first_series = payload["series"][0]
    assert first_series["name"] == "Revenue (B)"
    assert first_series["points"][0]["y"] == pytest.approx(105.0)


def test_planner_requested_plot_detects_variants() -> None:
    agent = FinanceAgent()

    result = agent._planner_requested_plot(  # type: ignore[attr-defined]
        planned_tools=["finance/plot"],
        fallback_tools=None,
        task_metadata=None,
    )

    assert result is True


@pytest.mark.asyncio
async def test_maybe_generate_plot_skips_without_tools() -> None:
    agent = FinanceAgent()
    task = AgentInput(task_id="task-1", prompt="report", metadata={})
    context = AgentContext(
        memory=object(),
        llm=object(),
        context=None,
        tools=None,
        scorer=None,
        planned_tools=["finance.plot"],
        fallback_tools=None,
        planner_reason=None,
    )
    trace: list[dict[str, str]] = []

    result = await agent._maybe_generate_plot(  # type: ignore[attr-defined]
        task,
        context=context,
        snapshot_result=None,
        tool_trace=trace,
    )

    assert result is None
    assert trace[-1]["reason"] == "tool_service_unavailable"
