from __future__ import annotations

import pytest

from app.agents.base import AgentContext
from app.agents.research import ResearchAgent
from app.schemas.agents import AgentInput
from tests.helpers.stubs import StubLLMService, StubMemoryService, StubToolService


@pytest.mark.asyncio
async def test_research_agent_enriches_missing_finance_metrics() -> None:
    agent = ResearchAgent()
    memory = StubMemoryService()
    llm = StubLLMService()
    tools = StubToolService()

    tools.set_response(
        "research.search",
        {
            "results": [
                {
                    "summary": "Apple expands buybacks",
                    "source": "newswire",
                }
            ]
        },
    )

    tools.set_response(
        "finance.snapshot",
        {
            "metrics": [
                {
                    "symbol": "AAPL",
                    "price": 210.5,
                    "market_cap": 3_000_000_000_000,
                    "volume": 120_000_000,
                    "fundamentals": {
                        "revenue_ttm": 380_000_000_000,
                        "revenue_growth": 0.08,
                        "eps": 6.72,
                    },
                    "updated_at": "2025-11-29T00:00:00Z",
                }
            ]
        },
        resolved_tool="finance/yfinance",
    )

    context = AgentContext(memory=memory, llm=llm, tools=tools, context=None, scorer=None)

    task = AgentInput(
        task_id="task-finance-gap",
        prompt="Provide Apple financial overview focusing on market cap, EPS, and revenue growth.",
        metadata={"symbols": ["AAPL"], "required_metrics": ["market cap", "revenue growth", "eps"]},
    )

    output = await agent.handle(task, context=context)

    finance_calls = [call for call in tools.calls if call[0].startswith("finance.snapshot")]
    assert finance_calls, "expected finance snapshot invocation when metrics missing"
    assert finance_calls[0][1]["symbols"] == ["AAPL"]

    resolved_metrics = output.metadata["resolved_metrics"]
    assert resolved_metrics["market_cap"] == 3_000_000_000_000
    assert resolved_metrics["revenue"] == 380_000_000_000
    assert resolved_metrics["revenue_growth"] == 0.08
    assert resolved_metrics["eps"] == pytest.approx(6.72)

    enrichment = output.metadata["enriched_metrics"]
    assert enrichment["missing"] == []
    assert enrichment["source"] == "finance/yfinance"
    assert enrichment["timestamp"] == "2025-11-29T00:00:00Z"

    assert output.summary.startswith("stubbed-response")


@pytest.mark.asyncio
async def test_research_agent_enriches_when_metrics_requested_without_finance_context() -> None:
    agent = ResearchAgent()
    memory = StubMemoryService()
    llm = StubLLMService()
    tools = StubToolService()

    tools.set_response("research.search", {"results": []})

    tools.set_response(
        "finance.snapshot",
        {
            "metrics": [
                {
                    "symbol": "GBX",
                    "price": 42.5,
                    "market_cap": 12_000_000_000,
                    "fundamentals": {
                        "revenue_ttm": 18_000_000_000,
                        "revenue_growth": 0.12,
                        "eps": 3.14,
                    },
                    "updated_at": "2025-11-29T12:00:00Z",
                }
            ]
        },
        resolved_tool="finance/yfinance",
    )

    context = AgentContext(memory=memory, llm=llm, tools=tools, context=None, scorer=None)

    task = AgentInput(
        task_id="task-finance-no-hints",
        prompt="Provide a narrative on Globex Corporation's community partnerships.",
        metadata={"required_metrics": ["eps", "revenue", "revenue growth rate"]},
    )

    output = await agent.handle(task, context=context)

    finance_calls = [call for call in tools.calls if call[0].startswith("finance.snapshot")]
    assert finance_calls, "expected finance snapshot invocation even without finance hints"
    assert "query" in finance_calls[0][1]

    resolved_metrics = output.metadata["resolved_metrics"]
    assert resolved_metrics["eps"] == pytest.approx(3.14)
    assert resolved_metrics["revenue"] == 18_000_000_000
    assert resolved_metrics["revenue_growth_rate"] == 0.12

    enrichment = output.metadata["enriched_metrics"]
    assert enrichment["missing"] == []
    assert enrichment["source"] == "finance/yfinance"
    assert enrichment["timestamp"] == "2025-11-29T12:00:00Z"

    assert output.summary.startswith("stubbed-response")
