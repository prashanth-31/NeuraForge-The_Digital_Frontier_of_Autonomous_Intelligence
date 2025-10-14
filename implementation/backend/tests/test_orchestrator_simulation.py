from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from app.agents.base import AgentContext
from app.orchestration.simulation import (
    SimulationHarness,
    SimulationReport,
    SimulationRunResult,
    SimulationScenario,
    summarize_reports,
)
from app.monitoring.orchestrator_report import build_evaluation_snapshot


@dataclass
class _DemoResult:
    status: str = "completed"
    consensus: float = 0.8


class _FakeOrchestrator:
    def __init__(self, *, result: _DemoResult | None = None) -> None:
        self._result = result or _DemoResult()

    async def route_task(self, task: dict, *, context: AgentContext, progress_cb=None) -> dict:
        if progress_cb is not None:
            await progress_cb({"event": "started", "task_id": task.get("id")})
        await asyncio.sleep(0)
        return {
            **task,
            "status": self._result.status,
            "outputs": [
                {
                    "agent": "demo",
                    "summary": f"Handled {task['prompt']}",
                    "confidence": 0.75,
                }
            ],
            "negotiation": {
                "status": "completed",
                "consensus": self._result.consensus,
                "metadata": {"strategy": "test"},
            },
            "guardrails": {"decisions": []},
            "plan": {"status": "planned"},
        }


def _context_factory(_: dict) -> AgentContext:
    return AgentContext(memory=None, llm=None)


@pytest.mark.asyncio
async def test_simulation_harness_produces_successful_report() -> None:
    scenario = SimulationScenario(
        name="unit-test",
        base_task={"prompt": "Collect KPIs"},
        repetitions=2,
        concurrency=2,
    )
    harness = SimulationHarness(_FakeOrchestrator(), context_factory=_context_factory)
    report = await harness.run(scenario)
    assert len(report.runs) == 2
    assert report.success_rate == pytest.approx(1.0)
    assert report.average_latency >= 0
    snapshot = build_evaluation_snapshot(report)
    assert snapshot.total_runs == 2
    assert snapshot.success_rate == pytest.approx(1.0)


def test_summarize_reports_handles_multiple_batches() -> None:
    scenario = SimulationScenario(name="rollup", base_task={"prompt": "A"})
    run = SimulationRunResult(
        task_id="demo",
        status="completed",
        latency_seconds=0.1,
        negotiation_consensus=0.7,
        guardrail_decisions=0,
        escalations=0,
        outputs=[],
        plan_status="planned",
        meta_summary=None,
        meta_confidence=None,
        meta_mode=None,
        meta_should_escalate=False,
        dispute_flagged=False,
    )
    report = SimulationReport(scenario=scenario, runs=[run])
    summary = summarize_reports([report])
    assert summary["total_runs"] == 1
    assert summary["success_rate"] == pytest.approx(1.0)
    assert summary["average_latency"] == pytest.approx(0.1)
