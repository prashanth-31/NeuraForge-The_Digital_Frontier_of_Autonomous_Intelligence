from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.core.config import PlanningSettings, SchedulingSettings
from app.orchestration.context import ContextAssemblyContract, ContextStage
from app.orchestration.guardrails import GuardrailManager
from app.orchestration.planner import DependencyTaskPlanner, PlannedStep, TaskPlan
from app.orchestration.scheduler import AsyncioTaskScheduler
from app.schemas.tasks import TaskRequest
from app.core.config import GuardrailSettings


@pytest.mark.asyncio
async def test_dependency_planner_creates_dependency_order() -> None:
    settings = PlanningSettings(enabled=True, max_subtasks=5, default_step_duration_minutes=15)
    planner = DependencyTaskPlanner(settings=settings)
    task = {
        "id": "task-123",
        "prompt": "Produce market analysis",
        "metadata": {
            "subtasks": [
                {
                    "id": "research",
                    "title": "Compile research",
                    "description": "Gather latest financial news",
                    "capability": "research",
                },
                {
                    "id": "analysis",
                    "title": "Financial analysis",
                    "description": "Summarize financial indicators",
                    "capability": "finance",
                    "depends_on": ["research"],
                },
            ]
        },
    }
    outputs = [
        {
            "agent": "research_agent",
            "summary": "Identified key market trends",
            "metadata": {
                "actions": [
                    {
                        "id": "summary",
                        "title": "Executive summary",
                        "description": "Draft executive summary",
                        "capability": "enterprise",
                        "depends_on": ["analysis"],
                    }
                ]
            },
        }
    ]
    plan = await planner.build_plan(task=task, outputs=outputs, negotiation=None)
    assert plan is not None
    assert [step.step_id for step in plan.steps] == ["research", "analysis", "summary"]
    assert plan.steps[1].depends_on == ["research"]
    assert plan.steps[2].depends_on == ["analysis"]
    assert plan.steps[0].agent == "research_agent"
    assert plan.metadata["strategy"] == "dependency"


@pytest.mark.asyncio
async def test_async_scheduler_assigns_concurrent_etas() -> None:
    scheduling = SchedulingSettings(max_concurrency=2, default_deadline_minutes=30)
    scheduler = AsyncioTaskScheduler(settings=scheduling)
    plan = TaskPlan(
        task_id="task-xyz",
        summary="Test plan",
        steps=[
            PlannedStep(step_id="a", title="A", agent="alpha", description="step a"),
            PlannedStep(step_id="b", title="B", agent="beta", description="step b", depends_on=["a"]),
            PlannedStep(step_id="c", title="C", agent="gamma", description="step c"),
        ],
    )
    scheduled = await scheduler.schedule(plan, start_time=datetime(2025, 10, 14, tzinfo=timezone.utc))
    assert all(step.eta_iso for step in scheduled.steps)
    assert scheduled.steps[0].deadline_iso is not None
    # dependency ensures b starts after a
    assert scheduled.steps[1].eta_iso >= scheduled.steps[0].deadline_iso


@pytest.mark.asyncio
async def test_guardrail_manager_flags_high_risk(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = GuardrailSettings(enabled=True, enforce_policies=True, enforce_safety_filters=False, risk_threshold=0.4)
    manager = GuardrailManager(settings=settings, store=None, llm_service=None)
    decision = await manager.evaluate_step(task_id="task", run_id=None, step={"description": "classified exploit", "risk_score": 0.9}, agent="alpha")
    assert decision.decision.value in {"escalate", "deny", "review"}


@pytest.mark.asyncio
async def test_context_contract_stage_overrides() -> None:
    class FakeAssembler:
        async def build(self, *, task: dict[str, str], agent: str | None = None):  # type: ignore[override]
            from app.services.retrieval import ContextBundle, ContextSnippet

            snippets = [
                ContextSnippet(source="episodic", content=f"detail-{idx}", metadata={}, score=1.0)
                for idx in range(5)
            ]
            return ContextBundle(query=task["prompt"], snippets=snippets, max_chars=2000)

    contract = ContextAssemblyContract(
        assembler=FakeAssembler(),
        stage_overrides={ContextStage.NEGOTIATION: {"top_snippets": 2, "max_chars": 100}},
    )
    bundle = await contract.build_for_stage(ContextStage.NEGOTIATION, task={"prompt": "test"}, agent=None)
    assert len(bundle.snippets) == 2
    assert bundle.max_chars == 100