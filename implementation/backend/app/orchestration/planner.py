from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable


@dataclass(slots=True)
class PlannedStep:
    step_id: str
    title: str
    agent: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    eta_iso: str | None = None


@dataclass(slots=True)
class TaskPlan:
    task_id: str
    summary: str
    steps: list[PlannedStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskPlanner:
    async def build_plan(
        self,
        *,
        task: dict[str, Any],
        outputs: Iterable[dict[str, Any]],
        negotiation: dict[str, Any] | None,
    ) -> TaskPlan | None:
        raise NotImplementedError


class SimpleTaskPlanner(TaskPlanner):
    def __init__(self, *, default_step_prefix: str = "step") -> None:
        self._default_step_prefix = default_step_prefix

    async def build_plan(
        self,
        *,
        task: dict[str, Any],
        outputs: Iterable[dict[str, Any]],
        negotiation: dict[str, Any] | None,
    ) -> TaskPlan | None:
        task_id = str(task.get("id") or task.get("task_id") or "")
        if not task_id:
            return None

        summary = "Orchestrated task plan"
        if negotiation and negotiation.get("outcome"):
            summary = str(negotiation["outcome"])

        steps: list[PlannedStep] = []
        for index, raw in enumerate(outputs):
            if not isinstance(raw, dict):
                continue
            agent_name = str(raw.get("agent", "unknown"))
            description = str(raw.get("summary") or raw.get("content") or "").strip()
            if not description:
                continue
            step_id = f"{task_id}-{self._default_step_prefix}-{index + 1}"
            depends_on = [steps[-1].step_id] if steps else []
            raw_metadata = raw.get("metadata")
            metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
            steps.append(
                PlannedStep(
                    step_id=step_id,
                    title=f"{agent_name} deliverable",
                    agent=agent_name,
                    description=description,
                    depends_on=list(depends_on),
                    metadata=metadata,
                )
            )

        if not steps:
            return None

        plan_metadata: dict[str, Any] = {
            "strategy": "simple",
            "negotiation": negotiation or {},
        }
        return TaskPlan(task_id=task_id, summary=summary, steps=steps, metadata=plan_metadata)


def clone_plan_with_schedule(plan: TaskPlan, *, timestamps: list[str]) -> TaskPlan:
    if len(timestamps) != len(plan.steps):
        raise ValueError("Timestamp list must align with plan steps")
    scheduled_steps = [replace(step, eta_iso=ts) for step, ts in zip(plan.steps, timestamps)]
    return TaskPlan(task_id=plan.task_id, summary=plan.summary, steps=scheduled_steps, metadata=dict(plan.metadata))
