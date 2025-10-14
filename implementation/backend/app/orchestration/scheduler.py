from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .planner import TaskPlan, clone_plan_with_schedule


class TaskScheduler:
    async def schedule(self, plan: TaskPlan, *, start_time: datetime | None = None) -> TaskPlan:
        raise NotImplementedError


class SequentialTaskScheduler(TaskScheduler):
    def __init__(self, *, step_interval: timedelta | None = None) -> None:
        self._step_interval = step_interval or timedelta(minutes=5)

    async def schedule(self, plan: TaskPlan, *, start_time: datetime | None = None) -> TaskPlan:
        if not plan.steps:
            return plan
        origin = start_time or datetime.now(timezone.utc)
        timestamps: list[str] = []
        current = origin
        for _ in plan.steps:
            timestamps.append(current.isoformat())
            current += self._step_interval
        return clone_plan_with_schedule(plan, timestamps=timestamps)


class InMemoryTaskScheduler(TaskScheduler):
    def __init__(self) -> None:
        self._scheduled: list[TaskPlan] = []

    @property
    def plans(self) -> Iterable[TaskPlan]:
        return tuple(self._scheduled)

    async def schedule(self, plan: TaskPlan, *, start_time: datetime | None = None) -> TaskPlan:
        scheduled = await SequentialTaskScheduler().schedule(plan, start_time=start_time)
        self._scheduled.append(scheduled)
        return scheduled
