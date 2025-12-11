from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Iterable, Mapping

from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_random_exponential

from ..core.config import SchedulingSettings
from ..orchestration.enums import LifecycleStatus

from .planner import PlannedStep, TaskPlan, clone_plan_with_schedule


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


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int
    base_backoff_seconds: float
    backoff_multiplier: float
    max_backoff_seconds: float

    @classmethod
    def from_settings(cls, settings: SchedulingSettings) -> "RetryPolicy":
        return cls(
            max_attempts=settings.default_retry_attempts,
            base_backoff_seconds=settings.base_backoff_seconds,
            backoff_multiplier=settings.backoff_multiplier,
            max_backoff_seconds=settings.max_backoff_seconds,
        )

    def merge(self, override: Mapping[str, object] | None) -> "RetryPolicy":
        if not override:
            return self

        def _coerce_int(value: object, default: int) -> int:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return default
            return default

        def _coerce_float(value: object, default: float) -> float:
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
            return default

        return RetryPolicy(
            max_attempts=_coerce_int(override.get("max_attempts"), self.max_attempts),
            base_backoff_seconds=_coerce_float(override.get("backoff_seconds"), self.base_backoff_seconds),
            backoff_multiplier=_coerce_float(override.get("backoff_multiplier"), self.backoff_multiplier),
            max_backoff_seconds=_coerce_float(override.get("max_backoff_seconds"), self.max_backoff_seconds),
        )


class AsyncioTaskScheduler(TaskScheduler):
    def __init__(
        self,
        *,
        settings: SchedulingSettings,
    ) -> None:
        self._settings = settings
        self._retry_defaults = RetryPolicy.from_settings(settings)

    async def schedule(self, plan: TaskPlan, *, start_time: datetime | None = None) -> TaskPlan:
        if not plan.steps:
            return plan
        now = start_time or datetime.now(timezone.utc)
        slot_available: list[datetime] = [now for _ in range(self._settings.max_concurrency)]
        completion_times: dict[str, datetime] = {}
        scheduled_steps: list[PlannedStep] = []

        for step in plan.steps:
            dependency_ready = [completion_times[dep] for dep in step.depends_on if dep in completion_times]
            ready_time = max(dependency_ready) if dependency_ready else now
            slot_index = min(range(len(slot_available)), key=lambda idx: slot_available[idx])
            slot_time = slot_available[slot_index]
            start_at = max(ready_time, slot_time)
            duration_minutes = self._extract_duration_minutes(step)
            completion_time = start_at + timedelta(minutes=duration_minutes)
            deadline_time = self._compute_deadline(step, start_at, completion_time)

            slot_available[slot_index] = completion_time
            completion_times[step.step_id] = completion_time

            retry_policy = self._retry_defaults.merge(step.retry_policy)
            merged_metadata = dict(step.metadata or {})
            merged_metadata.setdefault("lifecycle_status", LifecycleStatus.PLANNED.value)
            merged_metadata["retry_policy"] = {
                "max_attempts": retry_policy.max_attempts,
                "backoff_seconds": retry_policy.base_backoff_seconds,
                "backoff_multiplier": retry_policy.backoff_multiplier,
                "max_backoff_seconds": retry_policy.max_backoff_seconds,
            }

            scheduled_steps.append(
                replace(
                    step,
                    eta_iso=start_at.isoformat(),
                    deadline_iso=deadline_time.isoformat() if deadline_time else None,
                    metadata=merged_metadata,
                )
            )

        return TaskPlan(task_id=plan.task_id, summary=plan.summary, steps=scheduled_steps, metadata=dict(plan.metadata))

    async def dispatch(
        self,
        plan: TaskPlan,
        *,
        executor: Callable[[PlannedStep], Awaitable[None]],
        on_event: Callable[[str, PlannedStep, dict[str, object]], Awaitable[None]] | None = None,
    ) -> None:
        pending: dict[str, asyncio.Task[None]] = {}

        async def run_step(step: PlannedStep) -> None:
            for dep in step.depends_on:
                if dep in pending:
                    await pending[dep]

            retry_policy = self._retry_defaults.merge(step.retry_policy)
            attempts = retry_policy.max_attempts if retry_policy.max_attempts > 0 else 1
            max_arg = retry_policy.max_backoff_seconds if retry_policy.max_backoff_seconds > 0 else None
            multiplier = max(retry_policy.base_backoff_seconds, 1.0)
            exp_base = max(1.5, retry_policy.backoff_multiplier)
            if max_arg is not None:
                wait_strategy = wait_random_exponential(
                    multiplier=multiplier,
                    exp_base=exp_base,
                    max=max_arg,
                )
            else:
                wait_strategy = wait_random_exponential(
                    multiplier=multiplier,
                    exp_base=exp_base,
                )

            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(attempts),
                wait=wait_strategy,
                reraise=True,
            ):
                with attempt:
                    if on_event is not None:
                        await on_event("started", step, {"attempt": attempt.retry_state.attempt_number})
                    await executor(step)
                    if on_event is not None:
                        await on_event(
                            "completed",
                            step,
                            {
                                "attempt": attempt.retry_state.attempt_number,
                            },
                        )

        semaphore = asyncio.Semaphore(self._settings.max_concurrency)

        async def gated_executor(step: PlannedStep) -> None:
            async with semaphore:
                try:
                    await run_step(step)
                except RetryError as exc:
                    if on_event is not None:
                        await on_event(
                            "failed",
                            step,
                            {
                                "attempt": exc.last_attempt.attempt_number if exc.last_attempt else None,
                                "error": str(exc.last_attempt.exception() if exc.last_attempt else exc),
                            },
                        )
                    raise

        for step in plan.steps:
            pending[step.step_id] = asyncio.create_task(gated_executor(step))

        await asyncio.gather(*pending.values())

    def _extract_duration_minutes(self, step: PlannedStep) -> int:
        metadata = step.metadata or {}
        duration = metadata.get("duration_minutes") or metadata.get("estimated_minutes")
        if isinstance(duration, (int, float)):
            return max(1, int(duration))
        baseline = max(1, self._settings.default_deadline_minutes // 3)
        return baseline

    def _compute_deadline(
        self,
        step: PlannedStep,
        start_at: datetime,
        completion_time: datetime,
    ) -> datetime | None:
        if step.deadline_iso and step.deadline_iso.startswith("+"):
            try:
                seconds = int(step.deadline_iso.strip("+s"))
                return start_at + timedelta(seconds=seconds)
            except ValueError:
                return completion_time
        return completion_time
