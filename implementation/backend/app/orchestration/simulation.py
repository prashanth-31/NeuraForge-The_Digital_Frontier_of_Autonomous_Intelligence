from __future__ import annotations

import asyncio
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Awaitable, Callable, Iterable, Sequence

from .graph import Orchestrator
from ..agents.base import AgentContext


@dataclass(slots=True)
class SimulationScenario:
    """Defines a synthetic workload for orchestrator stress testing."""

    name: str
    base_task: dict[str, Any]
    variations: Sequence[dict[str, Any]] = field(default_factory=tuple)
    repetitions: int = 1
    concurrency: int = 1
    notes: str | None = None


@dataclass(slots=True)
class SimulationRunResult:
    task_id: str
    status: str
    latency_seconds: float
    negotiation_consensus: float | None
    guardrail_decisions: int
    escalations: int
    outputs: list[dict[str, Any]]
    plan_status: str | None
    meta_summary: str | None
    meta_confidence: float | None
    meta_mode: str | None
    meta_should_escalate: bool
    dispute_flagged: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "latency_seconds": self.latency_seconds,
            "negotiation_consensus": self.negotiation_consensus,
            "guardrail_decisions": self.guardrail_decisions,
            "escalations": self.escalations,
            "outputs": self.outputs,
            "plan_status": self.plan_status,
            "meta_summary": self.meta_summary,
            "meta_confidence": self.meta_confidence,
            "meta_mode": self.meta_mode,
            "meta_should_escalate": self.meta_should_escalate,
            "dispute_flagged": self.dispute_flagged,
        }


@dataclass(slots=True)
class SimulationReport:
    scenario: SimulationScenario
    runs: list[SimulationRunResult]

    @property
    def success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for run in self.runs if run.status == "completed") / len(self.runs)

    @property
    def average_latency(self) -> float:
        if not self.runs:
            return 0.0
        return mean(run.latency_seconds for run in self.runs)

    @property
    def escalation_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for run in self.runs if run.escalations > 0) / len(self.runs)

    def to_timeseries_payload(self) -> list[dict[str, Any]]:
        return [run.as_dict() for run in self.runs]


class SimulationHarness:
    """Runs orchestrator simulations using synthetic scenarios."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        *,
        context_factory: Callable[[dict[str, Any]], AgentContext],
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._context_factory = context_factory
        self._progress_callback = progress_callback

    async def run(self, scenario: SimulationScenario) -> SimulationReport:
        tasks = self._expand_tasks(scenario)
        semaphore = asyncio.Semaphore(max(1, scenario.concurrency))
        runs: list[SimulationRunResult] = []

        async def execute(payload: dict[str, Any]) -> None:
            async with semaphore:
                runs.append(await self._run_single(payload))

        await asyncio.gather(*(execute(task) for task in tasks))
        return SimulationReport(scenario=scenario, runs=runs)

    def _expand_tasks(self, scenario: SimulationScenario) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        variations = scenario.variations or (None,)
        for variation in variations:
            base = deepcopy(scenario.base_task)
            if variation:
                base = self._merge(base, variation)
            for index in range(max(1, scenario.repetitions)):
                task = deepcopy(base)
                task.setdefault("id", f"{scenario.name}-{uuid.uuid4()}-{index}")
                task.setdefault("metadata", {})
                task["metadata"].setdefault("scenario", scenario.name)
                tasks.append(task)
        return tasks

    async def _run_single(self, task: dict[str, Any]) -> SimulationRunResult:
        context = self._context_factory(task)
        start = time.perf_counter()
        result = await self._orchestrator.route_task(
            deepcopy(task),
            context=context,
            progress_cb=self._progress_callback,
        )
        latency = time.perf_counter() - start
        negotiation = result.get("negotiation") or {}
        guardrail_section = result.get("guardrails") or {}
        decisions = guardrail_section.get("decisions") or []
        escalations = sum(1 for entry in decisions if entry.get("decision") in {"review", "escalate"})
        meta_section = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        dispute_section = meta_section.get("dispute") if isinstance(meta_section.get("dispute"), dict) else {}
        run_result = SimulationRunResult(
            task_id=str(result.get("id") or task.get("id")),
            status=str(result.get("status") or "unknown"),
            latency_seconds=latency,
            negotiation_consensus=float(negotiation.get("consensus")) if negotiation.get("consensus") is not None else None,
            guardrail_decisions=len(decisions),
            escalations=escalations,
            outputs=[dict(item) for item in result.get("outputs", [])],
            plan_status=result.get("plan", {}).get("status") if isinstance(result.get("plan"), dict) else None,
            meta_summary=str(meta_section.get("summary")) if meta_section.get("summary") else None,
            meta_confidence=float(meta_section.get("confidence")) if meta_section.get("confidence") is not None else None,
            meta_mode=str(meta_section.get("mode")) if meta_section.get("mode") else None,
            meta_should_escalate=bool(meta_section.get("should_escalate")) if meta_section else False,
            dispute_flagged=bool(dispute_section.get("flagged")) if isinstance(dispute_section, dict) else False,
        )
        return run_result

    def _merge(self, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(base)
        for key, value in overrides.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged


def summarize_reports(reports: Iterable[SimulationReport]) -> dict[str, Any]:
    reports_list = list(reports)
    if not reports_list:
        return {"total_runs": 0, "success_rate": 0.0, "average_latency": 0.0, "escalation_rate": 0.0}
    total_runs = sum(len(report.runs) for report in reports_list)
    successes = sum(1 for report in reports_list for run in report.runs if run.status == "completed")
    avg_latency = mean(run.latency_seconds for report in reports_list for run in report.runs) if total_runs else 0.0
    escalation_events = sum(1 for report in reports_list for run in report.runs if run.escalations > 0)
    return {
        "total_runs": total_runs,
        "success_rate": successes / total_runs if total_runs else 0.0,
        "average_latency": avg_latency,
        "escalation_rate": escalation_events / total_runs if total_runs else 0.0,
    }
