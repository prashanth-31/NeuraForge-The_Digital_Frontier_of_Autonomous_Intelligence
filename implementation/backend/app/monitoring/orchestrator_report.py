from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from ..core.metrics import update_success_rate_from_history
from ..orchestration.simulation import SimulationReport


@dataclass(slots=True)
class EvaluationSnapshot:
    captured_at: datetime
    success_rate: float
    average_latency: float
    escalation_rate: float
    total_runs: int


def build_evaluation_snapshot(report: SimulationReport) -> EvaluationSnapshot:
    captured = datetime.now(timezone.utc)
    snapshot = EvaluationSnapshot(
        captured_at=captured,
        success_rate=report.success_rate,
        average_latency=report.average_latency,
        escalation_rate=report.escalation_rate,
        total_runs=len(report.runs),
    )
    update_success_rate_from_history([(captured, run.status == "completed") for run in report.runs])
    return snapshot


def combine_snapshots(snapshots: Iterable[EvaluationSnapshot]) -> EvaluationSnapshot:
    snapshots = list(snapshots)
    if not snapshots:
        return EvaluationSnapshot(
            captured_at=datetime.now(timezone.utc),
            success_rate=0.0,
            average_latency=0.0,
            escalation_rate=0.0,
            total_runs=0,
        )
    total_runs = sum(item.total_runs for item in snapshots)
    if total_runs == 0:
        return EvaluationSnapshot(
            captured_at=max(item.captured_at for item in snapshots),
            success_rate=0.0,
            average_latency=0.0,
            escalation_rate=0.0,
            total_runs=0,
        )
    weighted_success = sum(item.success_rate * item.total_runs for item in snapshots)
    weighted_latency = sum(item.average_latency * item.total_runs for item in snapshots)
    weighted_escalations = sum(item.escalation_rate * item.total_runs for item in snapshots)
    return EvaluationSnapshot(
        captured_at=max(item.captured_at for item in snapshots),
        success_rate=weighted_success / total_runs,
        average_latency=weighted_latency / total_runs,
        escalation_rate=weighted_escalations / total_runs,
        total_runs=total_runs,
    )
