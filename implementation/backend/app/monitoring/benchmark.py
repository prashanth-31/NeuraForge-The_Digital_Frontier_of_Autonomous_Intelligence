from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable

from ..core.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class AgentEvaluationResult:
    agent: str
    success: bool
    confidence: float
    notes: str = ""


@dataclass
class BenchmarkSummary:
    total_cases: int
    success_rate: float
    avg_confidence: float
    details: list[AgentEvaluationResult] = field(default_factory=list)


def summarize_benchmark(results: Iterable[AgentEvaluationResult]) -> BenchmarkSummary:
    results_list = list(results)
    total_cases = len(results_list)
    successes = sum(1 for result in results_list if result.success)
    success_rate = successes / total_cases if total_cases else 0.0
    avg_confidence = mean(result.confidence for result in results_list) if results_list else 0.0
    summary = BenchmarkSummary(
        total_cases=total_cases,
        success_rate=success_rate,
        avg_confidence=avg_confidence,
        details=results_list,
    )
    logger.info(
        "benchmark_summary",
        total_cases=summary.total_cases,
        success_rate=summary.success_rate,
        avg_confidence=summary.avg_confidence,
    )
    return summary
