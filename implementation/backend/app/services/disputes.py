from __future__ import annotations

from dataclasses import dataclass
from statistics import StatisticsError, mean, median, pstdev
from typing import Iterable, Sequence


@dataclass(slots=True)
class MetaConfidenceInput:
    agent: str
    confidence: float
    weight: float = 1.0


@dataclass(slots=True)
class MetaConfidenceStats:
    mean: float
    median: float
    stddev: float
    weighted_mean: float
    max_delta: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "stddev": round(self.stddev, 4),
            "weighted_mean": round(self.weighted_mean, 4),
            "max_delta": round(self.max_delta, 4),
        }


@dataclass(slots=True)
class DisputeAssessment:
    flagged: bool
    stats: MetaConfidenceStats
    supporting_agents: list[str]
    dissenting_agents: list[str]
    threshold: float
    severity: str

    def as_dict(self) -> dict[str, object]:
        return {
            "flagged": self.flagged,
            "stats": self.stats.as_dict(),
            "supporting_agents": self.supporting_agents,
            "dissenting_agents": self.dissenting_agents,
            "threshold": round(self.threshold, 4),
            "severity": self.severity,
        }


class MetaConfidenceScorer:
    """Aggregate agent confidence scores with optional historical weighting."""

    def score(self, inputs: Sequence[MetaConfidenceInput]) -> MetaConfidenceStats:
        if not inputs:
            return MetaConfidenceStats(mean=0.0, median=0.0, stddev=0.0, weighted_mean=0.0, max_delta=0.0)

        values = [max(0.0, min(1.0, item.confidence)) for item in inputs]
        weights = [max(item.weight, 0.0) for item in inputs]
        total_weight = sum(weights) or 1.0
        weighted_mean = sum(value * weight for value, weight in zip(values, weights)) / total_weight

        try:
            spread = pstdev(values)
        except StatisticsError:  # pragma: no cover - degenerate single value
            spread = 0.0

        max_delta = max(abs(value - weighted_mean) for value in values) if values else 0.0
        return MetaConfidenceStats(
            mean=mean(values),
            median=median(values),
            stddev=spread,
            weighted_mean=weighted_mean,
            max_delta=max_delta,
        )


class DisputeDetector:
    """Heuristic detector for conflicting agent outputs."""

    def __init__(
        self,
        *,
        consensus_delta_threshold: float,
        stddev_threshold: float,
    ) -> None:
        self._consensus_delta = max(0.0, consensus_delta_threshold)
        self._stddev_threshold = max(0.0, stddev_threshold)
        self._scorer = MetaConfidenceScorer()

    def evaluate(self, inputs: Sequence[MetaConfidenceInput]) -> DisputeAssessment:
        stats = self._scorer.score(inputs)
        supporting, dissenting = self._partition(inputs, stats.weighted_mean)
        severity = self._classify_severity(stats)
        flagged = bool(dissenting) and (
            stats.max_delta >= self._consensus_delta or stats.stddev >= self._stddev_threshold
        )
        return DisputeAssessment(
            flagged=flagged,
            stats=stats,
            supporting_agents=supporting,
            dissenting_agents=dissenting,
            threshold=self._consensus_delta,
            severity=severity,
        )

    def _partition(
        self,
        inputs: Sequence[MetaConfidenceInput],
        anchor: float,
    ) -> tuple[list[str], list[str]]:
        supporting: list[str] = []
        dissenting: list[str] = []
        for item in inputs:
            delta = abs(item.confidence - anchor)
            if delta <= self._consensus_delta:
                supporting.append(item.agent)
            else:
                dissenting.append(item.agent)
        return supporting, dissenting

    def _classify_severity(self, stats: MetaConfidenceStats) -> str:
        if stats.max_delta >= self._consensus_delta * 2 or stats.stddev >= self._stddev_threshold * 1.5:
            return "high"
        if stats.max_delta >= self._consensus_delta or stats.stddev >= self._stddev_threshold:
            return "medium"
        return "low"


def build_inputs_from_outputs(outputs: Iterable[dict[str, object]]) -> list[MetaConfidenceInput]:
    inputs: list[MetaConfidenceInput] = []
    for entry in outputs:
        if not isinstance(entry, dict):
            continue
        agent = str(entry.get("agent", "unknown"))
        confidence = float(entry.get("confidence", 0.0) or 0.0)
        history = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        historical_weight = float(history.get("historical_weight", 1.0) or 1.0)
        inputs.append(MetaConfidenceInput(agent=agent, confidence=confidence, weight=max(historical_weight, 0.1)))
    return inputs
