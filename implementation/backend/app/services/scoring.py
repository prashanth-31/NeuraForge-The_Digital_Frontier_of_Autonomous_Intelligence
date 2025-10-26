from __future__ import annotations

from dataclasses import dataclass

from app.core.config import ScoringSettings
from app.services.tools import ToolInvocationResult


@dataclass(slots=True)
class ConfidenceBreakdown:
    base: float
    evidence: float
    tool_reliability: float
    self_assessment: float

    def as_dict(self) -> dict[str, float]:
        return {
            "base": self.base,
            "evidence": self.evidence,
            "tool_reliability": self.tool_reliability,
            "self_assessment": self.self_assessment,
        }


@dataclass(slots=True)
class ConfidenceResult:
    score: float
    breakdown: ConfidenceBreakdown

    def as_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            "score": self.score,
            "breakdown": self.breakdown.as_dict(),
        }


class ConfidenceScorer:
    """Blend multiple signals into a normalized confidence value."""

    def __init__(self, settings: ScoringSettings) -> None:
        self._settings = settings

    def score(
        self,
        *,
        evidence_count: int,
        tool_result: ToolInvocationResult | None,
        self_assessment: float | None,
    ) -> ConfidenceResult:
        evidence_ratio = min(max(evidence_count, 0) / self._settings.max_evidence, 1.0)
        tool_component = self._tool_reliability(tool_result)
        self_assessment_component = self._clamp(self_assessment if self_assessment is not None else 0.5)

        base = self._settings.base_confidence
        evidence_value = self._settings.evidence_weight * evidence_ratio
        tool_value = self._settings.tool_reliability_weight * tool_component
        self_assessment_value = self._settings.self_assessment_weight * self_assessment_component

        total = self._clamp(base + evidence_value + tool_value + self_assessment_value)
        breakdown = ConfidenceBreakdown(
            base=round(base, 4),
            evidence=round(evidence_value, 4),
            tool_reliability=round(tool_value, 4),
            self_assessment=round(self_assessment_value, 4),
        )
        return ConfidenceResult(score=round(total, 4), breakdown=breakdown)

    def _tool_reliability(self, tool_result: ToolInvocationResult | None) -> float:
        if tool_result is None:
            return 0.5

        cache_bonus = 0.2 if tool_result.cached else 0.0
        latency_score = self._latency_factor(tool_result.latency)
        blended = latency_score + cache_bonus
        return self._clamp(blended)

    @staticmethod
    def _latency_factor(latency: float | None) -> float:
        if latency is None:
            return 0.6
        # Latency <= 1s considered excellent, >= 6s progressively worse.
        normalized = 1.1 - (latency / 6.0)
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def _clamp(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value