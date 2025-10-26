from app.core.config import ScoringSettings
from app.services.scoring import ConfidenceScorer
from app.services.tools import ToolInvocationResult


DEFAULT_SCORING = ScoringSettings(
    base_confidence=0.6,
    evidence_weight=0.2,
    tool_reliability_weight=0.15,
    self_assessment_weight=0.15,
    max_evidence=5,
)


def _make_tool_result(*, cached: bool, latency: float) -> ToolInvocationResult:
    return ToolInvocationResult(
        tool="test_tool",
        resolved_tool="catalog/test_tool",
        payload={"query": "data"},
        response={"results": []},
        cached=cached,
        latency=latency,
    )


def test_confidence_increases_with_cached_tool():
    scorer = ConfidenceScorer(DEFAULT_SCORING)
    baseline = scorer.score(evidence_count=1, tool_result=None, self_assessment=0.5)
    cached = scorer.score(evidence_count=1, tool_result=_make_tool_result(cached=True, latency=0.4), self_assessment=0.5)
    assert cached.score > baseline.score


def test_confidence_penalizes_high_latency():
    scorer = ConfidenceScorer(DEFAULT_SCORING)
    fast = scorer.score(evidence_count=1, tool_result=_make_tool_result(cached=False, latency=0.4), self_assessment=0.5)
    slow = scorer.score(evidence_count=1, tool_result=_make_tool_result(cached=False, latency=8.0), self_assessment=0.5)
    assert slow.score < fast.score


def test_confidence_clamped_between_zero_and_one():
    settings = ScoringSettings(
        base_confidence=1.0,
        evidence_weight=1.0,
        tool_reliability_weight=1.0,
        self_assessment_weight=1.0,
        max_evidence=1,
    )
    scorer = ConfidenceScorer(settings)
    result = scorer.score(evidence_count=10, tool_result=_make_tool_result(cached=True, latency=0.1), self_assessment=1.0)
    assert 0.0 <= result.score <= 1.0