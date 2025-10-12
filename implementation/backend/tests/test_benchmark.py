from app.monitoring.benchmark import AgentEvaluationResult, summarize_benchmark


def test_summarize_benchmark() -> None:
    results = [
        AgentEvaluationResult(agent="research", success=True, confidence=0.8),
        AgentEvaluationResult(agent="finance", success=False, confidence=0.6),
    ]
    summary = summarize_benchmark(results)
    assert summary.total_cases == 2
    assert summary.success_rate == 0.5
    assert summary.avg_confidence == 0.7
