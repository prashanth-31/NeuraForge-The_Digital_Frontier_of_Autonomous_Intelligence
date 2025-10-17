from __future__ import annotations

import httpx
import pytest

from app.core.metrics import (
    increment_guardrail_decision,
    observe_task_latency,
    record_review_oldest_ticket_age,
    record_review_ticket_counts,
)


@pytest.mark.asyncio
async def test_metrics_endpoint_includes_custom_series(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main

    monkeypatch.setattr(main.settings.observability, "prometheus_enabled", True, raising=False)

    observe_task_latency(entry_point="api", agent_count=7, latency=2.5)
    increment_guardrail_decision(decision="deny", policy_id="policy-ci")
    record_review_ticket_counts(open_count=4, in_review=1, resolved=0, dismissed=0, unassigned_open=2)
    record_review_oldest_ticket_age(seconds=7200.0)

    transport = httpx.ASGITransport(app=main.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/metrics")
    finally:
        await transport.aclose()

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/plain")

    body = response.text
    assert "neuraforge_task_latency_seconds_bucket" in body
    assert 'agent_count="5_plus"' in body
    assert "neuraforge_guardrail_decision_total" in body
    assert 'policy_id="policy-ci"' in body
    assert 'neuraforge_review_tickets{status="unassigned"}' in body
    assert "neuraforge_review_ticket_oldest_age_seconds" in body
