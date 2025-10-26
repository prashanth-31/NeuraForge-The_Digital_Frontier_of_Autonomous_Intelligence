from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from app.core.metrics import (
    increment_guardrail_decision,
    observe_task_latency,
    record_review_oldest_ticket_age,
    record_review_ticket_counts,
)


def test_observe_task_latency_records_by_agent_bucket():
    labels = {"entry_point": "test-suite", "agent_count": "5_plus"}
    before = REGISTRY.get_sample_value("neuraforge_task_latency_seconds_sum", labels) or 0.0

    observe_task_latency(entry_point="test-suite", agent_count=8, latency=3.5)

    after = REGISTRY.get_sample_value("neuraforge_task_latency_seconds_sum", labels)
    assert after is not None
    assert after == pytest.approx(before + 3.5, rel=1e-6)


def test_guardrail_decision_metrics_increment_together():
    labels = {"decision": "deny", "policy_id": "policy-test"}
    legacy_before = REGISTRY.get_sample_value("neuraforge_guardrail_decisions_total", labels) or 0.0
    primary_before = REGISTRY.get_sample_value("neuraforge_guardrail_decision_total", labels) or 0.0

    increment_guardrail_decision(decision="deny", policy_id="policy-test")

    legacy_after = REGISTRY.get_sample_value("neuraforge_guardrail_decisions_total", labels)
    primary_after = REGISTRY.get_sample_value("neuraforge_guardrail_decision_total", labels)

    assert legacy_after is not None
    assert primary_after is not None
    assert legacy_after == pytest.approx(legacy_before + 1.0)
    assert primary_after == pytest.approx(primary_before + 1.0)


def test_record_review_ticket_counts_tracks_unassigned_bucket():
    record_review_ticket_counts(open_count=3, in_review=1, resolved=0, dismissed=0, unassigned_open=2)

    gauge_value = REGISTRY.get_sample_value(
        "neuraforge_review_tickets",
        {"status": "unassigned"},
    )
    assert gauge_value == pytest.approx(2.0)


def test_record_review_oldest_ticket_age_sets_gauge():
    record_review_oldest_ticket_age(seconds=180.0)

    observed = REGISTRY.get_sample_value("neuraforge_review_ticket_oldest_age_seconds")
    assert observed == pytest.approx(180.0)
