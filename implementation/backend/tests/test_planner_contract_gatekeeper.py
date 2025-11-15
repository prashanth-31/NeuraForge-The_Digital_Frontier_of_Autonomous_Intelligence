import pytest

from app.orchestration.planner_contract import PlanGatekeeper, PlannerContractViolation


def _valid_payload() -> dict:
    return {
        "steps": [
            {
                "agent": "general_agent",
                "reason": "triage request",
                "tools": ["general.answer"],
                "fallback_tools": ["general.rephrase"],
                "confidence": 0.9,
            }
        ],
        "metadata": {"handoff_strategy": "sequential"},
        "confidence": 0.75,
    }


def test_gatekeeper_accepts_valid_plan() -> None:
    gatekeeper = PlanGatekeeper()
    plan = gatekeeper.enforce(_valid_payload(), raw_response="{}")

    assert plan.steps[0].agent == "general_agent"
    assert plan.metadata["contract_version"].startswith("planner.contract")
    assert 0.0 <= plan.confidence <= 1.0


def test_gatekeeper_rejects_missing_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    gatekeeper = PlanGatekeeper()
    payload = {"metadata": {}}
    metric: dict[str, str] = {}

    def _capture_metric(*, reason: str) -> None:
        metric["reason"] = reason

    monkeypatch.setattr(
        "app.orchestration.planner_contract.increment_plan_contract_failure",
        _capture_metric,
    )

    with pytest.raises(PlannerContractViolation):
        gatekeeper.enforce(payload, raw_response="{}")

    assert metric["reason"].startswith("steps")


def test_gatekeeper_rejects_invalid_tool_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    gatekeeper = PlanGatekeeper()
    payload = _valid_payload()
    payload["steps"][0]["tools"] = [{"tool": "general.answer"}]
    metric: dict[str, str] = {}

    def _capture_metric(*, reason: str) -> None:
        metric["reason"] = reason

    monkeypatch.setattr(
        "app.orchestration.planner_contract.increment_plan_contract_failure",
        _capture_metric,
    )

    with pytest.raises(PlannerContractViolation):
        gatekeeper.enforce(payload, raw_response="{}")

    assert "tools" in metric["reason"]


def test_gatekeeper_rejects_blank_tool_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    gatekeeper = PlanGatekeeper()
    payload = _valid_payload()
    payload["steps"][0]["tools"] = ["  "]
    metric: dict[str, str] = {}

    def _capture_metric(*, reason: str) -> None:
        metric["reason"] = reason

    monkeypatch.setattr(
        "app.orchestration.planner_contract.increment_plan_contract_failure",
        _capture_metric,
    )

    with pytest.raises(PlannerContractViolation):
        gatekeeper.enforce(payload, raw_response="{}")

    assert metric["reason"].startswith("steps")
