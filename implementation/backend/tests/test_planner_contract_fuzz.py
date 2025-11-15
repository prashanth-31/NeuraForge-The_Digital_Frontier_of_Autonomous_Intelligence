from __future__ import annotations

import copy
import random
from typing import Any, Callable

import pytest

from app.orchestration.planner_contract import PlanGatekeeper, PlannerContractViolation


def _valid_plan() -> dict[str, Any]:
    return {
        "steps": [
            {
                "agent": "general_agent",
                "reason": "triage",
                "tools": ["general.answer"],
                "fallback_tools": ["general.rephrase"],
                "confidence": 0.9,
            },
            {
                "agent": "research_agent",
                "reason": "gather context",
                "tools": ["research.search"],
                "fallback_tools": ["research.doc_loader"],
                "confidence": 0.82,
            },
        ],
        "metadata": {"handoff_strategy": "sequential"},
        "confidence": 0.8,
    }


def _invalid_mutations() -> list[Callable[[dict[str, Any], random.Random], None]]:
    return [
        lambda payload, _: payload.pop("steps", None),
        lambda payload, _: payload["steps"].append("not-a-step"),
        lambda payload, _: payload["steps"][0].pop("agent", None),
        lambda payload, _: payload["steps"][0].update({"tools": [{"id": "bad"}]}),
        lambda payload, _: payload["steps"][1].update({"fallback_tools": [""]}),
        lambda payload, _: payload.update({"confidence": 3.5}),
        lambda payload, rng: payload["steps"][0].update({"confidence": rng.uniform(-2, -0.1)}),
        lambda payload, _: payload.setdefault("unexpected", {"nested": {"value": object()}}),
        lambda payload, _: payload["metadata"].update({"handoff_strategy": 42}),
    ]


def _valid_mutations() -> list[Callable[[dict[str, Any], random.Random], None]]:
    return [
        lambda payload, rng: payload["steps"].append(
            {
                "agent": "finance_agent",
                "reason": f"financial review {rng.randint(1, 99)}",
                "tools": ["finance.snapshot"],
                "fallback_tools": [
                    "finance.snapshot.alpha",
                    "finance.snapshot.cached",
                    "finance.news",
                ],
                "confidence": 0.75,
            }
        ),
        lambda payload, rng: payload["steps"][0].update({"reason": f"triage-{rng.randint(10, 99)}"}),
        lambda payload, rng: payload["metadata"].update({"handoff_strategy": "sequential", "seed": rng.randint(0, 1000)}),
        lambda payload, _: payload["steps"][1].setdefault("notes", "supplemental context"),
    ]


@pytest.mark.parametrize("seed", [11, 29, 1337])
def test_gatekeeper_rejects_fuzzed_invalid_payloads(seed: int) -> None:
    gatekeeper = PlanGatekeeper()
    rng = random.Random(seed)
    mutations = _invalid_mutations()

    for _ in range(50):
        payload = copy.deepcopy(_valid_plan())
        mutation = rng.choice(mutations)
        mutation(payload, rng)
        with pytest.raises(PlannerContractViolation):
            gatekeeper.enforce(payload, raw_response="{}")


@pytest.mark.parametrize("seed", [3, 7, 101])
def test_gatekeeper_accepts_randomized_valid_payloads(seed: int) -> None:
    gatekeeper = PlanGatekeeper()
    rng = random.Random(seed)
    mutations = _valid_mutations()

    for _ in range(30):
        payload = copy.deepcopy(_valid_plan())
        for _ in range(rng.randint(0, len(mutations))):
            mutation = rng.choice(mutations)
            mutation(payload, rng)
        plan = gatekeeper.enforce(payload, raw_response="{}")
        assert plan.steps
        assert 0.0 <= plan.confidence <= 1.0
        assert plan.metadata["contract_version"].startswith("planner.contract")
