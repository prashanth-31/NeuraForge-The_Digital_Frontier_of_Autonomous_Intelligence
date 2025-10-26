from __future__ import annotations

import pytest

from app.agents.contracts import (
    AgentContract,
    get_contract,
    list_contracts,
    validate_agent_request,
    validate_agent_response,
)
from app.schemas.agents import AgentCapability, AgentInput, AgentOutput


def test_contract_registry_contains_all_capabilities() -> None:
    metadata = {item.capability for item in list_contracts()}
    assert metadata == {
        AgentCapability.RESEARCH,
        AgentCapability.FINANCE,
        AgentCapability.CREATIVE,
        AgentCapability.ENTERPRISE,
    }


def test_validate_agent_request_round_trip() -> None:
    payload = {
        "task_id": "task-1",
        "prompt": "Summarize the latest AI benchmarks",
        "metadata": {"priority": "high"},
        "prior_exchanges": [
            {"agent": "research_agent", "content": "Initial hypothesis", "confidence": 0.6},
        ],
    }
    result = validate_agent_request(AgentCapability.RESEARCH, payload)
    assert isinstance(result, AgentInput)
    assert result.task_id == "task-1"
    assert result.prior_exchanges[0].confidence == 0.6


def test_validate_agent_response_enforces_capability() -> None:
    payload = {
        "agent": "research_agent",
        "capability": AgentCapability.RESEARCH,
        "summary": "Concise report",
        "confidence": 0.8,
        "rationale": "Backed by citations",
        "evidence": [],
    }
    result = validate_agent_response(AgentCapability.RESEARCH, payload)
    assert isinstance(result, AgentOutput)
    assert result.summary == "Concise report"

    with pytest.raises(ValueError):
        validate_agent_response(AgentCapability.FINANCE, payload)


def test_get_contract_metadata_matches_registry() -> None:
    contract: AgentContract = get_contract(AgentCapability.CREATIVE)
    assert contract.metadata.name == "creative_agent"
    assert contract.metadata.supports_streaming is True
    assert "stylizer" in " ".join(contract.metadata.tools)
