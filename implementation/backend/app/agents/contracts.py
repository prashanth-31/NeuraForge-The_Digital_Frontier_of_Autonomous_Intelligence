from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from pydantic import BaseModel, ValidationError

from ..schemas.agents import AgentCapability, AgentContractMetadata, AgentInput, AgentOutput


@dataclass(slots=True)
class AgentContract:
    metadata: AgentContractMetadata
    input_model: type[BaseModel]
    output_model: type[BaseModel]

    def validate_request(self, payload: dict[str, Any]) -> AgentInput:
        data = self.input_model.model_validate(payload)
        if not isinstance(data, AgentInput):
            # allow subclasses, coerce to AgentInput for downstream logic
            return AgentInput.model_validate(data.model_dump())
        return data

    def validate_response(self, payload: dict[str, Any]) -> AgentOutput:
        data = self.output_model.model_validate(payload)
        if not isinstance(data, AgentOutput):
            return AgentOutput.model_validate(data.model_dump())
        return data


def _build_contracts() -> Dict[AgentCapability, AgentContract]:
    return {
        AgentCapability.GENERAL: AgentContract(
            metadata=AgentContractMetadata(
                name="general_agent",
                capability=AgentCapability.GENERAL,
                description="Provides broad overviews, clarifications, and triages tasks for specialist follow-up.",
                tools=["summarizer"],
                default_timeout_seconds=40,
            ),
            input_model=AgentInput,
            output_model=AgentOutput,
        ),
        AgentCapability.RESEARCH: AgentContract(
            metadata=AgentContractMetadata(
                name="research_agent",
                capability=AgentCapability.RESEARCH,
                description="Synthesizes research findings with citations and knowledge gaps.",
                tools=["mcp://search/duckduckgo", "summarizer"],
                default_timeout_seconds=45,
            ),
            input_model=AgentInput,
            output_model=AgentOutput,
        ),
        AgentCapability.FINANCE: AgentContract(
            metadata=AgentContractMetadata(
                name="finance_agent",
                capability=AgentCapability.FINANCE,
                description="Provides financial analysis leveraging market data and compliance checks.",
                tools=["mcp://finance/yfinance", "calculator"],
                default_timeout_seconds=60,
            ),
            input_model=AgentInput,
            output_model=AgentOutput,
        ),
        AgentCapability.CREATIVE: AgentContract(
            metadata=AgentContractMetadata(
                name="creative_agent",
                capability=AgentCapability.CREATIVE,
                description="Generates stylized content with tone guidance and brand alignment.",
                tools=["mcp://creative/stylizer"],
                default_timeout_seconds=75,
                supports_streaming=True,
            ),
            input_model=AgentInput,
            output_model=AgentOutput,
        ),
        AgentCapability.ENTERPRISE: AgentContract(
            metadata=AgentContractMetadata(
                name="enterprise_agent",
                capability=AgentCapability.ENTERPRISE,
                description="Delivers strategic recommendations grounded in policy and knowledge base references.",
                tools=["mcp://enterprise/notion", "enterprise/policy_checker"],
                default_timeout_seconds=90,
            ),
            input_model=AgentInput,
            output_model=AgentOutput,
        ),
    }


_CONTRACTS = _build_contracts()


def get_contract(capability: AgentCapability) -> AgentContract:
    return _CONTRACTS[capability]


def list_contracts() -> list[AgentContractMetadata]:
    return [contract.metadata for contract in _CONTRACTS.values()]


def validate_agent_request(capability: AgentCapability, payload: dict[str, Any]) -> AgentInput:
    contract = get_contract(capability)
    try:
        return contract.validate_request(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid payload for capability {capability.value}") from exc


def validate_agent_response(capability: AgentCapability, payload: dict[str, Any]) -> AgentOutput:
    contract = get_contract(capability)
    try:
        result = contract.validate_response(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid response for capability {capability.value}") from exc
    if result.capability != capability:
        raise ValueError(
            f"Response capability {result.capability.value} does not match requested capability {capability.value}"
        )
    return result