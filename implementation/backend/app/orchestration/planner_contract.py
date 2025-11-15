from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..core.logging import get_logger
from ..core.metrics import increment_plan_contract_failure

__all__ = [
    "PlanGatekeeper",
    "PlannerContractViolation",
    "PlannerPlan",
    "PlannedAgentStep",
    "PlannerPlanPayload",
    "PlannerStepPayload",
]

logger = get_logger(name=__name__)


class PlannerContractViolation(RuntimeError):
    """Raised when a planner payload violates the contract."""


def _validate_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("confidence must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError("confidence must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return numeric


class PlannerStepPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    agent: str = Field(..., min_length=1)
    reason: str = Field(default="")
    tools: list[str] = Field(default_factory=list)
    fallback_tools: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None)

    @field_validator("tools", "fallback_tools", mode="before")
    @classmethod
    def _coerce_tool_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("tool collections must be lists of strings")
        return list(value)

    @field_validator("tools", "fallback_tools")
    @classmethod
    def _validate_tools(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("tool names must be strings")
            trimmed = item.strip()
            if not trimmed:
                raise ValueError("tool names cannot be blank")
            normalized.append(trimmed)
        return normalized

    @field_validator("confidence")
    @classmethod
    def _validate_step_confidence(cls, value: Any) -> float | None:
        return _validate_confidence(value)


class PlannerPlanPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[PlannerStepPayload] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("metadata must be an object")
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            sanitized[key] = item
        return sanitized

    @field_validator("confidence")
    @classmethod
    def _validate_plan_confidence(cls, value: Any) -> float | None:
        return _validate_confidence(value)


@dataclass(slots=True)
class PlannedAgentStep:
    agent: str
    tools: list[str] = field(default_factory=list)
    fallback_tools: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 1.0


@dataclass(slots=True)
class PlannerPlan:
    steps: list[PlannedAgentStep]
    raw_response: str
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


def _metric_reason_from_error(error: ValidationError) -> str:
    details = error.errors()
    if not details:
        return "unknown"
    first = details[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    err_type = first.get("type", "validation")
    reason = f"{location}:{err_type}" if location else err_type
    return reason.replace(":", "_").replace(" ", "_") or "unknown"


class PlanGatekeeper:
    """Validates planner payloads against the shared contract."""

    def __init__(self, *, contract_version: str = "planner.contract.v1") -> None:
        self._contract_version = contract_version

    def enforce(self, payload: Mapping[str, Any], *, raw_response: str) -> PlannerPlan:
        try:
            plan_model = PlannerPlanPayload.model_validate(payload)
        except ValidationError as exc:
            reason = _metric_reason_from_error(exc)
            increment_plan_contract_failure(reason=reason)
            logger.warning("planner_contract_rejected", reason=reason, errors=exc.errors())
            raise PlannerContractViolation("Planner payload failed contract validation") from exc

        steps = [
            PlannedAgentStep(
                agent=step.agent,
                tools=list(step.tools),
                fallback_tools=list(step.fallback_tools),
                reason=step.reason,
                confidence=step.confidence if step.confidence is not None else 1.0,
            )
            for step in plan_model.steps
        ]

        metadata = dict(plan_model.metadata)
        metadata.setdefault("contract_version", self._contract_version)
        metadata.setdefault("validated_at", datetime.now(tz=timezone.utc).isoformat())

        confidence = plan_model.confidence if plan_model.confidence is not None else 1.0

        return PlannerPlan(
            steps=steps,
            raw_response=raw_response,
            metadata=metadata,
            confidence=confidence,
        )