from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from ..orchestration.state import OrchestratorEvent, OrchestratorRun


class OrchestratorRunTelemetry(BaseModel):
    agents_started: int = 0
    agents_completed: int = 0
    agents_failed: int = 0
    guardrail_events: int = 0
    tool_invocations: int = 0
    negotiation_events: int = 0
    last_event_at: datetime | None = None
    total_events: int = 0


class OrchestratorEventModel(BaseModel):
    sequence: int
    event_type: str
    agent: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @classmethod
    def from_domain(cls, event: OrchestratorEvent) -> "OrchestratorEventModel":
        return cls(
            sequence=event.sequence,
            event_type=event.event_type,
            agent=event.agent,
            payload=dict(event.payload),
            created_at=event.created_at,
        )


class OrchestratorRunModel(BaseModel):
    run_id: UUID
    task_id: str
    status: str
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, run: OrchestratorRun) -> "OrchestratorRunModel":
        return cls(
            run_id=run.run_id,
            task_id=run.task_id,
            status=run.status.value,
            state=dict(run.state),
            created_at=run.created_at,
            updated_at=run.updated_at,
        )


class OrchestratorRunDetail(OrchestratorRunModel):
    events: list[OrchestratorEventModel] = Field(default_factory=list)
    telemetry: OrchestratorRunTelemetry = Field(default_factory=OrchestratorRunTelemetry)

    @classmethod
    def from_domain(
        cls,
        run: OrchestratorRun,
        events: Sequence[OrchestratorEvent],
    ) -> "OrchestratorRunDetail":
        telemetry = cls._build_telemetry(events)
        return cls(
            **OrchestratorRunModel.from_domain(run).model_dump(),
            events=[OrchestratorEventModel.from_domain(event) for event in events],
            telemetry=telemetry,
        )

    @staticmethod
    def _build_telemetry(events: Sequence[OrchestratorEvent]) -> OrchestratorRunTelemetry:
        agents_started = 0
        agents_completed = 0
        agents_failed = 0
        guardrail_events = 0
        tool_invocations = 0
        negotiation_events = 0
        last_event_at: datetime | None = None

        for event in events:
            if event.event_type == "agent_started":
                agents_started += 1
            elif event.event_type == "agent_completed":
                agents_completed += 1
            elif event.event_type == "agent_failed":
                agents_failed += 1
            elif event.event_type == "guardrail_triggered":
                guardrail_events += 1
            elif event.event_type in {"tool_invoked", "tool_invocation"}:
                tool_invocations += 1
            elif event.event_type.startswith("negotiation_"):
                negotiation_events += 1

            last_event_at = event.created_at

        return OrchestratorRunTelemetry(
            agents_started=agents_started,
            agents_completed=agents_completed,
            agents_failed=agents_failed,
            guardrail_events=guardrail_events,
            tool_invocations=tool_invocations,
            negotiation_events=negotiation_events,
            last_event_at=last_event_at,
            total_events=len(events),
        )


__all__ = [
    "OrchestratorEventModel",
    "OrchestratorRunModel",
    "OrchestratorRunDetail",
]
