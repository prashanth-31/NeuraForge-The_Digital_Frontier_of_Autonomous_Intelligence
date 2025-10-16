from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from ..orchestration.state import OrchestratorEvent, OrchestratorRun


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

    @classmethod
    def from_domain(
        cls,
        run: OrchestratorRun,
        events: Sequence[OrchestratorEvent],
    ) -> "OrchestratorRunDetail":
        return cls(
            **OrchestratorRunModel.from_domain(run).model_dump(),
            events=[OrchestratorEventModel.from_domain(event) for event in events],
        )


__all__ = [
    "OrchestratorEventModel",
    "OrchestratorRunModel",
    "OrchestratorRunDetail",
]
