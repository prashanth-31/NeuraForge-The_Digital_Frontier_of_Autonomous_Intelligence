from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class OrchestratorStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OrchestratorRun(BaseModel):
    run_id: UUID = Field(default_factory=uuid4)
    task_id: str = Field(min_length=1)
    status: OrchestratorStatus = Field(default=OrchestratorStatus.PENDING)
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OrchestratorEvent(BaseModel):
    run_id: UUID
    sequence: int = Field(ge=0)
    event_type: str = Field(min_length=1)
    agent: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def new_run(task_id: str, *, state: dict[str, Any]) -> OrchestratorRun:
    return OrchestratorRun(task_id=task_id, status=OrchestratorStatus.RUNNING, state=state)
