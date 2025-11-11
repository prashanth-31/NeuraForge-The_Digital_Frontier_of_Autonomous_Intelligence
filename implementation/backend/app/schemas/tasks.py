from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .orchestrator import OrchestratorEventModel


class TaskRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    continuation_task_id: str | None = Field(default=None)


class TaskResponse(BaseModel):
    task_id: str
    status: str


class TaskResult(BaseModel):
    task_id: str
    agent: str
    content: dict[str, Any]
    confidence: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TaskStatusMetrics(BaseModel):
    agents_completed: int = 0
    agents_failed: int = 0
    guardrail_events: int = 0
    negotiation_rounds: int | None = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    run_id: str | None = None
    prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    negotiation: dict[str, Any] | None = None
    guardrails: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    dossier: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    metrics: TaskStatusMetrics = Field(default_factory=TaskStatusMetrics)
    last_error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    events: list[OrchestratorEventModel] = Field(default_factory=list)
