from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    task_id: str
    status: str


class TaskResult(BaseModel):
    task_id: str
    agent: str
    content: dict[str, Any]
    confidence: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
