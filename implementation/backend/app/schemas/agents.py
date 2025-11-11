from __future__ import annotations

from enum import Enum
from typing import Any, Annotated

from pydantic import BaseModel, Field, HttpUrl, confloat


class AgentCapability(str, Enum):
    GENERAL = "general"
    RESEARCH = "research"
    FINANCE = "finance"
    CREATIVE = "creative"
    ENTERPRISE = "enterprise"


class AgentExchange(BaseModel):
    agent: str
    type: str = Field(default="statement")
    content: str = Field(..., min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AgentInput(BaseModel):
    task_id: str
    prompt: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    context: str | None = None
    prior_exchanges: list[AgentExchange] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    label: str
    url: HttpUrl | None = None
    notes: str | None = None


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class AgentOutput(BaseModel):
    agent: str
    capability: AgentCapability
    summary: str = Field(..., min_length=1)
    confidence: Confidence
    rationale: str = Field(default="")
    evidence: list[EvidenceItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentContractMetadata(BaseModel):
    name: str
    capability: AgentCapability
    description: str
    tools: list[str] = Field(default_factory=list)
    default_timeout_seconds: int = Field(default=60, ge=1)
    supports_streaming: bool = False
