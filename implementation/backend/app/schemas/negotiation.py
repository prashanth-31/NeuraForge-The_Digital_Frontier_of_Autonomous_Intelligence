from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NegotiationEvidence(BaseModel):
    reference: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NegotiationProposal(BaseModel):
    agent: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None
    evidence: list[NegotiationEvidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NegotiationDecision(BaseModel):
    outcome: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    consensus: float = Field(ge=0.0, le=1.0)
    supporting_agents: list[str] = Field(default_factory=list)
    dissenting_agents: list[str] = Field(default_factory=list)
    rationale: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
