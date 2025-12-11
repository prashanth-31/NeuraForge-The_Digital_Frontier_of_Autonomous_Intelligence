from __future__ import annotations

from enum import Enum
from typing import Any, Annotated, Literal

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


# ──────────────────────────────────────────────────────────────────────────────
# Reasoning & Thinking Structures (NEW)
# ──────────────────────────────────────────────────────────────────────────────


class ReasoningStepType(str, Enum):
    """Type of reasoning step an agent can perform."""
    OBSERVATION = "observation"       # What the agent sees/reads
    ANALYSIS = "analysis"            # Breaking down information
    HYPOTHESIS = "hypothesis"        # Forming a theory
    DECISION = "decision"            # Making a choice
    TOOL_SELECTION = "tool_selection"  # Deciding which tool to use
    EVALUATION = "evaluation"        # Assessing results
    SYNTHESIS = "synthesis"          # Combining findings
    UNCERTAINTY = "uncertainty"      # Acknowledging unknowns


class ReasoningStep(BaseModel):
    """A single step in the agent's chain of thought."""
    step_type: ReasoningStepType
    thought: str = Field(..., min_length=1, description="The agent's thought at this step")
    evidence: str | None = Field(default=None, description="Supporting evidence or data")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    timestamp: str | None = Field(default=None, description="ISO timestamp of when this step occurred")


class KeyFinding(BaseModel):
    """A key finding or insight from the agent's analysis."""
    claim: str = Field(..., description="The main assertion or finding")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str | None = Field(default=None, description="Where this finding came from")
    contradictions: list[str] = Field(default_factory=list, description="Conflicting information")


class ToolConsideration(BaseModel):
    """Record of a tool the agent considered using."""
    tool_name: str
    reason: str = Field(default="", description="Why this tool was considered")
    selected: bool = Field(default=False, description="Whether the tool was ultimately used")
    rejection_reason: str | None = Field(default=None, description="Why it wasn't selected, if applicable")


class AgentOutput(BaseModel):
    """Enhanced agent output with full reasoning transparency."""
    agent: str
    capability: AgentCapability
    summary: str = Field(..., min_length=1)
    confidence: Confidence
    rationale: str = Field(default="")
    evidence: list[EvidenceItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # NEW: Reasoning transparency fields
    reasoning_steps: list[ReasoningStep] = Field(
        default_factory=list,
        description="Chain of thought steps the agent took"
    )
    key_findings: list[KeyFinding] = Field(
        default_factory=list,
        description="Key insights and findings from analysis"
    )
    tools_considered: list[ToolConsideration] = Field(
        default_factory=list,
        description="Tools the agent evaluated for this task"
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Areas where the agent is uncertain or needs more data"
    )
    suggested_followup: str | None = Field(
        default=None,
        description="What the agent recommends as a next step"
    )
    handoff_request: dict[str, Any] | None = Field(
        default=None,
        description="Request to hand off to another agent with specific context"
    )


class AgentContractMetadata(BaseModel):
    name: str
    capability: AgentCapability
    description: str
    tools: list[str] = Field(default_factory=list)
    default_timeout_seconds: int = Field(default=60, ge=1)
    supports_streaming: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Thinking Stream Events (NEW)
# ──────────────────────────────────────────────────────────────────────────────


class ThinkingEventType(str, Enum):
    """Types of thinking events that can be streamed."""
    THINKING = "agent_thinking"
    PLANNING = "agent_planning"
    TOOL_DECIDING = "agent_tool_deciding"
    TOOL_PROGRESS = "agent_tool_progress"
    EVALUATING = "agent_evaluating"
    FINDING = "agent_finding"
    UNCERTAINTY = "agent_uncertainty"
    COLLABORATION = "agent_collaboration"


class ThinkingEvent(BaseModel):
    """Event emitted when an agent is thinking/reasoning."""
    event_type: ThinkingEventType
    agent: str
    thought: str
    step_index: int | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
