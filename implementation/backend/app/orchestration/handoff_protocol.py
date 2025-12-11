"""
Agent Handoff Protocol

This module implements structured agent-to-agent handoffs, allowing agents
to explicitly request that another agent take over a task with specific
context and instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from pydantic import BaseModel, Field

from ..core.logging import get_logger
from ..schemas.agents import (
    AgentOutput,
    ThinkingEvent,
    ThinkingEventType,
)

if TYPE_CHECKING:
    from ..agents.base import BaseAgent, AgentContext
    from .planner_contract import PlannedAgentStep

logger = get_logger(name=__name__)


class HandoffPriority(str, Enum):
    """Priority level for handoff requests."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class HandoffReason(str, Enum):
    """Standard reasons for agent handoffs."""
    CAPABILITY_MISMATCH = "capability_mismatch"
    EXPERTISE_REQUIRED = "expertise_required"
    TOOL_ACCESS_NEEDED = "tool_access_needed"
    SUBTASK_DELEGATION = "subtask_delegation"
    VERIFICATION_NEEDED = "verification_needed"
    COLLABORATION = "collaboration"
    ESCALATION = "escalation"


class HandoffRequest(BaseModel):
    """Structured request for agent handoff."""
    target_agent: str = Field(..., description="Name of the target agent")
    reason: HandoffReason = Field(default=HandoffReason.EXPERTISE_REQUIRED)
    priority: HandoffPriority = Field(default=HandoffPriority.NORMAL)
    context: dict[str, Any] = Field(default_factory=dict, description="Context to pass to target agent")
    instructions: str = Field(default="", description="Specific instructions for the target agent")
    return_expected: bool = Field(default=False, description="Whether the source expects results back")
    preserve_history: bool = Field(default=True, description="Whether to pass conversation history")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HandoffRequest | None":
        """Create a HandoffRequest from a dictionary (e.g., from AgentOutput.handoff_request)."""
        if not data:
            return None
        try:
            # Handle string reason/priority
            reason = data.get("reason", "expertise_required")
            if isinstance(reason, str) and reason not in [r.value for r in HandoffReason]:
                reason = HandoffReason.EXPERTISE_REQUIRED
            elif isinstance(reason, str):
                reason = HandoffReason(reason)
            
            priority = data.get("priority", "normal")
            if isinstance(priority, str) and priority not in [p.value for p in HandoffPriority]:
                priority = HandoffPriority.NORMAL
            elif isinstance(priority, str):
                priority = HandoffPriority(priority)
            
            return cls(
                target_agent=data.get("target_agent", data.get("target", "")),
                reason=reason,
                priority=priority,
                context=data.get("context", {}),
                instructions=data.get("instructions", ""),
                return_expected=data.get("return_expected", False),
                preserve_history=data.get("preserve_history", True),
            )
        except Exception as exc:
            logger.warning("handoff_request_parse_failed", error=str(exc), data=data)
            return None


@dataclass
class HandoffResult:
    """Result of executing a handoff."""
    success: bool
    source_agent: str
    target_agent: str
    target_output: AgentOutput | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffChain:
    """Tracks a chain of handoffs for debugging and transparency."""
    handoffs: list[tuple[str, str, HandoffReason]] = field(default_factory=list)
    max_depth: int = 5
    
    def add_handoff(self, source: str, target: str, reason: HandoffReason) -> bool:
        """Add a handoff to the chain. Returns False if max depth exceeded."""
        if len(self.handoffs) >= self.max_depth:
            return False
        self.handoffs.append((source, target, reason))
        return True
    
    def has_cycle(self, target: str) -> bool:
        """Check if adding this target would create a cycle."""
        return target in [h[0] for h in self.handoffs]
    
    @property
    def depth(self) -> int:
        return len(self.handoffs)
    
    def to_summary(self) -> str:
        """Create a readable summary of the handoff chain."""
        if not self.handoffs:
            return "No handoffs"
        parts = []
        for source, target, reason in self.handoffs:
            parts.append(f"{source} → {target} ({reason.value})")
        return " | ".join(parts)


class HandoffProtocol:
    """
    Manages agent-to-agent handoffs with proper context passing and tracking.
    
    The protocol ensures:
    1. Proper validation of handoff requests
    2. Context preservation and enhancement
    3. Cycle detection and depth limiting
    4. Visibility through thinking events
    5. Result aggregation
    """
    
    def __init__(
        self,
        agents: dict[str, "BaseAgent"],
        thinking_emitter: Callable[[ThinkingEvent], Awaitable[None]] | None = None,
        max_handoff_depth: int = 5,
    ):
        self.agents = agents
        self.thinking_emitter = thinking_emitter
        self.max_handoff_depth = max_handoff_depth
        self._active_chains: dict[str, HandoffChain] = {}
    
    async def _emit_thinking(
        self,
        thought: str,
        event_type: ThinkingEventType = ThinkingEventType.COLLABORATION,
        source_agent: str | None = None,
        target_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a thinking event for handoff visibility."""
        if self.thinking_emitter is None:
            return
        event = ThinkingEvent(
            event_type=event_type,
            agent=source_agent or "orchestrator",
            thought=thought,
            metadata={
                **(metadata or {}),
                "target_agent": target_agent,
            },
        )
        await self.thinking_emitter(event)
    
    def validate_handoff(
        self,
        request: HandoffRequest,
        source_agent: str,
        chain_id: str,
    ) -> tuple[bool, str]:
        """
        Validate that a handoff request is acceptable.
        
        Returns (is_valid, error_message)
        """
        # Check target exists
        if request.target_agent not in self.agents:
            return False, f"Target agent '{request.target_agent}' not found"
        
        # Check for self-handoff
        if request.target_agent == source_agent:
            return False, "Cannot hand off to self"
        
        # Get or create chain
        chain = self._active_chains.get(chain_id, HandoffChain(max_depth=self.max_handoff_depth))
        
        # Check for cycles
        if chain.has_cycle(request.target_agent):
            return False, f"Handoff would create cycle: {request.target_agent} already in chain"
        
        # Check depth
        if chain.depth >= self.max_handoff_depth:
            return False, f"Maximum handoff depth ({self.max_handoff_depth}) exceeded"
        
        return True, ""
    
    async def prepare_handoff_context(
        self,
        request: HandoffRequest,
        source_output: AgentOutput,
        original_input: dict[str, Any],
        chain_id: str,
    ) -> dict[str, Any]:
        """
        Prepare the context for a handoff, enriching with relevant information.
        """
        chain = self._active_chains.get(chain_id, HandoffChain())
        
        context = {
            "original_prompt": original_input.get("prompt", ""),
            "handoff_instructions": request.instructions,
            "source_agent": source_output.agent,
            "source_summary": source_output.summary,
            "source_confidence": source_output.confidence,
            "handoff_reason": request.reason.value,
            "handoff_chain": chain.to_summary(),
            "chain_depth": chain.depth,
        }
        
        # Include source findings if available
        if source_output.key_findings:
            context["source_findings"] = [
                {"claim": f.claim, "confidence": f.confidence}
                for f in source_output.key_findings[:5]
            ]
        
        # Include source uncertainties
        if source_output.uncertainties:
            context["source_uncertainties"] = source_output.uncertainties[:3]
        
        # Merge in custom context from request
        if request.context:
            context["custom_context"] = request.context
        
        # Preserve history if requested
        if request.preserve_history and "messages" in original_input:
            context["conversation_history"] = original_input["messages"]
        
        return context
    
    async def execute_handoff(
        self,
        request: HandoffRequest,
        source_output: AgentOutput,
        original_input: dict[str, Any],
        context: "AgentContext",
        chain_id: str,
    ) -> HandoffResult:
        """
        Execute a handoff to the target agent.
        
        Args:
            request: The handoff request
            source_output: Output from the source agent
            original_input: Original task input
            context: Agent context for execution
            chain_id: Unique ID for this handoff chain
            
        Returns:
            HandoffResult with target agent's output
        """
        start_time = datetime.now(timezone.utc)
        source_agent = source_output.agent
        target_agent_name = request.target_agent
        
        # Validate
        is_valid, error_msg = self.validate_handoff(request, source_agent, chain_id)
        if not is_valid:
            return HandoffResult(
                success=False,
                source_agent=source_agent,
                target_agent=target_agent_name,
                error=error_msg,
            )
        
        # Get or create chain and add this handoff
        if chain_id not in self._active_chains:
            self._active_chains[chain_id] = HandoffChain(max_depth=self.max_handoff_depth)
        chain = self._active_chains[chain_id]
        chain.add_handoff(source_agent, target_agent_name, request.reason)
        
        await self._emit_thinking(
            f"Executing handoff from {source_agent} to {target_agent_name}: {request.reason.value}",
            source_agent=source_agent,
            target_agent=target_agent_name,
            metadata={"chain_depth": chain.depth, "instructions": request.instructions[:100]},
        )
        
        # Prepare enhanced context for target
        handoff_context = await self.prepare_handoff_context(
            request, source_output, original_input, chain_id
        )
        
        # Get target agent
        target_agent = self.agents[target_agent_name]
        
        # Prepare input for target agent
        from ..schemas.agents import AgentInput
        target_input = AgentInput(
            prompt=original_input.get("prompt", ""),
            context=[
                f"HANDOFF FROM {source_agent}:",
                f"Reason: {request.reason.value}",
                f"Instructions: {request.instructions}",
                f"Previous summary: {source_output.summary}",
                f"Chain: {chain.to_summary()}",
            ],
            tools=context.planned_tools or [],
        )
        
        # Execute target agent
        try:
            target_output = await target_agent.handle(target_input, context=context)
            
            end_time = datetime.now(timezone.utc)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            await self._emit_thinking(
                f"Handoff to {target_agent_name} completed with confidence {target_output.confidence:.2f}",
                source_agent=target_agent_name,
                metadata={"confidence": target_output.confidence, "execution_time_ms": execution_time_ms},
            )
            
            return HandoffResult(
                success=True,
                source_agent=source_agent,
                target_agent=target_agent_name,
                target_output=target_output,
                execution_time_ms=execution_time_ms,
                metadata={
                    "chain_depth": chain.depth,
                    "reason": request.reason.value,
                },
            )
            
        except Exception as exc:
            logger.exception("handoff_execution_failed", source=source_agent, target=target_agent_name)
            end_time = datetime.now(timezone.utc)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            return HandoffResult(
                success=False,
                source_agent=source_agent,
                target_agent=target_agent_name,
                error=str(exc),
                execution_time_ms=execution_time_ms,
            )
    
    def clear_chain(self, chain_id: str) -> None:
        """Clear a handoff chain after completion."""
        self._active_chains.pop(chain_id, None)
    
    def get_chain_summary(self, chain_id: str) -> str:
        """Get a summary of a handoff chain."""
        chain = self._active_chains.get(chain_id)
        if chain:
            return chain.to_summary()
        return "No active chain"


def create_handoff_request(
    target: str,
    reason: HandoffReason = HandoffReason.EXPERTISE_REQUIRED,
    instructions: str = "",
    context: dict[str, Any] | None = None,
    priority: HandoffPriority = HandoffPriority.NORMAL,
) -> dict[str, Any]:
    """
    Helper function for agents to create handoff request dictionaries.
    
    Example usage in an agent:
        output.handoff_request = create_handoff_request(
            target="finance_agent",
            reason=HandoffReason.TOOL_ACCESS_NEEDED,
            instructions="Analyze the financial metrics for AAPL",
            context={"symbol": "AAPL", "metrics": ["PE", "EPS"]},
        )
    """
    return {
        "target_agent": target,
        "reason": reason.value,
        "priority": priority.value,
        "context": context or {},
        "instructions": instructions,
        "return_expected": True,
        "preserve_history": True,
    }
