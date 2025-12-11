from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Awaitable, Protocol

from ..services.llm import LLMService
from ..services.memory import HybridMemoryService
from ..services.scoring import ConfidenceScorer
from ..services.tools import ToolService
from ..schemas.agents import (
    AgentCapability,
    AgentInput,
    AgentOutput,
    ReasoningStep,
    ReasoningStepType,
    KeyFinding,
    ToolConsideration,
    ThinkingEvent,
    ThinkingEventType,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..services.retrieval import ContextAssembler
    from ..orchestration.tool_chain import ToolChainExecutor, ToolChainBuilder


class Tool(Protocol):
    name: str

    async def run(self, **kwargs: Any) -> Any:
        ...


# Type alias for the thinking event emitter callback
ThinkingEmitter = Callable[[ThinkingEvent], Awaitable[None]]


@dataclass
class AgentContext:
    """Context provided to agents during task execution."""
    memory: HybridMemoryService
    llm: LLMService
    context: "ContextAssembler | None" = None
    tools: ToolService | None = None
    scorer: ConfidenceScorer | None = None
    planned_tools: list[str] | None = None
    fallback_tools: list[str] | None = None
    planner_reason: str | None = None
    
    # NEW: Thinking stream emitter for real-time reasoning visibility
    thinking_emitter: ThinkingEmitter | None = None
    
    # NEW: Tool chain executor for multi-step tool pipelines
    _tool_chain_executor: "ToolChainExecutor | None" = None
    
    async def emit_thinking(
        self,
        agent: str,
        thought: str,
        event_type: ThinkingEventType = ThinkingEventType.THINKING,
        step_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a thinking event if an emitter is configured."""
        if self.thinking_emitter is None:
            return
        event = ThinkingEvent(
            event_type=event_type,
            agent=agent,
            thought=thought,
            step_index=step_index,
            metadata=metadata or {},
        )
        await self.thinking_emitter(event)
    
    def create_tool_chain(self, agent_name: str = "tool_chain") -> "ToolChainBuilder":
        """
        Create a new tool chain builder for multi-step tool pipelines.
        
        Example:
            chain = ctx.create_tool_chain("research")
            chain.add(
                "research.search",
                lambda c: {"query": c["prompt"]},
                description="Search for information"
            ).then(
                "research.summarize",
                lambda result: {"content": result},
                description="Summarize findings"
            )
            result = await ctx.execute_tool_chain(chain.build(), {"prompt": query})
        """
        from ..orchestration.tool_chain import chain
        return chain()
    
    async def execute_tool_chain(
        self,
        steps: list,  # list[ToolChainStep]
        initial_context: dict[str, Any],
        agent_name: str = "tool_chain",
    ):  # -> ToolChainResult
        """Execute a tool chain with thinking event emission."""
        from ..orchestration.tool_chain import ToolChainExecutor
        
        if self.tools is None:
            raise ValueError("ToolService not available in context for tool chain execution")
        
        executor = ToolChainExecutor(
            tool_service=self.tools,
            thinking_emitter=self.thinking_emitter,
            agent_name=agent_name,
        )
        return await executor.execute(steps, initial_context)
        await self.thinking_emitter(event)


class BaseAgent(Protocol):
    name: str
    capability: AgentCapability
    system_prompt: str
    description: str
    tool_preference: list[str]
    fallback_agent: str | None
    confidence_bias: float

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        ...


def get_agent_schema(agents: list[BaseAgent]) -> list[dict[str, Any]]:
    return [
        {
            "name": agent.name,
            "description": agent.description,
            "tools": list(agent.tool_preference),
            "fallback_agent": agent.fallback_agent,
            "confidence_bias": agent.confidence_bias,
        }
        for agent in agents
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Reasoning Builder Helper (NEW)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ReasoningBuilder:
    """Helper class to build reasoning traces during agent execution."""
    agent_name: str
    context: AgentContext
    steps: list[ReasoningStep] = field(default_factory=list)
    findings: list[KeyFinding] = field(default_factory=list)
    tools_considered: list[ToolConsideration] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    _step_counter: int = 0
    
    async def think(
        self,
        thought: str,
        step_type: ReasoningStepType = ReasoningStepType.ANALYSIS,
        evidence: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Record and emit a thinking step."""
        from datetime import datetime, timezone
        
        self._step_counter += 1
        step = ReasoningStep(
            step_type=step_type,
            thought=thought,
            evidence=evidence,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.steps.append(step)
        
        # Emit thinking event
        await self.context.emit_thinking(
            agent=self.agent_name,
            thought=thought,
            event_type=ThinkingEventType.THINKING,
            step_index=self._step_counter,
            metadata={
                "step_type": step_type.value,
                "has_evidence": evidence is not None,
            },
        )
    
    async def consider_tool(
        self,
        tool_name: str,
        reason: str,
        selected: bool = False,
        rejection_reason: str | None = None,
    ) -> None:
        """Record tool consideration and emit event."""
        consideration = ToolConsideration(
            tool_name=tool_name,
            reason=reason,
            selected=selected,
            rejection_reason=rejection_reason,
        )
        self.tools_considered.append(consideration)
        
        await self.context.emit_thinking(
            agent=self.agent_name,
            thought=f"Considering tool '{tool_name}': {reason}" + (
                f" → Selected" if selected else f" → Rejected: {rejection_reason}" if rejection_reason else ""
            ),
            event_type=ThinkingEventType.TOOL_DECIDING,
            metadata={
                "tool": tool_name,
                "selected": selected,
            },
        )
    
    async def add_finding(
        self,
        claim: str,
        evidence: list[str] | None = None,
        confidence: float = 0.5,
        source: str | None = None,
    ) -> None:
        """Record a key finding and emit event."""
        finding = KeyFinding(
            claim=claim,
            evidence=evidence or [],
            confidence=confidence,
            source=source,
        )
        self.findings.append(finding)
        
        await self.context.emit_thinking(
            agent=self.agent_name,
            thought=f"Finding: {claim}",
            event_type=ThinkingEventType.FINDING,
            metadata={
                "confidence": confidence,
                "has_evidence": bool(evidence),
                "source": source,
            },
        )
    
    async def note_uncertainty(self, uncertainty: str) -> None:
        """Record and emit an uncertainty."""
        self.uncertainties.append(uncertainty)
        
        await self.context.emit_thinking(
            agent=self.agent_name,
            thought=f"Uncertainty: {uncertainty}",
            event_type=ThinkingEventType.UNCERTAINTY,
            metadata={},
        )
