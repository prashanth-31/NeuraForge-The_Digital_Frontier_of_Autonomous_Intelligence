"""
Dynamic Re-Planning Module

This module provides capability for the orchestrator to dynamically 
re-plan execution when agents report low confidence, new information
emerges, or execution conditions change.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Awaitable, Sequence

from ..core.logging import get_logger
from ..schemas.agents import (
    AgentOutput,
    ThinkingEvent,
    ThinkingEventType,
)
from .planner_contract import PlannedAgentStep, PlannerPlan

if TYPE_CHECKING:
    from ..agents.base import BaseAgent
    from .llm_planner import LLMOrchestrationPlanner

logger = get_logger(name=__name__)


class ReplanTrigger(str, Enum):
    """Reasons for triggering a re-plan."""
    LOW_CONFIDENCE = "low_confidence"
    AGENT_REQUEST = "agent_request"
    NEW_INFORMATION = "new_information"
    TOOL_FAILURE = "tool_failure"
    TIMEOUT = "timeout"
    USER_INTERRUPT = "user_interrupt"


@dataclass
class ReplanContext:
    """Context for re-planning decisions."""
    trigger: ReplanTrigger
    original_plan: PlannerPlan
    completed_steps: list[tuple[PlannedAgentStep, AgentOutput]]
    remaining_steps: list[PlannedAgentStep]
    current_step_index: int
    failure_info: dict[str, Any] | None = None
    agent_suggestion: str | None = None
    new_information: str | None = None


@dataclass
class ReplanResult:
    """Result of a re-planning operation."""
    should_replan: bool
    new_steps: list[PlannedAgentStep] | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ReplanPolicy:
    """
    Policy governing when and how re-planning should occur.
    
    This can be configured per-deployment to control the aggressiveness
    of dynamic re-planning.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.4,
        max_replans: int = 3,
        enable_tool_failure_replan: bool = True,
        enable_agent_suggestion_replan: bool = True,
        cooldown_seconds: float = 2.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_replans = max_replans
        self.enable_tool_failure_replan = enable_tool_failure_replan
        self.enable_agent_suggestion_replan = enable_agent_suggestion_replan
        self.cooldown_seconds = cooldown_seconds


class DynamicReplanner:
    """
    Handles dynamic re-planning during orchestration execution.
    
    The replanner monitors agent outputs and can trigger plan adjustments
    when confidence is low, tools fail, or agents explicitly request re-planning.
    """
    
    def __init__(
        self,
        planner: "LLMOrchestrationPlanner",
        agents: Sequence["BaseAgent"],
        policy: ReplanPolicy | None = None,
        thinking_emitter: Callable[[ThinkingEvent], Awaitable[None]] | None = None,
    ):
        self.planner = planner
        self.agents = agents
        self.policy = policy or ReplanPolicy()
        self.thinking_emitter = thinking_emitter
        self._replan_count = 0
        self._last_replan_time: datetime | None = None
    
    async def _emit_thinking(
        self,
        thought: str,
        event_type: ThinkingEventType = ThinkingEventType.PLANNING,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a thinking event for re-planning visibility."""
        if self.thinking_emitter is None:
            return
        event = ThinkingEvent(
            event_type=event_type,
            agent="orchestrator",
            thought=thought,
            metadata=metadata or {},
        )
        await self.thinking_emitter(event)
    
    def should_consider_replan(
        self,
        agent_output: AgentOutput,
        current_step: PlannedAgentStep,
    ) -> tuple[bool, ReplanTrigger | None]:
        """
        Determine if re-planning should be considered based on agent output.
        
        Returns (should_consider, trigger_reason)
        """
        # Check if we've exceeded max replans
        if self._replan_count >= self.policy.max_replans:
            logger.debug("replan_limit_reached", count=self._replan_count)
            return False, None
        
        # Check cooldown
        if self._last_replan_time:
            elapsed = (datetime.now(timezone.utc) - self._last_replan_time).total_seconds()
            if elapsed < self.policy.cooldown_seconds:
                logger.debug("replan_cooldown_active", elapsed=elapsed)
                return False, None
        
        # Check for low confidence
        if agent_output.confidence < self.policy.confidence_threshold:
            return True, ReplanTrigger.LOW_CONFIDENCE
        
        # Check for explicit re-plan request
        if (
            self.policy.enable_agent_suggestion_replan
            and agent_output.handoff_request
            and "replan" in agent_output.handoff_request.lower()
        ):
            return True, ReplanTrigger.AGENT_REQUEST
        
        # Check for uncertainties that might warrant re-planning
        if agent_output.uncertainties and len(agent_output.uncertainties) >= 3:
            return True, ReplanTrigger.NEW_INFORMATION
        
        return False, None
    
    def should_replan_on_tool_failure(
        self,
        tool_name: str,
        error: Exception,
        attempts: int,
    ) -> tuple[bool, ReplanTrigger | None]:
        """Determine if re-planning should occur after tool failure."""
        if not self.policy.enable_tool_failure_replan:
            return False, None
        
        if self._replan_count >= self.policy.max_replans:
            return False, None
        
        # Critical tools that warrant re-planning on failure
        critical_patterns = ["finance.", "research.search", "database."]
        is_critical = any(pattern in tool_name for pattern in critical_patterns)
        
        if is_critical and attempts >= 2:
            return True, ReplanTrigger.TOOL_FAILURE
        
        return False, None
    
    async def create_replan_context(
        self,
        trigger: ReplanTrigger,
        original_plan: PlannerPlan,
        completed_steps: list[tuple[PlannedAgentStep, AgentOutput]],
        remaining_steps: list[PlannedAgentStep],
        current_step_index: int,
        failure_info: dict[str, Any] | None = None,
        agent_output: AgentOutput | None = None,
    ) -> ReplanContext:
        """Build context for re-planning decision."""
        agent_suggestion = None
        new_information = None
        
        if agent_output:
            # Extract any suggested next steps from agent
            if agent_output.suggested_followup:
                agent_suggestion = "; ".join(agent_output.suggested_followup)
            
            # Extract new information from key findings
            if agent_output.key_findings:
                new_claims = [f.claim for f in agent_output.key_findings[:3]]
                new_information = "; ".join(new_claims)
        
        return ReplanContext(
            trigger=trigger,
            original_plan=original_plan,
            completed_steps=completed_steps,
            remaining_steps=remaining_steps,
            current_step_index=current_step_index,
            failure_info=failure_info,
            agent_suggestion=agent_suggestion,
            new_information=new_information,
        )
    
    async def evaluate_and_replan(
        self,
        context: ReplanContext,
        task: dict[str, Any],
    ) -> ReplanResult:
        """
        Evaluate the context and produce a re-plan if warranted.
        
        Args:
            context: The re-planning context
            task: The original task
            
        Returns:
            ReplanResult with new steps if re-planning occurred
        """
        await self._emit_thinking(
            f"Evaluating re-plan triggered by {context.trigger.value}",
            event_type=ThinkingEventType.EVALUATING,
            metadata={"trigger": context.trigger.value, "step_index": context.current_step_index},
        )
        
        # Build enhanced context for the planner
        completed_summaries = []
        for step, output in context.completed_steps:
            summary = {
                "agent": step.agent,
                "confidence": output.confidence,
                "findings": len(output.key_findings or []),
                "uncertainties": len(output.uncertainties or []),
            }
            completed_summaries.append(summary)
        
        # Construct modified task with execution context
        enhanced_task = dict(task)
        enhanced_task["_replan_context"] = {
            "trigger": context.trigger.value,
            "completed_steps": completed_summaries,
            "remaining_agents": [s.agent for s in context.remaining_steps],
            "current_step_index": context.current_step_index,
            "agent_suggestion": context.agent_suggestion,
            "new_information": context.new_information,
            "failure_info": context.failure_info,
        }
        
        # Request new plan from planner
        try:
            await self._emit_thinking(
                "Requesting new execution plan based on current context",
                event_type=ThinkingEventType.PLANNING,
            )
            
            prior_outputs = [
                {
                    "agent": step.agent,
                    "answer": output.answer,
                    "confidence": output.confidence,
                }
                for step, output in context.completed_steps
            ]
            
            new_plan = await self.planner.plan(
                task=enhanced_task,
                prior_outputs=prior_outputs,
                agents=self.agents,
            )
            
            # Update tracking
            self._replan_count += 1
            self._last_replan_time = datetime.now(timezone.utc)
            
            await self._emit_thinking(
                f"Re-plan complete: {len(new_plan.steps)} new steps planned",
                event_type=ThinkingEventType.PLANNING,
                metadata={"new_steps": len(new_plan.steps), "replan_count": self._replan_count},
            )
            
            return ReplanResult(
                should_replan=True,
                new_steps=new_plan.steps,
                reason=f"Re-planned due to {context.trigger.value}",
                metadata={
                    "replan_count": self._replan_count,
                    "trigger": context.trigger.value,
                    "original_remaining": len(context.remaining_steps),
                    "new_steps": len(new_plan.steps),
                },
            )
            
        except Exception as exc:
            logger.warning(
                "replan_failed",
                error=str(exc),
                trigger=context.trigger.value,
            )
            await self._emit_thinking(
                f"Re-plan failed: {exc}. Continuing with original plan.",
                event_type=ThinkingEventType.UNCERTAINTY,
            )
            return ReplanResult(
                should_replan=False,
                reason=f"Re-plan failed: {exc}",
            )
    
    def reset(self) -> None:
        """Reset re-plan tracking for a new orchestration run."""
        self._replan_count = 0
        self._last_replan_time = None


@dataclass
class ReplanCheckpoint:
    """
    Checkpoint for saving/restoring orchestration state during re-planning.
    
    This allows the orchestrator to safely attempt re-planning while
    maintaining ability to resume from a known good state.
    """
    step_index: int
    completed_outputs: list[AgentOutput]
    plan_snapshot: PlannerPlan
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def can_resume_from(self, current_index: int) -> bool:
        """Check if this checkpoint is valid for resumption."""
        return self.step_index <= current_index
