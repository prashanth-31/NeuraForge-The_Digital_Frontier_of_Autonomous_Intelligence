from __future__ import annotations

import asyncio
import copy
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping

try:
    from langgraph.graph import StateGraph
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    StateGraph = None  # type: ignore[misc,assignment]

from ..agents.base import AgentContext, BaseAgent
from ..agents.contracts import validate_agent_request, validate_agent_response
from ..core.config import PlanningSettings, TOOL_ENFORCEMENT_POLICY, get_settings
from ..core.logging import get_logger
from ..core.metrics import (
    increment_agent_event,
    increment_loop_abort,
    mark_orchestrator_run_completed,
    mark_orchestrator_run_started,
    observe_agent_latency,
    observe_negotiation_metrics,
    record_agent_tool_invocation,
    record_agent_tool_failure,
    record_agent_tool_policy,
    record_plan_metrics,
    record_sla_event,
)
from ..schemas.agents import AgentInput, AgentOutput
from ..services.scoring import OutputQualityScorer
from ..tools.exceptions import ToolError, ToolPolicyViolationError
from ..tools.registry import normalize_tool_name, tool_registry
from .context import ContextAssemblyContract, ContextSnapshot, ContextSnapshotStore, ContextStage
from .dossier import build_decision_dossier
from .guardrails import GuardrailDecisionType, GuardrailManager
from .llm_planner import LLMOrchestrationPlanner, PlannerError, PlannerPlan, PlannedAgentStep
from .lifecycle import LifecycleEvent, LifecycleStatus, TaskLifecycleStore
from .meta import MetaAgent, MetaResolution
from .negotiation import NegotiationEngine
from .planner import TaskPlanner
from .review import ReviewManager
from .routing import DynamicAgentRouter, RoutingDecision
from .scheduler import TaskScheduler
from .state import OrchestratorEvent, OrchestratorRun, OrchestratorStatus
from .store import OrchestratorStateStore
from .tool_policy import AgentToolPolicy, get_agent_tool_policy

logger = get_logger(name=__name__)


# Confidence scores below this value trigger general_agent fallback.
LOW_PLANNER_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class _RunTracker:
    run: OrchestratorRun | None
    entry_point: str
    sequence: int = 0
    lifecycle_sequence: int = 0
    started_at: float = 0.0
    negotiation_rounds: int = 0
    negotiation_strategy: str = "unknown"
    negotiation_consensus: float | None = None
    guardrail_decisions: int = 0
    escalations: int = 0
    tool_calls: int = 0
    planner_invocations: int = 0
    abort_reason: str | None = None

    def __post_init__(self) -> None:
        self.started_at = time.perf_counter()


class ToolFirstPolicyViolation(RuntimeError):
    """Raised when an agent completes without a successful tool invocation."""


class SafetyLimitExceeded(RuntimeError):
    """Raised when orchestration guardrails exceed configured limits."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class _ToolSession:
    agent_name: str
    tool_service: Any
    planned_tools: tuple[str, ...] = ()
    fallback_tools: tuple[str, ...] = ()
    planner_reason: str | None = None
    invocations: list[tuple[str, Any]] | None = None
    failures: list[dict[str, Any]] | None = None
    run_tracker: "_RunTracker" | None = None
    max_tool_calls: int | None = None
    policy: AgentToolPolicy | None = None

    def __post_init__(self) -> None:
        self.invocations = [] if self.invocations is None else self.invocations
        self.failures = [] if self.failures is None else self.failures
        self.attempts = 0

    def register_attempt(self, tool: str) -> None:
        if self.max_tool_calls is None or self.max_tool_calls <= 0 or self.run_tracker is None:
            return
        next_count = self.run_tracker.tool_calls + 1
        if next_count > self.max_tool_calls:
            raise SafetyLimitExceeded(
                "tool_call_limit",
                f"Exceeded maximum of {self.max_tool_calls} tool calls while invoking '{tool}'",
            )
        self.run_tracker.tool_calls = next_count

    @property
    def proxy(self) -> Any:
        return _ToolProxy(self)

    def record_success(self, tool: str, result: Any) -> None:
        self.attempts += 1
        self.invocations.append((tool, result))
        record_agent_tool_invocation(agent=self.agent_name, tool=tool, outcome="success")

    def record_failure(self, tool: str, error: Exception) -> None:
        self.attempts += 1
        failure_type = type(error).__name__
        canonical_error = isinstance(error, ToolError)
        record_agent_tool_failure(
            agent=self.agent_name,
            tool=tool,
            failure_type=failure_type,
            canonical=canonical_error,
        )
        self.failures.append(
            {
                "tool": tool,
                "error": str(error),
                "type": failure_type,
                "canonical": canonical_error,
            }
        )
        record_agent_tool_invocation(agent=self.agent_name, tool=tool, outcome="failure")

    @property
    def successful(self) -> bool:
        return bool(self.invocations)

    def adherence_summary(self) -> dict[str, Any]:
        planned_set = set(self.planned_tools)
        fallback_set = set(self.fallback_tools)
        success_tools = [tool for tool, _ in self.invocations]
        failure_tools = [entry["tool"] for entry in self.failures]
        primary_used = any(tool in planned_set for tool in success_tools)
        fallback_used = any(tool in fallback_set for tool in success_tools)
        unplanned_successes = [
            tool for tool in success_tools if tool not in planned_set and tool not in fallback_set
        ]
        classification = "none"
        if primary_used:
            classification = "primary"
        elif fallback_used:
            classification = "fallback"
        elif success_tools:
            classification = "unplanned"
        allowed = bool(success_tools) and (
            (not planned_set and not fallback_set) or primary_used or fallback_used
        )
        missing_primary = bool(planned_set) and not primary_used
        return {
            "planned_tools": sorted(planned_set),
            "fallback_tools": sorted(fallback_set),
            "successful_tools": success_tools,
            "failed_tools": failure_tools,
            "primary_used": primary_used,
            "fallback_used": fallback_used,
            "unplanned_successes": unplanned_successes,
            "allowed": allowed,
            "missing_primary": missing_primary,
            "classification": classification,
            "planner_reason": self.planner_reason,
        }

    @property
    def used_tools(self) -> list[str]:
        """Return list of successfully used tools."""
        return [tool for tool, _ in self.invocations]


class _ToolProxy:
    def __init__(self, session: _ToolSession) -> None:
        self._session = session

    async def invoke(self, tool: str, payload: dict[str, Any]) -> Any:
        self._session.register_attempt(tool)
        policy = self._session.policy
        if policy is not None and not policy.is_allowed(tool):
            error = ToolPolicyViolationError(
                f"Tool '{tool}' is not permitted for agent '{self._session.agent_name}' (policy restricted)."
            )
            self._session.record_failure(tool, error)
            raise error
        try:
            result = await self._session.tool_service.invoke(tool, payload)
        except Exception as exc:  # pragma: no cover - surfaced via agent logic
            self._session.record_failure(tool, exc)
            raise
        self._session.record_success(tool, result)
        return result

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._session.tool_service, name)


class Orchestrator:
    def __init__(
        self,
        *,
        agents: list[BaseAgent],
        state_store: OrchestratorStateStore | None = None,
        negotiation_engine: NegotiationEngine | None = None,
        planner: TaskPlanner | None = None,
        scheduler: TaskScheduler | None = None,
        context_contract: ContextAssemblyContract | None = None,
        snapshot_store: ContextSnapshotStore | None = None,
        lifecycle_store: TaskLifecycleStore | None = None,
        guardrail_manager: GuardrailManager | None = None,
        meta_agent: MetaAgent | None = None,
        review_manager: ReviewManager | None = None,
        orchestration_planner: LLMOrchestrationPlanner | None = None,
        planning_settings: PlanningSettings | None = None,
        agent_router: DynamicAgentRouter | None = None,
        output_quality_scorer: OutputQualityScorer | None = None,
    ):
        self.agents = agents
        self._state_store = state_store
        self._negotiation = negotiation_engine
        self._planner = planner
        self._scheduler = scheduler
        self._context_contract = context_contract
        self._snapshot_store = snapshot_store
        self._lifecycle_store = lifecycle_store
        self._guardrails = guardrail_manager
        self._meta_agent = meta_agent
        self._reviews = review_manager
        self._graph = self._build_graph(agents)
        self._current_context: AgentContext | None = None
        self._llm_planner = orchestration_planner
        self._active_plan_steps: list[PlannedAgentStep] | None
        self._active_plan_step_lookup: dict[str, PlannedAgentStep] | None
        self._roster_index: dict[str, BaseAgent]
        self._active_plan_steps = None
        self._active_plan_step_lookup = None
        self._roster_index = {}
        self._planning_settings = planning_settings or get_settings().planning
        self._max_tool_calls = max(0, int(self._planning_settings.max_tool_calls_per_run))
        self._max_planner_recursions = max(0, int(self._planning_settings.max_planner_recursions))
        self._max_run_seconds = max(0.0, float(self._planning_settings.max_run_seconds))
        self._active_run_tracker: _RunTracker | None = None
        self._agent_router = agent_router
        self._low_confidence_threshold = self._resolve_low_confidence_threshold()
        self._agent_tool_catalog: dict[str, set[str]] = {}
        self._output_quality_scorer = output_quality_scorer or OutputQualityScorer()

    def _reset_plan_state(self) -> None:
        self._active_plan_steps = None
        self._active_plan_step_lookup = None
        self._roster_index = {}
        self._agent_tool_catalog = {}

    def _resolve_low_confidence_threshold(self) -> float:
        value = getattr(self._planning_settings, "low_confidence_threshold", None)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return LOW_PLANNER_CONFIDENCE_THRESHOLD
        if not math.isfinite(numeric):
            return LOW_PLANNER_CONFIDENCE_THRESHOLD
        return max(0.0, min(1.0, numeric))

    def _is_low_confidence(self, confidence: float | None) -> bool:
        if confidence is None:
            return False
        return confidence < self._low_confidence_threshold

    def _should_early_exit(
        self,
        executed_agent: BaseAgent,
        latest_output: dict[str, Any] | None,
        execution_plan: list[tuple[int, PlannedAgentStep | None, BaseAgent]],
        current_step_index: int,
    ) -> bool:
        """
        Determine if we should skip subsequent general_agent calls after
        a specialist agent has provided a high-confidence response.
        
        This prevents redundant/duplicate responses where general_agent
        repeats triage after a specialist has already handled the task.
        """
        # Only consider specialist agents (not general_agent)
        specialist_agents = {"finance_agent", "research_agent", "creative_agent", "enterprise_agent"}
        if executed_agent.name not in specialist_agents:
            return False
        
        # Check if output has high confidence
        if latest_output is None:
            return False
        
        confidence = latest_output.get("confidence")
        if confidence is None:
            return False
        
        # High confidence threshold for early exit
        # Lowered to 0.75 to allow specialist agents to trigger early exit more often
        HIGH_CONFIDENCE_THRESHOLD = 0.75
        if confidence < HIGH_CONFIDENCE_THRESHOLD:
            return False
        
        # Check if there's a general_agent still pending in the plan
        remaining_general = any(
            agent.name == "general_agent" 
            for idx, _, agent in execution_plan 
            if idx > current_step_index
        )
        
        if remaining_general:
            logger.info(
                "early_exit_triggered",
                specialist=executed_agent.name,
                confidence=confidence,
                threshold=HIGH_CONFIDENCE_THRESHOLD,
            )
            return True
        
        return False

    async def _try_dynamic_replan(
        self,
        state: dict[str, Any],
        failed_agent: BaseAgent,
        error: Exception,
        context: AgentContext,
        run_tracker: _RunTracker | None,
        progress_cb: Callable[[dict[str, Any]], Awaitable[None]] | None,
        roster_index: dict[str, BaseAgent],
    ) -> dict[str, Any] | None:
        """
        Attempt dynamic re-planning after an agent failure.
        
        Tries to find an alternative agent that can handle the task.
        Returns recovery info if successful, None if re-planning failed.
        """
        # Don't re-plan for safety limit violations or tool policy errors
        if isinstance(error, (SafetyLimitExceeded, ToolFirstPolicyViolation)):
            logger.info(
                "dynamic_replan_skipped",
                task=state.get("id"),
                failed_agent=failed_agent.name,
                reason="non_recoverable_error",
                error_type=type(error).__name__,
            )
            return None
        
        # Check if we have a fallback agent defined
        fallback_name = getattr(failed_agent, "fallback_agent", None)
        if fallback_name and fallback_name in roster_index:
            fallback_agent = roster_index[fallback_name]
            
            logger.info(
                "dynamic_replan_attempting",
                task=state.get("id"),
                failed_agent=failed_agent.name,
                fallback_agent=fallback_name,
            )
            
            await self._record_event(
                run_tracker,
                "dynamic_replan_started",
                agent=failed_agent.name,
                payload={
                    "failed_error": str(error),
                    "fallback_agent": fallback_name,
                },
            )
            
            await self._notify(
                progress_cb,
                self._ensure_run_id(
                    run_tracker,
                    {
                        "event": "dynamic_replan",
                        "failed_agent": failed_agent.name,
                        "recovery_agent": fallback_name,
                        "task_id": state.get("id"),
                        "error": str(error),
                    },
                ),
            )
            
            # Try to execute the fallback agent
            try:
                tool_session = self._start_tool_session(
                    agent=fallback_agent,
                    base_context=context,
                    plan_step=None,
                    run_tracker=run_tracker,
                )
                agent_context = self._clone_context(
                    context,
                    tool_session,
                    None,
                    agent_name=fallback_agent.name,
                )
                agent_input = await self._prepare_agent_input(
                    state, agent=fallback_agent, context=agent_context
                )
                result = await fallback_agent.handle(agent_input, context=agent_context)
                self._record_output(state, agent=fallback_agent, result=result, planner_step=None)
                
                # Get the latest output to include in the agent_completed event
                latest_output = state.get("outputs", [])[-1] if state.get("outputs") else None
                
                logger.info(
                    "dynamic_replan_completed",
                    task=state.get("id"),
                    failed_agent=failed_agent.name,
                    recovery_agent=fallback_name,
                )
                
                await self._record_event(
                    run_tracker,
                    "dynamic_replan_success",
                    agent=fallback_name,
                    payload={
                        "original_failed_agent": failed_agent.name,
                        "original_error": str(error),
                    },
                )
                
                # Send agent_completed event for the recovery agent
                await self._notify(
                    progress_cb,
                    self._ensure_run_id(
                        run_tracker,
                        {
                            "event": "agent_completed",
                            "agent": fallback_name,
                            "task_id": state.get("id"),
                            "output": latest_output,
                            "is_recovery": True,
                            "original_failed_agent": failed_agent.name,
                        },
                    ),
                )
                
                return {
                    "recovery_agent": fallback_name,
                    "recovery_result": result,
                }
            except Exception as fallback_exc:
                logger.warning(
                    "dynamic_replan_fallback_failed",
                    task=state.get("id"),
                    failed_agent=failed_agent.name,
                    fallback_agent=fallback_name,
                    error=str(fallback_exc),
                )
                return None
        
        # No fallback agent available
        logger.info(
            "dynamic_replan_no_fallback",
            task=state.get("id"),
            failed_agent=failed_agent.name,
        )
        return None

    async def _handle_collaboration(
        self,
        target_agent_name: str,
        request: str,
        context_data: dict[str, Any],
        *,
        base_context: AgentContext,
        run_tracker: _RunTracker | None,
        roster_index: dict[str, BaseAgent],
        state: dict[str, Any],
    ) -> AgentOutput:
        """
        Handle a collaboration request from one agent to another.
        
        This is called when an agent needs help from a specialist during execution.
        It runs the target agent synchronously and returns its output.
        """
        if target_agent_name not in roster_index:
            raise ValueError(f"Unknown collaboration target: {target_agent_name}")
        
        target_agent = roster_index[target_agent_name]
        
        logger.info(
            "agent_collaboration_request",
            target_agent=target_agent_name,
            request_preview=request[:100],
            task_id=state.get("id"),
        )
        
        await self._record_event(
            run_tracker,
            "collaboration_request",
            agent=target_agent_name,
            payload={
                "request": request,
                "context_data": context_data,
            },
        )
        
        # Create a tool session for the target agent
        tool_session = self._start_tool_session(
            agent=target_agent,
            base_context=base_context,
            plan_step=None,
            run_tracker=run_tracker,
        )
        
        # Clone context for the target agent (without collaboration handler to prevent recursion)
        collab_context = AgentContext(
            memory=base_context.memory,
            llm=base_context.llm,
            context=base_context.context,
            tools=tool_session.proxy if tool_session else base_context.tools,
            scorer=base_context.scorer,
            planned_tools=None,  # Let the agent decide
            fallback_tools=None,
            planner_reason=f"Collaboration request: {request[:200]}",
            thinking_emitter=base_context.thinking_emitter,
            _tool_chain_executor=base_context._tool_chain_executor,
            _collaboration_handler=None,  # Prevent nested collaboration
        )
        
        # Create agent input from the collaboration request
        collab_input = AgentInput(
            request=request,
            context=context_data.get("context", {}),
            routing=context_data.get("routing", {}),
            session_id=state.get("session_id"),
            conversation_id=context_data.get("conversation_id"),
        )
        
        # Execute the target agent
        result = await target_agent.handle(collab_input, context=collab_context)
        
        logger.info(
            "agent_collaboration_completed",
            target_agent=target_agent_name,
            confidence=result.confidence,
        )
        
        await self._record_event(
            run_tracker,
            "collaboration_completed",
            agent=target_agent_name,
            payload={
                "confidence": result.confidence,
                "response_preview": (getattr(result, "response", None) or result.summary or "")[:200],
            },
        )
        
        return result

    async def _execute_single_agent(
        self,
        state: dict[str, Any],
        step_index: int,
        plan_step: PlannedAgentStep | None,
        planned_agent: BaseAgent,
        context: AgentContext,
        run_tracker: _RunTracker | None,
        progress_cb: Callable[[dict[str, Any]], Awaitable[None]] | None,
        roster_index: dict[str, BaseAgent],
        task_id: str,
        using_plan_steps: bool,
    ) -> dict[str, Any]:
        """
        Execute a single agent in the plan (used for parallel execution).
        
        Returns a dict with execution results:
        - early_exit_skipped: set of indices to skip due to early exit
        - agent: name of the executed agent
        - success: whether execution succeeded
        """
        task = state
        executed_agent = planned_agent
        resolved_step = plan_step
        fallback_agent: BaseAgent | None = None
        fallback_reason: str | None = None
        step_router_metadata: dict[str, Any] | None = None
        step_confidence = 1.0
        
        if plan_step is not None:
            step_confidence = self._coerce_plan_confidence(getattr(plan_step, "confidence", 1.0))
            plan_step.confidence = step_confidence
            
            if self._is_low_confidence(step_confidence):
                exclude = {plan_step.agent}
                router_candidate, router_meta = await self._router_pick_agent(
                    task,
                    list(roster_index.values()),
                    exclude=exclude,
                )
                candidate = router_candidate or roster_index.get("general_agent")
                if candidate is not None and candidate.name != planned_agent.name:
                    fallback_agent = candidate
                    fallback_reason = (
                        f"Planner confidence {step_confidence:.2f} below threshold "
                        f"{self._low_confidence_threshold:.2f}; rerouting to {candidate.name}."
                    )
                    executed_agent = candidate
                    resolved_step = PlannedAgentStep(
                        agent=candidate.name,
                        tools=[],
                        fallback_tools=list(plan_step.fallback_tools),
                        reason=(
                            f"{fallback_reason} Original agent: {plan_step.agent}. {plan_step.reason}"
                        ).strip(),
                        confidence=step_confidence,
                    )
                    step_router_metadata = router_meta if router_candidate is not None else None
        
        step_id = f"step_{step_index}"
        step_context = resolved_step
        step_metadata = self._build_planner_step_metadata(
            index=step_index,
            planned_step=plan_step,
            resolved_step=resolved_step,
            executed_agent=executed_agent,
            fallback_agent=fallback_agent,
            fallback_reason=fallback_reason,
            router_metadata=step_router_metadata,
        )
        event_step_metadata = step_metadata if using_plan_steps else None
        
        # Check early exit
        if self._should_early_exit(step_index, plan_step, state):
            return {"early_exit_skipped": {step_index}, "agent": executed_agent.name, "success": True}
        
        await self._notify(
            progress_cb,
            self._ensure_run_id(
                run_tracker,
                {
                    "event": "agent_started",
                    "agent": executed_agent.name,
                    "task_id": task.get("id"),
                    "step_id": step_id,
                    "planner_step": event_step_metadata,
                },
            ),
        )
        increment_agent_event(agent=executed_agent.name, event="started")
        logger.info("agent_started", agent=executed_agent.name, task=task.get("id"))
        start_time = time.perf_counter()
        
        await self._record_lifecycle(
            run_tracker,
            task_id=task_id,
            step_id=step_id,
            event_type="agent_started",
            status=LifecycleStatus.IN_PROGRESS,
            agent=executed_agent.name,
        )
        
        try:
            tool_session = self._start_tool_session(
                agent=executed_agent,
                base_context=context,
                plan_step=step_context,
                run_tracker=run_tracker,
            )
            
            # Build collaboration handler for this agent
            async def collab_handler(target: str, request: str, ctx_data: dict[str, Any]) -> AgentOutput:
                return await self._handle_collaboration(
                    target_agent_name=target,
                    request=request,
                    context_data=ctx_data,
                    base_context=context,
                    run_tracker=run_tracker,
                    roster_index=roster_index,
                    state=state,
                )
            
            agent_context = self._clone_context(
                context,
                tool_session,
                step_context,
                agent_name=executed_agent.name,
                collaboration_handler=collab_handler,
            )
            agent_input = await self._prepare_agent_input(state, agent=executed_agent, context=agent_context)
            result = await executed_agent.handle(agent_input, context=agent_context)
            
            # Evaluate output quality
            tools_used = list(tool_session.used_tools) if tool_session else None
            tools_expected = list(plan_step.tools) if plan_step and plan_step.tools else None
            quality_acceptable, quality_info = await self._evaluate_output_quality(
                state=state,
                agent=executed_agent,
                result=result,
                tools_used=tools_used,
                tools_expected=tools_expected,
                run_tracker=run_tracker,
                progress_cb=progress_cb,
            )
            
            # Add quality info to result metadata
            if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                result.metadata["quality_score"] = quality_info.get("score")
                result.metadata["quality_breakdown"] = quality_info.get("breakdown")
            
            self._record_output(state, agent=executed_agent, result=result, planner_step=step_metadata)
            
            await self._enforce_tool_first_policy(
                tool_session,
                agent=executed_agent,
                state=state,
                run_tracker=run_tracker,
                progress_cb=progress_cb,
            )
            
            duration = time.perf_counter() - start_time
            observe_agent_latency(agent=executed_agent.name, latency=duration)
            increment_agent_event(agent=executed_agent.name, event="completed")
            
            await self._record_lifecycle(
                run_tracker,
                task_id=task_id,
                step_id=step_id,
                event_type="agent_completed",
                status=LifecycleStatus.COMPLETED,
                agent=executed_agent.name,
                latency_ms=duration * 1000,
            )
            
            # Get the latest output to include in the agent_completed event
            latest_output = state.get("outputs", [])[-1] if state.get("outputs") else None
            
            await self._notify(
                progress_cb,
                self._ensure_run_id(
                    run_tracker,
                    {
                        "event": "agent_completed",
                        "agent": executed_agent.name,
                        "task_id": task.get("id"),
                        "step_id": step_id,
                        "planner_step": event_step_metadata,
                        "latency_ms": duration * 1000,
                        "output": latest_output,
                    },
                ),
            )
            
            return {"early_exit_skipped": set(), "agent": executed_agent.name, "success": True}
            
        except Exception as exc:
            duration = time.perf_counter() - start_time
            observe_agent_latency(agent=executed_agent.name, latency=duration)
            increment_agent_event(agent=executed_agent.name, event="failed")
            
            # Try dynamic re-planning
            replan_result = await self._try_dynamic_replan(
                state=state,
                failed_agent=executed_agent,
                error=exc,
                context=context,
                run_tracker=run_tracker,
                progress_cb=progress_cb,
                roster_index=roster_index,
            )
            
            if replan_result is not None:
                return {"early_exit_skipped": set(), "agent": executed_agent.name, "success": True}
            
            # Re-raise the exception to be caught by the parallel execution handler
            raise

    def _begin_run(self, run_tracker: _RunTracker | None, *, task: Mapping[str, Any]) -> _RunTracker:
        tracker = run_tracker
        if tracker is None:
            entry_point = str(task.get("source") or "api")
            tracker = _RunTracker(run=None, entry_point=entry_point)
        tracker.tool_calls = 0
        tracker.planner_invocations = 0
        tracker.abort_reason = None
        tracker.started_at = time.perf_counter()
        self._active_run_tracker = tracker
        return tracker

    def _complete_run(self) -> None:
        if self._active_run_tracker is not None:
            self._active_run_tracker.tool_calls = 0
            self._active_run_tracker.planner_invocations = 0
        self._active_run_tracker = None

    def _lookup_plan_step_for_agent(self, agent_name: str) -> PlannedAgentStep | None:
        if self._active_plan_step_lookup is not None:
            candidate = self._active_plan_step_lookup.get(agent_name)
            if candidate is not None:
                return candidate
        if not self._active_plan_steps:
            return None
        for step in self._active_plan_steps:
            if step.agent == agent_name:
                return step
        return None

    def _build_planner_step_metadata(
        self,
        *,
        index: int,
        planned_step: PlannedAgentStep | None,
        resolved_step: PlannedAgentStep | None,
        executed_agent: BaseAgent,
        fallback_agent: BaseAgent | None,
        fallback_reason: str | None,
        router_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        capability = executed_agent.capability
        executed_capability = capability.value if hasattr(capability, "value") else str(capability)
        metadata: dict[str, Any] = {
            "index": index,
            "executed_agent": executed_agent.name,
            "executed_agent_capability": executed_capability,
            "fallback_applied": fallback_agent is not None,
        }
        if planned_step is not None:
            metadata.update(
                {
                    "planned_agent": planned_step.agent,
                    "planned_reason": planned_step.reason,
                    "planned_tools": list(planned_step.tools),
                    "planned_fallback_tools": list(planned_step.fallback_tools),
                    "planned_confidence": self._coerce_plan_confidence(getattr(planned_step, "confidence", 1.0)),
                }
            )
        if resolved_step is not None:
            metadata.update(
                {
                    "executed_reason": resolved_step.reason,
                    "executed_tools": list(resolved_step.tools),
                    "executed_fallback_tools": list(resolved_step.fallback_tools),
                    "executed_confidence": self._coerce_plan_confidence(getattr(resolved_step, "confidence", 1.0)),
                }
            )
        if fallback_agent is not None:
            fallback_capability = fallback_agent.capability
            metadata["fallback_agent"] = fallback_agent.name
            metadata["fallback_agent_capability"] = (
                fallback_capability.value if hasattr(fallback_capability, "value") else str(fallback_capability)
            )
            metadata["fallback_reason"] = fallback_reason
            metadata["confidence_threshold"] = self._low_confidence_threshold
        if router_metadata is not None:
            metadata["router"] = dict(router_metadata)
        return metadata

    async def _router_decision(
        self,
        task: Mapping[str, Any],
        roster: Sequence[BaseAgent],
    ) -> RoutingDecision | None:
        if self._agent_router is None:
            return None
        try:
            return await self._agent_router.select(task=task, agents=roster)
        except Exception as exc:  # pragma: no cover - router best effort
            logger.warning("dynamic_router_failed", error=str(exc))
            return None

    async def _router_low_confidence_selection(
        self,
        task: Mapping[str, Any],
        roster: Sequence[BaseAgent],
    ) -> tuple[list[BaseAgent], dict[str, Any]] | None:
        decision = await self._router_decision(task, roster)
        if decision is None or not decision.agents:
            return None
        return list(decision.agents), self._router_metadata(decision)

    async def _router_pick_agent(
        self,
        task: Mapping[str, Any],
        roster: Sequence[BaseAgent],
        *,
        exclude: set[str] | None = None,
    ) -> tuple[BaseAgent | None, dict[str, Any] | None]:
        decision = await self._router_decision(task, roster)
        if decision is None or not decision.agents:
            return None, None
        for candidate in decision.agents:
            if exclude and candidate.name in exclude:
                continue
            return candidate, self._router_metadata(decision)
        return None, self._router_metadata(decision)

    @staticmethod
    def _router_metadata(decision: RoutingDecision) -> dict[str, Any]:
        return {
            "reason": decision.reason,
            "scores": decision.scores,
            "metadata": decision.metadata,
        }

    @staticmethod
    def _collect_tool_names(candidate: Any) -> set[str]:
        names: set[str] = set()

        def _add_with_variants(identifier: str | None) -> None:
            if not identifier or not isinstance(identifier, str):
                return
            token = identifier.strip()
            if not token:
                return
            normalized = Orchestrator._normalize_tool_identifier(token)
            if normalized:
                names.add(normalized)
            if "/" in token:
                variant = Orchestrator._normalize_tool_identifier(token.replace("/", "."))
                if variant:
                    names.add(variant)
            if "." in token:
                variant = Orchestrator._normalize_tool_identifier(token.replace(".", "/"))
                if variant:
                    names.add(variant)

        if candidate is None:
            return names
        if isinstance(candidate, Mapping):
            iterable = candidate.keys()
        elif isinstance(candidate, (list, tuple, set, frozenset)):
            iterable = candidate
        else:
            return names
        for item in iterable:
            if isinstance(item, str):
                _add_with_variants(item)
        return names

    @classmethod
    def _build_agent_tool_catalog(cls, roster: Sequence[BaseAgent]) -> dict[str, set[str]]:
        alias_map = tool_registry.aliases()
        reverse_aliases: dict[str, set[str]] = {}
        for alias, target in alias_map.items():
            normalized_target = cls._normalize_tool_identifier(target)
            normalized_alias = cls._normalize_tool_identifier(alias)
            if not normalized_target or not normalized_alias:
                continue
            reverse_aliases.setdefault(normalized_target, set()).add(normalized_alias)

        catalog: dict[str, set[str]] = {}
        for agent in roster:
            tool_names: set[str] = set()
            for attribute in ("tool_preference", "tool_candidates", "fallback_tools", "tools", "default_tools"):
                tool_names |= cls._collect_tool_names(getattr(agent, attribute, None))
            retry_config = getattr(agent, "tool_retry_config", None)
            if isinstance(retry_config, Mapping):
                tool_names |= cls._collect_tool_names(retry_config)
            resolved_names: set[str] = set()
            for name in tool_names:
                if not name:
                    continue
                resolved_names.add(name)
                canonical = tool_registry.resolve(name)
                normalized_canonical = cls._normalize_tool_identifier(canonical) if canonical else None
                if normalized_canonical:
                    resolved_names.add(normalized_canonical)
            alias_variants: set[str] = set()
            for key in resolved_names:
                alias_variants |= reverse_aliases.get(key, set())
            catalog[agent.name] = resolved_names | alias_variants
        return catalog

    @staticmethod
    def _normalize_tool_identifier(identifier: str | None) -> str | None:
        if not identifier or not isinstance(identifier, str):
            return None
        token = identifier.strip()
        if not token:
            return None
        try:
            return normalize_tool_name(token)
        except Exception:
            return token.replace("\\", "_").strip().lower()

    @classmethod
    def _tool_in_catalog(cls, tool: str, catalog: set[str] | None) -> bool:
        if not catalog:
            return False
        normalized = cls._normalize_tool_identifier(tool)
        return bool(normalized and normalized in catalog)

    def _validate_plan_tools(self, steps: Sequence[PlannedAgentStep]) -> None:
        if not steps:
            return
        sanitized_entries: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            catalog = self._agent_tool_catalog.get(step.agent)
            if catalog is None:
                continue
            missing = [tool for tool in step.tools if not self._tool_in_catalog(tool, catalog)]
            fallback_missing = [tool for tool in step.fallback_tools if not self._tool_in_catalog(tool, catalog)]
            if not missing and not fallback_missing:
                continue
            if missing:
                step.tools = [tool for tool in step.tools if tool not in missing]
            if fallback_missing:
                step.fallback_tools = [tool for tool in step.fallback_tools if tool not in fallback_missing]
            sanitized_entries.append(
                {
                    "agent": step.agent,
                    "step_index": index,
                    "removed_tools": sorted(set(missing)),
                    "removed_fallback_tools": sorted(set(fallback_missing)),
                    "available": sorted(catalog),
                }
            )
        if sanitized_entries:
            logger.warning("planner_invalid_tools_sanitized", entries=sanitized_entries)

    def _record_step_override(self, state: dict[str, Any], step_metadata: Mapping[str, Any]) -> None:
        routing = state.get("routing")
        if not isinstance(routing, dict):
            return
        metadata = routing.get("metadata")
        if not isinstance(metadata, dict):
            return
        planner_section = metadata.get("planner")
        if not isinstance(planner_section, dict):
            return
        overrides = planner_section.setdefault("step_overrides", [])
        if isinstance(overrides, list):
            overrides.append(dict(step_metadata))

    def _append_step_metadata(self, state: dict[str, Any], step_metadata: Mapping[str, Any]) -> None:
        shared_context = state.setdefault("shared_context", {"provenance": []})
        executed_steps = shared_context.setdefault("planner_steps", [])
        if isinstance(executed_steps, list):
            executed_steps.append(copy.deepcopy(dict(step_metadata)))
            if len(executed_steps) > 20:
                del executed_steps[:-20]

    def _build_graph(self, agents: list[BaseAgent]) -> Any:
        if StateGraph is None:
            logger.warning("langgraph_not_installed", agents=len(agents))
            return None
        graph = StateGraph(dict)  # type: ignore[arg-type]
        for agent in agents:
            async def _node(state: dict[str, Any], *, agent=agent) -> dict[str, Any]:
                if self._current_context is None:
                    raise RuntimeError("Agent context not set for orchestration")
                plan_step = self._lookup_plan_step_for_agent(agent.name)
                tool_session = self._start_tool_session(
                    agent=agent,
                    base_context=self._current_context,
                    plan_step=plan_step,
                    run_tracker=self._active_run_tracker,
                )
                agent_context = self._clone_context(
                    self._current_context,
                    tool_session,
                    plan_step,
                    agent_name=agent.name,
                )
                request = await self._prepare_agent_input(state, agent=agent, context=agent_context)
                output = await agent.handle(request, context=agent_context)
                updated_state = self._record_output(state, agent=agent, result=output, planner_step=None)
                await self._enforce_tool_first_policy(tool_session, agent=agent, state=updated_state)
                return updated_state

            graph.add_node(agent.name, _node)  # type: ignore[arg-type]
        graph.set_entry_point(agents[0].name if agents else "noop")
        for index in range(len(agents) - 1):
            graph.add_edge(agents[index].name, agents[index + 1].name)
        if agents:
            graph.set_finish_point(agents[-1].name)
        return graph

    def _start_tool_session(
        self,
        *,
        agent: BaseAgent,
        base_context: AgentContext,
        plan_step: PlannedAgentStep | None,
        run_tracker: _RunTracker | None = None,
    ) -> _ToolSession | None:
        if base_context.tools is None:
            return None
        if plan_step is None:
            plan_step = self._lookup_plan_step_for_agent(agent.name)
        planned = tuple(plan_step.tools) if plan_step and plan_step.tools else ()
        fallback = tuple(plan_step.fallback_tools) if plan_step and plan_step.fallback_tools else ()
        reason = plan_step.reason if plan_step and plan_step.reason else None
        resolved_tracker = run_tracker if run_tracker is not None else self._active_run_tracker
        max_calls = self._max_tool_calls if self._max_tool_calls > 0 else None
        policy = get_agent_tool_policy(agent.name)
        return _ToolSession(
            agent_name=agent.name,
            tool_service=base_context.tools,
            planned_tools=planned,
            fallback_tools=fallback,
            planner_reason=reason,
            run_tracker=resolved_tracker,
            max_tool_calls=max_calls,
            policy=policy,
        )

    def _clone_context(
        self,
        base_context: AgentContext,
        tool_session: _ToolSession | None,
        plan_step: PlannedAgentStep | None,
        *,
        agent_name: str,
        collaboration_handler: Callable[[str, str, dict[str, Any]], Awaitable[AgentOutput]] | None = None,
    ) -> AgentContext:
        planned_list = None
        fallback_list = None
        if plan_step is None:
            plan_step = self._lookup_plan_step_for_agent(agent_name)
        if plan_step is not None:
            if plan_step.tools:
                planned_list = list(plan_step.tools)
            if plan_step.fallback_tools:
                fallback_list = list(plan_step.fallback_tools)
        elif base_context.planned_tools:
            planned_list = list(base_context.planned_tools)
            fallback_list = list(base_context.fallback_tools or []) if base_context.fallback_tools else None
        tools_proxy = tool_session.proxy if tool_session is not None else base_context.tools
        planner_reason = plan_step.reason if plan_step and plan_step.reason else base_context.planner_reason
        return AgentContext(
            memory=base_context.memory,
            llm=base_context.llm,
            context=base_context.context,
            tools=tools_proxy,
            scorer=base_context.scorer,
            planned_tools=planned_list,
            fallback_tools=fallback_list,
            planner_reason=planner_reason,
            thinking_emitter=base_context.thinking_emitter,
            _tool_chain_executor=base_context._tool_chain_executor,
            _collaboration_handler=collaboration_handler,
        )

    def _register_abort(self, tracker: _RunTracker | None, *, reason: str) -> None:
        resolved = tracker or self._active_run_tracker
        if resolved is not None and resolved.abort_reason == reason:
            return
        if resolved is not None:
            resolved.abort_reason = reason
        increment_loop_abort(reason=reason)

    def _enforce_run_time_budget(self, tracker: _RunTracker | None) -> None:
        if self._max_run_seconds <= 0.0:
            return
        resolved = tracker or self._active_run_tracker
        if resolved is None:
            return
        elapsed = time.perf_counter() - resolved.started_at
        if elapsed > self._max_run_seconds:
            self._register_abort(resolved, reason="run_time_limit")
            raise SafetyLimitExceeded(
                "run_time_limit",
                (
                    f"Orchestration exceeded maximum duration of {self._max_run_seconds:.2f} seconds "
                    f"(elapsed {elapsed:.2f} seconds)."
                ),
            )

    def _enforce_planner_budget(self, tracker: _RunTracker | None) -> None:
        resolved = tracker or self._active_run_tracker
        if resolved is None:
            return
        resolved.planner_invocations += 1
        if self._max_planner_recursions > 0 and resolved.planner_invocations > self._max_planner_recursions:
            self._register_abort(resolved, reason="planner_recursion_limit")
            raise SafetyLimitExceeded(
                "planner_recursion_limit",
                f"Exceeded maximum of {self._max_planner_recursions} planner invocations while resolving task.",
            )

    async def _enforce_tool_first_policy(
        self,
        tool_session: _ToolSession | None,
        *,
        agent: BaseAgent,
        state: dict[str, Any],
        run_tracker: _RunTracker | None = None,
        progress_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        if tool_session is None:
            return
        
        # ════════════════════════════════════════════════════════════════════════════
        # SKIP TOOL POLICY FOR AGENTS THAT PASSED
        # When an agent returns confidence=0.0 with metadata.passed=True, it signals
        # that the task is not relevant to this agent. In this case, we should NOT
        # enforce tool-first policy since the agent correctly chose to pass.
        # ════════════════════════════════════════════════════════════════════════════
        outputs = state.get("outputs", [])
        latest_output = outputs[-1] if outputs else None
        if isinstance(latest_output, dict):
            confidence = latest_output.get("confidence", 1.0)
            metadata = latest_output.get("metadata")
            is_passed = (
                confidence == 0.0 and
                isinstance(metadata, dict) and
                metadata.get("passed") is True
            )
            if is_passed:
                logger.info(
                    "tool_policy_agent_passed",
                    agent=agent.name,
                    reason="agent_returned_pass_with_zero_confidence",
                )
                record_agent_tool_policy(agent=agent.name, outcome="passed_skipped")
                return
        
        summary = tool_session.adherence_summary()
        enforcement_required = TOOL_ENFORCEMENT_POLICY.get(agent.name, False)
        policy_mode = "enforced" if enforcement_required else "optional"

        if isinstance(latest_output, dict):
            metadata = latest_output.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                latest_output["metadata"] = metadata
            tools_used = metadata.setdefault("tools_used", [])
            planned_set = set(summary["planned_tools"])
            fallback_set = set(summary["fallback_tools"])
            for tool, result in tool_session.invocations:
                if not isinstance(tool, str):
                    continue
                role = "primary" if tool in planned_set else "fallback" if tool in fallback_set else "unplanned"
                descriptor = {
                    "tool": tool,
                    "resolved": getattr(result, "resolved_tool", tool),
                    "cached": getattr(result, "cached", False),
                    "latency": getattr(result, "latency", 0.0),
                    "planner_role": role,
                }
                if descriptor not in tools_used:
                    tools_used.append(descriptor)
            planner_meta = metadata.setdefault("planner_tools", {})
            planner_meta.update(
                {
                    "planned": summary["planned_tools"],
                    "fallback": summary["fallback_tools"],
                    "successful": summary["successful_tools"],
                    "failed": summary["failed_tools"],
                    "classification": summary["classification"],
                    "unplanned_successes": summary["unplanned_successes"],
                    "reason": summary["planner_reason"],
                }
            )
            policy_meta = metadata.get("tool_policy")
            if not isinstance(policy_meta, dict):
                policy_meta = {}
                metadata["tool_policy"] = policy_meta
            policy_meta.update(
                {
                    "mode": policy_mode,
                    "planner_expected": bool(summary["planned_tools"] or summary["fallback_tools"]),
                }
            )
            if tool_session.policy is not None:
                policy_meta.setdefault("allowed_patterns", list(tool_session.policy.allowed_patterns))
                if tool_session.policy.denied_patterns:
                    policy_meta.setdefault("denied_patterns", list(tool_session.policy.denied_patterns))

        planner_expected = bool(summary["planned_tools"] or summary["fallback_tools"])
        override = self._extract_tool_policy_override(state, agent_name=agent.name)
        
        # Also check for synthesis task conditions (prior outputs + synthesis keywords in prompt)
        is_synthesis_task = self._is_synthesis_task_from_state(state, agent_name=agent.name)
        if is_synthesis_task:
            logger.info(
                "tool_policy_synthesis_skip",
                agent=agent.name,
                reason="synthesis_task_with_prior_outputs",
            )
            record_agent_tool_policy(agent=agent.name, outcome="synthesis_skipped")
            return
        
        if not tool_session.successful:
            if override and bool(override.get("allow_skip")):
                outcome = "override_allowed"
                record_agent_tool_policy(agent=agent.name, outcome=outcome)
                payload = {
                    "attempts": tool_session.attempts,
                    "errors": tool_session.failures,
                    "planner": summary,
                    "override": dict(override),
                    "reason": override.get("reason", "agent_override"),
                }
                await self._record_event(run_tracker, "tool_policy_override", agent=agent.name, payload=payload)
                if progress_cb is not None:
                    event_payload = self._ensure_run_id(
                        run_tracker,
                        {
                            "event": "tool_policy_override",
                            "agent": agent.name,
                            "task_id": self._task_id_from_state(state),
                            **payload,
                        },
                    )
                    await self._notify(progress_cb, event_payload)
                return
            if not enforcement_required:
                if tool_session.attempts == 0:
                    if not planner_expected:
                        record_agent_tool_policy(agent=agent.name, outcome="optional_skipped")
                        return
                    if self._should_allow_optional_tool_skip(summary=summary, state=state, agent_name=agent.name):
                        record_agent_tool_policy(agent=agent.name, outcome="optional_skipped_planned")
                        return
                outcome = "optional_failed" if tool_session.attempts else "optional_missing"
            else:
                outcome = "violation_failed" if tool_session.attempts else "violation_missing"
            record_agent_tool_policy(agent=agent.name, outcome=outcome)
            payload = {
                "attempts": tool_session.attempts,
                "errors": tool_session.failures,
                "reason": "no_successful_invocation" if tool_session.attempts else "no_invocation",
                "planner": summary,
            }
            if tool_session.policy is not None:
                payload["policy"] = {
                    "allowed": list(tool_session.policy.allowed_patterns),
                    "denied": list(tool_session.policy.denied_patterns),
                }
            await self._record_event(run_tracker, "tool_policy_violation", agent=agent.name, payload=payload)
            if progress_cb is not None:
                event_payload = self._ensure_run_id(
                    run_tracker,
                    {
                        "event": "tool_policy_violation",
                        "agent": agent.name,
                        "task_id": self._task_id_from_state(state),
                        **payload,
                    },
                )
                await self._notify(progress_cb, event_payload)
            if not enforcement_required:
                logger.warning(
                    "tool_policy_optional_violation",
                    agent=agent.name,
                    task=self._task_id_from_state(state),
                    reason=payload["reason"],
                )
                if planner_expected:
                    planned_list = summary["planned_tools"] or summary["fallback_tools"]
                    raise ToolFirstPolicyViolation(
                        f"Agent '{agent.name}' must successfully invoke planner-selected tools {planned_list} before completing."
                    )
                raise ToolFirstPolicyViolation(
                    f"Agent '{agent.name}' must successfully invoke at least one tool before completing."
                )
            raise ToolFirstPolicyViolation(
                f"Agent '{agent.name}' must successfully invoke at least one tool before completing."
            )

        if planner_expected and not summary["allowed"]:
            if override and bool(override.get("allow_skip")):
                outcome = "override_allowed_planner_mismatch"
                record_agent_tool_policy(agent=agent.name, outcome=outcome)
                payload = {
                    "attempts": tool_session.attempts,
                    "errors": tool_session.failures,
                    "planner": summary,
                    "override": dict(override),
                    "reason": override.get("reason", "agent_override"),
                    "successful": summary["successful_tools"],
                }
                await self._record_event(run_tracker, "tool_policy_override", agent=agent.name, payload=payload)
                if progress_cb is not None:
                    event_payload = self._ensure_run_id(
                        run_tracker,
                        {
                            "event": "tool_policy_override",
                            "agent": agent.name,
                            "task_id": self._task_id_from_state(state),
                            **payload,
                        },
                    )
                    await self._notify(progress_cb, event_payload)
                return
            outcome = "violation_planner_unfulfilled"
            record_agent_tool_policy(agent=agent.name, outcome=outcome)
            payload = {
                "attempts": tool_session.attempts,
                "errors": tool_session.failures,
                "planner": summary,
                "reason": "missing_planned_tool" if summary["missing_primary"] else "unplanned_success",
            }
            await self._record_event(run_tracker, "tool_policy_violation", agent=agent.name, payload=payload)
            if progress_cb is not None:
                event_payload = self._ensure_run_id(
                    run_tracker,
                    {
                        "event": "tool_policy_violation",
                        "agent": agent.name,
                        "task_id": self._task_id_from_state(state),
                        **payload,
                    },
                )
                await self._notify(progress_cb, event_payload)
            if not enforcement_required:
                logger.warning(
                    "planner_tools_optional_violation",
                    agent=agent.name,
                    task=self._task_id_from_state(state),
                    planned=summary["planned_tools"],
                    fallback=summary["fallback_tools"],
                )
            planned_list = summary["planned_tools"] or summary["fallback_tools"]
            raise ToolFirstPolicyViolation(
                f"Agent '{agent.name}' must successfully invoke planned tools or fallbacks {planned_list} before completing."
            )

        if planner_expected:
            classification = summary["classification"]
            if classification == "primary":
                outcome = "compliant_primary"
            elif classification == "fallback":
                outcome = "compliant_fallback"
            else:
                outcome = "compliant_unplanned"
        else:
            outcome = "compliant"
        record_agent_tool_policy(agent=agent.name, outcome=outcome)

    @staticmethod
    def _is_synthesis_task_from_state(state: Mapping[str, Any], *, agent_name: str | None = None) -> bool:
        """
        Detect if this is a synthesis task OR standalone strategy task 
        that should skip tool enforcement.
        
        These tasks don't need new tool invocations because they either:
        - Combine prior agent outputs (synthesis)
        - Rely on general business knowledge (strategy/diagnostic plans)
        - Are project planning/architecture tasks (knowledge-based)
        - Are pure creative writing tasks (poems, stories, etc.)
        """
        prompt = str(state.get("prompt") or "").lower()
        outputs = state.get("outputs", [])
        
        # ═════════════════════════════════════════════════════════════════════════
        # CHECK PLANNER METADATA for strategy_task flag (fast path)
        # ═════════════════════════════════════════════════════════════════════════
        planner_metadata = state.get("planner_metadata") or state.get("metadata", {}).get("planner", {})
        if isinstance(planner_metadata, dict):
            attributes = planner_metadata.get("attributes", {})
            if attributes.get("strategy_task") is True:
                return True
        
        # ═════════════════════════════════════════════════════════════════════════
        # CREATIVE WRITING TASKS (creative_agent)
        # These are pure creative tasks, not data retrieval tasks
        # ═════════════════════════════════════════════════════════════════════════
        if agent_name == "creative_agent":
            creative_task_indicators = (
                # Explicit creative requests
                "write something creative", "something creative", "sounds like",
                "written by a human", "thoughtful human", "not an ai", "not an a.i.",
                "human touch", "more human", "less robotic", "less ai",
                "make it sound", "rewrite this", "creative writing",
                # Creative content types
                "write a poem", "write a story", "write a song", "write lyrics",
                "write a blog", "blog post", "write an essay", "marketing copy",
                "write a slogan", "write a tagline", "brainstorm",
                "compose a", "draft a", "create a story", "create a poem",
            )
            
            if any(indicator in prompt for indicator in creative_task_indicators):
                return True
        
        # ═════════════════════════════════════════════════════════════════════════
        # STANDALONE STRATEGY TASKS (enterprise_agent)
        # These are knowledge-based writing tasks, not data retrieval tasks
        # ═════════════════════════════════════════════════════════════════════════
        if agent_name == "enterprise_agent":
            strategy_task_indicators = (
                # Diagnostic/planning tasks
                "diagnostic plan", "recovery plan", "action plan", "implementation plan",
                "hypotheses", "30-day", "90-day", "roadmap",
                # Strategy documents
                "gtm strategy", "go-to-market", "business strategy", "growth strategy",
                "competitive strategy", "pricing strategy", "marketing strategy",
                # Proposals and pitches
                "pitch deck", "investor pitch", "executive summary", "business case",
                "grant proposal", "project proposal", "budget proposal",
                # Operational frameworks
                "diagnostic framework", "assessment framework", "evaluation criteria",
                "kpis", "metrics framework", "success criteria",
                # Analysis requests (without needing live data)
                "root cause analysis", "gap analysis", "swot analysis",
                "risk assessment", "impact analysis",
                # PROJECT PLANNING / ARCHITECTURE TASKS (knowledge-based)
                "break this into", "break into tasks", "technical tasks", "clear tasks",
                "assign agents", "assign suitable", "tech stack", "technology stack",
                "system architecture", "software architecture", "system design",
                "project breakdown", "project tasks", "build a system",
                "attendance management", "management system", "ai-powered",
            )
            
            if any(indicator in prompt for indicator in strategy_task_indicators):
                return True
            
            # CRM/ops analytics framing (not stock ticker CRM)
            crm_ops_indicators = (
                "crm data", "lead-to-close", "conversion rate", "sales pipeline",
                "customer data", "lead data", "funnel", "churn",
            )
            if any(indicator in prompt for indicator in crm_ops_indicators):
                return True
        
        # ═════════════════════════════════════════════════════════════════════════
        # SYNTHESIS TASKS - Combine prior agent outputs
        # ═════════════════════════════════════════════════════════════════════════
        if not outputs or len(outputs) < 1:
            return False
        
        # Check for synthesis indicators in the prompt
        synthesis_indicators = (
            "executive brief", "executive summary", "summarize", "combine",
            "synthesize", "distinguish facts", "facts vs sentiment",
            "produce a brief", "create a summary", "write a summary",
            "consolidate", "compile", "bring together", "finally produce",
        )
        
        if any(indicator in prompt for indicator in synthesis_indicators):
            # Check if there are meaningful prior outputs (not just passes)
            meaningful_outputs = [
                o for o in outputs 
                if isinstance(o, dict) and o.get("confidence", 0) > 0.5
            ]
            if meaningful_outputs:
                return True
        
        # Also check output metadata for synthesis_task flag
        for entry in reversed(outputs):
            if not isinstance(entry, dict):
                continue
            if agent_name and entry.get("agent") != agent_name:
                continue
            metadata = entry.get("metadata", {})
            if metadata.get("synthesis_task"):
                return True
            if agent_name:
                break
        
        return False

    @staticmethod
    def _should_allow_optional_tool_skip(*, summary: Mapping[str, Any], state: Mapping[str, Any], agent_name: str | None = None) -> bool:
        override = Orchestrator._extract_tool_policy_override(state, agent_name=agent_name)
        if override and bool(override.get("allow_skip")):
            return True
        planned_tools = tuple(summary.get("planned_tools") or ())
        fallback_tools = tuple(summary.get("fallback_tools") or ())
        if not planned_tools and not fallback_tools:
            return True

        prompt = str(state.get("prompt") or "").strip()
        greeting_like = False
        if prompt:
            prompt_lc = prompt.lower()
            greeting_terms = (
                "hi",
                "hello",
                "hey",
                "greetings",
                "welcome",
                "good morning",
                "good afternoon",
                "good evening",
            )
            greeting_like = any(
                prompt_lc == term or prompt_lc.startswith(f"{term} ")
                for term in greeting_terms
            )

        reason = str(summary.get("planner_reason") or "").lower()
        if any(keyword in reason for keyword in ("greet", "hello", "welcome", "triage", "intro")):
            greeting_like = True

        metadata = state.get("metadata")
        if isinstance(metadata, Mapping):
            intent = metadata.get("intent")
            if isinstance(intent, str) and intent.lower() in {"greeting", "smalltalk"}:
                greeting_like = True

        return greeting_like

    @staticmethod
    def _extract_tool_policy_override(state: Mapping[str, Any], *, agent_name: str | None = None) -> Mapping[str, Any] | None:
        outputs = state.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return None
        for entry in reversed(outputs):
            if not isinstance(entry, Mapping):
                continue
            if agent_name and entry.get("agent") != agent_name:
                continue
            metadata = entry.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            override = metadata.get("tool_policy_override")
            if isinstance(override, Mapping):
                return dict(override)
            if agent_name:
                # Stop searching once we've inspected the most recent record for this agent
                break
        return None

    async def route_task(
        self,
        task: dict[str, Any],
        *,
        context: AgentContext,
        progress_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        logger.info("task_routed", task=task.get("id"))
        entry_point = str(task.get("source") or "api")
        task_id = str(task.get("id") or task.get("task_id") or uuid.uuid4())
        task["id"] = task_id
        task.setdefault("outputs", [])
        self._reset_plan_state()
        run_tracker = _RunTracker(run=None, entry_point=entry_point)
        mark_orchestrator_run_started(entry_point=entry_point, task_id=task_id)
        if not self.agents:
            mark_orchestrator_run_completed(entry_point=entry_point, task_id=task_id, status="skipped", latency=0.0)
            return {"status": "noop", "outputs": [], **task}

        try:
            agent_sequence, routing_context, skipped_agents = await self._determine_agent_sequence(
                task,
                run_tracker=run_tracker,
            )
        except PlannerError as exc:
            error_message = str(exc) or "Planner failure"
            task.setdefault("outputs", [])
            task.setdefault("routing", {})
            planner_section = task["routing"].get("planner")
            task["status"] = "failed"
            task["error"] = error_message
            task.setdefault("failure", {})
            task["failure"].setdefault("type", "planner_failed")
            if planner_section and isinstance(planner_section, dict):
                planner_section.setdefault("status", "failed")
                planner_section.setdefault("error", error_message)
            await self._notify(
                progress_cb,
                self._ensure_run_id(
                    run_tracker,
                    {
                        "event": "planner_failed",
                        "task_id": task_id,
                        "error": error_message,
                    },
                ),
            )
            logger.error("planner_failed", task=task_id, error=error_message)
            await self._finalize_run(run_tracker, task, status=OrchestratorStatus.FAILED, error=error_message)
            return task

        task.setdefault("routing", {}).update(routing_context)
        await self._notify(
            progress_cb,
            self._ensure_run_id(
                run_tracker,
                {
                    "event": "routing_decided",
                    "task_id": task_id,
                    **{key: value for key, value in routing_context.items() if key != "available_agents"},
                },
            ),
        )
        logger.info(
            "agent_routing_decided",
            task=task_id,
            selected=routing_context.get("selected_agents", []),
            skipped=routing_context.get("skipped_agents", []),
            reason=routing_context.get("reason"),
        )

        if not agent_sequence:
            mark_orchestrator_run_completed(
                entry_point=entry_point,
                task_id=task_id,
                status="skipped",
                latency=time.perf_counter() - run_tracker.started_at,
            )
            task["status"] = "noop"
            task.setdefault("outputs", [])
            return task
        if self._state_store is not None:
            run = await self._state_store.start_run(task_id, state=task)
            task["run_id"] = str(run.run_id)
            run_tracker.run = run
            await self._update_state(run_tracker, task, status=OrchestratorStatus.RUNNING)
            await self._record_event(run_tracker, "run_started", payload={"task_id": task_id})
            await self._capture_snapshot(run_tracker, ContextStage.INTAKE, task)
        await self._record_event(run_tracker, "routing_decided", payload=routing_context)

        try:
            return await self._run_sequential(
                task,
                context=context,
                progress_cb=progress_cb,
                run_tracker=run_tracker,
                agent_sequence=agent_sequence,
                skipped_agents=skipped_agents,
            )
        finally:
            self._reset_plan_state()

    async def _run_sequential(
        self,
        task: dict[str, Any],
        *,
        context: AgentContext,
        progress_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        run_tracker: _RunTracker | None = None,
        agent_sequence: Sequence[BaseAgent] | None = None,
        skipped_agents: Sequence[BaseAgent] | None = None,
    ) -> dict[str, Any]:
        state = task
        state.setdefault("outputs", [])
        state.setdefault("shared_context", {"provenance": []})
        run_tracker = self._begin_run(run_tracker, task=state)
        self._current_context = context
        try:
            self._enforce_run_time_budget(run_tracker)
            task_id = str(state.get("id") or state.get("task_id") or "")
            failure: Exception | None = None
            plan_steps = list(self._active_plan_steps or [])
            using_plan_steps = bool(plan_steps)
            roster_index = self._roster_index if self._roster_index else {agent.name: agent for agent in self.agents}
            execution_plan: list[tuple[int, PlannedAgentStep | None, BaseAgent]] = []
            if using_plan_steps:
                for idx, plan_step in enumerate(plan_steps):
                    candidate = roster_index.get(plan_step.agent)
                    if candidate is None:
                        logger.error(
                            "planner_step_agent_missing",
                            task=task.get("id"),
                            step_index=idx,
                            planned_agent=plan_step.agent,
                        )
                        continue
                    execution_plan.append((idx, plan_step, candidate))
                if not execution_plan:
                    using_plan_steps = False
            if not execution_plan:
                derived_agents = list(agent_sequence or self.agents)
                execution_plan = [(idx, None, agent) for idx, agent in enumerate(derived_agents)]
            agents = [agent for _, _, agent in execution_plan]
            if skipped_agents:
                routing_state = state.get("routing") if isinstance(state.get("routing"), dict) else {}
                scores = routing_state.get("scores") if isinstance(routing_state.get("scores"), dict) else {}
                for skipped in skipped_agents:
                    await self._record_event(
                        run_tracker,
                        "agent_skipped",
                        agent=skipped.name,
                        payload={"score": scores.get(skipped.name)},
                    )
                    await self._notify(
                        progress_cb,
                        self._ensure_run_id(
                            run_tracker,
                            {
                                "event": "agent_skipped",
                                "agent": skipped.name,
                                "task_id": task.get("id"),
                                "score": scores.get(skipped.name),
                            },
                        ),
                    )
                logger.info(
                    "agents_skipped",
                    task=task.get("id"),
                    agents=[agent.name for agent in skipped_agents],
                )

            # Track indices to skip due to early exit
            early_exit_skipped_indices: set[int] = set()
            
            # Group steps by parallel_group for concurrent execution
            parallel_groups: dict[int | None, list[tuple[int, PlannedAgentStep | None, BaseAgent]]] = {}
            for step_info in execution_plan:
                step_index, plan_step, planned_agent = step_info
                group_id = getattr(plan_step, "parallel_group", None) if plan_step else None
                parallel_groups.setdefault(group_id, []).append(step_info)
            
            # Track completed agents for dependency checking
            completed_agents: set[str] = set()
            
            # Process each group
            sorted_group_ids = sorted(
                [g for g in parallel_groups.keys() if g is not None],
                key=lambda x: x if x is not None else float('inf')
            )
            # Add None group at the end (non-parallelizable steps)
            if None in parallel_groups:
                sorted_group_ids.append(None)
            
            for group_id in sorted_group_ids:
                group_steps = parallel_groups[group_id]
                
                # Filter out skipped steps and SORT BY INDEX to ensure correct order
                active_steps = sorted(
                    [(idx, step, agent) for idx, step, agent in group_steps
                     if idx not in early_exit_skipped_indices],
                    key=lambda x: x[0]  # Sort by step_index
                )
                
                if not active_steps:
                    continue
                
                # Check if all dependencies are met for this group
                if group_id is not None and len(active_steps) > 1:
                    # Run in parallel
                    logger.info(
                        "parallel_execution_started",
                        task=task.get("id"),
                        group_id=group_id,
                        agents=[agent.name for _, _, agent in active_steps],
                    )
                    
                    # Create tasks for parallel execution
                    parallel_tasks = []
                    for step_index, plan_step, planned_agent in active_steps:
                        task_coro = self._execute_single_agent(
                            state=state,
                            step_index=step_index,
                            plan_step=plan_step,
                            planned_agent=planned_agent,
                            context=context,
                            run_tracker=run_tracker,
                            progress_cb=progress_cb,
                            roster_index=roster_index,
                            task_id=task_id,
                            using_plan_steps=using_plan_steps,
                        )
                        parallel_tasks.append(task_coro)
                    
                    # Execute all agents in this group in parallel
                    results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
                    
                    # Process results and check for failures
                    for i, result in enumerate(results):
                        step_index, plan_step, planned_agent = active_steps[i]
                        if isinstance(result, Exception):
                            logger.error(
                                "parallel_agent_failed",
                                task=task.get("id"),
                                agent=planned_agent.name,
                                error=str(result),
                            )
                            failure = result
                            state["status"] = "failed"
                            state["error"] = str(result)
                            if not isinstance(result, SafetyLimitExceeded):
                                await self._finalize_run(run_tracker, state, status=OrchestratorStatus.FAILED, error=str(result))
                            raise result
                        else:
                            # Result contains the executed agent and any early exit info
                            exec_result = result or {}
                            if exec_result.get("early_exit_skipped"):
                                early_exit_skipped_indices.update(exec_result["early_exit_skipped"])
                            completed_agents.add(planned_agent.name)
                    
                    logger.info(
                        "parallel_execution_completed",
                        task=task.get("id"),
                        group_id=group_id,
                        agents=[agent.name for _, _, agent in active_steps],
                    )
                else:
                    # Run sequentially (either single step or no parallel group)
                    for step_index, plan_step, planned_agent in active_steps:
                        # Skip this step if it was marked for early exit
                        if step_index in early_exit_skipped_indices:
                            continue
                        
                        self._enforce_run_time_budget(run_tracker)
                        executed_agent = planned_agent
                        resolved_step = plan_step
                        fallback_agent: BaseAgent | None = None
                        fallback_reason: str | None = None
                        step_router_metadata: dict[str, Any] | None = None
                        step_confidence = 1.0
                        if plan_step is not None:
                            step_confidence = self._coerce_plan_confidence(getattr(plan_step, "confidence", 1.0))
                            plan_step.confidence = step_confidence
                            if self._is_low_confidence(step_confidence):
                                exclude = {plan_step.agent}
                                router_candidate, router_meta = await self._router_pick_agent(
                                    task,
                                    list(roster_index.values()),
                                    exclude=exclude,
                                )
                                candidate = router_candidate or roster_index.get("general_agent")
                                if candidate is not None and candidate.name != planned_agent.name:
                                    fallback_agent = candidate
                                    fallback_reason = (
                                        f"Planner confidence {step_confidence:.2f} below threshold "
                                        f"{self._low_confidence_threshold:.2f}; rerouting to {candidate.name}."
                                    )
                                    executed_agent = candidate
                                    resolved_step = PlannedAgentStep(
                                        agent=candidate.name,
                                        tools=[],
                                        fallback_tools=list(plan_step.fallback_tools),
                                        reason=(
                                            f"{fallback_reason} Original agent: {plan_step.agent}. {plan_step.reason}"
                                        ).strip(),
                                        confidence=step_confidence,
                                    )
                                    step_router_metadata = router_meta if router_candidate is not None else None
                                    logger.warning(
                                        "planner_step_low_confidence_fallback",
                                        task=task.get("id"),
                                        step_index=step_index,
                                        planned_agent=plan_step.agent,
                                        fallback_agent=candidate.name,
                                        confidence=step_confidence,
                                        threshold=self._low_confidence_threshold,
                                        router=step_router_metadata,
                                    )
                                else:
                                    resolved_step = replace(
                                        plan_step,
                                        confidence=step_confidence,
                                        tools=list(plan_step.tools),
                                        fallback_tools=list(plan_step.fallback_tools),
                                    )
                            else:
                                resolved_step = replace(
                                    plan_step,
                                    confidence=step_confidence,
                                    tools=list(plan_step.tools),
                                    fallback_tools=list(plan_step.fallback_tools),
                                )
                        step_context = resolved_step or plan_step
                        step_id = f"agent::{executed_agent.name}"
                        if using_plan_steps:
                            step_id = f"{step_id}::{step_index}"
                        step_metadata: dict[str, Any] | None = None
                        if plan_step is not None or step_context is not None:
                            step_metadata = self._build_planner_step_metadata(
                                index=step_index,
                                planned_step=plan_step,
                                resolved_step=step_context,
                                executed_agent=executed_agent,
                                fallback_agent=fallback_agent,
                                fallback_reason=fallback_reason,
                                router_metadata=step_router_metadata,
                            )
                            step_metadata.setdefault("step_id", step_id)
                            self._append_step_metadata(state, step_metadata)
                            if fallback_agent is not None:
                                self._record_step_override(state, step_metadata)
                            logger.info(
                                "planner_step_started",
                                task=task.get("id"),
                                step_index=step_index,
                                planned_agent=step_metadata.get("planned_agent"),
                                executed_agent=step_metadata.get("executed_agent"),
                                reason=step_metadata.get("executed_reason") or step_metadata.get("planned_reason"),
                                tools=step_metadata.get("executed_tools") or step_metadata.get("planned_tools"),
                                fallback_applied=step_metadata.get("fallback_applied"),
                            )
                        event_step_metadata = dict(step_metadata) if step_metadata is not None else None
                        if event_step_metadata is not None:
                            event_step_metadata["status"] = "started"
                        await self._record_event(
                            run_tracker,
                            "agent_started",
                            agent=executed_agent.name,
                            payload={"task_id": task.get("id"), "planner_step": event_step_metadata},
                        )
                        await self._notify(
                            progress_cb,
                            self._ensure_run_id(
                                run_tracker,
                                {
                                    "event": "agent_started",
                                    "agent": executed_agent.name,
                                    "task_id": task.get("id"),
                                    "step_id": step_id,
                                    "planner_step": event_step_metadata,
                                },
                            ),
                        )
                        increment_agent_event(agent=executed_agent.name, event="started")
                        logger.info("agent_started", agent=executed_agent.name, task=task.get("id"))
                        start_time = time.perf_counter()
                        await self._record_lifecycle(
                            run_tracker,
                            task_id=task_id,
                            step_id=step_id,
                            event_type="agent_started",
                            status=LifecycleStatus.IN_PROGRESS,
                            agent=executed_agent.name,
                        )
                        await self._capture_snapshot(
                            run_tracker,
                            ContextStage.EXECUTION,
                            state,
                            agent=executed_agent.name,
                            extra={"event": "agent_started", "planner_step": event_step_metadata},
                        )
                        try:
                            tool_session = self._start_tool_session(
                                agent=executed_agent,
                                base_context=context,
                                plan_step=step_context,
                                run_tracker=run_tracker,
                            )
                            
                            # Build collaboration handler for this agent
                            async def collab_handler(target: str, request: str, ctx_data: dict[str, Any]) -> AgentOutput:
                                return await self._handle_collaboration(
                                    target_agent_name=target,
                                    request=request,
                                    context_data=ctx_data,
                                    base_context=context,
                                    run_tracker=run_tracker,
                                    roster_index=roster_index,
                                    state=state,
                                )
                            
                            agent_context = self._clone_context(
                                context,
                                tool_session,
                                step_context,
                                agent_name=executed_agent.name,
                                collaboration_handler=collab_handler,
                            )
                            agent_input = await self._prepare_agent_input(state, agent=executed_agent, context=agent_context)
                            result = await executed_agent.handle(agent_input, context=agent_context)
                            
                            # Evaluate output quality
                            tools_used = list(tool_session.used_tools) if tool_session else None
                            tools_expected = list(plan_step.tools) if plan_step and plan_step.tools else None
                            quality_acceptable, quality_info = await self._evaluate_output_quality(
                                state=state,
                                agent=executed_agent,
                                result=result,
                                tools_used=tools_used,
                                tools_expected=tools_expected,
                                run_tracker=run_tracker,
                                progress_cb=progress_cb,
                            )
                            
                            # Add quality info to result metadata
                            if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                                result.metadata["quality_score"] = quality_info.get("score")
                                result.metadata["quality_breakdown"] = quality_info.get("breakdown")
                            
                            self._record_output(state, agent=executed_agent, result=result, planner_step=step_metadata)
                            await self._enforce_tool_first_policy(
                                tool_session,
                                agent=executed_agent,
                                state=state,
                                run_tracker=run_tracker,
                                progress_cb=progress_cb,
                            )
                        except Exception as exc:
                            duration = time.perf_counter() - start_time
                            observe_agent_latency(agent=executed_agent.name, latency=duration)
                            increment_agent_event(agent=executed_agent.name, event="failed")
                            
                            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                            # Dynamic Re-Planning: Try to recover from agent failure
                            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                            replan_result = await self._try_dynamic_replan(
                                state=state,
                                failed_agent=executed_agent,
                                error=exc,
                                context=context,
                                run_tracker=run_tracker,
                                progress_cb=progress_cb,
                                roster_index=roster_index,
                            )
                            
                            if replan_result is not None:
                                # Re-planning succeeded - continue execution
                                logger.info(
                                    "dynamic_replan_success",
                                    task=task.get("id"),
                                    failed_agent=executed_agent.name,
                                    recovery_agent=replan_result.get("recovery_agent"),
                                )
                                # Update state with re-planning result
                                completed_agents.add(executed_agent.name)
                                continue  # Move to next step
                            
                            # Re-planning failed - proceed with original failure handling
                            failure = exc
                            failed_step_metadata = dict(step_metadata) if step_metadata is not None else None
                            if failed_step_metadata is not None:
                                failed_step_metadata["status"] = "failed"
                            await self._record_lifecycle(
                                run_tracker,
                                task_id=task_id,
                                step_id=step_id,
                                event_type="agent_failed",
                                status=LifecycleStatus.FAILED,
                                agent=executed_agent.name,
                                latency_ms=duration * 1000,
                            )
                            await self._notify(
                                progress_cb,
                                self._ensure_run_id(
                                    run_tracker,
                                    {
                                        "event": "agent_failed",
                                        "agent": executed_agent.name,
                                        "task_id": task.get("id"),
                                        "error": str(exc),
                                        "latency": duration,
                                        "step_id": step_id,
                                        "planner_step": failed_step_metadata,
                                    },
                                ),
                            )
                            logger.exception(
                                "agent_failed",
                                agent=executed_agent.name,
                                task=task.get("id"),
                                duration=duration,
                            )
                            if step_metadata is not None:
                                logger.error(
                                    "planner_step_failed",
                                    task=task.get("id"),
                                    step_index=step_index,
                                    executed_agent=executed_agent.name,
                                    error=str(exc),
                                )
                            state["status"] = "failed"
                            state["error"] = str(exc)
                            await self._record_event(
                                run_tracker,
                                "agent_failed",
                                agent=executed_agent.name,
                                payload={"error": str(exc), "latency": duration, "planner_step": failed_step_metadata},
                            )
                            await self._capture_snapshot(
                                run_tracker,
                                ContextStage.EXECUTION,
                                state,
                                agent=executed_agent.name,
                                extra={"event": "agent_failed", "error": str(exc), "planner_step": failed_step_metadata},
                            )
                            if not isinstance(exc, SafetyLimitExceeded):
                                await self._finalize_run(run_tracker, state, status=OrchestratorStatus.FAILED, error=str(exc))
                            raise
                        duration = time.perf_counter() - start_time
                        observe_agent_latency(agent=executed_agent.name, latency=duration)
                        increment_agent_event(agent=executed_agent.name, event="completed")
                        logger.info(
                            "agent_completed",
                            agent=executed_agent.name,
                            task=task.get("id"),
                            duration=duration,
                            outputs=len(state.get("outputs", [])),
                        )
                        if step_metadata is not None:
                            logger.info(
                                "planner_step_completed",
                                task=task.get("id"),
                                step_index=step_index,
                                executed_agent=step_metadata.get("executed_agent"),
                                latency=duration,
                                fallback_applied=step_metadata.get("fallback_applied"),
                            )
                        latest_output = state.get("outputs", [])[-1] if state.get("outputs") else None
                        completed_step_metadata = dict(step_metadata) if step_metadata is not None else None
                        if completed_step_metadata is not None:
                            completed_step_metadata["status"] = "completed"
                        await self._notify(
                            progress_cb,
                            self._ensure_run_id(
                                run_tracker,
                                {
                                    "event": "agent_completed",
                                    "agent": executed_agent.name,
                                    "task_id": task.get("id"),
                                    "latency": duration,
                                    "output": latest_output,
                                    "step_id": step_id,
                                    "planner_step": completed_step_metadata,
                                },
                            ),
                        )
                        await self._record_event(
                            run_tracker,
                            "agent_completed",
                            agent=executed_agent.name,
                            payload={"latency": duration, "output": latest_output, "planner_step": completed_step_metadata},
                        )
                        await self._record_lifecycle(
                            run_tracker,
                            task_id=task_id,
                            step_id=step_id,
                            event_type="agent_completed",
                            status=LifecycleStatus.COMPLETED,
                            agent=executed_agent.name,
                            latency_ms=duration * 1000,
                        )
                        await self._capture_snapshot(
                            run_tracker,
                            ContextStage.EXECUTION,
                            state,
                            agent=executed_agent.name,
                            extra={"event": "agent_completed", "output": latest_output, "planner_step": completed_step_metadata},
                        )
                        await self._update_state(run_tracker, state)
                        self._enforce_run_time_budget(run_tracker)
                        
                        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        # Early Exit: Skip general_agent if specialist already responded
                        # with high confidence to avoid duplicate/redundant responses
                        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        if self._should_early_exit(
                            executed_agent=executed_agent,
                            latest_output=latest_output,
                            execution_plan=execution_plan,
                            current_step_index=step_index,
                        ):
                            skipped_remaining = [
                                (idx, step, agent)
                                for idx, step, agent in execution_plan
                                if idx > step_index and agent.name == "general_agent"
                            ]
                            for skip_idx, skip_step, skip_agent in skipped_remaining:
                                # Mark this index for skipping in subsequent iterations
                                early_exit_skipped_indices.add(skip_idx)
                                await self._record_event(
                                    run_tracker,
                                    "agent_skipped",
                                    agent=skip_agent.name,
                                    payload={"reason": "early_exit_high_confidence_specialist"},
                                )
                                await self._notify(
                                    progress_cb,
                                    self._ensure_run_id(
                                        run_tracker,
                                        {
                                            "event": "agent_skipped",
                                            "agent": skip_agent.name,
                                            "task_id": task.get("id"),
                                            "reason": "Specialist agent provided high-confidence response",
                                        },
                                    ),
                                )
                                logger.info(
                                    "agent_skipped_early_exit",
                                    task=task.get("id"),
                                    skipped_agent=skip_agent.name,
                                    reason="specialist_high_confidence",
                                    specialist=executed_agent.name,
                                    confidence=latest_output.get("confidence") if isinstance(latest_output, dict) else None,
                                )

            self._enforce_run_time_budget(run_tracker)
            if state.get("status") != "failed":
                meta_resolution: MetaResolution | None = None
                if self._negotiation is not None:
                    await self._apply_negotiation(state, run_tracker, progress_cb=progress_cb)
                else:
                    await self._generate_plan(state, run_tracker, progress_cb=progress_cb)
                should_run_meta, skip_reason = self._should_run_meta_pipeline(state)
                if should_run_meta:
                    meta_resolution = await self._synthesize_meta_resolution(state, run_tracker)
                    await self._assemble_dossier(state, run_tracker, meta_resolution=meta_resolution)
                else:
                    if skip_reason:
                        logger.info("meta_pipeline_skipped", task=task.get("id"), reason=skip_reason)
                        await self._record_event(
                            run_tracker,
                            "meta_pipeline_skipped",
                            payload={"reason": skip_reason},
                        )
                    meta_section = state.setdefault("meta", {})
                    meta_section.setdefault("status", "skipped")
                    if skip_reason:
                        meta_section.setdefault("reason", skip_reason)
                    dossier_section = state.setdefault("dossier", {})
                    dossier_section.setdefault("status", "skipped")
                    if skip_reason:
                        dossier_section.setdefault("reason", skip_reason)
                state["status"] = "completed"
                await self._finalize_run(
                    run_tracker,
                    state,
                    status=OrchestratorStatus.COMPLETED if failure is None else OrchestratorStatus.FAILED,
                    error=str(failure) if failure is not None else None,
                )
                return state
        except SafetyLimitExceeded as exc:
            self._register_abort(run_tracker, reason=exc.reason)
            logger.error(
                "safety_limit_exceeded",
                task=state.get("id"),
                reason=exc.reason,
                error=str(exc),
            )
            state["status"] = "failed"
            state["error"] = str(exc)
            failure_section = state.setdefault("failure", {})
            failure_section.setdefault("type", exc.reason)
            failure_section.setdefault("message", str(exc))
            await self._record_event(
                run_tracker,
                "run_aborted",
                payload={"reason": exc.reason, "message": str(exc)},
            )
            await self._capture_snapshot(
                run_tracker,
                ContextStage.EXECUTION,
                state,
                extra={"event": "run_aborted", "reason": exc.reason, "error": str(exc)},
            )
            await self._finalize_run(run_tracker, state, status=OrchestratorStatus.FAILED, error=str(exc))
            raise
        finally:
            self._current_context = None
            self._complete_run()

    async def _determine_agent_sequence(
        self,
        task: dict[str, Any],
        run_tracker: _RunTracker | None = None,
    ) -> tuple[list[BaseAgent], dict[str, Any], list[BaseAgent]]:
        roster = list(self.agents)
        self._active_plan_steps = None
        self._active_plan_step_lookup = None
        self._roster_index = {agent.name: agent for agent in roster}

        if not roster:
            planner_payload = {
                "strategy": "llm_orchestration",
                "steps": [],
                "selected_agents": [],
                "metadata": {},
                "raw_response": None,
                "status": "empty",
            }
            routing_payload = {
                "selected_agents": [],
                "available_agents": [],
                "skipped_agents": [],
                "scores": {},
                "reason": "planner.no_agents",
                "metadata": {"planner": planner_payload},
            }
            self._record_planner_metadata(task, planner_payload)
            return [], routing_payload, []

        if self._llm_planner is None:
            error_message = "LLM planner not configured"
            self._register_planner_failure(task, roster, status="failed", error=error_message)
            record_plan_metrics(strategy="llm_orchestration", status="failed")
            raise PlannerError(error_message)

        try:
            self._enforce_planner_budget(run_tracker)
            planner_plan = await self._llm_planner.plan(
                task=task,
                prior_outputs=self._coerce_prior_outputs(task),
                agents=roster,
                tool_aliases=self._collect_tool_aliases(task),
            )
        except SafetyLimitExceeded:
            raise
        except PlannerError as exc:
            error_message = str(exc)
            self._register_planner_failure(task, roster, status="failed", error=error_message)
            record_plan_metrics(strategy="llm_orchestration", status="failed")
            raise
        except Exception as exc:  # pragma: no cover - defensive guard for unexpected planner failures
            error_message = str(exc)
            self._register_planner_failure(task, roster, status="failed", error=error_message)
            record_plan_metrics(strategy="llm_orchestration", status="failed")
            raise PlannerError("Planner execution failed") from exc

        if not planner_plan.steps:
            empty_reason = "Planner produced empty plan"
            routing_payload = self._build_planner_routing_payload(
                planner_plan,
                roster,
                [],
                status="failed",
                reason="planner.empty_plan",
                error=empty_reason,
            )
            task.setdefault("routing", {}).update(routing_payload)
            self._annotate_task_with_plan(task, planner_plan, [], status="failed", error=empty_reason)
            record_plan_metrics(strategy="llm_orchestration", status="failed", steps=len(planner_plan.steps))
            raise PlannerError(empty_reason)

        self._agent_tool_catalog = self._build_agent_tool_catalog(roster)
        self._validate_plan_tools(planner_plan.steps)

        base_confidence = getattr(planner_plan, "confidence", None)
        metadata_confidence = planner_plan.metadata.get("confidence", base_confidence or 1.0)
        confidence = self._coerce_plan_confidence(metadata_confidence)
        planner_plan.confidence = confidence
        planner_plan.metadata["confidence"] = confidence
        if self._is_low_confidence(confidence):
            router_selection = await self._router_low_confidence_selection(task, roster)
            if router_selection is not None:
                selection, router_meta = router_selection
                planner_plan.metadata.setdefault("fallback_reason", "low_confidence_router")
                planner_plan.metadata.setdefault("confidence_threshold", self._low_confidence_threshold)
                planner_plan.metadata.setdefault("router_fallback", router_meta)
                logger.warning(
                    "planner_low_confidence_router",
                    task=task.get("id"),
                    confidence=confidence,
                    threshold=self._low_confidence_threshold,
                    agents=[agent.name for agent in selection],
                )
                routing_payload = self._build_planner_routing_payload(
                    planner_plan,
                    roster,
                    selection,
                    status="fallback",
                    reason="planner.low_confidence_router",
                )
                routing_payload.setdefault("metadata", {}).setdefault("router", router_meta)
                self._annotate_task_with_plan(task, planner_plan, selection, status="fallback")
                record_plan_metrics(
                    strategy="llm_orchestration",
                    status="fallback",
                    steps=len(planner_plan.steps),
                )
                skipped_agents = [agent for agent in roster if agent not in selection]
                return selection, routing_payload, skipped_agents
            fallback_agent = next((agent for agent in roster if agent.name == "general_agent"), None)
            if fallback_agent is not None:
                planner_plan.metadata.setdefault("fallback_reason", "low_confidence")
                planner_plan.metadata.setdefault("confidence_threshold", self._low_confidence_threshold)
                planner_plan.metadata.setdefault("fallback_selected_agents", [fallback_agent.name])
                logger.warning(
                    "planner_low_confidence_fallback",
                    task=task.get("id"),
                    confidence=confidence,
                    threshold=self._low_confidence_threshold,
                    fallback_agent=fallback_agent.name,
                )
                selection = [fallback_agent]
                routing_payload = self._build_planner_routing_payload(
                    planner_plan,
                    roster,
                    selection,
                    status="fallback",
                    reason="planner.low_confidence",
                )
                self._annotate_task_with_plan(task, planner_plan, selection, status="fallback")
                record_plan_metrics(
                    strategy="llm_orchestration",
                    status="fallback",
                    steps=len(planner_plan.steps),
                )
                skipped_agents = [agent for agent in roster if agent not in selection]
                return selection, routing_payload, skipped_agents
            logger.warning(
                "planner_low_confidence_no_general_agent",
                task=task.get("id"),
                confidence=confidence,
                threshold=self._low_confidence_threshold,
                roster=[agent.name for agent in roster],
            )

        self._active_plan_steps = list(planner_plan.steps)
        lookup: dict[str, PlannedAgentStep] = {}
        for step in planner_plan.steps:
            lookup.setdefault(step.agent, step)
        self._active_plan_step_lookup = lookup

        selection, missing_agents = self._resolve_planner_agents(planner_plan.steps, roster)
        if missing_agents:
            logger.error(
                "planner_unknown_agents",
                task=task.get("id"),
                agents=sorted({name for name in missing_agents}),
            )
            invalid_reason = "Planner referenced unknown agents"
            routing_payload = self._build_planner_routing_payload(
                planner_plan,
                roster,
                [],
                status="failed",
                reason="planner.unknown_agents",
                error=invalid_reason,
            )
            task.setdefault("routing", {}).update(routing_payload)
            self._annotate_task_with_plan(task, planner_plan, [], status="failed", error=invalid_reason)
            record_plan_metrics(strategy="llm_orchestration", status="failed", steps=len(planner_plan.steps))
            self._reset_plan_state()
            raise PlannerError(invalid_reason)
        if not selection:
            invalid_reason = "Planner referenced unknown agents"
            routing_payload = self._build_planner_routing_payload(
                planner_plan,
                roster,
                [],
                status="failed",
                reason="planner.unknown_agents",
                error=invalid_reason,
            )
            task.setdefault("routing", {}).update(routing_payload)
            self._annotate_task_with_plan(task, planner_plan, [], status="failed", error=invalid_reason)
            record_plan_metrics(strategy="llm_orchestration", status="failed", steps=len(planner_plan.steps))
            self._reset_plan_state()
            raise PlannerError(invalid_reason)

        metadata = self._build_planner_routing_payload(
            planner_plan,
            roster,
            selection,
            status="planned",
        )
        skipped_agents = [agent for agent in roster if agent not in selection]
        self._annotate_task_with_plan(task, planner_plan, selection, status="planned")
        record_plan_metrics(strategy="llm_orchestration", status="planned", steps=len(planner_plan.steps))
        return selection, metadata, skipped_agents

    def _coerce_prior_outputs(self, task: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        outputs = task.get("outputs")
        if not isinstance(outputs, Sequence):
            return []
        return [item for item in outputs if isinstance(item, Mapping)]

    @staticmethod
    def _coerce_plan_confidence(raw_value: Any) -> float:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(value):
            return 1.0
        return max(0.0, min(1.0, value))

    def _collect_tool_aliases(self, task: Mapping[str, Any]) -> dict[str, str] | None:
        metadata = task.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        aliases = metadata.get("tool_aliases")
        if not isinstance(aliases, Mapping):
            return None
        resolved: dict[str, str] = {}
        for key, value in aliases.items():
            if isinstance(key, str) and isinstance(value, str):
                resolved[key] = value
        return resolved or None

    def _resolve_planner_agents(
        self,
        steps: Sequence[PlannedAgentStep],
        roster: Sequence[BaseAgent],
    ) -> tuple[list[BaseAgent], list[str]]:
        index = {agent.name: agent for agent in roster}
        resolved: list[BaseAgent] = []
        missing: list[str] = []
        for step in steps:
            candidate = index.get(step.agent)
            if candidate is None:
                missing.append(step.agent)
                continue
            resolved.append(candidate)
        return resolved, missing

    def _build_planner_routing_payload(
        self,
    plan: PlannerPlan,
        roster: Sequence[BaseAgent],
        selection: Sequence[BaseAgent],
        *,
        status: str = "planned",
        reason: str = "planner.llm_selected",
        error: str | None = None,
    ) -> dict[str, Any]:
        scores: dict[str, float] = {}
        for position, agent in enumerate(selection):
            scores[agent.name] = max(0.0, 1.0 - (position * 0.1))
        for agent in roster:
            scores.setdefault(agent.name, 0.0)
        planner_steps = [
            {
                "agent": step.agent,
                "tools": step.tools,
                "fallback_tools": step.fallback_tools,
                "reason": step.reason,
                "confidence": getattr(step, "confidence", None),
            }
            for step in plan.steps
        ]
        planner_metadata = {
            "strategy": "llm_orchestration",
            "steps": planner_steps,
            "raw_response": plan.raw_response,
            "attributes": plan.metadata,
            "status": status,
        }
        if error:
            planner_metadata["error"] = error
        payload: dict[str, Any] = {
            "selected_agents": [agent.name for agent in selection],
            "available_agents": [agent.name for agent in roster],
            "skipped_agents": [agent.name for agent in roster if agent not in selection],
            "scores": scores,
            "reason": reason,
            "metadata": {"planner": planner_metadata},
        }
        return payload

    def _annotate_task_with_plan(
        self,
        task: dict[str, Any],
    plan: PlannerPlan,
        selection: Sequence[BaseAgent],
        *,
        status: str = "planned",
        error: str | None = None,
    ) -> None:
        planner_steps = [
            {
                "agent": step.agent,
                "tools": step.tools,
                "fallback_tools": step.fallback_tools,
                "reason": step.reason,
                "confidence": getattr(step, "confidence", None),
            }
            for step in plan.steps
        ]
        planner_payload = {
            "strategy": "llm_orchestration",
            "steps": planner_steps,
            "selected_agents": [agent.name for agent in selection],
            "metadata": plan.metadata,
            "raw_response": plan.raw_response,
            "status": status,
        }
        if error:
            planner_payload["error"] = error
        self._record_planner_metadata(task, planner_payload)

    def _register_planner_failure(
        self,
        task: dict[str, Any],
        roster: Sequence[BaseAgent],
        *,
        status: str,
        error: str,
        raw_response: str | None = None,
    ) -> dict[str, Any]:
        planner_payload = {
            "strategy": "llm_orchestration",
            "steps": [],
            "selected_agents": [],
            "metadata": {},
            "raw_response": raw_response,
            "status": status,
            "error": error,
        }
        self._record_planner_metadata(task, planner_payload)
        routing_payload = {
            "selected_agents": [],
            "available_agents": [agent.name for agent in roster],
            "skipped_agents": [agent.name for agent in roster],
            "scores": {agent.name: 0.0 for agent in roster},
            "reason": f"planner.{status}",
            "metadata": {"planner": planner_payload},
        }
        task.setdefault("routing", {}).update(routing_payload)
        return routing_payload

    def _record_planner_metadata(self, task: dict[str, Any], payload: Mapping[str, Any]) -> None:
        routing_section = task.setdefault("routing", {})
        routing_section["planner"] = copy.deepcopy(dict(payload))
        shared = task.setdefault("shared_context", {})
        shared.setdefault("provenance", [])
        shared["planner"] = copy.deepcopy(dict(payload))

    def _extract_planner_status(self, state: Mapping[str, Any]) -> str | None:
        candidates: list[Mapping[str, Any]] = []
        shared_context = state.get("shared_context")
        if isinstance(shared_context, Mapping):
            planner_section = shared_context.get("planner")
            if isinstance(planner_section, Mapping):
                candidates.append(planner_section)
        routing_section = state.get("routing")
        if isinstance(routing_section, Mapping):
            planner_section = routing_section.get("planner")
            if isinstance(planner_section, Mapping):
                candidates.append(planner_section)
        for candidate in candidates:
            status = candidate.get("status")
            if isinstance(status, str) and status:
                return status.lower()
        return None

    def _should_run_meta_pipeline(self, state: Mapping[str, Any]) -> tuple[bool, str | None]:
        outputs = state.get("outputs")
        if not isinstance(outputs, Sequence) or not outputs:
            return False, "no_outputs"
        if len(outputs) <= 1:
            if self._meta_agent is None:
                return False, "single_output"
        planner_status = self._extract_planner_status(state)
        if planner_status and planner_status not in {"planned", "scheduled"}:
            return False, f"planner_status_{planner_status}"
        return True, None

    async def _prepare_agent_input(
        self,
        state: dict[str, Any],
        *,
        agent: BaseAgent,
        context: AgentContext,
    ) -> AgentInput:
        context_text: str | None = None
        if self._context_contract is not None:
            bundle = await self._context_contract.build_for_stage(
                ContextStage.EXECUTION,
                task=state,
                agent=agent.name,
            )
            context_text = bundle.as_prompt_section()
        elif context.context is not None:
            bundle = await context.context.build(task=state, agent=agent.name)
            context_text = bundle.as_prompt_section()

        payload = {
            "task_id": str(state.get("id") or state.get("task_id") or ""),
            "prompt": state.get("prompt", ""),
            "metadata": self._build_metadata_payload(state),
            "context": context_text,
            "prior_exchanges": self._serialize_prior_exchanges(state.get("outputs", [])),
        }
        return validate_agent_request(agent.capability, payload)

    def _build_metadata_payload(self, state: dict[str, Any]) -> dict[str, Any]:
        raw_metadata = state.get("metadata")
        metadata: dict[str, Any] = {}
        if isinstance(raw_metadata, Mapping):
            metadata = copy.deepcopy(raw_metadata)
        shared_context = state.get("shared_context")
        if isinstance(shared_context, dict) and shared_context:
            metadata.setdefault("_shared_context", copy.deepcopy(shared_context))
        return metadata

    def _record_output(
        self,
        state: dict[str, Any],
        *,
        agent: BaseAgent,
        result: AgentOutput | dict[str, Any],
        planner_step: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(result, AgentOutput):
            validated = validate_agent_response(agent.capability, result.model_dump())
        else:
            validated = validate_agent_response(agent.capability, result)
        
        # Skip recording outputs from agents that passed (confidence=0, or [PASS] summary)
        output_confidence = validated.confidence if hasattr(validated, 'confidence') else 0
        output_summary = validated.summary if hasattr(validated, 'summary') else ""
        output_metadata = validated.metadata if hasattr(validated, 'metadata') else {}
        
        # Check if agent passed (zero confidence, [PASS] marker, or explicit passed flag)
        is_pass_response = (
            output_confidence == 0.0 or 
            output_summary == "[PASS]" or 
            output_metadata.get("passed") is True
        )
        
        if is_pass_response:
            logger.info(
                "skipped_passed_agent_output",
                agent=agent.name,
                reason="agent_passed",
            )
            return state
        
        output_dict = validated.model_dump()
        output_dict.setdefault("content", validated.summary)
        output_dict.setdefault("agent", agent.name)
        metadata_section = output_dict.setdefault("metadata", {})
        if planner_step is not None:
            metadata_section.setdefault("planner_step", copy.deepcopy(dict(planner_step)))
        state.setdefault("outputs", []).append(output_dict)
        shared = state.setdefault("shared_context", {"provenance": []})
        provenance: list[dict[str, Any]] = shared.setdefault("provenance", [])
        planner_step_entry = copy.deepcopy(dict(planner_step)) if planner_step is not None else None
        provenance.append(
            {
                "agent": agent.name,
                "capability": agent.capability.value,
                "summary": output_dict.get("summary"),
                "confidence": output_dict.get("confidence"),
                "metadata": output_dict.get("metadata", {}),
                "planner_step": planner_step_entry,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(provenance) > 20:
            del provenance[:-20]
        shared["last_writer"] = agent.name
        return state

    async def _evaluate_output_quality(
        self,
        state: dict[str, Any],
        agent: BaseAgent,
        result: AgentOutput,
        tools_used: list[str] | None = None,
        tools_expected: list[str] | None = None,
        run_tracker: _RunTracker | None = None,
        progress_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Evaluate the quality of an agent's output.
        
        Returns:
            Tuple of (is_acceptable, quality_info)
            - is_acceptable: True if quality meets threshold
            - quality_info: Quality scoring breakdown
        """
        request = state.get("request", state.get("input", ""))
        
        quality_result = self._output_quality_scorer.score(
            response=getattr(result, "response", None) or result.summary or "",
            request=request,
            tools_used=tools_used,
            tools_expected=tools_expected,
            confidence=result.confidence,
        )
        
        quality_info = quality_result.as_dict()
        
        logger.info(
            "output_quality_evaluated",
            agent=agent.name,
            task=state.get("id"),
            score=quality_result.score,
            should_retry=quality_result.should_retry,
            issues=quality_result.issues,
        )
        
        await self._record_event(
            run_tracker,
            "output_quality_scored",
            agent=agent.name,
            payload={
                "score": quality_result.score,
                "breakdown": quality_info.get("breakdown", {}),
                "issues": quality_result.issues,
                "should_retry": quality_result.should_retry,
            },
        )
        
        if quality_result.should_retry and progress_cb:
            await self._notify(
                progress_cb,
                self._ensure_run_id(
                    run_tracker,
                    {
                        "event": "output_quality_low",
                        "agent": agent.name,
                        "task_id": state.get("id"),
                        "quality_score": quality_result.score,
                        "issues": quality_result.issues,
                    },
                ),
            )
        
        return not quality_result.should_retry, quality_info

    @staticmethod
    def _serialize_prior_exchanges(outputs: list[Any]) -> list[dict[str, Any]]:
        exchanges: list[dict[str, Any]] = []
        if not outputs:
            return exchanges
        for item in outputs:
            if not isinstance(item, dict):
                continue
            exchanges.append(
                {
                    "agent": item.get("agent", "unknown"),
                    "content": item.get("summary") or item.get("content") or "",
                    "confidence": item.get("confidence"),
                }
            )
        return exchanges

    @staticmethod
    async def _notify(
        callback: Callable[[dict[str, Any]], Awaitable[None]] | None,
        payload: dict[str, Any],
    ) -> None:
        if callback is None:
            return
        await callback(payload)

    @staticmethod
    def _ensure_run_id(tracker: _RunTracker | None, payload: dict[str, Any]) -> dict[str, Any]:
        if tracker is None or tracker.run is None:
            return payload
        enriched = dict(payload)
        enriched.setdefault("run_id", str(tracker.run.run_id))
        return enriched

    async def _record_event(
        self,
        tracker: _RunTracker | None,
        event_type: str,
        *,
        agent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if tracker is None or self._state_store is None or tracker.run is None:
            return
        tracker.sequence += 1
        event = OrchestratorEvent(
            run_id=tracker.run.run_id,
            sequence=tracker.sequence,
            event_type=event_type,
            agent=agent,
            payload=payload or {},
        )
        await self._state_store.record_event(event)

    async def _record_lifecycle(
        self,
        tracker: _RunTracker | None,
        *,
        task_id: str,
        step_id: str,
        event_type: str,
        status: LifecycleStatus,
        agent: str | None = None,
        attempt: int = 0,
        latency_ms: float | None = None,
        eta: datetime | None = None,
        deadline: datetime | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if tracker is None or self._lifecycle_store is None or tracker.run is None:
            return
        tracker.lifecycle_sequence += 1
        event = LifecycleEvent(
            task_id=task_id,
            step_id=step_id,
            event_type=event_type,
            status=status,
            payload=payload or {},
            agent=agent,
            sequence=tracker.lifecycle_sequence,
            run_id=tracker.run.run_id,
            attempt=attempt,
            eta=eta,
            deadline=deadline,
            latency_ms=latency_ms,
        )
        await self._lifecycle_store.record(event)

    def _task_id_from_state(self, state: dict[str, Any]) -> str:
        value = state.get("id") or state.get("task_id")
        return str(value) if value else ""

    async def _capture_snapshot(
        self,
        tracker: _RunTracker | None,
        stage: ContextStage,
        state: dict[str, Any],
        *,
        agent: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._snapshot_store is None:
            return
        task_id = self._task_id_from_state(state)
        if not task_id:
            return
        payload: dict[str, Any] = {
            "prompt": state.get("prompt"),
            "metadata": state.get("metadata"),
            "outputs": state.get("outputs", [])[-3:],
            "negotiation": state.get("negotiation"),
            "plan_status": state.get("plan", {}).get("status") if isinstance(state.get("plan"), dict) else None,
        }
        if extra:
            payload.update(extra)
        snapshot = ContextSnapshot(
            task_id=task_id,
            stage=stage,
            payload=payload,
            agent=agent,
            run_id=tracker.run.run_id if tracker is not None and tracker.run is not None else None,
        )
        try:
            await self._snapshot_store.record(snapshot)
        except Exception as exc:  # pragma: no cover - snapshot failures should not stop orchestrator
            logger.warning("snapshot_record_failed", stage=stage.value, error=str(exc))

    async def _update_state(
        self,
        tracker: _RunTracker | None,
        state: dict[str, Any],
        *,
        status: OrchestratorStatus | None = None,
    ) -> None:
        if tracker is None or self._state_store is None or tracker.run is None:
            return
        await self._state_store.update_state(tracker.run.run_id, state=state, status=status)

    async def _finalize_run(
        self,
        tracker: _RunTracker | None,
        state: dict[str, Any],
        *,
        status: OrchestratorStatus,
        error: str | None = None,
    ) -> None:
        task_id = self._task_id_from_state(state)
        if tracker is not None:
            duration = time.perf_counter() - tracker.started_at
            mark_orchestrator_run_completed(
                entry_point=tracker.entry_point,
                task_id=task_id or None,
                status=status.value,
                latency=duration,
            )
        if tracker is None or self._state_store is None or tracker.run is None:
            return
        event_type = "run_completed" if status is OrchestratorStatus.COMPLETED else f"run_{status.value}"
        await self._record_event(tracker, event_type, payload={"error": error} if error else None)
        await self._state_store.finalize_run(tracker.run.run_id, state=state, status=status, error=error)

    async def _apply_negotiation(
        self,
        state: dict[str, Any],
        tracker: _RunTracker | None,
        *,
        progress_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        if self._negotiation is None:
            return
        self._enforce_run_time_budget(tracker)
        try:
            await self._record_event(tracker, "negotiation_started", payload={"outputs": len(state.get("outputs", []))})
            decision = await self._negotiation.decide(state, state.get("outputs", []))
        except Exception as exc:  # pragma: no cover - negotiation failure should not break orchestrator
            logger.exception("negotiation_failed", error=str(exc))
            state.setdefault("negotiation", {})
            state["negotiation"].update({"status": "failed", "error": str(exc)})
            await self._record_event(tracker, "negotiation_failed", payload={"error": str(exc)})
            await self._update_state(tracker, state)
            return

        serialized = decision.model_dump()
        serialized["status"] = "completed"
        state["negotiation"] = serialized
        await self._record_event(tracker, "negotiation_completed", payload=serialized)
        if tracker is not None:
            tracker.negotiation_rounds = len(state.get("outputs", []))
            metadata = serialized.get("metadata") if isinstance(serialized.get("metadata"), dict) else {}
            tracker.negotiation_strategy = str(metadata.get("strategy", "unknown")) if isinstance(metadata, dict) else "unknown"
            tracker.negotiation_consensus = decision.consensus
            observe_negotiation_metrics(
                strategy=tracker.negotiation_strategy,
                rounds=tracker.negotiation_rounds,
                consensus=decision.consensus,
            )
        await self._update_state(tracker, state)
        await self._capture_snapshot(tracker, ContextStage.NEGOTIATION, state, extra={"negotiation": serialized})
        await self._generate_plan(state, tracker, progress_cb=progress_cb)

    async def _generate_plan(
        self,
        state: dict[str, Any],
        tracker: _RunTracker | None,
        *,
        progress_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        if self._planner is None:
            return
        self._enforce_run_time_budget(tracker)
        try:
            plan = await self._planner.build_plan(
                task=state,
                outputs=state.get("outputs", []),
                negotiation=state.get("negotiation"),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("plan_generation_failed", error=str(exc))
            state.setdefault("plan", {"status": "failed", "error": str(exc)})
            record_plan_metrics(strategy="unknown", status="failed")
            await self._record_event(tracker, "plan_failed", payload={"error": str(exc)})
            await self._update_state(tracker, state)
            return

        if plan is None:
            record_plan_metrics(strategy="unknown", status="skipped")
            return

        if self._scheduler is not None:
            try:
                plan = await self._scheduler.schedule(plan)
            except Exception as exc:  # pragma: no cover - scheduler failure surfaces to state
                logger.exception("plan_scheduling_failed", error=str(exc))
                state["plan"] = {"status": "failed", "error": str(exc)}
                metadata_payload = plan.metadata if isinstance(plan.metadata, dict) else {}
                strategy = str(metadata_payload.get("strategy") or "unknown")
                record_plan_metrics(strategy=strategy, status="failed")
                await self._record_event(tracker, "plan_failed", payload={"error": str(exc)})
                await self._update_state(tracker, state)
                return

        if self._lifecycle_store is not None and tracker is not None and tracker.run is not None:
            await self._lifecycle_store.record_plan(plan, run_id=tracker.run.run_id)

        guardrail_decisions: list[dict[str, Any]] = []
        plan_status = "scheduled" if any(step.eta_iso for step in plan.steps) else "planned"
        if self._guardrails is not None:
            for step in plan.steps:
                decision = await self._guardrails.evaluate_step(
                    task_id=plan.task_id,
                    run_id=tracker.run.run_id if tracker is not None and tracker.run is not None else None,
                    step=asdict(step),
                    agent=step.agent,
                )
                guardrail_decisions.append({"step_id": step.step_id, **decision.model_dump()})
                if progress_cb is not None and decision.decision is not GuardrailDecisionType.ALLOW:
                    await self._notify(
                        progress_cb,
                        self._ensure_run_id(
                            tracker,
                            {
                                "event": "guardrail_triggered",
                                "task_id": plan.task_id,
                                "step_id": step.step_id,
                                "agent": step.agent,
                                "decision": decision.decision.value,
                                "reason": decision.reason,
                                "risk_score": decision.risk_score,
                                "policy_id": decision.policy_id,
                                "metadata": decision.metadata or {},
                            },
                        ),
                    )
                if tracker is not None and decision.decision in {GuardrailDecisionType.ESCALATE, GuardrailDecisionType.REVIEW}:
                    tracker.escalations += 1
                if tracker is not None:
                    tracker.guardrail_decisions += 1
                if decision.decision is GuardrailDecisionType.DENY:
                    plan_status = "blocked"
                elif decision.decision in {GuardrailDecisionType.ESCALATE, GuardrailDecisionType.REVIEW} and plan_status != "blocked":
                    plan_status = "requires_review"

        for step in plan.steps:
            try:
                eta = datetime.fromisoformat(step.eta_iso) if step.eta_iso else None
                deadline = datetime.fromisoformat(step.deadline_iso) if step.deadline_iso else None
            except ValueError:
                eta = None
                deadline = None
            if eta is None or deadline is None:
                record_sla_event(category="unspecified")
            elif deadline >= eta:
                record_sla_event(category="met")
            else:
                record_sla_event(category="violated")

        serialized = {
            "task_id": plan.task_id,
            "summary": plan.summary,
            "steps": [asdict(step) for step in plan.steps],
            "metadata": plan.metadata,
            "status": plan_status,
        }
        state["plan"] = serialized
        if guardrail_decisions:
            state.setdefault("guardrails", {})
            state["guardrails"].setdefault("decisions", []).extend(guardrail_decisions)
        await self._record_event(tracker, "plan_generated", payload=serialized)
        metadata_payload = plan.metadata if isinstance(plan.metadata, dict) else {}
        strategy = str(metadata_payload.get("strategy") or serialized.get("metadata", {}).get("strategy") or "unknown")
        record_plan_metrics(strategy=strategy, status=plan_status, steps=len(plan.steps))
        await self._update_state(tracker, state)
        await self._capture_snapshot(
            tracker,
            ContextStage.PLANNING,
            state,
            extra={"plan": serialized, "guardrails": guardrail_decisions},
        )

    async def _synthesize_meta_resolution(
        self,
        state: dict[str, Any],
        tracker: _RunTracker | None,
    ) -> MetaResolution | None:
        if self._meta_agent is None:
            return None
        self._enforce_run_time_budget(tracker)
        try:
            resolution = await self._meta_agent.synthesize(
                task=state,
                outputs=state.get("outputs", []),
                negotiation=state.get("negotiation"),
            )
        except Exception as exc:  # pragma: no cover - meta agent failures should surface but not crash
            logger.exception("meta_resolution_failed", error=str(exc))
            state.setdefault("meta", {})
            state["meta"].update({"status": "failed", "error": str(exc)})
            await self._record_event(tracker, "meta_resolution_failed", payload={"error": str(exc)})
            await self._update_state(tracker, state)
            await self._capture_snapshot(
                tracker,
                ContextStage.CONSOLIDATION,
                state,
                extra={"meta": {"status": "failed", "error": str(exc)}},
            )
            return None

        payload = {"status": "completed", **resolution.as_dict()}
        state["meta"] = payload
        if resolution.should_escalate:
            sources = state.setdefault("escalation", {}).setdefault("sources", [])
            if "meta_agent" not in sources:
                sources.append("meta_agent")
            state["escalation"].setdefault("status", "recommended")
            state["escalation"].setdefault("notes", "Meta-agent recommends human review.")
            if self._reviews is not None:
                try:
                    ticket = await self._reviews.ensure_ticket(
                        task_state=state,
                        resolution_summary=resolution.summary,
                        sources=sources,
                    )
                    state["escalation"]["ticket_id"] = str(ticket.ticket_id)
                except Exception as exc:  # pragma: no cover - escalation failures should not break run
                    logger.warning("review_ticket_creation_failed", error=str(exc))

        await self._record_event(tracker, "meta_resolution_completed", payload=payload)
        await self._update_state(tracker, state)
        await self._capture_snapshot(
            tracker,
            ContextStage.CONSOLIDATION,
            state,
            extra={"meta": payload},
        )
        return resolution

    async def _assemble_dossier(
        self,
        state: dict[str, Any],
        tracker: _RunTracker | None,
        *,
        meta_resolution: MetaResolution | None,
    ) -> None:
        self._enforce_run_time_budget(tracker)
        try:
            dossier = build_decision_dossier(state, meta_resolution=meta_resolution)
        except Exception as exc:  # pragma: no cover - dossier generation should not fail the run
            logger.exception("dossier_generation_failed", error=str(exc))
            state.setdefault("dossier", {})
            state["dossier"].update({"status": "failed", "error": str(exc)})
            await self._record_event(tracker, "dossier_failed", payload={"error": str(exc)})
            await self._update_state(tracker, state)
            return

        state["dossier"] = {
            "status": "available",
            "json": dossier.as_dict(),
            "markdown": dossier.as_markdown(),
        }
        await self._record_event(
            tracker,
            "dossier_created",
            payload={"summary": dossier.summary, "created_at": dossier.created_at.isoformat()},
        )
        await self._update_state(tracker, state)
