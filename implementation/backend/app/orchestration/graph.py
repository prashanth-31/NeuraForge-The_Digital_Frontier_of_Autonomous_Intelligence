from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

try:
    from langgraph.graph import StateGraph
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    StateGraph = None  # type: ignore[misc,assignment]

from ..agents.base import AgentContext, BaseAgent
from ..agents.contracts import validate_agent_request, validate_agent_response
from ..core.logging import get_logger
from ..core.metrics import (
    increment_agent_event,
    mark_orchestrator_run_completed,
    mark_orchestrator_run_started,
    observe_agent_latency,
    observe_negotiation_metrics,
    record_agent_tool_invocation,
    record_agent_tool_policy,
    record_plan_metrics,
    record_sla_event,
)
from ..schemas.agents import AgentInput, AgentOutput
from .context import ContextAssemblyContract, ContextSnapshot, ContextSnapshotStore, ContextStage
from .dossier import build_decision_dossier
from .guardrails import GuardrailDecisionType, GuardrailManager
from .llm_planner import LLMOrchestrationPlanner, PlannerError, PlannerExecutionPlan, PlannedAgentStep
from .lifecycle import LifecycleEvent, LifecycleStatus, TaskLifecycleStore
from .meta import MetaAgent, MetaResolution
from .negotiation import NegotiationEngine
from .planner import TaskPlanner
from .review import ReviewManager
from .scheduler import TaskScheduler
from .state import OrchestratorEvent, OrchestratorRun, OrchestratorStatus
from .store import OrchestratorStateStore

logger = get_logger(name=__name__)


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

    def __post_init__(self) -> None:
        self.started_at = time.perf_counter()


OPTIONAL_TOOL_AGENTS: set[str] = {"general_agent"}


class ToolFirstPolicyViolation(RuntimeError):
    """Raised when an agent completes without a successful tool invocation."""


@dataclass
class _ToolSession:
    agent_name: str
    tool_service: Any
    planned_tools: tuple[str, ...] = ()
    fallback_tools: tuple[str, ...] = ()
    planner_reason: str | None = None
    invocations: list[tuple[str, Any]] | None = None
    failures: list[tuple[str, str]] | None = None

    def __post_init__(self) -> None:
        self.invocations = [] if self.invocations is None else self.invocations
        self.failures = [] if self.failures is None else self.failures
        self.attempts = 0

    @property
    def proxy(self) -> Any:
        return _ToolProxy(self)

    def record_success(self, tool: str, result: Any) -> None:
        self.attempts += 1
        self.invocations.append((tool, result))
        record_agent_tool_invocation(agent=self.agent_name, tool=tool, outcome="success")

    def record_failure(self, tool: str, error: Exception) -> None:
        self.attempts += 1
        self.failures.append((tool, str(error)))
        record_agent_tool_invocation(agent=self.agent_name, tool=tool, outcome="failure")

    @property
    def successful(self) -> bool:
        return bool(self.invocations)

    def adherence_summary(self) -> dict[str, Any]:
        planned_set = set(self.planned_tools)
        fallback_set = set(self.fallback_tools)
        success_tools = [tool for tool, _ in self.invocations]
        failure_tools = [tool for tool, _ in self.failures]
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


class _ToolProxy:
    def __init__(self, session: _ToolSession) -> None:
        self._session = session

    async def invoke(self, tool: str, payload: dict[str, Any]) -> Any:
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
        self._active_plan_steps: dict[str, PlannedAgentStep] | None = None

    def _build_graph(self, agents: list[BaseAgent]) -> Any:
        if StateGraph is None:
            logger.warning("langgraph_not_installed", agents=len(agents))
            return None
        graph = StateGraph(dict)  # type: ignore[arg-type]
        for agent in agents:
            async def _node(state: dict[str, Any], *, agent=agent) -> dict[str, Any]:
                if self._current_context is None:
                    raise RuntimeError("Agent context not set for orchestration")
                plan_step = None
                if self._active_plan_steps is not None:
                    plan_step = self._active_plan_steps.get(agent.name)
                tool_session = self._start_tool_session(
                    agent=agent,
                    base_context=self._current_context,
                    plan_step=plan_step,
                )
                agent_context = self._clone_context(self._current_context, tool_session, plan_step)
                request = await self._prepare_agent_input(state, agent=agent, context=agent_context)
                output = await agent.handle(request, context=agent_context)
                updated_state = self._record_output(state, agent=agent, result=output)
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
    ) -> _ToolSession | None:
        if base_context.tools is None:
            return None
        planned = tuple(plan_step.tools) if plan_step and plan_step.tools else ()
        fallback = tuple(plan_step.fallback_tools) if plan_step and plan_step.fallback_tools else ()
        reason = plan_step.reason if plan_step and plan_step.reason else None
        return _ToolSession(
            agent_name=agent.name,
            tool_service=base_context.tools,
            planned_tools=planned,
            fallback_tools=fallback,
            planner_reason=reason,
        )

    @staticmethod
    def _clone_context(
        base_context: AgentContext,
        tool_session: _ToolSession | None,
        plan_step: PlannedAgentStep | None,
    ) -> AgentContext:
        planned_list = None
        fallback_list = None
        if plan_step is not None:
            if plan_step.tools:
                planned_list = list(plan_step.tools)
            if plan_step.fallback_tools:
                fallback_list = list(plan_step.fallback_tools)
        elif base_context.planned_tools:
            planned_list = list(base_context.planned_tools)
            fallback_list = list(base_context.fallback_tools or []) if base_context.fallback_tools else None
        tools_proxy = tool_session.proxy if tool_session is not None else base_context.tools
        return AgentContext(
            memory=base_context.memory,
            llm=base_context.llm,
            context=base_context.context,
            tools=tools_proxy,
            scorer=base_context.scorer,
            planned_tools=planned_list,
            fallback_tools=fallback_list,
            planner_reason=(plan_step.reason if plan_step and plan_step.reason else base_context.planner_reason),
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
        summary = tool_session.adherence_summary()

        outputs = state.get("outputs", [])
        latest_output = outputs[-1] if outputs else None
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

        if not tool_session.successful:
            allow_optional = (
                agent.name in OPTIONAL_TOOL_AGENTS
                and not summary["planned_tools"]
                and not summary["fallback_tools"]
                and tool_session.attempts == 0
            )
            if allow_optional:
                record_agent_tool_policy(agent=agent.name, outcome="optional_skipped")
                return

            if (
                agent.name in OPTIONAL_TOOL_AGENTS
                and tool_session.attempts == 0
                and self._should_allow_optional_tool_skip(summary=summary, state=state)
            ):
                record_agent_tool_policy(agent=agent.name, outcome="optional_skipped_planned")
                return
            outcome = "violation_failed" if tool_session.attempts else "violation_missing"
            record_agent_tool_policy(agent=agent.name, outcome=outcome)
            payload = {
                "attempts": tool_session.attempts,
                "errors": tool_session.failures,
                "reason": "no_successful_invocation" if tool_session.attempts else "no_invocation",
                "planner": summary,
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
            raise ToolFirstPolicyViolation(
                f"Agent '{agent.name}' must successfully invoke at least one tool before completing."
            )

        planned_expectations = bool(summary["planned_tools"] or summary["fallback_tools"])
        if planned_expectations and not summary["allowed"]:
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
            planned_list = summary["planned_tools"] or summary["fallback_tools"]
            raise ToolFirstPolicyViolation(
                f"Agent '{agent.name}' must successfully invoke planned tools or fallbacks {planned_list} before completing."
            )

        if planned_expectations:
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
    def _should_allow_optional_tool_skip(*, summary: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
        prompt = str(state.get("prompt") or "").strip()
        if prompt and len(prompt) <= 12 and "?" not in prompt:
            return True
        reason = str(summary.get("planner_reason") or "").lower()
        if any(keyword in reason for keyword in ("greet", "hello", "welcome", "triage", "intro")):
            return True
        metadata = state.get("metadata")
        if isinstance(metadata, Mapping):
            intent = metadata.get("intent")
            if isinstance(intent, str) and intent.lower() in {"greeting", "smalltalk"}:
                return True
        return False

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
        self._active_plan_steps = None
        run_tracker = _RunTracker(run=None, entry_point=entry_point)
        mark_orchestrator_run_started(entry_point=entry_point, task_id=task_id)
        if not self.agents:
            mark_orchestrator_run_completed(entry_point=entry_point, task_id=task_id, status="skipped", latency=0.0)
            return {"status": "noop", "outputs": [], **task}

        try:
            agent_sequence, routing_context, skipped_agents = await self._determine_agent_sequence(task)
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
            self._active_plan_steps = None

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
        agents = list(agent_sequence or self.agents)
        state = task
        state.setdefault("outputs", [])
        state.setdefault("shared_context", {"provenance": []})
        task_id = str(state.get("id") or state.get("task_id") or "")
        failure: Exception | None = None
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

        for agent in agents:
            await self._record_event(
                run_tracker,
                "agent_started",
                agent=agent.name,
                payload={"task_id": task.get("id")},
            )
            await self._notify(
                progress_cb,
                self._ensure_run_id(
                    run_tracker,
                    {
                        "event": "agent_started",
                        "agent": agent.name,
                        "task_id": task.get("id"),
                        "step_id": f"agent::{agent.name}",
                    },
                ),
            )
            increment_agent_event(agent=agent.name, event="started")
            logger.info("agent_started", agent=agent.name, task=task.get("id"))
            start_time = time.perf_counter()
            await self._record_lifecycle(
                run_tracker,
                task_id=task_id,
                step_id=f"agent::{agent.name}",
                event_type="agent_started",
                status=LifecycleStatus.IN_PROGRESS,
                agent=agent.name,
            )
            await self._capture_snapshot(
                run_tracker,
                ContextStage.EXECUTION,
                state,
                agent=agent.name,
                extra={"event": "agent_started"},
            )
            try:
                plan_step = None
                if self._active_plan_steps is not None:
                    plan_step = self._active_plan_steps.get(agent.name)
                tool_session = self._start_tool_session(
                    agent=agent,
                    base_context=context,
                    plan_step=plan_step,
                )
                agent_context = self._clone_context(context, tool_session, plan_step)
                agent_input = await self._prepare_agent_input(state, agent=agent, context=agent_context)
                result = await agent.handle(agent_input, context=agent_context)
                self._record_output(state, agent=agent, result=result)
                await self._enforce_tool_first_policy(
                    tool_session,
                    agent=agent,
                    state=state,
                    run_tracker=run_tracker,
                    progress_cb=progress_cb,
                )
            except Exception as exc:
                duration = time.perf_counter() - start_time
                observe_agent_latency(agent=agent.name, latency=duration)
                increment_agent_event(agent=agent.name, event="failed")
                failure = exc
                await self._record_lifecycle(
                    run_tracker,
                    task_id=task_id,
                    step_id=f"agent::{agent.name}",
                    event_type="agent_failed",
                    status=LifecycleStatus.FAILED,
                    agent=agent.name,
                    latency_ms=duration * 1000,
                )
                await self._notify(
                    progress_cb,
                    self._ensure_run_id(
                        run_tracker,
                        {
                            "event": "agent_failed",
                            "agent": agent.name,
                            "task_id": task.get("id"),
                            "error": str(exc),
                            "latency": duration,
                            "step_id": f"agent::{agent.name}",
                        },
                    ),
                )
                logger.exception(
                    "agent_failed",
                    agent=agent.name,
                    task=task.get("id"),
                    duration=duration,
                )
                state["status"] = "failed"
                state["error"] = str(exc)
                await self._record_event(
                    run_tracker,
                    "agent_failed",
                    agent=agent.name,
                    payload={"error": str(exc), "latency": duration},
                )
                await self._capture_snapshot(
                    run_tracker,
                    ContextStage.EXECUTION,
                    state,
                    agent=agent.name,
                    extra={"event": "agent_failed", "error": str(exc)},
                )
                await self._finalize_run(run_tracker, state, status=OrchestratorStatus.FAILED, error=str(exc))
                raise
            duration = time.perf_counter() - start_time
            observe_agent_latency(agent=agent.name, latency=duration)
            increment_agent_event(agent=agent.name, event="completed")
            logger.info(
                "agent_completed",
                agent=agent.name,
                task=task.get("id"),
                duration=duration,
                outputs=len(state.get("outputs", [])),
            )
            latest_output = state.get("outputs", [])[-1] if state.get("outputs") else None
            await self._notify(
                progress_cb,
                self._ensure_run_id(
                    run_tracker,
                    {
                        "event": "agent_completed",
                        "agent": agent.name,
                        "task_id": task.get("id"),
                        "latency": duration,
                        "output": latest_output,
                        "step_id": f"agent::{agent.name}",
                    },
                ),
            )
            await self._record_event(
                run_tracker,
                "agent_completed",
                agent=agent.name,
                payload={"latency": duration, "output": latest_output},
            )
            await self._record_lifecycle(
                run_tracker,
                task_id=task_id,
                step_id=f"agent::{agent.name}",
                event_type="agent_completed",
                status=LifecycleStatus.COMPLETED,
                agent=agent.name,
                latency_ms=duration * 1000,
            )
            await self._capture_snapshot(
                run_tracker,
                ContextStage.EXECUTION,
                state,
                agent=agent.name,
                extra={"event": "agent_completed", "output": latest_output},
            )
            await self._update_state(run_tracker, state)

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

    async def _determine_agent_sequence(self, task: dict[str, Any]) -> tuple[list[BaseAgent], dict[str, Any], list[BaseAgent]]:
        roster = list(self.agents)
        self._active_plan_steps = None

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
            planner_plan = await self._llm_planner.plan(
                task=task,
                prior_outputs=self._coerce_prior_outputs(task),
                agents=roster,
                tool_aliases=self._collect_tool_aliases(task),
            )
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

        selection = self._resolve_planner_agents(planner_plan.steps, roster)
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
            raise PlannerError(invalid_reason)

        metadata = self._build_planner_routing_payload(
            planner_plan,
            roster,
            selection,
            status="planned",
        )
        skipped_agents = [agent for agent in roster if agent not in selection]
        self._active_plan_steps = {step.agent: step for step in planner_plan.steps}
        self._annotate_task_with_plan(task, planner_plan, selection, status="planned")
        record_plan_metrics(strategy="llm_orchestration", status="planned", steps=len(planner_plan.steps))
        return selection, metadata, skipped_agents

    def _coerce_prior_outputs(self, task: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        outputs = task.get("outputs")
        if not isinstance(outputs, Sequence):
            return []
        return [item for item in outputs if isinstance(item, Mapping)]

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
    ) -> list[BaseAgent]:
        index = {agent.name: agent for agent in roster}
        resolved: list[BaseAgent] = []
        seen: set[str] = set()
        for step in steps:
            candidate = index.get(step.agent)
            if candidate is None or candidate.name in seen:
                continue
            resolved.append(candidate)
            seen.add(candidate.name)
        return resolved

    def _build_planner_routing_payload(
        self,
        plan: PlannerExecutionPlan,
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
        plan: PlannerExecutionPlan,
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

    def _record_output(self, state: dict[str, Any], *, agent: BaseAgent, result: AgentOutput | dict[str, Any]) -> dict[str, Any]:
        if isinstance(result, AgentOutput):
            validated = validate_agent_response(agent.capability, result.model_dump())
        else:
            validated = validate_agent_response(agent.capability, result)
        output_dict = validated.model_dump()
        output_dict.setdefault("content", validated.summary)
        output_dict.setdefault("agent", agent.name)
        state.setdefault("outputs", []).append(output_dict)
        shared = state.setdefault("shared_context", {"provenance": []})
        provenance: list[dict[str, Any]] = shared.setdefault("provenance", [])
        provenance.append(
            {
                "agent": agent.name,
                "capability": agent.capability.value,
                "summary": output_dict.get("summary"),
                "confidence": output_dict.get("confidence"),
                "metadata": output_dict.get("metadata", {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(provenance) > 20:
            del provenance[:-20]
        shared["last_writer"] = agent.name
        return state

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
