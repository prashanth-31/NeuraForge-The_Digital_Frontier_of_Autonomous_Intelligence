from __future__ import annotations

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
from ..tools.exceptions import ToolError, ToolPolicyViolationError
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
                token = item.strip()
                if token:
                    names.add(token)
        return names

    @classmethod
    def _build_agent_tool_catalog(cls, roster: Sequence[BaseAgent]) -> dict[str, set[str]]:
        catalog: dict[str, set[str]] = {}
        for agent in roster:
            tool_names: set[str] = set()
            for attribute in ("tool_preference", "tool_candidates", "fallback_tools", "tools", "default_tools"):
                tool_names |= cls._collect_tool_names(getattr(agent, attribute, None))
            retry_config = getattr(agent, "tool_retry_config", None)
            if isinstance(retry_config, Mapping):
                tool_names |= cls._collect_tool_names(retry_config)
            catalog[agent.name] = tool_names
        return catalog

    def _validate_plan_tools(self, steps: Sequence[PlannedAgentStep]) -> None:
        if not steps:
            return
        invalid_entries: list[dict[str, Any]] = []
        for step in steps:
            catalog = self._agent_tool_catalog.get(step.agent)
            if catalog is None:
                invalid_entries.append({"agent": step.agent, "missing": sorted(set(step.tools))})
                continue
            missing = [tool for tool in step.tools if tool not in catalog]
            fallback_missing = [tool for tool in step.fallback_tools if tool not in catalog]
            if missing or fallback_missing:
                invalid_entries.append(
                    {
                        "agent": step.agent,
                        "missing": sorted(set(missing + fallback_missing)),
                        "available": sorted(catalog),
                    }
                )
        if invalid_entries:
            logger.warning("planner_invalid_tools", entries=invalid_entries)
            raise PlannerError("Planner referenced tools that target agents do not expose")

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
        summary = tool_session.adherence_summary()
        enforcement_required = TOOL_ENFORCEMENT_POLICY.get(agent.name, False)
        policy_mode = "enforced" if enforcement_required else "optional"

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

            for step_index, plan_step, planned_agent in execution_plan:
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
                    agent_context = self._clone_context(
                        context,
                        tool_session,
                        step_context,
                        agent_name=executed_agent.name,
                    )
                    agent_input = await self._prepare_agent_input(state, agent=executed_agent, context=agent_context)
                    result = await executed_agent.handle(agent_input, context=agent_context)
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
