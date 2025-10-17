from __future__ import annotations

import time
import uuid
from datetime import datetime
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

try:
    from langgraph.graph import StateGraph
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    StateGraph = None  # type: ignore[misc,assignment]

from ..core.logging import get_logger
from ..core.metrics import (
    increment_agent_event,
    mark_orchestrator_run_completed,
    mark_orchestrator_run_started,
    observe_agent_latency,
    observe_negotiation_metrics,
    record_sla_event,
)
from ..agents.base import AgentContext, BaseAgent
from ..agents.contracts import validate_agent_request, validate_agent_response
from ..schemas.agents import AgentInput, AgentOutput
from .negotiation import NegotiationEngine
from .context import ContextAssemblyContract, ContextSnapshot, ContextSnapshotStore, ContextStage
from .guardrails import GuardrailDecisionType, GuardrailManager
from .lifecycle import LifecycleEvent, LifecycleStatus, TaskLifecycleStore
from .planner import TaskPlanner
from .scheduler import TaskScheduler
from .state import OrchestratorEvent, OrchestratorRun, OrchestratorStatus
from .store import OrchestratorStateStore
from .meta import MetaAgent, MetaResolution
from .dossier import DecisionDossier, build_decision_dossier
from .review import ReviewManager

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

    def _build_graph(self, agents: list[BaseAgent]) -> Any:
        if StateGraph is None:
            logger.warning("langgraph_not_installed", agents=len(agents))
            return None
        graph = StateGraph(dict)  # type: ignore[arg-type]
        for agent in agents:
            async def _node(state: dict[str, Any], *, agent=agent) -> dict[str, Any]:
                if self._current_context is None:
                    raise RuntimeError("Agent context not set for orchestration")
                request = await self._prepare_agent_input(state, agent=agent, context=self._current_context)
                output = await agent.handle(request, context=self._current_context)
                return self._record_output(state, agent=agent, result=output)

            graph.add_node(agent.name, _node)  # type: ignore[arg-type]
        graph.set_entry_point(agents[0].name if agents else "noop")
        for index in range(len(agents) - 1):
            graph.add_edge(agents[index].name, agents[index + 1].name)
        if agents:
            graph.set_finish_point(agents[-1].name)
        return graph

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
        run_tracker = _RunTracker(run=None, entry_point=entry_point)
        mark_orchestrator_run_started(entry_point=entry_point, task_id=task_id)
        if not self.agents:
            mark_orchestrator_run_completed(entry_point=entry_point, task_id=task_id, status="skipped", latency=0.0)
            return {"status": "noop", "outputs": [], **task}
        if self._state_store is not None:
            run = await self._state_store.start_run(task_id, state=task)
            task["run_id"] = str(run.run_id)
            run_tracker.run = run
            await self._update_state(run_tracker, task, status=OrchestratorStatus.RUNNING)
            await self._record_event(run_tracker, "run_started", payload={"task_id": task_id})
            await self._capture_snapshot(run_tracker, ContextStage.INTAKE, task)

        if self._graph is None or progress_cb is not None or self._state_store is not None:
            result = await self._run_sequential(
                task,
                context=context,
                progress_cb=progress_cb,
                run_tracker=run_tracker,
            )
            return result

        self._current_context = context
        try:
            executor = self._graph.compile()
            result = await executor.ainvoke(task)
            result.setdefault("outputs", [])
            if result.get("status") != "failed":
                if self._negotiation is not None and not result.get("negotiation"):
                        await self._apply_negotiation(result, run_tracker, progress_cb=progress_cb)
                elif "plan" not in result:
                        await self._generate_plan(result, run_tracker, progress_cb=progress_cb)
                meta_resolution = await self._synthesize_meta_resolution(result, run_tracker)
                await self._assemble_dossier(result, run_tracker, meta_resolution=meta_resolution)
        except Exception:
            mark_orchestrator_run_completed(
                entry_point=entry_point,
                task_id=task_id,
                status="failed",
                latency=time.perf_counter() - run_tracker.started_at,
            )
            raise
        finally:
            self._current_context = None
        result["status"] = "completed"
        mark_orchestrator_run_completed(
            entry_point=entry_point,
            task_id=task_id,
            status="completed",
            latency=time.perf_counter() - run_tracker.started_at,
        )
        return result

    async def _run_sequential(
        self,
        task: dict[str, Any],
        *,
        context: AgentContext,
        progress_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        run_tracker: _RunTracker | None = None,
    ) -> dict[str, Any]:
        state = task
        state.setdefault("outputs", [])
        task_id = str(state.get("id") or state.get("task_id") or "")
        failure: Exception | None = None
        for agent in self.agents:
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
                agent_input = await self._prepare_agent_input(state, agent=agent, context=context)
                result = await agent.handle(agent_input, context=context)
                self._record_output(state, agent=agent, result=result)
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
            if self._negotiation is not None:
                await self._apply_negotiation(state, run_tracker, progress_cb=progress_cb)
            else:
                await self._generate_plan(state, run_tracker, progress_cb=progress_cb)
            meta_resolution = await self._synthesize_meta_resolution(state, run_tracker)
            await self._assemble_dossier(state, run_tracker, meta_resolution=meta_resolution)
        state["status"] = "completed"
        await self._finalize_run(
            run_tracker,
            state,
            status=OrchestratorStatus.COMPLETED if failure is None else OrchestratorStatus.FAILED,
            error=str(failure) if failure is not None else None,
        )
        return state

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
            "metadata": state.get("metadata", {}),
            "context": context_text,
            "prior_exchanges": self._serialize_prior_exchanges(state.get("outputs", [])),
        }
        return validate_agent_request(agent.capability, payload)

    def _record_output(self, state: dict[str, Any], *, agent: BaseAgent, result: AgentOutput | dict[str, Any]) -> dict[str, Any]:
        if isinstance(result, AgentOutput):
            validated = validate_agent_response(agent.capability, result.model_dump())
        else:
            validated = validate_agent_response(agent.capability, result)
        output_dict = validated.model_dump()
        output_dict.setdefault("content", validated.summary)
        output_dict.setdefault("agent", agent.name)
        state.setdefault("outputs", []).append(output_dict)
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
            await self._record_event(tracker, "plan_failed", payload={"error": str(exc)})
            await self._update_state(tracker, state)
            return

        if plan is None:
            return

        if self._scheduler is not None:
            try:
                plan = await self._scheduler.schedule(plan)
            except Exception as exc:  # pragma: no cover - scheduler failure surfaces to state
                logger.exception("plan_scheduling_failed", error=str(exc))
                state["plan"] = {"status": "failed", "error": str(exc)}
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
