from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

try:
    from langgraph.graph import StateGraph
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    StateGraph = None  # type: ignore[misc,assignment]

from ..core.logging import get_logger
from ..core.metrics import increment_agent_event, observe_agent_latency
from ..agents.base import AgentContext, BaseAgent
from ..agents.contracts import validate_agent_request, validate_agent_response
from ..schemas.agents import AgentInput, AgentOutput
from .negotiation import NegotiationEngine
from .planner import TaskPlanner
from .scheduler import TaskScheduler
from .state import OrchestratorEvent, OrchestratorRun, OrchestratorStatus
from .store import OrchestratorStateStore

logger = get_logger(name=__name__)


@dataclass
class _RunTracker:
    run: OrchestratorRun
    sequence: int = 0


class Orchestrator:
    def __init__(
        self,
        *,
        agents: list[BaseAgent],
        state_store: OrchestratorStateStore | None = None,
        negotiation_engine: NegotiationEngine | None = None,
        planner: TaskPlanner | None = None,
        scheduler: TaskScheduler | None = None,
    ):
        self.agents = agents
        self._state_store = state_store
        self._negotiation = negotiation_engine
        self._planner = planner
        self._scheduler = scheduler
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
        if not self.agents:
            return {"status": "noop", "outputs": [], **task}
        task_id = str(task.get("id") or task.get("task_id") or uuid.uuid4())
        task["id"] = task_id
        task.setdefault("outputs", [])
        run_tracker: _RunTracker | None = None
        if self._state_store is not None:
            run = await self._state_store.start_run(task_id, state=task)
            task["run_id"] = str(run.run_id)
            run_tracker = _RunTracker(run=run)
            await self._update_state(run_tracker, task, status=OrchestratorStatus.RUNNING)
            await self._record_event(run_tracker, "run_started", payload={"task_id": task_id})

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
        finally:
            self._current_context = None
        result["status"] = "completed"
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
        failure: Exception | None = None
        for agent in self.agents:
            await self._record_event(
                run_tracker,
                "agent_started",
                agent=agent.name,
                payload={"task_id": task.get("id")},
            )
            await self._notify(progress_cb, {"agent": agent.name, "task_id": task.get("id"), "event": "started"})
            increment_agent_event(agent=agent.name, event="started")
            logger.info("agent_started", agent=agent.name, task=task.get("id"))
            start_time = time.perf_counter()
            try:
                agent_input = await self._prepare_agent_input(state, agent=agent, context=context)
                result = await agent.handle(agent_input, context=context)
                self._record_output(state, agent=agent, result=result)
            except Exception as exc:
                duration = time.perf_counter() - start_time
                observe_agent_latency(agent=agent.name, latency=duration)
                increment_agent_event(agent=agent.name, event="failed")
                failure = exc
                await self._notify(
                    progress_cb,
                    {
                        "agent": agent.name,
                        "task_id": task.get("id"),
                        "event": "failed",
                        "error": str(exc),
                        "latency": duration,
                    },
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
                {
                    "agent": agent.name,
                    "task_id": task.get("id"),
                    "event": "completed",
                    "latency": duration,
                    "output": latest_output,
                },
            )
            await self._record_event(
                run_tracker,
                "agent_completed",
                agent=agent.name,
                payload={"latency": duration, "output": latest_output},
            )
            await self._update_state(run_tracker, state)

        if state.get("status") != "failed" and self._negotiation is not None:
            await self._apply_negotiation(state, run_tracker)
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
        if context.context is not None:
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

    async def _record_event(
        self,
        tracker: _RunTracker | None,
        event_type: str,
        *,
        agent: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if tracker is None or self._state_store is None:
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

    async def _update_state(
        self,
        tracker: _RunTracker | None,
        state: dict[str, Any],
        *,
        status: OrchestratorStatus | None = None,
    ) -> None:
        if tracker is None or self._state_store is None:
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
        if tracker is None or self._state_store is None:
            return
        event_type = "run_completed" if status is OrchestratorStatus.COMPLETED else f"run_{status.value}"
        await self._record_event(tracker, event_type, payload={"error": error} if error else None)
        await self._state_store.finalize_run(tracker.run.run_id, state=state, status=status, error=error)

    async def _apply_negotiation(self, state: dict[str, Any], tracker: _RunTracker | None) -> None:
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
        await self._update_state(tracker, state)
        await self._generate_plan(state, tracker)

    async def _generate_plan(self, state: dict[str, Any], tracker: _RunTracker | None) -> None:
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

        serialized = {
            "task_id": plan.task_id,
            "summary": plan.summary,
            "steps": [asdict(step) for step in plan.steps],
            "metadata": plan.metadata,
            "status": "scheduled" if any(step.eta_iso for step in plan.steps) else "planned",
        }
        state["plan"] = serialized
        await self._record_event(tracker, "plan_generated", payload=serialized)
        await self._update_state(tracker, state)
