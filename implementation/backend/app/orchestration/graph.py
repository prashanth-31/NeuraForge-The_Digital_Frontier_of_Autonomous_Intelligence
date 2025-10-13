from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
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

logger = get_logger(name=__name__)


class Orchestrator:
    def __init__(self, *, agents: list[BaseAgent]):
        self.agents = agents
        self._graph = self._build_graph(agents)
        self._current_context: AgentContext | None = None

    def _build_graph(self, agents: list[BaseAgent]) -> Any:
        if StateGraph is None:
            logger.warning("langgraph_not_installed", agents=len(agents))
            return None
        graph = StateGraph(dict)
        for agent in agents:
            async def _node(state: dict[str, Any], *, agent=agent) -> dict[str, Any]:
                if self._current_context is None:
                    raise RuntimeError("Agent context not set for orchestration")
                request = await self._prepare_agent_input(state, agent=agent, context=self._current_context)
                output = await agent.handle(request, context=self._current_context)
                return self._record_output(state, agent=agent, result=output)

            graph.add_node(agent.name, _node)
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
        if self._graph is None or progress_cb is not None:
            return await self._run_sequential(task, context=context, progress_cb=progress_cb)

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
    ) -> dict[str, Any]:
        state = task
        state.setdefault("outputs", [])
        for agent in self.agents:
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
        state["status"] = "completed"
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
