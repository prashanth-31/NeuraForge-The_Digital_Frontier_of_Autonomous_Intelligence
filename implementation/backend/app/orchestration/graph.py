from __future__ import annotations

from typing import Any, Protocol

try:
    from langgraph.graph import StateGraph
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    StateGraph = None  # type: ignore[misc,assignment]

from ..core.logging import get_logger

logger = get_logger(name=__name__)


class AgentNode(Protocol):
    name: str

    async def handle(self, task: dict[str, Any]) -> dict[str, Any]:
        ...


class Orchestrator:
    def __init__(self, *, agents: list[AgentNode]):
        self.agents = agents
        self._graph = self._build_graph(agents)

    def _build_graph(self, agents: list[AgentNode]) -> Any:
        if StateGraph is None:
            logger.warning("langgraph_not_installed", agents=len(agents))
            return None
        graph = StateGraph(dict)
        for agent in agents:
            graph.add_node(agent.name, agent.handle)
        graph.set_entry_point(agents[0].name if agents else "noop")
        return graph

    async def route_task(self, task: dict[str, Any]) -> dict[str, Any]:
        logger.info("task_routed", task=task.get("id"))
        if self._graph is None:
            return {"status": "noop", "task": task}
        executor = self._graph.compile()
        return await executor.ainvoke(task)
