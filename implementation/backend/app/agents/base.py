from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ..services.llm import LLMService
from ..services.memory import HybridMemoryService
from ..services.scoring import ConfidenceScorer
from ..services.tools import ToolService
from ..schemas.agents import AgentCapability, AgentInput, AgentOutput

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..services.retrieval import ContextAssembler


class Tool(Protocol):
    name: str

    async def run(self, **kwargs: Any) -> Any:
        ...


@dataclass
class AgentContext:
    memory: HybridMemoryService
    llm: LLMService
    context: "ContextAssembler | None" = None
    tools: ToolService | None = None
    scorer: ConfidenceScorer | None = None
    planned_tools: list[str] | None = None
    fallback_tools: list[str] | None = None
    planner_reason: str | None = None


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
