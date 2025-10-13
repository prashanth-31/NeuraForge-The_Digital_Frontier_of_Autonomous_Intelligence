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


class BaseAgent(Protocol):
    name: str
    capability: AgentCapability

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        ...
