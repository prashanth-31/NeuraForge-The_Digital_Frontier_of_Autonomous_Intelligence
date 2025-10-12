from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..services.memory import HybridMemoryService


class Tool(Protocol):
    name: str

    async def run(self, **kwargs: Any) -> Any:
        ...


@dataclass
class AgentContext:
    memory: HybridMemoryService


class BaseAgent(Protocol):
    name: str

    async def handle(self, task: dict[str, Any], *, context: AgentContext) -> dict[str, Any]:
        ...
