from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.logging import get_logger
from ..services.memory import HybridMemoryService

logger = get_logger(name=__name__)


@dataclass
class ResearchAgent:
    memory: HybridMemoryService

    name: str = "research_agent"

    async def handle(self, task: dict[str, Any]) -> dict[str, Any]:
        logger.info("research_agent_task", task=task)
        await self.memory.store_working_memory(task.get("id", "unknown"), task.get("prompt", ""))
        return {
            "agent": self.name,
            "summary": "Research insights placeholder",
            "confidence": 0.7,
        }
