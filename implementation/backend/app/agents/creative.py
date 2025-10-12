from __future__ import annotations

from dataclasses import dataclass

from ..core.logging import get_logger
from ..services.memory import HybridMemoryService

logger = get_logger(name=__name__)


@dataclass
class CreativeAgent:
    memory: HybridMemoryService

    name: str = "creative_agent"

    async def handle(self, task: dict[str, str]) -> dict[str, str | float]:
        logger.info("creative_agent_task", task=task)
        return {
            "agent": self.name,
            "creative_output": f"Stylized response for: {task.get('prompt', '')}",
            "confidence": 0.72,
        }
