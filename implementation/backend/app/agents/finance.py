from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.logging import get_logger
from ..services.memory import HybridMemoryService

logger = get_logger(name=__name__)


@dataclass
class FinanceAgent:
    memory: HybridMemoryService

    name: str = "finance_agent"

    async def handle(self, task: dict[str, Any]) -> dict[str, Any]:
        logger.info("finance_agent_task", task=task)
        return {
            "agent": self.name,
            "forecast": "Finance forecast placeholder",
            "confidence": 0.65,
        }
