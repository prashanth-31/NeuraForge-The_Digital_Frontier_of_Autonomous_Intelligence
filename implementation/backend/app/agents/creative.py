from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agents.base import AgentContext
from ..core.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class CreativeAgent:
    name: str = "creative_agent"
    system_prompt: str = (
        "You are NeuraForge's Creative Agent. Craft compelling narratives, taglines, or storytelling"
        " snippets tailored to the task. Blend clarity with creativity."
    )

    async def handle(self, task: dict[str, Any], *, context: AgentContext) -> dict[str, Any]:
        logger.info("creative_agent_task", task=task)
        prompt = self._build_prompt(task)
        creative_output = await context.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.6,
        )

        task.setdefault("outputs", []).append(
            {
                "agent": self.name,
                "type": "creative_direction",
                "content": creative_output,
                "confidence": 0.7,
            }
        )
        return task

    def _build_prompt(self, task: dict[str, Any]) -> str:
        audience = task.get("metadata", {}).get("audience", "a general audience")
        prior = _collect_task_context(task.get("outputs", []))
        return (
            f"Core request:\n{task.get('prompt', '')}\n\n"
            f"Primary audience: {audience}\n\n"
            f"Previous agent notes:\n{prior}\n\n"
            "Deliver a vibrant, memorable piece (<=150 words) that aligns with the strategy."
        )


def _collect_task_context(outputs: list[dict[str, Any]] | None) -> str:
    if not outputs:
        return "(none)"
    return "\n".join(f"- {item.get('agent', 'unknown')}: {item.get('content', '')}" for item in outputs)
