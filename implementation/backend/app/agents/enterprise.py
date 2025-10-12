from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..agents.base import AgentContext
from ..core.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class EnterpriseAgent:
    name: str = "enterprise_agent"
    system_prompt: str = (
        "You are NeuraForge's Enterprise Agent. Translate insights into market positioning, "
        "operational tactics, and executive-ready recommendations."
    )

    async def handle(self, task: dict[str, Any], *, context: AgentContext) -> dict[str, Any]:
        logger.info("enterprise_agent_task", task=task)
        prompt = self._build_prompt(task)
        strategy = await context.llm.generate(prompt=prompt, system_prompt=self.system_prompt, temperature=0.15)

        task.setdefault("outputs", []).append(
            {
                "agent": self.name,
                "type": "enterprise_strategy",
                "content": strategy,
                "confidence": 0.74,
            }
        )
        return task

    def _build_prompt(self, task: dict[str, Any]) -> str:
        metadata = json.dumps(task.get("metadata", {}), indent=2, sort_keys=True)
        prior = _summarize_prior_outputs(task.get("outputs", []))
        return (
            f"Executive task:\n{task.get('prompt', '')}\n\n"
            f"Business metadata:\n{metadata}\n\n"
            f"Cross-agent insights:\n{prior}\n\n"
            "Deliver a numbered action plan (max 5 steps) with expected impact and confidence."
        )


def _summarize_prior_outputs(outputs: list[dict[str, Any]] | None) -> str:
    if not outputs:
        return "(none)"
    return "\n".join(f"- {item.get('agent', 'unknown')}: {item.get('content', '')}" for item in outputs)
