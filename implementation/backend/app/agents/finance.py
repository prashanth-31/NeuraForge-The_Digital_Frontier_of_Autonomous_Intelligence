from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..agents.base import AgentContext
from ..core.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class FinanceAgent:
    name: str = "finance_agent"
    system_prompt: str = (
        "You are NeuraForge's Finance Agent. Provide high-level forecasts, key risks, and actionable"
        " next steps using available context. Respond with structured bullet points."
    )

    async def handle(self, task: dict[str, Any], *, context: AgentContext) -> dict[str, Any]:
        logger.info("finance_agent_task", task=task)
        prompt = self._build_prompt(task)
        forecast = await context.llm.generate(prompt=prompt, system_prompt=self.system_prompt, temperature=0.2)

        task.setdefault("outputs", []).append(
            {
                "agent": self.name,
                "type": "financial_forecast",
                "content": forecast,
                "confidence": 0.68,
            }
        )
        return task

    def _build_prompt(self, task: dict[str, Any]) -> str:
        metadata = json.dumps(task.get("metadata", {}), indent=2, sort_keys=True)
        prior = _collect_prior_outputs(task.get("outputs", []))
        return (
            f"Business question:\n{task.get('prompt', '')}\n\n"
            f"Financial metadata:\n{metadata}\n\n"
            f"Earlier agent signals:\n{prior}\n\n"
            "Estimate revenue or cost implications, surface 2-3 headline metrics, and outline immediate actions."
        )


def _collect_prior_outputs(outputs: list[dict[str, Any]] | None) -> str:
    if not outputs:
        return "(none)"
    return "\n".join(f"- {item.get('agent', 'unknown')}: {item.get('content', '')}" for item in outputs)
