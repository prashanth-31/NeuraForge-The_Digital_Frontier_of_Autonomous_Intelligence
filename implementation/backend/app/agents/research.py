from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..agents.base import AgentContext
from ..core.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class ResearchAgent:
    name: str = "research_agent"
    system_prompt: str = (
        "You are NeuraForge's Research Agent. Derive concise, well-sourced insights from the task "
        "description and prior agent outputs. Highlight 2-3 key findings and note missing context."
    )

    async def handle(self, task: dict[str, Any], *, context: AgentContext) -> dict[str, Any]:
        logger.info("research_agent_task", task=task)
        context_section = None
        if context.context is not None:
            bundle = await context.context.build(task=task, agent=self.name)
            context_section = bundle.as_prompt_section()
        prompt = self._build_prompt(task, context_section=context_section)
        summary = await context.llm.generate(prompt=prompt, system_prompt=self.system_prompt, temperature=0.1)

        await context.memory.store_working_memory(task.get("id", "unknown"), summary)

        task.setdefault("outputs", []).append(
            {
                "agent": self.name,
                "type": "research_summary",
                "content": summary,
                "confidence": 0.72,
            }
        )
        return task

    def _build_prompt(self, task: dict[str, Any], context_section: str | None = None) -> str:
        metadata = json.dumps(task.get("metadata", {}), indent=2, sort_keys=True)
        history = _format_history(task.get("outputs", []))
        retrieved = context_section or "(no retrieved context)"
        return (
            f"Primary question:\n{task.get('prompt', 'No prompt provided.')}\n\n"
            f"Known metadata:\n{metadata}\n\n"
            f"Prior agent outputs (if any):\n{history}\n\n"
            f"Retrieved context:\n{retrieved}\n\n"
            "Summarize credible research-backed insights, cite sources inline, and call out missing data."
        )


def _format_history(outputs: list[dict[str, Any]] | None) -> str:
    if not outputs:
        return "(none)"
    lines = []
    for item in outputs:
        agent = item.get("agent", "unknown")
        snippet = item.get("content") or item.get("summary") or "(no content)"
        lines.append(f"- {agent}: {snippet}")
    return "\n".join(lines)
