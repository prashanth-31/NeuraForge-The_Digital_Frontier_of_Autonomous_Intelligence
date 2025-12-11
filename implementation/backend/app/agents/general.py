from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agents.base import AgentContext
from ..core.logging import get_logger
from ..core.metrics import observe_confidence_component
from ..schemas.agents import AgentCapability, AgentExchange, AgentInput, AgentOutput

logger = get_logger(name=__name__)


@dataclass
class GeneralistAgent:
    name: str = "general_agent"
    capability: AgentCapability = AgentCapability.GENERAL
    system_prompt: str = (
        "You are NeuraForge's Generalist Agent. Provide a concise, confident response that triages the "
        "request, highlights immediate next steps, and captures any clarifying questions. Keep it under "
        "200 words while sounding professional and action-oriented."
    )
    description: str = "First-responder agent that greets users, answers simple prompts, and routes work to specialists."
    tool_preference: list[str] = field(default_factory=list)
    fallback_agent: str | None = None
    confidence_bias: float = 0.75

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("general_agent_task", task=task.model_dump())
        context_section = task.context
        if context_section is None and context.context is not None:
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()

        prompt = self._build_prompt(task, context_section=context_section)
        summary = await context.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.2,
        )

        await context.memory.store_working_memory(task.task_id, summary)

        evidence_count = len(task.prior_exchanges)
        if context_section:
            evidence_count += 1

        self_assessment = min(0.55 + 0.07 * evidence_count, 0.9)
        confidence = 0.68
        confidence_breakdown: dict[str, float] | None = None
        if context.scorer is not None:
            scoring = context.scorer.score(
                evidence_count=evidence_count,
                tool_result=None,
                self_assessment=self_assessment,
            )
            confidence = scoring.score
            confidence_breakdown = scoring.breakdown.as_dict()
            for component, value in confidence_breakdown.items():
                observe_confidence_component(agent=self.name, component=component, value=value)

        metadata: dict[str, Any] = {
            "type": "general_brief",
            "audience": task.metadata.get("audience"),
            "priority": task.metadata.get("priority"),
        }
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=summary,
            confidence=confidence,
            rationale="Initial triage summary prepared without tool usage.",
            metadata=metadata,
        )

    def _build_prompt(self, task: AgentInput, *, context_section: str | None = None) -> str:
        metadata_repr = self._format_metadata(task.metadata)
        history = _format_history(task.prior_exchanges)
        retrieved = context_section or "(no retrieved context)"
        return (
            f"User prompt:\n{task.prompt}\n\n"
            f"Metadata:\n{metadata_repr}\n\n"
            f"Prior agent outputs:\n{history}\n\n"
            f"Retrieved context:\n{retrieved}\n\n"
            "Answer directly when possible, flag missing info, and suggest the next specialist step if needed."
        )

    @staticmethod
    def _format_metadata(metadata: dict[str, Any]) -> str:
        if not metadata:
            return "(none)"
        lines = [f"- {key}: {value}" for key, value in metadata.items()]
        return "\n".join(lines)


def _format_history(outputs: list[AgentExchange]) -> str:
    if not outputs:
        return "(none)"
    lines = []
    for item in outputs:
        snippet = item.content or "(no content)"
        suffix = f" (confidence {item.confidence:.2f})" if item.confidence is not None else ""
        lines.append(f"- {item.agent}: {snippet}{suffix}")
    return "\n".join(lines)


def _serialize_agent_input(task: AgentInput) -> dict[str, Any]:
    return {
        "id": task.task_id,
        "prompt": task.prompt,
        "metadata": task.metadata,
        "outputs": [
            {"agent": exchange.agent, "content": exchange.content, "confidence": exchange.confidence}
            for exchange in task.prior_exchanges
        ],
    }
