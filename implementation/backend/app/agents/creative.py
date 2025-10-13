from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agents.base import AgentContext
from ..core.logging import get_logger
from ..core.metrics import observe_confidence_component
from ..schemas.agents import AgentCapability, AgentExchange, AgentInput, AgentOutput
from ..services.tools import ToolDisabledError, ToolInvocationError, ToolInvocationResult

logger = get_logger(name=__name__)


@dataclass
class CreativeAgent:
    name: str = "creative_agent"
    capability: AgentCapability = AgentCapability.CREATIVE
    system_prompt: str = (
        "You are NeuraForge's Creative Agent. Craft compelling narratives, taglines, or storytelling"
        " snippets tailored to the task. Blend clarity with creativity."
    )

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("creative_agent_task", task=task.model_dump())
        context_section = task.context
        if context_section is None and context.context is not None:
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()
        tool_result = await self._maybe_invoke_tool(task, context=context)
        prompt = self._build_prompt(task, context_section=context_section, tool_result=tool_result)
        creative_output = await context.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.6,
        )

        evidence_count = len(task.prior_exchanges)
        if context_section:
            evidence_count += 1
        if tool_result is not None:
            evidence_count += 1

        base_self_assessment = 0.65 if tool_result is None else 0.75
        self_assessment = min(base_self_assessment + 0.05 * evidence_count, 0.95)
        confidence = 0.7
        confidence_breakdown: dict[str, float] | None = None
        if context.scorer is not None:
            scoring = context.scorer.score(
                evidence_count=evidence_count,
                tool_result=tool_result,
                self_assessment=self_assessment,
            )
            confidence = scoring.score
            confidence_breakdown = scoring.breakdown.as_dict()
            for component, value in confidence_breakdown.items():
                observe_confidence_component(agent=self.name, component=component, value=value)

        metadata: dict[str, Any] = {
            "type": "creative_direction",
            "audience": task.metadata.get("audience"),
        }
        if tool_result is not None:
            metadata["tool"] = self._tool_metadata(tool_result)
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=creative_output,
            confidence=confidence,
            rationale="Creative draft aligned with requested tone.",
            metadata=metadata,
        )

    def _build_prompt(
        self,
        task: AgentInput,
        *,
        context_section: str | None = None,
        tool_result: ToolInvocationResult | None = None,
    ) -> str:
        audience = task.metadata.get("audience", "a general audience")
        prior = _collect_task_context(task.prior_exchanges)
        retrieved = context_section or "(no retrieved context)"
        tone_checks = self._format_tool_feedback(tool_result)
        return (
            f"Core request:\n{task.prompt}\n\n"
            f"Primary audience: {audience}\n\n"
            f"Previous agent notes:\n{prior}\n\n"
            f"Retrieved context:\n{retrieved}\n\n"
            f"Tone guidance:\n{tone_checks}\n\n"
            "Deliver a vibrant, memorable piece (<=150 words) that aligns with the strategy."
        )

    async def _maybe_invoke_tool(self, task: AgentInput, *, context: AgentContext) -> ToolInvocationResult | None:
        if context.tools is None:
            return None
        payload = {
            "prompt": task.prompt,
            "audience": task.metadata.get("audience"),
        }
        try:
            return await context.tools.invoke("creative.tonecheck", payload)
        except (ToolDisabledError, ToolInvocationError) as exc:
            logger.warning("creative_tool_failure", error=str(exc))
            return None

    @staticmethod
    def _format_tool_feedback(tool_result: ToolInvocationResult | None) -> str:
        if tool_result is None:
            return "(no tone guidance)"
        response = tool_result.response
        if isinstance(response, dict):
            suggestions = response.get("suggestions")
            if isinstance(suggestions, list):
                return "\n".join(str(item) for item in suggestions)
        return str(response)

    @staticmethod
    def _tool_metadata(tool_result: ToolInvocationResult) -> dict[str, Any]:
        return {
            "name": tool_result.tool,
            "resolved": tool_result.resolved_tool,
            "cached": tool_result.cached,
            "latency": round(tool_result.latency, 4),
        }


def _collect_task_context(outputs: list[AgentExchange]) -> str:
    if not outputs:
        return "(none)"
    return "\n".join(
        f"- {item.agent}: {item.content or '(no content)'}"
        + (f" (confidence {item.confidence:.2f})" if item.confidence is not None else "")
        for item in outputs
    )


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
