from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..agents.base import AgentContext
from ..core.logging import get_logger
from ..core.metrics import observe_confidence_component
from ..schemas.agents import AgentCapability, AgentExchange, AgentInput, AgentOutput
from ..services.tools import ToolDisabledError, ToolInvocationError, ToolInvocationResult

logger = get_logger(name=__name__)


@dataclass
class EnterpriseAgent:
    name: str = "enterprise_agent"
    capability: AgentCapability = AgentCapability.ENTERPRISE
    system_prompt: str = (
        "You are NeuraForge's Enterprise Agent. Translate insights into market positioning, "
        "operational tactics, and executive-ready recommendations."
    )

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("enterprise_agent_task", task=task.model_dump())
        context_section = task.context
        if context_section is None and context.context is not None:
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()
        tool_result = await self._maybe_invoke_tool(task, context=context)
        prompt = self._build_prompt(task, context_section=context_section, tool_result=tool_result)
        strategy = await context.llm.generate(prompt=prompt, system_prompt=self.system_prompt, temperature=0.15)

        evidence_count = len(task.prior_exchanges)
        if context_section:
            evidence_count += 1
        if tool_result is not None:
            evidence_count += len(self._extract_actions(tool_result)) or 1

        self_assessment = min(0.62 + 0.1 * evidence_count, 0.97)
        confidence = 0.74
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

        metadata: dict[str, Any] = {"type": "enterprise_strategy"}
        if tool_result is not None:
            metadata["tool"] = self._tool_metadata(tool_result)
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=strategy,
            confidence=confidence,
            rationale="Executive recommendations synthesized from multi-agent context.",
            metadata=metadata,
        )

    def _build_prompt(
        self,
        task: AgentInput,
        *,
        context_section: str | None = None,
        tool_result: ToolInvocationResult | None = None,
    ) -> str:
        metadata = json.dumps(task.metadata, indent=2, sort_keys=True)
        prior = _summarize_prior_outputs(task.prior_exchanges)
        retrieved = context_section or "(no retrieved context)"
        tool_actions = self._format_tool_actions(tool_result)
        return (
            f"Executive task:\n{task.prompt}\n\n"
            f"Business metadata:\n{metadata}\n\n"
            f"Cross-agent insights:\n{prior}\n\n"
            f"Retrieved context:\n{retrieved}\n\n"
            f"Playbook suggestions:\n{tool_actions}\n\n"
            "Deliver a numbered action plan (max 5 steps) with expected impact and confidence."
        )

    async def _maybe_invoke_tool(self, task: AgentInput, *, context: AgentContext) -> ToolInvocationResult | None:
        if context.tools is None:
            return None
        payload = {
            "prompt": task.prompt,
            "metadata": task.metadata,
            "prior_outputs": [exchange.model_dump() for exchange in task.prior_exchanges],
        }
        try:
            return await context.tools.invoke("enterprise.playbook", payload)
        except (ToolDisabledError, ToolInvocationError) as exc:
            logger.warning("enterprise_tool_failure", error=str(exc))
            return None

    def _format_tool_actions(self, tool_result: ToolInvocationResult | None) -> str:
        actions = self._extract_actions(tool_result) if tool_result else []
        if not actions:
            return "(no playbook suggestions)"
        return "\n".join(
            f"- {item.get('action', 'action')} (impact: {item.get('impact', 'n/a')})"
            for item in actions
        )

    @staticmethod
    def _extract_actions(tool_result: ToolInvocationResult | None) -> list[dict[str, Any]]:
        if tool_result is None:
            return []
        response = tool_result.response
        actions = response.get("actions") if isinstance(response, dict) else None
        if isinstance(actions, list):
            return [item for item in actions if isinstance(item, dict)]
        return []

    @staticmethod
    def _tool_metadata(tool_result: ToolInvocationResult) -> dict[str, Any]:
        return {
            "name": tool_result.tool,
            "resolved": tool_result.resolved_tool,
            "cached": tool_result.cached,
            "latency": round(tool_result.latency, 4),
        }


def _summarize_prior_outputs(outputs: list[AgentExchange]) -> str:
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
