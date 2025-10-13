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
class FinanceAgent:
    name: str = "finance_agent"
    capability: AgentCapability = AgentCapability.FINANCE
    system_prompt: str = (
        "You are NeuraForge's Finance Agent. Provide high-level forecasts, key risks, and actionable"
        " next steps using available context. Respond with structured bullet points."
    )

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("finance_agent_task", task=task.model_dump())
        context_section = task.context
        if context_section is None and context.context is not None:
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()
        tool_result = await self._maybe_invoke_tool(task, context=context)
        prompt = self._build_prompt(task, context_section=context_section, tool_result=tool_result)
        forecast = await context.llm.generate(prompt=prompt, system_prompt=self.system_prompt, temperature=0.2)

        evidence_count = len(task.prior_exchanges)
        if context_section:
            evidence_count += 1
        if tool_result:
            evidence_count += max(len(self._extract_tool_metrics(tool_result)), 1)

        self_assessment = min(0.6 + 0.08 * evidence_count, 0.95)
        confidence = 0.68
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

        metadata: dict[str, Any] = {"type": "financial_forecast"}
        if tool_result is not None:
            metadata["tool"] = self._tool_metadata(tool_result)
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=forecast,
            confidence=confidence,
            rationale="Financial forecast generated from context and prior agent signals.",
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
        prior = _collect_prior_outputs(task.prior_exchanges)
        retrieved = context_section or "(no retrieved context)"
        tool_data = self._format_tool_metrics(tool_result)
        return (
            f"Business question:\n{task.prompt}\n\n"
            f"Financial metadata:\n{metadata}\n\n"
            f"Earlier agent signals:\n{prior}\n\n"
            f"Retrieved context:\n{retrieved}\n\n"
            f"Tool metrics:\n{tool_data}\n\n"
            "Estimate revenue or cost implications, surface 2-3 headline metrics, and outline immediate actions."
        )

    async def _maybe_invoke_tool(self, task: AgentInput, *, context: AgentContext) -> ToolInvocationResult | None:
        if context.tools is None:
            return None
        payload = {
            "query": task.prompt,
            "metadata": task.metadata,
            "signals": [exchange.model_dump() for exchange in task.prior_exchanges],
        }
        try:
            return await context.tools.invoke("finance.snapshot", payload)
        except (ToolDisabledError, ToolInvocationError) as exc:
            logger.warning("finance_tool_failure", error=str(exc))
            return None

    def _format_tool_metrics(self, tool_result: ToolInvocationResult | None) -> str:
        metrics = self._extract_tool_metrics(tool_result) if tool_result else []
        if not metrics:
            return "(tool metrics unavailable)"
        return "\n".join(
            f"- {item.get('name', 'metric')}: {item.get('value', 'n/a')} ({item.get('trend', 'stable')})"
            for item in metrics
        )

    @staticmethod
    def _extract_tool_metrics(tool_result: ToolInvocationResult | None) -> list[dict[str, Any]]:
        if tool_result is None:
            return []
        response = tool_result.response
        metrics = response.get("metrics") if isinstance(response, dict) else None
        if isinstance(metrics, list):
            return [item for item in metrics if isinstance(item, dict)]
        return []

    @staticmethod
    def _tool_metadata(tool_result: ToolInvocationResult) -> dict[str, Any]:
        return {
            "name": tool_result.tool,
            "resolved": tool_result.resolved_tool,
            "cached": tool_result.cached,
            "latency": round(tool_result.latency, 4),
        }


def _collect_prior_outputs(outputs: list[AgentExchange]) -> str:
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
