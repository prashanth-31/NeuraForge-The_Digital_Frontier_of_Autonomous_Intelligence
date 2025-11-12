from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..agents.base import AgentContext
from ..core.logging import get_logger
from ..core.metrics import observe_confidence_component
from ..schemas.agents import AgentCapability, AgentExchange, AgentInput, AgentOutput
from ..services.tools import ToolDisabledError, ToolInvocationError, ToolInvocationResult

logger = get_logger(name=__name__)


@dataclass
class GeneralistAgent:
    name: str = "general_agent"
    capability: AgentCapability = AgentCapability.GENERAL
    system_prompt: str = (
        "You are NeuraForge's Generalist Agent. Offer clear, well-structured answers, collect clarifying"
        " details, and flag when specialist agents should assist. Keep outputs actionable and concise."
    )
    description: str = "Handles greetings, casual queries, or triage before escalating as needed."
    tool_preference: list[str] = field(default_factory=list)
    fallback_agent: str | None = None
    confidence_bias: float = 0.6

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("general_agent_task", task=task.model_dump())
        context_section = task.context
        if context_section is None and context.context is not None:
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()

        tool_result = await self._maybe_invoke_tool(task, context=context)
        tool_section = self._format_tool_response(tool_result)

        prompt = self._build_prompt(task, context_section=context_section, tool_section=tool_section)
        response = await context.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.2,
        )

        await context.memory.store_working_memory(task.task_id, response)

        evidence_count = len(task.prior_exchanges)
        if context_section:
            evidence_count += 1
        if tool_result is not None:
            evidence_count += max(len(self._extract_tool_items(tool_result)), 1)

        self_assessment = min(0.6 + 0.05 * evidence_count, 0.85)
        confidence = 0.65
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
            "type": "general_overview",
            "specialist_hint": self._specialist_hint(task.metadata),
        }
        if tool_result is not None:
            metadata["tool"] = self._tool_metadata(tool_result)
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=response,
            confidence=confidence,
            rationale="High-level response based on prompt and available context.",
            metadata=metadata,
        )

    def _build_prompt(
        self,
        task: AgentInput,
        *,
        context_section: str | None,
        tool_section: str,
    ) -> str:
        metadata = json.dumps(self._filtered_metadata(task.metadata), indent=2, sort_keys=True)
        history = _format_history(task.prior_exchanges)
        retrieved = context_section or "(no retrieved context)"
        return (
            f"User prompt:\n{task.prompt}\n\n"
            f"Known metadata:\n{metadata}\n\n"
            f"Prior agent exchanges:\n{history}\n\n"
            f"Retrieved context:\n{retrieved}\n\n"
            f"Tool insights:\n{tool_section}\n\n"
            "Provide a direct answer. Note any follow-up questions or specialist hand-offs that may be useful."
        )

    async def _maybe_invoke_tool(self, task: AgentInput, *, context: AgentContext) -> ToolInvocationResult | None:
        if context.tools is None:
            return None
        payload = self._build_search_payload(task)
        if payload is None:
            return None
        try:
            return await context.tools.invoke("research.search", payload)
        except (ToolDisabledError, ToolInvocationError) as exc:
            logger.warning("general_tool_failure", error=str(exc))
            return None

    def _build_search_payload(self, task: AgentInput) -> dict[str, Any] | None:
        prompt_text = (task.prompt or "").strip()
        metadata_terms: list[str] = []
        if isinstance(task.metadata, dict):
            for key in ("entity", "topic", "subject", "company"):
                value = task.metadata.get(key)
                if isinstance(value, str) and value.strip():
                    metadata_terms.append(value.strip())
        query_parts = [prompt_text] if prompt_text else []
        query_parts.extend(metadata_terms)
        query = " ".join(part for part in query_parts if part).strip()
        if len(query) < 3:
            return None
        return {"query": query[:512], "max_results": 5}

    def _format_tool_response(self, tool_result: ToolInvocationResult | None) -> str:
        if tool_result is None:
            return "(tool unavailable)"
        items = self._extract_tool_items(tool_result)
        if not items:
            return json.dumps(tool_result.response, indent=2, sort_keys=True)
        lines = []
        for item in items:
            summary = item.get("summary") or item.get("title") or item.get("content", "(no summary)")
            source = item.get("source") or item.get("url")
            if source:
                lines.append(f"- {summary} [{source}]")
            else:
                lines.append(f"- {summary}")
        return "\n".join(lines)

    @staticmethod
    def _extract_tool_items(tool_result: ToolInvocationResult) -> list[dict[str, Any]]:
        response = tool_result.response
        if "results" in response and isinstance(response["results"], list):
            return [item for item in response["results"] if isinstance(item, dict)]
        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]
        return []

    @staticmethod
    def _tool_metadata(tool_result: ToolInvocationResult) -> dict[str, Any]:
        return {
            "name": tool_result.tool,
            "resolved": tool_result.resolved_tool,
            "cached": tool_result.cached,
            "latency": round(tool_result.latency, 4),
        }

    @staticmethod
    def _filtered_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        return {key: value for key, value in metadata.items() if key != "_shared_context"}

    @staticmethod
    def _specialist_hint(metadata: dict[str, Any]) -> str | None:
        if not isinstance(metadata, dict):
            return None
        requested = metadata.get("target_capabilities") or metadata.get("preferred_capabilities")
        if isinstance(requested, list):
            return ", ".join(str(item) for item in requested if item)
        if isinstance(requested, str) and requested.strip():
            return requested.strip()
        return None


def _format_history(outputs: list[AgentExchange]) -> str:
    if not outputs:
        return "(none)"
    lines: list[str] = []
    for item in outputs:
        suffix = f" (confidence {item.confidence:.2f})" if item.confidence is not None else ""
        lines.append(f"- {item.agent}: {item.content}{suffix}")
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
