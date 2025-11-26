from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..agents.base import AgentContext
from ..core.logging import get_logger
from ..core.metrics import observe_confidence_component
from ..schemas.agents import AgentCapability, AgentExchange, AgentInput, AgentOutput
from ..services.tools import ToolDisabledError, ToolInvocationError, ToolInvocationResult

logger = get_logger(name=__name__)


@dataclass
class ResearchAgent:
    name: str = "research_agent"
    capability: AgentCapability = AgentCapability.RESEARCH
    system_prompt: str = (
        "You are NeuraForge's Research Agent. Derive concise, well-sourced insights from the task "
        "description and prior agent outputs. Highlight 2-3 key findings, prioritize the most recent "
        "data available, and optionally surface a clearly labeled Historical Insight when older context "
        "materially explains the story. Always note missing context."
    )
    description: str = "Finds factual, evidence-backed information and compiles concise briefings."
    tool_preference: list[str] = field(default_factory=lambda: ["research.search", "research.summarizer"])
    tool_candidates: tuple[str, ...] = (
        "research.search",
        "research.summarizer",
        "research.doc_loader",
        "research/doc_loader",
    )
    fallback_agent: str | None = "enterprise_agent"
    confidence_bias: float = 0.9

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("research_agent_task", task=task.model_dump())
        context_section = task.context
        if context_section is None and context.context is not None:
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()

        tool_result = await self._maybe_invoke_tool(task, context=context)
        tool_section = self._format_tool_response(tool_result)
        prompt = self._build_prompt(task, context_section=context_section)
        summary = await context.llm.generate(
            prompt=f"{prompt}\n\nTool Findings:\n{tool_section}",
            system_prompt=self.system_prompt,
            temperature=0.1,
        )

        await context.memory.store_working_memory(task.task_id, summary)

        evidence_count = len(task.prior_exchanges)
        if context_section:
            evidence_count += 1
        if tool_result:
            evidence_count += max(len(self._extract_tool_items(tool_result)), 1)

        self_assessment = min(0.55 + 0.1 * evidence_count, 0.95)
        confidence = 0.72
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

        metadata: dict[str, Any] = {"type": "research_summary"}
        if tool_result is not None:
            metadata["tool"] = self._tool_metadata(tool_result)
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=summary,
            confidence=confidence,
            rationale="Stack-ranked findings consolidated from retrieved context.",
            metadata=metadata,
        )

    def _build_prompt(self, task: AgentInput, context_section: str | None = None) -> str:
        metadata_payload = self._filtered_metadata(task.metadata)
        metadata = json.dumps(metadata_payload, indent=2, sort_keys=True)
        history = _format_history(task.prior_exchanges)
        retrieved = context_section or "(no retrieved context)"
        return (
            f"Primary question:\n{task.prompt or 'No prompt provided.'}\n\n"
            f"Known metadata:\n{metadata}\n\n"
            f"Prior agent outputs (if any):\n{history}\n\n"
            f"Retrieved context:\n{retrieved}\n\n"
            "Summarize credible research-backed insights, cite sources inline, and call out missing data. "
            "If tools uncovered a remarkable historical pattern, append a short 'Historical Insight' note "
            "that references the relevant year and source."
        )

    async def _maybe_invoke_tool(self, task: AgentInput, *, context: AgentContext) -> ToolInvocationResult | None:
        if context.tools is None:
            return None
        payload = self._build_search_payload(task)
        if payload is None:
            logger.debug("research_search_payload_skipped", task_id=task.task_id)
            return None
        try:
            return await context.tools.invoke("research.search", payload)
        except (ToolDisabledError, ToolInvocationError) as exc:
            logger.warning("research_tool_failure", error=str(exc))
            return None

    def _build_search_payload(self, task: AgentInput) -> dict[str, Any] | None:
        prompt_text = (task.prompt or "").strip()
        metadata: dict[str, Any] = task.metadata if isinstance(task.metadata, dict) else {}

        keywords: list[str] = []
        extra_terms = metadata.get("keywords") or metadata.get("search_terms")
        if isinstance(extra_terms, str) and extra_terms.strip():
            keywords.extend([extra_terms.strip()])
        elif isinstance(extra_terms, list):
            keywords.extend(str(item).strip() for item in extra_terms if isinstance(item, str) and item.strip())

        if metadata.get("entity") and isinstance(metadata["entity"], str):
            keywords.append(metadata["entity"].strip())

        query_parts = [prompt_text] if prompt_text else []
        if keywords:
            query_parts.append(" ".join(keywords))

        query = " ".join(part for part in query_parts if part).strip()
        if len(query) < 3:
            return None

        query = self._ensure_current_year(query)

        payload: dict[str, Any] = {"query": query[:512]}
        region = metadata.get("region") or metadata.get("market")
        if isinstance(region, str) and region.strip():
            payload["region"] = region.strip()
        max_results = metadata.get("max_results") or metadata.get("search_results")
        if isinstance(max_results, int) and 1 <= max_results <= 20:
            payload["max_results"] = max_results
        return payload

    def _ensure_current_year(self, query: str) -> str:
        if re.search(r"\b20\d{2}\b", query):
            return query
        current_year = datetime.now(UTC).year
        return f"{query} {current_year}".strip()

    def _format_tool_response(self, tool_result: ToolInvocationResult | None) -> str:
        if tool_result is None:
            return "(tool unavailable)"
        items = self._extract_tool_items(tool_result)
        if not items:
            return json.dumps(tool_result.response, indent=2, sort_keys=True)
        lines = []
        for item in items:
            summary = item.get("summary") or item.get("title") or item.get("content", "(no summary)")
            source = item.get("source")
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
