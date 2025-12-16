from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from ..agents.base import AgentContext, ReasoningBuilder
from ..core.logging import get_logger
from ..core.metrics import observe_confidence_component
from ..schemas.agents import (
    AgentCapability,
    AgentExchange,
    AgentInput,
    AgentOutput,
    ReasoningStepType,
)
from ..services.llm import is_llm_unavailable
from ..services.tools import ToolDisabledError, ToolInvocationError, ToolInvocationResult
from ..mcp.symbols import extract_symbols_from_text

logger = get_logger(name=__name__)

FINANCE_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "finance",
    "financial",
    "market",
    "stock",
    "equity",
    "earnings",
    "invest",
    "valuation",
)

FINANCE_METRIC_KEYWORDS: dict[str, str] = {
    "market cap": "market_cap",
    "market capitalization": "market_cap",
    "market-cap": "market_cap",
    "share price": "price",
    "stock price": "price",
    "price": "price",
    "eps": "eps",
    "earnings per share": "eps",
    "volume": "volume",
    "trading volume": "volume",
    "revenue": "revenue",
    "revenue growth": "revenue_growth",
    "growth rate": "revenue_growth",
}

CRITICAL_FINANCE_FIELDS: tuple[str, ...] = ("price", "market_cap", "volume")
DEFAULT_FINANCE_METRICS: tuple[str, ...] = CRITICAL_FINANCE_FIELDS + ("eps", "revenue", "revenue_growth")

DEFAULT_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "price": ("price", "current_price", "regularMarketPrice", "last_price"),
    "market_cap": ("market_cap", "marketCap", "marketCapitalization"),
    "market_capitalization": ("market_cap", "marketCap", "marketCapitalization"),
    "eps": (
        "eps",
        "earnings_per_share",
        "fundamentals.guidance.forward_eps",
        "fundamentals.eps",
        "fundamentals.diluted_eps_ttm",
    ),
    "earnings_per_share": (
        "eps",
        "earnings_per_share",
        "fundamentals.guidance.forward_eps",
        "fundamentals.eps",
        "fundamentals.diluted_eps_ttm",
    ),
    "revenue": (
        "revenue",
        "fundamentals.trailing.revenue",
        "fundamentals.quarterly.0.total_revenue",
        "fundamentals.revenue_ttm",
    ),
    "revenue_growth": (
        "fundamentals.guidance.revenue_growth",
        "guidance.revenue_growth",
        "fundamentals.revenue_growth",
    ),
    "revenue_growth_rate": (
        "fundamentals.guidance.revenue_growth",
        "guidance.revenue_growth",
        "fundamentals.revenue_growth",
    ),
    "volume": ("volume", "regularMarketVolume", "averageDailyVolume10Day"),
    "day_high": ("day_high", "regularMarketDayHigh"),
    "day_low": ("day_low", "regularMarketDayLow"),
}


@dataclass
class ResearchAgent:
    name: str = "research_agent"
    capability: AgentCapability = AgentCapability.RESEARCH
    system_prompt: str = (
        "You are NeuraForge's Research Agent. Derive concise, well-sourced insights from the task "
        "description and prior agent outputs. Highlight 2-3 key findings, prioritize the most recent "
        "data available, and optionally surface a clearly labeled Historical Insight when older context "
        "materially explains the story. Always note missing context. "
        "IMPORTANT: When citing academic papers, ALWAYS include the full reference with authors, year, "
        "title, and URL/link. Format references at the end of your response in a 'References' section."
    )
    description: str = "Finds factual, evidence-backed information and compiles concise briefings."
    tool_preference: list[str] = field(default_factory=lambda: ["research.arxiv", "research.search", "research.wikipedia", "research.summarizer"])
    tool_candidates: tuple[str, ...] = (
        # Core research tools (all open source / free APIs)
        "research.search",            # DuckDuckGo - anonymous, free
        "research.summarizer",        # Text summarization
        "research.wikipedia",         # Wikipedia REST API - free
        "research.arxiv",             # arXiv API - open academic papers
        "research.doc_loader",        # Document parsing
        "research/doc_loader",        # Alias
        "research.vector_search",     # Qdrant vector search
        # Browser tools for web research
        "browser.open",               # HTTP fetching
        "browser.extract_text",       # HTML text extraction
        # Data analysis
        "dataframe.analyze",          # Pandas analytics
        "dataframe.transform",        # Data transformations
    )
    fallback_agent: str | None = "enterprise_agent"
    confidence_bias: float = 0.9
    finance_tool_candidates: tuple[str, ...] = (
        "finance.snapshot",
        "finance.snapshot.alpha",
        "finance.snapshot.cached",
    )
    default_required_metrics: tuple[str, ...] = DEFAULT_FINANCE_METRICS
    metric_aliases: dict[str, tuple[str, ...]] = field(default_factory=lambda: DEFAULT_METRIC_ALIASES.copy())

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("research_agent_task", task=task.model_dump())
        
        # Initialize reasoning builder for tracking thought process
        reasoning = ReasoningBuilder(agent_name=self.name, context=context)
        
        # Step 1: Analyze the research request
        await reasoning.think(
            f"Analyzing research request: '{task.prompt[:100]}...'",
            step_type=ReasoningStepType.OBSERVATION,
        )
        
        # Step 2: Gather context
        context_section = task.context
        if context_section is None and context.context is not None:
            await reasoning.think(
                "Assembling context from knowledge base and prior conversations",
                step_type=ReasoningStepType.ANALYSIS,
            )
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()

        # Step 3: Evaluate and invoke tools
        await reasoning.think(
            "Evaluating available research tools for data gathering",
            step_type=ReasoningStepType.TOOL_SELECTION,
        )
        
        # Check for tool availability
        if context.tools is None:
            await reasoning.consider_tool(
                tool_name="research.search",
                reason="Primary research tool for gathering external information",
                selected=False,
                rejection_reason="Tool service not available",
            )
            tool_result = None
        else:
            await reasoning.consider_tool(
                tool_name="research.search",
                reason="Can search for factual information and recent data",
                selected=True,
            )
            tool_result = await self._maybe_invoke_tool(task, context=context)
            if tool_result:
                await reasoning.add_finding(
                    claim=f"Research tool returned {len(self._extract_tool_items(tool_result))} results",
                    evidence=[str(tool_result.payload)[:200] + "..."] if tool_result.payload else [],
                    confidence=0.8,
                    source="research.search",
                )
        
        # Step 4: Check for financial metrics
        requested_metrics = self._requested_metrics_from_prompt(task)
        if requested_metrics:
            await reasoning.think(
                f"Identified {len(requested_metrics)} requested financial metrics: {', '.join(requested_metrics[:5])}",
                step_type=ReasoningStepType.ANALYSIS,
            )
        
        resolved_metrics = self._collect_metrics_from_tool(tool_result, requested_metrics)
        missing_metrics = [metric for metric in requested_metrics if metric not in resolved_metrics]
        
        if missing_metrics:
            await reasoning.note_uncertainty(
                f"Could not resolve all requested metrics. Missing: {', '.join(missing_metrics[:5])}"
            )

        enrichment: dict[str, Any] | None = None
        finance_context = self._should_attempt_finance_enrichment(
            task.metadata if isinstance(task.metadata, Mapping) else None,
            task.prompt,
        )
        should_enrich = bool(missing_metrics) and (finance_context or bool(requested_metrics))
        if should_enrich:
            await reasoning.think(
                f"Attempting finance enrichment for {len(missing_metrics)} missing metrics",
                step_type=ReasoningStepType.DECISION,
            )
            enrichment = await self._enrich_financial_metrics(
                task,
                context=context,
                requested_metrics=missing_metrics,
            )
            if enrichment is not None:
                resolved_metrics.update(enrichment.get("metrics", {}))
                remaining = [metric for metric in requested_metrics if metric not in resolved_metrics]
                enrichment["missing"] = remaining
                if tool_result is not None and enrichment.get("metrics"):
                    self._inject_metrics_into_tool_payload(tool_result, enrichment)
                    
                await reasoning.add_finding(
                    claim=f"Finance enrichment provided {len(enrichment.get('metrics', {}))} additional metrics",
                    evidence=list(enrichment.get("metrics", {}).keys()),
                    confidence=0.75,
                    source="finance_enrichment",
                )
        
        # Step 5: Synthesize findings
        await reasoning.think(
            "Synthesizing all gathered information into comprehensive research summary",
            step_type=ReasoningStepType.SYNTHESIS,
        )
        
        tool_section = self._format_tool_response(
            tool_result,
            resolved_metrics if resolved_metrics else None,
            self._build_metrics_context(tool_result, enrichment, resolved_metrics) if resolved_metrics else None,
        )
        prompt = self._build_prompt(task, context_section=context_section)
        compiled_prompt = f"{prompt}\n\nTool Findings:\n{tool_section}"
        if enrichment is not None and (enrichment.get("missing") or enrichment.get("reason")):
            compiled_prompt += f"\n\nSupplemental Metrics:\n{self._format_enrichment_section(enrichment)}"
        summary = await context.llm.generate(
            prompt=compiled_prompt,
            system_prompt=self.system_prompt,
            temperature=0.1,
        )

        # Handle LLM unavailability with fallback
        if is_llm_unavailable(summary):
            logger.warning("research_agent_llm_unavailable", task_id=task.task_id)
            summary = self._generate_fallback_response(task, tool_result, context_section, tool_section)
            await reasoning.note_uncertainty(
                "LLM temporarily unavailable - providing structured research data"
            )

        await context.memory.store_working_memory(task.task_id, summary)

        # Step 6: Calculate confidence
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

        await reasoning.think(
            f"Final confidence assessment: {confidence:.2f} based on {evidence_count} evidence sources",
            step_type=ReasoningStepType.EVALUATION,
            confidence=confidence,
        )

        metadata: dict[str, Any] = {"type": "research_summary"}
        if tool_result is not None:
            metadata["tool"] = self._tool_metadata(tool_result)
        if resolved_metrics:
            metadata["resolved_metrics"] = resolved_metrics
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown
        if enrichment is not None:
            metadata["enriched_metrics"] = enrichment

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=summary,
            confidence=confidence,
            rationale="Stack-ranked findings consolidated from retrieved context.",
            metadata=metadata,
            # Include reasoning transparency
            reasoning_steps=reasoning.steps,
            key_findings=reasoning.findings,
            tools_considered=reasoning.tools_considered,
            uncertainties=reasoning.uncertainties,
        )

    def _generate_fallback_response(
        self,
        task: AgentInput,
        tool_result: ToolInvocationResult | None,
        context_section: str | None,
        tool_section: str | None,
    ) -> str:
        """Generate a structured fallback response when LLM is unavailable."""
        response_parts = []
        
        # Header
        response_parts.append("## Research Summary (Structured Data)\n")
        
        # Include query
        if task.prompt:
            response_parts.append(f"**Query**: {task.prompt[:200]}\n")
        
        # Include tool data if available
        if tool_section:
            response_parts.append("### Search Results\n")
            response_parts.append(tool_section)
            response_parts.append("")
        elif tool_result and tool_result.response:
            response_parts.append("### Data Collected\n")
            if isinstance(tool_result.response, dict):
                for key, value in list(tool_result.response.items())[:10]:
                    if isinstance(value, (str, int, float)):
                        response_parts.append(f"- **{key}**: {value}")
            elif isinstance(tool_result.response, list):
                for i, item in enumerate(tool_result.response[:5]):
                    if isinstance(item, dict):
                        title = item.get("title") or item.get("name") or f"Result {i+1}"
                        response_parts.append(f"- {title}")
                        if item.get("description") or item.get("snippet"):
                            response_parts.append(f"  - {(item.get('description') or item.get('snippet', ''))[:150]}")
            response_parts.append("")
        
        # Context
        if context_section:
            response_parts.append("### Relevant Context\n")
            response_parts.append(f"{context_section[:500]}...\n")
        
        response_parts.append("\n---\n*Note: This is structured data as the LLM is temporarily unavailable. For synthesized analysis, please try again later.*")
        
        return "\n".join(response_parts)

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
            "Summarize credible research-backed insights, cite sources inline. "
            "Only mention missing information if it's directly relevant to the research question being asked. "
            "If tools uncovered a remarkable historical pattern, append a short 'Historical Insight' note "
            "that references the relevant year and source."
        )

    def _get_planned_tools(self, task: AgentInput) -> list[str]:
        """Extract planned tools from task metadata, falling back to defaults."""
        metadata = task.metadata if isinstance(task.metadata, Mapping) else {}
        shared_context = metadata.get("_shared_context", {})
        planner = shared_context.get("planner", {})
        steps = planner.get("steps", [])
        
        # Find our step in the plan
        for step in steps:
            if step.get("agent") == self.name:
                tools = step.get("tools", [])
                fallback_tools = step.get("fallback_tools", [])
                if tools or fallback_tools:
                    return list(tools) + list(fallback_tools)
        
        # If no tools from planner, detect query type and select appropriate tool
        prompt_lower = (task.prompt or "").lower()
        
        # Academic paper indicators → use arXiv
        academic_indicators = {
            "research paper", "research papers", "academic paper", "academic papers",
            "scientific paper", "journal article", "preprint", "arxiv",
            "literature review", "latest paper", "recent paper",
            "find papers", "search papers", "papers about", "papers on",
            "key findings", "multi-ai research", "ai research",
            "ml research", "machine learning research", "deep learning research",
        }
        if any(indicator in prompt_lower for indicator in academic_indicators):
            return ["research.arxiv", "research.search"]
        
        # Wikipedia indicators
        wiki_indicators = {
            "history of", "what is", "who is", "who was", "biography",
            "tell me about", "explain", "describe",
        }
        if any(indicator in prompt_lower for indicator in wiki_indicators):
            return ["research.wikipedia", "research.search"]
        
        # Default research tools
        return ["research.search"]

    def _build_tool_payload(self, tool_name: str, task: AgentInput) -> dict[str, Any] | None:
        """Build appropriate payload for the given tool."""
        prompt_text = (task.prompt or "").strip()
        if len(prompt_text) < 3:
            return None
        
        # Wikipedia expects {"title": ...}
        if "wikipedia" in tool_name.lower():
            # Extract the main topic from the prompt
            # Remove common prefixes like "Give me history of", "What is", etc.
            title = prompt_text
            prefixes_to_remove = [
                "give me history of",
                "give me the history of", 
                "what is the history of",
                "tell me about",
                "what is",
                "who is",
                "who was",
                "explain",
                "describe",
            ]
            title_lower = title.lower()
            for prefix in prefixes_to_remove:
                if title_lower.startswith(prefix):
                    title = title[len(prefix):].strip()
                    break
            # Remove trailing punctuation
            title = title.rstrip("?.!")
            return {"title": title[:200]} if title else None
        
        # ArXiv expects {"query": ...}
        if "arxiv" in tool_name.lower():
            return {"query": prompt_text[:512]}
        
        # Default search payload
        return self._build_search_payload(task)

    async def _maybe_invoke_tool(self, task: AgentInput, *, context: AgentContext) -> ToolInvocationResult | None:
        if context.tools is None:
            return None
        
        # PRIORITY 1: Use tools from context (set by orchestrator from current planner step)
        # This is the authoritative source for the current task's planned tools
        planned_tools: list[str] = []
        if context.planned_tools:
            planned_tools = list(context.planned_tools)
            if context.fallback_tools:
                planned_tools.extend(context.fallback_tools)
            logger.info(
                "research_agent_using_context_tools",
                task_id=task.task_id,
                planned_tools=planned_tools,
                source="context",
            )
        else:
            # PRIORITY 2: Fall back to metadata extraction (for backward compatibility)
            planned_tools = self._get_planned_tools(task)
            logger.info(
                "research_agent_planned_tools",
                task_id=task.task_id,
                planned_tools=planned_tools,
                source="metadata_fallback",
            )
        
        # Try planned tools in order with appropriate payloads
        for tool_name in planned_tools:
            payload = self._build_tool_payload(tool_name, task)
            if payload is None:
                logger.debug("research_payload_skipped", tool=tool_name, task_id=task.task_id)
                continue
            try:
                logger.info(
                    "research_agent_invoking_tool",
                    tool=tool_name,
                    task_id=task.task_id,
                    payload_keys=list(payload.keys()),
                )
                result = await context.tools.invoke(tool_name, payload)
                if result:
                    return result
            except (ToolDisabledError, ToolInvocationError) as exc:
                logger.warning("research_tool_failure", tool=tool_name, error=str(exc))
                continue
        
        # Fallback to research.search if none of the planned tools worked
        if "research.search" not in planned_tools:
            payload = self._build_search_payload(task)
            if payload:
                try:
                    return await context.tools.invoke("research.search", payload)
                except (ToolDisabledError, ToolInvocationError) as exc:
                    logger.warning("research_tool_failure", tool="research.search", error=str(exc))
        
        return None

    def _requested_metrics_from_prompt(self, task: AgentInput) -> list[str]:
        metadata = task.metadata if isinstance(task.metadata, Mapping) else {}
        metrics: list[str] = []

        def _add_metric(raw: str | None) -> None:
            normalized = self._normalize_metric_name(raw)
            if normalized:
                metrics.append(normalized)

        requested = metadata.get("required_metrics")
        if isinstance(requested, Sequence):
            for item in requested:
                if isinstance(item, str):
                    _add_metric(item)
        elif isinstance(requested, str):
            _add_metric(requested)

        def _scan_metadata_values(payload: Mapping[str, Any]) -> None:
            for value in payload.values():
                if isinstance(value, str):
                    metrics.extend(self._metrics_from_text(value))
                elif isinstance(value, Mapping):
                    _scan_metadata_values(value)
                elif isinstance(value, Sequence):
                    for item in value:
                        if isinstance(item, str):
                            metrics.extend(self._metrics_from_text(item))

        if metadata:
            _scan_metadata_values(metadata)

        metrics.extend(self._metrics_from_text(task.prompt))
        for exchange in task.prior_exchanges:
            metrics.extend(self._metrics_from_text(exchange.content))

        normalized = [self._normalize_metric_name(metric) for metric in metrics if metric]
        deduped = [metric for metric in normalized if metric]
        deduped = list(dict.fromkeys(deduped))

        if not deduped and self._should_attempt_finance_enrichment(metadata, task.prompt):
            deduped = list(self.default_required_metrics)

        return deduped

    def _collect_metrics_from_tool(
        self,
        tool_result: ToolInvocationResult | None,
        requested_metrics: Sequence[str],
    ) -> dict[str, Any]:
        if tool_result is None or not requested_metrics:
            return {}
        candidates: list[Mapping[str, Any]] = []
        items = self._extract_tool_items(tool_result)
        candidates.extend(item for item in items if isinstance(item, Mapping))
        if isinstance(tool_result.response, Mapping):
            candidates.append(tool_result.response)
        collected: dict[str, Any] = {}
        for metric in requested_metrics:
            for source in candidates:
                value = self._lookup_metric_value(source, metric)
                if value is not None:
                    collected[metric] = value
                    break
        return collected

    def _should_attempt_finance_enrichment(
        self,
        metadata: Mapping[str, Any] | None,
        prompt: str | None,
    ) -> bool:
        prompt_text = (prompt or "").lower()
        if any(keyword in prompt_text for keyword in FINANCE_CONTEXT_KEYWORDS):
            return True
        metadata = metadata or {}
        topic_fields = (metadata.get("category"), metadata.get("topic"), metadata.get("domain"))
        for field in topic_fields:
            if isinstance(field, str) and any(keyword in field.lower() for keyword in FINANCE_CONTEXT_KEYWORDS):
                return True
        return self._has_symbol_hint(metadata, prompt)

    def _has_symbol_hint(self, metadata: Mapping[str, Any], prompt: str | None) -> bool:
        """
        Check if the prompt/metadata contains hints of stock symbols.
        
        IMPORTANT: We must be conservative to avoid false positives like "AI", "ML", "LLM"
        which are common tech acronyms, NOT stock tickers.
        """
        # Common tech/AI acronyms that should NOT trigger finance enrichment
        NON_TICKER_ACRONYMS = {
            "ai", "ml", "llm", "nlp", "api", "ui", "ux", "db", "sql", "css", "html",
            "sdk", "ide", "gpu", "cpu", "ram", "ssd", "hdd", "iot", "ar", "vr", "xr",
            "crm", "erp", "saas", "paas", "iaas", "devops", "cicd", "qa", "pm", "hr",
            "b2b", "b2c", "roi", "kpi", "mvp", "poc", "rfp", "rfi", "sla", "nda",
        }
        
        keys = ("symbol", "ticker", "entity", "company")
        list_keys = ("symbols", "tickers", "companies")
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                # Check if it's a known non-ticker acronym
                if value.strip().lower() not in NON_TICKER_ACRONYMS:
                    return True
        for key in list_keys:
            value = metadata.get(key)
            if isinstance(value, Sequence):
                real_symbols = [
                    item for item in value 
                    if isinstance(item, str) and item.strip() and item.strip().lower() not in NON_TICKER_ACRONYMS
                ]
                if real_symbols:
                    return True
        
        # Use the symbol extractor but filter out non-ticker acronyms
        extracted = extract_symbols_from_text(prompt)
        if extracted:
            real_symbols = [s for s in extracted if s.lower() not in NON_TICKER_ACRONYMS]
            if real_symbols:
                return True
        
        # DO NOT use the aggressive regex fallback - it causes too many false positives
        # The old code: return bool(re.search(r"\b[A-Z]{2,5}\b", prompt_str))
        # This would match "AI", "ML", "LLM" etc. which are NOT stock tickers
        return False

    def _metrics_from_text(self, text: str | None) -> list[str]:
        if not text:
            return []
        lower = text.lower()
        results: list[str] = []
        for phrase, metric in FINANCE_METRIC_KEYWORDS.items():
            if phrase in lower:
                results.append(metric)
        return results

    @staticmethod
    def _normalize_metric_name(value: str | None) -> str | None:
        if not value:
            return None
        cleaned = value.strip().lower().replace(" ", "_")
        return cleaned or None

    async def _enrich_financial_metrics(
        self,
        task: AgentInput,
        *,
        context: AgentContext,
        requested_metrics: Sequence[str],
    ) -> dict[str, Any] | None:
        if not requested_metrics:
            return None
        enrichment: dict[str, Any] = {"metrics": {}, "missing": list(requested_metrics)}
        if context.tools is None:
            enrichment["reason"] = "tool_service_unavailable"
            return enrichment
        payload = self._build_finance_payload(task)
        if payload is None:
            enrichment["reason"] = "finance_payload_unavailable"
            return enrichment

        for alias in self.finance_tool_candidates:
            try:
                result = await context.tools.invoke(alias, payload)
            except (ToolDisabledError, ToolInvocationError) as exc:
                logger.debug("research_finance_tool_attempt_failed", tool=alias, error=str(exc))
                continue

            metrics = self._extract_finance_metrics(result)
            if not metrics:
                continue
            snapshot = metrics[0]
            collected = self._summarize_required_metrics(snapshot, requested_metrics)
            if collected:
                enrichment["metrics"].update(collected)

            timestamp_value = snapshot.get("updated_at")
            if isinstance(timestamp_value, datetime):
                timestamp_value = timestamp_value.isoformat()
            elif not isinstance(timestamp_value, str) and isinstance(result.response, Mapping):
                generated_at = result.response.get("generated_at")
                timestamp_value = generated_at if isinstance(generated_at, str) else None

            enrichment.update(
                {
                    "source": result.resolved_tool or alias,
                    "symbol": snapshot.get("symbol"),
                    "timestamp": timestamp_value,
                }
            )

            enrichment["missing"] = [metric for metric in requested_metrics if metric not in enrichment["metrics"]]

            if not enrichment["missing"]:
                enrichment.pop("reason", None)
                return enrichment

        if enrichment["metrics"]:
            enrichment.setdefault("reason", "partial_metrics_available")
        else:
            enrichment.setdefault("reason", "finance_tool_unavailable")
        return enrichment

    def _build_finance_payload(self, task: AgentInput) -> dict[str, Any] | None:
        metadata = task.metadata if isinstance(task.metadata, Mapping) else {}
        symbols = self._extract_symbols(metadata)
        if not symbols:
            symbols = self._infer_symbols_from_prompt(task.prompt)
        unique = self._unique_upper(symbols)
        if unique:
            return {"symbols": unique[:3]}
        for key in ("entity", "company"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return {"query": value.strip()[:120]}
        prompt_text = (task.prompt or "").strip()
        symbols_from_prompt = extract_symbols_from_text(prompt_text)
        if symbols_from_prompt:
            return {"symbols": symbols_from_prompt[:3]}
        return {"query": prompt_text[:120]} if prompt_text else None

    def _extract_symbols(self, metadata: Mapping[str, Any]) -> list[str]:
        symbols: list[str] = []
        list_keys = ("symbols", "tickers", "companies")
        for key in list_keys:
            entries = metadata.get(key)
            if isinstance(entries, Sequence):
                for item in entries:
                    if isinstance(item, str) and item.strip():
                        symbols.append(item.strip())
        single_keys = ("symbol", "ticker", "entity", "company")
        for key in single_keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                symbols.append(value.strip())
        return symbols

    def _infer_symbols_from_prompt(self, prompt: str | None) -> list[str]:
        if not prompt:
            return []
        inferred = extract_symbols_from_text(prompt)
        regex_matches = re.findall(r"\b[A-Z]{2,5}\b", prompt)
        combined: list[str] = []
        for symbol in inferred + regex_matches:
            candidate = symbol.upper()
            if candidate and candidate not in combined:
                combined.append(candidate)
        return combined

    @staticmethod
    def _unique_upper(symbols: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for symbol in symbols:
            candidate = symbol.upper()
            if candidate and candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered

    def _extract_finance_metrics(self, tool_result: ToolInvocationResult) -> list[dict[str, Any]]:
        response = tool_result.response if isinstance(tool_result.response, Mapping) else {}
        metrics = response.get("metrics") if isinstance(response, Mapping) else None
        if isinstance(metrics, list):
            return [item for item in metrics if isinstance(item, dict)]
        return []

    def _summarize_required_metrics(self, snapshot: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
        collected: dict[str, Any] = {}
        for metric in required:
            value = self._lookup_metric_value(snapshot, metric)
            if value is not None:
                collected[metric] = value
        return collected

    def _lookup_metric_value(self, snapshot: Mapping[str, Any], metric: str) -> Any:
        candidates = self.metric_aliases.get(metric) or self.metric_aliases.get(metric.lower()) or (metric,)
        for candidate in candidates:
            value = self._get_nested_value(snapshot, candidate)
            if value is not None:
                return value
        normalized = metric.lower().replace(" ", "_")
        if normalized != metric:
            candidates = self.metric_aliases.get(normalized)
            if candidates:
                for candidate in candidates:
                    value = self._get_nested_value(snapshot, candidate)
                    if value is not None:
                        return value
        return None

    def _get_nested_value(self, payload: Any, path: str) -> Any:
        if not path:
            return None
        current: Any = payload
        tokens = [token for token in path.replace("[", ".").replace("]", "").split(".") if token]
        for token in tokens:
            if isinstance(current, Mapping):
                if token in current:
                    current = current[token]
                    continue
                return None
            if isinstance(current, Sequence):
                if token.isdigit():
                    index = int(token)
                    if 0 <= index < len(current):
                        current = current[index]
                        continue
                return None
            return None
        return current

    def _inject_metrics_into_tool_payload(
        self,
        tool_result: ToolInvocationResult,
        enrichment: Mapping[str, Any],
    ) -> None:
        response = tool_result.response
        if not isinstance(response, dict):
            return
        finance_block = response.setdefault("finance_metrics", {})
        if not isinstance(finance_block, dict):
            return
        metrics_block = finance_block.setdefault("metrics", {})
        if not isinstance(metrics_block, dict):
            return
        source = enrichment.get("source")
        timestamp = enrichment.get("timestamp")
        if self._is_newer_timestamp(timestamp, finance_block.get("timestamp")):
            if timestamp:
                finance_block["timestamp"] = timestamp
            if source:
                finance_block["source"] = source
        for key, value in enrichment.get("metrics", {}).items():
            metrics_block[key] = value

    @staticmethod
    def _is_newer_timestamp(candidate: Any, existing: Any) -> bool:
        if candidate is None:
            return False
        if existing is None:
            return True
        try:
            candidate_dt = datetime.fromisoformat(candidate) if isinstance(candidate, str) else candidate
            existing_dt = datetime.fromisoformat(existing) if isinstance(existing, str) else existing
            if isinstance(candidate_dt, datetime) and isinstance(existing_dt, datetime):
                return candidate_dt > existing_dt
        except Exception:  # pragma: no cover - defensive
            return True
        return False

    def _build_metrics_context(
        self,
        tool_result: ToolInvocationResult | None,
        enrichment: Mapping[str, Any] | None,
        resolved_metrics: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not resolved_metrics:
            return None
        source = None
        timestamp = None
        if enrichment and enrichment.get("metrics"):
            source = enrichment.get("source")
            timestamp = enrichment.get("timestamp")
        if not source and tool_result is not None:
            source = tool_result.resolved_tool or tool_result.tool
        context_block = {"source": source, "timestamp": timestamp}
        if not source and not timestamp:
            return None
        return context_block

    def _format_enrichment_section(self, enrichment: dict[str, Any] | None) -> str:
        if enrichment is None:
            return "(supplemental metrics not requested)"
        metrics = enrichment.get("metrics") if isinstance(enrichment, Mapping) else None
        missing = enrichment.get("missing") if isinstance(enrichment, Mapping) else None
        reason = enrichment.get("reason") if isinstance(enrichment, Mapping) else None
        if isinstance(metrics, Mapping) and metrics:
            lines = [f"- {key}: {self._stringify_metric(value)}" for key, value in metrics.items()]
            source = enrichment.get("source") if isinstance(enrichment, Mapping) else None
            if source:
                lines.append(f"source: {source}")
            timestamp = enrichment.get("timestamp") if isinstance(enrichment, Mapping) else None
            if timestamp:
                lines.append(f"timestamp: {timestamp}")
            if missing:
                lines.append(f"missing: {', '.join(missing)}")
            return "\n".join(lines)
        if missing:
            line = f"missing: {', '.join(missing)}"
            if reason:
                return f"{line} (reason: {reason})"
            return line
        return reason or "(no supplemental metrics available)"

    @staticmethod
    def _stringify_metric(value: Any) -> str:
        if isinstance(value, (int, float)):
            if abs(value) >= 1_000_000_000_000:
                return f"{value / 1_000_000_000_000:.2f}T"
            if abs(value) >= 1_000_000_000:
                return f"{value / 1_000_000_000:.2f}B"
            if abs(value) >= 1_000_000:
                return f"{value / 1_000_000:.2f}M"
            return f"{value:,.2f}"
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

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

    def _format_tool_response(
        self,
        tool_result: ToolInvocationResult | None,
        resolved_metrics: Mapping[str, Any] | None = None,
        metrics_context: Mapping[str, Any] | None = None,
    ) -> str:
        if tool_result is None:
            base_lines = ["(tool unavailable)"]
        else:
            items = self._extract_tool_items(tool_result)
            if not items:
                payload = tool_result.response
                try:
                    serialized = json.dumps(payload, indent=2, sort_keys=True)
                except TypeError:
                    serialized = str(payload)
                base_lines = [serialized]
            else:
                base_lines = []
                tool_name = (tool_result.tool or "").lower()
                
                # Check if this is arXiv results - format with full academic citation
                is_arxiv = "arxiv" in tool_name
                
                for i, item in enumerate(items, 1):
                    if is_arxiv:
                        # Format arXiv papers with full citation
                        title = item.get("title", "Untitled")
                        authors_list = item.get("authors", [])
                        if isinstance(authors_list, list):
                            author_names = [a.get("name", "") if isinstance(a, dict) else str(a) for a in authors_list[:3]]
                            authors = ", ".join(author_names)
                            if len(authors_list) > 3:
                                authors += " et al."
                        else:
                            authors = "Unknown authors"
                        
                        published = item.get("published")
                        if published:
                            if isinstance(published, str):
                                year = published[:4]
                            else:
                                year = str(published)[:4]
                        else:
                            year = "n.d."
                        
                        pdf_url = item.get("pdf_url") or item.get("identifier", "")
                        summary = (item.get("summary") or "")[:300]
                        if len(item.get("summary", "")) > 300:
                            summary += "..."
                        category = item.get("primary_category", "")
                        
                        base_lines.append(f"\n**[{i}] {title}**")
                        base_lines.append(f"   Authors: {authors}")
                        base_lines.append(f"   Published: {year} | Category: {category}")
                        base_lines.append(f"   URL: {pdf_url}")
                        base_lines.append(f"   Abstract: {summary}")
                    else:
                        # Standard format for other tools
                        summary = item.get("summary") or item.get("title") or item.get("content", "(no summary)")
                        source = item.get("source") or item.get("url") or item.get("link")
                        line = f"- {summary}"
                        if source:
                            line = f"{line} [{source}]"
                        base_lines.append(line)
                        
                # Add references section for arXiv
                if is_arxiv and items:
                    base_lines.append("\n---")
                    base_lines.append("**References:**")
                    for i, item in enumerate(items, 1):
                        title = item.get("title", "Untitled")
                        pdf_url = item.get("pdf_url") or item.get("identifier", "")
                        base_lines.append(f"[{i}] {title} - {pdf_url}")
                        
        if resolved_metrics:
            base_lines.append("Finance metrics:")
            for key, value in resolved_metrics.items():
                base_lines.append(f"- {key}: {self._stringify_metric(value)}")
            if metrics_context:
                source = metrics_context.get("source")
                timestamp = metrics_context.get("timestamp")
                if source:
                    base_lines.append(f"  source: {source}")
                if timestamp:
                    base_lines.append(f"  timestamp: {timestamp}")
        return "\n".join(base_lines)

    @staticmethod
    def _extract_tool_items(tool_result: ToolInvocationResult) -> list[dict[str, Any]]:
        response = tool_result.response
        if isinstance(response, Mapping):
            results = response.get("results")
            if isinstance(results, list):
                return [item for item in results if isinstance(item, dict)]
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
