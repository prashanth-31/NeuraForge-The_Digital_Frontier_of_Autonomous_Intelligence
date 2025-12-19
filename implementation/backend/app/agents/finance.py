from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
import math
from typing import Any, Mapping, Sequence

from ..agents.base import AgentContext, ReasoningBuilder
from ..core.logging import get_logger
from ..core.metrics import observe_confidence_component
from ..mcp.symbols import extract_symbols_from_text
from ..schemas.agents import AgentCapability, AgentExchange, AgentInput, AgentOutput, ReasoningStepType
from ..services.llm import is_llm_unavailable
from ..services.tools import ToolDisabledError, ToolInvocationError, ToolInvocationResult

logger = get_logger(name=__name__)


@dataclass
class FinanceAgent:
    name: str = "finance_agent"
    capability: AgentCapability = AgentCapability.FINANCE
    system_prompt: str = (
        "You are NeuraForge's Finance Agent - a professional financial analyst. "
        "Provide clear, data-driven financial analysis using the tool metrics provided. "
        "When live tool metrics are available, treat them as the source of truth and cite their timestamps. "
        "Structure your response with: 1) Headline Metrics, 2) Key Observations, 3) Market Context, 4) Analysis Summary. "
        "Use professional financial language. Format numbers clearly (e.g., $451.45, 63.2M shares). "
        "The finance tools provide real-time quotes, fundamentals data (PE ratio, revenue, EPS, etc.), "
        "and up to 30 days of historical daily prices."
    )
    description: str = "Performs financial analysis, forecasting, and headline metric synthesis."
    tool_preference: list[str] = field(default_factory=lambda: ["finance.snapshot"])
    tool_candidates: tuple[str, ...] = (
        # Primary market data (free/open APIs)
        "finance.snapshot",           # Yahoo Finance - free
        "finance.snapshot.alpha",     # Alpha Vantage - free tier
        "finance.snapshot.cached",    # Stooq fallback - free
        # Analysis & visualization
        "finance.plot",               # Matplotlib charting
        "finance.analytics",          # Pandas analytics
        "finance/pandas",             # Data analysis
        "finance/csv_analyzer",       # CSV processing
        "finance/sentiment",          # FinBERT sentiment (HuggingFace - open source)
        # Cross-domain research support
        "research.search",            # DuckDuckGo for news context
        "research.wikipedia",         # Company background info
    )
    tool_retry_config: dict[str, tuple[int, float]] = field(
        default_factory=lambda: {
            "finance.snapshot": (3, 0.75),
            "finance.snapshot.alpha": (2, 1.0),
            "finance.snapshot.cached": (1, 0.0),
            "finance.plot": (1, 0.0),
            "finance/sentiment": (1, 0.0),
            "research.search": (2, 0.5),
        }
    )
    fallback_agent: str | None = "enterprise_agent"
    confidence_bias: float = 0.9

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("finance_agent_task", task=task.model_dump())
        
        # Initialize reasoning builder for tracking thought process
        reasoning = ReasoningBuilder(agent_name=self.name, context=context)
        
        # Check if this task is actually relevant to finance agent
        if not self._is_relevant_task(task):
            logger.info("finance_agent_pass", reason="not_a_finance_task", prompt=task.prompt[:100] if task.prompt else "")
            await reasoning.think(
                "Analyzed request - not relevant to financial data/analysis, passing to other agents",
                step_type=ReasoningStepType.EVALUATION,
            )
            return AgentOutput(
                agent=self.name,
                capability=self.capability,
                summary="[PASS]",  # Non-empty placeholder to pass validation
                confidence=0.0,  # Zero confidence signals to skip this response
                rationale="Task not relevant to finance agent - passing.",
                metadata={"passed": True, "reason": "not_a_finance_task"},
            )
        
        # Step 1: Analyze the incoming request
        await reasoning.think(
            f"Received financial analysis request: '{task.prompt[:100]}...'",
            step_type=ReasoningStepType.OBSERVATION,
        )
        
        logger.info(
            "finance_agent_context",
            planned_tools=context.planned_tools,
            fallback_tools=context.fallback_tools,
        )
        
        # Wrap main logic in try-except to prevent exceptions from triggering replan
        # This ensures we return a fallback response instead of raising
        try:
            return await self._execute_financial_analysis(task, context, reasoning)
        except Exception as exc:
            logger.error(
                "finance_agent_exception",
                error=str(exc),
                task_id=task.task_id,
                exc_info=True,
            )
            # Return a fallback response instead of raising
            fallback_response = self._generate_fallback_response(task, None, None)
            return AgentOutput(
                agent=self.name,
                capability=self.capability,
                summary=fallback_response,
                confidence=0.4,  # Lower confidence for fallback
                rationale="Generated rule-based response due to internal error.",
                metadata={"fallback": True, "error_recovered": True},
            )

    async def _execute_financial_analysis(
        self,
        task: AgentInput,
        context: AgentContext,
        reasoning: ReasoningBuilder,
    ) -> AgentOutput:
        """Execute the main financial analysis logic. Extracted for try-except wrapping."""
        # Step 2: Build context
        context_section = task.context
        if context_section is None and context.context is not None:
            await reasoning.think(
                "Assembling financial context from knowledge base",
                step_type=ReasoningStepType.ANALYSIS,
            )
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()
            if context_section:
                await reasoning.add_finding(
                    claim="Retrieved relevant financial context",
                    evidence=[context_section[:200] + "..."] if len(context_section) > 200 else [context_section],
                    confidence=0.7,
                    source="context_assembler",
                )
        
        # Step 3: Extract symbols and invoke tools
        symbols = extract_symbols_from_text(task.prompt or "")
        if symbols:
            await reasoning.think(
                f"Detected ticker symbols: {', '.join(symbols)} - will fetch live market data",
                step_type=ReasoningStepType.ANALYSIS,
                evidence=f"Symbols found: {symbols}",
            )
        
        tool_result, tool_trace = await self._maybe_invoke_tool(task, context=context)
        if tool_result is not None:
            await reasoning.consider_tool(
                tool_name=tool_result.tool,
                reason="Financial data tool for market metrics",
                selected=True,
            )
            metrics = self._extract_tool_metrics(tool_result)
            if metrics:
                # metrics is a list of dicts; format each dict as evidence
                evidence_strs = []
                for m in metrics[:5]:
                    if isinstance(m, dict):
                        evidence_strs.extend(f"{k}: {v}" for k, v in list(m.items())[:3])
                await reasoning.add_finding(
                    claim=f"Retrieved live market data with {len(metrics)} data points",
                    evidence=evidence_strs[:10],
                    confidence=0.9,
                    source=tool_result.tool,
                )
        elif context.planned_tools:
            await reasoning.consider_tool(
                tool_name=context.planned_tools[0] if context.planned_tools else "finance.snapshot",
                reason="Tool was planned but not invoked",
                selected=False,
                rejection_reason="No tool result obtained",
            )
        
        # Step 4: Generate plot if applicable
        plot_result = await self._maybe_generate_plot(
            task,
            context=context,
            snapshot_result=tool_result,
            tool_trace=tool_trace,
        )
        if plot_result is not None:
            await reasoning.consider_tool(
                tool_name="finance.plot",
                reason="Generated price chart visualization",
                selected=True,
            )
            await reasoning.add_finding(
                claim="Generated price chart visualization",
                confidence=0.85,
                source="finance.plot",
            )
        
        # Step 5: Generate analysis
        prompt = self._build_prompt(task, context_section=context_section, tool_result=tool_result)
        
        await reasoning.think(
            "Synthesizing financial analysis from market data and context",
            step_type=ReasoningStepType.SYNTHESIS,
        )
        
        forecast = await context.llm.generate(prompt=prompt, system_prompt=self.system_prompt, temperature=0.2)

        # Handle LLM unavailability - generate fallback response from tool data
        if is_llm_unavailable(forecast):
            logger.warning("finance_agent_llm_unavailable", task_id=task.task_id)
            forecast = self._generate_fallback_response(task, tool_result, context_section)
            await reasoning.note_uncertainty(
                "LLM temporarily unavailable - providing rule-based financial analysis"
            )

        # Step 6: Calculate confidence
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

        await reasoning.think(
            f"Calculated confidence: {confidence:.2f} based on {evidence_count} evidence sources",
            step_type=ReasoningStepType.EVALUATION,
            confidence=confidence,
        )

        # Step 7: Note any uncertainties
        if not tool_result:
            await reasoning.note_uncertainty(
                "No live market data available - analysis based on context only"
            )
        if confidence < 0.6:
            await reasoning.note_uncertainty(
                "Lower confidence - market conditions may require additional verification"
            )

        metadata: dict[str, Any] = {"type": "financial_forecast"}
        tools_metadata: list[dict[str, Any]] = []
        if tool_result is not None:
            primary_tool_meta = self._tool_metadata(tool_result)
            metadata["tool"] = primary_tool_meta
            tools_metadata.append(primary_tool_meta)
        if plot_result is not None:
            plot_meta = self._tool_metadata(plot_result)
            tools_metadata.append(plot_meta)
            metadata.setdefault("visualizations", []).append(
                {
                    "tool": plot_result.tool,
                    "title": plot_result.response.get("title"),
                    "mime_type": plot_result.response.get("mime_type"),
                    "image_base64": plot_result.response.get("image_base64"),
                    "legend": plot_result.response.get("legend"),
                    "points": plot_result.response.get("points"),
                }
            )
        if tools_metadata:
            metadata.setdefault("tools_used", tools_metadata)
        if tool_trace:
            metadata["tool_attempts"] = tool_trace
            metadata["tool_fallback_used"] = any(
                attempt.get("status") != "success" for attempt in tool_trace[:-1]
            )
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown
        override_payload = self._tool_policy_override(
            tool_result=tool_result,
            tool_trace=tool_trace,
            tools=context.tools,
            planned_tools=context.planned_tools,
            fallback_tools=context.fallback_tools,
            task_metadata=task.metadata if isinstance(task.metadata, dict) else None,
        )
        logger.info(
            "finance_agent_override_payload",
            has_override=bool(override_payload),
            override_reason=(override_payload or {}).get("reason"),
            planned_tools=context.planned_tools,
            fallback_tools=context.fallback_tools,
        )
        if override_payload is not None:
            metadata["tool_policy_override"] = override_payload

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=forecast,
            confidence=confidence,
            rationale="Financial forecast generated from context and prior agent signals.",
            metadata=metadata,
            # Include reasoning transparency
            reasoning_steps=reasoning.steps,
            key_findings=reasoning.findings,
            tools_considered=reasoning.tools_considered,
            uncertainties=reasoning.uncertainties,
        )

    def _is_meta_prompt(self, prompt: str) -> bool:
        """
        Detect if the prompt is a meta-prompt (role description/instruction)
        rather than an actual query to be processed.
        
        Meta-prompts like "You are FinResearch Agent, responsible for..." are
        instructions describing what the agent should be, not actual queries.
        """
        normalized = (prompt or "").lower().strip()
        if not normalized:
            return False
        
        # Meta-prompt patterns (role descriptions, instructions)
        meta_patterns = [
            r'^you are\s+\w+\s*(agent|assistant)',  # "You are X Agent..."
            r'^act as\s+',  # "Act as a..."
            r'^assume the role',  # "Assume the role of..."
            r'^your role is',  # "Your role is..."
            r'^you should be',  # "You should be..."
            r'^behave as',  # "Behave as..."
            r'^pretend you are',  # "Pretend you are..."
            r'^i want you to act',  # "I want you to act..."
            r'^from now on',  # "From now on, you are..."
        ]
        
        for pattern in meta_patterns:
            if re.search(pattern, normalized):
                # Check if this contains ONLY instructions without an actual query
                # If it has "first analyze", "then perform", etc., it's still just instructions
                instruction_only_indicators = [
                    "assume missing details",
                    "state all assumptions",
                    "present reasoning step",
                    "conclude with",
                    "structure your response",
                    "follow these guidelines",
                ]
                if any(indicator in normalized for indicator in instruction_only_indicators):
                    logger.info(
                        "finance_agent_meta_prompt_detected",
                        prompt_preview=normalized[:100],
                        pattern=pattern,
                    )
                    return True
        
        return False

    def _is_relevant_task(self, task: AgentInput) -> bool:
        """Check if this task is relevant to finance agent - handles ALL financial matters."""
        prompt = (task.prompt or "").lower()
        
        # ═══════════════════════════════════════════════════════════════════════
        # FIRST: Check for meta-prompts (role descriptions) - these need special handling
        # Even if planner selected us, we should not process pure instructions as queries
        # ═══════════════════════════════════════════════════════════════════════
        if self._is_meta_prompt(task.prompt):
            logger.info(
                "finance_agent_reject_meta_prompt",
                reason="prompt_is_role_description_not_query",
                prompt_preview=prompt[:100] if prompt else "",
            )
            return False
        
        # ═══════════════════════════════════════════════════════════════════════
        # SECOND: If planner explicitly selected this agent, trust it
        # ═══════════════════════════════════════════════════════════════════════
        metadata = task.metadata or {}
        shared_ctx = metadata.get("_shared_context", {})
        planner_data = shared_ctx.get("planner", {})
        selected_agents = planner_data.get("selected_agents", [])
        planner_steps = shared_ctx.get("planner_steps", [])
        
        # If planner explicitly selected finance_agent, accept the task
        if "finance_agent" in selected_agents:
            logger.debug("finance_agent_accept", reason="planner_selected", agents=selected_agents)
            return True
        
        # Check if any planner step is for finance_agent
        for step in planner_steps:
            if step.get("planned_agent") == "finance_agent" or step.get("executed_agent") == "finance_agent":
                logger.debug("finance_agent_accept", reason="planner_step_assigned")
                return True
        
        # ═══════════════════════════════════════════════════════════════════════
        # QUICK REJECTIONS - Things finance agent should NEVER handle
        # Use word boundary checks to avoid false positives (e.g., "hi" in "highlight")
        # ═══════════════════════════════════════════════════════════════════════
        immediate_reject_phrases = {
            # Meta questions (full phrases only)
            "what are your capabilities", "what can you do", "what do you do",
            "how can you help", "tell me about yourself", "who are you",
            "what features", "your capabilities",
            # Creative
            "write a poem", "write a story", "creative writing",
        }
        if any(reject in prompt for reject in immediate_reject_phrases):
            return False
        
        # Single-word rejections need word boundary check
        single_word_rejects = ["help", "hello", "hi", "greetings", "hey"]
        for word in single_word_rejects:
            # Use word boundary to avoid matching "hi" in "highlight"
            if re.search(rf'\b{word}\b', prompt):
                return False
        
        # ═══════════════════════════════════════════════════════════════════════
        # FINANCE TRIGGERS - Any financial topic should be handled by finance_agent
        # ═══════════════════════════════════════════════════════════════════════
        finance_triggers = {
            # General finance terms
            "financial", "finance", "money", "monetary",
            # Investment
            "investment", "invest", "investing", "portfolio", "allocation",
            "diversification", "diversify", "asset allocation",
            # Personal finance
            "salary", "income", "savings", "budget", "budgeting",
            "retirement", "pension", "401k", "ira", "roth",
            "tax", "taxes", "deduction", "exemption",
            "lpa", "lakhs", "crores",  # Indian salary context
            # Market data
            "stock", "stocks", "share", "shares", "equity", "equities",
            "ticker", "market cap", "pe ratio", "eps", "revenue",
            "earnings", "dividend", "yield", "valuation",
            "bull", "bear", "market",
            # Indices
            "nifty", "sensex", "dow", "nasdaq", "s&p",
            # Analysis
            "financial analysis", "financial health", "financial assessment",
            "profitability", "liquidity", "solvency",
            "cash flow", "burn rate", "runway",
            "balance sheet", "income statement", "profit and loss",
            "financial ratio", "debt ratio", "current ratio",
            "roa", "roe", "roce", "return on",
            # Planning
            "financial planning", "financial advice", "financial advise",
            "investment strategy", "investment advice",
            "where to invest", "how to invest", "best investment",
            "wealth", "wealth management", "wealth creation",
            # Risk
            "risk", "volatility", "hedging", "hedge",
            # Banking/Credit
            "loan", "emi", "interest rate", "credit", "debt",
            "mortgage", "insurance", "premium",
            # Crypto (if applicable)
            "crypto", "bitcoin", "ethereum", "cryptocurrency",
        }
        
        # ═══════════════════════════════════════════════════════════════════════
        # Non-finance indicators - CHECK FIRST to reject business strategy tasks
        # These tasks may contain "revenue", "growth" but are NOT finance tasks
        # ═══════════════════════════════════════════════════════════════════════
        non_finance_indicators = {
            # Business strategy (not finance analysis)
            "business proposal", "create a proposal", "write a proposal",
            "pitch deck", "executive summary", "company background",
            "business strategy", "startup pitch", "business idea",
            "college project", "hackathon", "value proposition",
            "target customers", "growth strategy", "revenue model",
            "ai business strategy", "business strategy agent",
            # Creative tasks
            "poem", "story", "creative", "love story", "shakespeare",
            "write a story", "write a poem", "compose",
        }
        
        # Check non-finance FIRST - reject if this is a business strategy/creative task
        if any(indicator in prompt for indicator in non_finance_indicators):
            logger.debug("finance_agent_reject", reason="non_finance_indicator_found", prompt=prompt[:100])
            return False
        
        # Now check finance triggers
        if any(trigger in prompt for trigger in finance_triggers):
            return True
        
        # Default to PASS for truly unrelated queries
        return False

    def _generate_fallback_response(
        self,
        task: AgentInput,
        tool_result: ToolInvocationResult | None,
        context_section: str | None,
    ) -> str:
        """Generate a rule-based fallback response when LLM is unavailable."""
        prompt = (task.prompt or "").lower()
        response_parts = []
        
        # Header
        response_parts.append("## Financial Analysis (Rule-Based)\n")
        
        # If we have tool data, format it
        if tool_result and tool_result.response:
            response_parts.append("### Market Data Summary\n")
            data = tool_result.response
            
            # Format stock data if available
            if isinstance(data, dict):
                if "currentPrice" in data:
                    response_parts.append(f"- **Current Price**: ${data.get('currentPrice', 'N/A')}")
                if "marketCap" in data:
                    mc = data.get("marketCap", 0)
                    if mc > 1e12:
                        response_parts.append(f"- **Market Cap**: ${mc/1e12:.2f}T")
                    elif mc > 1e9:
                        response_parts.append(f"- **Market Cap**: ${mc/1e9:.2f}B")
                    else:
                        response_parts.append(f"- **Market Cap**: ${mc/1e6:.2f}M")
                if "volume" in data:
                    response_parts.append(f"- **Volume**: {data.get('volume', 'N/A'):,}")
                if "52WeekHigh" in data:
                    response_parts.append(f"- **52-Week High**: ${data.get('52WeekHigh', 'N/A')}")
                if "52WeekLow" in data:
                    response_parts.append(f"- **52-Week Low**: ${data.get('52WeekLow', 'N/A')}")
            response_parts.append("")
        
        # Provide general advice based on query type
        if any(k in prompt for k in ["salary", "income", "lpa", "lakhs"]):
            response_parts.append("### General Investment Guidelines\n")
            response_parts.append("Based on standard financial planning principles:\n")
            response_parts.append("1. **Emergency Fund**: Maintain 6-12 months of expenses in liquid savings")
            response_parts.append("2. **50-30-20 Rule**: 50% needs, 30% wants, 20% savings/investments")
            response_parts.append("3. **Diversification**: Spread investments across asset classes")
            response_parts.append("4. **Tax Planning**: Maximize tax-advantaged accounts (PPF, ELSS, NPS)")
            response_parts.append("5. **Regular Review**: Rebalance portfolio quarterly\n")
        
        if any(k in prompt for k in ["investment", "invest", "portfolio"]):
            response_parts.append("### Investment Considerations\n")
            response_parts.append("- Risk tolerance varies by individual")
            response_parts.append("- Time horizon affects asset allocation")
            response_parts.append("- Consider consulting a certified financial advisor")
            response_parts.append("- Past performance doesn't guarantee future results\n")
        
        if context_section:
            response_parts.append("### Additional Context\n")
            response_parts.append(f"{context_section[:500]}...")
        
        # Disclaimer
        response_parts.append("\n---\n*Note: This is a rule-based response as the LLM is temporarily unavailable. For detailed analysis, please try again later.*")
        
        return "\n".join(response_parts)

    def _build_prompt(
        self,
        task: AgentInput,
        *,
        context_section: str | None = None,
        tool_result: ToolInvocationResult | None = None,
    ) -> str:
        metadata_clean = self._filtered_metadata(task.metadata)
        metadata_json = json.dumps(metadata_clean, indent=2, sort_keys=True)
        prior = _collect_prior_outputs(task.prior_exchanges)
        retrieved = context_section or "(no retrieved context)"
        tool_data = self._format_tool_metrics(tool_result)
        tool_json = self._format_tool_json(tool_result)
        recency_anchor = self._format_tool_recency(tool_result)
        handoff = self._format_handoff(task.metadata)
        return (
            f"Business question:\n{task.prompt}\n\n"
            f"Financial metadata:\n{metadata_json}\n\n"
            f"Shared context from peers:\n{handoff}\n\n"
            f"Earlier agent signals:\n{prior}\n\n"
            f"Retrieved context:\n{retrieved}\n\n"
            f"Tool recency anchor:\n{recency_anchor}\n\n"
            f"Tool metrics summary:\n{tool_data}\n\n"
            f"Raw tool metrics JSON (primary source, prefer over model priors):\n{tool_json}\n\n"
            "Estimate revenue or cost implications, surface 2-3 headline metrics, and outline immediate actions."
        )

    async def _maybe_invoke_tool(
        self,
        task: AgentInput,
        *,
        context: AgentContext,
    ) -> tuple[ToolInvocationResult | None, list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        if context.tools is None:
            return None, attempts

        payload = self._build_snapshot_payload(task)
        news_payload = self._build_news_search_payload(task)

        tool_queue: list[tuple[str, dict[str, Any]]] = []
        if payload is not None:
            tool_queue.extend((alias, payload) for alias in self.tool_candidates)
        # Use DuckDuckGo search for news instead of finance.news (which requires API key)
        if news_payload is not None:
            tool_queue.append(("research.search", news_payload))

        for alias, candidate_payload in tool_queue:
            result = await self._invoke_finance_tool(context, alias, candidate_payload, attempts)
            if result is not None:
                return result, attempts

        return None, attempts

    async def _invoke_finance_tool(
        self,
        context: AgentContext,
        alias: str,
        payload: dict[str, Any],
        attempts: list[dict[str, Any]],
    ) -> ToolInvocationResult | None:
        max_attempts, base_delay = self.tool_retry_config.get(alias, (1, 0.5))
        delay = base_delay
        for attempt in range(1, max_attempts + 1):
            try:
                result = await context.tools.invoke(alias, payload)
            except ToolDisabledError as exc:
                attempts.append({"tool": alias, "status": "disabled", "error": str(exc)})
                return None
            except ToolInvocationError as exc:
                attempts.append(
                    {
                        "tool": alias,
                        "status": "error",
                        "error": str(exc),
                        "attempt": attempt,
                    }
                )
                should_retry = attempt < max_attempts and self._should_retry(alias, exc)
                if should_retry:
                    sleep_for = max(delay, 0.2)
                    await asyncio.sleep(min(sleep_for, 6.0))
                    delay = delay * 1.5 if delay else base_delay or 0.5
                    continue
                if delay:
                    await asyncio.sleep(min(delay, 1.0))
                return None
            else:
                attempts.append({"tool": alias, "status": "success", "attempt": attempt})
                return result
        return None

    def _should_retry(self, alias: str, error: Exception) -> bool:
        if alias not in {"finance.snapshot", "finance.snapshot.alpha"}:
            return False
        message = str(error).lower()
        retry_keywords = (
            "429",
            "too many requests",
            "timeout",
            "circuit breaker",
            "http failure",
        )
        return any(keyword in message for keyword in retry_keywords)

    def _tool_policy_override(
        self,
        *,
        tool_result: ToolInvocationResult | None,
        tool_trace: list[dict[str, Any]],
        tools: Any,
        planned_tools: list[str] | None = None,
        fallback_tools: list[str] | None = None,
        task_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        planned = tuple(tool for tool in (planned_tools or []) if isinstance(tool, str))
        fallback = tuple(tool for tool in (fallback_tools or []) if isinstance(tool, str))
        if (not planned or not fallback) and task_metadata:
            derived_planned, derived_fallback = self._extract_planner_tools_from_metadata(task_metadata)
            if not planned and derived_planned:
                planned = tuple(derived_planned)
            if not fallback and derived_fallback:
                fallback = tuple(derived_fallback)
        planner_expected = bool(planned or fallback)

        if tool_result is None:
            payload: dict[str, Any] = {"allow_skip": True, "attempts": len(tool_trace)}
            if tools is None:
                payload["reason"] = "tool_service_unavailable"
            elif tool_trace:
                payload["reason"] = "tool_attempts_exhausted"
            else:
                payload["reason"] = "tool_payload_missing"
            if planner_expected and planned:
                payload["requested_tools"] = list(planned)
            last_error = next((attempt.get("error") for attempt in reversed(tool_trace) if attempt.get("error")), None)
            if last_error:
                payload["last_error"] = last_error
            return payload

        if not planner_expected:
            return None

        success_aliases = [entry.get("tool") for entry in tool_trace if entry.get("status") == "success"]
        planned_success = any(alias in planned for alias in success_aliases if alias)
        fallback_success = any(alias in fallback for alias in success_aliases if alias)
        if planned_success or fallback_success:
            return None

        supported_aliases = set(self.tool_candidates) | {"finance.news", "finance.plot"}
        unsupported_planned = [tool for tool in planned if tool not in supported_aliases]
        unsupported_fallback = [tool for tool in fallback if tool not in supported_aliases]
        failed_planned = [
            attempt
            for attempt in tool_trace
            if attempt.get("tool") in planned and attempt.get("status") != "success"
        ]
        payload: dict[str, Any] = {"allow_skip": True, "attempts": len(tool_trace)}
        success_alias = getattr(tool_result, "tool", None)

        if unsupported_planned or unsupported_fallback:
            payload["reason"] = "planner_tool_unsupported"
            if unsupported_planned:
                payload["unsupported_planned"] = unsupported_planned
            if unsupported_fallback:
                payload["unsupported_fallback"] = unsupported_fallback
        elif failed_planned:
            payload["reason"] = "planner_tool_outage"
            payload["failed_planned"] = [
                {
                    "tool": attempt.get("tool"),
                    "status": attempt.get("status"),
                    "error": attempt.get("error"),
                }
                for attempt in failed_planned
            ]
        elif success_alias and success_alias not in planned and success_alias not in fallback:
            payload["reason"] = "unplanned_tool_success"
            payload["replacement_tool"] = success_alias
        else:
            return None

        if success_alias:
            payload["tool_used"] = success_alias
        last_error = next((attempt.get("error") for attempt in reversed(tool_trace) if attempt.get("error")), None)
        if last_error:
            payload.setdefault("last_error", last_error)
        if planned:
            payload.setdefault("requested_tools", list(planned))
        return payload

    async def _maybe_generate_plot(
        self,
        task: AgentInput,
        *,
        context: AgentContext,
        snapshot_result: ToolInvocationResult | None,
        tool_trace: list[dict[str, Any]],
    ) -> ToolInvocationResult | None:
        alias = "finance.plot"
        metadata = task.metadata if isinstance(task.metadata, Mapping) else None
        if not self._planner_requested_plot(
            planned_tools=context.planned_tools,
            fallback_tools=context.fallback_tools,
            task_metadata=metadata,
        ):
            return None
        if context.tools is None:
            tool_trace.append({"tool": alias, "status": "skipped", "reason": "tool_service_unavailable"})
            return None
        if snapshot_result is None:
            tool_trace.append({"tool": alias, "status": "skipped", "reason": "snapshot_unavailable"})
            return None
        payload = self._build_plot_payload(snapshot_result)
        if payload is None:
            tool_trace.append({"tool": alias, "status": "skipped", "reason": "insufficient_plot_data"})
            return None
        return await self._invoke_finance_tool(context, alias, payload, tool_trace)

    def _planner_requested_plot(
        self,
        *,
        planned_tools: list[str] | None,
        fallback_tools: list[str] | None,
        task_metadata: Mapping[str, Any] | None,
    ) -> bool:
        variants = self._normalize_tools(planned_tools) | self._normalize_tools(fallback_tools)
        derived_planned, derived_fallback = self._extract_planner_tools_from_metadata(task_metadata)
        variants |= self._normalize_tools(derived_planned)
        variants |= self._normalize_tools(derived_fallback)
        return "finance.plot" in variants or "finance/plot" in variants

    def _build_plot_payload(self, snapshot_result: ToolInvocationResult) -> dict[str, Any] | None:
        metrics = self._extract_tool_metrics(snapshot_result)
        if not metrics:
            return None
        primary = metrics[0]
        fundamentals = primary.get("fundamentals")
        if not isinstance(fundamentals, Mapping):
            return None
        quarterly = fundamentals.get("quarterly")
        if not isinstance(quarterly, list):
            return None
        revenue_points = self._collect_quarterly_points(quarterly, "total_revenue")
        income_points = self._collect_quarterly_points(quarterly, "net_income")
        series: list[dict[str, Any]] = []
        if len(revenue_points) >= 2:
            series.append({"name": "Revenue (B)", "points": revenue_points})
        if len(income_points) >= 2:
            series.append({"name": "Net Income (B)", "points": income_points})
        if not series:
            return None
        title = primary.get("company_name") or primary.get("symbol") or "Finance Plot"
        return {
            "title": f"{title} quarterly trends"[:120],
            "y_label": "USD (billions)",
            "x_label": "Quarter",
            "series": series,
        }

    def _collect_quarterly_points(self, entries: Sequence[Any], key: str) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        filtered = [entry for entry in entries if isinstance(entry, Mapping)]
        trimmed = filtered[:6]
        for idx, entry in enumerate(reversed(trimmed)):
            value = self._safe_number(entry.get(key))
            if value is None:
                continue
            label = self._extract_period_label(entry, idx)
            points.append({"x": label, "y": value / 1_000_000_000})
        return points

    @staticmethod
    def _safe_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    @staticmethod
    def _extract_period_label(entry: Mapping[str, Any], fallback_index: int) -> str:
        for key in ("period", "end_date", "date"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"T-{fallback_index}"

    @staticmethod
    def _normalize_tools(tools: Sequence[str] | None) -> set[str]:
        normalized: set[str] = set()
        if not tools:
            return normalized
        for tool in tools:
            if not isinstance(tool, str):
                continue
            candidate = tool.strip()
            if not candidate:
                continue
            normalized.add(candidate)
            if "/" in candidate:
                normalized.add(candidate.replace("/", "."))
            if "." in candidate:
                normalized.add(candidate.replace(".", "/"))
        return normalized

    def _extract_planner_tools_from_metadata(
        self, metadata: Mapping[str, Any] | None
    ) -> tuple[list[str], list[str]]:
        if not isinstance(metadata, Mapping):
            return [], []

        def _normalize(value: Any) -> list[str]:
            if isinstance(value, str):
                return [token.strip() for token in value.split() if token.strip()]
            if isinstance(value, Sequence):
                cleaned: list[str] = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        cleaned.append(item.strip())
                return cleaned
            return []

        planned: list[str] = []
        fallback: list[str] = []

        candidates: list[Mapping[str, Any]] = []
        planner_step = metadata.get("planner_step")
        if isinstance(planner_step, Mapping):
            candidates.append(planner_step)
        shared_context = metadata.get("_shared_context")
        if isinstance(shared_context, Mapping):
            planner_steps = shared_context.get("planner_steps")
            if isinstance(planner_steps, Sequence):
                for entry in reversed(planner_steps):
                    if isinstance(entry, Mapping) and entry.get("planned_agent") == self.name:
                        candidates.append(entry)
                        break

        for candidate in candidates:
            planned = _normalize(candidate.get("executed_tools") or candidate.get("planned_tools")) or planned
            fallback = _normalize(candidate.get("planned_fallback_tools") or candidate.get("executed_fallback_tools")) or fallback
            if planned:
                break

        return planned, fallback

    def _build_news_search_payload(self, task: AgentInput) -> dict[str, Any] | None:
        """Build a DuckDuckGo search payload for financial news."""
        metadata: dict[str, Any] = task.metadata if isinstance(task.metadata, dict) else {}
        
        # Extract symbols/tickers for targeted news search
        symbols = self._extract_symbols(metadata)
        if not symbols:
            symbols.extend(self._infer_symbols_from_prompt(task.prompt))
        
        # Build search query
        prompt_text = (task.prompt or "").strip()
        if symbols:
            # Search for news about specific symbols
            query = f"{' '.join(symbols[:3])} stock news financial analysis"
        elif prompt_text:
            # Generic financial news search based on prompt
            query = f"{prompt_text[:100]} financial news"
        else:
            return None
        
        return {"query": query}

    def _build_news_payload(self, task: AgentInput) -> dict[str, Any] | None:
        """Legacy method - kept for compatibility."""
        return self._build_news_search_payload(task)

    def _build_snapshot_payload(self, task: AgentInput) -> dict[str, Any] | None:
        """
        Build payload for finance snapshot tool.
        
        IMPORTANT: Only returns a payload if we have explicit symbols/tickers to look up.
        We DO NOT try to derive company names from vague prompts - this prevents
        unnecessary stock data lookups for generic questions.
        """
        metadata: dict[str, Any] = task.metadata if isinstance(task.metadata, dict) else {}
        symbols = self._extract_symbols(metadata)
        if not symbols:
            symbols.extend(self._infer_symbols_from_prompt(task.prompt))

        unique_symbols = self._unique_upper(symbols)
        
        # Only build payload if we have actual symbols to look up
        # DO NOT fall back to company query - this causes unnecessary stock lookups
        if not unique_symbols:
            logger.debug(
                "finance_snapshot_payload_skipped",
                reason="no_symbols_found",
                prompt_preview=task.prompt[:100] if task.prompt else "",
            )
            return None
        
        payload: dict[str, Any] = {"symbols": unique_symbols[:5]}

        fields = metadata.get("fields")
        if isinstance(fields, list):
            cleaned = [str(field).strip() for field in fields if isinstance(field, str) and field.strip()]
            if cleaned:
                payload["fields"] = cleaned[:25]
        return payload

    def _extract_symbols(self, metadata: dict[str, Any]) -> list[str]:
        symbols: list[str] = []
        raw_symbols = metadata.get("symbols") or metadata.get("tickers")
        if isinstance(raw_symbols, list):
            symbols.extend(str(item).strip() for item in raw_symbols if isinstance(item, str) and item.strip())
        elif isinstance(raw_symbols, str) and raw_symbols.strip():
            symbols.append(raw_symbols.strip())

        for key in ("symbol", "ticker"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                symbols.append(value.strip())

        return symbols

    def _infer_symbols_from_prompt(self, prompt: str | None) -> list[str]:
        """
        Extract potential ticker symbols from prompt text.
        
        CONSERVATIVE approach: Only return symbols that are very likely to be tickers.
        We require the symbol to be in a financial context or from a known list.
        """
        if not prompt:
            return []
        
        # Use the centralized symbol extraction (more reliable)
        inferred = extract_symbols_from_text(prompt)
        unique: list[str] = []
        for symbol in inferred:
            if symbol and symbol not in unique:
                unique.append(symbol)
        
        # Common false positives to filter out
        false_positives = {
            "LPA", "CTC", "USD", "INR", "EUR", "GBP", "JPY", "AUD", "CAD",  # Salary/currency
            "ETF", "IPO", "ROI", "ROE", "ROA", "EPS", "CEO", "CFO", "CTO",  # Finance terms
            "FAQ", "API", "USA", "UK", "EU", "AI", "ML", "IT", "HR",  # Misc acronyms
            "MBA", "PHD", "BBA", "CA", "CPA", "CFA",  # Degrees
            "SIP", "PPF", "NPS", "FD", "RD", "MF",  # Investment products (India)
            "AM", "PM", "ID", "TM", "OR", "AND", "NOT", "BUT",  # Common words
        }
        
        # Filter out false positives
        unique = [s for s in unique if s.upper() not in false_positives]
        
        # Only do regex extraction if there's strong context suggesting a ticker lookup
        ticker_context_words = {"stock", "price", "ticker", "share", "shares", "quote", "trading"}
        prompt_lower = prompt.lower()
        has_ticker_context = any(word in prompt_lower for word in ticker_context_words)
        
        if has_ticker_context:
            # More aggressive extraction only when context clearly indicates ticker lookup
            ticker_candidates = re.findall(r"\b[A-Z]{2,5}\b", prompt)
            for candidate in ticker_candidates:
                if candidate not in unique and candidate not in false_positives:
                    unique.append(candidate)
        
        return unique

    def _derive_company_query(self, task: AgentInput, metadata: dict[str, Any]) -> str | None:
        candidates: list[str] = []
        meta_company = metadata.get("company") or metadata.get("entity")
        if isinstance(meta_company, str) and meta_company.strip():
            candidates.append(meta_company.strip())

        meta_companies = metadata.get("companies")
        if isinstance(meta_companies, list):
            candidates.extend(
                str(item).strip() for item in meta_companies if isinstance(item, str) and item.strip()
            )

        prompt_candidates = self._extract_company_candidates(task.prompt)
        candidates.extend(prompt_candidates)

        if candidates:
            return candidates[0][:120]

        prompt_text = (task.prompt or "").strip()
        return prompt_text[:120] if prompt_text else None

    def _extract_company_candidates(self, prompt: str | None) -> list[str]:
        if not prompt:
            return []
        stopwords = {
            "Financial",
            "Finance",
            "Outlook",
            "Review",
            "Update",
            "Analysis",
            "Overview",
            "Report",
            "Plan",
            "Strategy",
            "Investment",
            "Forecast",
            "Global",
            "Market",
        }
        matches = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", prompt)
        candidates: list[str] = []
        for match in matches:
            if match in stopwords:
                continue
            tokens = match.split()
            if all(token in stopwords for token in tokens):
                continue
            candidates.append(match)
        return candidates

    @staticmethod
    def _unique_upper(symbols: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for symbol in symbols:
            candidate = symbol.upper()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
        return ordered

    def _format_tool_metrics(self, tool_result: ToolInvocationResult | None) -> str:
        metrics = self._extract_tool_metrics(tool_result) if tool_result else []
        if not metrics:
            return "(tool metrics unavailable)"
        lines: list[str] = []
        for metric in metrics[:3]:
            symbol = metric.get("symbol") or "?"
            company = metric.get("company_name") or metric.get("long_name") or symbol
            price = metric.get("price") or metric.get("regularMarketPrice")
            change_pct = metric.get("change_percent") or metric.get("regularMarketChangePercent")
            change_value = metric.get("change") or metric.get("regularMarketChange")
            updated = metric.get("updated_at") or metric.get("regularMarketTime")
            previous_close = metric.get("previous_close") or metric.get("regularMarketPreviousClose")

            price_str = self._format_price(price, metric.get("currency"))
            change_parts: list[str] = []
            if change_value is not None:
                change_parts.append(self._format_price(change_value, metric.get("currency"), prefix="Δ"))
            if change_pct is not None:
                change_parts.append(f"{change_pct:+.2f}%")
            change_str = " ".join(change_parts) if change_parts else "flat"

            previous_str = self._format_price(previous_close, metric.get("currency")) if previous_close is not None else "n/a"
            timestamp = self._format_timestamp(updated)
            lines.append(f"- {company} ({symbol}): {price_str} ({change_str}) vs prev {previous_str} as of {timestamp}")
            fundamentals_summary = self._summarize_fundamentals(metric.get("fundamentals"), metric.get("currency"))
            if fundamentals_summary:
                lines.append(f"  {fundamentals_summary}")
            # Add historical price summary
            historical_summary = self._summarize_historical_prices(metric.get("historical_prices"), metric.get("currency"))
            if historical_summary:
                lines.append(f"  {historical_summary}")
        return "\n".join(lines)

    def _format_tool_json(self, tool_result: ToolInvocationResult | None) -> str:
        if tool_result is None or not isinstance(tool_result.response, dict):
            return "(no tool payload)"
        try:
            return json.dumps(tool_result.response, indent=2, sort_keys=True, default=str)
        except TypeError:
            return "(tool payload not serializable)"

    def _format_tool_recency(self, tool_result: ToolInvocationResult | None) -> str:
        if tool_result is None:
            return "Tool invocation unavailable"

        timestamps: list[datetime] = []
        for metric in self._extract_tool_metrics(tool_result):
            parsed = self._parse_timestamp(metric.get("updated_at"))
            if parsed:
                timestamps.append(parsed)

        response = tool_result.response if isinstance(tool_result.response, dict) else {}
        generated = self._parse_timestamp(response.get("generated_at"))
        if generated:
            timestamps.append(generated)

        if timestamps:
            latest = max(timestamps)
            return f"Latest tool data captured {latest.strftime('%Y-%m-%d %H:%M %Z')}"

        provider = getattr(tool_result, "resolved_tool", None) or tool_result.tool
        return f"Tool {provider} returned no timestamp; treat outputs as legacy context"

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
    def _format_price(value: Any, currency: str | None, prefix: str = "") -> str:
        try:
            if value is None:
                return "n/a"
            amount = float(value)
        except (TypeError, ValueError):
            return "n/a"
        unit = currency or "USD"
        prefix = f"{prefix} " if prefix else ""
        return f"{prefix}{unit} {amount:,.2f}"

    @staticmethod
    def _format_timestamp(raw: Any) -> str:
        if raw is None:
            return "unknown time"
        if isinstance(raw, (int, float)):
            try:
                dt = datetime.fromtimestamp(float(raw), tz=UTC)
            except (OverflowError, ValueError):
                return "unknown time"
            return dt.strftime("%Y-%m-%d %H:%M %Z")
        if isinstance(raw, str):
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return raw
            return dt.strftime("%Y-%m-%d %H:%M %Z")
        return str(raw)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=UTC)
            except (OverflowError, ValueError):
                return None
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @staticmethod
    def _tool_metadata(tool_result: ToolInvocationResult) -> dict[str, Any]:
        return {
            "name": tool_result.tool,
            "resolved": tool_result.resolved_tool,
            "cached": tool_result.cached,
            "latency": round(tool_result.latency, 4),
        }

    def _summarize_fundamentals(self, fundamentals: Any, fallback_currency: str | None) -> str:
        if not isinstance(fundamentals, dict) or not fundamentals:
            return ""
        currency = fundamentals.get("currency") or fallback_currency or "USD"
        parts: list[str] = []

        trailing = fundamentals.get("trailing")
        if isinstance(trailing, dict):
            revenue = trailing.get("revenue")
            net_income = trailing.get("net_income")
            trailing_parts: list[str] = []
            if revenue is not None:
                trailing_parts.append(f"TTM revenue {self._format_amount(revenue, currency)}")
            if net_income is not None:
                trailing_parts.append(f"TTM net income {self._format_amount(net_income, currency)}")
            if trailing_parts:
                parts.append("; ".join(trailing_parts))

        quarterly = fundamentals.get("quarterly")
        if isinstance(quarterly, list) and quarterly:
            latest = quarterly[0]
            if isinstance(latest, dict):
                period = latest.get("period", "recent quarter")
                revenue = latest.get("total_revenue")
                net_income = latest.get("net_income")
                quarter_parts: list[str] = []
                if revenue is not None:
                    quarter_parts.append(f"revenue {self._format_amount(revenue, currency)}")
                if net_income is not None:
                    quarter_parts.append(f"net income {self._format_amount(net_income, currency)}")
                if quarter_parts:
                    parts.append(f"Latest quarter ({period}): {', '.join(quarter_parts)}")

        guidance = fundamentals.get("guidance")
        if isinstance(guidance, dict) and guidance:
            guidance_parts: list[str] = []
            target_mean = guidance.get("target_mean_price")
            if target_mean is not None:
                guidance_parts.append(f"target price {self._format_price(target_mean, currency)}")
            forward_eps = guidance.get("forward_eps")
            if forward_eps is not None:
                try:
                    guidance_parts.append(f"forward EPS {float(forward_eps):.2f}")
                except (TypeError, ValueError):
                    pass
            revenue_growth = guidance.get("revenue_growth")
            if revenue_growth is not None:
                try:
                    guidance_parts.append(f"revenue growth {float(revenue_growth)*100:.1f}%")
                except (TypeError, ValueError):
                    pass
            earnings_growth = guidance.get("earnings_growth")
            if earnings_growth is not None:
                try:
                    guidance_parts.append(f"earnings growth {float(earnings_growth)*100:.1f}%")
                except (TypeError, ValueError):
                    pass
            if guidance_parts:
                parts.append("Guidance: " + "; ".join(guidance_parts))

        return "; ".join(parts)

    def _summarize_historical_prices(self, historical_prices: Any, currency: str | None) -> str:
        """Summarize historical price data for the LLM context."""
        if not isinstance(historical_prices, list) or not historical_prices:
            return ""
        
        currency_str = currency or "USD"
        valid_prices: list[tuple[str, float]] = []
        
        for entry in historical_prices:
            if not isinstance(entry, dict):
                continue
            date = entry.get("date")
            close = entry.get("close") or entry.get("adjusted_close")
            if date and close is not None:
                try:
                    valid_prices.append((str(date), float(close)))
                except (TypeError, ValueError):
                    continue
        
        if len(valid_prices) < 2:
            return ""
        
        # Sort by date (most recent first, should already be sorted)
        valid_prices.sort(key=lambda x: x[0], reverse=True)
        
        latest_date, latest_price = valid_prices[0]
        oldest_date, oldest_price = valid_prices[-1]
        
        # Calculate period return
        period_change = latest_price - oldest_price
        period_change_pct = (period_change / oldest_price * 100) if oldest_price else 0.0
        
        # Find high and low in period
        all_prices = [p[1] for p in valid_prices]
        period_high = max(all_prices)
        period_low = min(all_prices)
        
        # Recent trend (last 5 days vs previous 5 days)
        trend_desc = ""
        if len(valid_prices) >= 10:
            recent_5 = sum([p[1] for p in valid_prices[:5]]) / 5
            prev_5 = sum([p[1] for p in valid_prices[5:10]]) / 5
            if recent_5 > prev_5 * 1.01:
                trend_desc = ", trending up"
            elif recent_5 < prev_5 * 0.99:
                trend_desc = ", trending down"
            else:
                trend_desc = ", relatively flat"
        
        return (
            f"Historical ({oldest_date} to {latest_date}): "
            f"{len(valid_prices)} days, "
            f"{self._format_price(period_change, currency_str, prefix='Δ')} ({period_change_pct:+.1f}%), "
            f"range {self._format_price(period_low, currency_str)}-{self._format_price(period_high, currency_str)}"
            f"{trend_desc}"
        )

    @staticmethod
    def _format_amount(value: Any, currency: str | None) -> str:
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return "n/a"
        unit = currency or "USD"
        absolute = abs(amount)
        scale = 1.0
        suffix = ""
        for threshold, label in ((1_000_000_000_000.0, "T"), (1_000_000_000.0, "B"), (1_000_000.0, "M")):
            if absolute >= threshold:
                scale = threshold
                suffix = label
                break
        formatted = amount / scale if scale != 1.0 else amount
        if suffix:
            return f"{unit} {formatted:,.2f}{suffix}"
        return f"{unit} {formatted:,.2f}"

    @staticmethod
    def _filtered_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        return {key: value for key, value in metadata.items() if key != "_shared_context"}

    def _format_handoff(self, metadata: dict[str, Any]) -> str:
        shared = metadata.get("_shared_context")
        if not isinstance(shared, dict):
            return "(no shared context)"
        provenance = shared.get("provenance")
        if not isinstance(provenance, list) or not provenance:
            return "(no shared context)"
        lines: list[str] = []
        for item in provenance[-3:]:
            if not isinstance(item, dict):
                continue
            agent_name = item.get("agent", "agent")
            summary = item.get("summary") or "(no summary)"
            tool_meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            tool_info = tool_meta.get("tool") if isinstance(tool_meta, dict) else None
            if isinstance(tool_info, dict):
                resolved = tool_info.get("resolved")
            else:
                resolved = None
            tool_segment = f" via {resolved}" if resolved else ""
            lines.append(f"- {agent_name}{tool_segment}: {summary}")
        return "\n".join(lines) if lines else "(no shared context)"


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
