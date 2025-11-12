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
class FinanceAgent:
    name: str = "finance_agent"
    capability: AgentCapability = AgentCapability.FINANCE
    system_prompt: str = (
        "You are NeuraForge's Finance Agent. Provide high-level forecasts, key risks, and actionable"
        " next steps using available context. Respond with structured bullet points. When live tool"
        " metrics are available, treat them as the source of truth, cite their timestamps, and avoid"
        " repeating stale historical figures unless you explicitly flag them as legacy context."
    )
    description: str = "Performs financial analysis, forecasting, and headline metric synthesis."
    tool_preference: list[str] = field(default_factory=lambda: ["finance.snapshot"])
    fallback_agent: str | None = "enterprise_agent"
    confidence_bias: float = 0.9

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
        metadata_clean = self._filtered_metadata(task.metadata)
        metadata_json = json.dumps(metadata_clean, indent=2, sort_keys=True)
        prior = _collect_prior_outputs(task.prior_exchanges)
        retrieved = context_section or "(no retrieved context)"
        tool_data = self._format_tool_metrics(tool_result)
        tool_json = self._format_tool_json(tool_result)
        handoff = self._format_handoff(task.metadata)
        return (
            f"Business question:\n{task.prompt}\n\n"
            f"Financial metadata:\n{metadata_json}\n\n"
            f"Shared context from peers:\n{handoff}\n\n"
            f"Earlier agent signals:\n{prior}\n\n"
            f"Retrieved context:\n{retrieved}\n\n"
            f"Tool metrics summary:\n{tool_data}\n\n"
            f"Raw tool metrics JSON (primary source, prefer over model priors):\n{tool_json}\n\n"
            "Estimate revenue or cost implications, surface 2-3 headline metrics, and outline immediate actions."
        )

    async def _maybe_invoke_tool(self, task: AgentInput, *, context: AgentContext) -> ToolInvocationResult | None:
        if context.tools is None:
            return None
        payload = self._build_snapshot_payload(task)
        if payload is None:
            logger.debug("finance_snapshot_payload_skipped", task_id=task.task_id)
            return None
        try:
            return await context.tools.invoke("finance.snapshot", payload)
        except (ToolDisabledError, ToolInvocationError) as exc:
            logger.warning("finance_tool_failure", error=str(exc))
            return None

    def _build_snapshot_payload(self, task: AgentInput) -> dict[str, Any] | None:
        metadata: dict[str, Any] = task.metadata if isinstance(task.metadata, dict) else {}
        symbols = self._extract_symbols(metadata)
        if not symbols:
            symbols.extend(self._infer_symbols_from_prompt(task.prompt))

        unique_symbols = self._unique_upper(symbols)
        payload: dict[str, Any] = {}
        if unique_symbols:
            payload["symbols"] = unique_symbols[:5]
        else:
            query = self._derive_company_query(task, metadata)
            if query is None:
                return None
            payload["query"] = query

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
        if not prompt:
            return []
        ticker_candidates = re.findall(r"\b[A-Z]{1,5}\b", prompt)
        return [candidate for candidate in ticker_candidates if len(candidate) >= 2]

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
        return "\n".join(lines)

    def _format_tool_json(self, tool_result: ToolInvocationResult | None) -> str:
        if tool_result is None or not isinstance(tool_result.response, dict):
            return "(no tool payload)"
        try:
            return json.dumps(tool_result.response, indent=2, sort_keys=True, default=str)
        except TypeError:
            return "(tool payload not serializable)"

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
