from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class PlannedTool:
    alias: str
    resolved: str
    category: str
    description: str


# High-level onboarding plan derived from docs/mcp/category_onboarding.md
RESEARCH_TOOLS: List[PlannedTool] = [
    PlannedTool(alias="research.search", resolved="search/duckduckgo", category="research", description="DuckDuckGo web search"),
    PlannedTool(alias="research.arxiv", resolved="research/arxiv", category="research", description="ArXiv paper retrieval"),
    PlannedTool(alias="research.wikipedia", resolved="research/wikipedia", category="research", description="Wikipedia summaries"),
    PlannedTool(alias="research.doc_loader", resolved="research/doc_loader", category="research", description="Document ingestion"),
    PlannedTool(alias="research.qdrant", resolved="research/qdrant", category="research", description="Vector store retriever"),
    PlannedTool(alias="research.summarizer", resolved="research/summarizer", category="research", description="Summarisation helper"),
]

FINANCE_TOOLS: List[PlannedTool] = [
    PlannedTool(alias="finance.snapshot", resolved="finance/yfinance", category="finance", description="Yahoo Finance snapshot"),
    PlannedTool(alias="finance.analytics", resolved="finance/pandas", category="finance", description="Pandas analytics sandbox"),
    PlannedTool(alias="finance.plot", resolved="finance/plot", category="finance", description="Plotting utility"),
    PlannedTool(alias="finance.news", resolved="finance/coingecko_news", category="finance", description="CoinGecko news feed"),
    PlannedTool(alias="finance.csv", resolved="finance/csv", category="finance", description="CSV/Excel analyser"),
    PlannedTool(alias="finance.sentiment", resolved="finance/finbert", category="finance", description="FinBERT sentiment analysis"),
]

CREATIVE_TOOLS: List[PlannedTool] = [
    PlannedTool(alias="creative.tonecheck", resolved="creative/stylizer", category="creative", description="Prompt tone stylizer"),
    PlannedTool(alias="creative.tone_checker", resolved="creative/tone_checker", category="creative", description="Tone diagnostics"),
    PlannedTool(alias="creative.transcribe", resolved="creative/whisper_transcription", category="creative", description="Whisper transcription helper"),
    PlannedTool(alias="creative.image", resolved="creative/image_generator", category="creative", description="Image generation placeholder"),
]

ENTERPRISE_TOOLS: List[PlannedTool] = [
    PlannedTool(alias="enterprise.playbook", resolved="enterprise/playbook", category="enterprise", description="Composite playbook orchestrator"),
    PlannedTool(alias="enterprise.notion", resolved="enterprise/notion", category="enterprise", description="Notion knowledge connector"),
    PlannedTool(alias="enterprise.calendar", resolved="enterprise/calendar", category="enterprise", description="Calendar synchronization"),
    PlannedTool(alias="enterprise.policy", resolved="enterprise/policy_checker", category="enterprise", description="Policy compliance check"),
    PlannedTool(alias="enterprise.crm", resolved="enterprise/crm", category="enterprise", description="CRM account insights"),
]

ONBOARDING_TOOLS: List[PlannedTool] = RESEARCH_TOOLS + FINANCE_TOOLS + CREATIVE_TOOLS + ENTERPRISE_TOOLS


def planned_tools_by_category() -> Dict[str, List[PlannedTool]]:
    categories: Dict[str, List[PlannedTool]] = {}
    for tool in ONBOARDING_TOOLS:
        categories.setdefault(tool.category, []).append(tool)
    return categories


def all_planned_tools() -> Iterable[PlannedTool]:
    return list(ONBOARDING_TOOLS)


__all__ = [
    "PlannedTool",
    "RESEARCH_TOOLS",
    "FINANCE_TOOLS",
    "CREATIVE_TOOLS",
    "ENTERPRISE_TOOLS",
    "planned_tools_by_category",
    "all_planned_tools",
]
