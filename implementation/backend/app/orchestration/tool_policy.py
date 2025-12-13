from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

from ..tools.registry import normalize_tool_name


@dataclass(frozen=True)
class AgentToolPolicy:
    allowed_patterns: tuple[str, ...]
    denied_patterns: tuple[str, ...] = ()
    default_allow: bool = False

    def is_allowed(self, tool: str) -> bool:
        identifier = normalize_tool_name(tool)
        for pattern in self.denied_patterns:
            if fnmatch(identifier, pattern):
                return False
        if not self.allowed_patterns:
            return True
        if any(fnmatch(identifier, pattern) for pattern in self.allowed_patterns):
            return True
        return self.default_allow

    def filter_tools(self, tools: Sequence[str]) -> tuple[list[str], list[str]]:
        allowed: list[str] = []
        removed: list[str] = []
        for tool in tools:
            if self.is_allowed(tool):
                allowed.append(tool)
            else:
                removed.append(tool)
        return allowed, removed


_POLICY_DEFINITIONS: Mapping[str, Mapping[str, Sequence[str]]] = {
    # ════════════════════════════════════════════════════════════════════════════
    # GENERAL AGENT: First-line responder, handles greetings, clarifications,
    # and routes to specialists. Has access to basic research and summarization.
    # ════════════════════════════════════════════════════════════════════════════
    "general_agent": {
        "allowed": (
            # Research & Information
            "research.search",
            "research.summarizer",
            "research.wikipedia",
            "research.arxiv",
            # Browser & Web
            "browser.open",
            "browser.extract_text",
            # Creative (basic tone checking)
            "creative.tonecheck",
            "creative.tone_checker",
            # Memory & Context
            "memory.*",
            # Utils
            "utils.*",
        ),
        "denied": (
            "terminal.*",
            "code.*",
            "finance.*",
            "enterprise.*",
        ),
    },
    # ════════════════════════════════════════════════════════════════════════════
    # RESEARCH AGENT: Deep research, citations, document analysis, knowledge synthesis
    # ════════════════════════════════════════════════════════════════════════════
    "research_agent": {
        "allowed": (
            # Core Research Tools (Open Source)
            "research.search",           # DuckDuckGo - anonymous web search
            "research.summarizer",       # Text summarization
            "research.wikipedia",        # Wikipedia API - free
            "research.arxiv",            # arXiv - open academic papers
            "research.doc_loader",       # Document loading/parsing
            "research.vector_search",    # Qdrant vector search
            # Browser Tools
            "browser.open",              # HTTP fetching
            "browser.extract_text",      # HTML text extraction
            # Data Analysis
            "dataframe.*",               # Pandas-based analytics
            # Financial Data (read-only for research context)
            "finance.snapshot",          # Yahoo Finance - free
            "finance.snapshot.alpha",    # Alpha Vantage - free tier
            "finance.snapshot.cached",   # Cached quotes
            # Memory & Context
            "memory.*",
            # Utils
            "utils.*",
        ),
        "denied": (
            "creative.*",
            "terminal.*",
            "enterprise.*",
        ),
    },
    # ════════════════════════════════════════════════════════════════════════════
    # FINANCE AGENT: Financial analysis, market data, portfolio insights
    # All tools use open-source or free-tier APIs
    # ════════════════════════════════════════════════════════════════════════════
    "finance_agent": {
        "allowed": (
            # Core Finance Tools (Free/Open APIs)
            "finance.snapshot",          # Yahoo Finance - free
            "finance.snapshot.alpha",    # Alpha Vantage - free tier (5 calls/min)
            "finance.snapshot.cached",   # Stooq fallback - free
            "finance.indicators.*",      # Technical indicators
            "finance.plot",              # Matplotlib charting - open source
            "finance.analytics",         # Pandas analytics - open source
            "finance/pandas",            # Pandas data analysis
            "finance/csv_analyzer",      # CSV analysis
            "finance/sentiment",         # FinBERT sentiment - open source model
            "finance/finbert",           # FinBERT alias
            # Data Processing
            "dataframe.*",               # Pandas operations
            # Research Support (for news via DuckDuckGo)
            "research.search",           # DuckDuckGo for news context
            "research.wikipedia",        # Company background
            # Memory & Context
            "memory.*",
            # Utils
            "utils.*",
        ),
        "denied": (
            "creative.*",
            "terminal.*",
            "enterprise.*",
        ),
    },
    # ════════════════════════════════════════════════════════════════════════════
    # CREATIVE AGENT: Content creation, tone styling, prompt crafting
    # ════════════════════════════════════════════════════════════════════════════
    "creative_agent": {
        "allowed": (
            # Creative Tools
            "creative.*",                # All creative adapters
            # Browser & Research (for inspiration/reference)
            "browser.open",
            "browser.extract_text",
            "research.search",           # DuckDuckGo for research
            "research.wikipedia",        # Background info
            # Text Processing
            "text.*",
            # Memory
            "memory.*",
            # Utils
            "utils.*",
        ),
        "denied": (
            "finance.*",
            "terminal.*",
            "enterprise.*",
        ),
    },
    # ════════════════════════════════════════════════════════════════════════════
    # ENTERPRISE AGENT: Business strategy, playbooks, compliance, CRM
    # ════════════════════════════════════════════════════════════════════════════
    "enterprise_agent": {
        "allowed": (
            # Enterprise Tools
            "enterprise.*",              # All enterprise adapters
            # Browser & Research (for business intelligence)
            "browser.open",
            "browser.extract_text",
            "research.search",           # Market research
            "research.wikipedia",        # Industry background
            "research.arxiv",            # Academic business research
            # Data Analysis
            "dataframe.*",
            # Memory & Context
            "memory.*",
            # Planning
            "planning.*",
            # Utils
            "utils.*",
        ),
        "denied": (
            "finance.*",
            "creative.*",
            "terminal.*",
        ),
    },
}


@lru_cache(maxsize=16)
def get_agent_tool_policy(agent_name: str) -> AgentToolPolicy | None:
    definition = _POLICY_DEFINITIONS.get(agent_name)
    if definition is None:
        return None
    return AgentToolPolicy(
        allowed_patterns=tuple(str(pattern) for pattern in definition.get("allowed", ()) if pattern),
        denied_patterns=tuple(str(pattern) for pattern in definition.get("denied", ()) if pattern),
    )
