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
    "general_agent": {
        "allowed": (
            "research.search",
            "research.summarizer",
            "creative.tonecheck",
            "browser.open",
            "browser.summarize",
            "memory.*",
        ),
        "denied": (
            "terminal.*",
            "code.*",
            "finance.*",
        ),
    },
    "research_agent": {
        "allowed": (
            "research.search",
            "research.summarizer",
            "research.doc_loader",
            "browser.*",
            "pdf.*",
            "text.*",
            "finance.snapshot",
            "finance.snapshot.alpha",
            "finance.snapshot.cached",
        ),
        "denied": (
            "finance.indicators.*",
            "creative.*",
            "terminal.*",
        ),
    },
    "finance_agent": {
        "allowed": (
            "finance.snapshot",
            "finance.snapshot.alpha",
            "finance.snapshot.cached",
            "finance.indicators.*",
            "finance.plot",
            "dataframe.*",
            "memory.*",
        ),
        "denied": (
            "creative.*",
            "research.vector_search",
            "terminal.*",
        ),
    },
    "creative_agent": {
        "allowed": (
            "creative.*",
            "browser.*",
            "text.*",
        ),
        "denied": (
            "finance.*",
            "research.*",
            "terminal.*",
            "code.*",
        ),
    },
    "enterprise_agent": {
        "allowed": (
            "enterprise.*",
            "browser.*",
            "memory.*",
            "dataframe.*",
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
