from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple

from app.tools.registry import normalize_tool_name

__all__ = [
    "AgentToolPolicy",
    "get_agent_tool_policy",
    "filter_agent_tools",
    "AGENT_TOOL_POLICIES",
]


@dataclass(frozen=True)
class _PatternRule:
    base: str
    wildcard: bool

    def matches(self, normalized: str) -> bool:
        if self.wildcard:
            if not self.base:
                return True
            return normalized == self.base or normalized.startswith(f"{self.base}.")
        return normalized == self.base


def _compile_pattern(pattern: str) -> _PatternRule:
    value = (pattern or "").strip()
    if not value:
        raise ValueError("pattern cannot be blank")
    if value == "*":
        return _PatternRule(base="", wildcard=True)
    wildcard = value.endswith(".*")
    if wildcard:
        value = value[:-2]
    base = normalize_tool_name(value) if value else ""
    return _PatternRule(base=base, wildcard=wildcard)


def _unique_ordered(items: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


@dataclass(frozen=True)
class AgentToolPolicy:
    agent: str
    allowed_patterns: Tuple[str, ...] = field(default_factory=tuple)
    denied_patterns: Tuple[str, ...] = field(default_factory=tuple)
    _allowed_rules: Tuple[_PatternRule, ...] = field(init=False, repr=False)
    _denied_rules: Tuple[_PatternRule, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_allowed_rules", tuple(_compile_pattern(pattern) for pattern in self.allowed_patterns))
        object.__setattr__(self, "_denied_rules", tuple(_compile_pattern(pattern) for pattern in self.denied_patterns))

    def is_allowed(self, tool: str) -> bool:
        normalized = normalize_tool_name(tool)
        if any(rule.matches(normalized) for rule in self._denied_rules):
            return False
        if not self._allowed_rules:
            return True
        return any(rule.matches(normalized) for rule in self._allowed_rules)

    def filter_tools(self, tools: Sequence[str]) -> tuple[list[str], list[str]]:
        allowed: list[str] = []
        removed: list[str] = []
        for tool in tools:
            if self.is_allowed(tool):
                allowed.append(tool)
            else:
                removed.append(tool)
        return allowed, removed


UNIVERSAL_ALLOWED: Tuple[str, ...] = (
    "memory.*",
    "utils.*",
    "test.*",
)


def _policy(agent: str, allowed: Sequence[str], denied: Sequence[str] = ()) -> AgentToolPolicy:
    combined_allowed = _unique_ordered(list(UNIVERSAL_ALLOWED) + list(allowed))
    combined_denied = _unique_ordered(denied)
    return AgentToolPolicy(agent=agent, allowed_patterns=combined_allowed, denied_patterns=combined_denied)


AGENT_TOOL_POLICIES: dict[str, AgentToolPolicy] = {
    "enterprise_agent": _policy(
        "enterprise_agent",
        allowed=(
            "file.*",
            "http.fetch",
            "json.*",
            "text.*",
            "pdf.*",
            "document.*",
            "terminal.execute",
            "shell.execute",
            "process.*",
            "search.web",
            "enterprise.*",
            "browser.*",
            "dataframe.*",
        ),
        denied=(
            "finance.*",
            "research.vector_search",
            "research.embedding.create",
            "creative.*",
        ),
    ),
    "finance_agent": _policy(
        "finance_agent",
        allowed=(
            "http.fetch",
            "file.read",
            "json.*",
            "text.summarize",
            "pdf.extract_text",
            "pdf.extract_tables",
            "finance.*",
            "dataframe.*",
        ),
        denied=(
            "terminal.execute",
            "shell.execute",
            "process.*",
            "creative.*",
            "research.vector_search",
            "code.*",
        ),
    ),
    "research_agent": _policy(
        "research_agent",
        allowed=(
            "search.web",
            "search.academic",
            "vector.*",
            "pdf.*",
            "document.*",
            "file.*",
            "text.*",
            "json.*",
            "code.*",
            "research.*",
            "browser.*",
            "dataframe.*",
        ),
        denied=(
            "terminal.execute",
            "shell.execute",
            "process.*",
            "finance.*",
            "creative.*",
        ),
    ),
    "creative_agent": _policy(
        "creative_agent",
        allowed=(
            "file.read",
            "text.*",
            "json.*",
            "creative.*",
            "browser.*",
        ),
        denied=(
            "terminal.execute",
            "shell.execute",
            "process.*",
            "finance.*",
            "research.*",
            "vector.*",
            "code.*",
        ),
    ),
    "general_agent": AgentToolPolicy(agent="general_agent", allowed_patterns=("*",)),
}


def get_agent_tool_policy(agent: str) -> AgentToolPolicy | None:
    return AGENT_TOOL_POLICIES.get(agent)


def filter_agent_tools(agent: str, tools: Sequence[str]) -> tuple[list[str], list[str]]:
    policy = get_agent_tool_policy(agent)
    if policy is None:
        return list(tools), []
    return policy.filter_tools(tools)
