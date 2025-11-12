from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, Iterator, Tuple

__all__ = ["normalize_tool_name", "ToolRegistry", "tool_registry"]


_NAME_PATTERN = re.compile(r"[\\/\s]+")
_ALIAS_COLLAPSE = re.compile(r"\.+")


def normalize_tool_name(name: str) -> str:
    """Return a normalized identifier used for registry lookups."""
    if not isinstance(name, str):
        raise TypeError("Tool name must be a string")
    collapsed = _NAME_PATTERN.sub(".", name.strip())
    collapsed = _ALIAS_COLLAPSE.sub(".", collapsed)
    collapsed = collapsed.strip(".")
    return collapsed.lower()


class ToolRegistry:
    """Registry that manages tool adapters, aliases, and circuit breaker state."""

    def __init__(self) -> None:
        self._registry: Dict[str, Any] = {}
        self._canonical: Dict[str, str] = {}
        self._alias_index: Dict[str, str] = {}
        self._alias_display: Dict[str, str] = {}
        self._failure_threshold: int = 5
        self._cooldown_seconds: float = 30.0
        self._failures: Dict[str, int] = {}
        self._open_until: Dict[str, float] = {}

    def register(self, name: str, adapter: Any, *, aliases: Iterable[str] | None = None) -> None:
        key = normalize_tool_name(name)
        canonical = self._canonicalize(name)
        self._registry[key] = adapter
        self._canonical[key] = canonical
        self._failures.pop(key, None)
        self._open_until.pop(key, None)
        if aliases:
            for alias in aliases:
                self.register_alias(alias, canonical)

    def unregister(self, name: str) -> None:
        key = normalize_tool_name(name)
        if key not in self._registry:
            return
        del self._registry[key]
        self._failures.pop(key, None)
        self._open_until.pop(key, None)
        canonical = self._canonical.pop(key, None)
        if canonical is None:
            return
        for alias_key, target_key in list(self._alias_index.items()):
            if target_key == key:
                del self._alias_index[alias_key]
        for alias, target in list(self._alias_display.items()):
            if target == canonical:
                del self._alias_display[alias]

    def clear(self) -> None:
        self._registry.clear()
        self._canonical.clear()
        self._alias_index.clear()
        self._alias_display.clear()
        self._failures.clear()
        self._open_until.clear()

    def register_alias(self, alias: str, target: str) -> None:
        alias = alias.strip()
        target = target.strip()
        if not alias or not target:
            return
        key = normalize_tool_name(target)
        canonical = self._canonicalize(target)
        self._canonical.setdefault(key, canonical)
        self._store_alias(alias, key, canonical)
        alternate = self._alternate_form(alias)
        if alternate != alias:
            self._store_alias(alternate, key, canonical)

    def get(self, name: str) -> Any | None:
        key = self._resolve_key(name)
        if key is None:
            return None
        return self._registry.get(key)

    def resolve(self, name: str) -> str | None:
        normalized = normalize_tool_name(name)
        seen: set[str] = set()
        current = normalized
        while True:
            canonical = self._canonical.get(current)
            alias_key = self._alias_index.get(current)
            if alias_key is None:
                return canonical
            if alias_key in seen:
                return canonical
            seen.add(alias_key)
            current = alias_key

    def list(self) -> list[str]:
        return sorted(self._canonical[key] for key in self._registry)

    def aliases(self) -> dict[str, str]:
        return dict(sorted(self._alias_display.items()))

    def items(self) -> Iterator[Tuple[str, Any]]:
        for key, adapter in self._registry.items():
            yield self._canonical.get(key, key), adapter

    def configure_circuit(self, *, threshold: int, reset_seconds: float) -> None:
        self._failure_threshold = max(1, int(threshold))
        self._cooldown_seconds = max(0.0, float(reset_seconds))

    def record_failure(self, name: str) -> bool:
        key = self._resolve_tracking_key(name)
        if key is None:
            return False
        count = self._failures.get(key, 0) + 1
        self._failures[key] = count
        if count >= self._failure_threshold and self._cooldown_seconds > 0:
            self._open_until[key] = self._monotonic() + self._cooldown_seconds
            return True
        return False

    def record_success(self, name: str) -> None:
        key = self._resolve_tracking_key(name)
        if key is None:
            return
        self._failures.pop(key, None)
        self._open_until.pop(key, None)

    def is_circuit_open(self, name: str) -> bool:
        key = self._resolve_tracking_key(name)
        if key is None:
            return False
        expires_at = self._open_until.get(key)
        if expires_at is None:
            return False
        now = self._monotonic()
        if now >= expires_at:
            self._open_until.pop(key, None)
            self._failures.pop(key, None)
            return False
        return True

    def failure_count(self, name: str) -> int:
        key = self._resolve_tracking_key(name)
        if key is None:
            return 0
        return self._failures.get(key, 0)

    def _resolve_key(self, name: str) -> str | None:
        normalized = normalize_tool_name(name)
        if normalized in self._registry:
            return normalized
        alias_key = self._alias_index.get(normalized)
        if alias_key is not None and alias_key in self._registry:
            return alias_key
        return None

    def _resolve_tracking_key(self, name: str) -> str | None:
        normalized = normalize_tool_name(name)
        if normalized in self._canonical:
            return normalized
        alias_key = self._alias_index.get(normalized)
        if alias_key is not None:
            return alias_key
        return None

    def _store_alias(self, alias: str, key: str, canonical: str) -> None:
        alias = alias.strip()
        if not alias:
            return
        normalized_alias = normalize_tool_name(alias)
        self._alias_index[normalized_alias] = key
        self._canonical[normalized_alias] = canonical
        self._alias_display[alias] = canonical

    def _canonicalize(self, name: str) -> str:
        sanitized = (name or "").strip()
        sanitized = sanitized.replace("\\", "/")
        sanitized = sanitized.replace(" ", "/")
        sanitized = sanitized.replace(".", "/")
        sanitized = re.sub(r"/{2,}", "/", sanitized)
        sanitized = sanitized.strip("/")
        return sanitized.lower()

    def _alternate_form(self, alias: str) -> str:
        if "/" in alias:
            return alias.replace("/", ".")
        if "." in alias:
            return alias.replace(".", "/")
        return alias

    def _monotonic(self) -> float:
        return time.monotonic()


tool_registry = ToolRegistry()
