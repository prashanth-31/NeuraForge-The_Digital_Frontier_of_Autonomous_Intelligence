from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Mapping

from pydantic import BaseModel
from prometheus_client import Counter, Histogram

from app.core.config import get_settings

from .exceptions import CircuitBreakerOpenError, ToolInvocationError, ToolTimeoutError
from .registry import tool_registry

__all__ = [
    "TOOL_INVOCATIONS",
    "TOOL_LATENCY",
    "ToolInvocationError",
    "ToolTimeoutError",
    "CircuitBreakerOpenError",
    "MCPToolAdapter",
]


_settings = get_settings()
_runtime = getattr(_settings.tools, "runtime", None)
_DEFAULT_TIMEOUT = getattr(_runtime, "invocation_timeout_seconds", 8.0)
_DEFAULT_RETRIES = getattr(_runtime, "invocation_retries", 2)
_DEFAULT_CIRCUIT_THRESHOLD = getattr(_runtime, "circuit_break_failures", 5)
_DEFAULT_CIRCUIT_RESET = getattr(_runtime, "circuit_break_reset_seconds", 30.0)

tool_registry.configure_circuit(
    threshold=_DEFAULT_CIRCUIT_THRESHOLD,
    reset_seconds=_DEFAULT_CIRCUIT_RESET,
)

TOOL_INVOCATIONS = Counter(
    "mcp_tool_invocations_total",
    "Count of MCP tool invocations grouped by outcome.",
    ["tool", "outcome"],
)
TOOL_LATENCY = Histogram(
    "mcp_tool_latency_seconds",
    "Latency distribution for MCP tool invocations.",
    ["tool"],
)


class MCPToolAdapter:
    """Base adapter providing retries, timeout enforcement, and result normalization."""

    name: str = "unnamed.tool"
    default_timeout: float | None = None
    default_retries: int | None = None
    _backoff_base: float = 0.5

    async def invoke(
        self,
        payload: Mapping[str, Any],
        *,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> Dict[str, Any]:
        tool_name = self._canonical_tool_name()
        if tool_registry.is_circuit_open(tool_name):
            TOOL_INVOCATIONS.labels(tool=tool_name, outcome="circuit_open").inc()
            raise CircuitBreakerOpenError(f"Tool '{tool_name}' is temporarily unavailable")

        effective_timeout = self._effective_timeout(timeout)
        effective_retries = self._effective_retries(retries)

        attempt = 0
        last_exc: BaseException | None = None
        last_error: ToolInvocationError | None = None
        start = time.perf_counter()

        while attempt <= effective_retries:
            try:
                result = await asyncio.wait_for(self._run(dict(payload)), timeout=effective_timeout)
            except asyncio.TimeoutError as exc:
                last_exc = exc
                last_error = ToolTimeoutError(
                    f"Tool '{tool_name}' timed out after {attempt + 1} attempt(s)"
                )
                TOOL_INVOCATIONS.labels(tool=tool_name, outcome="timeout").inc()
                circuit_opened = tool_registry.record_failure(tool_name)
                if attempt >= effective_retries or circuit_opened:
                    TOOL_LATENCY.labels(tool=tool_name).observe(time.perf_counter() - start)
                    if circuit_opened:
                        raise CircuitBreakerOpenError(
                            f"Tool '{tool_name}' circuit breaker open after consecutive timeouts"
                        ) from exc
                    raise last_error from exc
            except Exception as exc:  # pragma: no cover - normalized below
                last_exc = exc
                last_error = ToolInvocationError(f"Tool '{tool_name}' failed: {exc}")
                TOOL_INVOCATIONS.labels(tool=tool_name, outcome="failure").inc()
                circuit_opened = tool_registry.record_failure(tool_name)
                if attempt >= effective_retries or circuit_opened:
                    TOOL_LATENCY.labels(tool=tool_name).observe(time.perf_counter() - start)
                    if circuit_opened:
                        raise CircuitBreakerOpenError(
                            f"Tool '{tool_name}' circuit breaker open due to repeated failures"
                        ) from exc
                    raise last_error from exc
            else:
                tool_registry.record_success(tool_name)
                TOOL_INVOCATIONS.labels(tool=tool_name, outcome="success").inc()
                duration = time.perf_counter() - start
                TOOL_LATENCY.labels(tool=tool_name).observe(duration)
                return self._normalize_result(result)

            attempt += 1
            backoff = self._backoff_base * (2**attempt)
            await asyncio.sleep(backoff)

        TOOL_LATENCY.labels(tool=tool_name).observe(time.perf_counter() - start)
        if last_error is not None:
            raise last_error from last_exc
        raise ToolInvocationError(f"Tool '{tool_name}' failed after retries")

    async def _run(self, payload: Dict[str, Any]) -> Any:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _canonical_tool_name(self) -> str:
        resolved = tool_registry.resolve(self.name)
        if resolved:
            return resolved
        cleaned = self.name.replace(".", "/")
        return cleaned.lower()

    def _effective_timeout(self, override: float | None) -> float:
        if override is not None:
            return max(0.1, float(override))
        if self.default_timeout is not None:
            return max(0.1, float(self.default_timeout))
        return max(0.1, float(_DEFAULT_TIMEOUT))

    def _effective_retries(self, override: int | None) -> int:
        if override is not None:
            return max(0, int(override))
        if self.default_retries is not None:
            return max(0, int(self.default_retries))
        return max(0, int(_DEFAULT_RETRIES))

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, BaseModel):
            payload = result.model_dump()
        elif isinstance(result, dict):
            payload = dict(result)
        else:
            try:
                json.dumps(result)
            except (TypeError, ValueError):
                return {"result": str(result)}
            return {"result": result}
        return self._json_safe_dict(payload)

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return self._json_safe(value.model_dump())
        if isinstance(value, dict):
            return self._json_safe_dict(value)
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return [self._json_safe(item) for item in sorted(value, key=lambda item: repr(item))]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    def _json_safe_dict(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for key, value in payload.items():
            safe[str(key)] = self._json_safe(value)
        return safe
