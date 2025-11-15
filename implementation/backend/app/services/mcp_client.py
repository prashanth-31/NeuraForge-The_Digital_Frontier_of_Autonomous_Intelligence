from __future__ import annotations

import asyncio
import random
import time
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from app.core import metrics
from app.core.logging import get_logger

logger = get_logger(name=__name__)


AuthHeaderProvider = Callable[[], Awaitable[dict[str, str] | None] | dict[str, str] | None]
RequestSigner = Callable[[str, str, Any | None, dict[str, Any] | None], Awaitable[dict[str, str] | None] | dict[str, str] | None]
InstrumentationHook = Callable[[dict[str, Any]], None]


class CircuitOpenError(RuntimeError):
    """Raised when the MCP circuit breaker is open."""


@dataclass(slots=True)
class MCPClientConfig:
    base_url: str
    timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    retry_jitter_seconds: float
    verify_ssl: bool
    default_headers: dict[str, str]
    circuit_breaker_threshold: int
    circuit_breaker_reset_seconds: float
    auth_header_provider: AuthHeaderProvider | None = None
    request_signer: RequestSigner | None = None
    instrumentation_hooks: tuple[InstrumentationHook, ...] = ()


class _CircuitBreaker:
    def __init__(self, threshold: int, reset_seconds: float) -> None:
        self._threshold = max(1, threshold)
        self._reset_seconds = max(1.0, reset_seconds)
        self._failure_count = 0
        self._opened_until = 0.0
        self._lock = asyncio.Lock()

    async def before_request(self, endpoint: str) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._opened_until > now:
                metrics.increment_mcp_circuit_open(endpoint=endpoint)
                raise CircuitOpenError(f"Circuit breaker open for endpoint '{endpoint}'")
            if self._opened_until and now >= self._opened_until:
                # Reset breaker after cool-down
                self._failure_count = 0
                self._opened_until = 0.0

    async def record_success(self) -> None:
        async with self._lock:
            self._failure_count = 0
            self._opened_until = 0.0

    async def record_failure(self, endpoint: str) -> None:
        async with self._lock:
            self._failure_count += 1
            if self._failure_count >= self._threshold:
                self._opened_until = time.monotonic() + self._reset_seconds
                self._failure_count = 0
                metrics.increment_mcp_circuit_trip(endpoint=endpoint)

    def status(self) -> dict[str, float | int | bool]:
        opened_for = max(0.0, self._opened_until - time.monotonic()) if self._opened_until else 0.0
        return {
            "is_open": self._opened_until > time.monotonic(),
            "seconds_until_close": opened_for if opened_for > 0 else 0.0,
            "failure_streak": self._failure_count,
        }


class MCPClient:
    RETRY_STATUS_RANGES = ((500, 599),)
    RETRY_STATUS_CODES = {408, 409, 425, 429, 502, 503, 504}

    def __init__(self, config: MCPClientConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.timeout_seconds),
            headers=config.default_headers,
            verify=config.verify_ssl,
        )
        self._breaker = _CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            reset_seconds=config.circuit_breaker_reset_seconds,
        )
        self._hooks = tuple(config.instrumentation_hooks or ())

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        trace_id: str | None = None,
    ) -> httpx.Response:
        endpoint_label = self._normalise_endpoint(path)
        await self._breaker.before_request(endpoint_label)

        attempt = 0
        backoff = self._config.retry_backoff_seconds
        last_exception: Exception | None = None
        while True:
            attempt += 1
            dynamic_headers = await self._resolve_auth_headers()
            signature_headers = await self._resolve_signature_headers(method, path, json, params)
            request_headers = self._compose_headers(headers, trace_id, dynamic_headers, signature_headers)
            attempt_context = {
                "method": method.upper(),
                "path": path,
                "endpoint": endpoint_label,
                "attempt": attempt,
                "max_retries": self._config.max_retries,
                "trace_id": trace_id,
            }
            self._emit_instrumentation("request.start", attempt_context)
            start = time.perf_counter()
            try:
                response = await self._client.request(
                    method,
                    path,
                    json=json,
                    params=params,
                    headers=request_headers,
                )
                latency = time.perf_counter() - start
                if self._should_retry_response(response) and attempt <= self._config.max_retries:
                    await self._breaker.record_failure(endpoint_label)
                    metrics.observe_mcp_request(
                        method=method,
                        endpoint=endpoint_label,
                        status=response.status_code,
                        success=False,
                        latency=latency,
                    )
                    metrics.increment_mcp_retry(
                        method=method,
                        endpoint=endpoint_label,
                        reason=f"status_{response.status_code}",
                    )
                    retry_context = {
                        **attempt_context,
                        "status": response.status_code,
                        "latency": latency,
                        "retry_in": backoff,
                    }
                    self._emit_instrumentation("request.retry", retry_context)
                    await asyncio.sleep(self._backoff_with_jitter(backoff))
                    backoff *= 2
                    continue

                await self._breaker.record_success()
                metrics.observe_mcp_request(
                    method=method,
                    endpoint=endpoint_label,
                    status=response.status_code,
                    success=response.is_success,
                    latency=latency,
                )
                success_context = {
                    **attempt_context,
                    "status": response.status_code,
                    "latency": latency,
                }
                self._emit_instrumentation("request.success", success_context)
                return response
            except httpx.RequestError as exc:
                latency = time.perf_counter() - start
                await self._breaker.record_failure(endpoint_label)
                metrics.observe_mcp_request(
                    method=method,
                    endpoint=endpoint_label,
                    status=None,
                    success=False,
                    latency=latency,
                )
                last_exception = exc
                error_context = {
                    **attempt_context,
                    "error": str(exc),
                    "latency": latency,
                }
                should_retry = attempt <= self._config.max_retries
                if should_retry:
                    metrics.increment_mcp_retry(method=method, endpoint=endpoint_label, reason="exception")
                    error_context["retry_in"] = backoff
                    self._emit_instrumentation("request.error", error_context)
                else:
                    self._emit_instrumentation("request.failed", error_context)
                if attempt > self._config.max_retries:
                    raise
                await asyncio.sleep(self._backoff_with_jitter(backoff))
                backoff *= 2

    async def _resolve_auth_headers(self) -> dict[str, str]:
        provider = self._config.auth_header_provider
        if not provider:
            return {}
        try:
            value = provider()
            if inspect.isawaitable(value):
                value = await value
            return dict(value or {})
        except Exception as exc:  # pragma: no cover - defensive logging
            self._emit_instrumentation("auth_provider.error", {"error": str(exc)})
            raise

    async def _resolve_signature_headers(
        self,
        method: str,
        path: str,
        payload: Any | None,
        params: dict[str, Any] | None,
    ) -> dict[str, str]:
        signer = self._config.request_signer
        if not signer:
            return {}
        try:
            value = signer(method.upper(), self._normalise_endpoint(path), payload, params)
            if inspect.isawaitable(value):
                value = await value
            return dict(value or {})
        except Exception as exc:  # pragma: no cover - defensive logging
            self._emit_instrumentation("request_signer.error", {"error": str(exc)})
            raise

    def _compose_headers(
        self,
        headers: dict[str, str] | None,
        trace_id: str | None,
        dynamic_headers: dict[str, str],
        signature_headers: dict[str, str],
    ) -> dict[str, str]:
        merged = dict(self._config.default_headers)
        if dynamic_headers:
            merged.update(dynamic_headers)
        if headers:
            merged.update(headers)
        if trace_id and "X-Trace-Id" not in merged:
            merged["X-Trace-Id"] = trace_id
        if signature_headers:
            merged.update(signature_headers)
        return merged

    def _should_retry_response(self, response: httpx.Response) -> bool:
        if response.status_code in self.RETRY_STATUS_CODES:
            return True
        for lower, upper in self.RETRY_STATUS_RANGES:
            if lower <= response.status_code <= upper:
                return True
        return False

    def _backoff_with_jitter(self, base_backoff: float) -> float:
        jitter = random.uniform(0.0, self._config.retry_jitter_seconds)
        return max(0.0, base_backoff + jitter)

    @staticmethod
    def _normalise_endpoint(path: str) -> str:
        if not path:
            return "root"
        if path.startswith("/"):
            return path
        return f"/{path}"

    def _emit_instrumentation(self, event: str, payload: dict[str, Any]) -> None:
        data = dict(payload)
        data["event"] = event
        for hook in self._hooks:
            try:
                hook(dict(data))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("mcp_instrumentation_hook_failed", hook_event=event, error=str(exc))
        log_payload = dict(data)
        if "event" in log_payload:
            log_payload["mcp_event"] = log_payload.pop("event")
        logger.debug("mcp_client_event", **log_payload)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "base_url": self._config.base_url,
            "circuit": self._breaker.status(),
            "timeouts": {
                "request_seconds": self._config.timeout_seconds,
                "retry_backoff_seconds": self._config.retry_backoff_seconds,
                "max_retries": self._config.max_retries,
            },
            "auth": {
                "default_headers": sorted(self._config.default_headers.keys()),
                "dynamic_provider": bool(self._config.auth_header_provider),
                "request_signer": bool(self._config.request_signer),
            },
        }


__all__ = ["MCPClient", "MCPClientConfig", "CircuitOpenError"]
