from __future__ import annotations

import asyncio
import json
import time
import base64
import hashlib
import hmac
import contextvars
import os
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Deque, Mapping, Sequence
from urllib.parse import quote, urlencode, urlparse

import httpx
from datetime import datetime, timezone

from app.core import metrics
from app.core.config import MCPToolSettings, get_settings
from app.core.logging import get_logger
from app.services.enterprise_playbook import (
    actions_from_notion,
    actions_from_policy,
    assemble_policy_document,
    derive_playbook_query,
    extract_policy_hints,
)
from app.services.mcp_client import CircuitOpenError, MCPClient, MCPClientConfig
from app.services.tool_onboarding import all_planned_tools
from app.tools.catalog_store import ToolCatalogEntry, tool_catalog_store
from app.tools.registry import tool_registry
from app.tools.validation import ToolPayloadValidationError, tool_payload_validator

logger = get_logger(name=__name__)


_tool_event_callback: contextvars.ContextVar[
    Callable[[dict[str, Any]], Awaitable[None]] | None
] = contextvars.ContextVar("tool_event_callback", default=None)


class ToolDisabledError(RuntimeError):
    """Raised when a tool is invoked while tooling is disabled."""


class ToolInvocationError(RuntimeError):
    """Raised when a tool invocation fails."""


class ToolConfigurationError(RuntimeError):
    """Raised when MCP tooling cannot be initialized."""


@dataclass(slots=True)
class MCPToolDescriptor:
    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    labels: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()


@dataclass(slots=True)
class ToolInvocationResult:
    tool: str
    payload: dict[str, Any]
    response: dict[str, Any]
    cached: bool
    latency: float
    resolved_tool: str


DEFAULT_TOOL_ALIASES: dict[str, str] = {
    "research.search": "search/tavily",
    "search/tavily": "search/duckduckgo",
    "research.arxiv": "research/arxiv",
    "research.wikipedia": "research/wikipedia",
    "research.doc_loader": "research/doc_loader",
    "research.qdrant": "research/qdrant",
    "research.summarizer": "research/summarizer",
    "finance.snapshot": "finance/yfinance",
    "finance.snapshot.alpha": "finance/alpha_vantage",
    "finance.snapshot.cached": "finance/cached_quotes",
    "finance.analytics": "finance/pandas",
    "finance.plot": "finance/plot",
    "finance.news": "finance/coingecko_news",
    "finance.csv": "finance/csv",
    "finance.sentiment": "finance/finbert",
    "creative.tonecheck": "creative/stylizer",
    "creative.tone_checker": "creative/tone_checker",
    "creative.transcribe": "creative/whisper_transcription",
    "creative.image": "creative/image_generator",
    "enterprise.playbook": "enterprise/playbook",
    "enterprise.policy": "enterprise/policy_checker",
    "enterprise.notion": "enterprise/notion",
    "enterprise.calendar": "enterprise/calendar",
    "enterprise.crm": "enterprise/crm",
}


class _RollingWindowRateLimiter:
    """Simple async-safe rolling window rate limiter."""

    def __init__(self, max_calls: int, period_seconds: int) -> None:
        self._max_calls = max_calls
        self._period = period_seconds
        self._timestamps: Deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            await self._prune_locked()
            if len(self._timestamps) < self._max_calls:
                self._timestamps.append(time.monotonic())
                return
            oldest = self._timestamps[0]
            wait_for = (oldest + self._period) - time.monotonic()
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        async with self._lock:
            await self._prune_locked()
            self._timestamps.append(time.monotonic())

    async def _prune_locked(self) -> None:
        boundary = time.monotonic() - self._period
        while self._timestamps and self._timestamps[0] < boundary:
            self._timestamps.popleft()


class _ResponseCache:
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._storage: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> dict[str, Any] | None:
        if self._ttl <= 0:
            return None
        async with self._lock:
            entry = self._storage.get(key)
            if not entry:
                return None
            timestamp, value = entry
            if time.monotonic() - timestamp > self._ttl:
                del self._storage[key]
                return None
            return value

    async def set(self, key: str, value: dict[str, Any]) -> None:
        if self._ttl <= 0:
            return
        async with self._lock:
            self._storage[key] = (time.monotonic(), value)


MAX_PAYLOAD_STRING_LENGTH = 4096


class ToolService:
    def __init__(self, settings: MCPToolSettings, http_client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._cache = _ResponseCache(settings.cache_ttl_seconds)
        self._rate_limiter = _RollingWindowRateLimiter(
            max_calls=settings.rate_limit.max_calls,
            period_seconds=settings.rate_limit.period_seconds,
        )
        self._alias_mapping = {**DEFAULT_TOOL_ALIASES, **(settings.aliases or {})}
        self._apply_finance_snapshot_override()
        self._register_aliases(self._alias_mapping)
        self._catalog: dict[str, MCPToolDescriptor] = {}
        self._catalog_expiry: float = 0.0
        self._catalog_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._last_client_event: dict[str, Any] | None = None
        self._http_client = http_client
        self._client = MCPClient(
            MCPClientConfig(
                base_url=settings.endpoint,
                timeout_seconds=settings.timeout_seconds,
                max_retries=settings.max_retries,
                retry_backoff_seconds=settings.retry_backoff_seconds,
                retry_jitter_seconds=settings.retry_jitter_seconds,
                verify_ssl=settings.verify_ssl,
                default_headers=self._build_default_headers(),
                circuit_breaker_threshold=settings.circuit_breaker_threshold,
                circuit_breaker_reset_seconds=settings.circuit_breaker_reset_seconds,
                auth_header_provider=self._build_auth_header_provider(),
                request_signer=self._build_request_signer(),
                instrumentation_hooks=(self._capture_client_event,),
            ),
            client=http_client,
        )
        self._last_health_status: str = "unknown"
        self._last_health_timestamp: float | None = None
        self._last_health_error: str | None = None
        self._last_error: str | None = None
        self._last_invocation: dict[str, Any] | None = None
        self._last_catalog_refresh: float | None = None
        self._payload_string_limit = MAX_PAYLOAD_STRING_LENGTH

    def _apply_finance_snapshot_override(self) -> None:
        raw_value = os.getenv("FINANCE_SNAPSHOT_PROVIDER") or ""
        providers = [segment.strip().lower() for segment in raw_value.split(",") if segment.strip()]
        if not providers:
            return
        provider_mapping = {
            "alpha_vantage": "finance/alpha_vantage",
            "alpha": "finance/alpha_vantage",
            "yfinance": "finance/yfinance",
            "yahoo": "finance/yfinance",
        }
        resolved: list[str] = []
        for provider in providers:
            target = provider_mapping.get(provider)
            if target and target not in resolved:
                resolved.append(target)
        if not resolved:
            return
        primary = resolved[0]
        self._alias_mapping["finance.snapshot"] = primary

        fallback = next((candidate for candidate in resolved[1:] if candidate != primary), None)
        if fallback is None:
            default_fallback = (
                "finance/yfinance" if primary == "finance/alpha_vantage" else "finance/alpha_vantage"
            )
            fallback = default_fallback if default_fallback != primary else None
        if fallback:
            self._alias_mapping["finance.snapshot.alpha"] = fallback

    def _register_aliases(self, mapping: Mapping[str, str]) -> None:
        for alias, target in mapping.items():
            tool_registry.register_alias(alias, target)

    @asynccontextmanager
    async def instrument(
        self,
        callback: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> AsyncIterator["ToolService"]:
        token = _tool_event_callback.set(callback)
        try:
            yield self
        finally:
            _tool_event_callback.reset(token)

    async def _emit_tool_event(self, payload: dict[str, Any]) -> None:
        callback = _tool_event_callback.get()
        if callback is None:
            return
        try:
            await callback(payload)
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("tool_event_callback_failed")

    async def initialize(self, *, validate: bool = True) -> None:
        if self._initialized and not validate:
            return
        async with self._init_lock:
            if self._initialized and not validate:
                return
            if not self._settings.enabled:
                self._initialized = True
                return
            if validate:
                await self._perform_health_check()
                await self.refresh_catalog(force=True)
            self._initialized = True

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._http_client is not None:
            await self._http_client.aclose()

    async def refresh_catalog(self, *, force: bool = False) -> dict[str, MCPToolDescriptor]:
        if not self._settings.enabled:
            return {}
        async with self._catalog_lock:
            now = time.monotonic()
            if not force and self._settings.catalog_refresh_seconds > 0 and now < self._catalog_expiry:
                return self._catalog
            descriptors = await self._fetch_catalog()
            self._catalog = {descriptor.name: descriptor for descriptor in descriptors}
            remote_aliases = {
                alias: descriptor.name
                for descriptor in descriptors
                for alias in descriptor.aliases
            }
            combined_aliases = dict(remote_aliases)
            combined_aliases.update(self._alias_mapping)
            if combined_aliases:
                self._register_aliases(combined_aliases)
            self._materialize_catalog_snapshot(descriptors, combined_aliases)
            if self._settings.catalog_refresh_seconds > 0:
                self._catalog_expiry = now + self._settings.catalog_refresh_seconds
            return self._catalog

    async def list_tools(self, *, force_refresh: bool = False) -> list[MCPToolDescriptor]:
        await self.initialize(validate=force_refresh)
        await self.refresh_catalog(force=force_refresh)
        return list(self._catalog.values())

    async def invoke(self, tool: str, payload: dict[str, Any]) -> ToolInvocationResult:
        if not self._settings.enabled:
            raise ToolDisabledError("Tooling layer is disabled by configuration.")

        await self.initialize(validate=False)
        if tool == "enterprise.playbook":
            return await self._invoke_enterprise_playbook(payload)

        return await self._invoke_standard(tool, payload)

    async def _invoke_standard(self, tool: str, payload: dict[str, Any]) -> ToolInvocationResult:
        resolved_tool = self._resolve_tool_identifier(tool)
        await self._ensure_tool_known(resolved_tool)

        cache_key = self._cache_key(resolved_tool, payload)
        cached_response = await self._cache.get(cache_key)
        if cached_response is not None:
            metrics.observe_tool_invocation(tool=tool, latency=0.0, cached=True)
            result = ToolInvocationResult(
                tool=tool,
                resolved_tool=resolved_tool,
                payload=payload,
                response=cached_response,
                cached=True,
                latency=0.0,
            )
            self._record_invocation(result)
            await self._emit_tool_event(
                {
                    "tool": tool,
                    "resolved_tool": resolved_tool,
                    "status": "success",
                    "cached": True,
                    "latency": 0.0,
                    "payload_keys": self._payload_keyset(payload),
                }
            )
            return result

        self._validate_payload(resolved_tool, payload)
        await self._rate_limiter.acquire()
        start = time.perf_counter()
        try:
            response = await self._dispatch(resolved_tool=resolved_tool, payload=payload)
        except ToolInvocationError as exc:  # pragma: no cover - passthrough for metrics
            metrics.increment_tool_error(tool=tool)
            metrics.increment_tool_call_failure(tool=tool, reason=type(exc).__name__)
            self._last_error = str(exc)
            await self._emit_tool_event(
                {
                    "tool": tool,
                    "resolved_tool": resolved_tool,
                    "status": "error",
                    "cached": False,
                    "error": str(exc),
                    "payload_keys": self._payload_keyset(payload),
                }
            )
            raise
        except httpx.HTTPError as exc:  # pragma: no cover - defensive guard
            metrics.increment_tool_error(tool=tool)
            metrics.increment_tool_call_failure(tool=tool, reason=type(exc).__name__)
            error_message = str(exc)
            self._last_error = error_message
            await self._emit_tool_event(
                {
                    "tool": tool,
                    "resolved_tool": resolved_tool,
                    "status": "error",
                    "cached": False,
                    "error": error_message,
                    "payload_keys": self._payload_keyset(payload),
                }
            )
            raise ToolInvocationError(f"Tool '{tool}' invocation failed") from exc

        latency = time.perf_counter() - start
        await self._cache.set(cache_key, response)
        metrics.observe_tool_invocation(tool=tool, latency=latency, cached=False)
        result = ToolInvocationResult(
            tool=tool,
            resolved_tool=resolved_tool,
            payload=payload,
            response=response,
            cached=False,
            latency=latency,
        )
        self._record_invocation(result)
        await self._emit_tool_event(
            {
                "tool": tool,
                "resolved_tool": resolved_tool,
                "status": "success",
                "cached": False,
                "latency": latency,
                "payload_keys": self._payload_keyset(payload),
            }
        )
        return result

    def _validate_payload(self, resolved_tool: str, payload: dict[str, Any]) -> None:
        descriptor = self._catalog.get(resolved_tool)
        catalog_entry = tool_catalog_store.entry_for(resolved_tool)
        try:
            tool_payload_validator.validate(
                resolved_tool=resolved_tool,
                payload=payload,
                descriptor=descriptor,
                catalog_entry=catalog_entry,
                max_string_length=self._payload_string_limit,
            )
        except ToolPayloadValidationError as exc:
            raise ToolInvocationError(str(exc)) from exc

    async def _invoke_enterprise_playbook(self, payload: dict[str, Any]) -> ToolInvocationResult:
        resolved_tool = "enterprise/playbook"
        cache_key = self._cache_key(resolved_tool, payload)
        cached_response = await self._cache.get(cache_key)
        if cached_response is not None:
            metrics.observe_tool_invocation(tool="enterprise.playbook", latency=0.0, cached=True)
            result = ToolInvocationResult(
                tool="enterprise.playbook",
                resolved_tool=resolved_tool,
                payload=payload,
                response=cached_response,
                cached=True,
                latency=0.0,
            )
            self._record_invocation(result)
            await self._emit_tool_event(
                {
                    "tool": "enterprise.playbook",
                    "resolved_tool": resolved_tool,
                    "status": "success",
                    "cached": True,
                    "latency": 0.0,
                    "payload_keys": self._payload_keyset(payload),
                    "composite": True,
                }
            )
            return result

        query = derive_playbook_query(payload)
        start = time.perf_counter()
        notion_result: ToolInvocationResult | None = None
        notion_error: str | None = None
        policy_result: ToolInvocationResult | None = None
        policy_error: str | None = None

        try:
            notion_result = await self._invoke_standard(
                "enterprise.notion",
                {"action": "search", "query": query},
            )
        except ToolInvocationError as exc:
            notion_error = str(exc)

        actions = actions_from_notion(notion_result.response) if notion_result else []

        if not actions:
            document = assemble_policy_document(payload)
            policies = extract_policy_hints(payload)
            try:
                policy_result = await self._invoke_standard(
                    "enterprise.policy",
                    {"document": document, "policies": policies},
                )
            except ToolInvocationError as exc:
                policy_error = str(exc)
            else:
                actions = actions_from_policy(policy_result.response)

        if not actions:
            message = "Enterprise playbook returned no actionable guidance"
            if notion_error or policy_error:
                details = ", ".join(filter(None, [notion_error, policy_error]))
                message = f"{message}: {details}"
            await self._emit_tool_event(
                {
                    "tool": "enterprise.playbook",
                    "resolved_tool": resolved_tool,
                    "status": "error",
                    "cached": False,
                    "error": message,
                    "payload_keys": self._payload_keyset(payload),
                    "composite": True,
                }
            )
            raise ToolInvocationError(message)

        response_payload = {
            "query": query,
            "actions": actions,
            "notion": {
                "results": notion_result.response.get("results", []) if notion_result else [],
                "cached": notion_result.cached if notion_result else None,
                "error": notion_error,
            },
            "policy": {
                "findings": policy_result.response.get("findings", []) if policy_result else [],
                "compliant": policy_result.response.get("compliant") if policy_result else None,
                "cached": policy_result.cached if policy_result else None,
                "error": policy_error,
            },
        }

        latency = time.perf_counter() - start
        await self._cache.set(cache_key, response_payload)
        metrics.observe_tool_invocation(tool="enterprise.playbook", latency=latency, cached=False)
        result = ToolInvocationResult(
            tool="enterprise.playbook",
            resolved_tool=resolved_tool,
            payload=payload,
            response=response_payload,
            cached=False,
            latency=latency,
        )
        self._record_invocation(result)
        await self._emit_tool_event(
            {
                "tool": "enterprise.playbook",
                "resolved_tool": resolved_tool,
                "status": "success",
                "cached": False,
                "latency": latency,
                "payload_keys": self._payload_keyset(payload),
                "composite": True,
                "notion_error": notion_error,
                "policy_error": policy_error,
            }
        )
        return result

    @staticmethod
    def _payload_keyset(payload: dict[str, Any]) -> list[str]:
        return sorted(str(key) for key in payload.keys())

    async def _dispatch(self, *, resolved_tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._settings.invoke_path_template.format(tool=quote(resolved_tool, safe="/:"))
        try:
            response = await self._client.request("POST", path, json=payload)
            response.raise_for_status()
        except CircuitOpenError as exc:
            raise ToolInvocationError(f"Tool '{resolved_tool}' circuit breaker open") from exc
        except httpx.HTTPError as exc:
            raise ToolInvocationError(f"Tool '{resolved_tool}' HTTP failure") from exc

        data = response.json()
        if isinstance(data, dict):
            return data
        # Normalize non-dict responses for downstream consistency
        return {"result": data}

    async def _perform_health_check(self) -> None:
        health_path = self._settings.healthcheck_path
        if not health_path:
            return
        try:
            response = await self._client.request("GET", health_path)
            response.raise_for_status()
            self._last_health_status = "ok"
            self._last_health_timestamp = time.time()
            self._last_health_error = None
        except CircuitOpenError as exc:
            self._last_health_status = "circuit_open"
            self._last_health_timestamp = time.time()
            self._last_health_error = str(exc)
            raise ToolConfigurationError("MCP circuit breaker open during health check") from exc
        except httpx.HTTPError as exc:
            self._last_health_status = "error"
            self._last_health_timestamp = time.time()
            self._last_health_error = str(exc)
            raise ToolConfigurationError("Failed MCP health check") from exc

    async def _fetch_catalog(self) -> list[MCPToolDescriptor]:
        try:
            response = await self._client.request("GET", self._settings.catalog_path)
            response.raise_for_status()
            self._last_catalog_refresh = time.time()
        except CircuitOpenError as exc:
            self._last_error = str(exc)
            raise ToolConfigurationError("MCP circuit breaker open while loading catalog") from exc
        except httpx.HTTPError as exc:
            self._last_error = str(exc)
            raise ToolConfigurationError("Unable to load MCP tool catalog") from exc

        payload = response.json()
        entries = payload.get("tools") if isinstance(payload, dict) else payload
        descriptors: list[MCPToolDescriptor] = []
        if isinstance(entries, list):
            for raw in entries:
                if not isinstance(raw, dict):
                    continue
                name = raw.get("name")
                if not isinstance(name, str):
                    continue
                descriptors.append(
                    MCPToolDescriptor(
                        name=name,
                        description=str(raw.get("description") or ""),
                        input_schema=raw.get("input_schema") if isinstance(raw.get("input_schema"), dict) else None,
                        output_schema=raw.get("output_schema") if isinstance(raw.get("output_schema"), dict) else None,
                        labels=tuple(label for label in raw.get("labels", []) if isinstance(label, str)),
                        aliases=tuple(alias for alias in raw.get("aliases", []) if isinstance(alias, str)),
                        capabilities=tuple(cap for cap in raw.get("capabilities", []) if isinstance(cap, str)),
                    )
                )
        else:
            logger.debug("mcp_catalog_unexpected_payload", payload=payload)
        return descriptors

    def _materialize_catalog_snapshot(
        self,
        descriptors: Sequence[MCPToolDescriptor],
        alias_map: Mapping[str, str],
    ) -> None:
        entries = [
            ToolCatalogEntry(
                name=descriptor.name,
                description=descriptor.description,
                labels=descriptor.labels,
                aliases=descriptor.aliases,
                capabilities=descriptor.capabilities,
                input_schema=descriptor.input_schema or {},
                output_schema=descriptor.output_schema or {},
            )
            for descriptor in descriptors
        ]
        snapshot = tool_catalog_store.sync(entries, aliases=alias_map, source="mcp_catalog_refresh")
        tool_catalog_store.persist(snapshot)

    async def _ensure_tool_known(self, resolved_tool: str) -> None:
        await self.refresh_catalog(force=False)
        if self._catalog and resolved_tool not in self._catalog:
            logger.warning("mcp_tool_not_listed", tool=resolved_tool)

    def _build_default_headers(self) -> dict[str, str]:
        return dict(self._settings.extra_headers)

    def _build_auth_header_provider(self):
        header_name = (self._settings.api_key_header or "Authorization").strip() or "Authorization"
        scheme = (self._settings.auth_scheme or "").strip()
        api_key = self._settings.api_key
        client_id = self._settings.client_id
        client_secret = self._settings.client_secret

        if scheme.lower() == "basic" and client_id and client_secret:
            basic_token = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")

            async def _basic_provider() -> dict[str, str]:
                return {header_name: f"Basic {basic_token}"}

            return _basic_provider

        if api_key:

            async def _bearer_provider() -> dict[str, str]:
                value = f"{scheme} {api_key}".strip() if scheme else api_key
                return {header_name: value}

            return _bearer_provider

        return None

    def _build_request_signer(self):
        secret = self._settings.signing_secret
        if not secret:
            return None
        header_name = (self._settings.signing_header or "X-MCP-Signature").strip() or "X-MCP-Signature"
        algorithm = self._settings.signing_algorithm
        if algorithm != "hmac-sha256":
            raise ToolConfigurationError(f"Unsupported signing algorithm: {algorithm}")

        secret_bytes = secret.encode("utf-8")

        def _canonical_json(value: Any | None) -> str:
            if value is None:
                return ""
            if isinstance(value, (str, int, float, bool)):
                return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
            return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

        def _canonical_params(params: dict[str, Any] | None) -> str:
            if not params:
                return ""
            flattened = []
            for key in sorted(params):
                value = params[key]
                flattened.append((key, json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value))
            return urlencode(flattened, doseq=True)

        def _sign(method: str, endpoint: str, payload: Any | None, params: dict[str, Any] | None) -> dict[str, str]:
            canonical = "\n".join(
                [
                    method.upper(),
                    endpoint,
                    _canonical_params(params),
                    _canonical_json(payload),
                ]
            )
            digest = hmac.new(secret_bytes, canonical.encode("utf-8"), hashlib.sha256).digest()
            signature = base64.b64encode(digest).decode("ascii")
            return {header_name: signature}

        return _sign

    def _record_invocation(self, result: ToolInvocationResult) -> None:
        self._last_invocation = {
            "tool": result.tool,
            "resolved": result.resolved_tool,
            "cached": result.cached,
            "latency": result.latency,
            "timestamp": time.time(),
        }
        self._last_error = None

    def _capture_client_event(self, payload: dict[str, Any]) -> None:
        event_payload = dict(payload)
        event_payload["timestamp"] = time.time()
        self._last_client_event = event_payload

    def get_diagnostics(self) -> dict[str, Any]:
        client_diag = self._client.diagnostics()
        return {
            "enabled": self._settings.enabled,
            "endpoint": self._settings.endpoint,
            "catalog_size": len(self._catalog),
            "aliases": tool_registry.aliases(),
            "last_health": {
                "status": self._last_health_status,
                "timestamp": self._iso_or_none(self._last_health_timestamp),
                "error": self._last_health_error,
            },
            "last_catalog_refresh": self._iso_or_none(self._last_catalog_refresh),
            "last_error": self._last_error,
            "last_invocation": self._format_last_invocation(),
            "circuit": client_diag["circuit"],
            "client": {
                "base_url": client_diag["base_url"],
                "timeouts": client_diag["timeouts"],
                "auth": client_diag.get("auth", {}),
            },
            "last_client_event": self._format_last_client_event(),
            "onboarding": self._format_onboarding_status(),
        }

    def _format_last_invocation(self) -> dict[str, Any] | None:
        if self._last_invocation is None:
            return None
        payload = dict(self._last_invocation)
        payload["timestamp"] = self._iso_or_none(payload.get("timestamp"))
        return payload

    def _format_last_client_event(self) -> dict[str, Any] | None:
        if self._last_client_event is None:
            return None
        payload = dict(self._last_client_event)
        payload["timestamp"] = self._iso_or_none(payload.get("timestamp"))
        return payload

    def _format_onboarding_status(self) -> dict[str, list[dict[str, Any]]]:
        catalog_keys = set(self._catalog.keys())
        status: dict[str, list[dict[str, Any]]] = {}
        for planned in all_planned_tools():
            alias_resolution = self._resolve_tool_identifier(planned.alias)
            resolved = alias_resolution or self._resolve_tool_identifier(planned.resolved)
            catalog_key = self._resolve_tool_identifier(planned.resolved)
            alias_registered = tool_registry.get(planned.alias) is not None or alias_resolution is not None
            status.setdefault(planned.category, []).append(
                {
                    "alias": planned.alias,
                    "resolved": resolved,
                    "expected_resolved": planned.resolved,
                    "alias_registered": alias_registered,
                    "catalog_present": catalog_key in catalog_keys,
                    "description": planned.description,
                }
            )
        return status

    @staticmethod
    def _cache_key(tool: str, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return f"{tool}:{serialized}"

    def _resolve_tool_identifier(self, tool: str) -> str:
        adapter = tool_registry.get(tool)
        if adapter is not None:
            canonical_name = getattr(adapter, "name", None)
            if isinstance(canonical_name, str) and canonical_name:
                return canonical_name
            return tool_registry.resolve(tool) or tool

        alias_map = tool_registry.aliases()
        current = tool
        seen: set[str] = set()
        while current in alias_map and current not in seen:
            seen.add(current)
            candidate = alias_map[current]
            candidate_adapter = tool_registry.get(candidate)
            if candidate_adapter is not None:
                canonical_name = getattr(candidate_adapter, "name", None)
                if isinstance(canonical_name, str) and canonical_name:
                    return canonical_name
                return tool_registry.resolve(candidate) or candidate
            current = candidate

        if current.startswith("mcp://"):
            parsed = urlparse(current)
            segments = [parsed.netloc]
            path = parsed.path.lstrip("/")
            if path:
                segments.append(path)
            return "/".join(segment for segment in segments if segment)
        if "." in current and "/" not in current:
            return current.replace(".", "/")
        return current

    @staticmethod
    def _iso_or_none(ts: float | None) -> str | None:
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


_tool_service: ToolService | None = None
_tool_service_lock = asyncio.Lock()


async def get_tool_service() -> ToolService:
    global _tool_service
    if _tool_service is not None:
        return _tool_service
    async with _tool_service_lock:
        if _tool_service is None:
            settings = get_settings().tools.mcp
            http_client: httpx.AsyncClient | None = None
            if settings.use_local_router:
                from app.main import app as fastapi_app  # lazy import to avoid circular dependency

                transport = httpx.ASGITransport(app=fastapi_app)
                http_client = httpx.AsyncClient(
                    transport=transport,
                    base_url=settings.endpoint.rstrip("/"),
                    timeout=httpx.Timeout(settings.timeout_seconds),
                    headers=dict(settings.extra_headers),
                    verify=settings.verify_ssl,
                )

            service = ToolService(settings, http_client=http_client)
            try:
                await service.initialize(validate=True)
            except Exception:
                await service.aclose()
                raise
            _tool_service = service
        return _tool_service
