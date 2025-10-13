from __future__ import annotations

import asyncio
import json
import time
import base64
import hashlib
import hmac
import contextvars
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Deque
from urllib.parse import quote, urlencode, urlparse

import httpx
from datetime import datetime, timezone

from app.core import metrics
from app.core.config import MCPToolSettings, get_settings
from app.core.logging import get_logger
from app.services.mcp_client import CircuitOpenError, MCPClient, MCPClientConfig
from app.services.tool_onboarding import all_planned_tools

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
    "research.arxiv": "research/arxiv",
    "research.wikipedia": "research/wikipedia",
    "research.doc_loader": "research/doc_loader",
    "research.qdrant": "research/qdrant",
    "research.summarizer": "research/summarizer",
    "finance.snapshot": "finance/yfinance",
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


class ToolService:
    def __init__(self, settings: MCPToolSettings) -> None:
        self._settings = settings
        self._cache = _ResponseCache(settings.cache_ttl_seconds)
        self._rate_limiter = _RollingWindowRateLimiter(
            max_calls=settings.rate_limit.max_calls,
            period_seconds=settings.rate_limit.period_seconds,
        )
        self._aliases = {**DEFAULT_TOOL_ALIASES, **(settings.aliases or {})}
        self._catalog: dict[str, MCPToolDescriptor] = {}
        self._catalog_expiry: float = 0.0
        self._catalog_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._last_client_event: dict[str, Any] | None = None
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
            )
        )
        self._last_health_status: str = "unknown"
        self._last_health_timestamp: float | None = None
        self._last_health_error: str | None = None
        self._last_error: str | None = None
        self._last_invocation: dict[str, Any] | None = None
        self._last_catalog_refresh: float | None = None

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

    async def refresh_catalog(self, *, force: bool = False) -> dict[str, MCPToolDescriptor]:
        if not self._settings.enabled:
            return {}
        async with self._catalog_lock:
            now = time.monotonic()
            if not force and self._settings.catalog_refresh_seconds > 0 and now < self._catalog_expiry:
                return self._catalog
            descriptors = await self._fetch_catalog()
            self._catalog = {descriptor.name: descriptor for descriptor in descriptors}
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

        await self._rate_limiter.acquire()
        start = time.perf_counter()
        try:
            response = await self._dispatch(resolved_tool=resolved_tool, payload=payload)
        except ToolInvocationError as exc:  # pragma: no cover - passthrough for metrics
            metrics.increment_tool_error(tool=tool)
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

        query = self._derive_playbook_query(payload)
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

        actions = self._actions_from_notion(notion_result.response) if notion_result else []

        if not actions:
            document = self._assemble_policy_document(payload)
            policies = self._extract_policy_hints(payload)
            try:
                policy_result = await self._invoke_standard(
                    "enterprise.policy",
                    {"document": document, "policies": policies},
                )
            except ToolInvocationError as exc:
                policy_error = str(exc)
            else:
                actions = self._actions_from_policy(policy_result.response)

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

    def _derive_playbook_query(self, payload: dict[str, Any]) -> str:
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in ("playbook_query", "topic", "keyword"):
                candidate = metadata.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()[:256]
        prompt = payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()[:256]
        return "enterprise strategy playbook"

    def _assemble_policy_document(self, payload: dict[str, Any]) -> str:
        sections: list[str] = []
        prompt = payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            sections.append(prompt.strip())
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and metadata:
            sections.append(json.dumps(metadata, sort_keys=True, indent=None))
        prior = payload.get("prior_outputs")
        if isinstance(prior, list):
            for item in prior[:5]:
                if not isinstance(item, dict):
                    continue
                agent = str(item.get("agent", "agent"))
                content = item.get("summary") or item.get("content") or ""
                if isinstance(content, str) and content.strip():
                    sections.append(f"{agent}: {content.strip()}")
        document = "\n\n".join(section for section in sections if section)
        if len(document) < 20:
            fallback = document or "Enterprise policy review context"
            document = (fallback + "\n" + fallback).strip()
        if len(document) < 20:
            document = document.ljust(20, ".")
        return document

    @staticmethod
    def _extract_policy_hints(payload: dict[str, Any]) -> list[str]:
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            policies = metadata.get("policies")
            if isinstance(policies, list):
                hints = [str(item).strip() for item in policies if isinstance(item, str) and item.strip()]
                if hints:
                    return hints
        return []

    @staticmethod
    def _actions_from_notion(response: dict[str, Any]) -> list[dict[str, Any]]:
        results = response.get("results") if isinstance(response, dict) else None
        if not isinstance(results, list):
            return []
        actions: list[dict[str, Any]] = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            title = entry.get("title") or entry.get("name") or entry.get("page_id") or "Playbook reference"
            snippet = entry.get("snippet") or entry.get("summary") or "Leverage documented best practice."
            action: dict[str, Any] = {
                "action": str(title),
                "impact": str(snippet),
                "origin": "notion",
            }
            page_id = entry.get("page_id")
            if isinstance(page_id, str):
                action["page_id"] = page_id
                action.setdefault("source", f"notion://{page_id}")
            url = entry.get("url") or entry.get("link")
            if isinstance(url, str):
                action["source"] = url
            actions.append(action)
        return actions

    @staticmethod
    def _actions_from_policy(response: dict[str, Any]) -> list[dict[str, Any]]:
        findings = response.get("findings") if isinstance(response, dict) else None
        actions: list[dict[str, Any]] = []
        if isinstance(findings, list):
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                status = str(finding.get("status", "")).lower()
                policy = finding.get("policy") or "policy"
                details = finding.get("details") or "Review policy guidance."
                if status and status != "pass":
                    actions.append(
                        {
                            "action": f"Mitigate policy risk: {policy}",
                            "impact": str(details),
                            "origin": "policy_checker",
                        }
                    )
        if actions:
            return actions
        return [
            {
                "action": "Confirm compliance readiness",
                "impact": "Policy checker returned no blocking findings.",
                "origin": "policy_checker",
            }
        ]

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
                    )
                )
        else:
            logger.debug("mcp_catalog_unexpected_payload", payload=payload)
        return descriptors

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
            "aliases": self._aliases,
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
            resolved = self._aliases.get(planned.alias, planned.resolved)
            status.setdefault(planned.category, []).append(
                {
                    "alias": planned.alias,
                    "resolved": resolved,
                    "expected_resolved": planned.resolved,
                    "alias_registered": planned.alias in self._aliases,
                    "catalog_present": resolved in catalog_keys,
                    "description": planned.description,
                }
            )
        return status

    @staticmethod
    def _cache_key(tool: str, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return f"{tool}:{serialized}"

    def _resolve_tool_identifier(self, tool: str) -> str:
        if tool in self._aliases:
            return self._aliases[tool]
        if tool.startswith("mcp://"):
            parsed = urlparse(tool)
            segments = [parsed.netloc]
            path = parsed.path.lstrip("/")
            if path:
                segments.append(path)
            return "/".join(segment for segment in segments if segment)
        if "." in tool and "/" not in tool:
            return tool.replace(".", "/")
        return tool

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
            service = ToolService(settings)
            try:
                await service.initialize(validate=True)
            except Exception:
                await service.aclose()
                raise
            _tool_service = service
        return _tool_service
