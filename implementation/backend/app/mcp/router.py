from __future__ import annotations

import time
import uuid
from functools import lru_cache
from typing import Any

import httpx
from fastapi import APIRouter, Body, HTTPException, status
from pydantic import ValidationError

from app.core.logging import get_logger
from app.tools.base import CircuitBreakerOpenError, ToolInvocationError
from app.tools.registry import tool_registry

from .adapters.base import MCPToolAdapter
from .adapters.creative import CREATIVE_ADAPTER_CLASSES
from .adapters.enterprise import ENTERPRISE_ADAPTER_CLASSES
from .adapters.finance import FINANCE_ADAPTER_CLASSES
from .adapters.research import RESEARCH_ADAPTER_CLASSES

router = APIRouter(prefix="/mcp", tags=["mcp"])
logger = get_logger(name=__name__)


ALL_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    RESEARCH_ADAPTER_CLASSES
    + FINANCE_ADAPTER_CLASSES
    + CREATIVE_ADAPTER_CLASSES
    + ENTERPRISE_ADAPTER_CLASSES
)


@lru_cache(maxsize=1)
def bootstrap_tool_registry() -> bool:
    for adapter_cls in ALL_ADAPTER_CLASSES:
        name = adapter_cls.name
        if tool_registry.get(name) is not None:
            continue
        adapter = adapter_cls()
        tool_registry.register(name, adapter)
    try:
        from app.services.tools import DEFAULT_TOOL_ALIASES
    except Exception:  # pragma: no cover - defensive import guard
        DEFAULT_TOOL_ALIASES = {}
    for alias, target in DEFAULT_TOOL_ALIASES.items():
        tool_registry.register_alias(alias, target)
    return True


@router.get("/health", summary="MCP health check")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/tools", summary="List available MCP tools")
async def list_tools() -> dict[str, Any]:
    bootstrap_tool_registry()
    descriptors: list[dict[str, Any]] = []
    for _, adapter in tool_registry.items():
        descriptor = adapter.descriptor()
        descriptors.append(
            {
                "name": descriptor.name,
                "description": descriptor.description,
                "input_schema": descriptor.input_schema,
                "output_schema": descriptor.output_schema,
                "labels": descriptor.labels,
            }
        )
    descriptors.sort(key=lambda payload: payload["name"])
    return {"tools": descriptors}


@router.post("/tools/{tool_name:path}/invoke", summary="Invoke MCP tool")
async def invoke_tool(
    tool_name: str,
    payload: dict[str, Any] = Body(..., embed=False),
) -> Any:
    bootstrap_tool_registry()
    adapter = tool_registry.get(tool_name)
    if adapter is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    trace_id = uuid.uuid4().hex
    started = time.perf_counter()
    canonical_name = getattr(adapter, "name", None) or tool_registry.resolve(tool_name) or tool_name
    logger.info(
        "mcp_tool_invocation_started",
        tool=canonical_name,
        requested=tool_name,
        trace_id=trace_id,
        payload_keys=sorted(payload.keys()),
    )

    try:
        result = await adapter.invoke(payload)
    except ValidationError as exc:
        duration = time.perf_counter() - started
        logger.warning(
            "mcp_tool_invocation_validation_error",
            tool=canonical_name,
            trace_id=trace_id,
            duration=duration,
            errors=len(exc.errors()),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        ) from exc
    except CircuitBreakerOpenError as exc:
        duration = time.perf_counter() - started
        logger.warning(
            "mcp_tool_circuit_open",
            tool=canonical_name,
            trace_id=trace_id,
            duration=duration,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ToolInvocationError as exc:
        duration = time.perf_counter() - started
        logger.warning(
            "mcp_tool_invocation_error",
            tool=canonical_name,
            trace_id=trace_id,
            duration=duration,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except httpx.HTTPError as exc:
        duration = time.perf_counter() - started
        logger.warning(
            "mcp_tool_invocation_http_error",
            tool=canonical_name,
            trace_id=trace_id,
            duration=duration,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - safety net for unexpected adapter errors
        duration = time.perf_counter() - started
        logger.exception(
            "mcp_tool_invocation_failed",
            tool=canonical_name,
            trace_id=trace_id,
            duration=duration,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    else:
        duration = time.perf_counter() - started
        logger.info(
            "mcp_tool_invocation_completed",
            tool=canonical_name,
            trace_id=trace_id,
            duration=duration,
            response_type=type(result).__name__,
        )
        return result
