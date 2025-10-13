from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

import httpx
from fastapi import APIRouter, Body, HTTPException, status
from pydantic import ValidationError

from .adapters.base import MCPToolAdapter
from .adapters.creative import CREATIVE_ADAPTER_CLASSES
from .adapters.enterprise import ENTERPRISE_ADAPTER_CLASSES
from .adapters.finance import FINANCE_ADAPTER_CLASSES
from .adapters.research import RESEARCH_ADAPTER_CLASSES

router = APIRouter(prefix="/mcp", tags=["mcp"])


def _instantiate_adapters(
    adapter_classes: Iterable[type[MCPToolAdapter]],
) -> dict[str, MCPToolAdapter]:
    registry: dict[str, MCPToolAdapter] = {}
    for adapter_cls in adapter_classes:
        name = adapter_cls.name
        if name in registry:
            raise RuntimeError(f"Duplicate MCP tool name detected: {name}")
        registry[name] = adapter_cls()
    return registry


ALL_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    RESEARCH_ADAPTER_CLASSES
    + FINANCE_ADAPTER_CLASSES
    + CREATIVE_ADAPTER_CLASSES
    + ENTERPRISE_ADAPTER_CLASSES
)


@lru_cache
def _get_adapters() -> dict[str, MCPToolAdapter]:
    return _instantiate_adapters(ALL_ADAPTER_CLASSES)


@router.get("/health", summary="MCP health check")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/tools", summary="List available MCP tools")
async def list_tools() -> dict[str, Any]:
    descriptors: list[dict[str, Any]] = []
    for adapter in _get_adapters().values():
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


@router.post("/tools/{tool_name}/invoke", summary="Invoke MCP tool")
async def invoke_tool(
    tool_name: str,
    payload: dict[str, Any] = Body(..., embed=False),
) -> Any:
    adapter = _get_adapters().get(tool_name)
    if adapter is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    try:
        return await adapter.invoke(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - safety net for unexpected adapter errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
