from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.mcp.router import bootstrap_tool_registry

from .registry import tool_registry

__all__ = ["router"]

router = APIRouter(tags=["tools"])


@router.get("/health", summary="Tool subsystem health check")
async def health() -> dict[str, object]:
    bootstrap_tool_registry()
    return {
        "status": "ok",
        "registered_tools": tool_registry.list(),
        "aliases": tool_registry.aliases(),
    }


@router.get("/metrics", summary="Tool metrics endpoint")
async def metrics() -> Response:
    bootstrap_tool_registry()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
