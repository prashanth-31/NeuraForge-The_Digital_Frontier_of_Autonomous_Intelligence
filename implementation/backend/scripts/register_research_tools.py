"""Register research MCP tools against the configured MCP router.

Usage:
    E:/NeuraForge-The_Digital_Frontier_of_Autonomous_Intelligence/implementation/backend/.venv/Scripts/python.exe scripts/register_research_tools.py

Environment requirements:
    TOOLS__MCP__ENDPOINT   Base URL to the MCP router (e.g. https://mcp-staging.neuraforge.ai)
    TOOLS__MCP__API_KEY    Bearer token with catalog admin privileges

The script is idempotent: it upserts each tool definition using PUT /tools/{name}.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

from app.services.tool_onboarding import RESEARCH_TOOLS


def _build_payload(tool_alias: str, description: str) -> dict[str, Any]:
    return {
        "name": tool_alias,
        "description": description,
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
    "labels": ["research", "open"],
    }


async def register_tools() -> None:
    base_url = os.environ.get("TOOLS__MCP__ENDPOINT")
    api_key = os.environ.get("TOOLS__MCP__API_KEY")
    if not base_url:
        raise RuntimeError("TOOLS__MCP__ENDPOINT is required")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=10.0) as client:
        for tool in RESEARCH_TOOLS:
            payload = _build_payload(tool.resolved, tool.description)
            response = await client.put(f"/tools/{tool.resolved}", content=json.dumps(payload))
            response.raise_for_status()
            print(f"Registered {tool.resolved}: {response.status_code}")


if __name__ == "__main__":
    asyncio.run(register_tools())
