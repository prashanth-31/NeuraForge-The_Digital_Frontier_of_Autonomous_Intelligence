# MCP Diagnostics Runbook

_Date:_ 2025-11-14

## Purpose

Provide repeatable steps for validating Model Context Protocol integrations when tool invocations fail or catalog data drifts.

## Pre-Checks

1. Confirm MCP support enabled: `SETTINGS__TOOLS__MCP__ENABLED=true` in active deployment.
2. Retrieve configured endpoint: `settings.tools.mcp.endpoint` via `/api/v1/diagnostics/mcp`.
3. Ensure network path to MCP router is reachable (`curl -I <endpoint>/health`).

## Catalog Verification

- Trigger snapshot refresh:
  ```bash
  python -m app.tools.catalog_store --refresh
  ```
- Compare latest snapshot with planner expectations:
  ```bash
  python -m pytest tests/test_tool_catalog_phase2.py
  python -m pytest tests/test_tool_reconciliation.py
  ```
- If mismatches arise, inspect `implementation/backend/app/tools/catalog_store.py` log output for missing aliases or authentication errors.

## Invocation Smoke Test

1. Run backend regression suite for MCP interactions:
   ```bash
   python -m pytest tests/test_tool_endpoints.py -k mcp
   ```
2. Execute ad-hoc invocation using ASGI transport:
   ```python
   import asyncio
   from app.tools.api import invoke_tool
   async def main():
       result = await invoke_tool("research.search", {"query": "diagnostics"})
       print(result)
   asyncio.run(main())
   ```
3. Check response latency in Prometheus metric `neuraforge_mcp_tool_latency_seconds`.

## Failure Triage

- **Authentication Errors:** Validate API key via `/api/v1/diagnostics/mcp`. Rotate credentials stored in Jenkins secret `MCP_ROUTER_TOKEN` if expired.
- **Schema Drift:** Run `python -m pytest tests/test_planner_contract_fuzz.py` to confirm planner outputs remain compatible with MCP tool IDs.
- **Network Timeouts:** Increase `TOOLS__MCP__TIMEOUT_SECONDS` temporarily and monitor retries; ensure Alertmanager suppresses duplicates with route `team="ops-memory"`.
- **Circuit Breaks:** Review `neuraforge_mcp_circuit_open_total`. Clear breaker by restarting MCP router once root cause fixed.

## Documentation

- Archive diagnostics in `docs/reports/mcp_incident_{date}.md` including tool names, error payloads, and resolution steps.
- Update `plan_llm_orchestration.md` tooling section with any new aliases or fallback strategies.
