# Tooling Remediation Plan

_Date: 2025-11-12_

High-level priorities (execute in order):
1. Tool name normalization & registry aliasing.
2. Adapter return normalization (guarantee JSON-safe dict via base adapter wrapper).
3. Health & metrics endpoints (`/health`, `/metrics`) for MCP services.
4. Unified timeout / retry / circuit-breaker handling for tool invocation.
5. Prometheus instrumentation (invocation counts, latency, outcomes).
6. Exception normalization (canonical `ToolError` hierarchy).
7. Registry discovery API + planner integration.
8. Idempotency support for side-effecting tools.
9. Unit + integration tests wired into CI.

---

## 1) Tool-name normalization (registry)

**Goal**  
Ensure planner, orchestrator, and adapters share the same canonical tool identifiers.

**Implementation**
- Create `backend/app/tools/registry.py`:
  ```python
  from typing import Dict, Any
  import re

  def normalize_tool_name(name: str) -> str:
    return re.sub(r"[/\s]+", ".", name.strip()).lower()

  class ToolRegistry:
    def __init__(self):
      self._registry: Dict[str, Any] = {}
      self._aliases: Dict[str, str] = {}

    def register(self, name: str, adapter: Any) -> None:
      key = normalize_tool_name(name)
      self._registry[key] = adapter
      self._aliases[name] = key
      self._aliases[name.replace(".", "/")] = key

    def get(self, name: str) -> Any | None:
      key = normalize_tool_name(name)
      return self._registry.get(key)

    def list(self) -> list[str]:
      return sorted(self._registry.keys())

  tool_registry = ToolRegistry()
  ```
- Update all adapter registrations and `ToolService.invoke` to use `tool_registry`.
- Planner prompt builder should enumerate `tool_registry.list()`.

**Tests**
- `backend/tests/test_tool_registry.py` covering slash/dot normalization and listing.

---

## 2) Adapter invoke wrapper + result normalization

**Goal**  
Guarantee adapters return JSON-serializable payloads, with retry/timeout wrapped in a shared base class.

**Implementation**
- Add `backend/app/tools/base.py`:
  ```python
  import asyncio, json, time
  from typing import Any, Dict
  from pydantic import BaseModel
  from prometheus_client import Counter, Histogram

  TOOL_INVOCATIONS = Counter("mcp_tool_invocations_total", "Tool invocations", ["tool", "outcome"])
  TOOL_LATENCY = Histogram("mcp_tool_latency_seconds", "Tool latency", ["tool"])

  class ToolInvocationError(Exception):
    pass

  class MCPToolAdapter:
    name = "unnamed.tool"

    async def _run(self, payload: Dict[str, Any]) -> Any:
      raise NotImplementedError

    async def invoke(self, payload: Dict[str, Any], *, timeout: float = 8.0, retries: int = 2) -> Dict[str, Any]:
      start = time.perf_counter()
      attempt = 0
      last_exc: Exception | None = None
      while attempt <= retries:
        try:
          res = await asyncio.wait_for(self._run(payload), timeout=timeout)
          if isinstance(res, BaseModel):
            out = res.model_dump()
          elif isinstance(res, dict):
            out = res
          else:
            try:
              json.dumps(res)
              out = res
            except Exception:
              out = {"result": str(res)}
          TOOL_INVOCATIONS.labels(tool=self.name, outcome="success").inc()
          TOOL_LATENCY.labels(tool=self.name).observe(time.perf_counter() - start)
          return {"resolved_tool": self.name, "result": out, "latency": time.perf_counter() - start, "cached": False}
        except asyncio.TimeoutError as exc:
          last_exc = exc
          TOOL_INVOCATIONS.labels(tool=self.name, outcome="timeout").inc()
        except Exception as exc:
          last_exc = exc
          TOOL_INVOCATIONS.labels(tool=self.name, outcome="failure").inc()
        attempt += 1
        await asyncio.sleep(0.5 * 2**attempt)
      TOOL_LATENCY.labels(tool=self.name).observe(time.perf_counter() - start)
      raise ToolInvocationError(f"{self.name} failed after {retries + 1} attempts: {last_exc}") from last_exc
  ```
- Wrap existing adapters with `MCPToolAdapter`.

**Tests**
- `test_tool_returns_json_serializable`, `test_invoke_retries` verifying fallback serialization and retry behaviour.

---

## 3) Health & metrics endpoints

**Goal**  
Expose status and metrics for MCP services.

**Implementation**
- Create `backend/app/tools/api.py` with FastAPI router:
  ```python
  from fastapi import APIRouter
  from fastapi.responses import Response
  from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
  from backend.app.tools.registry import tool_registry

  router = APIRouter()

  @router.get("/health")
  async def health():
    return {"status": "ok", "registered_tools": tool_registry.list()}

  @router.get("/metrics")
  async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
  ```
- Mount router in the MCP FastAPI app.

**Tests**
- `test_mcp_health_endpoint` ensures `/health` returns expected fields.

---

## 4) Timeouts, retries, circuit breaker

**Goal**  
Prevent stuck tool invocations and provide self-healing on repeated failures.

**Implementation**
- Extend `ToolRegistry` with failure tracking and circuit breaker state (open after N failures for X seconds).
- Within `MCPToolAdapter.invoke`, short-circuit if the registry reports the breaker is open; report success/failure to reset counters.
- Configurable defaults via `core/config.py` (`TOOL_INVOCATION_TIMEOUT`, `TOOL_INVOCATION_RETRIES`, `TOOL_CIRCUIT_BREAK_FAILS`, `TOOL_CIRCUIT_BREAK_SECONDS`).

**Tests**
- `test_timeouts_retries` simulating slow adapters.  
- `test_circuit_breaker_opens_and_recovers` validating breaker operation.

---

## 5) Prometheus instrumentation

**Goal**  
Capture invocation counts, latency, and outcome distribution per tool.

**Implementation**
- Metrics already defined in the base adapter; ensure `/metrics` exposes them and Grafana dashboards include panels for `mcp_tool_invocations_total` and `mcp_tool_latency_seconds`.

**Tests**
- `test_metrics_exposed` invoking a tool then scraping `/metrics` to confirm counters increment.

---

## 6) Exception normalization

**Goal**  
Provide consistent error types to orchestrator and callers.

**Implementation**
- Add `backend/app/tools/exceptions.py`:
  ```python
  class ToolError(Exception):
    pass

  class ToolTimeout(ToolError):
    pass

  class ToolNotFound(ToolError):
    pass
  ```
- Ensure adapters wrap backend-specific exceptions and re-raise these canonical types.  
- Orchestrator `_ToolSession` records failures based on `ToolError` hierarchy.

**Tests**
- `test_adapter_raises_toolerror` verifying orchestrator records failure metadata.  
- `test_tool_timeout_exception_path` ensures timeout maps to `ToolTimeout`.

---

## 7) Registry discovery & planner integration

**Goal**  
Keep planner prompts aligned with live tool registry.

**Implementation**
- `llm_planner.py` pulls `tool_registry.list()` and renders canonical tool descriptions in the prompt (name, brief summary, input/output schema).  
- Provide an API endpoint (optional) exposing registry metadata for ops visibility.

**Tests**
- `test_registry_listed_in_planner_prompt` ensuring prompt contains known tool names.  
- `test_tool_registry_list_api` for any new endpoint.

---

## 8) Idempotency for side-effecting tools

**Goal**  
Prevent duplicate side effects during retries.

**Implementation**
- Side-effecting adapters accept `idempotency_key` and persist results (Redis/Postgres) keyed by `{tool}:{key}`.  
- On invocation, return cached result if key already processed; otherwise execute and store the output.

**Tests**
- `test_tool_idempotency_key` verifying retry returns cached result without re-executing side effects.

---

## 9) Tests & CI integration

**Goal**  
Ensure new behaviours are covered and enforced.

**Implementation**
- Add unit and integration tests listed above under `backend/tests/`.  
- Update CI workflow (e.g., `.github/workflows/ci.yml`) to run the new suites and smoke-check `/health` + `/metrics` endpoints.

**Checklist**
- [ ] Tool registry resolves dot/slash names; planner prompt emits normalized names.  
- [ ] Adapters return JSON-safe dicts; orchestrator records metadata successfully.  
- [ ] `/health` and `/metrics` endpoints respond in dev/CI environments.  
- [ ] Timeouts, retries, and circuit breaker logic verified via tests.  
- [ ] Prometheus counters/histograms visible in metrics endpoint.  
- [ ] Canonical exceptions propagate through orchestrator telemetry.  
- [ ] Registry discovery wired into planner prompt builder.  
- [ ] Idempotency coverage for stateful tools.  
- [ ] All new tests run in CI and pass.

This plan replaces the earlier tooling remediation outline and should guide the implementation workstream for MCP tool hardening.
