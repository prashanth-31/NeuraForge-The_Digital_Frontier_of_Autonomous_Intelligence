from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Histogram

AGENT_LATENCY_SECONDS = Histogram(
    "neuraforge_agent_execution_latency_seconds",
    "Latency for each agent execution",
    labelnames=("agent",),
)

AGENT_EVENT_TOTAL = Counter(
    "neuraforge_agent_events_total",
    "Count of agent lifecycle events",
    labelnames=("agent", "event"),
)

MEMORY_CACHE_HITS_TOTAL = Counter(
    "neuraforge_memory_cache_hits_total",
    "Count of cache hits by layer",
    labelnames=("layer",),
)

MEMORY_CACHE_MISSES_TOTAL = Counter(
    "neuraforge_memory_cache_misses_total",
    "Count of cache misses by layer",
    labelnames=("layer",),
)

MEMORY_INGEST_TOTAL = Counter(
    "neuraforge_memory_ingest_records_total",
    "Number of records written to a memory store",
    labelnames=("store", "operation"),
)

MEMORY_CONSOLIDATION_DURATION_SECONDS = Histogram(
    "neuraforge_memory_consolidation_duration_seconds",
    "Duration of consolidation runs",
    labelnames=("status",),
)

MEMORY_CONSOLIDATION_ITEMS_TOTAL = Counter(
    "neuraforge_memory_consolidation_items_total",
    "Items processed during consolidation runs",
    labelnames=("status",),
)

RETRIEVAL_RESULTS_TOTAL = Counter(
    "neuraforge_retrieval_results_total",
    "Count of records returned during retrieval",
    labelnames=("source",),
)

CONTEXT_ASSEMBLY_CHARS = Histogram(
    "neuraforge_context_assembly_chars",
    "Character length of assembled context passed to agents",
    labelnames=("agent",),
)

TOOL_INVOCATIONS_TOTAL = Counter(
    "neuraforge_tool_invocations_total",
    "Number of tool invocations per tool",
    labelnames=("tool", "cached"),
)

TOOL_ERRORS_TOTAL = Counter(
    "neuraforge_tool_errors_total",
    "Tool invocation failures",
    labelnames=("tool",),
)

TOOL_LATENCY_SECONDS = Histogram(
    "neuraforge_tool_latency_seconds",
    "Latency for tool invocations",
    labelnames=("tool",),
)

MCP_REQUEST_TOTAL = Counter(
    "neuraforge_mcp_request_total",
    "Total MCP HTTP requests by endpoint and outcome",
    labelnames=("method", "endpoint", "outcome"),
)

MCP_REQUEST_LATENCY_SECONDS = Histogram(
    "neuraforge_mcp_request_latency_seconds",
    "Latency for MCP HTTP requests",
    labelnames=("method", "endpoint", "status"),
)

MCP_CIRCUIT_OPEN_TOTAL = Counter(
    "neuraforge_mcp_circuit_open_total",
    "Count of MCP circuit breaker blocks",
    labelnames=("endpoint",),
)

MCP_CIRCUIT_TRIP_TOTAL = Counter(
    "neuraforge_mcp_circuit_trip_total",
    "Count of MCP circuit breaker trips",
    labelnames=("endpoint",),
)

CONFIDENCE_COMPONENT_VALUE = Histogram(
    "neuraforge_confidence_component_value",
    "Observed component contributions to agent confidence scores",
    labelnames=("agent", "component"),
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def observe_agent_latency(*, agent: str, latency: float) -> None:
    AGENT_LATENCY_SECONDS.labels(agent=agent).observe(latency)


def increment_agent_event(*, agent: str, event: str) -> None:
    AGENT_EVENT_TOTAL.labels(agent=agent, event=event).inc()


def increment_cache_hit(*, layer: str) -> None:
    MEMORY_CACHE_HITS_TOTAL.labels(layer=layer).inc()


def increment_cache_miss(*, layer: str) -> None:
    MEMORY_CACHE_MISSES_TOTAL.labels(layer=layer).inc()


def increment_memory_ingest(*, store: str, operation: str) -> None:
    MEMORY_INGEST_TOTAL.labels(store=store, operation=operation).inc()


def observe_consolidation_run(*, status: str, duration: float, processed: int, embedded: int, skipped: int) -> None:
    MEMORY_CONSOLIDATION_DURATION_SECONDS.labels(status=status).observe(duration)
    if processed:
        MEMORY_CONSOLIDATION_ITEMS_TOTAL.labels(status="processed").inc(processed)
    if embedded:
        MEMORY_CONSOLIDATION_ITEMS_TOTAL.labels(status="embedded").inc(embedded)
    if skipped:
        MEMORY_CONSOLIDATION_ITEMS_TOTAL.labels(status="skipped").inc(skipped)


def increment_retrieval_results(*, source: str, count: int) -> None:
    if count:
        RETRIEVAL_RESULTS_TOTAL.labels(source=source).inc(count)


def observe_context_chars(*, agent: str, length: int) -> None:
    CONTEXT_ASSEMBLY_CHARS.labels(agent=agent).observe(length)


def bind_event_payload(agent: str, task_id: str, event: str, **extra: Any) -> dict[str, Any]:
    payload = {"agent": agent, "task_id": task_id, "event": event}
    payload.update(extra)
    return payload


def observe_tool_invocation(*, tool: str, latency: float, cached: bool) -> None:
    TOOL_INVOCATIONS_TOTAL.labels(tool=tool, cached=str(cached)).inc()
    TOOL_LATENCY_SECONDS.labels(tool=tool).observe(latency)


def increment_tool_error(*, tool: str) -> None:
    TOOL_ERRORS_TOTAL.labels(tool=tool).inc()


def observe_confidence_component(*, agent: str, component: str, value: float) -> None:
    CONFIDENCE_COMPONENT_VALUE.labels(agent=agent, component=component).observe(value)


def observe_mcp_request(*, method: str, endpoint: str, status: int | None, success: bool, latency: float) -> None:
    status_label = str(status) if status is not None else "error"
    outcome = "success" if success else "failure"
    MCP_REQUEST_TOTAL.labels(method=method.upper(), endpoint=endpoint, outcome=outcome).inc()
    MCP_REQUEST_LATENCY_SECONDS.labels(method=method.upper(), endpoint=endpoint, status=status_label).observe(latency)


def increment_mcp_circuit_open(*, endpoint: str) -> None:
    MCP_CIRCUIT_OPEN_TOTAL.labels(endpoint=endpoint).inc()


def increment_mcp_circuit_trip(*, endpoint: str) -> None:
    MCP_CIRCUIT_TRIP_TOTAL.labels(endpoint=endpoint).inc()
