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
