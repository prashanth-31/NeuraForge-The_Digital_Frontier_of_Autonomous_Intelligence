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


def observe_agent_latency(*, agent: str, latency: float) -> None:
    AGENT_LATENCY_SECONDS.labels(agent=agent).observe(latency)


def increment_agent_event(*, agent: str, event: str) -> None:
    AGENT_EVENT_TOTAL.labels(agent=agent, event=event).inc()


def bind_event_payload(agent: str, task_id: str, event: str, **extra: Any) -> dict[str, Any]:
    payload = {"agent": agent, "task_id": task_id, "event": event}
    payload.update(extra)
    return payload
