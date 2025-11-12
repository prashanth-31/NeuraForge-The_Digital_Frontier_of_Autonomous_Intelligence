from __future__ import annotations

from typing import Any

from datetime import datetime
from typing import Iterable

from prometheus_client import Counter, Gauge, Histogram

AGENT_LATENCY_SECONDS = Histogram(
    "neuraforge_agent_execution_latency_seconds",
    "Latency for each agent execution",
    labelnames=("agent",),
)

TASK_LATENCY_SECONDS = Histogram(
    "neuraforge_task_latency_seconds",
    "End-to-end task latency segmented by agent involvement",
    labelnames=("entry_point", "agent_count"),
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, float("inf")),
)

AGENT_EVENT_TOTAL = Counter(
    "neuraforge_agent_event_total",
    "Count of agent lifecycle events (started/completed/failed)",
    labelnames=("agent", "event"),
)

AGENT_TOOL_USAGE_TOTAL = Counter(
    "neuraforge_agent_tool_usage_total",
    "Agent tool invocation attempts grouped by outcome",
    labelnames=("agent", "tool", "outcome"),
)

AGENT_TOOL_POLICY_TOTAL = Counter(
    "neuraforge_agent_tool_policy_total",
    "Tool-first policy compliance outcomes per agent",
    labelnames=("agent", "outcome"),
)

AGENT_TOOL_FAILURE_TOTAL = Counter(
    "neuraforge_agent_tool_failure_total",
    "Tool invocation failures grouped by agent, tool, and failure category",
    labelnames=("agent", "tool", "failure_type", "canonical"),
)

ORCHESTRATOR_RUNS_TOTAL = Counter(
    "neuraforge_orchestrator_runs_total",
    "Total orchestrator runs by status",
    labelnames=("entry_point", "status"),
)

ORCHESTRATOR_RUN_LATENCY_SECONDS = Histogram(
    "neuraforge_orchestrator_run_latency_seconds",
    "End-to-end orchestrator runtime",
    labelnames=("entry_point",),
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 900, float("inf")),
)

ORCHESTRATOR_ACTIVE_GAUGE = Gauge(
    "neuraforge_orchestrator_runs_active",
    "Active orchestrator runs in flight",
    labelnames=("entry_point",),
)

NEGOTIATION_ROUNDS = Histogram(
    "neuraforge_negotiation_rounds",
    "Number of negotiation payloads considered per run",
    labelnames=("strategy",),
    buckets=(0, 1, 2, 3, 4, 5, 8, 13),
)

PLANNER_STEPS = Histogram(
    "neuraforge_planner_plan_steps",
    "Number of steps produced per generated task plan",
    labelnames=("strategy",),
    buckets=(0, 1, 2, 3, 4, 5, 8, 13, 21),
)

PLANNER_OUTCOMES_TOTAL = Counter(
    "neuraforge_planner_plan_total",
    "Count of planner outcomes grouped by status",
    labelnames=("strategy", "status"),
)

NEGOTIATION_CONSENSUS = Histogram(
    "neuraforge_negotiation_consensus",
    "Consensus score distribution",
    labelnames=("strategy",),
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)

GUARDRAIL_DECISIONS_TOTAL = Counter(
    "neuraforge_guardrail_decisions_total",
    "Guardrail decisions by outcome",
    labelnames=("decision", "policy_id"),
)

GUARDRAIL_DECISION_TOTAL = Counter(
    "neuraforge_guardrail_decision_total",
    "Guardrail decisions by outcome",
    labelnames=("decision", "policy_id"),
)

ORCHESTRATOR_ESCALATIONS_TOTAL = Counter(
    "neuraforge_orchestrator_escalations_total",
    "Escalations raised by orchestrator guardrails",
    labelnames=("policy_id",),
)

ORCHESTRATOR_SLA_EVENTS_TOTAL = Counter(
    "neuraforge_orchestrator_sla_events_total",
    "SLA compliance events",
    labelnames=("category",),
)

ORCHESTRATOR_OUTCOME_GAUGE = Gauge(
    "neuraforge_orchestrator_outcomes",
    "Rolling orchestrator outcomes (1 for success, 0 for failure)",
    labelnames=("task_id",),
)

ORCHESTRATOR_SUCCESS_RATE = Gauge(
    "neuraforge_orchestrator_success_rate",
    "Smoothed orchestrator success rate",
)

META_AGENT_RESOLUTION_LATENCY_SECONDS = Histogram(
    "neuraforge_meta_agent_resolution_latency_seconds",
    "Latency of meta-agent synthesis operations",
    labelnames=("mode",),
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")),
)

META_AGENT_DISPUTES_TOTAL = Counter(
    "neuraforge_meta_agent_disputes_total",
    "Count of disputes detected by the meta-agent",
    labelnames=("severity",),
)

META_AGENT_OVERRIDES_TOTAL = Counter(
    "neuraforge_meta_agent_overrides_total",
    "Meta-agent actions that triggered human overrides",
    labelnames=("action",),
)

REVIEW_TICKETS_GAUGE = Gauge(
    "neuraforge_review_tickets",
    "Current review ticket counts by status",
    labelnames=("status",),
)

REVIEW_TICKETS_OPEN_GAUGE = Gauge(
    "neuraforge_review_tickets_open",
    "Open review tickets awaiting action.",
)

REVIEW_TICKET_OLDEST_AGE_SECONDS = Gauge(
    "neuraforge_review_ticket_oldest_age_seconds",
    "Oldest observed open review ticket age in seconds.",
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

FINANCE_QUOTE_FALLBACK_TOTAL = Counter(
    "neuraforge_finance_quote_fallback_total",
    "Count of finance quote fallback attempts by provider and reason",
    labelnames=("provider", "reason"),
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


def observe_task_latency(*, entry_point: str, agent_count: int, latency: float) -> None:
    histogram_label = str(agent_count) if agent_count <= 5 else "5_plus"
    TASK_LATENCY_SECONDS.labels(entry_point=entry_point, agent_count=histogram_label).observe(latency)


def increment_agent_event(*, agent: str, event: str) -> None:
    AGENT_EVENT_TOTAL.labels(agent=agent, event=event).inc()


def record_agent_tool_invocation(*, agent: str, tool: str, outcome: str) -> None:
    AGENT_TOOL_USAGE_TOTAL.labels(agent=agent, tool=tool, outcome=outcome).inc()


def record_agent_tool_policy(*, agent: str, outcome: str) -> None:
    AGENT_TOOL_POLICY_TOTAL.labels(agent=agent, outcome=outcome).inc()


def record_agent_tool_failure(*, agent: str, tool: str, failure_type: str, canonical: bool) -> None:
    AGENT_TOOL_FAILURE_TOTAL.labels(
        agent=agent,
        tool=tool,
        failure_type=failure_type,
        canonical=str(canonical).lower(),
    ).inc()


def mark_orchestrator_run_started(*, entry_point: str, task_id: str | None = None) -> None:
    ORCHESTRATOR_ACTIVE_GAUGE.labels(entry_point=entry_point).inc()
    ORCHESTRATOR_RUNS_TOTAL.labels(entry_point, "started").inc()
    if task_id:
        ORCHESTRATOR_OUTCOME_GAUGE.labels(task_id=task_id).set(0)


def mark_orchestrator_run_completed(
    *,
    entry_point: str,
    task_id: str | None,
    status: str,
    latency: float,
) -> None:
    ORCHESTRATOR_ACTIVE_GAUGE.labels(entry_point=entry_point).dec()
    ORCHESTRATOR_RUNS_TOTAL.labels(entry_point, status).inc()
    ORCHESTRATOR_RUN_LATENCY_SECONDS.labels(entry_point=entry_point).observe(latency)
    if task_id:
        ORCHESTRATOR_OUTCOME_GAUGE.labels(task_id=task_id).set(1 if status == "completed" else 0)


def observe_negotiation_metrics(*, strategy: str, rounds: int, consensus: float | None) -> None:
    NEGOTIATION_ROUNDS.labels(strategy=strategy).observe(rounds)
    if consensus is not None:
        NEGOTIATION_CONSENSUS.labels(strategy=strategy).observe(max(0.0, min(1.0, consensus)))


def record_plan_metrics(*, strategy: str, status: str, steps: int | None = None) -> None:
    PLANNER_OUTCOMES_TOTAL.labels(strategy=strategy, status=status).inc()
    if steps is not None:
        PLANNER_STEPS.labels(strategy=strategy).observe(max(0, steps))


def increment_guardrail_decision(*, decision: str, policy_id: str | None) -> None:
    labels = {"decision": decision, "policy_id": policy_id or "none"}
    GUARDRAIL_DECISIONS_TOTAL.labels(**labels).inc()
    GUARDRAIL_DECISION_TOTAL.labels(**labels).inc()
    if decision in {"escalate", "review"}:
        ORCHESTRATOR_ESCALATIONS_TOTAL.labels(policy_id=policy_id or "none").inc()


def record_sla_event(*, category: str) -> None:
    ORCHESTRATOR_SLA_EVENTS_TOTAL.labels(category=category).inc()


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


def observe_meta_resolution_latency(*, mode: str, latency: float) -> None:
    META_AGENT_RESOLUTION_LATENCY_SECONDS.labels(mode=mode).observe(latency)


def increment_meta_dispute(*, severity: str) -> None:
    META_AGENT_DISPUTES_TOTAL.labels(severity=severity).inc()


def increment_meta_override(*, action: str) -> None:
    META_AGENT_OVERRIDES_TOTAL.labels(action=action).inc()


def record_review_ticket_counts(
    *,
    open_count: int,
    in_review: int,
    resolved: int,
    dismissed: int,
    unassigned_open: int = 0,
) -> None:
    REVIEW_TICKETS_GAUGE.labels(status="open").set(open_count)
    REVIEW_TICKETS_GAUGE.labels(status="in_review").set(in_review)
    REVIEW_TICKETS_GAUGE.labels(status="resolved").set(resolved)
    REVIEW_TICKETS_GAUGE.labels(status="dismissed").set(dismissed)
    REVIEW_TICKETS_GAUGE.labels(status="unassigned").set(unassigned_open)
    REVIEW_TICKETS_OPEN_GAUGE.set(open_count)


def record_review_oldest_ticket_age(*, seconds: float) -> None:
    REVIEW_TICKET_OLDEST_AGE_SECONDS.set(max(0.0, seconds))


def bind_event_payload(agent: str, task_id: str, event: str, **extra: Any) -> dict[str, Any]:
    payload = {"agent": agent, "task_id": task_id, "event": event}
    payload.update(extra)
    return payload


def observe_tool_invocation(*, tool: str, latency: float, cached: bool) -> None:
    TOOL_INVOCATIONS_TOTAL.labels(tool=tool, cached=str(cached)).inc()
    TOOL_LATENCY_SECONDS.labels(tool=tool).observe(latency)


def increment_tool_error(*, tool: str) -> None:
    TOOL_ERRORS_TOTAL.labels(tool=tool).inc()


def increment_finance_quote_fallback(*, provider: str, reason: str) -> None:
    FINANCE_QUOTE_FALLBACK_TOTAL.labels(provider=provider, reason=reason).inc()


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


def update_success_rate_from_history(results: Iterable[tuple[datetime, bool]], *, window: int = 50) -> None:
    history = list(results)
    if not history:
        return
    recent = history[-window:]
    success_ratio = sum(1 for _, ok in recent if ok) / len(recent)
    ORCHESTRATOR_SUCCESS_RATE.set(success_ratio)
