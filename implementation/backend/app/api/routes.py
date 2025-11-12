from __future__ import annotations

import asyncio
import copy
import json
import uuid
from contextlib import AsyncExitStack, suppress
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from ..agents.base import AgentContext, BaseAgent
from ..agents.creative import CreativeAgent
from ..agents.enterprise import EnterpriseAgent
from ..agents.finance import FinanceAgent
from ..agents.general import GeneralistAgent
from ..agents.research import ResearchAgent
from ..core.config import Settings, get_settings
from ..core.logging import get_logger
from ..core.rate_limit import rate_limit_dependency
from ..core.security import require_scopes
from ..dependencies import (
    get_hybrid_memory,
    get_orchestrator_state_store,
    get_review_manager,
    get_review_manager_singleton,
    get_task_queue,
)
from ..orchestration.context import ContextAssemblyContract, ContextSnapshotStore, ContextStage
from ..orchestration.guardrails import GuardrailManager, GuardrailStore
from ..orchestration.lifecycle import TaskLifecycleStore
from ..orchestration.graph import Orchestrator
from ..orchestration.negotiation import SimpleNegotiationEngine
from ..orchestration.planner import DependencyTaskPlanner, SimpleTaskPlanner
from ..orchestration.scheduler import AsyncioTaskScheduler, SequentialTaskScheduler
from ..orchestration.meta import MetaAgent
from ..orchestration.review import ReviewManager, ReviewStatus
from ..orchestration.llm_planner import LLMOrchestrationPlanner
from ..orchestration.state import OrchestratorEvent, OrchestratorRun
from ..orchestration.store import OrchestratorStateStore
from ..schemas.reviews import (
    ReviewAssignmentRequest,
    ReviewNoteCreate,
    ReviewResolutionRequest,
    ReviewStatusLiteral,
    ReviewTicketModel,
    ReviewMetricsResponse,
)
from ..schemas.tasks import TaskRequest, TaskResponse, TaskResult, TaskStatusMetrics, TaskStatusResponse
from ..schemas.orchestrator import OrchestratorEventModel, OrchestratorRunDetail
from ..services.embedding import EmbeddingService
from ..services.llm import LLMService
from ..services.memory import HybridMemoryService
from ..services.retrieval import ContextAssembler, RetrievalService
from ..services.scoring import ConfidenceScorer
from ..services.disputes import DisputeDetector, MetaConfidenceScorer
from ..services.tools import get_tool_service

logger = get_logger(name=__name__)

router = APIRouter()

REVIEW_READ_SCOPE = "reviews:read"
REVIEW_WRITE_SCOPE = "reviews:write"
REPORTS_READ_SCOPE = "reports:read"

rate_limit_task_submission = rate_limit_dependency("task_submission")
rate_limit_review_action = rate_limit_dependency("review_action")

MAX_TASK_EVENTS = 250
EVENT_ENVELOPE_VERSION = 1


async def _extract_json_body(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Request body must be a JSON object")
    if "payload" in payload and isinstance(payload["payload"], dict) and len(payload) == 1:
        return payload["payload"]
    return payload


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _scrub_outputs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    outputs: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            outputs.append(dict(item))
    return outputs


def _scrub_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _seed_conversation_state(state: dict[str, Any], previous: Mapping[str, Any]) -> None:
    prior_outputs = previous.get("outputs")
    if isinstance(prior_outputs, list):
        state["outputs"] = [dict(item) for item in prior_outputs if isinstance(item, dict)]

    shared_context = previous.get("shared_context")
    if isinstance(shared_context, Mapping):
        state["shared_context"] = copy.deepcopy(dict(shared_context))

    existing_metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
    previous_metadata = previous.get("metadata") if isinstance(previous.get("metadata"), Mapping) else {}
    merged_metadata = copy.deepcopy(dict(previous_metadata))
    merged_metadata.update(dict(existing_metadata))
    state["metadata"] = merged_metadata

    if isinstance(state.get("shared_context"), Mapping):
        state.setdefault("metadata", {})
        state["metadata"]["_shared_context"] = copy.deepcopy(dict(state["shared_context"]))
    else:
        state.setdefault("shared_context", {"provenance": []})


def _ensure_conversation_metadata(
    state: dict[str, Any],
    *,
    current_task_id: str,
    continuation_task_id: str | None = None,
) -> None:
    if not isinstance(state, dict):
        return

    metadata_section = state.get("metadata")
    if isinstance(metadata_section, dict):
        metadata = metadata_section
    elif isinstance(metadata_section, Mapping):
        metadata = dict(metadata_section)
        state["metadata"] = metadata
    else:
        metadata = {}
        state["metadata"] = metadata

    conversation_section = metadata.get("conversation")
    if isinstance(conversation_section, dict):
        conversation = conversation_section
    elif isinstance(conversation_section, Mapping):
        conversation = dict(conversation_section)
        metadata["conversation"] = conversation
    else:
        conversation = {}
        metadata["conversation"] = conversation

    if continuation_task_id is not None:
        continuation_ref = str(continuation_task_id)
        conversation["continuation_task_id"] = continuation_ref
        conversation["previous_task_id"] = continuation_ref

    root_value = conversation.get("root_task_id")
    if not isinstance(root_value, str) or not root_value:
        fallback_root = continuation_task_id if continuation_task_id is not None else current_task_id
        conversation["root_task_id"] = str(fallback_root)

    conversation["latest_task_id"] = str(current_task_id)

    shared_context_section = state.get("shared_context")
    if isinstance(shared_context_section, dict):
        shared_context = shared_context_section
    elif isinstance(shared_context_section, Mapping):
        shared_context = dict(shared_context_section)
        state["shared_context"] = shared_context
    else:
        shared_context = {}
        state["shared_context"] = shared_context

    shared_conversation_section = shared_context.get("conversation")
    if isinstance(shared_conversation_section, dict):
        shared_conversation = shared_conversation_section
    elif isinstance(shared_conversation_section, Mapping):
        shared_conversation = dict(shared_conversation_section)
        shared_context["conversation"] = shared_conversation
    else:
        shared_conversation = {}
        shared_context["conversation"] = shared_conversation

    for key in ("root_task_id", "latest_task_id", "continuation_task_id", "previous_task_id"):
        value = conversation.get(key)
        if value is not None:
            shared_conversation[key] = str(value)


def _normalize_task_identifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    if isinstance(value, uuid.UUID):
        return str(value)
    try:
        candidate = str(value)
    except Exception:
        return None
    candidate = candidate.strip()
    return candidate or None


def _extract_guardrail_data(result_state: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    guardrail_block = result_state.get("guardrails") if isinstance(result_state, dict) else None
    if not isinstance(guardrail_block, dict):
        return None, []
    decisions_raw = guardrail_block.get("decisions")
    decisions: list[dict[str, Any]] = []
    if isinstance(decisions_raw, list):
        for item in decisions_raw:
            if isinstance(item, dict):
                decisions.append(dict(item))
    return dict(guardrail_block), decisions


def _compute_task_metrics(
    result_state: dict[str, Any],
    events: list[OrchestratorEvent],
    guardrail_decisions: list[dict[str, Any]],
) -> TaskStatusMetrics:
    completed_agents = sum(1 for event in events if event.event_type == "agent_completed")
    failed_agents = sum(1 for event in events if event.event_type == "agent_failed")
    if completed_agents == 0:
        completed_agents = len(_scrub_outputs(result_state.get("outputs")))
    guardrail_events = len(guardrail_decisions)
    negotiation_rounds: int | None = None
    negotiation_block = result_state.get("negotiation")
    if isinstance(negotiation_block, dict):
        rounds_value = negotiation_block.get("rounds")
        if isinstance(rounds_value, int):
            negotiation_rounds = rounds_value
        elif isinstance(rounds_value, float):
            negotiation_rounds = int(round(rounds_value))
    return TaskStatusMetrics(
        agents_completed=completed_agents,
        agents_failed=failed_agents,
        guardrail_events=guardrail_events,
        negotiation_rounds=negotiation_rounds,
    )


def _condense_meta_summary(meta: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        return {}
    summary = meta.get("summary")
    confidence = meta.get("confidence")
    mode = meta.get("mode")
    should_escalate = meta.get("should_escalate")
    payload: dict[str, Any] = {}
    if isinstance(summary, str) and summary.strip():
        payload["summary"] = summary.strip()
    if isinstance(confidence, (int, float)):
        payload["confidence"] = float(confidence)
    if isinstance(mode, str) and mode:
        payload["mode"] = mode
    if isinstance(should_escalate, bool):
        payload["should_escalate"] = should_escalate
    return payload


def _build_report_payload(state: Mapping[str, Any]) -> dict[str, Any] | None:
    meta_section = state.get("meta") if isinstance(state, Mapping) else None
    meta_payload = _condense_meta_summary(meta_section) if isinstance(meta_section, Mapping) else {}

    dossier_block = state.get("dossier") if isinstance(state, Mapping) else None
    dossier_json = None
    if isinstance(dossier_block, Mapping):
        candidate = dossier_block.get("json")
        if isinstance(candidate, Mapping):
            dossier_json = candidate

    agent_perspectives: list[dict[str, Any]] = []
    if dossier_json is not None:
        raw_perspectives = dossier_json.get("agent_perspectives")
        if isinstance(raw_perspectives, Sequence):
            for item in raw_perspectives:
                if not isinstance(item, Mapping):
                    continue
                summary = str(item.get("summary") or item.get("content") or "").strip()
                if not summary:
                    continue
                perspective = {
                    "agent": item.get("agent"),
                    "summary": summary,
                }
                confidence = item.get("confidence")
                if isinstance(confidence, (int, float)):
                    perspective["confidence"] = float(confidence)
                metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else None
                if metadata:
                    perspective["metadata"] = dict(metadata)
                agent_perspectives.append(perspective)

    headline = None
    if dossier_json is not None:
        headline_candidate = dossier_json.get("headline")
        if isinstance(headline_candidate, str) and headline_candidate.strip():
            headline = headline_candidate.strip()

    if not agent_perspectives:
        outputs = state.get("outputs") if isinstance(state, Mapping) else None
        if isinstance(outputs, Sequence):
            for entry in outputs:
                if not isinstance(entry, Mapping):
                    continue
                summary = str(entry.get("summary") or entry.get("content") or "").strip()
                if not summary:
                    continue
                perspective = {
                    "agent": entry.get("agent"),
                    "summary": summary,
                }
                confidence = entry.get("confidence")
                if isinstance(confidence, (int, float)):
                    perspective["confidence"] = float(confidence)
                agent_perspectives.append(perspective)

    if headline is None:
        for perspective in agent_perspectives:
            summary = perspective.get("summary")
            if isinstance(summary, str) and summary.strip():
                headline = summary.strip()
                break

    if headline is None:
        prompt_value = state.get("prompt") if isinstance(state.get("prompt"), str) else None
        if prompt_value:
            headline = f"Task completed: {prompt_value[:80]}"

    if dossier_json is not None and not meta_payload:
        meta_resolution = dossier_json.get("meta_resolution")
        if isinstance(meta_resolution, Mapping):
            meta_payload = _condense_meta_summary(meta_resolution)

    plan_block = state.get("plan") if isinstance(state, Mapping) else None
    suggestions: list[dict[str, Any]] = []
    plan_status: str | None = None
    if isinstance(plan_block, Mapping):
        plan_status_value = plan_block.get("status")
        if isinstance(plan_status_value, str) and plan_status_value:
            plan_status = plan_status_value
        steps = plan_block.get("steps")
        if isinstance(steps, Sequence):
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                description = str(step.get("description") or step.get("detail") or "").strip()
                if not description:
                    continue
                suggestion: dict[str, Any] = {
                    "description": description,
                }
                step_id = step.get("step_id")
                if isinstance(step_id, str) and step_id:
                    suggestion["step_id"] = step_id
                title = step.get("title")
                if isinstance(title, str) and title:
                    suggestion["title"] = title
                agent_name = step.get("agent")
                if isinstance(agent_name, str) and agent_name:
                    suggestion["agent"] = agent_name
                depends_on = step.get("depends_on")
                if isinstance(depends_on, Sequence):
                    suggestion["depends_on"] = [str(item) for item in depends_on if item]
                metadata = step.get("metadata")
                if isinstance(metadata, Mapping):
                    capability = metadata.get("capability")
                    if capability:
                        suggestion["capability"] = capability
                suggestions.append(suggestion)

    report: dict[str, Any] = {}
    if headline:
        report["headline"] = headline
    if meta_payload:
        report["meta"] = meta_payload
    if agent_perspectives:
        report["agent_perspectives"] = agent_perspectives
    if suggestions:
        report["suggested_actions"] = suggestions
    if plan_status:
        report["plan_status"] = plan_status

    if not report:
        return None
    return report


def _task_status_response(
    *,
    task_id: str,
    result_state: dict[str, Any],
    run_record: OrchestratorRun | None,
    events: list[OrchestratorEvent],
) -> TaskStatusResponse:
    effective_task_id = str(result_state.get("id") or result_state.get("task_id") or task_id)
    status_value = str(result_state.get("status") or (run_record.status.value if run_record else "unknown"))
    prompt_value = result_state.get("prompt") if isinstance(result_state.get("prompt"), str) else None
    metadata = _scrub_metadata(result_state.get("metadata"))
    outputs = _scrub_outputs(result_state.get("outputs"))
    plan = result_state.get("plan") if isinstance(result_state.get("plan"), dict) else None
    negotiation = result_state.get("negotiation") if isinstance(result_state.get("negotiation"), dict) else None
    guardrails_payload, guardrail_decisions = _extract_guardrail_data(result_state)

    meta_section = result_state.get("meta") if isinstance(result_state.get("meta"), dict) else None
    dossier_section = result_state.get("dossier") if isinstance(result_state.get("dossier"), dict) else None
    report_candidate = result_state.get("report")
    report_block = report_candidate if isinstance(report_candidate, dict) else None
    if report_block is None:
        generated_report = _build_report_payload(result_state)
        if generated_report is not None:
            report_block = generated_report
    elif isinstance(report_block, dict):
        report_block = dict(report_block)

    last_error_value = result_state.get("error")
    last_error = str(last_error_value) if isinstance(last_error_value, str) and last_error_value else None

    run_id_value = result_state.get("run_id")
    run_id_str: str | None = None
    if isinstance(run_id_value, uuid.UUID):
        run_id_str = str(run_id_value)
    elif isinstance(run_id_value, str) and run_id_value:
        run_id_str = run_id_value
    elif run_record is not None:
        run_id_str = str(run_record.run_id)

    metrics = _compute_task_metrics(result_state, events, guardrail_decisions)

    created_at = run_record.created_at if run_record else _parse_datetime(result_state.get("created_at"))
    updated_at = run_record.updated_at if run_record else _parse_datetime(result_state.get("updated_at") or result_state.get("timestamp"))

    event_models = [OrchestratorEventModel.from_domain(event) for event in events]

    return TaskStatusResponse(
        task_id=effective_task_id,
        status=status_value,
        run_id=run_id_str,
        prompt=prompt_value,
        metadata=metadata,
        outputs=outputs,
        plan=plan,
        negotiation=negotiation,
        guardrails=guardrails_payload,
    meta=dict(meta_section) if meta_section is not None else None,
    dossier=dict(dossier_section) if dossier_section is not None else None,
    report=report_block,
        metrics=metrics,
        last_error=last_error,
        created_at=created_at,
        updated_at=updated_at,
        events=event_models,
    )


async def _build_orchestration_pipeline(
    *,
    agents: list[BaseAgent],
    settings: Settings,
    memory: HybridMemoryService,
    exit_stack: AsyncExitStack,
    state_store: OrchestratorStateStore | None,
    llm_service: LLMService,
    review_manager: ReviewManager | None,
) -> tuple[Orchestrator, EmbeddingService | None, ContextAssembler | None]:
    embedding_service: EmbeddingService | None = None
    context_assembler: ContextAssembler | None = None
    try:
        embedding_service = EmbeddingService.from_settings(settings, memory_service=memory)
        retrieval_service = RetrievalService.from_settings(settings, memory=memory, embedder=embedding_service)
        context_assembler = ContextAssembler(retrieval=retrieval_service)
    except Exception as exc:  # pragma: no cover - embedding optional
        logger.warning("embedding_initialization_failed", error=str(exc))
        embedding_service = None
        context_assembler = None

    lifecycle_store_ctx: TaskLifecycleStore | None = None
    lifecycle_candidate: TaskLifecycleStore | None = None
    try:
        lifecycle_candidate = TaskLifecycleStore.from_settings(settings)
        lifecycle_store_ctx = await exit_stack.enter_async_context(lifecycle_candidate.lifecycle())
    except Exception as exc:  # pragma: no cover - lifecycle persistence optional
        logger.warning("lifecycle_store_unavailable", error=str(exc))
        if lifecycle_candidate is not None:
            await lifecycle_candidate.close()
        lifecycle_store_ctx = None

    snapshot_store_ctx: ContextSnapshotStore | None = None
    snapshot_candidate: ContextSnapshotStore | None = None
    if settings.snapshots.enabled:
        try:
            snapshot_candidate = ContextSnapshotStore.from_settings(settings)
            snapshot_store_ctx = await exit_stack.enter_async_context(snapshot_candidate.lifecycle())
        except Exception as exc:  # pragma: no cover - optional snapshot persistence
            logger.warning("snapshot_store_unavailable", error=str(exc))
            if snapshot_candidate is not None:
                await snapshot_candidate.close()
            snapshot_store_ctx = None

    guardrail_store_ctx: GuardrailStore | None = None
    guardrail_manager: GuardrailManager | None = None
    guardrail_candidate: GuardrailStore | None = None
    if settings.guardrails.enabled:
        try:
            guardrail_candidate = GuardrailStore.from_settings(settings)
            guardrail_store_ctx = await exit_stack.enter_async_context(guardrail_candidate.lifecycle())
        except Exception as exc:  # pragma: no cover - optional guardrail persistence
            logger.warning("guardrail_store_unavailable", error=str(exc))
            if guardrail_candidate is not None:
                await guardrail_candidate.close()
            guardrail_store_ctx = None
        guardrail_manager = GuardrailManager(settings=settings.guardrails, store=guardrail_store_ctx, llm_service=llm_service)

    planner = DependencyTaskPlanner(settings=settings.planning) if settings.planning.enabled else SimpleTaskPlanner()
    if settings.scheduling.backend == "asyncio":
        scheduler = AsyncioTaskScheduler(settings=settings.scheduling)
    else:
        scheduler = SequentialTaskScheduler()

    context_contract = None
    if context_assembler is not None:
        overrides = {
            ContextStage.INTAKE: {"top_snippets": 4, "max_chars": settings.retrieval.max_context_chars},
            ContextStage.NEGOTIATION: {"top_snippets": 3, "max_chars": max(600, settings.retrieval.max_context_chars // 2)},
            ContextStage.PLANNING: {"top_snippets": 5, "max_chars": settings.retrieval.max_context_chars},
        }
        context_contract = ContextAssemblyContract(assembler=context_assembler, stage_overrides=overrides)

    meta_agent = None
    if settings.meta_agent.enabled and settings.environment == "production":
        meta_agent = MetaAgent(
            llm_service=llm_service,
            settings=settings.meta_agent,
            dispute_detector=DisputeDetector(
                consensus_delta_threshold=settings.meta_agent.consensus_delta_threshold,
                stddev_threshold=settings.meta_agent.stddev_threshold,
            ),
            confidence_scorer=MetaConfidenceScorer(),
        )

    orchestration_planner = LLMOrchestrationPlanner(settings=settings)

    orchestrator = Orchestrator(
        agents=agents,
        state_store=state_store,
        negotiation_engine=SimpleNegotiationEngine(),
        planner=planner,
        scheduler=scheduler,
        context_contract=context_contract,
        snapshot_store=snapshot_store_ctx,
        lifecycle_store=lifecycle_store_ctx,
        guardrail_manager=guardrail_manager,
        meta_agent=meta_agent,
        review_manager=review_manager,
        orchestration_planner=orchestration_planner,
    )
    return orchestrator, embedding_service, context_assembler


async def _fetch_result_state(
    task_id: str,
    *,
    memory: HybridMemoryService,
) -> dict[str, Any]:
    payload = await memory.fetch_ephemeral_memory(task_id)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task result unavailable")

    result_state = payload.get("result")
    if not isinstance(result_state, dict):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task result unavailable")
    return result_state


@router.get("/health", tags=["health"])
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/diagnostics/mcp", tags=["diagnostics"])
async def mcp_diagnostics(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    if not settings.tools.mcp.enabled:
        return {"enabled": False}
    try:
        service = await get_tool_service()
    except Exception as exc:  # pragma: no cover - returns snapshot on failure
        return {
            "enabled": True,
            "status": "unavailable",
            "error": str(exc),
        }
    return service.get_diagnostics()


@router.post("/submit_task", response_model=TaskResponse, tags=["tasks"])
async def submit_task(
    request: Request,
    _: None = Depends(rate_limit_task_submission),
    queue=Depends(get_task_queue),
    settings: Settings = Depends(get_settings),
) -> TaskResponse:
    raw_payload = await _extract_json_body(request)
    try:
        payload = TaskRequest(**raw_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()) from exc
    task_id = str(uuid.uuid4())
    task_payload = payload.model_dump()
    continuation_id = _normalize_task_identifier(task_payload.get("continuation_task_id"))

    async def _job() -> None:
        memory_service = HybridMemoryService.from_settings(settings)
        state_store_factory: OrchestratorStateStore | None = None
        review_manager = get_review_manager_singleton(settings)
        try:
            state_store_factory = OrchestratorStateStore.from_settings(settings)
        except RuntimeError as exc:  # pragma: no cover - optional persistence layer
            logger.warning("orchestrator_state_store_unavailable", error=str(exc))
        state = {
            "id": task_id,
            "prompt": task_payload["prompt"],
            "metadata": task_payload.get("metadata", {}),
            "outputs": [],
        }
        async with memory_service.lifecycle() as memory:
            exit_stack = AsyncExitStack()
            await exit_stack.__aenter__()
            try:
                state_store_context: OrchestratorStateStore | None = None
                if state_store_factory is not None:
                    try:
                        state_store_context = await exit_stack.enter_async_context(state_store_factory.lifecycle())
                    except Exception as exc:  # pragma: no cover - store failure should not break task
                        logger.warning("orchestrator_state_store_connect_failed", error=str(exc))
                        await state_store_factory.close()
                        state_store_context = None

                if continuation_id is not None:
                    try:
                        previous_episode = await memory.fetch_ephemeral_memory(continuation_id)
                    except Exception:  # pragma: no cover - defensive safety
                        previous_episode = None
                    if isinstance(previous_episode, Mapping):
                        prior_result = previous_episode.get("result")
                        if isinstance(prior_result, Mapping):
                            _seed_conversation_state(state, prior_result)

                _ensure_conversation_metadata(
                    state,
                    current_task_id=task_id,
                    continuation_task_id=continuation_id,
                )

                try:
                    llm_service = LLMService.from_settings(settings)
                except RuntimeError as exc:
                    logger.exception("llm_initialization_failed", error=str(exc))
                    failure = {
                        **state,
                        "status": "failed",
                        "error": "LLM service unavailable",
                    }
                    _ensure_conversation_metadata(
                        failure,
                        current_task_id=task_id,
                        continuation_task_id=continuation_id,
                    )
                    await memory.store_ephemeral_memory(task_id, {"result": failure})
                    return

                agents = [
                    GeneralistAgent(),
                    ResearchAgent(),
                    FinanceAgent(),
                    CreativeAgent(),
                    EnterpriseAgent(),
                ]
                embedding_service: EmbeddingService | None = None
                context_assembler: ContextAssembler | None = None
                orchestrator: Orchestrator
                orchestrator, embedding_service, context_assembler = await _build_orchestration_pipeline(
                    agents=agents,
                    settings=settings,
                    memory=memory,
                    exit_stack=exit_stack,
                    state_store=state_store_context,
                    llm_service=llm_service,
                    review_manager=review_manager,
                )

                try:
                    tool_service = None
                    if settings.tools.mcp.enabled:
                        try:
                            tool_service = await get_tool_service()
                        except Exception as exc:  # pragma: no cover - optional tool layer
                            logger.warning("tool_service_unavailable", error=str(exc))

                    agent_context = AgentContext(
                        memory=memory,
                        llm=llm_service,
                        context=context_assembler,
                        tools=tool_service,
                        scorer=ConfidenceScorer(settings.scoring),
                    )
                    result = await orchestrator.route_task(state, context=agent_context)
                    if isinstance(result, dict):
                        _ensure_conversation_metadata(
                            result,
                            current_task_id=task_id,
                            continuation_task_id=continuation_id,
                        )
                        report_candidate = result.get("report")
                        report_payload = report_candidate if isinstance(report_candidate, dict) else None
                        if report_payload is None:
                            generated_report = _build_report_payload(result)
                            if generated_report is not None:
                                result["report"] = generated_report
                    await memory.store_ephemeral_memory(task_id, {"result": result})
                finally:
                    if embedding_service is not None:
                        await embedding_service.aclose()
            finally:
                try:
                    await exit_stack.__aexit__(None, None, None)
                except Exception as exc:  # pragma: no cover - optional cleanup issues
                    logger.warning("orchestrator_cleanup_failed", error=str(exc))

    await queue.enqueue(_job)
    return TaskResponse(task_id=task_id, status="queued")


@router.post("/submit_task/stream", tags=["tasks"])
async def submit_task_stream(
    request: Request,
    _: None = Depends(rate_limit_task_submission),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    raw_payload = await _extract_json_body(request)
    try:
        payload = TaskRequest(**raw_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()) from exc
    task_id = str(uuid.uuid4())
    task_payload = payload.model_dump()
    continuation_id = _normalize_task_identifier(task_payload.get("continuation_task_id"))

    state = {
        "id": task_id,
        "prompt": task_payload["prompt"],
        "metadata": task_payload.get("metadata", {}),
        "outputs": [],
    }

    async def event_stream():
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        event_sequence = 0
        active_run_id: str | None = None

        def _normalize_event(event: str, payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal event_sequence, active_run_id
            body = dict(payload)

            timestamp_value = body.get("timestamp")
            if isinstance(timestamp_value, datetime):
                timestamp_value = timestamp_value.isoformat()
            if not isinstance(timestamp_value, str) or not timestamp_value:
                timestamp_value = _now_iso()

            run_value = body.get("run_id")
            if isinstance(run_value, uuid.UUID):
                run_value = str(run_value)
            if isinstance(run_value, str) and run_value:
                active_run_id = run_value
            elif active_run_id:
                run_value = active_run_id
            else:
                run_value = None

            task_identifier = body.get("task_id")
            if not isinstance(task_identifier, str) or not task_identifier:
                task_identifier = task_id

            normalized_payload: dict[str, Any] = {}
            for key, value in body.items():
                if key in {"task_id", "timestamp", "run_id"}:
                    continue
                if isinstance(value, datetime):
                    normalized_payload[key] = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    normalized_payload[key] = str(value)
                else:
                    normalized_payload[key] = value

            event_sequence += 1

            envelope: dict[str, Any] = {
                "version": EVENT_ENVELOPE_VERSION,
                "schema": "neuraforge.task-event.v1",
                "event": event,
                "type": event,
                "sequence": event_sequence,
                "task_id": task_identifier,
                "timestamp": timestamp_value,
                "payload": normalized_payload,
            }
            if run_value:
                envelope["run_id"] = run_value
                active_run_id = run_value

            return envelope

        async def emit(event: str, data: dict[str, Any]) -> None:
            if event == "__END__":
                await queue.put((event, {}))
                return
            await queue.put((event, _normalize_event(event, data)))

        async def progress(event_payload: dict[str, Any]) -> None:
            payload_copy = {key: value for key, value in event_payload.items() if key != "event"}
            payload_copy.setdefault("task_id", task_id)
            payload_copy.setdefault("timestamp", _now_iso())
            event_type = str(event_payload.get("event") or "telemetry_update")
            await emit(event_type, payload_copy)

        async def runner() -> None:
            nonlocal active_run_id
            memory_service = HybridMemoryService.from_settings(settings)
            state_store_factory: OrchestratorStateStore | None = None
            review_manager = get_review_manager_singleton(settings)
            try:
                try:
                    state_store_factory = OrchestratorStateStore.from_settings(settings)
                except RuntimeError as exc:  # pragma: no cover - optional persistence layer
                    logger.warning("orchestrator_state_store_unavailable", error=str(exc))

                async with memory_service.lifecycle() as memory:
                    exit_stack = AsyncExitStack()
                    await exit_stack.__aenter__()
                    try:
                        state_store_context: OrchestratorStateStore | None = None
                        if state_store_factory is not None:
                            try:
                                state_store_context = await exit_stack.enter_async_context(state_store_factory.lifecycle())
                            except Exception as exc:  # pragma: no cover - store failure should not break task
                                logger.warning("orchestrator_state_store_connect_failed", error=str(exc))
                                await state_store_factory.close()
                                state_store_context = None

                        if continuation_id is not None:
                            try:
                                previous_episode = await memory.fetch_ephemeral_memory(continuation_id)
                            except Exception:  # pragma: no cover - defensive safety
                                previous_episode = None
                            if isinstance(previous_episode, Mapping):
                                prior_result = previous_episode.get("result")
                                if isinstance(prior_result, Mapping):
                                    _seed_conversation_state(state, prior_result)

                        _ensure_conversation_metadata(
                            state,
                            current_task_id=task_id,
                            continuation_task_id=continuation_id,
                        )

                        await emit(
                            "task_started",
                            {"task_id": task_id, "timestamp": _now_iso(), "prompt": state["prompt"]},
                        )
                        try:
                            llm_service = LLMService.from_settings(settings)
                        except RuntimeError as exc:
                            logger.exception("llm_initialization_failed", error=str(exc))
                            failure = {
                                **state,
                                "status": "failed",
                                "error": "LLM service unavailable",
                                "timestamp": _now_iso(),
                            }
                            _ensure_conversation_metadata(
                                failure,
                                current_task_id=task_id,
                                continuation_task_id=continuation_id,
                            )
                            await memory.store_ephemeral_memory(task_id, {"result": failure})
                            await emit(
                                "task_failed",
                                {"task_id": task_id, "error": failure["error"], "timestamp": failure["timestamp"]},
                            )
                            return

                        agents = [
                            GeneralistAgent(),
                            ResearchAgent(),
                            FinanceAgent(),
                            CreativeAgent(),
                            EnterpriseAgent(),
                        ]
                        embedding_service: EmbeddingService | None = None
                        context_assembler: ContextAssembler | None = None
                        orchestrator: Orchestrator
                        orchestrator, embedding_service, context_assembler = await _build_orchestration_pipeline(
                            agents=agents,
                            settings=settings,
                            memory=memory,
                            exit_stack=exit_stack,
                            state_store=state_store_context,
                            llm_service=llm_service,
                            review_manager=review_manager,
                        )

                        result: dict[str, Any] | None = None

                        try:
                            tool_service = None
                            if settings.tools.mcp.enabled:
                                try:
                                    tool_service = await get_tool_service()
                                except Exception as exc:  # pragma: no cover - optional tool layer
                                    logger.warning("tool_service_unavailable", error=str(exc))

                            async def tool_event(payload: dict[str, Any]) -> None:
                                event_payload = dict(payload)
                                event_payload.setdefault("timestamp", _now_iso())
                                event_payload["task_id"] = task_id
                                if active_run_id:
                                    event_payload.setdefault("run_id", active_run_id)
                                await emit("tool_invoked", event_payload)

                            async def run_agents(active_tools: Any | None) -> dict[str, Any]:
                                agent_context = AgentContext(
                                    memory=memory,
                                    llm=llm_service,
                                    context=context_assembler,
                                    tools=active_tools,
                                    scorer=ConfidenceScorer(settings.scoring),
                                )
                                return await orchestrator.route_task(state, context=agent_context, progress_cb=progress)

                            if tool_service is not None:
                                await emit(
                                    "mcp_status",
                                    {
                                        "task_id": task_id,
                                        "timestamp": _now_iso(),
                                        "status": tool_service.get_diagnostics(),
                                    },
                                )
                                async with tool_service.instrument(tool_event):
                                    result = await run_agents(tool_service)
                            else:
                                result = await run_agents(None)
                        except Exception as exc:  # pragma: no cover - surfaced via SSE
                            failure = {
                                **state,
                                "status": "failed",
                                "error": str(exc),
                                "timestamp": _now_iso(),
                            }
                            _ensure_conversation_metadata(
                                failure,
                                current_task_id=task_id,
                                continuation_task_id=continuation_id,
                            )
                            await memory.store_ephemeral_memory(task_id, {"result": failure})
                            await emit(
                                "task_failed",
                                {"task_id": task_id, "error": failure["error"], "timestamp": failure["timestamp"]},
                            )
                            return
                        finally:
                            if embedding_service is not None:
                                await embedding_service.aclose()

                        if result is not None:
                            run_candidate = result.get("run_id") if isinstance(result, dict) else None
                            if isinstance(run_candidate, uuid.UUID):
                                active_run_id = str(run_candidate)
                                result["run_id"] = active_run_id
                            elif isinstance(run_candidate, str) and run_candidate:
                                active_run_id = run_candidate
                            if isinstance(result, dict):
                                _ensure_conversation_metadata(
                                    result,
                                    current_task_id=task_id,
                                    continuation_task_id=continuation_id,
                                )
                                report_payload = result.get("report") if isinstance(result.get("report"), dict) else None
                                if report_payload is None:
                                    generated_report = _build_report_payload(result)
                                    if generated_report is not None:
                                        result["report"] = generated_report
                            await memory.store_ephemeral_memory(task_id, {"result": result})
                            completion_payload: dict[str, Any] = {
                                "task_id": task_id,
                                "status": result.get("status", "completed"),
                                "timestamp": _now_iso(),
                                "run_id": active_run_id,
                            }
                            if isinstance(result, dict):
                                meta_block = result.get("meta")
                                if isinstance(meta_block, Mapping):
                                    condensed_meta = _condense_meta_summary(meta_block)
                                    if condensed_meta:
                                        completion_payload["meta"] = condensed_meta
                                report_candidate = result.get("report")
                                report_block = report_candidate if isinstance(report_candidate, dict) else None
                                if report_block is None:
                                    report_block = _build_report_payload(result) or None
                                    if report_block is not None:
                                        result.setdefault("report", report_block)
                                if report_block is not None:
                                    completion_payload["report"] = report_block
                            await emit(
                                "task_completed",
                                completion_payload,
                            )
                    finally:
                        try:
                            await exit_stack.__aexit__(None, None, None)
                        except Exception as exc:  # pragma: no cover - optional cleanup issues
                            logger.warning("orchestrator_cleanup_failed", error=str(exc))
            finally:
                if state_store_factory is not None:
                    await state_store_factory.close()
                await emit("__END__", {})

        runner_task = asyncio.create_task(runner())

        try:
            while True:
                event, data = await queue.get()
                if event == "__END__":
                    break
                yield _format_sse(event, data)
        except asyncio.CancelledError:
            runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await runner_task
            raise
        except Exception:
            runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await runner_task
            raise
        finally:
            if not runner_task.done():
                runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await runner_task

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse, tags=["tasks"])
async def get_task_status(
    task_id: str,
    memory: HybridMemoryService = Depends(get_hybrid_memory),
    state_store: OrchestratorStateStore | None = Depends(get_orchestrator_state_store),
) -> TaskStatusResponse:
    result_state: dict[str, Any] | None = None
    run_record: OrchestratorRun | None = None
    events: list[OrchestratorEvent] = []

    payload = await memory.fetch_ephemeral_memory(task_id)
    if isinstance(payload, dict):
        candidate = payload.get("result")
        if isinstance(candidate, dict):
            result_state = dict(candidate)

    if state_store is not None:
        run_uuid: uuid.UUID | None = None
        if result_state is not None:
            run_id_value = result_state.get("run_id")
            if isinstance(run_id_value, uuid.UUID):
                run_uuid = run_id_value
            elif isinstance(run_id_value, str) and run_id_value:
                try:
                    run_uuid = uuid.UUID(run_id_value)
                except ValueError:
                    run_uuid = None
        if run_uuid is not None:
            run_record = await state_store.get_run(run_uuid)
        if run_record is None:
            run_record = await state_store.get_latest_run_for_task(task_id)
        if run_record is not None:
            events = await state_store.list_events(run_record.run_id, limit=MAX_TASK_EVENTS)
            if result_state is None:
                result_state = dict(run_record.state)
            else:
                result_state.setdefault("status", run_record.status.value)
                result_state.setdefault("run_id", str(run_record.run_id))

    if result_state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    return _task_status_response(
        task_id=task_id,
        result_state=result_state,
        run_record=run_record,
        events=events,
    )


@router.get("/orchestrator/runs/{run_id}", response_model=OrchestratorRunDetail, tags=["orchestration"])
async def get_orchestrator_run(
    run_id: uuid.UUID,
    state_store: OrchestratorStateStore | None = Depends(get_orchestrator_state_store),
) -> OrchestratorRunDetail:
    if state_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator run persistence unavailable",
        )
    run_record = await state_store.get_run(run_id)
    if run_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    events = await state_store.list_events(run_id)
    return OrchestratorRunDetail.from_domain(run_record, events)


@router.get("/history/{task_id}", response_model=list[TaskResult], tags=["tasks"])
async def get_task_history(
    task_id: str,
    memory: HybridMemoryService = Depends(get_hybrid_memory),
) -> list[TaskResult]:
    result_state = await _fetch_result_state(task_id, memory=memory)
    outputs = result_state.get("outputs", []) if isinstance(result_state, dict) else []
    if not outputs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No results available")

    formatted: list[TaskResult] = []
    for item in outputs:
        raw_content = item.get("summary") or item.get("content") or ""
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        content: dict[str, Any] = {"text": raw_content, "metadata": metadata}

        formatted.append(
            TaskResult(
                task_id=task_id,
                agent=item.get("agent", "unknown"),
                content=content,
                confidence=float(item.get("confidence", 0.0)),
            )
        )

    return formatted


@router.get("/reports/{task_id}/dossier.json", tags=["tasks"])
async def get_task_dossier_json(
    task_id: str,
    memory: HybridMemoryService = Depends(get_hybrid_memory),
    _: dict[str, Any] = Depends(require_scopes(REPORTS_READ_SCOPE)),
) -> Response:
    result_state = await _fetch_result_state(task_id, memory=memory)
    dossier = result_state.get("dossier") if isinstance(result_state, dict) else None
    if not isinstance(dossier, dict):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Decision dossier unavailable")
    payload = dossier.get("json")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Decision dossier unavailable")
    return Response(
        content=json.dumps(payload),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=\"{task_id}-dossier.json\""},
    )


@router.get("/reports/{task_id}/dossier.md", tags=["tasks"])
async def get_task_dossier_markdown(
    task_id: str,
    memory: HybridMemoryService = Depends(get_hybrid_memory),
    _: dict[str, Any] = Depends(require_scopes(REPORTS_READ_SCOPE)),
) -> Response:
    result_state = await _fetch_result_state(task_id, memory=memory)
    dossier = result_state.get("dossier") if isinstance(result_state, dict) else None
    if not isinstance(dossier, dict):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Decision dossier unavailable")
    payload = dossier.get("markdown")
    if not isinstance(payload, str) or not payload.strip():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Decision dossier unavailable")
    return Response(
        content=payload,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=\"{task_id}-dossier.md\""},
    )


@router.get("/reviews/metrics", response_model=ReviewMetricsResponse, tags=["reviews"])
async def get_review_metrics(
    manager: ReviewManager = Depends(get_review_manager),
    _: dict[str, Any] = Depends(require_scopes(REVIEW_READ_SCOPE)),
    __: None = Depends(rate_limit_review_action),
) -> ReviewMetricsResponse:
    metrics = await manager.get_metrics()
    return ReviewMetricsResponse(**metrics)


@router.get("/reviews", response_model=list[ReviewTicketModel], tags=["reviews"])
async def list_reviews(
    status_filter: ReviewStatusLiteral | None = None,
    manager: ReviewManager = Depends(get_review_manager),
    _: dict[str, Any] = Depends(require_scopes(REVIEW_READ_SCOPE)),
    __: None = Depends(rate_limit_review_action),
) -> list[ReviewTicketModel]:
    try:
        status_enum = ReviewStatus(status_filter) if status_filter else None
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    tickets = await manager.list_tickets(status=status_enum)
    return [ReviewTicketModel.from_domain(ticket) for ticket in tickets]


@router.get("/reviews/{ticket_id}", response_model=ReviewTicketModel, tags=["reviews"])
async def get_review_ticket(
    ticket_id: uuid.UUID,
    manager: ReviewManager = Depends(get_review_manager),
    _: dict[str, Any] = Depends(require_scopes(REVIEW_READ_SCOPE)),
    __: None = Depends(rate_limit_review_action),
) -> ReviewTicketModel:
    ticket = await manager.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review ticket not found")
    return ReviewTicketModel.from_domain(ticket)


@router.post("/reviews/{ticket_id}/assign", response_model=ReviewTicketModel, tags=["reviews"])
async def assign_review_ticket(
    ticket_id: uuid.UUID,
    request: Request,
    manager: ReviewManager = Depends(get_review_manager),
    identity: dict[str, Any] = Depends(require_scopes(REVIEW_WRITE_SCOPE)),
    _: None = Depends(rate_limit_review_action),
) -> ReviewTicketModel:
    raw_payload = await _extract_json_body(request)
    try:
        payload = ReviewAssignmentRequest(**raw_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()) from exc
    reviewer_id = payload.reviewer_id or identity.get("sub")
    if not reviewer_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Reviewer id required")
    try:
        ticket = await manager.assign(ticket_id, reviewer_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review ticket not found") from None
    return ReviewTicketModel.from_domain(ticket)


@router.post("/reviews/{ticket_id}/notes", response_model=ReviewTicketModel, tags=["reviews"])
async def add_review_note(
    ticket_id: uuid.UUID,
    request: Request,
    manager: ReviewManager = Depends(get_review_manager),
    identity: dict[str, Any] = Depends(require_scopes(REVIEW_WRITE_SCOPE)),
    _: None = Depends(rate_limit_review_action),
) -> ReviewTicketModel:
    raw_payload = await _extract_json_body(request)
    try:
        payload = ReviewNoteCreate(**raw_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()) from exc
    author = payload.author or identity.get("sub")
    if not author:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Author required")
    note_content = (payload.content or "").strip()
    if not note_content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Note content required")
    try:
        await manager.add_note(ticket_id, author=author, content=note_content)
        ticket = await manager.get_ticket(ticket_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review ticket not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if ticket is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review ticket not found")
    return ReviewTicketModel.from_domain(ticket)


@router.patch("/reviews/{ticket_id}/unassign", response_model=ReviewTicketModel, tags=["reviews"])
async def unassign_review_ticket(
    ticket_id: uuid.UUID,
    manager: ReviewManager = Depends(get_review_manager),
    _: dict[str, Any] = Depends(require_scopes(REVIEW_WRITE_SCOPE)),
    __: None = Depends(rate_limit_review_action),
) -> ReviewTicketModel:
    try:
        ticket = await manager.unassign(ticket_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review ticket not found") from None
    return ReviewTicketModel.from_domain(ticket)


@router.post("/reviews/{ticket_id}/resolve", response_model=ReviewTicketModel, tags=["reviews"])
async def resolve_review_ticket(
    ticket_id: uuid.UUID,
    request: Request,
    manager: ReviewManager = Depends(get_review_manager),
    identity: dict[str, Any] = Depends(require_scopes(REVIEW_WRITE_SCOPE)),
    _: None = Depends(rate_limit_review_action),
) -> ReviewTicketModel:
    raw_payload = await _extract_json_body(request)
    try:
        payload = ReviewResolutionRequest(**raw_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()) from exc
    reviewer_id = payload.reviewer_id or identity.get("sub")
    if not reviewer_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Reviewer id required")
    try:
        ticket = await manager.resolve(
            ticket_id,
            status=payload.to_status(),
            reviewer=reviewer_id,
            summary=payload.summary,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review ticket not found") from None
    return ReviewTicketModel.from_domain(ticket)
