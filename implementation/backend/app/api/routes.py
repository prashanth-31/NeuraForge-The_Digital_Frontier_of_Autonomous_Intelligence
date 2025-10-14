from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from ..agents.base import AgentContext, BaseAgent
from ..agents.creative import CreativeAgent
from ..agents.enterprise import EnterpriseAgent
from ..agents.finance import FinanceAgent
from ..agents.research import ResearchAgent
from ..core.config import Settings, get_settings
from ..core.logging import get_logger
from ..core.security import require_roles
from ..dependencies import (
    get_hybrid_memory,
    get_review_manager,
    get_review_manager_singleton,
    get_task_queue,
)
from ..orchestration.graph import Orchestrator
from ..orchestration.context import ContextAssemblyContract, ContextSnapshotStore, ContextStage
from ..orchestration.guardrails import GuardrailManager, GuardrailStore
from ..orchestration.lifecycle import TaskLifecycleStore
from ..orchestration.negotiation import SimpleNegotiationEngine
from ..orchestration.planner import DependencyTaskPlanner, SimpleTaskPlanner
from ..orchestration.scheduler import AsyncioTaskScheduler, SequentialTaskScheduler
from ..orchestration.store import OrchestratorStateStore
from ..orchestration.meta import MetaAgent
from ..orchestration.review import ReviewManager, ReviewStatus
from ..schemas.reviews import (
    ReviewAssignmentRequest,
    ReviewNoteCreate,
    ReviewResolutionRequest,
    ReviewStatusLiteral,
    ReviewTicketModel,
)
from ..schemas.tasks import TaskRequest, TaskResponse, TaskResult
from ..services.embedding import EmbeddingService
from ..services.llm import LLMService
from ..services.memory import HybridMemoryService
from ..services.retrieval import ContextAssembler, RetrievalService
from ..services.scoring import ConfidenceScorer
from ..services.disputes import DisputeDetector, MetaConfidenceScorer
from ..services.tools import get_tool_service

logger = get_logger(name=__name__)

router = APIRouter()


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

                try:
                    llm_service = LLMService.from_settings(settings)
                except RuntimeError as exc:
                    logger.exception("llm_initialization_failed", error=str(exc))
                    failure = {
                        **state,
                        "status": "failed",
                        "error": "LLM service unavailable",
                    }
                    await memory.store_ephemeral_memory(task_id, {"result": failure})
                    return

                agents = [
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
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    raw_payload = await _extract_json_body(request)
    try:
        payload = TaskRequest(**raw_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()) from exc
    task_id = str(uuid.uuid4())
    task_payload = payload.model_dump()

    state = {
        "id": task_id,
        "prompt": task_payload["prompt"],
        "metadata": task_payload.get("metadata", {}),
        "outputs": [],
    }

    async def event_stream():
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

        async def emit(event: str, data: dict[str, Any]) -> None:
            await queue.put((event, data))

        async def progress(event_payload: dict[str, Any]) -> None:
            payload_copy = {key: value for key, value in event_payload.items() if key != "event"}
            payload_copy["timestamp"] = _now_iso()
            event_type = f"agent_{event_payload.get('event', 'update')}"
            await emit(event_type, payload_copy)

        async def runner() -> None:
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
                            await memory.store_ephemeral_memory(task_id, {"result": failure})
                            await emit(
                                "task_failed",
                                {"task_id": task_id, "error": failure["error"], "timestamp": failure["timestamp"]},
                            )
                            return

                        agents = [
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
                                await emit("tool_invocation", event_payload)

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
                            await memory.store_ephemeral_memory(task_id, {"result": result})
                            await emit(
                                "task_completed",
                                {
                                    "task_id": task_id,
                                    "status": result.get("status", "completed"),
                                    "timestamp": _now_iso(),
                                },
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
        finally:
            await runner_task

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


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


REVIEW_ROLES = ("reviewer", "review_admin")


@router.get("/reviews", response_model=list[ReviewTicketModel], tags=["reviews"])
async def list_reviews(
    status_filter: ReviewStatusLiteral | None = None,
    manager: ReviewManager = Depends(get_review_manager),
    _: dict[str, Any] = Depends(require_roles(*REVIEW_ROLES)),
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
    _: dict[str, Any] = Depends(require_roles(*REVIEW_ROLES)),
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
    identity: dict[str, Any] = Depends(require_roles(*REVIEW_ROLES)),
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
    identity: dict[str, Any] = Depends(require_roles(*REVIEW_ROLES)),
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
    _: dict[str, Any] = Depends(require_roles(*REVIEW_ROLES)),
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
    identity: dict[str, Any] = Depends(require_roles(*REVIEW_ROLES)),
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
