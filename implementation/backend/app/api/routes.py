from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ..agents.base import AgentContext
from ..agents.creative import CreativeAgent
from ..agents.enterprise import EnterpriseAgent
from ..agents.finance import FinanceAgent
from ..agents.research import ResearchAgent
from ..core.config import Settings, get_settings
from ..core.logging import get_logger
from ..dependencies import get_hybrid_memory, get_task_queue
from ..orchestration.graph import Orchestrator
from ..schemas.tasks import TaskRequest, TaskResponse, TaskResult
from ..services.embedding import EmbeddingService
from ..services.llm import LLMService
from ..services.memory import HybridMemoryService
from ..services.retrieval import ContextAssembler, RetrievalService
from ..services.scoring import ConfidenceScorer
from ..services.tools import get_tool_service

logger = get_logger(name=__name__)

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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
    payload: TaskRequest,
    queue=Depends(get_task_queue),
    settings: Settings = Depends(get_settings),
) -> TaskResponse:
    task_id = str(uuid.uuid4())
    task_payload = payload.model_dump()

    async def _job() -> None:
        memory_service = HybridMemoryService.from_settings(settings)
        orchestrator = Orchestrator(
            agents=[
                ResearchAgent(),
                FinanceAgent(),
                CreativeAgent(),
                EnterpriseAgent(),
            ]
        )
        state = {
            "id": task_id,
            "prompt": task_payload["prompt"],
            "metadata": task_payload.get("metadata", {}),
            "outputs": [],
        }
        async with memory_service.lifecycle() as memory:
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

    await queue.enqueue(_job)
    return TaskResponse(task_id=task_id, status="queued")


@router.post("/submit_task/stream", tags=["tasks"])
async def submit_task_stream(
    payload: TaskRequest,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    task_id = str(uuid.uuid4())
    task_payload = payload.model_dump()

    orchestrator = Orchestrator(
        agents=[
            ResearchAgent(),
            FinanceAgent(),
            CreativeAgent(),
            EnterpriseAgent(),
        ]
    )

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
            try:
                async with memory_service.lifecycle() as memory:
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

                    embedding_service: EmbeddingService | None = None
                    context_assembler: ContextAssembler | None = None
                    try:
                        embedding_service = EmbeddingService.from_settings(settings, memory_service=memory)
                        retrieval_service = RetrievalService.from_settings(settings, memory=memory, embedder=embedding_service)
                        context_assembler = ContextAssembler(retrieval=retrieval_service)
                    except Exception as exc:  # pragma: no cover - optional fallback
                        logger.warning("embedding_initialization_failed", error=str(exc))
                        embedding_service = None
                        context_assembler = None

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
    payload = await memory.fetch_ephemeral_memory(task_id)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    result_state = payload.get("result") if isinstance(payload, dict) else None
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
