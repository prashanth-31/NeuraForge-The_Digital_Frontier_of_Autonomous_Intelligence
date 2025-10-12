from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_hybrid_memory, get_task_queue
from ..orchestration.graph import Orchestrator
from ..schemas.tasks import TaskRequest, TaskResponse, TaskResult
from ..services.memory import HybridMemoryService

router = APIRouter()


@router.get("/health", tags=["health"])
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/submit_task", response_model=TaskResponse, tags=["tasks"])
async def submit_task(
    payload: TaskRequest,
    queue=Depends(get_task_queue),
    memory: HybridMemoryService = Depends(get_hybrid_memory),
) -> TaskResponse:
    task_id = str(uuid.uuid4())

    async def _job() -> None:
        # Placeholder orchestration call; in a full implementation this would
        # route through LangGraph with the configured agents.
        orchestrator = Orchestrator(
            agents=[],
        )
        result = await orchestrator.route_task({"id": task_id, **payload.dict()})
        await memory.store_ephemeral_memory(task_id, {"result": result})

    await queue.enqueue(_job)
    return TaskResponse(task_id=task_id, status="queued")


@router.get("/history/{task_id}", response_model=list[TaskResult], tags=["tasks"])
async def get_task_history(
    task_id: str,
    memory: HybridMemoryService = Depends(get_hybrid_memory),
) -> list[TaskResult]:
    # Placeholder implementation using semantic memory
    results: list[TaskResult] = []
    context = await memory.retrieve_context(query_vector=[0.0], limit=5)
    for item in context:
        if item.get("task_id") == task_id:
            results.append(
                TaskResult(
                    task_id=task_id,
                    agent=item.get("agent", "unknown"),
                    content=item.get("content", {}),
                    confidence=float(item.get("confidence", 0.0)),
                )
            )
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return results
