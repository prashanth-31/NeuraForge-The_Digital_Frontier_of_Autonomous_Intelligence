from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends

from .core.config import Settings, get_settings
from .queue.manager import TaskQueueManager
from .services.memory import HybridMemoryService


async def get_app_settings() -> AsyncIterator[Settings]:
    yield get_settings()


async def get_task_queue(
    settings: Settings = Depends(get_settings),
) -> AsyncIterator[TaskQueueManager]:
    queue = TaskQueueManager.from_settings(settings)
    async with queue.lifecycle():
        yield queue


async def get_hybrid_memory(
    settings: Settings = Depends(get_settings),
) -> AsyncIterator[HybridMemoryService]:
    memory = HybridMemoryService.from_settings(settings)
    async with memory.lifecycle():
        yield memory
