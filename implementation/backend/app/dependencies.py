from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends

from .core.config import Settings, get_settings
from .queue.manager import TaskQueueManager
from .services.llm import LLMService
from .services.memory import HybridMemoryService
from .services.notifications import ReviewNotificationService, log_notification
from .orchestration.review import ReviewManager, build_review_store


_review_manager_singleton: ReviewManager | None = None
_review_notification_service: ReviewNotificationService | None = None


def get_review_manager_singleton(settings: Settings) -> ReviewManager:
    global _review_manager_singleton, _review_notification_service
    if _review_manager_singleton is None:
        store = build_review_store(settings)
        notification_service = ReviewNotificationService(settings.escalation)
        if settings.escalation.audit_log_enabled:
            notification_service.subscribe(log_notification)
        _review_notification_service = notification_service
        _review_manager_singleton = ReviewManager(
            store=store,
            settings=settings.escalation,
            notifications=_review_notification_service,
        )
    return _review_manager_singleton


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


async def get_llm_service(
    settings: Settings = Depends(get_settings),
) -> AsyncIterator[LLMService]:
    yield LLMService.from_settings(settings)


async def get_review_manager(
    settings: Settings = Depends(get_settings),
) -> AsyncIterator[ReviewManager]:
    yield get_review_manager_singleton(settings)
