from __future__ import annotations

import asyncio
import contextlib
from asyncio import Queue
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

try:
    from redis.asyncio import Redis
except ModuleNotFoundError:  # pragma: no cover - optional redis dependency
    Redis = None  # type: ignore[misc,assignment]

from ..core.config import Settings
from ..core.logging import get_logger

logger = get_logger(name=__name__)


class TaskQueueManager:
    def __init__(self, *, redis: Redis | None = None) -> None:  # type: ignore[name-defined]
        self._redis = redis
        self._queue: Queue[Callable[[], Awaitable[Any]]] = Queue()
        self._consumer_task: asyncio.Task[None] | None = None

    async def enqueue(self, job: Callable[[], Awaitable[Any]]) -> None:
        await self._queue.put(job)
        logger.info("task_enqueued", queue_size=self._queue.qsize())

    async def _consumer(self) -> None:
        logger.info("task_queue_started")
        while True:
            job = await self._queue.get()
            try:
                await job()
            except Exception as exc:  # pragma: no cover - log unexpected execution errors
                logger.exception("task_execution_failed", error=str(exc))
            finally:
                self._queue.task_done()

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator["TaskQueueManager"]:
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def start(self) -> None:
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._consumer())

    async def stop(self) -> None:
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):  # type: ignore[name-defined]
                await self._consumer_task
            self._consumer_task = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "TaskQueueManager":
        if Redis and settings.redis.url:
            redis = Redis.from_url(str(settings.redis.url), db=settings.redis.task_queue_db)
            return cls(redis=redis)
        return cls()
