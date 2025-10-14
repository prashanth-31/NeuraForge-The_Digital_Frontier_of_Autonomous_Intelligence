from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, TYPE_CHECKING

import httpx

from ..core.config import EscalationSettings

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..orchestration.review import ReviewNote, ReviewTicket
else:  # pragma: no cover - runtime hints only
    ReviewNote = Any
    ReviewTicket = Any

logger = logging.getLogger(__name__)

ReviewEventType = Literal[
    "review.ticket.created",
    "review.ticket.assigned",
    "review.ticket.reassigned",
    "review.ticket.unassigned",
    "review.ticket.resolved",
    "review.ticket.dismissed",
    "review.ticket.note_added",
]


@dataclass(slots=True)
class ReviewEvent:
    event: ReviewEventType
    ticket: ReviewTicket
    note: ReviewNote | None = None
    actor: str | None = None

    def to_payload(self) -> dict[str, object]:
        base: dict[str, object] = {
            "event": self.event,
            "ticket": {
                "id": str(self.ticket.ticket_id),
                "status": self.ticket.status.value,
                "reviewer": self.ticket.assigned_to,
                "summary": self.ticket.summary,
                "sources": self.ticket.sources,
                "created_at": self.ticket.created_at.isoformat(),
                "updated_at": self.ticket.updated_at.isoformat(),
            },
        }
        if self.actor:
            base["actor"] = self.actor
        if self.note is not None:
            base["note"] = {
                "id": str(self.note.note_id),
                "author": self.note.author,
                "content": self.note.content,
                "created_at": self.note.created_at.isoformat(),
            }
        return base


Subscriber = Callable[[ReviewEvent], Awaitable[None]]


class ReviewNotificationService:
    def __init__(self, settings: EscalationSettings) -> None:
        self._settings = settings
        self._subscribers: set[Subscriber] = set()
        self._webhook_url = settings.notification_webhook_url
        self._http_timeout = settings.notification_timeout_seconds
        self._default_recipients = frozenset(settings.notification_recipients or [])

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.add(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.discard(subscriber)

    async def publish(self, event: ReviewEvent) -> None:
        if not self._settings.enabled:
            return

        payload = event.to_payload()
        tasks: list[Awaitable[object]] = []

        for subscriber in list(self._subscribers):
            tasks.append(self._safe_invoke(subscriber, event))

        if self._webhook_url:
            tasks.append(self._post_webhook(payload))

        if not tasks:
            logger.debug("review_notification_skipped", event=payload)
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("review_notification_error", error=str(result), exc_info=result)

    async def _safe_invoke(self, subscriber: Subscriber, event: ReviewEvent) -> None:
        try:
            await subscriber(event)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("review_subscriber_failed", subscriber=subscriber.__qualname__, error=str(exc))

    async def _post_webhook(self, payload: dict[str, object]) -> None:
        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                response = await client.post(
                    self._webhook_url,
                    content=json.dumps(
                        {
                            "recipients": list(self._default_recipients),
                            "payload": payload,
                        }
                    ),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
        except Exception as exc:  # pragma: no cover - webhook best effort
            logger.warning("review_webhook_failed", error=str(exc))


async def log_notification(event: ReviewEvent) -> None:
    logger.info("review_notification", event=event.event, ticket=str(event.ticket.ticket_id))