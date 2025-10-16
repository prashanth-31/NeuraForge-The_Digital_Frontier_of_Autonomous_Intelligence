from __future__ import annotations

import asyncio
from time import time
from typing import Awaitable, Callable

from fastapi import Depends, HTTPException, Request, status

try:
    from redis.asyncio import Redis
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Redis = None  # type: ignore[misc,assignment]

from .config import RateLimitRule, Settings, get_settings
from .logging import get_logger
from .security import decode_access_token

logger = get_logger(name=__name__)

_rate_limit_client: Redis | None = None  # type: ignore[name-defined]
_rate_limit_lock = asyncio.Lock()


async def _get_rate_limit_client(settings: Settings) -> Redis | None:  # type: ignore[name-defined]
    global _rate_limit_client
    if Redis is None:
        logger.warning("rate_limit_redis_unavailable")
        return None
    if _rate_limit_client is not None:
        return _rate_limit_client
    async with _rate_limit_lock:
        if _rate_limit_client is None:
            _rate_limit_client = Redis.from_url(str(settings.redis.url))
        return _rate_limit_client


async def _consume_bucket(
    *,
    redis: Redis | None,  # type: ignore[name-defined]
    key: str,
    rule: RateLimitRule,
) -> tuple[bool, int | None]:
    if redis is None:
        return True, None
    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, rule.window_seconds)
        if count > rule.capacity:
            ttl = await redis.ttl(key)
            return False, ttl if ttl and ttl > 0 else rule.window_seconds
        return True, None
    except Exception as exc:  # pragma: no cover - connection errors route here in tests
        logger.warning(
            "rate_limit_redis_error",
            error=str(exc),
            key=key,
            window=rule.window_seconds,
        )
        return True, None


def _resolve_subject(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except HTTPException:
        return None
    subject = payload.get("sub")
    if subject is None:
        return None
    return str(subject)


def _identifier_from_request(request: Request, subject: str | None) -> str:
    if subject:
        return subject
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    client = request.client
    if client and client.host:
        return client.host
    return "anonymous"


def rate_limit_dependency(rule_name: str) -> Callable[[Request, Settings], Awaitable[None]]:
    async def _dependency(
        request: Request,
        settings: Settings = Depends(get_settings),
    ) -> None:
        config = getattr(settings, "rate_limit", None)
        if config is None or not getattr(config, "enabled", False):
            return
        rule: RateLimitRule | None = getattr(config, rule_name, None)
        if rule is None:
            return
        redis = await _get_rate_limit_client(settings)
        subject = _resolve_subject(request)
        identifier = _identifier_from_request(request, subject)
        window_started = int(time() // max(1, rule.window_seconds))
        key = f"{config.namespace}:{rule_name}:{identifier}:{window_started}"
        allowed, retry_after = await _consume_bucket(redis=redis, key=key, rule=rule)
        if allowed:
            return
        headers = {"Retry-After": str(max(1, int(retry_after or rule.window_seconds)))}
        logger.warning(
            "rate_limit_exceeded",
            rule=rule_name,
            identifier=identifier,
            retry_after=headers["Retry-After"],
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests",
            headers=headers,
        )

    return _dependency


__all__ = ["rate_limit_dependency"]
