from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.tool_reconciliation import ToolReconciliationJob
from app.services.tools import ToolConfigurationError, get_tool_service

try:  # pragma: no cover - optional dependency for semantic store
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.http.models import Distance, VectorParams
except ModuleNotFoundError:  # pragma: no cover - qdrant optional in test environments
    AsyncQdrantClient = None  # type: ignore[assignment]
    UnexpectedResponse = None  # type: ignore[assignment]
    Distance = None  # type: ignore[assignment]
    VectorParams = None  # type: ignore[assignment]


logger = get_logger(name=__name__)


async def ensure_foundation_ready(settings: Settings) -> None:
    """Ensure datastore collections and external tooling are ready before serving."""

    try:
        await _ensure_qdrant_collection(settings)
    except Exception as exc:  # pragma: no cover - defensive guard during startup
        if settings.environment == "production":
            raise
        logger.warning(
            "qdrant_bootstrap_degraded",
            error=str(exc),
            environment=settings.environment,
        )

    try:
        await _wait_for_mcp_readiness(settings)
    except Exception as exc:  # pragma: no cover - defensive guard during startup
        if settings.environment == "production":
            raise
        logger.warning(
            "mcp_bootstrap_degraded",
            error=str(exc),
            environment=settings.environment,
        )


async def _ensure_qdrant_collection(settings: Settings) -> None:
    if AsyncQdrantClient is None or Distance is None or VectorParams is None:
        logger.warning("qdrant_client_unavailable", message="Skipping collection bootstrap; client library missing")
        return

    collection = settings.qdrant.collection_name
    dimension = settings.embedding.preferred_dimension or 384
    max_attempts = 5
    delay = 1.5
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        client = AsyncQdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key,
            prefer_grpc=False,
            timeout=10.0,
        )
        try:
            await _create_collection_if_missing(client, collection, dimension)
        except Exception as exc:  # pragma: no cover - network/service errors
            last_error = exc
            logger.warning(
                "qdrant_collection_probe_failed",
                collection=collection,
                attempt=attempt,
                error=str(exc),
            )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)
        else:
            logger.info("qdrant_collection_ready", collection=collection, dimension=dimension)
            return
        finally:
            with contextlib.suppress(Exception):
                await client.close()

    if last_error is not None:
        raise RuntimeError(f"Failed to ensure Qdrant collection '{collection}' is available") from last_error


async def _create_collection_if_missing(
    client: Any,
    collection: str,
    dimension: int,
) -> None:
    try:
        info = await client.get_collection(collection_name=collection)
    except UnexpectedResponse as exc:
        if exc.status_code != 404:  # type: ignore[attr-defined]
            raise
        await client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            on_disk_payload=True,
        )
        return

    config = getattr(info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    size: int | None = None
    if isinstance(vectors, dict):
        first_vector = next(iter(vectors.values()), None)
        size = getattr(first_vector, "size", None)
    elif hasattr(vectors, "size"):
        size = getattr(vectors, "size", None)

    if size and size != dimension:
        logger.warning(
            "qdrant_collection_dimension_mismatch",
            collection=collection,
            expected=dimension,
            actual=size,
        )


async def _wait_for_mcp_readiness(settings: Settings) -> None:
    if not settings.tools.mcp.enabled:
        logger.info("mcp_disabled", message="Skipping MCP readiness check")
        return

    deadline = time.monotonic() + 45.0
    delay = 1.0
    attempt = 0
    last_error: str | None = None

    while True:
        attempt += 1
        try:
            service = await get_tool_service()
            await service.initialize(validate=True)
            await ToolReconciliationJob.run_once()
        except ToolConfigurationError as exc:
            last_error = str(exc)
        except Exception as exc:  # pragma: no cover - defensive catch
            last_error = str(exc)
        else:
            logger.info("mcp_ready", attempt=attempt)
            return

        if time.monotonic() >= deadline:
            detail = last_error or "no diagnostic message"
            raise RuntimeError("MCP tooling failed health checks during startup") from ToolConfigurationError(detail)

        logger.warning("mcp_not_ready", attempt=attempt, retry_in=round(delay, 2), error=last_error)
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 6.0)