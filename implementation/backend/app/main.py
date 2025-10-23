from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .api.routes import router as api_router
from .core.audit import AuditLoggingMiddleware
from .core.config import get_settings
from .core.logging import configure_logging, get_logger
from .mcp.router import router as mcp_router
from .services.consolidation import ConsolidationJob

async def _consolidation_loop() -> None:
    interval = max(5, settings.consolidation.interval_seconds)
    while True:
        await asyncio.sleep(interval)
        try:
            await ConsolidationJob.run_once_from_settings(settings)
        except Exception as exc:  # pragma: no cover - background error logging
            logger.exception("consolidation_loop_failed", error=str(exc))


settings = get_settings()
configure_logging(settings.observability.log_level)
logger = get_logger(name=__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    task: asyncio.Task[None] | None = None
    if settings.consolidation.enabled:
        task = asyncio.create_task(_consolidation_loop())
        app.state.consolidation_task = task
    try:
        yield
    finally:
        task = getattr(app.state, "consolidation_task", None)
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):  # pragma: no cover - managed shutdown
                await task


app = FastAPI(title="NeuraForge Backend", version="0.1.0", lifespan=app_lifespan)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.api_v1_prefix)
app.include_router(mcp_router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"message": "NeuraForge backend running"}


if settings.observability.prometheus_enabled:

    @app.get("/metrics", tags=["observability"])
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
