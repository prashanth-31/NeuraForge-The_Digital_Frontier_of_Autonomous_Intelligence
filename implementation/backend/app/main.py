from __future__ import annotations

from fastapi import FastAPI

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .api.routes import router as api_router
from .core.config import get_settings
from .core.logging import configure_logging

settings = get_settings()
configure_logging(settings.observability.log_level)

app = FastAPI(title="NeuraForge Backend", version="0.1.0")
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"message": "NeuraForge backend running"}


if settings.observability.prometheus_enabled:

    @app.get("/metrics", tags=["observability"])
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
