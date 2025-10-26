from __future__ import annotations

import hashlib
from time import perf_counter
from typing import Iterable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from .logging import get_logger
from .security import decode_access_token


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Emit structured audit events for authenticated requests."""

    def __init__(self, app: ASGIApp, *, include_prefixes: Iterable[str] = ("/api/", "/metrics")) -> None:
        super().__init__(app)
        self._prefixes = tuple(include_prefixes)
        self._logger = get_logger(name="audit")

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        if self._prefixes and not any(request.url.path.startswith(prefix) for prefix in self._prefixes):
            return await call_next(request)

        start = perf_counter()
        payload_hash = None
        body = b""
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            body = await request.body()
            request._body = body  # type: ignore[attr-defined]
            if body:
                payload_hash = hashlib.sha256(body).hexdigest()

        subject = None
        roles: list[str] = []
        scopes: list[str] = []
        authorization = request.headers.get("authorization")
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
            if token:
                try:
                    payload = decode_access_token(token)
                except Exception:  # pragma: no cover - invalid token paths are logged downstream
                    payload = None
                if isinstance(payload, dict):
                    subject = str(payload.get("sub") or "") or None
                    raw_roles = payload.get("roles")
                    if isinstance(raw_roles, str):
                        if "," in raw_roles:
                            raw_roles = [segment.strip() for segment in raw_roles.split(",") if segment.strip()]
                        else:
                            raw_roles = [raw_roles]
                    if isinstance(raw_roles, (list, tuple, set)):
                        roles = [str(role) for role in raw_roles if role]
                    raw_scope = payload.get("scopes") or payload.get("scope")
                    if isinstance(raw_scope, str):
                        scopes = [segment for segment in raw_scope.split() if segment]
                    elif isinstance(raw_scope, (list, tuple, set)):
                        scopes = [str(item) for item in raw_scope if item]

        response = await call_next(request)
        duration = perf_counter() - start
        retry_after = response.headers.get("Retry-After")

        self._logger.info(
            "audit_log",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            subject=subject or "anonymous",
            roles=roles,
            scopes=scopes,
            payload_hash=payload_hash,
            content_length=len(body) if body else 0,
            client_ip=(request.client.host if request.client else None),
            duration_ms=round(duration * 1000, 3),
            retry_after=retry_after,
        )
        return response


__all__ = ["AuditLoggingMiddleware"]
