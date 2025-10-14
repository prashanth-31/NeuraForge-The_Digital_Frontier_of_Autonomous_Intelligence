from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, get_settings


def create_access_token(
    subject: str,
    *,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.auth.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {"exp": expire, "sub": subject}
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(
        payload,
        settings.auth.jwt_secret_key,
        algorithm=settings.auth.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.auth.jwt_secret_key,
            algorithms=[settings.auth.jwt_algorithm],
        )
    except jwt.PyJWTError as exc:  # pragma: no cover - error path covered by HTTPException
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        ) from exc


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_subject(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return decode_access_token(credentials.credentials)


def require_roles(*roles: str):
    def _dependency(
        token_payload: dict[str, Any] = Depends(get_current_subject),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        assigned_roles = token_payload.get("roles")
        if isinstance(assigned_roles, str):
            assigned_roles = [assigned_roles]
        if not isinstance(assigned_roles, (list, tuple, set)):
            assigned_roles = []
        normalized_roles = {str(role).lower() for role in assigned_roles}

        if settings.auth.superuser_email and token_payload.get("sub") == settings.auth.superuser_email:
            return token_payload

        if settings.auth.service_token and token_payload.get("sub") == settings.auth.service_token:
            return token_payload

        required = {role.lower() for role in roles if role}
        if required and not (normalized_roles & required):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return token_payload

    return _dependency
