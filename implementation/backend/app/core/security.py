from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, get_settings


ROLE_SCOPE_MAP: dict[str, set[str]] = {
    "reviewer": {"reviews:read", "reviews:write", "reports:read"},
    "review_admin": {"reviews:read", "reviews:write", "reviews:admin", "reports:read"},
    "observer": {"reviews:read", "reports:read"},
}


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


def get_optional_subject(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> dict[str, Any] | None:
    if credentials is None:
        return None
    try:
        return decode_access_token(credentials.credentials)
    except HTTPException:
        return None


def _is_unrestricted(token_payload: dict[str, Any], settings: Settings) -> bool:
    if settings.auth.superuser_email and token_payload.get("sub") == settings.auth.superuser_email:
        return True
    if settings.auth.service_token and token_payload.get("sub") == settings.auth.service_token:
        return True
    return False


def _normalize_roles(token_payload: dict[str, Any]) -> set[str]:
    assigned_roles = token_payload.get("roles")
    if isinstance(assigned_roles, str):
        if "," in assigned_roles:
            assigned_roles = [segment.strip() for segment in assigned_roles.split(",") if segment.strip()]
        else:
            assigned_roles = [assigned_roles]
    if not isinstance(assigned_roles, (list, tuple, set)):
        return set()
    return {str(role).lower() for role in assigned_roles if role}


def _resolve_scopes(token_payload: dict[str, Any]) -> set[str]:
    explicit_scopes: set[str] = set()
    for key in ("scope", "scopes"):
        claim = token_payload.get(key)
        if isinstance(claim, str):
            explicit_scopes.update(segment.strip() for segment in claim.split() if segment.strip())
        elif isinstance(claim, (list, tuple, set)):
            explicit_scopes.update(str(item).strip() for item in claim if str(item).strip())
    roles = _normalize_roles(token_payload)
    for role in roles:
        explicit_scopes.update(ROLE_SCOPE_MAP.get(role, set()))
    return {scope for scope in explicit_scopes if scope}


def require_roles(*roles: str):
    def _dependency(
        token_payload: dict[str, Any] = Depends(get_current_subject),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        if _is_unrestricted(token_payload, settings):
            return token_payload
        normalized_roles = _normalize_roles(token_payload)
        required = {role.lower() for role in roles if role}
        if required and not (normalized_roles & required):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return token_payload

    return _dependency


def require_scopes(*scopes: str):
    def _dependency(
        token_payload: dict[str, Any] = Depends(get_current_subject),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        if _is_unrestricted(token_payload, settings):
            return token_payload
        granted = _resolve_scopes(token_payload)
        required = {scope for scope in scopes if scope}
        if required and not required.issubset(granted):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return token_payload

    return _dependency


__all__ = [
    "create_access_token",
    "decode_access_token",
    "get_current_subject",
    "get_optional_subject",
    "require_roles",
    "require_scopes",
    "ROLE_SCOPE_MAP",
]
