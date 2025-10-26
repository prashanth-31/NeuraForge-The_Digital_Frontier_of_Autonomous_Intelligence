from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (UUID, Decimal)):
        return str(value)
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - fallback to string repr
            return str(value)
    return str(value)


def encode_jsonb(value: Any) -> str:
    """Serialize the given value into a JSON string suitable for jsonb columns."""
    try:
        return json.dumps(value)
    except TypeError:
        return json.dumps(value, default=_json_default)


def decode_jsonb(value: Any) -> Any:
    """Decode a jsonb column value into native Python structures."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    if hasattr(value, "items"):
        try:
            return dict(value.items())  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fallback to string parse
            return json.loads(json.dumps(value, default=_json_default))
    return value
