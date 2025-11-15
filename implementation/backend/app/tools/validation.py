from __future__ import annotations

import json
from typing import Any, Mapping

from app.core.logging import get_logger
from app.tools.catalog_store import ToolCatalogEntry

logger = get_logger(name=__name__)


class ToolPayloadValidationError(ValueError):
    """Raised when a tool payload violates schema or structural guards."""


class ToolPayloadValidator:
    def validate(
        self,
        *,
        resolved_tool: str,
        payload: Mapping[str, Any],
        descriptor: Any | None,
        catalog_entry: ToolCatalogEntry | None,
        max_string_length: int,
    ) -> None:
        schema = None
        if catalog_entry is not None and catalog_entry.input_schema:
            schema = catalog_entry.input_schema
        elif descriptor is not None and getattr(descriptor, "input_schema", None):
            schema = descriptor.input_schema
        if schema:
            self._validate_schema(resolved_tool, payload, schema)
        self._guard_payload_shape(resolved_tool, payload, max_length=max_string_length)
        self._ensure_serializable(resolved_tool, payload)

    def _validate_schema(self, tool: str, payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
        if not isinstance(schema, Mapping):
            return
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        additional_allowed = schema.get("additionalProperties", True)

        missing = [field for field in required if field not in payload]
        if missing:
            raise ToolPayloadValidationError(
                f"Payload for '{tool}' missing required fields: {', '.join(sorted(set(missing)))}"
            )

        if additional_allowed is False and isinstance(properties, Mapping):
            extraneous = [field for field in payload if field not in properties]
            if extraneous:
                raise ToolPayloadValidationError(
                    f"Payload for '{tool}' includes unsupported fields: {', '.join(sorted(set(extraneous)))}"
                )

    def _guard_payload_shape(self, tool: str, value: Any, *, max_length: int, key_path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                next_path = f"{key_path}.{key}" if key_path else str(key)
                self._guard_payload_shape(tool, item, max_length=max_length, key_path=next_path)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                next_path = f"{key_path}[{index}]" if key_path else f"[{index}]"
                self._guard_payload_shape(tool, item, max_length=max_length, key_path=next_path)
            return
        if isinstance(value, str) and len(value) > max_length:
            raise ToolPayloadValidationError(
                f"Payload field '{key_path or '<root>'}' for '{tool}' exceeds maximum length of {max_length} characters"
            )

    def _ensure_serializable(self, tool: str, payload: Mapping[str, Any]) -> None:
        try:
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise ToolPayloadValidationError(f"Payload for '{tool}' is not JSON-serializable") from exc


tool_payload_validator = ToolPayloadValidator()

__all__ = [
    "ToolPayloadValidator",
    "ToolPayloadValidationError",
    "tool_payload_validator",
]
