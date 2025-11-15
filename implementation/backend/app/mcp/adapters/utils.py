from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Mapping

from pydantic import BaseModel, Field

from .base import MCPToolAdapter


class UtilsHashInput(BaseModel):
    value: Any
    algorithm: str = Field("sha256", pattern=r"^(md5|sha1|sha256|sha512)$")

    model_config = {"extra": "forbid"}


class UtilsHashOutput(BaseModel):
    algorithm: str
    digest: str


class UtilsHashAdapter(MCPToolAdapter):
    name = "utils/hash"
    description = "Produces deterministic hashes for debugging pipelines."
    labels = ("utils", "hash")
    aliases = ("utils.hash",)
    capabilities = ("utility",)
    InputModel = UtilsHashInput
    OutputModel = UtilsHashOutput

    async def _invoke(self, payload_model: UtilsHashInput) -> Mapping[str, Any]:
        data = json.dumps(payload_model.value, sort_keys=True, default=str).encode("utf-8")
        digest = getattr(hashlib, payload_model.algorithm)(data).hexdigest()
        return {"algorithm": payload_model.algorithm, "digest": digest}


class UtilsTimestampInput(BaseModel):
    timezone: str | None = Field(default="UTC", pattern=r"^(UTC)$")

    model_config = {"extra": "forbid"}


class UtilsTimestampOutput(BaseModel):
    timestamp: str


class UtilsTimestampAdapter(MCPToolAdapter):
    name = "utils/timestamp"
    description = "Returns the current UTC timestamp in ISO-8601 format."
    labels = ("utils", "time")
    aliases = ("utils.timestamp",)
    capabilities = ("utility",)
    InputModel = UtilsTimestampInput
    OutputModel = UtilsTimestampOutput

    async def _invoke(self, payload_model: UtilsTimestampInput) -> Mapping[str, Any]:
        now = datetime.now(UTC)
        return {"timestamp": now.isoformat()}


class UtilsUUIDInput(BaseModel):
    namespace: str | None = Field(default=None, max_length=64)
    name: str | None = Field(default=None, max_length=256)

    model_config = {"extra": "forbid"}


class UtilsUUIDOutput(BaseModel):
    uuid: str


class UtilsUUIDAdapter(MCPToolAdapter):
    name = "utils/uuid"
    description = "Generates UUIDv4 values or deterministic UUIDv5 identifiers."
    labels = ("utils", "id")
    aliases = ("utils.uuid",)
    capabilities = ("utility",)
    InputModel = UtilsUUIDInput
    OutputModel = UtilsUUIDOutput

    async def _invoke(self, payload_model: UtilsUUIDInput) -> Mapping[str, Any]:
        if payload_model.namespace and payload_model.name:
            namespace = uuid.uuid5(uuid.NAMESPACE_DNS, payload_model.namespace)
            generated = uuid.uuid5(namespace, payload_model.name)
        else:
            generated = uuid.uuid4()
        return {"uuid": str(generated)}


class UtilsSanitizeHTMLInput(BaseModel):
    html: str = Field(..., min_length=1)

    model_config = {"extra": "forbid"}


class UtilsSanitizeHTMLOutput(BaseModel):
    sanitized: str


class UtilsSanitizeHTMLAdapter(MCPToolAdapter):
    name = "utils/sanitize_html"
    description = "Removes script/style tags and inline event handlers from HTML snippets."
    labels = ("utils", "html")
    aliases = ("utils.sanitize_html",)
    capabilities = ("utility",)
    InputModel = UtilsSanitizeHTMLInput
    OutputModel = UtilsSanitizeHTMLOutput

    async def _invoke(self, payload_model: UtilsSanitizeHTMLInput) -> Mapping[str, Any]:
        text = payload_model.html
        text = _remove_pattern(text, "<(script|style)(.|\\n)*?</(script|style)>")
        text = _remove_pattern(text, r"on[a-zA-Z]+\s*=\s*\".*?\"")
        text = _remove_pattern(text, r"on[a-zA-Z]+\s*=\s*'.*?'")
        return {"sanitized": text}


class UtilsMergeJSONInput(BaseModel):
    base: Mapping[str, Any]
    overlay: Mapping[str, Any]
    deep: bool = Field(True)

    model_config = {"extra": "forbid"}


class UtilsMergeJSONOutput(BaseModel):
    merged: Mapping[str, Any]


class UtilsMergeJSONAdapter(MCPToolAdapter):
    name = "utils/merge_json"
    description = "Merges JSON documents using shallow or deep strategy."
    labels = ("utils", "json")
    aliases = ("utils.merge_json",)
    capabilities = ("utility",)
    InputModel = UtilsMergeJSONInput
    OutputModel = UtilsMergeJSONOutput

    async def _invoke(self, payload_model: UtilsMergeJSONInput) -> Mapping[str, Any]:
        merged = _merge_dicts(dict(payload_model.base), dict(payload_model.overlay), deep=payload_model.deep)
        return {"merged": merged}


class UtilsCompareJSONInput(BaseModel):
    left: Mapping[str, Any]
    right: Mapping[str, Any]

    model_config = {"extra": "forbid"}


class UtilsCompareJSONDiff(BaseModel):
    path: str
    left: Any
    right: Any


class UtilsCompareJSONOutput(BaseModel):
    identical: bool
    differences: list[UtilsCompareJSONDiff]


class UtilsCompareJSONAdapter(MCPToolAdapter):
    name = "utils/compare_json"
    description = "Produces a structural diff between two JSON payloads."
    labels = ("utils", "json")
    aliases = ("utils.compare_json",)
    capabilities = ("utility",)
    InputModel = UtilsCompareJSONInput
    OutputModel = UtilsCompareJSONOutput

    async def _invoke(self, payload_model: UtilsCompareJSONInput) -> Mapping[str, Any]:
        differences: list[UtilsCompareJSONDiff] = []
        _diff(payload_model.left, payload_model.right, path="", collector=differences)
        return {
            "identical": not differences,
            "differences": [item.model_dump() for item in differences],
        }


def _merge_dicts(base: dict[str, Any], overlay: dict[str, Any], *, deep: bool) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value
    return result


def _diff(left: Mapping[str, Any], right: Mapping[str, Any], *, path: str, collector: list[UtilsCompareJSONDiff]) -> None:
    left_keys = set(left.keys())
    right_keys = set(right.keys())
    for key in sorted(left_keys | right_keys):
        candidate_path = f"{path}.{key}" if path else str(key)
        if key not in right:
            collector.append(UtilsCompareJSONDiff(path=candidate_path, left=left[key], right=None))
            continue
        if key not in left:
            collector.append(UtilsCompareJSONDiff(path=candidate_path, left=None, right=right[key]))
            continue
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            _diff(left_value, right_value, path=candidate_path, collector=collector)
            continue
        if left_value != right_value:
            collector.append(UtilsCompareJSONDiff(path=candidate_path, left=left_value, right=right_value))


def _remove_pattern(text: str, pattern: str) -> str:
    import re

    return re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)


UTILS_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    UtilsHashAdapter,
    UtilsTimestampAdapter,
    UtilsUUIDAdapter,
    UtilsSanitizeHTMLAdapter,
    UtilsMergeJSONAdapter,
    UtilsCompareJSONAdapter,
)


__all__ = [
    "UtilsHashAdapter",
    "UtilsTimestampAdapter",
    "UtilsUUIDAdapter",
    "UtilsSanitizeHTMLAdapter",
    "UtilsMergeJSONAdapter",
    "UtilsCompareJSONAdapter",
    "UTILS_ADAPTER_CLASSES",
]
