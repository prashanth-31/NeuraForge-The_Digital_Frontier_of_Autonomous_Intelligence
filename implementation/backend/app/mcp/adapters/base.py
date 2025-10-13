from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Mapping

from pydantic import BaseModel


@dataclass(slots=True, frozen=True)
class ToolDescriptor:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    labels: tuple[str, ...] = ()


class MCPToolAdapter(ABC):
    """Base adapter for MCP tools using pydantic models for validation."""

    name: ClassVar[str]
    description: ClassVar[str]
    labels: ClassVar[tuple[str, ...]] = ()
    InputModel: ClassVar[type[BaseModel]]
    OutputModel: ClassVar[type[BaseModel]]

    @classmethod
    def descriptor(cls) -> ToolDescriptor:
        return ToolDescriptor(
            name=cls.name,
            description=cls.description,
            input_schema=cls.InputModel.model_json_schema(),
            output_schema=cls.OutputModel.model_json_schema(),
            labels=tuple(cls.labels) if cls.labels else (),
        )

    @classmethod
    def all_descriptors(cls) -> Iterable[ToolDescriptor]:  # pragma: no cover - convenience hook
        yield cls.descriptor()

    async def invoke(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        model = self.InputModel.model_validate(payload)
        result = await self._invoke(model)
        return self.OutputModel.model_validate(result).model_dump(mode="json")

    @abstractmethod
    async def _invoke(self, payload_model: BaseModel) -> Mapping[str, Any]:
        ...


__all__ = ["ToolDescriptor", "MCPToolAdapter"]
