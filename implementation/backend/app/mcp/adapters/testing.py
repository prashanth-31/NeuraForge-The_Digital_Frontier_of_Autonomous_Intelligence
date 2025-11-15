from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime
from typing import Any, Mapping

from pydantic import BaseModel, Field

from app.tools.exceptions import ToolInvocationError

from .base import MCPToolAdapter


class TestEchoInput(BaseModel):
    payload: Mapping[str, Any]

    model_config = {"extra": "forbid"}


class TestEchoOutput(BaseModel):
    payload: Mapping[str, Any]
    echoed_at: datetime


class TestEchoAdapter(MCPToolAdapter):
    name = "test/echo"
    description = "Returns the provided payload for diagnostics."
    labels = ("test",)
    aliases = ("test.echo",)
    capabilities = ("test",)
    InputModel = TestEchoInput
    OutputModel = TestEchoOutput

    async def _invoke(self, payload_model: TestEchoInput) -> Mapping[str, Any]:
        return {"payload": dict(payload_model.payload), "echoed_at": datetime.now(UTC)}


class TestDelayInput(BaseModel):
    seconds: float = Field(0.1, ge=0.0, le=5.0)

    model_config = {"extra": "forbid"}


class TestDelayOutput(BaseModel):
    delayed: float


class TestDelayAdapter(MCPToolAdapter):
    name = "test/delay"
    description = "Introduces an async delay to test orchestration timeouts."
    labels = ("test",)
    aliases = ("test.delay",)
    capabilities = ("test",)
    InputModel = TestDelayInput
    OutputModel = TestDelayOutput

    async def _invoke(self, payload_model: TestDelayInput) -> Mapping[str, Any]:
        await asyncio.sleep(payload_model.seconds)
        return {"delayed": payload_model.seconds}


class TestFailInput(BaseModel):
    message: str = Field("Intentional failure for testing.")

    model_config = {"extra": "forbid"}


class TestFailAdapter(MCPToolAdapter):
    name = "test/fail"
    description = "Raises a controlled failure for resilience testing."
    labels = ("test",)
    aliases = ("test.fail",)
    capabilities = ("test",)
    InputModel = TestFailInput
    OutputModel = TestEchoOutput  # unused but required by base class

    async def _invoke(self, payload_model: TestFailInput) -> Mapping[str, Any]:  # type: ignore[override]
        raise ToolInvocationError(payload_model.message)


class TestRandomErrorInput(BaseModel):
    probability: float = Field(0.5, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class TestRandomErrorOutput(BaseModel):
    outcome: str
    seed: int


class TestRandomErrorAdapter(MCPToolAdapter):
    name = "test/random_error"
    description = "Throws randomised errors to exercise retry logic."
    labels = ("test",)
    aliases = ("test.random_error",)
    capabilities = ("test",)
    InputModel = TestRandomErrorInput
    OutputModel = TestRandomErrorOutput

    async def _invoke(self, payload_model: TestRandomErrorInput) -> Mapping[str, Any]:
        seed = random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        if rng.random() < payload_model.probability:
            raise ToolInvocationError("Randomised failure triggered")
        return {"outcome": "success", "seed": seed}


TESTING_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    TestEchoAdapter,
    TestDelayAdapter,
    TestFailAdapter,
    TestRandomErrorAdapter,
)


__all__ = [
    "TestEchoAdapter",
    "TestDelayAdapter",
    "TestFailAdapter",
    "TestRandomErrorAdapter",
    "TESTING_ADAPTER_CLASSES",
]
