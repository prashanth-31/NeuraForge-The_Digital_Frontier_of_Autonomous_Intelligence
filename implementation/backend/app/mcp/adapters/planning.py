from __future__ import annotations

from typing import Mapping

from pydantic import BaseModel, Field

from .base import MCPToolAdapter


class PlannerSuggestInput(BaseModel):
    task: str = Field(..., min_length=1, description="Natural language problem statement.")

    model_config = {"extra": "forbid"}


class PlannerSuggestOutput(BaseModel):
    suggestion: str
    rationale: str | None = None


class PlannerSuggestAdapter(MCPToolAdapter):
    name = "planner/suggest"
    description = "Suggests a high-level plan for a given task for agent chaining."
    labels = ("planner",)
    aliases = ("planner.suggest",)
    capabilities = ("planning",)
    InputModel = PlannerSuggestInput
    OutputModel = PlannerSuggestOutput

    async def _invoke(self, payload_model: PlannerSuggestInput) -> Mapping[str, str | None]:
        return {
            "suggestion": (
                "Break the task into small, verifiable steps, check available tools, "
                "and execute sequentially while monitoring for policy violations."
            ),
            "rationale": "Template guidance for orchestrator scaffolding.",
        }


class PlannerDecomposeInput(BaseModel):
    goal: str = Field(..., min_length=1)
    max_steps: int = Field(5, ge=1, le=10)

    model_config = {"extra": "forbid"}


class PlannerDecomposeOutput(BaseModel):
    steps: list[str]


class PlannerDecomposeAdapter(MCPToolAdapter):
    name = "planner/decompose"
    description = "Returns a list of generic plan steps for orchestration tests."
    labels = ("planner",)
    aliases = ("planner.decompose",)
    capabilities = ("planning",)
    InputModel = PlannerDecomposeInput
    OutputModel = PlannerDecomposeOutput

    async def _invoke(self, payload_model: PlannerDecomposeInput) -> Mapping[str, list[str]]:
        steps = [f"Step {idx + 1}: placeholder" for idx in range(payload_model.max_steps)]
        return {"steps": steps}


PLANNING_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    PlannerSuggestAdapter,
    PlannerDecomposeAdapter,
)


__all__ = [
    "PlannerSuggestAdapter",
    "PlannerDecomposeAdapter",
    "PLANNING_ADAPTER_CLASSES",
]
