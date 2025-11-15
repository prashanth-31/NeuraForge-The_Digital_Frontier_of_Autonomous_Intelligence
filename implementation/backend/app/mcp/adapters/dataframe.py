from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, model_validator

from app.tools.exceptions import ToolInvocationError

from .base import MCPToolAdapter


@dataclass(slots=True)
class _FrameEnvelope:
    frame: pd.DataFrame
    truncated: bool


def _load_frame(records: Sequence[Mapping[str, Any]] | None, csv_content: str | None) -> _FrameEnvelope:
    truncated = False
    if records is not None:
        frame = pd.DataFrame(list(records))
    else:
        assert csv_content is not None
        buffer = io.StringIO(csv_content)
        frame = pd.read_csv(buffer)
    if frame.shape[0] > 10_000:
        frame = frame.iloc[:10_000].copy()
        truncated = True
    return _FrameEnvelope(frame=frame, truncated=truncated)


class DataFrameLoadCSVInput(BaseModel):
    csv: str = Field(..., description="CSV payload as text.")
    sample_limit: int = Field(20, ge=1, le=200)
    include_preview: bool = Field(True)
    encoding: str = Field("utf-8")

    model_config = {"extra": "forbid"}


class DataFrameLoadCSVOutput(BaseModel):
    columns: list[str]
    row_count: int
    preview: list[dict[str, Any]]
    truncated: bool


class DataFrameLoadCSVAdapter(MCPToolAdapter):
    name = "dataframe/load_csv"
    description = "Parses CSV content into a structured table preview."
    labels = ("dataframe", "io")
    aliases = ("dataframe.load_csv",)
    capabilities = ("data", "csv")
    InputModel = DataFrameLoadCSVInput
    OutputModel = DataFrameLoadCSVOutput

    async def _invoke(self, payload_model: DataFrameLoadCSVInput) -> Mapping[str, Any]:
        buffer = io.StringIO(payload_model.csv)
        frame = pd.read_csv(buffer, encoding=payload_model.encoding)
        truncated = False
        if frame.shape[0] > payload_model.sample_limit:
            truncated = True
        head = frame.head(payload_model.sample_limit) if payload_model.include_preview else pd.DataFrame(columns=frame.columns)
        preview = head.to_dict(orient="records")
        return {
            "columns": list(frame.columns),
            "row_count": int(frame.shape[0]),
            "preview": preview,
            "truncated": truncated,
        }


class DataFrameDescribeInput(BaseModel):
    records: list[Mapping[str, Any]] | None = Field(default=None)
    csv: str | None = Field(default=None)
    include: str | None = Field(default=None, description="Optional pandas describe include parameter.")

    @model_validator(mode="after")
    def ensure_source(self) -> "DataFrameDescribeInput":
        if self.records is None and not self.csv:
            raise ValueError("Either records or csv must be provided")
        return self

    model_config = {"extra": "forbid"}


class DataFrameDescribeOutput(BaseModel):
    summary: Mapping[str, Mapping[str, Any]]
    truncated: bool


class DataFrameDescribeAdapter(MCPToolAdapter):
    name = "dataframe/describe"
    description = "Generates descriptive statistics for tabular data."
    labels = ("dataframe", "analytics")
    aliases = ("dataframe.describe",)
    capabilities = ("data", "statistics")
    InputModel = DataFrameDescribeInput
    OutputModel = DataFrameDescribeOutput

    async def _invoke(self, payload_model: DataFrameDescribeInput) -> Mapping[str, Any]:
        envelope = _load_frame(payload_model.records, payload_model.csv)
        frame = envelope.frame
        include = payload_model.include or "all"
        try:
            described = frame.describe(include=include).transpose()
        except ValueError:
            described = frame.describe(include="all").transpose()
        summary = described.fillna(0).to_dict(orient="index")
        return {
            "summary": summary,
            "truncated": envelope.truncated,
        }


class DataFramePlotInput(BaseModel):
    records: list[Mapping[str, Any]] | None = Field(default=None)
    csv: str | None = Field(default=None)
    x: str | None = Field(default=None)
    y: str | None = Field(default=None)
    kind: str = Field("line", pattern=r"^(line|bar|scatter)$")
    title: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def ensure_source(self) -> "DataFramePlotInput":
        if self.records is None and not self.csv:
            raise ValueError("Either records or csv must be provided")
        return self

    model_config = {"extra": "forbid"}


class DataFramePlotOutput(BaseModel):
    image_base64: str
    mime_type: str
    points: int


class DataFramePlotAdapter(MCPToolAdapter):
    name = "dataframe/plot"
    description = "Renders lightweight visualisations for tabular data."
    labels = ("dataframe", "visualisation")
    aliases = ("dataframe.plot",)
    capabilities = ("data", "visualisation")
    InputModel = DataFramePlotInput
    OutputModel = DataFramePlotOutput

    async def _invoke(self, payload_model: DataFramePlotInput) -> Mapping[str, Any]:
        envelope = _load_frame(payload_model.records, payload_model.csv)
        frame = envelope.frame
        if frame.empty:
            raise ToolInvocationError("Dataframe is empty; cannot plot")

        temp = frame.copy()
        if payload_model.x and payload_model.x in temp.columns:
            temp = temp.dropna(subset=[payload_model.x])
        if payload_model.y and payload_model.y in temp.columns:
            temp = temp.dropna(subset=[payload_model.y])
        if payload_model.x and payload_model.x not in temp.columns:
            raise ToolInvocationError(f"Column '{payload_model.x}' not found")
        if payload_model.y and payload_model.y not in temp.columns:
            raise ToolInvocationError(f"Column '{payload_model.y}' not found")

        figure, axis = plt.subplots(figsize=(6, 4))
        kind = payload_model.kind
        if kind == "line":
            temp.plot(x=payload_model.x, y=payload_model.y, ax=axis, legend=False)
        elif kind == "bar":
            temp.plot.bar(x=payload_model.x, y=payload_model.y, ax=axis, legend=False)
        else:
            if payload_model.y is None:
                raise ToolInvocationError("Scatter plots require the 'y' column to be specified")
            if payload_model.x and payload_model.x not in temp.columns:
                raise ToolInvocationError(f"Column '{payload_model.x}' not found for x-axis")
            if payload_model.y not in temp.columns:
                raise ToolInvocationError(f"Column '{payload_model.y}' not found for y-axis")
            x_values = temp[payload_model.x] if payload_model.x else range(len(temp))
            axis.scatter(x_values, temp[payload_model.y])
            axis.set_xlabel(payload_model.x or "index")
            axis.set_ylabel(payload_model.y or "value")
        axis.set_title(payload_model.title or "Dataframe Plot")
        axis.grid(True, alpha=0.2)
        buffer = io.BytesIO()
        figure.tight_layout()
        figure.savefig(buffer, format="png")
        plt.close(figure)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("ascii")
        return {
            "image_base64": image_base64,
            "mime_type": "image/png",
            "points": int(temp.shape[0]),
        }


DATAFRAME_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    DataFrameLoadCSVAdapter,
    DataFrameDescribeAdapter,
    DataFramePlotAdapter,
)


__all__ = [
    "DataFrameLoadCSVAdapter",
    "DataFrameDescribeAdapter",
    "DataFramePlotAdapter",
    "DATAFRAME_ADAPTER_CLASSES",
]
