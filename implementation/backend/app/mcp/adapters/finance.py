from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from datetime import UTC, datetime
from typing import Any, Literal, Sequence

import httpx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, HttpUrl, model_validator

from .base import MCPToolAdapter

YAHOO_QUOTE_ENDPOINT = "https://query1.finance.yahoo.com/v7/finance/quote"
YAHOO_SEARCH_ENDPOINT = "https://query2.finance.yahoo.com/v1/finance/search"
COINGECKO_NEWS_ENDPOINT = "https://api.coingecko.com/api/v3/news"
_FINBERT_MODEL_ID = "ProsusAI/finbert"


class YahooFinanceMetrics(BaseModel):
    symbol: str = Field(..., description="Ticker symbol requested.")
    company_name: str | None = Field(default=None, description="Long name provided by the data source.")
    currency: str | None = Field(default=None)
    price: float | None = Field(default=None, description="Last traded price.")
    change: float | None = Field(default=None, description="Absolute price change since previous close.")
    change_percent: float | None = Field(default=None, description="Percentage price change.")
    previous_close: float | None = Field(default=None)
    open: float | None = Field(default=None)
    day_low: float | None = Field(default=None)
    day_high: float | None = Field(default=None)
    volume: int | None = Field(default=None)
    fifty_two_week_low: float | None = Field(default=None)
    fifty_two_week_high: float | None = Field(default=None)
    market_cap: float | None = Field(default=None)
    updated_at: datetime | None = Field(default=None, description="Timestamp of the market data, if provided.")


class YahooFinanceRequest(BaseModel):
    symbols: list[str] | None = Field(default=None, min_length=1, max_length=20)
    query: str | None = Field(default=None, min_length=2, max_length=120)
    fields: list[str] | None = Field(default=None, description="Optional subset of fields to return for each symbol.")

    @model_validator(mode="after")
    def ensure_inputs(self) -> "YahooFinanceRequest":
        if not self.symbols and not (self.query and self.query.strip()):
            raise ValueError("Either symbols or query must be provided")
        return self

    model_config = {"extra": "forbid"}


class YahooFinanceResponse(BaseModel):
    requested: list[str]
    generated_at: datetime
    metrics: list[YahooFinanceMetrics]


class YahooFinanceSnapshotAdapter(MCPToolAdapter):
    name = "finance/yfinance"
    description = "Fetch quote snapshots for equities using Yahoo Finance's public endpoint."
    labels = ("finance", "open")
    InputModel = YahooFinanceRequest
    OutputModel = YahooFinanceResponse

    _lookup_cache: dict[str, tuple[float, list[str]]] = {}
    _lookup_cache_lock: asyncio.Lock = asyncio.Lock()
    _lookup_cache_ttl: float = 600.0
    _lookup_max_attempts: int = 4
    _lookup_base_delay: float = 0.5
    _quote_max_attempts: int = 4
    _quote_base_delay: float = 0.75

    async def _invoke(self, payload_model: YahooFinanceRequest) -> dict[str, Any]:
        symbols = payload_model.symbols
        if not symbols:
            assert payload_model.query is not None
            symbols = await self._lookup_symbols(payload_model.query)
            if not symbols:
                raise ValueError("No matching symbols found for query")

        quotes = await self._fetch_quotes(symbols)
        metrics: list[YahooFinanceMetrics] = []
        for entry in quotes:
            metrics.append(
                YahooFinanceMetrics(
                    symbol=entry.get("symbol", ""),
                    company_name=entry.get("longName") or entry.get("shortName"),
                    currency=entry.get("currency"),
                    price=_safe_float(entry.get("regularMarketPrice")),
                    change=_safe_float(entry.get("regularMarketChange")),
                    change_percent=_safe_float(entry.get("regularMarketChangePercent")),
                    previous_close=_safe_float(entry.get("regularMarketPreviousClose")),
                    open=_safe_float(entry.get("regularMarketOpen")),
                    day_low=_safe_float(entry.get("regularMarketDayLow")),
                    day_high=_safe_float(entry.get("regularMarketDayHigh")),
                    volume=_safe_int(entry.get("regularMarketVolume")),
                    fifty_two_week_low=_safe_float(entry.get("fiftyTwoWeekLow")),
                    fifty_two_week_high=_safe_float(entry.get("fiftyTwoWeekHigh")),
                    market_cap=_safe_float(entry.get("marketCap")),
                    updated_at=_safe_datetime(entry.get("regularMarketTime")),
                )
            )

        filtered_metrics = metrics
        if payload_model.fields:
            filtered_metrics = [
                YahooFinanceMetrics(**{field: metric.model_dump().get(field) for field in payload_model.fields if field in metric.model_dump()} | {"symbol": metric.symbol})
                for metric in metrics
            ]

        return {
            "requested": symbols,
            "generated_at": datetime.now(UTC),
            "metrics": [metric.model_dump() for metric in filtered_metrics],
        }

    async def _lookup_symbols(self, query: str) -> list[str]:
        normalized = query.strip().lower()
        now = time.monotonic()
        async with self._lookup_cache_lock:
            cached = self._lookup_cache.get(normalized)
            if cached and (now - cached[0]) < self._lookup_cache_ttl:
                return list(cached[1])

        delay = self._lookup_base_delay
        last_error: Exception | None = None
        for attempt in range(1, self._lookup_max_attempts + 1):
            try:
                payload = await self._perform_symbol_lookup(normalized)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code == 429:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 6.0)
                    continue
                break
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 6.0)
                continue
            else:
                quotes = payload.get("quotes", []) if isinstance(payload, dict) else []
                symbols: list[str] = []
                for entry in quotes:
                    symbol = entry.get("symbol") if isinstance(entry, dict) else None
                    if isinstance(symbol, str) and symbol.strip():
                        symbols.append(symbol.strip().upper())
                async with self._lookup_cache_lock:
                    self._lookup_cache[normalized] = (time.monotonic(), symbols)
                return symbols

        if last_error is not None:
            raise ValueError(f"Symbol lookup failed after retries: {last_error}") from last_error

        return []

    async def _perform_symbol_lookup(self, normalized_query: str) -> dict[str, Any]:
        params = {"q": normalized_query, "quotesCount": 5}
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(YAHOO_SEARCH_ENDPOINT, params=params)
            response.raise_for_status()
            return response.json()

    async def _fetch_quotes(self, symbols: Sequence[str]) -> list[dict[str, Any]]:
        params = {"symbols": ",".join(symbols)}
        delay = self._quote_base_delay
        last_error: Exception | None = None
        for attempt in range(1, self._quote_max_attempts + 1):
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(YAHOO_QUOTE_ENDPOINT, params=params)
            if response.status_code == 429:
                last_error = httpx.HTTPStatusError("rate limited", request=response.request, response=response)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 6.0)
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 6.0)
                continue
            payload = response.json()
            return payload.get("quoteResponse", {}).get("result", [])

        if last_error is not None:
            raise last_error

        return []


class PandasAnalyticsRequest(BaseModel):
    frame: list[dict[str, Any]] | None = Field(default=None, description="List of row dictionaries.")
    csv: str | None = Field(default=None, description="CSV content provided inline.")
    operations: list[Literal["summary", "correlation"]] = Field(default_factory=lambda: ["summary"])
    sample_size: int = Field(5, ge=1, le=50)

    @model_validator(mode="after")
    def validate_source(self) -> "PandasAnalyticsRequest":
        if not self.frame and not self.csv:
            raise ValueError("Either frame or csv must be provided")
        return self

    model_config = {"extra": "forbid"}


class ColumnSummary(BaseModel):
    column: str
    dtype: str
    non_null: int
    missing: int
    statistics: dict[str, float | int | None]


class PandasAnalyticsResponse(BaseModel):
    columns: list[ColumnSummary]
    sample: list[dict[str, Any]]
    correlation: dict[str, dict[str, float]] | None = None


class PandasAnalyticsAdapter(MCPToolAdapter):
    name = "finance/pandas"
    description = "Perform exploratory data analysis using pandas over ad-hoc datasets."
    labels = ("finance", "analytics")
    InputModel = PandasAnalyticsRequest
    OutputModel = PandasAnalyticsResponse

    async def _invoke(self, payload_model: PandasAnalyticsRequest) -> dict[str, Any]:
        frame = _load_dataframe(payload_model)

        describe = frame.describe(include="all", datetime_is_numeric=True).transpose()
        describe = describe.replace({np.nan: None})
        columns: list[ColumnSummary] = []
        for column in frame.columns:
            series = frame[column]
            stats_row = describe.loc[column] if column in describe.index else {}
            statistics = {
                key: (float(value) if isinstance(value, (float, int)) and value is not None else value)
                for key, value in stats_row.to_dict().items()
            }
            columns.append(
                ColumnSummary(
                    column=str(column),
                    dtype=str(series.dtype),
                    non_null=int(series.count()),
                    missing=int(series.isna().sum()),
                    statistics=statistics,
                )
            )

        sample = frame.head(payload_model.sample_size).fillna(value=None)
        correlation: dict[str, dict[str, float]] | None = None
        if "correlation" in payload_model.operations:
            corr = frame.corr(numeric_only=True).round(6)
            if not corr.empty:
                correlation = {
                    str(idx): {str(col): float(val) for col, val in row.items() if not np.isnan(val)}
                    for idx, row in corr.to_dict().items()
                }

        return {
            "columns": [column.model_dump() for column in columns],
            "sample": sample.to_dict(orient="records"),
            "correlation": correlation,
        }


class PlotPoint(BaseModel):
    x: float | datetime | str
    y: float


class PlotSeries(BaseModel):
    name: str = Field(..., description="Series label used in the legend.")
    points: list[PlotPoint] = Field(..., min_length=2)


class FinancePlotRequest(BaseModel):
    title: str = Field("Finance Plot", max_length=120)
    series: list[PlotSeries] = Field(..., min_length=1)
    y_label: str | None = Field(default=None)
    x_label: str | None = Field(default=None)

    model_config = {"extra": "forbid"}


class FinancePlotResponse(BaseModel):
    title: str
    mime_type: Literal["image/png"]
    image_base64: str
    legend: list[str]
    points: list[dict[str, Any]]


class FinancePlotAdapter(MCPToolAdapter):
    name = "finance/plot"
    description = "Generate quick visualization plots for financial series."
    labels = ("finance", "visualization")
    InputModel = FinancePlotRequest
    OutputModel = FinancePlotResponse

    async def _invoke(self, payload_model: FinancePlotRequest) -> dict[str, Any]:
        buffer = io.BytesIO()
        plt.switch_backend("Agg")
        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
        try:
            flattened_points: list[dict[str, Any]] = []
            for series in payload_model.series:
                xs = [_coerce_x(point.x) for point in series.points]
                ys = [point.y for point in series.points]
                ax.plot(xs, ys, label=series.name)
                flattened_points.append(
                    {
                        "series": series.name,
                        "points": [
                            {"x": _serialize_x(x), "y": y}
                            for x, y in zip(xs, ys)
                        ],
                    }
                )

            ax.set_title(payload_model.title)
            if payload_model.y_label:
                ax.set_ylabel(payload_model.y_label)
            if payload_model.x_label:
                ax.set_xlabel(payload_model.x_label)
            if len(payload_model.series) > 1:
                ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.autofmt_xdate()
            fig.canvas.draw()
            fig.savefig(buffer, format="png")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        finally:
            plt.close(fig)

        return {
            "title": payload_model.title,
            "mime_type": "image/png",
            "image_base64": encoded,
            "legend": [series.name for series in payload_model.series],
            "points": flattened_points,
        }


class CoinGeckoNewsRequest(BaseModel):
    categories: list[str] | None = Field(default=None, description="Optional category filters from CoinGecko news API.")
    limit: int = Field(5, ge=1, le=20)

    model_config = {"extra": "forbid"}


class CoinGeckoNewsArticle(BaseModel):
    title: str
    description: str | None = None
    url: HttpUrl
    source: str | None = None
    categories: list[str] = Field(default_factory=list)
    published_at: datetime


class CoinGeckoNewsResponse(BaseModel):
    articles: list[CoinGeckoNewsArticle]
    generated_at: datetime


class CoinGeckoNewsAdapter(MCPToolAdapter):
    name = "finance/coingecko_news"
    description = "Fetch latest cryptocurrency market news using the CoinGecko API."
    labels = ("finance", "news")
    InputModel = CoinGeckoNewsRequest
    OutputModel = CoinGeckoNewsResponse

    async def _invoke(self, payload_model: CoinGeckoNewsRequest) -> dict[str, Any]:
        headers = {}
        api_key = os.environ.get("COINGECKO_API_KEY")
        if api_key:
            headers["x-cg-pro-api-key"] = api_key
        params = {"page": 1, "per_page": payload_model.limit}
        if payload_model.categories:
            params["categories"] = ",".join(payload_model.categories)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(COINGECKO_NEWS_ENDPOINT, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()

        raw_articles = payload.get("data", [])
        articles: list[CoinGeckoNewsArticle] = []
        for article in raw_articles:
            categories = article.get("categories") or []
            if isinstance(categories, str):
                categories = [cat.strip() for cat in categories.split(",") if cat.strip()]
            articles.append(
                CoinGeckoNewsArticle(
                    title=article.get("title", "Untitled"),
                    description=article.get("description") or article.get("text"),
                    url=article.get("url"),
                    source=article.get("news_site") or article.get("source"),
                    categories=list(categories),
                    published_at=_safe_datetime(article.get("published_at") or article.get("updated_at")),
                )
            )

        return {
            "articles": [article.model_dump() for article in articles],
            "generated_at": datetime.now(UTC),
        }


class CSVAnalyzerRequest(BaseModel):
    content: str | None = Field(default=None, description="Raw CSV content.")
    base64_content: str | None = Field(default=None, description="Base64-encoded CSV content.")
    delimiter: str = Field(",", min_length=1, max_length=1)
    sample_size: int = Field(5, ge=1, le=50)

    @model_validator(mode="after")
    def validate_content(self) -> "CSVAnalyzerRequest":
        if not self.content and not self.base64_content:
            raise ValueError("Either content or base64_content must be provided")
        return self

    model_config = {"extra": "forbid"}


class CSVAnalyzerResponse(BaseModel):
    columns: list[ColumnSummary]
    sample: list[dict[str, Any]]


class CSVAnalyzerAdapter(MCPToolAdapter):
    name = "finance/csv"
    description = "Validate and summarize CSV datasets."
    labels = ("finance", "ingestion")
    InputModel = CSVAnalyzerRequest
    OutputModel = CSVAnalyzerResponse

    async def _invoke(self, payload_model: CSVAnalyzerRequest) -> dict[str, Any]:
        if payload_model.content:
            buffer = io.StringIO(payload_model.content)
        else:
            decoded = base64.b64decode(payload_model.base64_content or "", validate=True)
            buffer = io.StringIO(decoded.decode("utf-8"))
        frame = pd.read_csv(buffer, delimiter=payload_model.delimiter)
        frame = frame.replace({np.nan: None})

        columns = [
            ColumnSummary(
                column=str(col),
                dtype=str(frame[col].dtype),
                non_null=int(frame[col].count()),
                missing=int(frame[col].isna().sum()),
                statistics={},
            ).model_dump()
            for col in frame.columns
        ]
        sample = frame.head(payload_model.sample_size).to_dict(orient="records")
        return {
            "columns": columns,
            "sample": sample,
        }


class FinbertInput(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=20)
    top_k: int = Field(1, ge=1, le=3)

    model_config = {"extra": "forbid"}


class FinbertScore(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: float


class FinbertInference(BaseModel):
    text: str
    scores: list[FinbertScore]


class FinbertResponse(BaseModel):
    model: str
    generated_at: datetime
    inferences: list[FinbertInference]


class FinbertSentimentAdapter(MCPToolAdapter):
    name = "finance/finbert"
    description = "Run FinBERT sentiment analysis over financial text snippets."
    labels = ("finance", "nlp")
    InputModel = FinbertInput
    OutputModel = FinbertResponse

    _pipeline_lock: asyncio.Lock = asyncio.Lock()
    _pipeline: Any | None = None

    async def _invoke(self, payload_model: FinbertInput) -> dict[str, Any]:
        scorer = await self._get_pipeline()
        inferences: list[FinbertInference] = []
        for text in payload_model.texts:
            scores = await _run_in_thread(scorer, text, payload_model.top_k)
            inferences.append(
                FinbertInference(
                    text=text,
                    scores=[FinbertScore(label=item["label"].lower(), score=float(item["score"])) for item in scores],
                )
            )
        return {
            "model": _FINBERT_MODEL_ID,
            "generated_at": datetime.now(UTC),
            "inferences": [item.model_dump() for item in inferences],
        }

    async def _get_pipeline(self):
        async with self._pipeline_lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

                tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL_ID)
                model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL_ID)
                pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
                self._pipeline = pipeline
            except Exception:
                self._pipeline = _fallback_finbert_pipeline
            return self._pipeline


async def _run_in_thread(pipeline, text: str, top_k: int):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: pipeline(text, top_k=top_k))


def _fallback_finbert_pipeline(text: str, top_k: int = 1):
    tokens = text.lower().split()
    positive = sum(token in {"gain", "growth", "profit", "surge", "beat"} for token in tokens)
    negative = sum(token in {"loss", "drop", "decline", "miss", "debt"} for token in tokens)
    neutral = max(len(tokens) - positive - negative, 0)
    total = max(positive + negative + neutral, 1)
    scores = [
        {"label": "positive", "score": positive / total},
        {"label": "negative", "score": negative / total},
        {"label": "neutral", "score": neutral / total},
    ]
    scores.sort(key=lambda item: item["score"], reverse=True)
    return scores[:top_k]


def _load_dataframe(payload_model: PandasAnalyticsRequest) -> pd.DataFrame:
    if payload_model.frame is not None:
        return pd.DataFrame(payload_model.frame)
    assert payload_model.csv is not None
    buffer = io.StringIO(payload_model.csv)
    return pd.read_csv(buffer)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _coerce_x(value: float | datetime | str):
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value


def _serialize_x(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


FINANCE_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    YahooFinanceSnapshotAdapter,
    PandasAnalyticsAdapter,
    FinancePlotAdapter,
    CoinGeckoNewsAdapter,
    CSVAnalyzerAdapter,
    FinbertSentimentAdapter,
)


__all__ = [
    "YahooFinanceSnapshotAdapter",
    "PandasAnalyticsAdapter",
    "FinancePlotAdapter",
    "CoinGeckoNewsAdapter",
    "CSVAnalyzerAdapter",
    "FinbertSentimentAdapter",
    "FINANCE_ADAPTER_CLASSES",
]
