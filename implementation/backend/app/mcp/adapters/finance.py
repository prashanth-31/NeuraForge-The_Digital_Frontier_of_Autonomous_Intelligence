from __future__ import annotations

import asyncio
import copy
import base64
import io
import logging
import math
import os
import time
import csv
from datetime import UTC, datetime
from typing import Any, Literal, Mapping, Sequence

import httpx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, HttpUrl, model_validator

from app.core import metrics

from .base import MCPToolAdapter

YAHOO_QUOTE_ENDPOINT = "https://query1.finance.yahoo.com/v7/finance/quote"
YAHOO_CRUMB_ENDPOINT = "https://query1.finance.yahoo.com/v1/test/getcrumb"
YAHOO_DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0"
YAHOO_SEARCH_ENDPOINT = "https://query2.finance.yahoo.com/v1/finance/search"
COINGECKO_NEWS_ENDPOINT = "https://api.coingecko.com/api/v3/news"
_FINBERT_MODEL_ID = "ProsusAI/finbert"

# Minimal keyword-to-symbol mapping to avoid repeated remote lookups for common equities.
KEYWORD_SYMBOL_OVERRIDES: dict[str, list[str]] = {
    "amazon": ["AMZN"],
    "amzn": ["AMZN"],
    "apple": ["AAPL"],
    "aapl": ["AAPL"],
    "microsoft": ["MSFT"],
    "msft": ["MSFT"],
    "google": ["GOOGL", "GOOG"],
    "alphabet": ["GOOGL", "GOOG"],
    "meta": ["META"],
    "facebook": ["META"],
    "netflix": ["NFLX"],
    "tesla": ["TSLA"],
    "tsla": ["TSLA"],
    "nvidia": ["NVDA"],
    "nvda": ["NVDA"],
    "samsung": ["005930.KS", "SSNLF"],
    "samsung electronics": ["005930.KS"],
    "ssnlf": ["SSNLF"],
    "005930": ["005930.KS"],
}


logger = logging.getLogger(__name__)


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
    _quote_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
    _quote_cache_lock: asyncio.Lock = asyncio.Lock()
    _quote_cache_ttl: float = 900.0
    _quote_session_lock: asyncio.Lock = asyncio.Lock()
    _quote_session_cookies: httpx.Cookies | None = None
    _quote_session_crumb: str | None = None
    _quote_session_refreshed_at: float = 0.0
    _quote_session_ttl: float = 1800.0
    _crumb_max_attempts: int = 3
    _crumb_base_delay: float = 0.75

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

        enriched = []
        for metric in filtered_metrics:
            payload = metric.model_dump()
            fundamentals = _extract_fundamentals_payload(quotes, metric.symbol)
            if fundamentals is not None:
                payload.setdefault("fundamentals", fundamentals)
            enriched.append(payload)

        return {
            "requested": symbols,
            "generated_at": datetime.now(UTC),
            "metrics": enriched,
        }

    async def _lookup_symbols(self, query: str) -> list[str]:
        normalized = query.strip().lower()
        if not normalized:
            return []
        keyword_matches = _keyword_symbol_lookup(normalized)
        if keyword_matches:
            return keyword_matches
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
            keyword_matches = _keyword_symbol_lookup(normalized)
            if keyword_matches:
                return keyword_matches
            raise ValueError(f"Symbol lookup failed after retries: {last_error}") from last_error

        return _keyword_symbol_lookup(normalized)

    async def _perform_symbol_lookup(self, normalized_query: str) -> dict[str, Any]:
        params = {"q": normalized_query, "quotesCount": 5}
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(YAHOO_SEARCH_ENDPOINT, params=params)
            response.raise_for_status()
            return response.json()

    async def _fetch_quotes(self, symbols: Sequence[str]) -> list[dict[str, Any]]:
        symbols_tuple = tuple(sorted(symbols))
        cache_key = ",".join(symbols_tuple)
        cached = await self._get_cached_quotes(cache_key)
        if cached is not None:
            needs_fundamentals = any("fundamentals" not in entry for entry in cached)
            if needs_fundamentals:
                cached = await self._attach_fundamentals([dict(entry) for entry in cached], symbols_tuple)
                await self._set_quote_cache(cache_key, cached)
            return [dict(entry) for entry in cached]
        params = {"symbols": ",".join(symbols_tuple), "lang": "en-US", "region": "US"}
        delay = self._quote_base_delay
        last_error: Exception | None = None
        refresh_session = False
        for attempt in range(1, self._quote_max_attempts + 1):
            crumb, cookies = await self._ensure_quote_session(force_refresh=refresh_session)
            headers = {
                "User-Agent": YAHOO_DEFAULT_USER_AGENT,
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            }
            request_params = dict(params)
            if crumb:
                request_params["crumb"] = crumb
            async with httpx.AsyncClient(timeout=8.0, headers=headers, cookies=cookies) as client:
                response = await client.get(YAHOO_QUOTE_ENDPOINT, params=request_params)
            if response.status_code in (401, 403):
                refresh_session = True
                last_error = httpx.HTTPStatusError(
                    "unauthorized",
                    request=response.request,
                    response=response,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 6.0)
                continue
            refresh_session = False
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
            quotes = [dict(item) for item in payload.get("quoteResponse", {}).get("result", [])]
            quotes = await self._attach_fundamentals(quotes, symbols_tuple)
            await self._set_quote_cache(cache_key, quotes)
            return quotes

        if last_error is not None:
            fallback_quotes = await self._fetch_quotes_via_yfinance(symbols_tuple)
            if fallback_quotes:
                logger.warning(
                    "finance_yfinance_http_fallback",
                    extra={"symbols": symbols_tuple, "reason": str(last_error)},
                )
                fallback_quotes = await self._attach_fundamentals([dict(item) for item in fallback_quotes], symbols_tuple)
                await self._set_quote_cache(cache_key, fallback_quotes)
                metrics.increment_finance_quote_fallback(
                    provider="yfinance",
                    reason=_classify_fallback_reason(last_error),
                )
                return fallback_quotes

            stooq_quotes = await self._fetch_quotes_via_stooq(symbols_tuple)
            if stooq_quotes:
                logger.warning(
                    "finance_stooq_http_fallback",
                    extra={"symbols": symbols_tuple, "reason": str(last_error)},
                )
                await self._set_quote_cache(cache_key, stooq_quotes)
                metrics.increment_finance_quote_fallback(
                    provider="stooq",
                    reason=_classify_fallback_reason(last_error),
                )
                return [dict(item) for item in stooq_quotes]

            cached = await self._get_cached_quotes(cache_key)
            if cached is not None:
                metrics.increment_finance_quote_fallback(
                    provider="cache",
                    reason=_classify_fallback_reason(last_error),
                )
                return [dict(entry) for entry in cached]
            logger.error(
                "finance_quote_fetch_failed",
                extra={"symbols": symbols_tuple, "reason": str(last_error)},
            )
            return [{"symbol": symbol} for symbol in symbols_tuple]

        cached = await self._get_cached_quotes(cache_key)
        if cached is not None:
            return [dict(entry) for entry in cached]

        return [{"symbol": symbol} for symbol in symbols_tuple]

    async def _ensure_quote_session(self, *, force_refresh: bool = False) -> tuple[str | None, httpx.Cookies | None]:
        async with self._quote_session_lock:
            if force_refresh or self._quote_session_needs_refresh():
                return await self._refresh_quote_session_locked()
            cookies = copy.copy(self._quote_session_cookies) if self._quote_session_cookies is not None else None
            return self._quote_session_crumb, cookies

    def _quote_session_needs_refresh(self) -> bool:
        if self._quote_session_refreshed_at == 0.0:
            return True
        return (time.monotonic() - self._quote_session_refreshed_at) > self._quote_session_ttl

    async def _refresh_quote_session_locked(self) -> tuple[str | None, httpx.Cookies | None]:
        crumb: str | None = None
        cookies: httpx.Cookies | None = None
        previous_crumb = self._quote_session_crumb
        previous_cookies = self._quote_session_cookies
        try:
            crumb, cookies = await self._fetch_yahoo_crumb()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("finance_yahoo_crumb_fetch_failed", extra={"error": str(exc)})
            crumb = None
            cookies = None

        if crumb is None and previous_crumb is not None:
            crumb = previous_crumb
            cookies = previous_cookies

        if crumb is not None:
            self._quote_session_crumb = crumb
            self._quote_session_cookies = cookies
            self._quote_session_refreshed_at = time.monotonic()
        else:
            self._quote_session_refreshed_at = 0.0

        return crumb, copy.copy(cookies) if cookies is not None else None

    async def _fetch_yahoo_crumb(self) -> tuple[str | None, httpx.Cookies | None]:
        headers = {
            "User-Agent": YAHOO_DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
        delay = self._crumb_base_delay
        last_error: Exception | None = None
        for attempt in range(1, self._crumb_max_attempts + 1):
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                response = await client.get(YAHOO_CRUMB_ENDPOINT, params={"lang": "en-US", "region": "US"})
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
                crumb_value = response.text.strip() or None
                cookies = copy.copy(client.cookies) if client.cookies else None
                return crumb_value, cookies
        if last_error is not None:
            raise last_error
        return None, None

    async def _fetch_quotes_via_stooq(self, symbols: Sequence[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": YAHOO_DEFAULT_USER_AGENT}) as client:
            for symbol in symbols:
                stooq_symbol = symbol.lower()
                if "." not in stooq_symbol:
                    stooq_symbol = f"{stooq_symbol}.us"
                params = {"s": stooq_symbol, "i": "d"}
                try:
                    response = await client.get("https://stooq.com/q/d/l/", params=params)
                    response.raise_for_status()
                except httpx.HTTPError:
                    continue
                content = response.text.strip()
                if not content:
                    continue
                reader = csv.DictReader(content.splitlines())
                latest_row: dict[str, str] | None = None
                for row in reader:
                    latest_row = row
                if not latest_row:
                    continue
                try:
                    close_price = float(latest_row.get("Close", ""))
                except ValueError:
                    close_price = None
                try:
                    open_price = float(latest_row.get("Open", ""))
                except ValueError:
                    open_price = None
                try:
                    day_high = float(latest_row.get("High", ""))
                except ValueError:
                    day_high = None
                try:
                    day_low = float(latest_row.get("Low", ""))
                except ValueError:
                    day_low = None
                try:
                    volume = int(float(latest_row.get("Volume", "")))
                except ValueError:
                    volume = None
                payload = {
                    "symbol": symbol,
                    "regularMarketPrice": close_price,
                    "regularMarketOpen": open_price,
                    "regularMarketDayHigh": day_high,
                    "regularMarketDayLow": day_low,
                    "regularMarketVolume": volume,
                }
                results.append(payload)
        return results

    async def _fetch_quotes_via_yfinance(self, symbols: Sequence[str]) -> list[dict[str, Any]]:
        try:
            import yfinance as yf  # type: ignore[import-not-found]
        except ImportError:
            return []

        async def _gather() -> list[dict[str, Any]]:
            def _fetch_sync() -> list[dict[str, Any]]:
                tickers = yf.Tickers(" ".join(symbols))
                results: list[dict[str, Any]] = []
                for symbol in symbols:
                    ticker = tickers.tickers.get(symbol)
                    if ticker is None:
                        continue
                    try:
                        fast_info = dict(getattr(ticker, "fast_info", {}) or {})
                    except Exception:  # pragma: no cover - defensive guard
                        fast_info = {}
                    price = fast_info.get("last_price") or fast_info.get("regular_market_price")
                    prev_close = fast_info.get("previous_close") or fast_info.get("regular_market_previous_close")
                    open_price = fast_info.get("open") or fast_info.get("regular_market_open")
                    day_low = fast_info.get("day_low") or fast_info.get("regular_market_day_low")
                    day_high = fast_info.get("day_high") or fast_info.get("regular_market_day_high")
                    volume = fast_info.get("regular_market_volume") or fast_info.get("volume")
                    year_low = fast_info.get("year_low") or fast_info.get("fifty_two_week_low")
                    year_high = fast_info.get("year_high") or fast_info.get("fifty_two_week_high")
                    market_cap = fast_info.get("market_cap")
                    updated = fast_info.get("regular_market_time") or fast_info.get("last_market_time")

                    history_timestamp: datetime | None = None
                    need_history = any(
                        candidate is None for candidate in (price, prev_close, open_price, day_low, day_high, volume)
                    )
                    if need_history:
                        try:
                            history = ticker.history(period="5d", interval="1d")
                        except Exception:  # pragma: no cover - defensive guard
                            history = None
                        if history is not None and not history.empty:
                            last_row = history.iloc[-1]
                            history_timestamp = getattr(last_row, "name", None)
                            price = price if price is not None else last_row.get("Close")
                            open_price = open_price if open_price is not None else last_row.get("Open")
                            day_low = day_low if day_low is not None else last_row.get("Low")
                            day_high = day_high if day_high is not None else last_row.get("High")
                            volume = volume if volume is not None else last_row.get("Volume")
                            if prev_close is None:
                                if len(history) > 1:
                                    prev_close = history.iloc[-2].get("Close")
                                else:
                                    prev_close = last_row.get("Close")
                    if updated is None and history_timestamp is not None:
                        try:
                            updated = history_timestamp.to_pydatetime().timestamp()
                        except AttributeError:
                            try:
                                updated = datetime.fromisoformat(str(history_timestamp)).timestamp()
                            except Exception:  # pragma: no cover - defensive
                                updated = None

                    price = _safe_float(price)
                    prev_close = _safe_float(prev_close)
                    open_price = _safe_float(open_price)
                    day_low = _safe_float(day_low)
                    day_high = _safe_float(day_high)
                    volume = _safe_int(volume)
                    year_low = _safe_float(year_low)
                    year_high = _safe_float(year_high)

                    change = None
                    change_percent = None
                    if price is not None and prev_close not in (None, 0):
                        change = float(price) - float(prev_close)
                        if prev_close:
                            change_percent = (change / float(prev_close)) * 100.0

                    long_name = None
                    try:
                        info = ticker.get_info()
                    except Exception:  # pragma: no cover - defensive guard
                        info = {}
                    else:
                        long_name = info.get("longName") or info.get("shortName")

                    results.append(
                        {
                            "symbol": symbol,
                            "longName": long_name,
                            "currency": fast_info.get("currency") or fast_info.get("currency_code"),
                            "regularMarketPrice": price,
                            "regularMarketChange": change,
                            "regularMarketChangePercent": change_percent,
                            "regularMarketPreviousClose": prev_close,
                            "regularMarketOpen": open_price,
                            "regularMarketDayLow": day_low,
                            "regularMarketDayHigh": day_high,
                            "regularMarketVolume": volume,
                            "fiftyTwoWeekLow": year_low,
                            "fiftyTwoWeekHigh": year_high,
                            "marketCap": market_cap,
                            "regularMarketTime": updated,
                        }
                    )
                return results

            return await asyncio.to_thread(_fetch_sync)

        try:
            quotes = await _gather()
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("finance_yfinance_fallback_failed", extra={"symbols": symbols})
            return []

        now = datetime.now(UTC)
        normalized: list[dict[str, Any]] = []
        for entry in quotes:
            entry = dict(entry)
            if entry.get("regularMarketTime") is None:
                entry["regularMarketTime"] = now.timestamp()
            normalized.append(entry)
        return normalized

    async def _attach_fundamentals(
        self,
        quotes: list[dict[str, Any]],
        symbols: Sequence[str],
    ) -> list[dict[str, Any]]:
        fundamentals = await self._fetch_fundamentals(symbols)
        if not fundamentals:
            return quotes
        for entry in quotes:
            symbol = entry.get("symbol")
            if isinstance(symbol, str):
                payload = fundamentals.get(symbol)
                if payload:
                    entry["fundamentals"] = payload
        return quotes

    async def _fetch_fundamentals(self, symbols: Sequence[str]) -> dict[str, Any]:
        try:
            import yfinance as yf  # type: ignore[import-not-found]
        except ImportError:
            return {}

        symbols_tuple = tuple(symbols)
        if not symbols_tuple:
            return {}

        def _collect() -> dict[str, Any]:
            tickers = yf.Tickers(" ".join(symbols_tuple))
            collected: dict[str, Any] = {}
            for symbol in symbols_tuple:
                ticker = tickers.tickers.get(symbol)
                if ticker is None:
                    continue
                fundamentals = _build_fundamentals_snapshot(ticker)
                if fundamentals:
                    collected[symbol] = fundamentals
            return collected

        try:
            return await asyncio.to_thread(_collect)
        except Exception:  # pragma: no cover - fallback safety
            logger.exception("finance_fundamentals_fetch_failed", extra={"symbols": symbols_tuple})
            return {}

    async def _get_cached_quotes(self, key: str) -> list[dict[str, Any]] | None:
        async with self._quote_cache_lock:
            entry = self._quote_cache.get(key)
            if not entry:
                return None
            timestamp, payload = entry
            if time.monotonic() - timestamp > self._quote_cache_ttl:
                del self._quote_cache[key]
                return None
            return [dict(item) for item in payload]

    async def _set_quote_cache(self, key: str, payload: list[dict[str, Any]]) -> None:
        async with self._quote_cache_lock:
            self._quote_cache[key] = (time.monotonic(), [dict(item) for item in payload])


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


def _keyword_symbol_lookup(normalized_query: str) -> list[str]:
    if not normalized_query:
        return []
    matches: list[str] = []
    tokens = normalized_query.replace("-", " ").replace("/", " ").split()
    for token in tokens:
        symbols = KEYWORD_SYMBOL_OVERRIDES.get(token)
        if not symbols:
            continue
        for symbol in symbols:
            if symbol not in matches:
                matches.append(symbol)
    if matches:
        return matches
    for keyword, symbols in KEYWORD_SYMBOL_OVERRIDES.items():
        if keyword in normalized_query:
            for symbol in symbols:
                if symbol not in matches:
                    matches.append(symbol)
    return matches


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
        candidate = float(value)
        if math.isnan(candidate) or math.isinf(candidate):
            return None
        return candidate
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


def _extract_fundamentals_payload(quotes: Sequence[Mapping[str, Any]], symbol: str) -> dict[str, Any] | None:
    if not symbol:
        return None
    target = symbol.upper()
    for entry in quotes:
        entry_symbol = entry.get("symbol")
        if isinstance(entry_symbol, str) and entry_symbol.upper() == target:
            fundamentals = entry.get("fundamentals")
            if isinstance(fundamentals, Mapping) and fundamentals:
                return copy.deepcopy(dict(fundamentals))
    return None


def _build_fundamentals_snapshot(ticker: Any) -> dict[str, Any] | None:
    annual_entries = _extract_financial_entries(_safe_dataframe(getattr(ticker, "financials", None)), limit=4)
    quarterly_entries = _extract_financial_entries(
        _safe_dataframe(getattr(ticker, "quarterly_financials", None)),
        limit=6,
    )

    trailing_revenue = _compute_trailing_total(quarterly_entries, "total_revenue")
    trailing_net_income = _compute_trailing_total(quarterly_entries, "net_income")

    fundamentals: dict[str, Any] = {}
    if trailing_revenue is not None or trailing_net_income is not None:
        trailing_payload: dict[str, Any] = {}
        if trailing_revenue is not None:
            trailing_payload["revenue"] = trailing_revenue
        if trailing_net_income is not None:
            trailing_payload["net_income"] = trailing_net_income
        fundamentals["trailing"] = trailing_payload

    if annual_entries:
        fundamentals["annual"] = annual_entries
    if quarterly_entries:
        fundamentals["quarterly"] = quarterly_entries

    info: Mapping[str, Any] | None = None
    try:
        info_candidate = ticker.get_info()
    except Exception:  # pragma: no cover - defensive guard
        info_candidate = None
    if isinstance(info_candidate, Mapping):
        info = info_candidate
        currency = info.get("financialCurrency") or info.get("currency")
        if isinstance(currency, str) and currency:
            fundamentals["currency"] = currency
        guidance = _extract_guidance_fields(info)
        if guidance:
            fundamentals["guidance"] = guidance

    return fundamentals or None


def _extract_financial_entries(frame: pd.DataFrame | None, *, limit: int) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    entries: list[dict[str, Any]] = []
    columns = list(frame.columns)[:limit]
    for column in columns:
        record: dict[str, Any] = {"period": _format_period_label(column)}
        for source_key, alias in ("Total Revenue", "total_revenue"), ("Net Income", "net_income"):
            value = None
            if source_key in frame.index:
                try:
                    value = frame.loc[source_key, column]
                except Exception:  # pragma: no cover - guard against missing data
                    value = None
            record[alias] = _safe_float(value)
        entries.append(record)
    return entries


def _compute_trailing_total(entries: Sequence[Mapping[str, Any]], key: str, count: int = 4) -> float | None:
    values: list[float] = []
    for entry in entries:
        candidate = entry.get(key)
        if candidate is None:
            continue
        number = _safe_float(candidate)
        if number is None:
            continue
        values.append(number)
        if len(values) >= count:
            break
    if not values:
        return None
    return float(sum(values))


def _safe_dataframe(candidate: Any) -> pd.DataFrame | None:
    if isinstance(candidate, pd.DataFrame):
        return candidate
    return None


def _format_period_label(period: Any) -> str:
    if isinstance(period, pd.Timestamp):
        return period.to_pydatetime().date().isoformat()
    if isinstance(period, datetime):
        return period.date().isoformat()
    return str(period)


def _extract_guidance_fields(info: Mapping[str, Any]) -> dict[str, Any]:
    guidance_map = {
        "forward_eps": "forwardEps",
        "forward_pe": "forwardPE",
        "target_mean_price": "targetMeanPrice",
        "target_high_price": "targetHighPrice",
        "target_low_price": "targetLowPrice",
        "target_median_price": "targetMedianPrice",
        "revenue_growth": "revenueGrowth",
        "earnings_growth": "earningsGrowth",
    }
    guidance: dict[str, Any] = {}
    for key, source in guidance_map.items():
        value = info.get(source)
        numeric = _safe_float(value)
        if numeric is not None:
            guidance[key] = numeric
    return guidance


def _classify_fallback_reason(error: Exception | None) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code if error.response is not None else None
        if status == 429:
            return "rate_limited"
        if status == 401:
            return "unauthorized"
        if status == 403:
            return "forbidden"
        if status is not None:
            return f"http_{status}"
        return "http_error"
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    if isinstance(error, httpx.HTTPError):
        return "http_error"
    if isinstance(error, ValueError):
        return "value_error"
    return "error"
__all__ = [
    "YahooFinanceSnapshotAdapter",
    "PandasAnalyticsAdapter",
    "FinancePlotAdapter",
    "CoinGeckoNewsAdapter",
    "CSVAnalyzerAdapter",
    "FinbertSentimentAdapter",
    "FINANCE_ADAPTER_CLASSES",
]
