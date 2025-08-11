"""Financial tools wrapping Stooq and ExchangeRate.host."""
from __future__ import annotations

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..services import stooq, exchangerate_host


class PriceSeriesInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g., AAPL")


async def _price_series(symbol: str) -> str:
    rows = await stooq.daily_ohlc(symbol)
    if not rows:
        return "No data"
    last = rows[-1]
    return f"{symbol.upper()} last close {last.get('close')} on {last.get('date')}"


class FxConvertInput(BaseModel):
    amount: float = Field(..., gt=0)
    from_code: str = Field(..., min_length=3, max_length=3)
    to_code: str = Field(..., min_length=3, max_length=3)


async def _fx_convert(amount: float, from_code: str, to_code: str) -> str:
    data = await exchangerate_host.convert(amount, from_code, to_code)
    return f"{amount} {from_code.upper()} = {data.get('result')} {to_code.upper()}"


price_series_tool = StructuredTool.from_function(
    coroutine=_price_series,
    name="stooq_price_series",
    description="Fetch recent daily close for a ticker from Stooq.",
    args_schema=PriceSeriesInput,
)

fx_convert_tool = StructuredTool.from_function(
    coroutine=_fx_convert,
    name="fx_convert",
    description="Convert currency using ExchangeRate.host.",
    args_schema=FxConvertInput,
)
