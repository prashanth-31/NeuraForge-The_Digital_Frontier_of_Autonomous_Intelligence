import httpx
import pytest

from app.mcp.adapters.finance import YAHOO_QUOTE_ENDPOINT, YahooFinanceSnapshotAdapter


@pytest.mark.asyncio
async def test_yfinance_keyword_override(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = YahooFinanceSnapshotAdapter()

    async def fake_fetch(self: YahooFinanceSnapshotAdapter, symbols):
        assert symbols == ["AMZN"]
        return [
            {
                "symbol": "AMZN",
                "regularMarketPrice": 123.45,
                "regularMarketChange": 1.23,
                "regularMarketChangePercent": 1.0,
            }
        ]

    monkeypatch.setattr(YahooFinanceSnapshotAdapter, "_fetch_quotes", fake_fetch, raising=False)

    payload = adapter.InputModel(query="give me a financial report of amazon")
    result = await adapter._invoke(payload)
    assert result["requested"] == ["AMZN"]
    assert result["metrics"]
    assert result["metrics"][0]["symbol"] == "AMZN"


@pytest.mark.asyncio
async def test_yfinance_rate_limit_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = YahooFinanceSnapshotAdapter()

    class AlwaysRateLimitedClient:
        def __init__(self, *args, **kwargs):
            request = httpx.Request("GET", YAHOO_QUOTE_ENDPOINT)
            self._response = httpx.Response(status_code=429, request=request)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None):
            return self._response

    monkeypatch.setattr("app.mcp.adapters.finance.httpx.AsyncClient", AlwaysRateLimitedClient)

    payload = adapter.InputModel(symbols=["AMZN"])
    result = await adapter._invoke(payload)

    assert result["requested"] == ["AMZN"]
    assert result["metrics"]
    metric = result["metrics"][0]
    assert metric["symbol"] == "AMZN"
    assert metric["price"] is None
