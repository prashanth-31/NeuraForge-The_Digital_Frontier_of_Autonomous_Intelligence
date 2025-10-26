import base64

import pytest
from pytest_httpx import HTTPXMock

from app.mcp.adapters.research import (
    ArxivFetchAdapter,
    DocumentLoaderAdapter,
    DuckDuckGoSearchAdapter,
    QdrantRetrieverAdapter,
    SummarizerAdapter,
    WikipediaSummaryAdapter,
)


pytestmark = pytest.mark.asyncio


async def test_duckduckgo_search_parses_results(httpx_mock: HTTPXMock) -> None:
    adapter = DuckDuckGoSearchAdapter()
    payload = {
        "RelatedTopics": [
            {"Text": "NeuraForge", "FirstURL": "https://example.com", "Result": "<b>Result</b>"}
        ]
    }
    httpx_mock.add_response(json=payload)

    result = await adapter.invoke({"query": "neuraforge"})

    assert result["results"][0]["title"] == "NeuraForge"
    assert "<" not in result["results"][0]["snippet"]


async def test_arxiv_fetch_parses_entries(httpx_mock: HTTPXMock) -> None:
    adapter = ArxivFetchAdapter()
    xml_payload = """
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
            <entry>
                <id>http://arxiv.org/abs/1234.5678v1</id>
                <title> Sample Paper </title>
                <summary> Findings summary. </summary>
                <published>2024-01-01T00:00:00Z</published>
                <updated>2024-01-01T00:00:00Z</updated>
                <link href="http://arxiv.org/pdf/1234.5678v1" rel="related" title="pdf"/>
                <author><name>Doe, Jane</name></author>
                <arxiv:primary_category term="cs.AI"/>
            </entry>
        </feed>
    """
    httpx_mock.add_response(text=xml_payload)

    result = await adapter.invoke({"query": "cs"})

    assert result["results"][0]["primary_category"] == "cs.AI"


async def test_wikipedia_summary_adapter_handles_response(httpx_mock: HTTPXMock) -> None:
    adapter = WikipediaSummaryAdapter()
    payload = {
        "title": "NeuraForge",
        "extract": "NeuraForge is a platform.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Example"}},
        "timestamp": "2024-01-01T00:00:00Z",
    }
    httpx_mock.add_response(json=payload)

    result = await adapter.invoke({"title": "NeuraForge"})

    assert result["url"].startswith("https://")
    assert result["last_modified"].startswith("2024-01-01")


async def test_document_loader_accepts_base64() -> None:
    adapter = DocumentLoaderAdapter()
    content = base64.b64encode("sample document".encode("utf-8")).decode("ascii")

    result = await adapter.invoke({"content_base64": content, "chunk_size": 6})

    assert result["bytes_loaded"] == len("sample document".encode("utf-8"))
    assert len(result["chunks"]) >= 2


async def test_qdrant_retriever_invokes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = QdrantRetrieverAdapter()

    class _FakeSearchResult:
        def __init__(self) -> None:
            self.payload = {"text": "example"}
            self.score = 0.9

    class _FakeClient:
        async def search(self, *, collection_name: str, query_vector, limit: int, **kwargs):
            return [_FakeSearchResult()]

    async def _fake_embed(self, text: str) -> list[float]:  # type: ignore[override]
        return [0.1, 0.2, 0.3]

    async def _fake_get_client(cls):
        return _FakeClient()

    monkeypatch.setattr(QdrantRetrieverAdapter, "_get_client", classmethod(_fake_get_client))
    monkeypatch.setattr(QdrantRetrieverAdapter, "_embed_text", _fake_embed, raising=False)

    result = await adapter.invoke({"query": "example"})

    assert result["hits"][0]["payload"]["text"] == "example"


async def test_summarizer_respects_max_tokens() -> None:
    adapter = SummarizerAdapter()
    text = "Sentence one. Sentence two is longer." * 5

    result = await adapter.invoke({"text": text, "max_tokens": 20})

    assert result["tokens_used"] <= 20