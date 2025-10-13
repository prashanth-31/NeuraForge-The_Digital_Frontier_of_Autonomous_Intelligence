from __future__ import annotations

import asyncio
import base64
import hashlib
from datetime import UTC, datetime
from typing import Any, ClassVar, Sequence

import httpx
from pydantic import BaseModel, Field, HttpUrl, ValidationError, model_validator, parse_obj_as

from .base import MCPToolAdapter

try:  # pragma: no cover - optional dependency in some environments
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http import models as qmodels
except ModuleNotFoundError:  # pragma: no cover - gracefully degrade during tests
    AsyncQdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]


DIGEST_NAMESPACE = "neuraforge:doc-loader"


class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=512)
    region: str = Field("wt-wt", description="Region/locale code supported by DuckDuckGo.")
    max_results: int = Field(5, ge=1, le=20)

    model_config = {"extra": "forbid"}


class DuckDuckGoSearchResult(BaseModel):
    title: str
    snippet: str | None = None
    url: HttpUrl | None = None
    source: str | None = None


class DuckDuckGoSearchOutput(BaseModel):
    query: str
    fetched_at: datetime
    results: list[DuckDuckGoSearchResult]


class DuckDuckGoSearchAdapter(MCPToolAdapter):
    name = "search/duckduckgo"
    description = "Run anonymous DuckDuckGo web searches and return the top organic results."
    labels = ("research", "open")
    InputModel = DuckDuckGoSearchInput
    OutputModel = DuckDuckGoSearchOutput

    _endpoint = "https://duckduckgo.com/"
    _headers = {
        "User-Agent": "NeuraForge-MCP-Research/1.0",
        "Accept": "application/json",
    }

    async def _invoke(self, payload_model: DuckDuckGoSearchInput) -> dict[str, Any]:
        params = {
            "q": payload_model.query,
            "format": "json",
            "no_redirect": 1,
            "t": "neuraforge-mcp",
            "kl": payload_model.region,
        }
        async with httpx.AsyncClient(timeout=10.0, headers=self._headers) as client:
            response = await client.get(self._endpoint, params=params)
            response.raise_for_status()
            data = response.json()

        raw_topics = data.get("RelatedTopics") or []
        organic = self._flatten_topics(raw_topics)
        return {
            "query": payload_model.query,
            "fetched_at": datetime.now(UTC),
            "results": [entry.model_dump() for entry in organic[: payload_model.max_results]],
        }

    def _flatten_topics(self, topics: Sequence[Any]) -> list[DuckDuckGoSearchResult]:
        results: list[DuckDuckGoSearchResult] = []
        for item in topics:
            if isinstance(item, dict) and "Topics" in item:
                results.extend(self._flatten_topics(item.get("Topics", [])))
                continue
            if not isinstance(item, dict):
                continue
            url = item.get("FirstURL") or item.get("first_url")
            title = item.get("Text") or item.get("Title")
            snippet = item.get("Result") or item.get("Text")
            if isinstance(snippet, str):
                snippet = self._strip_html(snippet)
            try:
                result = DuckDuckGoSearchResult(
                    title=str(title) if title else "Untitled",
                    snippet=str(snippet) if snippet else None,
                    url=url,
                    source=item.get("Source") or "duckduckgo",
                )
            except ValidationError:  # pragma: no cover - defensive guard
                continue
            results.append(result)
        return results

    @staticmethod
    def _strip_html(value: str) -> str:
        # simple HTML tag stripper
        stripped = []
        inside = False
        for char in value:
            if char == "<":
                inside = True
                continue
            if char == ">":
                inside = False
                continue
            if not inside:
                stripped.append(char)
        return "".join(stripped)


class ArxivQueryInput(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    max_results: int = Field(5, ge=1, le=25)
    sort: str = Field("relevance", pattern="^(relevance|submittedDate)$")

    model_config = {"extra": "forbid"}


class ArxivAuthor(BaseModel):
    name: str


class ArxivEntry(BaseModel):
    identifier: str
    title: str
    summary: str
    published: datetime | None = None
    updated: datetime | None = None
    pdf_url: HttpUrl | None = None
    primary_category: str | None = None
    authors: list[ArxivAuthor]


class ArxivQueryOutput(BaseModel):
    query: str
    fetched_at: datetime
    results: list[ArxivEntry]


class ArxivFetchAdapter(MCPToolAdapter):
    name = "research/arxiv"
    description = "Query the arXiv API for scholarly articles."
    labels = ("research", "open")
    InputModel = ArxivQueryInput
    OutputModel = ArxivQueryOutput

    _endpoint = "https://export.arxiv.org/api/query"
    _rate_limiter: ClassVar[asyncio.Semaphore] = asyncio.Semaphore(3)

    async def _invoke(self, payload_model: ArxivQueryInput) -> dict[str, Any]:
        params = {
            "search_query": payload_model.query,
            "max_results": payload_model.max_results,
            "sortBy": payload_model.sort,
        }
        async with self._rate_limiter:
            async with httpx.AsyncClient(timeout=12.0, headers={"User-Agent": "NeuraForge-MCP-Research/1.0"}) as client:
                response = await client.get(self._endpoint, params=params)
                response.raise_for_status()
                payload = response.text

        entries = list(self._parse_atom(payload))
        return {
            "query": payload_model.query,
            "fetched_at": datetime.now(UTC),
            "results": [entry.model_dump() for entry in entries],
        }

    def _parse_atom(self, raw: str) -> Sequence[ArxivEntry]:
        import xml.etree.ElementTree as ET

        namespace = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        try:
            document = ET.fromstring(raw)
        except ET.ParseError:  # pragma: no cover - defensive guard
            return []

        results: list[ArxivEntry] = []
        for node in document.findall("atom:entry", namespace):
            identifier = node.findtext("atom:id", default="", namespaces=namespace)
            title = (node.findtext("atom:title", default="", namespaces=namespace) or "").strip()
            summary = (node.findtext("atom:summary", default="", namespaces=namespace) or "").strip()
            published = self._parse_datetime(node.findtext("atom:published", namespaces=namespace))
            updated = self._parse_datetime(node.findtext("atom:updated", namespaces=namespace))
            pdf_link = None
            for link in node.findall("atom:link", namespace):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_link = link.attrib.get("href")
                    break

            pdf_url = None
            if pdf_link:
                try:
                    pdf_url = parse_obj_as(HttpUrl, pdf_link)
                except ValidationError:
                    pdf_url = None

            primary_category = None
            cat_node = node.find("arxiv:primary_category", namespace)
            if cat_node is not None:
                primary_category = cat_node.attrib.get("term")

            authors: list[ArxivAuthor] = []
            for author in node.findall("atom:author", namespace):
                name = author.findtext("atom:name", default="", namespaces=namespace)
                if name:
                    authors.append(ArxivAuthor(name=name))

            results.append(
                ArxivEntry(
                    identifier=identifier,
                    title=title or "Untitled",
                    summary=summary,
                    published=published,
                    updated=updated,
                    pdf_url=pdf_url,
                    primary_category=primary_category,
                    authors=authors,
                )
            )
        return results

    @staticmethod
    def _parse_datetime(raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:  # pragma: no cover - defensive guard
            return None


class WikipediaSummaryInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    language: str = Field("en", pattern=r"^[a-z]{2,5}$")
    redirect: bool = Field(True)

    model_config = {"extra": "forbid"}


class WikipediaSummaryOutput(BaseModel):
    title: str
    summary: str
    description: str | None
    language: str
    url: HttpUrl | None
    last_modified: datetime | None


class WikipediaSummaryAdapter(MCPToolAdapter):
    name = "research/wikipedia"
    description = "Retrieve localized summaries from Wikipedia's REST API."
    labels = ("research", "open")
    InputModel = WikipediaSummaryInput
    OutputModel = WikipediaSummaryOutput

    async def _invoke(self, payload_model: WikipediaSummaryInput) -> dict[str, Any]:
        from urllib.parse import quote

        encoded = quote(payload_model.title, safe="")
        language = payload_model.language.lower()
        url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        headers = {
            "User-Agent": "NeuraForge-MCP-Research/1.0",
            "Accept": "application/json",
        }
        if not payload_model.redirect:
            headers["Accept"] = "application/json; redirect=false"
        async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        last_modified = data.get("timestamp") or data.get("items", [{}])[0].get("timestamp")
        return {
            "title": data.get("title") or payload_model.title,
            "summary": data.get("extract") or data.get("description") or "",
            "description": data.get("description"),
            "language": language,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page")
            or data.get("content_urls", {}).get("mobile", {}).get("page"),
            "last_modified": self._parse_timestamp(last_modified),
        }

    @staticmethod
    def _parse_timestamp(raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:  # pragma: no cover - defensive guard
            return None


class DocumentLoaderInput(BaseModel):
    source_url: HttpUrl | None = Field(default=None)
    content_base64: str | None = Field(default=None)
    max_bytes: int = Field(2_097_152, ge=1_024, le=8_388_608)
    chunk_size: int = Field(2_000, ge=4, le=8_000)
    encoding: str = Field("utf-8")

    @model_validator(mode="before")
    def validate_source(cls, data: Any) -> Any:
        if not isinstance(data, dict):  # pragma: no cover - defensive guard
            return data
        if not data.get("source_url") and not data.get("content_base64"):
            raise ValueError("Either source_url or content_base64 must be provided")
        return data

    model_config = {"extra": "forbid"}


class DocumentChunk(BaseModel):
    index: int
    text: str
    offset: int


class DocumentLoaderOutput(BaseModel):
    digest: str
    bytes_loaded: int
    chunks: list[DocumentChunk]


class DocumentLoaderAdapter(MCPToolAdapter):
    name = "research/doc_loader"
    description = "Load textual documents from URLs or base64 blobs and return chunked content."
    labels = ("research", "ingestion")
    InputModel = DocumentLoaderInput
    OutputModel = DocumentLoaderOutput

    async def _invoke(self, payload_model: DocumentLoaderInput) -> dict[str, Any]:
        raw_bytes = await self._read_payload(payload_model)
        text = raw_bytes.decode(payload_model.encoding, errors="replace")
        digest = hashlib.sha256(f"{DIGEST_NAMESPACE}:{text}".encode("utf-8")).hexdigest()
        chunks = self._chunk_text(text, payload_model.chunk_size)
        return {
            "digest": digest,
            "bytes_loaded": len(raw_bytes),
            "chunks": [chunk.model_dump() for chunk in chunks],
        }

    async def _read_payload(self, payload_model: DocumentLoaderInput) -> bytes:
        if payload_model.content_base64:
            data = base64.b64decode(payload_model.content_base64, validate=True)
            if len(data) > payload_model.max_bytes:
                raise ValueError("Decoded content exceeds max_bytes limit")
            return data

        assert payload_model.source_url is not None
        headers = {"User-Agent": "NeuraForge-MCP-Research/1.0"}
        async with httpx.AsyncClient(timeout=12.0, headers=headers) as client:
            response = await client.get(str(payload_model.source_url))
            response.raise_for_status()
            content_length = int(response.headers.get("Content-Length", "0") or 0)
            if content_length and content_length > payload_model.max_bytes:
                raise ValueError("Remote content exceeds max_bytes limit")
            data = response.content
        if len(data) > payload_model.max_bytes:
            raise ValueError("Remote content exceeds max_bytes limit")
        return data

    def _chunk_text(self, text: str, chunk_size: int) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for index, offset in enumerate(range(0, len(text), chunk_size)):
            segment = text[offset : offset + chunk_size]
            chunks.append(DocumentChunk(index=index, text=segment, offset=offset))
        return chunks


class QdrantRetrieverInput(BaseModel):
    query: str | None = Field(default=None, max_length=2_000)
    vector: list[float] | None = Field(default=None)
    limit: int = Field(5, ge=1, le=25)
    filters: list[tuple[str, str]] | None = Field(default=None, description="Simple equality filters as (key, value).")

    @model_validator(mode="before")
    def ensure_vector(cls, data: Any) -> Any:
        if not isinstance(data, dict):  # pragma: no cover - defensive guard
            return data
        if data.get("vector") is None and not data.get("query"):
            raise ValueError("Either query text or vector must be provided")
        return data

    model_config = {"extra": "forbid"}


class QdrantRetrieverHit(BaseModel):
    score: float | None
    payload: dict[str, Any]


class QdrantRetrieverOutput(BaseModel):
    hits: list[QdrantRetrieverHit]
    generated_at: datetime


class QdrantRetrieverAdapter(MCPToolAdapter):
    name = "research/qdrant"
    description = "Search the configured Qdrant vector store for semantic matches."
    labels = ("research", "memory")
    InputModel = QdrantRetrieverInput
    OutputModel = QdrantRetrieverOutput

    _client: ClassVar[Any | None] = None
    _client_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _collection: ClassVar[str | None] = None

    async def _invoke(self, payload_model: QdrantRetrieverInput) -> dict[str, Any]:
        vector = payload_model.vector
        if vector is None:
            vector = await self._embed_text(payload_model.query or "")

        client = await self._get_client()
        if client is None:
            return {"hits": [], "generated_at": datetime.now(UTC)}

        kwargs = {}
        query_filter = self._build_filter(payload_model.filters)
        if query_filter is not None:
            kwargs["query_filter"] = query_filter

        response = await client.search(
            collection_name=self._collection or "neura_tasks",
            query_vector=vector,
            limit=payload_model.limit,
            **kwargs,
        )
        hits: list[QdrantRetrieverHit] = []
        for item in response:
            payload = dict(getattr(item, "payload", {}) or {})
            score = getattr(item, "score", None)
            hits.append(QdrantRetrieverHit(score=score, payload=payload))
        return {
            "hits": [hit.model_dump() for hit in hits],
            "generated_at": datetime.now(UTC),
        }

    async def _embed_text(self, text: str) -> list[float]:
        from app.core.config import get_settings
        from app.services.embedding import EmbeddingService
        from app.services.memory import HybridMemoryService

        settings = get_settings()
        memory = HybridMemoryService.from_settings(settings)
        async with memory.lifecycle():
            embedder = EmbeddingService.from_settings(settings, memory_service=memory)
            try:
                record = await embedder.embed_text(text)
            finally:
                await embedder.aclose()
            return record.vector

    @classmethod
    async def _get_client(cls):
        if AsyncQdrantClient is None:
            return None
        async with cls._client_lock:
            if cls._client is not None:
                return cls._client
            from app.core.config import get_settings

            settings = get_settings()
            client = AsyncQdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)
            cls._collection = settings.qdrant.collection_name
            cls._client = client
            return cls._client

    def _build_filter(self, filters: list[tuple[str, str]] | None):
        if not filters or qmodels is None:
            return None
        conditions = []
        for key, value in filters:
            conditions.append(
                qmodels.FieldCondition(
                    key=key,
                    match=qmodels.MatchValue(value=value),
                )
            )
        if not conditions:
            return None
        return qmodels.Filter(must=conditions)


class SummarizerInput(BaseModel):
    text: str | None = Field(default=None)
    passages: list[str] | None = Field(default=None)
    max_tokens: int = Field(200, ge=10, le=800)
    focus: str | None = Field(default=None, description="Optional focus area to emphasise in summary.")

    @model_validator(mode="before")
    def ensure_content(cls, data: Any) -> Any:
        if not isinstance(data, dict):  # pragma: no cover - defensive guard
            return data
        if not data.get("text") and not data.get("passages"):
            raise ValueError("Either text or passages must be provided")
        return data

    model_config = {"extra": "forbid"}


class SummarizerOutput(BaseModel):
    summary: str
    tokens_used: int
    generated_at: datetime


class SummarizerAdapter(MCPToolAdapter):
    name = "research/summarizer"
    description = "Generate concise summaries from supplied passages without external API calls."
    labels = ("research", "synthesis")
    InputModel = SummarizerInput
    OutputModel = SummarizerOutput

    _timeout_seconds = 6.0

    async def _invoke(self, payload_model: SummarizerInput) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        start = loop.time()
        if hasattr(asyncio, "timeout"):
            async with asyncio.timeout(self._timeout_seconds):  # type: ignore[attr-defined]
                summary = await asyncio.to_thread(self._summarize, payload_model)
        else:  # pragma: no cover - fallback for older python versions
            summary = await asyncio.wait_for(asyncio.to_thread(self._summarize, payload_model), timeout=self._timeout_seconds)
        duration = loop.time() - start

        from app.core import metrics

        metrics.TOOL_LATENCY_SECONDS.labels(tool=self.name).observe(duration)

        tokens = len(summary.split())
        return {
            "summary": summary,
            "tokens_used": tokens,
            "generated_at": datetime.now(UTC),
        }

    def _summarize(self, payload_model: SummarizerInput) -> str:
        text = payload_model.text or "\n".join(payload_model.passages or [])
        sentences = self._split_sentences(text)
        if payload_model.focus:
            sentences = self._prioritize_focus(sentences, payload_model.focus)
        max_words = payload_model.max_tokens
        summary_sentences: list[str] = []
        total_words = 0
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
            if total_words + len(words) > max_words:
                break
            summary_sentences.append(sentence)
            total_words += len(words)
        if not summary_sentences:
            summary_sentences.append(sentences[0] if sentences else text[:200])
        return " ".join(summary_sentences).strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        terminators = {".", "!", "?"}
        sentences: list[str] = []
        buffer: list[str] = []
        for char in text:
            buffer.append(char)
            if char in terminators:
                sentence = "".join(buffer).strip()
                if sentence:
                    sentences.append(sentence)
                buffer.clear()
        if buffer:
            sentence = "".join(buffer).strip()
            if sentence:
                sentences.append(sentence)
        return sentences if sentences else [text.strip()]

    @staticmethod
    def _prioritize_focus(sentences: Sequence[str], focus: str) -> list[str]:
        keyword = focus.lower()
        prioritized = [s for s in sentences if keyword in s.lower()]
        remainder = [s for s in sentences if keyword not in s.lower()]
        return prioritized + remainder


RESEARCH_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    DuckDuckGoSearchAdapter,
    ArxivFetchAdapter,
    WikipediaSummaryAdapter,
    DocumentLoaderAdapter,
    QdrantRetrieverAdapter,
    SummarizerAdapter,
)


__all__ = [
    "DuckDuckGoSearchAdapter",
    "ArxivFetchAdapter",
    "WikipediaSummaryAdapter",
    "DocumentLoaderAdapter",
    "QdrantRetrieverAdapter",
    "SummarizerAdapter",
    "RESEARCH_ADAPTER_CLASSES",
]