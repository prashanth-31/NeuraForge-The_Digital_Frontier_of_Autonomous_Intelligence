from __future__ import annotations

import io
from tempfile import SpooledTemporaryFile

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.core.config import DocumentSettings
from app.services import document_parser
from app.services.document_ingestion import DocumentIngestionService
from app.services.document_parser import DocumentParseResult, parse_document


class _StubMemory:
    def __init__(self) -> None:
        self.working: dict[str, str] = {}
        self.ephemeral: dict[str, dict[str, object]] = {}

    async def store_working_memory(self, key: str, value: str, ttl: int | None = None) -> None:  # noqa: ARG002
        self.working[key] = value

    async def store_ephemeral_memory(self, task_id: str, payload: dict[str, object]) -> None:
        self.ephemeral[task_id] = payload

    async def fetch_ephemeral_memory(self, task_id: str) -> dict[str, object] | None:
        return self.ephemeral.get(task_id)

    async def fetch_working_memory(self, key: str) -> str | None:
        return self.working.get(key)


class _StubEmbedder:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls = 0

    async def embed_documents(
        self,
        documents: list[str],
        *,
        metadatas: list[dict[str, object]] | None = None,  # noqa: ARG002
        store: bool = False,  # noqa: ARG002
        ids: list[str] | None = None,  # noqa: ARG002
        collection: str | None = None,  # noqa: ARG002
        score: float = 1.0,  # noqa: ARG002
    ) -> list[object]:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("embedding unavailable")
        return [doc for doc in documents]


def _make_upload(*, filename: str, content_type: str, data: bytes) -> UploadFile:
    file = SpooledTemporaryFile()
    file.write(data)
    file.seek(0)
    headers = Headers({"content-type": content_type})
    return UploadFile(filename=filename, file=file, headers=headers)


@pytest.mark.asyncio
async def test_parse_document_handles_csv_upload() -> None:
    upload = _make_upload(filename="table.csv", content_type="text/csv", data=b"col1,col2\n1,2\n")
    try:
        result = await parse_document(upload)
    finally:
        await upload.close()
    assert "col1" in result.text
    assert result.metadata["extension"] == ".csv"
    assert result.metadata["line_count"] >= 2


@pytest.mark.asyncio
async def test_parse_document_reads_pdf_via_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePage:
        def extract_text(self) -> str:
            return "pdf body"

    class _FakePdf:
        def __init__(self) -> None:
            self.pages = [_FakePage()]

        def __enter__(self) -> "_FakePdf":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ARG002
            return False

    class _FakePlumber:
        def open(self, buffer: io.BytesIO) -> _FakePdf:  # noqa: ARG002
            return _FakePdf()

    monkeypatch.setattr(document_parser, "pdfplumber", _FakePlumber())

    upload = _make_upload(filename="doc.pdf", content_type="application/pdf", data=b"%PDF-fake")
    try:
        result = await parse_document(upload)
    finally:
        await upload.close()

    assert "pdf body" in result.text
    assert result.metadata["extension"] == ".pdf"


@pytest.mark.asyncio
async def test_document_ingestion_chunks_and_persists() -> None:
    memory = _StubMemory()
    embedder = _StubEmbedder()
    service = DocumentIngestionService(memory=memory, embedder=embedder, config=DocumentSettings(chunk_size=200, chunk_overlap=50))
    parsed = DocumentParseResult(
        text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5,
        metadata={
            "filename": "sample.pdf",
            "content_type": "application/pdf",
            "extension": ".pdf",
            "filesize_bytes": 128,
            "line_count": 10,
            "character_count": 320,
        },
    )

    result = await service.ingest(parsed)

    assert result.chunk_count >= 1
    assert result.document_id in memory.ephemeral
    stored_metadata = memory.ephemeral[result.document_id]
    assert isinstance(stored_metadata.get("chunk_keys"), list)
    assert len(stored_metadata["chunk_keys"]) == result.chunk_count
    chunk_entries = [key for key in memory.working if key.startswith(f"document:{result.document_id}:chunk")]
    assert len(chunk_entries) == result.chunk_count
    assert embedder.calls == 1


@pytest.mark.asyncio
async def test_document_ingestion_survives_embedding_failure() -> None:
    memory = _StubMemory()
    embedder = _StubEmbedder(should_fail=True)
    service = DocumentIngestionService(memory=memory, embedder=embedder, config=DocumentSettings())
    parsed = DocumentParseResult(
        text="alpha beta gamma delta",
        metadata={
            "filename": "fail.pdf",
            "content_type": "application/pdf",
            "extension": ".pdf",
            "filesize_bytes": 32,
            "line_count": 1,
            "character_count": 24,
        },
    )

    result = await service.ingest(parsed)

    assert result.chunk_count == 1
    assert embedder.calls == 1
    assert result.document_id in memory.ephemeral