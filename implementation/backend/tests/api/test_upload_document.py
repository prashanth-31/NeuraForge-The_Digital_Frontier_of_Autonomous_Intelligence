from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import pytest

from app.api import routes as routes_module
from app.main import app


class _StubLLM:
    async def generate(self, prompt: str, *, system_prompt: str | None = None, temperature: float | None = None) -> str:  # noqa: ARG002
        return "stub-analysis"


class _StubMemoryService:
    def __init__(self) -> None:
        self.ephemeral: dict[str, dict[str, object]] = {}
        self.working: dict[str, str] = {}

    async def store_ephemeral_memory(self, task_id: str, payload: dict[str, object]) -> None:
        self.ephemeral[task_id] = payload

    async def store_working_memory(self, key: str, value: str, *, ttl: int = 600) -> None:  # noqa: ARG002
        self.working[key] = value

    @asynccontextmanager
    async def lifecycle(self):  # noqa: D401
        yield self


@pytest.mark.asyncio
async def test_upload_document_returns_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_module.LLMService, "from_settings", classmethod(lambda cls, settings, *, model=None, client=None: _StubLLM()))  # type: ignore[arg-type]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/upload_document",
            files={"document": ("sample.txt", b"NeuraForge document upload test", "text/plain")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"] == "stub-analysis"
    assert payload["document"]["filename"] == "sample.txt"
    assert payload["document"]["line_count"] == 1


@pytest.mark.asyncio
async def test_upload_document_rejects_unsupported_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_module.LLMService, "from_settings", classmethod(lambda cls, settings, *, model=None, client=None: _StubLLM()))  # type: ignore[arg-type]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/upload_document",
            files={"document": ("sample.bin", b"binary", "application/octet-stream")},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_document_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    memory_stub = _StubMemoryService()

    class _HybridMemoryStub:
        @classmethod
        def from_settings(cls, settings):  # noqa: ANN001
            return memory_stub

    monkeypatch.setattr(routes_module.LLMService, "from_settings", classmethod(lambda cls, settings, *, model=None, client=None: _StubLLM()))  # type: ignore[arg-type]
    monkeypatch.setattr(routes_module, "HybridMemoryService", _HybridMemoryStub)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/upload_document",
            params={"persist": "true"},
            files={"document": ("persist.txt", b"NeuraForge persistence", "text/plain")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["persisted"] is True
    assert payload["memory_task_id"]
    assert memory_stub.ephemeral
    assert memory_stub.working
