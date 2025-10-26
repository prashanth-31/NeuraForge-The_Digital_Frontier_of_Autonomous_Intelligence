from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import pytest
from fastapi.testclient import TestClient

from app.api import routes as routes_module
from app.core import rate_limit as rate_limit_module
from app.core.config import RateLimitRule, get_settings
from app.core.security import create_access_token
from app.dependencies import get_hybrid_memory, get_task_queue
from app.main import app


class StubRedis:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.expiry: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        value = self.counts.get(key, 0) + 1
        self.counts[key] = value
        return value

    async def expire(self, key: str, seconds: int) -> None:  # noqa: ARG002
        self.expiry[key] = max(1, int(seconds))

    async def ttl(self, key: str) -> int:
        return self.expiry.get(key, 0)


class StubMemoryService:
    def __init__(self) -> None:
        self.ephemeral: dict[str, dict[str, Any]] = {}
        self.working: dict[str, list[str]] = {}

    @asynccontextmanager
    async def lifecycle(self):  # noqa: D401 - simple async context
        yield self

    async def store_ephemeral_memory(self, task_id: str, payload: dict[str, Any]) -> None:
        self.ephemeral[task_id] = payload

    async def fetch_ephemeral_memory(self, task_id: str) -> dict[str, Any] | None:
        return self.ephemeral.get(task_id)

    async def store_working_memory(self, key: str, value: str, *, ttl: int = 600) -> None:  # noqa: ARG002
        self.working.setdefault(key, []).append(value)

    async def store_working_batch(self, items: list[tuple[str, str]], *, ttl: int = 600) -> None:  # noqa: ARG002
        for key, value in items:
            await self.store_working_memory(key, value, ttl=ttl)

    async def fetch_recent_context(self, *, agent: str | None = None, limit: int = 5) -> list[dict[str, Any]]:  # noqa: ARG002
        return []

    async def retrieve_context(self, *, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:  # noqa: ARG002
        return []


class StubLLMService:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        return "stub-response"


class ImmediateQueue:
    def __init__(self) -> None:
        self.calls = 0

    async def enqueue(self, job: Callable[[], Awaitable[Any]]) -> None:
        self.calls += 1
        await job()


class StubEmbeddingService:
    async def embed_text(self, text: str, *, metadata: dict[str, Any] | None = None, store: bool = False):  # noqa: ARG002
        return SimpleNamespace(vector=[0.0])

    async def aclose(self) -> None:
        return None


@pytest.fixture()
def dossier_client(monkeypatch: pytest.MonkeyPatch):
    memory = StubMemoryService()

    async def override_memory():
        yield memory

    app.dependency_overrides[get_hybrid_memory] = override_memory

    client = TestClient(app)
    try:
        yield client, memory
    finally:
        app.dependency_overrides.clear()


def test_dossier_requires_reports_scope(dossier_client) -> None:
    client, memory = dossier_client
    task_id = "task-secured"
    memory.ephemeral[task_id] = {
        "result": {
            "dossier": {
                "json": {"summary": "secure"},
                "markdown": "# secure",
            }
        }
    }

    response = client.get(f"/api/v1/reports/{task_id}/dossier.json")
    assert response.status_code == 401

    limited_token = create_access_token("observer-0")
    response = client.get(
        f"/api/v1/reports/{task_id}/dossier.json",
        headers={"Authorization": f"Bearer {limited_token}"},
    )
    assert response.status_code == 403

    reviewer_token = create_access_token("reviewer-1", extra_claims={"roles": ["reviewer"]})
    response = client.get(
        f"/api/v1/reports/{task_id}/dossier.json",
        headers={"Authorization": f"Bearer {reviewer_token}"},
    )
    assert response.status_code == 200
    assert response.json() == {"summary": "secure"}


def test_task_submission_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_redis = StubRedis()
    monkeypatch.setattr(rate_limit_module, "_rate_limit_client", None)

    async def _redis_factory(settings):  # noqa: ANN001, ARG001
        return stub_redis

    monkeypatch.setattr(rate_limit_module, "_get_rate_limit_client", _redis_factory)

    settings = get_settings()
    original = settings.rate_limit.model_copy(deep=True)
    settings.rate_limit.enabled = True
    settings.rate_limit.namespace = "test:ratelimit"
    settings.rate_limit.task_submission = RateLimitRule(capacity=2, window_seconds=5)

    memory = StubMemoryService()
    llm = StubLLMService()
    queue = ImmediateQueue()

    monkeypatch.setattr(routes_module.HybridMemoryService, "from_settings", lambda _: memory)
    monkeypatch.setattr(routes_module.LLMService, "from_settings", lambda _: llm)
    monkeypatch.setattr(routes_module.EmbeddingService, "from_settings", lambda *_, **__: StubEmbeddingService())

    async def override_queue():
        yield queue

    async def override_memory():
        yield memory

    app.dependency_overrides[get_task_queue] = override_queue
    app.dependency_overrides[get_hybrid_memory] = override_memory

    token = create_access_token("submitter", extra_claims={"roles": ["reviewer"]})
    headers = {"Authorization": f"Bearer {token}"}

    client = TestClient(app)
    try:
        for _ in range(2):
            accepted = client.post(
                "/api/v1/submit_task",
                json={"prompt": "secure", "metadata": {}},
                headers=headers,
            )
            assert accepted.status_code == 200

        throttled = client.post(
            "/api/v1/submit_task",
            json={"prompt": "secure", "metadata": {}},
            headers=headers,
        )
        assert throttled.status_code == 429
        assert "Retry-After" in throttled.headers
    finally:
        app.dependency_overrides.clear()
        settings.rate_limit = original
        rate_limit_module._rate_limit_client = None