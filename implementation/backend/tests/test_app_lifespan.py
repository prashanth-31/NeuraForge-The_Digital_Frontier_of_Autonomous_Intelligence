from __future__ import annotations

import asyncio

import httpx
import pytest


@pytest.mark.asyncio
async def test_consolidation_loop_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main

    start_event = asyncio.Event()
    cancel_event = asyncio.Event()

    async def fake_loop() -> None:
        start_event.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:  # pragma: no cover - cancellation path exercised in test
            cancel_event.set()
            raise

    monkeypatch.setattr(main, "_consolidation_loop", fake_loop)
    monkeypatch.setattr(main.settings.consolidation, "enabled", True, raising=False)

    transport = httpx.ASGITransport(app=main.app)
    async with main.app.router.lifespan_context(main.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/")
            assert response.status_code == 200
            await asyncio.wait_for(start_event.wait(), timeout=1.0)

    await asyncio.wait_for(cancel_event.wait(), timeout=1.0)
    await transport.aclose()