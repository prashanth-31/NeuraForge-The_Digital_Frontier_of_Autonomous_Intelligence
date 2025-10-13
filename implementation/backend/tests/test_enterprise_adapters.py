from datetime import UTC, datetime, timedelta

import pytest

from app.mcp.adapters.enterprise import (
    CRMAdapter,
    CalendarSyncAdapter,
    NotionConnectorAdapter,
    PolicyCheckerAdapter,
)


pytestmark = pytest.mark.asyncio


async def test_notion_connector_fetch_requires_page_id() -> None:
    adapter = NotionConnectorAdapter()
    result = await adapter.invoke({"action": "fetch", "page_id": "abc123"})

    assert result["results"][0]["page_id"] == "abc123"


async def test_notion_connector_search_returns_results() -> None:
    adapter = NotionConnectorAdapter()
    result = await adapter.invoke({"action": "search", "query": "playbook"})

    assert len(result["results"]) == 2


async def test_calendar_sync_detects_overlap() -> None:
    adapter = CalendarSyncAdapter()
    start = datetime.now(tz=UTC)
    events = [
        {
            "summary": "Kickoff",
            "start": start,
            "end": start + timedelta(hours=1),
        },
        {
            "summary": "Review",
            "start": start + timedelta(minutes=30),
            "end": start + timedelta(hours=2),
        },
    ]

    result = await adapter.invoke({"calendar_id": "team", "events": events})

    assert result["conflicts"]


async def test_policy_checker_flags_keyword() -> None:
    adapter = PolicyCheckerAdapter()
    document = "This document references restricted data."

    result = await adapter.invoke({"document": document, "policies": ["restricted"]})

    assert result["findings"][0]["status"] == "flagged"
    assert result["compliant"] is False


async def test_crm_adapter_returns_profile() -> None:
    adapter = CRMAdapter()
    result = await adapter.invoke({"contact_email": "user@example.com", "include_history": True})

    assert result["profile_url"].startswith("https://crm.neuraforge.ai/contacts/")
    assert len(result["interactions"]) >= 1