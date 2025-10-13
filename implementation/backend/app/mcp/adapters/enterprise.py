from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, model_validator

from .base import MCPToolAdapter


class NotionConnectorInput(BaseModel):
    action: str = Field("fetch", pattern=r"^(fetch|search)$")
    page_id: str | None = Field(default=None, max_length=128)
    query: str | None = Field(default=None, max_length=256)

    @model_validator(mode="before")
    def enforce_requirements(cls, data: Any) -> Any:
        if not isinstance(data, dict):  # pragma: no cover - defensive guard
            return data
        action = data.get("action", "fetch")
        if action == "fetch" and not data.get("page_id"):
            raise ValueError("page_id is required for fetch action")
        if action == "search" and not data.get("query"):
            raise ValueError("query is required for search action")
        return data

    model_config = {"extra": "forbid"}


class NotionConnectorOutput(BaseModel):
    action: str
    results: list[dict[str, Any]]
    generated_at: datetime


class NotionConnectorAdapter(MCPToolAdapter):
    name = "enterprise/notion"
    description = "Fetches workspace metadata from a Notion-like store (stubbed)."
    labels = ("enterprise", "knowledge")
    InputModel = NotionConnectorInput
    OutputModel = NotionConnectorOutput

    async def _invoke(self, payload_model: NotionConnectorInput) -> dict[str, Any]:
        if payload_model.action == "fetch":
            results = [
                {
                    "page_id": payload_model.page_id,
                    "title": f"Page {payload_model.page_id}",
                    "last_updated": datetime.now(UTC).isoformat(),
                    "status": "active",
                }
            ]
        else:
            results = [
                {
                    "page_id": f"match-{index+1}",
                    "title": f"Result for {payload_model.query}",
                    "snippet": "High-level summary placeholder.",
                }
                for index in range(2)
            ]
        return {
            "action": payload_model.action,
            "results": results,
            "generated_at": datetime.now(UTC),
        }


class CalendarEvent(BaseModel):
    summary: str = Field(..., max_length=128)
    start: datetime
    end: datetime
    location: str | None = Field(default=None, max_length=128)

    @model_validator(mode="before")
    def validate_order(cls, data: Any) -> Any:
        if not isinstance(data, dict):  # pragma: no cover - defensive guard
            return data
        start = data.get("start")
        end = data.get("end")
        if start is not None and end is not None and end <= start:
            raise ValueError("Event end must occur after start")
        return data


class CalendarSyncInput(BaseModel):
    calendar_id: str = Field(..., max_length=64)
    events: list[CalendarEvent] = Field(..., min_length=1, max_length=20)

    model_config = {"extra": "forbid"}


class CalendarSyncOutput(BaseModel):
    calendar_id: str
    applied_events: int
    conflicts: list[str]
    generated_at: datetime


class CalendarSyncAdapter(MCPToolAdapter):
    name = "enterprise/calendar"
    description = "Validates and stages calendar events for synchronization."
    labels = ("enterprise", "scheduling")
    InputModel = CalendarSyncInput
    OutputModel = CalendarSyncOutput

    async def _invoke(self, payload_model: CalendarSyncInput) -> dict[str, Any]:
        conflicts: list[str] = []
        sorted_events = sorted(payload_model.events, key=lambda event: event.start)
        for previous, current in zip(sorted_events, sorted_events[1:]):
            if current.start < previous.end:
                conflicts.append(
                    f"Overlap between '{previous.summary}' and '{current.summary}'"
                )
        applied = len(payload_model.events) - len(conflicts)
        return {
            "calendar_id": payload_model.calendar_id,
            "applied_events": max(applied, 0),
            "conflicts": conflicts,
            "generated_at": datetime.now(UTC),
        }


class PolicyCheckerInput(BaseModel):
    document: str = Field(..., min_length=20, max_length=8_000)
    policies: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class PolicyCheckerFinding(BaseModel):
    policy: str
    status: str
    details: str


class PolicyCheckerOutput(BaseModel):
    findings: list[PolicyCheckerFinding]
    compliant: bool


class PolicyCheckerAdapter(MCPToolAdapter):
    name = "enterprise/policy_checker"
    description = "Performs lightweight policy compliance checks."
    labels = ("enterprise", "compliance")
    InputModel = PolicyCheckerInput
    OutputModel = PolicyCheckerOutput

    async def _invoke(self, payload_model: PolicyCheckerInput) -> dict[str, Any]:
        findings: list[PolicyCheckerFinding] = []
        text_lower = payload_model.document.lower()
        for policy in payload_model.policies or ["confidential", "restricted"]:
            marker = policy.lower()
            status = "pass" if marker not in text_lower else "flagged"
            details = "Keyword absent" if status == "pass" else "Keyword detected in document"
            findings.append(PolicyCheckerFinding(policy=policy, status=status, details=details))
        compliant = all(item.status == "pass" for item in findings)
        return {
            "findings": [finding.model_dump() for finding in findings],
            "compliant": compliant,
        }


class CRMAdapterInput(BaseModel):
    contact_email: str = Field(..., max_length=128)
    include_history: bool = Field(False)

    model_config = {"extra": "forbid"}


class CRMInteraction(BaseModel):
    occurred_at: datetime
    channel: str
    summary: str


class CRMAdapterOutput(BaseModel):
    contact_email: str
    status: str
    interactions: list[CRMInteraction]
    profile_url: HttpUrl | None


class CRMAdapter(MCPToolAdapter):
    name = "enterprise/crm"
    description = "Provides CRM contact roll-ups from deterministic fixtures."
    labels = ("enterprise", "crm")
    InputModel = CRMAdapterInput
    OutputModel = CRMAdapterOutput

    async def _invoke(self, payload_model: CRMAdapterInput) -> dict[str, Any]:
        base_interactions = [
            CRMInteraction(
                occurred_at=datetime(2024, 10, 1, tzinfo=UTC),
                channel="email",
                summary="Shared Q4 roadmap draft.",
            ),
            CRMInteraction(
                occurred_at=datetime(2024, 11, 15, tzinfo=UTC),
                channel="call",
                summary="Discussed renewal adjustments.",
            ),
        ]
        interactions = base_interactions if payload_model.include_history else base_interactions[:1]
        profile_url = f"https://crm.neuraforge.ai/contacts/{payload_model.contact_email}"
        return {
            "contact_email": payload_model.contact_email,
            "status": "active",
            "interactions": [interaction.model_dump() for interaction in interactions],
            "profile_url": profile_url,
        }


ENTERPRISE_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    NotionConnectorAdapter,
    CalendarSyncAdapter,
    PolicyCheckerAdapter,
    CRMAdapter,
)


__all__ = [
    "NotionConnectorAdapter",
    "CalendarSyncAdapter",
    "PolicyCheckerAdapter",
    "CRMAdapter",
    "ENTERPRISE_ADAPTER_CLASSES",
]