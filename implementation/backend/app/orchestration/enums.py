from __future__ import annotations

from enum import Enum


class LifecycleStatus(str, Enum):
    QUEUED = "queued"
    PLANNED = "planned"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GuardrailDecisionType(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"
    REVIEW = "review"


__all__ = ["LifecycleStatus", "GuardrailDecisionType"]
