from __future__ import annotations

import random
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Iterable, Optional
from uuid import UUID

try:
    import asyncpg
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore[assignment]

from ..core.config import GuardrailSettings, Settings
from ..core.metrics import increment_guardrail_decision
from ..core.logging import get_logger
from ..services.llm import LLMService
from .enums import GuardrailDecisionType

logger = get_logger(name=__name__)


@dataclass(slots=True)
class GuardrailDecision:
    decision: GuardrailDecisionType
    reason: str
    risk_score: float | None = None
    policy_id: str | None = None
    metadata: dict[str, Any] = None

    def model_dump(self) -> dict[str, Any]:  # compatibility with pydantic style usage
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "risk_score": self.risk_score,
            "policy_id": self.policy_id,
            "metadata": self.metadata or {},
        }


@dataclass(slots=True)
class PolicyRule:
    policy_id: str
    description: str
    on_violation: GuardrailDecisionType
    keywords: tuple[str, ...] = ()
    min_risk_score: float = 0.8
    evaluator: Callable[[dict[str, Any]], float] | None = None

    def evaluate(self, payload: dict[str, Any]) -> GuardrailDecision | None:
        score = self._score(payload)
        if score < self.min_risk_score:
            return None
        reason = f"Policy {self.policy_id} triggered with score {score:.2f}"
        return GuardrailDecision(
            decision=self.on_violation,
            reason=reason,
            risk_score=score,
            policy_id=self.policy_id,
            metadata={"keywords": self.keywords},
        )

    def _score(self, payload: dict[str, Any]) -> float:
        if self.evaluator is not None:
            try:
                return float(self.evaluator(payload))
            except Exception as exc:  # pragma: no cover - guard evaluator errors
                logger.warning("policy_evaluator_failed", policy=self.policy_id, error=str(exc))
                return 0.0
        text = " ".join(
            str(payload.get(key, "")) for key in ("description", "summary", "content", "title")
        ).lower()
        for keyword in self.keywords:
            if keyword.lower() in text:
                return 1.0
        return float(payload.get("risk_score") or 0.0)


class GuardrailStore:
    _INSERT = """
        INSERT INTO guardrail_events(task_id, run_id, decision, reason, risk_score, policy_id, agent, payload)
        VALUES($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
    """

    def __init__(self, pool: Any | None) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required for GuardrailStore")
        self._pool_or_factory = pool
        self._pool: asyncpg.Pool | None = None  # type: ignore[name-defined]

    @classmethod
    def from_settings(cls, settings: Settings) -> "GuardrailStore":
        if asyncpg is None:
            raise RuntimeError("asyncpg is not available")
        pool = asyncpg.create_pool(
            dsn=str(settings.postgres.dsn),
            min_size=settings.postgres.pool_min_size,
            max_size=settings.postgres.pool_max_size,
        )
        return cls(pool)

    async def record(self, *, task_id: str, run_id: UUID | None, decision: GuardrailDecision, agent: str | None, payload: dict[str, Any]) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                self._INSERT,
                task_id,
                run_id,
                decision.decision.value,
                decision.reason,
                decision.risk_score,
                decision.policy_id,
                agent,
                payload,
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _ensure_pool(self) -> asyncpg.Pool:  # type: ignore[name-defined]
        if self._pool is not None:
            return self._pool
        candidate = self._pool_or_factory
        if isinstance(candidate, asyncpg.Pool):
            self._pool = candidate
            return self._pool
        if hasattr(candidate, "__await__"):
            candidate = await candidate
        if isinstance(candidate, asyncpg.Pool):
            self._pool = candidate
            return self._pool
        raise RuntimeError("Invalid asyncpg pool supplied to GuardrailStore")

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator["GuardrailStore"]:
        try:
            await self._ensure_pool()
            yield self
        finally:
            await self.close()


class GuardrailManager:
    def __init__(
        self,
        *,
        settings: GuardrailSettings,
        policies: Iterable[PolicyRule] | None = None,
        store: GuardrailStore | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self._settings = settings
        self._policies = list(policies or self._default_policies())
        self._store = store
        self._llm = llm_service

    async def evaluate_step(
        self,
        *,
        task_id: str,
        run_id: UUID | None,
        step: dict[str, Any],
        agent: str | None = None,
    ) -> GuardrailDecision:
        decision = self._evaluate_policies(step)
        if decision.decision is GuardrailDecisionType.ALLOW and self._settings.enforce_safety_filters:
            decision = await self._safety_filter(step, decision)
        if self._settings.red_team_sampling_rate > 0:
            await self._maybe_enqueue_red_team(step)
        increment_guardrail_decision(decision=decision.decision.value, policy_id=decision.policy_id)
        if self._store is not None and self._settings.audit_log_enabled:
            await self._store.record(task_id=task_id, run_id=run_id, decision=decision, agent=agent, payload=step)
        return decision

    def _evaluate_policies(self, payload: dict[str, Any]) -> GuardrailDecision:
        threshold = self._settings.risk_threshold
        payload_risk = float(payload.get("risk_score") or 0.0)
        if payload_risk >= threshold:
            reason = f"Baseline risk {payload_risk:.2f} exceeds threshold {threshold:.2f}"
            return GuardrailDecision(decision=GuardrailDecisionType.REVIEW, reason=reason, risk_score=payload_risk)
        for policy in self._policies:
            decision = policy.evaluate(payload)
            if decision is not None:
                return decision
        return GuardrailDecision(decision=GuardrailDecisionType.ALLOW, reason="No guardrail triggered", risk_score=payload_risk)

    async def _safety_filter(self, payload: dict[str, Any], prior: GuardrailDecision) -> GuardrailDecision:
        if self._llm is None:
            return prior
        text = payload.get("description") or payload.get("summary") or ""
        if not text:
            return prior
        try:
            assessment = await self._llm.moderate(text)
        except AttributeError:  # pragma: no cover - moderation optional
            return prior
        except Exception as exc:  # pragma: no cover - LLM failure shouldn't block
            logger.warning("guardrail_safety_filter_failed", error=str(exc))
            return prior
        severity = float(assessment.get("severity", 0.0))
        if severity >= 0.9:
            return GuardrailDecision(
                decision=GuardrailDecisionType.DENY,
                reason="Safety filter denied content",
                risk_score=severity,
                metadata={"filter": "llm", "detail": assessment},
            )
        if severity >= 0.6:
            return GuardrailDecision(
                decision=GuardrailDecisionType.REVIEW,
                reason="Safety filter flagged content",
                risk_score=severity,
                metadata={"filter": "llm", "detail": assessment},
            )
        return prior

    async def _maybe_enqueue_red_team(self, payload: dict[str, Any]) -> None:
        if random.random() > self._settings.red_team_sampling_rate:
            return
        logger.info("guardrail_red_team_prompt", payload=payload)

    def _default_policies(self) -> list[PolicyRule]:
        return [
            PolicyRule(
                policy_id="compliance.high_risk",
                description="Escalate when high risk score provided",
                on_violation=GuardrailDecisionType.ESCALATE,
                min_risk_score=max(0.8, self._settings.risk_threshold),
            ),
            PolicyRule(
                policy_id="compliance.prohibited_keywords",
                description="Deny when prohibited keywords detected",
                on_violation=GuardrailDecisionType.DENY,
                keywords=("classified", "leak", "malicious"),
            ),
        ]


class InMemoryGuardrailStore(GuardrailStore):
    def __init__(self) -> None:  # pragma: no cover - fallback for tests
        self._records: list[dict[str, Any]] = []
        self._pool_or_factory = None
        self._pool = None

    async def record(self, *, task_id: str, run_id: UUID | None, decision: GuardrailDecision, agent: str | None, payload: dict[str, Any]) -> None:
        self._records.append(
            {
                "task_id": task_id,
                "run_id": run_id,
                "decision": decision,
                "agent": agent,
                "payload": payload,
            }
        )

    async def close(self) -> None:
        return

    async def _ensure_pool(self) -> asyncpg.Pool:  # type: ignore[name-defined]
        raise RuntimeError("In-memory guardrail store does not provide a database pool")

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator["InMemoryGuardrailStore"]:
        try:
            yield self
        finally:
            return
