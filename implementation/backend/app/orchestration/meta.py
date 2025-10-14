from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Sequence

from ..core.config import MetaAgentSettings
from ..core.logging import get_logger
from ..core.metrics import increment_meta_dispute, observe_meta_resolution_latency
from ..services.disputes import (
    DisputeAssessment,
    DisputeDetector,
    MetaConfidenceInput,
    MetaConfidenceScorer,
    MetaConfidenceStats,
    build_inputs_from_outputs,
)
from ..services.llm import LLMService

logger = get_logger(name=__name__)

ValidationCallable = Callable[["EvidenceItem"], Awaitable["ValidationOutcome"]]


@dataclass(slots=True)
class ValidationOutcome:
    name: str
    success: bool
    details: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "success": self.success, "details": self.details}


@dataclass(slots=True)
class EvidenceItem:
    agent: str
    summary: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    validation: list[ValidationOutcome] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "summary": self.summary,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "validation": [result.as_dict() for result in self.validation],
        }


@dataclass(slots=True)
class MetaResolution:
    summary: str
    confidence: float
    mode: str
    evidence: list[EvidenceItem]
    dispute: DisputeAssessment | None
    validation: list[ValidationOutcome]
    should_escalate: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "confidence": round(self.confidence, 4),
            "mode": self.mode,
            "should_escalate": self.should_escalate,
            "validation": [item.as_dict() for item in self.validation],
            "evidence": [item.as_dict() for item in self.evidence],
            "dispute": self.dispute.as_dict() if self.dispute else None,
        }


class MetaAgent:
    """LLM-backed synthesizer that reconciles agent outputs into a single recommendation."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        settings: MetaAgentSettings,
        dispute_detector: DisputeDetector,
        confidence_scorer: MetaConfidenceScorer | None = None,
        validators: Sequence[ValidationCallable] | None = None,
    ) -> None:
        self._llm = llm_service
        self._settings = settings
        self._dispute_detector = dispute_detector
        self._confidence_scorer = confidence_scorer or MetaConfidenceScorer()
        self._validators = tuple(validators or ())
        self._consensus_delta = settings.consensus_delta_threshold

    async def synthesize(
        self,
        *,
        task: dict[str, Any],
        outputs: Sequence[dict[str, Any]],
        negotiation: dict[str, Any] | None,
    ) -> MetaResolution:
        if not outputs:
            logger.info("meta_agent_no_outputs", task=task.get("id"))
            empty = MetaResolution(
                summary="No agent outputs available for synthesis.",
                confidence=0.0,
                mode="fallback",
                evidence=[],
                dispute=None,
                validation=[],
                should_escalate=False,
            )
            observe_meta_resolution_latency(mode=empty.mode, latency=0.0)
            return empty

        evidence_items = self._build_evidence(outputs)
        if not evidence_items:
            logger.warning("meta_agent_no_evidence", task=task.get("id"))
            empty = MetaResolution(
                summary="Agent outputs lacked usable summaries for synthesis.",
                confidence=0.0,
                mode="fallback",
                evidence=[],
                dispute=None,
                validation=[],
                should_escalate=False,
            )
            observe_meta_resolution_latency(mode=empty.mode, latency=0.0)
            return empty

        confidence_inputs: Sequence[MetaConfidenceInput] = build_inputs_from_outputs(outputs)
        stats = self._confidence_scorer.score(confidence_inputs)
        dispute = self._dispute_detector.evaluate(confidence_inputs)
        if dispute.flagged:
            increment_meta_dispute(severity=dispute.severity)

        prompt = self._build_prompt(
            task=task,
            outputs=evidence_items,
            negotiation=negotiation,
            stats=stats,
        )
        summary_mode = "llm"
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        try:
            summary = await self._llm.generate(prompt, system_prompt=self._settings.summary_prompt)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("meta_agent_summary_failed", error=str(exc))
            summary_mode = "fallback"
            summary = self._fallback_summary(evidence_items)
        latency = loop.time() - start_time
        observe_meta_resolution_latency(mode=summary_mode, latency=latency)

        if self._settings.validation_enabled:
            validation_summary, per_item_validations = await self._run_validations(evidence_items)
            for item in evidence_items:
                item.validation.extend(per_item_validations.get(item.agent, []))
        else:
            validation_summary = []

        final_confidence = stats.weighted_mean
        should_escalate = bool(dispute.flagged and self._settings.escalate_on_dispute)
        return MetaResolution(
            summary=summary.strip(),
            confidence=round(final_confidence, 4),
            mode=summary_mode,
            evidence=evidence_items,
            dispute=dispute,
            validation=validation_summary,
            should_escalate=should_escalate,
        )

    async def _run_validations(
        self,
        evidence: Sequence[EvidenceItem],
    ) -> tuple[list[ValidationOutcome], dict[str, list[ValidationOutcome]]]:
        if not self._validators:
            return [], {}
        summary: list[ValidationOutcome] = []
        per_item: dict[str, list[ValidationOutcome]] = {item.agent: [] for item in evidence}
        for validator in self._validators:
            try:
                results = await asyncio.gather(*(validator(item) for item in evidence), return_exceptions=True)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.exception("meta_agent_validator_failed", error=str(exc))
                continue
            for item, result in zip(evidence, results):
                if isinstance(result, Exception):
                    logger.error("meta_agent_validation_error", error=str(result))
                    continue
                summary.append(result)
                per_item.setdefault(item.agent, []).append(result)
        return summary, per_item

    def _build_prompt(
        self,
        *,
        task: dict[str, Any],
        outputs: Sequence[EvidenceItem],
        negotiation: dict[str, Any] | None,
        stats: MetaConfidenceStats,
    ) -> str:
        negotiation_lines: list[str] = []
        if negotiation:
            outcome = negotiation.get("outcome")
            rationale = negotiation.get("rationale")
            if outcome:
                negotiation_lines.append(f"Outcome: {outcome}")
            if rationale:
                negotiation_lines.append(f"Rationale: {rationale}")
        prompt_parts = [
            "Task Prompt:",
            str(task.get("prompt", "")),
            "\nAgent Evidence:",
        ]
        for item in outputs[: self._settings.max_evidence_items]:
            prompt_parts.append(f"- {item.agent} (confidence={item.confidence:.2f}): {item.summary}")
        prompt_parts.append(
            "\nConfidence statistics: "
            f"weighted_mean={stats.weighted_mean:.2f}, "
            f"stddev={stats.stddev:.2f}, "
            f"max_delta={stats.max_delta:.2f}."
        )
        prompt_parts.append(
            "\nInstructions: Summarize consensus, highlight disagreements, and recommend next steps."
        )
        if negotiation_lines:
            prompt_parts.append("\nNegotiation Summary:\n" + "\n".join(negotiation_lines))
        return "\n".join(prompt_parts)

    def _build_evidence(self, outputs: Iterable[dict[str, Any]]) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for entry in outputs:
            if not isinstance(entry, dict):
                continue
            summary = str(entry.get("summary") or entry.get("content") or "").strip()
            if not summary:
                continue
            metadata = entry.get("metadata")
            evidence.append(
                EvidenceItem(
                    agent=str(entry.get("agent", "unknown")),
                    summary=summary,
                    confidence=float(entry.get("confidence", 0.0) or 0.0),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        return evidence

    def _fallback_summary(self, evidence: Sequence[EvidenceItem]) -> str:
        if not evidence:
            return "No agent evidence available."
        ranked = sorted(evidence, key=lambda item: item.confidence, reverse=True)
        best = ranked[0]
        dissenters = [
            item for item in ranked[1:] if abs(item.confidence - best.confidence) > self._consensus_delta
        ]
        lines = [
            f"Primary recommendation from {best.agent}: {best.summary} (confidence={best.confidence:.2f})."
        ]
        if dissenters:
            dissent_text = "; ".join(f"{item.agent} ({item.confidence:.2f})" for item in dissenters)
            lines.append(f"Dissenting agents: {dissent_text}.")
        return " ".join(lines)
