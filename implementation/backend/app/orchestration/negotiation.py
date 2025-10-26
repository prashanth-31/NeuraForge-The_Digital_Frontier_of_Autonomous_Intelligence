from __future__ import annotations

from collections.abc import Sequence
from math import isclose
from typing import Any

from pydantic import ValidationError

from ..core.logging import get_logger
from ..schemas.negotiation import NegotiationDecision, NegotiationProposal

logger = get_logger(name=__name__)


class NegotiationEngine:
    async def decide(self, task: dict[str, Any], outputs: Sequence[dict[str, Any]]) -> NegotiationDecision:
        raise NotImplementedError


class SimpleNegotiationEngine(NegotiationEngine):
    def __init__(self, *, consensus_threshold: float = 0.15) -> None:
        self.consensus_threshold = consensus_threshold

    async def decide(self, task: dict[str, Any], outputs: Sequence[dict[str, Any]]) -> NegotiationDecision:
        proposals = self._parse_proposals(outputs)
        if not proposals:
            logger.warning("negotiation_no_proposals", task_id=task.get("id"))
            return NegotiationDecision(
                outcome="No agent proposals available",
                confidence=0.0,
                consensus=0.0,
                supporting_agents=[],
                dissenting_agents=[],
                rationale="All agents returned empty responses.",
                metadata={"strategy": "simple", "reason": "empty"},
            )

        top = max(proposals, key=lambda item: item.confidence)
        total_confidence = sum(item.confidence for item in proposals) or 1.0
        supporting = [
            item
            for item in proposals
            if isclose(item.confidence, top.confidence, abs_tol=self.consensus_threshold)
            or item.confidence >= top.confidence - self.consensus_threshold
        ]
        dissenting = [item for item in proposals if item not in supporting]

        consensus_value = sum(item.confidence for item in supporting) / total_confidence
        rationale_parts = [
            f"{item.agent}: {item.summary} (confidence={item.confidence:.2f})"
            for item in supporting
        ]
        dissent_parts = [
            f"{item.agent}: {item.summary} (confidence={item.confidence:.2f})"
            for item in dissenting
        ]

        metadata: dict[str, Any] = {
            "strategy": "simple",
            "consensus_threshold": self.consensus_threshold,
            "supporting_summaries": rationale_parts,
            "dissenting_summaries": dissent_parts,
        }

        rationale_lines: list[str] = []
        if rationale_parts:
            rationale_lines.append("Supporting agents:\n- " + "\n- ".join(rationale_parts))
        if dissent_parts:
            rationale_lines.append("Dissenting agents:\n- " + "\n- ".join(dissent_parts))

        return NegotiationDecision(
            outcome=top.summary,
            confidence=min(1.0, top.confidence if supporting else top.confidence * consensus_value),
            consensus=max(0.0, min(1.0, consensus_value)),
            supporting_agents=[item.agent for item in supporting],
            dissenting_agents=[item.agent for item in dissenting],
            rationale="\n\n".join(rationale_lines) or None,
            metadata=metadata,
        )

    @staticmethod
    def _parse_proposals(outputs: Sequence[dict[str, Any]]) -> list[NegotiationProposal]:
        proposals: list[NegotiationProposal] = []
        for entry in outputs:
            if not isinstance(entry, dict):
                continue
            payload = {
                "agent": entry.get("agent", "unknown"),
                "summary": entry.get("summary") or entry.get("content") or "",
                "confidence": float(entry.get("confidence") or 0.0),
                "rationale": entry.get("rationale"),
                "metadata": entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {},
            }
            if not payload["summary"]:
                continue
            try:
                proposals.append(NegotiationProposal(**payload))
            except ValidationError as exc:  # pragma: no cover - defensive guard
                logger.warning("negotiation_proposal_invalid", error=str(exc), payload=payload)
        return proposals
