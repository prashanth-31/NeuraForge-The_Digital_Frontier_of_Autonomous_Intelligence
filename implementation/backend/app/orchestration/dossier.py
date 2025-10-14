from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime to prevent cycles
    from .meta import MetaResolution


@dataclass(slots=True)
class AgentPerspective:
    agent: str
    summary: str
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "summary": self.summary,
            "confidence": round(self.confidence, 4),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class DecisionDossier:
    task_id: str
    prompt: str
    created_at: datetime
    headline: str
    agent_perspectives: Sequence[AgentPerspective]
    meta_resolution: Mapping[str, Any] | None
    negotiation: Mapping[str, Any] | None
    plan: Mapping[str, Any] | None
    guardrail_decisions: Sequence[Mapping[str, Any]]
    reviewer_notes: Sequence[str]
    escalation: Mapping[str, Any]

    @property
    def summary(self) -> str:
        return self.headline

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "created_at": self.created_at.isoformat(),
            "headline": self.headline,
            "agent_perspectives": [item.as_dict() for item in self.agent_perspectives],
            "meta_resolution": dict(self.meta_resolution) if self.meta_resolution is not None else None,
            "negotiation": dict(self.negotiation) if self.negotiation is not None else None,
            "plan": dict(self.plan) if self.plan is not None else None,
            "guardrail_decisions": [dict(decision) for decision in self.guardrail_decisions],
            "reviewer_notes": list(self.reviewer_notes),
            "escalation": dict(self.escalation),
        }

    def as_markdown(self) -> str:
        lines: list[str] = [
            f"# Decision Dossier for Task {self.task_id}",
            "",
            f"- Generated: {self.created_at.isoformat()}",
            f"- Prompt: {self.prompt}",
            "",
            "## Summary",
            self.headline or "No summary available.",
            "",
            "## Agent Perspectives",
        ]
        if self.agent_perspectives:
            for perspective in self.agent_perspectives:
                meta = perspective.metadata
                metadata_str = ""
                if meta:
                    key_values = ", ".join(f"{key}={value}" for key, value in meta.items())
                    metadata_str = f" ({key_values})"
                lines.append(
                    f"- **{perspective.agent}** (confidence={perspective.confidence:.2f}){metadata_str}: {perspective.summary}"
                )
        else:
            lines.append("- No agent outputs recorded.")
        lines.append("")

        lines.append("## Meta-Agent Resolution")
        if self.meta_resolution:
            lines.append(f"- Mode: {self.meta_resolution.get('mode', 'n/a')}")
            lines.append(f"- Confidence: {self.meta_resolution.get('confidence', 'n/a')}")
            lines.append(f"- Summary: {self.meta_resolution.get('summary', 'n/a')}")
            dispute = self.meta_resolution.get("dispute")
            if dispute:
                lines.append(f"- Dispute Severity: {dispute.get('severity', 'n/a')}")
                lines.append(f"- Dissenting Agents: {', '.join(dispute.get('dissenting_agents', [])) or 'None'}")
            lines.append(f"- Escalation Recommended: {self.meta_resolution.get('should_escalate', False)}")
        else:
            lines.append("- Meta-agent resolution unavailable.")
        lines.append("")

        lines.append("## Negotiation Overview")
        if self.negotiation:
            lines.append(f"- Outcome: {self.negotiation.get('outcome', 'n/a')}")
            if rationale := self.negotiation.get("rationale"):
                lines.append(f"- Rationale: {rationale}")
            if consensus := self.negotiation.get("consensus"):
                lines.append(f"- Consensus: {consensus}")
        else:
            lines.append("- Negotiation data unavailable.")
        lines.append("")

        lines.append("## Plan Snapshot")
        if self.plan:
            lines.append(f"- Status: {self.plan.get('status', 'n/a')}")
            if summary := self.plan.get("summary"):
                lines.append(f"- Summary: {summary}")
            steps = self.plan.get("steps")
            if isinstance(steps, Iterable):
                lines.append("")
                lines.append("### Planned Steps")
                for step in steps:
                    if not isinstance(step, Mapping):
                        continue
                    lines.append(
                        f"- {step.get('step_id', 'unknown')} -> {step.get('description', '')} (agent: {step.get('agent', 'n/a')})"
                    )
        else:
            lines.append("- No plan generated.")
        lines.append("")

        lines.append("## Guardrail Decisions")
        if self.guardrail_decisions:
            for decision in self.guardrail_decisions:
                lines.append(
                    f"- Step {decision.get('step_id', 'n/a')} -> {decision.get('decision', 'n/a')} (policy: {decision.get('policy_id', 'n/a')})"
                )
        else:
            lines.append("- No guardrail evaluations recorded.")
        lines.append("")

        lines.append("## Reviewer Notes")
        if self.reviewer_notes:
            for note in self.reviewer_notes:
                lines.append(f"- {note}")
        else:
            lines.append("- None")
        lines.append("")

        lines.append("## Escalation")
        if self.escalation:
            lines.append(f"- Status: {self.escalation.get('status', 'n/a')}")
            sources = self.escalation.get("sources")
            if isinstance(sources, Sequence):
                lines.append(f"- Sources: {', '.join(str(item) for item in sources) or 'n/a'}")
            if notes := self.escalation.get("notes"):
                lines.append(f"- Notes: {notes}")
        else:
            lines.append("- Not escalated.")

        return "\n".join(lines)


def _extract_guardrail_decisions(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    guardrails = state.get("guardrails")
    if not isinstance(guardrails, Mapping):
        return []
    decisions = guardrails.get("decisions")
    if not isinstance(decisions, Sequence):
        return []
    return [decision for decision in decisions if isinstance(decision, Mapping)]


def _collect_reviewer_notes(state: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    review = state.get("review")
    if isinstance(review, Mapping):
        raw = review.get("notes")
        if isinstance(raw, str) and raw.strip():
            notes.append(raw.strip())
        elif isinstance(raw, Sequence):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    notes.append(item.strip())
    escalation = state.get("escalation")
    if isinstance(escalation, Mapping):
        extra = escalation.get("notes")
        if isinstance(extra, str) and extra.strip():
            notes.append(extra.strip())
    return notes


def _build_agent_perspectives(outputs: Sequence[Mapping[str, Any]]) -> list[AgentPerspective]:
    perspectives: list[AgentPerspective] = []
    for entry in outputs:
        agent = str(entry.get("agent", "unknown"))
        summary = str(entry.get("summary") or entry.get("content") or "").strip()
        confidence = float(entry.get("confidence", 0.0) or 0.0)
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
        perspectives.append(
            AgentPerspective(agent=agent, summary=summary, confidence=confidence, metadata=metadata)
        )
    return perspectives


def _derive_headline(
    *,
    meta_resolution: Mapping[str, Any] | None,
    perspectives: Sequence[AgentPerspective],
) -> str:
    if meta_resolution:
        summary = str(meta_resolution.get("summary", "")).strip()
        if summary:
            return summary
    for perspective in perspectives:
        if perspective.summary:
            return perspective.summary
    return "No summary available."


def build_decision_dossier(
    state: Mapping[str, Any],
    *,
    meta_resolution: "MetaResolution" | Mapping[str, Any] | None = None,
) -> DecisionDossier:
    task_id = str(state.get("id") or state.get("task_id") or "unknown")
    prompt = str(state.get("prompt") or "")
    outputs = state.get("outputs")
    if isinstance(outputs, Sequence):
        perspectives = _build_agent_perspectives([entry for entry in outputs if isinstance(entry, Mapping)])
    else:
        perspectives = []

    plan = state.get("plan") if isinstance(state.get("plan"), Mapping) else None
    negotiation = state.get("negotiation") if isinstance(state.get("negotiation"), Mapping) else None
    meta_payload: Mapping[str, Any] | None
    if meta_resolution is not None:
        if hasattr(meta_resolution, "as_dict"):
            meta_payload = meta_resolution.as_dict()  # type: ignore[assignment]
        elif isinstance(meta_resolution, Mapping):
            meta_payload = dict(meta_resolution)
        else:
            meta_payload = None
    else:
        meta_payload = state.get("meta") if isinstance(state.get("meta"), Mapping) else None
    guardrail_decisions = _extract_guardrail_decisions(state)
    reviewer_notes = _collect_reviewer_notes(state)
    escalation = state.get("escalation") if isinstance(state.get("escalation"), Mapping) else {}
    headline = _derive_headline(meta_resolution=meta_payload, perspectives=perspectives)

    return DecisionDossier(
        task_id=task_id,
        prompt=prompt,
        created_at=datetime.now(timezone.utc),
        headline=headline,
        agent_perspectives=perspectives,
        meta_resolution=meta_payload,
        negotiation=negotiation,
        plan=plan,
        guardrail_decisions=guardrail_decisions,
        reviewer_notes=reviewer_notes,
        escalation=escalation,
    )
