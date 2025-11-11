from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

PLAYBOOK_DEFAULT_QUERY = "enterprise strategy playbook"


def derive_playbook_query(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("playbook_query", "topic", "keyword"):
            candidate = metadata.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()[:256]
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()[:256]
    return PLAYBOOK_DEFAULT_QUERY


def assemble_policy_document(payload: Mapping[str, Any]) -> str:
    sections: list[str] = []
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        sections.append(prompt.strip())
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and metadata:
        sections.append(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    prior = payload.get("prior_outputs")
    if isinstance(prior, Sequence):
        for item in list(prior)[:5]:
            if not isinstance(item, Mapping):
                continue
            agent = str(item.get("agent", "agent"))
            content = item.get("summary") or item.get("content") or ""
            if isinstance(content, str) and content.strip():
                sections.append(f"{agent}: {content.strip()}")
    document = "\n\n".join(section for section in sections if section)
    if len(document) < 20:
        fallback = document or "Enterprise policy review context"
        document = (fallback + "\n" + fallback).strip()
    if len(document) < 20:
        document = document.ljust(20, ".")
    return document


def extract_policy_hints(payload: Mapping[str, Any]) -> list[str]:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        policies = metadata.get("policies")
        if isinstance(policies, Sequence):
            hints = [str(item).strip() for item in policies if isinstance(item, str) and item.strip()]
            if hints:
                return hints
    return []


def actions_from_notion(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = response.get("results") if isinstance(response, Mapping) else None
    if not isinstance(results, Sequence):
        return []
    actions: list[dict[str, Any]] = []
    for entry in results:
        if not isinstance(entry, Mapping):
            continue
        title = entry.get("title") or entry.get("name") or entry.get("page_id") or "Playbook reference"
        snippet = entry.get("snippet") or entry.get("summary") or "Leverage documented best practice."
        action: dict[str, Any] = {
            "action": str(title),
            "impact": str(snippet),
            "origin": "notion",
        }
        page_id = entry.get("page_id")
        if isinstance(page_id, str):
            action["page_id"] = page_id
            action.setdefault("source", f"notion://{page_id}")
        url = entry.get("url") or entry.get("link")
        if isinstance(url, str):
            action["source"] = url
        actions.append(action)
    return actions


def actions_from_policy(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    findings = response.get("findings") if isinstance(response, Mapping) else None
    actions: list[dict[str, Any]] = []
    if isinstance(findings, Sequence):
        for finding in findings:
            if not isinstance(finding, Mapping):
                continue
            status = str(finding.get("status", "")).lower()
            policy = finding.get("policy") or "policy"
            details = finding.get("details") or "Review policy guidance."
            if status and status != "pass":
                actions.append(
                    {
                        "action": f"Mitigate policy risk: {policy}",
                        "impact": str(details),
                        "origin": "policy_checker",
                    }
                )
    if actions:
        return actions
    return [
        {
            "action": "Confirm compliance readiness",
            "impact": "Policy checker returned no blocking findings.",
            "origin": "policy_checker",
        }
    ]


__all__ = [
    "PLAYBOOK_DEFAULT_QUERY",
    "derive_playbook_query",
    "assemble_policy_document",
    "extract_policy_hints",
    "actions_from_notion",
    "actions_from_policy",
]
