from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[misc,assignment]

from ..agents.base import BaseAgent
from ..core.config import Settings, get_settings
from ..core.logging import get_logger
from ..schemas.agents import AgentCapability

logger = get_logger(name=__name__)


@dataclass(slots=True)
class CapabilityProfile:
    capability: AgentCapability
    description: str
    dependencies: tuple[AgentCapability, ...] = ()
    embedding: tuple[float, ...] | None = None


@dataclass(slots=True)
class RoutingDecision:
    agents: list[BaseAgent]
    scores: dict[str, float] = field(default_factory=dict)
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [agent.name for agent in self.agents]


class AgentRouter(Protocol):
    async def select(self, *, task: Mapping[str, Any], agents: Sequence[BaseAgent]) -> RoutingDecision:
        ...


class DynamicAgentRouter:
    """Hybrid semantic + heuristic agent selector with dependency-aware ordering."""

    _CAPABILITY_KEYWORDS: Mapping[AgentCapability, tuple[str, ...]] = {
        AgentCapability.RESEARCH: (
            "research",
            "investigate",
            "analysis",
            "analyze",
            "explain",
            "compare",
            "summarize",
            "overview",
            "evaluate",
            "insight",
            "report",
            "study",
            "deep dive",
            "whitepaper",
        ),
        AgentCapability.FINANCE: (
            "finance",
            "financial",
            "revenue",
            "roi",
            "budget",
            "profit",
            "loss",
            "forecast",
            "pricing",
            "cost",
            "cash flow",
            "expenses",
            "valuation",
            "market share",
            "investment",
        ),
        AgentCapability.CREATIVE: (
            "story",
            "narrative",
            "creative",
            "campaign",
            "copy",
            "tagline",
            "slogan",
            "script",
            "poem",
            "concept",
            "idea",
            "tone",
            "voice",
            "greeting",
            "write",
        ),
        AgentCapability.ENTERPRISE: (
            "enterprise",
            "roadmap",
            "strategy",
            "strategic",
            "executive",
            "stakeholder",
            "policy",
            "governance",
            "compliance",
            "initiative",
            "milestone",
            "timeline",
            "program",
            "portfolio",
            "transformation",
        ),
    }

    _GREETING_WORDS: set[str] = {
        "hi",
        "hello",
        "hey",
        "hiya",
        "howdy",
        "yo",
        "sup",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
    }

    _AGENT_NAME_KEYS = ("agents", "preferred_agents", "force_agents", "target_agents")
    _CAPABILITY_KEYS = ("capabilities", "preferred_capabilities", "target_capabilities")

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        capability_path: str | Path | None = None,
        model_path: str | None = None,
        alpha: float = 0.7,
        beta: float = 0.3,
        similarity_threshold: float = 0.55,
        combined_threshold: float = 0.6,
        max_agents: int = 4,
    ) -> None:
        self._settings = settings or get_settings()
        self._capability_path = Path(capability_path or Path(__file__).with_name("capabilities.json"))

        embedding_config = getattr(self._settings, "embedding", None)
        default_model = getattr(embedding_config, "default_model", None) if embedding_config else None
        preferred_dimension = getattr(embedding_config, "preferred_dimension", 768) if embedding_config else 768

        resolved_model = model_path or default_model or "models/all-mpnet-base-v2"
        self._model_path = str(resolved_model)
        self._dimension = int(preferred_dimension) if preferred_dimension else 768

        self._alpha = max(0.0, min(1.0, alpha))
        self._beta = max(0.0, min(1.0, beta))
        if self._alpha == 0.0 and self._beta == 0.0:
            self._alpha, self._beta = 0.7, 0.3

        self._similarity_threshold = similarity_threshold
        self._combined_threshold = combined_threshold
        self._max_agents = max(1, max_agents)

        self._model: SentenceTransformer | None = None
        self._profiles = self._load_profiles()

        logger.info(
            "dynamic_router_initialized",
            model=self._model_path,
            alpha=self._alpha,
            beta=self._beta,
            similarity_threshold=self._similarity_threshold,
            combined_threshold=self._combined_threshold,
            capability_path=str(self._capability_path),
        )

    async def select(self, *, task: Mapping[str, Any], agents: Sequence[BaseAgent]) -> RoutingDecision:
        if not agents:
            return RoutingDecision(agents=[], reason="empty-roster")

        metadata_raw = task.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
        limit = self._resolve_limit(metadata, len(agents))

        prompt_text = str(task.get("prompt") or "").strip()
        combined_blob, text_blobs = self._collect_text_blobs(prompt_text, metadata_raw)

        greeting_decision = self._try_greeting_shortcut(prompt_text, agents)
        if greeting_decision is not None:
            return greeting_decision

        requested_agents = self._extract_requested_agents(metadata_raw)
        requested_capabilities = self._extract_requested_capabilities(metadata_raw)

        prompt_vector = self._embed_text(combined_blob or prompt_text.lower())
        heuristic_scores = self._compute_heuristic_scores(text_blobs, requested_capabilities, agents)

        combined_scores: dict[str, float] = {}
        embedding_scores: dict[str, float] = {}
        component_scores: dict[str, Any] = {}
        selected_agents: list[BaseAgent] = []

        for agent in agents:
            capability = getattr(agent, "capability", None)
            if not isinstance(capability, AgentCapability):
                continue

            profile = self._profiles.get(capability)
            embedding_score = self._similarity(prompt_vector, profile.embedding if profile else None)
            heuristic_score = heuristic_scores.get(capability, 0.0)
            combined = (self._alpha * heuristic_score) + (self._beta * embedding_score)

            if agent.name.lower() in requested_agents or capability in requested_capabilities:
                combined = max(combined, 1.0)
                heuristic_score = max(heuristic_score, 1.0)
                embedding_score = max(embedding_score, self._similarity_threshold)

            embedding_scores[agent.name] = embedding_score
            combined_scores[agent.name] = combined
            component_scores[agent.name] = {
                "heuristic": heuristic_score,
                "embedding": embedding_score,
            }

            include_agent = (
                agent.name.lower() in requested_agents
                or capability in requested_capabilities
                or embedding_score >= self._similarity_threshold
                or combined >= self._combined_threshold
            )
            if include_agent:
                selected_agents.append(agent)

        if not selected_agents:
            fallback_agent = max(agents, key=lambda item: combined_scores.get(item.name, 0.0))
            selected_agents = [fallback_agent]

        limited_agents = self._apply_limit(selected_agents, combined_scores, limit)
        ordered_agents = self._apply_dependencies(limited_agents, agents, combined_scores)

        scores = {agent.name: combined_scores.get(agent.name, 0.0) for agent in ordered_agents}
        available = [agent.name for agent in agents]
        skipped = [agent.name for agent in agents if agent not in ordered_agents]

        metadata_payload = {
            "available_agents": available,
            "selected_agents": [agent.name for agent in ordered_agents],
            "skipped_agents": skipped,
            "scores": scores,
            "embedding_scores": embedding_scores,
            "component_scores": component_scores,
            "alpha": self._alpha,
            "beta": self._beta,
            "similarity_threshold": self._similarity_threshold,
            "combined_threshold": self._combined_threshold,
            "dependency_graph": self._dag_metadata(ordered_agents),
            "requested_agents": sorted(requested_agents),
            "requested_capabilities": [capability.value for capability in requested_capabilities],
            "limit": limit,
            "final_agent_count": len(ordered_agents),
        }

        return RoutingDecision(
            agents=ordered_agents,
            scores=scores,
            reason="dynamic.embedding_routing",
            metadata=metadata_payload,
        )

    def _resolve_limit(self, metadata: Mapping[str, Any], roster_size: int) -> int:
        limit = self._max_agents
        raw_value = metadata.get("max_agents") if metadata else None
        if isinstance(raw_value, (int, float)):
            limit = int(raw_value)
        elif isinstance(raw_value, str):
            try:
                limit = int(raw_value.strip())
            except ValueError:
                limit = self._max_agents

        allow_multi = metadata.get("allow_multi_agents") if metadata else None
        if isinstance(allow_multi, bool) and not allow_multi:
            limit = 1

        if limit <= 0:
            limit = 1
        return min(limit, max(1, roster_size))

    def _apply_limit(
        self,
        agents: Sequence[BaseAgent],
        scores: Mapping[str, float],
        limit: int,
    ) -> list[BaseAgent]:
        ordered = sorted(agents, key=lambda agent: scores.get(agent.name, 0.0), reverse=True)
        if len(ordered) <= limit:
            return list(ordered)
        return list(ordered[:limit])

    def _apply_dependencies(
        self,
        selected_agents: Sequence[BaseAgent],
        roster: Sequence[BaseAgent],
        scores: Mapping[str, float],
    ) -> list[BaseAgent]:
        if not selected_agents:
            return []

        capability_to_agent = {
            agent.capability: agent for agent in roster if isinstance(agent.capability, AgentCapability)
        }
        selected_capabilities = {
            agent.capability for agent in selected_agents if isinstance(agent.capability, AgentCapability)
        }
        expanded_capabilities = self._expand_dependencies(selected_capabilities)

        adjacency: dict[AgentCapability, set[AgentCapability]] = {}
        indegree: dict[AgentCapability, int] = {}
        for capability in expanded_capabilities:
            profile = self._profiles.get(capability)
            dependencies = set(profile.dependencies if profile else ()) & expanded_capabilities
            adjacency[capability] = set()
            indegree.setdefault(capability, 0)
            for dependency in dependencies:
                adjacency.setdefault(dependency, set()).add(capability)
                indegree[capability] = indegree.get(capability, 0) + 1
                indegree.setdefault(dependency, 0)

        scores_by_capability: dict[AgentCapability, float] = {}
        for capability, agent in capability_to_agent.items():
            scores_by_capability[capability] = scores.get(agent.name, 0.0)

        zero_indegree = [cap for cap, degree in indegree.items() if degree == 0]
        zero_indegree.sort(key=lambda cap: scores_by_capability.get(cap, 0.0), reverse=True)

        ordered_capabilities: list[AgentCapability] = []
        while zero_indegree:
            current = zero_indegree.pop(0)
            ordered_capabilities.append(current)
            for successor in adjacency.get(current, set()):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    zero_indegree.append(successor)
                    zero_indegree.sort(key=lambda cap: scores_by_capability.get(cap, 0.0), reverse=True)

        if len(ordered_capabilities) < len(expanded_capabilities):
            remaining = [cap for cap in expanded_capabilities if cap not in ordered_capabilities]
            remaining.sort(key=lambda cap: scores_by_capability.get(cap, 0.0), reverse=True)
            ordered_capabilities.extend(remaining)

        ordered_agents: list[BaseAgent] = []
        for capability in ordered_capabilities:
            agent = capability_to_agent.get(capability)
            if agent is None:
                continue
            if agent not in ordered_agents:
                ordered_agents.append(agent)
        return ordered_agents

    def _expand_dependencies(self, selected: set[AgentCapability]) -> set[AgentCapability]:
        resolved = set(selected)
        added = True
        while added:
            added = False
            for capability in list(resolved):
                profile = self._profiles.get(capability)
                if not profile:
                    continue
                for dependency in profile.dependencies:
                    if dependency not in resolved:
                        resolved.add(dependency)
                        added = True
        return resolved

    def _dag_metadata(self, ordered_agents: Sequence[BaseAgent]) -> list[dict[str, str]]:
        edges: list[dict[str, str]] = []
        for agent in ordered_agents:
            capability = getattr(agent, "capability", None)
            profile = self._profiles.get(capability) if isinstance(capability, AgentCapability) else None
            if not profile:
                continue
            for dependency in profile.dependencies:
                edges.append({"from": dependency.value, "to": capability.value})
        return edges

    def _collect_text_blobs(self, prompt: str, metadata: Any) -> tuple[str, list[str]]:
        blobs: list[str] = []
        if prompt:
            blobs.append(prompt.lower())
        for value in self._iter_metadata_values(metadata):
            if isinstance(value, str) and value.strip():
                blobs.append(value.strip().lower())
        combined = " ".join(blobs)
        return combined, blobs

    def _iter_metadata_values(self, metadata: Any) -> Iterable[Any]:
        if metadata is None:
            return []
        if isinstance(metadata, Mapping):
            values = list(metadata.values())
        elif isinstance(metadata, Iterable) and not isinstance(metadata, (bytes, bytearray, str)):
            values = list(metadata)
        else:
            return []
        for value in values:
            if isinstance(value, Mapping):
                yield from self._iter_metadata_values(value)
            elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
                yield from self._iter_metadata_values(list(value))
            else:
                yield value

    def _compute_heuristic_scores(
        self,
        text_blobs: Sequence[str],
        requested_capabilities: set[AgentCapability],
        agents: Sequence[BaseAgent],
    ) -> dict[AgentCapability, float]:
        combined_text = " \\n ".join(text_blobs)
        capability_scores: dict[AgentCapability, float] = {capability: 0.0 for capability in AgentCapability}
        if combined_text:
            for capability, keywords in self._CAPABILITY_KEYWORDS.items():
                hits = 0
                for keyword in keywords:
                    if keyword in combined_text:
                        hits += 1
                if hits > 0:
                    capability_scores[capability] = min(1.0, hits / 4.0)

        for capability in requested_capabilities:
            capability_scores[capability] = 1.0

        if not combined_text:
            available_capabilities = {
                agent.capability for agent in agents if isinstance(agent.capability, AgentCapability)
            }
            for capability in available_capabilities:
                capability_scores.setdefault(capability, 0.25)
        return capability_scores

    def _extract_requested_agents(self, metadata: Any) -> set[str]:
        names: set[str] = set()
        if not isinstance(metadata, Mapping):
            return names
        for key in self._AGENT_NAME_KEYS:
            for item in self._normalize_list(metadata.get(key)):
                names.add(item.lower())
        return names

    def _extract_requested_capabilities(self, metadata: Any) -> set[AgentCapability]:
        capabilities: set[AgentCapability] = set()
        if not isinstance(metadata, Mapping):
            return capabilities
        for key in self._CAPABILITY_KEYS:
            for item in self._normalize_list(metadata.get(key)):
                capability = self._coerce_capability(item)
                if capability is not None:
                    capabilities.add(capability)
        return capabilities

    def _normalize_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            tokens = re.split(r"[,;]\s*", value)
            return [token.strip() for token in tokens if token.strip()]
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            result: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    result.append(item.strip())
            return result
        return []

    def _coerce_capability(self, value: Any) -> AgentCapability | None:
        if isinstance(value, AgentCapability):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower().replace("_agent", "")
            for capability in AgentCapability:
                if normalized == capability.value or normalized == capability.name.lower():
                    return capability
        return None

    def _try_greeting_shortcut(
        self,
        prompt: str,
        agents: Sequence[BaseAgent],
    ) -> RoutingDecision | None:
        if not prompt:
            return None
        normalized = re.sub(r"[^a-z0-9\s]", " ", prompt.lower())
        words = [token for token in normalized.split() if token]
        if not words:
            return None
        greeting_hits = sum(1 for word in words if word in self._GREETING_WORDS)
        non_greeting = len(words) - greeting_hits
        if greeting_hits == 0 or len(words) > 6 or non_greeting > 1:
            return None
        creative_agent = next(
            (agent for agent in agents if getattr(agent, "capability", None) is AgentCapability.CREATIVE),
            None,
        )
        if creative_agent is None:
            return None
        scores = {agent.name: (1.0 if agent is creative_agent else 0.0) for agent in agents}
        metadata = {
            "tokens": words,
            "reason": "greeting",
        }
        return RoutingDecision(
            agents=[creative_agent],
            scores=scores,
            reason="dynamic.greeting",
            metadata=metadata,
        )

    def _load_profiles(self) -> dict[AgentCapability, CapabilityProfile]:
        profiles: dict[AgentCapability, CapabilityProfile] = {}
        try:
            payload = json.loads(self._capability_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("capability_profile_missing", path=str(self._capability_path))
            payload = {}
        except json.JSONDecodeError as exc:
            logger.warning("capability_profile_invalid", path=str(self._capability_path), error=str(exc))
            payload = {}

        for capability in AgentCapability:
            raw = payload.get(capability.value) or {}
            description = str(raw.get("description") or capability.value)
            dependencies_raw = raw.get("dependencies") or []
            dependencies: list[AgentCapability] = []
            for item in dependencies_raw:
                capability_dep = self._coerce_capability(item)
                if capability_dep is not None and capability_dep not in dependencies:
                    dependencies.append(capability_dep)
            profile = CapabilityProfile(
                capability=capability,
                description=description,
                dependencies=tuple(dependencies),
            )
            profile.embedding = self._embed_capability(description)
            profiles[capability] = profile
        return profiles

    def _ensure_model(self) -> SentenceTransformer | None:  # type: ignore[name-defined]
        if SentenceTransformer is None:
            logger.warning("sentence_transformers_not_available")
            return None
        if self._model is not None:
            return self._model
        try:
            self._model = SentenceTransformer(self._model_path)
        except Exception as exc:  # pragma: no cover - runtime environment guard
            logger.warning("embedding_model_unavailable", model=self._model_path, error=str(exc))
            self._model = None
        return self._model

    def _embed_capability(self, description: str) -> tuple[float, ...] | None:
        model = self._ensure_model()
        if model is None:
            return None
        vector = model.encode([description], normalize_embeddings=True)[0]
        return tuple(float(x) for x in vector.tolist())

    def _embed_text(self, text: str) -> tuple[float, ...]:
        model = self._ensure_model()
        if model is None:
            return tuple(0.0 for _ in range(self._dimension))
        vector = model.encode([text], normalize_embeddings=True)[0]
        return tuple(float(x) for x in vector.tolist())

    def _similarity(self, a: tuple[float, ...], b: tuple[float, ...] | None) -> float:
        if not a or not b:
            return 0.0
        if len(a) != len(b):
            length = min(len(a), len(b))
            a = a[:length]
            b = b[:length]
        numerator = sum(x * y for x, y in zip(a, b))
        if numerator == 0.0:
            return 0.0
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(-1.0, min(1.0, numerator / (norm_a * norm_b)))


__all__ = [
    "AgentRouter",
    "CapabilityProfile",
    "DynamicAgentRouter",
    "RoutingDecision",
]
