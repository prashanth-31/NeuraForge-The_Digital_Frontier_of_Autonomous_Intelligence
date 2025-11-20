from __future__ import annotations

import json
import math
import string
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..agents.base import BaseAgent, get_agent_schema
from ..core.config import Settings
from ..core.logging import get_logger
from ..schemas.agents import AgentCapability
from ..services.llm import LLMService
from ..services.tools import get_tool_service
from ..tools.registry import tool_registry
from .tool_policy import get_agent_tool_policy
from .planner_contract import (
    PlanGatekeeper,
    PlannerContractViolation,
    PlannerPlan,
    PlannedAgentStep,
)

logger = get_logger(name=__name__)

_DEFAULT_TOOL_ALIAS_CACHE: dict[str, str] | None = None
_CAPABILITY_CACHE: dict[str, dict[str, Any]] | None = None
_CAPABILITY_CACHE: dict[str, dict[str, Any]] | None = None


def _load_default_tool_aliases() -> dict[str, str]:
    global _DEFAULT_TOOL_ALIAS_CACHE
    if _DEFAULT_TOOL_ALIAS_CACHE is not None:
        return _DEFAULT_TOOL_ALIAS_CACHE
    try:
        from ..services.tools import DEFAULT_TOOL_ALIASES  # type: ignore circular-import
    except Exception:  # pragma: no cover - fallback if tooling layer unavailable
        _DEFAULT_TOOL_ALIAS_CACHE = {}
    else:
        _DEFAULT_TOOL_ALIAS_CACHE = {str(alias): str(target) for alias, target in DEFAULT_TOOL_ALIASES.items()}
    return _DEFAULT_TOOL_ALIAS_CACHE


def _load_capability_catalog() -> dict[str, dict[str, Any]]:
    global _CAPABILITY_CACHE
    if _CAPABILITY_CACHE is not None:
        return _CAPABILITY_CACHE
    path = Path(__file__).with_name("capabilities.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("planner_capabilities_missing", path=str(path))
        payload = {}
    except json.JSONDecodeError as exc:
        logger.warning("planner_capabilities_invalid", path=str(path), error=str(exc))
        payload = {}
    catalog: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, Mapping):
            continue
        catalog[str(key)] = {
            "description": str(value.get("description") or "") or "(no description)",
            "dependencies": [str(dep) for dep in value.get("dependencies", []) if isinstance(dep, str)],
        }
    _CAPABILITY_CACHE = catalog
    return catalog


class PlannerError(RuntimeError):
    """Raised when the orchestration planner cannot produce a valid plan."""


class LLMOrchestrationPlanner:
    """LLM-backed planner that selects agents and tools for orchestration."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm_service: LLMService | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm_service or LLMService.from_settings(settings, model=settings.planner_llm.model)
        self._system_prompt = settings.planner_llm.system_prompt
        self._temperature = settings.planner_llm.temperature
        self._gatekeeper = PlanGatekeeper()

    async def plan(
        self,
        *,
        task: Mapping[str, Any],
        prior_outputs: Sequence[Mapping[str, Any]],
        agents: Sequence[BaseAgent],
        tool_aliases: Mapping[str, str] | None = None,
    ) -> PlannerPlan:
        original_prompt = str(task.get("prompt") or "")
        capability_catalog = _load_capability_catalog()
        alias_snapshot = await self._resolve_tool_alias_snapshot()
        merged_aliases = self._merge_alias_sources(alias_snapshot, tool_aliases)
        prompt = self._build_prompt(
            task=task,
            prior_outputs=prior_outputs,
            agents=agents,
            tool_aliases=merged_aliases,
            capabilities=capability_catalog,
        )
        logger.debug("planner_prompt", prompt=prompt)
        response = await self._llm.generate(
            prompt=prompt,
            system_prompt=self._system_prompt,
            temperature=self._temperature,
        )
        trimmed = response.strip()

        try:
            payload = self._parse_json(trimmed)
            validated_plan = self._gatekeeper.enforce(payload, raw_response=trimmed)
        except (PlannerError, PlannerContractViolation) as exc:
            logger.exception("planner_output_invalid", error=str(exc), response=trimmed)
            heuristic_plan = self._build_keyword_plan(
                prompt=original_prompt,
                agents=agents,
                raw_response=trimmed,
                reason="planner_output_invalid",
                tool_aliases=tool_aliases,
            )
            if heuristic_plan is not None:
                return heuristic_plan
            fallback_plan = self._build_fallback_plan(
                agents=agents,
                raw_response=trimmed,
                reason="Planner output could not be parsed",
            )
            if fallback_plan is not None:
                return fallback_plan
            raise PlannerError("Planner produced invalid plan") from exc

        raw_steps = [
            PlannedAgentStep(
                agent=step.agent,
                tools=list(step.tools),
                fallback_tools=list(step.fallback_tools),
                reason=step.reason or "Planner selected agent",
                confidence=step.confidence,
            )
            for step in validated_plan.steps
        ]
        if not raw_steps:
            logger.warning("planner_empty_plan", response=trimmed)
            fallback_plan = self._build_fallback_plan(
                agents=agents,
                raw_response=trimmed,
                reason="Planner returned empty step list",
            )
            if fallback_plan is not None:
                return fallback_plan

        steps, adjustments = self._post_process_steps(
            original_prompt,
            raw_steps,
            tool_aliases=tool_aliases,
        )
        if not steps:
            logger.warning("planner_post_process_empty", response=trimmed)
            fallback_plan = self._build_fallback_plan(
                agents=agents,
                raw_response=trimmed,
                reason="Planner post-processing removed all steps",
            )
            if fallback_plan is not None:
                return fallback_plan

        metadata = dict(validated_plan.metadata)
        if adjustments:
            metadata["post_processing"] = adjustments

        confidence = self._coerce_plan_confidence(
            metadata.get("confidence", validated_plan.confidence)
        )
        metadata["confidence"] = confidence
        metadata.setdefault("handoff_strategy", "sequential")
        metadata.setdefault("schema_version", "v2")

        logger.info(
            "planner_plan_ready",
            steps=len(steps),
            confidence=confidence,
            adjustments=bool(adjustments),
        )
        return PlannerPlan(steps=steps, raw_response=trimmed, metadata=metadata, confidence=confidence)

    @staticmethod
    def _coerce_plan_confidence(raw_value: Any) -> float:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(value):
            return 1.0
        return max(0.0, min(1.0, value))

    def _parse_json(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise PlannerError("Planner response did not contain JSON")
            snippet = text[start : end + 1]
            return json.loads(snippet)

    def _build_prompt(
        self,
        *,
        task: Mapping[str, Any],
        prior_outputs: Sequence[Mapping[str, Any]],
        agents: Sequence[BaseAgent],
        tool_aliases: Mapping[str, str] | None,
        capabilities: Mapping[str, Mapping[str, Any]] | None,
    ) -> str:
        prompt = str(task.get("prompt") or "").strip()
        metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
        prior_summary = self._summarize_prior_outputs(prior_outputs)
        agent_catalog = self._summarize_agents(agents)
        agent_metadata = self._summarize_agent_metadata(agents)
        alias_catalog = self._summarize_aliases(tool_aliases)
        capability_graph = self._summarize_capabilities(capabilities)
        meta_section = json.dumps(metadata, indent=2, sort_keys=True) if metadata else "{}"
        instructions = (
            "Respond with a single JSON object matching this structure (do not include any text before or after it):\n"
            "{\n"
            "  \"steps\": [\n"
            "    {\"agent\": \"general_agent\", \"reason\": \"triage request\", \"tools\": [], \"fallback_tools\": [], \"confidence\": 0.85}\n"
            "  ],\n"
            "  \"metadata\": {\"handoff_strategy\": \"sequential\", \"notes\": \"short planner note\"},\n"
            "  \"confidence\": 0.85\n"
            "}\n"
            "Rules: Use only agents and tools listed. Respect the per-agent tool policy: finance_agent and research_agent must "
            "include at least one tool when selected. enterprise_agent, creative_agent, and general_agent may omit tools when they "
            "are only greeting, triaging, or when no tool would materially improve the answer. Only assign tools each agent can "
            "reasonably invoke. Limit to at most 4 agents total. Use general_agent by itself only for greetings or vague prompts; when the user specifies concrete deliverables, assign the relevant specialists even if the request is multi-part. "
            "If the prompt asks for financial performance, earnings, revenue, ratios, guidance, a ticker symbol, or contains finance-focused keywords (e.g., \"financial report\", \"earnings\", \"revenue\", \"profit\", \"loss\", \"forecast\", \"Netflix\", \"NFLX\"), include finance_agent with finance tools. "
            "If the task calls for external research, comparisons, citations, or fact-finding, include research_agent with research tools. "
            "If the user asks for storytelling, copywriting, slogans, creative briefs, or tone adjustments, include creative_agent with creative tools. "
            "If the user needs business strategy, GTM planning, operational guidance, or executive-ready recommendations, include enterprise_agent with enterprise tools. "
            "Only add specialist agents when their domain expertise is clearly required. Do not output a legacy \"agents\" array — you must use the steps structure shown above. "
            "\"metadata\" must be an object with short strings. Always include \"confidence\" as a float between 0.0 and 1.0 indicating how confident you are in the selected plan. "
            "Never return multiple JSON objects or extra commentary."
        )
        sections = [
            "Task Prompt:\n" + (prompt or "(none)"),
            "Task Metadata (JSON):\n" + meta_section,
            "Available Agents:\n" + agent_catalog,
            "Agent Metadata:\n" + agent_metadata,
            "Capabilities Graph:\n" + capability_graph,
            "Tool Aliases:\n" + alias_catalog,
            "Prior Outputs:\n" + prior_summary,
            "Planning Instructions:\n" + instructions,
        ]
        return "\n\n".join(sections)

    def _summarize_prior_outputs(self, outputs: Sequence[Mapping[str, Any]]) -> str:
        if not outputs:
            return "(no prior outputs)"
        lines: list[str] = []
        for item in outputs[-5:]:
            agent = str(item.get("agent") or "agent")
            summary = str(item.get("summary") or item.get("content") or "")[:250]
            confidence = item.get("confidence")
            confidence_part = f" (confidence {confidence:.2f})" if isinstance(confidence, (int, float)) else ""
            lines.append(f"- {agent}: {summary}{confidence_part}")
        return "\n".join(lines)

    def _summarize_agents(self, agents: Sequence[BaseAgent]) -> str:
        if not agents:
            return "(no agents configured)"
        lines: list[str] = []
        for agent in agents:
            capability = agent.capability.value if isinstance(agent.capability, AgentCapability) else str(agent.capability)
            description = getattr(agent, "system_prompt", "")[:160]
            lines.append(f"- {agent.name} (capability: {capability}) :: {description}")
        return "\n".join(lines)

    def _summarize_agent_metadata(self, agents: Sequence[BaseAgent]) -> str:
        schema = get_agent_schema(list(agents)) if agents else []
        if not schema:
            return "(no agent metadata)"
        lines: list[str] = []
        for entry in schema:
            tools = entry.get("tools") or []
            tool_summary = ", ".join(sorted({tool for tool in tools if tool})) or "none"
            fallback = entry.get("fallback_agent") or "none"
            confidence_bias = entry.get("confidence_bias")
            bias_str = f"{float(confidence_bias):.2f}" if isinstance(confidence_bias, (int, float)) else "n/a"
            description = str(entry.get("description") or "")
            lines.append(
                f"- {entry.get('name', 'agent')} :: {description} | tools: {tool_summary} | "
                f"fallback: {fallback} | confidence_bias: {bias_str}"
            )
        return "\n".join(lines)

    def _summarize_aliases(self, aliases: Mapping[str, str] | None) -> str:
        if aliases:
            resolved = {str(key): str(value) for key, value in aliases.items()}
        else:
            resolved = dict(tool_registry.aliases())
        lines = [f"- {alias} -> {target}" for alias, target in sorted(resolved.items())]
        return "\n".join(lines) if lines else "(no tool aliases)"

    def _summarize_capabilities(self, capabilities: Mapping[str, Mapping[str, Any]] | None) -> str:
        if not capabilities:
            return "(no capability metadata)"
        lines: list[str] = []
        for name in sorted(capabilities):
            entry = capabilities[name]
            description = str(entry.get("description") or "").strip() or "(no description)"
            dependencies = [dep.strip() for dep in entry.get("dependencies", []) if isinstance(dep, str) and dep.strip()]
            dependency_str = f" depends on {', '.join(dependencies)}" if dependencies else ""
            lines.append(f"- {name}: {description}{dependency_str}")
        return "\n".join(lines)

    def _merge_alias_sources(
        self,
        base_aliases: Mapping[str, str] | None,
        overrides: Mapping[str, str] | None,
    ) -> dict[str, str]:
        merged: dict[str, str] = {}
        if base_aliases:
            for key, value in base_aliases.items():
                merged[str(key)] = str(value)
        if overrides:
            for key, value in overrides.items():
                merged[str(key)] = str(value)
        return merged

    async def _resolve_tool_alias_snapshot(self) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        registry_aliases = tool_registry.aliases()
        for key, value in registry_aliases.items():
            alias_map[str(key)] = str(value)
        default_aliases = _load_default_tool_aliases()
        for key, value in default_aliases.items():
            alias_map.setdefault(str(key), str(value))
        if getattr(self._settings.tools.mcp, "enabled", False):
            try:
                service = await get_tool_service()
                diagnostics = service.get_diagnostics()
                remote_aliases = diagnostics.get("aliases") if isinstance(diagnostics, Mapping) else None
                if isinstance(remote_aliases, Mapping):
                    for key, value in remote_aliases.items():
                        alias_map[str(key)] = str(value)
            except Exception as exc:  # pragma: no cover - diagnostics best effort
                logger.warning("planner_alias_snapshot_failed", error=str(exc))
        return alias_map

    def _post_process_steps(
        self,
        prompt: str,
        steps: list[PlannedAgentStep],
        *,
        tool_aliases: Mapping[str, str] | None,
    ) -> tuple[list[PlannedAgentStep], dict[str, Any]]:
        adjustments: dict[str, Any] = {}
        if not steps:
            return steps, adjustments

        allowed_tools = self._allowed_tool_names(tool_aliases)
        updated_steps = [
            PlannedAgentStep(
                agent=step.agent,
                tools=list(step.tools),
                fallback_tools=list(step.fallback_tools),
                reason=step.reason,
                confidence=step.confidence,
            )
            for step in steps
        ]

        if self._is_simple_greeting(prompt) and updated_steps:
            general_step = next((step for step in updated_steps if step.agent == "general_agent"), None)
            selected = general_step or updated_steps[0]
            removed_agents = [step.agent for step in updated_steps if step is not selected]
            if removed_agents:
                logger.info(
                    "planner_trimmed_agents",
                    reason="simple_greeting",
                    prompt=prompt.strip(),
                    selected_agent=selected.agent,
                )
            adjustments = {
                "reason": "simple_greeting",
                "selected_agent": selected.agent,
                "removed_agents": removed_agents,
            }
            updated_steps = [
                PlannedAgentStep(
                    agent=selected.agent,
                    tools=list(selected.tools),
                    fallback_tools=list(selected.fallback_tools),
                    reason=selected.reason,
                    confidence=selected.confidence,
                )
            ]

        sanitized_entries: list[dict[str, Any]] = []
        canonicalized_entries: list[dict[str, Any]] = []
        for step in updated_steps:
            original_tools = list(step.tools)
            original_fallbacks = list(step.fallback_tools)

            step.tools, tool_replacements = self._canonicalize_agent_tools(step.agent, step.tools)
            step.fallback_tools, fallback_replacements = self._canonicalize_agent_tools(step.agent, step.fallback_tools)

            if step.agent == "creative_agent" and not self._prompt_mentions_audio(prompt):
                step.tools = [tool for tool in step.tools if tool != "creative.transcribe"]
                step.fallback_tools = [tool for tool in step.fallback_tools if tool != "creative.transcribe"]

            step.tools = [tool for tool in step.tools if tool in allowed_tools]
            step.fallback_tools = [tool for tool in step.fallback_tools if tool in allowed_tools]

            policy_filtered = False
            policy = get_agent_tool_policy(step.agent)
            if policy is not None:
                filtered_tools, removed_policy_tools = policy.filter_tools(step.tools)
                filtered_fallbacks, removed_policy_fallbacks = policy.filter_tools(step.fallback_tools)
                if removed_policy_tools or removed_policy_fallbacks:
                    policy_filtered = True
                step.tools = filtered_tools
                step.fallback_tools = filtered_fallbacks

            if tool_replacements or fallback_replacements:
                replacement_payload: list[dict[str, str]] = []
                for original, updated in tool_replacements:
                    replacement_payload.append({"from": original, "to": updated, "scope": "tools"})
                for original, updated in fallback_replacements:
                    replacement_payload.append({"from": original, "to": updated, "scope": "fallback"})
                if replacement_payload:
                    canonicalized_entries.append({"agent": step.agent, "replacements": replacement_payload})

            removed_tools = sorted(set(original_tools) - set(step.tools))
            removed_fallbacks = sorted(set(original_fallbacks) - set(step.fallback_tools))

            if step.agent == "general_agent" and not self._needs_creative_agent(prompt, []):
                if step.tools or step.fallback_tools:
                    removed_tools.extend(step.tools)
                    removed_fallbacks.extend(step.fallback_tools)
                    step.tools = []
                    step.fallback_tools = []

            if removed_tools or removed_fallbacks:
                entry = {
                    "agent": step.agent,
                    "removed_tools": sorted(set(removed_tools)),
                    "removed_fallbacks": sorted(set(removed_fallbacks)),
                }
                if policy_filtered:
                    entry["policy_filtered"] = True
                sanitized_entries.append(entry)

        if sanitized_entries:
            adjustments["sanitized_tools"] = sanitized_entries
        if canonicalized_entries:
            adjustments["canonicalized_tools"] = canonicalized_entries

        added_agents: list[str] = []

        if self._needs_finance_agent(prompt, updated_steps):
            finance_step = PlannedAgentStep(
                agent="finance_agent",
                tools=["finance.snapshot"],
                fallback_tools=["finance.snapshot.alpha", "finance.snapshot.cached", "finance.news"],
                reason="financial report requested",
                confidence=1.0,
            )
            updated_steps.append(finance_step)
            added_agents.append(finance_step.agent)
            logger.info(
                "planner_added_finance_agent",
                prompt=prompt.strip(),
                agent=finance_step.agent,
                tools=finance_step.tools,
            )

        if self._needs_research_agent(prompt, updated_steps):
            research_step = PlannedAgentStep(
                agent="research_agent",
                tools=["research.search", "research.summarizer"],
                fallback_tools=["research.doc_loader"],
                reason="research support requested",
                confidence=1.0,
            )
            updated_steps.append(research_step)
            added_agents.append(research_step.agent)
            logger.info(
                "planner_added_research_agent",
                prompt=prompt.strip(),
                agent=research_step.agent,
                tools=research_step.tools,
            )

        if self._needs_creative_agent(prompt, updated_steps):
            creative_step = PlannedAgentStep(
                agent="creative_agent",
                tools=["creative.tonecheck"],
                fallback_tools=["creative.image"],
                reason="creative deliverable requested",
                confidence=1.0,
            )
            updated_steps.append(creative_step)
            added_agents.append(creative_step.agent)
            logger.info(
                "planner_added_creative_agent",
                prompt=prompt.strip(),
                agent=creative_step.agent,
                tools=creative_step.tools,
            )

        if self._needs_enterprise_agent(prompt, updated_steps):
            enterprise_step = PlannedAgentStep(
                agent="enterprise_agent",
                tools=["enterprise.playbook"],
                fallback_tools=["enterprise.policy"],
                reason="business strategy requested",
                confidence=1.0,
            )
            updated_steps.append(enterprise_step)
            added_agents.append(enterprise_step.agent)
            logger.info(
                "planner_added_enterprise_agent",
                prompt=prompt.strip(),
                agent=enterprise_step.agent,
                tools=enterprise_step.tools,
            )

        if added_agents:
            adjustments.setdefault("added_agents", added_agents)

        return updated_steps, adjustments

    def _is_simple_greeting(self, prompt: str) -> bool:
        normalized = prompt.strip().lower()
        if not normalized:
            return True
        cleaned = normalized.translate(str.maketrans("", "", string.punctuation))
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return True

        greeting_aliases = {
            "hi",
            "hello",
            "hey",
            "hi there",
            "hello there",
            "good morning",
            "good afternoon",
            "good evening",
            "greetings",
        }
        if cleaned in greeting_aliases:
            return True

        tokens = cleaned.split()
        if len(tokens) <= 3 and tokens[0] in {"hi", "hello", "hey"}:
            allowed = {"hi", "hello", "hey", "there", "team", "everyone"}
            if all(token in allowed for token in tokens):
                return True

        if len(cleaned) <= 12 and "?" not in normalized and "!" not in normalized:
            if tokens and tokens[0] in {"hi", "hello", "hey", "greetings"}:
                return True

        return False

    def _needs_finance_agent(self, prompt: str, steps: Sequence[PlannedAgentStep]) -> bool:
        if any(step.agent == "finance_agent" for step in steps):
            return False

        original = (prompt or "").strip()
        normalized = original.lower()
        if not normalized:
            return False

        finance_keywords = {
            "financial",
            "finance",
            "financial report",
            "earnings",
            "revenue",
            "net income",
            "profit",
            "loss",
            "forecast",
            "guidance",
            "quarter",
            "q1",
            "q2",
            "q3",
            "q4",
            "fy",
            "yearly",
            "annual",
            "investment",
            "invest",
            "portfolio",
            "allocation",
            "treasury",
            "var",
            "value at risk",
            "sharpe",
            "nifty",
            "banknifty",
            "netflix",
            "nflx",
        }

        rupee_symbol = "\u20b9"

        if any(keyword in normalized for keyword in finance_keywords):
            return True

        if rupee_symbol in original:
            return True

        stripped_original = original.translate(str.maketrans("", "", string.punctuation))
        raw_tokens = [token for token in stripped_original.split() if token]
        if any(token.isalpha() and token.isupper() and 1 < len(token) <= 5 for token in raw_tokens):
            return True

        return False

    def _needs_research_agent(self, prompt: str, steps: Sequence[PlannedAgentStep]) -> bool:
        if any(step.agent == "research_agent" for step in steps):
            return False

        normalized = (prompt or "").strip().lower()
        if not normalized:
            return False

        research_keywords = {
            "research",
            "analyze",
            "analysis",
            "compare",
            "comparison",
            "benchmark",
            "study",
            "report",
            "citation",
            "sources",
            "market size",
            "industry",
            "whitepaper",
            "search",
            "web",
            "news",
            "trend",
            "trends",
        }

        if any(keyword in normalized for keyword in research_keywords):
            return True

        return "search the web" in normalized

    def _needs_creative_agent(self, prompt: str, steps: Sequence[PlannedAgentStep]) -> bool:
        if any(step.agent == "creative_agent" for step in steps):
            return False

        normalized = (prompt or "").strip().lower()
        if not normalized:
            return False

        creative_keywords = {
            "story",
            "storytelling",
            "narrative",
            "tagline",
            "slogan",
            "copy",
            "copywriting",
            "campaign",
            "creative",
            "tone",
            "branding",
        }

        return any(keyword in normalized for keyword in creative_keywords)

    def _needs_enterprise_agent(self, prompt: str, steps: Sequence[PlannedAgentStep]) -> bool:
        if any(step.agent == "enterprise_agent" for step in steps):
            return False

        normalized = (prompt or "").strip().lower()
        if not normalized:
            return False

        enterprise_keywords = {
            "go-to-market",
            "go to market",
            "gtm",
            "business plan",
            "strategy",
            "strategic",
            "enterprise",
            "enterprise idea",
            "roadmap",
            "operations",
            "operational",
            "operational insight",
            "operational insights",
            "executive",
            "board",
            "stakeholder",
            "market entry",
            "org design",
            "organizational",
            "pricing",
            "sales",
            "change management",
            "transformation",
            "expansion",
            "action plan",
            "b2b",
            "enterprise sales",
            "enterprise strategy",
        }

        if any(keyword in normalized for keyword in enterprise_keywords):
            return True

        return "operational insight" in normalized or "operational insights" in normalized

    def _allowed_tool_names(self, tool_aliases: Mapping[str, str] | None) -> set[str]:
        registry_aliases = dict(tool_registry.aliases())
        combined: dict[str, str] = {str(key): str(value) for key, value in registry_aliases.items()}
        if tool_aliases:
            for key, value in tool_aliases.items():
                combined[str(key)] = str(value)

        default_aliases = _load_default_tool_aliases()
        for key, value in default_aliases.items():
            combined.setdefault(str(key), str(value))

        canonical_registry_tools = list(tool_registry.list())
        canonical_defaults = [str(value) for value in default_aliases.values()]
        canonical_pool = canonical_registry_tools or canonical_defaults

        allowed: set[str] = set()

        def _add_variants(identifier: str) -> None:
            if not identifier:
                return
            allowed.add(identifier)
            if "." in identifier:
                allowed.add(identifier.replace(".", "/"))
            if "/" in identifier:
                allowed.add(identifier.replace("/", "."))

        for canonical in canonical_pool:
            _add_variants(canonical)

        for alias, target in combined.items():
            _add_variants(alias)
            _add_variants(target)

        return allowed

    def _canonicalize_agent_tools(
        self,
        agent: str,
        tools: list[str],
    ) -> tuple[list[str], list[tuple[str, str]]]:
        replacements: list[tuple[str, str]] = []
        normalized: list[str] = []
        seen: set[str] = set()
        for tool in tools:
            canonical = tool
            if agent == "creative_agent":
                canonical = self._canonicalize_creative_tool(tool)
            if canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)
            if canonical != tool:
                replacements.append((tool, canonical))
        return normalized, replacements

    @staticmethod
    def _canonicalize_creative_tool(tool: str) -> str:
        mapping = {
            "creative.tone_checker": "creative.tonecheck",
            "creative/tone_checker": "creative.tonecheck",
            "creative.stylizer": "creative.tonecheck",
        }
        return mapping.get(tool, tool)

    def _prompt_mentions_audio(self, prompt: str) -> bool:
        normalized = (prompt or "").strip().lower()
        if not normalized:
            return False
        audio_keywords = {
            "audio",
            "transcribe",
            "transcription",
            "voice memo",
            "voice note",
            "podcast",
            "recording",
        }
        return any(keyword in normalized for keyword in audio_keywords)

    def _build_fallback_plan(
        self,
        *,
        agents: Sequence[BaseAgent],
        raw_response: str,
        reason: str,
    ) -> PlannerPlan | None:
        fallback_agent = next((agent for agent in agents if agent.name == "general_agent"), None)
        if fallback_agent is None:
            fallback_agent = agents[0] if agents else None
        if fallback_agent is None:
            return None

        fallback_step = PlannedAgentStep(
            agent=fallback_agent.name,
            tools=[],
            fallback_tools=[],
            reason=reason if len(reason) >= 3 else "fallback triage",
            confidence=0.5,
        )
        logger.warning(
            "planner_fallback_default",
            agent=fallback_agent.name,
            reason=reason,
            confidence=0.0,
        )
        metadata = {
            "handoff_strategy": "sequential",
            "notes": "Fallback single-agent plan",
            "fallback_reason": reason,
            "confidence": 0.0,
            "schema_version": "v2",
        }
        return PlannerPlan(
            steps=[fallback_step],
            raw_response=raw_response,
            metadata=metadata,
            confidence=0.0,
        )

    def _build_keyword_plan(
        self,
        *,
        prompt: str,
        agents: Sequence[BaseAgent],
        raw_response: str,
        reason: str,
        tool_aliases: Mapping[str, str] | None,
    ) -> PlannerPlan | None:
        if not agents:
            return None

        roster = {agent.name: agent for agent in agents}
        if not roster:
            return None

        steps: list[PlannedAgentStep] = []

        def agent_available(name: str) -> bool:
            return name in roster

        def add_step(name: str, *, tools: Sequence[str], fallback_tools: Sequence[str], step_reason: str, confidence: float = 0.85) -> None:
            if agent_available(name):
                steps.append(
                    PlannedAgentStep(
                        agent=name,
                        tools=list(tools),
                        fallback_tools=list(fallback_tools),
                        reason=step_reason,
                        confidence=confidence,
                    )
                )

        # Prioritize specialist agents based on keyword heuristics so complex prompts don't fall back to general_agent only.
        research_needed = self._needs_research_agent(prompt, steps)
        finance_needed = self._needs_finance_agent(prompt, steps)
        enterprise_needed = self._needs_enterprise_agent(prompt, steps)
        creative_needed = self._needs_creative_agent(prompt, steps)

        if research_needed:
            add_step(
                "research_agent",
                tools=["research.search", "research.summarizer"],
                fallback_tools=["research.doc_loader"],
                step_reason="heuristic: prompt requests external research",
            )

        if finance_needed:
            add_step(
                "finance_agent",
                tools=["finance.snapshot"],
                fallback_tools=["finance.news"],
                step_reason="heuristic: prompt contains financial analysis",
            )

        if enterprise_needed:
            add_step(
                "enterprise_agent",
                tools=["enterprise.playbook"],
                fallback_tools=["enterprise.policy"],
                step_reason="heuristic: prompt focuses on operations/strategy",
            )

        if creative_needed:
            add_step(
                "creative_agent",
                tools=["creative.tonecheck"],
                fallback_tools=["creative.image"],
                step_reason="heuristic: prompt requests creative deliverables",
            )

        if agent_available("general_agent"):
            add_step(
                "general_agent",
                tools=[],
                fallback_tools=[],
                step_reason="heuristic: orchestration opener",
                confidence=0.8,
            )

        if not steps:
            logger.warning(
                "planner_keyword_fallback_empty",
                prompt_snippet=prompt[:240],
                research_flag=research_needed,
                finance_flag=finance_needed,
                enterprise_flag=enterprise_needed,
                creative_flag=creative_needed,
            )

        if not steps:
            return None

        processed_steps, adjustments = self._post_process_steps(
            prompt,
            steps,
            tool_aliases=tool_aliases,
        )

        if not processed_steps:
            logger.warning(
                "planner_keyword_postprocess_empty",
                prompt_snippet=prompt[:240],
                initial_agents=[step.agent for step in steps],
            )
            return None

        metadata: dict[str, Any] = {
            "handoff_strategy": "sequential",
            "notes": "Heuristic fallback plan",
            "fallback_reason": reason,
            "confidence": 0.85,
            "schema_version": "v2",
        }
        if adjustments:
            metadata["post_processing"] = adjustments

        logger.warning(
            "planner_keyword_fallback",
            prompt=prompt[:256],
            agents=[step.agent for step in processed_steps],
        )

        return PlannerPlan(
            steps=processed_steps,
            raw_response=raw_response,
            metadata=metadata,
            confidence=0.85,
        )

    def _extract_confidence(self, payload: Mapping[str, Any] | None) -> float:
        if not isinstance(payload, Mapping):
            return 1.0
        raw_value = payload.get("confidence", 1.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(value):
            return 1.0
        return max(0.0, min(1.0, value))