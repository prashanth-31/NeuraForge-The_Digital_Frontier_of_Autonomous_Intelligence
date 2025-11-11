from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field, ValidationError

from ..agents.base import BaseAgent
from ..core.config import Settings
from ..core.logging import get_logger
from ..schemas.agents import AgentCapability
from ..services.llm import LLMService
from ..services.tools import DEFAULT_TOOL_ALIASES

logger = get_logger(name=__name__)


class PlannerError(RuntimeError):
    """Raised when the orchestration planner cannot produce a valid plan."""


class PlannerAgentSpec(BaseModel):
    agent: str = Field(..., min_length=1, description="Agent identifier to execute.")
    tools: list[str] = Field(default_factory=list, description="Ordered list of tool aliases to attempt.")
    reason: str = Field(..., min_length=3, description="Rationale for selecting the agent.")
    fallback_tools: list[str] = Field(default_factory=list, description="Fallback tool aliases if primary list fails.")


class PlannerPlanModel(BaseModel):
    agents: list[PlannerAgentSpec] = Field(default_factory=list, description="Ordered agent execution plan.")
    handoff_strategy: str | None = Field(default="sequential", description="How agents exchange context.")
    notes: str | None = Field(default=None, description="Additional planner comments.")


@dataclass(slots=True)
class PlannedAgentStep:
    agent: str
    tools: list[str] = field(default_factory=list)
    fallback_tools: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class PlannerExecutionPlan:
    steps: list[PlannedAgentStep]
    raw_response: str
    metadata: dict[str, Any] = field(default_factory=dict)


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

    async def plan(
        self,
        *,
        task: Mapping[str, Any],
        prior_outputs: Sequence[Mapping[str, Any]],
        agents: Sequence[BaseAgent],
        tool_aliases: Mapping[str, str] | None = None,
    ) -> PlannerExecutionPlan:
        original_prompt = str(task.get("prompt") or "")
        prompt = self._build_prompt(task=task, prior_outputs=prior_outputs, agents=agents, tool_aliases=tool_aliases)
        logger.debug("planner_prompt", prompt=prompt)
        response = await self._llm.generate(
            prompt=prompt,
            system_prompt=self._system_prompt,
            temperature=self._temperature,
        )
        trimmed = response.strip()
        try:
            payload = self._parse_json(trimmed)
            normalized_payload = self._normalize_plan_payload(payload)
            plan_model = PlannerPlanModel.model_validate(normalized_payload)
        except (PlannerError, ValidationError) as exc:
            logger.exception("planner_output_invalid", error=str(exc), response=trimmed)
            fallback_plan = self._build_fallback_plan(
                agents=agents,
                raw_response=trimmed,
                reason="Planner output could not be parsed",
            )
            if fallback_plan is not None:
                return fallback_plan
            raise PlannerError("Planner produced invalid plan") from exc

        raw_steps = [self._to_step(spec) for spec in plan_model.agents if spec.agent]
        if not raw_steps:
            logger.warning("planner_empty_plan", response=trimmed)
            fallback_plan = self._build_fallback_plan(
                agents=agents,
                raw_response=trimmed,
                reason="Planner returned empty agent list",
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
                reason="Planner post-processing removed all agents",
            )
            if fallback_plan is not None:
                return fallback_plan

        metadata = {
            "handoff_strategy": plan_model.handoff_strategy or "sequential",
            "notes": plan_model.notes,
        }
        if adjustments:
            metadata["post_processing"] = adjustments
        return PlannerExecutionPlan(steps=steps, raw_response=trimmed, metadata=metadata)

    def _to_step(self, spec: PlannerAgentSpec) -> PlannedAgentStep:
        tools = [alias.strip() for alias in spec.tools if alias.strip()]
        fallback = [alias.strip() for alias in spec.fallback_tools if alias.strip()]
        return PlannedAgentStep(agent=spec.agent.strip(), tools=tools, fallback_tools=fallback, reason=spec.reason.strip())

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
    ) -> str:
        prompt = str(task.get("prompt") or "").strip()
        metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
        prior_summary = self._summarize_prior_outputs(prior_outputs)
        agent_catalog = self._summarize_agents(agents)
        alias_catalog = self._summarize_aliases(tool_aliases)
        meta_section = json.dumps(metadata, indent=2, sort_keys=True) if metadata else "{}"
        instructions = (
            "Respond with a single JSON object matching this structure (do not include any text before or after it):\n"
            "{\n"
            "  \"agents\": [\n"
            "    {\"agent\": \"general_agent\", \"tools\": [], \"fallback_tools\": [], \"reason\": \"triage request\"}\n"
            "  ],\n"
            "  \"handoff_strategy\": \"sequential\",\n"
            "  \"notes\": \"short planner note\"\n"
            "}\n"
            "Rules: Use only agents and tools listed. Specialist agents must include at least one tool. "
            "general_agent may omit tools when it is only greeting or triaging; only assign tools it can reasonably invoke. "
            "Limit to at most 4 agents total. When requests are broad or introductory, begin with general_agent before adding specialists. "
            "If the prompt asks for financial performance, earnings, revenue, ratios, guidance, a ticker symbol, or contains finance-focused keywords (e.g., \"financial report\", \"earnings\", \"revenue\", \"profit\", \"loss\", \"forecast\", \"Netflix\", \"NFLX\"), include finance_agent with finance tools. "
            "If the task calls for external research, comparisons, citations, or fact-finding, include research_agent with research tools. "
            "If the user asks for storytelling, copywriting, slogans, creative briefs, or tone adjustments, include creative_agent with creative tools. "
            "If the user needs business strategy, GTM planning, operational guidance, or executive-ready recommendations, include enterprise_agent with enterprise tools. "
            "Only add specialist agents when their domain expertise is clearly required. \"handoff_strategy\" must be a short string. "
            "\"notes\" must be a concise string. Never return multiple JSON objects or extra commentary."
        )
        sections = [
            "Task Prompt:\n" + (prompt or "(none)"),
            "Task Metadata (JSON):\n" + meta_section,
            "Available Agents:\n" + agent_catalog,
            "Tool Aliases:\n" + alias_catalog,
            "Prior Outputs:\n" + prior_summary,
            "Planning Instructions:\n" + instructions,
        ]
        return "\n\n".join(sections)

    def _normalize_plan_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        handoff = normalized.get("handoff_strategy")
        if handoff is not None and not isinstance(handoff, str):
            normalized["handoff_strategy"] = self._stringify_field(handoff)

        notes = normalized.get("notes")
        if notes is not None and not isinstance(notes, str):
            if isinstance(notes, Sequence) and not isinstance(notes, str):
                joined = "; ".join(str(item).strip() for item in notes if str(item).strip())
                normalized["notes"] = joined or self._stringify_field(notes)
            else:
                normalized["notes"] = self._stringify_field(notes)

        agents = normalized.get("agents")
        if isinstance(agents, Sequence):
            normalized_agents: list[dict[str, Any]] = []
            for spec in agents:
                if isinstance(spec, Mapping):
                    normalized_agents.append(self._normalize_agent_spec(spec))
            normalized["agents"] = normalized_agents
        else:
            normalized["agents"] = []

        return normalized

    def _normalize_agent_spec(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        agent_dict = dict(spec)
        agent_dict["agent"] = str(agent_dict.get("agent", "")).strip()
        reason_value = agent_dict.get("reason", "")
        agent_dict["reason"] = self._stringify_field(reason_value).strip()
        agent_dict["tools"] = self._normalize_tool_list(agent_dict.get("tools"))
        agent_dict["fallback_tools"] = self._normalize_tool_list(agent_dict.get("fallback_tools"))
        return agent_dict

    def _normalize_tool_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, Sequence):
            candidates = list(value)
        else:
            return []
        normalized: list[str] = []
        for item in candidates:
            if not isinstance(item, str):
                item = str(item)
            cleaned = item.strip()
            if cleaned:
                normalized.append(cleaned)
        return normalized

    @staticmethod
    def _stringify_field(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) or isinstance(value, Sequence):
            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(value)
        return str(value)

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

    def _summarize_aliases(self, aliases: Mapping[str, str] | None) -> str:
        if not aliases:
            resolved = DEFAULT_TOOL_ALIASES
        else:
            resolved = {**DEFAULT_TOOL_ALIASES, **dict(aliases)}
        lines = [f"- {alias} -> {target}" for alias, target in sorted(resolved.items())]
        return "\n".join(lines) if lines else "(no tool aliases)"

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

        if self._is_simple_greeting(prompt):
            general_step = next((step for step in steps if step.agent == "general_agent"), None)
            selected = general_step or steps[0]
            if len(steps) > 1 or selected is not steps[0]:
                logger.info(
                    "planner_trimmed_agents",
                    reason="simple_greeting",
                    prompt=prompt.strip(),
                    selected_agent=selected.agent,
                )
            adjustments = {
                "reason": "simple_greeting",
                "selected_agent": selected.agent,
                "removed_agents": [step.agent for step in steps if step.agent != selected.agent],
            }
            clone = PlannedAgentStep(
                agent=selected.agent,
                tools=list(selected.tools),
                fallback_tools=list(selected.fallback_tools),
                reason=selected.reason,
            )
            return [clone], adjustments

        allowed_tools = self._allowed_tool_names(tool_aliases)
        updated_steps = [
            PlannedAgentStep(
                agent=step.agent,
                tools=list(step.tools),
                fallback_tools=list(step.fallback_tools),
                reason=step.reason,
            )
            for step in steps
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
                sanitized_entries.append(
                    {
                        "agent": step.agent,
                        "removed_tools": sorted(set(removed_tools)),
                        "removed_fallbacks": sorted(set(removed_fallbacks)),
                    }
                )

        if sanitized_entries:
            adjustments["sanitized_tools"] = sanitized_entries
        if canonicalized_entries:
            adjustments["canonicalized_tools"] = canonicalized_entries

        added_agents: list[str] = []

        if self._needs_finance_agent(prompt, updated_steps):
            finance_step = PlannedAgentStep(
                agent="finance_agent",
                tools=["finance.snapshot"],
                fallback_tools=["finance.news"],
                reason="financial report requested",
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
            "netflix",
            "nflx",
        }

        if any(keyword in normalized for keyword in finance_keywords):
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
        }

        return any(keyword in normalized for keyword in research_keywords)

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
            "gtm",
            "business plan",
            "strategy",
            "roadmap",
            "operations",
            "executive",
            "board",
            "stakeholder",
            "market entry",
            "pricing",
            "sales",
        }

        return any(keyword in normalized for keyword in enterprise_keywords)

    def _allowed_tool_names(self, tool_aliases: Mapping[str, str] | None) -> set[str]:
        combined = dict(DEFAULT_TOOL_ALIASES)
        if tool_aliases:
            for key, value in tool_aliases.items():
                combined[str(key)] = str(value)
        allowed = set(combined.keys()) | set(combined.values())
        allowed.update(alias.replace(".", "/") for alias in combined.keys())
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
    ) -> PlannerExecutionPlan | None:
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
        )
        logger.warning(
            "planner_fallback_default",
            agent=fallback_agent.name,
            reason=reason,
        )
        metadata = {
            "handoff_strategy": "sequential",
            "notes": "Fallback single-agent plan",
            "fallback_reason": reason,
        }
        return PlannerExecutionPlan(steps=[fallback_step], raw_response=raw_response, metadata=metadata)