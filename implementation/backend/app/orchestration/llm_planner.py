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
        
        # ════════════════════════════════════════════════════════════════════════
        # FAST PATH: Bypass LLM for obvious patterns (greetings, simple questions)
        # This ensures correct routing without relying on the small LLM
        # ════════════════════════════════════════════════════════════════════════
        fast_path_plan = self._try_fast_path_routing(original_prompt)
        if fast_path_plan is not None:
            logger.info(
                "planner_fast_path_used",
                prompt=original_prompt[:100],
                agent=fast_path_plan.steps[0].agent if fast_path_plan.steps else "none",
            )
            return fast_path_plan
        
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
            max_tokens=self._settings.planner_llm.max_output_tokens,
        )
        trimmed = response.strip()

        try:
            payload = self._parse_json(trimmed)
            payload = self._sanitize_plan_payload(payload)
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

    def _sanitize_plan_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        # Allow more top-level keys to preserve LLM intent signals
        allowed_top_level = {"steps", "metadata", "confidence", "domains", "notes"}
        sanitized: dict[str, Any] = {}
        extra_top_level = [key for key in payload.keys() if key not in allowed_top_level]
        if extra_top_level:
            logger.info("planner_payload_extra_fields", extra_keys=sorted(extra_top_level))

        for key in allowed_top_level:
            if key in payload:
                sanitized[key] = payload[key]

        steps = payload.get("steps")
        cleaned_steps: list[dict[str, Any]] = []
        if isinstance(steps, Sequence):
            # Preserve justification field from LLM for better decision-making
            allowed_step_keys = {"agent", "reason", "tools", "fallback_tools", "confidence", "justification", "domain"}
            for index, entry in enumerate(steps):
                if not isinstance(entry, Mapping):
                    continue
                extra_fields = [field for field in entry.keys() if field not in allowed_step_keys]
                if extra_fields:
                    logger.debug(
                        "planner_step_extra_fields_preserved_in_metadata",
                        step_index=index,
                        extra_keys=sorted(extra_fields),
                    )
                step_data = {key: entry[key] for key in allowed_step_keys if key in entry}
                # Preserve extra LLM fields in step metadata for auditability
                if extra_fields:
                    step_data["_llm_extra"] = {k: entry[k] for k in extra_fields if k in entry}
                cleaned_steps.append(step_data)

        sanitized["steps"] = cleaned_steps
        if not isinstance(sanitized.get("metadata"), Mapping):
            sanitized["metadata"] = {}
        # Preserve LLM notes for post-processing decisions
        if "notes" in payload:
            sanitized["metadata"]["llm_notes"] = payload["notes"]
        return sanitized

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
        capabilities: Mapping[str, Mapping[str, Any]] | None = None,
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
            "    {\"agent\": \"research_agent\", \"reason\": \"find academic papers\", \"tools\": [\"research.arxiv\"], \"fallback_tools\": [\"research.search\"], \"confidence\": 0.95}\n"
            "  ],\n"
            "  \"metadata\": {\"handoff_strategy\": \"sequential\", \"notes\": \"academic research query\"},\n"
            "  \"confidence\": 0.92\n"
            "}\n"
            "\n"
            "╔═══════════════════════════════════════════════════════════════════════════════╗\n"
            "║           ⚠️  PRIORITY #1: CREATIVE AGENT FOR WRITING TASKS  ⚠️              ║\n"
            "╚═══════════════════════════════════════════════════════════════════════════════╝\n"
            "\n"
            "🎨 creative_agent → ALWAYS USE FOR:\n"
            "   • 'write a poem/story/song/blog post' → creative_agent + creative.tonecheck\n"
            "   • 'create a slogan/tagline/marketing copy' → creative_agent + creative.tonecheck\n"
            "   • 'brainstorm ideas' / 'help me brainstorm' → creative_agent + creative.tonecheck\n"
            "   • ANY task starting with 'write', 'create', 'compose', 'draft' + creative content\n"
            "\n"
            "⛔ NEVER use research_agent for writing tasks! 'Write a story about X' → creative_agent NOT research_agent\n"
            "\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            "                    TOOL SELECTION RULES\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            "\n"
            "research.arxiv → ONLY for: 'papers', 'academic', 'scientific', 'literature review', 'preprint', 'journal'\n"
            "research.search → For: news, current events, recent developments, general web info, 'search for'\n"
            "research.wikipedia → For: 'history of', 'what is', definitions, factual/encyclopedic info\n"
            "research.summarizer → For: 'summarize', 'key points', 'tldr', condensing text\n"
            "finance.snapshot → For: stock price, ticker, $SYMBOL, P/E ratio, earnings, market data\n"
            "creative.tonecheck → For: poems, stories, slogans, marketing copy, blog posts, brainstorming\n"
            "enterprise.playbook → For: business plan, proposal, strategy document, SWOT\n"
            "\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            "                    AGENT SELECTION RULES\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            "\n"
            "🎨 creative_agent → USE WHEN: 'write', 'poem', 'story', 'blog', 'slogan', 'tagline',\n"
            "                   'marketing', 'brainstorm', 'creative', 'compose', 'draft content'\n"
            "                   → Tool: creative.tonecheck\n"
            "\n"
            "🏠 general_agent → USE WHEN: greeting ('hi','hello'), 'what can you do', simple explanations,\n"
            "                  'explain how', unit conversions, basic calculations\n"
            "                  → NO tools needed\n"
            "\n"
            "🔬 research_agent → USE WHEN: 'research papers', 'academic', 'find information about',\n"
            "                   'news about', 'what is the history of', 'search for'\n"
            "                   → Tools: research.arxiv (papers ONLY), research.search (web), research.wikipedia\n"
            "\n"
            "💰 finance_agent → USE WHEN: 'stock price', 'ticker', '$TSLA', 'P/E ratio', 'earnings',\n"
            "                  'market data', 'financial analysis'\n"
            "                  → Tool: finance.snapshot\n"
            "\n"
            "🏢 enterprise_agent → USE WHEN: 'business proposal', 'strategy document', 'SWOT',\n"
            "                     'business plan', 'executive summary'\n"
            "                     → Tool: enterprise.playbook\n"
            "\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            "                    ⛔ NEGATIVE RULES (CRITICAL)\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            "\n"
            "⛔ research_agent FORBIDDEN FOR: writing tasks, poems, stories, slogans, brainstorming\n"
            "⛔ research.arxiv FORBIDDEN FOR: news, web search, current events, non-academic info\n"
            "⛔ finance_agent FORBIDDEN FOR: academic papers, writing tasks, explanations\n"
            "⛔ research_agent FORBIDDEN FOR: greetings, 'explain X', unit conversions, calculations\n"
            "\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            "                    MIXED INTENT PRIORITY (ACTION WORD WINS)\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            "\n"
            "The ACTION verb determines the agent:\n"
            "• 'Write about stocks' → creative_agent (ACTION=Write)\n"
            "• 'Write a story about robots' → creative_agent (ACTION=Write, NOT research)\n"
            "• 'Research papers about AI' → research_agent + research.arxiv\n"
            "• 'Get stock price of TSLA' → finance_agent + finance.snapshot\n"
            "• 'Explain how stocks work' → general_agent (ACTION=Explain)\n"
            "• 'Brainstorm app ideas' → creative_agent (ACTION=Brainstorm)\n"
            "\n"
            "IMPORTANT: Check 'ALL_AVAILABLE_TOOLS' in Agent Metadata for full tool selection!"
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
            # Show default tools and ALL available tools
            default_tools = entry.get("tools") or []
            all_tools = entry.get("all_tools") or default_tools
            
            default_summary = ", ".join(sorted({tool for tool in default_tools if tool})) or "none"
            all_tools_summary = ", ".join(sorted({tool for tool in all_tools if tool})) or "none"
            
            fallback = entry.get("fallback_agent") or "none"
            confidence_bias = entry.get("confidence_bias")
            bias_str = f"{float(confidence_bias):.2f}" if isinstance(confidence_bias, (int, float)) else "n/a"
            description = str(entry.get("description") or "")
            lines.append(
                f"- {entry.get('name', 'agent')} :: {description}\n"
                f"    default_tools: {default_summary}\n"
                f"    ALL_AVAILABLE_TOOLS: {all_tools_summary}\n"
                f"    fallback: {fallback} | confidence_bias: {bias_str}"
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
            
            # Include keywords for routing
            keywords = entry.get("keywords", [])
            keywords_str = f"\n    KEYWORDS: {', '.join(keywords)}" if keywords else ""
            
            # Include negative keywords (when NOT to use this agent)
            negative_keywords = entry.get("negative_keywords", [])
            negative_str = f"\n    DO NOT USE FOR: {', '.join(negative_keywords)}" if negative_keywords else ""
            
            # Include tool descriptions if available
            tools_info = entry.get("tools", {})
            if isinstance(tools_info, dict) and tools_info:
                tool_lines = [f"      • {tool}: {desc}" for tool, desc in tools_info.items()]
                tools_str = "\n    TOOLS:\n" + "\n".join(tool_lines)
            else:
                tools_str = ""
            
            lines.append(f"- {name}_agent: {description}{dependency_str}{keywords_str}{negative_str}{tools_str}")
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

    def _try_fast_path_routing(self, prompt: str) -> PlannerPlan | None:
        """
        Bypass the LLM for obvious routing patterns.
        Returns a PlannerPlan if fast path applies, None otherwise.
        """
        normalized = (prompt or "").strip().lower()
        cleaned = normalized.translate(str.maketrans("", "", string.punctuation))
        cleaned = " ".join(cleaned.split())
        
        if not cleaned:
            return None
        
        # ─────────────────────────────────────────────────────────────────────
        # GREETINGS → general_agent ONLY
        # ─────────────────────────────────────────────────────────────────────
        greeting_patterns = {
            "hi", "hello", "hey", "howdy", "hiya", "yo", "sup", "wassup",
            "hi there", "hello there", "hey there",
            "good morning", "good afternoon", "good evening", "good night",
            "how are you", "how are you doing", "hows it going", "how is it going",
            "whats up", "what is up", "how do you do", "how have you been",
            "nice to meet you", "pleased to meet you", "greetings",
            "how are things", "hows your day", "how is your day",
            "hi how are you", "hello how are you", "hey how are you",
            "im good", "i am good", "im fine", "i am fine", "im doing well",
            "thanks", "thank you", "thx", "ty",
        }
        
        if cleaned in greeting_patterns or any(cleaned.startswith(g + " ") for g in ["hi", "hello", "hey"]):
            return PlannerPlan(
                steps=[
                    PlannedAgentStep(
                        agent="general_agent",
                        tools=[],
                        fallback_tools=[],
                        reason="greeting detected",
                        confidence=1.0,
                    )
                ],
                raw_response="fast_path:greeting",
                metadata={"fast_path": "greeting", "handoff_strategy": "sequential"},
                confidence=1.0,
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # META QUESTIONS → general_agent ONLY
        # Questions about the system, capabilities, features, etc.
        # ─────────────────────────────────────────────────────────────────────
        meta_question_patterns = {
            "what are your capabilities", "what can you do", "what do you do",
            "what are you capable of", "what features do you have",
            "how can you help", "how can you help me", "what is your purpose",
            "what is neuraforge", "tell me about yourself", "who are you",
            "what agents are available", "what agents do you have",
            "list your capabilities", "show your capabilities",
            "help", "help me", "how does this work", "what is this",
        }
        
        if cleaned in meta_question_patterns or any(pattern in cleaned for pattern in [
            "your capabilities", "your features", "about yourself", "what you can do",
            "what can you", "what you do", "how do you work", "what are you",
        ]):
            return PlannerPlan(
                steps=[
                    PlannedAgentStep(
                        agent="general_agent",
                        tools=[],
                        fallback_tools=[],
                        reason="meta question about system capabilities",
                        confidence=1.0,
                    )
                ],
                raw_response="fast_path:meta_question",
                metadata={"fast_path": "meta_question", "handoff_strategy": "sequential"},
                confidence=1.0,
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # WIKIPEDIA / ENCYCLOPEDIA QUERIES → research_agent with research.wikipedia
        # For factual/biographical/historical queries
        # ─────────────────────────────────────────────────────────────────────
        wikipedia_indicators = [
            "who was", "who is", "what was", "what is the",
            "history of", "biography of", "tell me about",
            "give me history", "give a short biography",
            "explain what", "describe the", "when was",
            "where is", "where was", "define ",
        ]
        is_wikipedia_query = any(indicator in normalized for indicator in wikipedia_indicators)
        
        # Exclude if it looks like a creative request
        creative_exclusions = ["poem", "story", "song", "creative", "fun way", "simple way"]
        has_creative_intent = any(exc in normalized for exc in creative_exclusions)
        
        if is_wikipedia_query and not has_creative_intent:
            return PlannerPlan(
                steps=[
                    PlannedAgentStep(
                        agent="research_agent",
                        tools=["research.wikipedia"],
                        fallback_tools=["research.search"],
                        reason="factual/encyclopedic query",
                        confidence=1.0,
                    ),
                ],
                raw_response="fast_path:wikipedia",
                metadata={"fast_path": "wikipedia", "handoff_strategy": "sequential"},
                confidence=1.0,
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # CRM / OPS / DIAGNOSTIC TASKS → enterprise_agent ONLY (no tools needed)
        # These are knowledge-based strategy tasks, NOT stock ticker queries
        # ─────────────────────────────────────────────────────────────────────
        crm_ops_indicators = [
            "crm data", "lead-to-close", "conversion rate", "sales pipeline",
            "customer data", "lead data", "sales funnel", "churn rate",
            "diagnostic plan", "recovery plan", "30-day plan", "90-day plan",
            "hypotheses", "data needed", "structured diagnostic",
        ]
        is_crm_ops_query = any(indicator in normalized for indicator in crm_ops_indicators)
        
        if is_crm_ops_query:
            return PlannerPlan(
                steps=[
                    PlannedAgentStep(
                        agent="enterprise_agent",
                        tools=[],  # Strategy tasks don't need tools
                        fallback_tools=[],
                        reason="CRM/ops diagnostic - strategy task",
                        confidence=1.0,
                    ),
                ],
                raw_response="fast_path:crm_ops_diagnostic",
                metadata={"fast_path": "crm_ops_diagnostic", "handoff_strategy": "sequential", "strategy_task": True},
                confidence=1.0,
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # PROJECT PLANNING / ARCHITECTURE → enterprise_agent (no tools)
        # For software architecture, project breakdown, tech stack recommendations
        # These are knowledge-based tasks, not data retrieval tasks
        # ─────────────────────────────────────────────────────────────────────
        project_planning_indicators = [
            "break this into", "break it into", "break into tasks",
            "technical tasks", "clear tasks", "assign agents", "assign suitable",
            "tech stack", "technology stack", "suggest the stack",
            "system architecture", "software architecture", "system design",
            "project breakdown", "project tasks", "implementation plan",
            "build a system", "build an app", "develop a system",
            "ai-powered", "ai powered", "machine learning system",
            "attendance management", "management system",
        ]
        is_project_planning = any(indicator in normalized for indicator in project_planning_indicators)
        
        # Also check for combined signals
        has_build_verb = any(v in normalized for v in ["build", "develop", "create", "design"])
        has_planning_noun = any(n in normalized for n in ["system", "app", "application", "platform", "solution"])
        has_breakdown_request = any(b in normalized for b in ["break", "tasks", "steps", "phases"])
        
        if is_project_planning or (has_build_verb and has_planning_noun and has_breakdown_request):
            return PlannerPlan(
                steps=[
                    PlannedAgentStep(
                        agent="enterprise_agent",
                        tools=[],  # Project planning is knowledge-based, no tools needed
                        fallback_tools=[],
                        reason="project planning/architecture task",
                        confidence=1.0,
                    ),
                ],
                raw_response="fast_path:project_planning",
                metadata={"fast_path": "project_planning", "handoff_strategy": "sequential", "strategy_task": True},
                confidence=1.0,
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # CREATIVE WRITING → creative_agent ONLY
        # For poems, stories, creative writing, brainstorming, marketing copy, etc.
        # These are pure creative tasks that don't need data retrieval
        # ─────────────────────────────────────────────────────────────────────
        creative_writing_indicators = [
            "write something", "write me", "write a",
            "creative writing", "something creative",
            "sounds like", "written by a human", "thoughtful human", "not an ai",
            "write a poem", "write a story", "write a song", "write lyrics",
            "write a blog", "blog post", "write an essay",
            "write a slogan", "write a tagline", "marketing copy",
            "brainstorm", "help me brainstorm", "brainstorm ideas",
            "compose a", "draft a", "create a story", "create a poem",
            "storytelling", "creative piece", "creative content",
            "make it sound", "make this sound", "rewrite this",
            "human touch", "more human", "less robotic", "less ai",
        ]
        is_creative_writing = any(indicator in normalized for indicator in creative_writing_indicators)
        
        # Also check for action verb + creative intent
        has_creative_verb = any(v in cleaned.split()[:3] for v in ["write", "compose", "draft", "create", "brainstorm"])
        has_creative_noun = any(n in normalized for n in ["poem", "story", "song", "blog", "essay", "slogan", "tagline", "creative"])
        
        # Exclude if it looks like a research/data task
        research_exclusions = ["research", "papers", "academic", "scientific", "stock", "price", "financial", "analysis"]
        has_research_intent = any(exc in normalized for exc in research_exclusions)
        
        if (is_creative_writing or (has_creative_verb and has_creative_noun)) and not has_research_intent:
            return PlannerPlan(
                steps=[
                    PlannedAgentStep(
                        agent="creative_agent",
                        tools=["creative.tonecheck"],
                        fallback_tools=[],
                        reason="creative writing task",
                        confidence=1.0,
                    ),
                ],
                raw_response="fast_path:creative_writing",
                metadata={"fast_path": "creative_writing", "handoff_strategy": "sequential", "strategy_task": True},
                confidence=1.0,
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # BUSINESS PROPOSALS → enterprise_agent ONLY
        # Only for VERY SHORT, simple proposal requests (<40 chars).
        # Complex or multi-part requests go through LLM planner.
        # ─────────────────────────────────────────────────────────────────────
        
        # ─────────────────────────────────────────────────────────────────────
        # ACADEMIC PAPERS / RESEARCH → research_agent with research.arxiv
        # For queries about research papers, scientific literature, etc.
        # ─────────────────────────────────────────────────────────────────────
        academic_indicators = [
            "research paper", "research papers", "academic paper", "academic papers",
            "scientific paper", "scientific papers", "journal article", "journal articles",
            "preprint", "preprints", "arxiv", "literature review",
            "latest paper", "latest papers", "recent paper", "recent papers",
            "find papers", "search papers", "papers about", "papers on",
            "multi-ai research", "ai research paper", "ml research paper",
            "key findings of", "findings from", "what is this paper about",
        ]
        is_academic_query = any(indicator in normalized for indicator in academic_indicators)
        
        if is_academic_query:
            # Extract the search topic from the prompt
            return PlannerPlan(
                steps=[
                    PlannedAgentStep(
                        agent="research_agent",
                        tools=["research.arxiv"],
                        fallback_tools=["research.search"],
                        reason="academic/research paper query",
                        confidence=1.0,
                    ),
                    PlannedAgentStep(
                        agent="creative_agent",
                        tools=["creative.tonecheck"],
                        fallback_tools=[],
                        reason="synthesize research findings",
                        confidence=0.8,
                    ),
                ],
                raw_response="fast_path:academic_research",
                metadata={"fast_path": "academic_research", "handoff_strategy": "sequential"},
                confidence=1.0,
            )
        
        is_simple_proposal = (
            len(normalized) < 40 and  # Very short prompts only
            any(phrase in normalized for phrase in [
                "business proposal", "create a proposal", "write a proposal",
                "business plan", "pitch deck",
            ]) and
            not any(phrase in normalized for phrase in [
                "financial", "analysis", "assessment", "include", "also", "then",
                "swot", "ratio", "profitability", "liquidity", "cash flow",
                "full", "complete", "comprehensive", "detailed", "strategic",
            ])
        )
        if is_simple_proposal:
            return PlannerPlan(
                steps=[
                    PlannedAgentStep(
                        agent="enterprise_agent",
                        tools=["enterprise.playbook"],
                        fallback_tools=["enterprise.policy"],
                        reason="business proposal requested",
                        confidence=1.0,
                    )
                ],
                raw_response="fast_path:business_proposal",
                metadata={"fast_path": "business_proposal", "handoff_strategy": "sequential"},
                confidence=1.0,
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # STOCK QUOTES → finance_agent ONLY
        # Only for simple stock price lookups. Complex financial analysis
        # (ratios, assessments, etc.) should go through LLM planner.
        # ─────────────────────────────────────────────────────────────────────
        simple_stock_phrases = [
            "stock price of", "stock price for", "current price of",
            "what is the price of", "get stock price", "check stock price",
            "ticker for", "market cap of",
        ]
        # Only fast-path for VERY SHORT, simple stock lookups (<40 chars)
        is_simple_stock_query = (
            len(normalized) < 40 and
            any(phrase in normalized for phrase in simple_stock_phrases) and
            not any(phrase in normalized for phrase in [
                "assessment", "analysis", "ratio", "profitability", "liquidity",
                "strategic", "swot", "also", "then", "include", "full",
                "comprehensive", "detailed", "complete", "and",
            ])
        )
        if is_simple_stock_query:
            return PlannerPlan(
                steps=[
                    PlannedAgentStep(
                        agent="finance_agent",
                        tools=["finance.snapshot"],
                        fallback_tools=["finance.snapshot.alpha", "finance.news"],
                        reason="stock/financial data requested",
                        confidence=1.0,
                    )
                ],
                raw_response="fast_path:finance",
                metadata={"fast_path": "finance", "handoff_strategy": "sequential"},
                confidence=1.0,
            )
        
        # No fast path applies
        return None

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
        simple_greeting = self._is_simple_greeting(prompt)
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

        simple_greeting_selected = False
        if simple_greeting and updated_steps:
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
            simple_greeting_selected = True

        # ────────────────────────────────────────────────────────────────────────
        # Single-Agent Mode: ONLY for very short, clear single-domain intents
        # Complex multi-part queries should use multiple agents as planned by LLM
        # ────────────────────────────────────────────────────────────────────────
        single_agent_selected = False
        # Only enforce single-agent mode for SHORT prompts with clear single intent
        # Complex queries with multiple parts should use the LLM's multi-agent plan
        prompt_is_short = len(prompt) < 80
        prompt_is_simple = not any(word in prompt.lower() for word in [
            "also", "then", "include", "and also", "additionally", "furthermore",
            "first", "second", "third", "finally", "assessment", "full", "complete",
            "both", "multiple", "several", "comprehensive", "detailed",
        ])
        
        if not simple_greeting_selected and prompt_is_short and prompt_is_simple:
            single_agent = self._detect_single_agent_intent(prompt)
            if single_agent:
                target_step = next(
                    (s for s in updated_steps if s.agent == single_agent),
                    None
                )
                if target_step is None:
                    # Agent not in list, create a default step
                    target_step = self._create_default_step_for_agent(single_agent)
                    updated_steps = [target_step]
                else:
                    removed = [s.agent for s in updated_steps if s.agent != single_agent]
                    updated_steps = [target_step]
                    if removed:
                        adjustments["single_agent_mode"] = {
                            "selected": single_agent,
                            "removed": removed,
                            "reason": "clear_single_intent",
                        }
                        logger.info(
                            "planner_single_agent_mode",
                            prompt=prompt[:100],
                            selected=single_agent,
                            removed=removed,
                        )
                single_agent_selected = True

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

        if simple_greeting_selected:
            return updated_steps, adjustments

        if self._needs_finance_agent(prompt, updated_steps):
            finance_step = PlannedAgentStep(
                agent="finance_agent",
                tools=["finance.snapshot"],
                fallback_tools=["finance.snapshot.alpha", "finance.snapshot.cached", "finance.news"],
                reason="financial report requested",
                confidence=1.0,
            )
            updated_steps.insert(0, finance_step)
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
            general_index = next((idx for idx, step in enumerate(updated_steps) if step.agent == "general_agent"), len(updated_steps))
            updated_steps.insert(general_index, creative_step)
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
            general_index = next((idx for idx, step in enumerate(updated_steps) if step.agent == "general_agent"), len(updated_steps))
            updated_steps.insert(general_index, enterprise_step)
            added_agents.append(enterprise_step.agent)
            logger.info(
                "planner_added_enterprise_agent",
                prompt=prompt.strip(),
                agent=enterprise_step.agent,
                tools=enterprise_step.tools,
            )

        # Build priority order - research comes first as it provides foundational data
        priority_order: list[str] = []
        
        # For research-focused queries, research_agent should run FIRST
        if self._prompt_mentions_research(prompt):
            priority_order.append("research_agent")
        
        # Then specialized domain agents
        if self._prompt_mentions_finance(prompt):
            priority_order.append("finance_agent")
        if self._prompt_mentions_enterprise(prompt):
            priority_order.append("enterprise_agent")
        if self._prompt_mentions_creative(prompt):
            priority_order.append("creative_agent")

        if priority_order:
            original_order = [step.agent for step in updated_steps]
            prioritized_steps = self._reorder_by_priority(updated_steps, priority_order)
            new_order = [step.agent for step in prioritized_steps]
            if new_order != original_order:
                adjustments.setdefault(
                    "reordered_agents",
                    {
                        "priority": priority_order,
                        "original_order": original_order,
                        "updated_order": new_order,
                    },
                )
            updated_steps = prioritized_steps

        if added_agents:
            adjustments.setdefault("added_agents", added_agents)

        # ────────────────────────────────────────────────────────────────────────
        # Remove redundant general_agent when specialists can handle the task
        # This prevents duplicate "triage" responses after specialist responses
        # ────────────────────────────────────────────────────────────────────────
        specialist_agents = {"finance_agent", "research_agent", "creative_agent", "enterprise_agent"}
        has_specialist = any(step.agent in specialist_agents for step in updated_steps)
        has_general = any(step.agent == "general_agent" for step in updated_steps)
        
        if has_specialist and has_general and len(updated_steps) > 1:
            # Check if general_agent appears AFTER specialists - if so, remove it
            specialist_indices = [i for i, s in enumerate(updated_steps) if s.agent in specialist_agents]
            general_indices = [i for i, s in enumerate(updated_steps) if s.agent == "general_agent"]
            
            if specialist_indices and general_indices:
                # Remove general_agent entries that come AFTER specialists
                last_specialist = max(specialist_indices)
                trailing_general = [i for i in general_indices if i > last_specialist]
                
                if trailing_general:
                    removed_agents = []
                    updated_steps = [
                        step for i, step in enumerate(updated_steps)
                        if i not in trailing_general
                    ]
                    removed_agents = ["general_agent"] * len(trailing_general)
                    adjustments.setdefault("removed_redundant_general", {
                        "reason": "specialists_handle_task",
                        "removed_count": len(trailing_general),
                    })
                    logger.info(
                        "planner_removed_trailing_general_agent",
                        prompt=prompt[:100],
                        specialists=[s.agent for s in updated_steps if s.agent in specialist_agents],
                    )

        # ────────────────────────────────────────────────────────────────────────
        # Filter out agents with irrelevant reasons
        # The LLM sometimes selects agents with mismatched reasons. Remove them.
        # ────────────────────────────────────────────────────────────────────────
        updated_steps, irrelevant_removed = self._filter_irrelevant_agents(prompt, updated_steps)
        if irrelevant_removed:
            adjustments.setdefault("removed_irrelevant_agents", irrelevant_removed)

        # ────────────────────────────────────────────────────────────────────────
        # For business proposals/strategy docs, ALWAYS remove finance_agent
        # The _is_business_proposal_query already checks for real stock/market queries
        # Terms like "revenue model", "growth strategy" are NOT real finance tasks
        # ────────────────────────────────────────────────────────────────────────
        if self._is_business_proposal_query(prompt):
            # Remove finance_agent for pure business strategy/proposal tasks
            proposal_agents = [s for s in updated_steps if s.agent in {"enterprise_agent", "creative_agent", "general_agent"}]
            removed_agents = [s for s in updated_steps if s.agent not in {"enterprise_agent", "creative_agent", "general_agent"}]
            if proposal_agents and removed_agents:
                updated_steps = proposal_agents
                adjustments.setdefault("proposal_focused", {
                    "removed_agents": [s.agent for s in removed_agents],
                    "reason": "business_proposal_task",
                })
                logger.info(
                    "planner_removed_agents_for_proposal",
                    prompt=prompt[:100],
                    removed=[s.agent for s in removed_agents],
                )

        # ────────────────────────────────────────────────────────────────────────
        # For simple financial queries, keep only finance_agent
        # This is the final filter for focused single-domain queries
        # ────────────────────────────────────────────────────────────────────────
        if self._is_pure_finance_query(prompt):
            finance_only = [s for s in updated_steps if s.agent == "finance_agent"]
            if finance_only:
                removed_for_focus = [s.agent for s in updated_steps if s.agent != "finance_agent"]
                if removed_for_focus:
                    updated_steps = finance_only
                    adjustments.setdefault("focused_on_specialist", {
                        "domain": "finance",
                        "removed_agents": removed_for_focus,
                    })
                    logger.info(
                        "planner_focused_on_finance_only",
                        prompt=prompt[:100],
                        removed=removed_for_focus,
                    )

        # ────────────────────────────────────────────────────────────────────────
        # For pure research queries, prioritize research_agent
        # Creative can synthesize but shouldn't be the primary for research
        # ────────────────────────────────────────────────────────────────────────
        if self._is_pure_research_query(prompt):
            # Keep research_agent primary, but allow creative for synthesis if present
            research_first = [s for s in updated_steps if s.agent == "research_agent"]
            other_agents = [s for s in updated_steps if s.agent != "research_agent" and s.agent != "creative_agent"]
            creative_agents = [s for s in updated_steps if s.agent == "creative_agent"]
            
            if research_first and creative_agents:
                # Keep both but ensure research is first
                updated_steps = research_first + creative_agents
                if other_agents:
                    adjustments.setdefault("focused_on_research", {
                        "domain": "research",
                        "removed_agents": [s.agent for s in other_agents],
                    })
                    logger.info(
                        "planner_focused_on_research",
                        prompt=prompt[:100],
                        removed=[s.agent for s in other_agents],
                    )
            elif research_first:
                # Only research, remove others
                removed_for_focus = [s.agent for s in updated_steps if s.agent != "research_agent"]
                if removed_for_focus:
                    updated_steps = research_first
                    adjustments.setdefault("focused_on_specialist", {
                        "domain": "research",
                        "removed_agents": removed_for_focus,
                    })
                    logger.info(
                        "planner_focused_on_research_only",
                        prompt=prompt[:100],
                        removed=removed_for_focus,
                    )

        # ────────────────────────────────────────────────────────────────────────
        # FINAL CHECK: Only enforce single-agent for truly simple requests
        # Complex queries with multiple domains should use multiple agents
        # Uses LLM confidence to respect its multi-agent decisions
        # ────────────────────────────────────────────────────────────────────────
        if len(updated_steps) > 1:
            # Compute average confidence from LLM plan
            avg_confidence = sum((s.confidence or 0.0) for s in updated_steps) / len(updated_steps)
            
            # Check if this is genuinely a multi-agent request using domain scoring
            domain_scores = self._compute_domain_scores(prompt)
            multi_domain_detected = sum(1 for score in domain_scores.values() if score > 0.3) >= 2
            
            if self._requires_multi_agent(prompt) or multi_domain_detected:
                # Keep all agents - this is a legitimate multi-agent request
                logger.info(
                    "planner_multi_agent_approved",
                    prompt=prompt[:100],
                    agents=[s.agent for s in updated_steps],
                    domain_scores=domain_scores,
                    avg_confidence=round(avg_confidence, 2),
                )
            elif avg_confidence > 0.8:
                # Trust LLM's multi-agent decision if it's confident
                logger.info(
                    "planner_multi_agent_high_confidence",
                    prompt=prompt[:100],
                    agents=[s.agent for s in updated_steps],
                    avg_confidence=round(avg_confidence, 2),
                )
            elif len(prompt) < 50:
                # Only for VERY short prompts without multi-domain signals, reduce to single agent
                priority = ["enterprise_agent", "finance_agent", "research_agent", "creative_agent", "general_agent"]
                selected_step = None
                for agent_name in priority:
                    match = next((s for s in updated_steps if s.agent == agent_name), None)
                    if match:
                        selected_step = match
                        break
                if selected_step is None:
                    selected_step = updated_steps[0]
                
                removed = [s.agent for s in updated_steps if s.agent != selected_step.agent]
                if removed:
                    adjustments.setdefault("single_agent_enforcement", {
                        "selected": selected_step.agent,
                        "removed": removed,
                        "reason": "short_prompt_single_agent",
                        "avg_confidence": round(avg_confidence, 2),
                    })
                    logger.info(
                        "planner_enforced_single_agent",
                        prompt=prompt[:100],
                        selected=selected_step.agent,
                        removed=removed,
                        avg_confidence=round(avg_confidence, 2),
                    )
                    updated_steps = [selected_step]
            # else: keep all agents for medium-length prompts

        return updated_steps, adjustments
    
    def _compute_domain_scores(self, prompt: str) -> dict[str, float]:
        """
        Compute weighted domain scores for multi-agent detection.
        Returns scores between 0.0 and 1.0 for each domain.
        """
        normalized = (prompt or "").strip().lower()
        
        # Define keyword weights per domain
        domain_keywords = {
            "enterprise": {
                "high": ["business proposal", "strategic assessment", "corporate strategy", 
                         "swot", "business plan", "pitch deck", "executive summary"],
                "medium": ["strategy", "proposal", "recommendation", "operational", 
                           "competitive", "market opportunity", "go-to-market"],
                "low": ["business", "company", "organization", "growth"],
            },
            "finance": {
                "high": ["profitability ratio", "liquidity ratio", "efficiency ratio",
                         "financial health", "cash flow analysis", "financial assessment",
                         "stock price", "financial projection"],
                "medium": ["ratio", "revenue", "cost", "profit", "margin", "financial",
                           "burn rate", "income", "balance sheet"],
                "low": ["money", "budget", "expense", "funding"],
            },
            "research": {
                "high": ["research the", "find out about", "investigate", "compare products",
                         "current trends", "market research"],
                "medium": ["research", "compare", "trends", "study", "explore", "discover"],
                "low": ["learn", "understand", "find"],
            },
            "creative": {
                "high": ["write a poem", "write a story", "create a slogan", "make it catchy",
                         "creative writing", "artistic"],
                "medium": ["poem", "story", "creative", "catchy", "fun", "engaging"],
                "low": ["interesting", "exciting"],
            },
        }
        
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = 0.0
            for kw in keywords.get("high", []):
                if kw in normalized:
                    score += 0.4
            for kw in keywords.get("medium", []):
                if kw in normalized:
                    score += 0.15
            for kw in keywords.get("low", []):
                if kw in normalized:
                    score += 0.05
            scores[domain] = min(1.0, score)  # Cap at 1.0
        
        return scores
    
    def _requires_multi_agent(self, prompt: str) -> bool:
        """
        Detect if the prompt genuinely requires multiple agents.
        Multi-agent is appropriate for comprehensive requests spanning multiple domains.
        """
        normalized = (prompt or "").strip().lower()
        
        # Length-based heuristic: longer prompts often need multiple agents
        if len(normalized) > 200:
            return True
        
        # Explicit multi-step or multi-part requests
        multi_agent_triggers = {
            # Explicit multi-step
            "research and then write",
            "analyze and create",
            "find data and present",
            "compare multiple",
            "step 1", "step 2", "step 3",
            "first,", "then,", "finally,",
            "first ", "then ", "finally ",
            # Comprehensive analysis indicators
            "full assessment", "complete assessment", "comprehensive",
            "strategic assessment", "corporate strategy",
            "financial health check", "financial assessment",
            "full analysis", "complete analysis", "detailed analysis",
            "also include", "additionally", "furthermore",
            "as well as", "along with", "in addition to",
            # Multi-domain requests
            "profitability ratio", "liquidity ratio", "efficiency ratio",
            "swot", "competitive", "market opportunity",
            "risks", "recommendations", "roadmap",
        }
        
        if any(trigger in normalized for trigger in multi_agent_triggers):
            return True
        
        # Count domain keywords - if multiple domains mentioned, likely multi-agent
        domain_counts = {
            "enterprise": sum(1 for word in ["strategy", "proposal", "business", "swot", "operational", "recommendation"] if word in normalized),
            "finance": sum(1 for word in ["financial", "ratio", "profitability", "liquidity", "cash flow", "revenue", "cost"] if word in normalized),
            "research": sum(1 for word in ["research", "compare", "trends", "market", "competitor"] if word in normalized),
        }
        
        domains_with_hits = sum(1 for count in domain_counts.values() if count > 0)
        if domains_with_hits >= 2:
            return True
        
        return False

    def _is_simple_greeting(self, prompt: str) -> bool:
        normalized = prompt.strip().lower()
        if not normalized:
            return True
        cleaned = normalized.translate(str.maketrans("", "", string.punctuation))
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return True

        # Exact match greetings
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
            # Conversational greetings
            "how are you",
            "how are you doing",
            "hows it going",
            "how is it going",
            "whats up",
            "what is up",
            "how do you do",
            "nice to meet you",
            "pleased to meet you",
            "howdy",
            "hiya",
            "yo",
            "sup",
            "wassup",
            "hey there",
            "hello there how are you",
            "hi how are you",
            "hello how are you",
            "hey how are you",
            "how are things",
            "how have you been",
            "hows your day",
            "how is your day",
        }
        if cleaned in greeting_aliases:
            return True

        # Pattern-based greeting detection
        conversational_patterns = [
            "how are you",
            "how do you do",
            "hows it going",
            "how is it going",
            "whats up",
            "how are things",
            "how have you been",
            "nice to meet",
            "pleased to meet",
        ]
        if any(pattern in cleaned for pattern in conversational_patterns):
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

    def _select_finance_tools(self, prompt: str) -> dict[str, Any]:
        """Select appropriate finance tools based on the query type."""
        normalized = (prompt or "").lower()
        
        # Check for ticker symbols
        from app.mcp.symbols import extract_symbols_from_text
        symbols = extract_symbols_from_text(prompt)
        
        # Stock/market data queries - need real-time data
        if symbols or any(kw in normalized for kw in {"stock", "ticker", "share price", "market cap", "pe ratio", "eps"}):
            return {
                "primary": ["finance.snapshot", "finance.analytics"],
                "fallback": ["finance.news", "finance.plot"],
                "reason": "heuristic: stock/market data query - using snapshot and analytics tools",
            }
        
        # Investment strategy/advice queries - need analytics and planning
        if any(kw in normalized for kw in {"investment", "invest", "portfolio", "allocation", "diversif"}):
            return {
                "primary": ["finance.analytics", "finance.snapshot"],
                "fallback": ["finance.news"],
                "reason": "heuristic: investment strategy query - using analytics for planning",
            }
        
        # Personal finance queries (salary, budget, etc.)
        if any(kw in normalized for kw in {"salary", "income", "budget", "savings", "lpa", "lakhs", "retirement"}):
            return {
                "primary": ["finance.analytics"],
                "fallback": ["finance.snapshot"],
                "reason": "heuristic: personal finance query - using analytics for calculations",
            }
        
        # Financial analysis/ratios
        if any(kw in normalized for kw in {"ratio", "analysis", "health", "assessment", "cash flow", "balance sheet"}):
            return {
                "primary": ["finance.analytics", "finance.snapshot"],
                "fallback": ["finance.news"],
                "reason": "heuristic: financial analysis query - using analytics and snapshot",
            }
        
        # News/market trends
        if any(kw in normalized for kw in {"news", "trend", "market", "latest", "recent"}):
            return {
                "primary": ["finance.news", "finance.snapshot"],
                "fallback": ["finance.analytics"],
                "reason": "heuristic: market news/trends query - using news and snapshot",
            }
        
        # Default - general financial query
        return {
            "primary": ["finance.analytics", "finance.snapshot"],
            "fallback": ["finance.news"],
            "reason": "heuristic: general financial query - using analytics and snapshot",
        }

    def _needs_finance_agent(self, prompt: str, steps: Sequence[PlannedAgentStep]) -> bool:
        if any(step.agent == "finance_agent" for step in steps):
            return False
        return self._prompt_mentions_finance(prompt)

    def _prompt_mentions_finance(self, prompt: str) -> bool:
        if self._is_simple_greeting(prompt):
            return False
        original = (prompt or "").strip()
        normalized = original.lower()
        if not normalized:
            return False

        # ═══════════════════════════════════════════════════════════════════════
        # FINANCE TRIGGERS - Any financial topic should trigger finance_agent
        # ═══════════════════════════════════════════════════════════════════════
        finance_keywords = {
            # General finance terms
            "financial", "finance", "money", "monetary",
            # Investment
            "investment", "invest", "investing", "portfolio", "allocation",
            "diversification", "diversify", "asset allocation",
            # Personal finance
            "salary", "income", "savings", "budget", "budgeting",
            "retirement", "pension", "401k", "ira", "roth",
            "tax", "taxes", "deduction", "exemption",
            "lpa", "lakhs", "crores",  # Indian salary context
            # Market data
            "stock", "stocks", "share", "shares", "equity", "equities",
            "ticker", "market cap", "pe ratio", "eps", "revenue",
            "earnings", "dividend", "yield", "valuation",
            "bull", "bear", "market trend", "market trends",
            "stock trend", "stock trends", "trading trend",
            # Indices
            "nifty", "sensex", "dow", "nasdaq", "s&p", "banknifty",
            # Analysis
            "financial analysis", "financial health", "financial assessment",
            "profitability", "liquidity", "solvency",
            "cash flow", "burn rate", "runway",
            "balance sheet", "income statement", "profit and loss",
            "financial ratio", "debt ratio", "current ratio",
            "roa", "roe", "roce", "return on",
            # Planning
            "financial planning", "financial advice", "financial advise",
            "investment strategy", "investment advice",
            "where to invest", "how to invest", "best investment",
            "wealth", "wealth management", "wealth creation",
            # Risk
            "risk", "volatility", "hedging", "hedge",
            # Banking/Credit
            "loan", "emi", "interest rate", "credit", "debt",
            "mortgage", "insurance", "premium",
            # Crypto
            "crypto", "bitcoin", "ethereum", "cryptocurrency",
            # Quarters and fiscal
            "q1", "q2", "q3", "q4", "fy", "quarterly", "annual",
        }

        if any(keyword in normalized for keyword in finance_keywords):
            return True

        # Check for rupee symbol
        rupee_symbol = "\u20b9"
        if rupee_symbol in original:
            return True

        # Check for uppercase ticker symbols (2-5 chars)
        stripped_original = original.translate(str.maketrans("", "", string.punctuation))
        raw_tokens = [token for token in stripped_original.split() if token]
        if any(token.isalpha() and token.isupper() and 1 < len(token) <= 5 for token in raw_tokens):
            # Has potential ticker symbols
            return True

        return False

    def _is_pure_finance_query(self, prompt: str) -> bool:
        """
        Check if this is a pure/focused financial query that should only
        route to finance_agent without other agents.
        """
        if self._is_simple_greeting(prompt):
            return False
        normalized = (prompt or "").strip().lower()
        if not normalized:
            return False
        
        # Strong finance indicators
        finance_phrases = {
            "financial analysis",
            "stock price",
            "stock analysis",
            "earnings report",
            "financial report",
            "investment analysis",
            "portfolio analysis",
            "market cap",
            "pe ratio",
            "dividend",
            "revenue analysis",
            "profit analysis",
        }
        
        # Check for pure finance queries
        if any(phrase in normalized for phrase in finance_phrases):
            # Make sure it's NOT asking for creative/enterprise deliverables
            non_finance_indicators = {
                "proposal",
                "presentation",
                "pitch deck",
                "marketing",
                "campaign",
                "story",
                "creative",
                "strategy document",
                "playbook",
            }
            if not any(indicator in normalized for indicator in non_finance_indicators):
                return True
        
        return False

    def _is_pure_research_query(self, prompt: str) -> bool:
        """
        Check if this is a pure research query that should prioritize
        research_agent as the primary handler.
        """
        if self._is_simple_greeting(prompt):
            return False
        normalized = (prompt or "").strip().lower()
        if not normalized:
            return False
        
        # Strong research indicators
        research_phrases = {
            "research the",
            "research on",
            "what are the latest",
            "current trends",
            "latest trends",
            "analyze the trends",
            "investigate",
            "find out about",
            "look into",
            "search for information",
            "study on",
            "survey of",
            "state of the art",
            "what is happening",
        }
        
        # Check for pure research queries
        if any(phrase in normalized for phrase in research_phrases):
            # Make sure it's NOT asking for something that needs other specialists
            non_research_indicators = {
                "write a",
                "create a",
                "draft a",
                "proposal",
                "stock price",
                "ticker",
                "financial report",
                "earnings",
            }
            if not any(indicator in normalized for indicator in non_research_indicators):
                return True
        
        return False

    def _is_business_proposal_query(self, prompt: str) -> bool:
        """
        Check if this is a business proposal/strategy request that should
        NOT involve finance_agent (unless real stock/market data is needed).
        """
        if self._is_simple_greeting(prompt):
            return False
        normalized = (prompt or "").strip().lower()
        if not normalized:
            return False
        
        # Strong business proposal/strategy indicators
        proposal_phrases = {
            "business proposal",
            "create a proposal",
            "write a proposal",
            "draft a proposal",
            "business plan",
            "pitch deck",
            "investor pitch",
            "startup idea",
            "startup pitch",
            "market analysis",
            "go-to-market",
            "gtm strategy",
            "executive summary",
            "company background",
            "value proposition",
            "business model",
            "revenue model",
            "growth strategy",
            "business strategy",
            "ai business strategy",
            "business strategy agent",
            "college project",
            "hackathon",
            "target customers",
        }
        
        # Check for proposal/strategy queries
        if any(phrase in normalized for phrase in proposal_phrases):
            # Make sure it's NOT asking for REAL financial data (live stock prices, etc.)
            real_finance_indicators = {
                "stock price of",
                "current price of",
                "ticker symbol",
                "earnings report",
                "financial analysis of",
                "stock analysis of",
                "market cap of",
                "analyze the stock",
                "get stock",
                "fetch stock",
                "$tsla", "$aapl", "$googl", "$msft", "$amzn",  # Ticker symbols
            }
            if not any(indicator in normalized for indicator in real_finance_indicators):
                return True
        
        return False

    def _filter_irrelevant_agents(
        self, prompt: str, steps: list[PlannedAgentStep]
    ) -> tuple[list[PlannedAgentStep], list[dict[str, Any]]]:
        """
        Filter out agents that were selected with irrelevant reasons.
        The LLM sometimes hallucinates agent selections - this is a safety net.
        """
        removed: list[dict[str, Any]] = []
        normalized = (prompt or "").strip().lower()
        
        # Define reason-to-domain mappings
        irrelevant_reason_patterns = {
            # If the reason mentions these but prompt doesn't, remove the agent
            "business proposal": lambda p: "proposal" not in p and "business plan" not in p,
            "creative deliverable": lambda p: not self._prompt_mentions_creative(prompt),
            "marketing campaign": lambda p: "marketing" not in p and "campaign" not in p,
            "story": lambda p: "story" not in p and "narrative" not in p,
        }
        
        filtered_steps = []
        for step in steps:
            reason = (step.reason or "").lower()
            should_remove = False
            
            # Check if this agent's reason is irrelevant to the prompt
            for pattern, check_fn in irrelevant_reason_patterns.items():
                if pattern in reason and check_fn(normalized):
                    should_remove = True
                    removed.append({
                        "agent": step.agent,
                        "reason": step.reason,
                        "filter_reason": f"'{pattern}' not relevant to prompt",
                    })
                    logger.info(
                        "planner_filtered_irrelevant_agent",
                        agent=step.agent,
                        agent_reason=step.reason,
                        filter_reason=f"'{pattern}' not relevant to prompt",
                        prompt=prompt[:100],
                    )
                    break
            
            if not should_remove:
                filtered_steps.append(step)
        
        return filtered_steps, removed

    def _needs_research_agent(self, prompt: str, steps: Sequence[PlannedAgentStep]) -> bool:
        if any(step.agent == "research_agent" for step in steps):
            return False

        normalized = (prompt or "").strip().lower()
        if not normalized:
            return False

        # If finance_agent is already in the plan, don't add research for overlapping terms
        has_finance = any(step.agent == "finance_agent" for step in steps)
        if has_finance:
            # For finance queries, only add research if explicitly requested
            explicit_research_keywords = {
                "research paper",
                "academic",
                "whitepaper",
                "white paper",
                "search the web",
                "web search",
                "find sources",
                "find citations",
            }
            return any(keyword in normalized for keyword in explicit_research_keywords)

        # General research keywords (when no specialist is handling it)
        research_keywords = {
            "research",
            "compare",
            "comparison",
            "benchmark",
            "study",
            "citation",
            "sources",
            "market size",
            "industry analysis",
            "whitepaper",
            "web search",
            "search the web",
            "trend analysis",
        }

        if any(keyword in normalized for keyword in research_keywords):
            return True

        return False

    def _needs_creative_agent(self, prompt: str, steps: Sequence[PlannedAgentStep]) -> bool:
        """
        Creative agent should be added when there's an EXPLICIT request
        for creative content like poems, stories, creative rewriting, brainstorming.
        """
        if any(step.agent == "creative_agent" for step in steps):
            return False
        
        normalized = (prompt or "").strip().lower()
        
        # Trigger creative agent for explicit creative requests
        explicit_creative_triggers = {
            # Writing tasks
            "write a poem",
            "write a story",
            "write a song",
            "write a short story",
            "write me a poem",
            "write me a story",
            "write a blog",
            "write a blog post",
            "write content",
            "write creative",
            # Creation tasks
            "create a poem",
            "create a story",
            "create a slogan",
            "create a tagline",
            "create marketing",
            "create content",
            # Brainstorming
            "brainstorm",
            "brainstorming",
            "brainstorm ideas",
            "help me brainstorm",
            "generate ideas",
            "ideas for",
            # Style/tone
            "make it catchy",
            "make it fun",
            "make it creative",
            "make it poetic",
            "rewrite creatively",
            "give me a creative version",
            # Specific forms
            "write me a tagline",
            "write me a slogan",
            "write lyrics",
            "compose a poem",
            "compose a song",
            # Marketing content
            "marketing slogan",
            "marketing copy",
            "ad copy",
            "catchy slogan",
        }
        
        if any(trigger in normalized for trigger in explicit_creative_triggers):
            return True
        
        # Also check for action verbs + creative nouns
        action_verbs = {"write", "create", "compose", "draft", "make"}
        creative_nouns = {"poem", "story", "song", "slogan", "tagline", "lyrics", "blog", "content"}
        
        words = normalized.split()
        for i, word in enumerate(words):
            if word in action_verbs and i + 1 < len(words):
                # Check if next word or phrase contains creative noun
                remaining = " ".join(words[i+1:i+4])  # Look at next 3 words
                if any(noun in remaining for noun in creative_nouns):
                    return True
        
        return False

    def _prompt_mentions_creative(self, prompt: str) -> bool:
        """
        DEPRECATED - Use _needs_creative_agent instead.
        This is now very strict to prevent false triggers.
        """
        if self._is_simple_greeting(prompt):
            return False
        normalized = (prompt or "").strip().lower()
        if not normalized:
            return False

        # Only very explicit creative writing keywords
        creative_keywords = {
            "poem",
            "poetry",
            "sonnet",
            "verse",
            "lyrics",
            "lyric",
            "haiku",
            "limerick",
            "shakespeare",
            "rhyme",
        }

        return any(keyword in normalized for keyword in creative_keywords)

    def _needs_enterprise_agent(self, prompt: str, steps: Sequence[PlannedAgentStep]) -> bool:
        if any(step.agent == "enterprise_agent" for step in steps):
            return False
        return self._prompt_mentions_enterprise(prompt)

    def _prompt_mentions_enterprise(self, prompt: str) -> bool:
        if self._is_simple_greeting(prompt):
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
            "enterprise level",
            "enterprise-level",
            "enterprise grade",
            "enterprise-grade",
            "roadmap",
            "operations",
            "operational",
            "operational insight",
            "operational insights",
            "executive",
            "executive summary",
            "board",
            "board update",
            "stakeholder",
            "market entry",
            "org design",
            "organizational",
            "operating model",
            "pricing",
            "sales",
            "change management",
            "transformation",
            "expansion",
            "action plan",
            "b2b",
            "enterprise sales",
            "enterprise strategy",
            "corporate",
            "corporate strategy",
            "enterprise roadmap",
            "enterprise architecture",
            "enterprise solution",
            # Business planning (NOT personal finance - that's finance_agent)
            "business plan",
            "business case",
            "proposal",
            "pitch deck",
            # CRM and company data analysis
            "crm",
            "crm data",
            "customer relationship",
            "customer trends",
            "our company",
            "our sales",
            "our customers",
            "our data",
            "company's data",
            "company data",
            "internal data",
            "business report",
            "quarterly report",
            "sales data",
            "business analytics",
            "business intelligence",
        }

        if any(keyword in normalized for keyword in enterprise_keywords):
            return True

        return "operational insight" in normalized or "operational insights" in normalized

    def _prompt_mentions_research(self, prompt: str) -> bool:
        """Check if the prompt explicitly mentions research activities."""
        if self._is_simple_greeting(prompt):
            return False
        normalized = (prompt or "").strip().lower()
        if not normalized:
            return False

        research_keywords = {
            "research",
            "investigate",
            "investigation",
            "study",
            "analyze",
            "analysis",
            "compare",
            "comparison",
            "benchmark",
            "benchmarking",
            "explore",
            "find out",
            "look into",
            "search for",
            "survey",
            "review",
            "examine",
            "trends",
            "latest",
            "current state",
            "state of",
            "what is",
            "what are",
        }

        # Check for direct research keywords
        if any(keyword in normalized for keyword in research_keywords):
            # Exclude pure finance queries - they have their own agent
            if self._is_pure_finance_query(prompt):
                return False
            return True

        return False

    def _detect_single_agent_intent(self, prompt: str) -> str | None:
        """
        Detect if the prompt clearly maps to exactly one agent.
        Returns the agent name or None if multi-agent is needed.
        """
        normalized = (prompt or "").strip().lower()
        if not normalized:
            return None
        
        # Clear creative-only intents (poems, stories, brainstorming)
        creative_only_phrases = {
            "write a poem",
            "write a story",
            "write a short story",
            "write me a poem",
            "write me a story",
            "create a slogan",
            "create a tagline",
            "marketing slogan",
            "brainstorm ideas",
            "help me brainstorm",
            "write a blog post",
            "write lyrics",
            "compose a poem",
        }
        if any(phrase in normalized for phrase in creative_only_phrases):
            return "creative_agent"
        
        # Clear finance-only intents (stock quotes, price checks, market trends)
        finance_only_phrases = {
            "stock price of",
            "current price of",
            "what is the price of",
            "get me the price",
            "financial analysis of",
            "stock analysis of",
            "check the stock",
            "how is the stock",
            "market trends for",
            "latest market trends",
            "stock trends",
            "trading trends",
            "stock performance",
            "market performance",
            "stock outlook",
            "market outlook",
            "p/e ratio",
            "pe ratio",
            "compare the p/e",
            "compare p/e",
        }
        if any(phrase in normalized for phrase in finance_only_phrases):
            return "finance_agent"
        
        # Check for stock ticker + trends/performance pattern
        import re
        stock_trends_pattern = r'\b(trends?|performance|outlook|analysis)\b.{0,30}\b(stock|shares?|equity|[A-Z]{2,5})\b'
        reverse_pattern = r'\b([A-Z]{2,5}|stock|shares?|equity)\b.{0,30}\b(trends?|performance|outlook)\b'
        if re.search(stock_trends_pattern, normalized, re.IGNORECASE) or re.search(reverse_pattern, prompt, re.IGNORECASE):
            return "finance_agent"
        
        # Clear enterprise-only intents (business proposals, CRM, company data)
        enterprise_only_phrases = {
            "business proposal for",
            "write a business proposal",
            "create a business proposal",
            "draft a business proposal",
            "create a business plan",
            "write a business plan",
            "business report for",
            "quarterly business report",
            "quarterly report",
            "generate a report",
            "sales report",
            "crm data",
            "customer data",
            "customer trends",
            "analyze our",
            "our company",
            "our sales",
            "our customers",
            "our data",
            "company's data",
            "company data",
            "internal data",
            "business analytics",
            "operational report",
        }
        if any(phrase in normalized for phrase in enterprise_only_phrases):
            return "enterprise_agent"
        
        # Clear research-only intents
        research_only_phrases = {
            "research the latest",
            "what are the latest",
            "current trends in",
            "research about",
        }
        if any(phrase in normalized for phrase in research_only_phrases):
            return "research_agent"
        
        return None
    
    def _create_default_step_for_agent(self, agent_name: str) -> PlannedAgentStep:
        """Create a default step for a given agent."""
        defaults = {
            "finance_agent": PlannedAgentStep(
                agent="finance_agent",
                tools=["finance.snapshot"],
                fallback_tools=["finance.snapshot.alpha", "finance.news"],
                reason="financial data requested",
                confidence=1.0,
            ),
            "enterprise_agent": PlannedAgentStep(
                agent="enterprise_agent",
                tools=["enterprise.playbook"],
                fallback_tools=["enterprise.policy"],
                reason="business strategy requested",
                confidence=1.0,
            ),
            "research_agent": PlannedAgentStep(
                agent="research_agent",
                tools=["research.search", "research.summarizer"],
                fallback_tools=["research.doc_loader"],
                reason="research requested",
                confidence=1.0,
            ),
            "creative_agent": PlannedAgentStep(
                agent="creative_agent",
                tools=["creative.tonecheck"],
                fallback_tools=["creative.image"],
                reason="creative content requested",
                confidence=1.0,
            ),
            "general_agent": PlannedAgentStep(
                agent="general_agent",
                tools=[],
                fallback_tools=[],
                reason="general assistance",
                confidence=0.85,
            ),
        }
        return defaults.get(agent_name, defaults["general_agent"])

    @staticmethod
    def _reorder_by_priority(
        steps: Sequence[PlannedAgentStep],
        priority: Sequence[str],
    ) -> list[PlannedAgentStep]:
        if not steps:
            return []
        prioritized: list[PlannedAgentStep] = []
        consumed: set[int] = set()
        for agent_name in priority:
            for index, step in enumerate(steps):
                if index in consumed:
                    continue
                if step.agent == agent_name:
                    prioritized.append(step)
                    consumed.add(index)
                    break
        for index, step in enumerate(steps):
            if index not in consumed:
                prioritized.append(step)
        return prioritized

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
            "creative.writer": "creative.tonecheck",  # writer alias → tonecheck
            "creative.brainstorm": "creative.tonecheck",  # brainstorm alias → tonecheck
            "creative/writer": "creative.tonecheck",
            "creative/brainstorm": "creative.tonecheck",
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
        finance_needed = self._needs_finance_agent(prompt, steps)
        enterprise_needed = self._needs_enterprise_agent(prompt, steps)
        research_needed = self._needs_research_agent(prompt, steps)
        creative_needed = self._needs_creative_agent(prompt, steps)

        if finance_needed:
            # Determine appropriate tools based on query type
            finance_tools = self._select_finance_tools(prompt)
            add_step(
                "finance_agent",
                tools=finance_tools["primary"],
                fallback_tools=finance_tools["fallback"],
                step_reason=finance_tools["reason"],
            )

        if enterprise_needed:
            add_step(
                "enterprise_agent",
                tools=["enterprise.playbook"],
                fallback_tools=["enterprise.policy"],
                step_reason="heuristic: prompt focuses on operations/strategy",
            )

        if research_needed:
            add_step(
                "research_agent",
                tools=["research.search", "research.summarizer"],
                fallback_tools=["research.doc_loader"],
                step_reason="heuristic: prompt requests external research",
            )

        if creative_needed:
            add_step(
                "creative_agent",
                tools=["creative.tonecheck"],
                fallback_tools=["creative.image"],
                step_reason="heuristic: prompt requests creative deliverables",
            )

        # Only add general_agent if NO specialist was selected
        # This avoids redundant agent execution
        has_specialist = any(
            step.agent in {"finance_agent", "enterprise_agent", "research_agent", "creative_agent"}
            for step in steps
        )
        if not has_specialist and agent_available("general_agent"):
            add_step(
                "general_agent",
                tools=[],
                fallback_tools=[],
                step_reason="heuristic: no specialist matched, using general agent",
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