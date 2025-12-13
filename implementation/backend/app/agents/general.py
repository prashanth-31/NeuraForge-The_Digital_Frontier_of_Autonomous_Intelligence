from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agents.base import AgentContext, ReasoningBuilder
from ..core.logging import get_logger
from ..core.metrics import observe_confidence_component
from ..schemas.agents import (
    AgentCapability,
    AgentExchange,
    AgentInput,
    AgentOutput,
    ReasoningStepType,
)
from ..services.llm import is_llm_unavailable

logger = get_logger(name=__name__)


@dataclass
class GeneralistAgent:
    name: str = "general_agent"
    capability: AgentCapability = AgentCapability.GENERAL
    system_prompt: str = (
        "You are NeuraForge's Generalist Agent. Your primary role is to greet users, handle simple questions, "
        "and route complex work to specialists. IMPORTANT: If prior exchanges show specialist agents (finance_agent, "
        "research_agent, creative_agent, enterprise_agent) have already provided comprehensive responses, DO NOT "
        "repeat their work or ask clarifying questions they've already answered. Instead, provide a brief summary "
        "or acknowledge the specialists have handled the request. Keep responses under 200 words, professional and action-oriented."
    )
    description: str = "First-responder agent that greets users, answers simple prompts, and routes work to specialists."
    tool_preference: list[str] = field(default_factory=lambda: ["research.search", "research.wikipedia"])
    tool_candidates: tuple[str, ...] = (
        # Research tools for basic information retrieval
        "research.search",            # DuckDuckGo - free
        "research.summarizer",        # Text summarization
        "research.wikipedia",         # Wikipedia - free
        # Browser tools
        "browser.open",               # HTTP fetching
        "browser.extract_text",       # HTML extraction
        # Memory tools
        "memory.store",
        "memory.retrieve",
        "memory.timeline",
    )
    fallback_agent: str | None = None
    confidence_bias: float = 0.75

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("general_agent_task", task=task.model_dump())
        
        # Initialize reasoning builder for tracking thought process
        reasoning = ReasoningBuilder(agent_name=self.name, context=context)
        
        # Step 1: Analyze the incoming request
        await reasoning.think(
            f"Received request: '{task.prompt[:100]}...' - analyzing for intent and complexity",
            step_type=ReasoningStepType.OBSERVATION,
        )
        
        # Step 2: Check for prior context
        context_section = task.context
        if context_section is None and context.context is not None:
            await reasoning.think(
                "No immediate context provided, assembling context from memory and retrieval services",
                step_type=ReasoningStepType.ANALYSIS,
            )
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()
            if context_section:
                await reasoning.add_finding(
                    claim="Retrieved relevant context from knowledge base",
                    evidence=[context_section[:200] + "..."] if len(context_section) > 200 else [context_section],
                    confidence=0.7,
                    source="context_assembler",
                )

        # Step 3: Check for prior exchanges
        if task.prior_exchanges:
            await reasoning.think(
                f"Found {len(task.prior_exchanges)} prior agent exchanges - incorporating into response",
                step_type=ReasoningStepType.ANALYSIS,
                evidence=f"Agents involved: {', '.join(set(e.agent for e in task.prior_exchanges))}",
            )

        # Step 4: Generate response
        await reasoning.think(
            "Formulating response using LLM with assembled context and history",
            step_type=ReasoningStepType.SYNTHESIS,
        )
        
        prompt = self._build_prompt(task, context_section=context_section)
        summary = await context.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.2,
        )

        # Handle LLM unavailability with fallback
        if is_llm_unavailable(summary):
            logger.warning("general_agent_llm_unavailable", task_id=task.task_id)
            summary = self._generate_fallback_response(task, context_section)
            await reasoning.note_uncertainty(
                "LLM temporarily unavailable - providing basic greeting/acknowledgment"
            )

        await context.memory.store_working_memory(task.task_id, summary)

        # Step 5: Calculate confidence
        evidence_count = len(task.prior_exchanges)
        if context_section:
            evidence_count += 1

        self_assessment = min(0.55 + 0.07 * evidence_count, 0.9)
        confidence = 0.68
        confidence_breakdown: dict[str, float] | None = None
        if context.scorer is not None:
            scoring = context.scorer.score(
                evidence_count=evidence_count,
                tool_result=None,
                self_assessment=self_assessment,
            )
            confidence = scoring.score
            confidence_breakdown = scoring.breakdown.as_dict()
            for component, value in confidence_breakdown.items():
                observe_confidence_component(agent=self.name, component=component, value=value)

        await reasoning.think(
            f"Calculated confidence score: {confidence:.2f} based on {evidence_count} evidence sources",
            step_type=ReasoningStepType.EVALUATION,
            confidence=confidence,
        )

        # Step 6: Assess if handoff is needed
        if confidence < 0.5:
            await reasoning.note_uncertainty(
                "Low confidence response - may require specialist agent review"
            )
        
        metadata: dict[str, Any] = {
            "type": "general_brief",
            "audience": task.metadata.get("audience"),
            "priority": task.metadata.get("priority"),
        }
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=summary,
            confidence=confidence,
            rationale="Initial triage summary prepared without tool usage.",
            metadata=metadata,
            # Include reasoning transparency
            reasoning_steps=reasoning.steps,
            key_findings=reasoning.findings,
            tools_considered=reasoning.tools_considered,
            uncertainties=reasoning.uncertainties,
        )

    def _generate_fallback_response(
        self,
        task: AgentInput,
        context_section: str | None,
    ) -> str:
        """Generate a simple fallback response when LLM is unavailable."""
        prompt = (task.prompt or "").lower().strip()
        
        # Handle common greetings
        if any(g in prompt for g in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm NeuraForge's assistant. The language model is currently restarting. Please try your request again in a moment."
        
        # Handle meta questions
        if any(m in prompt for m in ["what can you do", "capabilities", "help"]):
            return (
                "I'm NeuraForge's multi-agent assistant. I can help with:\n"
                "- **Financial analysis** (investments, stocks, budgeting)\n"
                "- **Research** (web searches, information gathering)\n"
                "- **Enterprise content** (proposals, strategies)\n"
                "- **Creative writing** (poems, taglines, stories)\n\n"
                "*Note: The language model is temporarily unavailable. Please try again shortly.*"
            )
        
        # If specialists already responded
        specialist_agents = {"finance_agent", "research_agent", "creative_agent", "enterprise_agent"}
        specialist_responses = [ex for ex in task.prior_exchanges if ex.agent in specialist_agents]
        if specialist_responses:
            agent_names = ", ".join(set(ex.agent.replace("_agent", "") for ex in specialist_responses))
            return f"Your request has been processed by our specialist agents ({agent_names}). Their responses are shown above."
        
        # Generic fallback
        return (
            "I've received your request. The language model is temporarily restarting.\n\n"
            "Please try again in a few moments, or I can route your request to a specialist agent:\n"
            "- Ask about **finances** for financial analysis\n"
            "- Ask to **research** a topic for web searches\n"
            "- Ask for a **proposal** for business documents"
        )

    def _build_prompt(self, task: AgentInput, *, context_section: str | None = None) -> str:
        metadata_repr = self._format_metadata(task.metadata)
        history = _format_history(task.prior_exchanges)
        retrieved = context_section or "(no retrieved context)"
        
        # Check if specialists have already handled this
        specialist_agents = {"finance_agent", "research_agent", "creative_agent", "enterprise_agent"}
        specialist_responses = [ex for ex in task.prior_exchanges if ex.agent in specialist_agents]
        
        if specialist_responses:
            # Specialists have already responded - don't ask for clarification
            return (
                f"User prompt:\n{task.prompt}\n\n"
                f"Metadata:\n{metadata_repr}\n\n"
                f"Prior specialist responses:\n{history}\n\n"
                f"Retrieved context:\n{retrieved}\n\n"
                "IMPORTANT: Specialist agents have already provided comprehensive responses above. "
                "Do NOT ask clarifying questions or repeat their work. Either provide a brief acknowledgment "
                "that the request has been handled, or synthesize key points if helpful. Keep it very brief."
            )
        else:
            return (
                f"User prompt:\n{task.prompt}\n\n"
                f"Metadata:\n{metadata_repr}\n\n"
                f"Prior agent outputs:\n{history}\n\n"
                f"Retrieved context:\n{retrieved}\n\n"
                "Answer directly when possible, flag missing info, and suggest the next specialist step if needed."
            )

    @staticmethod
    def _format_metadata(metadata: dict[str, Any]) -> str:
        if not metadata:
            return "(none)"
        lines = [f"- {key}: {value}" for key, value in metadata.items()]
        return "\n".join(lines)


def _format_history(outputs: list[AgentExchange]) -> str:
    if not outputs:
        return "(none)"
    lines = []
    for item in outputs:
        snippet = item.content or "(no content)"
        suffix = f" (confidence {item.confidence:.2f})" if item.confidence is not None else ""
        lines.append(f"- {item.agent}: {snippet}{suffix}")
    return "\n".join(lines)


def _serialize_agent_input(task: AgentInput) -> dict[str, Any]:
    return {
        "id": task.task_id,
        "prompt": task.prompt,
        "metadata": task.metadata,
        "outputs": [
            {"agent": exchange.agent, "content": exchange.content, "confidence": exchange.confidence}
            for exchange in task.prior_exchanges
        ],
    }
