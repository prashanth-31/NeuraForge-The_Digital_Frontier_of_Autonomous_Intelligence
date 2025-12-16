from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agents.base import AgentContext, ReasoningBuilder
from ..core.logging import get_logger
from ..core.metrics import observe_confidence_component
from ..schemas.agents import AgentCapability, AgentExchange, AgentInput, AgentOutput, ReasoningStepType
from ..services.llm import is_llm_unavailable
from ..services.tools import ToolDisabledError, ToolInvocationError, ToolInvocationResult

logger = get_logger(name=__name__)


@dataclass
class CreativeAgent:
    name: str = "creative_agent"
    capability: AgentCapability = AgentCapability.CREATIVE
    system_prompt: str = (
        "You are NeuraForge's Creative Agent. Your role is to create engaging, creative content. "
        "When given financial data or technical information, transform it into easy-to-understand, "
        "creative explanations using metaphors, analogies, and vivid language. "
        "Focus on making complex topics accessible and memorable. "
        "Always provide creative, engaging content - never refuse or pass on a task."
    )
    description: str = "Creates expressive, poetic, or stylistic content tailored to the brief."
    tool_preference: list[str] = field(default_factory=lambda: ["creative.tonecheck", "creative.writer", "creative.stylizer"])
    tool_candidates: tuple[str, ...] = (
        # Creative tools - PRIMARY for writing
        "creative.writer",            # PRIMARY - Content generation (poems, stories, marketing)
        "creative.tonecheck",         # Tone analysis and content writing
        "creative.tone_checker",      # Tone evaluation
        "creative.stylizer",          # Prompt styling
        "creative.brainstorm",        # Idea generation
        "creative.transcribe",        # Audio transcription
        "creative.image",             # Image generation (placeholder)
        # Research for inspiration
        "research.search",            # DuckDuckGo for references
        "research.wikipedia",         # Background info
        # Browser for reference material
        "browser.open",               # HTTP fetching
        "browser.extract_text",       # HTML extraction
        # Memory
        "memory.store",
        "memory.retrieve",
    )
    fallback_tools: list[str] = field(default_factory=lambda: ["creative.image"])
    fallback_agent: str | None = "general_agent"
    confidence_bias: float = 0.8

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("creative_agent_task", task=task.model_dump())
        
        # Initialize reasoning builder for tracking thought process
        reasoning = ReasoningBuilder(agent_name=self.name, context=context)
        
        # Check if this task is actually relevant to creative agent
        if not self._is_relevant_task(task):
            logger.info("creative_agent_pass", reason="not_a_creative_task", prompt=task.prompt[:100] if task.prompt else "")
            await reasoning.think(
                "Analyzed request - not a creative writing task (poem, story, tagline), passing",
                step_type=ReasoningStepType.EVALUATION,
            )
            return AgentOutput(
                agent=self.name,
                capability=self.capability,
                summary="[PASS]",  # Non-empty placeholder to pass validation
                confidence=0.0,  # Zero confidence signals to skip this response
                rationale="Task not relevant to creative agent - passing.",
                metadata={"passed": True, "reason": "not_a_creative_task"},
            )
        
        # Step 1: Analyze the incoming request
        await reasoning.think(
            f"Received creative writing request: '{task.prompt[:100]}...'",
            step_type=ReasoningStepType.OBSERVATION,
        )
        
        # Step 2: Build context
        context_section = task.context
        if context_section is None and context.context is not None:
            await reasoning.think(
                "Gathering creative inspiration from knowledge base",
                step_type=ReasoningStepType.ANALYSIS,
            )
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()
            if context_section:
                await reasoning.add_finding(
                    claim="Retrieved creative context for inspiration",
                    confidence=0.6,
                    source="context_assembler",
                )
        
        # Step 3: Consider and use tools
        tool_result = await self._maybe_invoke_tool(task, context=context)
        if tool_result is not None:
            await reasoning.consider_tool(
                tool_name=tool_result.tool,
                reason="Creative enhancement tool",
                selected=True,
            )
            await reasoning.add_finding(
                claim=f"Tool '{tool_result.tool}' provided creative enhancement",
                confidence=0.75,
                source=tool_result.tool,
            )
        
        # Step 4: Generate creative output
        prompt = self._build_prompt(task, context_section=context_section, tool_result=tool_result)
        
        await reasoning.think(
            "Crafting creative content with expressive language and style",
            step_type=ReasoningStepType.SYNTHESIS,
        )
        
        creative_output = await context.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.6,
        )

        # Handle LLM unavailability with fallback
        if is_llm_unavailable(creative_output):
            logger.warning("creative_agent_llm_unavailable", task_id=task.task_id)
            creative_output = self._generate_fallback_response(task, tool_result, context_section)
            await reasoning.note_uncertainty(
                "LLM temporarily unavailable - providing placeholder creative content"
            )

        # Step 5: Calculate confidence
        evidence_count = len(task.prior_exchanges)
        if context_section:
            evidence_count += 1
        if tool_result is not None:
            evidence_count += 1

        base_self_assessment = 0.65 if tool_result is None else 0.75
        self_assessment = min(base_self_assessment + 0.05 * evidence_count, 0.95)
        confidence = 0.7
        confidence_breakdown: dict[str, float] | None = None
        if context.scorer is not None:
            scoring = context.scorer.score(
                evidence_count=evidence_count,
                tool_result=tool_result,
                self_assessment=self_assessment,
            )
            confidence = scoring.score
            confidence_breakdown = scoring.breakdown.as_dict()
            for component, value in confidence_breakdown.items():
                observe_confidence_component(agent=self.name, component=component, value=value)

        await reasoning.think(
            f"Evaluated creative output - confidence: {confidence:.2f}",
            step_type=ReasoningStepType.EVALUATION,
            confidence=confidence,
        )

        # Step 6: Note uncertainties
        if confidence < 0.6:
            await reasoning.note_uncertainty(
                "Creative output may not fully match intended tone or style"
            )

        metadata: dict[str, Any] = {
            "type": "creative_direction",
            "audience": task.metadata.get("audience"),
        }
        if tool_result is not None:
            metadata["tool"] = self._tool_metadata(tool_result)
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=creative_output,
            confidence=confidence,
            rationale="Creative draft aligned with requested tone.",
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
        tool_result: ToolInvocationResult | None,
        context_section: str | None,
    ) -> str:
        """Generate a template-based fallback response when LLM is unavailable."""
        prompt = (task.prompt or "").lower()
        response_parts = []
        
        # Detect creative type
        if "poem" in prompt or "poetry" in prompt:
            response_parts.append("## Creative Response (Template Poem)\n")
            response_parts.append("*Roses are red,*")
            response_parts.append("*Violets are blue,*")
            response_parts.append("*Our creative engine is recharging,*")
            response_parts.append("*But we'll be back for you!*\n")
        elif "tagline" in prompt or "slogan" in prompt:
            response_parts.append("## Creative Response (Template Taglines)\n")
            response_parts.append("Here are some placeholder taglines:\n")
            response_parts.append("1. \"Innovation Meets Imagination\"")
            response_parts.append("2. \"Where Ideas Come to Life\"")
            response_parts.append("3. \"Creating Tomorrow, Today\"")
            response_parts.append("4. \"Excellence in Every Expression\"\n")
        elif "story" in prompt:
            response_parts.append("## Creative Response (Story Framework)\n")
            response_parts.append("**Beginning**: Set the scene and introduce characters")
            response_parts.append("**Middle**: Present the conflict or challenge")
            response_parts.append("**Climax**: The turning point of the story")
            response_parts.append("**End**: Resolution and conclusion\n")
        else:
            response_parts.append("## Creative Response (Placeholder)\n")
            response_parts.append("Your creative request has been received.\n")
            response_parts.append("Key elements to consider:")
            response_parts.append("- Audience and tone")
            response_parts.append("- Core message")
            response_parts.append("- Emotional impact")
            response_parts.append("- Call to action\n")
        
        if tool_result and tool_result.response:
            response_parts.append("### Tool Enhancement Available\n")
            response_parts.append(f"Tool: {tool_result.tool}")
        
        if context_section:
            response_parts.append(f"\n### Context:\n{context_section[:300]}...")
        
        response_parts.append("\n---\n*Note: This is a placeholder as the LLM is temporarily unavailable. For full creative content, please try again later.*")
        
        return "\n".join(response_parts)
    
    def _is_relevant_task(self, task: AgentInput) -> bool:
        """Check if this task requires creative agent's expertise."""
        prompt = (task.prompt or "").lower()
        
        # ═══════════════════════════════════════════════════════════════════════
        # CHECK FOR EXPLICIT CREATIVE WRITING REQUESTS FIRST
        # These should ALWAYS be handled by creative agent
        # ═══════════════════════════════════════════════════════════════════════
        explicit_creative_requests = {
            "write something creative", "something creative", "sounds like",
            "written by a human", "thoughtful human", "not an ai",
            "human touch", "more human", "less robotic", "less ai",
            "make it sound", "rewrite this", "creative writing",
        }
        if any(phrase in prompt for phrase in explicit_creative_requests):
            return True
        
        # ═══════════════════════════════════════════════════════════════════════
        # QUICK REJECTIONS - Things creative agent should NEVER handle
        # These should go to research_agent instead
        # ═══════════════════════════════════════════════════════════════════════
        factual_indicators = {
            "who was", "who is", "what was", "what is the",
            "history of", "biography", "tell me about",
            "when was", "where is", "where was", "define",
            "explain what", "describe the", "give sources",
            "with sources", "with citations", "cite sources",
        }
        if any(indicator in prompt for indicator in factual_indicators):
            # Check if there's also a creative modifier - if not, reject
            creative_modifiers = {"poem", "story", "song", "creative", "fun", "simple way", "creatively"}
            if not any(mod in prompt for mod in creative_modifiers):
                return False
        
        # Check if planner explicitly selected this agent - respect the planner's decision
        # BUT only for creative tasks, not factual queries
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        shared_context = metadata.get("_shared_context", {})
        planner = shared_context.get("planner", {})
        selected_agents = planner.get("selected_agents", [])
        if self.name in selected_agents:
            return True  # Planner explicitly selected us
        
        # Check planner_steps for explicit selection
        planner_steps = shared_context.get("planner_steps", [])
        for step in planner_steps:
            if step.get("executed_agent") == self.name or step.get("planned_agent") == self.name:
                return True
        
        # Explicit creative triggers
        creative_triggers = {
            "poem", "poetry", "story", "song", "lyrics",
            "tagline", "slogan", "make it catchy", "make it fun",
            "make it creative", "creative version", "rewrite creatively",
            "write a poem", "compose", "creative brief",
            "creative way", "creatively", "fun way", "simple way",
            "explain simply", "easy to understand",
            "brainstorm", "ideas for", "blog post", "marketing copy",
        }
        
        if any(trigger in prompt for trigger in creative_triggers):
            return True
        
        # Non-creative indicators - pass if these are present without creative triggers
        non_creative_indicators = {
            "business proposal", "market analysis",
            "how are you", "hello", "hi", "greetings",
            "strategy", "roadmap", "operational",
        }
        
        if any(indicator in prompt for indicator in non_creative_indicators):
            return False
        
        return False  # Default to pass - only activate on explicit creative requests

    def _build_prompt(
        self,
        task: AgentInput,
        *,
        context_section: str | None = None,
        tool_result: ToolInvocationResult | None = None,
    ) -> str:
        audience = task.metadata.get("audience", "a general audience")
        prompt_lower = (task.prompt or "").lower()
        
        # Check if this is a pure creative fiction task (don't include prior business/finance context)
        pure_fiction_indicators = {
            "love story", "shakespeare", "write a story", "write a poem",
            "compose a", "tell a tale", "fairy tale", "fiction",
            "novel", "short story", "romance", "adventure story",
        }
        is_pure_fiction = any(ind in prompt_lower for ind in pure_fiction_indicators)
        
        # Check if this is a translation/conversion task that references previous content
        translation_indicators = {
            "translate", "convert to", "rewrite in", "in english",
            "in simple english", "modern english", "plain english",
            "the above", "above story", "previous", "that story",
        }
        is_translation = any(ind in prompt_lower for ind in translation_indicators)
        
        if is_pure_fiction:
            # For pure creative fiction, don't inject prior business/finance context
            tone_checks = self._format_tool_feedback(tool_result)
            return (
                f"Creative writing task:\n{task.prompt}\n\n"
                f"Primary audience: {audience}\n\n"
                f"Tone guidance:\n{tone_checks}\n\n"
                "Create an original, imaginative piece that captures the requested style and tone. "
                "Focus purely on creative storytelling without referencing external data or business content."
            )
        
        if is_translation and task.prior_exchanges:
            # For translation, include ONLY the most recent creative content from prior exchanges
            recent_creative = None
            for exchange in reversed(task.prior_exchanges):
                if exchange.agent == "creative_agent" and exchange.content:
                    # Skip if it looks like business content
                    content_lower = (exchange.content or "").lower()
                    if not any(biz in content_lower for biz in ["revenue", "market cap", "stock price", "subscription"]):
                        recent_creative = exchange.content
                        break
            
            if recent_creative:
                tone_checks = self._format_tool_feedback(tool_result)
                return (
                    f"Translation/Conversion task:\n{task.prompt}\n\n"
                    f"Content to translate/convert:\n{recent_creative}\n\n"
                    f"Primary audience: {audience}\n\n"
                    f"Tone guidance:\n{tone_checks}\n\n"
                    "Provide a clear, faithful translation or conversion of the content above. "
                    "Maintain the spirit and meaning while adapting to the requested style."
                )
        
        # For other creative tasks (e.g., explaining data creatively), include context
        prior = _collect_task_context(task.prior_exchanges)
        retrieved = context_section or "(no retrieved context)"
        tone_checks = self._format_tool_feedback(tool_result)
        return (
            f"Core request:\n{task.prompt}\n\n"
            f"Primary audience: {audience}\n\n"
            f"Previous agent notes:\n{prior}\n\n"
            f"Retrieved context:\n{retrieved}\n\n"
            f"Tone guidance:\n{tone_checks}\n\n"
            "Deliver a vibrant, memorable piece (<=150 words) that aligns with the strategy."
        )

    async def _maybe_invoke_tool(self, task: AgentInput, *, context: AgentContext) -> ToolInvocationResult | None:
        if context.tools is None:
            return None
        payload = {
            "prompt": task.prompt,
            "audience": task.metadata.get("audience"),
        }
        try:
            return await context.tools.invoke("creative.tonecheck", payload)
        except (ToolDisabledError, ToolInvocationError) as exc:
            logger.warning("creative_tool_failure", error=str(exc))
            return None

    @staticmethod
    def _format_tool_feedback(tool_result: ToolInvocationResult | None) -> str:
        if tool_result is None:
            return "(no tone guidance)"
        response = tool_result.response
        if isinstance(response, dict):
            suggestions = response.get("suggestions")
            if isinstance(suggestions, list):
                return "\n".join(str(item) for item in suggestions)
        return str(response)

    @staticmethod
    def _tool_metadata(tool_result: ToolInvocationResult) -> dict[str, Any]:
        return {
            "name": tool_result.tool,
            "resolved": tool_result.resolved_tool,
            "cached": tool_result.cached,
            "latency": round(tool_result.latency, 4),
        }


def _collect_task_context(outputs: list[AgentExchange]) -> str:
    if not outputs:
        return "(none)"
    return "\n".join(
        f"- {item.agent}: {item.content or '(no content)'}"
        + (f" (confidence {item.confidence:.2f})" if item.confidence is not None else "")
        for item in outputs
    )


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
