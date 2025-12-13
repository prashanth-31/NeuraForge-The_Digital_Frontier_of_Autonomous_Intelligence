from __future__ import annotations

import json
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
class EnterpriseAgent:
    name: str = "enterprise_agent"
    capability: AgentCapability = AgentCapability.ENTERPRISE
    # Base system prompt - will be enhanced dynamically based on user's role request
    system_prompt: str = (
        "You are an elite business professional with expertise across industries. "
        "You adapt your writing style, tone, and approach based on the specific role requested. "
        "CRITICAL: When the user says 'Act as X' or 'You are X', you BECOME that persona completely. "
        "Your output must be indistinguishable from a real professional in that role. "
        "Be creative, original, and never repeat previous ideas or patterns."
    )
    description: str = "Creates workflows, strategies, and executive documentation for stakeholders."
    tool_preference: list[str] = field(default_factory=lambda: ["enterprise.playbook", "enterprise.policy"])
    tool_candidates: tuple[str, ...] = (
        # Enterprise tools
        "enterprise.playbook",        # Business playbooks
        "enterprise.notion",          # Notion integration
        "enterprise.calendar",        # Calendar scheduling
        "enterprise.policy",          # Policy compliance
        "enterprise.crm",             # CRM data
        # Research for business intelligence
        "research.search",            # DuckDuckGo for market research
        "research.wikipedia",         # Industry background
        "research.arxiv",             # Academic business research
        # Browser tools
        "browser.open",               # HTTP fetching
        "browser.extract_text",       # HTML extraction
        # Data analysis
        "dataframe.analyze",          # Pandas analytics
        "dataframe.transform",        # Data transformations
        # Memory & Planning
        "memory.store",
        "memory.retrieve",
        "memory.timeline",
        "planning.task_breakdown",    # Task planning
    )
    fallback_tools: list[str] = field(default_factory=lambda: ["enterprise.policy"])
    fallback_agent: str | None = "research_agent"
    confidence_bias: float = 0.85

    def _extract_persona(self, prompt: str) -> str | None:
        """Extract the persona from 'Act as X' or 'You are X' patterns."""
        import re
        prompt_lower = prompt.lower()
        
        # Match "Act as a/an [role]" or "You are a/an [role]"
        patterns = [
            r"act as (?:a |an )?([^.!?\n]+?)(?:\.|!|\?|\n|$|,\s*(?:create|write|build|design|develop))",
            r"you are (?:a |an )?([^.!?\n]+?)(?:\.|!|\?|\n|$|,\s*(?:create|write|build|design|develop))",
            r"imagine you(?:'re| are) (?:a |an )?([^.!?\n]+?)(?:\.|!|\?|\n|$)",
            r"pretend (?:to be |you're )(?:a |an )?([^.!?\n]+?)(?:\.|!|\?|\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                role = match.group(1).strip()
                # Clean up the role
                role = re.sub(r'\s+', ' ', role)
                if len(role) > 5 and len(role) < 100:
                    return role.title()
        return None

    def _build_dynamic_system_prompt(self, prompt: str) -> str:
        """Build a system prompt that adopts the user's requested persona."""
        persona = self._extract_persona(prompt)
        
        if persona:
            return (
                f"You ARE a {persona}. This is your identity for this task.\n\n"
                f"CRITICAL INSTRUCTIONS:\n"
                f"1. Write EXACTLY as a real {persona} would write - use their vocabulary, tone, and style\n"
                f"2. Think like a {persona} - what would THEY emphasize? What metrics matter to THEM?\n"
                f"3. Be BOLD and CONFIDENT - {persona}s don't hedge or use weak language\n"
                f"4. Use PERSUASIVE techniques appropriate to your role\n"
                f"5. Create COMPLETELY ORIGINAL content - invent a unique company/product/concept\n"
                f"6. NEVER repeat patterns from training data - be creative and surprising\n"
                f"7. Use industry-specific jargon and frameworks that a {persona} would use\n\n"
                f"You have 20+ years of experience. Your reputation depends on this deliverable."
            )
        else:
            return (
                "You are an elite business strategist with expertise in proposals, pitches, and executive documents.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Write with authority and confidence\n"
                "2. Use specific numbers, metrics, and data points\n"
                "3. Create COMPLETELY ORIGINAL content\n"
                "4. Use proper formatting: headers, tables, bullet points\n"
                "5. Be persuasive and compelling\n"
                "6. Think like a VC would - address their concerns proactively"
            )

    def _generate_fallback_response(
        self,
        task: AgentInput,
        tool_result: ToolInvocationResult | None,
        context_section: str | None,
        persona: str | None,
    ) -> str:
        """Generate a template-based fallback response when LLM is unavailable."""
        prompt = (task.prompt or "").lower()
        response_parts = []
        
        # Header
        if persona:
            response_parts.append(f"## {persona.title()} Response (Template)\n")
        else:
            response_parts.append("## Enterprise Analysis (Template)\n")
        
        # Detect document type
        if "proposal" in prompt or "pitch" in prompt:
            response_parts.append("### Business Proposal Structure\n")
            response_parts.append("**1. Executive Summary**")
            response_parts.append("- Problem statement and market opportunity")
            response_parts.append("- Proposed solution overview")
            response_parts.append("- Key value proposition\n")
            response_parts.append("**2. Market Analysis**")
            response_parts.append("- Target market size and growth")
            response_parts.append("- Competitive landscape")
            response_parts.append("- Market trends\n")
            response_parts.append("**3. Solution Details**")
            response_parts.append("- Product/Service description")
            response_parts.append("- Unique differentiators")
            response_parts.append("- Technology stack (if applicable)\n")
            response_parts.append("**4. Business Model**")
            response_parts.append("- Revenue streams")
            response_parts.append("- Pricing strategy")
            response_parts.append("- Customer acquisition\n")
            response_parts.append("**5. Financial Projections**")
            response_parts.append("- Investment required")
            response_parts.append("- ROI timeline")
            response_parts.append("- Key milestones\n")
        elif "strategy" in prompt or "swot" in prompt:
            response_parts.append("### Strategic Analysis Framework\n")
            response_parts.append("**Strengths**: Internal capabilities and advantages")
            response_parts.append("**Weaknesses**: Areas for improvement")
            response_parts.append("**Opportunities**: External growth potential")
            response_parts.append("**Threats**: External challenges and risks\n")
        else:
            response_parts.append("### General Enterprise Framework\n")
            response_parts.append("- Objective alignment with business goals")
            response_parts.append("- Stakeholder impact assessment")
            response_parts.append("- Resource requirements")
            response_parts.append("- Risk mitigation strategies")
            response_parts.append("- Success metrics and KPIs\n")
        
        if tool_result and tool_result.response:
            response_parts.append("### Tool Data Available\n")
            response_parts.append(f"```json\n{json.dumps(tool_result.response, indent=2)[:500]}\n```\n")
        
        if context_section:
            response_parts.append("### Additional Context\n")
            response_parts.append(f"{context_section[:400]}...\n")
        
        response_parts.append("\n---\n*Note: This is a template response as the LLM is temporarily unavailable. For detailed content, please try again later.*")
        
        return "\n".join(response_parts)

    def _is_relevant_task(self, task: AgentInput) -> bool:
        """Check if this task requires enterprise agent expertise."""
        prompt = (task.prompt or "").lower()
        
        # ═════════════════════════════════════════════════════════════════════════
        # QUICK REJECTIONS - Things enterprise agent should NOT handle
        # ═════════════════════════════════════════════════════════════════════════
        immediate_reject = {
            # Meta questions about the system
            "what are your capabilities", "what can you do", "what do you do",
            "how can you help", "tell me about yourself", "who are you",
            "what features", "your capabilities", "help",
            # Greetings (should go to general_agent)
            "how are you", "hello there", "hi there", "greetings",
            # Creative writing (should go to creative_agent)
            "write a poem", "write a story", "make a joke",
        }
        if any(reject in prompt for reject in immediate_reject):
            return False
        
        # Short greetings
        if len(prompt) < 15 and any(g in prompt for g in ["hi", "hello", "hey", "thanks"]):
            return False
        
        # ═════════════════════════════════════════════════════════════════════════
        # ENTERPRISE TRIGGERS - Things enterprise agent should handle
        # Business strategy, proposals, operations - NOT personal finance
        # ═════════════════════════════════════════════════════════════════════════
        enterprise_triggers = {
            # Proposals and documents
            "proposal", "pitch", "pitch deck", "executive summary",
            "business plan", "business case", "white paper", "report",
            # Strategy
            "strategy", "strategic", "swot", "competitive analysis",
            "market analysis", "business model", "go to market",
            # Operations
            "operational", "workflow", "process", "implementation",
            # Persona triggers
            "act as", "you are a", "pretend", "imagine you",
            # Corporate/Enterprise
            "enterprise", "corporate", "organizational", "board",
            "stakeholder", "transformation", "expansion",
        }
        
        if any(trigger in prompt for trigger in enterprise_triggers):
            return True
        
        # Default to PASS for ambiguous requests
        # Let the planner decide which agent is best
        return False

    async def handle(self, task: AgentInput, *, context: AgentContext) -> AgentOutput:
        logger.info("enterprise_agent_task", task=task.model_dump())
        
        # Check if this task is actually relevant to enterprise agent
        if not self._is_relevant_task(task):
            logger.info("enterprise_agent_pass", reason="not_an_enterprise_task", prompt=task.prompt[:100] if task.prompt else "")
            return AgentOutput(
                agent=self.name,
                capability=self.capability,
                summary="[PASS]",
                confidence=0.0,
                rationale="Task not relevant to enterprise agent - passing.",
                metadata={"passed": True, "reason": "not_an_enterprise_task"},
            )
        
        # Initialize reasoning builder for tracking thought process
        reasoning = ReasoningBuilder(agent_name=self.name, context=context)
        
        # Step 1: Analyze the incoming request
        await reasoning.think(
            f"Received enterprise/business request: '{task.prompt[:100]}...' - analyzing for task type",
            step_type=ReasoningStepType.OBSERVATION,
        )
        
        # Step 2: Build context
        context_section = task.context
        if context_section is None and context.context is not None:
            await reasoning.think(
                "Assembling enterprise context from knowledge base",
                step_type=ReasoningStepType.ANALYSIS,
            )
            bundle = await context.context.build(task=_serialize_agent_input(task), agent=self.name)
            context_section = bundle.as_prompt_section()
            if context_section:
                await reasoning.add_finding(
                    claim="Retrieved enterprise context from knowledge base",
                    evidence=[context_section[:200] + "..."] if len(context_section) > 200 else [context_section],
                    confidence=0.7,
                    source="context_assembler",
                )
        
        # Step 3: Check for tool usage
        tool_result = await self._maybe_invoke_tool(task, context=context)
        if tool_result is not None:
            await reasoning.consider_tool(
                tool_name=tool_result.tool,
                reason=f"Enterprise tool invoked for enhanced analysis",
                selected=True,
            )
            await reasoning.add_finding(
                claim=f"Tool '{tool_result.tool}' provided additional data",
                confidence=0.8,
                source=tool_result.tool,
            )
        
        prompt = self._build_prompt(task, context_section=context_section, tool_result=tool_result)
        
        # Build dynamic system prompt based on user's requested persona
        dynamic_system_prompt = self._build_dynamic_system_prompt(task.prompt or "")
        persona = self._extract_persona(task.prompt or "")
        
        if persona:
            await reasoning.think(
                f"Detected persona request: '{persona}' - adapting response style",
                step_type=ReasoningStepType.ANALYSIS,
                evidence=f"Will adopt {persona} voice and expertise",
            )
        
        # Detect proposal/pitch requests - these need higher creativity
        prompt_lower = (task.prompt or "").lower()
        is_creative_task = any(phrase in prompt_lower for phrase in [
            "business proposal", "create a proposal", "write a proposal",
            "pitch deck", "investor pitch", "business plan", "startup idea",
            "act as", "you are", "imagine you", "pretend", "futuristic",
            "create a", "write a", "design a", "build a"
        ])
        
        if is_creative_task:
            await reasoning.think(
                "Task requires creative enterprise content - using higher temperature for originality",
                step_type=ReasoningStepType.SYNTHESIS,
            )
        
        # Higher temperature for creative tasks, use dynamic persona prompt
        temperature = 0.85 if is_creative_task else 0.3
        
        await reasoning.think(
            "Generating enterprise strategy with specialized prompt and context",
            step_type=ReasoningStepType.SYNTHESIS,
        )
        
        strategy = await context.llm.generate(
            prompt=prompt, 
            system_prompt=dynamic_system_prompt, 
            temperature=temperature,
            max_tokens=4096 if is_creative_task else 2048
        )

        # Handle LLM unavailability with fallback
        if is_llm_unavailable(strategy):
            logger.warning("enterprise_agent_llm_unavailable", task_id=task.task_id)
            strategy = self._generate_fallback_response(task, tool_result, context_section, persona)
            await reasoning.note_uncertainty(
                "LLM temporarily unavailable - providing template-based response"
            )

        # Step 4: Calculate confidence
        evidence_count = len(task.prior_exchanges)
        if context_section:
            evidence_count += 1
        if tool_result is not None:
            evidence_count += len(self._extract_actions(tool_result)) or 1

        self_assessment = min(0.62 + 0.1 * evidence_count, 0.97)
        confidence = 0.74
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
            f"Calculated confidence score: {confidence:.2f} based on {evidence_count} evidence sources",
            step_type=ReasoningStepType.EVALUATION,
            confidence=confidence,
        )

        # Step 5: Assess quality
        if confidence < 0.6:
            await reasoning.note_uncertainty(
                "Moderate confidence - recommendation may benefit from additional specialist review"
            )

        metadata: dict[str, Any] = {"type": "enterprise_strategy"}
        if tool_result is not None:
            metadata["tool"] = self._tool_metadata(tool_result)
        if confidence_breakdown is not None:
            metadata["confidence_breakdown"] = confidence_breakdown

        return AgentOutput(
            agent=self.name,
            capability=self.capability,
            summary=strategy,
            confidence=confidence,
            rationale="Executive recommendations synthesized from multi-agent context.",
            metadata=metadata,
            # Include reasoning transparency
            reasoning_steps=reasoning.steps,
            key_findings=reasoning.findings,
            tools_considered=reasoning.tools_considered,
            uncertainties=reasoning.uncertainties,
        )

    def _build_prompt(
        self,
        task: AgentInput,
        *,
        context_section: str | None = None,
        tool_result: ToolInvocationResult | None = None,
    ) -> str:
        import random
        import hashlib
        from datetime import datetime
        
        user_prompt = task.prompt or ""
        prompt_lower = user_prompt.lower()
        
        # Extract the persona if present
        persona = self._extract_persona(user_prompt)
        
        # Detect if this is a creative/proposal request
        is_creative_request = any(phrase in prompt_lower for phrase in [
            "business proposal", "create a proposal", "write a proposal",
            "pitch deck", "investor pitch", "business plan", "startup idea",
            "act as", "you are", "imagine you", "futuristic", "create a"
        ])
        
        if is_creative_request:
            # Generate a unique seed for originality
            seed = hashlib.md5(f"{datetime.now().isoformat()}{user_prompt}".encode()).hexdigest()[:8]
            
            # Industry pools for variety - avoid overused tech tropes
            industries = [
                "sustainable agriculture", "space logistics", "quantum computing",
                "regenerative medicine", "ocean exploration", "clean energy storage",
                "synthetic biology", "autonomous transportation", "digital identity",
                "precision nutrition", "carbon capture", "smart materials",
                "neural interfaces", "vertical farming", "drone delivery",
                "personalized education", "water purification", "waste-to-energy",
                "augmented reality retail", "predictive maintenance"
            ]
            
            # Pick a random industry hint to encourage variety
            industry_hint = random.choice(industries)
            
            # Build a VC-quality prompt with strong persona adoption
            instructions = f"""
=== CRITICAL: PERSONA ADOPTION ===
{f'You ARE a {persona}. Speak, think, and write EXACTLY as they would.' if persona else 'You are an elite pitch writer crafting investor-grade content.'}

=== ORIGINALITY REQUIREMENT (Seed: {seed}) ===
DO NOT create anything related to: sleep tech, brain interfaces, BCI, AI wellness, dream technology, neural headbands, or any variation thereof.
Instead, consider something in the realm of: {industry_hint} (but feel free to be creative).

=== VC-QUALITY STANDARDS ===
Real investors expect:
1. SPECIFIC NUMBERS - not "significant market" but "$47.3B market growing at 23% CAGR"
2. DEFENSIBLE MOATS - patents, network effects, switching costs, not just "first mover"  
3. UNIT ECONOMICS - CAC, LTV, payback period with realistic assumptions
4. TEAM CREDIBILITY - specific backgrounds, relevant exits, domain expertise
5. REALISTIC PROJECTIONS - bottoms-up math, not hockey sticks without justification
6. COMPETITIVE HONESTY - acknowledge real threats, show how you win anyway
7. CLEAR ASK - exactly how much, what for, what milestones it unlocks

=== WRITING STYLE ===
- BOLD opening hook that grabs attention in first sentence
- CONFIDENT tone without arrogance - "We will" not "We hope to"
- DATA-DRIVEN claims - every assertion backed by a number or source
- STORYTELLING - weave narrative through facts, make investors FEEL the problem
- URGENCY - why now? Why is this the moment?
- VISUAL formatting - tables, bullet points, clear hierarchy

=== STRUCTURE (Follow the user's requested sections if specified, otherwise use this) ===

**Executive Summary** (The hook - 2 paragraphs max, make them want to read more)

**The Problem** (Make it visceral - who suffers and why current solutions fail)

**Our Solution** (What we do, how it works, why it's 10x better)

**Market Opportunity** (TAM/SAM/SOM with methodology, not made-up numbers)

**Business Model** (Revenue streams, pricing, unit economics)

**Traction & Validation** (Users, revenue, partnerships, pilots - proof it works)

**Competitive Landscape** (Honest assessment, your unfair advantages)

**Go-to-Market Strategy** (How you acquire customers profitably)

**Team** (Why THIS team wins - relevant experience, exits, expertise)

**Financial Projections** (3-5 year P&L with assumptions)

**The Ask** (Amount, use of funds, milestones, expected runway)

**Risk Factors & Mitigation** (Show you've thought this through)

=== USER'S REQUEST ===
{user_prompt}

=== FINAL REMINDERS ===
- Create a COMPLETELY ORIGINAL company with a UNIQUE name
- Make it COMPELLING and PERSUASIVE
- Use SPECIFIC data and realistic numbers
- Format beautifully with Markdown
- This should be INVESTOR-READY quality
"""
            return instructions
        
        else:
            # For non-creative tasks, use standard business format
            metadata = json.dumps(task.metadata, indent=2, sort_keys=True)
            tool_actions = self._format_tool_actions(tool_result)
            prior = _summarize_prior_outputs(task.prior_exchanges)
            retrieved = context_section or "(no retrieved context)"
            
            return (
                f"Executive task:\n{user_prompt}\n\n"
                f"Business metadata:\n{metadata}\n\n"
                f"Cross-agent insights:\n{prior}\n\n"
                f"Retrieved context:\n{retrieved}\n\n"
                f"Playbook suggestions:\n{tool_actions}\n\n"
                f"Deliver a numbered action plan (max 5 steps) with expected impact and confidence."
            )

    async def _maybe_invoke_tool(self, task: AgentInput, *, context: AgentContext) -> ToolInvocationResult | None:
        if context.tools is None:
            return None
        payload = {
            "prompt": task.prompt,
            "metadata": task.metadata,
            "prior_outputs": [exchange.model_dump() for exchange in task.prior_exchanges],
        }
        try:
            return await context.tools.invoke("enterprise.playbook", payload)
        except (ToolDisabledError, ToolInvocationError) as exc:
            logger.warning("enterprise_tool_failure", error=str(exc))
            return None

    def _format_tool_actions(self, tool_result: ToolInvocationResult | None) -> str:
        actions = self._extract_actions(tool_result) if tool_result else []
        if not actions:
            return "(no playbook suggestions)"
        return "\n".join(
            f"- {item.get('action', 'action')} (impact: {item.get('impact', 'n/a')})"
            for item in actions
        )

    @staticmethod
    def _extract_actions(tool_result: ToolInvocationResult | None) -> list[dict[str, Any]]:
        if tool_result is None:
            return []
        response = tool_result.response
        actions = response.get("actions") if isinstance(response, dict) else None
        if isinstance(actions, list):
            return [item for item in actions if isinstance(item, dict)]
        return []

    @staticmethod
    def _tool_metadata(tool_result: ToolInvocationResult) -> dict[str, Any]:
        return {
            "name": tool_result.tool,
            "resolved": tool_result.resolved_tool,
            "cached": tool_result.cached,
            "latency": round(tool_result.latency, 4),
        }


def _summarize_prior_outputs(outputs: list[AgentExchange]) -> str:
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
