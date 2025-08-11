"""Base agent interfaces and utilities for NeuraForge agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    # Newer LangChain
    from langchain_core.language_models.llms import BaseLLM
except Exception:
    from langchain.llms.base import BaseLLM  # type: ignore


class AgentInput(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    content: str
    confidence_score: float = 0.8
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class BaseAgent:
    agent_id: str
    agent_name: str
    agent_type: str
    llm: BaseLLM
    callbacks: Optional[List[Any]] = None
    system_prompt: str = ""

    async def process(self, agent_input: AgentInput) -> AgentOutput:
        """Process the input and return output. To be implemented by subclasses."""
        raise NotImplementedError

    # Helper: build a single prompt from chat messages
    def build_prompt(self, agent_input: AgentInput, task_instructions: str) -> str:
        history = []
        for m in agent_input.messages[-8:]:  # limit history
            role = m.get("role", "user").upper()
            content = m.get("content", "").strip()
            if not content:
                continue
            history.append(f"{role}: {content}")
        history_text = "\n".join(history)
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Task: {task_instructions}\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"Assistant:"
        )
        return prompt
