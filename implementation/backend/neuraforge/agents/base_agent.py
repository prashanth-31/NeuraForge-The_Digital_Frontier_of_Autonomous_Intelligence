"""Base agent module for NeuraForge.

This module defines the base agent class and common agent functionality
that is shared across all domain-specific agents.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from langchain.llms.base import BaseLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)

class AgentInput(BaseModel):
    """Input schema for agent processing."""
    messages: List[Dict[str, str]] = Field(..., description="The conversation history")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    use_memory: bool = Field(True, description="Whether to use memory for context")
    stream: bool = Field(False, description="Whether to stream the response")

class AgentOutput(BaseModel):
    """Output schema for agent responses."""
    content: str = Field(..., description="The content of the response")
    agent_info: str = Field(..., description="Information about the agent")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

class BaseAgent(ABC):
    """Base class for all NeuraForge agents."""
    
    def __init__(
        self, 
        agent_id: str,
        agent_name: str,
        agent_type: str,
        llm: BaseLLM,
        prompt_template: str,
        tools: List[Callable] = None,
        callbacks: List[BaseCallbackHandler] = None
    ):
        """Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name for the agent
            agent_type: Type of agent (research, creative, finance, enterprise)
            llm: The language model to use
            prompt_template: The template for the agent's prompt
            tools: Optional list of tools the agent can use
            callbacks: Optional list of callback handlers
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.llm = llm
        self.tools = tools or []
        self.callbacks = callbacks or []
        
        # Create the LLM chain with the specified prompt template
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "chat_history"]
        )
        
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
            callbacks=self.callbacks
        )
        
        logger.info(f"Initialized {agent_type} agent: {agent_name} ({agent_id})")
    
    @abstractmethod
    async def process(self, agent_input: AgentInput) -> AgentOutput:
        """Process the input and generate a response.
        
        Args:
            agent_input: The input to process
            
        Returns:
            AgentOutput: The agent's response
        """
        pass
    
    def _format_chat_history(self, messages: List[Dict[str, str]]) -> str:
        """Format the chat history for inclusion in the prompt.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            str: Formatted chat history string
        """
        formatted_history = []
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                formatted_history.append(f"Human: {content}")
            elif role == "assistant":
                formatted_history.append(f"Assistant: {content}")
            elif role == "system":
                # Skip system messages in the formatted history
                continue
            else:
                formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate a confidence score for the response.
        
        This is a simplified implementation. In a real system, this would use
        more sophisticated methods like checking against facts, evaluating
        uncertainty expressions, etc.
        
        Args:
            response: The agent's response
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Simple heuristic: longer responses generally indicate more confidence
        # This is just a placeholder and should be replaced with a better method
        base_confidence = 0.7  # Start with a reasonable baseline
        
        # Penalize very short responses
        if len(response) < 50:
            base_confidence -= 0.2
        
        # Penalize responses with uncertainty markers
        uncertainty_phrases = [
            "I'm not sure", "uncertain", "might be", "possibly", 
            "I don't know", "unclear", "can't determine"
        ]
        
        for phrase in uncertainty_phrases:
            if phrase.lower() in response.lower():
                base_confidence -= 0.1
                
        # Ensure confidence is within bounds
        return max(0.1, min(0.99, base_confidence))
