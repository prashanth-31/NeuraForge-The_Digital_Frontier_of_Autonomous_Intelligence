"""Creative Agent for NeuraForge.

This module implements a specialized agent for handling creative tasks,
including content generation, writing, storytelling, and creative problem-solving.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from langchain.llms.base import BaseLLM
from langchain.callbacks.base import BaseCallbackHandler

from .base_agent import BaseAgent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class CreativeAgent(BaseAgent):
    """Specialized agent for creative and generative tasks."""
    
    def __init__(
        self, 
        agent_id: str,
        llm: BaseLLM,
        callbacks: List[BaseCallbackHandler] = None
    ):
        """Initialize the creative agent.
        
        Args:
            agent_id: Unique identifier for the agent
            llm: The language model to use
            callbacks: Optional list of callback handlers
        """
        # Creative agent-specific prompt template
        prompt_template = """
        You are a Creative Agent specializing in content generation, writing, and creative problem-solving.
        Your goal is to provide imaginative, original, and engaging content.
        
        You excel at:
        - Generating creative writing (stories, poems, scripts)
        - Brainstorming innovative ideas and solutions
        - Crafting engaging marketing and communication content
        - Adapting writing style to different tones and audiences
        - Providing creative perspectives on problems
        
        Always prioritize originality and creativity while maintaining coherence and purpose.
        
        Chat history:
        {chat_history}
        
        Human: {input}
        Creative Agent:
        """
        
        super().__init__(
            agent_id=agent_id,
            agent_name="Creative Agent",
            agent_type="creative",
            llm=llm,
            prompt_template=prompt_template,
            callbacks=callbacks
        )
        
        # Creative-specific tools could be added here
        # self.tools.append(metaphor_generator)
        # self.tools.append(story_structure_tool)
        
        logger.info(f"Creative Agent initialized with ID: {agent_id}")
    
    async def process(self, agent_input: AgentInput) -> AgentOutput:
        """Process the input and generate a creative response.
        
        Args:
            agent_input: The input to process
            
        Returns:
            AgentOutput: The creative agent's response
        """
        # Extract the latest user message
        user_messages = [msg["content"] for msg in agent_input.messages if msg["role"] == "user"]
        if not user_messages:
            return AgentOutput(
                content="I don't see any creative tasks. How can I help you with content creation or creative thinking?",
                agent_info="Creative Agent: Specialized in creative writing and innovative thinking",
                confidence_score=0.9
            )
        
        latest_user_message = user_messages[-1]
        
        # Format chat history for context
        chat_history = self._format_chat_history(agent_input.messages[:-1])  # Exclude the latest message
        
        # Process with LLM
        response = await self.chain.arun(
            input=latest_user_message,
            chat_history=chat_history
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(response)
        
        # Special adjustment for creative agent confidence
        # Creative agents should be more confident on creative requests
        if any(keyword in latest_user_message.lower() for keyword in 
               ["write", "create", "generate", "design", "story", "poem", "creative", "imagine", "brainstorm"]):
            confidence_score = min(0.95, confidence_score + 0.1)
        
        return AgentOutput(
            content=response,
            agent_info="Creative Agent: Specialized in creative writing and innovative thinking",
            confidence_score=confidence_score,
            metadata={
                "agent_type": "creative",
                "query_type": "generative",
                "tools_used": []  # Will be populated when tools are implemented
            }
        )
