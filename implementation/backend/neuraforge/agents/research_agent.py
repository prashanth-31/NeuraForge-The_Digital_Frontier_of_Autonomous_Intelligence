"""Research Agent for NeuraForge.

This module implements a specialized agent for handling research-related tasks,
including factual information retrieval, data analysis, and academic research.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from langchain.llms.base import BaseLLM
from langchain.callbacks.base import BaseCallbackHandler

from .base_agent import BaseAgent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """Specialized agent for research and factual information tasks."""
    
    def __init__(
        self, 
        agent_id: str,
        llm: BaseLLM,
        callbacks: List[BaseCallbackHandler] = None
    ):
        """Initialize the research agent.
        
        Args:
            agent_id: Unique identifier for the agent
            llm: The language model to use
            callbacks: Optional list of callback handlers
        """
        # Research agent-specific prompt template
        prompt_template = """
        You are a Research Agent specializing in factual information, data analysis, and scholarly knowledge.
        Your goal is to provide accurate, well-sourced information on any topic.
        
        You excel at:
        - Finding and presenting factual information
        - Analyzing data and explaining trends
        - Summarizing research papers and academic content
        - Providing balanced perspectives on complex topics
        - Citing sources and maintaining academic integrity
        
        When you don't know something, acknowledge it rather than making up information.
        Always prioritize accuracy over speculation.
        
        Chat history:
        {chat_history}
        
        Human: {input}
        Research Agent:
        """
        
        super().__init__(
            agent_id=agent_id,
            agent_name="Research Agent",
            agent_type="research",
            llm=llm,
            prompt_template=prompt_template,
            callbacks=callbacks
        )
        
        # Research-specific tools could be added here
        # self.tools.append(web_search_tool)
        # self.tools.append(academic_database_tool)
        
        logger.info(f"Research Agent initialized with ID: {agent_id}")
    
    async def process(self, agent_input: AgentInput) -> AgentOutput:
        """Process the input and generate a research-focused response.
        
        Args:
            agent_input: The input to process
            
        Returns:
            AgentOutput: The research agent's response
        """
        # Extract the latest user message
        user_messages = [msg["content"] for msg in agent_input.messages if msg["role"] == "user"]
        if not user_messages:
            return AgentOutput(
                content="I don't see any questions to research. How can I help you with your research needs?",
                agent_info="Research Agent: Specialized in factual information and data analysis",
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
        
        # Special adjustment for research agent confidence
        # Research agents should be more confident on factual questions
        if any(keyword in latest_user_message.lower() for keyword in 
               ["what is", "define", "explain", "who", "when", "where", "why", "how does", "statistics"]):
            confidence_score = min(0.95, confidence_score + 0.1)
        
        return AgentOutput(
            content=response,
            agent_info="Research Agent: Specialized in factual information and data analysis",
            confidence_score=confidence_score,
            metadata={
                "agent_type": "research",
                "query_type": "informational",
                "tools_used": []  # Will be populated when tools are implemented
            }
        )
