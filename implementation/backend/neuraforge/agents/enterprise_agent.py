"""Enterprise Agent for NeuraForge.

This module implements a specialized agent for handling enterprise and business tasks,
including business strategy, operational efficiency, and organizational management.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from langchain.llms.base import BaseLLM
from langchain.callbacks.base import BaseCallbackHandler

from .base_agent import BaseAgent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class EnterpriseAgent(BaseAgent):
    """Specialized agent for enterprise and business tasks."""
    
    def __init__(
        self, 
        agent_id: str,
        llm: BaseLLM,
        callbacks: List[BaseCallbackHandler] = None
    ):
        """Initialize the enterprise agent.
        
        Args:
            agent_id: Unique identifier for the agent
            llm: The language model to use
            callbacks: Optional list of callback handlers
        """
        # Enterprise agent-specific prompt template
        prompt_template = """
        You are an Enterprise Agent specializing in business strategy, operations, and organizational management.
        Your goal is to provide valuable insights and solutions for business challenges.
        
        You excel at:
        - Strategic business planning and analysis
        - Operational efficiency improvements
        - Project management and workflow optimization
        - Market analysis and competitive intelligence
        - Organizational development and leadership
        
        Focus on practical, implementable advice based on best practices.
        
        Chat history:
        {chat_history}
        
        Human: {input}
        Enterprise Agent:
        """
        
        super().__init__(
            agent_id=agent_id,
            agent_name="Enterprise Agent",
            agent_type="enterprise",
            llm=llm,
            prompt_template=prompt_template,
            callbacks=callbacks
        )
        
        # Enterprise-specific tools could be added here
        # self.tools.append(business_framework_tool)
        # self.tools.append(market_analysis_tool)
        
        logger.info(f"Enterprise Agent initialized with ID: {agent_id}")
    
    async def process(self, agent_input: AgentInput) -> AgentOutput:
        """Process the input and generate a business-focused response.
        
        Args:
            agent_input: The input to process
            
        Returns:
            AgentOutput: The enterprise agent's response
        """
        # Extract the latest user message
        user_messages = [msg["content"] for msg in agent_input.messages if msg["role"] == "user"]
        if not user_messages:
            return AgentOutput(
                content="I don't see any business questions. How can I help you with enterprise or organizational challenges?",
                agent_info="Enterprise Agent: Specialized in business strategy and operations",
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
        
        # Special adjustment for enterprise agent confidence
        # Enterprise agents should be more confident on business questions
        if any(keyword in latest_user_message.lower() for keyword in 
               ["business", "strategy", "company", "organization", "management", "process", "operation", 
                "project", "team", "leadership", "market", "customer"]):
            confidence_score = min(0.95, confidence_score + 0.1)
        
        return AgentOutput(
            content=response,
            agent_info="Enterprise Agent: Specialized in business strategy and operations",
            confidence_score=confidence_score,
            metadata={
                "agent_type": "enterprise",
                "query_type": "advisory",
                "tools_used": []  # Will be populated when tools are implemented
            }
        )
