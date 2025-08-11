"""Task Orchestration Layer for NeuraForge.

This module serves as the central brain of the NeuraForge system,
responsible for decomposing tasks and routing them to appropriate agents.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from fastapi import HTTPException

from langchain.chains.router import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM

from ..agents import (
    BaseAgent, AgentInput, AgentOutput,
    ResearchAgent, CreativeAgent, FinancialAgent, EnterpriseAgent
)

logger = logging.getLogger(__name__)

class TaskRequest(BaseModel):
    """Schema for task orchestration requests."""
    messages: List[Dict[str, str]] = Field(..., description="The conversation history")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional task metadata")
    stream: bool = Field(False, description="Whether to stream the response")

class AgentResponse(BaseModel):
    """Schema for agent responses."""
    content: str = Field(..., description="The content of the response")
    agent_id: str = Field(..., description="The ID of the agent that generated the response")
    agent_name: str = Field(..., description="The name of the agent that generated the response")
    agent_type: str = Field(..., description="The type of the agent (research, creative, financial, enterprise)")
    confidence_score: float = Field(..., description="The confidence score of the agent (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional response metadata")

class TaskOrchestrator:
    """Orchestrates tasks and routes them to appropriate agents."""
    
    def __init__(self, llm: BaseLLM, callbacks=None):
        """Initialize the task orchestrator.
        
        Args:
            llm: The language model to use for task decomposition and routing
            callbacks: Optional list of callback handlers
        """
        self.llm = llm
        self.callbacks = callbacks
        self.router_chain = self._create_router_chain()
        
        # Initialize specialized agents
        self.research_agent = ResearchAgent(
            agent_id=f"research-{uuid.uuid4().hex[:8]}",
            llm=llm,
            callbacks=callbacks
        )
        
        self.creative_agent = CreativeAgent(
            agent_id=f"creative-{uuid.uuid4().hex[:8]}",
            llm=llm,
            callbacks=callbacks
        )
        
        self.financial_agent = FinancialAgent(
            agent_id=f"financial-{uuid.uuid4().hex[:8]}",
            llm=llm,
            callbacks=callbacks
        )
        
        self.enterprise_agent = EnterpriseAgent(
            agent_id=f"enterprise-{uuid.uuid4().hex[:8]}",
            llm=llm,
            callbacks=callbacks
        )
        
        # Register agents in the registry
        self.agents = {
            self.research_agent.agent_id: self.research_agent,
            self.creative_agent.agent_id: self.creative_agent,
            self.financial_agent.agent_id: self.financial_agent,
            self.enterprise_agent.agent_id: self.enterprise_agent,
        }
        
        logger.info(f"TaskOrchestrator initialized with {len(self.agents)} agents")
        
    def _create_router_chain(self) -> LLMRouterChain:
        """Create the router chain for task decomposition."""
        router_template = """
        You are a task router for an advanced AI assistant with multiple specialized agents.
        Based on the user query, determine which agent would be best suited to handle it.
        
        The available agents are:
        1. Research Agent: Handles factual information, research questions, data analysis
        2. Creative Agent: Handles creative tasks, brainstorming, content creation
        3. Financial Agent: Handles financial questions, calculations, investment advice
        4. Enterprise Agent: Handles business strategy, operations, management questions
        
        If multiple agents could handle the task, select the one with the highest expertise.
        
        User query: {input}
        
        Provide your response in the following format:
        Agent: <agent_name>
        Confidence: <confidence_score_between_0_and_1>
        Reasoning: <brief_explanation>
        """
        
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"]
        )
        
        return LLMRouterChain.from_llm(
            llm=self.llm,
            prompt=router_prompt,
            parser=RouterOutputParser()
        )
    
    def _get_agent_by_type(self, agent_type: str) -> BaseAgent:
        """Get the appropriate agent based on the agent type.
        
        Args:
            agent_type: Type of agent to retrieve (research, creative, financial, enterprise)
            
        Returns:
            BaseAgent: The requested agent
        """
        agent_type = agent_type.lower()
        if "research" in agent_type:
            return self.research_agent
        elif "creative" in agent_type:
            return self.creative_agent
        elif "financ" in agent_type:
            return self.financial_agent
        elif "enterprise" in agent_type or "business" in agent_type:
            return self.enterprise_agent
        else:
            # Default to enterprise agent if no match
            logger.warning(f"No agent found for type {agent_type}, using enterprise agent as default")
            return self.enterprise_agent
    
    async def process_task(self, task: TaskRequest) -> AgentResponse:
        """Process a task by routing it to the appropriate agent.
        
        Args:
            task: The task request containing messages and metadata
            
        Returns:
            AgentResponse: The response from the selected agent
        """
        # Extract the latest user message
        user_messages = [msg["content"] for msg in task.messages if msg["role"] == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user messages found in task")
        
        latest_user_message = user_messages[-1]
        lower_msg = latest_user_message.lower()
        has_url = ("http://" in lower_msg) or ("https://" in lower_msg)
        time_keywords = [
            "latest",
            "today",
            "breaking",
            "release",
            "launch",
            "announce",
            "news",
            "timeline",
            "date",
            "update",
        ]
        is_time_sensitive = any(k in lower_msg for k in time_keywords)
        
        # Route the task to the appropriate agent
        try:
            # First determine which agent should handle this task
            # Pre-route override: force research for URLs or time-sensitive queries
            if has_url or is_time_sensitive:
                selected_agent_type = "research"
                router_result = {"destination": "research", "confidence": 0.9, "reasoning": "Time-sensitive/URL override"}
            else:
                router_result = self.router_chain.route(latest_user_message)
                selected_agent_type = router_result.get("destination", "enterprise").lower()
            confidence = float(router_result.get("confidence", 0.7))
            reasoning = router_result.get("reasoning", "Default routing")
            
            logger.info(f"Router selected agent type: {selected_agent_type} with confidence {confidence}")
            logger.debug(f"Routing reasoning: {reasoning}")
            
            # Get the appropriate agent
            selected_agent = self._get_agent_by_type(selected_agent_type)
            
            # Create agent input from task
            agent_input = AgentInput(
                messages=task.messages,
                user_id=task.user_id,
                metadata=task.metadata or {}
            )
            
            # Process with the selected agent
            logger.info(f"Processing task with agent: {selected_agent.agent_name}")
            agent_output = await selected_agent.process(agent_input)
            
            # Construct the response
            return AgentResponse(
                content=agent_output.content,
                agent_id=selected_agent.agent_id,
                agent_name=selected_agent.agent_name,
                agent_type=selected_agent.agent_type,
                confidence_score=agent_output.confidence_score,
                metadata=agent_output.metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing task: {str(e)}")
