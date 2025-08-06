"""NeuraForge API integration for orchestration.

This module connects the FastAPI endpoints with the Task Orchestration layer.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from ..orchestration.task_orchestrator import TaskOrchestrator, TaskRequest, AgentResponse
from ..llm.core import get_llm_model

logger = logging.getLogger(__name__)

# Global orchestrator instance to be initialized during startup
_orchestrator = None

def get_orchestrator() -> TaskOrchestrator:
    """Get the global orchestrator instance, initializing it if necessary.
    
    Returns:
        TaskOrchestrator: The global orchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        # Initialize the LLM
        llm = get_llm_model()
        # Initialize the orchestrator with the LLM
        _orchestrator = TaskOrchestrator(llm=llm)
        logger.info("Global orchestrator initialized")
    return _orchestrator

async def process_message(
    messages: List[Dict[str, str]],
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    stream: bool = False
) -> Dict[str, Any]:
    """Process a message through the orchestration layer.
    
    Args:
        messages: The conversation history
        user_id: Optional user identifier
        metadata: Optional task metadata
        stream: Whether to stream the response
        
    Returns:
        Dict[str, Any]: The response from the orchestrator
    """
    # Get the orchestrator
    orchestrator = get_orchestrator()
    
    # Create a task request
    task_request = TaskRequest(
        messages=messages,
        user_id=user_id,
        metadata=metadata or {},
        stream=stream
    )
    
    # Process the task
    logger.info(f"Processing message from user {user_id or 'anonymous'}")
    response = await orchestrator.process_task(task_request)
    
    # Convert to dictionary for API response
    return {
        "content": response.content,
        "agent": {
            "id": response.agent_id,
            "name": response.agent_name,
            "type": response.agent_type,
            "confidence": response.confidence_score
        },
        "metadata": response.metadata or {}
    }
