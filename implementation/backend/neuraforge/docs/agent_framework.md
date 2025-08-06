# NeuraForge Agent Framework Implementation

## Overview

The NeuraForge Agent Framework is the core of Phase 2 in the NeuraForge architecture. This document outlines the implemented components and their functionality.

## Architecture Components

### 1. Task Orchestration Layer

The Task Orchestration Layer is the central brain of NeuraForge, responsible for decomposing complex tasks and routing them to the appropriate specialized agents.

**Key Components:**
- `TaskOrchestrator`: Central component that routes user queries to the appropriate agent
- Router Chain: Uses LangChain's router to analyze user queries and select the best agent
- Agent Registry: Maintains the collection of available specialized agents

**Implementation:**
- Located in `neuraforge/orchestration/task_orchestrator.py`
- Uses LLM router for intelligent task routing based on content analysis
- Initializes and manages the lifecycle of all specialized agents

### 2. Specialized Agents

The Agent Framework includes several domain-specific agents, each specialized for particular types of tasks.

**Implemented Agents:**

1. **Research Agent**
   - Purpose: Handles factual information, research questions, and data analysis
   - Capabilities: Information retrieval, fact-checking, summarization
   - Implementation: `neuraforge/agents/research_agent.py`

2. **Creative Agent**
   - Purpose: Handles creative tasks, content generation, and brainstorming
   - Capabilities: Writing, ideation, artistic suggestions
   - Implementation: `neuraforge/agents/creative_agent.py`

3. **Financial Agent**
   - Purpose: Handles financial analysis, planning, and advice
   - Capabilities: Calculations, financial planning, investment guidance
   - Implementation: `neuraforge/agents/financial_agent.py`

4. **Enterprise Agent**
   - Purpose: Handles business strategy, operations, and organizational tasks
   - Capabilities: Business planning, operational analysis, management advice
   - Implementation: `neuraforge/agents/enterprise_agent.py`

### 3. Agent Base Class

The BaseAgent class provides common functionality for all agents, ensuring consistent interfaces and behavior.

**Key Features:**
- Standard processing pipeline for all agents
- Common methods for confidence calculation
- Unified input/output formats
- Implementation: `neuraforge/agents/base_agent.py`

### 4. API Integration

The API layer connects the orchestration system to the FastAPI endpoints, enabling frontend communication.

**Key Components:**
- API interface for routing requests to the orchestration layer
- WebSocket support for future streaming responses
- Updated response format with agent metadata
- Implementation: `neuraforge/api/integration.py`

## Data Flow

1. User sends a query via the frontend interface
2. API endpoint receives the request and formats it for the orchestrator
3. Task Orchestrator analyzes the query and selects the most appropriate agent
4. Selected agent processes the query and generates a response
5. Response is returned to the frontend with agent metadata
6. Frontend displays the response with appropriate agent visualization

## Frontend Updates

The frontend has been updated to support the new agent framework:

1. Updated API service to handle the new response format
2. Enhanced agent visualization with descriptive badges
3. Support for agent-specific styling and icons
4. Improved message handling in the chat context

## Next Steps

1. **Agent Tools**: Add specialized tools for each agent type
2. **Memory Integration**: Implement context-aware memory for more coherent conversations
3. **Agent Collaboration**: Enable multiple agents to collaborate on complex tasks
4. **User Preferences**: Allow users to explicitly select preferred agents
5. **Streaming Responses**: Implement real-time streaming of agent responses

## Usage Examples

### Research Agent
```python
# Example of using the Research Agent
from neuraforge.agents import ResearchAgent
from langchain.llms import BaseLLM

llm = get_llm_model()  # Get your LLM
research_agent = ResearchAgent(agent_id="research-1", llm=llm)

result = await research_agent.process({
    "messages": [{"role": "user", "content": "What is quantum computing?"}]
})
print(result.content)
```

### Task Orchestrator
```python
# Example of using the Task Orchestrator
from neuraforge.orchestration import TaskOrchestrator, TaskRequest
from langchain.llms import BaseLLM

llm = get_llm_model()  # Get your LLM
orchestrator = TaskOrchestrator(llm=llm)

response = await orchestrator.process_task(TaskRequest(
    messages=[{"role": "user", "content": "Create a marketing plan for a new product"}]
))
print(f"Response from {response.agent_name}: {response.content}")
```
