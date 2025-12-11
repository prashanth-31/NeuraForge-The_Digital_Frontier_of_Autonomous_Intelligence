"""
Orchestration Package

This package contains the core orchestration components for the multi-agent system:
- Graph-based orchestration (LangGraph StateGraph)
- LLM-powered planning
- Tool chain execution
- Dynamic re-planning
- Agent handoff protocol
- Parallel agent execution
"""

from .graph import Orchestrator
from .llm_planner import LLMOrchestrationPlanner, PlannerError
from .planner_contract import PlannedAgentStep, PlannerPlan, PlanGatekeeper
from .tool_policy import AgentToolPolicy, get_agent_tool_policy
from ..core.config import TOOL_ENFORCEMENT_POLICY

# New modules
from .tool_chain import (
    ToolChainExecutor,
    ToolChainBuilder,
    ToolChainStep,
    ToolChainResult,
    chain,
)
from .dynamic_replanner import (
    DynamicReplanner,
    ReplanPolicy,
    ReplanTrigger,
    ReplanContext,
    ReplanResult,
    ReplanCheckpoint,
)
from .handoff_protocol import (
    HandoffProtocol,
    HandoffRequest,
    HandoffResult,
    HandoffChain,
    HandoffReason,
    HandoffPriority,
    create_handoff_request,
)
from .parallel_executor import (
    ParallelExecutor,
    ParallelExecutionPlan,
    ParallelExecutionResult,
    ParallelAgentResult,
    ParallelizationStrategy,
    analyze_parallelization_opportunities,
)

__all__ = [
    # Core
    "Orchestrator",
    "LLMOrchestrationPlanner",
    "PlannerError",
    "PlannedAgentStep",
    "PlannerPlan",
    "PlanGatekeeper",
    "AgentToolPolicy",
    "get_agent_tool_policy",
    "TOOL_ENFORCEMENT_POLICY",
    # Tool Chain
    "ToolChainExecutor",
    "ToolChainBuilder",
    "ToolChainStep",
    "ToolChainResult",
    "chain",
    # Dynamic Re-planning
    "DynamicReplanner",
    "ReplanPolicy",
    "ReplanTrigger",
    "ReplanContext",
    "ReplanResult",
    "ReplanCheckpoint",
    # Handoff Protocol
    "HandoffProtocol",
    "HandoffRequest",
    "HandoffResult",
    "HandoffChain",
    "HandoffReason",
    "HandoffPriority",
    "create_handoff_request",
    # Parallel Execution
    "ParallelExecutor",
    "ParallelExecutionPlan",
    "ParallelExecutionResult",
    "ParallelAgentResult",
    "ParallelizationStrategy",
    "analyze_parallelization_opportunities",
]
