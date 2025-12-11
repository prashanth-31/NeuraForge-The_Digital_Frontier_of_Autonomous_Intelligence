"""
Parallel Agent Execution Module

This module provides capability for executing multiple agents concurrently
when their tasks are independent and can be parallelized.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Awaitable, Sequence

from ..core.logging import get_logger
from ..schemas.agents import (
    AgentInput,
    AgentOutput,
    ThinkingEvent,
    ThinkingEventType,
)

if TYPE_CHECKING:
    from ..agents.base import BaseAgent, AgentContext
    from .planner_contract import PlannedAgentStep

logger = get_logger(name=__name__)


class ParallelizationStrategy(str, Enum):
    """Strategy for parallelizing agent execution."""
    NONE = "none"  # Sequential execution (default)
    INDEPENDENT = "independent"  # Run independent agents in parallel
    FORK_JOIN = "fork_join"  # Fork to parallel, join results
    RACE = "race"  # Run in parallel, use first successful result


@dataclass
class ParallelExecutionPlan:
    """Plan for parallel agent execution."""
    strategy: ParallelizationStrategy
    groups: list[list[str]]  # Groups of agent names to run in parallel
    dependencies: dict[str, list[str]] = field(default_factory=dict)  # agent -> depends on
    merge_strategy: str = "concatenate"  # How to merge parallel results


@dataclass
class ParallelAgentResult:
    """Result from a single agent in parallel execution."""
    agent: str
    output: AgentOutput | None = None
    error: Exception | None = None
    execution_time_ms: float = 0.0
    success: bool = True


@dataclass
class ParallelExecutionResult:
    """Result of parallel agent execution."""
    strategy: ParallelizationStrategy
    results: list[ParallelAgentResult]
    merged_output: AgentOutput | None = None
    total_execution_time_ms: float = 0.0
    parallel_speedup: float = 1.0  # Ratio of sequential time / parallel time


class ParallelExecutor:
    """
    Executes multiple agents in parallel with various strategies.
    
    Supports:
    - Independent parallel execution
    - Fork-join patterns
    - Race conditions (first success wins)
    - Result merging
    """
    
    def __init__(
        self,
        agents: dict[str, "BaseAgent"],
        thinking_emitter: Callable[[ThinkingEvent], Awaitable[None]] | None = None,
        max_concurrency: int = 5,
        timeout_seconds: float = 60.0,
    ):
        self.agents = agents
        self.thinking_emitter = thinking_emitter
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _emit_thinking(
        self,
        thought: str,
        event_type: ThinkingEventType = ThinkingEventType.COLLABORATION,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a thinking event for parallel execution visibility."""
        if self.thinking_emitter is None:
            return
        event = ThinkingEvent(
            event_type=event_type,
            agent="orchestrator",
            thought=thought,
            metadata=metadata or {},
        )
        await self.thinking_emitter(event)
    
    async def _execute_single_agent(
        self,
        agent: "BaseAgent",
        agent_input: AgentInput,
        context: "AgentContext",
    ) -> ParallelAgentResult:
        """Execute a single agent with semaphore control."""
        start_time = datetime.now(timezone.utc)
        
        async with self._semaphore:
            try:
                await self._emit_thinking(
                    f"Starting parallel execution of {agent.name}",
                    metadata={"agent": agent.name},
                )
                
                output = await asyncio.wait_for(
                    agent.handle(agent_input, context=context),
                    timeout=self.timeout_seconds,
                )
                
                end_time = datetime.now(timezone.utc)
                execution_time_ms = (end_time - start_time).total_seconds() * 1000
                
                return ParallelAgentResult(
                    agent=agent.name,
                    output=output,
                    execution_time_ms=execution_time_ms,
                    success=True,
                )
                
            except asyncio.TimeoutError:
                end_time = datetime.now(timezone.utc)
                execution_time_ms = (end_time - start_time).total_seconds() * 1000
                logger.warning("parallel_agent_timeout", agent=agent.name)
                return ParallelAgentResult(
                    agent=agent.name,
                    error=TimeoutError(f"Agent {agent.name} timed out after {self.timeout_seconds}s"),
                    execution_time_ms=execution_time_ms,
                    success=False,
                )
                
            except Exception as exc:
                end_time = datetime.now(timezone.utc)
                execution_time_ms = (end_time - start_time).total_seconds() * 1000
                logger.exception("parallel_agent_failed", agent=agent.name)
                return ParallelAgentResult(
                    agent=agent.name,
                    error=exc,
                    execution_time_ms=execution_time_ms,
                    success=False,
                )
    
    async def execute_parallel(
        self,
        agent_names: list[str],
        agent_input: AgentInput,
        context: "AgentContext",
        strategy: ParallelizationStrategy = ParallelizationStrategy.INDEPENDENT,
    ) -> ParallelExecutionResult:
        """
        Execute multiple agents in parallel.
        
        Args:
            agent_names: Names of agents to execute
            agent_input: Input to pass to each agent
            context: Agent context
            strategy: Parallelization strategy
            
        Returns:
            ParallelExecutionResult with all agent outputs
        """
        start_time = datetime.now(timezone.utc)
        
        # Resolve agents
        agents_to_run = []
        for name in agent_names:
            if name in self.agents:
                agents_to_run.append(self.agents[name])
            else:
                logger.warning("parallel_agent_not_found", agent=name)
        
        if not agents_to_run:
            return ParallelExecutionResult(
                strategy=strategy,
                results=[],
                total_execution_time_ms=0.0,
            )
        
        await self._emit_thinking(
            f"Starting parallel execution of {len(agents_to_run)} agents: {[a.name for a in agents_to_run]}",
            event_type=ThinkingEventType.PLANNING,
            metadata={"agents": [a.name for a in agents_to_run], "strategy": strategy.value},
        )
        
        if strategy == ParallelizationStrategy.RACE:
            result = await self._execute_race(agents_to_run, agent_input, context)
        else:
            result = await self._execute_all(agents_to_run, agent_input, context, strategy)
        
        end_time = datetime.now(timezone.utc)
        result.total_execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Calculate speedup
        sequential_time = sum(r.execution_time_ms for r in result.results)
        if result.total_execution_time_ms > 0:
            result.parallel_speedup = sequential_time / result.total_execution_time_ms
        
        await self._emit_thinking(
            f"Parallel execution complete: {len([r for r in result.results if r.success])}/{len(result.results)} successful, "
            f"speedup: {result.parallel_speedup:.2f}x",
            event_type=ThinkingEventType.EVALUATING,
            metadata={
                "successful": len([r for r in result.results if r.success]),
                "total": len(result.results),
                "speedup": result.parallel_speedup,
            },
        )
        
        return result
    
    async def _execute_all(
        self,
        agents: list["BaseAgent"],
        agent_input: AgentInput,
        context: "AgentContext",
        strategy: ParallelizationStrategy,
    ) -> ParallelExecutionResult:
        """Execute all agents and wait for all to complete."""
        tasks = [
            self._execute_single_agent(agent, agent_input, context)
            for agent in agents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Merge outputs if using fork-join
        merged_output = None
        if strategy == ParallelizationStrategy.FORK_JOIN:
            merged_output = self._merge_outputs([r for r in results if r.success and r.output])
        
        return ParallelExecutionResult(
            strategy=strategy,
            results=list(results),
            merged_output=merged_output,
        )
    
    async def _execute_race(
        self,
        agents: list["BaseAgent"],
        agent_input: AgentInput,
        context: "AgentContext",
    ) -> ParallelExecutionResult:
        """Execute agents in race mode - first successful result wins."""
        tasks = {
            asyncio.create_task(self._execute_single_agent(agent, agent_input, context)): agent.name
            for agent in agents
        }
        
        results: list[ParallelAgentResult] = []
        winner: ParallelAgentResult | None = None
        
        pending = set(tasks.keys())
        
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                result = task.result()
                results.append(result)
                
                if result.success and winner is None:
                    winner = result
                    # Cancel remaining tasks
                    for remaining_task in pending:
                        remaining_task.cancel()
                    pending = set()
                    break
        
        return ParallelExecutionResult(
            strategy=ParallelizationStrategy.RACE,
            results=results,
            merged_output=winner.output if winner else None,
        )
    
    def _merge_outputs(
        self,
        results: list[ParallelAgentResult],
    ) -> AgentOutput | None:
        """Merge multiple agent outputs into a single output."""
        if not results:
            return None
        
        outputs = [r.output for r in results if r.output is not None]
        if not outputs:
            return None
        
        if len(outputs) == 1:
            return outputs[0]
        
        # Merge summaries
        merged_summary = "\n\n".join([
            f"[{o.agent}]: {o.summary}"
            for o in outputs
        ])
        
        # Merge findings
        merged_findings = []
        for output in outputs:
            if output.key_findings:
                merged_findings.extend(output.key_findings)
        
        # Merge uncertainties
        merged_uncertainties = []
        for output in outputs:
            if output.uncertainties:
                merged_uncertainties.extend(output.uncertainties)
        
        # Average confidence
        avg_confidence = sum(o.confidence for o in outputs) / len(outputs)
        
        # Create merged output using first output as base
        from ..schemas.agents import AgentCapability
        
        return AgentOutput(
            agent="parallel_merge",
            capability=AgentCapability.GENERAL,
            summary=merged_summary,
            confidence=avg_confidence,
            rationale=f"Merged results from {len(outputs)} agents",
            key_findings=merged_findings[:10],
            uncertainties=list(set(merged_uncertainties))[:5],
            metadata={
                "merged_from": [o.agent for o in outputs],
                "merge_count": len(outputs),
            },
        )


def analyze_parallelization_opportunities(
    steps: Sequence["PlannedAgentStep"],
) -> ParallelExecutionPlan:
    """
    Analyze a sequence of planned steps to identify parallelization opportunities.
    
    This function identifies which steps can be run in parallel based on
    their dependencies and tool requirements.
    """
    # Simple heuristic: agents with same tools can't run in parallel (tool contention)
    # Agents without explicit dependencies can potentially run in parallel
    
    groups: list[list[str]] = []
    current_group: list[str] = []
    used_tools: set[str] = set()
    
    for step in steps:
        step_tools = set(step.tools)
        
        # Check for tool conflict with current group
        if step_tools & used_tools:
            # Tool conflict - start new group
            if current_group:
                groups.append(current_group)
            current_group = [step.agent]
            used_tools = step_tools
        else:
            # No conflict - add to current group
            current_group.append(step.agent)
            used_tools.update(step_tools)
    
    if current_group:
        groups.append(current_group)
    
    # Determine strategy
    if len(groups) == len(steps):
        # All sequential
        strategy = ParallelizationStrategy.NONE
    elif any(len(g) > 1 for g in groups):
        # Some parallel opportunities
        strategy = ParallelizationStrategy.FORK_JOIN
    else:
        strategy = ParallelizationStrategy.NONE
    
    return ParallelExecutionPlan(
        strategy=strategy,
        groups=groups,
    )
