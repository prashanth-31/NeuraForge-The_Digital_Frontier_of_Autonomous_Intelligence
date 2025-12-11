"""
Tool Chain Execution Support

This module provides capability for agents to execute multi-step tool chains,
where the output of one tool becomes the input to another. This enables
more sophisticated reasoning patterns.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from ..core.logging import get_logger
from ..schemas.agents import ThinkingEvent, ThinkingEventType

logger = get_logger(name=__name__)


@dataclass
class ToolChainStep:
    """A single step in a tool chain."""
    tool_name: str
    payload_builder: Callable[[dict[str, Any]], dict[str, Any]]
    description: str = ""
    optional: bool = False
    retry_count: int = 0
    timeout_seconds: float | None = None


@dataclass
class ToolChainResult:
    """Result of executing a tool chain."""
    success: bool
    steps_completed: int
    total_steps: int
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    final_output: Any = None
    execution_time_ms: float = 0.0


class ToolChainExecutor:
    """
    Executes a chain of tools where each step can use results from previous steps.
    
    Example usage:
        chain = ToolChain()
        chain.add_step(
            "research.search",
            lambda ctx: {"query": ctx["prompt"]},
            description="Search for information"
        )
        chain.add_step(
            "finance.snapshot",
            lambda ctx: {"symbol": extract_symbol(ctx["step_0_result"])},
            description="Get financial data"
        )
        result = await executor.execute(chain, initial_context)
    """
    
    def __init__(
        self,
        tool_service: Any,
        thinking_emitter: Callable[[ThinkingEvent], Awaitable[None]] | None = None,
        agent_name: str = "tool_chain",
        max_retries: int = 2,
    ):
        self.tool_service = tool_service
        self.thinking_emitter = thinking_emitter
        self.agent_name = agent_name
        self.max_retries = max_retries
    
    async def _emit_thinking(
        self,
        thought: str,
        event_type: ThinkingEventType = ThinkingEventType.TOOL_PROGRESS,
        step_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a thinking event for tool chain progress."""
        if self.thinking_emitter is None:
            return
        event = ThinkingEvent(
            event_type=event_type,
            agent=self.agent_name,
            thought=thought,
            step_index=step_index,
            metadata=metadata or {},
        )
        await self.thinking_emitter(event)
    
    async def execute(
        self,
        steps: list[ToolChainStep],
        initial_context: dict[str, Any],
    ) -> ToolChainResult:
        """
        Execute a chain of tools sequentially, passing results forward.
        
        Args:
            steps: List of tool chain steps to execute
            initial_context: Starting context with initial data
            
        Returns:
            ToolChainResult with all step results and final output
        """
        start_time = datetime.now(timezone.utc)
        context = dict(initial_context)
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        steps_completed = 0
        
        await self._emit_thinking(
            f"Starting tool chain with {len(steps)} steps",
            event_type=ThinkingEventType.PLANNING,
            metadata={"total_steps": len(steps)},
        )
        
        for idx, step in enumerate(steps):
            step_name = step.tool_name
            
            await self._emit_thinking(
                f"Step {idx + 1}: Executing {step.description or step_name}",
                event_type=ThinkingEventType.TOOL_PROGRESS,
                step_index=idx + 1,
                metadata={"tool": step_name, "optional": step.optional},
            )
            
            # Build payload using context from previous steps
            try:
                payload = step.payload_builder(context)
            except Exception as exc:
                error_info = {
                    "step_index": idx,
                    "tool": step_name,
                    "phase": "payload_build",
                    "error": str(exc),
                }
                errors.append(error_info)
                logger.warning("tool_chain_payload_build_failed", **error_info)
                
                if not step.optional:
                    await self._emit_thinking(
                        f"Step {idx + 1} failed: Could not build payload - {exc}",
                        event_type=ThinkingEventType.UNCERTAINTY,
                        step_index=idx + 1,
                    )
                    break
                continue
            
            # Execute with retries
            result = None
            last_error = None
            retries = step.retry_count or self.max_retries
            
            for attempt in range(retries + 1):
                try:
                    if step.timeout_seconds:
                        result = await asyncio.wait_for(
                            self.tool_service.invoke(step_name, payload),
                            timeout=step.timeout_seconds,
                        )
                    else:
                        result = await self.tool_service.invoke(step_name, payload)
                    last_error = None
                    break
                except asyncio.TimeoutError as exc:
                    last_error = exc
                    logger.warning("tool_chain_step_timeout", tool=step_name, attempt=attempt + 1)
                except Exception as exc:
                    last_error = exc
                    logger.warning("tool_chain_step_failed", tool=step_name, attempt=attempt + 1, error=str(exc))
                    
                if attempt < retries:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
            
            if last_error is not None:
                error_info = {
                    "step_index": idx,
                    "tool": step_name,
                    "phase": "execution",
                    "error": str(last_error),
                    "attempts": retries + 1,
                }
                errors.append(error_info)
                
                if not step.optional:
                    await self._emit_thinking(
                        f"Step {idx + 1} failed after {retries + 1} attempts: {last_error}",
                        event_type=ThinkingEventType.UNCERTAINTY,
                        step_index=idx + 1,
                    )
                    break
                continue
            
            # Store result in context for next step
            step_result = {
                "step_index": idx,
                "tool": step_name,
                "success": True,
                "result": result.payload if hasattr(result, "payload") else result,
            }
            results.append(step_result)
            context[f"step_{idx}_result"] = step_result["result"]
            context["last_result"] = step_result["result"]
            steps_completed += 1
            
            await self._emit_thinking(
                f"Step {idx + 1} completed successfully",
                event_type=ThinkingEventType.FINDING,
                step_index=idx + 1,
                metadata={"tool": step_name},
            )
        
        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        success = steps_completed == len([s for s in steps if not s.optional])
        
        await self._emit_thinking(
            f"Tool chain completed: {steps_completed}/{len(steps)} steps successful",
            event_type=ThinkingEventType.EVALUATING,
            metadata={
                "success": success,
                "steps_completed": steps_completed,
                "execution_time_ms": execution_time_ms,
            },
        )
        
        return ToolChainResult(
            success=success,
            steps_completed=steps_completed,
            total_steps=len(steps),
            results=results,
            errors=errors,
            final_output=context.get("last_result"),
            execution_time_ms=execution_time_ms,
        )


class ToolChainBuilder:
    """Fluent builder for creating tool chains."""
    
    def __init__(self):
        self._steps: list[ToolChainStep] = []
    
    def add(
        self,
        tool_name: str,
        payload_builder: Callable[[dict[str, Any]], dict[str, Any]],
        description: str = "",
        optional: bool = False,
        retry_count: int = 0,
        timeout_seconds: float | None = None,
    ) -> "ToolChainBuilder":
        """Add a step to the chain."""
        self._steps.append(ToolChainStep(
            tool_name=tool_name,
            payload_builder=payload_builder,
            description=description,
            optional=optional,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        ))
        return self
    
    def then(
        self,
        tool_name: str,
        transform: Callable[[Any], dict[str, Any]],
        description: str = "",
        optional: bool = False,
    ) -> "ToolChainBuilder":
        """Add a step that uses the last result."""
        def payload_builder(ctx: dict[str, Any]) -> dict[str, Any]:
            return transform(ctx.get("last_result"))
        
        return self.add(
            tool_name=tool_name,
            payload_builder=payload_builder,
            description=description,
            optional=optional,
        )
    
    def build(self) -> list[ToolChainStep]:
        """Build the final step list."""
        return list(self._steps)


def chain() -> ToolChainBuilder:
    """Create a new tool chain builder."""
    return ToolChainBuilder()
