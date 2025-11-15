from __future__ import annotations


class ToolError(RuntimeError):
    """Base class for tooling-related failures."""


class ToolInvocationError(ToolError):
    """Raised when a tool invocation fails after retries or due to adapter errors."""


class ToolTimeoutError(ToolInvocationError):
    """Raised when a tool invocation exceeds the configured timeout."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool cannot be resolved."""


class CircuitBreakerOpenError(ToolInvocationError):
    """Raised when the circuit breaker prevents tool execution."""


class ToolPolicyViolationError(ToolError):
    """Raised when an agent attempts to invoke a tool outside of its allowlist."""
