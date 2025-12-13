from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..core.config import Settings
from ..core.logging import get_logger

try:  # pragma: no cover - optional heavy dependency
    from langchain_ollama import ChatOllama
except ModuleNotFoundError:  # pragma: no cover
    ChatOllama = None  # type: ignore[misc, assignment]

logger = get_logger(name=__name__)


def _build_base_url(host: str, port: int) -> str:
    if ":" in host.rsplit("/", maxsplit=1)[-1]:
        return host.rstrip("/")
    return f"{host.rstrip('/')}: {port}"


def _messages_from_text(
    prompt: str,
    system_prompt: str | None = None,
) -> Sequence[BaseMessage]:
    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    return messages


# ════════════════════════════════════════════════════════════════════════════════
# Retry Configuration for LLM calls
# ════════════════════════════════════════════════════════════════════════════════
LLM_MAX_RETRIES = 3
LLM_BASE_DELAY = 1.0  # seconds
LLM_MAX_DELAY = 10.0  # seconds
CIRCUIT_BREAKER_RESET_TIME = 30.0  # Reset circuit breaker after 30 seconds


# Marker for unavailable LLM responses
LLM_UNAVAILABLE_MARKER = "[LLM_UNAVAILABLE]"


def is_llm_unavailable(response: str) -> bool:
    """Check if a response indicates LLM was unavailable."""
    return response.startswith(LLM_UNAVAILABLE_MARKER) or response == "LLM generation temporarily unavailable."


@dataclass
class LLMService:
    """Thin LangChain-based client for interacting with local Ollama models."""

    settings: Settings
    _client: Any
    model: str
    default_system_prompt: str = (
        "You are NeuraForge's orchestration assistant. Be concise, structured, and cite key signals."
    )
    _client_cache: ClassVar[dict[str, Any]] = {}
    _consecutive_failures: ClassVar[int] = 0
    _max_consecutive_failures: ClassVar[int] = 5
    _last_failure_time: ClassVar[float] = 0.0

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        model: str | None = None,
        client: Any | None = None,
    ) -> "LLMService":
        model_name = model or settings.ollama.model
        if client is None:
            cache_key = f"{settings.ollama.host}:{settings.ollama.port}:{model_name}"
            cached = cls._client_cache.get(cache_key)
            if cached is None:
                if ChatOllama is None:  # pragma: no cover - handled in runtime logs
                    raise RuntimeError("langchain_ollama is not installed")
                base_url = _build_base_url(settings.ollama.host, settings.ollama.port)
                cached = ChatOllama(model=model_name, base_url=base_url, temperature=0.1)
                cls._client_cache[cache_key] = cached
            client = cached
        return cls(settings=settings, _client=client, model=model_name)

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """Generate text from the configured LLM, with retry and graceful degradation."""
        messages = _messages_from_text(prompt, system_prompt or self.default_system_prompt)
        client = self._client
        
        # Build options dict for temperature and max_tokens (num_predict for Ollama)
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens  # Ollama uses num_predict for max tokens
        effective_num_ctx = num_ctx if num_ctx is not None else getattr(self.settings.ollama, "num_ctx", None)
        if effective_num_ctx is not None:
            options["num_ctx"] = int(effective_num_ctx)
        if options and hasattr(client, "with_options"):
            client = client.with_options(**options)
        
        # Check if circuit breaker should reset (time-based reset)
        current_time = time.time()
        if LLMService._consecutive_failures >= LLMService._max_consecutive_failures:
            time_since_failure = current_time - LLMService._last_failure_time
            if time_since_failure >= CIRCUIT_BREAKER_RESET_TIME:
                logger.info(
                    "llm_circuit_breaker_reset",
                    time_since_failure=time_since_failure,
                    model=self.model,
                )
                LLMService._consecutive_failures = 0
            else:
                logger.warning(
                    "llm_circuit_breaker_open",
                    consecutive_failures=LLMService._consecutive_failures,
                    time_until_reset=CIRCUIT_BREAKER_RESET_TIME - time_since_failure,
                    model=self.model,
                )
                return f"{LLM_UNAVAILABLE_MARKER} LLM temporarily unavailable. Circuit breaker open."
        
        # Retry loop with exponential backoff
        last_error: Exception | None = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                result = await asyncio.wait_for(
                    client.ainvoke(messages),
                    timeout=60.0  # 60 second timeout per attempt
                )
                # Success - reset failure counter
                LLMService._consecutive_failures = 0
                return _extract_content(result)
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError("LLM request timed out after 60 seconds")
                LLMService._consecutive_failures += 1
                LLMService._last_failure_time = time.time()
                logger.warning(
                    "llm_generation_timeout",
                    attempt=attempt + 1,
                    max_attempts=LLM_MAX_RETRIES,
                    model=self.model,
                )
            except Exception as exc:
                last_error = exc
                LLMService._consecutive_failures += 1
                LLMService._last_failure_time = time.time()
                
                # Log the failure
                logger.warning(
                    "llm_generation_retry",
                    attempt=attempt + 1,
                    max_attempts=LLM_MAX_RETRIES,
                    error=str(exc),
                    model=self.model,
                )
                
                # Don't retry on certain errors - Ollama crashed
                error_str = str(exc).lower()
                if "llama runner process has terminated" in error_str:
                    logger.error(
                        "llm_ollama_crashed",
                        error=str(exc),
                        model=self.model,
                        host=self.settings.ollama.host,
                    )
                    # Try to restart Ollama by clearing cache and retrying
                    cache_key = f"{self.settings.ollama.host}:{self.settings.ollama.port}:{self.model}"
                    if cache_key in LLMService._client_cache:
                        del LLMService._client_cache[cache_key]
                    break
            
            if attempt < LLM_MAX_RETRIES - 1:
                # Exponential backoff
                delay = min(LLM_BASE_DELAY * (2 ** attempt), LLM_MAX_DELAY)
                await asyncio.sleep(delay)
        
        # All retries exhausted
        logger.error(
            "llm_generation_failed",
            error=str(last_error) if last_error else "Unknown error",
            model=self.model,
            host=self.settings.ollama.host,
            attempts=LLM_MAX_RETRIES,
        )
        return f"{LLM_UNAVAILABLE_MARKER} LLM generation failed after {LLM_MAX_RETRIES} attempts."

    async def chat(self, messages: Sequence[BaseMessage]) -> str:
        """Execute a chat interaction with pre-built messages."""
        try:
            result = await self._client.ainvoke(messages)
        except Exception as exc:  # pragma: no cover
            logger.exception("llm_chat_failed", error=str(exc))
            return "LLM generation temporarily unavailable."
        return _extract_content(result)

    async def moderate(self, text: str) -> dict[str, Any]:
        """Lightweight moderation heuristic returning severity score."""
        lowered = text.lower()
        keywords = ["attack", "exploit", "classified", "leak", "self-harm", "violence"]
        hits = [word for word in keywords if word in lowered]
        severity = min(1.0, 0.2 * len(hits))
        return {
            "severity": severity,
            "hits": hits,
            "length": len(text),
        }


def _extract_content(result: Any) -> str:
    if isinstance(result, AIMessage):
        content = result.content
        if isinstance(content, list):
            # Join list elements as string, handling dicts if present
            return " ".join(
                str(item) if not isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list):
            return " ".join(
                str(item) if not isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)
    return str(result)
