from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class LLMService:
    """Thin LangChain-based client for interacting with local Ollama models."""

    settings: Settings
    _client: Any
    default_system_prompt: str = (
        "You are NeuraForge's orchestration assistant. Be concise, structured, and cite key signals."
    )
    _client_cache: ClassVar[dict[str, Any]] = {}

    @classmethod
    def from_settings(cls, settings: Settings, *, client: Any | None = None) -> "LLMService":
        if client is None:
            cache_key = f"{settings.ollama.host}:{settings.ollama.port}:{settings.ollama.model}"
            cached = cls._client_cache.get(cache_key)
            if cached is None:
                if ChatOllama is None:  # pragma: no cover - handled in runtime logs
                    raise RuntimeError("langchain_ollama is not installed")
                base_url = _build_base_url(settings.ollama.host, settings.ollama.port)
                cached = ChatOllama(model=settings.ollama.model, base_url=base_url, temperature=0.1)
                cls._client_cache[cache_key] = cached
            client = cached
        return cls(settings=settings, _client=client)

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text from the configured LLM, with graceful degradation."""
        messages = _messages_from_text(prompt, system_prompt or self.default_system_prompt)
        client = self._client
        if temperature is not None and hasattr(client, "with_options"):
            client = client.with_options(temperature=temperature)
        try:
            result = await client.ainvoke(messages)
        except Exception as exc:  # pragma: no cover - network failures logged
            logger.exception(
                "llm_generation_failed",
                error=str(exc),
                model=self.settings.ollama.model,
                host=self.settings.ollama.host,
            )
            return "LLM generation temporarily unavailable."
        return _extract_content(result)

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
