from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseModel):
    url: RedisDsn = Field(
        "redis://localhost:6379/0",
        description="Connection URL for Redis instance (defaults to local development instance).",
    )
    task_queue_db: int = Field(1, ge=0, description="Redis database index for the task queue.")


class PostgresSettings(BaseModel):
    dsn: PostgresDsn = Field(
        "postgresql://postgres:postgres@localhost:5432/neuraforge",
        description="PostgreSQL DSN for episodic memory storage.",
    )
    pool_min_size: int = Field(1, ge=1)
    pool_max_size: int = Field(10, ge=1)


class QdrantSettings(BaseModel):
    url: str = Field("http://localhost:6333", description="Qdrant host URL.")
    api_key: str | None = Field(default=None, description="Optional Qdrant API key.")
    collection_name: str = Field("neura_tasks", min_length=1)


class OllamaSettings(BaseModel):
    host: str = Field("http://localhost", description="Base URL where Ollama is running.")
    port: int = Field(11434, ge=1, le=65535)
    model: str = Field("llama3", description="Default model served via Ollama.")


class AuthSettings(BaseModel):
    jwt_secret_key: str = Field("0" * 32, min_length=32)
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_expire_minutes: int = Field(60, ge=1)


class ObservabilitySettings(BaseModel):
    prometheus_enabled: bool = Field(True)
    grafana_enabled: bool = Field(True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class MemorySettings(BaseModel):
    read_preference: Literal["cache-first", "store-first", "cache-only", "store-only"] = "cache-first"
    working_memory_ttl: int = Field(600, ge=1)
    ephemeral_ttl: int = Field(600, ge=1)
    batch_size: int = Field(50, ge=1)
    redis_namespace: str = Field("neuraforge", min_length=1)


class EmbeddingSettings(BaseModel):
    default_model: str = Field("all-MiniLM-L6-v2", min_length=1)
    fallback_model: str = Field("nomic-embed-text", min_length=1)
    cache_namespace: str = Field("neuraforge:embedding", min_length=1)
    cache_ttl_seconds: int = Field(86_400, ge=0)
    preferred_dimension: int | None = Field(None, ge=1, description="Optional expected vector dimension.")
    cache_enabled: bool = Field(True)


class RetrievalSettings(BaseModel):
    semantic_limit: int = Field(5, ge=1)
    episodic_limit: int = Field(5, ge=0)
    max_context_chars: int = Field(2_000, ge=200)
    relevance_threshold: float = Field(0.0, ge=0.0)


class ConsolidationSettings(BaseModel):
    enabled: bool = Field(False)
    interval_seconds: int = Field(300, ge=30)
    batch_size: int = Field(25, ge=1)
    max_tasks: int = Field(100, ge=1)


class ToolRateLimitSettings(BaseModel):
    max_calls: int = Field(60, ge=1, description="Maximum tool invocations within the period.")
    period_seconds: int = Field(60, ge=1, description="Rolling window for rate limiting.")


class MCPToolSettings(BaseModel):
    endpoint: str = Field("http://localhost:6111", description="Base URL for the MCP router.")
    api_key: str | None = Field(default=None, description="Optional MCP authentication token.")
    api_key_header: str = Field("Authorization", description="Header name used when attaching API tokens.")
    auth_scheme: str = Field("Bearer", description="Auth scheme prefix applied to the API key (e.g. 'Bearer').")
    client_id: str | None = Field(default=None, description="Optional client identifier for basic auth/signing flows.")
    client_secret: str | None = Field(default=None, description="Optional client secret for basic auth/signing flows.")
    timeout_seconds: float = Field(15.0, ge=0.1)
    enabled: bool = Field(False)
    cache_ttl_seconds: int = Field(300, ge=0)
    rate_limit: ToolRateLimitSettings = Field(default_factory=ToolRateLimitSettings)  # type: ignore[arg-type]
    healthcheck_path: str = Field("/health", description="Relative path used to verify MCP availability.")
    catalog_path: str = Field("/tools", description="Relative path for listing available MCP tools.")
    invoke_path_template: str = Field(
        "/tools/{tool}/invoke",
        description="Path template for invoking a tool; the placeholder '{tool}' is replaced with the resolved tool id.",
    )
    catalog_refresh_seconds: int = Field(600, ge=0, description="TTL for cached MCP tool catalog before refresh.")
    verify_ssl: bool = Field(True, description="Whether to verify SSL certificates when calling MCP endpoints.")
    extra_headers: dict[str, str] = Field(default_factory=dict, description="Additional HTTP headers to include for MCP calls.")
    aliases: dict[str, str] = Field(
        default_factory=dict,
        description="Optional mapping of logical tool aliases to MCP tool identifiers (e.g. 'finance.snapshot' -> 'finance/yfinance').",
    )
    max_retries: int = Field(2, ge=0, description="Maximum retry attempts for MCP HTTP requests.")
    retry_backoff_seconds: float = Field(0.5, ge=0.0, description="Initial backoff delay between retries.")
    retry_jitter_seconds: float = Field(0.25, ge=0.0, description="Maximum jitter added to retry backoff.")
    circuit_breaker_threshold: int = Field(5, ge=1, description="Failures before the MCP circuit breaker opens.")
    circuit_breaker_reset_seconds: float = Field(30.0, ge=1.0, description="Cool-down period before circuit closes.")
    signing_secret: str | None = Field(default=None, description="Optional secret used to sign requests (HMAC).")
    signing_header: str = Field("X-MCP-Signature", description="Header name used to send the request signature.")
    signing_algorithm: Literal["hmac-sha256"] = Field(
        "hmac-sha256", description="Signing algorithm applied when signing_secret is configured."
    )


class ToolSettings(BaseModel):
    mcp: MCPToolSettings = Field(default_factory=MCPToolSettings)  # type: ignore[arg-type]


class ScoringSettings(BaseModel):
    base_confidence: float = Field(0.6, ge=0.0, le=1.0)
    evidence_weight: float = Field(0.2, ge=0.0, le=1.0)
    tool_reliability_weight: float = Field(0.15, ge=0.0, le=1.0)
    self_assessment_weight: float = Field(0.15, ge=0.0, le=1.0)
    max_evidence: int = Field(5, ge=1)


class Settings(BaseSettings):
    environment: Literal["local", "test", "production"] = Field("local")
    api_v1_prefix: str = Field("/api/v1")

    redis: RedisSettings = Field(default_factory=RedisSettings)  # type: ignore[arg-type]
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)  # type: ignore[arg-type]
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)  # type: ignore[arg-type]
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)  # type: ignore[arg-type]
    auth: AuthSettings = Field(default_factory=AuthSettings)  # type: ignore[arg-type]
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)  # type: ignore[arg-type]
    memory: MemorySettings = Field(default_factory=MemorySettings)  # type: ignore[arg-type]
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)  # type: ignore[arg-type]
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)  # type: ignore[arg-type]
    consolidation: ConsolidationSettings = Field(default_factory=ConsolidationSettings)  # type: ignore[arg-type]
    tools: ToolSettings = Field(default_factory=ToolSettings)  # type: ignore[arg-type]
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)  # type: ignore[arg-type]

    backend_base_url: str = Field("http://localhost:8000")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )


@lru_cache(maxsize=1)
def get_settings(overrides: dict[str, Any] | None = None) -> Settings:
    """Return cached settings, useful for dependency injection."""
    if overrides:
        return Settings(**overrides)
    return Settings()  # type: ignore[call-arg]
