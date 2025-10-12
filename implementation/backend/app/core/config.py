from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseModel):
    url: RedisDsn = Field(..., description="Connection URL for Redis instance.")
    task_queue_db: int = Field(1, ge=0, description="Redis database index for the task queue.")


class PostgresSettings(BaseModel):
    dsn: PostgresDsn = Field(..., description="PostgreSQL DSN for episodic memory storage.")
    pool_min_size: int = Field(1, ge=1)
    pool_max_size: int = Field(10, ge=1)


class QdrantSettings(BaseModel):
    url: str = Field(..., description="Qdrant host URL.")
    api_key: str | None = Field(default=None, description="Optional Qdrant API key.")
    collection_name: str = Field("neura_tasks", min_length=1)


class OllamaSettings(BaseModel):
    host: str = Field("http://localhost", description="Base URL where Ollama is running.")
    port: int = Field(11434, ge=1, le=65535)
    model: str = Field("llama3", description="Default model served via Ollama.")


class AuthSettings(BaseModel):
    jwt_secret_key: str = Field(..., min_length=32)
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


class Settings(BaseSettings):
    environment: Literal["local", "test", "production"] = Field("local")
    api_v1_prefix: str = Field("/api/v1")

    redis: RedisSettings
    postgres: PostgresSettings
    qdrant: QdrantSettings
    ollama: OllamaSettings
    auth: AuthSettings
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)  # type: ignore[arg-type]
    memory: MemorySettings = Field(default_factory=MemorySettings)  # type: ignore[arg-type]
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)  # type: ignore[arg-type]
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)  # type: ignore[arg-type]
    consolidation: ConsolidationSettings = Field(default_factory=ConsolidationSettings)  # type: ignore[arg-type]

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
