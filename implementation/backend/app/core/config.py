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


class Settings(BaseSettings):
    environment: Literal["local", "test", "production"] = Field("local")
    api_v1_prefix: str = Field("/api/v1")

    redis: RedisSettings
    postgres: PostgresSettings
    qdrant: QdrantSettings
    ollama: OllamaSettings
    auth: AuthSettings
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

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
