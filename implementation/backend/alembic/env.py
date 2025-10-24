from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, async_engine_from_config

from app.core.config import get_settings
from app.db.models import metadata

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


target_metadata = metadata


def _get_database_url() -> str:
    settings = get_settings()
    dsn = str(settings.postgres.dsn)
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    return dsn


def run_migrations_offline() -> None:
    url = _get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration: dict[str, Any] = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _get_database_url()

    connectable: AsyncEngine = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    async def async_run_migrations() -> None:
        try:
            async with connectable.connect() as connection:
                await connection.run_sync(do_run_migrations)
        finally:
            await connectable.dispose()

    def do_run_migrations(connection: Connection) -> None:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

    asyncio.run(async_run_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
