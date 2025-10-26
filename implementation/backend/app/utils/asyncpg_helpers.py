from __future__ import annotations

from typing import Any


async def ensure_pool_ready(pool: Any) -> Any:
    """Ensure an asyncpg pool is fully initialized before use."""
    initializer = getattr(pool, "_async__init__", None)
    initialized = getattr(pool, "_initialized", True)
    if callable(initializer) and not initialized:
        await initializer()
    return pool
