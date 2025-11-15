from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import random
import uuid
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.memory import EpisodeRecord, HybridMemoryService

logger = get_logger(name=__name__)

AGENT_CHOICES = ("enterprise_agent", "finance_agent", "research_agent", "creative_agent")


def _build_payload(task_id: str, agent: str, *, iteration: int) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "agent": agent,
        "iteration": iteration,
        "timestamp": datetime.now(UTC).isoformat(),
        "summary": f"Phase5 validation payload {iteration} for {agent}",
    }


async def _run_validation(*, settings: Settings, iterations: int, threshold: float) -> dict[str, Any]:
    memory = HybridMemoryService.from_settings(settings)
    stored_ids: list[str] = []
    direct_hits = 0
    coverage_hits = 0

    async with memory.lifecycle():
        for index in range(iterations):
            agent = random.choice(AGENT_CHOICES)
            task_id = f"phase5-validation-{uuid.uuid4().hex}"
            payload = _build_payload(task_id, agent, iteration=index)
            await memory.store_ephemeral_memory(task_id=task_id, payload=payload, agent=agent)
            stored_ids.append(task_id)

        for task_id in stored_ids:
            fetched = await memory.fetch_ephemeral_memory(task_id)
            if fetched and fetched.get("task_id") == task_id:
                direct_hits += 1

        recent_ids = await memory.enumerate_recent_task_ids(limit=iterations)
        coverage_hits = sum(1 for task_id in stored_ids if task_id in recent_ids)

    success_rate = direct_hits / iterations if iterations else 1.0
    coverage_rate = coverage_hits / iterations if iterations else 1.0
    meets_threshold = success_rate >= threshold and coverage_rate >= threshold

    return {
        "iterations": iterations,
        "success_rate": round(success_rate, 3),
        "coverage_rate": round(coverage_rate, 3),
        "threshold": threshold,
        "meets_threshold": meets_threshold,
    }


def _materialize_settings(args: argparse.Namespace) -> Settings:
    base = get_settings()
    payload = base.model_dump()

    if args.redis_url:
        payload.setdefault("redis", {})["url"] = args.redis_url

    if args.postgres_dsn:
        payload.setdefault("postgres", {})["dsn"] = args.postgres_dsn

    if args.qdrant_url:
        payload.setdefault("qdrant", {})["url"] = args.qdrant_url

    if args.qdrant_api_key is not None:
        payload.setdefault("qdrant", {})["api_key"] = args.qdrant_api_key or None
    else:
        qdrant_section = payload.get("qdrant")
        if isinstance(qdrant_section, dict) and not qdrant_section.get("api_key"):
            qdrant_section["api_key"] = None

    # Rehydrate Settings so downstream components see a fully validated configuration.
    return Settings(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 5 memory retrieval pipeline.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of episodic records to write/read.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Minimum acceptable success rate for retrieval checks.",
    )
    parser.add_argument("--redis-url", type=str, help="Override Redis connection URL (e.g. redis://localhost:16379/0).")
    parser.add_argument(
        "--postgres-dsn",
        type=str,
        help="Override Postgres DSN (e.g. postgresql://postgres:postgres@localhost:15432/neuraforge).",
    )
    parser.add_argument("--qdrant-url", type=str, help="Override Qdrant base URL (e.g. http://localhost:16333).")
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=None,
        help="Optional Qdrant API key override; pass an empty string to clear the configured key.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _materialize_settings(args)
    result = asyncio.run(
        _run_validation(settings=settings, iterations=args.iterations, threshold=args.threshold)
    )
    logger.info("memory_validation_result", **result)
    print(
        f"Iterations: {result['iterations']}\n"
        f"Success rate: {result['success_rate']:.3f}\n"
        f"Coverage rate: {result['coverage_rate']:.3f}\n"
        f"Threshold: {result['threshold']:.2f}\n"
        f"Meets threshold: {result['meets_threshold']}"
    )
    if not result["meets_threshold"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
