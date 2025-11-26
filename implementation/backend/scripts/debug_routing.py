from __future__ import annotations

import json
import os
from argparse import ArgumentParser

from fastapi.testclient import TestClient

from app.api import routes as routes_module
from app.dependencies import get_hybrid_memory, get_task_queue
from app.main import app
from tests.helpers.stubs import (
    ImmediateQueue,
    StubContextSnapshotStore,
    StubGuardrailStore,
    StubLLMService,
    StubMemoryService,
    StubOrchestratorStateStore,
    StubTaskLifecycleStore,
)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Submit a task and print orchestration routing metadata")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Draft a launch plan",
        help="Prompt to submit to /api/v1/submit_task",
    )
    parser.add_argument(
        "--metadata",
        default="{\"priority\": \"high\"}",
        help="JSON string for task metadata",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        metadata = json.loads(args.metadata)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid metadata JSON: {exc}") from exc

    memory = StubMemoryService()
    llm = StubLLMService()
    queue = ImmediateQueue()

    routes_module.HybridMemoryService.from_settings = lambda settings: memory
    routes_module.LLMService.from_settings = (
        lambda settings, *, model=None, client=None: llm  # noqa: ARG005
    )
    routes_module.OrchestratorStateStore.from_settings = lambda settings: StubOrchestratorStateStore()
    routes_module.TaskLifecycleStore.from_settings = lambda settings: StubTaskLifecycleStore()
    routes_module.ContextSnapshotStore.from_settings = lambda settings: StubContextSnapshotStore()
    routes_module.GuardrailStore.from_settings = lambda settings: StubGuardrailStore()

    async def override_queue():
        yield queue

    async def override_memory():
        yield memory

    app.dependency_overrides[get_task_queue] = override_queue
    app.dependency_overrides[get_hybrid_memory] = override_memory

    if not os.getenv("ALPHAVANTAGE_API_KEY"):
        # Use Alpha Vantage demo key locally so finance tools do not bail before fallback kicks in
        os.environ["ALPHAVANTAGE_API_KEY"] = "demo"

    client = TestClient(app)
    response = client.post(
        "/api/v1/submit_task",
        json={"prompt": args.prompt, "metadata": metadata},
    )
    print("status", response.status_code)
    print(response.json())
    print("llm calls", len(llm.calls))
    task_id = response.json()["task_id"]
    print("routing", memory.ephemeral[task_id]["result"].get("routing"))
    app.dependency_overrides.clear()


if __name__ == "__main__":
    main()
