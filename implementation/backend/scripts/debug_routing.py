from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import routes as routes_module
from app.dependencies import get_hybrid_memory, get_task_queue
from app.main import app
from tests.test_tasks import StubLLMService, StubMemoryService, ImmediateQueue


def main() -> None:
    memory = StubMemoryService()
    llm = StubLLMService()
    queue = ImmediateQueue()

    routes_module.HybridMemoryService.from_settings = lambda settings: memory
    routes_module.LLMService.from_settings = lambda settings: llm

    async def override_queue():
        yield queue

    async def override_memory():
        yield memory

    app.dependency_overrides[get_task_queue] = override_queue
    app.dependency_overrides[get_hybrid_memory] = override_memory

    client = TestClient(app)
    response = client.post(
        "/api/v1/submit_task",
        json={"prompt": "Draft a launch plan", "metadata": {"priority": "high"}},
    )
    print("status", response.status_code)
    print(response.json())
    print("llm calls", len(llm.calls))
    task_id = response.json()["task_id"]
    print("routing", memory.ephemeral[task_id]["result"].get("routing"))
    app.dependency_overrides.clear()


if __name__ == "__main__":
    main()
