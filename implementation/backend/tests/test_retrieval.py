from __future__ import annotations

import pytest

from app.services.embedding import EmbeddingRecord
from app.services.retrieval import ContextAssembler, RetrievalConfig, RetrievalService


from app.services.retrieval import HybridMemoryService

class StubMemory(HybridMemoryService):
    def __init__(self, semantic: list[dict[str, object]], episodic: list[dict[str, object]]) -> None:
        self.semantic = semantic
        self.episodic = episodic

    async def retrieve_context(self, *, query_vector: list[float], limit: int = 5) -> list[dict[str, object]]:  # noqa: ARG002
        return self.semantic[:limit]

    async def fetch_recent_context(
        self,
        *,
        agent: str | None = None,  # noqa: ARG002
        limit: int = 5,
    ) -> list[dict[str, object]]:
        return self.episodic[:limit]


from app.services.embedding import EmbeddingService

class StubEmbedder(EmbeddingService):
    def __init__(self) -> None:
        pass

    async def embed_text(
        self,
        text: str,
        *,
        metadata: dict[str, object] | None = None,  # noqa: ARG002
        store: bool = False,  # noqa: ARG002
        vector_id: str | None = None,  # noqa: ARG002
        collection: str | None = None,  # noqa: ARG002
        score: float = 1.0,  # noqa: ARG002
    ) -> EmbeddingRecord:
        return EmbeddingRecord(vector=[1.0, 0.0], metadata={"text": text}, cache_key="retrieval:test")


@pytest.mark.asyncio
async def test_retrieval_merges_semantic_and_episodic() -> None:
    memory = StubMemory(
        semantic=[{"text": "semantic insight", "_score": 0.92}],
        episodic=[{"result": {"content": "episodic recall"}}],
    )
    embedder = StubEmbedder()
    config = RetrievalConfig(semantic_limit=3, episodic_limit=3, max_context_chars=200)
    service = RetrievalService(memory=memory, embedder=embedder, config=config)

    result = await service.gather(query="plan a launch", agent="research_agent")

    assert len(result.semantic_hits) == 1
    assert len(result.episodic_hits) == 1

    assembler = ContextAssembler(retrieval=service, max_chars=200)
    bundle = await assembler.build(task={"prompt": "plan a launch", "outputs": []}, agent="research_agent")
    section = bundle.as_prompt_section()
    assert "semantic insight" in section
    assert "episodic recall" in section


@pytest.mark.asyncio
async def test_context_respects_relevance_threshold_and_budget() -> None:
    memory = StubMemory(
        semantic=[
            {"text": "high score", "_score": 0.85},
            {"text": "low score", "_score": 0.2},
        ],
        episodic=[{"content": "short episodic"}],
    )
    embedder = StubEmbedder()
    config = RetrievalConfig(semantic_limit=5, episodic_limit=1, max_context_chars=40, relevance_threshold=0.5)
    service = RetrievalService(memory=memory, embedder=embedder, config=config)

    assembler = ContextAssembler(retrieval=service, max_chars=40)
    bundle = await assembler.build(task={"prompt": "budget", "outputs": []}, agent="finance_agent")
    section = bundle.as_prompt_section()

    assert "high score" in section
    assert "low score" not in section
    assert len(section) <= 60  # ensure prompt stays within the configured budget