from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - optional heavy dependency
    SentenceTransformer = None  # type: ignore[misc,assignment]

from ..core.logging import get_logger

logger = get_logger(name=__name__)


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer | None:  # type: ignore[name-defined]
    if SentenceTransformer is None:
        logger.warning("sentence_transformers_not_available")
        return None
    return SentenceTransformer(model_name)


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    model = get_embedding_model()
    if model is None:
        return [[0.0] * 384 for _ in texts]
    embeddings = model.encode(list(texts), normalize_embeddings=True)
    return [embedding.tolist() for embedding in embeddings]  # type: ignore[return-value]
