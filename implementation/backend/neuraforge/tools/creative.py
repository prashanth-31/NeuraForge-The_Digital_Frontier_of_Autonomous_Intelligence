"""Creative tools wrapping Datamuse."""
from __future__ import annotations

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..services import datamuse


class SimilarMeaningInput(BaseModel):
    words: str = Field(..., description="Words to find similar meanings for")
    max_results: int = Field(10, ge=1, le=50)


async def _similar_meaning(words: str, max_results: int = 10) -> str:
    items = await datamuse.similar_meaning(words, max_results=max_results)
    return ", ".join(items)


class RhymesInput(BaseModel):
    word: str = Field(..., description="Word to rhyme")
    max_results: int = Field(10, ge=1, le=50)


async def _rhymes(word: str, max_results: int = 10) -> str:
    items = await datamuse.rhymes(word, max_results=max_results)
    return ", ".join(items)


similar_meaning_tool = StructuredTool.from_function(
    coroutine=_similar_meaning,
    name="datamuse_similar_meaning",
    description="Get words with similar meaning (for brainstorming and taglines).",
    args_schema=SimilarMeaningInput,
)

rhymes_tool = StructuredTool.from_function(
    coroutine=_rhymes,
    name="datamuse_rhymes",
    description="Get rhyming words (for slogans and creative writing).",
    args_schema=RhymesInput,
)
