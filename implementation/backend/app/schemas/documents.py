from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    filename: str = Field(..., description="Original document filename.")
    content_type: str | None = Field(None, description="Resolved content type.")
    extension: str | None = Field(None, description="File extension lowercased.")
    line_count: int = Field(..., ge=1, description="Number of text lines extracted from the document.")
    character_count: int = Field(..., ge=1, description="Number of characters available after parsing.")
    filesize_bytes: int = Field(..., ge=1, description="Raw document size in bytes.")


class DocumentIngestionInfo(BaseModel):
    document_id: str = Field(..., description="Identifier used to retrieve stored document context.")
    chunk_count: int = Field(..., ge=1, description="Number of chunks persisted for this document.")
    truncated: bool = Field(False, description="True when the document exceeded configured chunk limits.")
    preview: str | None = Field(None, description="Preview snippet returned after ingestion.")


class DocumentAnalysisResponse(BaseModel):
    output: str = Field(..., description="LLM-produced analysis of the uploaded document.")
    document: DocumentMetadata = Field(..., description="Metadata about the processed document.")
    truncated: bool = Field(False, description="Indicates whether the document content was truncated before analysis.")
    persisted: bool = Field(False, description="True when the document snapshot was persisted to memory services.")
    memory_task_id: str | None = Field(None, description="Task identifier used when persisting to memory.")
    preview: str | None = Field(None, description="First portion of the parsed text for UI previewing.")
    ingestion: DocumentIngestionInfo | None = Field(
        None,
        description="Details about the document chunks stored for downstream agent consumption.",
    )


__all__ = [
    "DocumentAnalysisResponse",
    "DocumentMetadata",
    "DocumentIngestionInfo",
]
