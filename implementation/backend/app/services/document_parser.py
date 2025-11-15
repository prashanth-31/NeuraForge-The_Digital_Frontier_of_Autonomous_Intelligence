from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile

try:  # pragma: no cover - optional dependency
    import pdfplumber  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    pdfplumber = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import docx  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    docx = None  # type: ignore[assignment]

MAX_DOCUMENT_BYTES = 8_000_000  # 8 MB ceiling for uploaded documents


class DocumentParseError(ValueError):
    """Raised when an uploaded document cannot be parsed."""


@dataclass(slots=True)
class DocumentParseResult:
    text: str
    metadata: dict[str, Any]


async def parse_document(upload: UploadFile) -> DocumentParseResult:
    """Parse an uploaded document into normalized text."""
    filename = upload.filename or "document"
    extension = Path(filename).suffix.lower()
    media_type = (upload.content_type or "").lower()

    raw_bytes = await _read_bytes(upload)
    text: str

    if extension == ".pdf" or "pdf" in media_type:
        text = _parse_pdf(raw_bytes)
    elif extension == ".docx" or media_type in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}:
        text = _parse_docx(raw_bytes)
    elif extension in {".csv"} or media_type == "text/csv":
        text = _parse_csv(raw_bytes)
    elif extension in {".json"} or media_type in {"application/json"}:
        text = _parse_json(raw_bytes)
    elif extension in {".md", ".markdown"} or media_type in {"text/markdown"}:
        text = _parse_text(raw_bytes)
    elif extension in {".txt"} or media_type.startswith("text/"):
        text = _parse_text(raw_bytes)
    else:
        raise DocumentParseError("Unsupported document type. Accepts PDF, DOCX, TXT, CSV, JSON, or Markdown.")

    normalized = _normalize_text(text)
    if not normalized:
        raise DocumentParseError("Document does not contain any extractable text.")

    metadata = {
        "filename": filename,
        "content_type": upload.content_type or "application/octet-stream",
        "extension": extension or None,
        "filesize_bytes": len(raw_bytes),
        "line_count": normalized.count("\n") + 1,
        "character_count": len(normalized),
    }

    return DocumentParseResult(text=normalized, metadata=metadata)


async def _read_bytes(upload: UploadFile) -> bytes:
    data = await upload.read()
    await upload.seek(0)
    if not data:
        raise DocumentParseError("Uploaded document is empty.")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise DocumentParseError("Uploaded document exceeds 8 MB limit.")
    return data


def _parse_pdf(payload: bytes) -> str:
    if pdfplumber is None:
        raise DocumentParseError("PDF support unavailable. Install pdfplumber to enable PDF parsing.")
    buffer = io.BytesIO(payload)
    text_parts: list[str] = []
    with pdfplumber.open(buffer) as pdf:  # type: ignore[arg-type]
        for page in pdf.pages:
            extracted = page.extract_text() or ""
            if extracted:
                text_parts.append(extracted)
    return "\n".join(text_parts)


def _parse_docx(payload: bytes) -> str:
    if docx is None:
        raise DocumentParseError("DOCX support unavailable. Install python-docx to enable DOCX parsing.")
    document = docx.Document(io.BytesIO(payload))  # type: ignore[call-arg]
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)


def _parse_csv(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(decoded))
    lines = [", ".join(row).strip() for row in reader]
    return "\n".join(line for line in lines if line)


def _parse_json(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise DocumentParseError("JSON document is malformed.") from exc
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def _parse_text(payload: bytes) -> str:
    return payload.decode("utf-8", errors="replace")


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x00", "")
    return cleaned.strip()


__all__ = [
    "DocumentParseError",
    "DocumentParseResult",
    "parse_document",
    "MAX_DOCUMENT_BYTES",
]
