from __future__ import annotations

import base64
import hashlib
import textwrap
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from .base import MCPToolAdapter


class PromptStylizerInput(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=2_000)
    tone: str | None = Field(default=None, max_length=64)
    audience: str | None = Field(default=None, max_length=128)

    model_config = {"extra": "forbid"}


class PromptStylizerOutput(BaseModel):
    styled_prompt: str
    suggestions: list[str]
    diagnostics: dict[str, Any]
    generated_at: datetime


class PromptStylizerAdapter(MCPToolAdapter):
    name = "creative/stylizer"
    description = "Rewrites prompts with brand-aligned tone and actionable suggestions."
    labels = ("creative", "tone")
    aliases = ("creative.tonecheck",)
    capabilities = ("creative", "prompt_styling")
    InputModel = PromptStylizerInput
    OutputModel = PromptStylizerOutput

    async def _invoke(self, payload_model: PromptStylizerInput) -> dict[str, Any]:
        tone = (payload_model.tone or "vibrant").lower()
        audience = payload_model.audience or "general audience"
        suggestions = [
            f"Emphasize benefits relevant to {audience}.",
            f"Apply a {tone} voice throughout.",
            "Close with a memorable hook.",
        ]
        styled = self._rewrite(payload_model.prompt, tone=tone, audience=audience)
        diagnostics = {
            "tone": tone,
            "audience": audience,
            "length_delta": len(styled) - len(payload_model.prompt),
        }
        return {
            "styled_prompt": styled,
            "suggestions": suggestions,
            "diagnostics": diagnostics,
            "generated_at": datetime.now(UTC),
        }

    def _rewrite(self, prompt: str, *, tone: str, audience: str) -> str:
        lines = textwrap.wrap(prompt.strip(), width=120)
        preface = f"[Tone: {tone}; Audience: {audience}]"
        return "\n".join([preface, *lines])


class ToneCheckInput(BaseModel):
    content: str = Field(..., min_length=5, max_length=4_000)
    desired_tone: str = Field("friendly", max_length=64)

    model_config = {"extra": "forbid"}


class ToneIssue(BaseModel):
    sentence: str
    issue: str
    recommendation: str


class ToneCheckOutput(BaseModel):
    score: float
    issues: list[ToneIssue]
    normalized_tone: str


class ToneCheckerAdapter(MCPToolAdapter):
    name = "creative/tone_checker"
    description = "Evaluates copy tone and suggests improvements."
    labels = ("creative", "quality")
    aliases = ("creative.tone_checker",)
    capabilities = ("creative", "tone_analysis")
    InputModel = ToneCheckInput
    OutputModel = ToneCheckOutput

    async def _invoke(self, payload_model: ToneCheckInput) -> dict[str, Any]:
        desired = payload_model.desired_tone.lower()
        sentences = [segment.strip() for segment in payload_model.content.replace("\n", " ").split(".") if segment.strip()]
        issues: list[ToneIssue] = []
        for sentence in sentences:
            lower = sentence.lower()
            if desired not in lower:
                issues.append(
                    ToneIssue(
                        sentence=sentence,
                        issue=f"Tone diverges from requested '{desired}'.",
                        recommendation=f"Blend in language conveying a {desired} attitude.",
                    )
                )
        score = max(0.1, 1.0 - 0.2 * len(issues)) if issues else 0.95
        return {
            "score": round(score, 2),
            "issues": [issue.model_dump() for issue in issues],
            "normalized_tone": desired,
        }


class WhisperTranscriptionInput(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio payload.")
    media_type: str = Field("audio/wav")
    language_hint: str | None = Field(default=None, max_length=8)

    model_config = {"extra": "forbid"}


class WhisperTranscriptionOutput(BaseModel):
    transcript: str
    detected_language: str | None
    checksum: str


class WhisperTranscriptionAdapter(MCPToolAdapter):
    name = "creative/whisper_transcription"
    description = "Performs lightweight transcription using heuristic decoding."
    labels = ("creative", "audio")
    aliases = ("creative.transcribe",)
    capabilities = ("creative", "transcription")
    InputModel = WhisperTranscriptionInput
    OutputModel = WhisperTranscriptionOutput

    async def _invoke(self, payload_model: WhisperTranscriptionInput) -> dict[str, Any]:
        raw = base64.b64decode(payload_model.audio_base64, validate=True)
        checksum = hashlib.sha256(raw).hexdigest()
        try:
            transcript = raw.decode("utf-8")
        except UnicodeDecodeError:
            transcript = "[non-text audio payload]"
        language = payload_model.language_hint or ("en" if transcript.isascii() else None)
        return {
            "transcript": transcript.strip(),
            "detected_language": language,
            "checksum": checksum,
        }


class ImageGeneratorInput(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=512)
    style: str | None = Field(default=None, max_length=32)
    width: int = Field(512, ge=128, le=1024)
    height: int = Field(512, ge=128, le=1024)

    model_config = {"extra": "forbid"}


class ImageGeneratorOutput(BaseModel):
    mime_type: str
    image_base64: str
    prompt: str
    style: str | None
    generated_at: datetime


class ImageGeneratorAdapter(MCPToolAdapter):
    name = "creative/image_generator"
    description = "Returns placeholder imagery metadata for downstream visualization flows."
    labels = ("creative", "visual")
    aliases = ("creative.image",)
    capabilities = ("creative", "image_generation")
    InputModel = ImageGeneratorInput
    OutputModel = ImageGeneratorOutput

    _PLACEHOLDER_PNG = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    async def _invoke(self, payload_model: ImageGeneratorInput) -> dict[str, Any]:
        # image generation is not available offline; provide deterministic placeholder
        return {
            "mime_type": "image/png",
            "image_base64": self._PLACEHOLDER_PNG,
            "prompt": payload_model.prompt,
            "style": payload_model.style,
            "generated_at": datetime.now(UTC),
        }


CREATIVE_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    PromptStylizerAdapter,
    ToneCheckerAdapter,
    WhisperTranscriptionAdapter,
    ImageGeneratorAdapter,
)


__all__ = [
    "PromptStylizerAdapter",
    "ToneCheckerAdapter",
    "WhisperTranscriptionAdapter",
    "ImageGeneratorAdapter",
    "CREATIVE_ADAPTER_CLASSES",
]