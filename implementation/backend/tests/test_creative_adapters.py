import base64

import pytest

from app.mcp.adapters.creative import (
    ImageGeneratorAdapter,
    PromptStylizerAdapter,
    ToneCheckerAdapter,
    WhisperTranscriptionAdapter,
)


pytestmark = pytest.mark.asyncio


async def test_prompt_stylizer_adds_preface() -> None:
    adapter = PromptStylizerAdapter()

    result = await adapter.invoke({"prompt": "Launch new feature", "tone": "bold"})

    assert result["styled_prompt"].startswith("[Tone: bold; Audience:")
    assert len(result["suggestions"]) >= 2


async def test_tone_checker_reports_issue() -> None:
    adapter = ToneCheckerAdapter()
    content = "This report is formal and direct."

    result = await adapter.invoke({"content": content, "desired_tone": "friendly"})

    assert result["score"] < 1.0
    assert result["issues"], "Expected tone issues to be reported"


async def test_whisper_transcription_decodes_utf8() -> None:
    adapter = WhisperTranscriptionAdapter()
    encoded = base64.b64encode("hello world".encode("utf-8")).decode("ascii")

    result = await adapter.invoke({"audio_base64": encoded, "language_hint": "en"})

    assert result["transcript"] == "hello world"
    assert result["checksum"]


async def test_image_generator_returns_placeholder() -> None:
    adapter = ImageGeneratorAdapter()

    result = await adapter.invoke({"prompt": "Create futuristic city"})

    assert result["mime_type"] == "image/png"
    assert len(result["image_base64"]) > 0