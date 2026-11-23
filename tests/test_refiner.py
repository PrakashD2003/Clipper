"""Unit tests for ContextRefiner boundary refinement (Gemini Flash and Ollama)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from streamclipper.analysis.context_window import ContextWindow, TimelineSegment
from streamclipper.config import Config
from streamclipper.scoring.refiner import ContextRefiner, RefinedMoment


@pytest.fixture
def dummy_context() -> ContextWindow:
    return ContextWindow(
        center_time=100.0,
        window_start=50.0,
        window_end=150.0,
        segments=[
            TimelineSegment(start=95.0, end=105.0, streamer_speech="I got the objective!"),
        ],
        total_chat_messages=12,
        avg_chat_rate=0.12,
        burst_timestamps=[98.0],
    )


def test_parse_response_clean_json() -> None:
    raw = json.dumps(
        {
            "exact_start": 94.5,
            "exact_end": 108.2,
            "peak_time": 99.0,
            "moment_type": "clutch",
            "title": "Clean Mission Pass",
            "description": "Streamer grabs the hard drive and survives.",
            "sub_clips": [{"start": 94.5, "end": 100.0, "label": "Setup"}],
            "reasoning": "Event triggers at 95s; chat explodes at 98s.",
            "confidence": 0.95,
        }
    )
    res = ContextRefiner._parse_response(raw)
    assert res.exact_start == 94.5
    assert res.exact_end == 108.2
    assert res.peak_time == 99.0
    assert res.moment_type == "clutch"
    assert res.title == "Clean Mission Pass"
    assert len(res.sub_clips) == 1
    assert res.confidence == 0.95


def test_parse_response_markdown_fences() -> None:
    raw = """```json
{
  "exact_start": 18796.0,
  "exact_end": 18840.0,
  "peak_time": 18810.0,
  "moment_type": "reaction",
  "title": "Speed Stunned by Rockstar Twist",
  "description": "Silent stunned cutscene.",
  "sub_clips": [
    {"start": 18796.0, "end": 18840.0, "label": "Cutscene"},
    {"start": 18880.0, "end": 18910.0, "label": "Reaction"}
  ],
  "reasoning": "Cutscene happens 80 seconds before scream.",
  "confidence": 0.99
}
```"""
    res = ContextRefiner._parse_response(raw)
    assert res.exact_start == 18796.0
    assert res.exact_end == 18840.0
    assert len(res.sub_clips) == 2


def test_parse_response_corrupt_falls_back_gracefully() -> None:
    res = ContextRefiner._parse_response("not valid json at all", default_start=100.0, default_end=130.0)
    assert res.exact_start == 100.0
    assert res.exact_end == 130.0
    assert res.sub_clips == []


def test_engine_resolution_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config()
    cfg._data.setdefault("refinement", {})["engine"] = "auto"
    cfg._data["refinement"]["gemini_api_key_env"] = "TEST_GEMINI_KEY"

    # Without key -> ollama
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)
    refiner = ContextRefiner(cfg)
    assert refiner._resolve_engine() == "ollama"

    # With key -> gemini
    monkeypatch.setenv("TEST_GEMINI_KEY", "dummy_key_12345")
    refiner = ContextRefiner(cfg)
    assert refiner._resolve_engine() == "gemini"


@pytest.mark.asyncio
async def test_refine_gemini_call(
    dummy_context: ContextWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config()
    cfg._data.setdefault("refinement", {})["engine"] = "gemini"
    cfg._data["refinement"]["gemini_api_key_env"] = "TEST_GEMINI_KEY"
    monkeypatch.setenv("TEST_GEMINI_KEY", "dummy_key")

    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "exact_start": 95.0,
        "exact_end": 115.0,
        "peak_time": 98.0,
        "moment_type": "hype",
        "title": "Gemini Refined Moment",
        "description": "Refined via Gemini Flash.",
        "confidence": 0.9,
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        refiner = ContextRefiner(cfg)
        result = await refiner.refine(dummy_context, proposed_start=90.0, proposed_end=120.0)
        assert result.exact_start == 95.0
        assert result.exact_end == 115.0
        assert result.title == "Gemini Refined Moment"
        assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_refine_ollama_call(
    dummy_context: ContextWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config()
    cfg._data.setdefault("refinement", {})["engine"] = "ollama"
    cfg._data.setdefault("llm", {})["model"] = "qwen3:4b"

    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {
            "content": json.dumps({
                "exact_start": 96.0,
                "exact_end": 116.0,
                "peak_time": 100.0,
                "moment_type": "fail",
                "title": "Ollama Refined Moment",
                "description": "Refined via Ollama.",
                "confidence": 0.85,
            })
        }
    }

    with patch("ollama.Client", return_value=mock_client):
        refiner = ContextRefiner(cfg)
        result = await refiner.refine(dummy_context, proposed_start=90.0, proposed_end=120.0)
        assert result.exact_start == 96.0
        assert result.exact_end == 116.0
        assert result.title == "Ollama Refined Moment"


@pytest.mark.asyncio
async def test_refine_mcp_engine_raises(dummy_context: ContextWindow) -> None:
    cfg = Config()
    cfg._data.setdefault("refinement", {})["engine"] = "mcp"
    refiner = ContextRefiner(cfg)
    with pytest.raises(ValueError, match="engine='mcp'"):
        await refiner.refine(dummy_context, 90.0, 120.0)


@pytest.mark.asyncio
async def test_refine_gemini_error_falls_back_to_ollama(
    dummy_context: ContextWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Gemini raises an exception in auto mode, it should fall back to Ollama."""
    cfg = Config()
    cfg._data.setdefault("refinement", {})["engine"] = "auto"
    cfg._data["refinement"]["gemini_api_key_env"] = "TEST_GEMINI_KEY"
    monkeypatch.setenv("TEST_GEMINI_KEY", "dummy_key")

    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = RuntimeError("Quota exhausted 429")

    mock_ollama = MagicMock()
    mock_ollama.chat.return_value = {
        "message": {
            "content": json.dumps({
                "exact_start": 94.0,
                "exact_end": 114.0,
                "peak_time": 100.0,
                "moment_type": "clutch",
                "title": "Ollama Fallback Pick",
                "description": "Fell back to Ollama.",
                "confidence": 0.8,
            })
        }
    }

    with (
        patch("google.genai.Client", return_value=mock_gemini),
        patch("ollama.Client", return_value=mock_ollama),
    ):
        refiner = ContextRefiner(cfg)
        res = await refiner.refine(dummy_context, proposed_start=90.0, proposed_end=120.0)
        assert res.exact_start == 94.0
        assert res.exact_end == 114.0
        assert res.title == "Ollama Fallback Pick"


@pytest.mark.asyncio
async def test_refine_both_fail_graceful_bounds_fallback(
    dummy_context: ContextWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both LLM engines fail, refine returns original bounds without crashing."""
    cfg = Config()
    cfg._data.setdefault("refinement", {})["engine"] = "auto"
    cfg._data["refinement"]["gemini_api_key_env"] = "TEST_GEMINI_KEY"
    monkeypatch.setenv("TEST_GEMINI_KEY", "dummy_key")

    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = RuntimeError("Gemini network error")

    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = ConnectionRefusedError("Ollama connection refused")

    with (
        patch("google.genai.Client", return_value=mock_gemini),
        patch("ollama.Client", return_value=mock_ollama),
    ):
        refiner = ContextRefiner(cfg)
        res = await refiner.refine(dummy_context, proposed_start=90.0, proposed_end=120.0)
        assert res.exact_start == 90.0
        assert res.exact_end == 120.0
        assert res.confidence == 0.0
        assert "Refinement failed" in res.reasoning


@pytest.mark.asyncio
async def test_refine_gemini_multimodal_call(
    dummy_context: ContextWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Gemini Flash multimodal call passes prompt + 2 contact sheet image parts."""
    cfg = Config()
    cfg._data.setdefault("refinement", {})["engine"] = "gemini"
    cfg._data["refinement"]["gemini_api_key_env"] = "TEST_GEMINI_KEY"
    monkeypatch.setenv("TEST_GEMINI_KEY", "dummy_key")

    s1 = tmp_path / "sheet_1.png"
    s2 = tmp_path / "sheet_2.png"
    s1.write_bytes(b"\x89PNG\r\n\x1a\nfake1")
    s2.write_bytes(b"\x89PNG\r\n\x1a\nfake2")
    dummy_context.contact_sheets = [s1, s2]

    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "exact_start": 92.0,
        "exact_end": 112.0,
        "peak_time": 98.0,
        "moment_type": "clutch",
        "title": "Visual Event Triggered",
        "confidence": 0.95,
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        refiner = ContextRefiner(cfg)
        result = await refiner.refine(dummy_context, proposed_start=90.0, proposed_end=120.0)

        assert result.exact_start == 92.0
        assert result.exact_end == 112.0
        assert result.title == "Visual Event Triggered"

        # Verify multimodal contents
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs.get("contents")
        assert isinstance(contents, list)
        assert len(contents) == 3  # [prompt, sheet1_part, sheet2_part]
        from google.genai import types
        assert isinstance(contents[1], types.Part)
        assert isinstance(contents[2], types.Part)


@pytest.mark.asyncio
async def test_refine_ollama_multimodal_call(
    dummy_context: ContextWindow, tmp_path: Path
) -> None:
    """Verify Ollama multimodal call passes images list and switches to vision model."""
    cfg = Config()
    cfg._data.setdefault("refinement", {})["engine"] = "ollama"
    cfg._data.setdefault("library", {}).setdefault("vision", {})["model"] = "qwen2.5vl:3b"

    s1 = tmp_path / "sheet_1.png"
    s2 = tmp_path / "sheet_2.png"
    s1.write_bytes(b"image1")
    s2.write_bytes(b"image2")
    dummy_context.contact_sheets = [s1, s2]

    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {
            "content": json.dumps({
                "exact_start": 91.0,
                "exact_end": 111.0,
                "peak_time": 95.0,
                "moment_type": "hype",
                "title": "Ollama Vision Refined",
                "confidence": 0.9,
            })
        }
    }

    with patch("ollama.Client", return_value=mock_client):
        refiner = ContextRefiner(cfg)
        result = await refiner.refine(dummy_context, proposed_start=90.0, proposed_end=120.0)

        assert result.exact_start == 91.0
        assert result.exact_end == 111.0

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["model"] == "qwen2.5vl:3b"
        msgs = call_kwargs["messages"]
        assert len(msgs) == 1
        assert "images" in msgs[0]
        assert msgs[0]["images"] == [s1, s2]


@pytest.mark.asyncio
async def test_refine_gemini_client_without_api_key_env(
    dummy_context: ContextWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When api key env is not set, genai.Client is called with default auth."""
    cfg = Config()
    cfg._data.setdefault("refinement", {})["engine"] = "gemini"
    cfg._data["refinement"]["gemini_api_key_env"] = "UNSET_KEY_ENV"
    monkeypatch.delenv("UNSET_KEY_ENV", raising=False)

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = json.dumps({
        "exact_start": 95.0,
        "exact_end": 115.0,
    })

    with patch("google.genai.Client", return_value=mock_client) as mock_cls:
        refiner = ContextRefiner(cfg)
        await refiner.refine(dummy_context, 90.0, 120.0)
        mock_cls.assert_called_once_with()



