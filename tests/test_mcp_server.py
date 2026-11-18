"""Unit tests for the StreamClipper MCP SDK v2 server (Package 5.1)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from mcp.server.mcpserver import MCPServer

from streamclipper.cache import StepCache
from streamclipper.config import Config, HardwareProfile
from streamclipper.mcp_server import main as mcp_main
from streamclipper.mcp_server import server as mcp_server
from streamclipper.mcp_server.server import mcp
from streamclipper.models import (
    ClipResult,
    Highlight,
    PipelineState,
    PipelineStatus,
    Platform,
    SignalType,
    StreamInfo,
    TimeRange,
    TranscriptSegment,
    TranscriptionResult,
    WordTimestamp,
)

EXPECTED_TOOLS = {
    "discover_streams",
    "download_stream_audio",
    "transcribe_audio",
    "analyze_signals",
    "get_transcript_chunks",
    "submit_highlights",
    "render_clip",
    "render_all_selected",
    "get_pipeline_status",
    "generate_post_metadata",
    "cleanup_temp",
}


def _cpu_profile() -> HardwareProfile:
    return HardwareProfile(
        tier="cpu_only",
        gpu_name="CPU",
        vram_gb=0.0,
        compute_capability="N/A",
        has_tensor_cores=False,
        has_nvenc=False,
        device="cpu",
        model_size="base",
        compute_type="int8",
        batch_size=4,
        llm_model="qwen3:4b",
        video_codec="libx264",
    )


def _make_config(tmp_path: Path) -> Config:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "general:",
                f"  output_dir: '{(tmp_path / 'output').as_posix()}'",
                f"  temp_dir: '{(tmp_path / 'temp').as_posix()}'",
                f"  cache_dir: '{(tmp_path / '.cache').as_posix()}'",
                "metadata:",
                "  hashtag_count: 10",
                "  include_streamer_tag: true",
                "  include_game_tag: true",
            ]
        ),
        encoding="utf-8",
    )
    return Config(user_config_path=cfg_file, hardware_profile=_cpu_profile())


def _extract_json(call_result: Any) -> dict[str, Any]:
    """Decode the JSON string payload returned in an MCP SDK v2 CallToolResult."""
    assert getattr(call_result, "is_error", False) is False
    assert call_result.content, "Expected non-empty content in CallToolResult"
    text = call_result.content[0].text
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    return parsed


def _make_sample_transcript() -> TranscriptionResult:
    """Create a 36-minute transcript spanning 3 distinct 12-minute windows."""
    segments = [
        TranscriptSegment(
            text="Whoa look at the water physics in Vice City!",
            start=100.0,
            end=115.0,
            words=[
                WordTimestamp("Whoa", 100.0, 101.0, 0.99),
                WordTimestamp("look", 101.0, 102.0, 0.98),
                WordTimestamp("at", 102.0, 103.0, 0.97),
                WordTimestamp("the", 103.0, 104.0, 0.97),
                WordTimestamp("water", 104.0, 106.0, 0.99),
                WordTimestamp("physics", 106.0, 110.0, 0.99),
            ],
        ),
        TranscriptSegment(
            text="Just driving quietly down the highway nothing happening.",
            start=900.0,
            end=920.0,
            words=[
                WordTimestamp("Just", 900.0, 902.0, 0.95),
                WordTimestamp("driving", 902.0, 906.0, 0.95),
            ],
        ),
        TranscriptSegment(
            text="NO WAY THAT HELICOPTER JUST EXPLODED ON THE ROOF!",
            start=1800.0,
            end=1840.0,
            words=[
                WordTimestamp("NO", 1800.0, 1802.0, 0.99),
                WordTimestamp("WAY", 1802.0, 1805.0, 0.99),
                WordTimestamp("HELICOPTER", 1810.0, 1818.0, 0.98),
                WordTimestamp("EXPLODED", 1820.0, 1835.0, 0.98),
            ],
        ),
    ]
    full_text = " ".join(s.text for s in segments)
    return TranscriptionResult(segments=segments, full_text=full_text, language="en")


@pytest.fixture
def mcp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Config, StepCache, Path]:
    """Patch MCP server config loader to use isolated temp/cache/output dirs."""
    config = _make_config(tmp_path)
    cache = StepCache(config)
    monkeypatch.setattr(mcp_server, "_get_config", lambda: config)
    return config, cache, tmp_path


# ---------------------------------------------------------------------------
# 1. MCP SDK v2 Server Registration & Schema Verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_server_registration_and_schemas() -> None:
    """Verify MCPServer instance and all 11 registered tools with valid input_schema."""
    assert isinstance(mcp, MCPServer)
    assert mcp.name == "streamclipper"

    tools = await mcp.list_tools()
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name.keys()) == EXPECTED_TOOLS
    assert len(tools) == 11

    for name, tool in by_name.items():
        schema = tool.input_schema
        assert isinstance(schema, dict), f"{name} missing input_schema dict"
        assert schema.get("type") == "object", f"{name} schema type must be object"
        assert isinstance(schema.get("properties"), dict), f"{name} missing properties"

    # Verify key defaults and YPP parameters in JSON schemas
    chunks_props = by_name["get_transcript_chunks"].input_schema["properties"]
    assert chunks_props["only_with_signals"]["default"] is True
    assert chunks_props["chunk_minutes"]["default"] == 12
    assert "audio_path" in chunks_props

    render_props = by_name["render_clip"].input_schema["properties"]
    for param in (
        "url",
        "start",
        "end",
        "hook_title",
        "moment_type",
        "narration_intro",
        "narration_outro",
        "context_badge",
        "sub_clips_json",
        "mode",
        "voice_intro_path",
        "voice_outro_path",
    ):
        assert param in render_props, f"render_clip missing parameter {param}"
    assert render_props["mode"]["default"] == "auto"

    meta_props = by_name["generate_post_metadata"].input_schema["properties"]
    assert meta_props["use_ollama"]["default"] is False


def test_mcp_main_invokes_mcp_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() delegates directly to mcp.run()."""
    run_mock = MagicMock()
    monkeypatch.setattr(mcp, "run", run_mock)
    mcp_main()
    run_mock.assert_called_once_with()


# ---------------------------------------------------------------------------
# 2. discover_streams
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_streams_success_and_error(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_streams = [
        StreamInfo(
            url="https://www.youtube.com/watch?v=vid12345678",
            title="GTA 6 Launch Stream",
            platform=Platform.YOUTUBE,
            streamer="TimTheTatman",
            game="GTA 6",
            duration_seconds=14400,
            view_count=85000,
            video_id="vid12345678",
        )
    ]

    class FakeDiscovery:
        def __init__(self, config: Config) -> None:
            self.config = config

        def discover(self, game: str | None = None, max_streams: int | None = None):
            assert game == "GTA 6"
            assert max_streams == 3
            return sample_streams

    monkeypatch.setattr(mcp_server, "YouTubeDiscovery", FakeDiscovery)

    res = await mcp.call_tool("discover_streams", {"game": "GTA 6", "max_results": 3})
    data = _extract_json(res)
    assert data["count"] == 1
    assert data["streams"][0]["video_id"] == "vid12345678"
    assert data["streams"][0]["view_count"] == 85000

    # Negative max_results returns structured JSON error
    bad_res = await mcp.call_tool("discover_streams", {"game": "GTA 6", "max_results": -1})
    bad_data = _extract_json(bad_res)
    assert "error" in bad_data


# ---------------------------------------------------------------------------
# 3. download_stream_audio + transcribe_audio + analyze_signals + StepCache linkage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_transcribe_analyze_via_audio_path_linkage(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify download -> transcribe (by audio_path) -> analyze (by audio_path) -> chunks (by audio_path)."""
    _, cache, tmp_path = mcp_env
    url = "https://www.youtube.com/watch?v=abc12345678"
    wav_path = tmp_path / "temp" / "abc12345678.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(b"RIFF0000WAVE")

    stream_info = StreamInfo(
        url=url,
        title="GTA 6 First Look",
        platform=Platform.YOUTUBE,
        streamer="Shroud",
        game="GTA 6",
        duration_seconds=3600,
        view_count=50000,
        video_id="abc12345678",
    )

    monkeypatch.setattr(
        mcp_server, "download_audio", lambda u, d: (wav_path, stream_info)
    )

    # Step 1: download_stream_audio persists StepCache with audio_path
    dl_res = await mcp.call_tool("download_stream_audio", {"url": url})
    dl_data = _extract_json(dl_res)
    assert dl_data["audio_path"] == str(wav_path)
    assert dl_data["status"] == "transcribing"

    cached_after_dl = cache.find_by_audio_path(wav_path)
    assert cached_after_dl is not None
    assert cached_after_dl.stream_info.url == url

    # Step 2: transcribe_audio resolves cache via audio_path ONLY (no url passed)
    transcript = _make_sample_transcript()

    class FakeTranscriber:
        def __init__(self, cfg: Config) -> None:
            pass

        def transcribe(self, path: Path) -> TranscriptionResult:
            assert path == wav_path
            return transcript

    monkeypatch.setattr(mcp_server, "Transcriber", FakeTranscriber)

    tr_res = await mcp.call_tool("transcribe_audio", {"audio_path": str(wav_path)})
    tr_data = _extract_json(tr_res)
    assert tr_data["url"] == url
    assert tr_data["cached"] is True
    assert tr_data["segment_count"] == 3
    assert tr_data["status"] == "analyzing"

    loaded = cache.load_state(url)
    assert loaded is not None and loaded.transcript is not None
    assert len(loaded.transcript.segments) == 3

    # Step 3: analyze_signals resolves cache via audio_path ONLY and saves loudness + chat spikes
    loud_spikes = [
        TimeRange(
            start=102.0,
            end=116.0,
            score=9.2,
            signal_type=SignalType.LOUDNESS,
            metadata={"peak_time": 108.5},
        ),
        TimeRange(
            start=1810.0,
            end=1830.0,
            score=8.8,
            signal_type=SignalType.LOUDNESS,
            metadata={"peak_time": 1822.0},
        ),
    ]
    chat_spikes = [
        TimeRange(
            start=1805.0,
            end=1835.0,
            score=9.5,
            signal_type=SignalType.CHAT_VELOCITY,
            metadata={
                "sample_messages": [
                    {"offset_seconds": 1815.0, "author": "fan1", "message": "NO WAY LMAO"},
                    {"offset_seconds": 1820.0, "author": "fan2", "message": "CLIP IT"},
                ]
            },
        )
    ]

    class FakeLoudness:
        def __init__(self, cfg: Config) -> None:
            pass

        def detect(self, path: Path) -> list[TimeRange]:
            assert path == wav_path
            return loud_spikes

    class FakeChatScraper:
        def load_cache(self, path: Path):
            return None

        def scrape(self, stream_url: str, temp_dir: Path) -> pd.DataFrame:
            assert stream_url == url
            return pd.DataFrame(
                [{"offset_seconds": 1815.0, "author": "fan1", "message": "NO WAY"}]
            )

        def save_cache(self, df: pd.DataFrame, path: Path) -> None:
            pass

    class FakeChatAnalyzer:
        def __init__(self, cfg: Config) -> None:
            pass

        def analyze(self, df: pd.DataFrame) -> list[TimeRange]:
            return chat_spikes

    monkeypatch.setattr(mcp_server, "LoudnessDetector", FakeLoudness)
    monkeypatch.setattr(mcp_server, "ChatScraper", FakeChatScraper)
    monkeypatch.setattr(mcp_server, "ChatVelocityAnalyzer", FakeChatAnalyzer)

    sig_res = await mcp.call_tool("analyze_signals", {"audio_path": str(wav_path)})
    sig_data = _extract_json(sig_res)
    assert sig_data["url"] == url
    assert sig_data["loudness_spike_count"] == 2
    assert sig_data["chat_spike_count"] == 1
    assert sig_data["overlap_count"] == 1
    assert sig_data["overlap_moments"][0]["start"] == 1810.0
    assert sig_data["overlap_moments"][0]["end"] == 1830.0
    assert sig_data["status"] == "scoring"

    # Step 4: get_transcript_chunks with only_with_signals=True filters dead-air middle chunk
    ch_res = await mcp.call_tool(
        "get_transcript_chunks",
        {"audio_path": str(wav_path), "chunk_minutes": 12, "only_with_signals": True},
    )
    ch_data = _extract_json(ch_res)
    assert ch_data["url"] == url
    assert ch_data["total_chunks"] == 3
    assert ch_data["returned_chunks"] == 2
    assert ch_data["filtered_out_chunks"] == 1
    assert len(ch_data["chunks"]) == 2
    assert "water physics" in ch_data["chunks"][0]["text"]
    assert len(ch_data["chunks"][0]["loudness_spikes"]) == 1
    assert "HELICOPTER" in ch_data["chunks"][1]["text"]
    assert len(ch_data["chunks"][1]["chat_spikes"]) == 1

    # With only_with_signals=False, all 3 chunks including quiet highway segment are returned
    all_ch_res = await mcp.call_tool(
        "get_transcript_chunks",
        {"url": url, "chunk_minutes": 12, "only_with_signals": False},
    )
    all_ch_data = _extract_json(all_ch_res)
    assert all_ch_data["returned_chunks"] == 3
    assert all_ch_data["filtered_out_chunks"] == 0


# ---------------------------------------------------------------------------
# 4. submit_highlights (YPP fields, dynamic fusion, cache persistence)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_highlights_parses_ypp_and_persists_cache(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    _, cache, _ = mcp_env
    url = "https://www.youtube.com/watch?v=hl123456789"
    state = PipelineState(
        stream_info=StreamInfo(
            url=url,
            title="GTA 6 Chaos",
            platform=Platform.YOUTUBE,
            streamer="KaiCenat",
            game="GTA 6",
            duration_seconds=3600,
            video_id="hl123456789",
        ),
        status=PipelineStatus.SCORING,
        transcript=_make_sample_transcript(),
        loudness_spikes=[
            TimeRange(
                start=1810.0,
                end=1830.0,
                score=9.0,
                signal_type=SignalType.LOUDNESS,
                metadata={"peak_time": 1821.5},
            )
        ],
        chat_spikes=[],  # Empty chat triggers dynamic weight renormalization
        analysis_complete=True,
    )
    cache.save_state(state)

    payload = json.dumps(
        [
            {
                "start": 1795.0,
                "end": 1845.0,
                "score": 9.2,
                "title": "HELICOPTER ROOF WIPEOUT!",
                "reason": "Insane explosion during heist escape",
                "best_quote": "NO WAY THAT HELICOPTER JUST EXPLODED!",
                "moment_type": "hype",
                "narration_intro": "Watch what happens three seconds after he lands on the roof.",
                "narration_outro": "Nobody in chat saw that chopper coming.",
                "context_badge": "🔥 Chaos Escape",
                "sub_clips": [
                    {"start": 1795.0, "end": 1805.0, "label": "Setup"},
                    {"start": 1815.0, "end": 1840.0, "label": "Payoff"},
                ],
            },
            {
                "start": 95.0,
                "end": 125.0,
                "score": 8.6,
                "title": "GTA 6 Water Physics Look Unreal",
                "reason": "Quiet graphics detail moment",
                "moment_type": "graphics_detail",
                "narration_intro": "Look closely at how the boat wake bends around the dock.",
                "narration_outro": "Rockstar really spent a decade on this water engine.",
                "context_badge": "🎨 Insane Detail",
                # Invalid sub_clips (overlapping) should be cleared cleanly
                "sub_clips": [
                    {"start": 95.0, "end": 110.0, "label": "A"},
                    {"start": 105.0, "end": 120.0, "label": "Overlap"},
                ],
            },
            # Invalid entry (end <= start) should be counted as rejected
            {"start": 500.0, "end": 490.0, "score": 5.0, "title": "Invalid"},
        ]
    )

    res = await mcp.call_tool("submit_highlights", {"url": url, "highlights": payload})
    data = _extract_json(res)
    assert data["accepted_count"] == 2
    assert data["rejected_count"] == 1
    assert data["status"] == "assembling"

    # First highlight (highest final_score) has valid sub_clips, peak_time, and dynamically fused score
    first = data["highlights"][0]
    assert first["title"] == "HELICOPTER ROOF WIPEOUT!"
    assert first["moment_type"] == "hype"
    assert len(first["sub_clips"]) == 2
    assert first["peak_time"] == 1821.5
    # Dynamic fusion with chat_spikes=[] -> (0.50/0.75)*9.2 + (0.25/0.75)*9.0 = 9.1333...
    assert abs(first["final_score"] - ((0.50 / 0.75) * 9.2 + (0.25 / 0.75) * 9.0)) < 1e-4

    # Second highlight: quiet AI pick >= 8.5 receives quiet-moment floor (0.85 * 8.6 = 7.31)
    second = data["highlights"][1]
    assert second["moment_type"] == "graphics_detail"
    assert second["sub_clips"] == []
    assert abs(second["final_score"] - (8.6 * 0.85)) < 1e-4

    # Verify persisted to StepCache
    reloaded = cache.load_state(url)
    assert reloaded is not None
    assert reloaded.status == PipelineStatus.ASSEMBLING
    assert len(reloaded.selected_highlights) == 2
    assert len(reloaded.selected_highlights[0].word_timestamps) > 0


@pytest.mark.asyncio
async def test_submit_highlights_invalid_json_returns_error(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    res = await mcp.call_tool(
        "submit_highlights",
        {"url": "https://youtube.com/watch?v=bad12345678", "highlights": "{not valid json"},
    )
    data = _extract_json(res)
    assert "error" in data
    assert "Invalid highlights JSON" in data["error"]


# ---------------------------------------------------------------------------
# 5. render_clip & render_all_selected (YPP modes + offline without Ollama)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_clip_story_narrative_raw_and_voice_overrides(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, tmp_path = mcp_env
    url = "https://www.youtube.com/watch?v=rnd12345678"

    # Ensure OllamaScorer is NEVER instantiated during MCP rendering
    def _forbid_ollama(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("OllamaScorer must not be called during MCP render_clip!")

    monkeypatch.setattr(mcp_server, "OllamaScorer", _forbid_ollama)

    seg_a = tmp_path / "temp" / "segments" / "rnd12345678_100_110.mp4"
    seg_b = tmp_path / "temp" / "segments" / "rnd12345678_120_140.mp4"
    seg_single = tmp_path / "temp" / "segments" / "rnd12345678_100_140.mp4"
    seg_a.parent.mkdir(parents=True, exist_ok=True)
    for p in (seg_a, seg_b, seg_single):
        p.write_bytes(b"mp4")

    multi_calls: list[list[tuple[float, float]]] = []
    single_calls: list[tuple[float, float]] = []
    assemble_calls: list[dict[str, Any]] = []

    def fake_download_multi(u, ranges, out_dir, buf):
        multi_calls.append(ranges)
        return [seg_a, seg_b]

    def fake_download_single(u, s, e, out_dir, buf):
        single_calls.append((s, e))
        return seg_single

    def fake_assemble(
        self,
        highlight,
        metadata,
        segment_paths,
        clip_index,
        mode,
        voice_intro_path,
        voice_outro_path,
        *,
        valid_beats=None,
        buffer_seconds=3.0,
    ):
        assemble_calls.append(
            {
                "mode": mode,
                "metadata": metadata,
                "segment_paths": segment_paths,
                "voice_intro_path": voice_intro_path,
                "voice_outro_path": voice_outro_path,
                "valid_beats": valid_beats,
            }
        )
        out_mp4 = tmp_path / "output" / f"clip_{clip_index:02d}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"rendered")
        return ClipResult(
            video_path=out_mp4,
            metadata_path=out_mp4.with_suffix(".json"),
            metadata=metadata,
            duration_seconds=30.0,
        )

    monkeypatch.setattr(mcp_server, "download_multiple_segments", fake_download_multi)
    monkeypatch.setattr(mcp_server, "download_segment", fake_download_single)
    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", fake_assemble)

    beats_json = json.dumps(
        [
            {"start": 100.0, "end": 110.0, "label": "Setup"},
            {"start": 120.0, "end": 140.0, "label": "Payoff"},
        ]
    )
    intro_wav = tmp_path / "my_intro.wav"
    outro_wav = tmp_path / "my_outro.wav"
    intro_wav.write_bytes(b"RIFF")
    outro_wav.write_bytes(b"RIFF")

    # 1) mode="auto" with 2 valid beats resolves to "story" and passes voiceover overrides
    story_res = await mcp.call_tool(
        "render_clip",
        {
            "url": url,
            "start": 100.0,
            "end": 140.0,
            "hook_title": "INSANE ESCAPE!",
            "game": "GTA 6",
            "streamer": "TimTheTatman",
            "moment_type": "clutch",
            "narration_intro": "Watch how he escapes five stars.",
            "narration_outro": "That jump saved the entire run.",
            "context_badge": "🔥 Clutch Escape",
            "sub_clips_json": beats_json,
            "mode": "auto",
            "voice_intro_path": str(intro_wav),
            "voice_outro_path": str(outro_wav),
        },
    )
    story_data = _extract_json(story_res)
    assert story_data["mode"] == "story"
    assert len(multi_calls) == 1
    assert assemble_calls[-1]["mode"] == "story"
    assert assemble_calls[-1]["voice_intro_path"] == intro_wav
    assert assemble_calls[-1]["voice_outro_path"] == outro_wav
    assert assemble_calls[-1]["metadata"].context_badge == "🔥 Clutch Escape"

    # 2) mode="narrative" ignores sub_clips and uses single cut
    narr_res = await mcp.call_tool(
        "render_clip",
        {
            "url": url,
            "start": 100.0,
            "end": 140.0,
            "hook_title": "CLEAN NARRATIVE",
            "narration_intro": "Here is the full uncut reaction.",
            "sub_clips_json": beats_json,
            "mode": "narrative",
        },
    )
    narr_data = _extract_json(narr_res)
    assert narr_data["mode"] == "narrative"
    assert len(single_calls) == 1
    assert assemble_calls[-1]["valid_beats"] == []

    # 3) mode="raw" clears narrative intro/outro/badge
    raw_res = await mcp.call_tool(
        "render_clip",
        {
            "url": url,
            "start": 100.0,
            "end": 140.0,
            "hook_title": "RAW CUT",
            "narration_intro": "Should be cleared",
            "narration_outro": "Should be cleared",
            "context_badge": "Should be cleared",
            "mode": "raw",
        },
    )
    raw_data = _extract_json(raw_res)
    assert raw_data["mode"] == "raw"
    raw_meta = assemble_calls[-1]["metadata"]
    assert raw_meta.narration_intro == ""
    assert raw_meta.narration_outro == ""
    assert raw_meta.context_badge == ""

    # 4) Invalid mode returns structured JSON error
    err_res = await mcp.call_tool(
        "render_clip",
        {
            "url": url,
            "start": 100.0,
            "end": 140.0,
            "hook_title": "BAD MODE",
            "mode": "turbo",
        },
    )
    err_data = _extract_json(err_res)
    assert "error" in err_data
    assert "Invalid transformation mode" in err_data["error"]


@pytest.mark.asyncio
async def test_render_all_selected_renders_and_completes_cache(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, cache, tmp_path = mcp_env
    url = "https://www.youtube.com/watch?v=all12345678"

    h1 = Highlight(
        start=100.0,
        end=145.0,
        score=9.0,
        title="Clip One",
        reason="Funny fail",
        final_score=9.0,
        moment_type="fail",
        narration_intro="Wait for the ending.",
        narration_outro="Instant karma.",
        context_badge="💀 Instant Karma",
    )
    h2 = Highlight(
        start=300.0,
        end=350.0,
        score=8.8,
        title="Clip Two",
        reason="Story cut",
        final_score=8.8,
        moment_type="clutch",
        sub_clips=[
            {"start": 300.0, "end": 310.0, "label": "Setup"},
            {"start": 330.0, "end": 350.0, "label": "Payoff"},
        ],
    )
    state = PipelineState(
        stream_info=StreamInfo(
            url=url,
            title="Full VOD",
            platform=Platform.YOUTUBE,
            streamer="xQc",
            game="GTA 6",
            duration_seconds=7200,
            video_id="all12345678",
        ),
        status=PipelineStatus.ASSEMBLING,
        highlights=[h1, h2],
        selected_highlights=[h1, h2],
    )
    cache.save_state(state)

    seg = tmp_path / "temp" / "segments" / "seg.mp4"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"mp4")

    monkeypatch.setattr(mcp_server, "download_segment", lambda *a, **kw: seg)
    monkeypatch.setattr(mcp_server, "download_multiple_segments", lambda *a, **kw: [seg, seg])

    def fake_assemble(self, highlight, metadata, segment_paths, clip_index, mode, *a, **kw):
        out_mp4 = tmp_path / "output" / f"clip_{clip_index:02d}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"mp4")
        return ClipResult(
            video_path=out_mp4,
            metadata_path=out_mp4.with_suffix(".json"),
            metadata=metadata,
            duration_seconds=35.0,
        )

    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", fake_assemble)

    res = await mcp.call_tool("render_all_selected", {"url": url, "mode": "auto"})
    data = _extract_json(res)
    assert data["rendered_count"] == 2
    assert data["status"] == "complete"
    assert data["clips"][0]["metadata"]["transformation_mode"] == "narrative"
    assert data["clips"][1]["metadata"]["transformation_mode"] == "story"

    reloaded = cache.load_state(url)
    assert reloaded is not None
    assert reloaded.status == PipelineStatus.COMPLETE
    assert len(reloaded.clips) == 2


# ---------------------------------------------------------------------------
# 6. get_pipeline_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_pipeline_status_empty_and_populated(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    _, cache, _ = mcp_env

    empty_res = await mcp.call_tool("get_pipeline_status", {})
    empty_data = _extract_json(empty_res)
    assert empty_data == {"count": 0, "pipelines": []}

    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url="https://youtube.com/watch?v=stat1111111",
                title="Stream A",
                platform=Platform.YOUTUBE,
                streamer="StreamerA",
                game="GTA 6",
                duration_seconds=3600,
                video_id="stat1111111",
            ),
            status=PipelineStatus.COMPLETE,
            analysis_complete=True,
        )
    )
    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url="https://youtube.com/watch?v=stat2222222",
                title="Stream B",
                platform=Platform.YOUTUBE,
                streamer="StreamerB",
                game="GTA 6",
                duration_seconds=7200,
                video_id="stat2222222",
            ),
            status=PipelineStatus.SCORING,
            analysis_complete=True,
        )
    )

    pop_res = await mcp.call_tool("get_pipeline_status", {})
    pop_data = _extract_json(pop_res)
    assert pop_data["count"] == 2
    statuses = {p["url"]: p["status"] for p in pop_data["pipelines"]}
    assert statuses["https://youtube.com/watch?v=stat1111111"] == "complete"
    assert statuses["https://youtube.com/watch?v=stat2222222"] == "scoring"


# ---------------------------------------------------------------------------
# 7. generate_post_metadata (offline default + optional Ollama)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_post_metadata_offline_default_and_ollama_opt_in(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Offline default must never touch OllamaScorer
    def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Ollama should not be called when use_ollama=False")

    monkeypatch.setattr(mcp_server, "OllamaScorer", _fail_if_called)

    res = await mcp.call_tool(
        "generate_post_metadata",
        {
            "hook_title": "HELICOPTER EXPLODED ON THE ROOF!",
            "streamer": "TimTheTatman",
            "game": "GTA 6",
            "description": "Wild five-star heist escape in Vice City",
        },
    )
    data = _extract_json(res)
    assert data["source"] == "offline"
    assert "GTA6" in data["hashtags"]
    assert "TimTheTatman" in data["hashtags"]
    assert len(data["hashtags"]) <= 10  # respects config metadata.hashtag_count=10
    assert all(not tag.startswith("#") for tag in data["hashtags"])
    assert "#GTA6" in data["hashtag_string"]
    assert "HELICOPTER EXPLODED ON THE ROOF!" in data["caption"]

    # Opt-in use_ollama=True uses OllamaScorer
    class FakeOllamaScorer:
        def __init__(self, cfg: Config) -> None:
            pass

        def _generate_caption_and_hashtags(self, h, streamer, game, hook_title):
            return "Ollama crafted caption!", ["GTA6", "TimTheTatman", "Viral"]

    monkeypatch.setattr(mcp_server, "OllamaScorer", FakeOllamaScorer)
    ollama_res = await mcp.call_tool(
        "generate_post_metadata",
        {
            "hook_title": "EPIC",
            "streamer": "TimTheTatman",
            "game": "GTA 6",
            "description": "Desc",
            "use_ollama": True,
        },
    )
    ollama_data = _extract_json(ollama_res)
    assert ollama_data["source"] == "ollama"
    assert ollama_data["caption"] == "Ollama crafted caption!"
    assert ollama_data["hashtags"] == ["GTA6", "TimTheTatman", "Viral"]


# ---------------------------------------------------------------------------
# 8. cleanup_temp (scoped stream, all_completed, and directory boundary safety)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_temp_stream_and_all_completed_safety(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    _, cache, tmp_path = mcp_env
    temp_dir = tmp_path / "temp"
    seg_dir = temp_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    url_done = "https://www.youtube.com/watch?v=done1111111"
    url_active = "https://www.youtube.com/watch?v=actv2222222"

    done_wav = temp_dir / "done1111111.wav"
    done_info = temp_dir / "done1111111.info.json"
    done_chat = temp_dir / "done1111111_chat.csv"
    done_live = temp_dir / "done1111111.live_chat.json"
    done_seg = seg_dir / "done1111111_100_140.mp4"
    active_wav = temp_dir / "actv2222222.wav"
    outside_file = tmp_path / "do_not_delete.wav"

    for f in (done_wav, done_info, done_chat, done_live, done_seg, active_wav, outside_file):
        f.write_bytes(b"12345")

    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url_done,
                title="Done Stream",
                platform=Platform.YOUTUBE,
                streamer="S1",
                game="GTA 6",
                duration_seconds=3600,
                video_id="done1111111",
            ),
            status=PipelineStatus.COMPLETE,
            audio_path=done_wav,
        )
    )
    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url_active,
                title="Active Stream",
                platform=Platform.YOUTUBE,
                streamer="S2",
                game="GTA 6",
                duration_seconds=3600,
                video_id="actv2222222",
            ),
            status=PipelineStatus.SCORING,
            audio_path=outside_file,  # Points outside temp_dir — must never be deleted
        )
    )

    # Empty url cleans only COMPLETE streams, leaving active stream untouched
    res_all = await mcp.call_tool("cleanup_temp", {})
    data_all = _extract_json(res_all)
    assert data_all["scope"] == "all_completed"
    assert data_all["cleaned_urls"] == [url_done]
    assert data_all["deleted_count"] == 5
    assert data_all["freed_bytes"] == 25
    assert not done_wav.exists()
    assert not done_seg.exists()
    assert active_wav.exists()
    assert outside_file.exists()

    # Specific url cleans that stream's files inside temp_dir, never deleting outside_file
    res_one = await mcp.call_tool("cleanup_temp", {"url": url_active})
    data_one = _extract_json(res_one)
    assert data_one["scope"] == "stream"
    assert data_one["deleted_count"] == 1
    assert not active_wav.exists()
    assert outside_file.exists()


# ---------------------------------------------------------------------------
# 9. Structured JSON Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unexpected_exception_returns_structured_json_error(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """All tools catch unexpected exceptions and return valid JSON with an 'error' key."""
    monkeypatch.setattr(
        mcp_server,
        "_get_config",
        lambda: (_ for _ in ()).throw(RuntimeError("Simulated disk failure")),
    )

    for tool_name, args in [
        ("discover_streams", {}),
        ("download_stream_audio", {"url": "https://youtube.com/watch?v=err12345678"}),
        ("transcribe_audio", {"url": "https://youtube.com/watch?v=err12345678"}),
        ("analyze_signals", {"url": "https://youtube.com/watch?v=err12345678"}),
        ("get_transcript_chunks", {"url": "https://youtube.com/watch?v=err12345678"}),
        ("submit_highlights", {"url": "https://youtube.com/watch?v=err12345678", "highlights": "[]"}),
        (
            "render_clip",
            {
                "url": "https://youtube.com/watch?v=err12345678",
                "start": 10.0,
                "end": 40.0,
                "hook_title": "Test",
            },
        ),
        ("render_all_selected", {"url": "https://youtube.com/watch?v=err12345678"}),
        ("get_pipeline_status", {}),
        (
            "generate_post_metadata",
            {"hook_title": "T", "streamer": "S", "game": "G", "description": "D"},
        ),
        ("cleanup_temp", {}),
    ]:
        res = await mcp.call_tool(tool_name, args)
        data = _extract_json(res)
        assert data == {"error": "Simulated disk failure"}, f"{tool_name} failed error contract"


@pytest.mark.asyncio
async def test_edge_cases_validation_and_cache_hydration(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify markdown-fenced submit_highlights, render_clip cache hydration, and validation error paths."""
    _, cache, tmp_path = mcp_env
    url = "https://www.youtube.com/watch?v=edg12345678"

    # Error paths on missing state / empty args / invalid bounds
    assert "error" in _extract_json(await mcp.call_tool("transcribe_audio", {}))
    assert "error" in _extract_json(await mcp.call_tool("analyze_signals", {}))
    assert "error" in _extract_json(await mcp.call_tool("get_transcript_chunks", {"url": url}))
    assert "error" in _extract_json(await mcp.call_tool("render_all_selected", {"url": url}))
    assert "error" in _extract_json(
        await mcp.call_tool(
            "render_clip",
            {"url": url, "start": 50.0, "end": 40.0, "hook_title": "Bad Bounds"},
        )
    )
    assert "error" in _extract_json(
        await mcp.call_tool(
            "render_clip",
            {
                "url": url,
                "start": 10.0,
                "end": 40.0,
                "hook_title": "Bad Beats",
                "sub_clips_json": '{"not": "array"}',
            },
        )
    )

    # Seed state with transcript + loudness + chat spikes, then test fenced submit_highlights
    state = PipelineState(
        stream_info=StreamInfo(
            url=url,
            title="Edge Stream",
            platform=Platform.YOUTUBE,
            streamer="Dr. Disrespect",
            game="GTA 6: Vice City!",
            duration_seconds=3600,
            video_id="edg12345678",
        ),
        status=PipelineStatus.SCORING,
        transcript=_make_sample_transcript(),
        loudness_spikes=[
            TimeRange(
                start=1810.0,
                end=1830.0,
                score=8.5,
                signal_type=SignalType.LOUDNESS,
                metadata={"peak_time": 1824.0},
            )
        ],
        chat_spikes=[
            TimeRange(
                start=1805.0,
                end=1835.0,
                score=9.0,
                signal_type=SignalType.CHAT_VELOCITY,
                metadata={
                    "sample_messages": [
                        {"offset_seconds": 1812.0, "author": "v1", "message": "INSANE!"}
                    ]
                },
            )
        ],
        analysis_complete=True,
    )
    cache.save_state(state)

    # Invalid chunk_minutes returns structured error
    assert "error" in _extract_json(
        await mcp.call_tool("get_transcript_chunks", {"url": url, "chunk_minutes": 0})
    )

    # Wrapped + markdown-fenced highlights JSON with unknown moment_type & hook_title alias
    fenced_payload = (
        "```json\n"
        + json.dumps(
            {
                "highlights": [
                    {
                        "start": 1795.0,
                        "end": 1845.0,
                        "score": 12.0,  # clamped to 10.0
                        "hook_title": "FENCED PICK",
                        "moment_type": "totally_unknown_type",
                    }
                ]
            }
        )
        + "\n```"
    )
    sub_data = _extract_json(
        await mcp.call_tool("submit_highlights", {"url": url, "highlights": fenced_payload})
    )
    assert sub_data["accepted_count"] == 1
    assert sub_data["highlights"][0]["title"] == "FENCED PICK"
    assert sub_data["highlights"][0]["score"] == 10.0
    assert sub_data["highlights"][0]["moment_type"] == "hype"

    # All-invalid payload returns error without wiping previously accepted highlights
    all_bad = _extract_json(
        await mcp.call_tool(
            "submit_highlights",
            {"url": url, "highlights": json.dumps([{"start": 20, "end": 10}])},
        )
    )
    assert "error" in all_bad
    assert len(cache.load_state(url).selected_highlights) == 1

    # render_clip hydrates word_timestamps, peak_time, chat_burst, game, streamer from StepCache
    seg = tmp_path / "temp" / "segments" / "edg12345678_1795_1845.mp4"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"mp4")
    monkeypatch.setattr(mcp_server, "download_segment", lambda *a, **kw: seg)

    captured: dict[str, Any] = {}

    def fake_assemble(self, highlight, metadata, segment_paths, clip_index, mode, *a, **kw):
        captured["highlight"] = highlight
        captured["metadata"] = metadata
        captured["clip_index"] = clip_index
        out_mp4 = tmp_path / "output" / f"clip_{clip_index:02d}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"mp4")
        return ClipResult(
            video_path=out_mp4,
            metadata_path=out_mp4.with_suffix(".json"),
            metadata=metadata,
            duration_seconds=50.0,
        )

    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", fake_assemble)

    rc_data = _extract_json(
        await mcp.call_tool(
            "render_clip",
            {
                "url": url,
                "start": 1795.0,
                "end": 1845.0,
                "hook_title": "HYDRATED CLIP",
            },
        )
    )
    assert rc_data["mode"] == "narrative"
    assert captured["clip_index"] == 1
    assert captured["highlight"].peak_time == 1824.0
    assert len(captured["highlight"].word_timestamps) == 4
    assert len(captured["highlight"].chat_burst) == 1
    assert captured["metadata"].streamer == "Dr. Disrespect"
    assert captured["metadata"].game == "GTA 6: Vice City!"
    assert "GTA6ViceCity" in captured["metadata"].hashtags
    assert "DrDisrespect" in captured["metadata"].hashtags
    assert len(cache.load_state(url).clips) == 1


# ---------------------------------------------------------------------------
# 10. Concurrency, Security, Partial Progress & Real Assembly Integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_transcribe_and_analyze_preserve_both_cache_updates(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent transcribe_audio + analyze_signals on the same URL must not clobber StepCache."""
    _, cache, tmp_path = mcp_env
    url = "https://www.youtube.com/watch?v=par12345678"
    wav_path = tmp_path / "temp" / "par12345678.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(b"RIFF0000WAVE")

    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url,
                title="Parallel Stream",
                platform=Platform.YOUTUBE,
                streamer="Tim",
                game="GTA 6",
                duration_seconds=3600,
                video_id="par12345678",
            ),
            status=PipelineStatus.TRANSCRIBING,
            audio_path=wav_path,
        )
    )

    class SlowTranscriber:
        def __init__(self, cfg: Config) -> None:
            pass

        def transcribe(self, path: Path) -> TranscriptionResult:
            time.sleep(0.15)
            return _make_sample_transcript()

    class FastLoudness:
        def __init__(self, cfg: Config) -> None:
            pass

        def detect(self, path: Path) -> list[TimeRange]:
            time.sleep(0.02)
            return [
                TimeRange(
                    start=1810.0,
                    end=1830.0,
                    score=9.1,
                    signal_type=SignalType.LOUDNESS,
                    metadata={"peak_time": 1820.0},
                )
            ]

    class FastChatScraper:
        def load_cache(self, path: Path):
            return None

        def scrape(self, stream_url: str, temp_dir: Path) -> pd.DataFrame:
            return pd.DataFrame(columns=["offset_seconds", "author", "message"])

        def save_cache(self, df: pd.DataFrame, path: Path) -> None:
            pass

    class FastChatAnalyzer:
        def __init__(self, cfg: Config) -> None:
            pass

        def analyze(self, df: pd.DataFrame) -> list[TimeRange]:
            return []

    monkeypatch.setattr(mcp_server, "Transcriber", SlowTranscriber)
    monkeypatch.setattr(mcp_server, "LoudnessDetector", FastLoudness)
    monkeypatch.setattr(mcp_server, "ChatScraper", FastChatScraper)
    monkeypatch.setattr(mcp_server, "ChatVelocityAnalyzer", FastChatAnalyzer)

    res_t, res_a = await asyncio.gather(
        mcp.call_tool("transcribe_audio", {"url": url}),
        mcp.call_tool("analyze_signals", {"url": url}),
    )
    data_t = _extract_json(res_t)
    data_a = _extract_json(res_a)
    assert "error" not in data_t
    assert "error" not in data_a

    final_state = cache.load_state(url)
    assert final_state is not None
    assert final_state.transcript is not None
    assert len(final_state.transcript.segments) == 3
    assert len(final_state.loudness_spikes) == 1
    assert final_state.analysis_complete is True
    assert final_state.status == PipelineStatus.SCORING


@pytest.mark.asyncio
async def test_gpu_lock_serializes_concurrent_transcribe_and_render(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shared GPU lock prevents concurrent WhisperX and video assembly from contending on VRAM."""
    _, _, tmp_path = mcp_env
    wav_path = tmp_path / "temp" / "gpu12345678.wav"
    seg_path = tmp_path / "temp" / "segments" / "gpu_seg.mp4"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(b"RIFF")
    seg_path.write_bytes(b"mp4")

    counter_lock = threading.Lock()
    active_gpu = 0
    max_concurrent_gpu = 0

    def _enter_gpu() -> None:
        nonlocal active_gpu, max_concurrent_gpu
        with counter_lock:
            active_gpu += 1
            max_concurrent_gpu = max(max_concurrent_gpu, active_gpu)

    def _exit_gpu() -> None:
        nonlocal active_gpu
        with counter_lock:
            active_gpu -= 1

    class GPUTranscriber:
        def __init__(self, cfg: Config) -> None:
            pass

        def transcribe(self, path: Path) -> TranscriptionResult:
            _enter_gpu()
            try:
                time.sleep(0.1)
                return _make_sample_transcript()
            finally:
                _exit_gpu()

    def gpu_assemble(self, highlight, metadata, segment_paths, clip_index, mode, *a, **kw):
        _enter_gpu()
        try:
            time.sleep(0.1)
            out_mp4 = tmp_path / "output" / f"clip_{clip_index:02d}.mp4"
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            out_mp4.write_bytes(b"mp4")
            return ClipResult(
                video_path=out_mp4,
                metadata_path=None,
                metadata=metadata,
                duration_seconds=15.0,
            )
        finally:
            _exit_gpu()

    monkeypatch.setattr(mcp_server, "Transcriber", GPUTranscriber)
    monkeypatch.setattr(mcp_server, "download_segment", lambda *a, **kw: seg_path)
    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", gpu_assemble)

    r1, r2 = await asyncio.gather(
        mcp.call_tool("transcribe_audio", {"audio_path": str(wav_path)}),
        mcp.call_tool(
            "render_clip",
            {
                "url": "https://www.youtube.com/watch?v=gpu12345678",
                "start": 10.0,
                "end": 25.0,
                "hook_title": "GPU TEST",
                "mode": "raw",
            },
        ),
    )
    assert "error" not in _extract_json(r1)
    assert "error" not in _extract_json(r2)
    assert max_concurrent_gpu == 1


@pytest.mark.asyncio
async def test_cleanup_temp_blocks_glob_injection_and_path_traversal(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    """cleanup_temp must resist glob wildcards ('*', '[a-z]*') and '../' path traversal."""
    _, cache, tmp_path = mcp_env
    temp_dir = tmp_path / "temp"
    seg_dir = temp_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    innocent_seg = seg_dir / "otherstream_10_20.mp4"
    innocent_chat = temp_dir / "otherstream.live_chat.json"
    outside_secret = tmp_path / "secret.wav"
    for f in (innocent_seg, innocent_chat, outside_secret):
        f.write_bytes(b"PROTECTED")

    for idx, malicious_vid in enumerate(("*", "[a-z]*stream", "../secret", r"..\secret")):
        bad_url = f"https://example.com/malicious-{idx}"
        cache.save_state(
            PipelineState(
                stream_info=StreamInfo(
                    url=bad_url,
                    title="Malicious",
                    platform=Platform.YOUTUBE,
                    streamer="Bad",
                    game="GTA 6",
                    duration_seconds=60,
                    video_id=malicious_vid,
                ),
                status=PipelineStatus.COMPLETE,
            )
        )
        res = _extract_json(await mcp.call_tool("cleanup_temp", {"url": bad_url}))
        assert "error" not in res
        assert res["deleted_count"] == 0

    assert innocent_seg.is_file()
    assert innocent_chat.is_file()
    assert outside_secret.is_file()


@pytest.mark.asyncio
async def test_render_all_selected_persists_partial_progress_on_mid_batch_failure(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-batch failure in render_all_selected persists rendered clips and reports partial output."""
    _, cache, tmp_path = mcp_env
    url = "https://www.youtube.com/watch?v=prt12345678"
    highlights = [
        Highlight(start=10.0, end=45.0, score=9.0, title="Clip 1", reason="First", final_score=9.0),
        Highlight(start=100.0, end=140.0, score=8.5, title="Clip 2", reason="Fails", final_score=8.5),
        Highlight(start=200.0, end=240.0, score=8.0, title="Clip 3", reason="Never", final_score=8.0),
    ]
    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url,
                title="Batch Fail",
                platform=Platform.YOUTUBE,
                streamer="S",
                game="GTA 6",
                duration_seconds=3600,
                video_id="prt12345678",
            ),
            status=PipelineStatus.ASSEMBLING,
            highlights=highlights,
            selected_highlights=highlights,
        )
    )

    seg = tmp_path / "temp" / "segments" / "prt.mp4"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"mp4")
    monkeypatch.setattr(mcp_server, "download_segment", lambda *a, **kw: seg)

    def flaky_assemble(self, highlight, metadata, segment_paths, clip_index, mode, *a, **kw):
        if clip_index == 2:
            raise RuntimeError("FFmpeg crashed on clip 2")
        out_mp4 = tmp_path / "output" / f"clip_{clip_index:02d}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"mp4")
        return ClipResult(
            video_path=out_mp4,
            metadata_path=None,
            metadata=metadata,
            duration_seconds=35.0,
        )

    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", flaky_assemble)

    data = _extract_json(await mcp.call_tool("render_all_selected", {"url": url}))
    assert "error" in data
    assert "FFmpeg crashed on clip 2" in data["error"]
    assert data["rendered_count"] == 1
    assert len(data["clips"]) == 1

    reloaded = cache.load_state(url)
    assert reloaded is not None
    assert reloaded.status == PipelineStatus.FAILED
    assert len(reloaded.clips) == 1


@pytest.mark.asyncio
async def test_submit_highlights_empty_resets_status_and_parses_string_timestamps(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    """submit_highlights handles MM:SS timestamps + explicit peak_time and resets stale status on []."""
    _, cache, _ = mcp_env
    url = "https://www.youtube.com/watch?v=str12345678"

    payload = json.dumps(
        [
            {
                "start": "01:30",
                "end": "02:20",
                "score": "8.9",
                "title": "MM:SS Beat Clip",
                "peak_time": "02:05",
                "sub_clips": [
                    {"start": "01:30", "end": "01:40", "label": "Setup"},
                    {"start": "01:55", "end": "02:20", "label": "Payoff"},
                ],
            }
        ]
    )
    sub = _extract_json(
        await mcp.call_tool("submit_highlights", {"url": url, "highlights": payload})
    )
    assert sub["accepted_count"] == 1
    assert sub["status"] == "assembling"
    hl = sub["highlights"][0]
    assert hl["start"] == 90.0
    assert hl["end"] == 140.0
    assert hl["peak_time"] == 125.0
    assert len(hl["sub_clips"]) == 2
    assert hl["sub_clips"][0]["start"] == 90.0
    assert hl["sub_clips"][1]["end"] == 140.0

    # Submitting empty list resets stale ASSEMBLING status instead of leaving ASSEMBLING with 0 clips
    cleared = _extract_json(
        await mcp.call_tool("submit_highlights", {"url": url, "highlights": "[]"})
    )
    assert cleared["accepted_count"] == 0
    assert cleared["status"] != "assembling"
    reloaded = cache.load_state(url)
    assert reloaded is not None
    assert reloaded.selected_highlights == []
    assert reloaded.status != PipelineStatus.ASSEMBLING


@pytest.mark.asyncio
async def test_render_clip_nonexistent_voice_path_and_uncached_collision_avoidance(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """render_clip validates voiceover paths exist and avoids overwriting clip_01 on uncached URLs."""
    config, _, tmp_path = mcp_env
    url = "https://www.youtube.com/watch?v=col12345678"

    # Missing voice_intro_path / voice_outro_path return structured errors
    err_intro = _extract_json(
        await mcp.call_tool(
            "render_clip",
            {
                "url": url,
                "start": 10.0,
                "end": 40.0,
                "hook_title": "Missing Intro",
                "voice_intro_path": str(tmp_path / "does_not_exist.wav"),
            },
        )
    )
    assert "error" in err_intro
    assert "voice_intro_path not found" in err_intro["error"]

    err_outro = _extract_json(
        await mcp.call_tool(
            "render_clip",
            {
                "url": url,
                "start": 10.0,
                "end": 40.0,
                "hook_title": "Missing Outro",
                "voice_outro_path": str(tmp_path / "does_not_exist_outro.wav"),
            },
        )
    )
    assert "error" in err_outro
    assert "voice_outro_path not found" in err_outro["error"]

    # Two uncached render_clip calls must allocate clip_01.mp4 then clip_02.mp4 (no overwrite)
    seg = tmp_path / "temp" / "segments" / "col.mp4"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"mp4")
    monkeypatch.setattr(mcp_server, "download_segment", lambda *a, **kw: seg)

    from streamclipper.output.organizer import FileOrganizer

    organizer = FileOrganizer(config)

    def org_assemble(self, highlight, metadata, segment_paths, clip_index, mode, *a, **kw):
        out_mp4 = organizer.get_clip_path(metadata.game, metadata.streamer, clip_index)
        out_mp4.write_bytes(f"clip-{clip_index}".encode("utf-8"))
        return ClipResult(
            video_path=out_mp4,
            metadata_path=None,
            metadata=metadata,
            duration_seconds=30.0,
        )

    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", org_assemble)

    c1 = _extract_json(
        await mcp.call_tool(
            "render_clip",
            {"url": url, "start": 10.0, "end": 40.0, "hook_title": "First", "mode": "raw"},
        )
    )
    c2 = _extract_json(
        await mcp.call_tool(
            "render_clip",
            {"url": url, "start": 50.0, "end": 80.0, "hook_title": "Second", "mode": "raw"},
        )
    )
    p1 = Path(c1["clip"]["video_path"])
    p2 = Path(c2["clip"]["video_path"])
    assert p1 != p2
    assert p1.name == "clip_01.mp4"
    assert p2.name == "clip_02.mp4"
    assert p1.read_bytes() == b"clip-1"
    assert p2.read_bytes() == b"clip-2"


@pytest.mark.asyncio
async def test_render_clip_real_assembly_end_to_end_narrative_and_story(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end MCP render_clip exercising real Pipeline._assemble_highlight (MoviePy/FFmpeg/ASS/Narrator)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed on PATH")

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "general:",
                f"  output_dir: '{(tmp_path / 'output').as_posix()}'",
                f"  temp_dir: '{(tmp_path / 'temp').as_posix()}'",
                f"  cache_dir: '{(tmp_path / '.cache').as_posix()}'",
                "video:",
                "  resolution: [360, 640]",
                "  fps: 15",
                "  codec: libx264",
            ]
        ),
        encoding="utf-8",
    )
    config = Config(user_config_path=cfg_file, hardware_profile=_cpu_profile())
    cache = StepCache(config)
    monkeypatch.setattr(mcp_server, "_get_config", lambda: config)

    seg_dir = tmp_path / "temp" / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_single = seg_dir / "e2e_single.mp4"
    seg_beat1 = seg_dir / "e2e_beat1.mp4"
    seg_beat2 = seg_dir / "e2e_beat2.mp4"
    voice_intro = tmp_path / "e2e_intro.wav"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=5:r=15",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(seg_single),
        ],
        check=True,
        capture_output=True,
    )
    for beat_path, freq in ((seg_beat1, 440), (seg_beat2, 550)):
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=green:s=640x360:d=11:r=15",
                "-f", "lavfi", "-i", f"sine=frequency={freq}:duration=11",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(beat_path),
            ],
            check=True,
            capture_output=True,
        )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=660:duration=1.2",
            "-ar", "24000", "-ac", "1",
            str(voice_intro),
        ],
        check=True,
        capture_output=True,
    )

    url = "https://www.youtube.com/watch?v=e2e12345678"
    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url,
                title="E2E Stream",
                platform=Platform.YOUTUBE,
                streamer="TimTheTatman",
                game="GTA 6",
                duration_seconds=600,
                video_id="e2e12345678",
            ),
            status=PipelineStatus.SCORING,
            transcript=TranscriptionResult(
                segments=[
                    TranscriptSegment(
                        text="Watch this crazy jump!",
                        start=100.0,
                        end=105.0,
                        words=[
                            WordTimestamp("Watch", 100.5, 101.5, 0.99),
                            WordTimestamp("jump!", 102.0, 104.0, 0.99),
                        ],
                    ),
                    TranscriptSegment(
                        text="Payoff explosion right now!",
                        start=120.0,
                        end=125.0,
                        words=[
                            WordTimestamp("Payoff", 120.5, 122.0, 0.99),
                            WordTimestamp("now!", 122.5, 124.5, 0.99),
                        ],
                    ),
                ],
                full_text="Watch this crazy jump! Payoff explosion right now!",
            ),
            loudness_spikes=[
                TimeRange(100.0, 105.0, 9.0, SignalType.LOUDNESS, {"peak_time": 102.5}),
                TimeRange(120.0, 125.0, 9.5, SignalType.LOUDNESS, {"peak_time": 122.5}),
            ],
            chat_spikes=[
                TimeRange(
                    100.0,
                    105.0,
                    8.5,
                    SignalType.CHAT_VELOCITY,
                    {
                        "sample_messages": [
                            {"offset_seconds": 101.5, "author": "fan1", "message": "NO WAY!"}
                        ]
                    },
                )
            ],
            analysis_complete=True,
        )
    )

    monkeypatch.setattr(mcp_server, "download_segment", lambda *a, **kw: seg_single)
    monkeypatch.setattr(
        mcp_server, "download_multiple_segments", lambda *a, **kw: [seg_beat1, seg_beat2]
    )

    # 1) Real narrative mode with custom voiceover WAV + ASS captions + ChatTicker + spike zoom
    narr = _extract_json(
        await mcp.call_tool(
            "render_clip",
            {
                "url": url,
                "start": 100.0,
                "end": 105.0,
                "hook_title": "CRAZY JUMP!",
                "moment_type": "hype",
                "narration_intro": "Watch this jump.",
                "context_badge": "Hype Moment",
                "mode": "narrative",
                "voice_intro_path": str(voice_intro),
            },
        )
    )
    assert "error" not in narr
    assert narr["mode"] == "narrative"
    narr_video = Path(narr["clip"]["video_path"])
    narr_meta = Path(narr["clip"]["metadata_path"])
    assert narr_video.is_file() and narr_video.stat().st_size > 1000
    assert narr_meta.is_file()
    meta_json = json.loads(narr_meta.read_text(encoding="utf-8"))
    assert meta_json["transformation_mode"] == "narrative"
    assert meta_json["context_badge"] == "Hype Moment"

    # 2) Real story mode with 2 beats stitched via compose_story_beats + remap_timeline_for_beats
    story = _extract_json(
        await mcp.call_tool(
            "render_clip",
            {
                "url": url,
                "start": 100.0,
                "end": 125.0,
                "hook_title": "STORY CUT!",
                "sub_clips_json": json.dumps(
                    [
                        {"start": 100.0, "end": 105.0, "label": "Setup"},
                        {"start": 120.0, "end": 125.0, "label": "Payoff"},
                    ]
                ),
                "mode": "auto",
            },
        )
    )
    assert "error" not in story
    assert story["mode"] == "story"
    story_video = Path(story["clip"]["video_path"])
    assert story_video.is_file() and story_video.stat().st_size > 1000
    assert story_video != narr_video
    assert abs(story["clip"]["duration_seconds"] - 10.0) <= 1.0


@pytest.mark.asyncio
async def test_analyze_signals_without_transcript_keeps_transcribing_status(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running analyze_signals before transcribe_audio must not advance status to SCORING."""
    _, cache, tmp_path = mcp_env
    url = "https://www.youtube.com/watch?v=ord12345678"
    wav_path = tmp_path / "temp" / "ord12345678.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(b"RIFF0000WAVE")

    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url,
                title="Order Test",
                platform=Platform.YOUTUBE,
                streamer="Tim",
                game="GTA 6",
                duration_seconds=3600,
                video_id="ord12345678",
            ),
            status=PipelineStatus.TRANSCRIBING,
            audio_path=wav_path,
        )
    )

    class FakeLoudness:
        def __init__(self, cfg: Config) -> None:
            pass

        def detect(self, path: Path) -> list[TimeRange]:
            return [
                TimeRange(
                    start=100.0,
                    end=115.0,
                    score=9.0,
                    signal_type=SignalType.LOUDNESS,
                    metadata={"peak_time": 108.0},
                )
            ]

    class FakeChatScraper:
        def load_cache(self, path: Path):
            return None

        def scrape(self, stream_url: str, temp_dir: Path) -> pd.DataFrame:
            return pd.DataFrame(columns=["offset_seconds", "author", "message"])

        def save_cache(self, df: pd.DataFrame, path: Path) -> None:
            pass

    class FakeChatAnalyzer:
        def __init__(self, cfg: Config) -> None:
            pass

        def analyze(self, df: pd.DataFrame) -> list[TimeRange]:
            return []

    class FakeTranscriber:
        def __init__(self, cfg: Config) -> None:
            pass

        def transcribe(self, path: Path) -> TranscriptionResult:
            return _make_sample_transcript()

    monkeypatch.setattr(mcp_server, "LoudnessDetector", FakeLoudness)
    monkeypatch.setattr(mcp_server, "ChatScraper", FakeChatScraper)
    monkeypatch.setattr(mcp_server, "ChatVelocityAnalyzer", FakeChatAnalyzer)
    monkeypatch.setattr(mcp_server, "Transcriber", FakeTranscriber)

    # 1) analyze_signals before transcribe_audio preserves TRANSCRIBING status
    sig = _extract_json(await mcp.call_tool("analyze_signals", {"url": url}))
    assert sig["status"] == "transcribing"
    after_sig = cache.load_state(url)
    assert after_sig is not None
    assert after_sig.analysis_complete is True
    assert after_sig.transcript is None
    assert after_sig.status == PipelineStatus.TRANSCRIBING

    # 2) Subsequent transcribe_audio promotes status cleanly to SCORING
    tr = _extract_json(await mcp.call_tool("transcribe_audio", {"url": url}))
    assert tr["status"] == "scoring"
    after_tr = cache.load_state(url)
    assert after_tr is not None
    assert after_tr.transcript is not None
    assert after_tr.analysis_complete is True
    assert after_tr.status == PipelineStatus.SCORING


@pytest.mark.asyncio
async def test_cleanup_temp_prevents_prefix_collision_across_streams(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    """cleanup_temp must not delete another stream's chat/segment files on ID prefix overlap."""
    _, cache, tmp_path = mcp_env
    temp_dir = tmp_path / "temp"
    seg_dir = temp_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    url_a = "https://example.com/stream-a"
    stream_a_chat = temp_dir / "stream.live_chat.json"
    stream_a_lang_chat = temp_dir / "stream.en.live_chat.json"
    stream_b_chat = temp_dir / "stream2.live_chat.json"
    stream_c_chat = temp_dir / "stream_2.live_chat.json"
    stream_a_seg = seg_dir / "stream_10_40.mp4"
    stream_a_sidecar = seg_dir / "stream_10_40.json"
    stream_b_seg = seg_dir / "stream_2_10_40.mp4"

    for f in (
        stream_a_chat,
        stream_a_lang_chat,
        stream_b_chat,
        stream_c_chat,
        stream_a_seg,
        stream_a_sidecar,
        stream_b_seg,
    ):
        f.write_bytes(b"DATA")

    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url_a,
                title="Stream A",
                platform=Platform.YOUTUBE,
                streamer="A",
                game="GTA 6",
                duration_seconds=600,
                video_id="stream",
            ),
            status=PipelineStatus.COMPLETE,
        )
    )

    res = _extract_json(await mcp.call_tool("cleanup_temp", {"url": url_a}))
    assert "error" not in res
    assert res["deleted_count"] == 4
    assert not stream_a_chat.exists()
    assert not stream_a_lang_chat.exists()
    assert not stream_a_seg.exists()
    assert not stream_a_sidecar.exists()
    # Other streams whose IDs share the "stream" prefix must remain untouched
    assert stream_b_chat.is_file()
    assert stream_c_chat.is_file()
    assert stream_b_seg.is_file()


@pytest.mark.asyncio
async def test_submit_highlights_preserves_explicit_peak_time_for_quiet_highlight(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    """Explicit peak_time must be preserved when stream has loudness spikes only outside the clip."""
    _, cache, _ = mcp_env
    url = "https://www.youtube.com/watch?v=pek12345678"
    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url,
                title="Peak Test",
                platform=Platform.YOUTUBE,
                streamer="Tim",
                game="GTA 6",
                duration_seconds=3600,
                video_id="pek12345678",
            ),
            status=PipelineStatus.SCORING,
            loudness_spikes=[
                TimeRange(
                    start=10.0,
                    end=20.0,
                    score=9.4,
                    signal_type=SignalType.LOUDNESS,
                    metadata={"peak_time": 15.0},
                )
            ],
            analysis_complete=True,
        )
    )

    payload = json.dumps(
        [
            {
                "start": 500.0,
                "end": 530.0,
                "score": 9.0,
                "title": "Quiet Punchline",
                "moment_type": "funny",
                "peak_time": 524.0,
            },
            {
                "start": 5.0,
                "end": 35.0,
                "score": 8.8,
                "title": "Loud Reaction",
                "moment_type": "hype",
                "peak_time": 29.0,
            },
        ]
    )
    res = _extract_json(
        await mcp.call_tool("submit_highlights", {"url": url, "highlights": payload})
    )
    assert res["accepted_count"] == 2
    by_title = {h["title"]: h for h in res["highlights"]}
    # Quiet highlight has no overlapping loudness spike -> honors explicit peak_time
    assert by_title["Quiet Punchline"]["peak_time"] == 524.0
    # Loud highlight overlaps spike at [10, 20] -> uses measured loudness peak_time
    assert by_title["Loud Reaction"]["peak_time"] == 15.0


def test_next_available_clip_index_terminates_for_constant_filename_template(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    """_next_available_clip_index must not hang when output.filename_template omits {index}."""
    from streamclipper.output.organizer import FileOrganizer

    config, _, _ = mcp_env
    config._data["output"]["filename_template"] = "{game}_{streamer}"
    organizer = FileOrganizer(config)
    existing = organizer.get_clip_path(game="GTA 6", streamer="Tim", index=1)
    existing.write_bytes(b"mp4")

    # Without a fixed-point check this would loop infinitely under _get_gpu_lock()
    idx = mcp_server._next_available_clip_index(config, "GTA 6", "Tim", min_index=1)
    assert idx == 1


@pytest.mark.asyncio
async def test_analyze_signals_blocks_chat_cache_path_traversal(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsafe stream_info.video_id must never write *_chat.csv outside general.temp_dir."""
    from streamclipper import pipeline as pipeline_mod
    from streamclipper.pipeline import Pipeline

    config, cache, _ = mcp_env
    temp_dir = Path(config.get("general.temp_dir"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = temp_dir / "safe_audio.wav"
    wav_path.write_bytes(b"RIFF")

    url = "https://www.twitch.tv/videos/traversal_test"
    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url,
                title="Traversal Test",
                platform=Platform.TWITCH,
                streamer="Tim",
                game="GTA 6",
                duration_seconds=60,
                video_id="../../escaped_chat",
            ),
            status=PipelineStatus.TRANSCRIBING,
            audio_path=wav_path,
        )
    )

    monkeypatch.setattr(mcp_server.LoudnessDetector, "detect", lambda self, p: [])
    monkeypatch.setattr(
        mcp_server.ChatScraper,
        "scrape",
        lambda self, u, d: pd.DataFrame(
            [{"offset_seconds": 1.0, "author": "u", "message": "pog"}]
        ),
    )
    monkeypatch.setattr(mcp_server.ChatVelocityAnalyzer, "analyze", lambda self, df: [])

    res = _extract_json(await mcp.call_tool("analyze_signals", {"url": url}))
    assert "error" not in res
    escaped_target = (temp_dir / "../../escaped_chat_chat.csv").resolve()
    assert not escaped_target.exists()
    expected_safe_csv = temp_dir / f"{cache.get_cache_key(url)}_chat.csv"
    assert expected_safe_csv.is_file()

    # Also verify Pipeline._step_analyze enforces the same containment
    expected_safe_csv.unlink()
    monkeypatch.setattr(pipeline_mod.LoudnessDetector, "detect", lambda self, p: [])
    monkeypatch.setattr(
        pipeline_mod.ChatScraper,
        "scrape",
        lambda self, u, d: pd.DataFrame(
            [{"offset_seconds": 1.0, "author": "u", "message": "pog"}]
        ),
    )
    monkeypatch.setattr(pipeline_mod.ChatVelocityAnalyzer, "analyze", lambda self, df: [])

    state = cache.load_state(url)
    assert state is not None
    pipe = Pipeline(config)
    await pipe._step_analyze(state)
    assert not escaped_target.exists()
    assert expected_safe_csv.is_file()


@pytest.mark.asyncio
async def test_submit_highlights_empty_gates_status_reset_on_transcript_and_analysis(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    """Clearing highlights via submit_highlights([]) must never advance status past missing prerequisites."""
    config, cache, _ = mcp_env
    temp_dir = Path(config.get("general.temp_dir"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = temp_dir / "gate1234567.wav"
    wav_path.write_bytes(b"RIFF")

    url = "https://www.youtube.com/watch?v=gate1234567"
    info = StreamInfo(
        url=url,
        title="Gate Test",
        platform=Platform.YOUTUBE,
        streamer="Tim",
        game="GTA 6",
        duration_seconds=120,
        video_id="gate1234567",
    )

    # Case 1: analysis_complete=True, transcript=None -> must reset to TRANSCRIBING (not SCORING)
    cache.save_state(
        PipelineState(
            stream_info=info,
            status=PipelineStatus.ASSEMBLING,
            audio_path=wav_path,
            transcript=None,
            analysis_complete=True,
        )
    )
    res1 = _extract_json(
        await mcp.call_tool("submit_highlights", {"url": url, "highlights": "[]"})
    )
    assert res1["status"] == "transcribing"
    reloaded1 = cache.load_state(url)
    assert reloaded1 is not None
    assert reloaded1.status is PipelineStatus.TRANSCRIBING

    # Case 2: transcript present, analysis_complete=False -> must reset to ANALYZING (not SCORING)
    transcript = TranscriptionResult(
        segments=[
            TranscriptSegment(
                text="hello world",
                start=0.0,
                end=5.0,
                words=[
                    WordTimestamp(text="hello", start=0.0, end=2.0),
                    WordTimestamp(text="world", start=2.1, end=5.0),
                ],
            )
        ],
        full_text="hello world",
        language="en",
    )
    cache.save_state(
        PipelineState(
            stream_info=info,
            status=PipelineStatus.ASSEMBLING,
            audio_path=wav_path,
            transcript=transcript,
            analysis_complete=False,
        )
    )
    res2 = _extract_json(
        await mcp.call_tool("submit_highlights", {"url": url, "highlights": "[]"})
    )
    assert res2["status"] == "analyzing"
    reloaded2 = cache.load_state(url)
    assert reloaded2 is not None
    assert reloaded2.status is PipelineStatus.ANALYZING

    # Case 3: both transcript and analysis_complete -> resets cleanly to SCORING
    cache.save_state(
        PipelineState(
            stream_info=info,
            status=PipelineStatus.ASSEMBLING,
            audio_path=wav_path,
            transcript=transcript,
            analysis_complete=True,
        )
    )
    res3 = _extract_json(
        await mcp.call_tool("submit_highlights", {"url": url, "highlights": "[]"})
    )
    assert res3["status"] == "scoring"
    reloaded3 = cache.load_state(url)
    assert reloaded3 is not None
    assert reloaded3.status is PipelineStatus.SCORING


@pytest.mark.asyncio
async def test_cleanup_temp_removes_orphaned_segments_after_wav_already_deleted(
    mcp_env: tuple[Config, StepCache, Path],
) -> None:
    """cleanup_temp must still remove orphaned segments/chat for non-YT streams after the WAV is gone."""
    config, cache, _ = mcp_env
    temp_dir = Path(config.get("general.temp_dir"))
    seg_dir = temp_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    twitch_url = "https://www.twitch.tv/videos/555444333"
    missing_wav = temp_dir / "v555444333.wav"  # already deleted earlier
    orphan_seg = seg_dir / "v555444333_10_40.mp4"
    orphan_chat = temp_dir / "v555444333_chat.csv"
    orphan_seg.write_bytes(b"mp4-bytes")
    orphan_chat.write_text("offset_seconds,author,message\n", encoding="utf-8")

    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=twitch_url,
                title="Twitch VOD",
                platform=Platform.TWITCH,
                streamer="Tim",
                game="GTA 6",
                duration_seconds=3600,
                video_id="",
            ),
            status=PipelineStatus.COMPLETE,
            audio_path=missing_wav,
        )
    )

    res = _extract_json(await mcp.call_tool("cleanup_temp", {"url": twitch_url}))
    assert "error" not in res
    assert res["deleted_count"] == 2
    assert not orphan_seg.exists()
    assert not orphan_chat.exists()


@pytest.mark.asyncio
async def test_submit_highlights_and_render_clip_midpoint_peak_time_without_loudness_spikes(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Midpoint peak_time fallback runs unconditionally when loudness_spikes is empty or uncached."""
    _, cache, tmp_path = mcp_env
    url = "https://www.youtube.com/watch?v=mid12345678"

    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url,
                title="Quiet Stream",
                platform=Platform.YOUTUBE,
                streamer="Kai",
                game="GTA 6",
                duration_seconds=3600,
                video_id="mid12345678",
            ),
            status=PipelineStatus.SCORING,
            loudness_spikes=[],
            chat_spikes=[],
            analysis_complete=True,
        )
    )

    payload = json.dumps(
        [
            {
                "start": 100.0,
                "end": 140.0,
                "score": 8.6,
                "title": "No Explicit Peak",
                "reason": "Quiet funny moment",
            },
            {
                "start": 200.0,
                "end": 260.0,
                "score": 8.8,
                "title": "Valid Explicit Peak",
                "reason": "Dry joke",
                "peak_time": 245.0,
            },
            {
                "start": 300.0,
                "end": 340.0,
                "score": 8.4,
                "title": "Out Of Range Explicit Peak",
                "reason": "Bad timestamp falls back to midpoint",
                "peak_time": 999.0,
            },
        ]
    )
    sub_res = _extract_json(
        await mcp.call_tool("submit_highlights", {"url": url, "highlights": payload})
    )
    assert "error" not in sub_res
    by_title = {h["title"]: h for h in sub_res["highlights"]}
    assert by_title["No Explicit Peak"]["peak_time"] == 120.0
    assert by_title["Valid Explicit Peak"]["peak_time"] == 245.0
    assert by_title["Out Of Range Explicit Peak"]["peak_time"] == 320.0

    # render_clip with cached state (empty loudness_spikes) AND with uncached URL
    seg = tmp_path / "temp" / "segments" / "mid.mp4"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"mp4")
    monkeypatch.setattr(mcp_server, "download_segment", lambda *a, **kw: seg)

    captured_peaks: list[float] = []

    def capture_assemble(self, highlight, metadata, segment_paths, clip_index, mode, *a, **kw):
        captured_peaks.append(highlight.peak_time)
        out_mp4 = tmp_path / "output" / f"mid_clip_{clip_index:02d}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"mp4")
        return ClipResult(
            video_path=out_mp4,
            metadata_path=None,
            metadata=metadata,
            duration_seconds=30.0,
        )

    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", capture_assemble)

    cached_render = _extract_json(
        await mcp.call_tool(
            "render_clip",
            {"url": url, "start": 100.0, "end": 140.0, "hook_title": "Cached Quiet", "mode": "raw"},
        )
    )
    assert "error" not in cached_render

    uncached_render = _extract_json(
        await mcp.call_tool(
            "render_clip",
            {
                "url": "https://www.youtube.com/watch?v=uncached999",
                "start": 50.0,
                "end": 90.0,
                "hook_title": "Uncached Quiet",
                "mode": "raw",
            },
        )
    )
    assert "error" not in uncached_render
    assert captured_peaks == [120.0, 70.0]


@pytest.mark.asyncio
async def test_concurrent_render_all_selected_and_render_clip_same_url_preserve_all_clips(
    mcp_env: tuple[Config, StepCache, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent render_all_selected + render_clip on the same URL must never overwrite each other."""
    config, cache, tmp_path = mcp_env
    seg = tmp_path / "temp" / "segments" / "conc.mp4"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"mp4")

    def slow_download(*a, **kw):
        time.sleep(0.03)
        return seg

    monkeypatch.setattr(mcp_server, "download_segment", slow_download)

    from streamclipper.output.organizer import FileOrganizer

    organizer = FileOrganizer(config)

    def org_assemble(self, highlight, metadata, segment_paths, clip_index, mode, *a, **kw):
        time.sleep(0.02)
        out_mp4 = organizer.get_clip_path(metadata.game, metadata.streamer, clip_index)
        out_mp4.write_bytes(f"{metadata.hook_title}-{clip_index}".encode("utf-8"))
        return ClipResult(
            video_path=out_mp4,
            metadata_path=None,
            metadata=metadata,
            duration_seconds=30.0,
        )

    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", org_assemble)

    # Test BOTH scheduling orders: batch-first and single-first
    for suffix, batch_first in (("orderA", True), ("orderB", False)):
        url = f"https://www.youtube.com/watch?v=cnc{suffix}1"
        h1 = Highlight(start=10.0, end=40.0, score=9.0, title="Batch 1", reason="r1", final_score=9.0)
        h2 = Highlight(start=60.0, end=90.0, score=8.7, title="Batch 2", reason="r2", final_score=8.7)
        cache.save_state(
            PipelineState(
                stream_info=StreamInfo(
                    url=url,
                    title=f"Concurrent {suffix}",
                    platform=Platform.YOUTUBE,
                    streamer=f"Streamer_{suffix}",
                    game="GTA 6",
                    duration_seconds=3600,
                    video_id=f"cnc{suffix}1",
                ),
                status=PipelineStatus.ASSEMBLING,
                highlights=[h1, h2],
                selected_highlights=[h1, h2],
            )
        )

        coro_batch = mcp.call_tool("render_all_selected", {"url": url, "mode": "raw"})
        coro_single = mcp.call_tool(
            "render_clip",
            {
                "url": url,
                "start": 200.0,
                "end": 230.0,
                "hook_title": "Bonus Single",
                "mode": "raw",
            },
        )
        if batch_first:
            r1, r2 = await asyncio.gather(coro_batch, coro_single)
            batch_res, single_res = _extract_json(r1), _extract_json(r2)
        else:
            r1, r2 = await asyncio.gather(coro_single, coro_batch)
            single_res, batch_res = _extract_json(r1), _extract_json(r2)

        assert "error" not in batch_res, batch_res
        assert "error" not in single_res, single_res
        assert batch_res["rendered_count"] == 2

        reloaded = cache.load_state(url)
        assert reloaded is not None
        assert len(reloaded.clips) == 3
        titles = {c.metadata.hook_title for c in reloaded.clips}
        assert titles == {"Batch 1", "Batch 2", "Bonus Single"}
        paths = {Path(c.video_path).resolve() for c in reloaded.clips}
        assert len(paths) == 3
        assert all(p.is_file() for p in paths)





