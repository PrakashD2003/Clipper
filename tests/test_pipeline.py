"""Unit and mocked end-to-end integration tests for the Pipeline orchestrator and CLI wiring."""
from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from rich.console import Console

import streamclipper.cli as cli_mod
import streamclipper.discovery.youtube as yt_mod
import streamclipper.pipeline as pipeline_mod
from streamclipper.cli import cli
from streamclipper.config import Config, HardwareProfile
from streamclipper.models import (
    ClipMetadata,
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
from streamclipper.pipeline import Pipeline, format_time


def _make_config(tmp_path: Path, overrides: dict[str, Any] | None = None) -> Config:
    """Create an isolated Config rooted in tmp_path with a deterministic HardwareProfile."""
    cfg_data: dict[str, Any] = {
        "general": {
            "cache_dir": str(tmp_path / ".cache"),
            "output_dir": str(tmp_path / "output"),
            "temp_dir": str(tmp_path / "temp"),
            "max_concurrent_streams": 2,
        }
    }
    if overrides:
        for section, vals in overrides.items():
            if section in cfg_data and isinstance(vals, dict):
                cfg_data[section].update(vals)
            else:
                cfg_data[section] = vals

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    hw = HardwareProfile(
        tier="cuda_mid_tensor",
        gpu_name="NVIDIA GeForce RTX 4050 Laptop GPU",
        vram_gb=6.44,
        compute_capability="8.9",
        has_tensor_cores=True,
        has_nvenc=True,
        device="cuda",
        model_size="large-v2",
        compute_type="int8_float16",
        batch_size=8,
        llm_model="qwen3:8b",
        video_codec="h264_nvenc",
    )
    return Config(user_config_path=cfg_path, hardware_profile=hw)


def _sample_stream_info(
    url: str = "https://youtube.com/watch?v=gta6_vod_01",
    streamer: str = "TimTheTatman",
    game: str = "GTA 6",
    video_id: str = "gta6_vod_01",
) -> StreamInfo:
    return StreamInfo(
        url=url,
        title="GTA 6 Launch Stream - Insane Heist!",
        platform=Platform.YOUTUBE,
        streamer=streamer,
        game=game,
        duration_seconds=7200,
        view_count=125000,
        video_id=video_id,
    )


def _sample_transcript() -> TranscriptionResult:
    w1 = WordTimestamp(text="WATCH", start=120.0, end=120.4, confidence=0.98)
    w2 = WordTimestamp(text="THIS!", start=120.5, end=121.0, confidence=0.97)
    return TranscriptionResult(
        segments=[
            TranscriptSegment(
                text="WATCH THIS!",
                start=120.0,
                end=121.0,
                words=[w1, w2],
            )
        ],
        full_text="WATCH THIS!",
        language="en",
    )


def _sample_highlights() -> list[Highlight]:
    return [
        Highlight(
            start=110.0,
            end=160.0,
            score=9.2,
            title="POLICE CHASE ROOFTOP JUMP",
            reason="Massive getaway leap across Vice City highway",
            best_quote="WATCH THIS!",
            loudness_score=8.8,
            chat_score=9.0,
            final_score=9.05,
            transcript_chunk="WATCH THIS!",
            word_timestamps=[
                WordTimestamp(text="WATCH", start=120.0, end=120.4, confidence=0.98),
                WordTimestamp(text="THIS!", start=120.5, end=121.0, confidence=0.97),
            ],
            moment_type="clutch",
            narration_intro="Five stars on his tail and zero escape routes left.",
            narration_outro="Somehow he threads the needle over six cop cars.",
            context_badge="🔥 Clutch Escape",
            peak_time=120.5,
            chat_burst=[{"author": "viewer1", "message": "NO WAY", "offset_seconds": 121.0}],
            sub_clips=[
                {"start": 110.0, "end": 124.0, "label": "Setup"},
                {"start": 140.0, "end": 160.0, "label": "Payoff"},
            ],
        ),
        Highlight(
            start=400.0,
            end=445.0,
            score=8.4,
            title="NPC KARMA MOMENT",
            reason="Instant karma after trash talking an NPC",
            best_quote="He can't catch me!",
            loudness_score=7.5,
            chat_score=8.1,
            final_score=8.1,
            transcript_chunk="He can't catch me!",
            moment_type="funny",
            narration_intro="Famous last words right before Vice City humbles him.",
            narration_outro="Chat did not let him live that one down.",
            context_badge="💀 Instant Karma",
            peak_time=425.0,
            sub_clips=[],
        ),
        Highlight(
            start=800.0,
            end=840.0,
            score=8.0,
            title="INSANE WATER PHYSICS",
            reason="Dynamic hurricane wave detail on the beach",
            best_quote="Look at those waves!",
            loudness_score=6.5,
            chat_score=7.8,
            final_score=7.58,
            transcript_chunk="Look at those waves!",
            moment_type="graphics_detail",
            narration_intro="Wait until you see what Rockstar did to the storm waves.",
            narration_outro="Even the spray reacts to the helicopter blades.",
            context_badge="🎨 Insane Detail",
            peak_time=820.0,
            sub_clips=[],
        ),
    ]


def _install_mock_stage_deps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    call_log: list[str],
    highlights: list[Highlight] | None = None,
    score_error: Exception | None = None,
    segment_calls: list[tuple[str, Any]] | None = None,
) -> Path:
    """Monkeypatch Phase 2 & Phase 3 collaborators imported in streamclipper.pipeline."""
    audio_file = tmp_path / "temp" / "gta6_vod_01.wav"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"RIFF....WAVE")

    def fake_download_audio(url: str, output_dir: Path) -> tuple[Path, StreamInfo]:
        call_log.append("download_audio")
        vid = url.split("=")[-1] if "=" in url else "vid001"
        path = Path(output_dir) / f"{vid}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF....WAVE")
        return path, _sample_stream_info(url=url, video_id=vid)

    class FakeTranscriber:
        def __init__(self, config: Config) -> None:
            self.config = config

        def transcribe(self, audio_path: Path) -> TranscriptionResult:
            call_log.append("transcribe")
            return _sample_transcript()

    class FakeLoudnessDetector:
        def __init__(self, config: Config) -> None:
            self.config = config

        def detect(self, audio_path: Path) -> list[TimeRange]:
            call_log.append("loudness")
            return [
                TimeRange(
                    start=118.0,
                    end=128.0,
                    score=9.1,
                    signal_type=SignalType.LOUDNESS,
                    metadata={"peak_time": 120.5},
                )
            ]

    class FakeChatScraper:
        def load_cache(self, path: Path) -> pd.DataFrame | None:
            return None

        def scrape(self, url: str, temp_dir: Path) -> pd.DataFrame:
            call_log.append("chat_scrape")
            return pd.DataFrame(
                [
                    {"offset_seconds": 120.0, "author": "viewer1", "message": "NO WAY"},
                    {"offset_seconds": 121.0, "author": "viewer2", "message": "INSANE"},
                ]
            )

        def save_cache(self, df: pd.DataFrame, path: Path) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)

    class FakeChatVelocityAnalyzer:
        def __init__(self, config: Config) -> None:
            self.config = config

        def analyze(self, chat_df: pd.DataFrame) -> list[TimeRange]:
            call_log.append("chat_velocity")
            return [
                TimeRange(
                    start=115.0,
                    end=130.0,
                    score=8.7,
                    signal_type=SignalType.CHAT_VELOCITY,
                    metadata={
                        "sample_messages": [
                            {"author": "viewer1", "message": "NO WAY", "offset_seconds": 120.0}
                        ]
                    },
                )
            ]

    hl_list = _sample_highlights() if highlights is None else highlights

    class FakeOllamaScorer:
        def __init__(self, config: Config) -> None:
            self.config = config

        def check_connection(self) -> bool:
            call_log.append("check_connection")
            if score_error is not None:
                raise score_error
            return True

        def score_and_select(
            self,
            transcript: TranscriptionResult,
            loudness_spikes: list[TimeRange],
            chat_spikes: list[TimeRange],
        ) -> list[Highlight]:
            call_log.append("score_and_select")
            return list(hl_list)

        def generate_metadata(
            self,
            highlight: Highlight,
            streamer: str,
            game: str,
            source_url: str,
        ) -> ClipMetadata:
            call_log.append("generate_metadata")
            return ClipMetadata(
                hook_title=highlight.title,
                caption=f"{streamer} in {game}!",
                hashtags=["GTA6", "Gaming"],
                streamer=streamer,
                game=game,
                source_url=source_url,
                clip_start=highlight.start,
                clip_end=highlight.end,
                highlight_score=highlight.final_score or highlight.score,
                moment_type=highlight.moment_type,
                narration_intro=highlight.narration_intro,
                narration_outro=highlight.narration_outro,
                context_badge=highlight.context_badge,
                transformation_mode=str(self.config.get("transformation.mode", "narrative")),
            )

    def fake_download_segment(
        url: str,
        start_seconds: float,
        end_seconds: float,
        output_dir: Path,
        buffer_seconds: float = 3.0,
    ) -> Path:
        call_log.append("download_segment")
        if segment_calls is not None:
            segment_calls.append(("single", (start_seconds, end_seconds)))
        p = Path(output_dir) / f"seg_{int(start_seconds)}_{int(end_seconds)}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"mp4-single")
        return p

    def fake_download_multiple_segments(
        url: str,
        segments: list[tuple[float, float]],
        output_dir: Path,
        buffer_seconds: float = 3.0,
    ) -> list[Path]:
        call_log.append("download_multiple_segments")
        if segment_calls is not None:
            segment_calls.append(("multi", list(segments)))
        paths: list[Path] = []
        for s, e in segments:
            p = Path(output_dir) / f"beat_{int(s)}_{int(e)}.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"mp4-beat")
            paths.append(p)
        return paths

    monkeypatch.setattr(pipeline_mod, "download_audio", fake_download_audio)
    monkeypatch.setattr(pipeline_mod, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(pipeline_mod, "LoudnessDetector", FakeLoudnessDetector)
    monkeypatch.setattr(pipeline_mod, "ChatScraper", FakeChatScraper)
    monkeypatch.setattr(pipeline_mod, "ChatVelocityAnalyzer", FakeChatVelocityAnalyzer)
    monkeypatch.setattr(pipeline_mod, "OllamaScorer", FakeOllamaScorer)
    monkeypatch.setattr(pipeline_mod, "download_segment", fake_download_segment)
    monkeypatch.setattr(
        pipeline_mod, "download_multiple_segments", fake_download_multiple_segments
    )
    return audio_file


# ---------------------------------------------------------------------------
# Helpers & _recover_failed_state Tests
# ---------------------------------------------------------------------------


class TestFormatTime:
    def test_formats_hours_minutes_seconds_and_clamps_negative(self) -> None:
        assert format_time(0) == "00:00:00"
        assert format_time(65.9) == "00:01:05"
        assert format_time(3661) == "01:01:01"
        assert format_time(-12.5) == "00:00:00"


class TestRecoverFailedState:
    def test_non_failed_states_are_unchanged(self) -> None:
        for status in (
            PipelineStatus.PENDING,
            PipelineStatus.DOWNLOADING,
            PipelineStatus.TRANSCRIBING,
            PipelineStatus.ANALYZING,
            PipelineStatus.SCORING,
            PipelineStatus.ASSEMBLING,
            PipelineStatus.COMPLETE,
        ):
            state = PipelineState(stream_info=_sample_stream_info(), status=status)
            recovered = Pipeline._recover_failed_state(state)
            assert recovered.status is status

    def test_failed_with_highlights_resets_to_assembling(self) -> None:
        state = PipelineState(
            stream_info=_sample_stream_info(),
            status=PipelineStatus.FAILED,
            highlights=_sample_highlights()[:1],
        )
        recovered = Pipeline._recover_failed_state(state)
        assert recovered.status is PipelineStatus.ASSEMBLING

    def test_failed_with_spikes_only_resets_to_scoring(self) -> None:
        state = PipelineState(
            stream_info=_sample_stream_info(),
            status=PipelineStatus.FAILED,
            transcript=_sample_transcript(),
            loudness_spikes=[TimeRange(start=10.0, end=20.0, score=8.0)],
        )
        recovered = Pipeline._recover_failed_state(state)
        assert recovered.status is PipelineStatus.SCORING

    def test_failed_with_chat_spikes_only_resets_to_scoring(self) -> None:
        state = PipelineState(
            stream_info=_sample_stream_info(),
            status=PipelineStatus.FAILED,
            chat_spikes=[
                TimeRange(
                    start=10.0,
                    end=20.0,
                    score=8.0,
                    signal_type=SignalType.CHAT_VELOCITY,
                )
            ],
        )
        recovered = Pipeline._recover_failed_state(state)
        assert recovered.status is PipelineStatus.SCORING

    def test_failed_with_transcript_only_resets_to_analyzing(self) -> None:
        state = PipelineState(
            stream_info=_sample_stream_info(),
            status=PipelineStatus.FAILED,
            transcript=_sample_transcript(),
        )
        recovered = Pipeline._recover_failed_state(state)
        assert recovered.status is PipelineStatus.ANALYZING

    def test_failed_with_existing_audio_resets_to_transcribing(self, tmp_path: Path) -> None:
        audio_path = tmp_path / "stream.wav"
        audio_path.write_bytes(b"wav")
        state = PipelineState(
            stream_info=_sample_stream_info(),
            status=PipelineStatus.FAILED,
            audio_path=audio_path,
        )
        recovered = Pipeline._recover_failed_state(state)
        assert recovered.status is PipelineStatus.TRANSCRIBING

    def test_failed_with_missing_audio_or_no_artifacts_resets_to_pending(
        self, tmp_path: Path
    ) -> None:
        missing_audio = tmp_path / "does_not_exist.wav"
        state = PipelineState(
            stream_info=_sample_stream_info(),
            status=PipelineStatus.FAILED,
            audio_path=missing_audio,
        )
        recovered = Pipeline._recover_failed_state(state)
        assert recovered.status is PipelineStatus.PENDING


# ---------------------------------------------------------------------------
# End-to-End Pipeline, Resume Caching, FAILED Retry & YPP Forwarding Tests
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline_transitions_pending_to_complete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        segment_calls: list[tuple[str, Any]] = []
        _install_mock_stage_deps(
            monkeypatch, tmp_path, call_log=call_log, segment_calls=segment_calls
        )

        config = _make_config(tmp_path)
        pipeline = Pipeline(config)
        url = "https://youtube.com/watch?v=gta6_vod_01"

        clips = await pipeline.process_stream(
            url,
            max_clips=2,
            game="GTA 6",
            streamer="CustomStreamer",
            interactive=False,
        )

        assert len(clips) == 2
        assert call_log == [
            "download_audio",
            "transcribe",
            "loudness",
            "chat_scrape",
            "chat_velocity",
            "check_connection",
            "score_and_select",
            "download_multiple_segments",  # highlight 1 has 2 sub_clips
            "generate_metadata",
            "download_segment",  # highlight 2 is single-cut
            "generate_metadata",
        ]
        assert segment_calls[0][0] == "multi"
        assert segment_calls[1][0] == "single"

        # Verify persisted final state in StepCache
        cached = pipeline.cache.load_state(url)
        assert cached is not None
        assert cached.status is PipelineStatus.COMPLETE
        assert cached.stream_info.streamer == "CustomStreamer"
        assert cached.stream_info.game == "GTA 6"
        assert len(cached.selected_highlights) == 2
        assert len(cached.clips) == 2

        # Verify companion metadata JSON written with YPP fields
        meta_json = json.loads(clips[0].metadata_path.read_text(encoding="utf-8"))
        assert meta_json["moment_type"] == "clutch"
        assert meta_json["transformation_mode"] == "narrative"
        assert meta_json["streamer"] == "CustomStreamer"

    @pytest.mark.asyncio
    async def test_resume_from_cached_analyzing_skips_download_and_transcribe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        audio_file = _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)

        config = _make_config(tmp_path)
        pipeline = Pipeline(config)
        url = "https://youtube.com/watch?v=resume_analyzing"

        pipeline.cache.save_state(
            PipelineState(
                stream_info=_sample_stream_info(url=url, video_id="resume_analyzing"),
                status=PipelineStatus.ANALYZING,
                audio_path=audio_file,
                transcript=_sample_transcript(),
            )
        )

        clips = await pipeline.process_stream(url, max_clips=1, interactive=False)
        assert len(clips) == 1
        assert "download_audio" not in call_log
        assert "transcribe" not in call_log
        assert "loudness" in call_log
        assert "chat_velocity" in call_log
        assert "score_and_select" in call_log

        cached = pipeline.cache.load_state(url)
        assert cached is not None and cached.status is PipelineStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_resume_from_cached_scoring_skips_download_transcribe_and_analyze(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        audio_file = _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)

        config = _make_config(tmp_path)
        pipeline = Pipeline(config)
        url = "https://youtube.com/watch?v=resume_scoring"

        pipeline.cache.save_state(
            PipelineState(
                stream_info=_sample_stream_info(url=url, video_id="resume_scoring"),
                status=PipelineStatus.SCORING,
                audio_path=audio_file,
                transcript=_sample_transcript(),
                loudness_spikes=[TimeRange(start=118.0, end=128.0, score=9.0)],
                chat_spikes=[
                    TimeRange(
                        start=115.0,
                        end=130.0,
                        score=8.5,
                        signal_type=SignalType.CHAT_VELOCITY,
                    )
                ],
            )
        )

        clips = await pipeline.process_stream(url, max_clips=1, interactive=False)
        assert len(clips) == 1
        assert "download_audio" not in call_log
        assert "transcribe" not in call_log
        assert "loudness" not in call_log
        assert "chat_scrape" not in call_log
        assert "chat_velocity" not in call_log
        assert "score_and_select" in call_log

    @pytest.mark.asyncio
    async def test_failed_state_saves_99_and_retry_recovers_to_correct_stage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        first_log: list[str] = []
        _install_mock_stage_deps(
            monkeypatch,
            tmp_path,
            call_log=first_log,
            score_error=RuntimeError("Ollama temporarily unavailable"),
        )

        config = _make_config(tmp_path)
        pipeline = Pipeline(config)
        url = "https://youtube.com/watch?v=retry_from_failed"

        with pytest.raises(RuntimeError, match="Ollama temporarily unavailable"):
            await pipeline.process_stream(url, max_clips=1, interactive=False)

        failed_state = pipeline.cache.load_state(url)
        assert failed_state is not None
        assert failed_state.status is PipelineStatus.FAILED
        assert int(failed_state.status) == 99
        assert failed_state.transcript is not None
        assert len(failed_state.loudness_spikes) == 1

        # Second attempt succeeds — must recover from FAILED (99) to SCORING (5),
        # skipping download, transcribe, and analyze, NOT skipping scoring/assembly!
        retry_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=retry_log)

        clips = await pipeline.process_stream(url, max_clips=1, interactive=False)
        assert len(clips) == 1
        assert "download_audio" not in retry_log
        assert "transcribe" not in retry_log
        assert "loudness" not in retry_log
        assert "chat_velocity" not in retry_log
        assert "check_connection" in retry_log
        assert "score_and_select" in retry_log

        final_state = pipeline.cache.load_state(url)
        assert final_state is not None and final_state.status is PipelineStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_mode_and_voice_paths_forwarded_to_assembly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        segment_calls: list[tuple[str, Any]] = []
        _install_mock_stage_deps(
            monkeypatch, tmp_path, call_log=call_log, segment_calls=segment_calls
        )

        config = _make_config(tmp_path)
        pipeline = Pipeline(config)

        recorded_assembly_kwargs: list[dict[str, Any]] = []
        original_assemble_highlight = pipeline._assemble_highlight

        def spy_assemble_highlight(
            highlight: Highlight,
            metadata: ClipMetadata,
            segment_paths: list[Path],
            clip_index: int,
            mode: str,
            voice_intro_path: Path | None,
            voice_outro_path: Path | None,
        ) -> ClipResult:
            recorded_assembly_kwargs.append(
                {
                    "mode": mode,
                    "voice_intro_path": voice_intro_path,
                    "voice_outro_path": voice_outro_path,
                    "meta_mode": metadata.transformation_mode,
                }
            )
            return original_assemble_highlight(
                highlight,
                metadata,
                segment_paths,
                clip_index,
                mode,
                voice_intro_path,
                voice_outro_path,
            )

        monkeypatch.setattr(pipeline, "_assemble_highlight", spy_assemble_highlight)

        intro_wav = tmp_path / "my_intro.wav"
        outro_wav = tmp_path / "my_outro.wav"

        clips = await pipeline.process_stream(
            "https://youtube.com/watch?v=raw_mode_check",
            max_clips=1,
            interactive=False,
            mode="raw",
            voice_intro_path=intro_wav,
            voice_outro_path=outro_wav,
        )

        assert len(clips) == 1
        assert recorded_assembly_kwargs == [
            {
                "mode": "raw",
                "voice_intro_path": intro_wav,
                "voice_outro_path": outro_wav,
                "meta_mode": "raw",
            }
        ]
        # In raw mode, even a highlight with sub_clips downloads a single cut
        assert segment_calls == [("single", (110.0, 160.0))]

    @pytest.mark.asyncio
    async def test_empty_highlights_completes_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log, highlights=[])

        pipeline = Pipeline(_make_config(tmp_path))
        url = "https://youtube.com/watch?v=no_highlights"
        clips = await pipeline.process_stream(url, interactive=False)
        assert clips == []
        cached = pipeline.cache.load_state(url)
        assert cached is not None and cached.status is PipelineStatus.COMPLETE


# ---------------------------------------------------------------------------
# Interactive vs Non-Interactive Highlight Selection Tests
# ---------------------------------------------------------------------------


class TestInteractiveSelection:
    def test_interactive_select_displays_moment_type_narration_intro_and_beats(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        pipeline.console = Console(record=True, width=180, force_terminal=False)
        highlights = _sample_highlights()

        monkeypatch.setattr(builtins, "input", lambda _prompt="": "1, 3, 1, 99, abc")
        picked = pipeline._interactive_select(highlights)

        assert len(picked) == 2
        assert picked[0].title == "POLICE CHASE ROOFTOP JUMP"
        assert picked[1].title == "INSANE WATER PHYSICS"

        rendered = pipeline.console.export_text()
        assert "Detected Highlights" in rendered
        assert "clutch" in rendered
        assert "funny" in rendered
        assert "graphics_detail" in rendered
        assert "2 cuts" in rendered
        assert "single" in rendered
        assert "Five stars on his tail" in rendered
        assert "Famous last words" in rendered

    def test_interactive_select_all_and_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        pipeline.console = Console(record=True, width=160, force_terminal=False)
        highlights = _sample_highlights()

        monkeypatch.setattr(builtins, "input", lambda _prompt="": "ALL")
        assert len(pipeline._interactive_select(highlights)) == 3

        assert pipeline._interactive_select([]) == []
        assert "No highlights found" in pipeline.console.export_text()

    @pytest.mark.asyncio
    async def test_process_stream_interactive_true_vs_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)
        pipeline = Pipeline(_make_config(tmp_path))
        pipeline.console = Console(record=True, width=180, force_terminal=False)

        # interactive=False must NEVER call input()
        def _forbid_input(_prompt: str = "") -> str:
            raise AssertionError("input() must not be called when interactive=False")

        monkeypatch.setattr(builtins, "input", _forbid_input)
        auto_clips = await pipeline.process_stream(
            "https://youtube.com/watch?v=auto_select",
            max_clips=2,
            interactive=False,
        )
        assert len(auto_clips) == 2

        # interactive=True uses user's selection from input()
        monkeypatch.setattr(builtins, "input", lambda _prompt="": "2")
        picked_clips = await pipeline.process_stream(
            "https://youtube.com/watch?v=manual_select",
            interactive=True,
        )
        assert len(picked_clips) == 1
        assert picked_clips[0].metadata.hook_title == "NPC KARMA MOMENT"


# ---------------------------------------------------------------------------
# Parallel Processing & Fault Isolation Tests
# ---------------------------------------------------------------------------


class TestParallelProcessing:
    @pytest.mark.asyncio
    async def test_parallel_forces_non_interactive_respects_concurrency_and_isolates_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pipeline = Pipeline(_make_config(tmp_path))

        def _forbid_input(_prompt: str = "") -> str:
            raise AssertionError("input() must never be called in parallel mode")

        monkeypatch.setattr(builtins, "input", _forbid_input)

        active = 0
        max_observed_active = 0
        interactive_flags: list[bool] = []

        async def fake_process_stream(
            url: str,
            max_clips: int | None = None,
            game: str | None = None,
            streamer: str | None = None,
            interactive: bool = False,
            mode: str | None = None,
            voice_intro_path: Path | None = None,
            voice_outro_path: Path | None = None,
        ) -> list[ClipResult]:
            nonlocal active, max_observed_active
            interactive_flags.append(interactive)
            active += 1
            max_observed_active = max(max_observed_active, active)
            try:
                await asyncio.sleep(0.03)
                if "bad_stream" in url:
                    raise RuntimeError(f"boom on {url}")
                return [
                    ClipResult(
                        video_path=tmp_path / f"{url.split('=')[-1]}.mp4",
                        metadata_path=tmp_path / f"{url.split('=')[-1]}.json",
                        metadata=ClipMetadata(
                            hook_title="Clip",
                            caption="Cap",
                            hashtags=["GTA6"],
                            streamer="S",
                            game=game or "GTA 6",
                            source_url=url,
                            clip_start=0.0,
                            clip_end=30.0,
                            highlight_score=9.0,
                            transformation_mode=mode or "narrative",
                        ),
                        duration_seconds=30.0,
                    )
                ]
            finally:
                active -= 1

        monkeypatch.setattr(pipeline, "process_stream", fake_process_stream)

        urls = [
            "https://youtube.com/watch?v=good_1",
            "https://youtube.com/watch?v=bad_stream",
            "https://youtube.com/watch?v=good_2",
            "https://youtube.com/watch?v=good_3",
        ]
        results = await pipeline.process_streams_parallel(
            urls,
            max_concurrent=2,
            max_clips=3,
            game="GTA 6",
            mode="story",
        )

        assert len(results) == 4
        assert interactive_flags == [False, False, False, False]
        assert max_observed_active <= 2

        assert isinstance(results[0], list) and len(results[0]) == 1
        assert isinstance(results[1], RuntimeError)
        assert "bad_stream" in str(results[1])
        assert isinstance(results[2], list) and len(results[2]) == 1
        assert isinstance(results[3], list) and len(results[3]) == 1


# ---------------------------------------------------------------------------
# CLI Wiring Tests (--mode, --voice-intro, --voice-outro)
# ---------------------------------------------------------------------------


class TestCLIPipelineWiring:
    def test_clip_command_forwards_mode_and_voice_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.safe_dump({"general": {"cache_dir": str(tmp_path / ".cache")}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(cli_mod, "console", Console(width=180, force_terminal=False))

        recorded: list[dict[str, Any]] = []

        class StubPipeline:
            def __init__(self, config: Config) -> None:
                self.config = config

            async def process_stream(self, **kwargs: Any) -> list[ClipResult]:
                recorded.append(kwargs)
                return [
                    ClipResult(
                        video_path=tmp_path / "clip_01.mp4",
                        metadata_path=tmp_path / "clip_01.json",
                        metadata=ClipMetadata(
                            hook_title="Test",
                            caption="Cap",
                            hashtags=["GTA6"],
                            streamer="Kai",
                            game="GTA 6",
                            source_url=kwargs["url"],
                            clip_start=10.0,
                            clip_end=45.0,
                            highlight_score=9.0,
                            transformation_mode=kwargs["mode"] or "narrative",
                        ),
                        duration_seconds=35.0,
                    )
                ]

        monkeypatch.setattr(pipeline_mod, "Pipeline", StubPipeline)

        intro_wav = tmp_path / "intro.wav"
        outro_wav = tmp_path / "outro.wav"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg_file),
                "clip",
                "https://youtube.com/watch?v=cli_test_1",
                "--max-clips",
                "3",
                "--game",
                "GTA 6",
                "--streamer",
                "Kai",
                "--no-interactive",
                "--mode",
                "story",
                "--voice-intro",
                str(intro_wav),
                "--voice-outro",
                str(outro_wav),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Generated 1 clip(s)" in result.output
        assert len(recorded) == 1
        assert recorded[0] == {
            "url": "https://youtube.com/watch?v=cli_test_1",
            "max_clips": 3,
            "game": "GTA 6",
            "streamer": "Kai",
            "interactive": False,
            "mode": "story",
            "voice_intro_path": intro_wav,
            "voice_outro_path": outro_wav,
        }

    def test_clip_command_rejects_invalid_mode(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.safe_dump({}), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg_file),
                "clip",
                "https://youtube.com/watch?v=bad_mode",
                "--mode",
                "invalid_mode",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value for '--mode'" in result.output

    def test_auto_command_discovers_and_invokes_parallel_pipeline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.safe_dump(
                {
                    "general": {
                        "cache_dir": str(tmp_path / ".cache"),
                        "max_concurrent_streams": 3,
                    },
                    "discovery": {"default_game": "GTA 6"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(cli_mod, "console", Console(width=180, force_terminal=False))

        class StubDiscovery:
            def __init__(self, config: Config) -> None:
                self.config = config

            def discover(self, game: str | None = None) -> list[StreamInfo]:
                return [
                    _sample_stream_info(url="https://youtube.com/watch?v=s1", video_id="s1"),
                    _sample_stream_info(url="https://youtube.com/watch?v=s2", video_id="s2"),
                    _sample_stream_info(url="https://youtube.com/watch?v=s3", video_id="s3"),
                ]

        parallel_calls: list[dict[str, Any]] = []

        class StubPipeline:
            def __init__(self, config: Config) -> None:
                self.config = config

            async def process_streams_parallel(
                self, urls: list[str], **kwargs: Any
            ) -> list[list[ClipResult] | BaseException]:
                parallel_calls.append({"urls": urls, **kwargs})
                return [
                    [
                        ClipResult(
                            video_path=tmp_path / "clip_01.mp4",
                            metadata_path=tmp_path / "clip_01.json",
                            metadata=ClipMetadata(
                                hook_title="Clip",
                                caption="Cap",
                                hashtags=["GTA6"],
                                streamer="S",
                                game="GTA 6",
                                source_url=urls[0],
                                clip_start=0.0,
                                clip_end=30.0,
                                highlight_score=9.0,
                                transformation_mode="raw",
                            ),
                            duration_seconds=30.0,
                        )
                    ],
                    RuntimeError("stream 2 failed"),
                ]

        monkeypatch.setattr(yt_mod, "YouTubeDiscovery", StubDiscovery)
        monkeypatch.setattr(pipeline_mod, "Pipeline", StubPipeline)

        intro_wav = tmp_path / "auto_intro.wav"
        outro_wav = tmp_path / "auto_outro.wav"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg_file),
                "auto",
                "--game",
                "GTA 6",
                "--max-streams",
                "2",
                "--max-clips",
                "4",
                "--mode",
                "raw",
                "--voice-intro",
                str(intro_wav),
                "--voice-outro",
                str(outro_wav),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "1 clip(s) generated across 1/2 stream(s)" in result.output
        assert len(parallel_calls) == 1
        assert parallel_calls[0] == {
            "urls": [
                "https://youtube.com/watch?v=s1",
                "https://youtube.com/watch?v=s2",
            ],
            "max_concurrent": 3,
            "max_clips": 4,
            "game": "GTA 6",
            "mode": "raw",
            "voice_intro_path": intro_wav,
            "voice_outro_path": outro_wav,
        }
