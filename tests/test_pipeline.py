"""Unit and mocked end-to-end integration tests for the Pipeline orchestrator and CLI wiring."""
from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path
import sys
import types
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

    def fake_download_audio(
        url: str, output_dir: Path, progress_callback: Any = None
    ) -> tuple[Path, StreamInfo]:
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
            max_clips: int | None = None,
            audio_path: Path | str | None = None,
        ) -> list[Highlight]:
            call_log.append("score_and_select")
            return list(hl_list)

        def generate_metadata(
            self,
            highlight: Highlight,
            streamer: str,
            game: str,
            source_url: str,
            mode: str | None = None,
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
        *,
        video_id: str | None = None,
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
        *,
        video_id: str | None = None,
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

    # Mock heavy video/FFmpeg Phase 4 operations for dummy 10-byte segment stubs while
    # letting the real Pipeline._assemble_highlight() and FileOrganizer run end-to-end.
    import streamclipper.assembly.composer as composer_mod
    import streamclipper.assembly.facecam as facecam_mod
    import streamclipper.assembly.hooks as hooks_mod
    import streamclipper.output.exporter as exporter_mod

    monkeypatch.setattr(facecam_mod.FacecamDetector, "detect", lambda self, path, **kw: None)
    monkeypatch.setattr(
        composer_mod.VideoComposer,
        "compose",
        lambda self, video_path, facecam, peak_offset=None, trim_range=None, face_center_x=None: {"kind": "single"},
    )
    monkeypatch.setattr(
        composer_mod.VideoComposer,
        "compose_story_beats",
        lambda self, segment_paths, facecam, beat_trim_ranges=None, peak_offset=None: {
            "kind": "multi_beat"
        },
    )
    monkeypatch.setattr(
        hooks_mod.HookTitleOverlay,
        "add_hook",
        lambda self, video_clip, hook_title, context_badge="": video_clip,
    )

    def _fake_export(self: Any, **kwargs: Any) -> Path:
        out_path = Path(kwargs["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"rendered-mp4")
        return out_path

    monkeypatch.setattr(exporter_mod.Exporter, "export", _fake_export)
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
            transcript=_sample_transcript(),
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

    def test_failed_with_analysis_or_spikes_but_no_transcript_never_skips_transcription(
        self, tmp_path: Path
    ) -> None:
        audio_path = tmp_path / "stream.wav"
        audio_path.write_bytes(b"wav")
        state_with_wav = PipelineState(
            stream_info=_sample_stream_info(),
            status=PipelineStatus.FAILED,
            audio_path=audio_path,
            transcript=None,
            analysis_complete=True,
            loudness_spikes=[TimeRange(start=10.0, end=20.0, score=8.0)],
            chat_spikes=[
                TimeRange(
                    start=10.0,
                    end=20.0,
                    score=8.0,
                    signal_type=SignalType.CHAT_VELOCITY,
                )
            ],
        )
        assert Pipeline._recover_failed_state(state_with_wav).status is PipelineStatus.TRANSCRIBING

        state_without_wav = PipelineState(
            stream_info=_sample_stream_info(),
            status=PipelineStatus.FAILED,
            audio_path=tmp_path / "missing.wav",
            transcript=None,
            analysis_complete=True,
            chat_spikes=[
                TimeRange(
                    start=10.0,
                    end=20.0,
                    score=8.0,
                    signal_type=SignalType.CHAT_VELOCITY,
                )
            ],
        )
        assert Pipeline._recover_failed_state(state_without_wav).status is PipelineStatus.PENDING

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
        # Default mode is 'auto': clip 1 has valid sub_clips -> 'story'; clip 2 is single-cut -> 'narrative'
        meta_json_0 = json.loads(clips[0].metadata_path.read_text(encoding="utf-8"))
        assert meta_json_0["moment_type"] == "clutch"
        assert meta_json_0["transformation_mode"] == "story"
        assert meta_json_0["streamer"] == "CustomStreamer"

        meta_json_1 = json.loads(clips[1].metadata_path.read_text(encoding="utf-8"))
        assert meta_json_1["transformation_mode"] == "narrative"

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
            **kwargs: Any,
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
                **kwargs,
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

    @pytest.mark.asyncio
    async def test_url_whitespace_normalized_and_empty_url_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)
        pipeline = Pipeline(_make_config(tmp_path))

        with pytest.raises(ValueError, match="non-empty"):
            await pipeline.process_stream("   ", interactive=False)
        assert pipeline.cache.list_all() == []

        raw_url = "  https://youtube.com/watch?v=ws_norm  "
        clean_url = raw_url.strip()
        clips = await pipeline.process_stream(raw_url, max_clips=1, interactive=False)
        assert len(clips) == 1

        # Re-running with stripped URL hits the exact same COMPLETE cache without re-running stages
        call_log.clear()
        clips_again = await pipeline.process_stream(clean_url, max_clips=1, interactive=False)
        assert len(clips_again) == 1
        assert call_log == []

    @pytest.mark.asyncio
    async def test_complete_state_with_empty_selection_skips_reselection(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)
        pipeline = Pipeline(_make_config(tmp_path))
        url = "https://youtube.com/watch?v=done_empty_selection"

        pipeline.cache.save_state(
            PipelineState(
                stream_info=_sample_stream_info(url=url, video_id="done_empty_selection"),
                status=PipelineStatus.COMPLETE,
                highlights=_sample_highlights(),
                selected_highlights=[],
                clips=[],
            )
        )

        def _forbid_input(_prompt: str = "") -> str:
            raise AssertionError("COMPLETE state must never re-prompt interactive selection")

        monkeypatch.setattr(builtins, "input", _forbid_input)
        clips = await pipeline.process_stream(url, interactive=True)
        assert clips == []
        assert call_log == []

        cached = pipeline.cache.load_state(url)
        assert cached is not None
        assert cached.status is PipelineStatus.COMPLETE
        assert cached.selected_highlights == []

    @pytest.mark.asyncio
    async def test_invalid_sub_clips_falls_back_to_single_segment_download(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bad_highlight = Highlight(
            start=100.0,
            end=150.0,
            score=9.0,
            title="MALFORMED BEATS",
            reason="Overlapping/invalid sub_clips should fall back to single cut",
            final_score=9.0,
            sub_clips=[
                {"start": 100.0, "end": 120.0, "label": "Setup"},
                {"start": 115.0, "end": 140.0, "label": "Overlapping Payoff"},
            ],
        )
        call_log: list[str] = []
        segment_calls: list[tuple[str, Any]] = []
        _install_mock_stage_deps(
            monkeypatch,
            tmp_path,
            call_log=call_log,
            highlights=[bad_highlight],
            segment_calls=segment_calls,
        )

        pipeline = Pipeline(_make_config(tmp_path))
        clips = await pipeline.process_stream(
            "https://youtube.com/watch?v=bad_beats",
            max_clips=1,
            interactive=False,
            mode="story",
        )
        assert len(clips) == 1
        assert segment_calls == [("single", (100.0, 150.0))]

    @pytest.mark.asyncio
    async def test_phase4_assembly_hook_single_and_multi_beat_contract(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify Phase 4 compose_story_beats, beat_trim_ranges, buffer offset math, and Exporter wiring."""
        call_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)

        compose_story_calls: list[dict[str, Any]] = []
        compose_single_calls: list[dict[str, Any]] = []
        export_calls: list[dict[str, Any]] = []

        facecam_mod = types.ModuleType("streamclipper.assembly.facecam")

        class StubFacecamDetector:
            def __init__(self, config: Config) -> None:
                self.config = config

            def detect(self, path, *, streamer=None) -> None:
                return None

        facecam_mod.FacecamDetector = StubFacecamDetector  # type: ignore[attr-defined]

        composer_mod = types.ModuleType("streamclipper.assembly.composer")

        def stub_remap(words, chat_burst, peak_time, beats):
            return words, chat_burst, 10.5

        class StubVideoComposer:
            def __init__(self, config: Config) -> None:
                self.config = config

            def compose_story_beats(
                self,
                segment_paths,
                facecam,
                beat_trim_ranges=None,
                peak_offset=None,
                face_center_x=None,
            ):
                compose_story_calls.append(
                    {
                        "segment_paths": segment_paths,
                        "facecam": facecam,
                        "beat_trim_ranges": beat_trim_ranges,
                        "peak_offset": peak_offset,
                    }
                )
                return {"kind": "multi_beat"}

            def compose(
                self, video_path, facecam, peak_offset=None, trim_range=None, face_center_x=None
            ):
                compose_single_calls.append(
                    {
                        "video_path": video_path,
                        "facecam": facecam,
                        "peak_offset": peak_offset,
                        "trim_range": trim_range,
                    }
                )
                # Mirror VideoComposer.compose(): expose the trim actually applied
                class _ComposedStub(dict):
                    pass

                result = _ComposedStub(kind="single")
                result._trim_start = trim_range[0] if trim_range else 0.0
                return result

        composer_mod.remap_timeline_for_beats = stub_remap  # type: ignore[attr-defined]
        composer_mod.VideoComposer = StubVideoComposer  # type: ignore[attr-defined]

        hooks_mod = types.ModuleType("streamclipper.assembly.hooks")

        class StubHookTitleOverlay:
            def __init__(self, config: Config) -> None:
                self.config = config

            def add_hook(self, video_clip, hook_title, context_badge=""):
                return {**video_clip, "hook_title": hook_title, "context_badge": context_badge}

        hooks_mod.HookTitleOverlay = StubHookTitleOverlay  # type: ignore[attr-defined]

        exporter_mod = types.ModuleType("streamclipper.output.exporter")

        class StubExporter:
            def __init__(self, config: Config) -> None:
                self.config = config

            def export(self, **kwargs: Any) -> Path:
                export_calls.append(kwargs)
                out_path = Path(kwargs["output_path"])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"rendered-mp4")
                return out_path

        exporter_mod.Exporter = StubExporter  # type: ignore[attr-defined]

        organizer_mod = types.ModuleType("streamclipper.output.organizer")

        class StubFileOrganizer:
            def __init__(self, config: Config) -> None:
                self.out_dir = Path(config.get("general.output_dir"))

            def get_clip_path(self, game: str, streamer: str, index: int) -> Path:
                return self.out_dir / f"clip_{index:02d}.mp4"

            def next_available_index(
                self, game: str, streamer: str, min_index: int = 1, date_str: str | None = None
            ) -> int:
                return min_index

            def save_clip(
                self, video_path: Path, metadata: ClipMetadata, transcript_text: str = ""
            ) -> ClipResult:
                meta_path = video_path.with_suffix(".json")
                meta_path.write_text(metadata.to_json(), encoding="utf-8")
                return ClipResult(
                    video_path=video_path,
                    metadata_path=meta_path,
                    metadata=metadata,
                    duration_seconds=34.0,
                )

        organizer_mod.FileOrganizer = StubFileOrganizer  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "streamclipper.assembly.facecam", facecam_mod)
        monkeypatch.setitem(sys.modules, "streamclipper.assembly.composer", composer_mod)
        monkeypatch.setitem(sys.modules, "streamclipper.assembly.hooks", hooks_mod)
        monkeypatch.setitem(sys.modules, "streamclipper.output.exporter", exporter_mod)
        monkeypatch.setitem(sys.modules, "streamclipper.output.organizer", organizer_mod)

        intro_wav = tmp_path / "human_intro.wav"
        outro_wav = tmp_path / "human_outro.wav"

        pipeline = Pipeline(
            _make_config(tmp_path, {"output": {"min_commentary_ratio": 0.0}})
        )
        clips = await pipeline.process_stream(
            "https://youtube.com/watch?v=phase4_contract",
            max_clips=2,
            interactive=False,
            mode="story",
            voice_intro_path=intro_wav,
            voice_outro_path=outro_wav,
        )

        assert len(clips) == 2

        # Highlight 1: multi-beat [110..124] & [140..160] with 3s download buffer -> local trims (3,17) & (3,23)
        assert len(compose_story_calls) == 1
        assert compose_story_calls[0]["beat_trim_ranges"] == [(3.0, 17.0), (3.0, 23.0)]
        assert compose_story_calls[0]["peak_offset"] == 10.5
        assert export_calls[0]["clip_start_offset"] == 0.0
        assert export_calls[0]["mode"] == "story"
        assert export_calls[0]["voice_intro_path"] == intro_wav
        assert export_calls[0]["voice_outro_path"] == outro_wav

        # Highlight 2: single-cut [400..445], peak=425.0, buffer=3.0 -> trim_range=(3.0, 48.0), clip_start_offset=400.0
        assert len(compose_single_calls) == 1
        assert compose_single_calls[0]["peak_offset"] == pytest.approx(28.0)
        assert compose_single_calls[0].get("trim_range") == (3.0, 48.0)
        assert export_calls[1]["clip_start_offset"] == pytest.approx(400.0)
        assert export_calls[1]["mode"] == "story"


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

    def test_clip_command_rejects_directory_for_voice_intro(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.safe_dump({}), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg_file),
                "clip",
                "https://youtube.com/watch?v=dir_voice",
                "--voice-intro",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        assert "directory" in result.output.lower()

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

        # L14: one failed stream makes `auto` exit non-zero after printing the summary
        assert result.exit_code == 1, result.output
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


# ---------------------------------------------------------------------------
# Audit Hardening & User Decision Regression Tests
# ---------------------------------------------------------------------------


class TestAuditHardeningAndModes:
    @pytest.mark.asyncio
    async def test_narrative_mode_forces_single_cut_even_when_sub_clips_exist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """mode='narrative' must keep single continuous cut + voiceover/overlays (no multi-beat stitching)."""
        call_log: list[str] = []
        segment_calls: list[tuple[str, Any]] = []
        _install_mock_stage_deps(
            monkeypatch, tmp_path, call_log=call_log, segment_calls=segment_calls
        )
        pipeline = Pipeline(_make_config(tmp_path))
        clips = await pipeline.process_stream(
            "https://youtube.com/watch?v=narrative_single",
            max_clips=1,
            interactive=False,
            mode="narrative",
        )
        assert len(clips) == 1
        assert segment_calls == [("single", (110.0, 160.0))]
        assert clips[0].metadata.transformation_mode == "narrative"
        assert clips[0].metadata.narration_intro != ""

    @pytest.mark.asyncio
    async def test_transformation_disabled_defaults_to_raw_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """transformation.enabled=false must resolve to raw mode (single cut, cleared narration/badge)."""
        call_log: list[str] = []
        segment_calls: list[tuple[str, Any]] = []
        _install_mock_stage_deps(
            monkeypatch, tmp_path, call_log=call_log, segment_calls=segment_calls
        )
        cfg = _make_config(tmp_path, overrides={"transformation": {"enabled": False, "mode": "story"}})
        pipeline = Pipeline(cfg)
        clips = await pipeline.process_stream(
            "https://youtube.com/watch?v=disabled_ypp",
            max_clips=1,
            interactive=False,
        )
        assert len(clips) == 1
        assert segment_calls == [("single", (110.0, 160.0))]
        assert clips[0].metadata.transformation_mode == "raw"
        assert clips[0].metadata.narration_intro == ""
        assert clips[0].metadata.narration_outro == ""
        assert clips[0].metadata.context_badge == ""

    @pytest.mark.asyncio
    async def test_analyze_failure_during_chat_does_not_skip_chat_on_retry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """If chat scraping fails after loudness succeeds, loudness is NOT prematurely committed."""
        call_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)

        class _FailingChatScraper:
            def load_cache(self, path: Path) -> None:
                return None

            def scrape(self, url: str, temp_dir: Path) -> Any:
                raise RuntimeError("transient chat error")

        monkeypatch.setattr(pipeline_mod, "ChatScraper", _FailingChatScraper)
        pipeline = Pipeline(_make_config(tmp_path))
        url = "https://youtube.com/watch?v=chat_fail_retry"

        with pytest.raises(RuntimeError, match="transient chat error"):
            await pipeline.process_stream(url, max_clips=1, interactive=False)

        failed = pipeline.cache.load_state(url)
        assert failed is not None
        assert failed.status is PipelineStatus.FAILED
        assert failed.loudness_spikes == []
        assert failed.analysis_complete is False

        # On retry, recovery must return to ANALYZING and run chat analysis
        retry_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=retry_log)
        clips = await pipeline.process_stream(url, max_clips=1, interactive=False)
        assert len(clips) == 1
        assert "chat_scrape" in retry_log
        assert "chat_velocity" in retry_log

    def test_recover_failed_state_handles_zero_spike_analysis_and_missing_audio(
        self, tmp_path: Path
    ) -> None:
        # Zero-spike completed analysis must recover to SCORING instead of re-running ANALYZING
        zero_spike_state = PipelineState(
            stream_info=_sample_stream_info(),
            status=PipelineStatus.FAILED,
            transcript=_sample_transcript(),
            analysis_complete=True,
        )
        assert Pipeline._recover_failed_state(zero_spike_state).status is PipelineStatus.SCORING

        # Transcript present + deleted temp audio must reset to PENDING (breaking FileNotFoundError loop)
        missing_wav_state = PipelineState(
            stream_info=_sample_stream_info(),
            status=PipelineStatus.FAILED,
            audio_path=tmp_path / "deleted.wav",
            transcript=_sample_transcript(),
        )
        assert Pipeline._recover_failed_state(missing_wav_state).status is PipelineStatus.PENDING

    def test_assemble_highlight_reraises_non_phase4_import_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Class-name mismatches or missing third-party deps must NOT silently fall back."""
        broken_organizer = types.ModuleType("streamclipper.output.organizer")
        # Deliberately omit FileOrganizer to simulate API mismatch
        monkeypatch.setitem(sys.modules, "streamclipper.assembly.composer", types.ModuleType("streamclipper.assembly.composer"))
        sys.modules["streamclipper.assembly.composer"].VideoComposer = object  # type: ignore[attr-defined]
        sys.modules["streamclipper.assembly.composer"].remap_timeline_for_beats = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "streamclipper.assembly.facecam", types.ModuleType("streamclipper.assembly.facecam"))
        sys.modules["streamclipper.assembly.facecam"].FacecamDetector = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "streamclipper.assembly.hooks", types.ModuleType("streamclipper.assembly.hooks"))
        sys.modules["streamclipper.assembly.hooks"].HookTitleOverlay = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "streamclipper.output.exporter", types.ModuleType("streamclipper.output.exporter"))
        sys.modules["streamclipper.output.exporter"].Exporter = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "streamclipper.output.organizer", broken_organizer)

        pipeline = Pipeline(_make_config(tmp_path))
        h = _sample_highlights()[0]
        meta = ClipMetadata(
            hook_title=h.title, caption="Cap", hashtags=["GTA6"],
            streamer="S", game="GTA 6", source_url="u",
            clip_start=h.start, clip_end=h.end, highlight_score=9.0,
        )
        with pytest.raises(ImportError):
            pipeline._assemble_highlight(h, meta, [tmp_path / "s.mp4"], 1, "narrative", None, None)

    @pytest.mark.asyncio
    async def test_parallel_streams_serialize_gpu_stages_and_preserve_detected_game_and_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Parallel runs never overlap GPU stages; detected game is preserved; force=True reruns."""
        call_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)

        # Override download_audio to return detected game='Valorant'
        def _dl_with_game(url: str, output_dir: Path) -> tuple[Path, StreamInfo]:
            call_log.append("download_audio")
            vid = url.split("=")[-1]
            p = Path(output_dir) / f"{vid}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"RIFF....WAVE")
            return p, _sample_stream_info(url=url, video_id=vid, game="Valorant")

        monkeypatch.setattr(pipeline_mod, "download_audio", _dl_with_game)

        active_gpu = 0
        max_gpu_overlap = 0

        class _GpuTrackingTranscriber:
            def __init__(self, config: Config) -> None:
                pass

            def transcribe(self, audio_path: Path) -> TranscriptionResult:
                nonlocal active_gpu, max_gpu_overlap
                import time

                active_gpu += 1
                max_gpu_overlap = max(max_gpu_overlap, active_gpu)
                try:
                    time.sleep(0.03)
                    return _sample_transcript()
                finally:
                    active_gpu -= 1

        monkeypatch.setattr(pipeline_mod, "Transcriber", _GpuTrackingTranscriber)

        cfg = _make_config(
            tmp_path,
            overrides={
                "output": {
                    "structure": "{streamer}/{game}",
                    "save_metadata_json": False,
                }
            },
        )
        pipeline = Pipeline(cfg)
        urls = [
            "https://youtube.com/watch?v=gpu_1",
            "https://youtube.com/watch?v=gpu_2",
        ]
        results = await pipeline.process_streams_parallel(urls, max_concurrent=2, max_clips=1)
        assert max_gpu_overlap == 1
        assert all(isinstance(r, list) and len(r) == 1 for r in results)

        # Detected game 'Valorant' was NOT overwritten by default 'GTA 6'
        first_clip = results[0][0]  # type: ignore[index]
        assert first_clip.metadata.game == "Valorant"
        # save_metadata_json=False returns metadata_path=None instead of nonexistent Path
        assert first_clip.metadata_path is None

        # force=True clears COMPLETE cache and re-runs pipeline stages
        call_log.clear()
        rerun = await pipeline.process_stream(urls[0], max_clips=1, interactive=False, force=True)
        assert len(rerun) == 1
        assert "download_audio" in call_log
        assert "score_and_select" in call_log

    @pytest.mark.asyncio
    async def test_process_stream_redownloads_missing_audio_at_analyzing_without_retranscribing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """If cached state is FAILED/ANALYZING with transcript but audio_path is None, re-download WAV and skip transcription."""
        call_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)
        cfg = _make_config(tmp_path)
        pipeline = Pipeline(cfg)

        url = "https://youtube.com/watch?v=missing_audio_path"
        state = PipelineState(
            stream_info=StreamInfo(
                url=url,
                title="Test",
                platform=Platform.YOUTUBE,
                streamer="S",
                game="G",
                duration_seconds=100,
            ),
            status=PipelineStatus.FAILED,
            audio_path=None,
            transcript=_sample_transcript(),
        )
        pipeline.cache.save_state(state)

        clips = await pipeline.process_stream(url, max_clips=1, interactive=False)
        assert len(clips) == 1
        assert "download_audio" in call_log
        assert "transcribe" not in call_log
        assert "loudness" in call_log

    def test_invalid_mode_override_raises_and_fallback_sanitizes_filename_template(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={"output": {"filename_template": "../../bad_{unknown}"}},
        )
        pipeline = Pipeline(cfg)
        with pytest.raises(ValueError, match="Invalid transformation mode"):
            pipeline._resolve_base_mode("invalid_mode")

        seg_path = tmp_path / "seg.mp4"
        seg_path.write_bytes(b"mp4")
        h = _sample_highlights()[1]
        meta = ClipMetadata(
            hook_title="Hook",
            caption="Cap",
            hashtags=["GTA6"],
            streamer="TestStreamer",
            game="GTA 6",
            source_url="https://youtu.be/1",
            clip_start=h.start,
            clip_end=h.end,
            highlight_score=8.5,
            transformation_mode="narrative",
        )
        res = pipeline._fallback_save_segment(h, meta, [seg_path], clip_index=1)
        assert res.metadata_path is not None
        assert res.metadata_path.name == "clip_01.json"
        assert tmp_path in res.metadata_path.resolve().parents

    @pytest.mark.asyncio
    async def test_process_stream_fails_fast_on_invalid_mode_before_cache_clear_or_download(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Invalid mode must raise ValueError immediately without wiping cache or downloading audio."""
        call_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)
        cfg = _make_config(tmp_path)
        pipeline = Pipeline(cfg)

        url = "https://youtube.com/watch?v=invalid_mode_fast_fail"
        cached = PipelineState(
            stream_info=StreamInfo(
                url=url,
                title="Cached Stream",
                platform=Platform.YOUTUBE,
                streamer="S",
                game="GTA 6",
                duration_seconds=1800,
                video_id="vid_fast",
            ),
            status=PipelineStatus.ANALYZING,
            transcript=_sample_transcript(),
        )
        pipeline.cache.save_state(cached)

        with pytest.raises(ValueError, match="Invalid transformation mode"):
            await pipeline.process_stream(url, mode="bogus_mode", force=True)

        assert call_log == []
        # Existing cache must remain intact (not wiped by force=True before mode validation)
        reloaded = pipeline.cache.load_state(url)
        assert reloaded is not None
        assert reloaded.status is PipelineStatus.ANALYZING

    @pytest.mark.asyncio
    async def test_process_stream_recovers_failed_state_when_analysis_ran_before_transcription(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """FAILED state with MCP-completed analysis and no transcript must transcribe then score."""
        call_log: list[str] = []
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=call_log)
        cfg = _make_config(tmp_path)
        pipeline = Pipeline(cfg)

        wav_path = tmp_path / "temp" / "pre_analyzed.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"RIFF")

        url = "https://youtube.com/watch?v=mcp_pre_analyzed"
        pipeline.cache.save_state(
            PipelineState(
                stream_info=StreamInfo(
                    url=url,
                    title="MCP Pre-Analyzed Stream",
                    platform=Platform.YOUTUBE,
                    streamer="S",
                    game="GTA 6",
                    duration_seconds=1800,
                    video_id="mcp_pre_analyzed",
                ),
                status=PipelineStatus.FAILED,
                audio_path=wav_path,
                transcript=None,
                loudness_spikes=[
                    TimeRange(start=118.0, end=123.0, score=8.8, metadata={"peak_time": 120.5})
                ],
                analysis_complete=True,
            )
        )

        clips = await pipeline.process_stream(url, max_clips=1, interactive=False)
        assert len(clips) == 1
        assert "download_audio" not in call_log
        assert "transcribe" in call_log
        assert "loudness" not in call_log  # already analysis_complete=True
        assert "score_and_select" in call_log
        final_state = pipeline.cache.load_state(url)
        assert final_state is not None
        assert final_state.status is PipelineStatus.COMPLETE


# ---------------------------------------------------------------------------
# Review fixes: H2 / M6 / M4 / L12 / L13 / L22 / G9 / G10 / A9 regressions
# ---------------------------------------------------------------------------


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


class TestReviewFixes:
    @pytest.mark.asyncio
    async def test_h2_existing_clip_is_not_overwritten(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=[])
        existing = tmp_path / "output" / "GTA 6" / "CustomStreamer" / _today() / "clip_01.mp4"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"old")
        pipeline = Pipeline(_make_config(tmp_path))
        clips = await pipeline.process_stream(
            "https://youtube.com/watch?v=h2",
            max_clips=1,
            game="GTA 6",
            streamer="CustomStreamer",
        )
        assert clips[0].video_path.name == "clip_02.mp4"
        assert existing.read_bytes() == b"old"

    @pytest.mark.asyncio
    async def test_m6_larger_max_clips_reruns_scoring(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_log: list[str] = []
        _install_mock_stage_deps(
            monkeypatch, tmp_path, call_log=call_log, highlights=_sample_highlights()[:1]
        )
        pipeline = Pipeline(_make_config(tmp_path))
        url = "https://youtube.com/watch?v=m6"
        await pipeline.process_stream(url, max_clips=1)
        assert call_log.count("score_and_select") == 1
        roundtrip = PipelineState.from_dict(
            PipelineState(
                stream_info=_sample_stream_info(), requested_max_clips=4
            ).to_dict()
        )
        assert roundtrip.requested_max_clips == 4
        clips = await pipeline.process_stream(url, max_clips=3)
        assert call_log.count("score_and_select") == 2
        assert len(clips) == 1

    @pytest.mark.asyncio
    async def test_m4_chat_scrape_failure_propagates_and_is_not_marked_complete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audio = _install_mock_stage_deps(monkeypatch, tmp_path, call_log=[])

        class BoomScraper:
            def load_cache(self, path: Path) -> None:
                return None

            def scrape(self, url: str, temp_dir: Path) -> pd.DataFrame:
                raise RuntimeError("scrape down")

        monkeypatch.setattr(pipeline_mod, "ChatScraper", BoomScraper)
        pipeline = Pipeline(_make_config(tmp_path))
        state = PipelineState(
            stream_info=_sample_stream_info(), audio_path=audio, transcript=_sample_transcript()
        )
        with pytest.raises(RuntimeError, match="scrape down"):
            await pipeline._step_analyze(state)
        assert state.analysis_complete is False

    @pytest.mark.asyncio
    async def test_m4_empty_chat_writes_header_only_csv_and_is_not_rescraped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from streamclipper.download.chat import ChatScraper, _empty_chat_df

        audio = _install_mock_stage_deps(monkeypatch, tmp_path, call_log=[])
        scrapes: list[str] = []

        class EmptyScraper(ChatScraper):
            def scrape(self, url: str, temp_dir: Path) -> pd.DataFrame:
                scrapes.append(url)
                return _empty_chat_df()

        monkeypatch.setattr(pipeline_mod, "ChatScraper", EmptyScraper)
        pipeline = Pipeline(_make_config(tmp_path))
        info = _sample_stream_info()
        for _ in range(2):
            state = PipelineState(stream_info=info, audio_path=audio)
            await pipeline._step_analyze(state)
        csv_path = pipeline.temp_dir / f"{info.video_id}_chat.csv"
        assert csv_path.is_file()
        assert csv_path.read_text(encoding="utf-8").strip().startswith("offset_seconds")
        assert len(scrapes) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "engine,expect_locked", [("edge-tts", False), ("kokoro", True)]
    )
    async def test_l12_assembly_lock_depends_on_tts_engine(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engine: str,
        expect_locked: bool,
    ) -> None:
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=[])
        pipeline = Pipeline(_make_config(tmp_path, {"transformation": {"tts_engine": engine}}))
        observed: list[bool] = []
        original = Pipeline._assemble_highlight

        def spy(self: Pipeline, *a: Any, **kw: Any) -> ClipResult:
            observed.append(self._gpu_lock.locked())
            return original(self, *a, **kw)

        monkeypatch.setattr(Pipeline, "_assemble_highlight", spy)
        await pipeline.process_stream("https://youtube.com/watch?v=l12", max_clips=1)
        assert observed == [expect_locked]

    def test_l13_close_failure_is_logged_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        class Bad:
            def close(self) -> None:
                raise OSError("close-fail-xyz")

        with caplog.at_level(logging.DEBUG, logger="streamclipper.pipeline"):
            Pipeline._close_composed_handles(Bad())
        assert "close-fail-xyz" in caplog.text

    @pytest.mark.asyncio
    async def test_l22_ignored_beats_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=[])
        pipeline = Pipeline(_make_config(tmp_path))
        beats = [{"start": 110.0, "end": 124.0}, {"start": 140.0, "end": 160.0}]
        monkeypatch.setattr(
            Pipeline, "_resolve_highlight_mode", lambda self, b, h: ("narrative", beats)
        )
        with caplog.at_level(logging.INFO, logger="streamclipper.pipeline"):
            await pipeline.process_stream("https://youtube.com/watch?v=l22", max_clips=1)
        assert "Ignoring 2 validated sub_clips" in caplog.text

    @pytest.mark.asyncio
    async def test_g9_resume_does_not_rerender_finished_clips(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import streamclipper.output.exporter as exporter_mod

        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=[])
        renders: list[str] = []
        fail_second = {"on": True}

        def export(self: Any, **kwargs: Any) -> Path:
            renders.append(Path(kwargs["output_path"]).name)
            if len(renders) == 2 and fail_second["on"]:
                raise RuntimeError("render died")
            out = Path(kwargs["output_path"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"mp4")
            return out

        monkeypatch.setattr(exporter_mod.Exporter, "export", export)
        pipeline = Pipeline(_make_config(tmp_path))
        url = "https://youtube.com/watch?v=g9"
        with pytest.raises(RuntimeError, match="render died"):
            await pipeline.process_stream(url, max_clips=2)
        fail_second["on"] = False
        clips = await pipeline.process_stream(url, max_clips=2)
        assert len(clips) == 2
        assert len(renders) == 3  # clip 1 rendered once, clip 2 twice

    @pytest.mark.asyncio
    async def test_g10_skip_high_music_skips_flagged_highlight(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        hls = _sample_highlights()
        hls[0].music_risk = "HIGH_MUSIC_DETECTED"
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=[], highlights=hls)
        pipeline = Pipeline(_make_config(tmp_path, {"analysis": {"skip_high_music": True}}))
        clips = await pipeline.process_stream("https://youtube.com/watch?v=g10", max_clips=2)
        assert [c.metadata.hook_title for c in clips] == ["NPC KARMA MOMENT"]

    @pytest.mark.asyncio
    async def test_g10_default_keeps_flagged_highlight(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        hls = _sample_highlights()
        hls[0].music_risk = "HIGH_MUSIC_DETECTED"
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=[], highlights=hls)
        pipeline = Pipeline(_make_config(tmp_path))
        clips = await pipeline.process_stream("https://youtube.com/watch?v=g10b", max_clips=2)
        assert len(clips) == 2

    @pytest.mark.asyncio
    async def test_a9_cross_process_gpu_lock_creates_and_releases_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_mock_stage_deps(monkeypatch, tmp_path, call_log=[])
        pipeline = Pipeline(
            _make_config(tmp_path, {"general": {"cross_process_gpu_lock": True}})
        )
        await pipeline.process_stream("https://youtube.com/watch?v=a9", max_clips=1)
        lock_path = tmp_path / ".cache" / "gpu.lock"
        assert lock_path.is_file()
        fh = await asyncio.wait_for(asyncio.to_thread(pipeline._acquire_file_lock), 5)
        pipeline._release_file_lock(fh)
        assert fh.closed
