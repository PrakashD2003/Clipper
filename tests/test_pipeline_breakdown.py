"""Unit and integration tests for YPP Breakdown mode routing, buffer trimming, and pre-flight copyright badges in Pipeline."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from streamclipper.config import Config, HardwareProfile
from streamclipper.models import (
    ClipMetadata,
    ClipResult,
    Highlight,
    PipelineState,
    PipelineStatus,
    Platform,
    StreamInfo,
    WordTimestamp,
)
from streamclipper.pipeline import Pipeline


def _make_config(tmp_path: Path) -> Config:
    hw = HardwareProfile(
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
    return Config(hardware_profile=hw)


def _make_highlight(
    moment_type: str = "hype",
    sub_clips: list[dict[str, Any]] | None = None,
    breakdown_commentary: str = "",
    music_risk: str = "CLEAN",
) -> Highlight:
    return Highlight(
        start=100.0,
        end=145.0,
        score=8.5,
        title="Test Highlight",
        reason="Action climax",
        moment_type=moment_type,
        sub_clips=sub_clips or [],
        breakdown_commentary=breakdown_commentary,
        music_risk=music_risk,
    )


class TestPipelineBreakdownModeResolution:
    """Verify _resolve_highlight_mode auto-selection rules."""

    def test_auto_resolves_breakdown_for_moment_types(self, tmp_path: Path) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        for m_type in ("analysis", "fail", "graphics_detail", "lore_secret"):
            h = _make_highlight(moment_type=m_type)
            mode, beats = pipeline._resolve_highlight_mode("auto", h)
            assert mode == "breakdown"
            assert beats == []

    def test_auto_resolves_breakdown_when_commentary_present(self, tmp_path: Path) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        # Even if moment_type is hype or clutch, explicit breakdown commentary triggers breakdown
        h = _make_highlight(
            moment_type="hype",
            breakdown_commentary="Watch how the ragdoll physics glitches into the palm tree.",
        )
        mode, beats = pipeline._resolve_highlight_mode("auto", h)
        assert mode == "breakdown"
        assert beats == []

    def test_auto_resolves_story_when_beats_present_and_not_breakdown(
        self, tmp_path: Path
    ) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        beats_data = [
            {"start": 100.0, "end": 115.0, "label": "Setup"},
            {"start": 130.0, "end": 145.0, "label": "Payoff"},
        ]
        h = _make_highlight(moment_type="clutch", sub_clips=beats_data)
        mode, beats = pipeline._resolve_highlight_mode("auto", h)
        assert mode == "story"
        assert len(beats) == 2

    def test_auto_resolves_narrative_for_clean_single(self, tmp_path: Path) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        h = _make_highlight(moment_type="clutch", sub_clips=[])
        mode, beats = pipeline._resolve_highlight_mode("auto", h)
        assert mode == "narrative"
        assert beats == []

    def test_auto_resolves_narrative_when_both_narration_intro_and_outro_present(
        self, tmp_path: Path
    ) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        h = _make_highlight(moment_type="fail")
        h.narration_intro = "Watch this streamer fail."
        h.narration_outro = "He never saw that wall coming."
        h.breakdown_commentary = ""
        mode, beats = pipeline._resolve_highlight_mode("auto", h)
        assert mode == "narrative"
        assert beats == []

    def test_explicit_breakdown_mode_returns_breakdown(self, tmp_path: Path) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        h = _make_highlight(moment_type="hype")
        mode, beats = pipeline._resolve_highlight_mode("breakdown", h)
        assert mode == "breakdown"
        assert beats == []

    def test_raw_mode_logs_ypp_demonetization_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        h = _make_highlight(moment_type="fail")
        with caplog.at_level(logging.WARNING):
            mode, beats = pipeline._resolve_highlight_mode("raw", h)
            assert mode == "raw"
            assert beats == []
            assert any(
                "reused content policies" in record.message
                for record in caplog.records
            )


class TestPipelineBreakdownAssembly:
    """Verify breakdown mode assembly routes through compose_instant_replay."""

    def test_assemble_highlight_breakdown_invokes_instant_replay(
        self, tmp_path: Path
    ) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        h = _make_highlight(
            moment_type="fail",
            breakdown_commentary="Watch the instant physics glitch on impact.",
        )
        h.peak_time = 120.0

        meta = ClipMetadata(
            hook_title="Epic Wipeout",
            caption="Wait for it...",
            hashtags=["GTA6"],
            streamer="Fleet",
            game="GTA 6",
            source_url="https://youtube.com/watch?v=xyz",
            clip_start=100.0,
            clip_end=145.0,
            highlight_score=8.5,
            moment_type="fail",
            transformation_mode="breakdown",
            breakdown_commentary="Watch the instant physics glitch on impact.",
        )

        dummy_seg = tmp_path / "seg.mp4"
        dummy_seg.write_bytes(b"dummy-video")

        fake_clip = MagicMock()
        fake_clip.duration = 45.0
        fake_replay_clip = MagicMock()
        fake_replay_clip.duration = 55.0

        with (
            patch("streamclipper.assembly.facecam.FacecamDetector.detect", return_value=None),
            patch("streamclipper.assembly.composer.VideoComposer.compose", return_value=fake_clip) as mock_compose,
            patch(
                "streamclipper.assembly.composer.VideoComposer.compose_instant_replay",
                return_value=(fake_replay_clip, 45.0),
            ) as mock_replay,
            patch("streamclipper.assembly.hooks.HookTitleOverlay.add_hook", return_value=fake_replay_clip),
            patch("streamclipper.output.exporter.Exporter.export", return_value=tmp_path / "out.mp4") as mock_export,
            patch(
                "streamclipper.output.organizer.FileOrganizer.save_clip",
                return_value=ClipResult(
                    video_path=tmp_path / "out.mp4",
                    metadata_path=tmp_path / "out.json",
                    metadata=meta,
                    duration_seconds=55.0,
                ),
            ),
        ):
            res = pipeline._assemble_highlight(
                highlight=h,
                metadata=meta,
                segment_paths=[dummy_seg],
                clip_index=1,
                mode="breakdown",
                voice_intro_path=None,
                voice_outro_path=None,
            )

            assert mock_compose.called
            # Verify single-clip buffer trim was passed
            _, compose_kw = mock_compose.call_args
            assert compose_kw.get("trim_range") == (3.0, 48.0)

            # Verify instant replay was assembled with peak accurately adjusted for lead padding
            assert mock_replay.called
            _, replay_kw = mock_replay.call_args
            assert replay_kw.get("peak_offset") == pytest.approx(20.0)
            assert mock_export.called
            export_kw = mock_export.call_args[1]
            assert export_kw.get("mode") == "breakdown"
            assert export_kw.get("breakdown_commentary") == "Watch the instant physics glitch on impact."
            assert export_kw.get("replay_start") == 45.0
            assert export_kw.get("clip_start_offset") == 100.0


class TestPipelineInteractiveSelectMusicBadge:
    """Verify ⚠️ MUSIC_DETECTED badge is shown on high-risk candidates."""

    def test_interactive_select_displays_music_badge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rich.console import Console

        pipeline = Pipeline(_make_config(tmp_path))
        pipeline.console = Console(record=True, width=120, force_terminal=False)

        h_clean = _make_highlight(moment_type="clutch", music_risk="CLEAN")
        h_music = _make_highlight(moment_type="hype", music_risk="HIGH_MUSIC_DETECTED")
        h_music.title = "RADIO MUSIC MOMENT"

        monkeypatch.setattr("builtins.input", lambda _: "all")

        selected = pipeline._interactive_select([h_clean, h_music])
        assert len(selected) == 2

        rendered = pipeline.console.export_text()
        assert "MUSIC_DETECTED" in rendered

    def test_assemble_highlight_breakdown_freeze_frame_style(
        self, tmp_path: Path
    ) -> None:
        import yaml

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.safe_dump(
                {
                    "transformation": {
                        "breakdown_style": "freeze_frame",
                        "breakdown": {
                            "freeze_duration": 4.5,
                            "freeze_zoom": 1.40,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        hw = HardwareProfile(
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
        config = Config(user_config_path=cfg_file, hardware_profile=hw)
        pipeline = Pipeline(config)

        h = _make_highlight(
            moment_type="fail",
            breakdown_commentary="Look at the exact frame of the collision.",
        )
        h.peak_time = 120.0

        meta = ClipMetadata(
            hook_title="Epic Freeze",
            caption="Wait for it...",
            hashtags=["GTA6"],
            streamer="Fleet",
            game="GTA 6",
            source_url="https://youtube.com/watch?v=xyz",
            clip_start=100.0,
            clip_end=145.0,
            highlight_score=8.5,
            moment_type="fail",
            transformation_mode="breakdown",
            breakdown_commentary="Look at the exact frame of the collision.",
        )

        dummy_seg = tmp_path / "seg.mp4"
        dummy_seg.write_bytes(b"dummy-video")

        fake_clip = MagicMock()
        fake_clip.duration = 45.0
        fake_freeze_clip = MagicMock()
        fake_freeze_clip.duration = 49.5

        with (
            patch("streamclipper.assembly.facecam.FacecamDetector.detect", return_value=None),
            patch("streamclipper.assembly.composer.VideoComposer.compose", return_value=fake_clip),
            patch(
                "streamclipper.assembly.composer.VideoComposer.compose_freeze_breakdown",
                return_value=fake_freeze_clip,
            ) as mock_freeze,
            patch("streamclipper.assembly.hooks.HookTitleOverlay.add_hook", return_value=fake_freeze_clip),
            patch("streamclipper.output.exporter.Exporter.export", return_value=tmp_path / "out.mp4"),
            patch(
                "streamclipper.output.organizer.FileOrganizer.save_clip",
                return_value=ClipResult(
                    video_path=tmp_path / "out.mp4",
                    metadata_path=tmp_path / "out.json",
                    metadata=meta,
                    duration_seconds=49.5,
                ),
            ),
        ):
            pipeline._assemble_highlight(
                highlight=h,
                metadata=meta,
                segment_paths=[dummy_seg],
                clip_index=1,
                mode="breakdown",
                voice_intro_path=None,
                voice_outro_path=None,
            )

            assert mock_freeze.called
            _, freeze_kw = mock_freeze.call_args
            assert freeze_kw.get("freeze_duration") == 4.5
            assert freeze_kw.get("zoom") == 1.40


class TestScorerBreakdownAndCopyright:
    """Verify Scorer parses breakdown fields and applies copyright risk penalty."""

    def test_ollama_scorer_parses_breakdown_fields(self, tmp_path: Path) -> None:
        from streamclipper.scoring.ollama_scorer import OllamaScorer

        scorer = OllamaScorer(_make_config(tmp_path))
        mock_response = """[
            {
                "start": 10.0,
                "end": 45.0,
                "score": 9,
                "moment_type": "fail",
                "title": "INSANE CRASH GLITCH",
                "reason": "Vehicle launched into stratosphere",
                "best_quote": "WHAT JUST HAPPENED",
                "breakdown_commentary": "Notice the collision mesh inverted on contact.",
                "breakdown_callout": "Hitbox Inversion"
            }
        ]"""
        highlights = scorer._parse_response(mock_response, chunk_offset=0.0)
        assert len(highlights) == 1
        hl = highlights[0]
        assert hl.breakdown_commentary == "Notice the collision mesh inverted on contact."
        assert hl.breakdown_callout == "Hitbox Inversion"
        assert hl.moment_type == "fail"

    def test_copyright_risk_penalty_applied_during_scoring(self, tmp_path: Path) -> None:
        from streamclipper.models import TranscriptionResult, TranscriptSegment
        from streamclipper.scoring.scorer import Scorer

        # Mock candidates
        h_clean = _make_highlight(moment_type="hype", music_risk="CLEAN")
        h_clean.score = 9.0
        h_clean.start = 100.0
        h_clean.end = 145.0

        h_music = _make_highlight(moment_type="hype", music_risk="HIGH_MUSIC_DETECTED")
        h_music.score = 9.0
        h_music.start = 200.0
        h_music.end = 245.0

        class DummyScorer(Scorer):
            def _score_chunk(self, chunk, loudness, chat):
                return [h_clean, h_music]

        config = _make_config(tmp_path)
        scorer = DummyScorer(config)

        transcript = TranscriptionResult(
            segments=[
                TranscriptSegment(text="Hello world", start=100.0, end=145.0, words=[])
            ],
            full_text="Hello world",
        )

        dummy_wav = tmp_path / "audio.wav"
        dummy_wav.write_bytes(b"dummy-audio-bytes")

        with patch("streamclipper.analysis.copyright_risk.CopyrightRiskDetector.detect_highlight") as mock_detect:
            # First highlight returns clean, second returns HIGH_MUSIC_DETECTED
            mock_detect.side_effect = [
                ("CLEAN", 0.10),
                ("HIGH_MUSIC_DETECTED", 0.85),
            ]
            selected = scorer.score_and_select(transcript, [], [], audio_path=dummy_wav)
            assert len(selected) == 2
            clean_hl = next(h for h in selected if h.music_risk == "CLEAN")
            music_hl = next(h for h in selected if h.music_risk == "HIGH_MUSIC_DETECTED")
            assert clean_hl.final_score == 9.0
            # 0.80x penalty applied: 9.0 * 0.80 = 7.20
            assert music_hl.final_score == pytest.approx(7.20)


class TestCLIBreakdownIntegration:
    """Verify CLI accepts --mode breakdown."""

    def test_cli_mode_breakdown_option(self, tmp_path: Path) -> None:
        from click.testing import CliRunner
        from streamclipper.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["clip", "--help"])
        assert result.exit_code == 0
        assert "breakdown" in result.output
        assert "--voice-breakdown" in result.output

        auto_result = runner.invoke(cli, ["auto", "--help"])
        assert auto_result.exit_code == 0
        assert "breakdown" in auto_result.output
        assert "--voice-breakdown" in auto_result.output

    def test_process_streams_parallel_accepts_voice_breakdown_path(
        self, tmp_path: Path
    ) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        voice_bd = tmp_path / "custom_bd.wav"
        voice_bd.write_bytes(b"RIFF....WAVE")

        captured_kwargs: dict[str, Any] = {}

        async def _fake_process(url: str, **kwargs: Any) -> list[Any]:
            captured_kwargs.update(kwargs)
            return []

        with patch.object(pipeline, "process_stream", side_effect=_fake_process):
            results = asyncio.run(
                pipeline.process_streams_parallel(
                    ["https://youtube.com/watch?v=abc"],
                    voice_breakdown_path=voice_bd,
                )
            )

        assert len(results) == 1
        assert captured_kwargs.get("voice_breakdown_path") == voice_bd

    def test_ollama_scorer_generate_metadata_breakdown_mode(
        self, tmp_path: Path
    ) -> None:
        from streamclipper.scoring.ollama_scorer import OllamaScorer

        config = _make_config(tmp_path)
        scorer = OllamaScorer(config)

        h_fail = _make_highlight(moment_type="fail")
        h_fail.reason = "Crash into wall"

        with (
            patch.object(scorer, "generate_narrative", return_value=("Intro", "Outro", "Badge")),
            patch.object(scorer, "_generate_caption_and_hashtags", return_value=("Caption", ["GTA6"])),
            patch.object(scorer, "_generate_hook_title", return_value="Epic Crash"),
        ):
            meta = scorer.generate_metadata(
                h_fail,
                streamer="GamerFleet",
                game="GTA 6",
                source_url="https://youtube.com/watch?v=123",
                mode="breakdown",
            )
            assert meta.transformation_mode == "breakdown"
            assert meta.breakdown_commentary != ""


