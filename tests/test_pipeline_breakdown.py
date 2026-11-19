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
        for m_type in ("analysis", "graphics_detail", "lore_secret"):
            h = _make_highlight(moment_type=m_type)
            mode, beats = pipeline._resolve_highlight_mode("auto", h)
            assert mode == "breakdown"
            assert beats == []

    def test_auto_keeps_funny_fail_on_narrative(self, tmp_path: Path) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        h = _make_highlight(moment_type="fail")
        mode, _ = pipeline._resolve_highlight_mode("auto", h)
        assert mode == "narrative"

    def test_auto_prefers_story_beats_over_moment_type_breakdown(
        self, tmp_path: Path
    ) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        beats_data = [
            {"start": 100.0, "end": 115.0, "label": "Setup"},
            {"start": 130.0, "end": 145.0, "label": "Payoff"},
        ]
        h = _make_highlight(moment_type="analysis", sub_clips=beats_data)
        mode, beats = pipeline._resolve_highlight_mode("auto", h)
        assert mode == "story"
        assert len(beats) == 2

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
        fake_clip._trim_start = 3.0
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
        # v3 defaults drop early chunks / high-music picks; disable so the penalty path is exercised.
        config._data.setdefault("analysis", {}).update(
            {"skip_intro_minutes": 0, "skip_high_music": False}
        )
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




class TestBreakdownReviewRegressions:
    """Regressions for freeze-frame sync, cue windows, and source dedup."""

    def test_shift_timeline_after_freeze_moves_only_post_peak_events(self) -> None:
        words = [
            WordTimestamp(text="before", start=405.0, end=405.5, confidence=0.9),
            WordTimestamp(text="after", start=412.0, end=412.5, confidence=0.9),
        ]
        chat = [{"message": "early", "offset_seconds": 404.0}, {"message": "late", "time": 415.0}]
        new_words, new_chat = Pipeline._shift_timeline_after(
            words, chat, clip_start_offset=400.0, at=10.0, delta=4.0
        )
        assert [w.start for w in new_words] == [405.0, 416.0]
        assert new_chat[0]["offset_seconds"] == 404.0
        assert new_chat[1]["time"] == 419.0
        assert words[1].start == 412.0  # inputs not mutated

    def test_breakdown_windows_place_commentary_at_freeze_point(self) -> None:
        from streamclipper.assembly.narrator import TransformativeNarrator

        win = TransformativeNarrator.compute_breakdown_windows(
            intro_dur=3.0, outro_dur=3.0, breakdown_dur=4.0, clip_duration=30.0,
            replay_start_time=12.0,
        )
        assert win["eff_intro"] == 3.0
        assert win["breakdown_start"] == 12.0
        assert win["breakdown_end"] == 16.0
        assert win["outro_start"] == 27.0

    def test_breakdown_windows_outro_never_overlaps_breakdown(self) -> None:
        from streamclipper.assembly.narrator import TransformativeNarrator

        win = TransformativeNarrator.compute_breakdown_windows(
            intro_dur=None, outro_dur=6.0, breakdown_dur=5.0, clip_duration=20.0,
            replay_start_time=16.0,
        )
        assert win["breakdown_end"] == 20.0
        assert win["outro_start"] >= win["breakdown_end"]

    def test_instant_replay_source_dedup_never_calls_clip_eq(self) -> None:
        from streamclipper.assembly import composer as composer_mod

        class NoEq:
            def __eq__(self, other):  # pragma: no cover - must not be called
                raise AssertionError("Clip.__eq__ must not be used for dedup")

            __hash__ = object.__hash__

        a, b = NoEq(), NoEq()
        main = MagicMock(duration=10.0, w=1080, h=1920)
        main._source_clips = [a, b]
        main._source_clip = a
        with (
            patch.object(composer_mod, "CompositeVideoClip", return_value=MagicMock(audio=None)),
            patch.object(composer_mod, "concatenate_videoclips", return_value=MagicMock()),
        ):
            combined, _ = composer_mod.compose_instant_replay(
                main, peak_offset=5.0, badge_clip=MagicMock()
            )
        assert combined._source_clips[0] is a
        assert combined._source_clips[1] is b
        assert sum(1 for s in combined._source_clips if s is a) == 1


# ---------------------------------------------------------------------------
# Output Quality v3 - Batch E (pipeline integration)
# ---------------------------------------------------------------------------


def _v3_meta(**overrides: Any) -> ClipMetadata:
    base: dict[str, Any] = dict(
        hook_title="Epic Wipeout",
        caption="Wait for it...",
        hashtags=["GTA6"],
        streamer="Fleet",
        game="GTA 6",
        source_url="https://youtube.com/watch?v=xyz",
        clip_start=100.0,
        clip_end=145.0,
        highlight_score=8.5,
        moment_type="funny",
        transformation_mode="narrative",
        narration_intro="Look at this absolute chaos right here.",
        narration_outro="And that is why you never trust the brakes.",
    )
    base.update(overrides)
    return ClipMetadata(**base)


def _run_assemble(
    tmp_path: Path,
    *,
    config: Config | None = None,
    meta: ClipMetadata | None = None,
    mode: str = "narrative",
    facecam: Any = None,
    highlight: Highlight | None = None,
    **assemble_kwargs: Any,
) -> dict[str, Any]:
    """Run Pipeline._assemble_highlight with every heavy collaborator mocked; return call info."""
    cfg = config or _make_config(tmp_path)
    pipeline = Pipeline(cfg)
    h = highlight or _make_highlight(moment_type="funny")
    h.peak_time = 120.0
    meta = meta or _v3_meta()
    seg = tmp_path / "seg.mp4"
    seg.write_bytes(b"dummy-video")
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")
    clip = MagicMock()
    clip.duration = 45.0
    clip._trim_start = 3.0
    clip._layout = "LAYOUT"
    with (
        patch(
            "streamclipper.assembly.facecam.FacecamDetector.detect", return_value=facecam
        ) as m_detect,
        patch(
            "streamclipper.assembly.composer.VideoComposer.compose", return_value=clip
        ) as m_compose,
        patch(
            "streamclipper.assembly.composer.VideoComposer.compose_instant_replay",
            return_value=(clip, 45.0),
        ) as m_replay,
        patch(
            "streamclipper.assembly.hooks.HookTitleOverlay.add_hook", return_value=clip
        ) as m_hook,
        patch("streamclipper.output.exporter.Exporter.export", return_value=out) as m_export,
        patch(
            "streamclipper.output.organizer.FileOrganizer.save_clip",
            return_value=ClipResult(
                video_path=out,
                metadata_path=tmp_path / "out.json",
                metadata=meta,
                duration_seconds=45.0,
            ),
        ),
    ):
        result = pipeline._assemble_highlight(
            highlight=h,
            metadata=meta,
            segment_paths=[seg],
            clip_index=1,
            mode=mode,
            voice_intro_path=None,
            voice_outro_path=None,
            **assemble_kwargs,
        )
    return {
        "result": result,
        "detect": m_detect,
        "compose": m_compose,
        "replay": m_replay,
        "hook": m_hook,
        "export": m_export,
        "pipeline": pipeline,
        "out": out,
    }


class TestOutputQualityV3Pipeline:
    def test_E1_facecam_detect_gets_segments_and_streamer(self, tmp_path: Path) -> None:
        info = _run_assemble(tmp_path)
        args, kwargs = info["detect"].call_args
        assert isinstance(args[0], list)
        assert kwargs["streamer"] == "Fleet"

    def test_E2_gate_skips_non_gameplay_without_facecam(self, tmp_path: Path) -> None:
        info = _run_assemble(tmp_path, gate_non_gameplay=True, facecam=None)
        assert info["result"] is None
        assert not info["export"].called

    def test_E2_gate_reaction_layout_with_facecam_and_ok_moment(self, tmp_path: Path) -> None:
        info = _run_assemble(tmp_path, gate_non_gameplay=True, facecam=MagicMock())
        assert info["compose"].call_args[1]["layout_kind"] == "reaction"
        bad = _run_assemble(
            tmp_path,
            gate_non_gameplay=True,
            facecam=MagicMock(),
            meta=_v3_meta(moment_type="analysis"),
        )
        assert bad["result"] is None

    def test_E3_reaction_moment_uses_reaction_layout_and_passes_layout(
        self, tmp_path: Path
    ) -> None:
        info = _run_assemble(
            tmp_path, facecam=MagicMock(), meta=_v3_meta(moment_type="reaction")
        )
        assert info["compose"].call_args[1]["layout_kind"] == "reaction"
        assert info["hook"].call_args[1]["layout"] == "LAYOUT"
        assert info["export"].call_args[1]["layout"] == "LAYOUT"

    def test_E3_default_layout_kind_not_forced(self, tmp_path: Path) -> None:
        info = _run_assemble(tmp_path, facecam=MagicMock())
        assert "layout_kind" not in info["compose"].call_args[1]

    def test_E4_callout_voiced_at_40_percent_of_moment(self, tmp_path: Path) -> None:
        meta = _v3_meta()
        meta.callout = "Watch the left tire"  # type: ignore[attr-defined]
        info = _run_assemble(tmp_path, meta=meta)
        kw = info["export"].call_args[1]
        assert kw["callout"] == "Watch the left tire"
        assert kw["callout_start"] == pytest.approx(0.4 * 45.0)

    def test_E4_callout_disabled_by_config(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        cfg._data.setdefault("transformation", {})["callout_enabled"] = False
        meta = _v3_meta()
        meta.callout = "Watch the left tire"  # type: ignore[attr-defined]
        info = _run_assemble(tmp_path, config=cfg, meta=meta)
        assert "callout" not in info["export"].call_args[1]

    def test_E5_budget_shrinks_replay_before_moment(self, tmp_path: Path) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        plan = pipeline._plan_budget(
            moment_dur=50.0, mode="breakdown", replay_window=5.0, replay_speed=0.5
        )
        assert plan["moment"] == 50.0  # moment untouched while the replay can shrink
        assert plan["replay_window"] == pytest.approx(4.5)
        assert plan["total"] <= 59.0 + 1e-6

    def test_E5_budget_trims_moment_when_replay_at_minimum(self, tmp_path: Path) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        plan = pipeline._plan_budget(
            moment_dur=80.0, mode="breakdown", replay_window=5.0, replay_speed=0.5
        )
        assert plan["replay_window"] == pytest.approx(2.0)
        assert plan["total"] <= 59.0 + 1e-6

    def test_E5_low_commentary_trims_moment_and_writes_ratio(self, tmp_path: Path) -> None:
        import json

        pipeline = Pipeline(_make_config(tmp_path))
        plan = pipeline._plan_budget(
            moment_dur=40.0,
            mode="narrative",
            narration_seconds=6.0,
            callout_seconds=1.0,
        )
        assert plan["moment"] < 40.0
        raw = pipeline._plan_budget(moment_dur=40.0, mode="raw")
        assert raw["moment"] == 40.0
        out = tmp_path / "clip.json"
        out.write_text("{}", encoding="utf-8")
        pipeline._annotate_clip_json(
            ClipResult(
                video_path=tmp_path / "clip.mp4",
                metadata_path=out,
                metadata=_v3_meta(),
                duration_seconds=1.0,
            ),
            {"commentary_ratio": plan["coverage"]},
        )
        assert json.loads(out.read_text(encoding="utf-8"))["commentary_ratio"] == plan["coverage"]

    def test_E5_hard_trim_cuts_overlong_clip(self, tmp_path: Path) -> None:
        pipeline = Pipeline(_make_config(tmp_path))
        video = tmp_path / "long.mp4"
        video.write_bytes(b"x")
        with (
            patch.object(Pipeline, "_probe_duration", return_value=70.0),
            patch("subprocess.run") as run,
            patch("os.replace") as repl,
        ):
            assert pipeline._hard_trim(video, 59.0) == 59.0
        cmd = run.call_args[0][0]
        assert "-t" in cmd and cmd[cmd.index("-t") + 1].startswith("59")
        assert repl.called

    def test_E6_high_music_ducks_full_clip(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        cfg._data.setdefault("analysis", {})["skip_high_music"] = False  # E6 needs skip disabled
        info = _run_assemble(
            tmp_path,
            config=cfg,
            highlight=_make_highlight(music_risk="HIGH_MUSIC_DETECTED"),
        )
        assert info["export"].call_args[1]["duck_full"] is True
        clean = _run_assemble(tmp_path, config=cfg)
        assert "duck_full" not in clean["export"].call_args[1]

    def test_E8_queued_clip_uses_queued_path(self, tmp_path: Path) -> None:
        with patch(
            "streamclipper.output.organizer.FileOrganizer.get_queued_path",
            return_value=tmp_path / "queued" / "clip.mp4",
        ) as queued:
            info = _run_assemble(tmp_path, queued=True)
        assert queued.called
        assert info["export"].call_args[1]["output_path"] == tmp_path / "queued" / "clip.mp4"

    def test_E10_blocked_streamer_returns_empty(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        cfg._data.setdefault("discovery", {})["blocked_streamers"] = ["BadStreamer"]
        pipeline = Pipeline(cfg)
        clips = asyncio.run(
            pipeline.process_stream(
                "https://youtube.com/watch?v=blocked1", streamer="badstreamer", force=True
            )
        )
        assert clips == []

    def test_E11_post_file_written_next_to_clip(self, tmp_path: Path) -> None:
        info = _run_assemble(tmp_path)
        post = tmp_path / "out_post.txt"
        assert post.is_file()
        text = post.read_text(encoding="utf-8")
        assert "Original stream by Fleet" in text
        assert "Altered or synthetic content" in text
        assert info["result"] is not None

    def test_E11_post_file_can_be_disabled(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        cfg._data.setdefault("output", {})["write_post_file"] = False
        _run_assemble(tmp_path, config=cfg)
        assert not (tmp_path / "out_post.txt").exists()

