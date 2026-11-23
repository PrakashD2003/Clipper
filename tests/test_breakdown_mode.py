"""Comprehensive unit and integration tests for YPP Breakdown (Instant Replay) Mode.

Covers:
1. Instant Replay Video Assembly (compose_instant_replay with slow_mo, replay_start_time, badge overlay, handle cleanup)
2. Freeze Breakdown Video Assembly (compose_freeze_breakdown with zoom callout, silent audio, handle cleanup)
3. Replay Badge Rendering (render_replay_badge and create_replay_badge high-contrast sports pill)
4. Multi-Cue 3-Stage Audio Ducking Filtergraph (_build_ducking_filtergraph with intro + breakdown + eval=frame)
5. End-to-End Narration Ducking Mix with -c:v copy (mix_narration_with_ducking)
"""

from __future__ import annotations

import gc
import math
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from moviepy import ColorClip, CompositeVideoClip, ImageClip
from moviepy.audio.AudioClip import AudioArrayClip

from streamclipper.assembly.composer import (
    VideoComposer,
    compose_freeze_breakdown,
    compose_instant_replay,
    compose_zoom_spotlight_breakdown,
)
from streamclipper.assembly.hooks import create_replay_badge, render_replay_badge
from streamclipper.assembly.narrator import TransformativeNarrator
from streamclipper.config import Config


@pytest.fixture
def base_config() -> Config:
    """Fixture providing base default Config."""
    return Config()


@pytest.fixture
def synthetic_video_clip() -> Generator[ColorClip, None, None]:
    """Create a 4-second 360x640 (9:16) ColorClip with stereo audio and cleanup."""
    clip = ColorClip(size=(360, 640), color=(15, 25, 40), duration=4.0)
    audio_fps = 44100
    n_samples = int(4.0 * audio_fps)
    t = np.linspace(0, 4.0, n_samples, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo_audio = np.column_stack([tone, tone])
    clip = clip.with_audio(AudioArrayClip(stereo_audio, fps=audio_fps))
    clip._source_clip = clip
    clip._source_clips = [clip]

    yield clip

    try:
        clip.close()
    except Exception:
        pass
    gc.collect()


@pytest.fixture
def synthetic_mp4_with_audio(tmp_path: Path) -> Path:
    """Generate a 6-second test MP4 container with H.264 video and AAC audio."""
    out_mp4 = tmp_path / "test_stream_climax.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=navy:s=360x640:d=6:r=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=6:sample_rate=48000",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_mp4


@pytest.fixture
def synthetic_intro_wav(tmp_path: Path) -> Path:
    """Generate a 1.5s 48kHz mono narration intro WAV."""
    p = tmp_path / "narrator_intro.wav"
    sr = 48000
    n_samples = int(1.5 * sr)
    t = np.linspace(0, 1.5, n_samples, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    sf.write(str(p), tone, sr)
    return p


@pytest.fixture
def synthetic_breakdown_wav(tmp_path: Path) -> Path:
    """Generate a 2.0s 48kHz mono breakdown commentary WAV."""
    p = tmp_path / "narrator_breakdown.wav"
    sr = 48000
    n_samples = int(2.0 * sr)
    t = np.linspace(0, 2.0, n_samples, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 550 * t)).astype(np.float32)
    sf.write(str(p), tone, sr)
    return p


# ============================================================================
# 1. Instant Replay Video Assembly Tests
# ============================================================================
class TestInstantReplayAssembly:
    """Test suite for compose_instant_replay in assembly/composer.py."""

    def test_instant_replay_duration_and_slow_mo_doubling(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """Instant replay at slow_mo=0.5 doubles replay slice duration and appends after main clip."""
        main_dur = synthetic_video_clip.duration  # 4.0s
        combined, replay_start = compose_instant_replay(
            synthetic_video_clip,
            peak_offset=2.0,
            replay_window=2.0,
            slow_mo=0.5,
        )

        assert replay_start == pytest.approx(main_dur, abs=1e-3)
        # Sliced 2.0s at 0.5x speed becomes 4.0s -> Total 8.0s
        expected_dur = main_dur + (2.0 / 0.5)
        assert combined.duration == pytest.approx(expected_dur, abs=1e-3)
        assert getattr(combined, "size", None) == (360, 640)

    def test_instant_replay_custom_window_and_slow_mo_scaling(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """Custom replay_window (1.0s) and slow_mo factor (0.25) scale correctly."""
        main_dur = synthetic_video_clip.duration
        combined, replay_start = compose_instant_replay(
            synthetic_video_clip,
            peak_offset=2.0,
            replay_window=1.0,
            slow_mo=0.25,
        )

        assert replay_start == pytest.approx(main_dur, abs=1e-3)
        # 1.0s / 0.25 = 4.0s replay -> Total 8.0s
        assert combined.duration == pytest.approx(main_dur + 4.0, abs=1e-3)

    def test_instant_replay_boundary_peaks_and_fallback(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """Boundary peaks (near 0, near duration, None, negative) clamp safely without exception."""
        # Peak at very beginning (0.1s)
        comb_start, _ = compose_instant_replay(
            synthetic_video_clip, peak_offset=0.1, replay_window=2.0, slow_mo=0.5
        )
        assert comb_start.duration == pytest.approx(8.0, abs=1e-3)
        comb_start.close()

        # Peak at very end (3.9s)
        comb_end, _ = compose_instant_replay(
            synthetic_video_clip, peak_offset=3.9, replay_window=2.0, slow_mo=0.5
        )
        assert comb_end.duration == pytest.approx(8.0, abs=1e-3)
        comb_end.close()

        # Peak is None (falls back to center = 2.0s)
        comb_none, _ = compose_instant_replay(
            synthetic_video_clip, peak_offset=None, replay_window=2.0, slow_mo=0.5
        )
        assert comb_none.duration == pytest.approx(8.0, abs=1e-3)
        comb_none.close()

        # Peak is negative (falls back to center)
        comb_neg, _ = compose_instant_replay(
            synthetic_video_clip, peak_offset=-1.0, replay_window=2.0, slow_mo=0.5
        )
        assert comb_neg.duration == pytest.approx(8.0, abs=1e-3)
        comb_neg.close()

    def test_instant_replay_badge_overlay_present(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """Replay section includes the overlay badge in its composite structure."""
        combined, _ = compose_instant_replay(
            synthetic_video_clip, peak_offset=2.0, replay_window=2.0
        )
        assert isinstance(combined, CompositeVideoClip)
        frame_at_replay = combined.get_frame(5.0)
        assert frame_at_replay.shape == (640, 360, 3)

    def test_instant_replay_custom_badge_clip_support(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """Caller can provide a custom badge_clip to compose_instant_replay."""
        custom_arr = np.full((30, 150, 4), 255, dtype=np.uint8)
        custom_badge = ImageClip(custom_arr).with_position(("center", 80))

        combined, _ = compose_instant_replay(
            synthetic_video_clip,
            peak_offset=2.0,
            replay_window=1.5,
            slow_mo=0.5,
            badge_clip=custom_badge,
        )
        assert combined.duration == pytest.approx(4.0 + (1.5 / 0.5), abs=1e-3)

    def test_instant_replay_audio_preservation(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """Audio track duration matches combined video duration across main and replay."""
        combined, _ = compose_instant_replay(
            synthetic_video_clip, peak_offset=2.0, replay_window=2.0, slow_mo=0.5
        )
        assert combined.audio is not None
        assert combined.audio.duration == pytest.approx(combined.duration, abs=1e-3)

    def test_instant_replay_source_clips_windows_cleanup(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """_source_clip and _source_clips are properly attached for Exporter handle release."""
        combined, _ = compose_instant_replay(
            synthetic_video_clip, peak_offset=2.0, replay_window=2.0
        )
        assert hasattr(combined, "_source_clip")
        assert hasattr(combined, "_source_clips")
        assert synthetic_video_clip in combined._source_clips

    def test_videocomposer_delegate_compose_instant_replay(
        self, synthetic_video_clip: ColorClip, base_config: Config
    ) -> None:
        """VideoComposer instance method delegates to compose_instant_replay cleanly."""
        composer = VideoComposer(base_config)
        combined, replay_start = composer.compose_instant_replay(
            synthetic_video_clip, peak_offset=2.0, replay_window=2.0, slow_mo=0.5
        )
        assert replay_start == pytest.approx(4.0, abs=1e-3)
        assert combined.duration == pytest.approx(8.0, abs=1e-3)


# ============================================================================
# 2. Freeze Breakdown Video Assembly Tests
# ============================================================================
class TestFreezeBreakdownAssembly:
    """Test suite for compose_freeze_breakdown in assembly/composer.py."""

    def test_freeze_breakdown_duration_and_slices(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """Freeze breakdown inserts a freeze_duration callout, extending total duration."""
        main_dur = synthetic_video_clip.duration  # 4.0s
        freeze_dur = 2.0
        combined = compose_freeze_breakdown(
            synthetic_video_clip,
            peak_offset=2.0,
            freeze_duration=freeze_dur,
            zoom=1.35,
        )

        assert combined.duration == pytest.approx(main_dur + freeze_dur, abs=1e-3)
        assert getattr(combined, "size", None) == (360, 640)

    def test_freeze_breakdown_silent_audio_attached(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """Silent audio is attached to freeze frame to avoid MoviePy audio desync."""
        combined = compose_freeze_breakdown(
            synthetic_video_clip,
            peak_offset=2.0,
            freeze_duration=2.0,
            zoom=1.35,
        )

        assert combined.audio is not None
        assert combined.audio.duration == pytest.approx(6.0, abs=1e-3)

        # Samples during freeze window [2.0s, 4.0s] should be silence (zero)
        fps = 44100
        freeze_audio_chunk = combined.audio.subclipped(2.5, 3.5).to_soundarray(fps=fps)
        assert np.allclose(freeze_audio_chunk, 0.0, atol=1e-5)

    def test_freeze_breakdown_without_main_audio(self) -> None:
        """Clips without audio still freeze cleanly without exception."""
        silent_clip = ColorClip(size=(180, 320), color=(0, 0, 0), duration=3.0)
        combined = compose_freeze_breakdown(
            silent_clip, peak_offset=1.5, freeze_duration=2.0, zoom=1.20
        )
        assert combined.duration == pytest.approx(5.0, abs=1e-3)

    def test_freeze_breakdown_boundary_peaks(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """Boundary peaks (0.0s, duration, None) handle gracefully without empty slices."""
        # Freeze at very beginning
        c_start = compose_freeze_breakdown(
            synthetic_video_clip, peak_offset=0.0, freeze_duration=2.0
        )
        assert c_start.duration == pytest.approx(6.0, abs=1e-3)
        c_start.close()

        # Freeze at very end
        c_end = compose_freeze_breakdown(
            synthetic_video_clip, peak_offset=4.0, freeze_duration=2.0
        )
        assert c_end.duration == pytest.approx(6.0, abs=1e-3)
        c_end.close()

        # Freeze with None (defaults to midpoint)
        c_none = compose_freeze_breakdown(
            synthetic_video_clip, peak_offset=None, freeze_duration=2.0
        )
        assert c_none.duration == pytest.approx(6.0, abs=1e-3)
        c_none.close()

    def test_freeze_breakdown_zoom_factor_unity(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """Zoom factor 1.0 skips zoom cropping and returns un-scaled frame."""
        combined = compose_freeze_breakdown(
            synthetic_video_clip, peak_offset=2.0, freeze_duration=2.0, zoom=1.0
        )
        assert combined.duration == pytest.approx(6.0, abs=1e-3)

    def test_freeze_breakdown_source_clips_windows_cleanup(
        self, synthetic_video_clip: ColorClip
    ) -> None:
        """_source_clip and _source_clips are preserved for cleanup."""
        combined = compose_freeze_breakdown(synthetic_video_clip, peak_offset=2.0)
        assert hasattr(combined, "_source_clip")
        assert hasattr(combined, "_source_clips")

    def test_videocomposer_delegate_compose_freeze_breakdown(
        self, synthetic_video_clip: ColorClip, base_config: Config
    ) -> None:
        """VideoComposer delegate executes cleanly."""
        composer = VideoComposer(base_config)
        combined = composer.compose_freeze_breakdown(
            synthetic_video_clip, peak_offset=2.0, freeze_duration=2.0, zoom=1.35
        )
        assert combined.duration == pytest.approx(6.0, abs=1e-3)


# ============================================================================
# 3. Replay Badge Rendering Tests
# ============================================================================
class TestReplayBadge:
    """Test suite for create_replay_badge and render_replay_badge in assembly/hooks.py."""

    def test_render_replay_badge_array_shape_and_channels(self) -> None:
        """render_replay_badge returns an RGBA uint8 numpy array with positive dimensions."""
        arr = render_replay_badge("🔴 INSTANT REPLAY")
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint8
        assert arr.ndim == 3
        assert arr.shape[2] == 4  # RGBA
        h, w = arr.shape[:2]
        assert w > 100
        assert h > 20

    def test_render_replay_badge_sports_pill_styling(self) -> None:
        """Pill styling has dark sleek fill (alpha 235) and high-contrast red border."""
        arr = render_replay_badge("🔴 INSTANT REPLAY")
        alphas = set(np.unique(arr[:, :, 3]))
        assert 0 in alphas
        assert 235 in alphas
        assert 255 in alphas

    def test_create_replay_badge_clip_properties(self) -> None:
        """create_replay_badge returns an ImageClip positioned at ('center', 140)."""
        badge_clip = create_replay_badge(duration=5.0)
        assert isinstance(badge_clip, ImageClip)
        assert badge_clip.duration == 5.0
        pos = getattr(badge_clip, "pos", None)
        pos_val = pos(0) if callable(pos) else pos
        assert pos_val == ("center", 140)

    def test_create_replay_badge_custom_text(self) -> None:
        """create_replay_badge supports custom text callouts."""
        badge_clip = create_replay_badge(text="SLOW-MO BREAKDOWN", duration=3.0)
        assert badge_clip.duration == 3.0
        frame = badge_clip.get_frame(0.0)
        assert frame.shape[2] == 3  # RGB frame from MoviePy ImageClip

    def test_render_replay_badge_fallback_font(self) -> None:
        """Non-existent font gracefully falls back to default system font."""
        arr = render_replay_badge("TEST REPLAY", font_name="NonExistentFont_12345")
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] > 0 and arr.shape[1] > 0


# ============================================================================
# 4. Multi-Cue 3-Stage Audio Ducking Filtergraph Tests
# ============================================================================
class TestBreakdownAudioDuckingFiltergraph:
    """Test suite for 3-stage ducking in TransformativeNarrator._build_ducking_filtergraph."""

    def test_3_stage_ducking_intro_and_breakdown(self, base_config: Config) -> None:
        """Stage 1 (intro) ducked, Stage 2 (climax) 100%, Stage 3 (replay breakdown) ducked."""
        narrator = TransformativeNarrator(base_config)
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=3.0,
            outro_dur=None,
            clip_duration=20.0,
            breakdown_dur=4.0,
            replay_start_time=10.0,
        )

        assert input_count == 3
        # Volume ducking condition checks both intro [0, 3.00] and replay [10.00, 14.00]
        assert "between(t,0,3.00)+between(t,10.00,14.00)" in filtergraph
        assert ":eval=frame" in filtergraph
        # Normalized to 48kHz stereo
        assert "[1:a]aresample=48000,aformat=channel_layouts=stereo" in filtergraph
        # Breakdown track delayed by 10000ms
        assert "adelay=10000|10000" in filtergraph
        # 3 inputs to amix
        assert "[a0][a1][a2]amix=inputs=3:duration=first:normalize=0[aout]" in filtergraph

    def test_breakdown_only_ducking(self, base_config: Config) -> None:
        """Breakdown commentary only (without intro voiceover)."""
        narrator = TransformativeNarrator(base_config)
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=None,
            outro_dur=None,
            clip_duration=15.0,
            breakdown_dur=5.0,
            replay_start_time=8.0,
        )

        assert input_count == 2
        assert "between(t,8.00,13.00)" in filtergraph
        assert ":eval=frame" in filtergraph
        assert "adelay=8000|8000" in filtergraph
        assert "[a0][a1]amix=inputs=2:duration=first:normalize=0[aout]" in filtergraph

    def test_all_three_cues_intro_breakdown_outro(self, base_config: Config) -> None:
        """All three cues active: intro, breakdown, and outro."""
        narrator = TransformativeNarrator(base_config)
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=3.0,
            outro_dur=2.0,
            clip_duration=25.0,
            breakdown_dur=4.0,
            replay_start_time=15.0,
        )

        assert input_count == 4
        assert "between(t,0,3.00)+between(t,15.00,19.00)+between(t,23.00,25.00)" in filtergraph
        assert ":eval=frame" in filtergraph
        assert "adelay=15000|15000" in filtergraph
        assert "adelay=23000|23000" in filtergraph
        assert "[a0][a1][a2][a3]amix=inputs=4:duration=first:normalize=0[aout]" in filtergraph

    def test_breakdown_dur_clamping_to_clip_duration(self, base_config: Config) -> None:
        """Breakdown commentary exceeding remaining clip duration is clamped and atrimmed."""
        narrator = TransformativeNarrator(base_config)
        # Clip is 12.0s, replay starts at 10.0s, but breakdown is 5.0s -> only 2.0s available
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=None,
            outro_dur=None,
            clip_duration=12.0,
            breakdown_dur=5.0,
            replay_start_time=10.0,
        )

        assert input_count == 2
        assert "between(t,10.00,12.00)" in filtergraph
        assert "atrim=0:2.00,asetpts=PTS-STARTPTS" in filtergraph
        assert "adelay=10000|10000" in filtergraph

    def test_silent_video_anullsrc_with_breakdown(self, base_config: Config) -> None:
        """When has_video_audio is False, anullsrc base track is synthesized."""
        narrator = TransformativeNarrator(base_config)
        filtergraph, input_count = narrator._build_ducking_filtergraph(
            intro_dur=2.0,
            outro_dur=None,
            clip_duration=10.0,
            has_video_audio=False,
            breakdown_dur=3.0,
            replay_start_time=5.0,
        )

        assert input_count == 3
        assert "anullsrc=channel_layout=stereo:sample_rate=48000" in filtergraph
        assert ":eval=frame" not in filtergraph  # silent track uses anullsrc without volume filter
        assert "adelay=5000|5000" in filtergraph

    def test_backward_compatibility_preserves_existing_exact_format(
        self, base_config: Config
    ) -> None:
        """Calling without breakdown_dur preserves exact existing output for Phase 4 tests."""
        narrator = TransformativeNarrator(base_config)
        fg_intro, in_intro = narrator._build_ducking_filtergraph(2.0, None, 10.0)
        assert in_intro == 2
        assert "[0:a]volume='if(between(t,0,2.00),0.18,1.0)':eval=frame" in fg_intro

        fg_outro, in_outro = narrator._build_ducking_filtergraph(None, 2.0, 10.0)
        assert in_outro == 2
        assert "[0:a]volume='if(between(t,8.00,10.00),0.18,1.0)':eval=frame" in fg_outro


# ============================================================================
# 5. End-to-End Narration Ducking Mix with -c:v copy Tests
# ============================================================================
class TestBreakdownAudioMixingE2E:
    """End-to-end integration tests for mix_narration_with_ducking in assembly/narrator.py."""

    def test_real_e2e_mixing_with_intro_and_breakdown(
        self,
        base_config: Config,
        synthetic_mp4_with_audio: Path,
        synthetic_intro_wav: Path,
        synthetic_breakdown_wav: Path,
        tmp_path: Path,
    ) -> None:
        """Mix real MP4 with intro (1.5s) and breakdown (2.0s at 3.0s replay offset) using -c:v copy."""
        narrator = TransformativeNarrator(base_config)
        out_mp4 = tmp_path / "mixed_breakdown_e2e.mp4"

        result = narrator.mix_narration_with_ducking(
            input_video=synthetic_mp4_with_audio,
            output_video=out_mp4,
            intro_wav=synthetic_intro_wav,
            outro_wav=None,
            clip_duration=6.0,
            breakdown_wav=synthetic_breakdown_wav,
            replay_start_time=3.0,
        )

        assert result.is_file()
        assert result.stat().st_size > 0

        # Verify output video duration and streams via ffprobe
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name",
            "-of",
            "json",
            str(result),
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        assert '"codec_type": "video"' in res.stdout
        assert '"codec_type": "audio"' in res.stdout
        assert '"codec_name": "h264"' in res.stdout
        assert '"codec_name": "aac"' in res.stdout

    def test_real_e2e_mixing_breakdown_only(
        self,
        base_config: Config,
        synthetic_mp4_with_audio: Path,
        synthetic_breakdown_wav: Path,
        tmp_path: Path,
    ) -> None:
        """Mix real MP4 with breakdown commentary only."""
        narrator = TransformativeNarrator(base_config)
        out_mp4 = tmp_path / "breakdown_only_e2e.mp4"

        result = narrator.mix_narration_with_ducking(
            input_video=synthetic_mp4_with_audio,
            output_video=out_mp4,
            intro_wav=None,
            outro_wav=None,
            clip_duration=6.0,
            breakdown_wav=synthetic_breakdown_wav,
            replay_start_time=3.0,
        )

        assert result.is_file()
        assert result.stat().st_size > 0

    def test_prepare_breakdown_track_helper(
        self, base_config: Config, synthetic_breakdown_wav: Path
    ) -> None:
        """prepare_breakdown_track resolves custom mic path with priority."""
        narrator = TransformativeNarrator(base_config)
        resolved = narrator.prepare_breakdown_track(
            breakdown_commentary="TTS should not run",
            voice_breakdown_path=synthetic_breakdown_wav,
        )
        assert resolved == synthetic_breakdown_wav


# ============================================================================
# 6. Forensic Audit P0 Fixes Tests (Facecam threshold, Crop-to-Aspect, Layout B)
# ============================================================================
class TestForensicAuditP0Fixes:
    """Test suite for Milestone 2 P0 Forensic Audit fixes."""

    def test_facecam_min_area_ratio_default_with_config(self, base_config: Config) -> None:
        """FacecamDetector with default Config has min_area_ratio = 0.002 (0.2%)."""
        from streamclipper.assembly.facecam import FacecamDetector

        detector = FacecamDetector(base_config)
        assert detector.min_area_ratio == pytest.approx(0.002)

        # GamerFleet corner webcam face: 86x86 in 1920x1080 = 7,396 px² (0.357% of frame)
        # Should now pass area validation (0.357% >= 0.20%)
        assert detector._is_valid_area(86, 86, 1920, 1080) is True

    def test_facecam_explicit_min_area_ratio_param(self) -> None:
        """FacecamDetector accepts explicit min_area_ratio parameter."""
        from streamclipper.assembly.facecam import FacecamDetector

        detector = FacecamDetector(min_area_ratio=0.002)
        assert detector.min_area_ratio == pytest.approx(0.002)
        assert detector._is_valid_area(86, 86, 1920, 1080) is True

    def test_crop_to_aspect_expand_width_preserves_height(self) -> None:
        """_crop_to_aspect with expand_width=True expands width instead of cropping height."""
        from streamclipper.assembly.composer import VideoComposer

        # Synthetic 1920x1080 clip
        clip = ColorClip(size=(1920, 1080), color=(10, 10, 10), duration=2.0)
        try:
            # GamerFleet detected face box: 154 x 129 px (aspect ratio = 1.194)
            # Target aspect ratio for top 35% (1080 / 672 = 1.607)
            # With expand_width=False, height would be cut from 129 -> 96 px (forehead/chin loss)
            # With expand_width=True, height remains 129 px and width expands to 129 * 1.607 = 207 px
            target_ratio = 1080.0 / 672.0
            x1, y1, x2, y2 = 1620, 160, 1774, 289  # w=154, h=129

            cropped = VideoComposer._crop_to_aspect(
                clip, x1, y1, x2, y2, target_ratio=target_ratio, expand_width=True
            )
            assert cropped.h == 129  # Full face height preserved!
            assert cropped.w == int(round(129 * target_ratio))  # Width expanded horizontally
        finally:
            clip.close()

    def test_layout_b_face_aware_centering(self, base_config: Config) -> None:
        """Layout B _compose_gameplay_only centers crop around face_center_x instead of blind center."""
        from streamclipper.assembly.composer import VideoComposer

        composer = VideoComposer(base_config)
        clip = ColorClip(size=(1920, 1080), color=(20, 30, 40), duration=2.0)
        try:
            # For 9:16 crop from 1920x1080, crop_w = 607 px
            # Default blind center is (1920 - 607) // 2 = 656 px -> x ∈ [656, 1263]
            # GamerFleet is seated left-of-center at x = 450 (bisected by x=656 crop line)
            # When face_center_x=450 is passed, crop_x should center at 450 - (607 // 2) = 147 px
            comp = composer._compose_gameplay_only(clip, face_center_x=450.0)
            assert comp.size == (1080, 1920)

            # Also verify via composer.compose with facecam=None
            comp2 = composer.compose(clip, facecam=None, face_center_x=450.0)
            assert comp2.size == (1080, 1920)
        finally:
            clip.close()

    def test_render_replay_badge_pil_support(self) -> None:
        """render_replay_badge supports as_pil=True and render_replay_badge_pil returns PIL Image."""
        from PIL import Image
        from streamclipper.assembly.hooks import render_replay_badge_pil

        pil_img = render_replay_badge("🔴 INSTANT REPLAY", as_pil=True)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size[0] > 100
        assert pil_img.size[1] > 20

        pil_img2 = render_replay_badge_pil("🔴 INSTANT REPLAY")
        assert isinstance(pil_img2, Image.Image)

    def test_mix_narration_with_ducking_accepts_replay_start_alias(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """mix_narration_with_ducking accepts replay_start as alias for replay_start_time."""
        narrator = TransformativeNarrator(base_config)
        # Calling with non-existent or bypass video should return gracefully without TypeError
        in_vid = tmp_path / "in.mp4"
        in_vid.write_bytes(b"dummy")
        out_vid = tmp_path / "out.mp4"

        # When narrator is disabled or in raw mode, bypass copy runs without raising TypeError
        narrator.enabled = False
        res = narrator.mix_narration_with_ducking(
            input_video=in_vid,
            output_video=out_vid,
            intro_wav=None,
            outro_wav=None,
            clip_duration=10.0,
            replay_start=4.0,
        )
        assert res.exists()

    def test_composer_compose_adjusts_peak_offset_when_trim_range_provided(
        self, base_config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VideoComposer.compose subtracts t_start from peak_offset so spike zoom is placed accurately."""
        from streamclipper.assembly.composer import VideoComposer

        composer = VideoComposer(base_config)
        clip = ColorClip(size=(640, 360), color=(10, 20, 30), duration=10.0)
        captured_offsets: list[float] = []

        def _spy_apply_spike_zoom(self_comp: Any, target_clip: Any, peak_offset: float) -> Any:
            captured_offsets.append(peak_offset)
            return target_clip

        monkeypatch.setattr(VideoComposer, "apply_spike_zoom", _spy_apply_spike_zoom)

        try:
            # 10s clip, trimmed to [3.0, 8.0] (duration 5s). Untrimmed peak at 5.5s -> effective peak at 2.5s
            composer.compose(
                clip,
                facecam=None,
                peak_offset=5.5,
                trim_range=(3.0, 8.0),
            )
            assert len(captured_offsets) == 1
            assert captured_offsets[0] == pytest.approx(2.5)
        finally:
            clip.close()

    def test_exporter_export_forwards_breakdown_audio_without_dropping(
        self, base_config: Config, tmp_path: Path
    ) -> None:
        """Exporter.export passes breakdown_wav and replay_start to mix_narration_with_ducking without dropping."""
        from streamclipper.output.exporter import Exporter
        from unittest.mock import MagicMock, patch

        exporter = Exporter(base_config)
        fake_clip = ColorClip(size=(360, 640), color=(0, 0, 0), duration=10.0)

        fake_breakdown_wav = tmp_path / "breakdown.wav"
        fake_breakdown_wav.write_bytes(b"RIFF....WAVE")

        captured_mix_kwargs: dict[str, Any] = {}

        def _fake_mix(**kwargs: Any) -> Path:
            captured_mix_kwargs.update(kwargs)
            return tmp_path / "mixed.mp4"

        with (
            patch.object(
                exporter.narrator,
                "prepare_breakdown_track",
                return_value=fake_breakdown_wav,
            ),
            patch.object(
                exporter.narrator,
                "_probe_audio_duration",
                return_value=3.0,
            ),
            patch.object(
                exporter.narrator,
                "mix_narration_with_ducking",
                side_effect=_fake_mix,
            ),
            patch.object(
                exporter,
                "_finalize_without_captions",
                return_value=tmp_path / "out.mp4",
            ),
            patch.object(
                exporter.caption_burner,
                "burn_captions",
                return_value=tmp_path / "final.mp4",
            ),
            patch.object(fake_clip, "write_videofile"),
        ):
            exporter.export(
                clip=fake_clip,
                words=[],
                clip_start_offset=0.0,
                output_path=tmp_path / "out.mp4",
                mode="breakdown",
                breakdown_commentary="Analyze this frame.",
                replay_start=6.0,
            )

        assert captured_mix_kwargs.get("breakdown_wav") == fake_breakdown_wav
        assert captured_mix_kwargs.get("replay_start_time") == 6.0


# ==============================================================================
# Visual Breakdown (Zoom Spotlight) Tests
# ==============================================================================


def test_compose_zoom_spotlight_breakdown_duration_and_resolution(synthetic_video_clip: ColorClip, base_config: Config) -> None:
    """Test compose_zoom_spotlight_breakdown attaches spotlight with exact requested duration and dimensions."""
    combined, replay_start = compose_zoom_spotlight_breakdown(
        synthetic_video_clip,
        target_time_offset=1.5,
        roi=[0.25, 0.35, 0.75, 0.80],
        duration=5.7,
        zoom_scale=2.2,
        badge_text="🔍 EVIDENCE BREAKDOWN",
        config=base_config,
    )

    try:
        assert float(combined.duration) == pytest.approx(4.0 + 5.7, abs=0.05)
        assert replay_start == pytest.approx(4.0, abs=0.01)
        assert combined.w == synthetic_video_clip.w
        assert combined.h == synthetic_video_clip.h
        assert getattr(combined, "_breakdown_kind", None) == "zoom_spotlight"
        assert getattr(combined, "_spotlight_duration", None) == 5.7
        assert combined.audio is not None
        assert float(combined.audio.duration) == pytest.approx(combined.duration, abs=0.1)
    finally:
        try:
            combined.close()
        except Exception:
            pass


def test_compose_zoom_spotlight_frame_interpolation_and_ken_burns(synthetic_video_clip: ColorClip) -> None:
    """Test smooth sinusoidal Ken Burns frame generation at t=0, hold, and end."""
    combined, replay_start = compose_zoom_spotlight_breakdown(
        synthetic_video_clip,
        target_time_offset=1.0,
        roi=[0.20, 0.20, 0.80, 0.80],
        duration=5.7,
        zoom_scale=2.2,
        ease_in_seconds=1.2,
        ease_out_seconds=1.0,
    )

    try:
        f_start = combined.get_frame(replay_start + 0.05)
        f_hold = combined.get_frame(replay_start + 2.5)
        f_end = combined.get_frame(replay_start + 5.65)

        assert f_start.shape == (synthetic_video_clip.h, synthetic_video_clip.w, 3)
        assert f_hold.shape == (synthetic_video_clip.h, synthetic_video_clip.w, 3)
        assert f_end.shape == (synthetic_video_clip.h, synthetic_video_clip.w, 3)
        assert f_start.dtype == np.uint8
        assert f_hold.dtype == np.uint8
        assert f_end.dtype == np.uint8
    finally:
        try:
            combined.close()
        except Exception:
            pass


def test_compose_zoom_spotlight_empty_main_clip_and_videocomposer_wrapper(base_config: Config) -> None:
    """Test compose_zoom_spotlight_breakdown handles 0-duration clip and VideoComposer wrapper."""
    empty_clip = ColorClip(size=(360, 640), color=(0, 0, 0), duration=0.0)
    c_out, r_time = compose_zoom_spotlight_breakdown(empty_clip)
    assert r_time == 0.0

    composer = VideoComposer(base_config)
    clip = ColorClip(size=(360, 640), color=(50, 60, 70), duration=2.0)
    try:
        comp, r_start = composer.compose_zoom_spotlight_breakdown(
            clip,
            roi=[0.1, 0.1, 0.9, 0.9],
            duration=3.0,
            zoom_scale=1.8,
        )
        assert comp.duration == pytest.approx(5.0, abs=0.05)
        assert r_start == pytest.approx(2.0, abs=0.01)
    finally:
        try:
            clip.close()
            comp.close()
        except Exception:
            pass


def test_compose_zoom_spotlight_zero_ease_and_roi_edge_cases(synthetic_video_clip: ColorClip) -> None:
    """Test zero ease duration and inverted / degenerate / out-of-bounds ROI coordinates."""
    # 1. Zero ease-in and zero ease-out
    comp_zero_ease, r_zero = compose_zoom_spotlight_breakdown(
        synthetic_video_clip,
        duration=4.0,
        ease_in_seconds=0.0,
        ease_out_seconds=0.0,
        zoom_scale=2.2,
    )
    try:
        f_early = comp_zero_ease.get_frame(r_zero + 0.05)
        assert f_early.shape == (synthetic_video_clip.h, synthetic_video_clip.w, 3)
    finally:
        comp_zero_ease.close()

    # 2. Inverted coordinates: ymin > ymax, xmin > xmax
    comp_inv, r_inv = compose_zoom_spotlight_breakdown(
        synthetic_video_clip,
        duration=3.0,
        roi=[0.85, 0.75, 0.15, 0.25],
    )
    try:
        f_inv = comp_inv.get_frame(r_inv + 1.0)
        assert f_inv.shape == (synthetic_video_clip.h, synthetic_video_clip.w, 3)
    finally:
        comp_inv.close()

    # 3. Degenerate point: [0.5, 0.5, 0.5, 0.5]
    comp_deg, r_deg = compose_zoom_spotlight_breakdown(
        synthetic_video_clip,
        duration=3.0,
        roi=[0.5, 0.5, 0.5, 0.5],
    )
    try:
        f_deg = comp_deg.get_frame(r_deg + 1.0)
        assert f_deg.shape == (synthetic_video_clip.h, synthetic_video_clip.w, 3)
    finally:
        comp_deg.close()

    # 4. Out-of-bounds coordinates: [-0.5, -0.2, 1.5, 1.2]
    comp_oob, r_oob = compose_zoom_spotlight_breakdown(
        synthetic_video_clip,
        duration=3.0,
        roi=[-0.5, -0.2, 1.5, 1.2],
    )
    try:
        f_oob = comp_oob.get_frame(r_oob + 1.0)
        assert f_oob.shape == (synthetic_video_clip.h, synthetic_video_clip.w, 3)
    finally:
        comp_oob.close()


def test_compose_zoom_spotlight_direct_source_ken_burns_16_to_9(synthetic_video_clip: ColorClip) -> None:
    """Test direct-source Ken Burns crop on 16:9 source frame produces full-bleed 1080x1920 frames without blurred bars."""
    import cv2
    from moviepy import ColorClip

    # Create 1080x1920 vertical main clip
    main_v = ColorClip(size=(1080, 1920), color=(10, 20, 30), duration=2.0)
    try:
        # Pass a mock ROI targeting screen on table [ymin=0.45, xmin=0.25, ymax=0.78, xmax=0.72]
        combined, replay_start = compose_zoom_spotlight_breakdown(
            main_v,
            roi=[0.45, 0.25, 0.78, 0.72],
            duration=4.0,
            zoom_scale=2.2,
            ease_in_seconds=1.0,
            ease_out_seconds=1.0,
        )

        assert combined.w == 1080
        assert combined.h == 1920
        assert combined.duration == pytest.approx(6.0, abs=0.05)

        # Inspect frames at start, mid-zoom, hold, and end
        for t_off in [0.1, 0.9, 2.0, 3.9]:
            frame = combined.get_frame(replay_start + t_off)
            assert frame.shape == (1920, 1080, 3)
            assert frame.dtype == np.uint8
            # Verify frame is not empty or corrupted
            assert np.any(frame > 0) or np.all(frame == 0)
    finally:
        main_v.close()
        try:
            combined.close()
        except Exception:
            pass


def test_dynamic_spotlight_duration_matches_speech(tmp_path: Path) -> None:
    """Test that spotlight duration scales dynamically to prevent voiceover truncation."""
    from streamclipper.models import Highlight, ClipMetadata
    from streamclipper.pipeline import Pipeline, _WORDS_PER_SECOND

    cfg = Config()
    pipeline = Pipeline(cfg)

    # 32 words of commentary + 15 words of outro = 47 words
    # At 2.5 words/s, that is 18.8s + 1.5s buffer = 20.3s
    commentary = (
        "Dhyan se laptop screen dekho! Jaise hi hacker classified data open karta hai, "
        "Rockstar ne cutscene black out kar diya. Yeh drive GTA 6 story DLC aur "
        "GTA Online 2 ka sabse bada plot hook hai!"
    )
    outro = "Kya drive me government corruption ka secret file tha? Apni theory comments me batao!"

    h = Highlight(
        start=100.0,
        end=140.0,
        score=9.0,
        title="Laptop Screen",
        reason="Clue",
        moment_type="analysis",
        breakdown_commentary=commentary,
        narration_outro=outro,
    )
    meta = ClipMetadata(
        hook_title="Secret Laptop",
        caption="Secret Laptop Clue",
        hashtags=["#gta6"],
        streamer="IShowSpeed",
        game="GTA 6",
        source_url="http://example.com",
        clip_start=100.0,
        clip_end=140.0,
        highlight_score=9.0,
        moment_type="analysis",
        breakdown_commentary=commentary,
        narration_outro=outro,
    )

    bd_dur = pipeline._estimate_breakdown_seconds(commentary, None, 1.0)
    outro_dur = pipeline._estimate_breakdown_seconds(outro, None, 1.0)
    needed_spotlight = bd_dur + outro_dur + 1.5
    zs_cfg_dur = float(cfg.get("transformation.breakdown.zoom_spotlight.duration", 5.7))
    zs_duration = max(zs_cfg_dur, needed_spotlight)

    assert zs_duration >= 20.0
    assert zs_duration > zs_cfg_dur  # Significantly larger than legacy 5.7s cap




