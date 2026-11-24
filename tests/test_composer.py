"""Unit tests for 9:16 Video Composer, conditional story beats, and spike zoom."""

import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure worktree src is prioritized over editable .venv installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
import yaml
from moviepy import ColorClip, CompositeVideoClip
from moviepy.audio.AudioClip import AudioArrayClip

from streamclipper.assembly.composer import (
    VideoComposer,
    remap_timeline_for_beats,
    remap_timestamps,
)
from streamclipper.assembly.layout import Box
from streamclipper.config import Config, HardwareProfile
from streamclipper.models import FacecamRegion, WordTimestamp


def _make_config(
    tmp_path: Path, overrides: dict[str, Any] | None = None
) -> Config:
    """Create an isolated Config rooted in tmp_path with a deterministic HardwareProfile."""
    cfg_data: dict[str, Any] = {
        "general": {
            "cache_dir": str(tmp_path / ".cache"),
            "output_dir": str(tmp_path / "output"),
            "temp_dir": str(tmp_path / "temp"),
            "max_concurrent_streams": 2,
        },
        "video": {
            "resolution": [1080, 1920],
            "fps": 30,
        },
        "transformation": {
            "punch_in_zoom": 1.20,
            "zoom_duration_seconds": 3.5,
        },
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


def _make_dummy_clip(
    duration: float = 5.0,
    size: tuple[int, int] = (1920, 1080),
    color: tuple[int, int, int] = (50, 100, 150),
    with_audio: bool = True,
) -> ColorClip:
    """Create a synthetic ColorClip with optional dummy audio."""
    clip = ColorClip(size=size, color=color, duration=duration)
    if with_audio:
        fps = 44100
        samples = int(duration * fps)
        arr = np.zeros((samples, 2), dtype=np.float32)
        audio = AudioArrayClip(arr, fps=fps)
        clip = clip.with_audio(audio)
    return clip


# ============================================================================
# 1. Initialization and Configuration Tests
# ============================================================================


class TestVideoComposerInit:
    """Verify VideoComposer initialization and config parsing."""

    def test_default_config_resolution_and_zoom(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        assert composer.width == 1080
        assert composer.height == 1920
        assert composer.fps == 30
        assert composer.punch_in_zoom == pytest.approx(1.20)
        assert composer.zoom_duration == pytest.approx(3.5)

    def test_custom_config_resolution_and_zoom(self, tmp_path: Path):
        overrides = {
            "video": {"resolution": [720, 1280], "fps": 60},
            "transformation": {
                "punch_in_zoom": 1.50,
                "zoom_duration_seconds": 2.0,
            },
        }
        cfg = _make_config(tmp_path, overrides=overrides)
        composer = VideoComposer(cfg)
        assert composer.width == 720
        assert composer.height == 1280
        assert composer.fps == 60
        assert composer.punch_in_zoom == pytest.approx(1.50)
        assert composer.zoom_duration == pytest.approx(2.0)


# ============================================================================
# 2. Aspect-Preserving Cropping Tests (_crop_to_aspect)
# ============================================================================


class TestCropToAspect:
    """Verify _crop_to_aspect eliminates horizontal/vertical stretching."""

    def test_crop_square_region_to_wide_aspect(self):
        clip = _make_dummy_clip(size=(1920, 1080))
        target_ratio = 1080.0 / 672.0  # ~1.607
        cropped = VideoComposer._crop_to_aspect(
            clip, x1=1400, y1=100, x2=1800, y2=500, target_ratio=target_ratio
        )
        assert cropped.w == 400
        assert cropped.h == 249
        ratio = cropped.w / cropped.h
        assert ratio == pytest.approx(target_ratio, rel=1e-2)

    def test_crop_wide_region_to_narrow_aspect(self):
        clip = _make_dummy_clip(size=(1920, 1080))
        target_ratio = 1.0  # square target
        cropped = VideoComposer._crop_to_aspect(
            clip, x1=200, y1=100, x2=800, y2=300, target_ratio=target_ratio
        )
        assert cropped.w == 200
        assert cropped.h == 200
        assert cropped.w / cropped.h == pytest.approx(1.0)

    def test_crop_coordinates_boundary_clamped(self):
        clip = _make_dummy_clip(size=(1920, 1080))
        cropped = VideoComposer._crop_to_aspect(
            clip, x1=-50, y1=-20, x2=2000, y2=1200, target_ratio=1.0
        )
        assert cropped.w <= clip.w
        assert cropped.h <= clip.h


# ============================================================================
# 3. Layout A & B Composition Tests
# ============================================================================


class TestLayoutComposition:
    """Verify 9:16 vertical layout composition with and without facecam."""

    def test_layout_a_with_facecam_stack(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        clip = _make_dummy_clip(duration=4.0, size=(1920, 1080))
        facecam = FacecamRegion(x=1400, y=50, w=400, h=400)

        comp = composer._compose_with_facecam(clip, facecam)
        assert comp.size == (1080, 1920)
        assert comp.duration == pytest.approx(4.0)
        assert comp.audio is not None

    def test_layout_b_gameplay_only(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        clip = _make_dummy_clip(duration=4.0, size=(1920, 1080))

        comp = composer._compose_gameplay_only(clip)
        assert comp.size == (1080, 1920)
        assert comp.duration == pytest.approx(4.0)
        assert comp.audio is not None

    def test_compose_routes_facecam_vs_gameplay(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        clip = _make_dummy_clip(duration=3.0, size=(1920, 1080))
        facecam = FacecamRegion(x=100, y=50, w=300, h=300)

        result_a = composer.compose(clip, facecam=facecam)
        assert result_a.size == (1080, 1920)
        assert result_a._source_clip is clip
        assert result_a._source_clips == [clip]

        result_b = composer.compose(clip, facecam=None)
        assert result_b.size == (1080, 1920)
        assert result_b._source_clip is clip
        assert result_b._source_clips == [clip]

    def test_full_bleed_stacked_composition(self, tmp_path: Path):
        cfg = _make_config(tmp_path, overrides={"video": {"game_fit": "full_bleed"}})
        composer = VideoComposer(cfg)
        assert composer.game_fit == "full_bleed"
        clip = _make_dummy_clip(duration=3.0, size=(1920, 1080))
        facecam = FacecamRegion(x=100, y=50, w=300, h=300)
        comp = composer.compose(clip, facecam=facecam)
        assert comp._layout.kind == "stacked"
        assert comp._layout.facecam_box.h == 640
        assert comp._layout.game_box.h == 1280
        assert comp._layout.hook_band.y == 540
        assert comp._layout.caption_band.y == 1120
        # Exactly 2 layers (facecam + game), no redundant blur background needed because canvas is 100% covered
        assert len(comp.clips) == 2

    def test_full_bleed_game_layer_aspect_preserving_crop(self, tmp_path: Path):
        cfg = _make_config(tmp_path, overrides={"video": {"game_fit": "full_bleed"}})
        composer = VideoComposer(cfg)
        clip = _make_dummy_clip(duration=1.0, size=(1920, 1080))
        # 1080x1280 box (ratio 27:32 = 0.84375)
        layer = composer._game_layer(clip, Box(0, 640, 1080, 1280))
        assert layer.size == (1080, 1280)
        assert layer.pos(0) == (0, 640)

    def test_stacked_full_and_balanced_include_blur_background(self, tmp_path: Path):
        clip = _make_dummy_clip(duration=1.0, size=(1920, 1080))
        facecam = FacecamRegion(x=100, y=50, w=300, h=300)

        # stacked + full
        cfg_full = _make_config(tmp_path, overrides={"video": {"game_fit": "full"}})
        comp_full = VideoComposer(cfg_full).compose(clip, facecam=facecam)
        # 3 layers: blur background + facecam + game
        assert len(comp_full.clips) == 3

        # stacked + balanced
        cfg_bal = _make_config(tmp_path, overrides={"video": {"game_fit": "balanced"}})
        comp_bal = VideoComposer(cfg_bal).compose(clip, facecam=facecam)
        # 3 layers: blur background + facecam + game
        assert len(comp_bal.clips) == 3


# ============================================================================
# 4. Spike Punch-in Zoom Tests (apply_spike_zoom)
# ============================================================================


class TestApplySpikeZoom:
    """Verify center punch-in zoom behavior, boundaries, and audio preservation."""

    def test_zoom_applied_around_peak_offset(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        clip = _make_dummy_clip(duration=10.0, size=(1080, 1920))
        clip._source_clip = clip
        clip._source_clips = [clip]

        zoomed = composer.apply_spike_zoom(clip, peak_offset=4.0)
        assert zoomed.size == (1080, 1920)
        assert zoomed.duration == pytest.approx(10.0)
        assert zoomed.audio is not None
        assert zoomed._source_clip is clip
        assert zoomed._source_clips == [clip]

    def test_zoom_near_beginning(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        clip = _make_dummy_clip(duration=6.0, size=(1080, 1920))

        zoomed = composer.apply_spike_zoom(clip, peak_offset=0.2)
        assert zoomed.size == (1080, 1920)
        assert zoomed.duration == pytest.approx(6.0)

    def test_zoom_disabled_when_factor_is_one(self, tmp_path: Path):
        cfg = _make_config(
            tmp_path, overrides={"transformation": {"punch_in_zoom": 1.0}}
        )
        composer = VideoComposer(cfg)
        clip = _make_dummy_clip(duration=5.0, size=(1080, 1920))

        result = composer.apply_spike_zoom(clip, peak_offset=2.5)
        assert result is clip

    def test_zoom_skipped_when_peak_offset_none_or_negative(
        self, tmp_path: Path
    ):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        clip = _make_dummy_clip(duration=5.0, size=(1080, 1920))

        assert composer.apply_spike_zoom(clip, peak_offset=None) is clip
        assert composer.apply_spike_zoom(clip, peak_offset=-1.0) is clip

    def test_zoom_skipped_when_peak_beyond_duration(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        clip = _make_dummy_clip(duration=5.0, size=(1080, 1920))

        result = composer.apply_spike_zoom(clip, peak_offset=15.0)
        assert result is clip


# ============================================================================
# 5. Multi-Beat Story Stitching Tests (compose_story_beats)
# ============================================================================


class TestComposeStoryBeats:
    """Verify multi-beat stitching, MoviePy 2.2.1 duration clamping, and fallback."""

    def test_fallback_to_compose_on_single_segment(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        c1 = _make_dummy_clip(duration=4.0, size=(1920, 1080))

        res = composer.compose_story_beats([c1], facecam=None)
        assert res.size == (1080, 1920)
        assert res.duration == pytest.approx(4.0)
        assert res._source_clip is c1
        assert res._source_clips == [c1]

    def test_stitch_two_beats_with_clamping(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        c1 = _make_dummy_clip(duration=4.0, size=(1920, 1080))
        c2 = _make_dummy_clip(duration=5.0, size=(1920, 1080))

        # Beat 1: trim (0.5, 3.5) -> 3.0s
        # Beat 2: trim (1.0, 99.0) -> clamped to (1.0, 5.0) -> 4.0s
        res = composer.compose_story_beats(
            segment_paths=[c1, c2],
            facecam=None,
            beat_trim_ranges=[(0.5, 3.5), (1.0, 99.0)],
            peak_offset=4.5,
        )
        assert res.size == (1080, 1920)
        assert res.duration == pytest.approx(7.0)
        assert res.audio is not None
        assert res._source_clip is c1
        assert res._source_clips == [c1, c2]

    def test_stitch_three_beats_without_trim_ranges(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        c1 = _make_dummy_clip(duration=2.0, size=(1920, 1080))
        c2 = _make_dummy_clip(duration=3.0, size=(1920, 1080))
        c3 = _make_dummy_clip(duration=1.5, size=(1920, 1080))

        res = composer.compose_story_beats(
            segment_paths=[c1, c2, c3],
            facecam=None,
            beat_trim_ranges=None,
        )
        assert res.size == (1080, 1920)
        assert res.duration == pytest.approx(6.5)
        assert len(res._source_clips) == 3

    def test_invalid_trim_range_handled_safely(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        c1 = _make_dummy_clip(duration=3.0, size=(1920, 1080))
        c2 = _make_dummy_clip(duration=3.0, size=(1920, 1080))

        # t_end <= t_start -> should safely fall back to full clip
        res = composer.compose_story_beats(
            segment_paths=[c1, c2],
            facecam=None,
            beat_trim_ranges=[(2.0, 1.0), (0.0, 2.0)],
        )
        assert res.size == (1080, 1920)
        assert res.duration == pytest.approx(5.0)

    def test_empty_segments_raises_value_error(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)
        with pytest.raises(ValueError, match="No segment paths provided"):
            composer.compose_story_beats([], facecam=None)


# ============================================================================
# 6. Timeline Remapping Tests (remap_timeline_for_beats)
# ============================================================================


class TestRemapTimelineForBeats:
    """Verify stream-absolute to stitched timeline translation and dead-air filtering."""

    def test_words_and_chat_shifted_and_dead_air_dropped(self):
        beats = [
            {"start": 10.0, "end": 20.0, "label": "Setup"},
            {"start": 30.0, "end": 40.0, "label": "Payoff"},
        ]
        words = [
            WordTimestamp(text="beat1_word", start=12.0, end=13.0),
            WordTimestamp(text="dead_air_word", start=25.0, end=26.0),
            WordTimestamp(text="beat2_word", start=32.0, end=33.5),
        ]
        chat = [
            {
                "author": "u1",
                "message": "msg1",
                "offset_seconds": 15.0,
            },
            {
                "author": "u2",
                "message": "dead_air_msg",
                "offset_seconds": 25.0,
            },
            {
                "author": "u3",
                "message": "msg2",
                "offset_seconds": 35.0,
            },
        ]
        peak_time = 35.0

        remapped_words, remapped_chat, peak_offset = remap_timeline_for_beats(
            words, chat, peak_time, beats
        )

        # Beat 1 is 10s (0.0 .. 10.0), Beat 2 is 10s (10.0 .. 20.0)
        assert len(remapped_words) == 2
        assert remapped_words[0].text == "beat1_word"
        assert remapped_words[0].start == pytest.approx(2.0)
        assert remapped_words[0].end == pytest.approx(3.0)

        assert remapped_words[1].text == "beat2_word"
        assert remapped_words[1].start == pytest.approx(12.0)
        assert remapped_words[1].end == pytest.approx(13.5)

        assert len(remapped_chat) == 2
        assert remapped_chat[0]["message"] == "msg1"
        assert remapped_chat[0]["offset_seconds"] == pytest.approx(5.0)

        assert remapped_chat[1]["message"] == "msg2"
        assert remapped_chat[1]["offset_seconds"] == pytest.approx(15.0)

        assert peak_offset == pytest.approx(15.0)

    def test_peak_in_dead_air_returns_none(self):
        beats = [
            {"start": 10.0, "end": 20.0},
            {"start": 30.0, "end": 40.0},
        ]
        words: list[WordTimestamp] = []
        chat: list[dict[str, Any]] = []
        peak_time = 25.0  # inside dead air

        _, _, peak_offset = remap_timeline_for_beats(
            words, chat, peak_time, beats
        )
        assert peak_offset is None

    def test_peak_zero_or_negative_returns_none(self):
        beats = [{"start": 10.0, "end": 20.0}]
        _, _, peak_offset = remap_timeline_for_beats([], [], 0.0, beats)
        assert peak_offset is None
        _, _, peak_offset = remap_timeline_for_beats([], [], -5.0, beats)
        assert peak_offset is None

    def test_empty_beats_returns_original(self):
        words = [WordTimestamp(text="test", start=1.0, end=2.0)]
        chat = [{"author": "u", "message": "m", "offset_seconds": 1.5}]
        peak_time = 5.0

        r_words, r_chat, r_peak = remap_timeline_for_beats(
            words, chat, peak_time, []
        )
        assert len(r_words) == 1
        assert len(r_chat) == 1
        assert r_peak == pytest.approx(5.0)

    def test_chat_timestamp_field_variations(self):
        beats = [{"start": 100.0, "end": 120.0}]
        chat = [
            {"author": "u1", "message": "m1", "timestamp": 105.0},
            {"author": "u2", "message": "m2", "time": 110.0},
            {"author": "u3", "message": "m3", "offset": 115.0},
        ]
        _, r_chat, _ = remap_timeline_for_beats([], chat, 0.0, beats)
        assert len(r_chat) == 3
        assert r_chat[0]["timestamp"] == pytest.approx(5.0)
        assert r_chat[1]["time"] == pytest.approx(10.0)
        assert r_chat[2]["offset"] == pytest.approx(15.0)

    def test_remap_timestamps_alias(self):
        """remap_timestamps helper accurately wraps remap_timeline_for_beats."""
        words = [WordTimestamp(text="hello", start=10.0, end=11.0)]
        beats = [{"start": 10.0, "end": 15.0}]
        chat = [{"author": "u", "message": "hi", "offset_seconds": 12.0}]
        r_words, r_chat, r_peak = remap_timestamps(
            words=words, sub_clips=beats, chat_burst=chat, peak_time=13.0
        )
        assert len(r_words) == 1
        assert r_words[0].start == pytest.approx(0.0)
        assert r_chat[0]["offset_seconds"] == pytest.approx(2.0)
        assert r_peak == pytest.approx(3.0)

        # Also supports raw dict words and beats= keyword argument
        dict_words = [{"word": "dict_word", "start": 11.0, "end": 12.0, "score": 0.95}]
        r_words2, _, _ = remap_timestamps(words=dict_words, beats=beats)
        assert len(r_words2) == 1
        assert r_words2[0].text == "dict_word"
        assert r_words2[0].start == pytest.approx(1.0)
        assert r_words2[0].confidence == pytest.approx(0.95)

    def test_A5_spike_zoom_only_inside_game_box(self, tmp_path: Path):
        """Spike zoom changes pixels inside game_box only; facecam stays locked."""
        cfg = _make_config(
            tmp_path,
            overrides={
                "video": {"game_fit": "full_bleed"},
                "transformation": {"punch_in_zoom": 1.20},
            },
        )
        composer = VideoComposer(cfg)
        # Left half red, right half blue so a centre punch-in visibly changes game pixels.
        def _frame(t):
            f = np.zeros((1080, 1920, 3), dtype=np.uint8)
            f[:, :640] = (255, 0, 0)
            f[:, 640:] = (0, 0, 255)
            return f

        from moviepy import VideoClip

        clip = VideoClip(_frame, duration=6.0).with_fps(30)
        facecam = FacecamRegion(x=1400, y=50, w=400, h=400)

        comp = composer.compose(clip, facecam=facecam, peak_offset=2.5)
        assert comp.size == (1080, 1920)
        assert comp.duration == pytest.approx(6.0)
        assert comp._layout.kind == "stacked"
        gb = comp._layout.game_box
        assert (gb.y, gb.h) == (640, 1280)

        plain = composer.compose(clip, facecam=facecam)
        zoomed_frame = comp.get_frame(2.5)
        plain_frame = plain.get_frame(2.5)
        # Outside game_box (facecam area 0..640): identical
        assert np.array_equal(zoomed_frame[:640], plain_frame[:640])
        # Inside game_box the punch-in occurred (the red/blue edge moved)
        assert not np.array_equal(
            zoomed_frame[gb.y : gb.y2, :], plain_frame[gb.y : gb.y2, :]
        )


# ============================================================================
# 7. Real MP4 File Handling and Windows Cleanup Tests
# ============================================================================


class TestRealFileHandlingAndWindowsCleanup:
    """Verify VideoComposer with actual on-disk MP4 files and safe handle release."""

    def test_compose_with_real_mp4_files(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)

        p1 = tmp_path / "clip1.mp4"
        p2 = tmp_path / "clip2.mp4"

        c1 = ColorClip(size=(640, 360), color=(200, 0, 0), duration=2.0)
        c1.write_videofile(str(p1), fps=24, logger=None)
        c2 = ColorClip(size=(640, 360), color=(0, 200, 0), duration=2.0)
        c2.write_videofile(str(p2), fps=24, logger=None)

        # Single compose
        facecam = FacecamRegion(x=400, y=20, w=150, h=150)
        res_single = composer.compose(p1, facecam=facecam, peak_offset=1.0)
        assert res_single.size == (1080, 1920)
        assert hasattr(res_single, "_source_clip")
        assert hasattr(res_single, "_source_clips")
        assert len(res_single._source_clips) == 1
        res_single._source_clip.close()

        # Story beats compose
        res_beats = composer.compose_story_beats(
            [p1, p2],
            facecam=None,
            beat_trim_ranges=[(0.0, 1.5), (0.5, 2.0)],
            peak_offset=2.0,
        )
        assert res_beats.size == (1080, 1920)
        assert res_beats.duration == pytest.approx(3.0)
        assert len(res_beats._source_clips) == 2
        for sc in res_beats._source_clips:
            sc.close()

        # Deleting the files succeeds on Windows because handles were closed
        p1.unlink()
        p2.unlink()
        assert not p1.exists()
        assert not p2.exists()

    def test_compose_story_beats_closes_opened_handles_on_partial_failure(
        self, tmp_path: Path
    ):
        """If beat 2 fails to open, beat 1's VideoFileClip handle is closed so Windows can delete it."""
        cfg = _make_config(tmp_path)
        composer = VideoComposer(cfg)

        p1 = tmp_path / "beat1_ok.mp4"
        c1 = ColorClip(size=(320, 180), color=(0, 128, 255), duration=1.0)
        c1.write_videofile(str(p1), fps=15, logger=None)

        missing_p2 = tmp_path / "beat2_missing.mp4"
        with pytest.raises(Exception):
            composer.compose_story_beats([p1, missing_p2], facecam=None)

        # Unlinking p1 proves beat 1's MoviePy handle was closed on exception
        p1.unlink()
        assert not p1.exists()

    def test_crop_to_aspect_out_of_bounds_box_stays_within_frame(self):
        """Out-of-bounds x1 >= clip.w or y1 >= clip.h must never produce x2 > clip.w / y2 > clip.h."""
        clip = ColorClip(size=(640, 360), color=(10, 20, 30), duration=1.0)
        cropped = VideoComposer._crop_to_aspect(
            clip, x1=900, y1=500, x2=1000, y2=600, target_ratio=1.5
        )
        assert cropped.w >= 1 and cropped.h >= 1
        assert cropped.w <= clip.w and cropped.h <= clip.h

    def test_apply_spike_zoom_uses_spec_window_and_skips_zero_duration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """apply_spike_zoom must use z_end = min(duration, z_start + zoom_duration) and skip zoom_duration=0."""
        import streamclipper.assembly.composer as composer_mod

        captured_slices: list[Any] = []
        orig_concat = composer_mod.concatenate_videoclips

        def _spy_concat(slices: list[Any], **kwargs: Any) -> Any:
            captured_slices.extend(slices)
            return orig_concat(slices, **kwargs)

        monkeypatch.setattr(composer_mod, "concatenate_videoclips", _spy_concat)

        cfg = _make_config(
            tmp_path,
            overrides={"transformation": {"punch_in_zoom": 1.2, "zoom_duration_seconds": 3.5}},
        )
        composer = VideoComposer(cfg)
        clip = _make_dummy_clip(duration=10.0, size=(360, 640))
        zoomed = composer.apply_spike_zoom(clip, peak_offset=4.0)
        assert zoomed.duration == pytest.approx(10.0)
        # Expect slices: [0.0..3.5], [3.5..7.0], [7.0..10.0] -> exact 3.5s zoom duration
        assert [round(s.duration, 2) for s in captured_slices] == [3.5, 3.5, 3.0]

        cfg_zero = _make_config(
            tmp_path,
            overrides={"transformation": {"punch_in_zoom": 1.2, "zoom_duration_seconds": 0.0}},
        )
        composer_zero = VideoComposer(cfg_zero)
        assert composer_zero.apply_spike_zoom(clip, peak_offset=4.0) is clip

    def test_remap_timeline_handles_malformed_chat_and_trailing_zero_duration_beat(self):
        """remap_timeline_for_beats skips non-dict/bad chat timestamps and keeps end boundary on last valid beat."""
        words = [WordTimestamp(text="boom", start=10.0, end=11.0)]
        chat = [
            {"offset_seconds": 20.0, "message": "boundary"},
            {"offset_seconds": "not-a-number", "message": "bad"},
            {"offset_seconds": True, "message": "bool"},
            "not_a_dict",
        ]
        beats = [{"start": 10.0, "end": 20.0}, {"start": 30.0, "end": 30.0}]
        rw, rc, rp = remap_timeline_for_beats(words, chat, peak_time=20.0, beats=beats)  # type: ignore[arg-type]
        assert len(rw) == 1
        assert len(rc) == 1
        assert rc[0]["message"] == "boundary"
        assert rc[0]["offset_seconds"] == pytest.approx(10.0)
        assert rp == pytest.approx(10.0)
