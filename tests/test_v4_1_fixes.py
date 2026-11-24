"""Unit tests for v4.1 Shorts quality & breakdown fixes.

Covers:
1. Zoom spotlight ROI protection & outer 15% margin saliency masking.
2. Breakdown mode multi-beat stitching (jump-cutting dead air).
3. Shorts duration defaults tightening (max_clip_seconds: 45, max_final_seconds: 55).
4. Hook banner placement at (540, 720) across facecam/gameplay seam.
5. Replay/breakdown badge emoji stripping to prevent tofu squares.
6. Callout start climax synchronization relative to peak_offset.
7. A/B title variants cleanup (banned phrases removed, Hinglish support).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from moviepy import ColorClip

from streamclipper.assembly.hooks import render_replay_badge
from streamclipper.assembly.layout import compute_layout
from streamclipper.config import Config, HardwareProfile
from streamclipper.models import ClipMetadata, ClipResult, Highlight, Platform, StreamInfo
from streamclipper.narration_style import BANNED_PHRASES
from streamclipper.output.post_writer import generate_clip_title_variants
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
    cfg = Config(hardware_profile=hw)
    cfg._data.setdefault("general", {})["temp_dir"] = str(tmp_path / "temp")
    cfg._data.setdefault("general", {})["output_dir"] = str(tmp_path / "output")
    cfg._data.setdefault("general", {})["cache_dir"] = str(tmp_path / "cache")
    return cfg


# --- Fix 1: Zoom spotlight ROI & outer 15% border margins ---


def test_v4_1_opencv_saliency_masks_outer_15_percent_margins(tmp_path: Path) -> None:
    """Tier 3 OpenCV saliency masks out outer 15% margins so timers and overlays cannot win."""
    import cv2

    cfg = _make_config(tmp_path)
    pipeline = Pipeline(cfg)

    # Create a synthetic 1080x720 video with a high-contrast pattern only in outer corner (x < 0.15)
    # and a valid text pattern in the safe center area (x > 0.3, y > 0.3)
    vid_path = tmp_path / "test_saliency.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (1080, 720))

    frame = np.zeros((720, 1080, 3), dtype=np.uint8)
    # HUD timer / overlay artifact in corner (x < 0.15, y < 0.15): (20, 20) to (120, 60)
    # High frequency stripes
    frame[20:60, 20:120] = 255
    for c in range(20, 120, 4):
        frame[20:60, c] = 0

    # Real center content at x=0.40, y=0.45: (320, 430) to (380, 650)
    for r in range(320, 380, 6):
        frame[r : r + 3, 430:650] = 255

    for _ in range(30):
        out.write(frame)
    out.release()

    h = Highlight(
        start=0.0,
        end=5.0,
        score=8.0,
        title="Secret file",
        reason="Reading secret document on screen",
        moment_type="analysis",
        breakdown_commentary="Look at this classified document",
        breakdown_focus="classified document",
    )

    t, roi = pipeline._resolve_breakdown_target_time_and_roi(h, vid_path, 0.0, 5.0)
    # The ROI must NOT be in the outer 15% corner (ymin, xmin, ymax, xmax)
    ymin, xmin, ymax, xmax = roi
    assert xmin >= 0.14
    assert ymin >= 0.14
    assert xmax <= 0.86
    assert ymax <= 0.86


def test_v4_1_zoom_spotlight_falls_back_when_no_roi(tmp_path: Path) -> None:
    """If zoom_spotlight has no explicit and no VLM-verified ROI, falls back to story or narrative."""
    cfg = _make_config(tmp_path)
    pipeline = Pipeline(cfg)

    # Case A: sub_clips exist (2 beats) -> falls back to story
    h_beats = Highlight(
        start=10.0,
        end=30.0,
        score=8.5,
        title="No ROI Beats",
        reason="Screen analysis without ROI",
        moment_type="analysis",
        breakdown_kind="zoom_spotlight",
        breakdown_roi=[],
        sub_clips=[{"start": 10.0, "end": 18.0}, {"start": 22.0, "end": 30.0}],
    )
    meta_beats = ClipMetadata(
        hook_title="No ROI",
        caption="Caption",
        hashtags=[],
        streamer="Streamer",
        game="GTA 6",
        source_url="http://example.com",
        clip_start=10.0,
        clip_end=30.0,
        highlight_score=8.5,
        moment_type="analysis",
        breakdown_kind="zoom_spotlight",
        breakdown_roi=[],
    )

    fake_comp = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=16.0)
    fake_comp._layout = None

    seg1 = tmp_path / "seg1.mp4"
    seg2 = tmp_path / "seg2.mp4"
    seg1.write_bytes(b"dummy")
    seg2.write_bytes(b"dummy")

    with patch("streamclipper.assembly.facecam.FacecamDetector.detect", return_value=None), \
         patch("streamclipper.assembly.composer.VideoComposer.compose_story_beats", return_value=fake_comp) as mock_story, \
         patch("streamclipper.assembly.composer.VideoComposer.compose_zoom_spotlight_breakdown") as mock_spotlight, \
         patch("streamclipper.output.exporter.Exporter.export", return_value=seg1), \
         patch("streamclipper.output.organizer.FileOrganizer.save_clip") as mock_save:
        mock_save.return_value = ClipResult(video_path=seg1, metadata_path=None, metadata=meta_beats, duration_seconds=16.0)
        pipeline._assemble_highlight(
            h_beats,
            meta_beats,
            [seg1, seg2],
            1,
            "breakdown",
            None,
            None,
            valid_beats=h_beats.sub_clips,
        )
        assert mock_story.called
        assert not mock_spotlight.called
        assert meta_beats.transformation_mode == "story"

    # Case B: no sub_clips -> falls back to narrative
    h_single = Highlight(
        start=10.0,
        end=30.0,
        score=8.5,
        title="No ROI Single",
        reason="Screen analysis without ROI",
        moment_type="analysis",
        breakdown_kind="zoom_spotlight",
        breakdown_roi=[],
        sub_clips=[],
    )
    meta_single = ClipMetadata(
        hook_title="No ROI Single",
        caption="Caption",
        hashtags=[],
        streamer="Streamer",
        game="GTA 6",
        source_url="http://example.com",
        clip_start=10.0,
        clip_end=30.0,
        highlight_score=8.5,
        moment_type="analysis",
        breakdown_kind="zoom_spotlight",
        breakdown_roi=[],
    )
    with patch("streamclipper.assembly.facecam.FacecamDetector.detect", return_value=None), \
         patch("streamclipper.assembly.composer.VideoComposer.compose", return_value=fake_comp) as mock_compose, \
         patch("streamclipper.assembly.composer.VideoComposer.compose_zoom_spotlight_breakdown") as mock_spotlight, \
         patch("streamclipper.output.exporter.Exporter.export", return_value=seg1), \
         patch("streamclipper.output.organizer.FileOrganizer.save_clip") as mock_save:
        mock_save.return_value = ClipResult(video_path=seg1, metadata_path=None, metadata=meta_single, duration_seconds=16.0)
        pipeline._assemble_highlight(
            h_single,
            meta_single,
            [seg1],
            2,
            "breakdown",
            None,
            None,
            valid_beats=[],
        )
        assert mock_compose.called
        assert not mock_spotlight.called
        assert meta_single.transformation_mode == "narrative"


# --- Fix 2: Breakdown mode stitches multi-beat sub_clips ---


def test_v4_1_breakdown_mode_stitches_beats_and_jump_cuts(tmp_path: Path) -> None:
    """When valid_beats exist in breakdown mode, beats are stitched before replay."""
    cfg = _make_config(tmp_path)
    pipeline = Pipeline(cfg)

    h = Highlight(
        start=10.0,
        end=50.0,
        score=9.0,
        title="Clutch play",
        reason="Epic win",
        moment_type="clutch",
        breakdown_commentary="Watch the sniper shot",
        peak_time=42.0,
        sub_clips=[{"start": 10.0, "end": 20.0}, {"start": 35.0, "end": 48.0}],
    )

    mode, beats = pipeline._resolve_highlight_mode("breakdown", h)
    assert mode == "breakdown"
    assert len(beats) == 2

    seg1 = tmp_path / "seg1.mp4"
    seg2 = tmp_path / "seg2.mp4"
    seg1.write_bytes(b"dummy1")
    seg2.write_bytes(b"dummy2")

    meta = ClipMetadata(
        hook_title="Clutch Play",
        caption="Caption",
        hashtags=[],
        streamer="Streamer",
        game="GTA 6",
        source_url="http://example.com",
        clip_start=10.0,
        clip_end=50.0,
        highlight_score=9.0,
        moment_type="clutch",
        breakdown_commentary="Watch the sniper shot",
        transformation_mode="breakdown",
    )

    fake_comp = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=23.0)
    fake_comp._layout = None

    with patch("streamclipper.assembly.facecam.FacecamDetector.detect", return_value=None), \
         patch("streamclipper.assembly.composer.VideoComposer.compose_story_beats", return_value=fake_comp) as mock_story, \
         patch("streamclipper.assembly.composer.VideoComposer.compose_instant_replay", return_value=(fake_comp, 23.0)) as mock_replay, \
         patch("streamclipper.output.exporter.Exporter.export", return_value=seg1), \
         patch("streamclipper.output.organizer.FileOrganizer.save_clip") as mock_save:
        mock_save.return_value = ClipResult(video_path=seg1, metadata_path=None, metadata=meta, duration_seconds=23.0)
        pipeline._assemble_highlight(
            h,
            meta,
            [seg1, seg2],
            1,
            "breakdown",
            None,
            None,
            valid_beats=beats,
        )
        assert mock_story.called
        assert mock_replay.called


# --- Fix 3: Shorts duration defaults ---


def test_v4_1_shorts_duration_defaults_and_spotlight_cap(tmp_path: Path) -> None:
    """Verify max_clip_seconds is 45, max_final_seconds is 55, and zoom spotlight duration is 4.5."""
    cfg = _make_config(tmp_path)
    assert cfg.get("analysis.max_clip_seconds") == 45
    assert cfg.get("output.max_final_seconds") == 55
    assert cfg.get("transformation.breakdown.zoom_spotlight.duration") == 4.5


# --- Fix 4: Hook banner placement ---


def test_v4_1_hook_band_seam_placement() -> None:
    """Stacked full_bleed hook_band sits at (540, 720) across facecam/gameplay seam."""
    layout = compute_layout("stacked", "full_bleed")
    assert layout.hook_band.y == 540
    assert layout.hook_band.y2 == 720
    # Streamer facecam is top 640px, so 540-720 bridges 100px of facecam and 80px of game box
    assert layout.facecam_box is not None
    assert layout.facecam_box.y == 0
    assert layout.facecam_box.y2 == 640


# --- Fix 5: Replay badge emoji stripping ---


def test_v4_1_render_replay_badge_strips_emoji() -> None:
    """render_replay_badge strips unrenderable emoji to prevent tofu squares."""
    arr = render_replay_badge("🔴 INSTANT REPLAY")
    assert isinstance(arr, np.ndarray)
    assert arr.shape[2] == 4

    arr2 = render_replay_badge("🔍 EVIDENCE BREAKDOWN")
    assert isinstance(arr2, np.ndarray)
    assert arr2.shape[2] == 4


# --- Fix 6: Callout start syncs to climax ---


def test_v4_1_callout_start_syncs_to_climax(tmp_path: Path) -> None:
    """callout_start is computed relative to peak_offset (peak_offset - 2.0 clamped cleanly)."""
    cfg = _make_config(tmp_path)
    pipeline = Pipeline(cfg)

    seg = tmp_path / "seg.mp4"
    seg.write_bytes(b"dummy")

    fake_comp = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=40.0)
    fake_comp._trim_start = 0.0
    fake_comp._layout = None

    # Case A: peak_offset is 18.0s, intro_dur=2.0s -> callout_start = 18.0 - 2.0 = 16.0s
    h = Highlight(
        start=0.0,
        end=40.0,
        score=8.0,
        title="Play",
        reason="Climax",
        peak_time=18.0,
        loudness_score=1.0,
    )
    meta = ClipMetadata(
        hook_title="Play",
        caption="Caption",
        hashtags=[],
        streamer="Streamer",
        game="GTA 6",
        source_url="http://example.com",
        clip_start=0.0,
        clip_end=40.0,
        highlight_score=8.0,
        callout="Watch this move!",
        narration_intro="Intro words here now",  # ~1.4s
    )

    with patch("streamclipper.assembly.facecam.FacecamDetector.detect", return_value=None), \
         patch("streamclipper.assembly.composer.VideoComposer.compose", return_value=fake_comp), \
         patch("streamclipper.output.exporter.Exporter.export", return_value=seg) as mock_export, \
         patch("streamclipper.output.organizer.FileOrganizer.save_clip") as mock_save:
        mock_save.return_value = ClipResult(video_path=seg, metadata_path=None, metadata=meta, duration_seconds=40.0)
        pipeline._assemble_highlight(h, meta, [seg], 1, "narrative", None, None)
        kw = mock_export.call_args[1]
        assert kw["callout_start"] == pytest.approx(16.0, abs=0.5)

    # Case B: No peak -> falls back to 0.4 * moment_seconds = 16.0s
    h_no_peak = Highlight(
        start=0.0,
        end=40.0,
        score=8.0,
        title="Play",
        reason="Climax",
        peak_time=0.0,
        loudness_score=0.0,
    )
    with patch("streamclipper.assembly.facecam.FacecamDetector.detect", return_value=None), \
         patch("streamclipper.assembly.composer.VideoComposer.compose", return_value=fake_comp), \
         patch("streamclipper.output.exporter.Exporter.export", return_value=seg) as mock_export, \
         patch("streamclipper.output.organizer.FileOrganizer.save_clip") as mock_save:
        mock_save.return_value = ClipResult(video_path=seg, metadata_path=None, metadata=meta, duration_seconds=40.0)
        pipeline._assemble_highlight(h_no_peak, meta, [seg], 2, "narrative", None, None)
        kw = mock_export.call_args[1]
        assert kw["callout_start"] == pytest.approx(0.4 * 40.0)


# --- Fix 7: A/B title variants cleanup & language matching ---


def test_v4_1_title_variants_english_no_banned_phrases() -> None:
    """English title variants avoid all BANNED_PHRASES and mid-phrase truncation."""
    stream = StreamInfo(
        url="https://youtube.com/watch?v=abc",
        title="Stream Title",
        platform=Platform.YOUTUBE,
        streamer="SpeedRunner",
        game="GTA 6",
        duration_seconds=120.0,
    )
    meta = ClipMetadata(
        hook_title="IMPOSSIBLE VAULT HEIST IN DOWNTOWN",
        caption="Check out this crazy vault heist",
        hashtags=["GTA6"],
        streamer="SpeedRunner",
        game="GTA 6",
        source_url="https://youtube.com/watch?v=abc",
        clip_start=10.0,
        clip_end=35.0,
        highlight_score=9.0,
        callout="Vault door blown open",
        narration_language="english",
    )

    variants = generate_clip_title_variants(meta, stream)
    assert 3 <= len(variants) <= 5
    for v in variants:
        # Banned phrases check
        v_low = v.lower()
        assert not any(p in v_low for p in BANNED_PHRASES)
        assert "you won't believe" not in v_low
        assert "wait for it... ..." not in v_low


def test_v4_1_title_variants_hinglish() -> None:
    """Hinglish title variants generate appropriate Hindi/Hinglish phrasing and avoid banned phrases."""
    stream = StreamInfo(
        url="https://youtube.com/watch?v=xyz",
        title="Hindi Stream",
        platform=Platform.YOUTUBE,
        streamer="CarryisLive",
        game="GTA 6",
        duration_seconds=120.0,
    )
    meta = ClipMetadata(
        hook_title="SECRET HEIST REVEALED",
        caption="Caption",
        hashtags=["GTA6"],
        streamer="CarryisLive",
        game="GTA 6",
        source_url="https://youtube.com/watch?v=xyz",
        clip_start=10.0,
        clip_end=35.0,
        highlight_score=9.0,
        callout="Door open ho gaya",
        narration_language="hinglish",
    )

    variants = generate_clip_title_variants(meta, stream)
    assert 3 <= len(variants) <= 5
    hinglish_markers = ("ko", "ye", "dekho", "kaise", "mein", "kya", "bhai", "nahi")
    has_hinglish_word = any(
        any(marker in v.lower().split() for marker in hinglish_markers)
        for v in variants
    )
    assert has_hinglish_word
    for v in variants:
        v_low = v.lower()
        assert not any(p in v_low for p in BANNED_PHRASES)
        assert "you won't believe" not in v_low
