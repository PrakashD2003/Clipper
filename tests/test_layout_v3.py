"""Output Quality v3 Batch A: layout geometry, composer layouts, captions bands, hooks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from moviepy import ColorClip, VideoClip

from streamclipper.assembly.captions import AssCaptionBurner
from streamclipper.assembly.composer import (
    VideoComposer,
    compose_freeze_breakdown,
    compose_instant_replay,
)
from streamclipper.assembly.hooks import HookTitleOverlay, create_replay_badge
from streamclipper.assembly.layout import compute_layout
from streamclipper.models import FacecamRegion, WordTimestamp
from tests.test_composer import _make_config, _make_dummy_clip

FACECAM = FacecamRegion(x=1400, y=50, w=400, h=400)


def _boxes(g):
    return (
        (g.hook_band.y, g.hook_band.y + g.hook_band.h),
        None if g.facecam_box is None else (g.facecam_box.y, g.facecam_box.y + g.facecam_box.h),
        (g.game_box.y, g.game_box.y + g.game_box.h),
        (g.caption_band.y, g.caption_band.y + g.caption_band.h),
    )


def test_A1_compute_layout_table():
    assert _boxes(compute_layout("stacked", "full")) == ((0, 180), (180, 660), (660, 1268), (1268, 1920))
    assert _boxes(compute_layout("stacked", "balanced")) == ((0, 180), (180, 680), (680, 1490), (1490, 1920))
    assert _boxes(compute_layout("stacked", "full_bleed")) == ((540, 720), (0, 640), (640, 1920), (1120, 1460))
    # Default without explicit game_fit is full_bleed
    assert _boxes(compute_layout("stacked")) == ((540, 720), (0, 640), (640, 1920), (1120, 1460))
    assert _boxes(compute_layout("reaction", "balanced")) == ((0, 180), (0, 960), (960, 1568), (1568, 1920))
    assert _boxes(compute_layout("reaction", "full_bleed")) == ((0, 180), (0, 960), (960, 1568), (1568, 1920))
    assert _boxes(compute_layout("blur_fill", "full")) == ((180, 520), None, (560, 1259), (1275, 1800))
    assert _boxes(compute_layout("blur_fill", "full_bleed")) == ((180, 520), None, (560, 1259), (1275, 1800))
    assert _boxes(compute_layout("blur_fill", "balanced")) == ((300, 540), None, (555, 1365), (1365, 1920))
    g = compute_layout("stacked", "balanced")
    assert (g.game_box.w, g.game_box.h) == (1080, 810)
    g_fb = compute_layout("stacked", "full_bleed")
    assert (g_fb.game_box.w, g_fb.game_box.h) == (1080, 1280)
    assert (g_fb.facecam_box.w, g_fb.facecam_box.h) == (1080, 640)
    with pytest.raises(ValueError):
        compute_layout("nope")


def test_A2_stacked_layout_bands(tmp_path: Path):
    # Default is full_bleed: zero black void, full canvas covered
    composer = VideoComposer(_make_config(tmp_path))
    clip = _make_dummy_clip(duration=2.0, size=(1920, 1080), color=(200, 40, 40))
    comp = composer.compose(clip, FACECAM)
    assert comp.size == (1080, 1920)
    assert comp._layout.kind == "stacked"
    frame = comp.get_frame(0.5)
    assert frame[1000, 540].tolist() == [200, 40, 40]  # game area
    assert frame[500, 540].tolist() == [200, 40, 40]  # facecam area
    assert frame[1700, 540].tolist() == [200, 40, 40]  # game fills bottom 66.7% (zero black void)

    # For stacked/full: blur background rendered behind canvas, eliminating black voids
    comp_full = VideoComposer(_make_config(tmp_path, overrides={"video": {"game_fit": "full"}}))
    res_full = comp_full.compose(clip, FACECAM)
    frame_full = res_full.get_frame(0.5)
    assert frame_full[1000, 540].tolist() == [200, 40, 40]  # game area
    assert frame_full[500, 540].tolist() == [200, 40, 40]  # facecam area
    assert frame_full[1700, 540].tolist() == [100, 20, 20]  # blurred/darkened ~0.5 background under caption band (no black void)

    # For stacked/balanced: blur background also rendered behind canvas
    comp_bal = VideoComposer(_make_config(tmp_path, overrides={"video": {"game_fit": "balanced"}}))
    res_bal = comp_bal.compose(clip, FACECAM)
    frame_bal = res_bal.get_frame(0.5)
    assert frame_bal[100, 540].tolist() == [100, 20, 20]  # blurred/darkened top hook band (no black void)
    assert frame_bal[1700, 540].tolist() == [100, 20, 20]  # blurred/darkened bottom caption band (no black void)


def test_A2_balanced_game_fit_crops_4x3(tmp_path: Path):
    composer = VideoComposer(_make_config(tmp_path, overrides={"video": {"game_fit": "balanced"}}))
    clip = _make_dummy_clip(duration=2.0, size=(1920, 1080))
    comp = composer.compose(clip, FACECAM)
    assert comp._layout.game_box.h == 810


def test_A3_blur_fill_background(tmp_path: Path):
    composer = VideoComposer(_make_config(tmp_path))

    def _frame(t):
        return np.full((1080, 1920, 3), 200, dtype=np.uint8)

    clip = VideoClip(_frame, duration=2.0).with_fps(30)
    comp = composer.compose(clip, None)
    assert comp._layout.kind == "blur_fill"
    frame = comp.get_frame(0.5)
    assert frame[900, 540].tolist() == [200, 200, 200]  # sharp game in game_box
    bg = frame[100, 540]
    assert 90 <= int(bg[0]) <= 110  # darkened ~0.5
    assert frame[1700, 540].mean() > 50  # blurred bg extends under caption band


def test_A4_reaction_layout(tmp_path: Path):
    composer = VideoComposer(_make_config(tmp_path))
    clip = _make_dummy_clip(duration=2.0, size=(1920, 1080), color=(10, 200, 10))
    comp = composer.compose(clip, FACECAM, layout_kind="reaction")
    assert comp._layout.kind == "reaction"
    assert comp._layout.facecam_box.h == 960
    frame = comp.get_frame(0.5)
    assert frame[500, 540].tolist() == [10, 200, 10]


def test_A5_story_beats_zoom_inside_game_box(tmp_path: Path):
    composer = VideoComposer(_make_config(tmp_path))
    a = _make_dummy_clip(duration=3.0, size=(1920, 1080))
    b = _make_dummy_clip(duration=3.0, size=(1920, 1080))
    result = composer.compose_story_beats([a, b], FACECAM, peak_offset=1.0, layout_kind="stacked")
    assert result._layout.kind == "stacked"
    assert result.size == (1080, 1920)


def test_A6_layout_propagates_through_replay_and_freeze(tmp_path: Path):
    composer = VideoComposer(_make_config(tmp_path))
    clip = _make_dummy_clip(duration=6.0, size=(1920, 1080))
    comp = composer.compose(clip, FACECAM, peak_offset=2.0)
    assert comp._layout is not None
    combined, _ = compose_instant_replay(comp, 2.0)
    assert combined._layout == comp._layout
    assert compose_freeze_breakdown(comp, 2.0)._layout == comp._layout
    badge = create_replay_badge(duration=1.0, layout=comp._layout)
    assert badge.pos(0)[1] == comp._layout.game_box.y + 16


def _ass(tmp_path, layout, **kw):
    burner = AssCaptionBurner(_make_config(tmp_path))
    words = [WordTimestamp(text="hello", start=0.5, end=0.9, confidence=1.0)]
    out = burner.generate_ass(
        words,
        0.0,
        tmp_path / "x.ass",
        chat_burst=[{"message": "lol", "offset_seconds": 1.0}],
        layout=layout,
        **kw,
    )
    return out.read_text(encoding="utf-8")


def test_A7_captions_in_caption_band_and_callout(tmp_path: Path):
    layout = compute_layout("stacked", "full")
    text = _ass(tmp_path, layout, callout="No way", callout_window=(1.0, 2.0))
    hormozi = next(l for l in text.splitlines() if l.startswith("Style: Hormozi"))
    margin_v = int(hormozi.split(",")[-2])
    bottom_y = 1920 - margin_v
    assert layout.caption_band.y < bottom_y <= layout.caption_band.y + layout.caption_band.h
    ticker = next(l for l in text.splitlines() if l.startswith("Style: ChatTicker"))
    assert int(ticker.split(",")[-2]) > layout.caption_band.y + layout.caption_band.h * 0.5
    assert "Style: Callout" in text
    assert "NO WAY" in text
    plain = _ass(tmp_path, None)
    assert "Style: Callout" not in plain


def test_A8_hook_centered_in_hook_band(tmp_path: Path):
    layout = compute_layout("stacked", "full")
    overlay = HookTitleOverlay(_make_config(tmp_path))
    base = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=5.0)
    result = overlay.add_hook(base, "BIG HOOK", context_badge="Clutch", layout=layout)
    ys = [c.pos(0)[1] for c in result.clips[1:]]
    assert all(0 <= y < 180 for y in ys)
    assert ys[0] < ys[1]  # hook above badge
    solo = overlay.add_hook(base, "BIG HOOK", layout=layout)
    y = solo.clips[1].pos(0)[1]
    h = solo.clips[1].size[1]
    assert abs((y + h / 2) - 90) <= 2
