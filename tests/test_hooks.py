"""Unit tests for HookTitleOverlay and adaptive context badge overlay (Package 4.4)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure current worktree src is prioritized over external editable install paths
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# If streamclipper was already imported by earlier test files, extend __path__
# so that the local worktree modules resolve cleanly.
if "streamclipper" in sys.modules:
    pkg_dir = str(_SRC_DIR / "streamclipper")
    if pkg_dir not in sys.modules["streamclipper"].__path__:
        sys.modules["streamclipper"].__path__.insert(0, pkg_dir)
if "streamclipper.assembly" in sys.modules:
    assembly_dir = str(_SRC_DIR / "streamclipper" / "assembly")
    if assembly_dir not in sys.modules["streamclipper.assembly"].__path__:
        sys.modules["streamclipper.assembly"].__path__.insert(0, assembly_dir)

import numpy as np
import pytest
from moviepy import AudioClip, ColorClip, CompositeVideoClip

from streamclipper.assembly.hooks import (
    HookTitleOverlay,
    _load_font,
    _wrap_text,
    render_context_badge,
    render_hook_banner,
)
from streamclipper.config import Config


class DummyConfig:
    """Mock Config helper for tests."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._settings = settings or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)


def _make_dummy_audio(duration: float = 5.0) -> AudioClip:
    """Create a lightweight synthetic audio clip."""
    return AudioClip(
        lambda t: np.sin(2 * np.pi * 440 * np.asarray(t)),
        duration=duration,
        fps=22050,
    )


class TestHookTitleOverlayConfig:
    """Tests for configuration parsing in HookTitleOverlay."""

    def test_init_defaults(self) -> None:
        cfg = DummyConfig()
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]
        assert overlay.enabled is True
        assert overlay.duration == 4.0
        assert overlay.font_size == 50
        assert overlay.fade_in is True
        assert overlay.font_name == "Arial-Bold"
        assert overlay.color == "white"
        assert overlay.stroke_color == "black"
        assert overlay.stroke_width == 3
        assert overlay.position == "top"

    def test_init_custom_config(self) -> None:
        cfg = DummyConfig({
            "hook_title.enabled": False,
            "hook_title.duration_seconds": 6,
            "hook_title.font_size": 42,
            "hook_title.fade_in": False,
            "hook_title.font": "segoeui.ttf",
            "hook_title.color": "yellow",
            "hook_title.stroke_color": "navy",
            "hook_title.stroke_width": 2,
            "hook_title.position": "center",
        })
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]
        assert overlay.enabled is False
        assert overlay.duration == 6.0
        assert overlay.font_size == 42
        assert overlay.fade_in is False
        assert overlay.font_name == "segoeui.ttf"
        assert overlay.color == "yellow"
        assert overlay.stroke_color == "navy"
        assert overlay.stroke_width == 2
        assert overlay.position == "center"


class TestPillowRendering:
    """Tests for Pillow-based rendering of banners and badges without ImageMagick."""

    def test_render_hook_banner_shape_and_channels(self) -> None:
        arr = render_hook_banner("INSANE ESCAPE", max_width=800, font_size=40)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert arr.shape[2] == 4  # RGBA
        assert arr.dtype == np.uint8
        # Height and width must be positive
        assert arr.shape[0] > 20
        assert arr.shape[1] > 100
        # Background alpha is semi-transparent (~185), text pixels reach 255
        assert arr[:, :, 3].max() == 255
        assert (arr[:, :, 3] > 0).any()

    def test_render_hook_banner_text_wrapping(self) -> None:
        short_banner = render_hook_banner("SHORT", max_width=800, font_size=40)
        long_banner = render_hook_banner(
            "THIS IS A VERY LONG HOOK TITLE THAT SHOULD WRAP ACROSS MULTIPLE LINES CLEANLY",
            max_width=400,
            font_size=40,
        )
        assert long_banner.shape[0] > short_banner.shape[0]
        assert long_banner.shape[1] <= 450  # Fits comfortably within requested width bound

    def test_render_context_badge_shape_and_channels(self) -> None:
        badge = render_context_badge("🔥 Clutch Escape", font_size=26)
        assert isinstance(badge, np.ndarray)
        assert badge.ndim == 3
        assert badge.shape[2] == 4  # RGBA
        assert badge.dtype == np.uint8
        assert badge.shape[0] > 15
        assert badge.shape[1] > 50
        assert badge.shape[0] < 80  # Compact pill height
        assert badge[:, :, 3].max() == 255

    @pytest.mark.parametrize(
        "badge_text",
        [
            "💀 Instant Karma",
            "🎨 Insane Detail",
            "🔥 Clutch Escape",
            "🧠 Hidden Detail",
            "⚡ High Energy",
            "😂 NPC Chaos",
        ],
    )
    def test_render_context_badge_emojis(self, badge_text: str) -> None:
        arr = render_context_badge(badge_text, font_size=24)
        assert arr.ndim == 3
        assert arr.shape[2] == 4
        assert arr.shape[0] > 0 and arr.shape[1] > 0

    def test_wrap_text_single_word_and_empty(self) -> None:
        font = _load_font(None, 40)
        assert _wrap_text("", font, 500) == []
        assert _wrap_text("   \n   ", font, 500) == []
        single = _wrap_text("SUPERLONGWORDEXCEEDINGBOUNDS", font, 100)
        assert len(single) == 1
        assert single[0] == "SUPERLONGWORDEXCEEDINGBOUNDS"


class TestAddHookOverlay:
    """Tests for HookTitleOverlay.add_hook method with synthetic clips."""

    def test_add_hook_both_enabled(self) -> None:
        cfg = DummyConfig({"hook_title.enabled": True, "hook_title.duration_seconds": 4})
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(10, 20, 30)).with_duration(8.0)
        audio = _make_dummy_audio(8.0)
        base = base.with_audio(audio)

        result = overlay.add_hook(base, "POLICE CHASE ESCAPE", context_badge="🔥 Clutch Escape")

        assert isinstance(result, CompositeVideoClip)
        assert result.size == (720, 1280)
        assert result.duration == 8.0
        # Exactly 3 clips: base video + hook banner + badge pill
        assert len(result.clips) == 3

        # Overlay durations clamped to self.duration (4.0s)
        hook_clip = result.clips[1]
        badge_clip = result.clips[2]
        assert hook_clip.duration == 4.0
        assert badge_clip.duration == 4.0

        # Hook banner positioned near the top (center, 120)
        assert hook_clip.pos(0) == ("center", 120)

        # Badge pill positioned below the hook banner (badge_y > 120)
        badge_pos = badge_clip.pos(0)
        assert badge_pos[0] == "center"
        assert isinstance(badge_pos[1], (int, float))
        assert badge_pos[1] > 120

        # Audio is preserved
        assert result.audio is not None
        assert result.audio == base.audio

    def test_add_hook_duration_clamped_to_shorter_video(self) -> None:
        cfg = DummyConfig({"hook_title.enabled": True, "hook_title.duration_seconds": 5.0})
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(2.5)
        result = overlay.add_hook(base, "QUICK MOMENT", context_badge="⚡ Fast")

        assert len(result.clips) == 3
        assert result.clips[1].duration == 2.5
        assert result.clips[2].duration == 2.5

    def test_add_hook_disappears_after_duration(self) -> None:
        cfg = DummyConfig({
            "hook_title.enabled": True,
            "hook_title.duration_seconds": 3.0,
            "hook_title.fade_in": False,
        })
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(6.0)
        result = overlay.add_hook(base, "SHORT TITLE", context_badge="💀 Karma")

        frame_active = result.get_frame(1.5)
        # While active, top area contains non-zero banner pixels
        assert (frame_active[120:250, :] > 0).any()

        frame_expired = result.get_frame(4.5)
        # After 3.0s, overlay has disappeared; base video is pure black (0, 0, 0)
        assert (frame_expired[120:250, :] == 0).all()

    def test_add_hook_disabled_with_context_badge(self) -> None:
        cfg = DummyConfig({"hook_title.enabled": False})
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(5, 5, 5)).with_duration(5.0)
        result = overlay.add_hook(base, "IGNORED HOOK", context_badge="🎨 Insane Detail")

        assert isinstance(result, CompositeVideoClip)
        # Exactly 2 clips: base video + badge pill (hook banner skipped)
        assert len(result.clips) == 2
        badge_clip = result.clips[1]
        assert badge_clip.duration == 4.0
        # Positioned near top when no hook banner is present
        assert badge_clip.pos(0) == ("center", 120)

    def test_add_hook_disabled_with_empty_badge_returns_unchanged(self) -> None:
        cfg = DummyConfig({"hook_title.enabled": False})
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(5.0)
        result1 = overlay.add_hook(base, "SKIPPED", context_badge="")
        assert result1 is base

        result2 = overlay.add_hook(base, "SKIPPED", context_badge="   ")
        assert result2 is base

    def test_add_hook_enabled_empty_strings_returns_unchanged(self) -> None:
        cfg = DummyConfig({"hook_title.enabled": True})
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(5.0)
        result = overlay.add_hook(base, "", context_badge="")
        assert result is base

        result_whitespace = overlay.add_hook(base, "   ", context_badge="   ")
        assert result_whitespace is base

    def test_add_hook_enabled_empty_badge_only_hook_layer(self) -> None:
        cfg = DummyConfig({"hook_title.enabled": True})
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(5.0)
        result = overlay.add_hook(base, "SOLO HOOK", context_badge="")

        assert isinstance(result, CompositeVideoClip)
        # Exactly 2 clips: base video + hook banner
        assert len(result.clips) == 2
        assert result.clips[1].pos(0) == ("center", 120)

    def test_add_hook_position_center(self) -> None:
        cfg = DummyConfig({
            "hook_title.enabled": True,
            "hook_title.position": "center",
        })
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(5.0)
        result = overlay.add_hook(base, "CENTER HOOK", context_badge="🔥 Badge")

        assert len(result.clips) == 3
        hook_clip = result.clips[1]
        badge_clip = result.clips[2]
        assert hook_clip.pos(0) == ("center", "center")
        badge_pos = badge_clip.pos(0)
        assert badge_pos[0] == "center"
        assert badge_pos[1] > (1280 // 2)


class TestAudioAndHandlePreservation:
    """Tests for audio preservation and Windows file handle tracking."""

    def test_audio_preservation(self) -> None:
        cfg = DummyConfig()
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(4.0)
        audio = _make_dummy_audio(4.0)
        base = base.with_audio(audio)

        result = overlay.add_hook(base, "AUDIO TEST", "BADGE")
        assert result.audio is not None
        assert result.audio == base.audio

    def test_audio_none_when_absent(self) -> None:
        cfg = DummyConfig()
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(4.0)
        assert base.audio is None

        result = overlay.add_hook(base, "NO AUDIO", "BADGE")
        assert result.audio is None

    def test_source_clip_references_propagated(self) -> None:
        cfg = DummyConfig()
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        mock_source_1 = ColorClip((100, 100), color=(0, 0, 0)).with_duration(2.0)
        mock_source_2 = ColorClip((100, 100), color=(0, 0, 0)).with_duration(2.0)

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(4.0)
        base._source_clip = mock_source_1
        base._source_clips = [mock_source_1, mock_source_2]

        result = overlay.add_hook(base, "HANDLE TEST", "🔥 Badge")
        assert getattr(result, "_source_clip", None) is mock_source_1
        assert getattr(result, "_source_clips", None) == [mock_source_1, mock_source_2]

    def test_source_clip_defaults_when_missing(self) -> None:
        cfg = DummyConfig()
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(4.0)
        # Base clip does not have _source_clip / _source_clips set yet
        assert not hasattr(base, "_source_clip")

        result = overlay.add_hook(base, "DEFAULT HANDLE TEST", "🔥 Badge")
        assert getattr(result, "_source_clip", None) is base
        assert getattr(result, "_source_clips", None) == [base]


class TestFadeInEffect:
    """Tests for fade-in effect on overlay layers."""

    def test_fade_in_enabled(self) -> None:
        cfg = DummyConfig({"hook_title.fade_in": True, "hook_title.duration_seconds": 4.0})
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(4.0)
        result = overlay.add_hook(base, "FADE TEST", "🔥 Badge")

        hook_clip = result.clips[1]
        assert hook_clip.mask is not None

        # At t=0, mask should be 0.0 (fade in starts from transparent)
        m0 = hook_clip.mask.get_frame(0.0)
        assert float(m0.max()) == pytest.approx(0.0, abs=1e-4)

        # At t=1.0s, mask should reach full opacity
        m1 = hook_clip.mask.get_frame(1.0)
        assert float(m1.max()) > 0.5

    def test_fade_in_disabled(self) -> None:
        cfg = DummyConfig({"hook_title.fade_in": False, "hook_title.duration_seconds": 4.0})
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(4.0)
        result = overlay.add_hook(base, "NO FADE TEST", "🔥 Badge")

        hook_clip = result.clips[1]
        assert hook_clip.mask is not None

        # At t=0, mask should already be at full alpha (>0.5)
        m0 = hook_clip.mask.get_frame(0.0)
        assert float(m0.max()) > 0.5

    def test_zero_duration_overlay_returns_video_unchanged(self) -> None:
        cfg = DummyConfig({"hook_title.enabled": True, "hook_title.duration_seconds": 0.0})
        overlay = HookTitleOverlay(cfg)  # type: ignore[arg-type]

        base = ColorClip((720, 1280), color=(0, 0, 0)).with_duration(4.0)
        result = overlay.add_hook(base, "SHOULD SKIP", "🔥 Badge")
        assert result is base
