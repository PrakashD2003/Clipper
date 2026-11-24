"""Hook title banner and adaptive context badge overlay using Pillow and MoviePy v2."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from moviepy import CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont

from ..config import Config
from .layout import LayoutGeometry
from .textutil import strip_unrenderable

logger = logging.getLogger(__name__)


def _load_font(font_name: str | None, size: int) -> ImageFont.ImageFont:
    """Load a TrueType font with fallbacks to system fonts and the default font."""
    candidates: list[str] = []
    if font_name:
        candidates.extend([font_name, f"{font_name}.ttf"])
    candidates.extend([
        "arialbd.ttf",
        "arial.ttf",
        "segoeui.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _split_oversized_word(
    word: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_w: int,
    stroke_width: int = 0,
) -> list[str]:
    """Split a single token wider than max_w into character chunks that fit within max_w."""
    chunks: list[str] = []
    current = ""
    for ch in word:
        candidate = current + ch
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
        if (bbox[2] - bbox[0]) <= max_w or not current:
            current = candidate
        else:
            chunks.append(current)
            current = ch
    if current:
        chunks.append(current)
    return chunks


def _wrap_text(
    text: str,
    font: ImageFont.ImageFont,
    max_w: int,
    stroke_width: int = 0,
    break_long_words: bool = False,
) -> list[str]:
    """Wrap text so that every line fits within max_w width.

    When ``break_long_words=True``, single tokens wider than ``max_w`` are split
    into character chunks so rendered banners/badges never overflow frame bounds.
    """
    tmp = Image.new("RGBA", (1, 1))
    d_tmp = ImageDraw.Draw(tmp)

    paragraphs = text.split("\n")
    lines: list[str] = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue
        current_line: list[str] = []
        for word in words:
            if break_long_words:
                word_bbox = d_tmp.textbbox(
                    (0, 0), word, font=font, stroke_width=stroke_width
                )
                if (word_bbox[2] - word_bbox[0]) > max_w:
                    if current_line:
                        lines.append(" ".join(current_line))
                        current_line = []
                    pieces = _split_oversized_word(
                        word, d_tmp, font, max_w, stroke_width
                    )
                    lines.extend(pieces[:-1])
                    current_line = [pieces[-1]]
                    continue

            test_line = " ".join(current_line + [word])
            bbox = d_tmp.textbbox((0, 0), test_line, font=font, stroke_width=stroke_width)
            if (bbox[2] - bbox[0]) <= max_w or not current_line:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))

    return lines


def render_hook_banner(
    text: str,
    max_width: int,
    font_name: str = "Arial-Bold",
    font_size: int = 50,
    color: str = "white",
    stroke_color: str = "black",
    stroke_width: int = 3,
) -> np.ndarray:
    """Render hook title text inside a rounded dark banner into an RGBA numpy array."""
    text = strip_unrenderable(text)
    font = _load_font(font_name, font_size)
    pad_x = 32
    pad_y = 18
    line_spacing = max(6, int(font_size * 0.2))
    content_max_w = max(200, max_width - 2 * pad_x)

    lines = _wrap_text(
        text, font, content_max_w, stroke_width=stroke_width, break_long_words=True
    )
    if not lines:
        lines = [text]

    tmp = Image.new("RGBA", (1, 1))
    d_tmp = ImageDraw.Draw(tmp)
    line_metrics: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for line in lines:
        b = d_tmp.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        line_metrics.append((b[2] - b[0], b[3] - b[1], b))

    max_line_w = max(m[0] for m in line_metrics)
    total_line_h = sum(m[1] for m in line_metrics) + (len(lines) - 1) * line_spacing

    banner_w = max_line_w + 2 * pad_x
    banner_h = total_line_h + 2 * pad_y

    img = Image.new("RGBA", (banner_w, banner_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [(0, 0), (banner_w - 1, banner_h - 1)],
        radius=min(20, banner_h // 2),
        fill=(0, 0, 0, 185),
    )

    curr_y = pad_y
    for line, (lw, lh, b) in zip(lines, line_metrics):
        tx = (banner_w - lw) // 2 - b[0]
        ty = curr_y - b[1]
        draw.text(
            (tx, ty),
            line,
            font=font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )
        curr_y += lh + line_spacing

    return np.array(img, dtype=np.uint8)


def render_context_badge(
    text: str,
    font_size: int = 28,
    font_name: str | None = None,
    max_width: int | None = None,
) -> np.ndarray:
    """Render a compact context badge pill into an RGBA numpy array."""
    text = strip_unrenderable(text)
    font = _load_font(font_name, font_size)
    tmp = Image.new("RGBA", (1, 1))
    d_tmp = ImageDraw.Draw(tmp)

    pad_x = 22
    pad_y = 8
    line_spacing = max(4, int(font_size * 0.2))

    if max_width is not None:
        content_max_w = max(80, int(max_width) - 2 * pad_x)
        lines = _wrap_text(
            text, font, content_max_w, stroke_width=0, break_long_words=True
        ) or [text]
    else:
        lines = [text]

    line_metrics: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for line in lines:
        b = d_tmp.textbbox((0, 0), line, font=font)
        line_metrics.append((max(1, b[2] - b[0]), max(1, b[3] - b[1]), b))

    max_line_w = max(m[0] for m in line_metrics)
    total_line_h = sum(m[1] for m in line_metrics) + (len(lines) - 1) * line_spacing

    badge_w = max_line_w + 2 * pad_x
    badge_h = total_line_h + 2 * pad_y

    img = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [(0, 0), (badge_w - 1, badge_h - 1)],
        radius=min(badge_h // 2, 24) if len(lines) > 1 else badge_h // 2,
        fill=(25, 25, 30, 215),
        outline=(255, 255, 255, 75),
        width=1,
    )
    curr_y = pad_y
    for line, (lw, lh, b) in zip(lines, line_metrics):
        tx = (badge_w - lw) // 2 - b[0]
        ty = curr_y - b[1]
        draw.text((tx, ty), line, font=font, fill="white")
        curr_y += lh + line_spacing

    return np.array(img, dtype=np.uint8)


def render_replay_badge(
    text: str = "🔴 INSTANT REPLAY",
    font_size: int = 28,
    font_name: str | None = None,
    max_width: int | None = None,
    as_pil: bool = False,
) -> np.ndarray | Image.Image:
    """Render a high-contrast sports broadcast 'INSTANT REPLAY' badge pill into an RGBA numpy array or PIL Image."""
    text = strip_unrenderable(text)
    font = _load_font(font_name, font_size)
    tmp = Image.new("RGBA", (1, 1))
    d_tmp = ImageDraw.Draw(tmp)

    pad_x = 28
    pad_y = 12

    bbox = d_tmp.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    badge_w = text_w + 2 * pad_x
    badge_h = text_h + 2 * pad_y

    img = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [(0, 0), (badge_w - 1, badge_h - 1)],
        radius=min(18, badge_h // 2),
        fill=(18, 18, 24, 235),
        outline=(230, 35, 35, 255),
        width=2,
    )

    tx = (badge_w - text_w) // 2 - bbox[0]
    ty = (badge_h - text_h) // 2 - bbox[1]
    draw.text((tx, ty), text, font=font, fill="white")

    if as_pil:
        return img
    return np.array(img, dtype=np.uint8)


def render_replay_badge_pil(
    text: str = "🔴 INSTANT REPLAY",
    font_size: int = 28,
    font_name: str | None = None,
    max_width: int | None = None,
) -> Image.Image:
    """Render a high-contrast sports broadcast 'INSTANT REPLAY' badge pill into a PIL Image."""
    res = render_replay_badge(
        text=text,
        font_size=font_size,
        font_name=font_name,
        max_width=max_width,
        as_pil=True,
    )
    return res  # type: ignore[return-value]


DEFAULT_REPLAY_BADGE_TEXT = "🔴 INSTANT REPLAY"


def create_replay_badge(
    text: str | None = None,
    width: int = 1080,
    height: int = 1920,
    font_size: int = 28,
    font_name: str | None = None,
    duration: float | None = None,
    config: Config | None = None,
    layout: LayoutGeometry | None = None,
) -> ImageClip:
    """Render a high-contrast sports broadcast replay badge pill as an ImageClip positioned at ('center', 140).

    When ``text`` is None it comes from ``transformation.replay_badge_text`` (default literal fallback).
    """
    if text is None:
        text = DEFAULT_REPLAY_BADGE_TEXT
        if config is not None:
            text = str(
                config.get("transformation.replay_badge_text", DEFAULT_REPLAY_BADGE_TEXT)
                or DEFAULT_REPLAY_BADGE_TEXT
            )
    arr = render_replay_badge(
        text=text,
        font_size=font_size,
        font_name=font_name,
        max_width=int(width * 0.90) if width is not None else None,
    )
    badge_y = 140 if layout is None else layout.game_box.y + 16  # top of game_box
    clip = ImageClip(arr).with_position(("center", badge_y))
    if duration is not None and duration > 0:
        clip = clip.with_duration(duration)
    return clip


def render_streamer_tag(
    text: str,
    font_size: int = 26,
    font_name: str | None = None,
    max_width: int | None = None,
) -> np.ndarray:
    """Small semi-transparent pill with the streamer credit ("@name") as an RGBA array."""
    text = strip_unrenderable(text)
    font = _load_font(font_name, font_size)
    pad_x, pad_y = 20, 8
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    b = tmp.textbbox((0, 0), text, font=font)
    tw, th = b[2] - b[0], b[3] - b[1]
    if max_width is not None and tw + 2 * pad_x > max_width:
        keep = max(4, int(len(text) * max_width / (tw + 2 * pad_x)) - 1)
        text = text[:keep].rstrip()
        b = tmp.textbbox((0, 0), text, font=font)
        tw, th = b[2] - b[0], b[3] - b[1]
    w, h = tw + 2 * pad_x, th + 2 * pad_y
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(0, 0), (w - 1, h - 1)], radius=h // 2, fill=(0, 0, 0, 120))
    draw.text((pad_x - b[0], pad_y - b[1]), text, font=font, fill=(255, 255, 255, 235))
    return np.array(img, dtype=np.uint8)


class HookTitleOverlay:
    """Renders hook title banner and adaptive context badge overlays using MoviePy v2."""

    def __init__(self, config: Config):
        self.config = config
        self.enabled: bool = bool(config.get("hook_title.enabled", True))
        self.duration: float = float(config.get("hook_title.duration_seconds", 4))
        self.font_size: int = int(config.get("hook_title.font_size", 50))
        self.fade_in: bool = bool(config.get("hook_title.fade_in", True))
        self.font_name: str = config.get("hook_title.font", "Arial-Bold")
        self.color: str = config.get("hook_title.color", "white")
        self.stroke_color: str = config.get("hook_title.stroke_color", "black")
        self.stroke_width: int = int(config.get("hook_title.stroke_width", 3))
        self.position: str = config.get("hook_title.position", "top")
        self.persistent_title: bool = bool(config.get("hooks.persistent_title", True))
        self.streamer_tag: bool = bool(config.get("hooks.streamer_tag", True))

    def add_hook(
        self,
        video_clip: CompositeVideoClip,
        hook_title: str,
        context_badge: str = "",
        layout: LayoutGeometry | None = None,
        *,
        streamer: str = "",
    ) -> CompositeVideoClip:
        """Add hook title banner and optional adaptive context_badge pill to the video.

        On the blur_fill layout the title stays on screen for the whole clip when
        ``hooks.persistent_title`` is set, and a small "@streamer" credit pill is placed at the
        bottom of the hook band when ``hooks.streamer_tag`` is set and ``streamer`` is given.
        """
        clean_hook = (hook_title or "").strip()
        clean_badge = (context_badge or "").strip()

        should_render_hook = self.enabled and bool(clean_hook)
        should_render_badge = bool(clean_badge)
        is_blur = layout is not None and layout.kind == "blur_fill"
        clean_streamer = (streamer or "").strip().lstrip("@")
        should_render_tag = bool(
            is_blur
            and self.streamer_tag
            and clean_streamer
            and (should_render_hook or should_render_badge)
        )

        if not should_render_hook and not should_render_badge:
            return video_clip

        video_size = getattr(video_clip, "size", None) or (1080, 1920)
        video_w, video_h = video_size

        video_duration = getattr(video_clip, "duration", None)
        if video_duration is not None:
            overlay_duration = min(self.duration, float(video_duration))
        else:
            overlay_duration = self.duration
        if is_blur and self.persistent_title and video_duration is not None:
            overlay_duration = float(video_duration)
        overlay_duration = max(0.0, overlay_duration)
        if overlay_duration <= 0.0:
            return video_clip

        layers: list[Any] = [video_clip]
        hook_banner_h = 0
        tag_arr = None
        tag_reserved = 0
        if should_render_tag:
            tag_arr = render_streamer_tag(
                f"@{clean_streamer}",
                font_size=max(18, int(self.font_size * 0.5)),
                font_name=self.font_name,
                max_width=int(video_w * 0.6),
            )
            tag_reserved = tag_arr.shape[0] + 16  # tag height + gap above it

        if should_render_hook:
            max_w = int(video_w * 0.90)
            hook_arr = render_hook_banner(
                text=clean_hook,
                max_width=max_w,
                font_name=self.font_name,
                font_size=self.font_size,
                color=self.color,
                stroke_color=self.stroke_color,
                stroke_width=self.stroke_width,
            )
            hook_banner_h = hook_arr.shape[0]

            hook_clip = ImageClip(hook_arr).with_duration(overlay_duration)
            if layout is not None:
                hb = layout.hook_band
                hook_y = hb.y + max(0, (hb.h - tag_reserved - hook_banner_h) // 2)
                hook_clip = hook_clip.with_position(("center", hook_y))
            elif self.position == "center":
                hook_clip = hook_clip.with_position(("center", "center"))
            else:
                hook_clip = hook_clip.with_position(("center", 120))

            if self.fade_in and overlay_duration > 0:
                fade_dur = min(0.5, overlay_duration / 2.0)
                if fade_dur > 0:
                    try:
                        from moviepy.video.fx import CrossFadeIn
                        hook_clip = hook_clip.with_effects([CrossFadeIn(fade_dur)])
                    except Exception as e:
                        logger.debug("Failed to apply CrossFadeIn to hook clip: %s", e)

            layers.append(hook_clip)

        if should_render_badge:
            badge_font_size = max(20, int(self.font_size * 0.55))
            badge_arr = render_context_badge(
                text=clean_badge,
                font_size=badge_font_size,
                font_name=self.font_name,
                max_width=int(video_w * 0.90),
            )
            badge_clip = ImageClip(badge_arr).with_duration(overlay_duration)

            if layout is not None:
                hb = layout.hook_band
                if should_render_hook:
                    # Hook + badge stack centred as one block inside hook_band.
                    block_h = hook_banner_h + 12 + badge_arr.shape[0]
                    top = hb.y + max(0, (hb.h - tag_reserved - block_h) // 2)
                    hook_y = top
                    badge_y = top + hook_banner_h + 12
                    layers[1] = layers[1].with_position(("center", hook_y))
                else:
                    badge_y = hb.y + max(0, (hb.h - tag_reserved - badge_arr.shape[0]) // 2)
            elif should_render_hook and self.position != "center":
                badge_y = 120 + hook_banner_h + 16
            elif should_render_hook and self.position == "center":
                badge_y = (video_h // 2) + (hook_banner_h // 2) + 16
            else:
                badge_y = 120

            badge_clip = badge_clip.with_position(("center", badge_y))

            if self.fade_in and overlay_duration > 0:
                fade_dur = min(0.5, overlay_duration / 2.0)
                if fade_dur > 0:
                    try:
                        from moviepy.video.fx import CrossFadeIn
                        badge_clip = badge_clip.with_effects([CrossFadeIn(fade_dur)])
                    except Exception as e:
                        logger.debug("Failed to apply CrossFadeIn to badge clip: %s", e)

            layers.append(badge_clip)

        if tag_arr is not None and layout is not None:
            hb = layout.hook_band
            tag_y = hb.y + hb.h - tag_arr.shape[0] - 6
            tag_clip = (
                ImageClip(tag_arr)
                .with_duration(overlay_duration)
                .with_position(("center", tag_y))
            )
            layers.append(tag_clip)

        result = CompositeVideoClip(layers, size=video_size)
        if getattr(video_clip, "audio", None) is not None:
            result = result.with_audio(video_clip.audio)

        # Propagate Windows file-lock handles for Exporter cleanup
        source_clip = getattr(video_clip, "_source_clip", None)
        result._source_clip = source_clip if source_clip is not None else video_clip

        source_clips = getattr(video_clip, "_source_clips", None)
        if source_clips is not None:
            result._source_clips = list(source_clips)
        elif result._source_clip is not None:
            result._source_clips = [result._source_clip]
        else:
            result._source_clips = [video_clip]

        return result
