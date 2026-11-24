"""Hormozi-style ASS caption generation, live chat burst ticker, and FFmpeg libass burn-in."""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from ..config import Config
from ..models import WordTimestamp
from .layout import LayoutGeometry
from .textutil import strip_unrenderable

logger = logging.getLogger(__name__)

_NAMED_ASS_COLORS: dict[str, str] = {
    "white": "&H00FFFFFF",
    "black": "&H00000000",
    "yellow": "&H0000FFFF",
    "gold": "&H0000D7FF",
    "red": "&H000000FF",
    "cyan": "&H00FFFF00",
}

_HEX6_RE = re.compile(r"[0-9a-fA-F]{6}")
_ASS_CONTROL_RE = re.compile(r"\\([Nnh])")
_VALID_CAPTION_STYLES = frozenset(
    {"hormozi", "single_word", "sentence", "bounce", "pop", "karaoke", "box"}
)
_VALID_ANIMATIONS = frozenset(
    {"none", "pop", "bounce", "sweep", "color_sweep"}
)

MOMENT_TYPE_CAPTION_STYLES: dict[str, str] = {
    "funny": "bounce",
    "chat_chaos": "bounce",
    "fail": "bounce",
    "hype": "pop",
    "clutch": "pop",
    "graphics_detail": "box",
    "analysis": "box",
    "lore_secret": "box",
}


def resolve_caption_style(
    caption_style: str | None = None,
    moment_type: str | None = None,
    default_style: str = "hormozi",
) -> str:
    """Resolve the effective subtitle style from an explicit override or moment_type.

    Priority:
    1. Explicit style override (e.g. 'bounce', 'pop', 'box', 'karaoke', 'hormozi') if not 'auto'.
    2. Automatic mapping based on moment_type:
       - funny, chat_chaos, fail -> bounce
       - hype, clutch -> pop
       - graphics_detail, analysis, lore_secret -> box
    3. Fallback to default_style (e.g. config captions.style).
    """
    clean_style = str(caption_style or "").strip().lower()
    if clean_style and clean_style != "auto":
        if clean_style in _VALID_CAPTION_STYLES:
            return clean_style
        return clean_style

    clean_moment = str(moment_type or "").strip().lower()
    if clean_moment in MOMENT_TYPE_CAPTION_STYLES:
        return MOMENT_TYPE_CAPTION_STYLES[clean_moment]

    return default_style or "hormozi"


_FONT_CANDIDATES: dict[str, list[str]] = {
    "impact": ["impact.ttf", "Impact.ttf", "impact.TTF", "Impact"],
    "arial black": ["ariblk.ttf", "Arial Black.ttf", "ariblk.TTF", "ariblk", "Arial Black"],
    "arial": ["arial.ttf", "Arial.ttf", "arial.TTF", "arial"],
}


def _is_font_installed(font_family: str) -> bool:
    """Check if a font family is installed on the local system."""
    clean = font_family.strip().lower()
    candidates = _FONT_CANDIDATES.get(clean, [f"{clean}.ttf", clean])
    try:
        from PIL import ImageFont

        for cand in candidates:
            try:
                ImageFont.truetype(cand, 10)
                return True
            except OSError:
                continue
    except ImportError:
        pass

    windir = os.environ.get("WINDIR", r"C:\Windows")
    win_fonts = Path(windir) / "Fonts"
    if win_fonts.is_dir():
        for cand in candidates:
            if (win_fonts / cand).exists():
                return True

    # Check Windows per-user font directory (%LOCALAPPDATA%\Microsoft\Windows\Fonts)
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        user_fonts = Path(local_app_data) / "Microsoft" / "Windows" / "Fonts"
        if user_fonts.is_dir():
            for cand in candidates:
                if (user_fonts / cand).exists():
                    return True

    # Check Linux / Unix font paths if running on non-Windows
    for sys_font_dir in (Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts"):
        if sys_font_dir.is_dir():
            for cand in candidates:
                if list(sys_font_dir.glob(f"**/{cand}")):
                    return True
    return False


def resolve_font(font_name: str, fallback: str = "Arial Black") -> str:
    """Resolve font name with automatic fallback (e.g. Impact -> Arial Black).

    Preserves explicit custom fonts while ensuring missing 'Impact' falls back cleanly.
    """
    raw = (font_name or "").strip()
    if not raw:
        return fallback
    clean_lower = raw.lower()
    if clean_lower == "impact":
        if _is_font_installed("impact"):
            return "Impact"
        if _is_font_installed("arial black"):
            return "Arial Black"
        return fallback or "Arial Black"
    return raw


def _chunk_words(words: list[str], max_chunk_size: int = 3) -> list[list[str]]:
    """Split words into natural 2-3 word chunks, borrowing to prevent 1-word trailing orphans."""
    if not words:
        return []
    effective_max = max(1, max_chunk_size)
    if len(words) <= effective_max:
        return [words]
    chunks = [words[i : i + effective_max] for i in range(0, len(words), effective_max)]
    if len(chunks) > 1 and len(chunks[-1]) == 1 and len(chunks[-2]) > 2:
        chunks[-1].insert(0, chunks[-2].pop())
    return chunks


def _safe_int(
    value: Any,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Coerce config value to int safely (rejecting None/bool/non-numeric)."""
    if value is None or isinstance(value, bool):
        result = default
    else:
        try:
            fval = float(value)
            result = int(fval) if math.isfinite(fval) else default
        except (TypeError, ValueError):
            result = default
    if minimum is not None and result < minimum:
        return minimum
    if maximum is not None and result > maximum:
        return maximum
    return result


def _safe_float(
    value: Any,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce config value to finite float safely (rejecting None/bool/NaN/inf)."""
    if value is None or isinstance(value, bool):
        result = default
    else:
        try:
            fval = float(value)
            result = fval if math.isfinite(fval) else default
        except (TypeError, ValueError):
            result = default
    if minimum is not None and result < minimum:
        result = minimum
    if maximum is not None and result > maximum:
        result = maximum
    return result


def _escape_ffmpeg_ass_filename(filename: str) -> str:
    """Escape a relative ASS filename for FFmpeg's ``-vf ass=<filename>`` filtergraph.

    Applies FFmpeg's two-level escaping (filter option parser, then filtergraph parser)
    while leaving standard filenames like ``clip_01.ass`` completely untouched.
    """
    # Level 1 (ass filter option parser): escape \, ', :, = and any leading spaces
    inner = (
        filename.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace("=", "\\=")
    )
    leading_spaces = len(inner) - len(inner.lstrip(" "))
    if leading_spaces:
        inner = ("\\ " * leading_spaces) + inner[leading_spaces:]

    # Level 2 (filtergraph parser): escape \, ', ,, ;, [, ]
    return (
        inner.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _sec_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format H:MM:SS.cc without 60.00 rollover overflow."""
    val = float(seconds)
    if not math.isfinite(val) or val < 0.0:
        val = 0.0
    total_cs = round(val * 100)
    hrs, rem_cs = divmod(total_cs, 360000)
    mins, rem_cs = divmod(rem_cs, 6000)
    secs, cs = divmod(rem_cs, 100)
    return f"{hrs}:{mins:02d}:{secs:02d}.{cs:02d}"


def _hex_to_ass_bgr(hex_color: str, default: str = "&H00D7FF&") -> str:
    """Convert CSS hex color '#RRGGBB' into ASS inline BGR override tag '&HBBGGRR&'."""
    cleaned = str(hex_color or "").strip().removeprefix("#")
    if not _HEX6_RE.fullmatch(cleaned):
        return default
    r, g, b = cleaned[0:2], cleaned[2:4], cleaned[4:6]
    return f"&H{b}{g}{r}&".upper()


def _resolve_highlight_bgr(color: str, default: str = "&H00E6FF&") -> str:
    """Resolve highlight color from hex (#RRGGBB), named color, or ASS (&HBBGGRR& / &HAABBGGRR)."""
    raw = str(color or "").strip()
    if not raw:
        return default
    clean = raw.rstrip("&")
    if clean.startswith("&H") or clean.startswith("&h"):
        hex_part = clean[2:]
        if len(hex_part) == 6 and bool(_HEX6_RE.fullmatch(hex_part)):
            return f"&H{hex_part.upper()}&"
        if len(hex_part) == 8 and bool(re.fullmatch(r"[0-9a-fA-F]{8}", hex_part)):
            return f"&H{hex_part[2:].upper()}&"
    lower_raw = raw.lower()
    if lower_raw in _NAMED_ASS_COLORS:
        named = _NAMED_ASS_COLORS[lower_raw]
        return f"&H{named[4:]}&"
    bgr = _hex_to_ass_bgr(raw, default="")
    if bgr:
        return bgr
    return default


def _color_to_ass_style(color: str, default: str = "&H00FFFFFF") -> str:
    """Convert CSS named color, '#RRGGBB', or '&HAABBGGRR' into ASS style color '&HAABBGGRR'."""
    raw = str(color or "").strip()
    if not raw:
        return default
    clean_ass = raw.rstrip("&")
    if clean_ass.startswith("&H") or clean_ass.startswith("&h"):
        if len(clean_ass) == 10 and bool(re.fullmatch(r"[0-9a-fA-F]{8}", clean_ass[2:])):
            return clean_ass.upper()
        if len(clean_ass) == 8 and bool(_HEX6_RE.fullmatch(clean_ass[2:])):
            return f"&H00{clean_ass[2:].upper()}"
    lower_raw = raw.lower()
    if lower_raw in _NAMED_ASS_COLORS:
        return _NAMED_ASS_COLORS[lower_raw]
    bgr = _hex_to_ass_bgr(raw, default="")
    if bgr:
        return "&H00" + bgr[2:-1]
    return default


def _color_to_ass_style_with_alpha(
    color: str, opacity: float | None = None, default: str = "&H00000000"
) -> str:
    """Convert CSS named color, '#RRGGBB', or ASS color into '&HAABBGGRR' format with opacity.

    Opacity is between 0.0 (transparent) and 1.0 (fully opaque).
    In ASS style colors, the alpha byte 00 is opaque and FF is fully transparent.
    """
    raw = str(color or "").strip()
    if not raw:
        return default

    alpha_hex: str | None = None
    if opacity is not None and math.isfinite(opacity):
        clamped_op = max(0.0, min(1.0, float(opacity)))
        alpha_int = round((1.0 - clamped_op) * 255)
        alpha_hex = f"{alpha_int:02X}"

    clean_ass = raw.rstrip("&")
    if clean_ass.startswith("&H") or clean_ass.startswith("&h"):
        if len(clean_ass) == 10 and bool(re.fullmatch(r"[0-9a-fA-F]{8}", clean_ass[2:])):
            if alpha_hex is not None:
                return f"&H{alpha_hex}{clean_ass[4:].upper()}"
            return clean_ass.upper()
        if len(clean_ass) == 8 and bool(_HEX6_RE.fullmatch(clean_ass[2:])):
            a_str = alpha_hex if alpha_hex is not None else "00"
            return f"&H{a_str}{clean_ass[2:].upper()}"

    clean_hex = raw.removeprefix("#").strip()
    if len(clean_hex) == 8 and bool(re.fullmatch(r"[0-9a-fA-F]{8}", clean_hex)):
        r, g, b, a = clean_hex[0:2], clean_hex[2:4], clean_hex[4:6], clean_hex[6:8]
        if alpha_hex is None:
            css_a = int(a, 16)
            alpha_hex = f"{(255 - css_a):02X}"
        return f"&H{alpha_hex}{b}{g}{r}".upper()

    lower_raw = raw.lower()
    if lower_raw in _NAMED_ASS_COLORS:
        base = _NAMED_ASS_COLORS[lower_raw]
        if alpha_hex is not None:
            return f"&H{alpha_hex}{base[4:]}".upper()
        return base

    bgr = _hex_to_ass_bgr(clean_hex, default="")
    if bgr:
        bgr_val = bgr[2:-1]
        a_str = alpha_hex if alpha_hex is not None else "00"
        return f"&H{a_str}{bgr_val}".upper()

    return default


def _escape_ass_text(text: str) -> str:
    """Sanitize newlines and escape ASS override delimiters safely for libass."""
    sanitized = strip_unrenderable(text)
    sanitized = sanitized.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    sanitized = sanitized.replace("{", r"\{").replace("}", r"\}")
    # Neutralize ASS control codes (\N, \n, \h) without doubling literal backslashes in libass
    return _ASS_CONTROL_RE.sub("\\\\\u200b\\1", sanitized)


class AssCaptionBurner:
    """Generates Hormozi/single-word/sentence ASS subtitles and burns them via FFmpeg libass."""

    MAX_CHAT_TICKER_MESSAGES = 5
    CHAT_TICKER_DURATION_SECONDS = 2.5

    def __init__(self, config: Config) -> None:
        self.config = config
        raw_style = str(config.get("captions.style", "hormozi") or "hormozi").strip().lower()
        if raw_style not in _VALID_CAPTION_STYLES:
            logger.warning(
                "Unknown captions.style %r; defaulting to 'hormozi'.", raw_style
            )
            raw_style = "hormozi"
        self.style = raw_style

        self.words_per_group = _safe_int(
            config.get("captions.words_per_group", 3), 3, minimum=1
        )
        raw_font = str(config.get("captions.font", "Impact") or "Impact")
        clean_font = (
            raw_font.replace(",", " ").replace("\r", " ").replace("\n", " ").strip()
            or "Impact"
        )
        self.font = resolve_font(clean_font)
        self.font_size = _safe_int(config.get("captions.font_size", 82), 82, minimum=1)
        self.primary_color = _color_to_ass_style(
            config.get("captions.color", "white"), default="&H00FFFFFF"
        )
        self.stroke_color = _color_to_ass_style(
            config.get("captions.stroke_color", "black"), default="&H00000000"
        )
        stroke_val = _safe_float(
            config.get("captions.stroke_width", 8), 8.0, minimum=0.0
        )
        self.stroke_width: int | float = (
            int(stroke_val) if stroke_val.is_integer() else stroke_val
        )

        raw_highlight = str(config.get("captions.highlight_color", "#FFE600") or "#FFE600")
        self.highlight_bgr = _resolve_highlight_bgr(raw_highlight, default="&H00E6FF&")
        self._ticker_font = self.font  # ChatTicker keeps the base font; presets never touch it

        # Animation mode: none | pop | bounce | sweep | color_sweep
        user_specified_anim = False
        if hasattr(config, "is_user_specified") and callable(config.is_user_specified):
            user_specified_anim = config.is_user_specified("captions.animation")

        if user_specified_anim:
            raw_anim = str(config.get("captions.animation", "bounce") or "bounce").strip().lower()
        else:
            if self.style == "pop":
                raw_anim = "pop"
            elif self.style == "bounce":
                raw_anim = "bounce"
            elif self.style in ("sentence", "single_word", "karaoke", "box"):
                raw_anim = "none"
            else:
                raw_anim = str(config.get("captions.animation", "bounce") or "bounce").strip().lower()
        if raw_anim not in _VALID_ANIMATIONS:
            logger.warning("Unknown captions.animation %r; defaulting to 'bounce'.", raw_anim)
            raw_anim = "bounce"
        self.animation = raw_anim

        self.animation_scale = _safe_float(
            config.get("captions.animation_scale", 1.30), 1.30, minimum=1.0, maximum=3.0
        )
        self.animation_duration_ms = _safe_int(
            config.get("captions.animation_duration_ms", 140), 140, minimum=20, maximum=2000
        )

        # Drop shadow configuration
        shadow_val = _safe_float(
            config.get("captions.shadow_offset", 6.0), 6.0, minimum=0.0, maximum=50.0
        )
        self.shadow_offset: int | float = (
            int(shadow_val) if shadow_val.is_integer() else shadow_val
        )
        self.shadow_color = _color_to_ass_style_with_alpha(
            config.get("captions.shadow_color", "&H90000000"), default="&H90000000"
        )

        # Background box configuration (BorderStyle=3)
        self.box_style = bool(config.get("captions.box_style", False)) or (self.style == "box")
        box_pad_val = _safe_float(
            config.get("captions.box_padding", 6), 6.0, minimum=0.0, maximum=100.0
        )
        self.box_padding: int | float = (
            int(box_pad_val) if box_pad_val.is_integer() else box_pad_val
        )
        self.box_opacity = _safe_float(
            config.get("captions.box_opacity", 0.60), 0.60, minimum=0.0, maximum=1.0
        )
        raw_box_color = str(config.get("captions.box_color", "black") or "black")
        self.box_color = _color_to_ass_style_with_alpha(
            raw_box_color, opacity=self.box_opacity, default="&H99000000"
        )

        self.codec = str(config.get("video.codec", "libx264") or "libx264").strip().lower()
        resolution = config.get("video.resolution", [1080, 1920])
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            self.width = _safe_int(resolution[0], 1080, minimum=1)
            self.height = _safe_int(resolution[1], 1920, minimum=1)
        else:
            self.width, self.height = 1080, 1920

        pos_y_ratio = _safe_float(
            config.get("captions.position_y_ratio", 0.75), 0.75, minimum=0.0, maximum=1.0
        )
        self.margin_v = round(self.height * (1.0 - pos_y_ratio))
        self.show_chat_ticker = bool(config.get("transformation.show_chat_ticker", True))

    def generate_ass(
        self,
        *args: Any,
        style_preset: dict[str, Any] | None = None,
        caption_style: str | None = None,
        moment_type: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Write the .ass file; `style_preset`, `caption_style`, or `moment_type`
        overrides the word-caption and narration-card styles only (ChatTicker is left alone)."""
        eff_preset = dict(style_preset) if style_preset else {}

        target_style: str | None = None
        if caption_style and str(caption_style).strip().lower() != "auto":
            target_style = str(caption_style).strip().lower()
        elif moment_type and str(moment_type).strip().lower() in MOMENT_TYPE_CAPTION_STYLES:
            target_style = MOMENT_TYPE_CAPTION_STYLES[str(moment_type).strip().lower()]

        if target_style and target_style in _VALID_CAPTION_STYLES:
            eff_preset["style"] = target_style
            if target_style == "bounce":
                eff_preset["animation"] = "bounce"
            elif target_style == "pop":
                eff_preset["animation"] = "pop"
            elif target_style == "box":
                eff_preset["box_style"] = True
                eff_preset["animation"] = "none"
            elif target_style == "karaoke":
                eff_preset["animation"] = "none"

        if not eff_preset:
            return self._generate_ass(*args, **kwargs)
        saved = (
            self.style,
            self.font,
            self.primary_color,
            self.stroke_color,
            self.stroke_width,
            self.highlight_bgr,
            self.shadow_offset,
            self.shadow_color,
            self.box_style,
            self.box_padding,
            self.box_opacity,
            self.box_color,
            self.animation,
            self.animation_scale,
            self.animation_duration_ms,
        )
        try:
            if "style" in eff_preset and eff_preset["style"] is not None:
                raw_s = str(eff_preset["style"]).strip().lower()
                if raw_s in _VALID_CAPTION_STYLES:
                    self.style = raw_s
                    if self.style == "bounce":
                        self.animation = "bounce"
                    elif self.style == "pop":
                        self.animation = "pop"
                    elif self.style == "box":
                        self.box_style = True
                        self.animation = "none"
                    elif self.style == "karaoke":
                        self.animation = "none"
            font = re.sub(r"[,\r\n]+", " ", str(eff_preset.get("font") or "")).strip()
            if font:
                self.font = resolve_font(font)
            primary_col = eff_preset.get("primary_color") or eff_preset.get("color")
            if primary_col:
                self.primary_color = _color_to_ass_style(
                    primary_col, default=self.primary_color
                )
            outline_col = eff_preset.get("outline_color") or eff_preset.get("stroke_color")
            if outline_col:
                self.stroke_color = _color_to_ass_style(
                    outline_col, default=self.stroke_color
                )
            sw_raw = eff_preset.get("stroke_width") if "stroke_width" in eff_preset else eff_preset.get("outline_width")
            if sw_raw is not None:
                sw_val = _safe_float(sw_raw, float(self.stroke_width), minimum=0.0)
                self.stroke_width = int(sw_val) if sw_val.is_integer() else sw_val
            if eff_preset.get("highlight_color"):
                self.highlight_bgr = _resolve_highlight_bgr(
                    str(eff_preset["highlight_color"]), default=self.highlight_bgr
                )
            if "shadow_offset" in eff_preset and eff_preset["shadow_offset"] is not None:
                s_val = _safe_float(eff_preset["shadow_offset"], float(self.shadow_offset), minimum=0.0)
                self.shadow_offset = int(s_val) if s_val.is_integer() else s_val
            if eff_preset.get("shadow_color"):
                self.shadow_color = _color_to_ass_style_with_alpha(
                    eff_preset["shadow_color"], default=self.shadow_color
                )
            if "box_style" in eff_preset and eff_preset["box_style"] is not None:
                self.box_style = bool(eff_preset["box_style"])
            if "box_padding" in eff_preset and eff_preset["box_padding"] is not None:
                bp_val = _safe_float(eff_preset["box_padding"], float(self.box_padding), minimum=0.0)
                self.box_padding = int(bp_val) if bp_val.is_integer() else bp_val
            if "box_opacity" in eff_preset and eff_preset["box_opacity"] is not None:
                self.box_opacity = _safe_float(eff_preset["box_opacity"], self.box_opacity, minimum=0.0, maximum=1.0)
                if not eff_preset.get("box_color"):
                    self.box_color = _color_to_ass_style_with_alpha(
                        self.box_color, opacity=self.box_opacity, default=self.box_color
                    )
            if eff_preset.get("box_color"):
                self.box_color = _color_to_ass_style_with_alpha(
                    eff_preset["box_color"], opacity=self.box_opacity, default=self.box_color
                )
            if "animation" in eff_preset and eff_preset["animation"] is not None:
                raw_a = str(eff_preset["animation"]).strip().lower()
                if raw_a in _VALID_ANIMATIONS:
                    self.animation = raw_a
            if "animation_scale" in eff_preset and eff_preset["animation_scale"] is not None:
                self.animation_scale = _safe_float(eff_preset["animation_scale"], self.animation_scale, minimum=1.0, maximum=3.0)
            if "animation_duration_ms" in eff_preset and eff_preset["animation_duration_ms"] is not None:
                self.animation_duration_ms = _safe_int(eff_preset["animation_duration_ms"], self.animation_duration_ms, minimum=20, maximum=2000)
            return self._generate_ass(*args, **kwargs)
        finally:
            (
                self.style,
                self.font,
                self.primary_color,
                self.stroke_color,
                self.stroke_width,
                self.highlight_bgr,
                self.shadow_offset,
                self.shadow_color,
                self.box_style,
                self.box_padding,
                self.box_opacity,
                self.box_color,
                self.animation,
                self.animation_scale,
                self.animation_duration_ms,
            ) = saved

    def _render_animated_tokens(
        self,
        tokens: list[str],
        active_idx: int,
        dur_ms: int,
        primary_inline_bgr: str,
        animation_override: str | None = None,
    ) -> str:
        """Render a list of tokens with active token highlighted and animated."""
        anim = animation_override or self.animation
        scale_pct = round(self.animation_scale * 100)
        rendered_tokens: list[str] = []
        for j, token in enumerate(tokens):
            if j == active_idx:
                safe_token = token + "\u200b" if token.endswith("\\") else token
                if anim == "bounce":
                    t1 = min(round(self.animation_duration_ms * 0.5), max(10, dur_ms // 3))
                    t2 = min(self.animation_duration_ms, max(t1 + 10, (2 * dur_ms) // 3))
                    t2 = min(t2, dur_ms)
                    if t1 >= t2:
                        t1 = max(5, dur_ms // 2)
                        t2 = max(t1 + 1, dur_ms)
                    anim_open = (
                        rf"{{\c{self.highlight_bgr}\fscx100\fscy100"
                        rf"\t(0,{t1},\fscx{scale_pct}\fscy{scale_pct})"
                        rf"\t({t1},{t2},\fscx100\fscy100)}}"
                    )
                    anim_close = rf"{{\c{primary_inline_bgr}\fscx100\fscy100}}"
                    rendered_tokens.append(anim_open + safe_token + anim_close)
                elif anim == "pop":
                    t_pop = min(self.animation_duration_ms, max(10, dur_ms // 2))
                    t_pop = min(t_pop, dur_ms)
                    anim_open = (
                        rf"{{\c{self.highlight_bgr}\fscx{scale_pct}\fscy{scale_pct}"
                        rf"\t(0,{t_pop},\fscx100\fscy100)}}"
                    )
                    anim_close = rf"{{\c{primary_inline_bgr}\fscx100\fscy100}}"
                    rendered_tokens.append(anim_open + safe_token + anim_close)
                elif anim in ("sweep", "color_sweep"):
                    sweep_ms = min(self.animation_duration_ms, dur_ms)
                    anim_open = (
                        rf"{{\c{primary_inline_bgr}"
                        rf"\t(0,{sweep_ms},\c{self.highlight_bgr})}}"
                    )
                    anim_close = rf"{{\c{primary_inline_bgr}}}"
                    rendered_tokens.append(anim_open + safe_token + anim_close)
                else:
                    rendered_tokens.append(
                        rf"{{\c{self.highlight_bgr}}}"
                        + safe_token
                        + rf"{{\c{primary_inline_bgr}}}"
                    )
            else:
                rendered_tokens.append(token)
        return " ".join(rendered_tokens)

    def _generate_narration_chunk_events(
        self,
        text: str,
        start_time: float,
        end_time: float,
        primary_inline_bgr: str,
    ) -> list[str]:
        """Split narration text into natural 2-3 word chunks and emit sequential ASS dialogue events.

        Distributes chunks across [start_time, end_time] with active word highlighting and pop/bounce animation.
        """
        raw_words = [_escape_ass_text(w.strip().upper()) for w in str(text or "").split() if w.strip()]
        if not raw_words or end_time <= start_time:
            return []

        chunk_size = max(1, self.words_per_group)
        chunks = _chunk_words(raw_words, max_chunk_size=chunk_size)
        total_dur = end_time - start_time
        total_words = sum(len(c) for c in chunks)
        if total_words == 0 or total_dur <= 0.0:
            return []

        word_dur = total_dur / total_words
        events: list[str] = []
        cur_time = start_time
        word_counter = 0

        narration_anim = self.animation if self.animation in ("pop", "bounce") else "pop"

        for chunk in chunks:
            upper_chunk = chunk
            if self.style == "sentence":
                chunk_start = cur_time
                word_counter += len(upper_chunk)
                chunk_end = end_time if word_counter == total_words else start_time + (word_counter * word_dur)
                cur_time = chunk_end
                start_ts = _sec_to_ass_time(chunk_start)
                end_ts = _sec_to_ass_time(chunk_end)
                chunk_text = " ".join(upper_chunk)
                t_dur = max(20, round((chunk_end - chunk_start) * 1000))
                scale_pct = round(self.animation_scale * 100)
                if narration_anim == "bounce":
                    t1 = min(round(self.animation_duration_ms * 0.5), max(10, t_dur // 3))
                    t2 = min(self.animation_duration_ms, max(t1 + 10, (2 * t_dur) // 3))
                    t2 = min(t2, t_dur)
                    if t1 >= t2:
                        t1 = max(5, t_dur // 2)
                        t2 = max(t1 + 1, t_dur)
                    anim_tag = (
                        rf"{{\fscx100\fscy100"
                        rf"\t(0,{t1},\fscx{scale_pct}\fscy{scale_pct})"
                        rf"\t({t1},{t2},\fscx100\fscy100)}}"
                    )
                    events.append(f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{anim_tag}{chunk_text}")
                elif narration_anim == "pop":
                    t_pop = min(self.animation_duration_ms, max(10, t_dur // 2))
                    t_pop = min(t_pop, t_dur)
                    anim_tag = rf"{{\fscx{scale_pct}\fscy{scale_pct}\t(0,{t_pop},\fscx100\fscy100)}}"
                    events.append(f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{anim_tag}{chunk_text}")
                else:
                    events.append(f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{chunk_text}")
                continue

            for idx in range(len(upper_chunk)):
                w_start = cur_time
                word_counter += 1
                w_end = end_time if word_counter == total_words else start_time + (word_counter * word_dur)
                cur_time = w_end

                start_ts = _sec_to_ass_time(w_start)
                end_ts = _sec_to_ass_time(w_end)
                dur_ms = max(20, round((w_end - w_start) * 1000))

                line_text = self._render_animated_tokens(
                    upper_chunk,
                    active_idx=idx,
                    dur_ms=dur_ms,
                    primary_inline_bgr=primary_inline_bgr,
                    animation_override=narration_anim,
                )
                events.append(f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{line_text}")

        return events

    def _generate_ass(
        self,
        words: list[WordTimestamp],
        clip_start_offset: float,
        ass_path: Path,
        chat_burst: list[dict[str, Any]] | None = None,
        narration_intro: str = "",
        narration_outro: str = "",
        intro_dur: float | None = None,
        outro_dur: float | None = None,
        clip_duration: float | None = None,
        breakdown_callout: str = "",
        breakdown_window: tuple[float | None, float | None] | None = None,
        layout: LayoutGeometry | None = None,
        callout: str = "",
        callout_window: tuple[float, float] | None = None,
        narration_beats_windows: list[tuple[float, float, str]] | None = None,
    ) -> Path:
        """Write an .ass subtitle file honoring captions.style, optional ChatTicker events, and narration windows."""
        target_path = Path(ass_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        bgr_raw = self.highlight_bgr.strip().replace("&", "")
        if bgr_raw.startswith("H") or bgr_raw.startswith("h"):
            bgr_raw = bgr_raw[1:]
        if len(bgr_raw) == 8:
            secondary_style_color = f"&H{bgr_raw.upper()}"
        elif len(bgr_raw) == 6:
            secondary_style_color = f"&H00{bgr_raw.upper()}"
        else:
            secondary_style_color = "&H0000E6FF"
        hormozi_margin_v = self.margin_v
        ticker_align, ticker_margin_v = 9, 260
        caption_center: tuple[int, int] | None = None
        callout_style = "Hormozi"
        callout_style_line = ""
        if layout is not None:
            # Captions/cards sit in the upper part of caption_band, ticker in the lower part;
            # nothing is placed over game_box.
            cb = layout.caption_band
            hormozi_margin_v = max(0, self.height - int(cb.y + 0.38 * cb.h))
            ticker_margin_v = int(cb.y + 0.72 * cb.h)
            if layout.kind == "blur_fill":
                # Word captions / narration cards centre vertically in the caption band.
                caption_center = (self.width // 2, int(cb.y + 0.5 * cb.h))
                hormozi_margin_v = max(0, self.height - int(cb.y + 0.5 * cb.h))
                ticker_margin_v = int(cb.y + 0.74 * cb.h)
            callout_style = "Callout"
            callout_style_line = (
                f"Style: Callout,{self.font},{max(1, int(self.font_size * 0.8))},"
                f"{self.primary_color},{secondary_style_color},{self.stroke_color},{self.shadow_color},"
                f"-1,0,0,0,100,100,0,0,1,{self.stroke_width},{self.shadow_offset},8,60,60,{int(cb.y + 8)},1"
            )
        # Note: For BorderStyle=3 (opaque box) on ChatTicker, OutlineColour controls the box fill!
        if self.box_style:
            hormozi_border_style = 3
            hormozi_outline = self.box_padding
            hormozi_shadow = 0
            hormozi_outline_col = self.box_color
            hormozi_back_col = self.box_color
        else:
            hormozi_border_style = 1
            hormozi_outline = self.stroke_width
            hormozi_shadow = self.shadow_offset
            hormozi_outline_col = self.stroke_color
            hormozi_back_col = self.shadow_color

        if self.style == "karaoke":
            style_primary_col = secondary_style_color
            style_secondary_col = self.primary_color
        else:
            style_primary_col = self.primary_color
            style_secondary_col = secondary_style_color

        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {self.width}
PlayResY: {self.height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hormozi,{self.font},{self.font_size},{style_primary_col},{style_secondary_col},{hormozi_outline_col},{hormozi_back_col},-1,0,0,0,100,100,0,0,{hormozi_border_style},{hormozi_outline},{hormozi_shadow},2,60,60,{hormozi_margin_v},1
Style: ChatTicker,{self._ticker_font},34,&H00FFFFFF,&H00FFFFFF,&H99000000,&H99000000,-1,0,0,0,100,100,0,0,3,2,0,{ticker_align},40,40,{ticker_margin_v},1
{callout_style_line}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events: list[str] = []
        primary_inline_bgr = "&H" + self.primary_color[4:].rstrip("&") + "&"

        # Determine narration exclusion windows to prevent muffled streamer words during ducked speech
        intro_cutoff = (
            float(intro_dur)
            if (intro_dur is not None and float(intro_dur) > 0.05)
            else 0.0
        )
        outro_start_cutoff = (
            max(intro_cutoff, float(clip_duration - outro_dur))
            if (
                outro_dur is not None
                and float(outro_dur) > 0.05
                and clip_duration is not None
                and float(clip_duration) > float(outro_dur)
            )
            else float("inf")
        )

        # Emit tone-matched chunked narration intro cards during ducked intro window
        if narration_intro and intro_cutoff > 0.0:
            events.extend(
                self._generate_narration_chunk_events(
                    narration_intro, 0.0, intro_cutoff, primary_inline_bgr
                )
            )

        # Filter streamer words: suppress speech during ducked intro & outro windows and narration beat windows
        cleaned_words: list[WordTimestamp] = []
        for w in words:
            if isinstance(w, dict):
                text_val = str(w.get("text") or w.get("word") or "").strip()
                s_val = float(w.get("start", 0.0))
                e_val = float(w.get("end", 0.0))
                w_obj = WordTimestamp(
                    text=text_val,
                    start=s_val,
                    end=e_val,
                    confidence=float(w.get("confidence", 0.0)),
                )
            else:
                w_obj = w
                text_val = str(w_obj.text or "").strip()
                s_val = float(w_obj.start)

            if not text_val:
                continue
            if intro_cutoff > 0.0 and (s_val - clip_start_offset) < intro_cutoff:
                continue
            if (s_val - clip_start_offset) >= outro_start_cutoff:
                continue
            if (
                breakdown_window
                and breakdown_window[0] is not None
                and breakdown_window[1] is not None
                and float(breakdown_window[0])
                <= (s_val - clip_start_offset)
                < float(breakdown_window[1])
            ):
                continue
            if narration_beats_windows:
                w_start_rel = s_val - clip_start_offset
                w_end_rel = float(w_obj.end) - clip_start_offset
                if any(
                    not (w_end_rel <= b_start or w_start_rel >= b_end)
                    for b_start, b_end, _ in narration_beats_windows
                ):
                    continue
            cleaned_words.append(w_obj)

        if narration_beats_windows:
            for b_start, b_end, b_text in narration_beats_windows:
                if b_text and float(b_end) > float(b_start):
                    events.extend(
                        self._generate_narration_chunk_events(
                            b_text, float(b_start), float(b_end), primary_inline_bgr
                        )
                    )

        effective_group_size = (
            1 if self.style == "single_word" else max(1, self.words_per_group)
        )
        groups = [
            cleaned_words[i : i + effective_group_size]
            for i in range(0, len(cleaned_words), effective_group_size)
        ]

        primary_inline_bgr = "&H" + self.primary_color[4:].rstrip("&") + "&"
        for group in groups:
            upper_words = [_escape_ass_text(w.text.strip().upper()) for w in group]
            if self.style == "sentence":
                start_sec = max(0.0, group[0].start - clip_start_offset)
                end_sec = max(start_sec, group[-1].end - clip_start_offset)
                start_ts = _sec_to_ass_time(start_sec)
                end_ts = _sec_to_ass_time(end_sec)
                text_content = " ".join(upper_words)
                t_dur = max(20, round((end_sec - start_sec) * 1000))
                scale_pct = round(self.animation_scale * 100)
                if self.animation == "bounce":
                    t1 = min(round(self.animation_duration_ms * 0.5), max(10, t_dur // 3))
                    t2 = min(self.animation_duration_ms, max(t1 + 10, (2 * t_dur) // 3))
                    t2 = min(t2, t_dur)
                    if t1 >= t2:
                        t1 = max(5, t_dur // 2)
                        t2 = max(t1 + 1, t_dur)
                    anim_tag = (
                        rf"{{\fscx100\fscy100"
                        rf"\t(0,{t1},\fscx{scale_pct}\fscy{scale_pct})"
                        rf"\t({t1},{t2},\fscx100\fscy100)}}"
                    )
                    events.append(
                        f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{anim_tag}{text_content}"
                    )
                elif self.animation == "pop":
                    t_pop = min(self.animation_duration_ms, max(10, t_dur // 2))
                    t_pop = min(t_pop, t_dur)
                    anim_tag = rf"{{\fscx{scale_pct}\fscy{scale_pct}\t(0,{t_pop},\fscx100\fscy100)}}"
                    events.append(
                        f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{anim_tag}{text_content}"
                    )
                else:
                    events.append(
                        f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{text_content}"
                    )
                continue

            if self.style == "karaoke":
                start_sec = max(0.0, group[0].start - clip_start_offset)
                raw_max_end = max(float(w.end) for w in group)
                parts: list[str] = []
                total_cs = 0
                for i, w in enumerate(group):
                    raw_text = _escape_ass_text(w.text.strip().upper())
                    w_text = raw_text + "\u200b" if raw_text.endswith("\\") else raw_text
                    dur_val = w.end - w.start
                    dur_cs = max(1, round(dur_val * 100)) if math.isfinite(dur_val) else 1
                    if i > 0:
                        prev_end = group[i - 1].end
                        gap_val = w.start - prev_end
                        gap_sec = max(0.0, gap_val) if math.isfinite(gap_val) else 0.0
                        gap_cs = round(gap_sec * 100)
                        if gap_cs > 0:
                            parts.append(rf"{{\k{gap_cs}}} ")
                            total_cs += gap_cs
                        else:
                            parts.append(" ")
                    parts.append(rf"{{\kf{dur_cs}}}{w_text}")
                    total_cs += dur_cs
                end_sec = max(
                    start_sec,
                    raw_max_end - clip_start_offset,
                    start_sec + (total_cs / 100.0),
                )
                start_ts = _sec_to_ass_time(start_sec)
                end_ts = _sec_to_ass_time(end_sec)
                events.append(
                    f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{''.join(parts)}"
                )
                continue

            for idx, active_word in enumerate(group):
                start_sec = max(0.0, active_word.start - clip_start_offset)
                end_sec = max(start_sec, active_word.end - clip_start_offset)
                start_ts = _sec_to_ass_time(start_sec)
                end_ts = _sec_to_ass_time(end_sec)
                dur_ms = max(20, round((end_sec - start_sec) * 1000))

                line_text = self._render_animated_tokens(
                    upper_words,
                    active_idx=idx,
                    dur_ms=dur_ms,
                    primary_inline_bgr=primary_inline_bgr,
                )
                events.append(
                    f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{line_text}"
                )

        # Emit on-screen breakdown callout while the breakdown commentary plays
        if breakdown_callout and str(breakdown_callout).strip() and breakdown_window:
            b_start, b_end = breakdown_window
            if b_start is not None and b_end is not None and float(b_end) > float(b_start):
                safe_callout = _escape_ass_text(str(breakdown_callout).strip().upper())
                events.append(
                    f"Dialogue: 0,{_sec_to_ass_time(float(b_start))},{_sec_to_ass_time(float(b_end))},"
                    f"Hormozi,,0,0,0,,{{\\c{self.highlight_bgr}}}{safe_callout}"
                )

        # Short punchy callout card voiced over the moment
        if callout and str(callout).strip() and callout_window:
            c_start, c_end = callout_window
            if c_start is not None and c_end is not None and float(c_end) > float(c_start):
                safe_c = _escape_ass_text(str(callout).strip().upper())
                events.append(
                    f"Dialogue: 0,{_sec_to_ass_time(float(c_start))},{_sec_to_ass_time(float(c_end))},"
                    f"{callout_style},,0,0,0,,{{\\c{self.highlight_bgr}}}{safe_c}"
                )

        # Emit tone-matched chunked narration outro cards during ducked outro window
        if (
            narration_outro
            and outro_start_cutoff < float("inf")
            and clip_duration is not None
            and float(clip_duration) > outro_start_cutoff
        ):
            events.extend(
                self._generate_narration_chunk_events(
                    narration_outro, outro_start_cutoff, float(clip_duration), primary_inline_bgr
                )
            )

        if self.show_chat_ticker and chat_burst:
            emitted = 0
            for msg in chat_burst:
                if emitted >= self.MAX_CHAT_TICKER_MESSAGES:
                    break
                if not isinstance(msg, dict):
                    continue
                # Accept both "message" (native chat_velocity schema) and "text"
                # (fallback schema also accepted by HighlightScorer._extract_chat_burst).
                raw_msg_val = msg.get("message")
                if raw_msg_val is None or not str(raw_msg_val).strip():
                    raw_msg_val = msg.get("text")
                raw_message = str(raw_msg_val or "").strip()
                if not raw_message:
                    continue

                # Match timestamp field variations handled by composer.remap_timeline_for_beats
                raw_offset = msg.get("offset_seconds")
                if raw_offset is None:
                    raw_offset = msg.get("timestamp")
                if raw_offset is None:
                    raw_offset = msg.get("time")
                if raw_offset is None:
                    raw_offset = msg.get("offset")
                try:
                    offset_val = (
                        float(raw_offset)
                        if raw_offset is not None and not isinstance(raw_offset, bool)
                        else float(clip_start_offset)
                    )
                    if not math.isfinite(offset_val):
                        offset_val = float(clip_start_offset)
                except (TypeError, ValueError):
                    offset_val = float(clip_start_offset)

                msg_offset = max(0.0, offset_val - clip_start_offset)
                msg_start = _sec_to_ass_time(msg_offset)
                msg_end = _sec_to_ass_time(msg_offset + self.CHAT_TICKER_DURATION_SECONDS)

                raw_author = str(msg.get("author") or "").strip() or "viewer"
                author = _escape_ass_text(raw_author[:20])
                text = _escape_ass_text(raw_message[:60])
                events.append(
                    f"Dialogue: 1,{msg_start},{msg_end},ChatTicker,,0,0,0,,{author}: {text}"
                )
                emitted += 1

        if caption_center is not None:
            marker = ",Hormozi,,0,0,0,,"
            pos = "{\\an5\\pos(%d,%d)}" % caption_center
            events = [e.replace(marker, marker + pos, 1) if marker in e else e for e in events]

        body = "\n".join(events) + "\n" if events else ""
        target_path.write_text(header + body, encoding="utf-8")
        return target_path

    def _build_ffmpeg_cmd(
        self,
        input_video: Path,
        ass_filename: str,
        output_video: Path,
        codec: str,
        crf: int,
        preset: str | None = None,
        bitrate: str | None = None,
        maxrate: str | None = None,
        bufsize: str | None = None,
        pix_fmt: str | None = None,
    ) -> list[str]:
        """Build FFmpeg command with codec-appropriate quality flags."""
        normalized_codec = str(codec or "libx264").strip().lower()
        if normalized_codec == "h264_nvenc":
            eff_preset = preset or "p4"
            v_args = ["-c:v", "h264_nvenc", "-preset", eff_preset, "-cq", str(crf)]
            if bitrate:
                v_args.extend(["-rc:v", "vbr", "-b:v", str(bitrate)])
                if maxrate:
                    v_args.extend(["-maxrate", str(maxrate)])
                if bufsize:
                    v_args.extend(["-bufsize", str(bufsize)])
                v_args.extend(["-spatial-aq", "1", "-temporal-aq", "1"])
            if pix_fmt:
                v_args.extend(["-pix_fmt", str(pix_fmt)])
        else:
            v_args = ["-c:v", "libx264", "-crf", str(crf)]
            if preset and preset not in ("p1", "p2", "p3", "p4", "p5", "p6", "p7"):
                v_args.extend(["-preset", str(preset)])
            if pix_fmt:
                v_args.extend(["-pix_fmt", str(pix_fmt)])

        escaped_ass = _escape_ffmpeg_ass_filename(ass_filename)
        return [
            "ffmpeg",
            "-y",
            "-i",
            str(Path(input_video).resolve()),
            "-vf",
            f"ass={escaped_ass}",
            *v_args,
            "-c:a",
            "copy",
            str(Path(output_video).resolve()),
        ]

    def burn_captions(
        self,
        input_video: Path,
        ass_path: Path,
        output_video: Path,
        crf: int = 19,
        codec: str | None = None,
        preset: str | None = None,
        bitrate: str | None = None,
        maxrate: str | None = None,
        bufsize: str | None = None,
        pix_fmt: str | None = None,
    ) -> Path:
        """Burn .ass captions into video in a single Windows-safe FFmpeg pass.

        Uses cwd=ass_path.parent with relative ass filename to avoid Windows drive-letter
        and filtergraph escaping issues. Falls back from h264_nvenc to libx264 on error.
        """
        input_path = Path(input_video)
        ass_file = Path(ass_path)
        output_path = Path(output_video)
        if input_path.resolve() == output_path.resolve():
            raise ValueError(
                "input_video and output_video must resolve to different paths."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selected_codec = str(codec or self.codec).strip().lower()
        selected_preset = preset or self.config.get("video.preset")
        if selected_codec != "h264_nvenc" and selected_preset in ("p1", "p2", "p3", "p4", "p5", "p6", "p7"):
            selected_preset = self.config.get("video.libx264_preset", "medium")
        selected_bitrate = bitrate or self.config.get("video.bitrate")
        selected_maxrate = maxrate or self.config.get("video.maxrate")
        selected_bufsize = bufsize or self.config.get("video.bufsize")
        selected_pix_fmt = pix_fmt or self.config.get("video.pix_fmt", "yuv420p")

        work_dir = str(ass_file.resolve().parent)
        cmd = self._build_ffmpeg_cmd(
            input_path,
            ass_file.name,
            output_path,
            selected_codec,
            crf,
            preset=selected_preset,
            bitrate=selected_bitrate,
            maxrate=selected_maxrate,
            bufsize=selected_bufsize,
            pix_fmt=selected_pix_fmt,
        )
        try:
            subprocess.run(
                cmd,
                cwd=work_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr_raw = exc.stderr
            if isinstance(stderr_raw, bytes):
                stderr_raw = stderr_raw.decode("utf-8", errors="replace")
            logger.warning(
                "FFmpeg caption burn-in failed (codec=%s): %s",
                selected_codec,
                (stderr_raw or "")[-500:],
            )
            if selected_codec != "h264_nvenc":
                raise
            logger.warning(
                "FFmpeg h264_nvenc caption burn-in failed; falling back to libx264."
            )
            fallback_cmd = self._build_ffmpeg_cmd(
                input_path,
                ass_file.name,
                output_path,
                "libx264",
                crf,
                preset=self.config.get("video.libx264_preset", "medium"),
                pix_fmt=selected_pix_fmt,
            )
            subprocess.run(
                fallback_cmd,
                cwd=work_dir,
                check=True,
                capture_output=True,
            )
        return output_path
