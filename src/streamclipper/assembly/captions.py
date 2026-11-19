"""Hormozi-style ASS caption generation, live chat burst ticker, and FFmpeg libass burn-in."""
from __future__ import annotations

import logging
import math
from pathlib import Path
import re
import subprocess
from typing import Any

from ..config import Config
from ..models import WordTimestamp

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
_VALID_CAPTION_STYLES = frozenset({"hormozi", "single_word", "sentence"})


def _safe_int(value: Any, default: int, *, minimum: int | None = None) -> int:
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


def _color_to_ass_style(color: str, default: str = "&H00FFFFFF") -> str:
    """Convert CSS named color or '#RRGGBB' into ASS style color '&H00BBGGRR'."""
    raw = str(color or "").strip().lower()
    if raw in _NAMED_ASS_COLORS:
        return _NAMED_ASS_COLORS[raw]
    bgr = _hex_to_ass_bgr(raw, default="")
    if bgr:
        return "&H00" + bgr[2:-1]
    return default


def _escape_ass_text(text: str) -> str:
    """Sanitize newlines and escape ASS override delimiters safely for libass."""
    sanitized = "" if text is None else str(text)
    sanitized = sanitized.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    sanitized = sanitized.replace("{", r"\{").replace("}", r"\}")
    # Neutralize ASS control codes (\N, \n, \h) without doubling literal backslashes in libass
    return _ASS_CONTROL_RE.sub("\\\\\u200b\\1", sanitized)


class AssCaptionBurner:
    """Generates Hormozi/single-word/sentence ASS subtitles and burns them via FFmpeg libass."""

    MAX_CHAT_TICKER_MESSAGES = 5
    CHAT_TICKER_DURATION_SECONDS = 2.5

    def __init__(self, config: Config) -> None:
        raw_style = str(config.get("captions.style", "hormozi") or "hormozi").strip().lower()
        if raw_style not in _VALID_CAPTION_STYLES:
            logger.warning(
                "Unknown captions.style %r; defaulting to 'hormozi'.", raw_style
            )
            raw_style = "hormozi"
        self.style = raw_style

        self.words_per_group = _safe_int(
            config.get("captions.words_per_group", 4), 4, minimum=1
        )
        raw_font = str(config.get("captions.font", "Arial") or "Arial")
        self.font = (
            raw_font.replace(",", " ").replace("\r", " ").replace("\n", " ").strip()
            or "Arial"
        )
        self.font_size = _safe_int(config.get("captions.font_size", 72), 72, minimum=1)
        self.primary_color = _color_to_ass_style(
            config.get("captions.color", "white"), default="&H00FFFFFF"
        )
        self.stroke_color = _color_to_ass_style(
            config.get("captions.stroke_color", "black"), default="&H00000000"
        )
        stroke_val = _safe_float(
            config.get("captions.stroke_width", 4), 4.0, minimum=0.0
        )
        self.stroke_width: int | float = (
            int(stroke_val) if stroke_val.is_integer() else stroke_val
        )

        raw_highlight = str(config.get("captions.highlight_color", "#FFD700") or "#FFD700")
        highlight_bgr = _hex_to_ass_bgr(raw_highlight, default="")
        if not highlight_bgr:
            named_style = _NAMED_ASS_COLORS.get(raw_highlight.strip().lower())
            highlight_bgr = f"&H{named_style[4:]}&" if named_style else "&H00D7FF&"
        self.highlight_bgr = highlight_bgr

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
        words: list[WordTimestamp],
        clip_start_offset: float,
        ass_path: Path,
        chat_burst: list[dict[str, Any]] | None = None,
        narration_intro: str = "",
        narration_outro: str = "",
        intro_dur: float | None = None,
        outro_dur: float | None = None,
        clip_duration: float | None = None,
    ) -> Path:
        """Write an .ass subtitle file honoring captions.style, optional ChatTicker events, and narration windows."""
        target_path = Path(ass_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        secondary_style_color = "&H00" + self.highlight_bgr[2:-1]
        # Note: For BorderStyle=3 (opaque box) on ChatTicker, OutlineColour controls the box fill!
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {self.width}
PlayResY: {self.height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hormozi,{self.font},{self.font_size},{self.primary_color},{secondary_style_color},{self.stroke_color},&H80000000,-1,0,0,0,100,100,0,0,1,{self.stroke_width},0,2,60,60,{self.margin_v},1
Style: ChatTicker,{self.font},34,&H00FFFFFF,&H00FFFFFF,&H99000000,&H99000000,-1,0,0,0,100,100,0,0,3,2,0,9,40,40,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events: list[str] = []

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

        # Emit tone-matched narration intro card during ducked intro window
        if narration_intro and intro_cutoff > 0.0:
            intro_start_ts = _sec_to_ass_time(0.0)
            intro_end_ts = _sec_to_ass_time(intro_cutoff)
            safe_intro = _escape_ass_text(str(narration_intro).strip().upper())
            events.append(
                f"Dialogue: 0,{intro_start_ts},{intro_end_ts},Hormozi,,0,0,0,,{{\\c{self.highlight_bgr}}}{safe_intro}"
            )

        # Filter streamer words: suppress speech during ducked intro & outro windows
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
            cleaned_words.append(w_obj)

        effective_group_size = (
            1 if self.style == "single_word" else max(1, self.words_per_group)
        )
        groups = [
            cleaned_words[i : i + effective_group_size]
            for i in range(0, len(cleaned_words), effective_group_size)
        ]

        primary_inline_bgr = "&H" + self.primary_color[4:] + "&"
        for group in groups:
            upper_words = [_escape_ass_text(w.text.strip().upper()) for w in group]
            if self.style == "sentence":
                start_sec = max(0.0, group[0].start - clip_start_offset)
                end_sec = max(start_sec, group[-1].end - clip_start_offset)
                start_ts = _sec_to_ass_time(start_sec)
                end_ts = _sec_to_ass_time(end_sec)
                events.append(
                    f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{' '.join(upper_words)}"
                )
                continue

            for idx, active_word in enumerate(group):
                start_sec = max(0.0, active_word.start - clip_start_offset)
                end_sec = max(start_sec, active_word.end - clip_start_offset)
                start_ts = _sec_to_ass_time(start_sec)
                end_ts = _sec_to_ass_time(end_sec)
                rendered_tokens: list[str] = []
                for j, token in enumerate(upper_words):
                    if j == idx:
                        safe_token = token + "\u200b" if token.endswith("\\") else token
                        rendered_tokens.append(
                            rf"{{\c{self.highlight_bgr}}}"
                            + safe_token
                            + rf"{{\c{primary_inline_bgr}}}"
                        )
                    else:
                        rendered_tokens.append(token)
                line_text = " ".join(rendered_tokens)
                events.append(
                    f"Dialogue: 0,{start_ts},{end_ts},Hormozi,,0,0,0,,{line_text}"
                )

        # Emit tone-matched narration outro card during ducked outro window
        if (
            narration_outro
            and outro_start_cutoff < float("inf")
            and clip_duration is not None
        ):
            outro_start_ts = _sec_to_ass_time(outro_start_cutoff)
            outro_end_ts = _sec_to_ass_time(clip_duration)
            safe_outro = _escape_ass_text(str(narration_outro).strip().upper())
            events.append(
                f"Dialogue: 0,{outro_start_ts},{outro_end_ts},Hormozi,,0,0,0,,{{\\c{self.highlight_bgr}}}{safe_outro}"
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
    ) -> list[str]:
        """Build FFmpeg command with codec-appropriate quality flags."""
        normalized_codec = str(codec or "libx264").strip().lower()
        if normalized_codec == "h264_nvenc":
            v_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)]
        else:
            v_args = ["-c:v", "libx264", "-crf", str(crf)]

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
        crf: int = 23,
        codec: str | None = None,
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
        work_dir = str(ass_file.resolve().parent)
        cmd = self._build_ffmpeg_cmd(
            input_path, ass_file.name, output_path, selected_codec, crf
        )
        try:
            subprocess.run(
                cmd,
                cwd=work_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            if selected_codec != "h264_nvenc":
                raise
            logger.warning(
                "FFmpeg h264_nvenc caption burn-in failed; falling back to libx264."
            )
            fallback_cmd = self._build_ffmpeg_cmd(
                input_path, ass_file.name, output_path, "libx264", crf
            )
            subprocess.run(
                fallback_cmd,
                cwd=work_dir,
                check=True,
                capture_output=True,
            )
        return output_path
