"""Long-form 16:9 topic video assembly: blocks, narration mixing, chapters, thumbnail, and ASS burn-in."""
from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from streamclipper.analysis.copyright_risk import CopyrightRiskDetector
from streamclipper.assembly.captions import (
    _escape_ass_text,
    _escape_ffmpeg_ass_filename,
    _sec_to_ass_time,
)
from streamclipper.assembly.narrator import TransformativeNarrator
from streamclipper.cache import StepCache
from streamclipper.config import Config
from streamclipper.download.audio import _extract_youtube_id_from_url
from streamclipper.download.video import download_segment
from streamclipper.output.exporter import _close_clip_handles
from streamclipper.output.topic_post import write_long_post
from streamclipper.topics.models import EvidenceClip, TopicPlan, TopicPoint
from streamclipper.topics.planner import plan_dir, save_plan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for audio / probe / synthetic audio
# ---------------------------------------------------------------------------

def _probe_duration(media_path: Path) -> float:
    """Probe duration in seconds of a video or audio file using ffprobe.
    Prefers video stream duration if a video stream is present, else format duration.
    """
    try:
        cmd_v = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(Path(media_path).resolve()),
        ]
        res_v = subprocess.run(cmd_v, capture_output=True, text=True)
        if res_v.returncode == 0 and res_v.stdout.strip():
            dur_v = float(res_v.stdout.strip())
            if math.isfinite(dur_v) and dur_v > 0:
                return dur_v

        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(Path(media_path).resolve()),
        ]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        dur = float(res.stdout.strip())
        return dur if math.isfinite(dur) and dur > 0 else 0.0
    except Exception as exc:
        logger.debug("ffprobe failed on %s: %s; trying soundfile or fallback", media_path, exc)
        try:
            import soundfile as sf
            info = sf.info(str(media_path))
            return float(info.duration)
        except Exception:
            return 0.0


def _ensure_synthetic_wav(wav_path: Path, duration_seconds: float = 3.0) -> Path:
    """Generate a clean synthetic WAV file if TTS did not produce one (e.g. in test runs)."""
    wav_path = Path(wav_path)
    if wav_path.is_file() and wav_path.stat().st_size > 44:
        return wav_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 48000
    total_samples = max(int(sample_rate * duration_seconds), sample_rate)
    # Generate 440 Hz gentle sine wave
    t = np.linspace(0, duration_seconds, total_samples, endpoint=False)
    waveform = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(waveform.tobytes())
    return wav_path


def _format_chapter_time(seconds: float) -> str:
    """Format seconds into YouTube chapter timestamp (mm:ss or h:mm:ss)."""
    val = max(0.0, float(seconds))
    total_s = int(round(val))
    hrs, rem_s = divmod(total_s, 3600)
    mins, secs = divmod(rem_s, 60)
    if hrs > 0:
        return f"{hrs}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _attach_bgm_to_audio(
    base_audio: Any,
    duration: float,
    bgm_file: Path | str | None,
    bgm_volume_db: float,
    opened_clips: list[Any],
) -> Any:
    """Attach background music bed under voiceover audio, ducked by -4 dB.

    Returns base_audio unchanged if bgm_file is None or does not exist on disk.
    Loops via AudioLoop(duration=duration) if shorter than duration,
    or subclips to (0, duration) if longer.
    """
    if bgm_file is None:
        return base_audio

    bgm_path = Path(bgm_file)
    if not bgm_path.is_file() or bgm_path.stat().st_size == 0:
        return base_audio

    dur = float(duration)
    if dur <= 0:
        return base_audio

    try:
        from moviepy import AudioFileClip, CompositeAudioClip
        from moviepy.audio.fx import AudioLoop

        bgm_clip = AudioFileClip(str(bgm_path))
        opened_clips.append(bgm_clip)

        bgm_dur = float(bgm_clip.duration or 0.0)
        if bgm_dur > 0 and bgm_dur < dur:
            bgm_audio = bgm_clip.with_effects([AudioLoop(duration=dur)])
            opened_clips.append(bgm_audio)
        elif bgm_dur > dur:
            bgm_audio = bgm_clip.subclipped(0, dur)
            opened_clips.append(bgm_audio)
        else:
            bgm_audio = bgm_clip

        ducked_vol = 10.0 ** ((float(bgm_volume_db) - 4.0) / 20.0)
        bgm_audio = bgm_audio.with_volume_scaled(ducked_vol)
        opened_clips.append(bgm_audio)

        composite = CompositeAudioClip([base_audio, bgm_audio])
        opened_clips.append(composite)
        return composite
    except Exception as exc:
        logger.warning("Failed loading BGM bed '%s' for narration audio: %s", bgm_file, exc)
        return base_audio


# ---------------------------------------------------------------------------
# Visual processing: Dark blurred chapter card & Ken-Burns Zoom
# ---------------------------------------------------------------------------

def _render_chapter_card_image(
    source_clip_path: Path,
    heading_text: str,
    target_w: int = 1920,
    target_h: int = 1080,
) -> Image.Image:
    """Extract a frame from source clip, apply Gaussian blur + darken, and draw heading."""
    frame_img: Image.Image | None = None
    try:
        from moviepy import VideoFileClip
        with VideoFileClip(str(source_clip_path)) as vclip:
            dur = float(vclip.duration or 1.0)
            t = max(0.0, min(dur / 2.0, max(0.0, dur - 0.05)))
            raw_frame = vclip.get_frame(t)
            frame_img = Image.fromarray(raw_frame)
    except Exception as exc:
        logger.debug("Failed to extract frame via MoviePy: %s; using blank dark frame", exc)

    if frame_img is None:
        frame_img = Image.new("RGB", (target_w, target_h), (20, 20, 25))
    else:
        frame_img = frame_img.resize((target_w, target_h), Image.Resampling.BILINEAR)

    # Apply blur
    blurred = frame_img.filter(ImageFilter.GaussianBlur(radius=18))

    # Darken overlay
    dark_overlay = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 160))
    blurred = blurred.convert("RGBA")
    blurred.alpha_composite(dark_overlay)
    base_img = blurred.convert("RGB")

    # Draw centered heading
    draw = ImageDraw.Draw(base_img)
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except Exception:
        font = ImageFont.load_default()

    clean_text = heading_text.strip()
    # Word wrap heading
    words = clean_text.split()
    lines: list[str] = []
    current_line: list[str] = []
    for w in words:
        if len(" ".join(current_line + [w])) > 30 and current_line:
            lines.append(" ".join(current_line))
            current_line = [w]
        else:
            current_line.append(w)
    if current_line:
        lines.append(" ".join(current_line))

    # Center text vertically & horizontally
    line_h = 76
    total_text_h = len(lines) * line_h
    start_y = max(20, (target_h - total_text_h) // 2)

    for idx, line in enumerate(lines):
        y = start_y + idx * line_h
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = max(20, (target_w - tw) // 2)
        # Text with dark stroke
        draw.text((x, y), line, font=font, fill=(255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0))

    return base_img


def _apply_ken_burns_zoom(
    clip: Any,
    max_zoom: float = 1.08,
    blur_radius: int = 0,
    dim_factor: float = 1.0,
) -> Any:
    """Apply smooth center Ken-Burns zoom over clip duration, with optional blur and dimming."""
    dur = float(clip.duration or 1.0)
    w, h = clip.size
    if (max_zoom <= 1.001 and blur_radius <= 0 and dim_factor >= 0.999) or dur <= 0.0:
        return clip

    def transform_frame(get_frame: Any, t: float) -> np.ndarray:
        progress = min(1.0, max(0.0, t / dur))
        zoom = 1.0 + (max_zoom - 1.0) * progress
        frame = get_frame(t)
        if zoom > 1.001:
            crop_w = max(1, min(w, int(w / zoom)))
            crop_h = max(1, min(h, int(h / zoom)))
            x1 = (w - crop_w) // 2
            y1 = (h - crop_h) // 2
            cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
            pil_img = Image.fromarray(cropped).resize((w, h), Image.Resampling.BILINEAR)
        else:
            pil_img = Image.fromarray(frame)

        if blur_radius > 0:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        if dim_factor < 0.999:
            from PIL import ImageEnhance
            pil_img = ImageEnhance.Brightness(pil_img).enhance(dim_factor)
        return np.array(pil_img, dtype=np.uint8)

    return clip.transform(transform_frame)


def _fit_to_canvas(clip: Any, target_w: int = 1920, target_h: int = 1080) -> Any:
    """Resize/crop clip to exact 16:9 canvas dimensions."""
    w, h = clip.size
    if (w, h) == (target_w, target_h):
        return clip
    scale = max(target_w / w, target_h / h)
    nw = round(w * scale)
    nh = round(h * scale)
    scaled = clip.resized((nw, nh))
    if nw > target_w or nh > target_h:
        x1 = (nw - target_w) // 2
        y1 = (nh - target_h) // 2
        return scaled.cropped(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)
    return scaled


def _resolve_local_clip_path(clip: EvidenceClip, temp_base: Path) -> Path | None:
    """Resolve an EvidenceClip to an already-downloaded local file path if present.

    Checks:
    1. Direct local file path in ``clip.url`` (or ``file://`` URI).
    2. Pre-downloaded B-roll files in ``<temp_base>/broll/`` matching the clip range or source_id.
    """
    raw_url = str(getattr(clip, "url", "") or "").strip()
    if raw_url and not raw_url.lower().startswith(("http://", "https://")):
        cand_str = raw_url[7:] if raw_url.lower().startswith("file://") else raw_url
        cand_path = Path(cand_str)
        if cand_path.is_file() and cand_path.stat().st_size > 0:
            return cand_path

    broll_dir = Path(temp_base) / "broll"
    if broll_dir.is_dir():
        vid = str(getattr(clip, "source_id", "") or "").strip()
        if not vid and raw_url:
            vid = _extract_youtube_id_from_url(raw_url) or ""
        safe_vid = re.sub(r"[^A-Za-z0-9_-]", "_", vid) if vid else ""
        if safe_vid:
            start_ms = round(max(0.0, float(getattr(clip, "start", 0.0) or 0.0)) * 1000)
            end_ms = round(max(0.0, float(getattr(clip, "end", 0.0) or 0.0)) * 1000)
            video_exts = (".mp4", ".webm", ".mkv", ".mov")
            for ext in video_exts:
                exact = broll_dir / f"{safe_vid}_{start_ms}_{end_ms}_b0{ext}"
                if exact.is_file() and exact.stat().st_size > 0:
                    return exact
            for cand in sorted(broll_dir.glob(f"{safe_vid}_*")):
                if cand.is_file() and cand.suffix.lower() in video_exts and cand.stat().st_size > 0:
                    return cand

    return None


def _build_broll_montage(
    candidate_paths: list[Path | str],
    target_duration: float,
    target_w: int,
    target_h: int,
    kenburns_max_zoom: float,
    opened_clips: list[Any],
    *,
    slice_min: float = 7.0,
    slice_max: float = 10.0,
    blur_radius: int = 0,
    dim_factor: float = 1.0,
) -> Any:
    """Build a multi-clip non-looping B-roll montage covering target_duration.

    Cycles round-robin across all distinct valid clips in ``candidate_paths``, taking
    7–10s slices per clip (advancing each clip's cursor so repeated visits show fresh
    segments) so the visual changes regularly and never loops the same clip back-to-back.
    """
    from moviepy import AudioClip, ColorClip, VideoFileClip, concatenate_videoclips

    target_dur = max(0.1, float(target_duration))
    s_min = max(1.0, float(slice_min))
    s_max = max(s_min, float(slice_max))
    base_slice = (s_min + s_max) / 2.0  # 8.5s default

    # Deduplicate valid candidate files while preserving order
    unique_paths: list[Path] = []
    seen_resolved: set[str] = set()
    for raw_p in candidate_paths or []:
        if not raw_p:
            continue
        p = Path(raw_p)
        if not p.is_file() or p.stat().st_size == 0:
            continue
        key = str(p.resolve()).lower()
        if key not in seen_resolved:
            seen_resolved.add(key)
            unique_paths.append(p)

    # Open each unique clip once
    sources: list[dict[str, Any]] = []
    for p in unique_paths:
        try:
            vclip = VideoFileClip(str(p))
            opened_clips.append(vclip)
            dur = float(vclip.duration or 0.0)
            if math.isfinite(dur) and dur >= 0.2:
                sources.append({"clip": vclip, "dur": dur, "cursor": 0.0, "path": p})
        except Exception as exc:
            logger.warning("Failed opening B-roll clip '%s': %s", p, exc)

    if not sources:
        fallback = ColorClip(size=(target_w, target_h), color=(20, 20, 30), duration=target_dur)
        opened_clips.append(fallback)
        return fallback

    slices: list[Any] = []
    accumulated = 0.0
    src_idx = 0
    num_sources = len(sources)
    has_any_audio = any(s["clip"].audio is not None for s in sources)

    while accumulated < target_dur - 0.02:
        remaining = target_dur - accumulated
        src = sources[src_idx]
        src_idx = (src_idx + 1) % num_sources
        dur = src["dur"]

        # Reset cursor if remaining tail in this clip is too tiny
        avail = dur - src["cursor"]
        if avail < min(2.0, dur * 0.5) or avail <= 0.15:
            src["cursor"] = 0.0
            avail = dur

        if remaining <= s_max:
            desired = remaining
        elif (remaining - base_slice) < 3.0:
            desired = min(max(s_min * 0.5, remaining / 2.0), s_max)
        else:
            desired = base_slice

        take = min(desired, avail, remaining)
        if take <= 0.05:
            src["cursor"] = 0.0
            take = min(desired, dur, remaining)

        sub_start = max(0.0, min(src["cursor"], max(0.0, dur - 0.05)))
        sub_end = min(dur, sub_start + max(0.05, take))
        actual_take = sub_end - sub_start
        if actual_take <= 0.02:
            break

        src["cursor"] = sub_end
        if src["cursor"] >= dur - 0.1:
            src["cursor"] = 0.0

        sub = src["clip"].subclipped(sub_start, sub_end)
        opened_clips.append(sub)

        fitted = _fit_to_canvas(sub, target_w, target_h)
        if fitted is not sub:
            opened_clips.append(fitted)

        if has_any_audio and fitted.audio is None:
            silent_a = AudioClip(lambda t: [0.0, 0.0], duration=actual_take)
            opened_clips.append(silent_a)
            fitted = fitted.with_audio(silent_a)
            opened_clips.append(fitted)

        zoomed_slice = _apply_ken_burns_zoom(
            fitted,
            max_zoom=kenburns_max_zoom,
            blur_radius=blur_radius,
            dim_factor=dim_factor,
        )
        if zoomed_slice is not fitted:
            opened_clips.append(zoomed_slice)

        slices.append(zoomed_slice)
        accumulated += actual_take

        # Guard against infinite loops on degenerate 0-duration clips
        if len(slices) > 200:
            break

    if not slices:
        fallback = ColorClip(size=(target_w, target_h), color=(20, 20, 30), duration=target_dur)
        opened_clips.append(fallback)
        return fallback

    if len(slices) == 1:
        montage = slices[0]
    else:
        montage = concatenate_videoclips(slices, method="compose")
        opened_clips.append(montage)

    if (montage.duration or 0.0) > target_dur:
        montage = montage.subclipped(0.0, target_dur)
        opened_clips.append(montage)

    return montage


# ---------------------------------------------------------------------------
# Thumbnail Generator (1280x720)
# ---------------------------------------------------------------------------

def _score_thumbnail_frame(frame: np.ndarray) -> float:
    """Score a frame by Laplacian sharpness and brightness balance (penalize near-black/washed-out)."""
    try:
        if frame.ndim == 3:
            if frame.shape[2] >= 3:
                gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
            else:
                gray = frame[:, :, 0]
        elif frame.ndim == 2:
            gray = frame
        else:
            return 0.0

        # Laplacian sharpness variance: cv2.Laplacian(..., cv2.CV_64F).var()
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Brightness balance: penalize near-black (< 35) or washed-out (> 220) frames
        mean_lum = float(np.mean(gray))
        dist = abs(mean_lum - 128.0) / 128.0
        brightness_factor = max(0.01, 1.0 - (dist ** 2))
        if mean_lum < 35.0:
            brightness_factor *= max(0.01, mean_lum / 35.0)
        elif mean_lum > 220.0:
            brightness_factor *= max(0.01, (255.0 - mean_lum) / 35.0)

        return lap_var * brightness_factor + (brightness_factor * 1e-4)
    except Exception as exc:
        logger.debug("Frame scoring failed: %s", exc)
        return 0.0


def generate_topic_thumbnail(
    frame_clip_path: Path,
    thumbnail_text: str,
    output_path: Path,
    target_w: int = 1280,
    target_h: int = 720,
) -> Path:
    """Generate 1280x720 topic video thumbnail with frame-scored selection, pop enhancements, and safe zone text."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_ratios = [0.20, 0.35, 0.50, 0.65, 0.80]
    candidate_frames: list[np.ndarray] = []

    # 1. Sample 5 evenly spaced frames (20%, 35%, 50%, 65%, 80%) across the clip
    try:
        from moviepy import VideoFileClip
        with VideoFileClip(str(frame_clip_path)) as vclip:
            dur = float(vclip.duration or 1.0)
            for r in sample_ratios:
                t = max(0.01, min(dur * r, max(0.01, dur - 0.05)))
                raw_frame = vclip.get_frame(t)
                if raw_frame is not None:
                    candidate_frames.append(raw_frame)
    except Exception as exc:
        logger.debug("Failed to extract frames for thumbnail via MoviePy: %s; trying cv2/FFmpeg", exc)

    if not candidate_frames:
        try:
            cap = cv2.VideoCapture(str(frame_clip_path.resolve()))
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if total_frames > 0:
                    for r in sample_ratios:
                        target_f = max(0, min(int(total_frames * r), total_frames - 1))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                        ret, bgr = cap.read()
                        if ret and bgr is not None:
                            candidate_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                cap.release()
        except Exception:
            pass

    if not candidate_frames:
        temp_thumb = output_path.with_name(f"{output_path.stem}_raw.jpg")
        try:
            dur = _probe_duration(frame_clip_path) or 1.0
            for r in sample_ratios:
                t_seek = max(0.01, min(dur * r, max(0.01, dur - 0.05)))
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{t_seek:.2f}",
                    "-i", str(frame_clip_path.resolve()),
                    "-vframes", "1",
                    "-f", "image2",
                    str(temp_thumb.resolve()),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                if temp_thumb.is_file():
                    img = Image.open(temp_thumb).convert("RGB")
                    candidate_frames.append(np.array(img))
                    temp_thumb.unlink(missing_ok=True)
        except Exception:
            pass

    # Score each frame by Laplacian sharpness + brightness balance and pick the best
    if candidate_frames:
        best_frame = max(candidate_frames, key=_score_thumbnail_frame)
        base_img = Image.fromarray(best_frame)
    else:
        base_img = Image.new("RGB", (target_w, target_h), (15, 20, 30))

    base_img = base_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 2. Apply subtle Pillow ImageEnhance.Contrast(1.18) and ImageEnhance.Color(1.22) so frame pops at mobile size
    base_img = ImageEnhance.Contrast(base_img).enhance(1.18)
    base_img = ImageEnhance.Color(base_img).enhance(1.22)

    # Gradient: subtle darkening across the top safe zone to make text pop
    base_rgba = base_img.convert("RGBA")
    gradient = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(gradient)
    for y_coord in range(target_h):
        factor = max(0.0, 1.0 - (y_coord / (target_h * 0.55)))
        alpha = int(30 + 130 * factor)
        g_draw.line([(0, y_coord), (target_w, y_coord)], fill=(0, 0, 0, alpha))
    base_rgba.alpha_composite(gradient)
    final_img = base_rgba.convert("RGB")

    # 3. Cap thumbnail text to 3 words max, render in bold yellow with thick black stroke + subtle drop shadow
    draw = ImageDraw.Draw(final_img)
    words = thumbnail_text.strip().split()
    if len(words) > 3:
        words = words[:3]
    display_text = " ".join(words).upper()

    # Dynamic font scaling to guarantee text fits within target_w - 140
    max_text_w = target_w - 140
    font_size = 92
    font = None
    while font_size >= 36:
        try:
            candidate_font = ImageFont.truetype("impact.ttf", font_size)
        except Exception:
            try:
                candidate_font = ImageFont.truetype("arialbd.ttf", font_size)
            except Exception:
                candidate_font = ImageFont.load_default()
                font = candidate_font
                break
        bbox = draw.textbbox((0, 0), display_text, font=candidate_font)
        tw = bbox[2] - bbox[0]
        if tw <= max_text_w:
            font = candidate_font
            break
        font_size -= 6

    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), display_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Upper-center safe zone (never bottom-right where YouTube puts the duration badge)
    x = max(60, (target_w - tw) // 2)
    y = 60
    stroke_w = max(4, font_size // 11)
    shadow_offset = max(4, font_size // 14)

    if display_text:
        # Subtle drop shadow
        draw.text(
            (x + shadow_offset, y + shadow_offset),
            display_text,
            font=font,
            fill=(0, 0, 0),
            stroke_width=stroke_w + 1,
            stroke_fill=(0, 0, 0),
        )

        # Bold yellow text with thick black stroke
        draw.text(
            (x, y),
            display_text,
            font=font,
            fill=(255, 235, 59),
            stroke_width=stroke_w,
            stroke_fill=(0, 0, 0),
        )

    final_img.save(str(output_path), "JPEG", quality=92)
    logger.info("Saved topic thumbnail to %s (1280x720)", output_path)
    return output_path


# ---------------------------------------------------------------------------
# ASS Caption & Lower-Third Credit File Builder
# ---------------------------------------------------------------------------

def _chunk_narration_text(text: str, duration: float) -> list[tuple[float, float, str]]:
    """Break narration text into 2-line max chunks distributed over duration."""
    clean = text.strip()
    if not clean or duration <= 0:
        return []
    words = clean.split()
    if len(words) <= 7:
        return [(0.0, duration, clean)]

    # Group into ~8-12 words per chunk
    chunks: list[str] = []
    curr: list[str] = []
    for w in words:
        curr.append(w)
        if len(curr) >= 10 and w.endswith((".", "!", "?", ",", ";", ":")):
            chunks.append(" ".join(curr))
            curr = []
        elif len(curr) >= 13:
            chunks.append(" ".join(curr))
            curr = []
    if curr:
        chunks.append(" ".join(curr))

    num = max(1, len(chunks))
    chunk_dur = duration / num
    results: list[tuple[float, float, str]] = []
    for i, chk in enumerate(chunks):
        st = i * chunk_dur
        et = min(duration, (i + 1) * chunk_dur)
        # Format into max 2 lines with \N if needed
        chk_words = chk.split()
        if len(chk_words) > 7:
            half = len(chk_words) // 2
            formatted_text = " ".join(chk_words[:half]) + r"\N" + " ".join(chk_words[half:])
        else:
            formatted_text = chk
        results.append((st, et, formatted_text))
    return results


def _build_longform_ass(
    narration_windows: list[tuple[float, float, str]],
    credit_windows: list[tuple[float, float, str]],
    output_ass: Path,
    font_name: str = "Arial",
) -> Path:
    """Generate ASS subtitle file with 16:9 Narration and Lower-Third Credit styles."""
    output_ass = Path(output_ass)
    output_ass.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "[Script Info]\n"
        "Title: StreamClipper Topic Video\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Narration16x9,{font_name},42,&H00FFFFFF,&H000000FF,&H99000000,&H99000000,1,0,0,0,100,100,0,0,3,3,0,2,80,80,75,1\n"
        f"Style: Credit16x9,{font_name},28,&H00FFFFFF,&H000000FF,&H99000000,&H99000000,0,0,0,0,100,100,0,0,3,2,0,1,60,60,60,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []

    # 1. Narration events
    for win_start, win_end, text in narration_windows:
        win_dur = max(0.1, win_end - win_start)
        chunks = _chunk_narration_text(text, win_dur)
        for c_start, c_end, c_text in chunks:
            abs_start = win_start + c_start
            abs_end = min(win_end, win_start + c_end)
            if abs_end <= abs_start:
                continue
            escaped = r"\N".join(_escape_ass_text(part) for part in c_text.split(r"\N"))
            events.append(
                f"Dialogue: 0,{_sec_to_ass_time(abs_start)},{_sec_to_ass_time(abs_end)},"
                f"Narration16x9,,0,0,0,,{escaped}"
            )

    # 2. Lower-third credit events
    for cred_start, cred_end, cred_text in credit_windows:
        if cred_end <= cred_start:
            continue
        escaped_cred = _escape_ass_text(cred_text)
        events.append(
            f"Dialogue: 0,{_sec_to_ass_time(cred_start)},{_sec_to_ass_time(cred_end)},"
            f"Credit16x9,,0,0,0,,{escaped_cred}"
        )

    # Sort events by start time
    events.sort(key=lambda line: line.split(",")[1])
    body = "\n".join(events) + "\n"
    output_ass.write_text(header + body, encoding="utf-8")
    return output_ass


# ---------------------------------------------------------------------------
# Core Renderer: render_long_video
# ---------------------------------------------------------------------------

def render_long_video(config: Config, plan: TopicPlan, *, progress: Any = None, draft: bool = False) -> dict[str, Any]:
    """Render a 16:9 long topic video from an approved TopicPlan.

    Performs:
    1. Status checks (refuses unapproved plans; sets rendering/rendered/failed).
    2. Downloads evidence segments (buffer_seconds=0.0) into <temp_dir>/topics/<plan_id>/.
    3. Copyright pre-flight checks using source cached audio; swaps alternates or drops clips.
    4. Cap enforcement (<= 20s, official <= 10s).
    5. Block assembly (cards for numbered/countdown; Ken-Burns ducked B-roll narration; full-volume evidence).
    6. Per-block loudnorm via TransformativeNarrator.normalize_audio.
    7. Concat (with 0.4s crossfade between chapters in story mode) + single final lossy encode
       with loudnorm (-14 LUFS) and ASS caption/credit burn.
    8. Chapters file (>= 3 chapters, first 00:00, each >= 10s), thumbnail (1280x720), and post file.
    9. Updates plan.outputs and saves plan.
    """
    if plan.status not in ("approved", "rendered"):
        raise ValueError(
            f"Topic plan '{plan.plan_id}' must be 'approved' to render (current: '{plan.status}')."
        )

    render_config = config
    if draft:
        render_config = copy.copy(config)
        render_config._data = copy.deepcopy(config._data)
        render_config._data.setdefault("transformation", {})["tts_engine"] = "edge-tts"
        render_config._data.setdefault("transformation", {})["hinglish_tts_engines"] = ["edge-tts"]
    config = render_config

    orig_status = plan.status
    if not draft:
        plan.status = "rendering"
        save_plan(config, plan)

    out_dir = plan_dir(config, plan.game, plan.topic_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_base = Path(config.get("general.temp_dir", "./temp"))
    temp_dir = temp_base / "topics" / plan.plan_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Enforce caps and run Copyright Risk Detector on evidence clips
        detector = CopyrightRiskDetector(config)
        step_cache = StepCache(config)

        for pt in plan.points:
            valid_evidence: list[EvidenceClip] = []
            used_alts: set[tuple[str, float, float]] = set()

            for clip in pt.evidence:
                local_clip_path = _resolve_local_clip_path(clip, temp_base)
                clip.start = max(0.0, float(clip.start))
                if local_clip_path is not None and float(clip.end) <= clip.start:
                    probed_dur = _probe_duration(local_clip_path) or 10.0
                    clip.end = clip.start + probed_dur
                else:
                    clip.end = max(clip.start + 0.1, float(clip.end))
                max_s = 10.0 if clip.kind == "official" else 20.0
                if (clip.end - clip.start) > max_s:
                    clip.end = clip.start + max_s

                # Copyright check
                risk = "UNKNOWN"
                try:
                    state = step_cache.load_state(clip.url)
                    if state is not None and getattr(state, "audio_path", None) and Path(state.audio_path).is_file():
                        risk, _ = detector.detect(state.audio_path, clip.start, clip.end)
                    else:
                        logger.warning("No cached stream audio found for %s; skipping copyright check", clip.url)
                except Exception as exc:
                    logger.warning("Copyright detector error for %s: %s; skipping check", clip.url, exc)

                if risk == "HIGH_MUSIC_DETECTED":
                    logger.warning(
                        "HIGH_MUSIC_DETECTED on clip %s (%.1f-%.1f); checking alternates",
                        clip.source_id, clip.start, clip.end,
                    )
                    swapped = False
                    for alt in pt.alternates:
                        alt_key = (alt.url, round(alt.start, 2), round(alt.end, 2))
                        if alt_key in used_alts:
                            continue
                        alt.start = max(0.0, float(alt.start))
                        alt.end = max(alt.start + 0.1, float(alt.end))
                        alt_max = 10.0 if alt.kind == "official" else 20.0
                        if (alt.end - alt.start) > alt_max:
                            alt.end = alt.start + alt_max
                        alt_risk = "UNKNOWN"
                        try:
                            alt_state = step_cache.load_state(alt.url)
                            if alt_state is not None and getattr(alt_state, "audio_path", None) and Path(alt_state.audio_path).is_file():
                                alt_risk, _ = detector.detect(alt_state.audio_path, alt.start, alt.end)
                        except Exception:
                            pass
                        if alt_risk != "HIGH_MUSIC_DETECTED":
                            valid_evidence.append(alt)
                            used_alts.add(alt_key)
                            swapped = True
                            logger.info("Swapped alternate clip %s for point %d", alt.source_id, pt.index)
                            break
                    if not swapped:
                        logger.warning("All alternates failed for point %d; dropping clip", pt.index)
                else:
                    valid_evidence.append(clip)

            if not valid_evidence:
                raise RuntimeError(
                    f"Point {pt.index} ('{pt.heading}') has no valid evidence clips after copyright filtering."
                )
            pt.evidence = valid_evidence

        # 2. Download segments into <temp_dir>/topics/<plan_id>/ (reusing local/broll clips)
        all_clips: list[EvidenceClip] = []
        for pt in plan.points:
            all_clips.extend(pt.evidence)

        downloaded_files: dict[tuple[str, float, float], Path] = {}
        total_clips = len(all_clips)
        for idx, clip in enumerate(all_clips, start=1):
            if progress and hasattr(progress, "stage"):
                progress.stage(f"downloading {idx}/{total_clips}")
            clip_key = (clip.url, round(clip.start, 2), round(clip.end, 2))
            if clip_key not in downloaded_files:
                local_p = _resolve_local_clip_path(clip, temp_base)
                if local_p is not None:
                    downloaded_files[clip_key] = local_p
                else:
                    path = download_segment(
                        clip.url,
                        clip.start,
                        clip.end,
                        temp_dir,
                        buffer_seconds=0.0,
                        video_id=clip.source_id or None,
                    )
                    downloaded_files[clip_key] = path

        # Resolve or download alternate / B-roll clips for multi-clip narration montages
        for pt in plan.points:
            for alt in getattr(pt, "alternates", []) or []:
                local_alt = _resolve_local_clip_path(alt, temp_base)
                alt.start = max(0.0, float(alt.start))
                if local_alt is not None and float(alt.end) <= alt.start:
                    alt.end = alt.start + (_probe_duration(local_alt) or 10.0)
                else:
                    alt.end = max(alt.start + 0.1, float(alt.end))
                alt_max = 10.0 if alt.kind == "official" else 20.0
                if (alt.end - alt.start) > alt_max:
                    alt.end = alt.start + alt_max
                alt_key = (alt.url, round(alt.start, 2), round(alt.end, 2))
                if alt_key in downloaded_files:
                    continue
                if local_alt is not None:
                    downloaded_files[alt_key] = local_alt
                elif alt.url and alt.url.strip().lower().startswith(("http://", "https://")):
                    try:
                        alt_path = download_segment(
                            alt.url,
                            alt.start,
                            alt.end,
                            temp_dir,
                            buffer_seconds=0.0,
                            video_id=alt.source_id or None,
                        )
                        downloaded_files[alt_key] = alt_path
                    except Exception as exc:
                        logger.warning(
                            "Optional alternate clip download failed for point %d (%s): %s",
                            pt.index,
                            alt.url,
                            exc,
                        )

        # 3. Voice narration blocks
        narrator = TransformativeNarrator(config)
        voiced_narrations: dict[str, tuple[Path, float]] = {}

        # Cold open
        if plan.cold_open.strip():
            co_wav = temp_dir / "narration_cold_open.wav"
            res_co = narrator.synthesize_speech(plan.cold_open, co_wav, language=plan.language)
            final_co_wav = res_co or co_wav
            if not final_co_wav.is_file() or final_co_wav.stat().st_size == 0:
                raise RuntimeError("TTS synthesis failed for cold_open")
            co_dur = _probe_duration(final_co_wav) or (len(plan.cold_open.split()) / 2.7)
            voiced_narrations["cold_open"] = (final_co_wav, co_dur)

        # Points
        for pt in plan.points:
            if progress and hasattr(progress, "stage"):
                progress.stage(f"voicing point {pt.index}/{len(plan.points)}")
            if pt.narration.strip():
                pt_wav = temp_dir / f"narration_point_{pt.index}.wav"
                res_pt = narrator.synthesize_speech(pt.narration, pt_wav, language=plan.language)
                final_pt_wav = res_pt or pt_wav
                if not final_pt_wav.is_file() or final_pt_wav.stat().st_size == 0:
                    raise RuntimeError(f"TTS synthesis failed for point {pt.index}")
                pt_dur = _probe_duration(final_pt_wav) or (len(pt.narration.split()) / 2.7)
                voiced_narrations[f"point_{pt.index}"] = (final_pt_wav, pt_dur)

        # Outro
        if plan.outro.strip():
            out_wav = temp_dir / "narration_outro.wav"
            res_out = narrator.synthesize_speech(plan.outro, out_wav, language=plan.language)
            final_out_wav = res_out or out_wav
            if not final_out_wav.is_file() or final_out_wav.stat().st_size == 0:
                raise RuntimeError("TTS synthesis failed for outro")
            out_dur = _probe_duration(final_out_wav) or (len(plan.outro.split()) / 2.7)
            voiced_narrations["outro"] = (final_out_wav, out_dur)

        # 4. Build intermediate video blocks
        from moviepy import (
            AudioClip,
            AudioFileClip,
            ColorClip,
            CompositeAudioClip,
            CompositeVideoClip,
            ImageClip,
            VideoFileClip,
            afx,
            concatenate_videoclips,
        )
        from moviepy.video.fx import CrossFadeIn, CrossFadeOut
        import moviepy.video.fx as vfx

        target_w = int(config.get("longform.width", 1920) or 1920)
        target_h = int(config.get("longform.height", 1080) or 1080)
        fps = int(config.get("longform.fps", 30) or 30)
        card_seconds = float(config.get("longform.card_seconds", 1.2) or 1.2)
        transition_sfx_path = str(config.get("longform.transition_sfx_path", "") or "")
        broll_volume = float(config.get("longform.broll_volume", 0.12) or 0.12)
        credit_seconds = float(config.get("longform.credit_seconds", 4.0) or 4.0)
        kenburns_max_zoom = float(config.get("longform.kenburns_max_zoom", 1.08) or 1.08)

        # Resolve BGM track once near start of block assembly
        bgm_file = narrator._resolve_bgm_file() if hasattr(narrator, "_resolve_bgm_file") else None
        bgm_volume_db = float(
            getattr(narrator, "bgm_volume_db", None)
            if getattr(narrator, "bgm_volume_db", None) is not None
            else (config.get("transformation.bgm_volume_db", -26.0) or -26.0)
        )

        # Helper to render a block to ultrafast MP4 and apply per-block loudnorm
        def render_block_mp4(
            clip: Any,
            name: str,
            clips_to_close: list[Any] | None = None,
            normalize: bool = True,
        ) -> Path:
            raw_out = temp_dir / f"{name}_raw.mp4"
            final_out = temp_dir / f"{name}.mp4"
            try:
                clip.write_videofile(
                    str(raw_out),
                    fps=fps,
                    codec="libx264",
                    audio_codec="aac",
                    preset="ultrafast",
                    ffmpeg_params=["-crf", "18"],
                    temp_audiofile_path=str(temp_dir),
                    logger=None,
                )
            finally:
                _close_clip_handles(clip)
                if clips_to_close:
                    for c in clips_to_close:
                        _close_clip_handles(c)

            if normalize:
                # Per-block loudnorm using narrator helper
                try:
                    narrator.normalize_audio(raw_out, final_out)
                    raw_out.unlink(missing_ok=True)
                except Exception as exc:
                    logger.warning("Per-block audio normalization fallback for %s: %s", name, exc)
                    if raw_out.is_file():
                        shutil.move(str(raw_out), str(final_out))
            else:
                # Non-normalized block (e.g. chapter card): ensure 48kHz stereo AAC stream format
                # without destroying the configured -12dB SFX or -26dB BGM bed level via loudnorm
                try:
                    cmd = [
                        "ffmpeg", "-y", "-i", str(raw_out),
                        "-map", "0:v", "-map", "0:a:0",
                        "-af", "aresample=48000,aformat=channel_layouts=stereo",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        str(final_out),
                    ]
                    subprocess.run(
                        cmd, check=True, capture_output=True, text=True,
                        encoding="utf-8", errors="replace",
                    )
                    raw_out.unlink(missing_ok=True)
                except Exception as exc:
                    logger.warning("Audio remux fallback for non-normalized %s: %s", name, exc)
                    if raw_out.is_file():
                        shutil.move(str(raw_out), str(final_out))

            return final_out

        # Find best B-roll (highest heat evidence clip)
        highest_heat_clip: EvidenceClip | None = None
        max_heat: float = -1.0
        for pt in plan.points:
            for clip in pt.evidence:
                heat_val = float(getattr(clip, "heat", 0.0) or 0.0)
                if heat_val > max_heat:
                    max_heat = heat_val
                    highest_heat_clip = clip

        if highest_heat_clip is None and plan.points and plan.points[0].evidence:
            highest_heat_clip = plan.points[0].evidence[0]

        best_broll_clip_path = (
            downloaded_files.get((highest_heat_clip.url, round(highest_heat_clip.start, 2), round(highest_heat_clip.end, 2)))
            if highest_heat_clip else None
        )
        fallback_clip_path = best_broll_clip_path or (next(iter(downloaded_files.values())) if downloaded_files else None)

        # Structure chapters: each chapter contains a list of block MP4 paths and timeline metadata
        # A chapter corresponds to:
        # Chapter 0 (optional): Cold open
        # Chapter 1..N: Each TopicPoint (card if numbered + narration + evidence clips)
        # Chapter N+1 (optional): Outro
        class ChapterAssembly:
            def __init__(self, name: str, heading: str) -> None:
                self.name = name
                self.heading = heading
                self.block_files: list[Path] = []
                self.narration_events: list[tuple[float, float, str]] = []  # (rel_start, rel_end, text)
                self.credit_events: list[tuple[float, float, str]] = []     # (rel_start, rel_end, text)
                self.duration: float = 0.0

        chapters_assembled: list[ChapterAssembly] = []
        block_counter: int = 0
        total_narration_dur: float = 0.0

        # --- Chapter: Cold Open ---
        if "cold_open" in voiced_narrations:
            block_counter += 1
            if progress and hasattr(progress, "stage"):
                progress.stage(f"rendering block {block_counter}")
            wav_path, nar_dur = voiced_narrations["cold_open"]
            chap_co = ChapterAssembly("cold_open", "Intro")

            opened_co: list[Any] = []
            co_candidates: list[Path | str] = []
            if fallback_clip_path:
                co_candidates.append(fallback_clip_path)
            for p_cand in downloaded_files.values():
                if p_cand and p_cand not in co_candidates:
                    co_candidates.append(p_cand)

            zoomed = _build_broll_montage(
                co_candidates,
                nar_dur,
                target_w,
                target_h,
                kenburns_max_zoom,
                opened_co,
            )

            nar_audio = AudioFileClip(str(wav_path))
            opened_co.append(nar_audio)
            if zoomed.audio is not None:
                ducked_broll_audio = zoomed.audio.with_volume_scaled(broll_volume)
                opened_co.append(ducked_broll_audio)
                mixed_audio = CompositeAudioClip([ducked_broll_audio, nar_audio])
            else:
                mixed_audio = nar_audio

            mixed_audio = _attach_bgm_to_audio(mixed_audio, nar_dur, bgm_file, bgm_volume_db, opened_co)

            zoomed = zoomed.with_audio(mixed_audio).with_duration(nar_dur)
            block_file = render_block_mp4(zoomed, f"block_{block_counter:02d}_cold_open", clips_to_close=opened_co)
            actual_dur = _probe_duration(block_file) or nar_dur

            chap_co.block_files.append(block_file)
            chap_co.narration_events.append((0.0, actual_dur, plan.cold_open))
            chap_co.duration = actual_dur
            chapters_assembled.append(chap_co)
            total_narration_dur += actual_dur

        # --- Chapters: Points ---
        for pt in plan.points:
            chap_pt = ChapterAssembly(f"pt_{pt.index}", pt.heading)
            rel_t = 0.0

            pt_first_clip = pt.evidence[0] if pt.evidence else None
            pt_clip_path = (
                downloaded_files.get((pt_first_clip.url, round(pt_first_clip.start, 2), round(pt_first_clip.end, 2)))
                if pt_first_clip else fallback_clip_path
            ) or fallback_clip_path

            # Chapter Card (Numbered or Countdown shapes only)
            if plan.shape != "story":
                block_counter += 1
                if progress and hasattr(progress, "stage"):
                    progress.stage(f"rendering block {block_counter}")

                card_img = _render_chapter_card_image(
                    pt_clip_path, pt.heading, target_w=target_w, target_h=target_h
                )
                card_img_clip = ImageClip(np.array(card_img)).with_duration(card_seconds)
                clips_to_close: list[Any] = [card_img_clip]

                sfx_attached = False
                sfx_file = Path(transition_sfx_path) if transition_sfx_path.strip() else None
                if sfx_file is not None and sfx_file.is_file() and sfx_file.stat().st_size > 0:
                    try:
                        sfx_clip = AudioFileClip(str(sfx_file))
                        clips_to_close.append(sfx_clip)
                        sfx_vol = 10.0 ** (-12.0 / 20.0)
                        sfx_audio = sfx_clip.with_volume_scaled(sfx_vol)
                        clips_to_close.append(sfx_audio)
                        if sfx_audio.duration and sfx_audio.duration > card_seconds:
                            sfx_audio = sfx_audio.subclipped(0, card_seconds)
                            clips_to_close.append(sfx_audio)
                        card_clip = card_img_clip.with_audio(sfx_audio)
                        sfx_attached = True
                    except Exception as sfx_exc:
                        logger.warning(
                            "Failed loading transition SFX '%s': %s; falling back to visual fade and BGM bed",
                            sfx_file,
                            sfx_exc,
                        )

                if not sfx_attached:
                    card_faded = card_img_clip.with_effects([CrossFadeIn(0.2), CrossFadeOut(0.2)])
                    card_clip = CompositeVideoClip([card_faded], size=(target_w, target_h)).with_duration(card_seconds)
                    clips_to_close.extend([card_faded, card_clip])

                    bgm_attached = False
                    if bgm_file is not None and Path(bgm_file).is_file():
                        try:
                            bgm_clip = AudioFileClip(str(bgm_file))
                            clips_to_close.append(bgm_clip)
                            bgm_vol = 10.0 ** (bgm_volume_db / 20.0)
                            bgm_audio = bgm_clip.with_volume_scaled(bgm_vol)
                            clips_to_close.append(bgm_audio)
                            if bgm_audio.duration and bgm_audio.duration < card_seconds:
                                bgm_audio = bgm_audio.with_effects([afx.AudioLoop(duration=card_seconds)])
                                clips_to_close.append(bgm_audio)
                            elif bgm_audio.duration and bgm_audio.duration > card_seconds:
                                bgm_audio = bgm_audio.subclipped(0, card_seconds)
                                clips_to_close.append(bgm_audio)
                            card_clip = card_clip.with_audio(bgm_audio)
                            bgm_attached = True
                        except Exception as bgm_exc:
                            logger.warning("Failed loading BGM bed '%s' for chapter card: %s", bgm_file, bgm_exc)

                    if not bgm_attached:
                        card_clip = card_clip.with_audio(AudioClip(lambda t: [0.0, 0.0], duration=card_seconds))

                card_file = render_block_mp4(
                    card_clip,
                    f"block_{block_counter:02d}_pt_{pt.index}_card",
                    clips_to_close=clips_to_close,
                    normalize=False,
                )
                card_actual_dur = _probe_duration(card_file) or card_seconds

                chap_pt.block_files.append(card_file)
                rel_t += card_actual_dur

            # Narration Block
            if f"point_{pt.index}" in voiced_narrations:
                block_counter += 1
                if progress and hasattr(progress, "stage"):
                    progress.stage(f"rendering block {block_counter}")
                pt_wav, pt_dur = voiced_narrations[f"point_{pt.index}"]

                opened_pt: list[Any] = []
                # Build a multi-clip non-looping B-roll montage for this point's narration.
                # Stop back-to-back duplicate footage: exclude pt.evidence[0] when other clips exist.
                # Candidate priority:
                # (a) pt.alternates (including local/downloaded B-roll clips)
                # (b) pt.evidence[1:] (if point has >= 2 evidence clips)
                # (c) distinct downloaded clips from other points (alternates + evidence)
                # (d) fallback to pt_clip_path with blur+dim only if no other clip exists
                pt_ev_keys = {(ev.url, round(ev.start, 2), round(ev.end, 2)) for ev in pt.evidence}
                pt_ev0_resolved = (
                    str(Path(pt_clip_path).resolve()).lower()
                    if pt_clip_path and Path(pt_clip_path).is_file()
                    else ""
                )

                candidate_paths: list[Path | str] = []
                seen_cand_keys: set[str] = set()

                def _add_candidate(cand: Path | None) -> None:
                    if not cand or not Path(cand).is_file():
                        return
                    r_key = str(Path(cand).resolve()).lower()
                    if r_key == pt_ev0_resolved:
                        return
                    if r_key not in seen_cand_keys:
                        seen_cand_keys.add(r_key)
                        candidate_paths.append(Path(cand))

                # (a) pt.alternates + local/broll clips
                for alt in getattr(pt, "alternates", []) or []:
                    alt_key = (alt.url, round(alt.start, 2), round(alt.end, 2))
                    if alt_key not in pt_ev_keys:
                        cand_path = downloaded_files.get(alt_key) or _resolve_local_clip_path(alt, temp_base)
                        _add_candidate(cand_path)

                # (b) pt.evidence[1:]
                for ev_extra in pt.evidence[1:]:
                    ev_key = (ev_extra.url, round(ev_extra.start, 2), round(ev_extra.end, 2))
                    _add_candidate(downloaded_files.get(ev_key))

                # (c) non-overlapping / distinct clips from other points if needed
                for other_pt in plan.points:
                    if other_pt.index == pt.index:
                        continue
                    for other_alt in getattr(other_pt, "alternates", []) or []:
                        other_alt_key = (other_alt.url, round(other_alt.start, 2), round(other_alt.end, 2))
                        _add_candidate(downloaded_files.get(other_alt_key) or _resolve_local_clip_path(other_alt, temp_base))
                    for other_ev in other_pt.evidence:
                        other_key = (other_ev.url, round(other_ev.start, 2), round(other_ev.end, 2))
                        _add_candidate(downloaded_files.get(other_key))

                apply_dim_blur = False
                if not candidate_paths:
                    apply_dim_blur = True
                    if pt_clip_path and Path(pt_clip_path).is_file():
                        candidate_paths.append(Path(pt_clip_path))

                zoomed = _build_broll_montage(
                    candidate_paths,
                    pt_dur,
                    target_w,
                    target_h,
                    kenburns_max_zoom,
                    opened_pt,
                    blur_radius=6 if apply_dim_blur else 0,
                    dim_factor=0.65 if apply_dim_blur else 1.0,
                )

                pt_audio = AudioFileClip(str(pt_wav))
                opened_pt.append(pt_audio)
                if zoomed.audio is not None:
                    ducked_broll_audio = zoomed.audio.with_volume_scaled(broll_volume)
                    opened_pt.append(ducked_broll_audio)
                    mixed_audio = CompositeAudioClip([ducked_broll_audio, pt_audio])
                else:
                    mixed_audio = pt_audio

                mixed_audio = _attach_bgm_to_audio(mixed_audio, pt_dur, bgm_file, bgm_volume_db, opened_pt)

                zoomed = zoomed.with_audio(mixed_audio).with_duration(pt_dur)
                nar_block_file = render_block_mp4(zoomed, f"block_{block_counter:02d}_pt_{pt.index}_narration", clips_to_close=opened_pt)
                nar_actual_dur = _probe_duration(nar_block_file) or pt_dur

                chap_pt.block_files.append(nar_block_file)
                chap_pt.narration_events.append((rel_t, rel_t + nar_actual_dur, pt.narration))
                total_narration_dur += nar_actual_dur
                rel_t += nar_actual_dur

            # Evidence Clips (full volume, original audio)
            for ev_idx, ev_clip in enumerate(pt.evidence, start=1):
                block_counter += 1
                if progress and hasattr(progress, "stage"):
                    progress.stage(f"rendering block {block_counter}")
                ev_path = downloaded_files.get(
                    (ev_clip.url, round(ev_clip.start, 2), round(ev_clip.end, 2))
                ) or fallback_clip_path

                opened_ev: list[Any] = []
                ev_dur = max(0.1, ev_clip.end - ev_clip.start)
                ev_video = VideoFileClip(str(ev_path)) if ev_path else ColorClip(size=(target_w, target_h), color=(10, 10, 10), duration=ev_dur)
                opened_ev.append(ev_video)
                ev_fitted = _fit_to_canvas(ev_video, target_w, target_h)
                ev_actual_dur = min(ev_dur, float(ev_fitted.duration or ev_dur))
                ev_trimmed = ev_fitted.subclipped(0.0, ev_actual_dur)

                if ev_trimmed.audio is None:
                    ev_trimmed = ev_trimmed.with_audio(AudioClip(lambda t: [0.0, 0.0], duration=ev_actual_dur))

                ev_block_file = render_block_mp4(ev_trimmed, f"block_{block_counter:02d}_pt_{pt.index}_ev_{ev_idx}", clips_to_close=opened_ev)
                actual_ev_dur = _probe_duration(ev_block_file) or ev_actual_dur
                chap_pt.block_files.append(ev_block_file)

                # Lower-third credit
                if ev_clip.kind == "official":
                    raw_ch = (ev_clip.channel or "").strip().lstrip("@")
                    cred_text = f"Official footage: {raw_ch}" if raw_ch else f"Official {plan.game} footage"
                else:
                    ch_clean = (ev_clip.channel or "").strip().lstrip("@")
                    cred_text = f"@{ch_clean}" if ch_clean else "@Unknown"

                actual_cred_dur = min(credit_seconds, actual_ev_dur)
                chap_pt.credit_events.append((rel_t, rel_t + actual_cred_dur, cred_text))
                rel_t += actual_ev_dur

            chap_pt.duration = rel_t
            chapters_assembled.append(chap_pt)

        # --- Chapter: Outro ---
        if "outro" in voiced_narrations:
            block_counter += 1
            if progress and hasattr(progress, "stage"):
                progress.stage(f"rendering block {block_counter}")
            out_wav, out_dur = voiced_narrations["outro"]
            chap_out = ChapterAssembly("outro", "Outro")

            opened_out: list[Any] = []
            out_candidates: list[Path | str] = []
            if fallback_clip_path:
                out_candidates.append(fallback_clip_path)
            for p_cand in downloaded_files.values():
                if p_cand and p_cand not in out_candidates:
                    out_candidates.append(p_cand)

            zoomed = _build_broll_montage(
                out_candidates,
                out_dur,
                target_w,
                target_h,
                kenburns_max_zoom,
                opened_out,
            )

            out_audio = AudioFileClip(str(out_wav))
            opened_out.append(out_audio)
            if zoomed.audio is not None:
                ducked_broll_audio = zoomed.audio.with_volume_scaled(broll_volume)
                opened_out.append(ducked_broll_audio)
                mixed_audio = CompositeAudioClip([ducked_broll_audio, out_audio])
            else:
                mixed_audio = out_audio

            mixed_audio = _attach_bgm_to_audio(mixed_audio, out_dur, bgm_file, bgm_volume_db, opened_out)

            zoomed = zoomed.with_audio(mixed_audio).with_duration(out_dur)
            block_file = render_block_mp4(zoomed, f"block_{block_counter:02d}_outro", clips_to_close=opened_out)
            out_actual_dur = _probe_duration(block_file) or out_dur

            chap_out.block_files.append(block_file)
            chap_out.narration_events.append((0.0, out_actual_dur, plan.outro))
            chap_out.duration = out_actual_dur
            chapters_assembled.append(chap_out)
            total_narration_dur += out_actual_dur

        # 5. Concatenate intermediate blocks into full video
        # In story shape: 0.4s crossfade marks chapters!
        # In numbered/countdown: standard concat demuxer (hard cuts).
        concat_video = temp_dir / "concatenated.mp4"
        chapters_data: list[tuple[float, str]] = []
        narration_windows: list[tuple[float, float, str]] = []
        credit_windows: list[tuple[float, float, str]] = []
        t_timeline = 0.0

        if plan.shape == "story" and len(chapters_assembled) > 1:
            # First assemble each chapter into a chapter-level video if it has multiple blocks
            chapter_video_files: list[Path] = []
            for ch_idx, chap in enumerate(chapters_assembled):
                if len(chap.block_files) == 1:
                    chapter_video_files.append(chap.block_files[0])
                else:
                    ch_concat = temp_dir / f"chapter_{ch_idx:02d}_concat.mp4"
                    ch_list = temp_dir / f"chapter_{ch_idx:02d}_list.txt"
                    ch_list.write_text("\n".join(f"file '{p.resolve()}'" for p in chap.block_files), encoding="utf-8")
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(ch_list.resolve()), "-c", "copy", str(ch_concat.resolve())],
                        check=True,
                        capture_output=True,
                    )
                    chapter_video_files.append(ch_concat)

            # Story shape: 0.4s crossfade between chapter videos
            story_clips = [VideoFileClip(str(p)) for p in chapter_video_files]
            try:
                # Clamp durations to exact video stream duration to prevent MoviePy EOF frame warnings
                clamped_clips = []
                for c in story_clips:
                    v_dur = _probe_duration(Path(c.filename))
                    safe_dur = max(0.5, v_dur - (2.0 / fps)) if v_dur > 0 else float(c.duration or 1.0)
                    clamped_clips.append(c.subclipped(0.0, safe_dur))
                cf_clips = [clamped_clips[0]] + [c.with_effects([vfx.CrossFadeIn(0.4)]) for c in clamped_clips[1:]]
                comp = concatenate_videoclips(cf_clips, method="compose", padding=-0.4)
                comp.write_videofile(
                    str(concat_video),
                    fps=fps,
                    codec="libx264",
                    audio_codec="aac",
                    preset="ultrafast",
                    ffmpeg_params=["-crf", "18"],
                    temp_audiofile_path=str(temp_dir),
                    logger=None,
                )
                _close_clip_handles(comp)
            finally:
                for c in story_clips:
                    _close_clip_handles(c)

            # Build synchronized timeline accounting for the 0.4s crossfade per chapter transition
            for ch_idx, chap in enumerate(chapters_assembled):
                if ch_idx > 0:
                    t_timeline = max(0.0, t_timeline - 0.4)
                ch_start = t_timeline
                chapters_data.append((ch_start, chap.heading))

                for st, et, txt in chap.narration_events:
                    narration_windows.append((ch_start + st, ch_start + et, txt))
                for st, et, txt in chap.credit_events:
                    credit_windows.append((ch_start + st, ch_start + et, txt))

                t_timeline += chap.duration
        else:
            # Numbered / Countdown: Concat all intermediate blocks via FFmpeg concat demuxer
            all_blocks: list[Path] = []
            for ch_idx, chap in enumerate(chapters_assembled):
                ch_start = t_timeline
                chapters_data.append((ch_start, chap.heading))

                for st, et, txt in chap.narration_events:
                    narration_windows.append((ch_start + st, ch_start + et, txt))
                for st, et, txt in chap.credit_events:
                    credit_windows.append((ch_start + st, ch_start + et, txt))

                all_blocks.extend(chap.block_files)
                t_timeline += chap.duration

            concat_list = temp_dir / "concat_list.txt"
            concat_lines = [f"file '{p.resolve()}'" for p in all_blocks]
            concat_list.write_text("\n".join(concat_lines), encoding="utf-8")
            cmd_concat = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list.resolve()),
                "-c", "copy",
                str(concat_video.resolve()),
            ]
            subprocess.run(cmd_concat, check=True, capture_output=True)

        # 6. Build Chapters list and enforce YouTube rules
        # YouTube needs:
        # - First chapter at 00:00
        # - >= 3 chapters
        # - Each chapter >= 10s apart
        # - CRITICAL: Do not drop point headings; adjust/space timestamps
        if not chapters_data:
            chapters_data = [(0.0, "Intro")]

        # Ensure first is at 0.0
        if chapters_data[0][0] > 0.0:
            chapters_data.insert(0, (0.0, "Intro"))
        else:
            chapters_data[0] = (0.0, chapters_data[0][1])

        # Space chapters by >= 10s without dropping headings
        spaced_chapters: list[tuple[float, str]] = []
        prev_t = -10.0
        for t_ch, h_ch in chapters_data:
            target_t = max(t_ch, prev_t + 10.0)
            spaced_chapters.append((target_t, h_ch))
            prev_t = target_t

        # Ensure at least 3 chapters
        total_time = _probe_duration(concat_video) or t_timeline
        while len(spaced_chapters) < 3:
            target_t = prev_t + 10.0
            spaced_chapters.append((target_t, f"Chapter {len(spaced_chapters) + 1}"))
            prev_t = target_t

        validated_chapters = [(_format_chapter_time(t), h) for t, h in spaced_chapters]

        chapters_file = out_dir / "chapters.txt"
        chapters_lines = [f"{t_str} {h_str}" for t_str, h_str in validated_chapters]
        chapters_file.write_text("\n".join(chapters_lines) + "\n", encoding="utf-8")
        logger.info("Wrote %d YouTube chapters to %s", len(validated_chapters), chapters_file)

        # 7. Generate ASS file (narration captions + credits)
        ass_path = temp_dir / "longform_burned.ass"
        captions_font = str(config.get("captions.font", "Arial") or "Arial")
        _build_longform_ass(narration_windows, credit_windows, ass_path, font_name=captions_font)

        # 8. Single final lossy encode with loudnorm (-14 LUFS) and ASS burn
        if progress and hasattr(progress, "stage"):
            progress.stage("final encode")

        video_filename = "long_draft.mp4" if draft else "long.mp4"
        final_long_mp4 = out_dir / video_filename
        codec = str(config.get("video.codec", "libx264") or "libx264").strip().lower()
        crf = int(config.get("video.crf", 23) or 23)

        def build_final_cmd(selected_codec: str) -> list[str]:
            escaped_ass = _escape_ffmpeg_ass_filename(ass_path.name)
            v_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)] if selected_codec == "h264_nvenc" else ["-c:v", "libx264", "-crf", str(crf)]
            return [
                "ffmpeg", "-y",
                "-i", str(concat_video.resolve()),
                "-vf", f"ass={escaped_ass}",
                "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000",
                *v_args,
                "-c:a", "aac",
                "-b:a", "192k",
                str(final_long_mp4.resolve()),
            ]

        cmd_final = build_final_cmd(codec)
        work_dir = str(ass_path.resolve().parent)
        try:
            subprocess.run(cmd_final, cwd=work_dir, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            if codec == "h264_nvenc":
                logger.warning("h264_nvenc final encode failed; falling back to libx264")
                fallback_cmd = build_final_cmd("libx264")
                subprocess.run(fallback_cmd, cwd=work_dir, check=True, capture_output=True)
            else:
                raise

        # 9. Generate Thumbnail (1280x720) from highest heat evidence clip's midpoint
        thumb_source_path = fallback_clip_path
        thumbnail_file = out_dir / "thumbnail.jpg"
        thumb_text = plan.thumbnail_text or plan.video_title or plan.topic_title or "TOPIC BREAKDOWN"
        generate_topic_thumbnail(thumb_source_path, thumb_text, thumbnail_file)

        # 10. Generate long_post.txt
        post_file = write_long_post(config, plan, chapters_lines)

        # 11. Final duration and narration share probes & warnings
        final_dur = _probe_duration(final_long_mp4) or total_time
        narration_share = (total_narration_dur / final_dur) if final_dur > 0 else 0.0

        if final_dur < 480.0 or final_dur > 600.0:
            logger.warning(
                "Longform video duration %.1fs is outside the recommended 480-600s range.",
                final_dur,
            )
        if narration_share < 0.40:
            logger.warning(
                "Longform narration share %.1f%% is below the recommended 40%% threshold.",
                narration_share * 100.0,
            )

        # 12. Update plan outputs and save
        out_key = "long_draft" if draft else "long"
        plan.outputs[out_key] = str(final_long_mp4.resolve())
        plan.outputs["chapters"] = str(chapters_file.resolve())
        plan.outputs["thumbnail"] = str(thumbnail_file.resolve())
        if draft:
            draft_post = out_dir / "long_draft_post.txt"
            if post_file.is_file():
                import shutil
                shutil.copy2(post_file, draft_post)
                post_file = draft_post
            plan.outputs["long_draft_post"] = str(post_file.resolve())
        else:
            plan.outputs["long_post"] = str(post_file.resolve())
            plan.status = "rendered"
        save_plan(config, plan)

        return {
            "status": "rendered",
            "plan_id": plan.plan_id,
            "video_path": str(final_long_mp4.resolve()),
            "duration": final_dur,
            "narration_share": narration_share,
            "chapters_path": str(chapters_file.resolve()),
            "thumbnail_path": str(thumbnail_file.resolve()),
            "post_path": str(post_file.resolve()),
            "outputs": plan.outputs,
            "clips": [{"video_path": str(final_long_mp4.resolve()), "plan_id": plan.plan_id}],
            "draft": draft,
        }

    except Exception:
        if not draft:
            plan.status = "failed"
            save_plan(config, plan)
        raise
