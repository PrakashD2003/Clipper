"""Post-render video reviewer: automated technical QA, silence detection, and multimodal visual contact sheet."""
from __future__ import annotations

import io
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def _parse_fps(fps_str: str) -> float:
    """Parse FFprobe r_frame_rate or avg_frame_rate string (e.g. '30/1', '29.97/1') to float."""
    if not fps_str or fps_str == "0/0":
        return 0.0
    if "/" in fps_str:
        parts = fps_str.split("/", 1)
        try:
            den = float(parts[1])
            return round(float(parts[0]) / den, 2) if den != 0.0 else 0.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return round(float(fps_str), 2)
    except ValueError:
        return 0.0


def _format_timestamp_label(seconds: float) -> str:
    """Format seconds into MM:SS label."""
    total = max(0, int(seconds))
    mins = total // 60
    secs = total % 60
    return f"{mins:02d}:{secs:02d}"


def _stamp_timestamp_badge(
    img: Image.Image,
    seconds: float,
    *,
    text_color: tuple[int, int, int] = (255, 255, 0),  # Bright yellow
) -> Image.Image:
    """Stamp an exact MM:SS timestamp in bright yellow with a dark background box."""
    draw = ImageDraw.Draw(img)
    label = _format_timestamp_label(seconds)
    try:
        font = ImageFont.load_default(size=18)
    except (TypeError, AttributeError):
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((10, 8), label, font=font)
        draw.rectangle(
            [bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2],
            fill=(0, 0, 0),
        )
    except Exception:
        draw.rectangle([6, 6, 80, 28], fill=(0, 0, 0))
    draw.text((10, 8), label, fill=text_color, font=font)
    return img


def _grab_frame_at_timestamp(
    video_path: Path,
    timestamp: float,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    """Extract a single frame from video at timestamp using OpenCV or FFmpeg fallback."""
    # 1. Try OpenCV first
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path.resolve()))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp) * 1000.0))
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
                return im.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    except Exception as exc:
        logger.debug("OpenCV frame grab failed at %.2fs: %s", timestamp, exc)

    # 2. FFmpeg fallback (piped directly to stdout, eliminating temp file contention on Windows)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path.resolve()),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0 and res.stdout:
            with Image.open(io.BytesIO(res.stdout)) as im:
                im_rgb = im.convert("RGB")
                return im_rgb.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    except Exception as exc:
        logger.debug("FFmpeg frame grab failed at %.2fs: %s", timestamp, exc)

    # 3. Blank fallback frame
    return Image.new("RGB", (tile_width, tile_height), color=(20, 20, 20))


def generate_review_contact_sheet(
    video_path: Path,
    duration: float,
    width: int,
    height: int,
    output_path: Path | None = None,
) -> Path:
    """Generate a labeled 3x4 (12 tiles) visual review contact sheet with yellow MM:SS stamps.

    Saves to `<video_stem>_review_sheet.jpg` beside the video by default.
    """
    sheet_path = output_path or video_path.parent / f"{video_path.stem}_review_sheet.jpg"
    rows = 3
    cols = 4
    num_tiles = 12

    # Determine tile aspect ratio from video resolution
    aspect = (height / width) if (width > 0 and height > 0) else (9.0 / 16.0)
    if aspect > 1.0:
        # Vertical video (e.g. 9:16 Shorts)
        tile_width = 270
        tile_height = max(16, int(round(tile_width * aspect)))
    else:
        # Horizontal video (e.g. 16:9 Long-Form)
        tile_width = 480
        tile_height = max(16, int(round(tile_width * aspect)))

    sheet_width = tile_width * cols
    sheet_height = tile_height * rows

    # Evenly space 12 frame timestamps across video duration
    dur = max(0.1, duration)
    timestamps = [round((i + 0.5) * (dur / num_tiles), 2) for i in range(num_tiles)]

    canvas = Image.new("RGB", (sheet_width, sheet_height), color=(0, 0, 0))

    for idx, ts in enumerate(timestamps):
        tile_img = _grab_frame_at_timestamp(video_path, ts, tile_width, tile_height)
        _stamp_timestamp_badge(tile_img, ts, text_color=(255, 255, 0))

        r = idx // cols
        c = idx % cols
        x = c * tile_width
        y = r * tile_height
        canvas.paste(tile_img, (x, y))

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(sheet_path, "JPEG", quality=90)
    return sheet_path


def _load_narration_timeline(video_path: Path, metadata_path: Path | None = None) -> list[dict[str, Any]]:
    """Load companion metadata (<stem>.json or plan.json) and format a structured narration timeline."""
    target_meta = metadata_path
    if target_meta is None or not Path(target_meta).is_file():
        # Check beside video
        direct_json = video_path.with_suffix(".json")
        if direct_json.is_file():
            target_meta = direct_json
        elif video_path.stem.endswith("_draft"):
            non_draft_json = video_path.with_name(f"{video_path.stem[:-6]}.json")
            if non_draft_json.is_file():
                target_meta = non_draft_json

    if target_meta is None or not Path(target_meta).is_file():
        # Check for plan.json in parent dir
        parent_plan = video_path.parent / "plan.json"
        if parent_plan.is_file():
            target_meta = parent_plan

    if target_meta is None or not Path(target_meta).is_file():
        return []

    try:
        data = json.loads(Path(target_meta).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Failed reading metadata JSON %s: %s", target_meta, exc)
        return []

    timeline: list[dict[str, Any]] = []

    # Check if TopicPlan JSON
    if "points" in data and isinstance(data["points"], list):
        # Long-form or Topic Short
        is_short = video_path.stem.startswith("short_")
        if is_short and "shorts" in data and isinstance(data["shorts"], list):
            # Locate matching short plan
            for s in data["shorts"]:
                slot = str(s.get("slot", "")).strip().lower()
                if slot and (slot in video_path.stem.lower() or f"short_{slot}" in video_path.stem.lower()):
                    if s.get("narration_intro"):
                        timeline.append({"role": "intro", "text": s["narration_intro"], "start": 0.0})
                    if s.get("callout"):
                        timeline.append({"role": "callout", "text": s["callout"], "start": None})
                    if s.get("narration_outro"):
                        timeline.append({"role": "outro", "text": s["narration_outro"], "start": None})
                    break
        else:
            # Long-form Topic Plan
            if data.get("cold_open"):
                timeline.append({"role": "cold_open", "text": data["cold_open"], "start": 0.0})

            # Check if chapters.txt exists for exact timestamps
            chapters_file = video_path.parent / "chapters.txt"
            chapter_timestamps: dict[str, float] = {}
            if chapters_file.is_file():
                for line in chapters_file.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        t_str, h_str = parts
                        # Parse mm:ss or hh:mm:ss
                        subparts = [float(p) for p in t_str.split(":")]
                        if len(subparts) == 2:
                            sec = subparts[0] * 60 + subparts[1]
                        elif len(subparts) == 3:
                            sec = subparts[0] * 3600 + subparts[1] * 60 + subparts[2]
                        else:
                            sec = 0.0
                        chapter_timestamps[h_str.lower().strip()] = sec

            for pt in data.get("points", []):
                heading = pt.get("heading", "")
                t_sec = chapter_timestamps.get(heading.lower().strip())
                timeline.append({
                    "role": f"point_{pt.get('index', 0)}",
                    "heading": heading,
                    "text": pt.get("narration", ""),
                    "start": t_sec,
                })

            if data.get("outro"):
                t_outro = chapter_timestamps.get("outro")
                timeline.append({"role": "outro", "text": data["outro"], "start": t_outro})

        if timeline:
            return timeline

    # ClipMetadata (Short / Single Highlight)
    narration_beats = data.get("narration_beats")
    if narration_beats and isinstance(narration_beats, list):
        for b in narration_beats:
            if isinstance(b, dict):
                at_sec = b.get("at_second")
                if at_sec is None:
                    at_sec = b.get("start", 0.0)
                dur = b.get("duration")
                end_sec = round(float(at_sec) + float(dur), 2) if dur is not None else None
                timeline.append({
                    "role": b.get("role") or b.get("kind") or "beat",
                    "text": b.get("text", ""),
                    "start": float(at_sec),
                    "end": end_sec,
                })

    if not timeline:
        if data.get("narration_intro"):
            timeline.append({"role": "intro", "text": data["narration_intro"], "start": 0.0})
        if data.get("callout"):
            timeline.append({"role": "callout", "text": data["callout"], "start": data.get("callout_start")})
        if data.get("breakdown_commentary"):
            timeline.append({
                "role": "breakdown",
                "text": data["breakdown_commentary"],
                "start": data.get("breakdown_target_time"),
            })
        if data.get("narration_outro"):
            timeline.append({"role": "outro", "text": data["narration_outro"], "start": data.get("outro_start")})

    if not timeline and "sub_clips" in data and isinstance(data["sub_clips"], list):
        for sc in data["sub_clips"]:
            if isinstance(sc, dict):
                timeline.append({
                    "role": "sub_clip",
                    "heading": str(sc.get("label", "")),
                    "start": float(sc.get("start", 0.0)),
                    "end": float(sc.get("end", 0.0)),
                })

    return timeline


def inspect_rendered_video(video_path: Path | str, metadata_path: Path | str | None = None) -> dict[str, Any]:
    """Inspect a rendered video for technical quality, volume/silence compliance, and visual alignment.

    Probes:
    - Duration, resolution, fps, audio stream presence via ffprobe.
    - Silence gaps >= 1.5s via FFmpeg silencedetect.
    - Audio volume levels (mean_volume, max_volume) via FFmpeg volumedetect.
    - Flags technical red flags (dead silence >= 1.5s, quiet < -28dB, clipping >= 0.0dB, bounds).
    - Generates 3x4 (12 tiles) visual review contact sheet with bright yellow MM:SS stamps.
    - Loads companion metadata (<stem>.json or plan.json) for script-to-visual narration timeline.
    """
    vpath = Path(video_path).resolve()
    mpath = Path(metadata_path).resolve() if metadata_path is not None else None

    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    # 1. FFprobe video & audio stream metadata
    cmd_probe = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(vpath),
    ]
    try:
        probe_res = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
        probe_json = json.loads(probe_res.stdout)
    except Exception as exc:
        raise RuntimeError(f"FFprobe failed on {vpath}: {exc}") from exc

    streams = probe_json.get("streams", [])
    format_info = probe_json.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int(video_stream.get("width", 0)) if video_stream else 0
    height = int(video_stream.get("height", 0)) if video_stream else 0
    fps = _parse_fps(video_stream.get("r_frame_rate", "") or video_stream.get("avg_frame_rate", "")) if video_stream else 0.0

    dur_raw = format_info.get("duration") or (video_stream.get("duration") if video_stream else None) or 0.0
    try:
        duration = round(float(dur_raw), 2)
    except (ValueError, TypeError):
        duration = 0.0

    has_audio = audio_stream is not None

    # 2. Audio quality: silencedetect and volumedetect
    silence_gaps: list[dict[str, float]] = []
    mean_volume: float | None = None
    max_volume: float | None = None
    red_flags: list[str] = []

    if not has_audio:
        red_flags.append("Fatal: Video contains no audio stream.")
    else:
        cmd_audio = [
            "ffmpeg",
            "-y",
            "-i",
            str(vpath),
            "-af",
            "silencedetect=noise=-45dB:d=1.5,volumedetect",
            "-f",
            "null",
            "-",
        ]
        try:
            audio_proc = subprocess.run(cmd_audio, capture_output=True, text=True)
            stderr_out = audio_proc.stderr or ""

            # Parse volumedetect
            def _parse_vol_str(val_str: str) -> float | None:
                cleaned = val_str.strip().lower()
                if "inf" in cleaned:
                    return -99.0 if "-" in cleaned else 0.0
                try:
                    return round(float(cleaned), 1)
                except ValueError:
                    return None

            mean_match = re.search(r"mean_volume:\s*([-\d.]+|-\s*inf|\binf\b)\s*dB", stderr_out, re.IGNORECASE)
            if mean_match:
                mean_volume = _parse_vol_str(mean_match.group(1))

            max_match = re.search(r"max_volume:\s*([-\d.]+|-\s*inf|\binf\b)\s*dB", stderr_out, re.IGNORECASE)
            if max_match:
                max_volume = _parse_vol_str(max_match.group(1))

            # Parse silencedetect
            cur_silence_start: float | None = None
            for line in stderr_out.splitlines():
                s_start = re.search(r"silence_start:\s*([-\d.]+)", line)
                if s_start:
                    cur_silence_start = float(s_start.group(1))
                s_end = re.search(r"silence_end:\s*([-\d.]+)\s*\|\s*silence_duration:\s*([-\d.]+)", line)
                if s_end:
                    end_val = float(s_end.group(1))
                    dur_val = float(s_end.group(2))
                    st_val = cur_silence_start if cur_silence_start is not None else max(0.0, end_val - dur_val)
                    if dur_val >= 1.5:
                        silence_gaps.append({
                            "start": round(st_val, 2),
                            "end": round(end_val, 2),
                            "duration": round(dur_val, 2),
                        })
                    cur_silence_start = None

            # Handle trailing silence reaching EOF
            if cur_silence_start is not None and duration > cur_silence_start:
                trailing_dur = duration - cur_silence_start
                if trailing_dur >= 1.5:
                    silence_gaps.append({
                        "start": round(cur_silence_start, 2),
                        "end": round(duration, 2),
                        "duration": round(trailing_dur, 2),
                    })

        except Exception as exc:
            logger.debug("Audio quality probe failed on %s: %s", vpath, exc)

    # 3. Automatic Red Flag Analysis
    for gap in silence_gaps:
        red_flags.append(
            f"Silence gap >= 1.5s detected: {gap['duration']}s from {gap['start']}s to {gap['end']}s"
        )

    if mean_volume is not None and mean_volume < -28.0:
        red_flags.append(f"Audio is very quiet: mean volume is {mean_volume:.1f} dB (< -28 dB threshold)")

    if max_volume is not None and max_volume >= 0.0:
        red_flags.append(f"Audio is clipping: max volume is {max_volume:.1f} dB (>= 0.0 dB threshold)")

    # Duration boundary checks: Shorts vs Long-form
    is_vertical = height > width
    is_long = "long" in vpath.stem.lower() or (width >= height and not ("clip" in vpath.stem.lower() or "short" in vpath.stem.lower()))
    is_short = not is_long and (is_vertical or "clip" in vpath.stem.lower() or "short" in vpath.stem.lower())

    if is_short:
        if duration > 55.0:
            red_flags.append(f"Short duration {duration:.1f}s exceeds 55s target bound")
    elif is_long:
        if duration < 420.0:
            red_flags.append(f"Long-form duration {duration:.1f}s is below 420s (7 min) target bound")
        elif duration > 660.0:
            red_flags.append(f"Long-form duration {duration:.1f}s exceeds 660s (11 min) target bound")

    # 4. Generate Visual Review Contact Sheet
    contact_sheet_path = generate_review_contact_sheet(
        video_path=vpath,
        duration=duration,
        width=width,
        height=height,
    )

    # 5. Extract Narration Timeline
    narration_timeline = _load_narration_timeline(vpath, metadata_path=mpath)

    audio_stats = {
        "has_audio": has_audio,
        "mean_volume_db": mean_volume,
        "max_volume_db": max_volume,
    }

    return {
        "video_path": str(vpath),
        "contact_sheet_path": str(contact_sheet_path),
        "duration": duration,
        "resolution": [width, height],
        "width": width,
        "height": height,
        "fps": fps,
        "audio_stats": audio_stats,
        "silence_gaps": silence_gaps,
        "red_flags": red_flags,
        "narration_timeline": narration_timeline,
    }
