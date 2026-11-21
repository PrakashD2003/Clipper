"""Low-resolution proxy video download and frame contact sheet builder for visual indexing."""
from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from streamclipper.analysis.content_gate import DEFAULT_STATIC_THRESHOLD
from streamclipper.config import Config
from streamclipper.download._ytdlp_opts import apply_ytdlp_auth_opts
from streamclipper.models import SerializableModel

logger = logging.getLogger(__name__)


@dataclass
class ContactSheet(SerializableModel):
    """A contact sheet image with a grid of sampled video frames and timestamps."""

    path: Path = Path("")
    start: float = 0.0
    end: float = 0.0
    timestamps: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContactSheet":
        raw_ts = data.get("timestamps") or []
        return cls(
            path=Path(str(data.get("path") or "")),
            start=float(data.get("start") or 0.0),
            end=float(data.get("end") or 0.0),
            timestamps=[float(t) for t in raw_ts],
        )


def download_proxy_video(
    url: str,
    out_dir: Path | str,
    *,
    height: int = 360,
) -> Path:
    """Download a low-resolution proxy video for visual contact sheets.

    Uses yt-dlp format 'bv*[height<=360]/b[height<=360]/worst' and caches at
    '<out_dir>/proxy.mp4'. Reuses the existing file if already present.
    """
    clean_url = str(url or "").strip()
    if not clean_url:
        raise ValueError("URL cannot be empty")

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = target_dir / "proxy.mp4"

    if proxy_path.is_file() and proxy_path.stat().st_size > 0:
        logger.debug("Reusing existing proxy video at %s", proxy_path)
        return proxy_path

    fmt = f"bv*[height<={height}]/b[height<={height}]/worst"
    ydl_opts: dict[str, Any] = {
        "format": fmt,
        "outtmpl": str(target_dir / "proxy.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    apply_ytdlp_auth_opts(ydl_opts)

    import yt_dlp

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([clean_url])

    if proxy_path.is_file() and proxy_path.stat().st_size > 0:
        return proxy_path

    # Check for alternative video extensions if not saved directly as .mp4
    video_exts = {".webm", ".mkv", ".avi", ".mov", ".ts", ".flv"}
    for cand in target_dir.glob("proxy.*"):
        if (
            cand.is_file()
            and cand.stat().st_size > 0
            and cand.suffix.lower() in video_exts
            and cand.resolve() != proxy_path.resolve()
        ):
            try:
                # Remux or rename to proxy.mp4
                cmd = ["ffmpeg", "-y", "-i", str(cand), "-c", "copy", str(proxy_path)]
                subprocess.run(cmd, capture_output=True, check=True)
                cand.unlink(missing_ok=True)
                if proxy_path.is_file() and proxy_path.stat().st_size > 0:
                    return proxy_path
            except Exception:
                shutil.move(str(cand), str(proxy_path))
                return proxy_path

    return proxy_path


def _get_video_duration(video_path: Path) -> float:
    """Probe duration in seconds of a video file via ffprobe or cv2 fallback."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dur = float(res.stdout.strip())
        if dur > 0.0:
            return dur
    except Exception:
        pass

    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dur = float(res.stdout.strip())
        if dur > 0.0:
            return dur
    except Exception:
        pass

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        if fps > 0.0 and count > 0.0:
            return float(count / fps)
    except Exception:
        pass

    return 0.0


def _grab_tile_frame(video_path: Path, timestamp: float, temp_path: Path) -> Image.Image | None:
    """Extract a single frame from video at timestamp using ffmpeg -ss."""
    temp_path.unlink(missing_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(temp_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, check=True)
        if temp_path.is_file() and temp_path.stat().st_size > 0:
            with Image.open(temp_path) as im:
                return im.convert("RGB")
    except Exception as exc:
        logger.debug("Failed grabbing frame at %s from %s: %s", timestamp, video_path, exc)
    return None


def _format_timestamp_label(seconds: float) -> str:
    """Format seconds into MM:SS display label."""
    s = max(0, int(round(seconds)))
    mins = s // 60
    secs = s % 60
    return f"{mins:02d}:{secs:02d}"


def build_contact_sheets(
    config: Config,
    source_id: str,
    *,
    interval_seconds: float | None = None,
    video_path: Path | None = None,
) -> list[ContactSheet]:
    """Build grid contact sheets for a source video and return metadata.

    One sheet per library.frames.sheet_seconds (default 90s) with a grid of
    library.frames.grid (default 3x3) frames. Each tile is labelled with its MM:SS timestamp.
    Sheets whose frames are static are skipped. Results are cached on disk.
    """
    temp_dir = Path(config.get("general.temp_dir", "./temp"))
    frames_dir = temp_dir / "frames" / source_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = frames_dir / "sheets.json"
    if manifest_path.is_file() and manifest_path.stat().st_size > 0:
        try:
            cached_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(cached_data, list):
                cached_sheets = [ContactSheet.from_dict(d) for d in cached_data]
                if not cached_sheets or all(s.path.is_file() for s in cached_sheets):
                    return cached_sheets
        except Exception as exc:
            logger.debug("Failed loading sheets manifest %s: %s", manifest_path, exc)

    if video_path is not None and Path(video_path).is_file():
        actual_video = Path(video_path)
    else:
        actual_video = frames_dir / "proxy.mp4"

    if not actual_video.is_file():
        # Check if source exists in index
        from streamclipper.library.index import LibraryIndex

        src = LibraryIndex(config).get_source(source_id)
        if src and src.url:
            p_height = int(config.get("library.frames.proxy_height", 360))
            download_proxy_video(src.url, frames_dir, height=p_height)

    if not actual_video.is_file():
        logger.warning("No video file found to build contact sheets for source %s", source_id)
        return []

    duration = _get_video_duration(actual_video)
    if duration <= 0.0:
        # Fallback to source duration
        from streamclipper.library.index import LibraryIndex

        src = LibraryIndex(config).get_source(source_id)
        if src:
            duration = float(src.duration_seconds or 0.0)

    if duration <= 0.0:
        return []

    sheet_seconds = float(
        interval_seconds
        if interval_seconds is not None
        else config.get("library.frames.sheet_seconds", 90)
    )
    grid = int(config.get("library.frames.grid", 3))
    sheet_width = int(config.get("library.frames.sheet_width", 1152))
    static_threshold = float(
        config.get("analysis.content_gate.static_threshold", DEFAULT_STATIC_THRESHOLD)
    )

    num_tiles = max(1, grid * grid)
    tile_width = max(16, sheet_width // grid)

    results: list[ContactSheet] = []
    current_start = 0.0

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        while current_start < duration:
            current_end = min(current_start + sheet_seconds, duration)
            if current_end - current_start < 0.5:
                break

            sheet_dur = current_end - current_start
            step = sheet_dur / num_tiles
            timestamps = [round(current_start + i * step, 2) for i in range(num_tiles)]

            start_label = f"{int(current_start)}" if current_start.is_integer() else f"{current_start:.1f}"
            sheet_filename = f"sheet_{start_label}.jpg"
            sheet_path = frames_dir / sheet_filename

            tile_images: list[Image.Image] = []
            gray_arrays: list[np.ndarray] = []

            for idx, ts in enumerate(timestamps):
                tile_tmp = tmp_dir / f"tile_{idx:02d}.jpg"
                img = _grab_tile_frame(actual_video, ts, tile_tmp)
                if img is None:
                    # Synthetic fallback blank frame
                    img = Image.new("RGB", (tile_width, int(tile_width * 9 / 16)), color=(0, 0, 0))

                tile_images.append(img)
                # Convert to grayscale numpy for static detection
                gray = np.array(img.convert("L"), dtype=np.float32)
                gray_arrays.append(gray)

            # Static screen check across consecutive sampled frames
            is_static = False
            if len(gray_arrays) >= 2:
                diffs: list[float] = []
                for i in range(1, len(gray_arrays)):
                    # Ensure matching shapes before subtracting
                    g1 = gray_arrays[i - 1]
                    g2 = gray_arrays[i]
                    if g1.shape != g2.shape:
                        continue
                    diffs.append(float(np.mean(np.abs(g2 - g1))))

                if diffs and (float(np.mean(diffs)) < static_threshold):
                    is_static = True

            if is_static:
                logger.debug(
                    "Skipping static contact sheet for source %s at [%.1f, %.1f]",
                    source_id,
                    current_start,
                    current_end,
                )
                current_start += sheet_seconds
                continue

            # Compute tile height based on aspect ratio
            first_w, first_h = tile_images[0].size
            aspect = (first_h / first_w) if first_w > 0 else (9.0 / 16.0)
            tile_height = max(16, int(round(tile_width * aspect)))
            sheet_height = tile_height * grid

            canvas = Image.new("RGB", (sheet_width, sheet_height), color=(0, 0, 0))

            for idx, (img, ts) in enumerate(zip(tile_images, timestamps)):
                resized = img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                draw = ImageDraw.Draw(resized)

                # Label tile with MM:SS timestamp
                lbl = _format_timestamp_label(ts)
                # Badge background in top-left
                try:
                    bbox = draw.textbbox((10, 8), lbl)
                    draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill=(0, 0, 0))
                except Exception:
                    draw.rectangle([6, 6, 60, 24], fill=(0, 0, 0))
                draw.text((10, 8), lbl, fill=(255, 255, 255))

                r = idx // grid
                c = idx % grid
                canvas.paste(resized, (c * tile_width, r * tile_height))

            canvas.save(sheet_path, "JPEG", quality=85)
            sheet_obj = ContactSheet(
                path=sheet_path,
                start=current_start,
                end=current_end,
                timestamps=timestamps,
            )
            results.append(sheet_obj)
            current_start += sheet_seconds

    # Save manifest for fast caching
    try:
        manifest_path.write_text(
            json.dumps([s.to_dict() for s in results], indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("Failed writing sheets manifest %s: %s", manifest_path, exc)

    return results
