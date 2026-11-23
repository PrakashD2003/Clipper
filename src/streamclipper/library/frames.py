"""Low-resolution proxy video download and frame contact sheet builder for visual indexing."""
from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    """Format seconds into MM:SS (or HH:MM:SS) display label."""
    if not math.isfinite(seconds):
        return "00:00"
    s = max(0, int(round(seconds)))
    if s < 3600:
        mins = s // 60
        secs = s % 60
        return f"{mins:02d}:{secs:02d}"
    hrs = s // 3600
    mins = (s % 3600) // 60
    secs = s % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _stamp_tile_timestamp(
    img: Image.Image,
    seconds: float,
    *,
    text_color: tuple[int, int, int] = (255, 255, 0),
) -> Image.Image:
    """Stamp a high-contrast timestamp badge on the top-left of a tile."""
    draw = ImageDraw.Draw(img)
    lbl = _format_timestamp_label(seconds)
    try:
        font = ImageFont.load_default(size=16)
    except (TypeError, AttributeError):
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((10, 8), lbl, font=font)
        draw.rectangle(
            [bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill=(0, 0, 0)
        )
    except Exception:
        draw.rectangle([6, 6, 80, 26], fill=(0, 0, 0))
    draw.text((10, 8), lbl, fill=text_color, font=font)
    return img


def extract_moment_contact_sheets(
    url: str,
    center_time: float,
    temp_dir: Path | str,
    *,
    window_before: float = 120.0,
    window_after: float = 60.0,
    height: int = 360,
    grid: int = 3,
    sheet_width: int = 1152,
) -> list[ContactSheet]:
    """Extract dual 90-second 3x3 contact sheets for candidate moment inspection.

    Downloads a lightweight 360p proxy video slice for [center - window_before, center + window_after]
    using yt-dlp download_ranges (or uses local video if url points to an existing file).
    Generates:
      - Sheet 1: Buildup & Trigger [center - 120s to -30s] (9 tiles, 1 every 10s)
      - Sheet 2: Climax & Reaction [center - 30s to +60s] (9 tiles, 1 every 10s)
    Each tile is stamped with a high-contrast yellow MM:SS (or HH:MM:SS) timestamp label.
    Returns list of ContactSheet objects with file paths and tile timestamps.
    """
    clean_url = str(url or "").strip()
    if not clean_url:
        raise ValueError("URL cannot be empty")
    center = float(center_time)
    if center < 0:
        raise ValueError("center_time must be >= 0")

    wb = float(window_before)
    wa = float(window_after)
    slice_start = max(0.0, center - wb)
    slice_end = center + wa

    source_key = hashlib.sha256(clean_url.encode("utf-8")).hexdigest()[:12]
    target_dir = Path(temp_dir) / "moment_frames" / source_key / f"{int(center)}_{int(wb)}_{int(wa)}"
    target_dir.mkdir(parents=True, exist_ok=True)
    sheet1_path = target_dir / "sheet_1_buildup.png"
    sheet2_path = target_dir / "sheet_2_climax.png"
    manifest_path = target_dir / "manifest.json"

    if manifest_path.is_file() and sheet1_path.is_file() and sheet2_path.is_file():
        try:
            cached_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(cached_data, list) and len(cached_data) == 2:
                cached_sheets = [ContactSheet.from_dict(d) for d in cached_data]
                if all(s.path.is_file() and s.path.stat().st_size > 0 for s in cached_sheets):
                    return cached_sheets
        except Exception as exc:
            logger.debug("Failed loading moment sheets manifest %s: %s", manifest_path, exc)

    local_cand = Path(clean_url)
    if local_cand.is_file() and local_cand.stat().st_size > 0:
        actual_video = local_cand
    else:
        proxy_path = target_dir / "proxy_slice.mp4"
        if not (proxy_path.is_file() and proxy_path.stat().st_size > 0):
            fmt = f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/worst"
            ydl_opts: dict[str, Any] = {
                "format": fmt,
                "outtmpl": str(target_dir / "proxy_slice.%(ext)s"),
                "merge_output_format": "mp4",
                "force_keyframes_at_cuts": True,
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "overwrites": True,
                "retries": 3,
                "fragment_retries": 3,
                "extractor_retries": 3,
            }
            apply_ytdlp_auth_opts(ydl_opts)

            import yt_dlp

            ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
                None, [(slice_start, slice_end)]
            )
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([clean_url])
            except Exception as exc:
                logger.warning(
                    "Failed downloading proxy video slice for %s [%.1f-%.1f]: %s",
                    clean_url,
                    slice_start,
                    slice_end,
                    exc,
                )
                return []

            if not (proxy_path.is_file() and proxy_path.stat().st_size > 0):
                video_exts = {".webm", ".mkv", ".avi", ".mov", ".ts", ".flv"}
                for cand in target_dir.glob("proxy_slice.*"):
                    if (
                        cand.is_file()
                        and cand.stat().st_size > 0
                        and cand.suffix.lower() in video_exts
                        and cand.resolve() != proxy_path.resolve()
                    ):
                        try:
                            cmd = ["ffmpeg", "-y", "-i", str(cand), "-c", "copy", str(proxy_path)]
                            subprocess.run(cmd, capture_output=True, check=True)
                            cand.unlink(missing_ok=True)
                            if proxy_path.is_file() and proxy_path.stat().st_size > 0:
                                break
                        except Exception:
                            shutil.move(str(cand), str(proxy_path))
                            break

        if not (proxy_path.is_file() and proxy_path.stat().st_size > 0):
            logger.warning("No proxy slice video file found at %s", proxy_path)
            return []
        actual_video = proxy_path

    v_dur = _get_video_duration(actual_video)
    proxy_cand = target_dir / "proxy_slice.mp4"
    if actual_video == proxy_cand or actual_video.name.startswith("proxy_slice"):
        file_offset = slice_start
    else:
        file_offset = 0.0

    num_tiles = max(1, grid * grid)

    # Sheet 1: Buildup & Trigger [center - window_before, center - 30.0]
    s1_start = max(0.0, center - wb)
    s1_end = max(s1_start, center - 30.0)
    if s1_end <= s1_start:
        s1_end = max(s1_start + 1.0, center)
    s1_dur = max(0.1, s1_end - s1_start)
    step1 = s1_dur / num_tiles
    s1_timestamps = [round(s1_start + i * step1, 2) for i in range(num_tiles)]

    # Sheet 2: Climax & Reaction [center - 30.0, center + window_after]
    s2_start = max(0.0, center - 30.0)
    s2_end = max(s2_start + 0.1, center + wa)
    s2_dur = max(0.1, s2_end - s2_start)
    step2 = s2_dur / num_tiles
    s2_timestamps = [round(s2_start + i * step2, 2) for i in range(num_tiles)]

    tile_width = max(16, sheet_width // grid)
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        def _render_sheet(timestamps: list[float], out_png: Path) -> None:
            tile_images: list[Image.Image] = []
            for idx, ts in enumerate(timestamps):
                tile_tmp = tmp_dir / f"tile_{idx:02d}.jpg"
                seek_time = max(0.0, ts - file_offset)
                if v_dur > 0:
                    seek_time = min(seek_time, max(0.0, v_dur - 0.05))
                img = _grab_tile_frame(actual_video, seek_time, tile_tmp)
                if img is None:
                    img = Image.new("RGB", (tile_width, int(tile_width * 9 / 16)), color=(0, 0, 0))
                tile_images.append(img)

            first_w, first_h = tile_images[0].size
            aspect = (first_h / first_w) if first_w > 0 else (9.0 / 16.0)
            tile_height = max(16, int(round(tile_width * aspect)))
            sheet_height = tile_height * grid

            canvas = Image.new("RGB", (sheet_width, sheet_height), color=(0, 0, 0))
            for idx, (img, ts) in enumerate(zip(tile_images, timestamps)):
                resized = img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                _stamp_tile_timestamp(resized, ts, text_color=(255, 255, 0))
                r = idx // grid
                c = idx % grid
                canvas.paste(resized, (c * tile_width, r * tile_height))

            canvas.save(out_png, "PNG")

        _render_sheet(s1_timestamps, sheet1_path)
        _render_sheet(s2_timestamps, sheet2_path)

    sheet1 = ContactSheet(path=sheet1_path, start=s1_start, end=s1_end, timestamps=s1_timestamps)
    sheet2 = ContactSheet(path=sheet2_path, start=s2_start, end=s2_end, timestamps=s2_timestamps)
    results = [sheet1, sheet2]

    try:
        manifest_path.write_text(
            json.dumps([s.to_dict() for s in results], indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("Failed writing moment sheets manifest %s: %s", manifest_path, exc)

    return results


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
                _stamp_tile_timestamp(resized, ts, text_color=(255, 255, 255))
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
