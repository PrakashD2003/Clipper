"""Targeted gameplay B-roll search, segment download, and 3-frame preview strip extraction."""
from __future__ import annotations

import logging
import math
from pathlib import Path
import re
import subprocess
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yt_dlp

from streamclipper.config import Config
from streamclipper.download._ytdlp_opts import apply_ytdlp_auth_opts
from streamclipper.download.video import VideoDownloadError, download_segment

logger = logging.getLogger(__name__)

_POSITIVE_BROLL_TERMS: tuple[str, ...] = (
    "gameplay",
    "no commentary",
    "showcase",
    "4k",
    "60fps",
    "detail",
    "details",
    "trailer",
    "comparison",
    "walkthrough",
)

_NEGATIVE_BROLL_TERMS: tuple[str, ...] = (
    "live stream",
    "livestream",
    "full stream",
    "reaction",
    "reacting",
    "reacts",
    "facecam",
    "podcast",
    "stream vod",
)


def _resolve_watch_url(entry: dict[str, Any], video_id: str) -> str:
    for key in ("webpage_url", "url"):
        cand = entry.get(key)
        if isinstance(cand, str) and cand.strip().startswith(("http://", "https://")):
            return cand.strip()
    return f"https://www.youtube.com/watch?v={video_id}"


def _resolve_channel_name(entry: dict[str, Any]) -> str:
    for key in ("channel", "uploader", "creator"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "Unknown"


def _score_broll_candidate(entry: dict[str, Any]) -> tuple[bool, float]:
    """Score a yt-dlp search entry for clean gameplay B-roll suitability.

    Prefers short, clean gameplay showcase videos (15s–10m) and penalizes/avoids
    livestreams and facecam/reaction clutter.
    """
    if not isinstance(entry, dict):
        return (False, -1.0)

    vid = str(entry.get("id") or "").strip()
    if not vid:
        return (False, -1.0)

    if entry.get("is_live") is True:
        return (False, -1.0)
    live_status = str(entry.get("live_status") or "").strip().lower()
    if live_status in ("is_live", "is_upcoming", "post_live"):
        return (False, -1.0)

    raw_dur = entry.get("duration")
    try:
        duration = float(raw_dur) if raw_dur is not None and not isinstance(raw_dur, bool) else 0.0
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration < 0.0:
        duration = 0.0

    score = 0.0
    if 15.0 <= duration <= 600.0:
        score += 50.0
        if 30.0 <= duration <= 300.0:
            score += 20.0
    elif 0.0 < duration < 15.0:
        score -= 50.0
    elif duration > 1800.0:
        score -= 50.0
    elif duration > 600.0:
        score -= 30.0

    # Prefer 16:9 landscape gameplay; strongly penalize vertical Shorts
    raw_url = str(entry.get("webpage_url") or entry.get("url") or "").lower()
    if "/shorts/" in raw_url:
        score -= 80.0
    try:
        w = float(entry.get("width") or 0.0)
        h = float(entry.get("height") or 0.0)
        if w > 0.0 and h > 0.0:
            if h > w:
                score -= 80.0
            elif w >= h:
                score += 10.0
    except (TypeError, ValueError):
        pass

    title = str(entry.get("title") or "").lower()
    for term in _POSITIVE_BROLL_TERMS:
        if term in title:
            score += 15.0
    for term in _NEGATIVE_BROLL_TERMS:
        if term in title:
            score -= 25.0
    if re.search(r"\blive\b", title):
        score -= 20.0
    if re.search(r"\bvod\b", title):
        score -= 20.0

    raw_views = entry.get("view_count")
    try:
        views = float(raw_views) if raw_views is not None and not isinstance(raw_views, bool) else 0.0
    except (TypeError, ValueError):
        views = 0.0
    if math.isfinite(views) and views > 0:
        score += min(10.0, math.log10(max(1.0, views)))

    return (True, score)


def _pick_segment_bounds(entry: dict[str, Any], clip_seconds: float) -> tuple[float, float]:
    """Select a targeted [start, end] range from the most-replayed point or video midpoint."""
    target_len = max(1.0, float(clip_seconds))
    raw_dur = entry.get("duration")
    try:
        duration = float(raw_dur) if raw_dur is not None and not isinstance(raw_dur, bool) else 0.0
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration < 0.0:
        duration = 0.0

    heatmap = entry.get("heatmap")
    if isinstance(heatmap, list) and heatmap:
        best_item: dict[str, Any] | None = None
        best_val = -1.0
        for item in heatmap:
            if not isinstance(item, dict):
                continue
            try:
                st = float(item.get("start_time", -1.0))
                et = float(item.get("end_time", -1.0))
                val = float(item.get("value", 0.0))
            except (TypeError, ValueError):
                continue
            if math.isfinite(st) and math.isfinite(et) and math.isfinite(val) and et > st >= 0.0:
                if val > best_val:
                    best_val = val
                    best_item = {"start_time": st, "end_time": et}

        if best_item is not None:
            mid = (best_item["start_time"] + best_item["end_time"]) / 2.0
            start = max(0.0, mid - (target_len / 2.0))
            if duration > 0.0:
                start = min(start, max(0.0, duration - target_len))
                end = min(duration, start + target_len)
            else:
                end = start + target_len
            if end > start:
                return (round(start, 2), round(end, 2))

    if duration > target_len:
        start = max(0.0, (duration - target_len) / 2.0)
        end = min(duration, start + target_len)
    elif duration > 0.5:
        start = 0.0
        end = min(duration, target_len)
    else:
        start = 0.0
        end = target_len

    return (round(start, 2), round(end, 2))


def _extract_preview_strip(
    clip_path: Path,
    preview_path: Path,
    *,
    num_frames: int = 3,
    tile_w: int = 426,
    tile_h: int = 240,
) -> Path | None:
    """Extract a horizontal 3-frame preview thumbnail strip (<stem>_preview.jpg)."""
    clip_path = Path(clip_path)
    preview_path = Path(preview_path)
    if not clip_path.is_file() or clip_path.stat().st_size == 0:
        return None

    ratios = [0.15, 0.50, 0.85][: max(1, num_frames)]
    sampled: list[tuple[float, Image.Image]] = []

    cap = None
    try:
        cap = cv2.VideoCapture(str(clip_path.resolve()))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            dur = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0
            if total_frames > 0:
                for r in ratios:
                    frame_idx = max(0, min(int(total_frames * r), total_frames - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, bgr = cap.read()
                    if ret and bgr is not None:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        ts = (dur * r) if dur > 0 else (r * 10.0)
                        sampled.append((ts, Image.fromarray(rgb)))
    except Exception as exc:
        logger.debug("OpenCV preview extraction failed for %s: %s", clip_path, exc)
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    if len(sampled) < len(ratios):
        sampled.clear()
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(clip_path.resolve()),
            ]
            probe = subprocess.run(probe_cmd, capture_output=True, text=True)
            dur = float(probe.stdout.strip()) if probe.returncode == 0 and probe.stdout.strip() else 10.0
            if not math.isfinite(dur) or dur <= 0.0:
                dur = 10.0
            for idx, r in enumerate(ratios):
                t_seek = max(0.05, min(dur * r, max(0.05, dur - 0.1)))
                tmp_frame = preview_path.with_name(f"{preview_path.stem}_f{idx}.jpg")
                try:
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{t_seek:.2f}",
                        "-i", str(clip_path.resolve()),
                        "-vframes", "1",
                        "-f", "image2",
                        str(tmp_frame.resolve()),
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    if tmp_frame.is_file():
                        with Image.open(tmp_frame) as im:
                            sampled.append((t_seek, im.convert("RGB")))
                finally:
                    tmp_frame.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("FFmpeg preview extraction failed for %s: %s", clip_path, exc)

    if not sampled:
        return None

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    strip = Image.new("RGB", (tile_w * len(sampled), tile_h), (12, 12, 16))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for idx, (ts, frame_img) in enumerate(sampled):
        tile = frame_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        x_off = idx * tile_w
        strip.paste(tile, (x_off, 0))
        label = f"{ts:.1f}s"
        draw.text(
            (x_off + 10, tile_h - 28),
            label,
            font=font,
            fill=(255, 235, 59),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    strip.save(str(preview_path), "JPEG", quality=88)
    return preview_path if preview_path.is_file() else None


def fetch_topic_broll(
    config: Config,
    query: str,
    game: str = "GTA 6",
    max_clips: int = 2,
    clip_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    """Search YouTube for short, clean gameplay B-roll, download targeted segments, and extract 3-frame preview strips.

    Returns a list of dicts containing clip_path, preview_sheet_path, url, channel,
    source_id, title, kind, role, start, end, and duration.
    """
    clean_query = str(query or "").strip()
    if not clean_query:
        raise ValueError("query is required")

    clean_game = str(game or "").strip()
    if clean_game and clean_game.lower() not in clean_query.lower():
        search_query = f"{clean_game} {clean_query}"
    else:
        search_query = clean_query

    limit = int(max_clips)
    if limit <= 0:
        return []

    seg_len = max(1.0, min(float(clip_seconds), 20.0))

    temp_base = Path(config.get("general.temp_dir", "./temp"))
    broll_dir = temp_base / "broll"
    broll_dir.mkdir(parents=True, exist_ok=True)

    search_n = max(8, limit * 4)
    search_term = f"ytsearch{search_n}:{search_query}"

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
        "ignoreerrors": True,
    }
    proxy = str(config.get("discovery.proxy", "") or "").strip()
    if proxy:
        ydl_opts["proxy"] = proxy
    apply_ytdlp_auth_opts(ydl_opts)

    entries: list[dict[str, Any]] = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_term, download=False)
        if isinstance(info, dict) and isinstance(info.get("entries"), list):
            entries = [e for e in info["entries"] if isinstance(e, dict)]
    except (yt_dlp.utils.DownloadError, yt_dlp.utils.ExtractorError, OSError) as exc:
        logger.warning("B-roll yt-dlp search failed for %r: %s", search_term, exc)
        return []

    ranked: list[tuple[float, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for entry in entries:
        ok, score = _score_broll_candidate(entry)
        if not ok:
            continue
        vid = str(entry.get("id") or "").strip()
        if not vid or vid in seen_ids:
            continue
        seen_ids.add(vid)
        ranked.append((score, entry))

    ranked.sort(key=lambda item: item[0], reverse=True)

    results: list[dict[str, Any]] = []
    for _, entry in ranked:
        if len(results) >= limit:
            break

        video_id = str(entry.get("id") or "").strip()
        watch_url = _resolve_watch_url(entry, video_id)
        channel = _resolve_channel_name(entry)
        title = str(entry.get("title") or "").strip()
        try:
            raw_dur = float(entry.get("duration") or 0.0)
        except (TypeError, ValueError):
            raw_dur = 0.0

        start, end = _pick_segment_bounds(entry, seg_len)
        if end <= start:
            continue

        try:
            clip_path = download_segment(
                watch_url,
                start,
                end,
                broll_dir,
                buffer_seconds=0.0,
                video_id=video_id,
            )
        except (VideoDownloadError, Exception) as exc:
            logger.warning("Failed downloading B-roll candidate %s (%s): %s", video_id, watch_url, exc)
            continue

        if not clip_path or not Path(clip_path).is_file():
            continue

        clip_path = Path(clip_path)
        preview_target = clip_path.with_name(f"{clip_path.stem}_preview.jpg")
        preview_path = _extract_preview_strip(clip_path, preview_target)

        results.append(
            {
                "source_id": video_id,
                "clip_path": str(clip_path.resolve()),
                "url": watch_url,
                "channel": channel,
                "title": title,
                "kind": "gameplay",
                "role": "broll",
                "start": start,
                "end": end,
                "duration": round(max(0.0, end - start), 2),
                "video_duration": round(max(0.0, raw_dur), 2),
                "preview_sheet_path": str(preview_path.resolve()) if preview_path and preview_path.is_file() else "",
                "query": search_query,
            }
        )

    return results
