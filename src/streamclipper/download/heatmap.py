"""YouTube "Most replayed" heatmap fetcher (free via yt-dlp's ``heatmap`` info field)."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def normalize_heatmap(raw: Any) -> list[dict[str, float]]:
    """Normalize yt-dlp ``heatmap`` entries to ``[{"start", "end", "value"}]`` (value 0..1).

    Accepts yt-dlp's ``start_time`` / ``end_time`` keys (and ``start`` / ``end``). Malformed
    entries are skipped; values above 1.0 are scaled by the maximum. Never raises.
    """
    if not isinstance(raw, list):
        return []
    segments: list[dict[str, float]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            start = float(entry.get("start_time", entry.get("start")))
            end = float(entry.get("end_time", entry.get("end")))
            value = float(entry.get("value"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end) and math.isfinite(value)):
            continue
        if start < 0.0 or end <= start or value < 0.0:
            continue
        segments.append({"start": start, "end": end, "value": value})
    if not segments:
        return []
    peak = max(s["value"] for s in segments)
    if peak > 1.0:
        for s in segments:
            s["value"] = s["value"] / peak
    return segments


def fetch_heatmap(url: str, info: dict | None = None) -> list[dict[str, float]]:
    """Return normalized heatmap segments for a video, or ``[]`` when unavailable.

    If ``info`` (a yt-dlp info dict) is given it is used with no network access; otherwise
    yt-dlp extracts the info (metadata only). Never raises: new / low-view videos have no
    heatmap, and any extraction failure is logged at debug level.
    """
    try:
        if isinstance(info, dict):
            return normalize_heatmap(info.get("heatmap"))
        if not isinstance(url, str) or not url.strip():
            return []
        import yt_dlp

        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            extracted = ydl.extract_info(url.strip(), download=False)
        if not isinstance(extracted, dict):
            return []
        return normalize_heatmap(extracted.get("heatmap"))
    except Exception as exc:
        logger.debug("Heatmap fetch failed for %s: %s", url, exc)
        return []


def _read_json(path: Path) -> Any:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return None


def load_cached_heatmap(temp_dir: Path | str, video_id: str) -> list[dict[str, float]]:
    """Read a heatmap from ``<temp_dir>/<video_id>.info.json`` or ``<video_id>_heatmap.json``."""
    if not video_id:
        return []
    base = Path(temp_dir)
    data = _read_json(base / f"{video_id}.info.json")
    if isinstance(data, dict):
        segments = normalize_heatmap(data.get("heatmap"))
        if segments:
            return segments
    return normalize_heatmap(_read_json(base / f"{video_id}_heatmap.json"))


def get_heatmap(
    url: str,
    temp_dir: Path | str,
    video_id: str,
    info: dict | None = None,
) -> list[dict[str, float]]:
    """Heatmap from ``info``, then local cache files, then yt-dlp (result cached on success)."""
    if isinstance(info, dict):
        segments = normalize_heatmap(info.get("heatmap"))
        if segments:
            return segments
    segments = load_cached_heatmap(temp_dir, video_id)
    if segments:
        return segments
    segments = fetch_heatmap(url)
    if segments and video_id:
        try:
            base = Path(temp_dir)
            base.mkdir(parents=True, exist_ok=True)
            (base / f"{video_id}_heatmap.json").write_text(
                json.dumps(segments), encoding="utf-8"
            )
        except OSError as exc:
            logger.debug("Could not cache heatmap: %s", exc)
    return segments
