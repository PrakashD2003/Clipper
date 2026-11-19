"""Selective video segment downloader using yt-dlp download_ranges."""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import yt_dlp

from .audio import _extract_youtube_id_from_url

logger = logging.getLogger(__name__)

_DEFAULT_RETRIES = 3

VIDEO_FORMAT_1080P = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
_ALT_VIDEO_EXTENSIONS = (".webm", ".mkv", ".mov", ".m4v", ".ts")


class VideoDownloadError(RuntimeError):
    """Raised when video metadata extraction or segment download fails."""


def _validate_url(url: str) -> str:
    """Validate that the stream URL is a non-empty string."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Invalid URL: URL must be a non-empty string.")
    return url.strip()


def _validate_range(
    start_seconds: float,
    end_seconds: float,
    buffer_seconds: float,
) -> tuple[float, float, float]:
    """Validate segment bounds and buffer duration."""
    if (
        isinstance(start_seconds, bool)
        or isinstance(end_seconds, bool)
        or isinstance(buffer_seconds, bool)
    ):
        raise ValueError("Segment timestamps and buffer_seconds must be numeric, not bool.")

    start = float(start_seconds)
    end = float(end_seconds)
    buf = float(buffer_seconds)

    if not math.isfinite(start):
        raise ValueError(f"start_seconds must be finite, got {start_seconds!r}")
    if not math.isfinite(end):
        raise ValueError(f"end_seconds must be finite, got {end_seconds!r}")
    if not math.isfinite(buf):
        raise ValueError(f"buffer_seconds must be finite, got {buffer_seconds!r}")
    if start < 0.0:
        raise ValueError(f"start_seconds must be >= 0.0, got {start_seconds!r}")
    if end <= start:
        raise ValueError(
            f"end_seconds ({end_seconds!r}) must be greater than start_seconds ({start_seconds!r})"
        )
    if buf < 0.0:
        raise ValueError(f"buffer_seconds must be >= 0.0, got {buffer_seconds!r}")

    return start, end, buf


def _classify_video_error(url: str, exc: Exception) -> VideoDownloadError:
    """Wrap yt-dlp exceptions into a descriptive VideoDownloadError."""
    raw_message = str(exc).strip()
    lower = raw_message.lower()

    if any(
        token in lower
        for token in ("confirm you're not a bot", "confirm you’re not a bot")
    ):
        return VideoDownloadError(
            f"YouTube bot-check blocked the request; try cookies or retry later ({url}): {raw_message}"
        )

    if "http error 429" in lower or "too many requests" in lower:
        return VideoDownloadError(
            f"Rate limited by the server (HTTP 429); retry later ({url}): {raw_message}"
        )

    if any(
        token in lower
        for token in ("private video", "this video is private", "members-only", "members only")
    ):
        return VideoDownloadError(
            f"Video is private or restricted and cannot be downloaded ({url}): {raw_message}"
        )

    if any(
        token in lower
        for token in (
            "confirm your age",
            "age-restricted",
            "age restricted",
            "sign in to confirm your age",
            "login required",
            "use --cookies",
        )
    ):
        return VideoDownloadError(
            f"Video is age-restricted or requires authentication ({url}): {raw_message}"
        )

    if any(
        token in lower
        for token in (
            "not available in your country",
            "available in your country",
            "blocked it in your country",
            "geo-restricted",
            "geo restricted",
            "not available in your region",
        )
    ):
        return VideoDownloadError(
            f"Video is region-locked and unavailable in your location ({url}): {raw_message}"
        )

    if any(
        token in lower
        for token in (
            "video unavailable",
            "has been removed",
            "video has been deleted",
            "no longer available",
            "account associated with this video has been terminated",
            "isn't available",
        )
    ):
        return VideoDownloadError(
            f"Video is deleted or unavailable ({url}): {raw_message}"
        )

    if any(
        token in lower
        for token in (
            "is not a valid url",
            "not a valid url",
            "unsupported url",
            "incomplete youtube id",
            "truncated id",
            "unable to download webpage",
        )
    ):
        return VideoDownloadError(
            f"Invalid or unsupported stream URL ({url}): {raw_message}"
        )

    return VideoDownloadError(f"Failed to download video segment from {url}: {raw_message}")


def _extract_video_id(url: str) -> str:
    """Resolve the canonical video ID without downloading media."""
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "retries": _DEFAULT_RETRIES,
        "fragment_retries": _DEFAULT_RETRIES,
        "extractor_retries": _DEFAULT_RETRIES,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except (yt_dlp.utils.DownloadError, yt_dlp.utils.ExtractorError) as exc:
        raise _classify_video_error(url, exc) from exc

    if not isinstance(info, dict) or not info.get("id"):
        raise VideoDownloadError(f"Could not determine video ID for URL: {url}")

    return str(info["id"])


def _resolve_video_id(url: str, video_id: str | None) -> str:
    """Use a caller-provided id, else parse offline, else fall back to network extraction."""
    if video_id:
        return re.sub(r"[^A-Za-z0-9_-]", "_", str(video_id))
    local_id = _extract_youtube_id_from_url(url)
    if local_id:
        return local_id
    return _extract_video_id(url)


def _segment_sidecar_matches(
    sidecar_path: Path, start_seconds: float, end_seconds: float, buffer_seconds: float
) -> bool:
    """Return True only if the sidecar exists and matches requested bounds & buffer."""
    if not sidecar_path.is_file():
        return False
    try:
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return (
            abs(float(meta["start_seconds"]) - start_seconds) < 1e-3
            and abs(float(meta["end_seconds"]) - end_seconds) < 1e-3
            and abs(float(meta["buffer_seconds"]) - buffer_seconds) < 1e-3
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return False


def _download_segment_with_id(
    url: str,
    video_id: str,
    start_seconds: float,
    end_seconds: float,
    output_dir: Path,
    buffer_seconds: float,
) -> Path:
    """Download a single range-bounded MP4 segment using a pre-resolved video_id."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / (
        f"{video_id}_{round(start_seconds * 1000)}_{round(end_seconds * 1000)}"
        f"_b{round(buffer_seconds * 1000)}.mp4"
    )
    sidecar_path = output_path.with_suffix(".meta.json")
    sidecar_matches = _segment_sidecar_matches(
        sidecar_path, start_seconds, end_seconds, buffer_seconds
    )
    if sidecar_matches:
        if output_path.is_file():
            logger.info("Cached video segment found, skipping download: %s", output_path)
            return output_path
        if sidecar_path.is_file():
            for alt_ext in _ALT_VIDEO_EXTENSIONS:
                alt_cached = output_path.with_suffix(alt_ext)
                if alt_cached.is_file():
                    logger.info(
                        "Cached alternate-container video segment found, skipping download: %s",
                        alt_cached,
                    )
                    return alt_cached
    else:
        # Sidecar metadata mismatched requested bounds/buffer: remove stale files before re-downloading
        # so an old .mp4 cannot mask a newly downloaded alternate container (.webm/.mkv).
        for stale in [output_path, *(output_path.with_suffix(ext) for ext in _ALT_VIDEO_EXTENSIONS)]:
            if stale.is_file():
                try:
                    stale.unlink()
                except OSError:
                    pass

    actual_start = max(0.0, start_seconds - buffer_seconds)
    actual_end = end_seconds + buffer_seconds

    ydl_opts: dict[str, Any] = {
        "format": VIDEO_FORMAT_1080P,
        "merge_output_format": "mp4",
        "postprocessors": [
            {
                "key": "FFmpegVideoRemuxer",
                "preferedformat": "mp4",
            }
        ],
        "outtmpl": str(output_path.with_suffix(".%(ext)s")),
        "download_ranges": yt_dlp.utils.download_range_func(
            None, [(actual_start, actual_end)]
        ),
        "force_keyframes_at_cuts": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
        "retries": _DEFAULT_RETRIES,
        "fragment_retries": _DEFAULT_RETRIES,
        "extractor_retries": _DEFAULT_RETRIES,
    }

    logger.info(
        "Downloading video segment %s [%.2fs..%.2fs] (padded [%.2fs..%.2fs]) -> %s",
        video_id,
        start_seconds,
        end_seconds,
        actual_start,
        actual_end,
        output_path,
    )
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except (
        yt_dlp.utils.DownloadError,
        yt_dlp.utils.ExtractorError,
        yt_dlp.utils.PostProcessingError,
    ) as exc:
        raise _classify_video_error(url, exc) from exc

    resolved_path = output_path
    if not resolved_path.is_file():
        for alt_ext in _ALT_VIDEO_EXTENSIONS:
            candidate = output_path.with_suffix(alt_ext)
            if candidate.is_file():
                logger.warning(
                    "yt-dlp wrote alternate container %s instead of %s; returning actual file.",
                    candidate.name,
                    output_path.name,
                )
                resolved_path = candidate
                break

    if resolved_path.is_file():
        try:
            sidecar_path.write_text(
                json.dumps(
                    {
                        "start_seconds": round(start_seconds, 3),
                        "end_seconds": round(end_seconds, 3),
                        "buffer_seconds": round(buffer_seconds, 3),
                    }
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    return resolved_path


def download_segment(
    url: str,
    start_seconds: float,
    end_seconds: float,
    output_dir: Path,
    buffer_seconds: float = 3.0,
    *,
    video_id: str | None = None,
) -> Path:
    """Download a specific time segment from a video as a 1080p MP4.

    Applies ``±buffer_seconds`` padding (with start clamped at ``0.0``) via
    ``yt_dlp.utils.download_range_func`` and ``force_keyframes_at_cuts=True``.
    Saves to ``{output_dir}/{video_id}_{int(start_seconds)}_{int(end_seconds)}.mp4``
    and skips downloading if that file already exists.
    """
    clean_url = _validate_url(url)
    start, end, buf = _validate_range(start_seconds, end_seconds, buffer_seconds)
    resolved_id = _resolve_video_id(clean_url, video_id)
    return _download_segment_with_id(
        url=clean_url,
        video_id=resolved_id,
        start_seconds=start,
        end_seconds=end,
        output_dir=output_dir,
        buffer_seconds=buf,
    )


def download_multiple_segments(
    url: str,
    segments: list[tuple[float, float]],
    output_dir: Path,
    buffer_seconds: float = 3.0,
    *,
    video_id: str | None = None,
) -> list[Path]:
    """Download multiple time segments from the same video efficiently.

    Resolves the ``video_id`` once for the stream and downloads each uncached
    ``(start_seconds, end_seconds)`` segment in order with ``±buffer_seconds``
    padding, enabling Phase 4 conditional Multi-Beat Story Mode without
    downloading dead air between beats.
    """
    clean_url = _validate_url(url)
    if isinstance(buffer_seconds, bool) or not math.isfinite(float(buffer_seconds)) or float(buffer_seconds) < 0.0:
        raise ValueError(f"buffer_seconds must be a finite number >= 0.0, got {buffer_seconds!r}")

    if not segments:
        return []

    validated_segments = [
        _validate_range(start, end, buffer_seconds) for start, end in segments
    ]
    resolved_id = _resolve_video_id(clean_url, video_id)

    paths: list[Path] = []
    for start, end, buf in validated_segments:
        path = _download_segment_with_id(
            url=clean_url,
            video_id=resolved_id,
            start_seconds=start,
            end_seconds=end,
            output_dir=output_dir,
            buffer_seconds=buf,
        )
        paths.append(path)

    return paths
