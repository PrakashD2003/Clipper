"""Selective video segment downloader using yt-dlp download_ranges."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yt_dlp

logger = logging.getLogger(__name__)

VIDEO_FORMAT_1080P = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"


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
    start = float(start_seconds)
    end = float(end_seconds)
    buf = float(buffer_seconds)

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
            "sign in",
            "login required",
            "cookies",
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
            "region",
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
            "deleted",
            "no longer available",
            "terminated",
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
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except (yt_dlp.utils.DownloadError, yt_dlp.utils.ExtractorError) as exc:
        raise _classify_video_error(url, exc) from exc

    if not isinstance(info, dict) or not info.get("id"):
        raise VideoDownloadError(f"Could not determine video ID for URL: {url}")

    return str(info["id"])


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

    output_path = out_dir / f"{video_id}_{int(start_seconds)}_{int(end_seconds)}.mp4"
    if output_path.is_file():
        logger.info("Cached video segment found, skipping download: %s", output_path)
        return output_path

    actual_start = max(0.0, start_seconds - buffer_seconds)
    actual_end = end_seconds + buffer_seconds

    ydl_opts: dict[str, Any] = {
        "format": VIDEO_FORMAT_1080P,
        "merge_output_format": "mp4",
        "outtmpl": str(output_path.with_suffix(".%(ext)s")),
        "download_ranges": yt_dlp.utils.download_range_func(
            None, [(actual_start, actual_end)]
        ),
        "force_keyframes_at_cuts": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
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

    return output_path


def download_segment(
    url: str,
    start_seconds: float,
    end_seconds: float,
    output_dir: Path,
    buffer_seconds: float = 3.0,
) -> Path:
    """Download a specific time segment from a video as a 1080p MP4.

    Applies ``±buffer_seconds`` padding (with start clamped at ``0.0``) via
    ``yt_dlp.utils.download_range_func`` and ``force_keyframes_at_cuts=True``.
    Saves to ``{output_dir}/{video_id}_{int(start_seconds)}_{int(end_seconds)}.mp4``
    and skips downloading if that file already exists.
    """
    clean_url = _validate_url(url)
    start, end, buf = _validate_range(start_seconds, end_seconds, buffer_seconds)
    video_id = _extract_video_id(clean_url)
    return _download_segment_with_id(
        url=clean_url,
        video_id=video_id,
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
) -> list[Path]:
    """Download multiple time segments from the same video efficiently.

    Resolves the ``video_id`` once for the stream and downloads each uncached
    ``(start_seconds, end_seconds)`` segment in order with ``±buffer_seconds``
    padding, enabling Phase 4 conditional Multi-Beat Story Mode without
    downloading dead air between beats.
    """
    clean_url = _validate_url(url)
    if float(buffer_seconds) < 0.0:
        raise ValueError(f"buffer_seconds must be >= 0.0, got {buffer_seconds!r}")

    if not segments:
        return []

    validated_segments = [
        _validate_range(start, end, buffer_seconds) for start, end in segments
    ]
    video_id = _extract_video_id(clean_url)

    paths: list[Path] = []
    for start, end, buf in validated_segments:
        path = _download_segment_with_id(
            url=clean_url,
            video_id=video_id,
            start_seconds=start,
            end_seconds=end,
            output_dir=output_dir,
            buffer_seconds=buf,
        )
        paths.append(path)

    return paths
