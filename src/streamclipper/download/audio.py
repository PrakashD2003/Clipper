"""Audio-only stream downloader producing 16kHz mono PCM WAV files via yt-dlp."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Callable

import yt_dlp

from ..models import Platform, StreamInfo

logger = logging.getLogger(__name__)

_DEFAULT_RETRIES = 3

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|live/|shorts/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})(?:[?&#/]|$)"
)


def _safe_non_negative_int(value: Any) -> int:
    """Safely coerce a metadata field to a non-negative int, returning 0 on invalid/non-finite inputs."""
    if value is None or isinstance(value, bool):
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(numeric) or numeric < 0.0:
        return 0
    return int(numeric)


def _extract_youtube_id_from_url(url: str) -> str | None:
    """Extract an 11-character YouTube video ID locally without network calls, if recognizable."""
    match = _YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


class AudioDownloadError(RuntimeError):
    """Raised when stream metadata extraction or audio download fails."""


def _validate_url(url: str) -> str:
    """Ensure the provided URL is a non-empty string and strip surrounding whitespace."""
    if not isinstance(url, str) or not url.strip():
        raise AudioDownloadError("Invalid URL: URL must be a non-empty string.")
    return url.strip()


def _classify_download_error(url: str, exc: Exception) -> AudioDownloadError:
    """Translate yt-dlp exceptions into descriptive, user-facing AudioDownloadError messages."""
    raw_message = str(exc).strip()
    lower = raw_message.lower()

    if any(
        token in lower
        for token in ("confirm you're not a bot", "confirm you’re not a bot")
    ):
        return AudioDownloadError(
            f"YouTube bot-check blocked the request; try cookies or retry later ({url}): {raw_message}"
        )

    if "http error 429" in lower or "too many requests" in lower:
        return AudioDownloadError(
            f"Rate limited by the server (HTTP 429); retry later ({url}): {raw_message}"
        )

    if any(
        token in lower
        for token in ("private video", "this video is private", "members-only", "members only")
    ):
        return AudioDownloadError(
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
        return AudioDownloadError(
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
        return AudioDownloadError(
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
        return AudioDownloadError(
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
        return AudioDownloadError(
            f"Invalid or unsupported stream URL ({url}): {raw_message}"
        )

    return AudioDownloadError(f"Failed to download audio from {url}: {raw_message}")


def _detect_platform(info: dict[str, Any], url: str) -> Platform:
    """Infer the streaming platform from yt-dlp extractor metadata or URL."""
    extractor = str(info.get("extractor_key") or info.get("extractor") or "").lower()
    url_lower = url.lower()

    if "twitch" in extractor or "twitch.tv" in url_lower:
        return Platform.TWITCH
    if "kick" in extractor or "kick.com" in url_lower:
        return Platform.KICK
    return Platform.YOUTUBE


def _info_dict_to_stream_info(info: dict[str, Any], url: str) -> StreamInfo:
    """Convert a yt-dlp info dictionary into a StreamInfo dataclass."""
    title = str(info.get("title") or "")
    streamer = str(
        info.get("uploader") or info.get("channel") or info.get("creator") or "Unknown"
    )
    game = str(info.get("game") or "")

    duration_seconds = _safe_non_negative_int(info.get("duration"))
    view_count = _safe_non_negative_int(info.get("view_count"))

    video_id = str(info.get("id") or "")
    platform = _detect_platform(info, url)

    return StreamInfo(
        url=url,
        title=title,
        platform=platform,
        streamer=streamer,
        game=game,
        duration_seconds=duration_seconds,
        view_count=view_count,
        video_id=video_id,
    )


def _find_cookie_file() -> str | None:
    """Find a Netscape cookies.txt file in the environment or project tree."""
    import os

    env_val = os.environ.get("YOUTUBE_COOKIES_FILE") or os.environ.get("YT_COOKIES_FILE")
    if env_val and Path(env_val).is_file():
        return str(Path(env_val).resolve())
    for cand in (
        Path("cookies.txt"),
        Path(__file__).resolve().parents[3] / "cookies.txt",
    ):
        if cand.is_file():
            return str(cand.resolve())
    return None


def extract_stream_info(url: str) -> StreamInfo:
    """Extract stream metadata without downloading media."""
    clean_url = _validate_url(url)
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "retries": _DEFAULT_RETRIES,
        "fragment_retries": _DEFAULT_RETRIES,
        "extractor_retries": _DEFAULT_RETRIES,
    }
    cookie_file = _find_cookie_file()
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
    except (yt_dlp.utils.DownloadError, yt_dlp.utils.ExtractorError) as exc:
        raise _classify_download_error(clean_url, exc) from exc

    if not isinstance(info, dict):
        raise AudioDownloadError(f"Could not extract stream metadata for URL: {clean_url}")

    if info.get("live_status") in {"is_live", "is_upcoming"} or info.get("is_live") is True:
        raise AudioDownloadError(
            f"Stream is live or upcoming and cannot be downloaded yet ({clean_url})"
        )

    return _info_dict_to_stream_info(info, clean_url)


def download_audio(
    url: str,
    output_dir: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, StreamInfo]:
    """Download 16kHz mono PCM WAV audio and extract stream metadata.

    Returns ``(audio_path, stream_info)``. If ``{output_dir}/{video_id}.wav``
    already exists on disk, skips downloading and returns the cached path.
    """
    clean_url = _validate_url(url)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Offline cache fast-path: if local YouTube ID matches both cached WAV and sidecar StreamInfo,
    # avoid a redundant live yt-dlp metadata network call.
    local_vid = _extract_youtube_id_from_url(clean_url)
    if local_vid:
        cached_wav = out_dir / f"{local_vid}.wav"
        cached_meta = out_dir / f"{local_vid}.info.json"
        if cached_wav.is_file() and cached_meta.is_file():
            try:
                loaded_info = StreamInfo.from_dict(
                    json.loads(cached_meta.read_text(encoding="utf-8"))
                )
                loaded_info.url = clean_url
                logger.info(
                    "Cached 16kHz WAV audio + metadata found, skipping network call: %s",
                    cached_wav,
                )
                return cached_wav, loaded_info
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
                pass

    stream_info = extract_stream_info(clean_url)
    if not stream_info.video_id:
        raise AudioDownloadError(f"Could not determine video ID for URL: {clean_url}")

    audio_path = out_dir / f"{stream_info.video_id}.wav"
    meta_sidecar = out_dir / f"{stream_info.video_id}.info.json"
    if audio_path.is_file():
        logger.info("Cached 16kHz WAV audio found, skipping download: %s", audio_path)
        try:
            meta_sidecar.write_text(stream_info.to_json(), encoding="utf-8")
        except OSError:
            pass
        return audio_path, stream_info

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": _DEFAULT_RETRIES,
        "fragment_retries": _DEFAULT_RETRIES,
        "extractor_retries": _DEFAULT_RETRIES,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "postprocessor_args": {
            "extractaudio": ["-ar", "16000", "-ac", "1"],
        },
    }
    cookie_file = _find_cookie_file()
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file
    if progress_callback is not None:
        ydl_opts["progress_hooks"] = [progress_callback]

    logger.info("Downloading 16kHz mono WAV audio for %s -> %s", stream_info.video_id, audio_path)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clean_url])
    except (
        yt_dlp.utils.DownloadError,
        yt_dlp.utils.ExtractorError,
        yt_dlp.utils.PostProcessingError,
    ) as exc:
        raise _classify_download_error(clean_url, exc) from exc

    if audio_path.is_file():
        try:
            meta_sidecar.write_text(stream_info.to_json(), encoding="utf-8")
        except OSError:
            pass

    return audio_path, stream_info
