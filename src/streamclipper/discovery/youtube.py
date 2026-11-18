"""YouTube stream discovery using yt-dlp ytsearch."""
from __future__ import annotations

import logging
from typing import Any

import yt_dlp

from ..config import Config
from ..models import Platform, StreamInfo

logger = logging.getLogger(__name__)


def _coerce_non_negative_int(value: Any) -> int | None:
    """Safely coerce a metadata field to a non-negative int, or return None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _resolve_stream_url(entry: dict[str, Any], video_id: str) -> str:
    """Extract a canonical YouTube watch URL from a yt-dlp search entry."""
    for key in ("webpage_url", "url"):
        candidate = entry.get(key)
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped.startswith(("http://", "https://")):
                return stripped
    return f"https://www.youtube.com/watch?v={video_id}"


def _resolve_streamer(entry: dict[str, Any]) -> str:
    """Resolve the streamer/channel name with safe fallbacks."""
    for key in ("uploader", "channel", "creator"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "Unknown"


class YouTubeDiscovery:
    """Discover high-view, long-form YouTube gaming streams via yt-dlp."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.default_game: str = str(config.get("discovery.default_game", "GTA 6"))
        self.min_viewers: int = int(config.get("discovery.min_viewers", 1000))
        self.max_streams: int = int(config.get("discovery.max_streams", 10))
        self.min_duration: int = int(
            float(config.get("discovery.min_duration_hours", 1)) * 3600
        )
        raw_keywords = config.get("discovery.game_keywords", []) or []
        self.keywords: list[str] = [
            str(kw).strip() for kw in raw_keywords if isinstance(kw, str) and kw.strip()
        ]

    def _build_queries(self, game: str | None) -> tuple[str, list[str]]:
        """Resolve the target game label and deduplicated search queries."""
        explicit_game = game.strip() if isinstance(game, str) and game.strip() else None
        target_game = explicit_game or self.default_game.strip() or "GTA 6"

        use_configured_keywords = (
            explicit_game is None
            or target_game.lower() == self.default_game.strip().lower()
        )
        raw_queries = (
            self.keywords if (use_configured_keywords and self.keywords) else [target_game]
        )

        queries: list[str] = []
        seen_lower: set[str] = set()
        for q in raw_queries:
            cleaned = q.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in seen_lower:
                seen_lower.add(key)
                queries.append(cleaned)

        return target_game, queries or [target_game]

    def discover(self, game: str | None = None) -> list[StreamInfo]:
        """Find top recent streams for a game."""
        target_game, queries = self._build_queries(game)

        all_results: list[dict[str, Any]] = []
        for query in queries:
            all_results.extend(self._search_youtube(query))

        return self._filter_streams(all_results, target_game)

    def _search_youtube(self, query: str, max_results: int = 20) -> list[dict[str, Any]]:
        """Raw yt-dlp search using extract_flat=False."""
        clean_query = query.strip()
        if not clean_query or max_results <= 0:
            return []

        search_term = f"ytsearch{max_results}:{clean_query} live stream"
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "skip_download": True,
            "ignoreerrors": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_term, download=False)
        except (yt_dlp.utils.DownloadError, yt_dlp.utils.ExtractorError) as exc:
            logger.warning("YouTube discovery search failed for %r: %s", search_term, exc)
            return []

        if not isinstance(info, dict):
            return []

        entries = info.get("entries")
        if not isinstance(entries, list):
            return []

        return [entry for entry in entries if isinstance(entry, dict)]

    def _filter_streams(self, results: list[dict[str, Any]], game: str) -> list[StreamInfo]:
        """Filter for actual streams (long duration, high views)."""
        if self.max_streams <= 0:
            return []

        seen_ids: set[str] = set()
        streams: list[StreamInfo] = []

        for entry in results:
            if not isinstance(entry, dict):
                continue

            video_id = str(entry.get("id") or "").strip()
            if not video_id or video_id in seen_ids:
                continue

            duration = _coerce_non_negative_int(entry.get("duration"))
            if duration is None or duration < self.min_duration:
                continue

            view_count = _coerce_non_negative_int(entry.get("view_count"))
            if view_count is None or view_count < self.min_viewers:
                continue

            seen_ids.add(video_id)
            streams.append(
                StreamInfo(
                    url=_resolve_stream_url(entry, video_id),
                    title=str(entry.get("title") or ""),
                    platform=Platform.YOUTUBE,
                    streamer=_resolve_streamer(entry),
                    game=game,
                    duration_seconds=duration,
                    view_count=view_count,
                    video_id=video_id,
                )
            )

        streams.sort(key=lambda s: s.view_count, reverse=True)
        return streams[: self.max_streams]
