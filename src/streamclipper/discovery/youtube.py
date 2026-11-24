"""YouTube stream discovery using yt-dlp ytsearch."""
from __future__ import annotations

import calendar
import logging
import math
import re
import time
from typing import Any

import yt_dlp

from ..config import Config
from ..models import Platform, StreamInfo

logger = logging.getLogger(__name__)


def _coerce_non_negative_int(value: Any) -> int | None:
    """Safely coerce a metadata field to a non-negative finite int, or return None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0.0:
        return None
    return int(numeric)


# Language codes accepted by yt-dlp's YouTube extractor `lang` arg (case-sensitive).
# Passing anything else (e.g. "en-US") makes every search fail with
# "Unsupported language code", which `ignoreerrors` silently turns into 0 results.
_YTDLP_YOUTUBE_LANGS: frozenset[str] = frozenset(
    "af az id ms bs ca cs da de et en-IN en-GB en es es-419 es-US eu fil fr fr-CA "
    "gl hr zu is it sw lv lt hu nl no uz pl pt-PT pt ro sq sk sl sr-Latn fi sv vi "
    "tr be bg ky kk mk mn ru sr uk el hy iw ur ar fa ne mr hi as bn pa gu or ta te "
    "kn ml si th lo my ka am km zh-CN zh-TW zh-HK ja ko".split()
)
_YTDLP_LANGS_BY_LOWER: dict[str, str] = {code.lower(): code for code in _YTDLP_YOUTUBE_LANGS}


def _to_ytdlp_youtube_lang(language: str) -> str:
    """Map a BCP-47 tag (e.g. "en-US", "ja_JP") to a yt-dlp-supported YouTube lang code."""
    tag = (language or "").strip().replace("_", "-")
    exact = _YTDLP_LANGS_BY_LOWER.get(tag.lower())
    if exact:
        return exact
    base = tag.split("-", 1)[0].lower()
    if base == "he":
        base = "iw"
    return _YTDLP_LANGS_BY_LOWER.get(base, "en")


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
        raw_lang = config.get("discovery.language", "en-US")
        self.language: str = str(raw_lang).strip() if raw_lang is not None else "en-US"
        if not self.language:
            self.language = "en-US"
        raw_region = config.get("discovery.region", "US")
        self.region: str = str(raw_region).strip() if raw_region is not None else "US"
        if not self.region:
            self.region = "US"
        raw_proxy = config.get("discovery.proxy", "")
        self.proxy: str = str(raw_proxy or "").strip()
        self.max_age_days: float = float(config.get("discovery.max_age_days", 30))
        self.require_title_match: bool = bool(
            config.get("discovery.require_title_match", True)
        )

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

    def discover(
        self,
        game: str | None = None,
        max_streams: int | None = None,
        *,
        language: str | None = None,
        region: str | None = None,
    ) -> list[StreamInfo]:
        """Find top recent streams for a game."""
        effective_max = (
            self.max_streams if max_streams is None else max(0, int(max_streams))
        )
        if effective_max <= 0:
            return []

        effective_lang = (
            str(language).strip()
            if language is not None and str(language).strip()
            else self.language
        )
        effective_region = (
            str(region).strip()
            if region is not None and str(region).strip()
            else self.region
        )

        target_game, queries = self._build_queries(game)
        per_query_limit = max(20, effective_max * 2)

        all_results: list[dict[str, Any]] = []
        for query in queries:
            all_results.extend(
                self._search_youtube(
                    query,
                    max_results=per_query_limit,
                    language=effective_lang,
                    region=effective_region,
                )
            )

        return self._filter_streams(all_results, target_game, max_streams=effective_max)

    def _search_youtube(
        self,
        query: str,
        max_results: int = 20,
        *,
        language: str | None = None,
        region: str | None = None,
    ) -> list[dict[str, Any]]:
        """Raw yt-dlp search using extract_flat=False."""
        clean_query = query.strip()
        if not clean_query or max_results <= 0:
            return []

        effective_lang = (
            str(language).strip()
            if language is not None and str(language).strip()
            else self.language
        )
        effective_region = (
            str(region).strip()
            if region is not None and str(region).strip()
            else self.region
        )

        if effective_lang.lower().startswith("hi"):
            search_term = f"ytsearch{max_results}:{clean_query} hindi live stream"
        else:
            search_term = f"ytsearch{max_results}:{clean_query} live stream"

        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "skip_download": True,
            "ignoreerrors": True,
            "http_headers": {"Accept-Language": f"{effective_lang},en;q=0.9"},
            "geo_bypass_country": effective_region,
            "extractor_args": {
                "youtube": {"lang": [_to_ytdlp_youtube_lang(effective_lang)]}
            },
        }
        if self.proxy:
            ydl_opts["proxy"] = self.proxy

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

    def _is_too_old(self, entry: dict[str, Any]) -> bool:
        """Return True when the entry's known publish date is older than max_age_days."""
        if self.max_age_days <= 0:
            return False
        published: float | None = None
        ts = entry.get("release_timestamp")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool) and math.isfinite(ts):
            published = float(ts)
        else:
            upload_date = entry.get("upload_date")
            if isinstance(upload_date, str) and re.fullmatch(r"\d{8}", upload_date.strip()):
                try:
                    published = calendar.timegm(
                        time.strptime(upload_date.strip(), "%Y%m%d")
                    )
                except ValueError:
                    published = None
        if published is None:
            return False
        return (time.time() - published) > self.max_age_days * 86400.0

    def _title_matches(self, title: str, game: str) -> bool:
        """Require at least one game keyword token (len >= 3) in the title."""
        tokens: set[str] = set()
        for phrase in [game, *self.keywords]:
            for tok in re.split(r"[^0-9a-zA-Z]+", str(phrase).lower()):
                if len(tok) >= 3:
                    tokens.add(tok)
        if not tokens:
            return True
        lower_title = title.lower()
        return any(tok in lower_title for tok in tokens)

    def _filter_streams(
        self,
        results: list[dict[str, Any]],
        game: str,
        max_streams: int | None = None,
    ) -> list[StreamInfo]:
        """Filter for actual streams (long duration, high views)."""
        limit = self.max_streams if max_streams is None else max(0, int(max_streams))
        if limit <= 0:
            return []

        seen_ids: set[str] = set()
        streams: list[StreamInfo] = []
        raw_blocked = self.config.get("discovery.blocked_streamers", []) or []
        if isinstance(raw_blocked, str):
            raw_blocked = [raw_blocked]
        blocked = {str(b).strip().lower() for b in raw_blocked if str(b).strip()}

        for entry in results:
            if not isinstance(entry, dict):
                continue

            video_id = str(entry.get("id") or "").strip()
            if not video_id or video_id in seen_ids:
                continue

            if blocked:
                names = {
                    str(entry.get(k) or "").strip().lower()
                    for k in ("uploader", "channel", "creator")
                }
                if names & blocked:
                    continue

            duration = _coerce_non_negative_int(entry.get("duration"))
            if duration is None or duration < self.min_duration:
                continue

            view_count = _coerce_non_negative_int(entry.get("view_count"))
            if view_count is None or view_count < self.min_viewers:
                continue

            live_status = entry.get("live_status")
            if live_status is not None and live_status != "was_live":
                continue
            if live_status is None and entry.get("was_live") is not None and not entry.get(
                "was_live"
            ):
                continue

            if self._is_too_old(entry):
                continue

            if self.require_title_match and not self._title_matches(
                str(entry.get("title") or ""), game
            ):
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
        return streams[:limit]
