"""MCP SDK v2 server exposing StreamClipper pipeline capabilities as MCP tools."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, AsyncIterator
import weakref

from mcp.server.mcpserver import MCPServer

from ..analysis.chat_velocity import ChatVelocityAnalyzer
from ..analysis.loudness import LoudnessDetector
from ..analysis.transcriber import Transcriber
from ..cache import StepCache
from ..config import Config
from ..language import language_instruction, resolve_narration_language
from ..narration_style import check_narration, narration_guide
from ..discovery.youtube import YouTubeDiscovery
from ..download.audio import _extract_youtube_id_from_url, download_audio
from ..download.chat import ChatScraper
from ..download.video import download_multiple_segments, download_segment
from ..models import (
    ClipMetadata,
    ClipResult,
    Highlight,
    PipelineState,
    PipelineStatus,
    Platform,
    StreamInfo,
    TimeRange,
)
from ..output.organizer import FileOrganizer
from ..pipeline import (
    Pipeline,
    _acquire_file_lock,
    _release_file_lock,
    _sanitize_path_segment,
)
from ..scoring.ollama_scorer import (
    VALID_MOMENT_TYPES,
    OllamaScorer,
    _as_str,
    _clean_llm_text,
    _parse_score,
    _parse_seconds,
)
from ..scoring.scorer import Scorer, TranscriptChunk

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = (
    "Always render with render_clip / render_all_selected. Never assemble clips yourself with "
    "ffmpeg, subtitle files or system TTS voices — those outputs skip the narration, captions, "
    "loudness and copyright checks. "
    "Workflow: download_stream_audio -> transcribe_audio -> analyze_signals -> "
    "get_transcript_chunks -> submit_highlights -> render_all_selected (or render_clip). "
    "Required fields: every non-raw highlight needs a non-empty narration_intro AND "
    "narration_outro (1-2 short sentences, grounded in the transcript, in the reported narration "
    "language). Write narration following narration_guide from get_transcript_chunks "
    "(energetic, interactive gamer voice, grounded in the transcript); generic, ungrounded or "
    "non-interactive lines are rejected with a problems list so you can rewrite and resubmit. "
    "submit_highlights requires a transcript (run transcribe_audio first) and rejects "
    "highlights that start before analysis.skip_intro_minutes. Use mode 'breakdown' (instant "
    "replay) ONLY for moment_type clutch, fail, graphics_detail or analysis with a "
    "breakdown_commentary; reactions (hype, funny, chat_chaos, lore_secret) get narrative/story."
)

mcp = MCPServer("streamclipper", instructions=SERVER_INSTRUCTIONS)

_LOOP_GPU_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Lock
] = weakref.WeakKeyDictionary()
_LOOP_URL_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()


def _get_gpu_lock() -> asyncio.Lock:
    """Return the shared GPU serialization lock bound to the active event loop."""
    loop = asyncio.get_running_loop()
    lock = _LOOP_GPU_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _LOOP_GPU_LOCKS[loop] = lock
    return lock


@contextlib.asynccontextmanager
async def _gpu_guard(config: Config) -> AsyncIterator[None]:
    """Hold the in-process GPU lock and (when enabled) cross-process file lock."""
    async with _get_gpu_lock():
        if not bool(config.get("general.cross_process_gpu_lock", False)):
            yield
            return
        lock_dir = Path(config.get("general.cache_dir", "./.cache"))
        fh = await asyncio.to_thread(_acquire_file_lock, lock_dir)
        try:
            yield
        finally:
            await asyncio.to_thread(_release_file_lock, fh)


def _get_url_lock(cache_key: str) -> asyncio.Lock:
    """Return a per-stream cache lock bound to the active event loop."""
    loop = asyncio.get_running_loop()
    locks = _LOOP_URL_LOCKS.get(loop)
    if locks is None:
        locks = {}
        _LOOP_URL_LOCKS[loop] = locks
    lock = locks.get(cache_key)
    if lock is None:
        lock = asyncio.Lock()
        locks[cache_key] = lock
    return lock

_DEFAULT_CLIP_HASHTAGS: tuple[str, ...] = (
    "Gaming",
    "GamingClips",
    "StreamClips",
    "Shorts",
    "YouTubeShorts",
    "Livestream",
    "Highlights",
    "Viral",
    "Clips",
    "FYP",
    "GamingCommunity",
    "Streamer",
    "FunnyMoments",
)


class _MCPScorer(Scorer):
    """Concrete Scorer exposing shared chunking, signal fusion, and YPP helpers without Ollama."""

    def _score_chunk(
        self,
        chunk: TranscriptChunk,
        loudness: list[TimeRange],
        chat: list[TimeRange],
    ) -> list[Highlight]:
        return []


def _get_config() -> Config:
    """Load the active StreamClipper configuration."""
    return Config()


def _get_cache(config: Config) -> StepCache:
    """Create a StepCache backed by the configured cache directory."""
    return StepCache(config)


def _json_ok(payload: dict[str, Any]) -> str:
    """Serialize a tool result dict to a formatted JSON string."""
    return json.dumps(payload, indent=2)


def _state_narration_language(config: Config, state: Any) -> str:
    """Resolve english/hinglish from a cached PipelineState (streamer + transcript language)."""
    if state is None:
        return resolve_narration_language(config, None, None)
    transcript = getattr(state, "transcript", None)
    return resolve_narration_language(
        config,
        getattr(state.stream_info, "streamer", None),
        getattr(transcript, "language", None) if transcript is not None else None,
    )


def _variety_state(config: Config) -> Any:
    from ..output.variety import VarietyState

    return VarietyState(
        Path(config.get("general.cache_dir", "./.cache")) / "variety_state.json",
        hook_styles=list(config.get("transformation.hook_styles", None) or []) or None,
    )


def _recent_outros(config: Config, n: int = 20) -> list[str]:
    """Recent outro lines from the VarietyState (empty when none / unreadable)."""
    try:
        return _variety_state(config).recent_outros(n)
    except Exception:
        return []


def _next_hook_style(config: Config) -> str:
    try:
        return _variety_state(config).next_hook_style()
    except Exception:
        return ""


def _chat_texts(messages: list[Any]) -> list[str]:
    out: list[str] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            text = _as_str(msg.get("text") or msg.get("message"))
            if text:
                out.append(text)
    return out


def _json_error(message: str) -> str:
    """Serialize a structured JSON error payload."""
    return json.dumps({"error": message}, indent=2)


def _resolve_state(
    cache: StepCache, url: str = "", audio_path: str = ""
) -> PipelineState | None:
    """Resolve a cached PipelineState by stream URL or via StepCache.find_by_audio_path."""
    clean_url = url.strip() if isinstance(url, str) else ""
    clean_audio = audio_path.strip() if isinstance(audio_path, str) else ""

    if clean_url:
        state = cache.load_state(clean_url)
        if state is not None:
            return state

    if clean_audio:
        state = cache.find_by_audio_path(clean_audio)
        if state is not None:
            return state

    return None


def _compute_signal_overlaps(
    loudness_spikes: list[TimeRange], chat_spikes: list[TimeRange]
) -> list[dict[str, Any]]:
    """Compute time intervals where loudness and chat velocity spikes overlap."""
    overlaps: list[dict[str, Any]] = []
    for ls in loudness_spikes:
        for cs in chat_spikes:
            ov_start = max(ls.start, cs.start)
            ov_end = min(ls.end, cs.end)
            if ov_start < ov_end:
                overlaps.append(
                    {
                        "start": round(ov_start, 3),
                        "end": round(ov_end, 3),
                        "loudness_score": ls.score,
                        "chat_score": cs.score,
                        "combined_score": round((ls.score + cs.score) / 2.0, 3),
                    }
                )
    overlaps.sort(key=lambda item: item["combined_score"], reverse=True)
    return overlaps


def _has_valid_peak_time(spike: TimeRange) -> bool:
    """Return True if a loudness spike carries a valid finite numeric metadata['peak_time']."""
    candidate = (spike.metadata or {}).get("peak_time")
    if candidate is None or isinstance(candidate, bool):
        return False
    try:
        return math.isfinite(float(candidate))
    except (TypeError, ValueError):
        return False


def _normalize_moment_type(value: Any) -> str:
    """Normalize a moment_type string to a valid taxonomy value."""
    normalized = _as_str(value, "hype").lower()
    return normalized if normalized in VALID_MOMENT_TYPES else "hype"


def _parse_raw_sub_clips(raw_sub_clips: Any) -> list[dict[str, Any]]:
    """Parse raw beat dicts into normalized start/end/label dictionaries."""
    if not isinstance(raw_sub_clips, list):
        return []
    parsed: list[dict[str, Any]] = []
    for beat in raw_sub_clips:
        if not isinstance(beat, dict):
            return []
        b_start = _parse_seconds(beat.get("start"))
        b_end = _parse_seconds(beat.get("end"))
        if b_start is None or b_end is None or b_end <= b_start:
            return []
        parsed.append(
            {
                "start": b_start,
                "end": b_end,
                "label": _as_str(beat.get("label")),
            }
        )
    return parsed


def _finalize_hashtags_offline(
    config: Config, raw_tags: list[Any], streamer: str, game: str
) -> list[str]:
    """Normalize, deduplicate, enforce game/streamer tags, and cap to metadata.hashtag_count."""
    max_tags = max(0, int(config.get("metadata.hashtag_count", 15)))
    if max_tags == 0:
        return []

    hashtags: list[str] = []
    seen_lower: set[str] = set()

    def _append_tag(raw: Any) -> None:
        cleaned = re.sub(r"\W+", "", _as_str(raw).lstrip("#").strip())
        if cleaned and cleaned.lower() not in seen_lower:
            seen_lower.add(cleaned.lower())
            hashtags.append(cleaned)

    required_tags: list[str] = []
    if config.get("metadata.include_game_tag", True) and game:
        game_tag = re.sub(r"\W+", "", game)
        if game_tag:
            required_tags.append(game_tag)

    if config.get("metadata.include_streamer_tag", True) and streamer:
        streamer_tag = re.sub(r"\W+", "", streamer)
        if streamer_tag:
            required_tags.append(streamer_tag)

    for req in required_tags:
        _append_tag(req)

    for tag in raw_tags:
        _append_tag(tag)

    if len(hashtags) <= max_tags:
        return hashtags

    required_lower = {r.lower() for r in required_tags}
    kept_required = [t for t in hashtags if t.lower() in required_lower][:max_tags]
    remaining_slots = max_tags - len(kept_required)
    kept_regular = [t for t in hashtags if t.lower() not in required_lower][:remaining_slots]
    keep_set = {t.lower() for t in kept_required} | {t.lower() for t in kept_regular}
    return [t for t in hashtags if t.lower() in keep_set][:max_tags]


def _build_offline_post_metadata(
    config: Config,
    hook_title: str,
    streamer: str,
    game: str,
    description: str,
) -> dict[str, Any]:
    """Build social post caption draft and normalized hashtags without Ollama."""
    clean_hook = _as_str(hook_title) or "Stream Highlight"
    clean_streamer = _as_str(streamer)
    clean_game = _as_str(game)
    clean_desc = _as_str(description)

    game_tag = re.sub(r"\W+", "", clean_game)
    streamer_tag = re.sub(r"\W+", "", clean_streamer)

    candidate_tags: list[str] = []
    if game_tag:
        candidate_tags.extend([game_tag, f"{game_tag}Clips", f"{game_tag}Highlights"])
    if streamer_tag:
        candidate_tags.extend([streamer_tag, f"{streamer_tag}Clips"])
    candidate_tags.extend(_DEFAULT_CLIP_HASHTAGS)

    hashtags = _finalize_hashtags_offline(config, candidate_tags, clean_streamer, clean_game)
    hashtag_string = " ".join(f"#{tag}" for tag in hashtags)

    caption_parts: list[str] = [clean_hook]
    if clean_desc and clean_desc.lower() != clean_hook.lower():
        caption_parts.append(clean_desc)
    credit_parts: list[str] = []
    if clean_streamer:
        credit_parts.append(f"Streamer: {clean_streamer}")
    if clean_game:
        credit_parts.append(f"Game: {clean_game}")
    if credit_parts:
        caption_parts.append(" | ".join(credit_parts))
    caption = "\n".join(caption_parts)

    return {
        "hook_title": clean_hook,
        "caption": caption,
        "hashtags": hashtags,
        "hashtag_string": hashtag_string,
        "streamer": clean_streamer,
        "game": clean_game,
        "source": "offline",
    }


def _is_within_directory(path: Path, base_dir: Path) -> bool:
    """Return True if resolved path resides inside resolved base_dir."""
    try:
        path.resolve().relative_to(base_dir.resolve())
        return True
    except (ValueError, OSError):
        return False


def _is_safe_id_prefix(value: str) -> bool:
    """Return True if value is a safe single-segment filename prefix (no separators or '..')."""
    if not value or not value.strip():
        return False
    if "/" in value or "\\" in value or ":" in value or ".." in value:
        return False
    return True


_SEGMENT_RANGE_SUFFIX_RE = re.compile(r"^\d+_\d+\.")


def _collect_temp_files_for_stream(
    temp_dir: Path,
    cache: StepCache,
    url: str,
    state: PipelineState | None,
) -> list[Path]:
    """Identify temporary audio, chat, and segment files belonging to a stream.

    Uses exact filename prefix/suffix matching (never glob wildcard expansion) and
    enforces strict containment within ``temp_dir`` on every returned path.
    """
    candidates: list[Path] = []
    video_ids: set[str] = set()

    if state is not None and _is_safe_id_prefix(state.stream_info.video_id):
        video_ids.add(state.stream_info.video_id)

    extracted_id = _extract_youtube_id_from_url(url)
    if extracted_id and _is_safe_id_prefix(extracted_id):
        video_ids.add(extracted_id)

    cache_key = cache.get_cache_key(url) if url else ""

    if state is not None and state.audio_path is not None:
        audio_candidate = Path(state.audio_path)
        if _is_within_directory(audio_candidate, temp_dir):
            if audio_candidate.is_file():
                candidates.append(audio_candidate)
            if _is_safe_id_prefix(audio_candidate.stem):
                video_ids.add(audio_candidate.stem)

    if temp_dir.is_dir():
        for vid in video_ids:
            for direct_name in (f"{vid}.wav", f"{vid}.info.json", f"{vid}_chat.csv"):
                p = temp_dir / direct_name
                if p.is_file() and _is_within_directory(p, temp_dir):
                    candidates.append(p)

        if video_ids:
            for entry in temp_dir.iterdir():
                if not entry.is_file() or not _is_within_directory(entry, temp_dir):
                    continue
                if entry.name.endswith(".live_chat.json") and any(
                    entry.name.startswith(f"{vid}.") for vid in video_ids
                ):
                    candidates.append(entry)

        if cache_key and _is_safe_id_prefix(cache_key):
            key_chat = temp_dir / f"{cache_key}_chat.csv"
            if key_chat.is_file() and _is_within_directory(key_chat, temp_dir):
                candidates.append(key_chat)

        segments_dir = temp_dir / "segments"
        if segments_dir.is_dir() and video_ids:
            for seg_file in segments_dir.iterdir():
                if not seg_file.is_file() or not _is_within_directory(seg_file, temp_dir):
                    continue
                for vid in video_ids:
                    prefix = f"{vid}_"
                    if seg_file.name.startswith(prefix) and _SEGMENT_RANGE_SUFFIX_RE.match(
                        seg_file.name[len(prefix) :]
                    ):
                        candidates.append(seg_file)
                        break

    unique: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        if not _is_within_directory(p, temp_dir):
            continue
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _next_available_clip_index(
    config: Config, game: str, streamer: str, min_index: int = 1
) -> int:
    """Find the lowest clip index >= min_index that does not overwrite an existing output MP4.

    Terminates immediately if ``output.filename_template`` does not vary with ``{index}``.
    """
    organizer = FileOrganizer(config)
    idx = max(1, int(min_index))
    path = organizer.get_clip_path(game=game, streamer=streamer, index=idx)
    while path.exists():
        next_idx = idx + 1
        next_path = organizer.get_clip_path(game=game, streamer=streamer, index=next_idx)
        if next_path == path:
            break
        idx = next_idx
        path = next_path
    return idx


def _has_rendered_clip(clips: list[ClipResult], highlight: Highlight) -> bool:
    """True if a clip with matching start/end (+-0.01s) already exists on disk."""
    for clip in clips:
        meta = clip.metadata
        if (
            abs(float(meta.clip_start) - highlight.start) <= 0.01
            and abs(float(meta.clip_end) - highlight.end) <= 0.01
            and Path(clip.video_path).is_file()
        ):
            return True
    return False


def _merge_clip_results(
    existing: list[ClipResult], added: list[ClipResult]
) -> list[ClipResult]:
    """Merge newly rendered clips with existing cached clips, deduplicating by resolved video_path."""
    merged: list[ClipResult] = []
    seen_paths: dict[str, int] = {}
    for clip in [*existing, *added]:
        key = str(Path(clip.video_path).resolve()).lower()
        if key in seen_paths:
            merged[seen_paths[key]] = clip
        else:
            seen_paths[key] = len(merged)
            merged.append(clip)
    return merged


async def _render_single_highlight(
    pipeline: Pipeline,
    config: Config,
    url: str,
    highlight: Highlight,
    streamer_name: str,
    game_name: str,
    clip_index: int | None,
    base_mode: str,
    voice_intro_path: Path | None = None,
    voice_outro_path: Path | None = None,
    *,
    min_clip_index: int = 1,
    voice_breakdown_path: Path | None = None,
    layout: str = "",
    language: str = "english",
    notes: list[dict[str, Any]] | None = None,
) -> tuple[ClipResult, str]:
    """Download segment(s), build offline ClipMetadata, and assemble one highlight.

    When an explicit "breakdown" request is downgraded because the moment is unsuitable, a
    `{"mode_adjusted": ..., "reason": ...}` dict is appended to *notes* (if provided).
    """
    clip_mode, valid_beats = pipeline._resolve_highlight_mode(base_mode, highlight)
    if base_mode == "breakdown" and clip_mode != "breakdown" and notes is not None:
        notes.append(
            {
                "mode_adjusted": f"breakdown -> {clip_mode}",
                "reason": pipeline._breakdown_ineligible_reason(highlight),
            }
        )
    segments_dir = pipeline.temp_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    buffer_seconds = 3.0

    use_multi_beat = clip_mode == "story" and len(valid_beats) >= 2
    if use_multi_beat:
        beat_ranges = [
            (float(beat["start"]), float(beat["end"])) for beat in valid_beats
        ]
        segment_paths = await asyncio.to_thread(
            download_multiple_segments,
            url,
            beat_ranges,
            segments_dir,
            buffer_seconds,
        )
    else:
        single_path = await asyncio.to_thread(
            download_segment,
            url,
            highlight.start,
            highlight.end,
            segments_dir,
            buffer_seconds,
        )
        segment_paths = [single_path]

    post_meta = _build_offline_post_metadata(
        config,
        hook_title=highlight.title,
        streamer=streamer_name,
        game=game_name,
        description=highlight.reason or highlight.best_quote,
    )
    score = highlight.final_score if highlight.final_score > 0.0 else highlight.score
    metadata = ClipMetadata(
        hook_title=post_meta["hook_title"],
        caption=post_meta["caption"],
        hashtags=post_meta["hashtags"],
        streamer=streamer_name,
        game=game_name,
        source_url=url,
        clip_start=highlight.start,
        clip_end=highlight.end,
        highlight_score=score,
        moment_type=_normalize_moment_type(highlight.moment_type),
        narration_intro="" if clip_mode == "raw" else _as_str(highlight.narration_intro),
        narration_outro="" if clip_mode == "raw" else _as_str(highlight.narration_outro),
        context_badge="" if clip_mode == "raw" else _as_str(highlight.context_badge),
        transformation_mode=clip_mode,
        breakdown_commentary="" if clip_mode == "raw" else _as_str(highlight.breakdown_commentary),
        breakdown_callout="" if clip_mode == "raw" else _as_str(highlight.breakdown_callout),
        music_risk=highlight.music_risk,
        music_confidence=highlight.music_confidence,
        narration_language=language,
    )
    # `callout` lands on ClipMetadata in Batch D; set it as an attribute so this works pre/post merge.
    metadata.callout = "" if clip_mode == "raw" else _as_str(getattr(highlight, "callout", ""))

    extra_kwargs: dict[str, Any] = {}
    if layout:
        extra_kwargs["layout_kind"] = layout

    async with _gpu_guard(config):
        effective_index = (
            clip_index
            if clip_index is not None
            else _next_available_clip_index(
                config, game_name, streamer_name, min_index=min_clip_index
            )
        )
        clip_result = await asyncio.to_thread(
            pipeline._assemble_highlight,
            highlight,
            metadata,
            segment_paths,
            effective_index,
            clip_mode,
            voice_intro_path,
            voice_outro_path,
            valid_beats=valid_beats,
            buffer_seconds=buffer_seconds,
            voice_breakdown_path=voice_breakdown_path,
            **extra_kwargs,
        )

    if getattr(clip_result, "quarantined", False):
        logger.warning(
            "Clip quarantined (%s); skipping post file: %s",
            getattr(clip_result, "quarantine_reason", ""),
            clip_result.video_path,
        )
    elif config.get("output.write_post_file", True):
        try:
            from ..output.post_writer import write_post_file
        except ImportError:
            write_post_file = None  # module not merged yet
        if write_post_file is not None:
            try:
                stream_info = StreamInfo(
                    url=url,
                    title="",
                    platform=Platform.YOUTUBE,
                    streamer=streamer_name,
                    game=game_name,
                    duration_seconds=0,
                )
                await asyncio.to_thread(
                    write_post_file, Path(clip_result.video_path), metadata, stream_info
                )
            except Exception:
                logger.warning("write_post_file failed", exc_info=True)
    return clip_result, clip_mode


# ---------------------------------------------------------------------------
# MCP Tools (11 tools registered on MCPServer)
# ---------------------------------------------------------------------------


@mcp.tool()
async def discover_streams(game: str = "GTA 6", max_results: int = 10) -> str:
    """Find top recent streams on YouTube for a given game."""
    try:
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 0:
            return _json_error(
                f"max_results must be a non-negative integer, got {max_results!r}"
            )
        config = _get_config()
        discovery = YouTubeDiscovery(config)
        streams = await asyncio.to_thread(
            discovery.discover, game=game, max_streams=max_results
        )
        return _json_ok(
            {
                "game": game,
                "count": len(streams),
                "streams": [s.to_dict() for s in streams],
            }
        )
    except Exception as exc:
        logger.exception("discover_streams failed")
        return _json_error(str(exc))


@mcp.tool()
async def download_stream_audio(url: str) -> str:
    """Download 16kHz mono WAV audio from a YouTube stream/VOD URL and save initial StepCache state."""
    try:
        clean_url = url.strip() if isinstance(url, str) else ""
        if not clean_url:
            return _json_error("Invalid URL: url must be a non-empty string.")

        config = _get_config()
        cache = _get_cache(config)
        temp_dir = Path(config.get("general.temp_dir", "./temp"))
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio_path, downloaded_info = await asyncio.to_thread(
            download_audio, clean_url, temp_dir
        )

        default_game = str(config.get("discovery.default_game", "GTA 6"))
        async with _get_url_lock(cache.get_cache_key(clean_url)):
            existing = cache.load_state(clean_url)
            if existing is not None:
                resolved_game = existing.stream_info.game or downloaded_info.game or default_game
                existing.stream_info = StreamInfo(
                    url=clean_url,
                    title=downloaded_info.title or existing.stream_info.title,
                    platform=downloaded_info.platform,
                    streamer=existing.stream_info.streamer or downloaded_info.streamer,
                    game=resolved_game,
                    duration_seconds=(
                        downloaded_info.duration_seconds
                        or existing.stream_info.duration_seconds
                    ),
                    view_count=downloaded_info.view_count or existing.stream_info.view_count,
                    video_id=(
                        downloaded_info.video_id
                        or existing.stream_info.video_id
                        or (_extract_youtube_id_from_url(clean_url) or "")
                    ),
                )
                Pipeline._recover_failed_state(existing)
                existing.audio_path = Path(audio_path)
                if (
                    existing.status < PipelineStatus.TRANSCRIBING
                    or existing.status == PipelineStatus.FAILED
                ):
                    existing.status = (
                        PipelineStatus.ANALYZING
                        if existing.transcript is not None
                        else PipelineStatus.TRANSCRIBING
                    )
                state = existing
            else:
                stream_info = StreamInfo(
                    url=clean_url,
                    title=downloaded_info.title,
                    platform=downloaded_info.platform,
                    streamer=downloaded_info.streamer,
                    game=downloaded_info.game or default_game,
                    duration_seconds=downloaded_info.duration_seconds,
                    view_count=downloaded_info.view_count,
                    video_id=(
                        downloaded_info.video_id
                        or (_extract_youtube_id_from_url(clean_url) or "")
                    ),
                )
                state = PipelineState(
                    stream_info=stream_info,
                    status=PipelineStatus.TRANSCRIBING,
                    audio_path=Path(audio_path),
                )

            cache.save_state(state)
        return _json_ok(
            {
                "url": clean_url,
                "audio_path": str(audio_path),
                "status": state.status.name.lower(),
                "stream_info": state.stream_info.to_dict(),
            }
        )
    except Exception as exc:
        logger.exception("download_stream_audio failed")
        return _json_error(str(exc))


@mcp.tool()
async def transcribe_audio(url: str = "", audio_path: str = "") -> str:
    """Transcribe downloaded audio using WhisperX with word-level timestamps.

    Resolves cached PipelineState via `url` or `StepCache.find_by_audio_path(audio_path)`.
    """
    try:
        clean_url = url.strip() if isinstance(url, str) else ""
        clean_audio = audio_path.strip() if isinstance(audio_path, str) else ""
        if not clean_url and not clean_audio:
            return _json_error("Provide either 'url' or 'audio_path'.")

        config = _get_config()
        cache = _get_cache(config)
        initial_state = _resolve_state(cache, url=clean_url, audio_path=clean_audio)

        target_audio: Path | None = None
        if clean_audio:
            target_audio = Path(clean_audio)
        elif initial_state is not None and initial_state.audio_path is not None:
            target_audio = Path(initial_state.audio_path)

        if target_audio is None:
            return _json_error(
                "No audio_path found for the provided URL. Run download_stream_audio first."
            )
        if not target_audio.is_file():
            return _json_error(f"Audio file not found: {target_audio}")

        transcriber = Transcriber(config)
        async with _gpu_guard(config):
            transcript = await asyncio.to_thread(transcriber.transcribe, target_audio)

        resolved_url = (
            initial_state.stream_info.url if initial_state is not None else clean_url
        )
        state: PipelineState | None = None
        if resolved_url:
            async with _get_url_lock(cache.get_cache_key(resolved_url)):
                # Reload latest state to avoid clobbering concurrent analyze_signals updates
                fresh_state = cache.load_state(resolved_url) or initial_state
                if fresh_state is None:
                    fresh_state = PipelineState(
                        stream_info=StreamInfo(
                            url=resolved_url,
                            title="",
                            platform=Platform.YOUTUBE,
                            streamer="",
                            game=str(config.get("discovery.default_game", "GTA 6")),
                            duration_seconds=0,
                            video_id=_extract_youtube_id_from_url(resolved_url) or "",
                        ),
                        status=PipelineStatus.ANALYZING,
                        audio_path=target_audio,
                        transcript=transcript,
                    )
                else:
                    Pipeline._recover_failed_state(fresh_state)
                    fresh_state.transcript = transcript
                    if fresh_state.audio_path is None or (
                        clean_audio and not Path(fresh_state.audio_path).is_file()
                    ):
                        fresh_state.audio_path = target_audio
                    if (
                        fresh_state.status < PipelineStatus.ANALYZING
                        or fresh_state.status == PipelineStatus.FAILED
                    ):
                        fresh_state.status = (
                            PipelineStatus.SCORING
                            if fresh_state.analysis_complete
                            else PipelineStatus.ANALYZING
                        )
                cache.save_state(fresh_state)
                state = fresh_state

        word_count = sum(len(seg.words) for seg in transcript.segments)
        duration_seconds = transcript.segments[-1].end if transcript.segments else 0.0

        return _json_ok(
            {
                "url": resolved_url,
                "audio_path": str(target_audio),
                "language": transcript.language,
                "segment_count": len(transcript.segments),
                "word_count": word_count,
                "duration_seconds": round(duration_seconds, 3),
                "preview": transcript.full_text[:500],
                "cached": state is not None,
                "status": state.status.name.lower() if state is not None else "analyzing",
            }
        )
    except Exception as exc:
        logger.exception("transcribe_audio failed")
        return _json_error(str(exc))


@mcp.tool()
async def analyze_signals(url: str = "", audio_path: str = "") -> str:
    """Run streamed loudness spike detection and yt-dlp chat velocity analysis, saving results to StepCache."""
    try:
        clean_url = url.strip() if isinstance(url, str) else ""
        clean_audio = audio_path.strip() if isinstance(audio_path, str) else ""
        if not clean_url and not clean_audio:
            return _json_error("Provide either 'url' or 'audio_path'.")

        config = _get_config()
        cache = _get_cache(config)
        initial_state = _resolve_state(cache, url=clean_url, audio_path=clean_audio)

        target_url = initial_state.stream_info.url if initial_state is not None else clean_url
        target_audio: Path | None = None
        if clean_audio:
            target_audio = Path(clean_audio)
        elif initial_state is not None and initial_state.audio_path is not None:
            target_audio = Path(initial_state.audio_path)

        if target_audio is None:
            return _json_error(
                "No audio_path found for signal analysis. Run download_stream_audio first or pass audio_path."
            )
        if not target_audio.is_file():
            return _json_error(f"Audio file not found: {target_audio}")

        loudness_detector = LoudnessDetector(config)
        loudness_spikes = await asyncio.to_thread(
            loudness_detector.detect, target_audio
        )

        chat_spikes: list[TimeRange] = []
        if target_url:
            temp_dir = Path(config.get("general.temp_dir", "./temp"))
            temp_dir.mkdir(parents=True, exist_ok=True)
            scraper = ChatScraper()
            raw_video_id = (
                initial_state.stream_info.video_id
                if (initial_state is not None and initial_state.stream_info.video_id)
                else (_extract_youtube_id_from_url(target_url) or "")
            )
            video_id = (
                _sanitize_path_segment(
                    raw_video_id, fallback=cache.get_cache_key(target_url)
                )
                if _is_safe_id_prefix(raw_video_id)
                else cache.get_cache_key(target_url)
            )
            chat_cache_path = temp_dir / f"{video_id}_chat.csv"
            if not _is_within_directory(chat_cache_path, temp_dir):
                chat_cache_path = temp_dir / f"{cache.get_cache_key(target_url)}_chat.csv"

            def _fetch_chat() -> Any:
                cached = scraper.load_cache(chat_cache_path)
                if cached is not None:
                    return cached
                df = scraper.scrape(target_url, temp_dir)
                # Header-only CSV marks "no chat" so later runs do not re-scrape
                scraper.save_cache(df, chat_cache_path)
                return df

            try:
                chat_df = await asyncio.to_thread(_fetch_chat)
            except Exception as chat_exc:
                logger.exception("analyze_signals chat scrape failed")
                return _json_error(
                    f"Chat scrape failed; analysis not marked complete, retry later: {chat_exc}"
                )
            chat_analyzer = ChatVelocityAnalyzer(config)
            chat_spikes = await asyncio.to_thread(chat_analyzer.analyze, chat_df)

        state: PipelineState | None = None
        if target_url:
            async with _get_url_lock(cache.get_cache_key(target_url)):
                # Reload latest state to avoid clobbering concurrent transcribe_audio updates
                fresh_state = cache.load_state(target_url) or initial_state
                if fresh_state is None:
                    fresh_state = PipelineState(
                        stream_info=StreamInfo(
                            url=target_url,
                            title="",
                            platform=Platform.YOUTUBE,
                            streamer="",
                            game=str(config.get("discovery.default_game", "GTA 6")),
                            duration_seconds=0,
                            video_id=_extract_youtube_id_from_url(target_url) or "",
                        ),
                        status=PipelineStatus.TRANSCRIBING,
                        audio_path=target_audio,
                        loudness_spikes=loudness_spikes,
                        chat_spikes=chat_spikes,
                        analysis_complete=True,
                    )
                else:
                    Pipeline._recover_failed_state(fresh_state)
                    fresh_state.loudness_spikes = loudness_spikes
                    fresh_state.chat_spikes = chat_spikes
                    fresh_state.analysis_complete = True
                    if fresh_state.audio_path is None or (
                        clean_audio and not Path(fresh_state.audio_path).is_file()
                    ):
                        fresh_state.audio_path = target_audio
                    if (
                        fresh_state.status < PipelineStatus.SCORING
                        or fresh_state.status == PipelineStatus.FAILED
                    ):
                        fresh_state.status = (
                            PipelineStatus.SCORING
                            if fresh_state.transcript is not None
                            else PipelineStatus.TRANSCRIBING
                        )
                cache.save_state(fresh_state)
                state = fresh_state

        overlaps = _compute_signal_overlaps(loudness_spikes, chat_spikes)
        return _json_ok(
            {
                "url": target_url,
                "audio_path": str(target_audio),
                "loudness_spike_count": len(loudness_spikes),
                "chat_spike_count": len(chat_spikes),
                "overlap_count": len(overlaps),
                "loudness_spikes": [s.to_dict() for s in loudness_spikes],
                "chat_spikes": [s.to_dict() for s in chat_spikes],
                "overlap_moments": overlaps,
                "cached": state is not None,
                "status": state.status.name.lower() if state is not None else "transcribing",
            }
        )
    except Exception as exc:
        logger.exception("analyze_signals failed")
        return _json_error(str(exc))


@mcp.tool()
async def get_transcript_chunks(
    url: str = "",
    chunk_minutes: int = 12,
    only_with_signals: bool = True,
    audio_path: str = "",
) -> str:
    """Get timestamped transcript divided into chunks for highlight scoring by the MCP client LLM.

    Resolves cached PipelineState via `url` or `StepCache.find_by_audio_path(audio_path)`.
    When only_with_signals=True (default), returns only chunks overlapping at least one
    loudness or chat spike (reduces token usage by 60-80%).
    The response includes narration_guide + avoid_phrases: write narration following
    narration_guide (energetic, interactive gamer voice, grounded in the transcript).
    """
    try:
        clean_url = url.strip() if isinstance(url, str) else ""
        clean_audio = audio_path.strip() if isinstance(audio_path, str) else ""
        if not clean_url and not clean_audio:
            return _json_error("Provide either 'url' or 'audio_path'.")

        config = _get_config()
        cache = _get_cache(config)
        state = _resolve_state(cache, url=clean_url, audio_path=clean_audio)
        if state is None:
            return _json_error(
                "No cached pipeline state found for the provided url or audio_path."
            )
        if state.transcript is None:
            return _json_error(
                "Transcript is not available in cache. Run transcribe_audio first."
            )
        narration_lang = _state_narration_language(config, state)

        scorer = _MCPScorer(config)
        all_chunks = scorer._chunk_transcript(
            state.transcript, chunk_minutes=chunk_minutes
        )
        candidates = scorer.get_candidate_chunks(
            state.transcript,
            state.loudness_spikes,
            state.chat_spikes,
            chunk_minutes=chunk_minutes,
            only_with_signals=only_with_signals,
        )

        chunks_payload = [
            {
                "chunk_index": idx,
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
                "loudness_spikes": [s.to_dict() for s in loud],
                "chat_spikes": [s.to_dict() for s in chat],
            }
            for idx, (chunk, loud, chat) in enumerate(candidates)
        ]

        payload: dict[str, Any] = {
            "url": state.stream_info.url,
            "chunk_minutes": chunk_minutes,
            "only_with_signals": only_with_signals,
            "analysis_complete": state.analysis_complete,
            "total_chunks": len(all_chunks),
            "returned_chunks": len(chunks_payload),
            "filtered_out_chunks": max(0, len(all_chunks) - len(chunks_payload)),
            "chunks": chunks_payload,
            "narration_language": narration_lang,
        }
        if narration_lang == "hinglish":
            payload["narration_language_instruction"] = language_instruction(narration_lang)
        avoid = _recent_outros(config)
        payload["avoid_phrases"] = avoid
        payload["narration_guide"] = narration_guide(
            narration_lang, _next_hook_style(config), avoid
        )
        if (
            only_with_signals
            and not state.analysis_complete
            and not state.loudness_spikes
            and not state.chat_spikes
        ):
            payload["hint"] = (
                "Signal analysis has not been run yet; call analyze_signals first "
                "or pass only_with_signals=False."
            )
        return _json_ok(payload)
    except Exception as exc:
        logger.exception("get_transcript_chunks failed")
        return _json_error(str(exc))


@mcp.tool()
async def submit_highlights(
    url: str,
    highlights: str,
    allow_without_transcript: bool = False,
    narration_strict: bool = True,
) -> str:
    """Submit highlight selections from the MCP client LLM for rendering.

    Run transcribe_audio first: a transcript is REQUIRED unless allow_without_transcript=true.
    Each highlight needs start, end, title, moment_type, a non-empty narration_intro AND
    narration_outro (written in the narration language this tool reports; a highlight without
    them is rejected unless it sets "mode": "raw"), plus optional context_badge, sub_clips,
    peak_time. Highlights that start before analysis.skip_intro_minutes are rejected.
    breakdown_commentary / breakdown_callout are only used by "breakdown" (instant replay),
    which is allowed ONLY for moment_type clutch, fail, graphics_detail or analysis and needs a
    non-empty breakdown_commentary; reactions (hype, funny, chat_chaos, lore_secret) get
    narrative/story instead. Always render with render_clip / render_all_selected; never
    assemble clips yourself with ffmpeg, subtitle files or system TTS voices.

    Write narration following narration_guide from get_transcript_chunks (energetic,
    interactive gamer voice, grounded in the transcript). Every non-raw highlight's narration is
    quality-checked (banned generic phrases, repeats, long sentences, not grounded in the
    transcript/chat, non-interactive outro); failing highlights are rejected with a `problems`
    list so you can rewrite and resubmit. Pass narration_strict=false to accept them with the
    problems returned as warnings instead.
    """
    try:
        clean_url = url.strip() if isinstance(url, str) else ""
        if not clean_url:
            return _json_error("Invalid URL: url must be a non-empty string.")
        if not isinstance(highlights, str) or not highlights.strip():
            return _json_error("highlights must be a non-empty JSON string.")

        cleaned_json = _clean_llm_text(highlights)
        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError as exc:
            return _json_error(f"Invalid highlights JSON: {exc}")

        if isinstance(parsed, list):
            raw_items = parsed
        elif isinstance(parsed, dict):
            if isinstance(parsed.get("highlights"), list):
                raw_items = parsed["highlights"]
            elif isinstance(parsed.get("clips"), list):
                raw_items = parsed["clips"]
            elif "start" in parsed and "end" in parsed:
                raw_items = [parsed]
            else:
                return _json_error(
                    "highlights JSON object must contain a 'highlights' list or highlight fields."
                )
        else:
            return _json_error(
                f"Expected JSON array or object for highlights, got {type(parsed).__name__}."
            )

        config = _get_config()
        cache = _get_cache(config)
        scorer = _MCPScorer(config)

        async with _get_url_lock(cache.get_cache_key(clean_url)):
            state = cache.load_state(clean_url)
            if state is None:
                state = PipelineState(
                    stream_info=StreamInfo(
                        url=clean_url,
                        title="",
                        platform=Platform.YOUTUBE,
                        streamer="",
                        game=str(config.get("discovery.default_game", "GTA 6")),
                        duration_seconds=0,
                        video_id=_extract_youtube_id_from_url(clean_url) or "",
                    ),
                    status=PipelineStatus.SCORING,
                )

            if state.transcript is None and not allow_without_transcript:
                return _json_error(
                    "No transcript cached for this stream. Run transcribe_audio first so "
                    "narration and captions can use what the streamer actually says, then "
                    "submit highlights again (or pass allow_without_transcript=true to "
                    "submit without one)."
                )

            accepted: list[Highlight] = []
            rejected_count = 0
            rejected_reasons: list[dict[str, Any]] = []
            narration_warnings: list[dict[str, Any]] = []
            recent_outros = _recent_outros(config)
            skip_intro_seconds = float(getattr(scorer, "skip_intro_seconds", 0.0) or 0.0)
            narr_lang = _state_narration_language(config, state)

            for item in raw_items:
                if not isinstance(item, dict):
                    rejected_count += 1
                    continue

                start = _parse_seconds(item.get("start"))
                end = _parse_seconds(item.get("end"))
                if start is None or end is None or end <= start:
                    rejected_count += 1
                    continue

                if skip_intro_seconds > 0 and start < skip_intro_seconds:
                    rejected_count += 1
                    rejected_reasons.append(
                        {
                            "start": start,
                            "end": end,
                            "reason": (
                                f"starts at {start:.0f}s, before analysis.skip_intro_minutes "
                                f"({skip_intro_seconds / 60:g} min = {skip_intro_seconds:.0f}s): "
                                "that part of the stream is pre-game/intro. Pick a moment later."
                            ),
                        }
                    )
                    continue

                item_mode = _as_str(item.get("mode") or item.get("transformation_mode")).lower()
                if item_mode != "raw":
                    missing = [
                        f
                        for f in ("narration_intro", "narration_outro")
                        if not _as_str(item.get(f))
                    ]
                    if missing:
                        rejected_count += 1
                        rejected_reasons.append(
                            {
                                "start": start,
                                "end": end,
                                "reason": (
                                    f"missing {' and '.join(missing)}: every non-raw clip must "
                                    "open with an intro voiceover and close with an outro. "
                                    "Write them (1-2 short sentences each, grounded in what "
                                    "the streamer says) in the narration language "
                                    f"'{narr_lang}' and resubmit this highlight."
                                ),
                            }
                        )
                        continue

                score = _parse_score(item.get("score"))
                title = (
                    _as_str(item.get("title"))
                    or _as_str(item.get("hook_title"))
                    or "Stream Highlight"
                )
                reason = _as_str(item.get("reason"))
                best_quote = _as_str(item.get("best_quote"))
                moment_type = _normalize_moment_type(item.get("moment_type"))
                narration_intro = _as_str(item.get("narration_intro"))
                narration_outro = _as_str(item.get("narration_outro"))
                context_badge = _as_str(item.get("context_badge"))
                breakdown_commentary = _as_str(item.get("breakdown_commentary"))
                breakdown_callout = _as_str(item.get("breakdown_callout"))
                callout = _as_str(item.get("callout"))

                raw_beats = _parse_raw_sub_clips(item.get("sub_clips"))
                validated_beats = Scorer._validate_sub_clips(
                    raw_beats,
                    start,
                    end,
                    max_duration=scorer.max_clip_seconds,
                )

                highlight = Highlight(
                    start=start,
                    end=end,
                    score=score,
                    title=title,
                    reason=reason,
                    best_quote=best_quote,
                    moment_type=moment_type,
                    narration_intro=narration_intro,
                    narration_outro=narration_outro,
                    context_badge=context_badge,
                    sub_clips=validated_beats,
                    breakdown_commentary=breakdown_commentary,
                    breakdown_callout=breakdown_callout,
                )
                highlight.callout = callout
                stitched = sum(float(b["end"]) - float(b["start"]) for b in validated_beats)
                if not (
                    validated_beats
                    and scorer.min_clip_seconds <= stitched <= scorer.max_clip_seconds
                ):
                    # Single cut: widen short moments / trim long ones instead of rejecting.
                    highlight.sub_clips = []
                    fit_peak = _parse_seconds(item.get("peak_time"))
                    highlight.peak_time = fit_peak if fit_peak is not None else 0.0
                    fit = scorer.fit_highlight_bounds(
                        highlight,
                        state.loudness_spikes,
                        stream_duration=float(state.stream_info.duration_seconds or 0) or None,
                    )
                    highlight.peak_time = 0.0
                    if fit.rejected:
                        rejected_count += 1
                        rejected_reasons.append(
                            {"start": start, "end": end, "reason": fit.adjustment}
                        )
                        continue
                    start, end = highlight.start, highlight.end
                if state.audio_path and Path(state.audio_path).is_file():
                    try:
                        from ..analysis.copyright_risk import CopyrightRiskDetector

                        detector = CopyrightRiskDetector(config)
                        if detector.enabled:
                            highlight.word_timestamps = scorer._extract_words_for_range(state.transcript, highlight.start, highlight.end) if state.transcript is not None else []
                            risk, conf = await asyncio.to_thread(
                                detector.detect_highlight, state.audio_path, highlight
                            )
                            highlight.music_risk = risk
                            highlight.music_confidence = conf
                    except Exception as det_exc:
                        logger.warning("Copyright risk detection failed: %s", det_exc)
                        highlight.music_risk = "UNKNOWN"
                        highlight.music_confidence = 0.0
                highlight.final_score = scorer._compute_weighted_score(
                    highlight, state.loudness_spikes, state.chat_spikes
                )

                if state.transcript is not None:
                    highlight.word_timestamps = scorer._extract_words_for_range(
                        state.transcript, highlight.start, highlight.end
                    )
                    scoped_text = scorer._extract_transcript_for_range(
                        state.transcript, highlight.start, highlight.end
                    )
                    if scoped_text:
                        highlight.transcript_chunk = scoped_text

                overlapping_loudness = [
                    s
                    for s in state.loudness_spikes
                    if s.end > highlight.start
                    and s.start < highlight.end
                    and _has_valid_peak_time(s)
                ]
                explicit_peak = _parse_seconds(item.get("peak_time"))
                if overlapping_loudness:
                    highlight.peak_time = scorer._extract_peak_time(
                        highlight, overlapping_loudness
                    )
                elif explicit_peak is not None and start <= explicit_peak <= end:
                    highlight.peak_time = explicit_peak
                else:
                    highlight.peak_time = scorer._extract_peak_time(highlight, [])
                highlight.chat_burst = scorer._extract_chat_burst(
                    highlight, state.chat_spikes
                )
                if item_mode != "raw":
                    problems = check_narration(
                        {
                            "narration_intro": narration_intro,
                            "narration_outro": narration_outro,
                            "callout": callout,
                        },
                        transcript_excerpt=_as_str(highlight.transcript_chunk),
                        chat_samples=_chat_texts(highlight.chat_burst),
                        avoid_phrases=recent_outros,
                    )
                    if problems and narration_strict:
                        rejected_count += 1
                        rejected_reasons.append(
                            {
                                "start": start,
                                "end": end,
                                "reason": (
                                    "narration failed the quality check: rewrite it following "
                                    "narration_guide (energetic, interactive gamer voice, "
                                    "grounded in the transcript) and resubmit this highlight."
                                ),
                                "problems": problems,
                            }
                        )
                        continue
                    if problems:
                        narration_warnings.append(
                            {"start": start, "end": end, "problems": problems}
                        )
                accepted.append(highlight)

            if raw_items and not accepted:
                return _json_error(
                    "No valid highlights found in payload (check start/end timestamps)."
                )

            pre_filter_count = len(accepted)
            accepted = Scorer._deduplicate(accepted)
            accepted.sort(key=lambda h: h.final_score, reverse=True)
            accepted = accepted[: max(0, int(scorer.max_clips))]
            dropped_count = pre_filter_count - len(accepted)
            state.highlights = list(accepted)
            state.selected_highlights = list(accepted)
            if accepted:
                state.status = PipelineStatus.ASSEMBLING
            elif (
                state.status >= PipelineStatus.ASSEMBLING
                and state.status != PipelineStatus.FAILED
            ):
                if state.transcript is not None and state.analysis_complete:
                    state.status = PipelineStatus.SCORING
                elif state.transcript is not None:
                    state.status = PipelineStatus.ANALYZING
                elif state.audio_path is not None and Path(state.audio_path).is_file():
                    state.status = PipelineStatus.TRANSCRIBING
                else:
                    state.status = PipelineStatus.PENDING
            cache.save_state(state)

        submit_lang = _state_narration_language(config, state)
        submit_payload: dict[str, Any] = {
            "url": clean_url,
            "accepted_count": len(accepted),
            "rejected_count": rejected_count,
            "rejected": rejected_reasons,
            "dropped_count": dropped_count,
            "status": state.status.name.lower(),
            "highlights": [h.to_dict() for h in accepted],
            "narration_language": submit_lang,
        }
        if submit_lang == "hinglish":
            submit_payload["narration_language_instruction"] = language_instruction(submit_lang)
        submit_warnings: list[Any] = []
        if state.transcript is None:
            submit_warnings.append(
                "No transcript cached for this stream: speech captions will be empty and "
                "narration can't quote the streamer. Run transcribe_audio first."
            )
        submit_warnings.extend(narration_warnings)
        if submit_warnings:
            submit_payload["warnings"] = submit_warnings
        return _json_ok(submit_payload)
    except Exception as exc:
        logger.exception("submit_highlights failed")
        return _json_error(str(exc))


@mcp.tool()
async def render_clip(
    url: str,
    start: float,
    end: float,
    hook_title: str,
    game: str = "",
    streamer: str = "",
    moment_type: str = "hype",
    narration_intro: str = "",
    narration_outro: str = "",
    context_badge: str = "",
    sub_clips_json: str = "[]",
    mode: str | None = None,
    voice_intro_path: str = "",
    voice_outro_path: str = "",
    breakdown_commentary: str = "",
    breakdown_callout: str = "",
    voice_breakdown_path: str = "",
    callout: str = "",
    layout: str = "",
    narration_strict: bool = True,
) -> str:
    """Render a single highlight into a final 9:16 vertical clip (mode: auto | narrative | story | breakdown | raw) with YPP transformative narration, zoom, and ASS captions+chat ticker.

    ALWAYS render with this tool or render_all_selected. NEVER assemble clips yourself with
    ffmpeg, subtitle files or system TTS voices: those outputs skip the narration, captions,
    loudness and copyright checks.

    Required for every non-raw mode: narration_intro AND narration_outro (1-2 short sentences
    each, grounded in what the streamer says, in the stream's narration language). Mode
    "breakdown" (instant replay) is only allowed for moment_type clutch, fail, graphics_detail
    or analysis, with non-empty breakdown_commentary and a peak inside the clip; otherwise it is
    rendered as narrative/story and the response says so (mode_adjusted + reason). Use
    narrative/story (or auto) for reactions: hype, funny, chat_chaos, lore_secret.

    Write narration following narration_guide from get_transcript_chunks (energetic,
    interactive gamer voice, grounded in the transcript). Given narration text is
    quality-checked and rejected with a problems list unless narration_strict=false.
    """
    try:
        clean_url = url.strip() if isinstance(url, str) else ""
        if not clean_url:
            return _json_error("Invalid URL: url must be a non-empty string.")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0.0
            or float(end) <= float(start)
        ):
            return _json_error(
                f"Invalid clip range: start={start!r}, end={end!r} (require 0 <= start < end)."
            )

        config = _get_config()
        cache = _get_cache(config)
        pipeline = Pipeline(config)
        scorer = _MCPScorer(config)

        base_mode = pipeline._resolve_base_mode(mode)

        if base_mode != "raw":
            missing_narr = [
                name
                for name, val, vpath in (
                    ("narration_intro", narration_intro, voice_intro_path),
                    ("narration_outro", narration_outro, voice_outro_path),
                )
                if not _as_str(val) and not _as_str(vpath)
            ]
            if missing_narr:
                return _json_error(
                    f"Missing {' and '.join(missing_narr)}: every non-raw clip must open with an "
                    "intro voiceover and close with an outro. Write them (1-2 short sentences "
                    "each, grounded in what the streamer says, in the stream's narration "
                    "language) and call render_clip again, or use mode='raw' for an "
                    "un-narrated cut."
                )

            if narration_strict and (_as_str(narration_intro) or _as_str(narration_outro)):
                nq_state = cache.load_state(clean_url)
                nq_excerpt = ""
                nq_chat: list[str] = []
                if nq_state is not None:
                    probe_h = Highlight(
                        start=float(start), end=float(end), score=0.0, title="", reason=""
                    )
                    if nq_state.transcript is not None:
                        nq_excerpt = scorer._extract_transcript_for_range(
                            nq_state.transcript, float(start), float(end)
                        )
                    nq_chat = _chat_texts(
                        scorer._extract_chat_burst(probe_h, nq_state.chat_spikes)
                    )
                nq_problems = check_narration(
                    {
                        "narration_intro": _as_str(narration_intro),
                        "narration_outro": _as_str(narration_outro),
                        "callout": _as_str(callout),
                    },
                    transcript_excerpt=nq_excerpt,
                    chat_samples=nq_chat,
                    avoid_phrases=_recent_outros(config),
                )
                if nq_problems:
                    return _json_error(
                        "Narration failed the quality check: "
                        + "; ".join(nq_problems)
                        + ". Rewrite following narration_guide from get_transcript_chunks "
                        "(energetic, interactive gamer voice, grounded in the transcript) and "
                        "call render_clip again, or pass narration_strict=false to render as is."
                    )

        skip_intro_seconds = float(config.get("analysis.skip_intro_minutes", 5) or 0) * 60.0
        if skip_intro_seconds > 0 and float(start) < skip_intro_seconds:
            return _json_error(
                f"start={float(start):.0f}s is before analysis.skip_intro_minutes "
                f"({skip_intro_seconds / 60:g} min = {skip_intro_seconds:.0f}s): that part of the "
                "stream is pre-game/intro. Pick a moment later."
            )

        layout_kind = _as_str(layout).lower()
        if layout_kind not in ("", "stacked", "blur_fill", "reaction"):
            return _json_error(
                f"Invalid layout: {layout!r} (use '' for auto, stacked, blur_fill or reaction)."
            )

        parsed_sub_clips: list[dict[str, Any]] = []
        if isinstance(sub_clips_json, str) and sub_clips_json.strip():
            try:
                raw_beats_data = json.loads(_clean_llm_text(sub_clips_json))
            except json.JSONDecodeError as exc:
                return _json_error(f"Invalid sub_clips_json: {exc}")
            if not isinstance(raw_beats_data, list):
                return _json_error("sub_clips_json must decode to a JSON array.")
            parsed_sub_clips = _parse_raw_sub_clips(raw_beats_data)

        validated_beats = Scorer._validate_sub_clips(
            parsed_sub_clips,
            float(start),
            float(end),
            max_duration=scorer.max_clip_seconds,
        )

        bounds_adjustment = ""
        stitched = sum(float(b["end"]) - float(b["start"]) for b in validated_beats)
        if not (
            validated_beats and scorer.min_clip_seconds <= stitched <= scorer.max_clip_seconds
        ):
            # Single cut: widen short moments / trim long ones instead of rejecting.
            validated_beats = []
            cached_state = cache.load_state(clean_url)
            probe = Highlight(start=float(start), end=float(end), score=0.0, title=hook_title, reason="")
            fit = scorer.fit_highlight_bounds(
                probe,
                cached_state.loudness_spikes if cached_state is not None else [],
                stream_duration=(
                    float(cached_state.stream_info.duration_seconds or 0) or None
                    if cached_state is not None
                    else None
                ),
            )
            if fit.rejected:
                return _json_error(f"Rejected clip range: {fit.adjustment}.")
            start, end = probe.start, probe.end
            bounds_adjustment = probe.bounds_adjustment

        intro_path = (
            Path(voice_intro_path.strip())
            if isinstance(voice_intro_path, str) and voice_intro_path.strip()
            else None
        )
        if intro_path is not None and not intro_path.is_file():
            return _json_error(f"voice_intro_path not found: {intro_path}")

        outro_path = (
            Path(voice_outro_path.strip())
            if isinstance(voice_outro_path, str) and voice_outro_path.strip()
            else None
        )
        if outro_path is not None and not outro_path.is_file():
            return _json_error(f"voice_outro_path not found: {outro_path}")

        breakdown_voice = (
            Path(voice_breakdown_path.strip())
            if isinstance(voice_breakdown_path, str) and voice_breakdown_path.strip()
            else None
        )
        if breakdown_voice is not None and not breakdown_voice.is_file():
            return _json_error(f"voice_breakdown_path not found: {breakdown_voice}")

        cache_key = cache.get_cache_key(clean_url)
        async with _get_url_lock(cache_key):
            state = cache.load_state(clean_url)
            resolved_game = (
                _as_str(game)
                or (state.stream_info.game if state is not None else "")
                or str(config.get("discovery.default_game", "GTA 6"))
            )
            resolved_streamer = (
                _as_str(streamer)
                or (state.stream_info.streamer if state is not None else "")
                or "Unknown"
            )

            highlight = Highlight(
                start=float(start),
                end=float(end),
                score=8.0,
                title=_as_str(hook_title) or "Stream Highlight",
                reason="",
                moment_type=_normalize_moment_type(moment_type),
                narration_intro=_as_str(narration_intro),
                narration_outro=_as_str(narration_outro),
                context_badge=_as_str(context_badge),
                sub_clips=validated_beats,
                breakdown_commentary=_as_str(breakdown_commentary),
                breakdown_callout=_as_str(breakdown_callout),
                bounds_adjustment=bounds_adjustment,
            )
            highlight.callout = _as_str(callout)

            if state is not None:
                highlight.final_score = scorer._compute_weighted_score(
                    highlight, state.loudness_spikes, state.chat_spikes
                )
                if state.transcript is not None:
                    highlight.word_timestamps = scorer._extract_words_for_range(
                        state.transcript, highlight.start, highlight.end
                    )
                    scoped_text = scorer._extract_transcript_for_range(
                        state.transcript, highlight.start, highlight.end
                    )
                    if scoped_text:
                        highlight.transcript_chunk = scoped_text
                highlight.peak_time = scorer._extract_peak_time(
                    highlight, state.loudness_spikes
                )
                highlight.chat_burst = scorer._extract_chat_burst(
                    highlight, state.chat_spikes
                )
            else:
                highlight.final_score = highlight.score
                highlight.peak_time = scorer._extract_peak_time(highlight, [])

            min_clip_index = (len(state.clips) + 1) if state is not None else 1
            render_notes: list[dict[str, Any]] = []
            clip_result, resolved_mode = await _render_single_highlight(
                pipeline=pipeline,
                config=config,
                url=clean_url,
                highlight=highlight,
                streamer_name=resolved_streamer,
                game_name=resolved_game,
                clip_index=None,
                base_mode=base_mode,
                voice_intro_path=intro_path,
                voice_outro_path=outro_path,
                min_clip_index=min_clip_index,
                voice_breakdown_path=breakdown_voice,
                layout=layout_kind,
                notes=render_notes,
                language=resolve_narration_language(
                    config,
                    resolved_streamer,
                    state.transcript.language
                    if state is not None and state.transcript is not None
                    else None,
                ),
            )

            fresh_state = cache.load_state(clean_url)
            if fresh_state is not None and not getattr(clip_result, "quarantined", False):
                fresh_state.clips = _merge_clip_results(fresh_state.clips, [clip_result])
                cache.save_state(fresh_state)

        render_payload: dict[str, Any] = {
            "url": clean_url,
            "mode": resolved_mode,
            "bounds_adjustment": bounds_adjustment,
            "quarantined": bool(getattr(clip_result, "quarantined", False)),
            "quarantine_reason": getattr(clip_result, "quarantine_reason", ""),
            "clip": clip_result.to_dict(),
        }
        if render_notes:
            render_payload.update(render_notes[0])
        return _json_ok(render_payload)
    except Exception as exc:
        logger.exception("render_clip failed")
        return _json_error(str(exc))


@mcp.tool()
async def render_all_selected(
    url: str, mode: str | None = None, force: bool = False
) -> str:
    """Render all selected highlights stored in StepCache for a stream.

    ALWAYS render with this tool or render_clip. NEVER assemble clips yourself with ffmpeg,
    subtitle files or system TTS voices: those outputs skip the narration, captions, loudness
    and copyright checks. Highlights must carry narration_intro/narration_outro (set via
    submit_highlights); "breakdown" is only honoured for moment_type clutch, fail,
    graphics_detail or analysis (others are rendered as narrative/story and reported in
    `adjusted`).
    """
    rendered: list[ClipResult] = []
    clean_url = ""
    try:
        clean_url = url.strip() if isinstance(url, str) else ""
        if not clean_url:
            return _json_error("Invalid URL: url must be a non-empty string.")

        config = _get_config()
        cache = _get_cache(config)
        pipeline = Pipeline(config)
        base_mode = pipeline._resolve_base_mode(mode)
        cache_key = cache.get_cache_key(clean_url)

        async with _get_url_lock(cache_key):
            state = cache.load_state(clean_url)
            if state is None:
                return _json_error(f"No cached pipeline state found for URL: {clean_url}")
            if not state.selected_highlights:
                return _json_error(
                    f"No selected highlights found in cache for {clean_url}. Run submit_highlights first."
                )

            streamer_name = state.stream_info.streamer or "Unknown"
            game_name = state.stream_info.game or str(
                config.get("discovery.default_game", "GTA 6")
            )
            selected_highlights = list(state.selected_highlights)
            existing_clips = list(state.clips)
            base_index = len(existing_clips)

            skipped_count = 0
            quarantined_clips: list[ClipResult] = []
            rendered_modes: list[str] = []
            adjusted_notes: list[dict[str, Any]] = []
            try:
                for idx, highlight in enumerate(selected_highlights, start=1):
                    if not force and _has_rendered_clip(existing_clips + rendered, highlight):
                        skipped_count += 1
                        continue
                    clip_result, clip_mode = await _render_single_highlight(
                        pipeline=pipeline,
                        config=config,
                        url=clean_url,
                        highlight=highlight,
                        streamer_name=streamer_name,
                        game_name=game_name,
                        clip_index=None,
                        base_mode=base_mode,
                        min_clip_index=base_index + idx,
                        language=_state_narration_language(config, state),
                        notes=adjusted_notes,
                    )
                    if getattr(clip_result, "quarantined", False):
                        quarantined_clips.append(clip_result)
                        continue
                    rendered.append(clip_result)
                    rendered_modes.append(clip_mode)
                    # Persist incremental clip progress so mid-batch failures never orphan rendered files
                    state.clips = _merge_clip_results(existing_clips, rendered)
                    state.status = PipelineStatus.ASSEMBLING
                    cache.save_state(state)

                state.clips = _merge_clip_results(existing_clips, rendered)
                state.status = PipelineStatus.COMPLETE
                cache.save_state(state)
            except Exception as exc:
                logger.exception("render_all_selected failed")
                if rendered:
                    try:
                        state.clips = _merge_clip_results(existing_clips, rendered)
                        state.status = PipelineStatus.FAILED
                        cache.save_state(state)
                    except Exception:
                        logger.exception(
                            "Failed to persist partial clips after render_all_selected error"
                        )
                    return _json_ok(
                        {
                            "error": str(exc),
                            "url": clean_url,
                            "rendered_count": len(rendered),
                            "clips": [c.to_dict() for c in rendered],
                        }
                    )
                return _json_error(str(exc))

        return _json_ok(
            {
                "url": clean_url,
                "mode": base_mode,
                "rendered_count": len(rendered),
                "skipped_count": skipped_count,
                "quarantined_count": len(quarantined_clips),
                "quarantined": [c.to_dict() for c in quarantined_clips],
                "adjusted": adjusted_notes,
                "status": state.status.name.lower(),
                "clips": [
                    {**c.to_dict(), "mode": m}
                    for c, m in zip(rendered, rendered_modes)
                ],
            }
        )
    except Exception as exc:
        logger.exception("render_all_selected failed")
        return _json_error(str(exc))


@mcp.tool()
async def get_pipeline_status() -> str:
    """Return status summary of all cached pipelines."""
    try:
        config = _get_config()
        cache = _get_cache(config)
        states = cache.list_all()

        pipelines_summary = [
            {
                "url": s.stream_info.url,
                "title": s.stream_info.title,
                "streamer": s.stream_info.streamer,
                "game": s.stream_info.game,
                "status": s.status.name.lower(),
                "status_code": int(s.status),
                "audio_path": str(s.audio_path) if s.audio_path else None,
                "analysis_complete": s.analysis_complete,
                "highlights_count": len(s.highlights),
                "clips_count": len(s.clips),
            }
            for s in states
        ]

        return _json_ok(
            {
                "count": len(pipelines_summary),
                "pipelines": pipelines_summary,
            }
        )
    except Exception as exc:
        logger.exception("get_pipeline_status failed")
        return _json_error(str(exc))


@mcp.tool()
async def generate_post_metadata(
    hook_title: str,
    streamer: str,
    game: str,
    description: str,
    use_ollama: bool = False,
) -> str:
    """Build normalized hashtags and social media caption draft without requiring Ollama (set use_ollama=True to invoke local Ollama)."""
    try:
        config = _get_config()
        if not use_ollama:
            payload = _build_offline_post_metadata(
                config=config,
                hook_title=hook_title,
                streamer=streamer,
                game=game,
                description=description,
            )
            return _json_ok(payload)

        clean_hook = _as_str(hook_title) or "Stream Highlight"
        clean_streamer = _as_str(streamer)
        clean_game = _as_str(game)
        clean_desc = _as_str(description)

        scorer = OllamaScorer(config)
        synthetic = Highlight(
            start=0.0,
            end=30.0,
            score=8.0,
            title=clean_hook,
            reason=clean_desc,
        )
        async with _gpu_guard(config):
            caption, hashtags = await asyncio.to_thread(
                scorer._generate_caption_and_hashtags,
                synthetic,
                clean_streamer,
                clean_game,
                clean_hook,
            )
        return _json_ok(
            {
                "hook_title": clean_hook,
                "caption": caption,
                "hashtags": hashtags,
                "hashtag_string": " ".join(f"#{tag}" for tag in hashtags),
                "streamer": clean_streamer,
                "game": clean_game,
                "source": "ollama",
            }
        )
    except Exception as exc:
        logger.exception("generate_post_metadata failed")
        return _json_error(str(exc))


@mcp.tool()
async def cleanup_temp(url: str = "") -> str:
    """Delete temporary audio/video files for a stream (or all completed streams)."""
    try:
        config = _get_config()
        cache = _get_cache(config)
        temp_dir = Path(config.get("general.temp_dir", "./temp"))
        clean_url = url.strip() if isinstance(url, str) else ""

        deleted_files: list[str] = []
        failed_files: list[dict[str, str]] = []
        freed_bytes = 0

        def _delete_paths(paths: list[Path]) -> None:
            nonlocal freed_bytes
            for path in paths:
                if not path.is_file():
                    continue
                try:
                    size = path.stat().st_size
                    path.unlink()
                    deleted_files.append(str(path))
                    freed_bytes += size
                except OSError as err:
                    failed_files.append({"path": str(path), "error": str(err)})

        if clean_url:
            async with _get_url_lock(cache.get_cache_key(clean_url)):
                state = cache.load_state(clean_url)
                targets = _collect_temp_files_for_stream(
                    temp_dir, cache, clean_url, state
                )
                _delete_paths(targets)
            payload: dict[str, Any] = {
                "scope": "stream",
                "url": clean_url,
                "deleted_count": len(deleted_files),
                "deleted_files": deleted_files,
                "freed_bytes": freed_bytes,
            }
            if failed_files:
                payload["failed_files"] = failed_files
            return _json_ok(payload)

        completed_states = [
            s for s in cache.list_all() if s.status == PipelineStatus.COMPLETE
        ]
        cleaned_urls: list[str] = []
        seen_paths: set[str] = set()
        all_targets: list[Path] = []
        for state in completed_states:
            stream_url = state.stream_info.url
            cleaned_urls.append(stream_url)
            for p in _collect_temp_files_for_stream(temp_dir, cache, stream_url, state):
                key = str(p.resolve()).lower()
                if key not in seen_paths:
                    seen_paths.add(key)
                    all_targets.append(p)

        _delete_paths(all_targets)
        payload = {
            "scope": "all_completed",
            "cleaned_urls": cleaned_urls,
            "deleted_count": len(deleted_files),
            "deleted_files": deleted_files,
            "freed_bytes": freed_bytes,
        }
        if failed_files:
            payload["failed_files"] = failed_files
        return _json_ok(payload)
    except Exception as exc:
        logger.exception("cleanup_temp failed")
        return _json_error(str(exc))
