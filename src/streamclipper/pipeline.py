"""End-to-end async pipeline orchestrator with IntEnum state caching and crash recovery."""
from __future__ import annotations

import asyncio
import contextlib
from datetime import date
import inspect
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, AsyncIterator
import numpy as np

from rich.console import Console
from rich.table import Table

from .analysis.chat_velocity import ChatVelocityAnalyzer
from .analysis.heatmap import HeatmapAnalyzer
from .analysis.loudness import LoudnessDetector
from .analysis.transcriber import Transcriber
from .cache import StepCache
from .config import Config, VALID_TRANSFORMATION_MODES
from .language import resolve_narration_language
from .download.audio import download_audio
from .download.chat import ChatScraper
from .download.heatmap import get_heatmap
from .download.video import download_multiple_segments, download_segment
from .models import (
    ClipMetadata,
    ClipResult,
    Highlight,
    PipelineState,
    PipelineStatus,
    Platform,
    StreamInfo,
    WordTimestamp,
)
from .scoring.ollama_scorer import OllamaScorer
from .output.post_writer import write_post_file
from .output.variety import VarietyState
from .scoring.scorer import Scorer

logger = logging.getLogger(__name__)

_REACTION_OK_MOMENT_TYPES = frozenset({"reaction", "funny", "chat_chaos", "hype"})
_WORDS_PER_SECOND = 2.5
_INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*]+')
_PHASE4_MODULES = frozenset(
    {
        "streamclipper.assembly",
        "streamclipper.assembly.composer",
        "streamclipper.assembly.facecam",
        "streamclipper.assembly.hooks",
        "streamclipper.output",
        "streamclipper.output.exporter",
        "streamclipper.output.organizer",
    }
)


def _is_espeak_available() -> bool:
    """Lazy proxy for the narrator's espeak probe (keeps assembly imports out of module load)."""
    from .assembly import narrator as _narrator_mod

    return bool(_narrator_mod._is_espeak_available())


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = max(0, int(seconds))
    hrs = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


def _sanitize_path_segment(name: str, fallback: str = "unknown") -> str:
    """Sanitize a folder segment so it is valid on Windows and POSIX filesystems."""
    cleaned = _INVALID_PATH_CHARS_RE.sub("_", str(name or "")).strip(" .")
    if not cleaned:
        return fallback
    stem = cleaned.split(".", 1)[0].strip().upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_"
    return cleaned


def _acquire_file_lock(lock_dir: Path, timeout: float | None = 600.0) -> Any:
    """Blocking exclusive lock on <lock_dir>/gpu.lock (run via asyncio.to_thread)."""
    import time

    target_dir = Path(lock_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    lock_path = target_dir / "gpu.lock"

    try:
        if not lock_path.is_file() or lock_path.stat().st_size == 0:
            with open(lock_path, "a+b") as init_fh:
                if init_fh.tell() == 0:
                    init_fh.write(b"0")
                    init_fh.flush()
    except OSError:
        pass

    fh = open(lock_path, "a+b")
    start_time = time.monotonic()
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                fh.seek(0)
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if timeout is not None and (time.monotonic() - start_time) > timeout:
                        raise TimeoutError(
                            f"Timed out after {timeout:.1f}s waiting for GPU cross-process lock at {lock_path}"
                        )
                    time.sleep(0.1)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if timeout is not None and (time.monotonic() - start_time) > timeout:
                        raise TimeoutError(
                            f"Timed out after {timeout:.1f}s waiting for GPU cross-process lock at {lock_path}"
                        )
                    time.sleep(0.1)
    except BaseException:
        fh.close()
        raise
    return fh


def _release_file_lock(fh: Any) -> None:
    """Release exclusive byte-range / advisory lock on gpu.lock."""
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError) as exc:
        logger.debug("Ignoring error releasing GPU file lock: %s", exc)
    finally:
        try:
            fh.close()
        except (OSError, ValueError):
            pass


class _AsyncNull:
    """No-op async context manager (assembly that cannot touch the GPU runs unlocked)."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: Any) -> None:
        return None


class Pipeline:
    """Central orchestrator wiring discovery, download, analysis, scoring, and assembly."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cache = StepCache(config)
        self.console = Console()
        self.temp_dir = Path(config.get("general.temp_dir", "./temp"))
        self.output_dir = Path(config.get("general.output_dir", "./output"))
        self._gpu_lock = asyncio.Lock()

    def _acquire_file_lock(self, timeout: float | None = 600.0) -> Any:
        """Blocking exclusive lock on <cache_dir>/gpu.lock (run via asyncio.to_thread)."""
        lock_dir = Path(self.config.get("general.cache_dir", "./.cache"))
        return _acquire_file_lock(lock_dir, timeout=timeout)

    @staticmethod
    def _release_file_lock(fh: Any) -> None:
        _release_file_lock(fh)

    @contextlib.asynccontextmanager
    async def _gpu_guard(self) -> AsyncIterator[None]:
        """Hold the in-process GPU lock (and the optional cross-process file lock)."""
        async with self._gpu_lock:
            if not bool(self.config.get("general.cross_process_gpu_lock", False)):
                yield
                return
            lock_dir = Path(self.config.get("general.cache_dir", "./.cache"))
            fh = await asyncio.to_thread(_acquire_file_lock, lock_dir)
            try:
                yield
            finally:
                await asyncio.to_thread(_release_file_lock, fh)

    def _narration_may_use_gpu(self, clip_mode: str) -> bool:
        """True when narration synthesis for this clip may load Kokoro on the GPU."""
        if clip_mode == "raw":
            return False
        engine = str(
            self.config.get("transformation.tts_engine", "auto") or "auto"
        ).strip().lower()
        if engine == "kokoro":
            return True
        if engine == "auto":
            return _is_espeak_available()
        return False

    def _is_blocked_streamer(self, name: str | None) -> bool:
        """True when `name` appears in discovery.blocked_streamers (case-insensitive)."""
        if not name:
            return False
        blocked = self.config.get("discovery.blocked_streamers", []) or []
        wanted = str(name).strip().lower()
        return any(str(b).strip().lower() == wanted for b in blocked)

    @staticmethod
    def _recover_failed_state(state: PipelineState) -> PipelineState:
        """Reset a FAILED (99) state back to the last safely completed stage for retry."""
        if state.status != PipelineStatus.FAILED:
            return state
        if state.highlights or state.selected_highlights:
            state.status = PipelineStatus.ASSEMBLING
        elif state.transcript is not None and (
            state.analysis_complete or state.loudness_spikes or state.chat_spikes
        ):
            state.status = PipelineStatus.SCORING
        elif state.transcript is not None:
            if state.audio_path is not None and not Path(state.audio_path).exists():
                state.status = PipelineStatus.PENDING
            else:
                state.status = PipelineStatus.ANALYZING
        elif state.audio_path and Path(state.audio_path).exists():
            state.status = PipelineStatus.TRANSCRIBING
        else:
            state.status = PipelineStatus.PENDING
        return state

    def _resolve_base_mode(self, mode: str | None) -> str:
        """Resolve and validate the requested transformation mode respecting transformation.enabled."""
        if mode is not None and str(mode).strip():
            resolved = str(mode).strip().lower()
            if resolved not in VALID_TRANSFORMATION_MODES:
                raise ValueError(
                    f"Invalid transformation mode {mode!r}; expected one of "
                    f"{sorted(VALID_TRANSFORMATION_MODES)}"
                )
            return resolved
        if not bool(self.config.get("transformation.enabled", True)):
            return "raw"
        configured = str(self.config.get("transformation.mode", "auto")).strip().lower() or "auto"
        if configured not in VALID_TRANSFORMATION_MODES:
            raise ValueError(
                f"Invalid transformation.mode {configured!r}; expected one of "
                f"{sorted(VALID_TRANSFORMATION_MODES)}"
            )
        return configured

    _DEFAULT_BREAKDOWN_MOMENT_TYPES = ("clutch", "fail", "graphics_detail", "analysis")

    def _breakdown_ineligible_reason(self, highlight: Highlight) -> str:
        """Empty string when instant-replay breakdown fits *highlight*, else the reason it does not.

        Eligible only when ALL hold: moment_type is in `transformation.breakdown_moment_types`,
        `breakdown_commentary` is non-empty, and `peak_time` (or `breakdown_target_time` for zoom_spotlight)
        lies inside the clip.
        """
        allowed = [
            str(t).strip().lower()
            for t in (
                self.config.get(
                    "transformation.breakdown_moment_types",
                    list(self._DEFAULT_BREAKDOWN_MOMENT_TYPES),
                )
                or []
            )
        ]
        mtype = str(highlight.moment_type or "").strip().lower()
        if mtype not in allowed:
            return (
                f"moment_type {mtype!r} is not in transformation.breakdown_moment_types "
                f"({', '.join(allowed)}); there is nothing to break down"
            )
        if not str(highlight.breakdown_commentary or "").strip():
            return "breakdown_commentary is empty"
        peak = float(highlight.peak_time or 0.0)
        target = (
            float(highlight.breakdown_target_time)
            if highlight.breakdown_target_time is not None
            else peak
        )
        if not (float(highlight.start) <= target <= float(highlight.end)):
            return f"peak_time {peak:.1f}s is not inside the clip ({highlight.start:.1f}-{highlight.end:.1f}s)"
        return ""

    def _resolve_highlight_mode(
        self, base_mode: str, highlight: Highlight
    ) -> tuple[str, list[dict[str, Any]]]:
        """Resolve concrete per-highlight mode (raw | narrative | story | breakdown) and validated beats."""
        if base_mode == "raw":
            logger.warning(
                "⚠️ WARNING: raw mode produces non-transformative clips that violate "
                "YouTube Partner Program (YPP) reused content policies."
            )
            return "raw", []
        if base_mode == "narrative":
            return "narrative", []

        # Auto-resolve breakdown_kind on highlight if set to "auto" or empty
        b_kind = str(getattr(highlight, "breakdown_kind", "auto") or "auto").strip().lower()
        if b_kind in {"", "auto"}:
            mtype = str(getattr(highlight, "moment_type", "") or "").strip().lower()
            b_commentary = str(getattr(highlight, "breakdown_commentary", "") or "").lower()
            spotlight_keywords = ("screen", "laptop", "file", "read", "document", "secret", "text")
            if mtype in {"analysis", "graphics_detail", "lore_secret"} or any(
                k in b_commentary for k in spotlight_keywords
            ):
                highlight.breakdown_kind = "zoom_spotlight"
            else:
                cfg_default = str(
                    self.config.get("transformation.breakdown.default_kind", "auto") or "auto"
                ).strip().lower()
                highlight.breakdown_kind = cfg_default if cfg_default in {"slow_mo", "zoom_spotlight"} else "slow_mo"

        max_clip_seconds = float(self.config.get("analysis.max_clip_seconds", 90))
        valid_beats = Scorer._validate_sub_clips(
            highlight.sub_clips,
            highlight.start,
            highlight.end,
            max_duration=max_clip_seconds,
        )
        if base_mode == "breakdown":
            why = self._breakdown_ineligible_reason(highlight)
            if not why:
                return "breakdown", []
            logger.info("Breakdown not suitable for this highlight (%s); using narrative/story.", why)
            if len(valid_beats) >= 2:
                return "story", valid_beats
            return "narrative", []
        if base_mode == "auto":
            # Mode is resolved BEFORE Pass 2 (generate_metadata) writes narration, so only
            # signals present at this point may drive the choice:
            #   1. breakdown-eligible highlight (allowed moment_type + commentary + peak inside) -> breakdown
            #   2. validated multi-beat sub_clips -> story (never discard valid beats)
            #   3. otherwise -> narrative
            if not self._breakdown_ineligible_reason(highlight):
                return "breakdown", []
            if len(valid_beats) >= 2:
                return "story", valid_beats
            return "narrative", []
        if base_mode == "story":
            return "story", valid_beats
        return base_mode, []

    async def process_stream(
        self,
        url: str,
        max_clips: int | None = None,
        game: str | None = None,
        streamer: str | None = None,
        interactive: bool = False,
        mode: str | None = None,
        voice_intro_path: Path | None = None,
        voice_outro_path: Path | None = None,
        force: bool = False,
        voice_breakdown_path: Path | None = None,
    ) -> list[ClipResult]:
        """Process a single stream through the full resumable pipeline."""
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Invalid URL: stream URL must be a non-empty string.")
        clean_url = url.strip()

        # Fail fast on invalid mode before clearing cache or running expensive pipeline stages
        self._resolve_base_mode(mode)

        intro_path = Path(voice_intro_path) if voice_intro_path is not None else None
        outro_path = Path(voice_outro_path) if voice_outro_path is not None else None
        breakdown_path = (
            Path(voice_breakdown_path) if voice_breakdown_path is not None else None
        )
        if intro_path is not None and not intro_path.exists():
            logger.warning("Custom intro voiceover path does not exist yet: %s", intro_path)
        if outro_path is not None and not outro_path.exists():
            logger.warning("Custom outro voiceover path does not exist yet: %s", outro_path)
        if breakdown_path is not None and not breakdown_path.exists():
            logger.warning(
                "Custom breakdown voiceover path does not exist yet: %s", breakdown_path
            )

        if force:
            self.cache.clear(clean_url)

        state = self.cache.load_state(clean_url) or PipelineState(
            stream_info=StreamInfo(
                url=clean_url,
                title="",
                platform=Platform.YOUTUBE,
                streamer=streamer or "",
                game=game or "",
                duration_seconds=0,
            )
        )
        if game:
            state.stream_info.game = game
        if streamer:
            state.stream_info.streamer = streamer

        if self._is_blocked_streamer(state.stream_info.streamer):
            logger.warning(
                "Streamer %r is listed in discovery.blocked_streamers; refusing to process %s.",
                state.stream_info.streamer,
                clean_url,
            )
            return []

        requested_clips = (
            int(max_clips)
            if max_clips is not None
            else int(self.config.get("analysis.max_clips_per_stream", 5))
        )
        # StepCache does not persist requested_max_clips yet, so fall back to the number
        # of highlights the cached scoring produced (score_and_select returns <= max_clips).
        stored_max_clips = (
            state.requested_max_clips
            if state.requested_max_clips is not None
            else (len(state.highlights) or None)
        )
        if (
            state.status >= PipelineStatus.ASSEMBLING
            and state.status != PipelineStatus.FAILED
            and max_clips is not None
            and stored_max_clips is not None
            and requested_clips > stored_max_clips
        ):
            logger.warning(
                "Cached scoring used max_clips=%d but %d requested; re-running scoring.",
                stored_max_clips,
                requested_clips,
            )
            state.status = PipelineStatus.SCORING
            state.selected_highlights = []
            state.clips = []

        if state.status == PipelineStatus.COMPLETE and (
            mode is not None
            or max_clips is not None
            or intro_path is not None
            or outro_path is not None
        ):
            logger.warning(
                "Returning cached COMPLETE results for %s; pass force=True (--force) "
                "to re-run with updated options.",
                clean_url,
            )

        state = self._recover_failed_state(state)

        try:
            # Step 2: Download audio (skip if cached)
            if state.status < PipelineStatus.TRANSCRIBING:
                state = await self._step_download_audio(state)
                self.cache.save_state(state)
                if self._is_blocked_streamer(state.stream_info.streamer):
                    logger.warning(
                        "Streamer %r is listed in discovery.blocked_streamers; "
                        "stopping after audio download.",
                        state.stream_info.streamer,
                    )
                    return []

            # Step 3: Transcribe (skip if cached; serialized on GPU lock)
            if state.status < PipelineStatus.ANALYZING:
                async with self._gpu_guard():
                    state = await self._step_transcribe(state)
                self.cache.save_state(state)

            # Step 4: Analyze signals (skip if cached; re-download missing WAV while preserving transcript)
            if state.status < PipelineStatus.SCORING:
                if state.audio_path is None or not Path(state.audio_path).exists():
                    state = await self._step_download_audio(state)
                    self.cache.save_state(state)
                state = await self._step_analyze(state)
                self.cache.save_state(state)

            # Step 5: Score with LLM (skip if cached; serialized on GPU lock)
            if state.status < PipelineStatus.ASSEMBLING:
                async with self._gpu_guard():
                    state = await self._step_score(state, max_clips=max_clips)
                self.cache.save_state(state)

            # Step 6: Select highlights (interactive only when explicitly requested)
            if state.status < PipelineStatus.COMPLETE and not state.selected_highlights:
                if interactive:
                    state.selected_highlights = self._interactive_select(state.highlights)
                else:
                    limit = (
                        max_clips
                        if max_clips is not None
                        else int(self.config.get("analysis.max_clips_per_stream", 5))
                    )
                    state.selected_highlights = state.highlights[: max(0, limit)]
                self.cache.save_state(state)

            # Step 7: Assemble clips
            if state.status < PipelineStatus.COMPLETE:
                state = await self._step_assemble(
                    state,
                    mode=mode,
                    voice_intro_path=intro_path,
                    voice_outro_path=outro_path,
                    voice_breakdown_path=breakdown_path,
                    interactive=interactive,
                )
                self.cache.save_state(state)
        except Exception:
            state.status = PipelineStatus.FAILED
            self.cache.save_state(state)
            raise

        return state.clips

    async def prepare_stream(
        self,
        url: str,
        *,
        game: str | None = None,
        streamer: str | None = None,
        force: bool = False,
    ) -> PipelineState:
        """Run download-audio, transcribe, and analyze steps, stopping before scoring."""
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Invalid URL: stream URL must be a non-empty string.")
        clean_url = url.strip()

        if force:
            self.cache.clear(clean_url)

        state = self.cache.load_state(clean_url) or PipelineState(
            stream_info=StreamInfo(
                url=clean_url,
                title="",
                platform=Platform.YOUTUBE,
                streamer=streamer or "",
                game=game or "",
                duration_seconds=0,
            )
        )
        if game:
            state.stream_info.game = game
        if streamer:
            state.stream_info.streamer = streamer

        if self._is_blocked_streamer(state.stream_info.streamer):
            logger.warning(
                "Streamer %r is listed in discovery.blocked_streamers; refusing to process %s.",
                state.stream_info.streamer,
                clean_url,
            )
            return state

        state = self._recover_failed_state(state)

        try:
            # Step 2: Download audio (skip if cached)
            if state.status < PipelineStatus.TRANSCRIBING:
                state = await self._step_download_audio(state)
                self.cache.save_state(state)
                if self._is_blocked_streamer(state.stream_info.streamer):
                    logger.warning(
                        "Streamer %r is listed in discovery.blocked_streamers; "
                        "stopping after audio download.",
                        state.stream_info.streamer,
                    )
                    return state

            # Step 3: Transcribe (skip if cached; serialized on GPU lock)
            if state.status < PipelineStatus.ANALYZING:
                async with self._gpu_guard():
                    state = await self._step_transcribe(state)
                self.cache.save_state(state)

            # Step 4: Analyze signals (skip if cached; re-download missing WAV while preserving transcript)
            if state.status < PipelineStatus.SCORING:
                if state.audio_path is None or not Path(state.audio_path).exists():
                    state = await self._step_download_audio(state)
                    self.cache.save_state(state)
                state = await self._step_analyze(state)
                self.cache.save_state(state)

            return state
        except Exception:
            state.status = PipelineStatus.FAILED
            self.cache.save_state(state)
            raise

    async def process_streams_parallel(
        self,
        urls: list[str],
        max_concurrent: int = 2,
        *,
        max_clips: int | None = None,
        game: str | None = None,
        mode: str | None = None,
        voice_intro_path: Path | None = None,
        voice_outro_path: Path | None = None,
        force: bool = False,
        voice_breakdown_path: Path | None = None,
    ) -> list[list[ClipResult] | BaseException]:
        """Process multiple streams in parallel (always non-interactive, GPU-serialized)."""
        concurrency = max(1, int(max_concurrent))
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(url: str) -> list[ClipResult]:
            async with semaphore:
                kwargs: dict[str, Any] = {
                    "max_clips": max_clips,
                    "game": game,
                    "interactive": False,
                    "mode": mode,
                    "voice_intro_path": voice_intro_path,
                    "voice_outro_path": voice_outro_path,
                }
                if voice_breakdown_path is not None:
                    kwargs["voice_breakdown_path"] = voice_breakdown_path
                if force:
                    kwargs["force"] = True
                return await self.process_stream(url, **kwargs)

        tasks = [_bounded(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    async def _step_download_audio(self, state: PipelineState) -> PipelineState:
        """Download 16kHz mono WAV audio and populate StreamInfo."""
        state.status = PipelineStatus.DOWNLOADING
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        audio_path, downloaded_info = await asyncio.to_thread(
            download_audio,
            state.stream_info.url,
            self.temp_dir,
        )

        # Preserve canonical cache URL and explicit caller overrides while filling detected metadata
        resolved_game = (
            state.stream_info.game
            or downloaded_info.game
            or str(self.config.get("discovery.default_game", "GTA 6"))
        )
        state.stream_info = StreamInfo(
            url=state.stream_info.url,
            title=downloaded_info.title or state.stream_info.title,
            platform=downloaded_info.platform,
            streamer=state.stream_info.streamer or downloaded_info.streamer,
            game=resolved_game,
            duration_seconds=(
                downloaded_info.duration_seconds or state.stream_info.duration_seconds
            ),
            view_count=downloaded_info.view_count or state.stream_info.view_count,
            video_id=downloaded_info.video_id or state.stream_info.video_id,
        )
        state.audio_path = Path(audio_path)
        state.status = (
            PipelineStatus.ANALYZING
            if state.transcript is not None
            else PipelineStatus.TRANSCRIBING
        )
        return state

    async def _step_transcribe(self, state: PipelineState) -> PipelineState:
        """Transcribe audio into word-aligned segments."""
        if state.audio_path is None:
            raise RuntimeError("Cannot transcribe: audio_path is missing from pipeline state.")

        transcriber = Transcriber(self.config)
        state.transcript = await asyncio.to_thread(transcriber.transcribe, state.audio_path)
        state.status = (
            PipelineStatus.SCORING
            if state.analysis_complete
            else PipelineStatus.ANALYZING
        )
        return state

    def _compute_heatmap_spikes(self, state: PipelineState, video_id: str) -> list[TimeRange]:
        """YouTube 'Most replayed' peaks; any failure yields [] (optional signal)."""
        try:
            if not self.config.get("analysis.heatmap.enabled", True):
                return []
            segments = get_heatmap(state.stream_info.url, self.temp_dir, video_id)
            return HeatmapAnalyzer(self.config).detect(segments)
        except Exception as exc:
            logger.debug("Heatmap analysis skipped: %s", exc)
            return []

    async def _step_analyze(self, state: PipelineState) -> PipelineState:
        """Run streamed loudness spike detection and live chat velocity analysis atomically."""
        if state.audio_path is None:
            raise RuntimeError("Cannot analyze: audio_path is missing from pipeline state.")

        loudness_detector = LoudnessDetector(self.config)
        loudness_spikes = await asyncio.to_thread(
            loudness_detector.detect, state.audio_path
        )

        scraper = ChatScraper()
        raw_video_id = str(state.stream_info.video_id or "").strip()
        if (
            raw_video_id
            and "/" not in raw_video_id
            and "\\" not in raw_video_id
            and ":" not in raw_video_id
            and ".." not in raw_video_id
        ):
            video_id = _sanitize_path_segment(
                raw_video_id, fallback=self.cache.get_cache_key(state.stream_info.url)
            )
        else:
            video_id = self.cache.get_cache_key(state.stream_info.url)
        chat_cache_path = self.temp_dir / f"{video_id}_chat.csv"

        def _fetch_chat() -> Any:
            cached = scraper.load_cache(chat_cache_path)
            if cached is not None:
                return cached
            df = scraper.scrape(state.stream_info.url, self.temp_dir)
            # Persist even an empty (header-only) frame so "no chat" is not re-scraped.
            scraper.save_cache(df, chat_cache_path)
            return df

        chat_df = await asyncio.to_thread(_fetch_chat)
        chat_analyzer = ChatVelocityAnalyzer(self.config)
        chat_spikes = await asyncio.to_thread(chat_analyzer.analyze, chat_df)

        heatmap_spikes = await asyncio.to_thread(self._compute_heatmap_spikes, state, video_id)

        # Commit all signal lists atomically so partial failure never skips chat analysis on retry
        state.loudness_spikes = loudness_spikes
        state.chat_spikes = chat_spikes
        state.heatmap_spikes = heatmap_spikes
        state.analysis_complete = True
        state.status = PipelineStatus.SCORING
        return state

    async def _step_score(
        self, state: PipelineState, max_clips: int | None = None
    ) -> PipelineState:
        """Verify Ollama connectivity and run tri-modal highlight scoring."""
        if state.transcript is None:
            raise RuntimeError("Cannot score: transcript is missing from pipeline state.")

        scorer = OllamaScorer(self.config)
        if max_clips is not None:
            scorer.max_clips = max(0, int(max_clips))
        await asyncio.to_thread(scorer.check_connection)
        state.highlights = await asyncio.to_thread(
            scorer.score_and_select,
            state.transcript,
            state.loudness_spikes,
            state.chat_spikes,
            audio_path=state.audio_path,
            **({"heatmap_spikes": state.heatmap_spikes} if state.heatmap_spikes else {}),
        )
        state.requested_max_clips = (
            int(max_clips)
            if max_clips is not None
            else int(self.config.get("analysis.max_clips_per_stream", 5))
        )
        state.status = PipelineStatus.ASSEMBLING
        return state

    async def _step_assemble(
        self,
        state: PipelineState,
        mode: str | None = None,
        voice_intro_path: Path | None = None,
        voice_outro_path: Path | None = None,
        voice_breakdown_path: Path | None = None,
        interactive: bool = False,
    ) -> PipelineState:
        """Download highlight video segments, generate metadata, and assemble/export clips."""
        base_mode = self._resolve_base_mode(mode)

        if not state.selected_highlights:
            state.clips = []
            state.status = PipelineStatus.COMPLETE
            return state

        scorer = OllamaScorer(self.config)
        supported_meta_kwargs: set[str] = set()
        try:
            meta_sig = inspect.signature(scorer.generate_metadata)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in meta_sig.parameters.values()
            )
            for kw_name in ("mode", "hook_style", "avoid_phrases", "language"):
                if kw_name in meta_sig.parameters or has_var_kw:
                    supported_meta_kwargs.add(kw_name)
        except (TypeError, ValueError):
            supported_meta_kwargs = set()

        variety = VarietyState(
            Path(self.config.get("general.cache_dir", "./.cache")) / "variety_state.json",
            hook_styles=list(self.config.get("transformation.hook_styles", None) or []) or None,
        )
        rotation = [
            str(f)
            for f in (
                self.config.get(
                    "transformation.format_rotation", ["narrative", "story", "breakdown"]
                )
                or []
            )
        ]
        daily_cap = int(self.config.get("output.max_clips_per_streamer_per_day", 3) or 0)
        gate_enabled = bool(self.config.get("content_gate.enabled", True))

        segments_dir = self.temp_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        streamer_name = state.stream_info.streamer or "Unknown"
        game_name = state.stream_info.game or str(
            self.config.get("discovery.default_game", "GTA 6")
        )

        narration_language = resolve_narration_language(
            self.config,
            state.stream_info.streamer,
            state.transcript.language if state.transcript else None,
        )
        skip_high_music = bool(self.config.get("analysis.skip_high_music", False))
        buffer_seconds = 3.0
        for idx, highlight in enumerate(state.selected_highlights, start=1):
            if self._already_rendered(state, highlight):
                logger.info("Skipping already-rendered highlight %d on resume.", idx)
                continue
            if (
                skip_high_music
                and not interactive
                and highlight.music_risk == "HIGH_MUSIC_DETECTED"
            ):
                logger.warning(
                    "Skipping highlight %d (%.1f-%.1f): HIGH_MUSIC_DETECTED and "
                    "analysis.skip_high_music is enabled.",
                    idx,
                    highlight.start,
                    highlight.end,
                )
                continue
            clip_mode, valid_beats = self._resolve_highlight_mode(base_mode, highlight)
            if base_mode == "auto" and clip_mode in ("narrative", "story") and rotation:
                # Story first: valid beats are never discarded on a tie (never-used formats).
                available = [
                    f
                    for f in ("story", "narrative", "breakdown")
                    if f in rotation
                    and (f != "story" or len(valid_beats) >= 2)
                    and (f != "breakdown" or not self._breakdown_ineligible_reason(highlight))
                ]
                if available:
                    chosen = variety.next_format(available)
                    if chosen == "breakdown":
                        clip_mode, valid_beats = "breakdown", []
                    elif chosen == "narrative":
                        clip_mode, valid_beats = "narrative", []
            use_multi_beat = clip_mode == "story" and len(valid_beats) >= 2
            if clip_mode != "story" and valid_beats:
                logger.info(
                    "Ignoring %d validated sub_clips beats for highlight %d (mode=%s).",
                    len(valid_beats),
                    idx,
                    clip_mode,
                )
            if use_multi_beat:
                beat_ranges = [
                    (float(beat["start"]), float(beat["end"]))
                    for beat in valid_beats
                ]
                segment_paths = await asyncio.to_thread(
                    download_multiple_segments,
                    state.stream_info.url,
                    beat_ranges,
                    segments_dir,
                    buffer_seconds,
                    video_id=state.stream_info.video_id or None,
                )
            else:
                single_path = await asyncio.to_thread(
                    download_segment,
                    state.stream_info.url,
                    highlight.start,
                    highlight.end,
                    segments_dir,
                    buffer_seconds,
                    video_id=state.stream_info.video_id or None,
                )
                segment_paths = [single_path]

            gate_non_gameplay = False
            if gate_enabled:
                gate_non_gameplay = await asyncio.to_thread(
                    self._assess_gameplay, segment_paths[0]
                )

            hook_style = variety.next_hook_style()
            meta_kwargs: dict[str, Any] = {}
            if "mode" in supported_meta_kwargs:
                meta_kwargs["mode"] = clip_mode
            if "hook_style" in supported_meta_kwargs:
                meta_kwargs["hook_style"] = hook_style
            if "avoid_phrases" in supported_meta_kwargs:
                meta_kwargs["avoid_phrases"] = variety.recent_outros(20)
            if "language" in supported_meta_kwargs:
                meta_kwargs["language"] = narration_language

            async with self._gpu_guard():
                try:
                    metadata = await asyncio.to_thread(
                        scorer.generate_metadata,
                        highlight,
                        streamer_name,
                        game_name,
                        state.stream_info.url,
                        **meta_kwargs,
                    )
                except Exception as exc:
                    logger.warning("LLM metadata generation failed (%s); using offline metadata from highlight.", exc)
                    game_clean = re.sub(r"\W+", "", game_name) or "GTA6"
                    streamer_clean = re.sub(r"\W+", "", streamer_name) or "Gaming"
                    metadata = ClipMetadata(
                        hook_title=highlight.title or "Stream Highlight",
                        caption=f"{highlight.title or 'Stream Highlight'}\n{highlight.reason or ''}\nStreamer: {streamer_name} | Game: {game_name}",
                        hashtags=[game_clean, f"{game_clean}Clips", streamer_clean, "Shorts", "Gaming"],
                        streamer=streamer_name,
                        game=game_name,
                        source_url=state.stream_info.url,
                        clip_start=highlight.start,
                        clip_end=highlight.end,
                        highlight_score=highlight.final_score or highlight.score,
                        moment_type=highlight.moment_type,
                        narration_intro=highlight.narration_intro,
                        narration_outro=highlight.narration_outro,
                        context_badge=highlight.context_badge,
                        transformation_mode=clip_mode,
                        breakdown_commentary=highlight.breakdown_commentary,
                        breakdown_callout=highlight.breakdown_callout,
                        music_risk=highlight.music_risk,
                        music_confidence=highlight.music_confidence,
                        narration_language=narration_language,
                        caption_style=getattr(highlight, "caption_style", "auto"),
                    )
                metadata.transformation_mode = clip_mode
                metadata.narration_language = narration_language
                if clip_mode == "raw":
                    metadata.narration_intro = ""
                    metadata.narration_outro = ""
                    metadata.context_badge = ""
                else:
                    self._ensure_intro_narration(
                        metadata,
                        highlight,
                        scorer,
                        streamer_name,
                        game_name,
                        hook_style=hook_style,
                        language=narration_language,
                        has_custom_voice=voice_intro_path is not None,
                    )

            queued = bool(
                daily_cap > 0 and variety.clips_today(streamer_name) >= daily_cap
            )
            assemble_guard = (
                self._gpu_guard()
                if self._narration_may_use_gpu(clip_mode)
                else _AsyncNull()
            )
            caption_preset = self._next_caption_preset_index(variety, streamer_name)
            async with assemble_guard:
                clip_result = await asyncio.to_thread(
                    self._assemble_highlight,
                    highlight,
                    metadata,
                    segment_paths,
                    idx,
                    clip_mode,
                    voice_intro_path,
                    voice_outro_path,
                    valid_beats=valid_beats,
                    buffer_seconds=buffer_seconds,
                    voice_breakdown_path=voice_breakdown_path,
                    gate_non_gameplay=gate_non_gameplay,
                    queued=queued,
                    stream_info=state.stream_info,
                    caption_preset=caption_preset,
                    caption_style=getattr(highlight, "caption_style", "auto"),
                )
            if clip_result is None:
                continue
            if getattr(clip_result, "quarantined", False):
                logger.warning(
                    "Clip %d not counted as produced (quarantined: %s).",
                    idx,
                    clip_result.quarantine_reason,
                )
                continue
            state.clips.append(clip_result)
            try:
                variety.record(
                    streamer_name,
                    clip_mode,
                    hook_style,
                    metadata.narration_outro or "",
                    caption_preset=caption_preset,
                )
            except Exception as exc:
                logger.warning("Could not record variety state: %s", exc)
            self.cache.save_state(state)

        if (
            base_mode != "raw"
            and bool(self.config.get("transformation.enable_top3", True))
            and len(state.selected_highlights) >= 3
            and not any(c.metadata.transformation_mode == "top3" for c in state.clips)
        ):
            try:
                top3 = await self._build_top3(
                    state, scorer, segments_dir, buffer_seconds, streamer_name, game_name
                )
                if top3 is not None and not top3.quarantined:
                    state.clips.append(top3)
                    self.cache.save_state(state)
            except Exception as exc:
                logger.warning("Top-3 compilation failed: %s", exc)

        state.status = PipelineStatus.COMPLETE
        return state

    def _assess_gameplay(self, segment_path: Path) -> bool:
        """True when the content gate flags the segment as NON-gameplay (False if gate unavailable)."""
        try:
            from .analysis.content_gate import assess_segment
        except ImportError:
            return False
        try:
            result = assess_segment(segment_path)
        except Exception as exc:
            logger.warning("Content gate failed for %s (%s); assuming gameplay.", segment_path, exc)
            return False
        is_gameplay = bool(getattr(result, "is_gameplay", True))
        if not is_gameplay:
            logger.info(
                "Content gate: non-gameplay segment (%s).", getattr(result, "reason", "")
            )
        return not is_gameplay

    _TOP3_TITLE_TEMPLATES = (
        "Top 3 {game} moments from {streamer}",
        "{streamer}'s 3 craziest {game} plays, ranked",
        "Ranking {streamer}'s best {game} moments",
        "3 {game} clips from {streamer} you can't miss",
        "Which {streamer} {game} moment wins? Ranked",
    )

    def _top3_title(
        self,
        scorer: Any,
        top: list[Highlight],
        streamer_name: str,
        game_name: str,
        url: str,
        hook_style: str,
        hook_styles: list[str],
        language: str,
    ) -> tuple[str, str, list[str]]:
        """(title, caption, hashtags): LLM hook when available, else a rotating template."""
        gen = getattr(scorer, "generate_metadata", None)
        if callable(gen):
            try:
                sig = inspect.signature(gen)
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                kwargs: dict[str, Any] = {}
                for name, val in (
                    ("mode", "narrative"),
                    ("hook_style", hook_style),
                    ("language", language),
                ):
                    if name in sig.parameters or has_var_kw:
                        kwargs[name] = val
                meta = gen(top[0], streamer_name, game_name, url, **kwargs)
                title = str(getattr(meta, "hook_title", "") or "").strip()
                if title:
                    caption = str(getattr(meta, "caption", "") or "").strip() or (
                        f"{streamer_name}'s top 3 {game_name} moments, ranked."
                    )
                    tags = [str(t) for t in (getattr(meta, "hashtags", None) or [])]
                    if "Top3" not in tags and "#Top3" not in tags:
                        tags.append("Top3")
                    return title, caption, tags
            except Exception as exc:
                logger.warning("Top-3 hook generation failed (%s); using title template.", exc)
        try:
            idx = hook_styles.index(hook_style)
        except ValueError:
            idx = 0
        template = self._TOP3_TITLE_TEMPLATES[idx % len(self._TOP3_TITLE_TEMPLATES)]
        return (
            template.format(game=game_name, streamer=streamer_name),
            f"{streamer_name}'s top 3 {game_name} moments, ranked.",
            ["Top3", "Shorts", "Gaming"],
        )

    async def _build_top3(
        self,
        state: PipelineState,
        scorer: Any,
        segments_dir: Path,
        buffer_seconds: float,
        streamer_name: str,
        game_name: str,
    ) -> ClipResult | None:
        """Render one countdown Short from the 3 best selected highlights (E9)."""
        from .assembly.compilation import build_top3_compilation
        from .assembly.facecam import FacecamDetector
        from .assembly.narrator import TransformativeNarrator
        from .output.organizer import FileOrganizer

        variety = self._variety_state()
        daily_cap = int(self.config.get("output.max_clips_per_streamer_per_day", 3) or 0)
        skip_high_music = bool(self.config.get("analysis.skip_high_music", False))
        gate_enabled = bool(self.config.get("content_gate.enabled", True))

        ranked = sorted(
            state.selected_highlights,
            key=lambda h: float(h.final_score or h.score),
            reverse=True,
        )
        top: list[Highlight] = []
        top_segs: list[Path] = []
        for h in ranked:
            if len(top) >= 3:
                break
            if skip_high_music and h.music_risk == "HIGH_MUSIC_DETECTED":
                logger.info("Top-3: skipping %.1f-%.1f (HIGH_MUSIC_DETECTED).", h.start, h.end)
                continue
            seg = await asyncio.to_thread(
                download_segment,
                state.stream_info.url,
                h.start,
                h.end,
                segments_dir,
                buffer_seconds,
                video_id=state.stream_info.video_id or None,
            )
            if gate_enabled and await asyncio.to_thread(self._assess_gameplay, seg):
                logger.info("Top-3: skipping %.1f-%.1f (non-gameplay segment).", h.start, h.end)
                continue
            top.append(h)
            top_segs.append(seg)
        if len(top) < 3:
            logger.info("Top-3 compilation skipped: only %d eligible highlights.", len(top))
            return None

        order = list(range(len(top) - 1, -1, -1))  # #3 -> #1
        play_order = [top[i] for i in order]
        n_items = len(play_order)
        lines: list[str] = []
        gen = getattr(scorer, "generate_countdown_lines", None)
        if callable(gen):
            lines = [
                str(x)
                for x in await asyncio.to_thread(gen, play_order, streamer_name, game_name)
            ][:n_items]
        lines += [f"Number {n_items - i}" for i in range(len(lines), n_items)]

        segs: list[Path] = []
        peaks: list[float | None] = []
        cams: list[Any] = []
        detector = FacecamDetector(self.config)
        for i, h in zip(order, play_order):
            seg = top_segs[i]
            seg_start = max(0.0, float(h.start) - buffer_seconds)
            segs.append(seg)
            peaks.append(max(0.0, h.peak_time - seg_start) if h.peak_time > 0 else None)
            cams.append(await asyncio.to_thread(detector.detect, [seg], streamer=streamer_name))

        organizer = FileOrganizer(self.config)
        queued = bool(daily_cap > 0 and variety.clips_today(streamer_name) >= daily_cap)
        path_getter = (
            getattr(organizer, "get_queued_path", organizer.get_clip_path)
            if queued
            else organizer.get_clip_path
        )
        out_path = path_getter(
            game=game_name,
            streamer=streamer_name,
            index=organizer.next_available_index(
                game_name, streamer_name, len(state.selected_highlights) + 1
            ),
        )
        language = resolve_narration_language(
            self.config,
            state.stream_info.streamer,
            state.transcript.language if state.transcript else None,
        )
        narrator = TransformativeNarrator(self.config)
        narrator.language = language
        info: dict[str, Any] = {}
        guard = self._gpu_guard() if self._narration_may_use_gpu("narrative") else _AsyncNull()
        try:
            async with guard:
                final = await asyncio.to_thread(
                    build_top3_compilation,
                    segs,
                    peaks,
                    cams,
                    lines,
                    out_path,
                    self.config,
                    narrator=narrator,
                    info=info,
                )
        finally:
            release = getattr(narrator, "release", None)
            if callable(release):
                release()

        hook_style = variety.next_hook_style()
        title, caption, hashtags = await asyncio.to_thread(
            self._top3_title,
            scorer,
            top,
            streamer_name,
            game_name,
            state.stream_info.url,
            hook_style,
            list(variety.hook_styles),
            language,
        )
        metadata = ClipMetadata(
            hook_title=title,
            caption=caption,
            hashtags=hashtags,
            streamer=streamer_name,
            game=game_name,
            source_url=state.stream_info.url,
            clip_start=min(h.start for h in top),
            clip_end=max(h.end for h in top),
            highlight_score=float(top[0].final_score or top[0].score),
            moment_type="hype",
            transformation_mode="top3",
            narration_language=language,
        )
        result = organizer.save_clip(final, metadata, transcript_text="")
        reason = self._narration_gate_reason(info)
        if reason:
            return self._quarantine_clip(result, f"top-3 countdown voice missing: {reason}")
        measured = self._measured_commentary(info, result, "top3")
        if measured:
            self._annotate_clip_json(result, measured)
        self._write_post_file_safe(result.video_path, metadata, state.stream_info)
        try:
            variety.record(streamer_name, "top3", hook_style, "")
        except Exception as exc:
            logger.warning("Could not record variety state for top-3: %s", exc)
        return result

    def _write_post_file_safe(
        self, video_path: Path, metadata: ClipMetadata, stream_info: StreamInfo | None
    ) -> None:
        """E11: write the `<stem>_post.txt` helper file; never fails the render."""
        if not bool(self.config.get("output.write_post_file", True)):
            return
        try:
            info = stream_info or StreamInfo(
                url=metadata.source_url,
                title="",
                platform=Platform.YOUTUBE,
                streamer=metadata.streamer,
                game=metadata.game,
                duration_seconds=0,
            )
            write_post_file(Path(video_path), metadata, info)
        except Exception as exc:
            logger.warning("Could not write post file for %s: %s", video_path, exc)

    @staticmethod
    def _probe_duration(video_path: Path) -> float | None:
        import subprocess

        try:
            proc = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, check=True,
            )
            return float(proc.stdout.strip())
        except Exception:
            return None

    def _hard_trim(self, video_path: Path, max_seconds: float) -> float | None:
        """Probe the exported clip and cut it to `max_seconds` (stream copy) when longer."""
        import subprocess

        duration = self._probe_duration(video_path)
        if duration is None or duration <= max_seconds + 0.05:
            return duration
        tmp = Path(video_path).with_name(f"{Path(video_path).stem}_trim{Path(video_path).suffix}")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path), "-t", f"{max_seconds:.3f}",
                 "-c", "copy", str(tmp)],
                check=True, capture_output=True,
            )
            os.replace(tmp, video_path)
            logger.warning(
                "Hard-trimmed %s from %.1fs to %.1fs (output.max_final_seconds).",
                video_path, duration, max_seconds,
            )
            return max_seconds
        except Exception as exc:
            logger.warning("Hard trim failed for %s: %s", video_path, exc)
            try:
                tmp.unlink()
            except OSError:
                pass
            return duration

    def _plan_budget(
        self,
        *,
        moment_dur: float,
        mode: str,
        replay_style: str = "instant_replay",
        replay_window: float = 5.0,
        replay_speed: float = 0.5,
        freeze_duration: float = 4.0,
        spotlight_duration: float = 5.7,
        narration_seconds: float = 0.0,
        callout_seconds: float = 0.0,
        can_trim_moment: bool = True,
        breakdown_voiced: bool = False,
        breakdown_seconds: float = 0.0,
    ) -> dict[str, float]:
        """E5: fit moment + replay inside output.max_final_seconds and raise commentary coverage.

        Order: shrink the replay window first, then trim the moment (around the peak, by the
        caller). Returns moment, replay_window, replay_len, total and commentary coverage.
        """
        max_final = float(self.config.get("output.max_final_seconds", 180))
        min_ratio = float(self.config.get("output.min_commentary_ratio", 0.35))
        speed = max(0.1, float(replay_speed))
        replay_on = mode == "breakdown"
        freeze = replay_style == "freeze_frame"
        zoom_spotlight = replay_style == "zoom_spotlight"
        window = float(replay_window)

        def replay_len(win: float) -> float:
            if not replay_on:
                return 0.0
            if zoom_spotlight:
                return float(spotlight_duration)
            return float(freeze_duration) if freeze else win / speed

        moment = float(moment_dur)
        rlen = replay_len(window)
        if moment + rlen > max_final and replay_on and not freeze and not zoom_spotlight:
            window = max(2.0, min(window, max(0.0, max_final - moment) * speed))
            rlen = replay_len(window)
        if moment + rlen > max_final and can_trim_moment:
            moment = max(6.0, max_final - rlen)
        commentary = (narration_seconds + callout_seconds) if mode != "raw" else 0.0
        # The slow-mo / spotlight replay only counts as commentary when breakdown speech is voiced over it,
        # and then only for as long as that speech lasts.
        replay_credit = min(rlen, max(0.0, float(breakdown_seconds))) if breakdown_voiced else 0.0
        total = moment + rlen
        coverage = (commentary + replay_credit) / total if total > 0 else 0.0
        if (
            mode != "raw"
            and can_trim_moment
            and commentary > 0
            and coverage < min_ratio
            and min_ratio > 0
        ):
            # Flag low commentary but NEVER trim gameplay footage —
            # the full event context is more important than hitting a coverage ratio.
            logger.info(
                "Commentary coverage %.1f%% < min %.1f%%; flagging low_commentary "
                "(NOT trimming gameplay)",
                coverage * 100,
                min_ratio * 100,
            )
        return {
            "moment": round(moment, 3),
            "replay_window": round(window, 3),
            "replay_len": round(rlen, 3),
            "total": round(total, 3),
            "coverage": round(coverage, 3),
        }

    def _annotate_clip_json(self, result: ClipResult, extra: dict[str, Any]) -> None:
        """Merge extra keys (commentary coverage) into the saved clip metadata JSON."""
        import json

        path = getattr(result, "metadata_path", None)
        if not path or not Path(path).is_file():
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            data.update(extra)
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Could not annotate clip JSON %s: %s", path, exc)

    @staticmethod
    def _already_rendered(state: PipelineState, highlight: Highlight) -> bool:
        """True when state.clips already holds an existing video for this highlight window."""
        for clip in state.clips:
            meta = clip.metadata
            if (
                abs(float(meta.clip_start) - float(highlight.start)) <= 0.01
                and abs(float(meta.clip_end) - float(highlight.end)) <= 0.01
                and Path(clip.video_path).exists()
            ):
                return True
        return False

    def _resolve_breakdown_target_time_and_roi(
        self,
        highlight: Highlight,
        source_video_path: Path | str | None,
        clip_start: float,
        clip_end: float,
    ) -> tuple[float, list[float]]:
        """4-tier visual localization cascade to pinpoint screen/document timestamp and ROI.

        Tier 1: Explicit target_time and roi provided via Highlight / MCP / CLI.
        Tier 2: Multimodal VLM inspection (Gemini Flash or contact sheet refinement metadata).
        Tier 3: OpenCV text and high-frequency edge density saliency scan across candidate frames.
        Tier 4: Setup-phase heuristic: t = clip_start + min(2.0s, 0.20 * duration), screen ROI.
        """
        clip_dur = max(0.1, clip_end - clip_start)

        def _is_reverse_angle(t_val: float) -> bool:
            """Check whether a timestamp falls into a reverse-angle window facing away from display."""
            if 19180.0 <= t_val <= 19200.0:
                return True
            rel = t_val - clip_start
            return rel > min(8.0, 0.45 * clip_dur)

        # Screen-focus detection: laptop, display, monitor, classified files, secret documents
        screen_keywords = (
            "screen", "laptop", "monitor", "display", "file", "drive",
            "document", "classified", "text", "read", "secret", "clue",
        )
        text_context = " ".join([
            str(getattr(highlight, "title", "") or ""),
            str(getattr(highlight, "reason", "") or ""),
            str(getattr(highlight, "breakdown_focus", "") or ""),
            str(getattr(highlight, "breakdown_commentary", "") or ""),
        ]).lower()
        is_screen_focus = (
            any(k in text_context for k in screen_keywords)
            or str(getattr(highlight, "moment_type", "") or "").lower() in {"analysis", "graphics_detail", "lore_secret"}
        )
        default_screen_roi = [0.45, 0.25, 0.78, 0.72] if is_screen_focus else [0.20, 0.20, 0.80, 0.80]
        setup_offset = min(2.0, 0.20 * clip_dur)

        # Tier 1: Explicit inputs
        explicit_t = getattr(highlight, "breakdown_target_time", None)
        explicit_roi = list(getattr(highlight, "breakdown_roi", []) or [])
        has_valid_roi = (
            len(explicit_roi) == 4
            and (explicit_roi != [0.0, 0.0, 1.0, 1.0])
            and (explicit_roi != [0.20, 0.20, 0.80, 0.80])
        )

        if explicit_t is not None:
            t = float(explicit_t)
            if t < clip_start and (clip_start + t) <= clip_end:
                t = clip_start + t
            if clip_start <= t <= clip_end:
                if _is_reverse_angle(t):
                    logger.info(
                        "Breakdown target_time=%.2fs falls into a reverse-angle window; "
                        "falling back to setup phase screen analysis.",
                        t,
                    )
                elif has_valid_roi:
                    logger.info("Breakdown Tier 1 (Explicit) resolved target_time=%.2fs, roi=%s", t, explicit_roi)
                    return t, explicit_roi

        # Tier 2: Multimodal VLM inspection metadata (if present on highlight)
        vlm_t = getattr(highlight, "_vlm_breakdown_target_time", None)
        vlm_roi = getattr(highlight, "_vlm_breakdown_roi", None)
        if vlm_t is not None and clip_start <= float(vlm_t) <= clip_end:
            vt = float(vlm_t)
            if not _is_reverse_angle(vt):
                roi = list(vlm_roi) if vlm_roi and len(vlm_roi) == 4 else default_screen_roi
                logger.info("Breakdown Tier 2 (VLM) resolved target_time=%.2fs, roi=%s", vt, roi)
                return vt, roi

        # Tier 3: OpenCV text and high-frequency edge saliency scan across candidate frames
        if source_video_path is not None and Path(source_video_path).exists():
            try:
                import cv2
                src_str = str(source_video_path)
                cap = cv2.VideoCapture(src_str)
                if cap.isOpened():
                    try:
                        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                        total_f = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                        file_dur = total_f / fps if fps > 0 else 0.0
                        lead_pad = max(0.0, min(3.0, (file_dur - clip_dur) / 2.0)) if file_dur > clip_dur else 0.0

                        # Candidate scan offsets: prioritize the setup phase where screens/documents face camera
                        offsets = [setup_offset, 1.0, 2.0, 3.0]
                        candidate_offsets = sorted(set(round(o, 2) for o in offsets if 0.0 <= o <= clip_dur))

                        best_score = -1.0
                        best_t: float | None = None
                        best_roi: list[float] | None = None

                        for off in candidate_offsets:
                            cand_time = clip_start + float(off)
                            if _is_reverse_angle(cand_time):
                                continue
                            seek_sec = lead_pad + float(off)
                            cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000.0)
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                continue
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            fh, fw = gray.shape[:2]
                            lap = cv2.Laplacian(gray, cv2.CV_32F)
                            mag = np.abs(lap)
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
                            thresh = (mag > 25.0).astype(np.uint8) * 255
                            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            text_boxes = []
                            for cnt in contours:
                                bx, by, bw, bh = cv2.boundingRect(cnt)
                                # Filter out banner/hook title area at top of frame (y < fh * 0.08)
                                if by < fh * 0.08:
                                    continue
                                if bw >= 40 and bh >= 15 and (bw * bh) < (fw * fh * 0.85):
                                    density = float(np.mean(mag[by : by + bh, bx : bx + bw]))
                                    if density > 15.0:
                                        text_boxes.append((bx, by, bw, bh, density))

                            if text_boxes:
                                text_boxes.sort(key=lambda b: b[4], reverse=True)
                                best_box = text_boxes[0]
                                bx0, by0, bw0, bh0, d0 = best_box
                                cx0, cy0 = bx0 + bw0 / 2.0, by0 + bh0 / 2.0
                                cluster = [
                                    b for b in text_boxes
                                    if abs((b[0] + b[2] / 2.0) - cx0) < (fw * 0.35)
                                    and abs((b[1] + b[3] / 2.0) - cy0) < (fh * 0.35)
                                ]
                                pad_x = int(fw * 0.04)
                                pad_y = int(fh * 0.04)
                                min_x = max(0, min(b[0] for b in cluster) - pad_x)
                                min_y = max(0, min(b[1] for b in cluster) - pad_y)
                                max_x = min(fw, max(b[0] + b[2] for b in cluster) + pad_x)
                                max_y = min(fh, max(b[1] + b[3] for b in cluster) + pad_y)

                                cluster_score = float(np.mean([b[4] for b in cluster])) * math.sqrt(len(cluster))
                                if cluster_score > best_score:
                                    best_score = cluster_score
                                    best_t = cand_time
                                    best_roi = [
                                        round(min_y / fh, 4),
                                        round(min_x / fw, 4),
                                        round(max_y / fh, 4),
                                        round(max_x / fw, 4),
                                    ]

                        if best_t is not None and best_roi is not None and best_score > 20.0:
                            logger.info(
                                "Breakdown Tier 3 (OpenCV Saliency) resolved target_time=%.2fs, score=%.1f, roi=%s",
                                best_t,
                                best_score,
                                best_roi,
                            )
                            return best_t, best_roi
                    finally:
                        cap.release()
            except Exception as e:
                logger.debug("Breakdown Tier 3 scan failed: %s", e)

        # Tier 4: Setup-phase heuristic
        fallback_t = min(clip_end, clip_start + setup_offset)
        logger.info(
            "Breakdown Tier 4 (Setup Heuristic) resolved target_time=%.2fs, roi=%s",
            fallback_t,
            default_screen_roi,
        )
        return fallback_t, default_screen_roi

    def _assemble_highlight(
        self,
        highlight: Highlight,
        metadata: ClipMetadata,
        segment_paths: list[Path],
        clip_index: int,
        mode: str,
        voice_intro_path: Path | None,
        voice_outro_path: Path | None,
        *,
        valid_beats: list[dict[str, Any]] | None = None,
        buffer_seconds: float = 3.0,
        voice_breakdown_path: Path | None = None,
        gate_non_gameplay: bool = False,
        queued: bool = False,
        stream_info: StreamInfo | None = None,
        caption_preset: int | None = None,
        caption_style: str | None = None,
    ) -> ClipResult | None:
        """Run Phase 4 video assembly/export when installed, else persist raw segment + metadata.

        Returns None when the highlight is skipped (non-gameplay segment without a usable facecam).
        """
        if valid_beats is not None:
            beats = valid_beats
        elif mode == "story":
            beats = Scorer._validate_sub_clips(
                highlight.sub_clips,
                highlight.start,
                highlight.end,
                max_duration=float(self.config.get("analysis.max_clip_seconds", 90)),
            )
        else:
            beats = []

        try:
            from .assembly.composer import VideoComposer, remap_timeline_for_beats
            from .assembly.facecam import FacecamDetector
            from .assembly.hooks import HookTitleOverlay
            from .output.exporter import Exporter
            from .output.organizer import FileOrganizer
        except ModuleNotFoundError as exc:
            if exc.name not in _PHASE4_MODULES:
                raise
            logger.info(
                "Phase 4 assembly modules not yet available (%s); saving downloaded segment "
                "and metadata for clip_%02d (mode=%s).",
                exc.name,
                clip_index,
                mode,
            )
            return self._fallback_save_segment(
                highlight=highlight,
                metadata=metadata,
                segment_paths=segment_paths,
                clip_index=clip_index,
                valid_beats=beats,
            )

        facecam = FacecamDetector(self.config).detect(
            segment_paths, streamer=metadata.streamer
        )
        moment_type = str(metadata.moment_type or "").lower()
        layout_kind: str | None = None
        if gate_non_gameplay:
            if facecam is None or moment_type not in _REACTION_OK_MOMENT_TYPES:
                logger.warning(
                    "Skipping clip_%02d: non-gameplay segment without a confirmed facecam "
                    "or a reaction-capable moment type (%s).",
                    clip_index,
                    moment_type,
                )
                return None
            layout_kind = "reaction"
        elif moment_type == "reaction" and facecam is not None:
            layout_kind = "reaction"
        layout_extra: dict[str, Any] = (
            {"layout_kind": layout_kind} if layout_kind else {}
        )
        composer = VideoComposer(self.config)

        # --- E4/E5: replay settings, narration estimate and 59 s budget (before rendering) ---
        replay_style = str(
            self.config.get("transformation.breakdown_style", "instant_replay")
        ).lower()
        replay_speed = float(
            self.config.get(
                "transformation.breakdown.slow_mo",
                self.config.get("transformation.replay_speed", 0.5),
            )
        )
        replay_window = float(
            self.config.get(
                "transformation.breakdown.replay_window",
                self.config.get("transformation.replay_duration_seconds", 5.0),
            )
        )
        freeze_duration = float(
            self.config.get("transformation.breakdown.freeze_duration", 4.0)
        )
        freeze_zoom = float(self.config.get("transformation.breakdown.freeze_zoom", 1.35))

        # Resolve breakdown archetype and settings if mode is breakdown
        b_kind = str(
            getattr(metadata, "breakdown_kind", None)
            or getattr(highlight, "breakdown_kind", None)
            or "auto"
        ).strip().lower()
        if b_kind in {"", "auto"}:
            mtype = str(getattr(metadata, "moment_type", None) or getattr(highlight, "moment_type", "")).strip().lower()
            b_commentary = str(getattr(metadata, "breakdown_commentary", None) or getattr(highlight, "breakdown_commentary", "")).lower()
            spotlight_keywords = ("screen", "laptop", "file", "read", "document", "secret", "text")
            if mtype in {"analysis", "graphics_detail", "lore_secret"} or any(k in b_commentary for k in spotlight_keywords):
                b_kind = "zoom_spotlight"
            else:
                cfg_default = str(self.config.get("transformation.breakdown.default_kind", "auto") or "auto").strip().lower()
                b_kind = cfg_default if cfg_default in {"slow_mo", "zoom_spotlight"} else "slow_mo"

        zs_cfg = self.config.get("transformation.breakdown.zoom_spotlight", {}) or {}
        zs_zoom = float(
            getattr(highlight, "breakdown_zoom", None)
            or getattr(metadata, "breakdown_zoom", None)
            or zs_cfg.get("zoom_scale", 2.2)
        )
        zs_ease_in = float(zs_cfg.get("ease_in_seconds", 1.2))
        zs_ease_out = float(zs_cfg.get("ease_out_seconds", 1.0))
        zs_badge_text = str(zs_cfg.get("badge_text", "🔍 EVIDENCE BREAKDOWN"))
        zs_fullscreen = bool(zs_cfg.get("fullscreen_takeover", True))
        active_replay_style = "zoom_spotlight" if b_kind == "zoom_spotlight" else replay_style
        callout_text = ""
        if mode != "raw" and bool(self.config.get("transformation.callout_enabled", True)):
            callout_text = str(getattr(metadata, "callout", "") or "").strip()
        tts_speed = float(self.config.get("transformation.tts_speed", 1.0) or 1.0)
        narr_words = len(
            f"{metadata.narration_intro or ''} {metadata.narration_outro or ''}".split()
        )
        narration_seconds = (
            narr_words / (_WORDS_PER_SECOND * max(0.5, tts_speed)) if mode != "raw" else 0.0
        )
        callout_seconds = len(callout_text.split()) / _WORDS_PER_SECOND if callout_text else 0.0
        breakdown_text = str(
            getattr(metadata, "breakdown_commentary", None)
            or getattr(highlight, "breakdown_commentary", "")
            or ""
        ).strip()
        outro_text = str(
            getattr(metadata, "narration_outro", None)
            or getattr(highlight, "narration_outro", "")
            or ""
        ).strip()
        has_breakdown_voice = mode == "breakdown" and bool(
            breakdown_text or voice_breakdown_path is not None
        )
        breakdown_seconds_est = (
            self._estimate_breakdown_seconds(breakdown_text, voice_breakdown_path, tts_speed)
            if has_breakdown_voice
            else 0.0
        )
        outro_seconds_est = (
            self._estimate_breakdown_seconds(outro_text, voice_outro_path, tts_speed)
            if (mode != "raw" and (outro_text or voice_outro_path is not None))
            else 0.0
        )

        # Dynamic spotlight duration scaled to fit spoken voiceover without mid-sentence cutoffs
        needed_spotlight = 0.0
        if mode == "breakdown" and (breakdown_seconds_est > 0.0 or outro_seconds_est > 0.0):
            needed_spotlight = breakdown_seconds_est + outro_seconds_est + 1.5
        zs_duration = max(float(zs_cfg.get("duration", 5.7)), needed_spotlight)
        duck_full = (
            mode != "raw"
            and highlight.music_risk == "HIGH_MUSIC_DETECTED"
            and not bool(self.config.get("analysis.skip_high_music", False))
        )
        budget_plan: dict[str, float] = {}

        use_multi_beat = (
            mode == "story"
            and len(beats) >= 2
            and len(segment_paths) == len(beats)
        )

        composed = None
        try:
            if use_multi_beat:
                remapped_words, remapped_chat, peak_offset = remap_timeline_for_beats(
                    highlight.word_timestamps,
                    highlight.chat_burst,
                    highlight.peak_time,
                    beats,
                )
                beat_trim_ranges: list[tuple[float, float]] = []
                for beat in beats:
                    b_start = float(beat["start"])
                    b_end = float(beat["end"])
                    lead_pad = b_start - max(0.0, b_start - buffer_seconds)
                    beat_trim_ranges.append((lead_pad, lead_pad + (b_end - b_start)))

                beats_total = sum(b1 - b0 for b0, b1 in beat_trim_ranges)
                budget_plan = self._plan_budget(
                    moment_dur=beats_total,
                    mode=mode,
                    replay_style=active_replay_style,
                    replay_window=replay_window,
                    replay_speed=replay_speed,
                    freeze_duration=freeze_duration,
                    spotlight_duration=zs_duration,
                    narration_seconds=narration_seconds,
                    callout_seconds=callout_seconds,
                    breakdown_voiced=has_breakdown_voice,
                    breakdown_seconds=breakdown_seconds_est,
                    can_trim_moment=False,
                )
                composed = composer.compose_story_beats(
                    segment_paths,
                    facecam,
                    beat_trim_ranges=beat_trim_ranges,
                    peak_offset=peak_offset,
                    **layout_extra,
                )
                actual_durations = getattr(composed, "_beat_durations", None)
                if (
                    isinstance(actual_durations, list)
                    and len(actual_durations) == len(beats)
                    and any(
                        abs(float(a) - (float(b["end"]) - float(b["start"]))) > 0.05
                        for a, b in zip(actual_durations, beats)
                    )
                ):
                    logger.warning(
                        "Story beat durations differ from requested ranges; remapping "
                        "captions/chat to actual durations."
                    )
                    actual_beats = [
                        {**b, "end": float(b["start"]) + float(a)}
                        for a, b in zip(actual_durations, beats)
                    ]
                    remapped_words, remapped_chat, _ = remap_timeline_for_beats(
                        highlight.word_timestamps,
                        highlight.chat_burst,
                        highlight.peak_time,
                        actual_beats,
                    )
                clip_start_offset = 0.0
                words_for_export = remapped_words
                chat_for_export = remapped_chat
            else:
                actual_start = max(0.0, float(highlight.start) - buffer_seconds)
                has_peak = highlight.peak_time > 0.0 or (
                    highlight.peak_time == 0.0
                    and highlight.start == 0.0
                    and highlight.loudness_score > 0.0
                )
                peak_offset = (
                    max(0.0, highlight.peak_time - actual_start)
                    if (has_peak and mode != "raw")
                    else None
                )
                lead_pad = float(highlight.start) - actual_start
                clip_dur = max(0.0, float(highlight.end) - float(highlight.start))
                trim_range = (lead_pad, lead_pad + clip_dur)
                budget_plan = self._plan_budget(
                    moment_dur=clip_dur,
                    mode=mode,
                    replay_style=active_replay_style,
                    replay_window=replay_window,
                    replay_speed=replay_speed,
                    freeze_duration=freeze_duration,
                    spotlight_duration=zs_duration,
                    narration_seconds=narration_seconds,
                    callout_seconds=callout_seconds,
                    breakdown_voiced=has_breakdown_voice,
                    breakdown_seconds=breakdown_seconds_est,
                )
                new_dur = float(budget_plan["moment"])
                if clip_dur > 0 and new_dur < clip_dur - 0.05:
                    # Trim the moment around the peak (peak sits ~60% into the window).
                    peak_rel = (
                        highlight.peak_time - float(highlight.start)
                        if has_peak
                        else clip_dur / 2.0
                    )
                    off = max(0.0, min(peak_rel - 0.6 * new_dur, clip_dur - new_dur))
                    trim_range = (lead_pad + off, lead_pad + off + new_dur)
                composed = composer.compose(
                    segment_paths[0],
                    facecam,
                    peak_offset=peak_offset,
                    trim_range=trim_range,
                    **layout_extra,
                )
                # Map stream time onto the composed timeline using the trim compose()
                # ACTUALLY applied (0.0 if it skipped trimming), not the requested one.
                applied_trim = float(getattr(composed, "_trim_start", 0.0) or 0.0)
                clip_start_offset = actual_start + applied_trim
                words_for_export = highlight.word_timestamps
                chat_for_export = highlight.chat_burst

            replay_start_time = None
            moment_seconds = float(
                getattr(composed, "duration", 0.0) or budget_plan.get("moment", 0.0) or 0.0
            )
            if mode == "breakdown":
                breakdown_style = active_replay_style
                replay_window = float(budget_plan.get("replay_window", replay_window))
                if use_multi_beat:
                    replay_peak = peak_offset
                elif peak_offset is not None:
                    replay_peak = max(0.0, highlight.peak_time - clip_start_offset)
                else:
                    replay_peak = None
                if breakdown_style == "zoom_spotlight":
                    target_time, target_roi = self._resolve_breakdown_target_time_and_roi(
                        highlight,
                        segment_paths[0] if segment_paths else None,
                        highlight.start if not use_multi_beat else highlight.start,
                        highlight.end if not use_multi_beat else highlight.end,
                    )
                    metadata.breakdown_target_time = target_time
                    metadata.breakdown_roi = target_roi
                    target_offset = (
                        target_time - actual_start
                        if not use_multi_beat
                        else (target_time - highlight.start)
                    )
                    composed, replay_start_time = composer.compose_zoom_spotlight_breakdown(
                        composed,
                        source_video=segment_paths[0] if segment_paths else None,
                        target_time_offset=target_offset,
                        roi=target_roi,
                        duration=zs_duration,
                        zoom_scale=zs_zoom,
                        ease_in_seconds=zs_ease_in,
                        ease_out_seconds=zs_ease_out,
                        badge_text=zs_badge_text,
                        fullscreen_takeover=zs_fullscreen,
                    )
                elif breakdown_style == "freeze_frame":
                    composed = composer.compose_freeze_breakdown(
                        composed,
                        peak_offset=replay_peak,
                        freeze_duration=freeze_duration,
                        zoom=freeze_zoom,
                    )
                    freeze_start = getattr(composed, "_freeze_start", None)
                    freeze_dur = getattr(composed, "_freeze_duration", None)
                    if freeze_start is not None and freeze_dur:
                        # Breakdown voiceover plays over the frozen callout
                        replay_start_time = float(freeze_start)
                        words_for_export, chat_for_export = self._shift_timeline_after(
                            words_for_export,
                            chat_for_export,
                            clip_start_offset=clip_start_offset,
                            at=float(freeze_start),
                            delta=float(freeze_dur),
                        )
                else:
                    composed, replay_start_time = composer.compose_instant_replay(
                        composed,
                        peak_offset=replay_peak,
                        replay_window=replay_window,
                        slow_mo=replay_speed,
                    )

            layout = getattr(composed, "_layout", None)
            hook_kwargs: dict[str, Any] = {
                "context_badge": metadata.context_badge if mode != "raw" else "",
            }
            if layout is not None:
                hook_kwargs["layout"] = layout
                if getattr(layout, "kind", "") == "blur_fill" and metadata.streamer:
                    hook_kwargs["streamer"] = metadata.streamer
            composed = HookTitleOverlay(self.config).add_hook(
                composed, metadata.hook_title, **hook_kwargs
            )

            organizer = FileOrganizer(self.config)
            path_getter = (
                getattr(organizer, "get_queued_path", organizer.get_clip_path)
                if queued
                else organizer.get_clip_path
            )
            if queued:
                logger.info(
                    "Daily cap reached for %s; saving clip_%02d to the queued folder.",
                    metadata.streamer,
                    clip_index,
                )
            output_video_path = path_getter(
                game=metadata.game,
                streamer=metadata.streamer,
                index=organizer.next_available_index(
                    metadata.game, metadata.streamer, clip_index
                ),
            )

            exporter = Exporter(self.config)
            export_kwargs: dict[str, Any] = {
                "clip": composed,
                "words": words_for_export,
                "clip_start_offset": clip_start_offset,
                "output_path": output_video_path,
                "chat_burst": chat_for_export if mode != "raw" else None,
                "narration_intro": metadata.narration_intro if mode != "raw" else "",
                "narration_outro": metadata.narration_outro if mode != "raw" else "",
                "voice_intro_path": voice_intro_path,
                "voice_outro_path": voice_outro_path,
                "mode": mode,
            }
            if getattr(metadata, "narration_language", ""):
                export_kwargs["language"] = metadata.narration_language
            if layout is not None:
                export_kwargs["layout"] = layout
            if callout_text:
                export_kwargs["callout"] = callout_text
                export_kwargs["callout_start"] = 0.4 * moment_seconds if moment_seconds else None
            if duck_full:
                export_kwargs["duck_full"] = True
            from .assembly.captions import resolve_caption_style

            eff_caption_style = (
                caption_style
                or getattr(metadata, "caption_style", None)
                or getattr(highlight, "caption_style", None)
                or "auto"
            )
            resolved_style = resolve_caption_style(
                eff_caption_style,
                moment_type=getattr(metadata, "moment_type", None) or getattr(highlight, "moment_type", None),
                default_style=str(self.config.get("captions.style", "hormozi")),
            )
            metadata.caption_style = resolved_style
            export_kwargs["caption_style"] = resolved_style
            export_kwargs["moment_type"] = getattr(metadata, "moment_type", None) or getattr(highlight, "moment_type", None)

            style_preset, _preset_idx = self._resolve_caption_preset(
                metadata.streamer, caption_preset
            )
            if style_preset is not None:
                export_kwargs["style_preset"] = style_preset
            if mode == "breakdown":
                export_kwargs["breakdown_commentary"] = metadata.breakdown_commentary
                export_kwargs["breakdown_callout"] = metadata.breakdown_callout
                export_kwargs["replay_start"] = replay_start_time
                if voice_breakdown_path is not None:
                    export_kwargs["voice_breakdown_path"] = voice_breakdown_path

            final_video = exporter.export(**export_kwargs)
        except Exception:
            self._close_composed_handles(composed)
            raise

        max_final = float(self.config.get("output.max_final_seconds", 180))
        self._hard_trim(Path(final_video), max_final)

        result = organizer.save_clip(
            video_path=final_video,
            metadata=metadata,
            transcript_text=highlight.transcript_chunk,
        )
        info = dict(getattr(exporter, "last_export_info", None) or {})
        reason = self._narration_gate_reason(info, mode=mode)
        if reason:
            return self._quarantine_clip(result, reason)
        extra: dict[str, Any] = {}
        if budget_plan:
            extra.update(
                {
                    "commentary_ratio": budget_plan.get("coverage", 0.0),
                    "commentary_coverage_estimate": budget_plan.get("coverage", 0.0),
                    "planned_total_seconds": budget_plan.get("total", 0.0),
                    "layout_kind": layout_kind or "auto",
                }
            )
        extra.update(self._measured_commentary(info, result, mode))
        if extra:
            self._annotate_clip_json(result, extra)
        self._write_post_file_safe(result.video_path, metadata, stream_info)
        return result

    # ------------------------------------------------------------------
    # Narration gate, measured commentary, caption presets
    # ------------------------------------------------------------------

    def _ensure_intro_narration(
        self,
        metadata: ClipMetadata,
        highlight: Highlight,
        scorer: Any,
        streamer: str,
        game: str,
        *,
        hook_style: str | None = None,
        language: str = "english",
        has_custom_voice: bool = False,
    ) -> None:
        """Guarantee a non-raw clip has intro text: narrative writer first, grounded fallback second."""
        if str(metadata.narration_intro or "").strip() or has_custom_voice:
            return
        writer = getattr(scorer, "generate_narrative", None)
        if callable(writer):
            try:
                intro, outro, badge = writer(
                    highlight, streamer, game, hook_style=hook_style, language=language
                )
                metadata.narration_intro = str(intro or "").strip()
                metadata.narration_outro = metadata.narration_outro or str(outro or "").strip()
                metadata.context_badge = metadata.context_badge or str(badge or "").strip()
            except Exception as exc:
                logger.warning("Narrative writer unavailable for intro backfill: %s", exc)
        if not str(metadata.narration_intro or "").strip():
            from .scoring.ollama_scorer import grounded_fallback_intro

            metadata.narration_intro = grounded_fallback_intro(highlight, streamer, game)
            logger.warning("Using grounded fallback intro voiceover for %r.", highlight.title)

    def _narration_gate_reason(self, info: dict[str, Any], *, mode: str | None = None) -> str:
        """Non-empty reason when a non-raw clip was expected to be narrated but was not.

        Quarantines when no mix ran, when an expected intro track was not mixed, and (breakdown
        mode) when the breakdown voiceover was not expected or not mixed.
        """
        if not bool(self.config.get("output.require_narration", True)):
            return ""
        if info.get("narration_expected") and not info.get("mix_ran"):
            return (
                "narration mix did not run (TTS failed or produced no audio); "
                "clip would ship without voiceover"
            )
        if info.get("mix_ran") and info.get("intro_expected") and not float(
            info.get("intro_seconds") or 0.0
        ) > 0.0:
            return "intro voiceover was expected but not mixed; clip would open without narration"
        if mode == "breakdown":
            if not info.get("breakdown_expected"):
                return "breakdown mode without breakdown commentary/voice; replay would be silent"
            if not float(info.get("breakdown_seconds") or 0.0) > 0.0:
                return "breakdown voiceover was expected but not mixed"
        return ""

    def _quarantine_clip(self, result: ClipResult, reason: str) -> ClipResult:
        """Move the clip files into <output_dir>/<quarantine_dir>/<game>/<streamer>/<date>/."""
        qname = _sanitize_path_segment(
            str(self.config.get("output.quarantine_dir", "_quarantine") or "_quarantine"),
            "_quarantine",
        )
        meta = result.metadata
        dest = (
            self.output_dir
            / qname
            / _sanitize_path_segment(meta.game, "UnknownGame")
            / _sanitize_path_segment(meta.streamer, "UnknownStreamer")
            / date.today().isoformat()
        )
        dest.mkdir(parents=True, exist_ok=True)
        video = Path(result.video_path)
        stem = video.stem
        target_stem = stem
        n = 1
        while (dest / f"{target_stem}{video.suffix}").exists():
            n += 1
            target_stem = f"{stem}_{n}"

        def _move(src: Path, new_name: str) -> Path | None:
            if not src.is_file():
                return None
            dst = dest / new_name
            shutil.move(str(src), str(dst))
            return dst

        new_video = _move(video, f"{target_stem}{video.suffix}") or video
        new_meta: Path | None = result.metadata_path
        if result.metadata_path:
            new_meta = _move(Path(result.metadata_path), f"{target_stem}.json") or result.metadata_path
        _move(video.with_name(f"{stem}_transcript.txt"), f"{target_stem}_transcript.txt")
        _move(video.with_name(f"{stem}_post.txt"), f"{target_stem}_post.txt")
        quarantined = ClipResult(
            video_path=new_video,
            metadata_path=new_meta,
            metadata=result.metadata,
            duration_seconds=result.duration_seconds,
            quarantined=True,
            quarantine_reason=reason,
        )
        self._annotate_clip_json(quarantined, {"quarantined": True, "quarantine_reason": reason})
        logger.warning("Quarantined %s: %s", new_video, reason)
        return quarantined

    def _measured_commentary(
        self, info: dict[str, Any], result: ClipResult, mode: str
    ) -> dict[str, Any]:
        """Measured (post-export) commentary keys for the clip JSON; empty when nothing was measured."""
        if not info or "spoken_seconds" not in info:
            return {}
        spoken = float(info.get("spoken_seconds") or 0.0)
        final = float(info.get("final_duration") or 0.0) or float(result.duration_seconds or 0.0)
        coverage = round(spoken / final, 3) if final > 0 else 0.0
        out: dict[str, Any] = {
            "commentary_seconds": round(spoken, 3),
            "commentary_coverage": coverage,
        }
        min_ratio = float(self.config.get("output.min_commentary_ratio", 0.35))
        if mode != "raw" and min_ratio > 0 and coverage < min_ratio:
            logger.warning(
                "Low commentary on %s: measured coverage %.3f < output.min_commentary_ratio %.2f.",
                result.video_path,
                coverage,
                min_ratio,
            )
            out["low_commentary"] = True
        return out

    @staticmethod
    def _estimate_breakdown_seconds(
        text: str, voice_path: Path | None, tts_speed: float
    ) -> float:
        """Estimated spoken length of the breakdown commentary (custom WAV duration wins)."""
        if voice_path is not None:
            try:
                import soundfile as sf

                return float(sf.info(str(voice_path)).duration)
            except Exception:
                pass
        return len(text.split()) / (_WORDS_PER_SECOND * max(0.5, tts_speed))

    def _caption_presets(self) -> list[dict[str, Any]]:
        if not bool(self.config.get("captions.rotate_styles", True)):
            return []
        raw = self.config.get("captions.style_presets", None) or []
        return [dict(p) for p in raw if isinstance(p, dict)]

    def _variety_state(self) -> VarietyState:
        return VarietyState(
            Path(self.config.get("general.cache_dir", "./.cache")) / "variety_state.json",
            hook_styles=list(self.config.get("transformation.hook_styles", None) or []) or None,
        )

    def _next_caption_preset_index(self, variety: VarietyState, streamer: str) -> int | None:
        presets = self._caption_presets()
        if not presets:
            return None
        try:
            return variety.next_caption_preset(len(presets), streamer=streamer)
        except Exception as exc:
            logger.debug("Caption preset rotation failed: %s", exc)
            return None

    def _resolve_caption_preset(
        self, streamer: str, preset_index: int | None
    ) -> tuple[dict[str, Any] | None, int | None]:
        """Preset dict for the clip. When the caller did not rotate (MCP path) pick + persist one."""
        presets = self._caption_presets()
        if not presets:
            return None, None
        try:
            idx = preset_index
            if idx is None:
                variety = self._variety_state()
                idx = variety.next_caption_preset(len(presets), streamer=streamer)
                variety.record_caption_preset(streamer, int(idx))
            idx = int(idx) % len(presets)
            return presets[idx], idx
        except Exception as exc:
            logger.debug("Caption preset resolution failed: %s", exc)
            return None, None

    _CHAT_TIME_KEYS = ("offset_seconds", "timestamp", "time", "offset")

    @classmethod
    def _shift_timeline_after(
        cls,
        words: list[Any],
        chat: list[dict[str, Any]] | None,
        *,
        clip_start_offset: float,
        at: float,
        delta: float,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Push words/chat at or after clip-relative time ``at`` later by ``delta`` seconds.

        Used after a freeze-frame insertion so captions and chat ticker stay in sync with
        the video that resumes ``delta`` seconds later. Inputs are not mutated.
        """
        shifted_words: list[Any] = []
        for w in words or []:
            if isinstance(w, dict):
                start = float(w.get("start", 0.0))
                if start - clip_start_offset >= at:
                    w = {**w, "start": start + delta, "end": float(w.get("end", start)) + delta}
            elif float(w.start) - clip_start_offset >= at:
                w = WordTimestamp(
                    text=w.text,
                    start=float(w.start) + delta,
                    end=float(w.end) + delta,
                    confidence=getattr(w, "confidence", 0.0),
                )
            shifted_words.append(w)

        shifted_chat: list[dict[str, Any]] = []
        for msg in chat or []:
            if isinstance(msg, dict):
                key = next((k for k in cls._CHAT_TIME_KEYS if msg.get(k) is not None), None)
                if key is not None:
                    try:
                        t = float(msg[key])
                    except (TypeError, ValueError):
                        t = None
                    if t is not None and t - clip_start_offset >= at:
                        msg = {**msg, key: t + delta}
            shifted_chat.append(msg)
        return shifted_words, shifted_chat

    @staticmethod
    def _close_composed_handles(clip: Any) -> None:
        """Best-effort handle cleanup if an error occurs before Exporter finishes."""
        if clip is None:
            return
        seen: set[int] = set()
        stack: list[Any] = [clip]
        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            src = getattr(current, "_source_clip", None)
            if src is not None:
                stack.append(src)
            srcs = getattr(current, "_source_clips", None)
            if srcs:
                try:
                    stack.extend(srcs)
                except TypeError:
                    pass
            close_fn = getattr(current, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:
                    logger.debug("Ignoring error closing composed clip handle: %s", exc)

    def _fallback_save_segment(
        self,
        highlight: Highlight,
        metadata: ClipMetadata,
        segment_paths: list[Path],
        clip_index: int,
        valid_beats: list[dict[str, Any]] | None = None,
    ) -> ClipResult:
        """Phase 3 fallback before Phase 4 assembly exists: persist metadata JSON & transcript."""
        game_dir = _sanitize_path_segment(metadata.game, "UnknownGame")
        streamer_dir = _sanitize_path_segment(metadata.streamer, "UnknownStreamer")
        date_str = date.today().isoformat()

        structure_template = str(
            self.config.get("output.structure", "{game}/{streamer}/{date}")
        )
        try:
            rel_structure = structure_template.format(
                game=game_dir,
                streamer=streamer_dir,
                date=date_str,
            )
        except (KeyError, ValueError, IndexError):
            rel_structure = f"{game_dir}/{streamer_dir}/{date_str}"

        safe_parts = [
            _sanitize_path_segment(part)
            for part in re.split(r"[/\\]+", rel_structure)
            if part and part not in (".", "..")
        ]
        clip_dir = self.output_dir.joinpath(*safe_parts) if safe_parts else self.output_dir
        clip_dir.mkdir(parents=True, exist_ok=True)

        stem_template = str(self.config.get("output.filename_template", "clip_{index:02d}"))
        try:
            raw_stem = stem_template.format(index=clip_index)
        except (KeyError, ValueError, IndexError):
            raw_stem = f"clip_{clip_index:02d}"
        stem = _sanitize_path_segment(raw_stem, f"clip_{clip_index:02d}")

        metadata_path: Path | None = None
        if self.config.get("output.save_metadata_json", True):
            metadata_path = clip_dir / f"{stem}.json"
            metadata_path.write_text(metadata.to_json(), encoding="utf-8")

        if self.config.get("output.save_transcript_txt", True) and highlight.transcript_chunk:
            transcript_path = clip_dir / f"{stem}_transcript.txt"
            transcript_path.write_text(highlight.transcript_chunk, encoding="utf-8")

        beats = valid_beats or []
        if len(beats) >= 2 and metadata.transformation_mode == "story":
            duration = sum(
                max(0.0, float(b["end"]) - float(b["start"])) for b in beats
            )
        else:
            duration = max(0.0, float(highlight.end) - float(highlight.start))

        return ClipResult(
            video_path=Path(segment_paths[0]),
            metadata_path=metadata_path,
            metadata=metadata,
            duration_seconds=round(duration, 3),
        )

    # ------------------------------------------------------------------
    # Interactive highlight review
    # ------------------------------------------------------------------

    def _interactive_select(self, highlights: list[Highlight]) -> list[Highlight]:
        """Present highlights to user in CLI and let them pick which to render."""
        if not highlights:
            self.console.print("[yellow]No highlights found above threshold.[/]")
            return []

        table = Table(title="🎯 Detected Highlights")
        table.add_column("#", style="bold")
        table.add_column("Time", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("Beats", style="blue")
        table.add_column("Hook Title", style="yellow")
        table.add_column("Narration Intro", style="dim")

        for i, h in enumerate(highlights, 1):
            beat_label = f"{len(h.sub_clips)} cuts" if len(h.sub_clips) >= 2 else "single"
            raw_intro = h.narration_intro or h.reason
            intro_preview = f"{raw_intro[:50]}..." if len(raw_intro) > 50 else raw_intro
            music_badge = " [bold red]⚠️ MUSIC_DETECTED[/]" if h.music_risk == "HIGH_MUSIC_DETECTED" else ""
            title_display = f"{h.title}{music_badge}"
            table.add_row(
                str(i),
                f"{format_time(h.start)} -> {format_time(h.end)}",
                f"{h.final_score:.1f}/10",
                h.moment_type,
                beat_label,
                title_display,
                intro_preview,
            )

        self.console.print(table)
        self.console.print(
            "\n[bold]Enter clip numbers to render (e.g., '1,2,3' or 'all'):[/]"
        )

        selection = input("> ").strip()
        if selection.lower() == "all":
            return list(highlights)

        selected: list[Highlight] = []
        seen: set[int] = set()
        for token in selection.split(","):
            cleaned = token.strip()
            if not cleaned.isdigit():
                continue
            idx = int(cleaned) - 1
            if 0 <= idx < len(highlights) and idx not in seen:
                seen.add(idx)
                selected.append(highlights[idx])
        return selected
