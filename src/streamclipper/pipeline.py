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
from typing import Any, AsyncIterator

from rich.console import Console
from rich.table import Table

from .analysis.chat_velocity import ChatVelocityAnalyzer
from .analysis.loudness import LoudnessDetector
from .analysis.transcriber import Transcriber
from .cache import StepCache
from .config import Config, VALID_TRANSFORMATION_MODES
from .download.audio import download_audio
from .download.chat import ChatScraper
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
from .scoring.scorer import Scorer

logger = logging.getLogger(__name__)

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

    def _acquire_file_lock(self) -> Any:
        """Blocking exclusive lock on <cache_dir>/gpu.lock (run via asyncio.to_thread)."""
        import time

        lock_dir = Path(self.config.get("general.cache_dir", "./.cache"))
        lock_dir.mkdir(parents=True, exist_ok=True)
        fh = open(lock_dir / "gpu.lock", "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0, os.SEEK_END)
                if fh.tell() == 0:
                    fh.write(b"0")
                    fh.flush()
                while True:
                    fh.seek(0)
                    try:
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.25)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except BaseException:
            fh.close()
            raise
        return fh

    @staticmethod
    def _release_file_lock(fh: Any) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            logger.debug("Ignoring error releasing GPU file lock: %s", exc)
        finally:
            fh.close()

    @contextlib.asynccontextmanager
    async def _gpu_guard(self) -> AsyncIterator[None]:
        """Hold the in-process GPU lock (and the optional cross-process file lock)."""
        async with self._gpu_lock:
            if not bool(self.config.get("general.cross_process_gpu_lock", False)):
                yield
                return
            fh = await asyncio.to_thread(self._acquire_file_lock)
            try:
                yield
            finally:
                await asyncio.to_thread(self._release_file_lock, fh)

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

    _AUTO_BREAKDOWN_MOMENT_TYPES = frozenset({"analysis", "graphics_detail", "lore_secret"})

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
        if base_mode == "breakdown":
            return "breakdown", []

        max_clip_seconds = float(self.config.get("analysis.max_clip_seconds", 90))
        valid_beats = Scorer._validate_sub_clips(
            highlight.sub_clips,
            highlight.start,
            highlight.end,
            max_duration=max_clip_seconds,
        )
        if base_mode == "auto":
            # Mode is resolved BEFORE Pass 2 (generate_metadata) writes narration, so only
            # signals present at this point may drive the choice:
            #   1. explicit breakdown commentary (Pass 1 / MCP host) -> breakdown
            #   2. validated multi-beat sub_clips -> story (never discard valid beats)
            #   3. detail/analysis moment types -> breakdown (funny `fail` keeps witty narrative)
            #   4. otherwise -> narrative
            if highlight.breakdown_commentary:
                return "breakdown", []
            if len(valid_beats) >= 2:
                return "story", valid_beats
            if (
                highlight.moment_type in self._AUTO_BREAKDOWN_MOMENT_TYPES
                and not (highlight.narration_intro and highlight.narration_outro)
            ):
                return "breakdown", []
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

        # Commit both signal lists atomically so partial failure never skips chat analysis on retry
        state.loudness_spikes = loudness_spikes
        state.chat_spikes = chat_spikes
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
        supports_mode_kwarg = False
        try:
            meta_sig = inspect.signature(scorer.generate_metadata)
            supports_mode_kwarg = "mode" in meta_sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in meta_sig.parameters.values()
            )
        except (TypeError, ValueError):
            supports_mode_kwarg = False

        segments_dir = self.temp_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        streamer_name = state.stream_info.streamer or "Unknown"
        game_name = state.stream_info.game or str(
            self.config.get("discovery.default_game", "GTA 6")
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

            async with self._gpu_guard():
                if supports_mode_kwarg:
                    metadata = await asyncio.to_thread(
                        scorer.generate_metadata,
                        highlight,
                        streamer_name,
                        game_name,
                        state.stream_info.url,
                        mode=clip_mode,
                    )
                else:
                    metadata = await asyncio.to_thread(
                        scorer.generate_metadata,
                        highlight,
                        streamer_name,
                        game_name,
                        state.stream_info.url,
                    )
                metadata.transformation_mode = clip_mode
                if clip_mode == "raw":
                    metadata.narration_intro = ""
                    metadata.narration_outro = ""
                    metadata.context_badge = ""

            assemble_guard = (
                self._gpu_guard()
                if self._narration_may_use_gpu(clip_mode)
                else _AsyncNull()
            )
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
                )
            state.clips.append(clip_result)
            self.cache.save_state(state)

        state.status = PipelineStatus.COMPLETE
        return state

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
    ) -> ClipResult:
        """Run Phase 4 video assembly/export when installed, else persist raw segment + metadata."""
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

        facecam = FacecamDetector(self.config).detect(segment_paths[0])
        composer = VideoComposer(self.config)

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

                composed = composer.compose_story_beats(
                    segment_paths,
                    facecam,
                    beat_trim_ranges=beat_trim_ranges,
                    peak_offset=peak_offset,
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
                composed = composer.compose(
                    segment_paths[0],
                    facecam,
                    peak_offset=peak_offset,
                    trim_range=trim_range,
                )
                # Map stream time onto the composed timeline using the trim compose()
                # ACTUALLY applied (0.0 if it skipped trimming), not the requested one.
                applied_trim = float(getattr(composed, "_trim_start", 0.0) or 0.0)
                clip_start_offset = actual_start + applied_trim
                words_for_export = highlight.word_timestamps
                chat_for_export = highlight.chat_burst

            replay_start_time = None
            if mode == "breakdown":
                breakdown_style = str(
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
                freeze_zoom = float(
                    self.config.get("transformation.breakdown.freeze_zoom", 1.35)
                )
                if use_multi_beat:
                    replay_peak = peak_offset
                elif peak_offset is not None:
                    replay_peak = max(0.0, highlight.peak_time - clip_start_offset)
                else:
                    replay_peak = None
                if breakdown_style == "freeze_frame":
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

            composed = HookTitleOverlay(self.config).add_hook(
                composed,
                metadata.hook_title,
                context_badge=metadata.context_badge if mode != "raw" else "",
            )

            organizer = FileOrganizer(self.config)
            output_video_path = organizer.get_clip_path(
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

        return organizer.save_clip(
            video_path=final_video,
            metadata=metadata,
            transcript_text=highlight.transcript_chunk,
        )

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
