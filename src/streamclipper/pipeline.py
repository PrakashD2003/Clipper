"""End-to-end async pipeline orchestrator with IntEnum state caching and crash recovery."""
from __future__ import annotations

import asyncio
from datetime import date
import inspect
import logging
from pathlib import Path
import re
from typing import Any

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


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = max(0, int(seconds))
    hrs = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _sanitize_path_segment(name: str, fallback: str = "unknown") -> str:
    """Sanitize a folder segment so it is valid on Windows and POSIX filesystems."""
    cleaned = _INVALID_PATH_CHARS_RE.sub("_", str(name or "")).strip(" .")
    return cleaned or fallback


class Pipeline:
    """Central orchestrator wiring discovery, download, analysis, scoring, and assembly."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cache = StepCache(config)
        self.console = Console()
        self.temp_dir = Path(config.get("general.temp_dir", "./temp"))
        self.output_dir = Path(config.get("general.output_dir", "./output"))
        self._gpu_lock = asyncio.Lock()

    @staticmethod
    def _recover_failed_state(state: PipelineState) -> PipelineState:
        """Reset a FAILED (99) state back to the last safely completed stage for retry."""
        if state.status != PipelineStatus.FAILED:
            return state
        if state.highlights or state.selected_highlights:
            state.status = PipelineStatus.ASSEMBLING
        elif state.analysis_complete or state.loudness_spikes or state.chat_spikes:
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

    def _resolve_highlight_mode(
        self, base_mode: str, highlight: Highlight
    ) -> tuple[str, list[dict[str, Any]]]:
        """Resolve concrete per-highlight mode (raw | narrative | story) and validated beats."""
        if base_mode == "raw":
            return "raw", []
        if base_mode == "narrative":
            return "narrative", []

        max_clip_seconds = float(self.config.get("analysis.max_clip_seconds", 90))
        valid_beats = Scorer._validate_sub_clips(
            highlight.sub_clips,
            highlight.start,
            highlight.end,
            max_duration=max_clip_seconds,
        )
        if base_mode == "auto":
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
    ) -> list[ClipResult]:
        """Process a single stream through the full resumable pipeline."""
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Invalid URL: stream URL must be a non-empty string.")
        clean_url = url.strip()

        # Fail fast on invalid mode before clearing cache or running expensive pipeline stages
        self._resolve_base_mode(mode)

        intro_path = Path(voice_intro_path) if voice_intro_path is not None else None
        outro_path = Path(voice_outro_path) if voice_outro_path is not None else None
        if intro_path is not None and not intro_path.exists():
            logger.warning("Custom intro voiceover path does not exist yet: %s", intro_path)
        if outro_path is not None and not outro_path.exists():
            logger.warning("Custom outro voiceover path does not exist yet: %s", outro_path)

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
                async with self._gpu_lock:
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
                async with self._gpu_lock:
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
        state.status = PipelineStatus.ANALYZING
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
        video_id = state.stream_info.video_id or self.cache.get_cache_key(
            state.stream_info.url
        )
        chat_cache_path = self.temp_dir / f"{video_id}_chat.csv"

        def _fetch_chat() -> Any:
            cached = scraper.load_cache(chat_cache_path)
            if cached is not None:
                return cached
            df = scraper.scrape(state.stream_info.url, self.temp_dir)
            if not df.empty:
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
        )
        state.status = PipelineStatus.ASSEMBLING
        return state

    async def _step_assemble(
        self,
        state: PipelineState,
        mode: str | None = None,
        voice_intro_path: Path | None = None,
        voice_outro_path: Path | None = None,
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

        clips: list[ClipResult] = []
        buffer_seconds = 3.0
        for idx, highlight in enumerate(state.selected_highlights, start=1):
            clip_mode, valid_beats = self._resolve_highlight_mode(base_mode, highlight)
            use_multi_beat = clip_mode == "story" and len(valid_beats) >= 2
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
                )
            else:
                single_path = await asyncio.to_thread(
                    download_segment,
                    state.stream_info.url,
                    highlight.start,
                    highlight.end,
                    segments_dir,
                    buffer_seconds,
                )
                segment_paths = [single_path]

            async with self._gpu_lock:
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
                )
            clips.append(clip_result)

        state.clips = clips
        state.status = PipelineStatus.COMPLETE
        return state

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
            composed = composer.compose(
                segment_paths[0],
                facecam,
                peak_offset=peak_offset,
            )
            clip_start_offset = actual_start
            words_for_export = highlight.word_timestamps
            chat_for_export = highlight.chat_burst

        composed = HookTitleOverlay(self.config).add_hook(
            composed,
            metadata.hook_title,
            context_badge=metadata.context_badge if mode != "raw" else "",
        )

        organizer = FileOrganizer(self.config)
        output_video_path = organizer.get_clip_path(
            game=metadata.game,
            streamer=metadata.streamer,
            index=clip_index,
        )

        exporter = Exporter(self.config)
        final_video = exporter.export(
            clip=composed,
            words=words_for_export,
            clip_start_offset=clip_start_offset,
            output_path=output_video_path,
            chat_burst=chat_for_export if mode != "raw" else None,
            narration_intro=metadata.narration_intro if mode != "raw" else "",
            narration_outro=metadata.narration_outro if mode != "raw" else "",
            voice_intro_path=voice_intro_path,
            voice_outro_path=voice_outro_path,
            mode=mode,
        )

        return organizer.save_clip(
            video_path=final_video,
            metadata=metadata,
            transcript_text=highlight.transcript_chunk,
        )

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
            table.add_row(
                str(i),
                f"{format_time(h.start)} -> {format_time(h.end)}",
                f"{h.final_score:.1f}/10",
                h.moment_type,
                beat_label,
                h.title,
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
