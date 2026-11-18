"""Abstract Scorer interface with signal fusion, transcript chunking, and YPP metadata validation."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..models import TranscriptionResult, TimeRange, Highlight, WordTimestamp
from ..config import Config


@dataclass
class TranscriptChunk:
    """A contiguous window of transcript text with its time boundaries."""
    start: float
    end: float
    text: str


class Scorer(ABC):
    """Abstract base for highlight scoring.

    Implements shared infrastructure — transcript chunking, signal fusion,
    deduplication, and YPP metadata attachment — leaving only the LLM-specific
    ``_score_chunk`` to concrete subclasses (``OllamaScorer``, MCP server, etc.).
    """

    def __init__(self, config: Config):
        self.config = config
        self.chunk_minutes = config.get("analysis.chunk_minutes", 12)
        self.min_score = config.get("analysis.min_highlight_score", 7.0)
        self.max_clips = config.get("analysis.max_clips_per_stream", 5)
        self.min_clip_seconds = config.get("analysis.min_clip_seconds", 30)
        self.max_clip_seconds = config.get("analysis.max_clip_seconds", 90)
        self.weights = config.get(
            "analysis.weights",
            {"loudness": 0.25, "chat_velocity": 0.25, "llm_score": 0.50},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_candidate_chunks(
        self,
        transcript: TranscriptionResult,
        loudness_spikes: list[TimeRange],
        chat_spikes: list[TimeRange],
        chunk_minutes: int | None = None,
        only_with_signals: bool = True,
    ) -> list[tuple[TranscriptChunk, list[TimeRange], list[TimeRange]]]:
        """Chunk transcript and pair each chunk with overlapping loudness/chat spikes.

        Shared by ``Scorer.score_and_select()`` and the MCP server's
        ``get_transcript_chunks`` tool.
        """
        chunks = self._chunk_transcript(transcript, chunk_minutes=chunk_minutes)
        candidates: list[tuple[TranscriptChunk, list[TimeRange], list[TimeRange]]] = []
        for chunk in chunks:
            chunk_loudness = self._find_overlapping(loudness_spikes, chunk)
            chunk_chat = self._find_overlapping(chat_spikes, chunk)
            if only_with_signals and not chunk_loudness and not chunk_chat:
                continue
            candidates.append((chunk, chunk_loudness, chunk_chat))
        return candidates

    def score_and_select(
        self,
        transcript: TranscriptionResult,
        loudness_spikes: list[TimeRange],
        chat_spikes: list[TimeRange],
    ) -> list[Highlight]:
        """Full scoring pipeline: chunk -> score -> fuse -> dedupe -> rank -> select."""
        chunks = self._chunk_transcript(transcript)

        raw_highlights: list[Highlight] = []
        for chunk in chunks:
            chunk_loudness = self._find_overlapping(loudness_spikes, chunk)
            chunk_chat = self._find_overlapping(chat_spikes, chunk)
            highlights = self._score_chunk(chunk, chunk_loudness, chunk_chat)
            raw_highlights.extend(highlights)

        for h in raw_highlights:
            h.final_score = self._compute_weighted_score(h, loudness_spikes, chat_spikes)

        deduped = self._deduplicate(raw_highlights)

        selected = sorted(
            [h for h in deduped if h.final_score >= self.min_score],
            key=lambda h: h.final_score,
            reverse=True,
        )[: self.max_clips]

        for h in selected:
            h.word_timestamps = self._extract_words_for_range(transcript, h.start, h.end)
            h.peak_time = self._extract_peak_time(h, loudness_spikes)
            h.chat_burst = self._extract_chat_burst(h, chat_spikes)
            h.sub_clips = self._validate_sub_clips(h.sub_clips, h.start, h.end)

        return selected

    # ------------------------------------------------------------------
    # Abstract method — subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _score_chunk(
        self,
        chunk: TranscriptChunk,
        loudness: list[TimeRange],
        chat: list[TimeRange],
    ) -> list[Highlight]:
        """LLM-specific implementation. Subclass for Ollama, MCP, etc."""
        ...

    # ------------------------------------------------------------------
    # Transcript chunking
    # ------------------------------------------------------------------

    def _chunk_transcript(
        self,
        transcript: TranscriptionResult,
        chunk_minutes: int | None = None,
        overlap_seconds: float = 60.0,
    ) -> list[TranscriptChunk]:
        """Split transcript into *chunk_minutes*-minute windows with *overlap_seconds* overlap.

        Each chunk collects segment text whose time range falls within or
        overlaps the chunk window.  Overlap ensures moments near chunk
        boundaries are scored by at least one chunk.
        """
        if not transcript.segments:
            return []

        minutes = chunk_minutes if chunk_minutes is not None else self.chunk_minutes
        chunk_duration = minutes * 60.0  # seconds

        total_start = transcript.segments[0].start
        total_end = transcript.segments[-1].end

        # Step size = chunk_duration - overlap_seconds
        step = max(chunk_duration - overlap_seconds, 1.0)

        chunks: list[TranscriptChunk] = []
        window_start = total_start
        while window_start < total_end:
            window_end = min(window_start + chunk_duration, total_end)
            # Collect segments that overlap this window
            texts: list[str] = []
            for seg in transcript.segments:
                if seg.end > window_start and seg.start < window_end:
                    texts.append(seg.text)
            if texts:
                chunks.append(TranscriptChunk(
                    start=window_start,
                    end=window_end,
                    text=" ".join(texts),
                ))
            window_start += step
        return chunks

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_overlapping(
        spikes: list[TimeRange], chunk: TranscriptChunk
    ) -> list[TimeRange]:
        """Return spikes whose time range overlaps the chunk window."""
        return [s for s in spikes if s.end > chunk.start and s.start < chunk.end]

    @staticmethod
    def _signal_overlap_score(
        highlight: Highlight, spikes: list[TimeRange]
    ) -> float:
        """Compute signal support score (0.0–10.0) based on temporal overlap.

        For each spike that overlaps the highlight, compute the fraction of the
        spike covered by the highlight window, weight it by the spike's own
        score, and take the maximum.  Clamped to ``[0.0, 10.0]``.
        """
        if not spikes:
            return 0.0

        h_start, h_end = highlight.start, highlight.end
        best = 0.0
        for spike in spikes:
            # Check overlap
            overlap_start = max(h_start, spike.start)
            overlap_end = min(h_end, spike.end)
            if overlap_start >= overlap_end:
                continue
            spike_duration = spike.end - spike.start
            if spike_duration <= 0:
                continue
            overlap_fraction = (overlap_end - overlap_start) / spike_duration
            score = overlap_fraction * spike.score
            if score > best:
                best = score
        return min(best, 10.0)

    def _compute_weighted_score(
        self,
        highlight: Highlight,
        loudness_spikes: list[TimeRange],
        chat_spikes: list[TimeRange],
    ) -> float:
        """Fuse LLM score with signal scores using configured weights.

        Formula::

            final = w_llm × highlight.score
                  + w_loudness × overlap_score(highlight, loudness_spikes)
                  + w_chat × overlap_score(highlight, chat_spikes)
        """
        loudness_overlap = self._signal_overlap_score(highlight, loudness_spikes)
        chat_overlap = self._signal_overlap_score(highlight, chat_spikes)
        highlight.loudness_score = loudness_overlap
        highlight.chat_score = chat_overlap

        return (
            self.weights["llm_score"] * highlight.score
            + self.weights["loudness"] * loudness_overlap
            + self.weights["chat_velocity"] * chat_overlap
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(
        highlights: list[Highlight], iou_threshold: float = 0.3
    ) -> list[Highlight]:
        """Remove overlapping highlights, keeping the higher ``final_score``.

        Uses Intersection-over-Union (IoU) on time intervals::

            IoU = overlap_duration / union_duration

        When IoU ≥ *iou_threshold*, the lower-scoring highlight is discarded.
        """
        if not highlights:
            return []

        # Sort descending by final_score so higher-scored highlights are kept
        sorted_hl = sorted(highlights, key=lambda h: h.final_score, reverse=True)
        kept: list[Highlight] = []

        for candidate in sorted_hl:
            is_duplicate = False
            for keeper in kept:
                # IoU calculation on 1-D intervals
                overlap_start = max(candidate.start, keeper.start)
                overlap_end = min(candidate.end, keeper.end)
                overlap = max(0.0, overlap_end - overlap_start)

                union = (
                    (candidate.end - candidate.start)
                    + (keeper.end - keeper.start)
                    - overlap
                )
                if union <= 0:
                    continue
                iou = overlap / union
                if iou >= iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(candidate)

        return kept

    # ------------------------------------------------------------------
    # Word timestamp extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_words_for_range(
        transcript: TranscriptionResult, start: float, end: float
    ) -> list[WordTimestamp]:
        """Extract word-level timestamps that fall within ``[start, end]``."""
        words: list[WordTimestamp] = []
        for seg in transcript.segments:
            if seg.end <= start or seg.start >= end:
                continue
            for w in seg.words:
                if w.start >= start and w.end <= end:
                    words.append(w)
        return words

    # ------------------------------------------------------------------
    # YPP metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_peak_time(
        highlight: Highlight, loudness_spikes: list[TimeRange]
    ) -> float:
        """Return the exact peak second from overlapping loudness spikes.

        Uses ``spike.metadata["peak_time"]`` when available, selecting the
        spike with the highest score.  Falls back to the highlight midpoint.
        """
        h_start, h_end = highlight.start, highlight.end
        best_score = -1.0
        best_peak: float | None = None

        for spike in loudness_spikes:
            # Check overlap
            if spike.end <= h_start or spike.start >= h_end:
                continue
            peak_candidate = spike.metadata.get("peak_time")
            if peak_candidate is not None and spike.score > best_score:
                best_score = spike.score
                best_peak = float(peak_candidate)

        if best_peak is not None:
            return best_peak
        # Fallback to midpoint
        return (h_start + h_end) / 2.0

    @staticmethod
    def _extract_chat_burst(
        highlight: Highlight, chat_spikes: list[TimeRange]
    ) -> list[dict[str, Any]]:
        """Return up to 5 representative chat messages from overlapping chat spikes."""
        h_start, h_end = highlight.start, highlight.end
        messages: list[dict[str, Any]] = []

        for spike in chat_spikes:
            if spike.end <= h_start or spike.start >= h_end:
                continue
            sample = spike.metadata.get("sample_messages", [])
            messages.extend(sample)

        # Deduplicate by message text and return at most 5
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for msg in messages:
            text = str(msg.get("message", msg.get("text", "")))
            if text not in seen:
                seen.add(text)
                unique.append(msg)
        return unique[:5]

    @staticmethod
    def _validate_sub_clips(
        sub_clips: list[dict[str, Any]], start: float, end: float
    ) -> list[dict[str, Any]]:
        """Validate multi-beat sub_clips; return ``[]`` on any violation.

        Rules:
        - At least 2 beats (``len(sub_clips) >= 2``).
        - Each beat: ``start <= beat["start"] < beat["end"] <= end``.
        - Each beat duration >= 3.0 seconds.
        - Beats are strictly ordered and non-overlapping
          (``prev_end <= next_start``).
        - Total stitched duration >= 10.0 seconds (or ``min_clip_seconds``
          when available, with a floor of 10.0).

        Any violation clears to ``[]`` for safe single-cut fallback.
        """
        if not sub_clips or len(sub_clips) < 2:
            return []

        total_duration = 0.0
        prev_end: float | None = None

        for beat in sub_clips:
            try:
                b_start = float(beat["start"])
                b_end = float(beat["end"])
            except (KeyError, TypeError, ValueError):
                return []

            # Boundary check
            if b_start < start or b_end > end:
                return []
            # Must be a forward interval
            if b_start >= b_end:
                return []
            # Minimum beat duration
            if (b_end - b_start) < 3.0:
                return []
            # Non-overlapping and ordered
            if prev_end is not None and b_start < prev_end:
                return []

            total_duration += b_end - b_start
            prev_end = b_end

        # Minimum total stitched duration
        if total_duration < 10.0:
            return []

        return list(sub_clips)
