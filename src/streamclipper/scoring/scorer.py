"""Abstract Scorer interface with signal fusion, transcript chunking, and YPP metadata validation."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
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
        max_clips: int | None = None,
    ) -> list[Highlight]:
        """Full scoring pipeline: chunk -> score -> filter duration -> fuse -> dedupe -> rank -> select."""
        chunks = self._chunk_transcript(transcript)

        raw_highlights: list[Highlight] = []
        for chunk in chunks:
            chunk_loudness = self._find_overlapping(loudness_spikes, chunk)
            chunk_chat = self._find_overlapping(chat_spikes, chunk)
            highlights = self._score_chunk(chunk, chunk_loudness, chunk_chat)
            raw_highlights.extend(highlights)

        valid_highlights: list[Highlight] = []
        for h in raw_highlights:
            if not (
                math.isfinite(h.start)
                and math.isfinite(h.end)
                and h.start >= 0.0
                and h.end > h.start
            ):
                continue

            validated_beats = self._validate_sub_clips(
                h.sub_clips,
                h.start,
                h.end,
                max_duration=self.max_clip_seconds,
            )
            h.sub_clips = validated_beats

            if validated_beats:
                effective_duration = sum(
                    float(b["end"]) - float(b["start"]) for b in validated_beats
                )
            else:
                effective_duration = h.end - h.start

            if effective_duration < self.min_clip_seconds or effective_duration > self.max_clip_seconds:
                continue

            h.final_score = self._compute_weighted_score(h, loudness_spikes, chat_spikes)
            valid_highlights.append(h)

        deduped = self._deduplicate(valid_highlights)

        limit = self.max_clips if max_clips is None else max(0, int(max_clips))
        selected = sorted(
            [h for h in deduped if h.final_score >= self.min_score],
            key=lambda h: h.final_score,
            reverse=True,
        )[:limit]

        for h in selected:
            h.word_timestamps = self._extract_words_for_range(transcript, h.start, h.end)
            scoped_text = self._extract_transcript_for_range(transcript, h.start, h.end)
            if scoped_text:
                h.transcript_chunk = scoped_text
            h.peak_time = self._extract_peak_time(h, loudness_spikes)
            h.chat_burst = self._extract_chat_burst(h, chat_spikes)

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
        """Split transcript into *chunk_minutes*-minute timestamped windows with bounded overlap.

        Each segment line is formatted as ``[start.1fs - end.1fs] text`` so LLMs
        (Ollama or MCP clients) can pinpoint exact clip and sub_clip boundaries.
        """
        minutes = chunk_minutes if chunk_minutes is not None else self.chunk_minutes
        if (
            isinstance(minutes, bool)
            or not isinstance(minutes, (int, float))
            or not math.isfinite(float(minutes))
            or float(minutes) <= 0.0
        ):
            raise ValueError(f"chunk_minutes must be finite and > 0, got {minutes!r}")
        if (
            isinstance(overlap_seconds, bool)
            or not isinstance(overlap_seconds, (int, float))
            or not math.isfinite(float(overlap_seconds))
        ):
            raise ValueError(
                f"overlap_seconds must be finite, got {overlap_seconds!r}"
            )

        if not transcript.segments:
            return []

        chunk_duration = float(minutes) * 60.0  # seconds

        total_start = transcript.segments[0].start
        total_end = transcript.segments[-1].end

        # Cap overlap to <= 25% of chunk_duration so small chunk_minutes (e.g. 1 min)
        # never collapses step size to 1 second and explodes chunk count.
        effective_overlap = max(0.0, min(float(overlap_seconds), chunk_duration * 0.25))
        step = max(chunk_duration - effective_overlap, 1.0)

        chunks: list[TranscriptChunk] = []
        window_start = total_start
        while window_start < total_end:
            window_end = min(window_start + chunk_duration, total_end)
            # Collect timestamped segment lines overlapping this window
            lines: list[str] = []
            for seg in transcript.segments:
                if seg.end > window_start and seg.start < window_end:
                    lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
            if lines:
                chunks.append(
                    TranscriptChunk(
                        start=window_start,
                        end=window_end,
                        text="\n".join(lines),
                    )
                )
            if window_end >= total_end:
                break
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
        """Fuse LLM score with available signal modalities using dynamically renormalized weights.

        When a signal modality is unavailable for the stream (e.g. VOD has no live
        chat replay so ``chat_spikes`` is empty), its weight is redistributed
        proportionally across the available modalities so highlights are not
        unfairly penalized. Additionally, top-tier AI picks (``highlight.score >= 8.5``)
        receive a quiet-moment floor (``0.85 * highlight.score``) so hilarious or
        visual-detail moments without loud yelling can still pass selection while
        ranking below signal-backed picks of the same LLM score.
        """
        loudness_overlap = self._signal_overlap_score(highlight, loudness_spikes)
        chat_overlap = self._signal_overlap_score(highlight, chat_spikes)
        highlight.loudness_score = loudness_overlap
        highlight.chat_score = chat_overlap

        raw_llm_score = float(highlight.score) if math.isfinite(float(highlight.score)) else 0.0
        llm_score = min(10.0, max(0.0, raw_llm_score))

        w_llm = float(self.weights.get("llm_score", 0.50))
        w_loud = float(self.weights.get("loudness", 0.25)) if loudness_spikes else 0.0
        w_chat = float(self.weights.get("chat_velocity", 0.25)) if chat_spikes else 0.0
        weight_sum = w_llm + w_loud + w_chat

        if weight_sum > 0.0:
            weighted = (
                (w_llm / weight_sum) * llm_score
                + (w_loud / weight_sum) * loudness_overlap
                + (w_chat / weight_sum) * chat_overlap
            )
        else:
            weighted = 0.0

        if w_llm > 0.0 and llm_score >= 8.5:
            weighted = max(weighted, min(10.0, llm_score * 0.85))

        return min(10.0, max(0.0, weighted))

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
        """Extract word-level timestamps overlapping ``[start, end]`` (including boundary-straddling words)."""
        words: list[WordTimestamp] = []
        for seg in transcript.segments:
            if seg.end <= start or seg.start >= end:
                continue
            for w in seg.words:
                if (w.end > start and w.start < end) or (
                    w.start == w.end and start <= w.start <= end
                ):
                    words.append(w)
        return words

    @staticmethod
    def _extract_transcript_for_range(
        transcript: TranscriptionResult, start: float, end: float
    ) -> str:
        """Extract spoken transcript text scoped specifically to ``[start, end]``."""
        texts: list[str] = []
        for seg in transcript.segments:
            if seg.end <= start or seg.start >= end:
                continue
            if seg.words:
                seg_words = [
                    w.text
                    for w in seg.words
                    if (w.end > start and w.start < end)
                    or (w.start == w.end and start <= w.start <= end)
                ]
                if seg_words:
                    texts.append(" ".join(seg_words))
            elif seg.text.strip():
                texts.append(seg.text.strip())
        return " ".join(texts).strip()

    # ------------------------------------------------------------------
    # YPP metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_peak_time(
        highlight: Highlight, loudness_spikes: list[TimeRange]
    ) -> float:
        """Return the exact peak second from overlapping loudness spikes, clamped to ``[start, end]``.

        Uses ``spike.metadata["peak_time"]`` when available, preferring spikes whose
        peak falls inside the highlight range and selecting the highest-scored spike.
        Falls back to the highlight midpoint.
        """
        h_start, h_end = highlight.start, highlight.end
        best_in_range_score = -1.0
        best_in_range_peak: float | None = None
        best_any_score = -1.0
        best_any_peak: float | None = None

        for spike in loudness_spikes:
            if spike.end <= h_start or spike.start >= h_end:
                continue
            peak_candidate = spike.metadata.get("peak_time")
            if peak_candidate is None:
                continue
            try:
                peak_val = float(peak_candidate)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(peak_val):
                continue

            if h_start <= peak_val <= h_end:
                if spike.score > best_in_range_score:
                    best_in_range_score = spike.score
                    best_in_range_peak = peak_val
            elif spike.score > best_any_score:
                best_any_score = spike.score
                best_any_peak = max(h_start, min(h_end, peak_val))

        if best_in_range_peak is not None:
            return best_in_range_peak
        if best_any_peak is not None:
            return best_any_peak
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
        sub_clips: list[dict[str, Any]],
        start: float,
        end: float,
        max_duration: float | None = None,
    ) -> list[dict[str, Any]]:
        """Validate multi-beat sub_clips; return ``[]`` on any violation.

        Rules:
        - ``start`` and ``end`` are finite and ``start < end``.
        - At least 2 beats (``len(sub_clips) >= 2``).
        - Each beat has finite non-boolean numeric timestamps:
          ``start <= beat["start"] < beat["end"] <= end``.
        - Each beat duration >= 3.0 seconds.
        - Beats are strictly ordered and non-overlapping
          (``prev_end <= next_start``).
        - Total stitched duration >= 10.0 seconds and (when ``max_duration`` is
          provided) ``<= max_duration``.

        Any violation clears to ``[]`` for safe single-cut fallback.
        """
        if (
            not sub_clips
            or len(sub_clips) < 2
            or not (math.isfinite(start) and math.isfinite(end))
            or end <= start
        ):
            return []

        total_duration = 0.0
        prev_end: float | None = None
        normalized: list[dict[str, Any]] = []

        for beat in sub_clips:
            if not isinstance(beat, dict):
                return []
            raw_start = beat.get("start")
            raw_end = beat.get("end")
            if isinstance(raw_start, bool) or isinstance(raw_end, bool):
                return []
            try:
                b_start = float(raw_start)  # type: ignore[arg-type]
                b_end = float(raw_end)      # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError):
                return []

            if not (math.isfinite(b_start) and math.isfinite(b_end)):
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
            normalized_beat = dict(beat)
            normalized_beat["start"] = b_start
            normalized_beat["end"] = b_end
            normalized.append(normalized_beat)

        # Total stitched duration bounds
        if total_duration < 10.0:
            return []
        if max_duration is not None and total_duration > max_duration:
            return []

        return normalized
