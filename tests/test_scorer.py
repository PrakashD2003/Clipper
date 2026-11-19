"""Unit tests for Phase 3 Scorer Interface, YPP Narrative Models, Prompt Templates & OllamaScorer."""
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import ollama
from ollama._types import GenerateResponse, ListResponse
import pytest

from streamclipper.models import (
    Highlight,
    ClipMetadata,
    TimeRange,
    TranscriptionResult,
    TranscriptSegment,
    WordTimestamp,
    SignalType,
)
from streamclipper.scoring.ollama_scorer import OllamaConnectionError, OllamaScorer
from streamclipper.scoring.scorer import Scorer, TranscriptChunk, fit_clip_range
from streamclipper.config import Config, HardwareProfile


# ---------------------------------------------------------------------------
# Minimal concrete Scorer for testing (returns pre-canned highlights)
# ---------------------------------------------------------------------------

class _StubScorer(Scorer):
    """Concrete scorer that returns pre-set highlights from _score_chunk."""

    def __init__(self, config: Config, canned: list[Highlight] | None = None):
        super().__init__(config)
        self._canned = canned or []

    def _score_chunk(self, chunk, loudness, chat):  # noqa: D401
        return [h for h in self._canned if h.start >= chunk.start and h.end <= chunk.end]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config(tmp_path: Path) -> Config:
    """Return a Config loaded from default config.default.yaml."""
    cfg = Config(user_config_path=tmp_path / "nonexistent.yaml")
    cfg._data.setdefault("analysis", {})["skip_intro_minutes"] = 0  # D2 off for legacy tests
    cfg._data["analysis"]["max_clip_seconds"] = 90  # pre-v3 clip cap for legacy tests
    return cfg


@pytest.fixture()
def long_transcript() -> TranscriptionResult:
    """Build a 48-minute transcript with 1-minute segments, each containing words."""
    segments: list[TranscriptSegment] = []
    for i in range(48):
        start = float(i * 60)
        end = float((i + 1) * 60)
        words = [
            WordTimestamp(text=f"word{j}", start=start + j * 2, end=start + j * 2 + 1.5, confidence=0.9)
            for j in range(5)
        ]
        segments.append(TranscriptSegment(
            text=f"Segment {i} content here with some spoken words",
            start=start,
            end=end,
            words=words,
        ))
    return TranscriptionResult(segments=segments, full_text="full text placeholder")


@pytest.fixture()
def loudness_spikes() -> list[TimeRange]:
    """Loudness spikes at specific positions with peak_time metadata."""
    return [
        TimeRange(start=120.0, end=140.0, score=8.5, signal_type=SignalType.LOUDNESS,
                  metadata={"peak_time": 130.0}),
        TimeRange(start=600.0, end=620.0, score=9.0, signal_type=SignalType.LOUDNESS,
                  metadata={"peak_time": 610.0}),
        TimeRange(start=1800.0, end=1820.0, score=7.0, signal_type=SignalType.LOUDNESS,
                  metadata={"peak_time": 1810.0}),
    ]


@pytest.fixture()
def chat_spikes() -> list[TimeRange]:
    """Chat velocity spikes with sample_messages metadata."""
    return [
        TimeRange(start=125.0, end=145.0, score=7.5, signal_type=SignalType.CHAT_VELOCITY,
                  metadata={"sample_messages": [
                      {"author": "viewer1", "message": "WHAT"},
                      {"author": "viewer2", "message": "NO WAY"},
                      {"author": "viewer3", "message": "LMAO"},
                  ]}),
        TimeRange(start=610.0, end=630.0, score=8.0, signal_type=SignalType.CHAT_VELOCITY,
                  metadata={"sample_messages": [
                      {"author": "viewer4", "message": "PogChamp"},
                      {"author": "viewer5", "message": "INSANE"},
                  ]}),
    ]


def _make_highlight(start: float, end: float, score: float, **kwargs: Any) -> Highlight:
    """Helper to create a Highlight with sensible defaults."""
    return Highlight(
        start=start, end=end, score=score,
        title=kwargs.get("title", "Test Title"),
        reason=kwargs.get("reason", "Test reason"),
        best_quote=kwargs.get("best_quote", ""),
        moment_type=kwargs.get("moment_type", "hype"),
        narration_intro=kwargs.get("narration_intro", ""),
        narration_outro=kwargs.get("narration_outro", ""),
        context_badge=kwargs.get("context_badge", ""),
        sub_clips=kwargs.get("sub_clips", []),
    )


# ===========================================================================
# Test: Backward-compatible model extensions
# ===========================================================================

class TestModelBackwardCompat:
    """Verify that adding YPP fields to Highlight and ClipMetadata is backward-compatible."""

    def test_highlight_from_dict_without_ypp_fields(self):
        """Phase 1 data (no YPP fields) deserializes with correct defaults."""
        data = {
            "start": 10.0, "end": 50.0, "score": 8.0,
            "title": "Epic Moment", "reason": "Great reaction",
        }
        h = Highlight.from_dict(data)
        assert h.moment_type == "hype"
        assert h.narration_intro == ""
        assert h.narration_outro == ""
        assert h.context_badge == ""
        assert h.peak_time == 0.0
        assert h.chat_burst == []
        assert h.sub_clips == []

    def test_highlight_from_dict_with_ypp_fields(self):
        """Full YPP data round-trips through from_dict."""
        data = {
            "start": 10.0, "end": 50.0, "score": 9.0,
            "title": "Clutch Moment", "reason": "Insane save",
            "moment_type": "clutch",
            "narration_intro": "Watch this clutch save closely",
            "narration_outro": "He actually pulled that off live",
            "context_badge": "🔥 Clutch Escape",
            "peak_time": 35.5,
            "chat_burst": [{"author": "u1", "message": "NO WAY"}],
            "sub_clips": [
                {"start": 10.0, "end": 25.0, "label": "Setup"},
                {"start": 35.0, "end": 50.0, "label": "Payoff"},
            ],
        }
        h = Highlight.from_dict(data)
        assert h.moment_type == "clutch"
        assert h.peak_time == 35.5
        assert len(h.chat_burst) == 1
        assert len(h.sub_clips) == 2

    def test_highlight_serialization_round_trip(self):
        """Highlight with YPP fields serializes and deserializes correctly."""
        h = _make_highlight(0.0, 60.0, 8.0, moment_type="funny",
                            narration_intro="So this guy walks into a bar",
                            context_badge="💀 Instant Karma",
                            sub_clips=[{"start": 0.0, "end": 20.0, "label": "Setup"},
                                       {"start": 40.0, "end": 60.0, "label": "Payoff"}])
        d = h.to_dict()
        h2 = Highlight.from_dict(d)
        assert h2.moment_type == "funny"
        assert h2.narration_intro == h.narration_intro
        assert h2.context_badge == h.context_badge
        assert len(h2.sub_clips) == 2

    def test_clip_metadata_from_dict_without_ypp_fields(self):
        """Phase 1 ClipMetadata data deserializes with YPP defaults."""
        data = {
            "hook_title": "Title", "caption": "Cap", "hashtags": ["tag"],
            "streamer": "s", "game": "g", "source_url": "u",
            "clip_start": 0.0, "clip_end": 60.0, "highlight_score": 8.0,
        }
        cm = ClipMetadata.from_dict(data)
        assert cm.moment_type == "hype"
        assert cm.transformation_mode == "narrative"
        assert cm.narration_intro == ""
        assert cm.narration_outro == ""
        assert cm.context_badge == ""

    def test_clip_metadata_with_ypp_fields_round_trip(self):
        """ClipMetadata with YPP fields round-trips through serialization."""
        data = {
            "hook_title": "Title", "caption": "Cap", "hashtags": ["tag"],
            "streamer": "s", "game": "g", "source_url": "u",
            "clip_start": 0.0, "clip_end": 60.0, "highlight_score": 8.0,
            "moment_type": "fail", "narration_intro": "intro text",
            "narration_outro": "outro text", "context_badge": "💀 RIP",
            "transformation_mode": "story",
        }
        cm = ClipMetadata.from_dict(data)
        assert cm.moment_type == "fail"
        assert cm.transformation_mode == "story"
        d = cm.to_dict()
        cm2 = ClipMetadata.from_dict(d)
        assert cm2.moment_type == "fail"
        assert cm2.transformation_mode == "story"


# ===========================================================================
# Test: Transcript chunking — 12-minute windows with 1-minute overlap
# ===========================================================================

class TestTranscriptChunking:
    """Verify 12-minute chunking with 1-minute overlap."""

    def test_chunk_duration_and_overlap(self, config: Config, long_transcript: TranscriptionResult):
        """Chunks should be ~12 minutes long, stepping by 11 minutes (12 - 1 overlap)."""
        scorer = _StubScorer(config)
        chunks = scorer._chunk_transcript(long_transcript)

        # 48-minute transcript:  chunks start at 0, 660, 1320, 1980
        # (window = 720s, step = 660s)
        assert len(chunks) >= 4

        for chunk in chunks:
            duration = chunk.end - chunk.start
            # Each chunk should be at most 720 seconds (12 min)
            assert duration <= 720.0 + 0.1
            # Minimum useful chunk duration
            assert duration > 0

    def test_overlap_between_consecutive_chunks(self, config: Config, long_transcript: TranscriptionResult):
        """Consecutive chunks should overlap by ~60 seconds."""
        scorer = _StubScorer(config)
        chunks = scorer._chunk_transcript(long_transcript)

        if len(chunks) >= 2:
            for i in range(len(chunks) - 1):
                overlap = chunks[i].end - chunks[i + 1].start
                # Should be ~60s overlap (1 minute)
                assert overlap > 0, "Consecutive chunks must overlap"
                assert abs(overlap - 60.0) < 1.0, f"Overlap should be ~60s, got {overlap}"

    def test_custom_chunk_minutes(self, config: Config, long_transcript: TranscriptionResult):
        """Custom chunk_minutes parameter should change window size."""
        scorer = _StubScorer(config)
        chunks_6min = scorer._chunk_transcript(long_transcript, chunk_minutes=6)
        chunks_12min = scorer._chunk_transcript(long_transcript, chunk_minutes=12)

        # 6-minute windows should produce more chunks than 12-minute
        assert len(chunks_6min) > len(chunks_12min)

    def test_empty_transcript_returns_no_chunks(self, config: Config):
        """An empty transcript should produce zero chunks."""
        scorer = _StubScorer(config)
        empty = TranscriptionResult(segments=[], full_text="")
        chunks = scorer._chunk_transcript(empty)
        assert chunks == []

    def test_chunks_contain_text(self, config: Config, long_transcript: TranscriptionResult):
        """Every chunk should have non-empty text from overlapping segments."""
        scorer = _StubScorer(config)
        chunks = scorer._chunk_transcript(long_transcript)
        for chunk in chunks:
            assert chunk.text, f"Chunk [{chunk.start}, {chunk.end}] has empty text"


# ===========================================================================
# Test: get_candidate_chunks with signal filtering
# ===========================================================================

class TestGetCandidateChunks:
    """Verify get_candidate_chunks filters dead-air chunks when only_with_signals=True."""

    def test_only_with_signals_filters_dead_chunks(
        self, config: Config, long_transcript: TranscriptionResult,
        loudness_spikes: list[TimeRange], chat_spikes: list[TimeRange],
    ):
        """Chunks with no overlapping signals should be filtered out."""
        scorer = _StubScorer(config)
        candidates = scorer.get_candidate_chunks(
            long_transcript, loudness_spikes, chat_spikes, only_with_signals=True,
        )
        # Not all chunks have signals; should have fewer candidates than total chunks
        all_chunks = scorer._chunk_transcript(long_transcript)
        assert len(candidates) < len(all_chunks)

        # Every candidate chunk must have at least one signal
        for chunk, loud, chat in candidates:
            assert loud or chat, f"Chunk [{chunk.start}, {chunk.end}] has no signals"

    def test_all_chunks_returned_when_signals_disabled(
        self, config: Config, long_transcript: TranscriptionResult,
        loudness_spikes: list[TimeRange], chat_spikes: list[TimeRange],
    ):
        """When only_with_signals=False, all chunks should be returned."""
        scorer = _StubScorer(config)
        candidates = scorer.get_candidate_chunks(
            long_transcript, loudness_spikes, chat_spikes, only_with_signals=False,
        )
        all_chunks = scorer._chunk_transcript(long_transcript)
        assert len(candidates) == len(all_chunks)


# ===========================================================================
# Test: Signal overlap score computation
# ===========================================================================

class TestSignalOverlapScore:
    """Verify _signal_overlap_score math."""

    def test_full_overlap_returns_spike_score(self, config: Config):
        """When highlight fully contains a spike, overlap fraction = 1.0."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 200.0, 8.0)
        spikes = [TimeRange(start=120.0, end=140.0, score=9.0)]
        score = scorer._signal_overlap_score(h, spikes)
        # overlap = full spike (20s), fraction = 1.0, result = 1.0 * 9.0 = 9.0
        assert abs(score - 9.0) < 0.01

    def test_partial_overlap_proportional(self, config: Config):
        """Partial overlap should scale by fraction of spike covered."""
        scorer = _StubScorer(config)
        h = _make_highlight(130.0, 200.0, 8.0)
        spikes = [TimeRange(start=120.0, end=140.0, score=10.0)]
        score = scorer._signal_overlap_score(h, spikes)
        # overlap = [130, 140] = 10s, spike_duration = 20s, fraction = 0.5
        # result = 0.5 * 10.0 = 5.0
        assert abs(score - 5.0) < 0.01

    def test_no_overlap_returns_zero(self, config: Config):
        """Non-overlapping spike should contribute 0.0."""
        scorer = _StubScorer(config)
        h = _make_highlight(200.0, 300.0, 8.0)
        spikes = [TimeRange(start=100.0, end=120.0, score=10.0)]
        score = scorer._signal_overlap_score(h, spikes)
        assert score == 0.0

    def test_multiple_spikes_takes_best(self, config: Config):
        """When multiple spikes overlap, the best score wins."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 200.0, 8.0)
        spikes = [
            TimeRange(start=110.0, end=130.0, score=5.0),  # full overlap, score=5.0
            TimeRange(start=150.0, end=170.0, score=9.0),  # full overlap, score=9.0
        ]
        score = scorer._signal_overlap_score(h, spikes)
        assert abs(score - 9.0) < 0.01

    def test_clamped_to_ten(self, config: Config):
        """Score must never exceed 10.0."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 200.0, 8.0)
        spikes = [TimeRange(start=100.0, end=200.0, score=15.0)]
        score = scorer._signal_overlap_score(h, spikes)
        assert score == 10.0

    def test_empty_spikes_returns_zero(self, config: Config):
        """Empty spike list must return 0.0."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 200.0, 8.0)
        assert scorer._signal_overlap_score(h, []) == 0.0


# ===========================================================================
# Test: Weighted score fusion
# ===========================================================================

class TestWeightedScoreFusion:
    """Verify _compute_weighted_score = 0.50 * llm + 0.25 * loudness + 0.25 * chat."""

    def test_weighted_fusion_formula(self, config: Config):
        """Check exact weighted fusion math."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 200.0, 8.0)
        loudness = [TimeRange(start=110.0, end=130.0, score=6.0)]  # full overlap -> 6.0
        chat = [TimeRange(start=150.0, end=170.0, score=4.0)]      # full overlap -> 4.0

        final = scorer._compute_weighted_score(h, loudness, chat)
        expected = 0.50 * 8.0 + 0.25 * 6.0 + 0.25 * 4.0
        assert abs(final - expected) < 0.001
        assert abs(final - 6.5) < 0.001

    def test_weighted_score_with_no_signals(self, config: Config):
        """When both loudness and chat modalities are unavailable, weights renormalize to llm_score (1.0)."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 200.0, 10.0)
        final = scorer._compute_weighted_score(h, [], [])
        assert abs(final - 10.0) < 0.001

    def test_weighted_score_renormalizes_when_chat_unavailable(self, config: Config):
        """Without live chat replay (chat_spikes=[]), llm (0.50) & loudness (0.25) renormalize to 2/3 & 1/3."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 150.0, 8.0)
        loudness = [TimeRange(start=110.0, end=130.0, score=10.0)]
        final = scorer._compute_weighted_score(h, loudness, [])
        expected = (2.0 / 3.0) * 8.0 + (1.0 / 3.0) * 10.0
        assert abs(final - expected) < 0.001
        assert final >= 7.0

    def test_weighted_score_quiet_top_tier_ai_pick_passes_while_signal_backed_ranks_higher(
        self, config: Config
    ):
        """A quiet 9.0 AI pick gets a 0.85*score floor (7.65 >= 7.0), while a signal-backed 9.0 still beats it."""
        scorer = _StubScorer(config)
        distant_loud = [TimeRange(start=500.0, end=520.0, score=9.0)]
        distant_chat = [TimeRange(start=505.0, end=525.0, score=9.0)]

        quiet_gem = _make_highlight(100.0, 145.0, 9.0, moment_type="funny")
        loud_hype = _make_highlight(490.0, 540.0, 9.0, moment_type="hype")
        quiet_mediocre = _make_highlight(100.0, 145.0, 7.5, moment_type="funny")

        s_quiet = scorer._compute_weighted_score(quiet_gem, distant_loud, distant_chat)
        s_loud = scorer._compute_weighted_score(loud_hype, distant_loud, distant_chat)
        s_mediocre = scorer._compute_weighted_score(quiet_mediocre, distant_loud, distant_chat)

        assert s_quiet == pytest.approx(7.65)
        assert s_loud == pytest.approx(9.0)
        assert s_loud > s_quiet >= 7.0
        assert s_mediocre < 7.0

    def test_weighted_score_sets_loudness_and_chat_fields(self, config: Config):
        """_compute_weighted_score should set h.loudness_score and h.chat_score."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 200.0, 8.0)
        loudness = [TimeRange(start=120.0, end=140.0, score=7.0)]
        chat = [TimeRange(start=150.0, end=170.0, score=5.0)]
        scorer._compute_weighted_score(h, loudness, chat)
        assert h.loudness_score == 7.0
        assert h.chat_score == 5.0


# ===========================================================================
# Test: peak_time and chat_burst extraction
# ===========================================================================

class TestPeakTimeAndChatBurst:
    """Verify _extract_peak_time and _extract_chat_burst."""

    def test_peak_time_from_loudness_metadata(self, config: Config, loudness_spikes: list[TimeRange]):
        """peak_time should come from the highest-scored overlapping spike's metadata."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 200.0, 8.0)
        peak = scorer._extract_peak_time(h, loudness_spikes)
        # spike at [120, 140] with peak_time=130.0 overlaps
        assert peak == 130.0

    def test_peak_time_fallback_to_midpoint(self, config: Config):
        """When no overlapping spike has peak_time metadata, use midpoint."""
        scorer = _StubScorer(config)
        h = _make_highlight(500.0, 560.0, 8.0)
        peak = scorer._extract_peak_time(h, [])
        assert peak == 530.0  # (500 + 560) / 2

    def test_peak_time_selects_highest_scored_spike(self, config: Config):
        """When multiple spikes overlap, the peak_time from the highest-scored wins."""
        scorer = _StubScorer(config)
        h = _make_highlight(590.0, 650.0, 8.0)
        spikes = [
            TimeRange(start=595.0, end=615.0, score=5.0, metadata={"peak_time": 605.0}),
            TimeRange(start=600.0, end=625.0, score=9.0, metadata={"peak_time": 612.0}),
        ]
        peak = scorer._extract_peak_time(h, spikes)
        assert peak == 612.0

    def test_chat_burst_extraction(self, config: Config, chat_spikes: list[TimeRange]):
        """chat_burst should collect sample_messages from overlapping chat spikes."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 200.0, 8.0)
        burst = scorer._extract_chat_burst(h, chat_spikes)
        # spike at [125, 145] overlaps, has 3 messages
        assert len(burst) == 3
        assert burst[0]["message"] == "WHAT"

    def test_chat_burst_max_five(self, config: Config):
        """chat_burst should return at most 5 messages."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 300.0, 8.0)
        spikes = [
            TimeRange(start=110.0, end=130.0, score=5.0, signal_type=SignalType.CHAT_VELOCITY,
                      metadata={"sample_messages": [
                          {"author": f"u{i}", "message": f"msg{i}"} for i in range(10)
                      ]}),
        ]
        burst = scorer._extract_chat_burst(h, spikes)
        assert len(burst) == 5

    def test_chat_burst_empty_when_no_overlap(self, config: Config):
        """No overlapping chat spikes should return empty list."""
        scorer = _StubScorer(config)
        h = _make_highlight(1000.0, 1100.0, 8.0)
        burst = scorer._extract_chat_burst(h, [])
        assert burst == []


# ===========================================================================
# Test: sub_clips validation
# ===========================================================================

class TestSubClipsValidation:
    """Verify _validate_sub_clips valid and invalid scenarios."""

    def test_valid_two_beat_sub_clips(self, config: Config):
        """Two valid beats should pass validation."""
        scorer = _StubScorer(config)
        sub_clips = [
            {"start": 10.0, "end": 20.0, "label": "Setup"},
            {"start": 25.0, "end": 40.0, "label": "Payoff"},
        ]
        result = scorer._validate_sub_clips(sub_clips, start=10.0, end=40.0)
        assert len(result) == 2
        assert result[0]["label"] == "Setup"

    def test_valid_three_beat_sub_clips(self, config: Config):
        """Three valid beats should pass validation."""
        scorer = _StubScorer(config)
        sub_clips = [
            {"start": 0.0, "end": 10.0, "label": "Setup"},
            {"start": 15.0, "end": 25.0, "label": "Escalation"},
            {"start": 30.0, "end": 45.0, "label": "Payoff"},
        ]
        result = scorer._validate_sub_clips(sub_clips, start=0.0, end=45.0)
        assert len(result) == 3

    def test_single_beat_returns_empty(self, config: Config):
        """Fewer than 2 beats should return []."""
        scorer = _StubScorer(config)
        sub_clips = [{"start": 10.0, "end": 25.0, "label": "Only"}]
        result = scorer._validate_sub_clips(sub_clips, start=10.0, end=25.0)
        assert result == []

    def test_empty_sub_clips_returns_empty(self, config: Config):
        """Empty sub_clips input should return []."""
        scorer = _StubScorer(config)
        assert scorer._validate_sub_clips([], start=0.0, end=60.0) == []

    def test_beat_outside_range_returns_empty(self, config: Config):
        """Beat extending outside [start, end] should fail."""
        scorer = _StubScorer(config)
        sub_clips = [
            {"start": 5.0, "end": 15.0, "label": "Setup"},  # starts before highlight start
            {"start": 20.0, "end": 30.0, "label": "Payoff"},
        ]
        result = scorer._validate_sub_clips(sub_clips, start=10.0, end=30.0)
        assert result == []

    def test_beat_too_short_returns_empty(self, config: Config):
        """Beats shorter than 3.0 seconds should fail."""
        scorer = _StubScorer(config)
        sub_clips = [
            {"start": 10.0, "end": 12.0, "label": "Setup"},  # only 2s
            {"start": 20.0, "end": 30.0, "label": "Payoff"},
        ]
        result = scorer._validate_sub_clips(sub_clips, start=10.0, end=30.0)
        assert result == []

    def test_overlapping_beats_return_empty(self, config: Config):
        """Overlapping beats should fail validation."""
        scorer = _StubScorer(config)
        sub_clips = [
            {"start": 10.0, "end": 25.0, "label": "Setup"},
            {"start": 20.0, "end": 35.0, "label": "Payoff"},  # overlaps with beat 1
        ]
        result = scorer._validate_sub_clips(sub_clips, start=10.0, end=35.0)
        assert result == []

    def test_total_duration_too_short_returns_empty(self, config: Config):
        """Total stitched duration < 10s should fail."""
        scorer = _StubScorer(config)
        sub_clips = [
            {"start": 10.0, "end": 13.5, "label": "Setup"},   # 3.5s
            {"start": 20.0, "end": 23.5, "label": "Payoff"},  # 3.5s = total 7.0s < 10.0s
        ]
        result = scorer._validate_sub_clips(sub_clips, start=10.0, end=30.0)
        assert result == []

    def test_missing_keys_returns_empty(self, config: Config):
        """Missing start/end keys should fail gracefully."""
        scorer = _StubScorer(config)
        sub_clips = [
            {"label": "Setup"},  # no start/end
            {"start": 20.0, "end": 30.0, "label": "Payoff"},
        ]
        result = scorer._validate_sub_clips(sub_clips, start=0.0, end=30.0)
        assert result == []

    def test_unordered_beats_return_empty(self, config: Config):
        """Beats in reverse order should fail validation."""
        scorer = _StubScorer(config)
        sub_clips = [
            {"start": 30.0, "end": 45.0, "label": "Payoff"},
            {"start": 10.0, "end": 25.0, "label": "Setup"},
        ]
        result = scorer._validate_sub_clips(sub_clips, start=10.0, end=45.0)
        assert result == []


# ===========================================================================
# Test: Deduplication (IoU-based)
# ===========================================================================

class TestDeduplication:
    """Verify _deduplicate merges overlapping highlights keeping the highest score."""

    def test_non_overlapping_kept(self, config: Config):
        """Non-overlapping highlights should all be kept."""
        scorer = _StubScorer(config)
        highlights = [
            _make_highlight(0.0, 60.0, 8.0),
            _make_highlight(100.0, 160.0, 7.0),
        ]
        for h in highlights:
            h.final_score = h.score
        result = scorer._deduplicate(highlights)
        assert len(result) == 2

    def test_overlapping_keeps_higher_score(self, config: Config):
        """Overlapping highlights should keep only the higher-scored one."""
        scorer = _StubScorer(config)
        h1 = _make_highlight(0.0, 60.0, 9.0)
        h1.final_score = 9.0
        h2 = _make_highlight(10.0, 70.0, 7.0)
        h2.final_score = 7.0
        # IoU = overlap(50s) / union(70s) ≈ 0.714 > 0.3 threshold
        result = scorer._deduplicate([h1, h2])
        assert len(result) == 1
        assert result[0].final_score == 9.0

    def test_exact_duplicate_keeps_one(self, config: Config):
        """Identical time ranges should keep only one (IoU = 1.0)."""
        scorer = _StubScorer(config)
        h1 = _make_highlight(100.0, 200.0, 8.0)
        h1.final_score = 8.0
        h2 = _make_highlight(100.0, 200.0, 6.0)
        h2.final_score = 6.0
        result = scorer._deduplicate([h1, h2])
        assert len(result) == 1
        assert result[0].final_score == 8.0

    def test_empty_input(self, config: Config):
        """Empty input should return empty list."""
        scorer = _StubScorer(config)
        assert scorer._deduplicate([]) == []

    def test_low_iou_keeps_both(self, config: Config):
        """Highlights with IoU < threshold should both survive."""
        scorer = _StubScorer(config)
        # h1: [0, 100], h2: [80, 200]
        # overlap: [80, 100] = 20s, union: 200s, IoU = 20/200 = 0.1 < 0.3
        h1 = _make_highlight(0.0, 100.0, 8.0)
        h1.final_score = 8.0
        h2 = _make_highlight(80.0, 200.0, 7.0)
        h2.final_score = 7.0
        result = scorer._deduplicate([h1, h2])
        assert len(result) == 2


# ===========================================================================
# Test: Word timestamp extraction
# ===========================================================================

class TestWordExtraction:
    """Verify _extract_words_for_range extracts correct words."""

    def test_extract_words_in_range(self, config: Config, long_transcript: TranscriptionResult):
        """Should extract only words whose timestamps fall within the range."""
        scorer = _StubScorer(config)
        # Segment 0: start=0.0, end=60.0, words at [0, 1.5], [2, 3.5], ...
        words = scorer._extract_words_for_range(long_transcript, 0.0, 60.0)
        assert len(words) > 0
        for w in words:
            assert w.start >= 0.0
            assert w.end <= 60.0

    def test_extract_no_words_outside_range(self, config: Config, long_transcript: TranscriptionResult):
        """Range with no matching segments should return empty."""
        scorer = _StubScorer(config)
        words = scorer._extract_words_for_range(long_transcript, 5000.0, 5100.0)
        assert words == []


# ===========================================================================
# Test: Full score_and_select pipeline
# ===========================================================================

class TestScoreAndSelect:
    """Verify the end-to-end score_and_select pipeline."""

    def test_pipeline_attaches_metadata(
        self, config: Config, long_transcript: TranscriptionResult,
        loudness_spikes: list[TimeRange], chat_spikes: list[TimeRange],
    ):
        """Selected highlights should have peak_time, chat_burst, and word_timestamps populated."""
        canned = [
            _make_highlight(110.0, 160.0, 9.0),  # overlaps loudness[0] and chat[0]
        ]
        scorer = _StubScorer(config, canned=canned)
        results = scorer.score_and_select(long_transcript, loudness_spikes, chat_spikes)

        if results:
            h = results[0]
            assert h.peak_time != 0.0, "peak_time should be populated"
            assert isinstance(h.chat_burst, list)
            assert isinstance(h.word_timestamps, list)
            assert isinstance(h.sub_clips, list)


# ===========================================================================
# Test: Prompt templates exist and have required placeholders
# ===========================================================================

class TestPromptTemplates:
    """Verify all 4 prompt templates exist and contain expected placeholders."""

    PROMPTS_DIR = Path(__file__).parent.parent / "src" / "streamclipper" / "prompts"

    def test_highlight_scorer_template(self):
        """highlight_scorer.txt must exist with required placeholders."""
        path = self.PROMPTS_DIR / "highlight_scorer.txt"
        assert path.exists(), f"Missing {path}"
        content = path.read_text(encoding="utf-8")
        for placeholder in [
            "{chunk_minutes}", "{start_time}", "{end_time}",
            "{transcript_text}", "{loudness_data}", "{chat_data}",
        ]:
            assert placeholder in content, f"Missing placeholder {placeholder}"
        # Must mention moment_type, narration_intro, narration_outro, context_badge, sub_clips
        for keyword in ["moment_type", "narration_intro", "narration_outro", "context_badge", "sub_clips"]:
            assert keyword in content, f"Missing keyword {keyword}"

    def test_narrative_writer_template(self):
        """narrative_writer.txt must exist with moment_type tone matching."""
        path = self.PROMPTS_DIR / "narrative_writer.txt"
        assert path.exists(), f"Missing {path}"
        content = path.read_text(encoding="utf-8")
        for placeholder in [
            "{streamer}", "{game}", "{moment_type}",
            "{hook_title}", "{description}", "{quote}",
        ]:
            assert placeholder in content, f"Missing placeholder {placeholder}"
        # Must mention all moment types for adaptive tone
        for mtype in ["funny", "fail", "graphics_detail", "hype", "clutch", "chat_chaos", "lore_secret", "analysis"]:
            assert mtype in content, f"Missing moment_type '{mtype}'"

    def test_hook_generator_template(self):
        """hook_generator.txt must exist with required placeholders."""
        path = self.PROMPTS_DIR / "hook_generator.txt"
        assert path.exists(), f"Missing {path}"
        content = path.read_text(encoding="utf-8")
        for placeholder in ["{streamer}", "{game}", "{description}", "{quote}"]:
            assert placeholder in content, f"Missing placeholder {placeholder}"

    def test_caption_writer_template(self):
        """caption_writer.txt must exist with required placeholders."""
        path = self.PROMPTS_DIR / "caption_writer.txt"
        assert path.exists(), f"Missing {path}"
        content = path.read_text(encoding="utf-8")
        for placeholder in [
            "{streamer}", "{game}", "{hook_title}",
            "{description}", "{hashtag_count}",
        ]:
            assert placeholder in content, f"Missing placeholder {placeholder}"


# ===========================================================================
# Package 3.2 — OllamaScorer (Standalone Mode)
# ===========================================================================


def _make_hw_profile(tier: str, llm_model: str) -> HardwareProfile:
    """Create a deterministic HardwareProfile for testing auto model resolution."""
    return HardwareProfile(
        tier=tier,
        gpu_name="Test GPU",
        vram_gb=6.0,
        compute_capability="8.9" if tier == "cuda_mid_tensor" else "7.5",
        has_tensor_cores=(tier == "cuda_mid_tensor"),
        has_nvenc=True,
        device="cuda",
        model_size="large-v2" if tier == "cuda_mid_tensor" else "medium",
        compute_type="int8_float16" if tier == "cuda_mid_tensor" else "int8",
        batch_size=8 if tier == "cuda_mid_tensor" else 4,
        llm_model=llm_model,
        video_codec="h264_nvenc",
    )


class TestOllamaScorerInit:
    """Verify OllamaScorer wires HardwareProfile-resolved model and ollama.Client."""

    @pytest.mark.parametrize(
        "tier,expected_model",
        [
            ("cuda_mid_tensor", "qwen3:8b"),
            ("cuda_mid_no_tensor", "qwen3:4b"),
        ],
    )
    def test_uses_hardware_resolved_llm_model_and_client_args(
        self, tmp_path: Path, tier: str, expected_model: str
    ):
        """Auto llm.model resolves via HardwareProfile and passes host/timeout to ollama.Client."""
        cfg = Config(
            user_config_path=tmp_path / "nonexistent.yaml",
            hardware_profile=_make_hw_profile(tier, expected_model),
        )
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            scorer = OllamaScorer(cfg)

        assert scorer.model == expected_model
        assert scorer.base_url == "http://localhost:11434"
        assert scorer.timeout == 120.0
        mock_client_cls.assert_called_once_with(
            host="http://localhost:11434", timeout=120.0
        )

    def test_respects_user_llm_override(self, tmp_path: Path):
        """Explicit user llm.model and base_url override HardwareProfile defaults."""
        user_cfg = tmp_path / "config.yaml"
        user_cfg.write_text(
            "llm:\n  model: 'mistral:7b'\n  base_url: 'http://127.0.0.1:11435'\n  timeout_seconds: 60\n",
            encoding="utf-8",
        )
        cfg = Config(
            user_config_path=user_cfg,
            hardware_profile=_make_hw_profile("cuda_mid_tensor", "qwen3:8b"),
        )
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            scorer = OllamaScorer(cfg)

        assert scorer.model == "mistral:7b"
        assert scorer.base_url == "http://127.0.0.1:11435"
        assert scorer.timeout == 60.0
        mock_client_cls.assert_called_once_with(
            host="http://127.0.0.1:11435", timeout=60.0
        )


class TestOllamaScorerResponseParsing:
    """Verify _parse_response handles lists, wrapper objects, YPP fields, and edge cases."""

    @pytest.fixture()
    def scorer(self, config: Config) -> OllamaScorer:
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client"):
            return OllamaScorer(config)

    def test_parses_raw_json_list_with_ypp_fields_and_sub_clips(self, scorer: OllamaScorer):
        """Raw JSON list populates moment_type, narrative fields, context_badge, and sub_clips."""
        payload = [
            {
                "start": 120.0,
                "end": 180.0,
                "score": 9,
                "moment_type": "graphics_detail",
                "title": "Streamer spots insane water reflection",
                "reason": "Stops mid-chase to zoom in on dynamic puddle physics.",
                "best_quote": "WAIT LOOK AT THE REFLECTION IN THAT PUDDLE!",
                "narration_intro": "Watch the street reflection closely as he pulls around this corner.",
                "narration_outro": "Even mid-chase, that real-time ray-traced puddle stopped the whole stream.",
                "context_badge": "🎨 Insane Detail",
                "sub_clips": [
                    {"start": 120.0, "end": 135.0, "label": "Setup"},
                    {"start": 155.0, "end": 180.0, "label": "Payoff"},
                ],
            }
        ]
        highlights = scorer._parse_response(json.dumps(payload), chunk_offset=0.0)
        assert len(highlights) == 1
        h = highlights[0]
        assert h.start == 120.0
        assert h.end == 180.0
        assert h.score == 9.0
        assert h.moment_type == "graphics_detail"
        assert h.narration_intro.startswith("Watch the street reflection")
        assert h.narration_outro.startswith("Even mid-chase")
        assert h.context_badge == "🎨 Insane Detail"
        assert len(h.sub_clips) == 2
        assert h.sub_clips[0] == {"start": 120.0, "end": 135.0, "label": "Setup"}

    def test_parses_wrapped_highlights_object_with_think_tags_and_fences(
        self, scorer: OllamaScorer
    ):
        """Wrapped {'highlights': [...]} payload with Qwen3 <think> and markdown fences parses cleanly."""
        wrapped = {
            "highlights": [
                {
                    "start": 400.0,
                    "end": 455.0,
                    "score": 8.5,
                    "moment_type": "fail",
                    "title": "Instant karma after bragging on stream",
                    "reason": "Claims he never crashes right before flipping the car.",
                    "best_quote": "I literally cannot crash this thing—",
                    "narration_intro": "He really said he never crashes two seconds before this turn.",
                    "narration_outro": "Classic streamer jinx delivered in under three seconds flat.",
                    "context_badge": "💀 Instant Karma",
                    "sub_clips": [],
                }
            ]
        }
        raw = f"<think>Analyzing spikes at 430s...</think>\n```json\n{json.dumps(wrapped)}\n```"
        highlights = scorer._parse_response(raw, chunk_offset=0.0)
        assert len(highlights) == 1
        assert highlights[0].moment_type == "fail"
        assert highlights[0].context_badge == "💀 Instant Karma"
        assert highlights[0].sub_clips == []

    def test_normalizes_invalid_moment_type_and_clamps_score(self, scorer: OllamaScorer):
        """Unknown moment_type falls back to 'hype' and scores clamp to [0, 10]."""
        payload = [
            {
                "start": 10.0,
                "end": 50.0,
                "score": 15.0,
                "moment_type": "totally_made_up",
                "title": "Wild Moment",
                "reason": "Something wild happened",
            }
        ]
        highlights = scorer._parse_response(json.dumps(payload), chunk_offset=0.0)
        assert len(highlights) == 1
        assert highlights[0].moment_type == "hype"
        assert highlights[0].score == 10.0

    def test_shifts_relative_timestamps_when_chunk_offset_positive(
        self, scorer: OllamaScorer
    ):
        """Chunk-relative timestamps (end <= chunk_offset) are shifted to absolute stream time."""
        payload = {
            "highlights": [
                {
                    "start": 20.0,
                    "end": 75.0,
                    "score": 8.0,
                    "moment_type": "clutch",
                    "title": "Last second escape",
                    "reason": "Dodges 5 cops with 1 HP",
                    "sub_clips": [
                        {"start": 20.0, "end": 35.0, "label": "Setup"},
                        {"start": 50.0, "end": 75.0, "label": "Payoff"},
                    ],
                }
            ]
        }
        highlights = scorer._parse_response(json.dumps(payload), chunk_offset=660.0)
        assert len(highlights) == 1
        assert highlights[0].start == 680.0
        assert highlights[0].end == 735.0
        assert highlights[0].sub_clips[0]["start"] == 680.0
        assert highlights[0].sub_clips[1]["end"] == 735.0

    def test_empty_responses_and_invalid_ranges(self, scorer: OllamaScorer):
        """Empty list/object returns [] and inverted ranges are skipped."""
        assert scorer._parse_response("[]", chunk_offset=0.0) == []
        assert scorer._parse_response("{}", chunk_offset=0.0) == []
        bad_range = [{"start": 60.0, "end": 40.0, "score": 8, "title": "Bad", "reason": "x"}]
        assert scorer._parse_response(json.dumps(bad_range), chunk_offset=0.0) == []


class TestOllamaScorerScoreChunkAndRetries:
    """Verify _score_chunk retries on malformed JSON and integrates with Scorer pipeline."""

    def test_retries_on_malformed_json_and_recovers(
        self, config: Config, loudness_spikes: list[TimeRange], chat_spikes: list[TimeRange]
    ):
        """Malformed JSON on attempts 1 & 2 recovers cleanly on attempt 3."""
        valid_payload = json.dumps(
            {
                "highlights": [
                    {
                        "start": 115.0,
                        "end": 165.0,
                        "score": 9,
                        "moment_type": "funny",
                        "title": "Streamer loses it at NPC dialogue",
                        "reason": "Unexpected NPC roast triggers 20-second laugh attack.",
                        "best_quote": "Did he really just say that?!",
                        "narration_intro": "Wait until you hear what this random NPC says next.",
                        "narration_outro": "Chat could not stop replaying that roast for five minutes.",
                        "context_badge": "😂 NPC Roast",
                        "sub_clips": [],
                    }
                ]
            }
        )
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.generate.side_effect = [
                GenerateResponse(response="{not valid json"),
                {"response": '{"unexpected_key": 123}'},
                GenerateResponse(response=valid_payload),
            ]
            scorer = OllamaScorer(config)
            chunk = TranscriptChunk(start=0.0, end=720.0, text="Did he really just say that?!")
            highlights = scorer._score_chunk(chunk, loudness_spikes[:1], chat_spikes[:1])

        assert mock_client.generate.call_count == 3
        assert len(highlights) == 1
        assert highlights[0].moment_type == "funny"
        assert highlights[0].context_badge == "😂 NPC Roast"
        # Verify format="json" and temperature were forwarded
        first_kwargs = mock_client.generate.call_args_list[0].kwargs
        assert first_kwargs["format"] == "json"
        assert first_kwargs["model"] == scorer.model
        assert first_kwargs["options"]["temperature"] == scorer.temperature

    def test_returns_empty_after_three_malformed_attempts(self, config: Config):
        """Exhausting all 3 retries on malformed JSON degrades gracefully to []."""
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.generate.return_value = {"response": "sorry I cannot output JSON"}
            scorer = OllamaScorer(config)
            chunk = TranscriptChunk(start=0.0, end=720.0, text="Some transcript text")
            result = scorer._score_chunk(chunk, [], [])

        assert result == []
        assert mock_client.generate.call_count == 3

    def test_connection_error_during_generate_propagates_with_model_hint(self, config: Config):
        """Model-not-found / connection errors raise OllamaConnectionError without JSON retries."""
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.generate.side_effect = ollama.ResponseError(
                "model not found", status_code=404
            )
            scorer = OllamaScorer(config)
            chunk = TranscriptChunk(start=0.0, end=720.0, text="Hello stream")
            with pytest.raises(OllamaConnectionError) as exc_info:
                scorer._score_chunk(chunk, [], [])

        assert mock_client.generate.call_count == 1
        assert scorer.model in str(exc_info.value)
        assert f"ollama pull {scorer.model}" in str(exc_info.value)

    def test_end_to_end_score_and_select_with_ollama(
        self,
        config: Config,
        long_transcript: TranscriptionResult,
        loudness_spikes: list[TimeRange],
        chat_spikes: list[TimeRange],
    ):
        """OllamaScorer inherits score_and_select and attaches peak_time, chat_burst, validated sub_clips."""
        chunk_response = json.dumps(
            [
                {
                    "start": 115.0,
                    "end": 165.0,
                    "score": 9.0,
                    "moment_type": "hype",
                    "title": "Massive explosion sends car flying",
                    "reason": "Huge reaction paired with chat burst.",
                    "best_quote": "NO WAY!",
                    "narration_intro": "Watch the highway intersection closely right before chaos erupts.",
                    "narration_outro": "That physics launch had the entire lobby screaming.",
                    "context_badge": "🔥 Pure Chaos",
                    "sub_clips": [
                        {"start": 115.0, "end": 128.0, "label": "Setup"},
                        {"start": 140.0, "end": 165.0, "label": "Payoff"},
                    ],
                }
            ]
        )

        def _fake_generate(**kwargs: Any) -> dict[str, str]:
            # Only return highlight on first chunk containing 0..720s
            if "(0.0s)" in kwargs["prompt"]:
                return {"response": chunk_response}
            return {"response": "[]"}

        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client_cls.return_value.generate.side_effect = _fake_generate
            scorer = OllamaScorer(config)
            selected = scorer.score_and_select(long_transcript, loudness_spikes, chat_spikes)

        assert len(selected) == 1
        h = selected[0]
        assert h.peak_time == 130.0
        assert len(h.chat_burst) == 3
        assert len(h.sub_clips) == 2
        assert h.final_score >= config.get("analysis.min_highlight_score", 7.0)


class TestOllamaScorerNarrativeAndMetadata:
    """Verify generate_narrative() and generate_metadata() prompt workflows."""

    def test_generate_narrative_uses_moment_type_and_retries(self, config: Config):
        """generate_narrative sends moment_type in prompt and recovers from malformed JSON."""
        narrative_json = json.dumps(
            {
                "narration_intro": "Look closely at the neon reflections on the wet sidewalk here.",
                "narration_outro": "Rockstar modeled every single ripple. Did you notice that?",
                "context_badge": "🎨 Insane Detail",
            }
        )
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.generate.side_effect = [
                {"response": "not json"},
                GenerateResponse(response=narrative_json),
            ]
            scorer = OllamaScorer(config)
            h = _make_highlight(
                100.0, 150.0, 9.0,
                title="He freezes over puddle graphics",
                reason="Zooms in on dynamic street reflections.",
                best_quote="Bro look at that water!",
                moment_type="graphics_detail",
            )
            intro, outro, badge = scorer.generate_narrative(h, "Tim", "GTA 6")

        assert mock_client.generate.call_count == 2
        sent_prompt = mock_client.generate.call_args_list[0].kwargs["prompt"]
        assert "graphics_detail" in sent_prompt
        assert "Tim" in sent_prompt
        assert intro == "Look closely at the neon reflections on the wet sidewalk here."
        assert outro == "Rockstar modeled every single ripple. Did you notice that?"
        assert badge == "🎨 Insane Detail"

    def test_generate_metadata_backfills_missing_narrative_fields(self, config: Config):
        """When highlight lacks narrative fields, generate_metadata calls hook, narrative, and caption prompts."""
        narrative_json = json.dumps(
            {
                "narration_intro": "He swore he had this getaway completely under control.",
                "narration_outro": "Instant karma hit him fast. Ever done that?",
                "context_badge": "💀 Instant Karma",
            }
        )
        caption_json = json.dumps(
            {
                "caption": "Tim really jinxed himself in record time on this heist. Drop a comment if you saw that coming!",
                "hashtags": ["#GTA6", "Tim", "#GamingClips"],
            }
        )
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.generate.side_effect = [
                GenerateResponse(response='"Tim instantly regrets saying this in GTA 6"'),
                GenerateResponse(response=narrative_json),
                GenerateResponse(response=caption_json),
            ]
            scorer = OllamaScorer(config)
            h = _make_highlight(
                200.0, 250.0, 8.8,
                title="Fallback title",
                reason="Brags about driving skills right before crashing into a pole.",
                best_quote="I am literally the best driver in Vice City.",
                moment_type="fail",
            )
            h.final_score = 8.9
            meta = scorer.generate_metadata(
                h, streamer="Tim", game="GTA 6", source_url="https://youtu.be/abc123"
            )

        assert mock_client.generate.call_count == 3
        assert meta.hook_title == "Tim instantly regrets saying this in GTA 6"
        assert meta.narration_intro.startswith("He swore he had this getaway")
        assert meta.narration_outro.startswith("Instant karma hit him")
        assert meta.context_badge == "💀 Instant Karma"
        assert meta.hashtags == ["GTA6", "Tim", "GamingClips"]
        assert meta.moment_type == "fail"
        assert meta.highlight_score == 8.9
        assert meta.transformation_mode == "narrative"

    def test_generate_metadata_preserves_existing_narrative_without_extra_call(
        self, config: Config
    ):
        """When intro, outro, and badge are already present, narrative_writer is skipped."""
        caption_json = json.dumps(
            {
                "caption": "Chat went completely feral during this chase. Follow for daily clips!",
                "hashtags": ["GTA6", "xQc"],
            }
        )
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.generate.side_effect = [
                GenerateResponse(response="xQc cannot believe chat tricked him"),
                GenerateResponse(response=caption_json),
            ]
            scorer = OllamaScorer(config)
            h = _make_highlight(
                300.0, 350.0, 8.5,
                reason="Chat convinces xQc to press eject mid-flight.",
                moment_type="chat_chaos",
                narration_intro="Chat told him this button activates turbo boost.",
                narration_outro="Five thousand viewers trolled him at the exact same second.",
                context_badge="💬 Chat Trolled Him",
            )
            meta = scorer.generate_metadata(
                h, streamer="xQc", game="GTA 6", source_url="https://youtu.be/xyz789"
            )

        assert mock_client.generate.call_count == 2
        assert meta.narration_intro == "Chat told him this button activates turbo boost."
        assert meta.narration_outro == "Five thousand viewers trolled him at the exact same second."
        assert meta.context_badge == "💬 Chat Trolled Him"


class TestOllamaScorerCheckConnection:
    """Verify check_connection() checks server reachability and resolved model availability."""

    def test_check_connection_succeeds_with_ollama_list_response(self, tmp_path: Path):
        """check_connection returns True when resolved model is in ListResponse."""
        cfg = Config(
            user_config_path=tmp_path / "nonexistent.yaml",
            hardware_profile=_make_hw_profile("cuda_mid_tensor", "qwen3:8b"),
        )
        list_resp = ListResponse(models=[ListResponse.Model(model="qwen3:8b")])
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client_cls.return_value.list.return_value = list_resp
            scorer = OllamaScorer(cfg)
            assert scorer.check_connection() is True

    def test_check_connection_succeeds_with_dict_response_and_latest_tag(self, tmp_path: Path):
        """check_connection accepts dict list payload and implicit :latest tag equivalence."""
        user_cfg = tmp_path / "config.yaml"
        user_cfg.write_text("llm:\n  model: 'llama3'\n", encoding="utf-8")
        cfg = Config(user_config_path=user_cfg)
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client_cls.return_value.list.return_value = {
                "models": [{"name": "llama3:latest"}]
            }
            scorer = OllamaScorer(cfg)
            assert scorer.check_connection() is True

    def test_check_connection_raises_when_model_missing(self, tmp_path: Path):
        """Missing resolved model raises OllamaConnectionError with exact `ollama pull` command."""
        cfg = Config(
            user_config_path=tmp_path / "nonexistent.yaml",
            hardware_profile=_make_hw_profile("cuda_mid_tensor", "qwen3:8b"),
        )
        list_resp = ListResponse(models=[ListResponse.Model(model="qwen3:4b")])
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client_cls.return_value.list.return_value = list_resp
            scorer = OllamaScorer(cfg)
            with pytest.raises(OllamaConnectionError) as exc_info:
                scorer.check_connection()

        msg = str(exc_info.value)
        assert "qwen3:8b" in msg
        assert "ollama pull qwen3:8b" in msg

    def test_check_connection_raises_when_server_unreachable(self, tmp_path: Path):
        """Unreachable Ollama server raises OllamaConnectionError mentioning model and URL."""
        cfg = Config(
            user_config_path=tmp_path / "nonexistent.yaml",
            hardware_profile=_make_hw_profile("cuda_mid_no_tensor", "qwen3:4b"),
        )
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client_cls.return_value.list.side_effect = ConnectionError("Connection refused")
            scorer = OllamaScorer(cfg)
            with pytest.raises(OllamaConnectionError) as exc_info:
                scorer.check_connection()

        msg = str(exc_info.value)
        assert "qwen3:4b" in msg
        assert "http://localhost:11434" in msg
        assert "ollama serve" in msg


class TestOllamaScorerEdgeCasesAndHardening:
    """Regression tests for null JSON fields, non-finite numbers, offset bounds, tags, and raw mode."""

    @pytest.fixture()
    def scorer(self, config: Config) -> OllamaScorer:
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client"):
            return OllamaScorer(config)

    def test_null_json_fields_never_stringify_to_none(self, scorer: OllamaScorer):
        """Explicit JSON null values map to empty strings / defaults, never literal 'None'."""
        payload = [
            {
                "start": 100.0,
                "end": 150.0,
                "score": 8.0,
                "moment_type": None,
                "title": None,
                "reason": None,
                "best_quote": None,
                "narration_intro": None,
                "narration_outro": None,
                "context_badge": None,
                "sub_clips": [
                    {"start": 100.0, "end": 120.0, "label": None},
                    {"start": 130.0, "end": 150.0, "label": "Payoff"},
                ],
            }
        ]
        highlights = scorer._parse_response(json.dumps(payload), chunk_offset=0.0)
        assert len(highlights) == 1
        h = highlights[0]
        assert h.moment_type == "hype"
        assert h.title == "Stream Highlight"
        assert h.reason == ""
        assert h.best_quote == ""
        assert h.narration_intro == ""
        assert h.narration_outro == ""
        assert h.context_badge == ""
        assert h.sub_clips[0]["label"] == ""

    def test_null_metadata_in_loudness_and_chat_formatters(self, scorer: OllamaScorer):
        """None peak_time or None chat message values do not crash or emit 'None'."""
        loud = [TimeRange(start=10.0, end=20.0, score=8.0, metadata={"peak_time": None})]
        chat = [
            TimeRange(
                start=10.0,
                end=20.0,
                score=7.5,
                metadata={"sample_messages": [{"message": None, "text": "POG"}, {"message": None}]},
            )
        ]
        loud_str = scorer._format_loudness_data(loud)
        chat_str = scorer._format_chat_data(chat)
        assert "peak: 15.0s" in loud_str
        assert "'POG'" in chat_str
        assert "None" not in chat_str

    def test_relative_offset_shift_does_not_overshift_out_of_window_timestamps(
        self, scorer: OllamaScorer
    ):
        """M2: on chunk 3 (offset=1320s) any range ending before the offset is chunk-relative and shifts."""
        payload = [
            {"start": 20.0, "end": 75.0, "score": 8.0, "title": "Relative", "reason": "r"},
            {"start": 900.0, "end": 950.0, "score": 8.0, "title": "Hallucinated", "reason": "r"},
        ]
        highlights = scorer._parse_response(json.dumps(payload), chunk_offset=1320.0)
        assert len(highlights) == 2
        assert highlights[0].start == 1340.0
        assert highlights[0].end == 1395.0
        assert highlights[1].start == 2220.0
        assert highlights[1].end == 2270.0

    def test_rejects_nan_infinity_bool_and_negative_timestamps(self, scorer: OllamaScorer):
        """NaN, Infinity, booleans, and negative starts are skipped; bad scores become 0.0."""
        raw = (
            '['
            '{"start": NaN, "end": 50.0, "score": 8, "title": "A", "reason": "r"},'
            '{"start": 10.0, "end": Infinity, "score": 8, "title": "B", "reason": "r"},'
            '{"start": -5.0, "end": 30.0, "score": 8, "title": "C", "reason": "r"},'
            '{"start": true, "end": 30.0, "score": 8, "title": "D", "reason": "r"},'
            '{"start": 10.0, "end": 50.0, "score": NaN, "title": "Valid", "reason": "r"},'
            '{"start": 60.0, "end": 100.0, "score": true, "title": "BoolScore", "reason": "r"}'
            ']'
        )
        highlights = scorer._parse_response(raw, chunk_offset=0.0)
        assert len(highlights) == 2
        assert highlights[0].title == "Valid"
        assert highlights[0].score == 0.0
        assert highlights[1].title == "BoolScore"
        assert highlights[1].score == 0.0

    def test_parses_mm_ss_and_hh_mm_ss_timestamp_strings(self, scorer: OllamaScorer):
        """String timestamps like '02:15', '01:02:30', and '120.5s' are parsed accurately."""
        payload = [
            {
                "start": "02:15",
                "end": "03:00",
                "score": 8.5,
                "title": "Formatted Time",
                "reason": "MM:SS string timestamps",
                "sub_clips": [
                    {"start": "135.0s", "end": "02:30", "label": "Setup"},
                    {"start": "02:40", "end": "03:00", "label": "Payoff"},
                ],
            },
            {
                "start": "01:02:30",
                "end": "01:03:15",
                "score": 9.0,
                "title": "Hour Timestamp",
                "reason": "HH:MM:SS string timestamps",
            },
        ]
        highlights = scorer._parse_response(json.dumps(payload), chunk_offset=0.0)
        assert len(highlights) == 2
        assert highlights[0].start == 135.0
        assert highlights[0].end == 180.0
        assert highlights[0].sub_clips[0]["start"] == 135.0
        assert highlights[0].sub_clips[0]["end"] == 150.0
        assert highlights[1].start == 3750.0
        assert highlights[1].end == 3795.0

    def test_hashtags_honor_game_and_streamer_flags_and_deduplicate(self, tmp_path: Path):
        """include_game_tag / include_streamer_tag append missing tags and deduplicate case-insensitively."""
        caption_json = json.dumps(
            {
                "caption": "Wild getaway clip!",
                "hashtags": ["#Viral", "viral", None, "#GamingClips"],
            }
        )
        cfg = Config(user_config_path=tmp_path / "nonexistent.yaml")
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.generate.side_effect = [
                GenerateResponse(response="Hook Title"),
                GenerateResponse(response=caption_json),
            ]
            scorer = OllamaScorer(cfg)
            h = _make_highlight(
                10.0, 50.0, 8.0,
                narration_intro="intro", narration_outro="outro", context_badge="🔥 Badge",
            )
            meta = scorer.generate_metadata(h, streamer="xQc", game="GTA 6", source_url="https://youtu.be/1")

        assert meta.hashtags == ["Viral", "GamingClips", "GTA6", "xQc"]

    def test_raw_or_disabled_transformation_skips_narrative_backfill(self, tmp_path: Path):
        """When transformation.enabled=false, transformation_mode is 'raw' and narrative_writer is skipped."""
        user_cfg = tmp_path / "config.yaml"
        user_cfg.write_text("transformation:\n  enabled: false\n  mode: 'narrative'\n", encoding="utf-8")
        cfg = Config(user_config_path=user_cfg)
        caption_json = json.dumps({"caption": "Clean clip!", "hashtags": ["GTA6", "Tim"]})

        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.generate.side_effect = [
                GenerateResponse(response="Tim reacts to Vice City"),
                GenerateResponse(response=caption_json),
            ]
            scorer = OllamaScorer(cfg)
            h = _make_highlight(10.0, 50.0, 8.5, reason="Pure gameplay clip")
            meta = scorer.generate_metadata(h, streamer="Tim", game="GTA 6", source_url="https://youtu.be/2")

        assert mock_client.generate.call_count == 2  # hook + caption only, no narrative call
        assert meta.transformation_mode == "raw"
        assert meta.narration_intro == ""
        assert meta.narration_outro == ""
        assert meta.context_badge == ""

    def test_narrative_and_caption_retries_reinforce_json_instruction(self, config: Config):
        """Retry attempts in generate_narrative and caption generation append strict JSON reminder."""
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.generate.side_effect = [
                {"response": "bad json"},
                GenerateResponse(
                    response=json.dumps(
                        {
                            "narration_intro": "Watch closely here.",
                            "narration_outro": "Was that timing unreal or what?",
                            "context_badge": "🔥 Timing",
                        }
                    )
                ),
            ]
            scorer = OllamaScorer(config)
            h = _make_highlight(10.0, 50.0, 8.0, reason="Crazy timing")
            intro, outro, badge = scorer.generate_narrative(h, "Tim", "GTA 6")

        assert intro == "Watch closely here."
        assert outro == "Was that timing unreal or what?"
        assert badge == "🔥 Timing"
        retry_prompt = mock_client.generate.call_args_list[1].kwargs["prompt"]
        assert "IMPORTANT: Return ONLY a valid JSON object" in retry_prompt

    def test_chunk_transcript_includes_timestamps_no_explosion_and_no_tail_dup(
        self, config: Config, long_transcript: TranscriptionResult
    ):
        """Chunks contain [start - end] timestamps, stay bounded at chunk_minutes=1, and skip redundant tail."""
        scorer = _StubScorer(config)

        # 12-min transcript should yield exactly 1 chunk (no redundant trailing [660, 720])
        twelve_min = TranscriptionResult(
            segments=long_transcript.segments[:12], full_text="twelve min"
        )
        chunks_12 = scorer._chunk_transcript(twelve_min, chunk_minutes=12)
        assert len(chunks_12) == 1
        assert "[0.0s - 60.0s]" in chunks_12[0].text

        # chunk_minutes=1 on 48-min transcript must NOT explode to ~2880 chunks (step=45s -> ~64 chunks)
        chunks_1 = scorer._chunk_transcript(long_transcript, chunk_minutes=1)
        assert 40 <= len(chunks_1) <= 80

        with pytest.raises(ValueError, match="chunk_minutes"):
            scorer._chunk_transcript(long_transcript, chunk_minutes=0)

    def test_score_and_select_enforces_clip_bounds_scopes_transcript_and_max_clips_override(
        self, config: Config, long_transcript: TranscriptionResult
    ):
        """Widens short moments, rejects ranges past the hard ceiling, keeps stitched beats, scopes transcript, honors max_clips."""
        canned = [
            _make_highlight(10.0, 20.0, 9.5, title="Too Short"),          # 10s < 30s -> widened to 30s
            _make_highlight(100.0, 400.0, 9.5, title="Too Long"),         # 300s > 180s ceiling -> rejected
            _make_highlight(60.0, 105.0, 9.2, title="Valid Single"),      # 45s -> kept
            _make_highlight(
                300.0, 420.0, 9.0, title="Valid MultiBeat",
                sub_clips=[
                    {"start": 300.0, "end": 325.0, "label": "Setup"},
                    {"start": 385.0, "end": 415.0, "label": "Payoff"},
                ],  # stitched = 55s (within 30..90) -> kept
            ),
        ]
        # Add 5 more valid highlights to test max_clips override > config default (5)
        for idx in range(5):
            base = 600.0 + idx * 120.0
            canned.append(_make_highlight(base, base + 40.0, 8.8, title=f"Extra {idx}"))

        scorer = _StubScorer(config, canned=canned)
        scorer.min_clip_seconds = 30
        selected = scorer.score_and_select(long_transcript, [], [], max_clips=7)
        titles = [h.title for h in selected]
        assert "Too Short" in titles
        short = next(h for h in selected if h.title == "Too Short")
        assert short.end - short.start == pytest.approx(30.0)
        assert "widened" in short.bounds_adjustment
        assert "Too Long" not in titles
        assert "Valid Single" in titles
        assert "Valid MultiBeat" in titles
        assert len(selected) == 7

        single = next(h for h in selected if h.title == "Valid Single")
        assert "Segment 1" in single.transcript_chunk or "word0" in single.transcript_chunk
        assert "Segment 40" not in single.transcript_chunk

    def test_validate_sub_clips_rejects_nan_bool_and_oversized_duration(self, config: Config):
        scorer = _StubScorer(config)
        assert scorer._validate_sub_clips(
            [{"start": float("nan"), "end": 20.0}, {"start": 30.0, "end": 45.0}],
            start=0.0, end=50.0,
        ) == []
        assert scorer._validate_sub_clips(
            [{"start": True, "end": 20.0}, {"start": 30.0, "end": 45.0}],
            start=0.0, end=50.0,
        ) == []
        assert scorer._validate_sub_clips(
            [{"start": 0.0, "end": 60.0}, {"start": 70.0, "end": 120.0}],
            start=0.0, end=120.0, max_duration=90.0,
        ) == []

    def test_peak_time_clamped_and_boundary_straddling_words_included(self, config: Config):
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 145.0, 8.5)
        spikes = [TimeRange(start=85.0, end=110.0, score=9.0, metadata={"peak_time": 88.0})]
        peak = scorer._extract_peak_time(h, spikes)
        assert 100.0 <= peak <= 145.0

        tr = TranscriptionResult(
            segments=[
                TranscriptSegment(
                    text="boundary word",
                    start=99.0,
                    end=101.0,
                    words=[
                        WordTimestamp(text="straddle", start=99.8, end=100.3, confidence=0.9),
                        WordTimestamp(text="inside", start=100.4, end=100.9, confidence=0.95),
                    ],
                )
            ],
            full_text="straddle inside",
        )
        words = scorer._extract_words_for_range(tr, 100.0, 145.0)
        assert [w.text for w in words] == ["straddle", "inside"]

    def test_ollama_think_false_late_chunk1_offset_hashtag_cap_and_explicit_raw_mode(
        self, tmp_path: Path
    ):
        user_cfg = tmp_path / "config.yaml"
        user_cfg.write_text("metadata:\n  hashtag_count: 4\n", encoding="utf-8")
        cfg = Config(user_config_path=user_cfg)

        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.generate.side_effect = [
                GenerateResponse(response="Hook Title"),
                GenerateResponse(
                    response=json.dumps(
                        {
                            "caption": "Cap!",
                            "hashtags": ["t1", "t2", "t3", "t4", "t5", "t6"],
                        }
                    )
                ),
            ]
            scorer = OllamaScorer(cfg)

            # M2: [650, 700] ends after chunk_offset=660 so it is treated as absolute (unchanged)
            late_payload = json.dumps(
                [{"start": 650.0, "end": 700.0, "score": 8.5, "title": "Late", "reason": "r"}]
            )
            parsed = scorer._parse_response(late_payload, chunk_offset=660.0)
            assert parsed[0].start == 650.0
            assert parsed[0].end == 700.0

            # Explicit mode='raw' skips narrative_writer LLM call and clears narration/badge fields
            h = _make_highlight(
                100.0, 145.0, 9.0,
                narration_intro="Should be cleared",
                narration_outro="Should be cleared",
                context_badge="🔥 Badge",
            )
            meta = scorer.generate_metadata(
                h, streamer="xQc", game="GTA 6", source_url="https://youtu.be/1", mode="raw"
            )
            assert mock_client.generate.call_count == 2
            assert mock_client.generate.call_args_list[0].kwargs["think"] is False
            assert meta.transformation_mode == "raw"
            assert meta.narration_intro == ""
            assert meta.narration_outro == ""
            assert meta.context_badge == ""
            # Hashtags capped to 4 while preserving required GTA6 & xQc tags
            assert len(meta.hashtags) == 4
            assert "GTA6" in meta.hashtags
            assert "xQc" in meta.hashtags

    def test_zero_llm_weight_nan_chunk_minutes_transcript_range_and_subclip_normalization(
        self, config: Config
    ):
        scorer = _StubScorer(config)

        # 1. w_llm == 0.0 with no active signals returns 0.0 (does not leak raw LLM score or 8.5 floor)
        scorer.weights = {"llm_score": 0.0, "loudness": 0.5, "chat_velocity": 0.5}
        h = _make_highlight(10.0, 50.0, 9.5)
        assert scorer._compute_weighted_score(h, [], []) == 0.0

        # 2. _extract_transcript_for_range does not fall through to full seg.text when seg.words don't overlap
        tr = TranscriptionResult(
            segments=[
                TranscriptSegment(
                    text="OUTSIDE SEGMENT TEXT",
                    start=95.0,
                    end=105.0,
                    words=[WordTimestamp(text="early", start=95.0, end=99.0, confidence=0.9)],
                )
            ],
            full_text="OUTSIDE SEGMENT TEXT",
        )
        assert scorer._extract_transcript_for_range(tr, 100.0, 145.0) == ""

        # 3. _chunk_transcript rejects NaN / Inf chunk_minutes and overlap_seconds
        with pytest.raises(ValueError, match="chunk_minutes"):
            scorer._chunk_transcript(tr, chunk_minutes=float("nan"))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="overlap_seconds"):
            scorer._chunk_transcript(tr, chunk_minutes=1, overlap_seconds=float("inf"))

        # 4. _validate_sub_clips normalizes numeric string timestamps to floats
        beats = scorer._validate_sub_clips(
            [
                {"start": "10.0", "end": "20.0", "label": "setup"},
                {"start": "30.0", "end": "45.0", "label": "payoff"},
            ],
            start=0.0,
            end=50.0,
        )
        assert len(beats) == 2
        assert isinstance(beats[0]["start"], float) and beats[0]["start"] == 10.0
        assert isinstance(beats[1]["end"], float) and beats[1]["end"] == 45.0

    def test_ollama_chunk1_relative_subclips_all_shift_and_invalid_mode_raises(
        self, config: Config
    ):
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            # First call raises TypeError for unexpected 'think', second succeeds
            mock_client.generate.side_effect = [
                TypeError("generate() got an unexpected keyword argument 'think'"),
                GenerateResponse(response="Hook Title"),
            ]
            scorer = OllamaScorer(config)

            # Parent [600, 700] on Chunk 1 (offset=660) with beat 2 starting at 670 >= 660:
            # BOTH beats must shift by +660 to stay aligned with parent [1260, 1360].
            payload = json.dumps(
                [
                    {
                        "start": 600.0,
                        "end": 700.0,
                        "score": 9.0,
                        "title": "Late Story",
                        "reason": "great setup and payoff",
                        "sub_clips": [
                            {"start": 600.0, "end": 620.0, "label": "Setup"},
                            {"start": 670.0, "end": 700.0, "label": "Payoff"},
                        ],
                    }
                ]
            )
            parsed = scorer._parse_response(payload, chunk_offset=660.0)
            assert parsed[0].start == 600.0
            assert parsed[0].end == 700.0
            assert parsed[0].sub_clips == [
                {"start": 600.0, "end": 620.0, "label": "Setup"},
                {"start": 670.0, "end": 700.0, "label": "Payoff"},
            ]
            assert len(
                Scorer._validate_sub_clips(parsed[0].sub_clips, parsed[0].start, parsed[0].end)
            ) == 2

            # Invalid explicit mode raises ValueError
            with pytest.raises(ValueError, match="Invalid transformation mode"):
                scorer.generate_metadata(
                    parsed[0],
                    streamer="xQc",
                    game="GTA 6",
                    source_url="https://youtu.be/1",
                    mode="turbo",
                )
            assert mock_client.generate.call_count == 2
            assert "think" not in mock_client.generate.call_args_list[1].kwargs

    def test_extract_chat_burst_handles_null_non_dict_and_blank_messages(
        self, config: Config
    ):
        """_extract_chat_burst tolerates None/non-list sample_messages, non-dict entries, and null/blank text."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 160.0, 9.0)
        spikes = [
            TimeRange(
                start=105.0,
                end=115.0,
                score=7.0,
                signal_type=SignalType.CHAT_VELOCITY,
                metadata={"sample_messages": None},
            ),
            TimeRange(
                start=115.0,
                end=125.0,
                score=8.0,
                signal_type=SignalType.CHAT_VELOCITY,
                metadata={"sample_messages": "corrupt_not_a_list"},
            ),
            TimeRange(
                start=125.0,
                end=140.0,
                score=9.0,
                signal_type=SignalType.CHAT_VELOCITY,
                metadata={
                    "sample_messages": [
                        None,
                        "raw_string",
                        {"message": ""},
                        {"message": "   "},
                        {"message": None, "text": "POG"},
                        {"message": None, "text": "LMAO"},
                        {"message": "POG"},  # duplicate of text-fallback entry
                    ]
                },
            ),
        ]
        burst = scorer._extract_chat_burst(h, spikes)
        assert burst == [
            {"message": None, "text": "POG"},
            {"message": None, "text": "LMAO"},
        ]

    def test_extract_peak_time_ignores_bool_and_overlap_ignores_inf_score(
        self, config: Config
    ):
        """Boolean peak_time falls back to midpoint; non-finite spike score is ignored."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 160.0, 9.0)
        spikes = [
            TimeRange(
                start=110.0,
                end=130.0,
                score=float("inf"),
                signal_type=SignalType.LOUDNESS,
                metadata={"peak_time": True},
            )
        ]
        assert scorer._extract_peak_time(h, spikes) == 130.0
        assert scorer._signal_overlap_score(h, spikes) == 0.0

    def test_score_and_select_short_sub_clips_falls_back_to_single_cut(
        self, config: Config
    ):
        """Valid 60s highlight with 16s stitched sub_clips (<30s min_clip_seconds) falls back to single cut."""
        highlights = [
            # 60s parent (valid) + 16s stitched sub_clips (< 30s) -> keep as single cut with sub_clips=[]
            _make_highlight(
                100.0,
                160.0,
                9.5,
                title="Short Beats Valid Parent",
                sub_clips=[
                    {"start": 100.0, "end": 108.0, "label": "Setup"},
                    {"start": 145.0, "end": 153.0, "label": "Payoff"},
                ],
            ),
            # 20s parent (too short) + 16s stitched sub_clips (< 30s) -> single cut widened to 30s
            _make_highlight(
                200.0,
                220.0,
                9.5,
                title="Too Short Overall",
                sub_clips=[
                    {"start": 200.0, "end": 208.0, "label": "Setup"},
                    {"start": 212.0, "end": 220.0, "label": "Payoff"},
                ],
            ),
            # 120s parent + 45s stitched sub_clips (within 30..90) -> kept as multi-beat story clip
            _make_highlight(
                300.0,
                420.0,
                9.0,
                title="Valid Story Beats",
                sub_clips=[
                    {"start": 300.0, "end": 320.0, "label": "Setup"},
                    {"start": 395.0, "end": 420.0, "label": "Payoff"},
                ],
            ),
        ]
        tr = TranscriptionResult(
            segments=[TranscriptSegment(text="clip talk", start=0.0, end=500.0)],
            full_text="clip talk",
        )
        scorer = _StubScorer(config, canned=highlights)
        scorer.min_clip_seconds = 30
        selected = scorer.score_and_select(tr, [], [])
        by_title = {h.title: h for h in selected}
        assert "Short Beats Valid Parent" in by_title
        assert by_title["Short Beats Valid Parent"].sub_clips == []
        widened = by_title["Too Short Overall"]
        assert widened.sub_clips == []
        assert widened.end - widened.start == pytest.approx(30.0)
        assert "Valid Story Beats" in by_title
        assert len(by_title["Valid Story Beats"].sub_clips) == 2

    def test_ollama_parse_response_clears_sub_clips_on_any_malformed_beat(
        self, config: Config
    ):
        """A 3-beat sub_clips list with one malformed beat clears to [] instead of silently dropping it."""
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client"):
            scorer = OllamaScorer(config)
            payload = json.dumps(
                [
                    {
                        "start": 100.0,
                        "end": 160.0,
                        "score": 9.0,
                        "title": "Corrupt Middle Beat",
                        "reason": "test",
                        "sub_clips": [
                            {"start": 100.0, "end": 115.0, "label": "Setup"},
                            {"start": "not_a_time", "end": 135.0, "label": "Mid"},
                            {"start": 140.0, "end": 160.0, "label": "Payoff"},
                        ],
                    }
                ]
            )
            parsed = scorer._parse_response(payload)
            assert len(parsed) == 1
            assert parsed[0].sub_clips == []





class TestMusicPenaltyAffectsSelection:
    """Copyright-risk penalty must run before ranking so it changes what is selected."""

    def test_music_flagged_top_pick_is_displaced_by_clean_candidate(
        self, config: Config, long_transcript: TranscriptionResult, tmp_path: Path
    ) -> None:
        loud = _make_highlight(60.0, 100.0, 9.0, title="Music")
        clean = _make_highlight(600.0, 640.0, 8.6, title="Clean")
        scorer = _StubScorer(config, canned=[loud, clean])
        scorer.max_clips = 1
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFF")

        def fake_detect(self, audio_path, h):
            return ("HIGH_MUSIC_DETECTED", 0.9) if h.title == "Music" else ("CLEAN", 0.0)

        with (
            patch(
                "streamclipper.analysis.copyright_risk.CopyrightRiskDetector.detect_highlight",
                fake_detect,
            ),
            patch.object(scorer, "_compute_weighted_score", side_effect=lambda h, *_: h.score),
        ):
            selected = scorer.score_and_select(long_transcript, [], [], audio_path=audio)

        assert [h.title for h in selected] == ["Clean"]

    def test_detector_receives_word_timestamps_before_detection(
        self, config: Config, long_transcript: TranscriptionResult, tmp_path: Path
    ) -> None:
        """H1: word_timestamps must be populated before detect_highlight runs."""
        h = _make_highlight(60.0, 100.0, 9.0, title="Speech")
        expected = Scorer._extract_words_for_range(long_transcript, 60.0, 100.0)
        assert expected, "fixture must contain words in the highlight range"
        scorer = _StubScorer(config, canned=[h])
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFF")
        seen: list[int] = []

        def fake_detect(self, audio_path, hl):
            seen.append(len(hl.word_timestamps))
            return ("CLEAN", 0.0)

        with (
            patch(
                "streamclipper.analysis.copyright_risk.CopyrightRiskDetector.detect_highlight",
                fake_detect,
            ),
            patch.object(scorer, "_compute_weighted_score", side_effect=lambda h, *_: h.score),
        ):
            scorer.score_and_select(long_transcript, [], [], audio_path=audio)

        assert seen == [len(expected)]


# ===========================================================================
# Reviewed-defect regression tests (X1, M1, M2, M3, M10, L7, L23, G3, G4)
# ===========================================================================

def _set_cache_dir(config: Config, path: Path) -> None:
    config._data.setdefault("general", {})["cache_dir"] = str(path)


class TestReviewedDefects:
    def test_X1_dedupe_runs_after_copyright_penalty(
        self, config: Config, long_transcript: TranscriptionResult, tmp_path: Path
    ) -> None:
        a = _make_highlight(60.0, 100.0, 8.6, title="A")
        b = _make_highlight(65.0, 105.0, 8.0, title="B")
        scorer = _StubScorer(config, canned=[a, b])
        _set_cache_dir(config, tmp_path)
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFF")

        def fake_detect(self, audio_path, h):
            return ("HIGH_MUSIC_DETECTED", 0.9) if h.title == "A" else ("CLEAN", 0.0)

        with (
            patch(
                "streamclipper.analysis.copyright_risk.CopyrightRiskDetector.detect_highlight",
                fake_detect,
            ),
            patch.object(scorer, "_compute_weighted_score", side_effect=lambda h, *_: h.score),
        ):
            selected = scorer.score_and_select(long_transcript, [], [], audio_path=audio)
        assert [h.title for h in selected] == ["B"]

    def test_M1_chat_burst_filters_range_sorts_dedupes_caps(self) -> None:
        h = _make_highlight(100.0, 140.0, 9.0)
        msgs = [
            {"message": "late", "offset_seconds": 500.0},
            {"message": "c", "offset_seconds": 130.0},
            {"message": "a", "offset_seconds": 105.0},
            {"message": "a", "offset_seconds": 106.0},
            {"message": "b", "time": 110.0},
            {"message": "d", "offset": 120.0},
            {"message": "e", "offset_seconds": 125.0},
            {"message": "f", "offset_seconds": 135.0},
            {"message": "nooffset"},
        ]
        spike = TimeRange(start=100.0, end=140.0, score=8.0, signal_type=SignalType.CHAT_VELOCITY,
                          metadata={"sample_messages": msgs})
        burst = [m["message"] for m in Scorer._extract_chat_burst(h, [spike])]
        assert burst == ["a", "b", "d", "e", "c"]
        assert "late" not in burst

    def test_M2_chunk_offset_shift_only_when_end_before_offset(self, config: Config) -> None:
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client"):
            scorer = OllamaScorer(config)
        payload = json.dumps([
            {"start": 655.0, "end": 700.0, "score": 8,
             "sub_clips": [{"start": 655.0, "end": 665.0}, {"start": 680.0, "end": 700.0}]},
            {"start": 10.0, "end": 40.0, "score": 8,
             "sub_clips": [{"start": 10.0, "end": 20.0}, {"start": 30.0, "end": 40.0}]},
        ])
        hs = scorer._parse_response(payload, chunk_offset=660.0)
        assert (hs[0].start, hs[0].end) == (655.0, 700.0)
        assert hs[0].sub_clips[0]["start"] == 655.0
        assert (hs[1].start, hs[1].end) == (670.0, 700.0)
        assert hs[1].sub_clips[0]["start"] == 670.0 and hs[1].sub_clips[1]["end"] == 700.0

    def test_M3_timeout_retries_once_then_returns_empty(self, config: Config) -> None:
        import httpx

        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
            client = cls.return_value
            client.generate.side_effect = httpx.ReadTimeout("timed out")
            scorer = OllamaScorer(config)
            chunk = TranscriptChunk(start=0.0, end=60.0, text="hello")
            assert scorer._score_chunk(chunk, [], []) == []
            assert client.generate.call_count == 2

    def test_M3_connection_error_still_raises(self, config: Config) -> None:
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
            cls.return_value.generate.side_effect = ConnectionError("refused")
            scorer = OllamaScorer(config)
            with pytest.raises(OllamaConnectionError):
                scorer._score_chunk(TranscriptChunk(0.0, 60.0, "hello"), [], [])
            assert cls.return_value.generate.call_count == 1

    def test_M10_scoring_detection_error_marks_unknown_no_penalty(
        self, config: Config, long_transcript: TranscriptionResult, tmp_path: Path, caplog
    ) -> None:
        h = _make_highlight(60.0, 100.0, 9.0)
        scorer = _StubScorer(config, canned=[h])
        _set_cache_dir(config, tmp_path)
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFF")

        def boom(self, audio_path, hl):
            raise RuntimeError("decode fail")

        with (
            patch("streamclipper.analysis.copyright_risk.CopyrightRiskDetector.detect_highlight", boom),
            patch.object(scorer, "_compute_weighted_score", side_effect=lambda h, *_: h.score),
            caplog.at_level("WARNING"),
        ):
            selected = scorer.score_and_select(long_transcript, [], [], audio_path=audio)
        assert len(selected) == 1
        assert selected[0].music_risk == "UNKNOWN"
        assert selected[0].music_confidence == 0.0
        assert selected[0].final_score == 9.0
        assert any("Copyright risk detection failed" in r.message for r in caplog.records)

    def test_L7_debug_log_when_highlight_overlaps_no_spikes(self, config: Config, caplog) -> None:
        scorer = _StubScorer(config)
        h = _make_highlight(500.0, 540.0, 7.0)
        far = TimeRange(start=0.0, end=5.0, score=9.0, signal_type=SignalType.LOUDNESS)
        with caplog.at_level("DEBUG", logger="streamclipper.scoring.scorer"):
            scorer._compute_weighted_score(h, [far], [])
        assert any("overlaps no loudness/chat spikes" in r.message for r in caplog.records)

    def test_L23_provider_warning_and_narration_style_in_prompt(self, config: Config, caplog) -> None:
        config._data["llm"]["provider"] = "openai"
        config._data.setdefault("transformation", {})["narration_style"] = "deadpan"
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
            with caplog.at_level("WARNING"):
                scorer = OllamaScorer(config)
            assert any("llm.provider" in r.message for r in caplog.records)
            cls.return_value.generate.return_value = {
                "response": '{"narration_intro": "i", "narration_outro": "o", "context_badge": "b"}'
            }
            scorer.generate_narrative(_make_highlight(0.0, 40.0, 8.0), "Streamer", "GTA 6")
            prompt = cls.return_value.generate.call_args.kwargs["prompt"]
        assert "deadpan" in prompt

    def test_G3_checkpoint_skips_completed_chunks_on_rerun(
        self, config: Config, long_transcript: TranscriptionResult, tmp_path: Path
    ) -> None:
        _set_cache_dir(config, tmp_path)
        calls: list[float] = []

        class _Flaky(_StubScorer):
            fail = True

            def _score_chunk(self, chunk, loudness, chat):
                calls.append(chunk.start)
                if self.fail and len(calls) == 2:
                    raise RuntimeError("boom")
                return [_make_highlight(chunk.start + 5.0, chunk.start + 45.0, 9.0, title=f"c{chunk.start}")]

        scorer = _Flaky(config)
        with pytest.raises(RuntimeError):
            scorer.score_and_select(long_transcript, [], [])
        assert len(list(tmp_path.glob("scoring_ckpt_*.json"))) == 1
        first_chunk_start = calls[0]

        calls.clear()
        scorer.fail = False
        scorer.score_and_select(long_transcript, [], [])
        assert first_chunk_start not in calls
        assert list(tmp_path.glob("scoring_ckpt_*.json")) == []

    def test_G4_empty_chunk_text_not_sent_to_scorer(self, config: Config, tmp_path: Path) -> None:
        _set_cache_dir(config, tmp_path)
        seen: list[str] = []

        class _Rec(_StubScorer):
            def _chunk_transcript(self, transcript, chunk_minutes=None, overlap_seconds=60.0):
                return [TranscriptChunk(0.0, 60.0, "   "), TranscriptChunk(60.0, 120.0, "real")]

            def _score_chunk(self, chunk, loudness, chat):
                seen.append(chunk.text)
                return []

        tr = TranscriptionResult(
            segments=[TranscriptSegment(text="x", start=0.0, end=120.0, words=[])], full_text="x"
        )
        _Rec(config).score_and_select(tr, [], [])
        assert seen == ["real"]



class TestFitClipRange:
    """Clip length limits widen/trim instead of rejecting; only the hard ceiling rejects."""

    def test_short_range_widened_setup_weighted(self) -> None:
        fit = fit_clip_range(100.0, 110.0, min_s=30, max_s=90, hard_max_s=180)
        assert not fit.rejected
        assert fit.end - fit.start == pytest.approx(30.0)
        assert 100.0 - fit.start > fit.end - 110.0  # more setup than aftermath

    def test_widen_clamped_at_stream_start_and_end(self) -> None:
        at_start = fit_clip_range(0.0, 5.0, min_s=30, max_s=90, hard_max_s=180)
        assert at_start.start == 0.0 and at_start.end == pytest.approx(30.0)
        at_end = fit_clip_range(
            995.0, 1000.0, min_s=30, max_s=90, hard_max_s=180, stream_duration=1000.0
        )
        assert at_end.end == 1000.0 and at_end.start == pytest.approx(970.0)

    def test_long_range_trimmed_around_peak(self) -> None:
        fit = fit_clip_range(0.0, 150.0, min_s=30, max_s=90, hard_max_s=180, peak_time=140.0)
        assert fit.end - fit.start == pytest.approx(90.0)
        assert fit.end == pytest.approx(150.0)  # window stays inside the original range
        assert fit.start <= 140.0 <= fit.end
        mid = fit_clip_range(0.0, 150.0, min_s=30, max_s=90, hard_max_s=180)
        assert (mid.start, mid.end) == (pytest.approx(30.0), pytest.approx(120.0))

    def test_beyond_hard_ceiling_rejected(self) -> None:
        fit = fit_clip_range(0.0, 7200.0, min_s=30, max_s=90, hard_max_s=180)
        assert fit.rejected and "hard_max_clip_seconds" in fit.adjustment

    def test_adjust_disabled_restores_strict_reject(self) -> None:
        assert fit_clip_range(0.0, 10.0, min_s=30, max_s=90, hard_max_s=180, adjust=False).rejected
        assert fit_clip_range(0.0, 120.0, min_s=30, max_s=90, hard_max_s=180, adjust=False).rejected
        ok = fit_clip_range(0.0, 45.0, min_s=30, max_s=90, hard_max_s=180, adjust=False)
        assert not ok.rejected and ok.adjustment == ""


# ===========================================================================
# Output Quality v3 — Batch D
# ===========================================================================


class TestBatchD:
    def test_D1_models_callout_roundtrip_and_defaults(self) -> None:
        h = Highlight.from_dict({"start": 1, "end": 2, "score": 5})
        assert h.callout == ""
        h.callout = "Wait for the wall"
        assert Highlight.from_dict(h.to_dict()).callout == "Wait for the wall"
        m = ClipMetadata.from_dict(
            {"clip_start": 0, "clip_end": 1, "highlight_score": 1, "callout": "x y z w"}
        )
        assert m.callout == "x y z w"
        assert ClipMetadata.from_dict(
            {"clip_start": 0, "clip_end": 1, "highlight_score": 1}
        ).callout == ""

    def test_D2_skip_intro_drops_early_chunks_and_highlights(
        self, config: Config, long_transcript: TranscriptionResult
    ) -> None:
        config._data["analysis"]["skip_intro_minutes"] = 5
        scorer = _StubScorer(
            config, canned=[_make_highlight(110.0, 160.0, 9.0)]
        )
        for chunk in scorer._chunk_transcript(long_transcript):
            for line in chunk.text.splitlines():
                assert float(line.split("[")[1].split("s")[0]) >= 300.0
        assert scorer.score_and_select(long_transcript, [], []) == []

    def test_D3_prompt_rejects_intro_and_allows_reaction(self) -> None:
        txt = (Path(__file__).parent.parent / "src/streamclipper/prompts/highlight_scorer.txt").read_text(encoding="utf-8")
        assert "reaction" in txt and "countdown" in txt.lower() and "cutscene" in txt.lower()
        from streamclipper.scoring.ollama_scorer import VALID_MOMENT_TYPES
        assert "reaction" in VALID_MOMENT_TYPES

    def test_D4_banned_phrase_triggers_retry(self, config: Config) -> None:
        bad = json.dumps({"narration_intro": "The chat went wild here.", "narration_outro": "Ok.", "context_badge": "B", "callout": "a b c d"})
        good = json.dumps({"narration_intro": "He eyes the red truck.", "narration_outro": "Honestly, I'd have bailed. Would you?", "context_badge": "Good", "callout": "Watch the red truck"})
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
            cls.return_value.generate.side_effect = [GenerateResponse(response=bad), GenerateResponse(response=good)]
            scorer = OllamaScorer(config)
            h = _make_highlight(0, 30, 8.0)
            intro, outro, badge = scorer.generate_narrative(h, "Tim", "GTA 6")
        assert cls.return_value.generate.call_count == 2
        assert intro == "He eyes the red truck." and h.callout == "Watch the red truck"

    def test_D4_avoid_phrases_and_placeholders(self, config: Config) -> None:
        reply = json.dumps({"narration_intro": "Tim hates this truck.", "narration_outro": "Fair.", "context_badge": "B"})
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
            cls.return_value.generate.side_effect = [GenerateResponse(response=reply)] * 3
            scorer = OllamaScorer(config)
            h = _make_highlight(0, 30, 8.0)
            h.transcript_chunk = "that purple sedan again"
            scorer.generate_narrative(h, "Tim", "GTA 6", avoid_phrases=["hates this truck"], hook_style="question")
        assert cls.return_value.generate.call_count == 3  # 1 + max 2 retries
        sent = cls.return_value.generate.call_args_list[0].kwargs["prompt"]
        assert "purple sedan" in sent and "hates this truck" in sent and "question" in sent

    def test_D5_generate_metadata_hook_style_and_callout(self, config: Config) -> None:
        narrative = json.dumps({"narration_intro": "Tim eyes the exit.", "narration_outro": "Brave. Dumb. Agree?", "context_badge": "B", "callout": "Here comes the truck"})
        caption = json.dumps({"caption": "c", "hashtags": ["a"]})
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
            cls.return_value.generate.side_effect = [
                GenerateResponse(response="Tim bails?"),
                GenerateResponse(response=narrative),
                GenerateResponse(response=caption),
            ]
            scorer = OllamaScorer(config)
            meta = scorer.generate_metadata(
                _make_highlight(0, 30, 8.0), "Tim", "GTA 6", "http://x",
                hook_style="question", avoid_phrases=[],
            )
        assert "question" in cls.return_value.generate.call_args_list[0].kwargs["prompt"]
        assert meta.callout == "Here comes the truck"

    def test_D6_generate_countdown_lines(self, config: Config) -> None:
        hs = [_make_highlight(0, 20, 8.0, title=f"T{i}") for i in range(3)]
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
            cls.return_value.generate.return_value = GenerateResponse(
                response=json.dumps({"lines": ["third", "second", "first"]})
            )
            scorer = OllamaScorer(config)
            assert scorer.generate_countdown_lines(hs, "Tim", "GTA 6") == ["third", "second", "first"]
        with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
            cls.return_value.generate.return_value = GenerateResponse(response="junk")
            scorer = OllamaScorer(config)
            assert len(scorer.generate_countdown_lines(hs, "Tim", "GTA 6")) == 3
