"""Unit tests for Package 3.1 — Scorer Interface, YPP Adaptive Narrative Models & Prompt Templates."""
import math
from pathlib import Path
from typing import Any

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
from streamclipper.scoring.scorer import Scorer, TranscriptChunk
from streamclipper.config import Config


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
    return Config(user_config_path=tmp_path / "nonexistent.yaml")


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
        """With no overlapping signals, final = 0.50 * llm_score."""
        scorer = _StubScorer(config)
        h = _make_highlight(100.0, 200.0, 10.0)
        final = scorer._compute_weighted_score(h, [], [])
        assert abs(final - 5.0) < 0.001

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
