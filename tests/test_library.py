"""Tests for topic video source library, SQLite FTS5 index, ingest, and CLI (T1)."""
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from rich.console import Console

import streamclipper.cli as cli_mod
from streamclipper.cache import StepCache
from streamclipper.cli import cli
from streamclipper.config import Config
from streamclipper.library.index import LibraryIndex, _escape_fts5_query
from streamclipper.library.ingest import add_source, index_cached_states
from streamclipper.library.models import LibrarySource, SearchHit
from streamclipper.models import (
    PipelineState,
    PipelineStatus,
    Platform,
    SignalType,
    StreamInfo,
    TimeRange,
    TranscriptSegment,
    TranscriptionResult,
)


def _make_config(tmp_path: Path, **extra: dict) -> Config:
    """Create a test Config pointing cache and temp to tmp_path."""
    cache_dir = tmp_path / ".cache"
    temp_dir = tmp_path / "temp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "general": {
            "cache_dir": str(cache_dir),
            "temp_dir": str(temp_dir),
        },
        "library": {
            "db_path": str(cache_dir / "library.sqlite"),
            "window_seconds": 30,
            "window_step_seconds": 20,
            "max_chat_chars_per_window": 600,
            "heat_boost": 0.3,
        },
    }
    data.update(extra)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return Config(user_config_path=cfg_file)


def _build_test_state(
    url: str = "https://youtube.com/watch?v=test1234",
    title: str = "Test Stream",
    channel: str = "TestStreamer",
    game: str = "GTA6",
    duration: int = 120,
    language: str = "en",
    segments: list[TranscriptSegment] | None = None,
    loudness_spikes: list[TimeRange] | None = None,
    heatmap_spikes: list[TimeRange] | None = None,
) -> PipelineState:
    """Helper to build PipelineState in code without network or GPU."""
    segs = segments or []
    full_text = " ".join(s.text for s in segs)
    return PipelineState(
        stream_info=StreamInfo(
            url=url,
            title=title,
            platform=Platform.YOUTUBE,
            streamer=channel,
            game=game,
            duration_seconds=duration,
            video_id=url.split("v=")[-1] if "v=" in url else "test1234",
        ),
        status=PipelineStatus.SCORING,
        transcript=TranscriptionResult(
            segments=segs,
            full_text=full_text,
            language=language,
        ),
        loudness_spikes=loudness_spikes or [],
        heatmap_spikes=heatmap_spikes or [],
        analysis_complete=True,
    )


def test_T1_schema_creation_and_version(tmp_path: Path) -> None:
    """Verify SQLite database, tables, triggers, and schema_version are initialized."""
    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    assert index.get_schema_version() == 1
    assert index.db_path.is_file()

    with index._get_connection() as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]
        assert "sources" in tables
        assert "segments" in tables
        assert "segments_fts" in tables
        assert "frame_notes" in tables
        assert "frame_notes_fts" in tables
        assert "schema_version" in tables


def test_T1_windowing_midpoint_capping_signals_and_skips(tmp_path: Path) -> None:
    """Test windowing: midpoint text assignment, chat capping, heat, loudness, chat_rate, empty window skips."""
    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    segments = [
        # Midpoint = 10.0 -> falls into [0, 30)
        TranscriptSegment(start=5.0, end=15.0, text="Entering the diamond casino vault", words=[]),
        # Midpoint = 25.0 -> falls into [0, 30) and [20, 50)
        TranscriptSegment(start=24.0, end=26.0, text="Secret entrance discovered", words=[]),
        # Midpoint = 105.0 -> falls into [100, 130) and [80, 110)
        TranscriptSegment(start=100.0, end=110.0, text="Escape chopper waiting on the roof", words=[]),
    ]

    # Create chat DataFrame
    long_msg = "hype " * 200  # 1000 characters
    chat_df = pd.DataFrame(
        [
            {"offset_seconds": 6.0, "author": "alice", "message": "hello casino"},
            {"offset_seconds": 12.0, "author": "bob", "message": long_msg},
            {"offset_seconds": 25.0, "author": "carol", "message": "vault is open"},
        ]
    )

    heatmap_spikes = [
        TimeRange(start=8.0, end=12.0, score=8.5, signal_type=SignalType.HEATMAP, metadata={"heat": 0.85})
    ]
    loudness_spikes = [
        TimeRange(start=7.0, end=11.0, score=9.2, signal_type=SignalType.LOUDNESS)
    ]

    state = _build_test_state(
        duration=130,
        segments=segments,
        loudness_spikes=loudness_spikes,
        heatmap_spikes=heatmap_spikes,
    )

    source = index.index_state(state, kind="stream", chat_df=chat_df)
    assert source.segment_count > 0

    with index._get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM segments WHERE source_id = ? ORDER BY start;",
            (source.source_id,),
        ).fetchall()

    # Window [0, 30]: has Seg 1 & 2 midpoints, chat, heat, loudness
    w0 = next(r for r in rows if r["start"] == 0.0)
    assert "Entering the diamond casino vault" in w0["text"]
    assert "Secret entrance discovered" in w0["text"]
    assert "Escape chopper" not in w0["text"]
    # Chat capped at 600
    assert len(w0["chat_text"]) <= 600
    assert "hello casino" in w0["chat_text"]
    assert pytest.approx(w0["heat"], 0.01) == 0.85
    assert pytest.approx(w0["loudness"], 0.01) == 9.2
    # 3 messages in 30s = 6 messages/min
    assert pytest.approx(w0["chat_rate"], 0.01) == 6.0

    # Window [40, 70]: no text, no chat, no heat, no loudness -> MUST BE SKIPPED
    starts = [r["start"] for r in rows]
    assert 40.0 not in starts
    assert 60.0 not in starts

    # Window [80, 110]: midpoint 85 is present
    w80 = next(r for r in rows if r["start"] == 80.0)
    assert "Escape chopper waiting on the roof" in w80["text"]


def test_T1_idempotent_reindex(tmp_path: Path) -> None:
    """Re-indexing a source must replace its rows cleanly without duplication."""
    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    state = _build_test_state(
        segments=[TranscriptSegment(start=5.0, end=15.0, text="First pass on the casino", words=[])]
    )

    src1 = index.index_state(state, kind="stream")
    initial_added_at = src1.added_at
    assert src1.segment_count >= 1

    # Re-index with updated text
    state.transcript.segments = [
        TranscriptSegment(start=5.0, end=15.0, text="Second pass updated casino info", words=[])
    ]
    src2 = index.index_state(state, kind="stream")

    # Source ID and added_at must be preserved
    assert src2.source_id == src1.source_id
    assert src2.added_at == initial_added_at

    # Only one source in table
    sources = index.list_sources()
    assert len(sources) == 1
    assert sources[0].source_id == src1.source_id

    # FTS5 search matches only the new text
    hits_old = index.search("First")
    assert len(hits_old) == 0
    hits_new = index.search("Second")
    assert len(hits_new) == 1


def test_T1_search_ranking_and_heat_boost(tmp_path: Path) -> None:
    """Term matches beat non-matches; heat boost reorders ties."""
    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    # State with two distinct non-overlapping casino moments
    state = _build_test_state(
        url="https://youtube.com/watch?v=ranktest",
        duration=200,
        segments=[
            # Moment A: high heat (heat = 0.95)
            TranscriptSegment(start=10.0, end=20.0, text="Breaking into the casino vault", words=[]),
            # Moment B: low heat (heat = 0.10)
            TranscriptSegment(start=120.0, end=130.0, text="Walking past the casino front door", words=[]),
            # Moment C: no casino keyword
            TranscriptSegment(start=160.0, end=170.0, text="Flying an airplane over the mountains", words=[]),
        ],
        heatmap_spikes=[
            TimeRange(start=10.0, end=25.0, score=9.5, signal_type=SignalType.HEATMAP, metadata={"heat": 0.95}),
            TimeRange(start=120.0, end=135.0, score=1.0, signal_type=SignalType.HEATMAP, metadata={"heat": 0.10}),
        ],
    )
    index.index_state(state)

    hits = index.search("casino")
    # Term match beats non-match: Moment C must not be in hits
    assert len(hits) == 2
    for h in hits:
        assert "airplane" not in h.snippet

    # Moment A (high heat) must rank higher than Moment B (low heat)
    assert hits[0].start < 60.0
    assert hits[0].heat > hits[1].heat
    assert hits[0].score > hits[1].score


def test_T1_hinglish_and_devanagari_search(tmp_path: Path) -> None:
    """Devanagari and Roman Hinglish keywords are indexed and searchable via unicode61."""
    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    state = _build_test_state(
        url="https://youtube.com/watch?v=multilang",
        duration=120,
        segments=[
            TranscriptSegment(start=5.0, end=15.0, text="यह एक बहुत बड़ा कैसीनो है भाई", words=[]),
            TranscriptSegment(start=45.0, end=55.0, text="bhai yeh secret location bilkul mast hai", words=[]),
        ],
    )
    index.index_state(state)

    dev_hits = index.search("कैसीनो")
    assert len(dev_hits) >= 1
    assert "कैसीनो" in dev_hits[0].snippet or "यह" in dev_hits[0].snippet

    hing_hits = index.search("bilkul mast")
    assert len(hing_hits) >= 1
    assert "secret location" in hing_hits[0].snippet


def test_T1_search_filters(tmp_path: Path) -> None:
    """Search correctly filters by game, kind, and channel."""
    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    s1 = _build_test_state(
        url="https://youtube.com/watch?v=src1",
        channel="AlphaGamer",
        game="GTA6",
        segments=[TranscriptSegment(start=5.0, end=15.0, text="Special docks heist secret", words=[])],
    )
    s2 = _build_test_state(
        url="https://youtube.com/watch?v=src2",
        channel="BetaPlayer",
        game="GTA6",
        segments=[TranscriptSegment(start=5.0, end=15.0, text="Special docks heist secret", words=[])],
    )
    s3 = _build_test_state(
        url="https://youtube.com/watch?v=src3",
        channel="AlphaGamer",
        game="Cyberpunk",
        segments=[TranscriptSegment(start=5.0, end=15.0, text="Special docks heist secret", words=[])],
    )

    index.index_state(s1, kind="stream")
    index.index_state(s2, kind="gameplay")
    index.index_state(s3, kind="official")

    # Filter by game
    hits_gta = index.search("docks", game="GTA6")
    assert len(hits_gta) == 2
    for h in hits_gta:
        assert h.source_id in (s1.stream_info.video_id, StepCache(config).get_cache_key(s1.stream_info.url), StepCache(config).get_cache_key(s2.stream_info.url))

    # Filter by kind
    hits_gameplay = index.search("docks", kinds=["gameplay"])
    assert len(hits_gameplay) == 1
    assert hits_gameplay[0].kind == "gameplay"

    # Filter by channel
    hits_beta = index.search("docks", channels=["BetaPlayer"])
    assert len(hits_beta) == 1
    assert hits_beta[0].channel == "BetaPlayer"


def test_T1_overlapping_hits_merged(tmp_path: Path) -> None:
    """Overlapping search hits within the same source merge into one continuous span."""
    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    # Segments that produce overlapping windows matching 'secret'
    # Window [0, 30] and Window [20, 50]
    segments = [
        TranscriptSegment(start=10.0, end=15.0, text="Find the secret casino room", words=[]),
        TranscriptSegment(start=25.0, end=30.0, text="The secret vault is unlockable here", words=[]),
        # Distinct far-away window [100, 130]
        TranscriptSegment(start=110.0, end=115.0, text="Another secret item across the map", words=[]),
    ]

    state = _build_test_state(duration=150, segments=segments)
    index.index_state(state)

    hits = index.search("secret")
    # Must merge the overlapping [0, 30] and [20, 50] into [0, 50]
    assert len(hits) == 2
    merged_hit = next(h for h in hits if h.start == 0.0)
    assert merged_hit.end >= 50.0
    assert "casino room" in merged_hit.snippet or "vault" in merged_hit.snippet


def test_T1_sql_injection_safe(tmp_path: Path) -> None:
    """FTS5 search gracefully handles SQL injection patterns and syntax-breaking chars."""
    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    state = _build_test_state(
        segments=[TranscriptSegment(start=5.0, end=15.0, text="Safe normal casino text", words=[])]
    )
    index.index_state(state)

    # Various adversarial / injection patterns
    test_queries = [
        "\"casino' OR 1=1 --\"",
        "'; DROP TABLE segments; --",
        "\"",
        "\"\"\"",
        "'",
        "AND OR NOT",
        "***",
        "  ",
        "((()))",
    ]

    for q in test_queries:
        hits = index.search(q)
        assert isinstance(hits, list)

    # Verify tables still exist
    sources = index.list_sources()
    assert len(sources) == 1


def test_T1_index_cached_states(tmp_path: Path) -> None:
    """index_cached_states picks up states saved with StepCache and ignores incomplete ones."""
    config = _make_config(tmp_path)
    cache = StepCache(config)

    # State 1: Complete with transcript
    s1 = _build_test_state(
        url="https://youtube.com/watch?v=cached1",
        game="GTA6",
        title="Cached Stream 1",
        segments=[TranscriptSegment(start=5.0, end=15.0, text="Secret casino passage", words=[])],
    )
    cache.save_state(s1)

    # State 2: Complete with transcript, different game
    s2 = _build_test_state(
        url="https://youtube.com/watch?v=cached2",
        game="RDR2",
        title="Cached Stream 2",
        segments=[TranscriptSegment(start=5.0, end=15.0, text="Cowboy heist in Saint Denis", words=[])],
    )
    cache.save_state(s2)

    # State 3: Pending/no transcript
    s3 = PipelineState(
        stream_info=StreamInfo(
            url="https://youtube.com/watch?v=cached3",
            title="Incomplete",
            platform=Platform.YOUTUBE,
            streamer="IncompleteStreamer",
            game="GTA6",
            duration_seconds=0,
        ),
        status=PipelineStatus.PENDING,
        transcript=None,
    )
    cache.save_state(s3)

    # Index all cached states
    indexed = index_cached_states(config)
    assert len(indexed) == 2
    urls = [src.url for src in indexed]
    assert s1.stream_info.url in urls
    assert s2.stream_info.url in urls
    assert s3.stream_info.url not in urls

    # Test filtering by game
    indexed_gta = index_cached_states(config, game="GTA6")
    assert len(indexed_gta) == 1
    assert indexed_gta[0].url == s1.stream_info.url


@pytest.mark.asyncio
async def test_T1_prepare_stream_stops_before_scoring(tmp_path: Path) -> None:
    """prepare_stream runs download, transcribe, and analyze, then stops before scoring."""
    from streamclipper.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    call_order: list[str] = []

    async def mock_download(state: PipelineState) -> PipelineState:
        call_order.append("download")
        state.audio_path = tmp_path / "audio.wav"
        state.audio_path.write_bytes(b"RIFFdummywav")
        state.status = PipelineStatus.TRANSCRIBING
        return state

    async def mock_transcribe(state: PipelineState) -> PipelineState:
        call_order.append("transcribe")
        state.transcript = TranscriptionResult(
            segments=[TranscriptSegment(start=0.0, end=10.0, text="Hello world", words=[])],
            full_text="Hello world",
        )
        state.status = PipelineStatus.ANALYZING
        return state

    async def mock_analyze(state: PipelineState) -> PipelineState:
        call_order.append("analyze")
        state.loudness_spikes = []
        state.chat_spikes = []
        state.heatmap_spikes = []
        state.analysis_complete = True
        state.status = PipelineStatus.SCORING
        return state

    mock_score = AsyncMock(side_effect=RuntimeError("Scoring should not be called"))
    mock_assemble = AsyncMock(side_effect=RuntimeError("Assemble should not be called"))

    with patch.object(pipeline, "_step_download_audio", side_effect=mock_download), \
         patch.object(pipeline, "_step_transcribe", side_effect=mock_transcribe), \
         patch.object(pipeline, "_step_analyze", side_effect=mock_analyze), \
         patch.object(pipeline, "_step_score", mock_score), \
         patch.object(pipeline, "_step_assemble", mock_assemble):

        state = await pipeline.prepare_stream("https://youtube.com/watch?v=prepare123", game="GTA6")

    assert call_order == ["download", "transcribe", "analyze"]
    mock_score.assert_not_called()
    mock_assemble.assert_not_called()
    assert state.status == PipelineStatus.SCORING
    assert state.transcript is not None


def test_T1_cli_library_list_and_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI commands library list and library search render rich tables properly."""
    config = _make_config(tmp_path)
    cfg_file = tmp_path / "config.yaml"
    monkeypatch.setattr(cli_mod, "console", Console(width=160, force_terminal=False))

    index = LibraryIndex(config)
    state = _build_test_state(
        url="https://youtube.com/watch?v=clitest",
        channel="CliStreamer",
        game="GTA6",
        title="Grand Theft Auto VI Exploration",
        duration=180,
        segments=[TranscriptSegment(start=10.0, end=25.0, text="Exploring the mysterious casino vaults", words=[])],
        heatmap_spikes=[TimeRange(start=10.0, end=20.0, score=8.0, signal_type=SignalType.HEATMAP, metadata={"heat": 0.8})],
    )
    index.index_state(state)

    runner = CliRunner()

    # 1. library list
    res_list = runner.invoke(cli, ["--config", str(cfg_file), "library", "list"])
    assert res_list.exit_code == 0
    assert "CliStreamer" in res_list.output
    assert "GTA" in res_list.output or "Exploration" in res_list.output

    # 2. library search
    res_search = runner.invoke(cli, ["--config", str(cfg_file), "library", "search", "casino"])
    assert res_search.exit_code == 0
    assert "CliStreamer" in res_search.output
    assert "casino" in res_search.output


def test_T1_cli_library_add_and_remove(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI commands library add and library remove function via CliRunner."""
    config = _make_config(tmp_path)
    cfg_file = tmp_path / "config.yaml"
    monkeypatch.setattr(cli_mod, "console", Console(width=160, force_terminal=False))

    mock_source = LibrarySource(
        source_id="testsourceid123",
        url="https://youtube.com/watch?v=addtest",
        title="Added Video",
        channel="NewChannel",
        game="GTA6",
        kind="stream",
        segment_count=5,
    )

    runner = CliRunner()

    # 1. library add (mocking add_source)
    with patch("streamclipper.library.add_source", AsyncMock(return_value=mock_source)):
        res_add = runner.invoke(
            cli,
            [
                "--config",
                str(cfg_file),
                "library",
                "add",
                "https://youtube.com/watch?v=addtest",
                "--game",
                "GTA6",
            ],
        )
        assert res_add.exit_code == 0
        assert "Indexed source" in res_add.output
        assert "testsourceid123" in res_add.output

    # 2. library remove
    index = LibraryIndex(config)
    state = _build_test_state(url="https://youtube.com/watch?v=removetest")
    src = index.index_state(state)

    res_remove = runner.invoke(
        cli,
        ["--config", str(cfg_file), "library", "remove", src.source_id],
    )
    assert res_remove.exit_code == 0
    assert f"Removed source {src.source_id}" in res_remove.output
    assert index.get_source(src.source_id) is None


@pytest.mark.asyncio
async def test_T1_edge_cases_and_error_handling(tmp_path: Path) -> None:
    """Explicitly test empty inputs, boundary values, error paths, and string filters."""
    from streamclipper.pipeline import Pipeline

    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    # 1. Non-existent source
    assert index.get_source("non_existent_id") is None
    assert index.remove_source("non_existent_id") is False

    # 2. Empty query and whitespace
    assert index.search("") == []
    assert index.search("   ") == []
    assert index.search("---") == []

    # 3. Non-matching game filter
    assert index.list_sources(game="NonExistentGame") == []

    # 4. Zero duration, empty segments state
    empty_state = _build_test_state(duration=0, segments=[])
    src_empty = index.index_state(empty_state)
    assert src_empty.segment_count == 0

    # 5. Pipeline prepare_stream invalid URL
    pipeline = Pipeline(config)
    with pytest.raises(ValueError, match="Invalid URL"):
        await pipeline.prepare_stream("")

    with pytest.raises(ValueError, match="Invalid URL"):
        await pipeline.prepare_stream("   ")

    # 6. prepare_stream blocked streamer
    config._data["discovery"] = {"blocked_streamers": ["BlockedStreamer"]}
    pipeline_blocked = Pipeline(config)
    blocked_state = await pipeline_blocked.prepare_stream(
        "https://youtube.com/watch?v=blocked",
        streamer="BlockedStreamer",
    )
    assert blocked_state.status == PipelineStatus.PENDING

    # 7. Kinds and channels as str instead of list
    state_valid = _build_test_state(
        url="https://youtube.com/watch?v=valid1",
        channel="StreamerX",
        segments=[TranscriptSegment(start=1.0, end=10.0, text="Found the casino", words=[])],
        heatmap_spikes=[TimeRange(start=1.0, end=5.0, score=5.0, signal_type=SignalType.HEATMAP, metadata=None)],
    )
    index.index_state(state_valid, kind="official")
    hits_str_filter = index.search(
        "casino",
        kinds="official",
        channels="StreamerX",
        limit=1,
    )
    assert len(hits_str_filter) == 1
    assert hits_str_filter[0].kind == "official"
    assert hits_str_filter[0].channel == "StreamerX"

    # 8. Negative search limit
    assert index.search("casino", limit=-5) == []

    # 9. Window step zero clamp (prevent infinite loop)
    cfg_zero_step = _make_config(tmp_path / "zero_step", library={"window_step_seconds": 0, "window_seconds": 0})
    idx_zero_step = LibraryIndex(cfg_zero_step)
    src_clamped = idx_zero_step.index_state(state_valid)
    assert src_clamped.segment_count > 0

    # 10. Float-string parsing in from_dict
    src_float_str = LibrarySource.from_dict({"duration_seconds": "120.5", "segment_count": "5.0"})
    assert src_float_str.duration_seconds == 120
    assert src_float_str.segment_count == 5


def test_T1_connection_closed_and_file_unlocked(tmp_path: Path) -> None:
    """Verify SQLite connections are closed and no file locks linger on Windows."""
    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    state = _build_test_state(
        segments=[TranscriptSegment(start=1.0, end=10.0, text="Checking connection lock", words=[])]
    )
    src = index.index_state(state)
    _ = index.get_source(src.source_id)
    _ = index.list_sources()
    _ = index.search("connection")
    _ = index.get_schema_version()

    # On Windows, if a connection is not closed, unlinking or renaming raises WinError 32
    assert index.db_path.is_file()
    # Attempting to open for exclusive read/write or unlink should succeed if connection is closed
    with open(index.db_path, "a+b") as f:
        f.seek(0)
        assert len(f.read(16)) > 0


def test_T1_snippet_cleaning_and_no_triple_ellipses(tmp_path: Path) -> None:
    """Snippet merging across adjacent windows should not produce triple ellipses or unstripped dots."""
    config = _make_config(tmp_path)
    index = LibraryIndex(config)

    # Windows [0, 30] and [20, 50] overlapping
    segments = [
        TranscriptSegment(start=5.0, end=15.0, text="First part of casino secret...", words=[]),
        TranscriptSegment(start=25.0, end=35.0, text="...second part of casino secret.", words=[]),
    ]
    state = _build_test_state(duration=60, segments=segments)
    index.index_state(state)

    hits = index.search("casino")
    assert len(hits) == 1
    snippet = hits[0].snippet
    assert "... ... ..." not in snippet
    assert "......." not in snippet
    assert "\n" not in snippet
    assert "\r" not in snippet


def test_T1_cli_duration_hours_formatting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI library list formats multi-hour streams as H:MM:SS."""
    config = _make_config(tmp_path)
    cfg_file = tmp_path / "config.yaml"
    monkeypatch.setattr(cli_mod, "console", Console(width=160, force_terminal=False))

    index = LibraryIndex(config)
    # Stream with 7500s duration (2h 05m 00s)
    state = _build_test_state(
        url="https://youtube.com/watch?v=longvod",
        duration=7500,
        segments=[TranscriptSegment(start=10.0, end=20.0, text="Long vod heist moment", words=[])]
    )
    index.index_state(state)

    runner = CliRunner()
    res = runner.invoke(cli, ["--config", str(cfg_file), "library", "list"])
    assert res.exit_code == 0
    assert "2:05:00" in res.output


def test_T1_kind_case_normalization(tmp_path: Path) -> None:
    """Kind is normalized to lowercase in models and ingest."""
    src = LibrarySource.from_dict({"source_id": "s1", "kind": "Official"})
    assert src.kind == "official"

    hit = SearchHit.from_dict({"source_id": "s1", "kind": "GAMEPLAY"})
    assert hit.kind == "gameplay"

