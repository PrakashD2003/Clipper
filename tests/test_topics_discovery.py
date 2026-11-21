"""Tests for topic video discovery, ledger, evidence evaluation, and library MCP tools (T3)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import yaml
from click.testing import CliRunner

from streamclipper.cli import cli
from streamclipper.config import Config
from streamclipper.library.index import LibraryIndex
from streamclipper.library.models import LibrarySource
from streamclipper.mcp_server.server import mcp
from streamclipper.models import (
    PipelineState,
    PipelineStatus,
    Platform,
    StreamInfo,
    TranscriptSegment,
    TranscriptionResult,
)
from streamclipper.topics.discover import (
    _extract_json,
    _extract_snippet_window,
    _fallback_term_clusters,
    _format_time,
    _slugify,
    evaluate_topic,
    suggest_topics_local,
    topic_material,
)
from streamclipper.topics.ledger import TopicLedger, _jaccard, _tokenize
from streamclipper.topics.models import EvidenceRef, TopicCandidate


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
            "frames": {
                "note_weight": 0.8,
            },
        },
        "topics": {
            "min_evidence": 6,
            "min_sources": 2,
            "max_evidence": 12,
            "similarity_threshold": 0.5,
        },
    }
    data.update(extra)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return Config(user_config_path=cfg_file)


def _build_state(
    url: str,
    channel: str,
    game: str,
    duration: int,
    segments: list[TranscriptSegment],
) -> PipelineState:
    full_text = " ".join(s.text for s in segments)
    video_id = url.split("v=")[-1] if "v=" in url else "vid"
    return PipelineState(
        stream_info=StreamInfo(
            url=url,
            title=f"Stream from {channel}",
            platform=Platform.YOUTUBE,
            streamer=channel,
            game=game,
            duration_seconds=duration,
            video_id=video_id,
        ),
        status=PipelineStatus.SCORING,
        transcript=TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language="en",
        ),
        loudness_spikes=[],
        heatmap_spikes=[],
        analysis_complete=True,
    )


@pytest.fixture
def sample_library(tmp_path: Path) -> tuple[Config, LibraryIndex]:
    """Populate a test SQLite library with 3 distinct sources (EN, Hinglish, Gameplay)."""
    cfg = _make_config(tmp_path)
    index = LibraryIndex(cfg)

    # Source 1: English streamer (distinct segments with "Vice City docks cargo ship")
    segs1 = [
        TranscriptSegment(start=10.0, end=20.0, text="bro look at this chat guys lol Vice City docks cargo ship secret entrance"),
        TranscriptSegment(start=150.0, end=160.0, text="Entering Vice City docks waterfront area now"),
        TranscriptSegment(start=300.0, end=310.0, text="Secret cargo container at Vice City docks"),
        TranscriptSegment(start=450.0, end=460.0, text="Security cameras near Vice City docks warehouse"),
        TranscriptSegment(start=600.0, end=610.0, text="Escape route behind Vice City docks pier"),
    ]
    s1 = _build_state("https://youtube.com/watch?v=src1", "AlphaGamer", "GTA6", 800, segs1)
    index.index_state(s1, kind="stream")

    # Source 2: Hinglish streamer (distinct segments with "Vice City docks cargo ship" + Hinglish filler)
    segs2 = [
        TranscriptSegment(start=20.0, end=30.0, text="bhai Vice City docks pe cargo ship hai bilkul secret mast"),
        TranscriptSegment(start=180.0, end=190.0, text="arre dekho Vice City docks mein underground tunnel mil gaya"),
        TranscriptSegment(start=350.0, end=360.0, text="yeh cargo ship ka crane check karo Vice City docks"),
        TranscriptSegment(start=500.0, end=510.0, text="police boats patrol kar rahi hai near Vice City docks"),
    ]
    s2 = _build_state("https://youtube.com/watch?v=src2", "DesiPlayer", "GTA6", 800, segs2)
    index.index_state(s2, kind="stream")

    # Source 3: Gameplay video with visual frame notes
    segs3 = [
        TranscriptSegment(start=10.0, end=20.0, text="Ocean drive coastline patrol"),
        TranscriptSegment(start=500.0, end=510.0, text="Vice City docks shipping facility harbor view"),
    ]
    s3 = _build_state("https://youtube.com/watch?v=src3", "SilentWalker", "GTA6", 800, segs3)
    src3_obj = index.index_state(s3, kind="gameplay")

    # Add frame notes to source 3 (distinct non-overlapping timestamps)
    index.add_frame_notes(
        src3_obj.source_id,
        [
            {"t": 60.0, "note": "Waterfront cranes and cargo ship at Vice City docks"},
            {"t": 250.0, "note": "Cargo containers stacked along the harbor docks"},
            {"t": 650.0, "note": "Vice City docks security gate and watchtower"},
        ],
    )

    return cfg, index


def test_T3_ledger_atomic_and_similarity(tmp_path: Path) -> None:
    """Verify TopicLedger atomic recording, case-insensitive entries, last_shape, and token Jaccard similarity."""
    cfg = _make_config(tmp_path)
    ledger = TopicLedger(cfg)

    # Initially empty
    assert ledger.entries("GTA6") == []
    assert ledger.last_shape("GTA6") == ""
    assert ledger.find_similar("Any Title", ["kw1"], "GTA6") is None

    # Record first topic
    e1 = ledger.record(
        topic_id="hidden-docks-secrets",
        title="Every Hidden Detail in Vice City Docks",
        game="GTA6",
        keywords=["vice city", "docks", "cargo ship", "secrets"],
        shape="numbered",
        plan_id="plan-001",
    )
    assert e1["topic_id"] == "hidden-docks-secrets"
    assert ledger.path.is_file()

    # Verify atomic file contents
    raw = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert len(raw["entries"]) == 1
    assert raw["entries"][0]["title"] == "Every Hidden Detail in Vice City Docks"

    # Last shape
    assert ledger.last_shape("GTA6") == "numbered"

    # Record second topic for different game
    ledger.record(
        topic_id="rdr-mountains",
        title="Mount Hagen Mystery",
        game="RDR2",
        keywords=["mountains", "snow"],
        shape="story",
    )
    assert ledger.last_shape("RDR2") == "story"
    assert ledger.last_shape("GTA6") == "numbered"

    # Game filter check
    gta_entries = ledger.entries("gta6")
    assert len(gta_entries) == 1
    assert gta_entries[0]["topic_id"] == "hidden-docks-secrets"

    # find_similar: Jaccard >= 0.5 match
    # Similar keywords and title
    sim = ledger.find_similar(
        "Hidden Details in Vice City Docks Area",
        ["vice city", "docks", "cargo ship"],
        game="GTA6",
    )
    assert sim is not None
    assert sim["topic_id"] == "hidden-docks-secrets"

    # Unrelated topic: Jaccard < 0.5 -> None
    unrelated = ledger.find_similar(
        "Complete Guide to Motorcycle Customization",
        ["bikes", "customs", "tuning"],
        game="GTA6",
    )
    assert unrelated is None

    # Different game should not match
    no_game_match = ledger.find_similar(
        "Every Hidden Detail in Vice City Docks",
        ["vice city", "docks"],
        game="RDR2",
    )
    assert no_game_match is None


def test_T3_topic_material_stopwords_and_snippets(sample_library: tuple[Config, LibraryIndex]) -> None:
    """Verify topic_material extracts salient terms, excludes EN+Hinglish filler, and includes snippets and stats."""
    cfg, _ = sample_library

    mat = topic_material(cfg, "GTA6", max_terms=20)
    assert mat["game"] == "GTA6"

    stats = mat["stats"]
    assert stats["sources"] == 3
    assert stats["channels"] == 3
    assert "stream" in stats["kinds"]
    assert "gameplay" in stats["kinds"]
    assert stats["total_hours"] > 0.0

    terms = mat["terms"]
    assert len(terms) > 0

    term_words = [t["term"] for t in terms]

    # Salient terms must include core topic words
    assert any("dock" in t or "vice city" in t or "cargo" in t for t in term_words)

    # Stopwords and filler MUST NOT appear as extracted terms
    banned_words = {"bro", "bhai", "chat", "guys", "lol", "hai", "pe", "dekho", "look", "this"}
    for tw in term_words:
        assert tw not in banned_words, f"Filler word '{tw}' was not excluded!"

    # Check snippet formatting
    found_snippets = False
    for t_item in terms:
        if t_item["snippets"]:
            found_snippets = True
            snip = t_item["snippets"][0]
            assert "channel" in snip
            assert "time" in snip
            assert ":" in snip["time"]  # MM:SS format
            assert "text" in snip
            assert "formatted" in snip
            break
    assert found_snippets, "Expected at least one term to have formatted snippets"


def test_T3_evaluate_topic_viability_and_sources(sample_library: tuple[Config, LibraryIndex]) -> None:
    """Verify evaluate_topic counts distinct sources/channels, applies viability thresholds, and clamps max_evidence."""
    cfg, _ = sample_library

    # 1. Viable candidate with search terms across multiple sources
    candidate = evaluate_topic(
        cfg,
        "GTA6",
        "Every Hidden Detail in the Vice City Docks",
        ["vice city", "docks", "cargo ship", "waterfront"],
    )

    assert candidate.topic_id == "every-hidden-detail-in-the-vice-city-docks"
    assert candidate.title == "Every Hidden Detail in the Vice City Docks"
    assert candidate.game == "GTA6"
    assert candidate.source_count >= 2
    assert candidate.channel_count >= 2
    assert len(candidate.evidence) >= 6
    assert candidate.is_viable is True
    assert candidate.already_made is False

    # Check evidence structure
    for ev in candidate.evidence:
        assert isinstance(ev, EvidenceRef)
        assert ev.source_id
        assert ev.end >= ev.start
        assert ev.snippet

    # 2. Non-viable candidate: term only in 1 source or very low evidence
    weak_cand = evaluate_topic(
        cfg,
        "GTA6",
        "Ocean Drive Coastline Mystery",
        ["ocean drive", "coastline"],
    )
    # Only in source 3, fewer than 6 evidence spans
    assert weak_cand.source_count <= 1 or len(weak_cand.evidence) < 6
    assert weak_cand.is_viable is False


def test_T3_already_made_flagging(sample_library: tuple[Config, LibraryIndex]) -> None:
    """Verify evaluate_topic sets already_made=True and similar_to from ledger."""
    cfg, _ = sample_library
    ledger = TopicLedger(cfg)

    ledger.record(
        topic_id="vice-city-docks-breakdown",
        title="Every Hidden Detail in the Vice City Docks",
        game="GTA6",
        keywords=["vice city", "docks", "cargo ship"],
        shape="numbered",
    )

    # Evaluate identical/similar topic
    cand = evaluate_topic(
        cfg,
        "GTA6",
        "Every Hidden Secret in Vice City Docks",
        ["vice city", "docks", "cargo ship"],
    )
    assert cand.already_made is True
    assert cand.similar_to == "vice-city-docks-breakdown"

    # Evaluate distinct topic
    cand_new = evaluate_topic(
        cfg,
        "GTA6",
        "Police Helicopter AI Patrol Routes",
        ["helicopter", "patrol", "routes"],
    )
    assert cand_new.already_made is False
    assert cand_new.similar_to == ""


def test_T3_ollama_discovery_paths(sample_library: tuple[Config, LibraryIndex]) -> None:
    """Verify suggest_topics_local with mocked Ollama: valid JSON, fenced JSON, and fallback on error."""
    cfg, _ = sample_library

    # 1. Valid JSON from Ollama
    valid_response = {
        "message": {
            "content": json.dumps([
                {
                    "title": "Every Hidden Detail in the Vice City Docks",
                    "keywords": ["vice city", "docks", "cargo ship"],
                    "angle": "Investigating the waterfront secrets and cargo ships",
                }
            ])
        }
    }
    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = valid_response
        mock_client_cls.return_value = mock_instance

        candidates = suggest_topics_local(cfg, "GTA6", limit=5)
        assert len(candidates) >= 1
        assert candidates[0].title == "Every Hidden Detail in the Vice City Docks"
        assert candidates[0].is_viable is True

    # 2. Fenced markdown JSON from Ollama
    fenced_content = (
        "```json\n"
        + json.dumps([
            {
                "title": "Every Hidden Detail in the Vice City Docks",
                "keywords": ["vice city", "docks", "cargo ship"],
                "angle": "Waterfront mystery",
            }
        ])
        + "\n```"
    )
    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {"message": {"content": fenced_content}}
        mock_client_cls.return_value = mock_instance

        candidates = suggest_topics_local(cfg, "GTA6", limit=5)
        assert len(candidates) >= 1
        assert candidates[0].is_viable is True

    # 3. Garbage / failure from Ollama -> Fallback to term clusters
    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.side_effect = RuntimeError("Ollama connection failed")
        mock_client_cls.return_value = mock_instance

        # Should fall back cleanly without raising
        candidates = suggest_topics_local(cfg, "GTA6", limit=5)
        assert isinstance(candidates, list)
        for c in candidates:
            assert c.is_viable is True


def test_T3_fallback_term_clusters_unit() -> None:
    """Verify _fallback_term_clusters produces meaningful proposals from top terms."""
    terms = [
        {"term": "vice city", "sources": 3},
        {"term": "docks", "sources": 3},
        {"term": "cargo ship", "sources": 2},
        {"term": "security gate", "sources": 2},
    ]
    clusters = _fallback_term_clusters(terms)
    assert len(clusters) == 1
    c0 = clusters[0]
    assert "Vice City" in c0["title"]
    assert len(c0["keywords"]) == 4
    assert "angle" in c0


@pytest.mark.asyncio
async def test_T3_mcp_library_tools(sample_library: tuple[Config, LibraryIndex], monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MCP tools search_library, list_library, get_topic_material, record_topic_candidates, and suggest_topics."""
    cfg, index = sample_library

    # Patch Config loader in server
    monkeypatch.setattr("streamclipper.mcp_server.server.Config", lambda *a, **k: cfg)

    # 1. list_library
    res_list = await mcp.call_tool("list_library", {"game": "GTA6"})
    data_list = json.loads(res_list.content[0].text)
    assert data_list["total"] == 3
    assert len(data_list["sources"]) == 3

    # 2. search_library
    res_search = await mcp.call_tool("search_library", {"query": "docks", "game": "GTA6"})
    data_search = json.loads(res_search.content[0].text)
    assert data_search["total"] > 0
    assert any("dock" in h["snippet"].lower() for h in data_search["hits"])

    # 3. get_topic_material
    res_mat = await mcp.call_tool("get_topic_material", {"game": "GTA6", "max_terms": 10})
    data_mat = json.loads(res_mat.content[0].text)
    assert data_mat["game"] == "GTA6"
    assert "instructions" in data_mat
    assert "record_topic_candidates" in data_mat["instructions"]
    assert len(data_mat["terms"]) > 0

    # 4. record_topic_candidates
    candidates_payload = [
        {
            "title": "Every Hidden Detail in the Vice City Docks",
            "keywords": ["vice city", "docks", "cargo ship"],
            "angle": "Investigating waterfront docks",
        },
        {
            "title": "Random Street Driving",
            "keywords": ["random", "street"],
        },
    ]
    res_record = await mcp.call_tool(
        "record_topic_candidates",
        {"game": "GTA6", "candidates": json.dumps(candidates_payload)},
    )
    data_record = json.loads(res_record.content[0].text)
    assert data_record["count"] == 2
    cand0 = data_record["candidates"][0]
    # Viable candidate is first
    assert cand0["title"] == "Every Hidden Detail in the Vice City Docks"
    assert cand0["source_count"] >= 2
    assert len(cand0["evidence"]) >= 6

    # 5. suggest_topics with mocked Ollama
    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {
            "message": {
                "content": json.dumps([{
                    "title": "Every Hidden Detail in the Vice City Docks",
                    "keywords": ["vice city", "docks", "cargo ship"],
                }])
            }
        }
        mock_client_cls.return_value = mock_instance

        res_suggest = await mcp.call_tool("suggest_topics", {"game": "GTA6", "limit": 5})
        data_suggest = json.loads(res_suggest.content[0].text)
        assert data_suggest["game"] == "GTA6"
        assert len(data_suggest["candidates"]) >= 1


@pytest.mark.asyncio
async def test_T3_mcp_library_add_source_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify library_add_source starts a background job and validates source kind."""
    cfg = _make_config(tmp_path)
    monkeypatch.setattr("streamclipper.mcp_server.server.Config", lambda *a, **k: cfg)

    # Kind validation: invalid kind returns error
    res_bad = await mcp.call_tool(
        "library_add_source",
        {"url": "https://youtube.com/watch?v=123", "game": "GTA6", "kind": "podcast"},
    )
    err = json.loads(res_bad.content[0].text)
    assert "error" in err
    assert "Allowed kinds" in err["error"]

    # Valid kind starts job
    mock_source = LibrarySource(
        source_id="src_test",
        url="https://youtube.com/watch?v=valid123",
        title="Test Title",
        channel="TestChannel",
        game="GTA6",
        kind="stream",
    )
    with patch("streamclipper.library.ingest.add_source", new_callable=AsyncMock) as mock_add:
        mock_add.return_value = mock_source

        res_ok = await mcp.call_tool(
            "library_add_source",
            {"url": "https://youtube.com/watch?v=valid123", "game": "GTA6", "kind": "stream"},
        )
        data = json.loads(res_ok.content[0].text)
        assert data["status"] == "started"
        assert "job_id" in data


def test_T3_cli_topics_suggest_and_ledger(sample_library: tuple[Config, LibraryIndex], tmp_path: Path) -> None:
    """Verify CLI commands topics suggest and topics ledger render rich tables properly."""
    cfg, _ = sample_library
    cfg_file = tmp_path / "config.yaml"

    runner = CliRunner()

    # 1. Record an entry into ledger
    ledger = TopicLedger(cfg)
    ledger.record(
        topic_id="vice-city-docks",
        title="Vice City Docks Secrets",
        game="GTA6",
        keywords=["vice city", "docks"],
        shape="numbered",
        plan_id="plan-123",
    )

    # 2. topics ledger (option and positional)
    res_ledger = runner.invoke(cli, ["--config", str(cfg_file), "topics", "ledger", "--game", "GTA6"])
    assert res_ledger.exit_code == 0
    assert "vice-city-docks" in res_ledger.output
    assert "Vice City Docks Secrets" in res_ledger.output
    assert "numbered" in res_ledger.output

    res_ledger_pos = runner.invoke(cli, ["--config", str(cfg_file), "topics", "ledger", "GTA6"])
    assert res_ledger_pos.exit_code == 0
    assert "vice-city-docks" in res_ledger_pos.output

    # 3. topics suggest (mocking Ollama to return the viable topic, option and positional)
    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {
            "message": {
                "content": json.dumps([{
                    "title": "Every Hidden Detail in the Vice City Docks",
                    "keywords": ["vice city", "docks", "cargo ship"],
                }])
            }
        }
        mock_client_cls.return_value = mock_instance

        res_suggest = runner.invoke(cli, ["--config", str(cfg_file), "topics", "suggest", "--game", "GTA6"])
        assert res_suggest.exit_code == 0
        assert "Vice City Docks" in res_suggest.output
        assert "Suggested Topic Videos" in res_suggest.output

        res_suggest_pos = runner.invoke(cli, ["--config", str(cfg_file), "topics", "suggest", "GTA6"])
        assert res_suggest_pos.exit_code == 0
        assert "Vice City Docks" in res_suggest_pos.output


def test_T3_evidence_source_diversity_starvation(tmp_path: Path) -> None:
    """Verify evaluate_topic does not allow a single source to starve other sources of evidence slots."""
    cfg = _make_config(tmp_path)
    index = LibraryIndex(cfg)

    # Source 1: 12 distinct segments spaced apart (not merged) with high keyword density
    segs1 = [
        TranscriptSegment(start=float(i * 150), end=float((i * 150) + 15), text=f"Vice City docks container shipment {i}")
        for i in range(12)
    ]
    s1 = _build_state("https://youtube.com/watch?v=src1_long", "HighDensityStreamer", "GTA6", 2000, segs1)
    index.index_state(s1, kind="stream")

    # Source 2: 4 distinct segments spaced apart with valid evidence
    segs2 = [
        TranscriptSegment(start=float(i * 150), end=float((i * 150) + 15), text=f"Vice City docks patrol boat {i}")
        for i in range(4)
    ]
    s2 = _build_state("https://youtube.com/watch?v=src2_short", "SecondStreamer", "GTA6", 1000, segs2)
    index.index_state(s2, kind="stream")

    # Evaluate topic with max_evidence = 12
    cand = evaluate_topic(cfg, "GTA6", "Vice City Docks Investigation", ["vice city docks"])
    assert len(cand.evidence) <= 12
    # Both sources must be represented in the evidence
    represented_sources = {e.source_id for e in cand.evidence}
    assert len(represented_sources) >= 2, f"Source 2 was starved! Represented: {represented_sources}"
    assert cand.source_count >= 2
    assert cand.is_viable is True


def test_T3_ollama_conversational_and_bracketed_json(sample_library: tuple[Config, LibraryIndex]) -> None:
    """Verify suggest_topics_local parses conversational prefix LLM output without failing back."""
    cfg, _ = sample_library

    conversational_content = (
        "Here are the top curiosity-driven video proposals based on the library:\n\n"
        "[\n"
        "  {\n"
        '    "title": "Every Hidden Detail in the Vice City Docks",\n'
        '    "keywords": ["vice city", "docks", "cargo ship"],\n'
        '    "angle": "Comprehensive look at harbor secrets"\n'
        "  }\n"
        "]\n\n"
        "I hope these provide great video ideas!"
    )

    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {"message": {"content": conversational_content}}
        mock_client_cls.return_value = mock_instance

        candidates = suggest_topics_local(cfg, "GTA6", limit=5)
        assert len(candidates) >= 1
        assert candidates[0].title == "Every Hidden Detail in the Vice City Docks"
        assert candidates[0].description == "Comprehensive look at harbor secrets"


def test_T3_configurable_viability_and_string_keywords(tmp_path: Path) -> None:
    """Verify topics.min_evidence and topics.min_sources user config overrides are respected and string keywords handled."""
    # Configure low thresholds: min_evidence=3, min_sources=1
    cfg = _make_config(tmp_path, topics={"min_evidence": 3, "min_sources": 1, "max_evidence": 8, "similarity_threshold": 0.5})
    index = LibraryIndex(cfg)

    segs = [
        TranscriptSegment(start=10.0, end=20.0, text="secret docks location one"),
        TranscriptSegment(start=100.0, end=110.0, text="secret docks location two"),
        TranscriptSegment(start=200.0, end=210.0, text="secret docks location three"),
    ]
    s = _build_state("https://youtube.com/watch?v=solo_src", "SoloStreamer", "GTA6", 400, segs)
    index.index_state(s, kind="stream")

    # Pass keywords as comma-separated string
    cand = evaluate_topic(cfg, "GTA6", "Secret Docks Investigation", "secret docks, location")
    assert cand.keywords == ["secret docks", "location"]
    assert len(cand.evidence) == 3
    assert cand.source_count == 1
    # Under configured thresholds (min_evidence=3, min_sources=1), candidate should be viable!
    assert cand.is_viable is True
    # to_dict contains is_viable
    d = cand.to_dict()
    assert d["is_viable"] is True


def test_T3_ledger_rerecord_last_shape(tmp_path: Path) -> None:
    """Verify that updating an existing topic moves it to the most recent position so last_shape reflects it."""
    cfg = _make_config(tmp_path)
    ledger = TopicLedger(cfg)

    # Record topic A
    ledger.record(topic_id="topic-a", title="Topic A", game="GTA6", shape="numbered")
    assert ledger.last_shape("GTA6") == "numbered"

    # Record topic B
    ledger.record(topic_id="topic-b", title="Topic B", game="GTA6", shape="story")
    assert ledger.last_shape("GTA6") == "story"

    # Re-record topic A with countdown shape
    ledger.record(topic_id="topic-a", title="Topic A Updated", game="GTA6", shape="countdown")
    assert ledger.last_shape("GTA6") == "countdown"


def test_T3_topic_material_cooccurrence_and_snippets_centering(tmp_path: Path) -> None:
    """Verify topic_material records co-occurrences and centers snippets around terms in long segments."""
    cfg = _make_config(tmp_path)
    index = LibraryIndex(cfg)

    # Segment with term deep into a 200+ char text
    long_prefix = "We are currently driving down the southern highway on our way to the next waypoint enjoying the sunshine and scenery "
    deep_term = "and suddenly we discovered the hidden submarine bunker under the bridge!"
    full_text = long_prefix + deep_term

    segs = [
        TranscriptSegment(start=50.0, end=75.0, text=full_text),
        TranscriptSegment(start=120.0, end=140.0, text="The hidden submarine bunker has armed guards"),
    ]
    s = _build_state("https://youtube.com/watch?v=sub_test", "DeepSearcher", "GTA6", 300, segs)
    index.index_state(s, kind="stream")

    mat = topic_material(cfg, "GTA6", max_terms=10)
    terms = mat["terms"]
    sub_term = next((t for t in terms if "submarine" in t["term"]), None)
    assert sub_term is not None
    # Check snippet centering
    snip_text = sub_term["snippets"][0]["text"]
    assert "submarine" in snip_text.lower(), f"Snippet truncated the term: {snip_text}"
    # Check co-occurring terms exist
    assert "co_occurring" in sub_term

