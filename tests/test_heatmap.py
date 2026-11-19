"""Tests for the YouTube "Most replayed" heatmap signal (fetch, peaks, fusion, state, MCP)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from streamclipper.analysis.heatmap import HeatmapAnalyzer, detect_heatmap_peaks
from streamclipper.cache import StepCache
from streamclipper.config import Config
from streamclipper.download import heatmap as dl_heatmap
from streamclipper.download.heatmap import fetch_heatmap, get_heatmap, load_cached_heatmap
from streamclipper.models import (
    Highlight,
    PipelineState,
    PipelineStatus,
    Platform,
    SignalType,
    StreamInfo,
    TimeRange,
)
from streamclipper.scoring.scorer import Scorer

from tests.test_mcp_server import _extract_json, _make_sample_transcript, mcp_env  # noqa: F401
from streamclipper.mcp_server.server import mcp

RAW = [
    {"start_time": 0.0, "end_time": 100.0, "value": 0.2},
    {"start_time": 100.0, "end_time": 200.0, "value": 1.0},
    {"start_time": 200.0, "end_time": 300.0, "value": 0.1},
]


class _Stub(Scorer):
    def _score_chunk(self, chunk, loudness, chat):  # pragma: no cover
        return []


@pytest.fixture()
def config(tmp_path: Path) -> Config:
    return Config(user_config_path=tmp_path / "nonexistent.yaml")


def _h(start: float, end: float, score: float) -> Highlight:
    return Highlight(start=start, end=end, score=score, title="t", reason="r")


# --- H1 fetch ---------------------------------------------------------------

def test_H1_fetch_from_info_dict_no_network() -> None:
    with patch("yt_dlp.YoutubeDL", side_effect=AssertionError("network used")):
        segs = fetch_heatmap("https://youtu.be/x", info={"heatmap": RAW})
    assert segs[1] == {"start": 100.0, "end": 200.0, "value": 1.0}
    assert len(segs) == 3


def test_H1_fetch_from_cached_info_json(tmp_path: Path) -> None:
    (tmp_path / "vid123.info.json").write_text(json.dumps({"heatmap": RAW}), encoding="utf-8")
    segs = load_cached_heatmap(tmp_path, "vid123")
    assert [s["value"] for s in segs] == [0.2, 1.0, 0.1]
    with patch.object(dl_heatmap, "fetch_heatmap", side_effect=AssertionError("network used")):
        assert get_heatmap("https://youtu.be/x", tmp_path, "vid123") == segs


def test_H1_missing_heatmap_returns_empty() -> None:
    assert fetch_heatmap("u", info={"title": "x"}) == []
    assert fetch_heatmap("u", info={"heatmap": None}) == []
    assert fetch_heatmap("", info=None) == []


def test_H1_ytdlp_raising_returns_empty() -> None:
    with patch("yt_dlp.YoutubeDL", side_effect=RuntimeError("boom")):
        assert fetch_heatmap("https://youtu.be/x") == []


def test_H1_fetch_via_ytdlp_and_cache_written(tmp_path: Path) -> None:
    class _Y:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=False): return {"heatmap": RAW}

    with patch("yt_dlp.YoutubeDL", _Y):
        segs = get_heatmap("https://youtu.be/x", tmp_path, "vid9")
    assert len(segs) == 3
    assert (tmp_path / "vid9_heatmap.json").is_file()


# --- H2 peaks ---------------------------------------------------------------

def test_H2_peak_detection_merge_and_metadata() -> None:
    segs = [{"start": i * 100.0, "end": (i + 1) * 100.0, "value": v}
            for i, v in enumerate([0.1, 0.1, 0.8, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])]
    peaks = detect_heatmap_peaks(segs, threshold_std=1.0, min_value=0.5)
    assert len(peaks) == 1  # adjacent bins merged
    p = peaks[0]
    assert (p.start, p.end) == (200.0, 400.0)
    assert p.signal_type == SignalType.HEATMAP
    assert p.score == pytest.approx(10.0)
    assert p.metadata["source"] == "heatmap"
    assert p.metadata["peak_time"] == pytest.approx(350.0)
    assert p.metadata["heat"] == pytest.approx(1.0)


def test_H2_no_peaks_when_flat_or_below_min(config: Config) -> None:
    flat = [{"start": i * 10.0, "end": (i + 1) * 10.0, "value": 0.3} for i in range(10)]
    assert detect_heatmap_peaks(flat) == []
    assert detect_heatmap_peaks(None) == []
    assert HeatmapAnalyzer(config).detect([]) == []


# --- H3 state ---------------------------------------------------------------

def test_H3_state_roundtrip_and_default(config: Config) -> None:
    info = StreamInfo(url="https://youtu.be/rt", title="t", platform=Platform.YOUTUBE,
                      streamer="s", game="g", duration_seconds=1000, video_id="rt")
    spike = TimeRange(start=1.0, end=2.0, score=8.0, signal_type=SignalType.HEATMAP,
                      metadata={"source": "heatmap", "heat": 0.8})
    state = PipelineState(stream_info=info, status=PipelineStatus.SCORING, heatmap_spikes=[spike])
    assert PipelineState.from_dict(state.to_dict()).heatmap_spikes == [spike]
    legacy = state.to_dict()
    legacy.pop("heatmap_spikes")
    assert PipelineState.from_dict(legacy).heatmap_spikes == []

    cache = StepCache(config)
    cache.save_state(state)
    loaded = cache.load_state(info.url)
    assert loaded is not None and loaded.heatmap_spikes == [spike]


# --- H4 fusion --------------------------------------------------------------

def test_H4_fusion_uses_heatmap(config: Config) -> None:
    s = _Stub(config)
    h = _h(100.0, 150.0, 8.0)
    loud = [TimeRange(start=110.0, end=130.0, score=6.0)]
    chat = [TimeRange(start=110.0, end=130.0, score=4.0)]
    heat = [TimeRange(start=0.0, end=500.0, score=9.0, signal_type=SignalType.HEATMAP)]
    final = s._compute_weighted_score(h, loud, chat, heatmap_spikes=heat)
    assert final == pytest.approx(0.4 * 8.0 + 0.2 * 6.0 + 0.2 * 4.0 + 0.2 * 9.0)


def test_H4_fusion_renormalizes_without_heatmap(config: Config) -> None:
    s = _Stub(config)
    h = _h(100.0, 150.0, 8.0)
    loud = [TimeRange(start=110.0, end=130.0, score=6.0)]
    chat = [TimeRange(start=110.0, end=130.0, score=4.0)]
    final = s._compute_weighted_score(h, loud, chat)
    assert final == pytest.approx(0.5 * 8.0 + 0.25 * 6.0 + 0.25 * 4.0)  # pre-heatmap behaviour
    assert s._compute_weighted_score(h, loud, chat, heatmap_spikes=[]) == pytest.approx(final)


def test_H4_default_weights_sum_to_one(config: Config) -> None:
    w = config.get("analysis.weights")
    assert w == {"loudness": 0.20, "chat_velocity": 0.20, "heatmap": 0.20, "llm_score": 0.40}
    assert sum(w.values()) == pytest.approx(1.0)


def test_H4_legacy_three_key_user_weights_still_valid(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    p.write_text("analysis:\n  weights: {loudness: 0.25, chat_velocity: 0.25, llm_score: 0.5}\n",
                 encoding="utf-8")
    cfg = Config(user_config_path=p)
    assert cfg.get("analysis.weights.heatmap") == 0.0


def test_H4_quiet_high_heat_floor(config: Config) -> None:
    s = _Stub(config)
    far_loud = [TimeRange(start=5000.0, end=5010.0, score=9.0)]
    far_chat = [TimeRange(start=6000.0, end=6010.0, score=9.0)]
    hot = [TimeRange(start=0.0, end=500.0, score=8.0, signal_type=SignalType.HEATMAP,
                     metadata={"heat": 0.8})]
    warm = [TimeRange(start=0.0, end=500.0, score=6.0, signal_type=SignalType.HEATMAP,
                      metadata={"heat": 0.6})]
    h = _h(100.0, 150.0, 7.5)
    floored = s._compute_weighted_score(h, far_loud, far_chat, heatmap_spikes=hot)
    assert floored >= 0.85 * 7.5 - 1e-9
    unfloored = s._compute_weighted_score(_h(100.0, 150.0, 7.5), far_loud, far_chat,
                                          heatmap_spikes=warm)
    assert unfloored < 0.85 * 7.5
    low_llm = s._compute_weighted_score(_h(100.0, 150.0, 6.5), far_loud, far_chat,
                                        heatmap_spikes=hot)
    assert low_llm < 0.85 * 6.5 + 1.0  # no floor below LLM score 7


def test_H4_get_candidate_chunks_heatmap_counts_as_signal(config: Config) -> None:
    s = _Stub(config)
    transcript = _make_sample_transcript()
    none = s.get_candidate_chunks(transcript, [], [], only_with_signals=True)
    assert none == []
    heat = [TimeRange(start=880.0, end=960.0, score=8.0, signal_type=SignalType.HEATMAP)]
    got = s.get_candidate_chunks(transcript, [], [], only_with_signals=True, heatmap_spikes=heat)
    assert len(got) == 1 and got[0][0].start <= 900.0 < got[0][0].end


# --- H5 MCP ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_H5_mcp_chunks_surface_quiet_high_heat_chunk(mcp_env) -> None:  # noqa: F811
    _, cache, _ = mcp_env
    url = "https://www.youtube.com/watch?v=heat1234567"
    cache.save_state(PipelineState(
        stream_info=StreamInfo(url=url, title="t", platform=Platform.YOUTUBE, streamer="S",
                               game="GTA 6", duration_seconds=3600, video_id="heat1234567"),
        status=PipelineStatus.SCORING,
        transcript=_make_sample_transcript(),
        analysis_complete=True,
        heatmap_spikes=[TimeRange(start=880.0, end=960.0, score=9.0,
                                  signal_type=SignalType.HEATMAP,
                                  metadata={"source": "heatmap", "heat": 0.9})],
    ))
    data = _extract_json(await mcp.call_tool("get_transcript_chunks", {"url": url}))
    assert data["returned_chunks"] == 1
    chunk = data["chunks"][0]
    assert chunk["heat"] == pytest.approx(0.9)
    assert chunk["loudness_spikes"] == [] and chunk["chat_spikes"] == []
    assert "hint" not in data

    tools = {t.name: t for t in await mcp.list_tools()}
    assert "heat" in (tools["get_transcript_chunks"].description or "")
