"""Tests for targeted gameplay B-roll search, download, and preview strip extraction (v4.4)."""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import patch

from PIL import Image
import pytest

from streamclipper.config import Config
from streamclipper.topics.broll import (
    _extract_preview_strip,
    _pick_segment_bounds,
    _score_broll_candidate,
    fetch_topic_broll,
)


def _create_test_mp4(path: Path, duration: float = 4.0, width: int = 320, height: int = 180) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={width}x{height}:rate=15",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        str(path.resolve()),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _make_cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg._data.setdefault("general", {})["temp_dir"] = str(tmp_path / "temp")
    cfg._data["general"]["cache_dir"] = str(tmp_path / "cache")
    cfg._data["general"]["output_dir"] = str(tmp_path / "output")
    return cfg


def test_broll_candidate_scoring_prefers_clean_gameplay_and_rejects_live() -> None:
    live_entry = {"id": "live1", "title": "GTA 6 Live Stream", "duration": 180, "is_live": True}
    ok_live, _ = _score_broll_candidate(live_entry)
    assert ok_live is False

    short_clean = {
        "id": "clean1",
        "title": "GTA 6 Alligator Swamp Gameplay 4K No Commentary",
        "duration": 120,
        "view_count": 50000,
    }
    reaction_long = {
        "id": "react1",
        "title": "Streamer Reaction Facecam Full Stream VOD",
        "duration": 7200,
        "view_count": 500000,
    }
    ok_c, score_c = _score_broll_candidate(short_clean)
    ok_r, score_r = _score_broll_candidate(reaction_long)
    assert ok_c is True and ok_r is True
    assert score_c > score_r

    vertical_short = {
        "id": "vert1",
        "title": "GTA 6 Alligator Swamp Gameplay 4K No Commentary",
        "duration": 45,
        "width": 1080,
        "height": 1920,
        "webpage_url": "https://www.youtube.com/shorts/vert1",
        "view_count": 500000,
    }
    ok_v, score_v = _score_broll_candidate(vertical_short)
    assert ok_v is True
    assert score_c > score_v


def test_pick_segment_bounds_uses_heatmap_or_midpoint() -> None:
    with_heatmap = {
        "duration": 120.0,
        "heatmap": [
            {"start_time": 10.0, "end_time": 20.0, "value": 0.2},
            {"start_time": 80.0, "end_time": 90.0, "value": 0.95},
        ],
    }
    st, et = _pick_segment_bounds(with_heatmap, 10.0)
    assert st == 80.0 and et == 90.0

    no_heatmap = {"duration": 100.0}
    st2, et2 = _pick_segment_bounds(no_heatmap, 10.0)
    assert st2 == 45.0 and et2 == 55.0


def test_extract_preview_strip_creates_3_frame_image(tmp_path: Path) -> None:
    vid = _create_test_mp4(tmp_path / "clip.mp4", duration=3.0)
    preview = tmp_path / "clip_preview.jpg"
    out = _extract_preview_strip(vid, preview)
    assert out is not None and out.is_file()
    with Image.open(out) as im:
        assert im.size == (426 * 3, 240)


def test_fetch_topic_broll_returns_structured_metadata_and_previews(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)

    fake_entries = [
        {
            "id": "live_skip",
            "title": "GTA 6 Live Stream",
            "duration": 180,
            "is_live": True,
            "channel": "LiveGuy",
            "webpage_url": "https://www.youtube.com/watch?v=live_skip",
        },
        {
            "id": "gator_vid",
            "title": "GTA 6 Alligator Swamp Gameplay 4K No Commentary",
            "duration": 120.0,
            "view_count": 25000,
            "channel": "ViceShowcase",
            "webpage_url": "https://www.youtube.com/watch?v=gator_vid",
            "heatmap": [{"start_time": 50.0, "end_time": 60.0, "value": 0.9}],
        },
        {
            "id": "shark_vid",
            "title": "GTA 6 Underwater Sharks Gameplay Showcase",
            "duration": 90.0,
            "view_count": 15000,
            "channel": "OceanClips",
            "webpage_url": "https://www.youtube.com/watch?v=shark_vid",
        },
    ]

    class MockYDL:
        def __init__(self, opts: dict[str, Any]) -> None:
            self.opts = opts

        def __enter__(self) -> "MockYDL":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def extract_info(self, term: str, download: bool = False) -> dict[str, Any]:
            assert term.startswith("ytsearch")
            assert "GTA 6" in term
            assert "alligator swamp" in term
            return {"entries": fake_entries}

    def mock_download_segment(url: str, start: float, end: float, out_dir: Path, **kwargs: Any) -> Path:
        vid = kwargs.get("video_id") or "vid"
        target = Path(out_dir) / f"{vid}_{round(start * 1000)}_{round(end * 1000)}_b0.mp4"
        return _create_test_mp4(target, duration=3.0)

    with patch("streamclipper.topics.broll.yt_dlp.YoutubeDL", MockYDL), \
         patch("streamclipper.topics.broll.download_segment", side_effect=mock_download_segment):
        results = fetch_topic_broll(cfg, "alligator swamp gameplay", game="GTA 6", max_clips=2, clip_seconds=10.0)

    assert len(results) == 2
    first = results[0]
    assert first["source_id"] == "gator_vid"
    assert first["url"] == "https://www.youtube.com/watch?v=gator_vid"
    assert first["channel"] == "ViceShowcase"
    assert first["kind"] == "gameplay"
    assert first["role"] == "broll"
    assert first["start"] == 50.0
    assert first["end"] == 60.0
    assert Path(first["clip_path"]).is_file()
    assert Path(first["clip_path"]).parent == (tmp_path / "temp" / "broll").resolve()
    assert Path(first["preview_sheet_path"]).is_file()
    assert first["preview_sheet_path"].endswith("_preview.jpg")


def test_fetch_topic_broll_edge_cases(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    with pytest.raises(ValueError, match="query is required"):
        fetch_topic_broll(cfg, "   ")
    assert fetch_topic_broll(cfg, "alligator", max_clips=0) == []


def test_v45_fetch_topic_broll_appends_no_commentary_and_skips_countdown_reaction_titles(tmp_path: Path) -> None:
    """fetch_topic_broll appends 'no commentary' to queries and skips countdown/review/reaction titles."""
    from streamclipper.topics.broll import _enrich_broll_search_query, _is_excluded_broll_title

    assert _enrich_broll_search_query("vice city driving", game="GTA 6") == "GTA 6 vice city driving no commentary gameplay"
    assert _enrich_broll_search_query("swamp gameplay", game="GTA 6") == "GTA 6 swamp gameplay no commentary"
    assert _enrich_broll_search_query("GTA 6 swamp no commentary", game="GTA 6") == "GTA 6 swamp no commentary"

    assert _is_excluded_broll_title("Top 10 Details You Missed in GTA 6") is True
    assert _is_excluded_broll_title("Top 20 Hidden Secrets in GTA 6") is True
    assert _is_excluded_broll_title("22 Hidden Details in GTA 6") is True
    assert _is_excluded_broll_title("15 Things You Didn't Know About GTA 6") is True
    assert _is_excluded_broll_title("10 Details You Missed in GTA 6 Trailer 2") is True
    assert _is_excluded_broll_title("GTA 6 Vehicles Ranked") is True
    assert _is_excluded_broll_title("15 Things You Missed in GTA 6 Trailer 2") is True
    assert _is_excluded_broll_title("GTA 6 Trailer 2 Breakdown") is True
    assert _is_excluded_broll_title("GTA 6 Gameplay Analysis") is True
    assert _is_excluded_broll_title("GTA 6 Map Explained") is True
    assert _is_excluded_broll_title("GTA 6 Honest Review & Impressions") is True
    assert _is_excluded_broll_title("Streamer Reaction to GTA 6") is True
    assert _is_excluded_broll_title("GTA 6 Swamp Gameplay 4K No Commentary") is False

    cfg = _make_cfg(tmp_path)
    seen_terms: list[str] = []
    fake_entries = [
        {"id": "top20", "title": "Top 20 Things You Missed in GTA 6", "duration": 180, "view_count": 999999},
        {"id": "cnt22", "title": "22 Hidden Details in GTA 6", "duration": 180, "view_count": 950000},
        {"id": "brk1", "title": "GTA 6 Trailer 2 Breakdown & Analysis", "duration": 180, "view_count": 910000},
        {"id": "top10", "title": "Top 10 GTA 6 Interiors Ranked", "duration": 160, "view_count": 888888},
        {"id": "review1", "title": "GTA 6 Graphics Review", "duration": 200, "view_count": 777777},
        {"id": "react1", "title": "GTA 6 Trailer 2 Reaction", "duration": 150, "view_count": 666666},
        {
            "id": "clean_raw",
            "title": "GTA 6 Night Driving 4K No Commentary Gameplay",
            "duration": 120.0,
            "view_count": 15000,
            "channel": "RawGameplays",
            "webpage_url": "https://www.youtube.com/watch?v=clean_raw",
        },
    ]

    class MockYDL:
        def __init__(self, opts: dict[str, Any]) -> None:
            self.opts = opts

        def __enter__(self) -> "MockYDL":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def extract_info(self, term: str, download: bool = False) -> dict[str, Any]:
            seen_terms.append(term)
            return {"entries": fake_entries}

    def mock_download_segment(url: str, start: float, end: float, out_dir: Path, **kwargs: Any) -> Path:
        vid = kwargs.get("video_id") or "vid"
        target = Path(out_dir) / f"{vid}_{round(start * 1000)}_{round(end * 1000)}_b0.mp4"
        return _create_test_mp4(target, duration=3.0)

    with patch("streamclipper.topics.broll.yt_dlp.YoutubeDL", MockYDL), \
         patch("streamclipper.topics.broll.download_segment", side_effect=mock_download_segment):
        results = fetch_topic_broll(cfg, "night driving", game="GTA 6", max_clips=3, clip_seconds=10.0)

    assert len(seen_terms) == 1
    assert "no commentary gameplay" in seen_terms[0].lower()
    assert len(results) == 1
    assert results[0]["source_id"] == "clean_raw"
    assert Path(results[0]["preview_sheet_path"]).is_file()
