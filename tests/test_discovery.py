"""Unit tests for YouTube stream discovery (Package 2.6)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
import yt_dlp

from streamclipper.config import Config, HardwareProfile
from streamclipper.discovery import YouTubeDiscovery
from streamclipper.models import Platform, StreamInfo

CPU_PROFILE = HardwareProfile(
    tier="cpu_only",
    gpu_name="CPU",
    vram_gb=0.0,
    compute_capability="N/A",
    has_tensor_cores=False,
    has_nvenc=False,
    device="cpu",
    model_size="base",
    compute_type="int8",
    batch_size=4,
    llm_model="qwen3:4b",
    video_codec="libx264",
)


def _make_config(tmp_path: Path, discovery_overrides: dict[str, Any] | None = None) -> Config:
    """Create a deterministic Config instance with optional discovery overrides."""
    if not discovery_overrides:
        return Config(user_config_path=tmp_path / "missing.yaml", hardware_profile=CPU_PROFILE)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"discovery": discovery_overrides}), encoding="utf-8")
    return Config(user_config_path=cfg_path, hardware_profile=CPU_PROFILE)


def _make_entry(
    video_id: str,
    *,
    title: str = "Long Stream VOD",
    uploader: str = "TopStreamer",
    duration: int | None = 7200,
    view_count: int | None = 25_000,
    webpage_url: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": video_id,
        "title": title,
        "uploader": uploader,
        "duration": duration,
        "view_count": view_count,
    }
    if webpage_url is not None:
        entry["webpage_url"] = webpage_url
    else:
        entry["webpage_url"] = f"https://www.youtube.com/watch?v={video_id}"
    return entry


def _patch_yt_dlp_search(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses_by_query: dict[str, Any] | None = None,
    default_entries: list[Any] | None = None,
    side_effect: Exception | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Patch yt_dlp.YoutubeDL and capture options and search terms."""
    captured_opts: list[dict[str, Any]] = []
    captured_queries: list[str] = []

    def _factory(opts: dict[str, Any]) -> MagicMock:
        captured_opts.append(opts)
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.__exit__.return_value = False

        def _extract_info(search_term: str, download: bool = False) -> Any:
            assert download is False
            captured_queries.append(search_term)
            if side_effect is not None:
                raise side_effect
            if responses_by_query is not None and search_term in responses_by_query:
                resp = responses_by_query[search_term]
                if isinstance(resp, Exception):
                    raise resp
                return resp
            return {"entries": default_entries if default_entries is not None else []}

        inst.extract_info.side_effect = _extract_info
        return inst

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)
    return captured_opts, captured_queries


# ---------------------------------------------------------------------------
# Search & Query Dispatch Tests
# ---------------------------------------------------------------------------


class TestYouTubeSearch:
    def test_search_youtube_uses_ytsearch_and_extract_flat_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path)
        discovery = YouTubeDiscovery(cfg)

        entries = [_make_entry("vid001"), None, _make_entry("vid002")]
        captured_opts, captured_queries = _patch_yt_dlp_search(
            monkeypatch, default_entries=entries
        )

        results = discovery._search_youtube("GTA 6", max_results=15)

        assert captured_queries == ["ytsearch15:GTA 6 live stream"]
        assert len(captured_opts) == 1
        assert captured_opts[0]["extract_flat"] is False
        assert captured_opts[0]["skip_download"] is True
        assert captured_opts[0]["ignoreerrors"] is True
        assert len(results) == 2
        assert [r["id"] for r in results] == ["vid001", "vid002"]

    def test_discover_uses_default_game_and_configured_keywords(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path)
        discovery = YouTubeDiscovery(cfg)

        captured_opts, captured_queries = _patch_yt_dlp_search(
            monkeypatch,
            responses_by_query={
                "ytsearch20:GTA 6 live stream": {
                    "entries": [_make_entry("v1", view_count=10_000)]
                },
                "ytsearch20:Grand Theft Auto 6 live stream": {
                    "entries": [_make_entry("v2", view_count=50_000)]
                },
                "ytsearch20:GTA VI live stream": {
                    "entries": [
                        _make_entry("v1", view_count=10_000),  # duplicate across keywords
                        _make_entry("v3", view_count=30_000),
                    ]
                },
            },
        )

        streams = discovery.discover()

        assert captured_queries == [
            "ytsearch20:GTA 6 live stream",
            "ytsearch20:Grand Theft Auto 6 live stream",
            "ytsearch20:GTA VI live stream",
        ]
        assert len(captured_opts) == 3
        assert [s.video_id for s in streams] == ["v2", "v3", "v1"]
        assert all(s.game == "GTA 6" for s in streams)
        assert all(s.platform == Platform.YOUTUBE for s in streams)

    def test_discover_for_custom_game_ignores_default_keywords(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path)
        discovery = YouTubeDiscovery(cfg)

        _, captured_queries = _patch_yt_dlp_search(
            monkeypatch,
            default_entries=[
                _make_entry("er01", title="Elden Ring DLC Run", view_count=42_000, duration=10_800)
            ],
        )

        streams = discovery.discover(game="Elden Ring")

        assert captured_queries == ["ytsearch20:Elden Ring live stream"]
        assert len(streams) == 1
        assert streams[0].game == "Elden Ring"
        assert streams[0].video_id == "er01"
        assert streams[0].duration_seconds == 10_800
        assert streams[0].view_count == 42_000

    def test_search_youtube_handles_download_error_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path)
        discovery = YouTubeDiscovery(cfg)

        _patch_yt_dlp_search(
            monkeypatch,
            responses_by_query={
                "ytsearch20:GTA 6 live stream": yt_dlp.utils.DownloadError("network timeout"),
                "ytsearch20:Grand Theft Auto 6 live stream": {
                    "entries": [_make_entry("survivor", view_count=9_000)]
                },
                "ytsearch20:GTA VI live stream": {"entries": []},
            },
        )

        streams = discovery.discover()
        assert [s.video_id for s in streams] == ["survivor"]


# ---------------------------------------------------------------------------
# Filtering, Deduplication, Sorting, and Limit Tests
# ---------------------------------------------------------------------------


class TestFilterStreams:
    def test_filters_short_videos_and_low_view_videos(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"min_duration_hours": 1, "min_viewers": 1000, "max_streams": 10},
        )
        discovery = YouTubeDiscovery(cfg)

        raw_results = [
            _make_entry("short_vid", duration=3599, view_count=50_000),   # < 1 hour -> drop
            _make_entry("live_none_dur", duration=None, view_count=50_000),  # None dur -> drop
            _make_entry("low_views", duration=7200, view_count=999),      # < 1000 views -> drop
            _make_entry("none_views", duration=7200, view_count=None),    # None views -> drop
            _make_entry("missing_id", duration=7200, view_count=10_000) | {"id": ""},
            _make_entry("boundary_ok", duration=3600, view_count=1000),   # exact boundary -> keep
            _make_entry("great_vod", duration=14_400, view_count=120_000),  # keep
        ]

        streams = discovery._filter_streams(raw_results, "GTA 6")

        assert [s.video_id for s in streams] == ["great_vod", "boundary_ok"]
        assert streams[0].duration_seconds == 14_400
        assert streams[0].view_count == 120_000
        assert streams[1].duration_seconds == 3600
        assert streams[1].view_count == 1000

    def test_deduplicates_by_video_id(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        discovery = YouTubeDiscovery(cfg)

        raw_results = [
            # Invalid entry with same ID should not block a later valid entry
            _make_entry("dup1", duration=100, view_count=5000),
            _make_entry("dup1", title="First Valid", duration=7200, view_count=15_000),
            _make_entry("dup1", title="Duplicate Valid", duration=7200, view_count=15_000),
            _make_entry("unique2", title="Second Stream", duration=5400, view_count=20_000),
        ]

        streams = discovery._filter_streams(raw_results, "GTA 6")

        assert len(streams) == 2
        assert [s.video_id for s in streams] == ["unique2", "dup1"]
        assert streams[1].title == "First Valid"

    def test_sorts_by_view_count_descending_and_respects_max_streams(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path, {"max_streams": 3, "min_viewers": 1000})
        discovery = YouTubeDiscovery(cfg)

        raw_results = [
            _make_entry("v1", view_count=5_000),
            _make_entry("v2", view_count=250_000),
            _make_entry("v3", view_count=12_000),
            _make_entry("v4", view_count=90_000),
            _make_entry("v5", view_count=1_500),
        ]

        streams = discovery._filter_streams(raw_results, "Valorant")

        assert len(streams) == 3
        assert [s.video_id for s in streams] == ["v2", "v4", "v3"]
        assert [s.view_count for s in streams] == [250_000, 90_000, 12_000]
        assert all(isinstance(s, StreamInfo) for s in streams)

    def test_metadata_fallbacks_for_url_and_streamer(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        discovery = YouTubeDiscovery(cfg)

        raw_results = [
            {
                "id": "chan_only",
                "title": "Channel Fallback",
                "channel": "ChannelHost",
                "duration": 4000,
                "view_count": 3000,
            },
            {
                "id": "creator_only",
                "title": "Creator Fallback",
                "creator": "CreatorHost",
                "url": "not-an-http-url",
                "duration": 4000,
                "view_count": 2000,
            },
            {
                "id": "unknown_host",
                "title": "No Host",
                "duration": 4000,
                "view_count": 1500,
            },
        ]

        streams = discovery._filter_streams(raw_results, "GTA 6")

        assert [s.streamer for s in streams] == ["ChannelHost", "CreatorHost", "Unknown"]
        assert [s.url for s in streams] == [
            "https://www.youtube.com/watch?v=chan_only",
            "https://www.youtube.com/watch?v=creator_only",
            "https://www.youtube.com/watch?v=unknown_host",
        ]
