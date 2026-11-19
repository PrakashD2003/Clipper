"""Unit tests for YouTube Discovery Regional, Language, and Proxy Configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
import yt_dlp

from streamclipper.config import Config, HardwareProfile
from streamclipper.discovery import YouTubeDiscovery
from streamclipper.models import Platform

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


def _patch_yt_dlp_search(
    monkeypatch: pytest.MonkeyPatch,
    default_entries: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Patch yt_dlp.YoutubeDL and capture passed options."""
    captured_opts: list[dict[str, Any]] = []

    def _factory(opts: dict[str, Any]) -> MagicMock:
        captured_opts.append(opts)
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.__exit__.return_value = False

        def _extract_info(search_term: str, download: bool = False) -> Any:
            assert download is False
            return {"entries": default_entries if default_entries is not None else []}

        inst.extract_info.side_effect = _extract_info
        return inst

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)
    return captured_opts


class TestDiscoveryGeoAndLanguage:
    """Test suite verifying regional, language, and proxy options in YouTubeDiscovery."""

    def test_discovery_default_geo_and_language_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify default language ('en-US'), region ('US'), and no proxy."""
        cfg = _make_config(tmp_path)
        discovery = YouTubeDiscovery(cfg)

        assert discovery.language == "en-US"
        assert discovery.region == "US"
        assert discovery.proxy == ""

        captured_opts = _patch_yt_dlp_search(monkeypatch)
        discovery._search_youtube("GTA 6", max_results=10)

        assert len(captured_opts) == 1
        opts = captured_opts[0]
        assert opts["http_headers"] == {"Accept-Language": "en-US,en;q=0.9"}
        assert opts["geo_bypass_country"] == "US"
        assert opts["extractor_args"] == {"youtube": {"lang": ["en"]}}
        assert "proxy" not in opts

    def test_discovery_custom_language_and_region(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify custom language and region settings are passed to yt-dlp."""
        cfg = _make_config(
            tmp_path,
            {"language": "ja-JP", "region": "JP", "default_game": "GTA 6"},
        )
        discovery = YouTubeDiscovery(cfg)

        assert discovery.language == "ja-JP"
        assert discovery.region == "JP"
        assert discovery.proxy == ""

        captured_opts = _patch_yt_dlp_search(monkeypatch)
        discovery._search_youtube("GTA 6", max_results=10)

        assert len(captured_opts) == 1
        opts = captured_opts[0]
        assert opts["http_headers"] == {"Accept-Language": "ja-JP,en;q=0.9"}
        assert opts["geo_bypass_country"] == "JP"
        assert opts["extractor_args"] == {"youtube": {"lang": ["ja"]}}
        assert "proxy" not in opts

    def test_discovery_proxy_injection(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify non-empty proxy is injected into yt-dlp options."""
        proxy_url = "http://proxy.internal:8080"
        cfg = _make_config(
            tmp_path,
            {"proxy": proxy_url},
        )
        discovery = YouTubeDiscovery(cfg)

        assert discovery.proxy == proxy_url

        captured_opts = _patch_yt_dlp_search(monkeypatch)
        discovery._search_youtube("GTA 6", max_results=10)

        assert len(captured_opts) == 1
        opts = captured_opts[0]
        assert opts["proxy"] == proxy_url
        assert opts["geo_bypass_country"] == "US"

    def test_discovery_empty_proxy_omitted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify empty or whitespace-only proxy string is omitted from yt-dlp options."""
        for empty_val in ("", "   ", "\t\n"):
            captured_opts = _patch_yt_dlp_search(monkeypatch)
            cfg = _make_config(tmp_path, {"proxy": empty_val})
            discovery = YouTubeDiscovery(cfg)

            assert discovery.proxy == ""
            discovery._search_youtube("GTA 6", max_results=5)

            assert len(captured_opts) == 1
            assert "proxy" not in captured_opts[0]

    def test_discovery_fallback_on_missing_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify None or blank config values gracefully fall back to defaults."""
        cfg = _make_config(
            tmp_path,
            {"language": None, "region": "  ", "proxy": None},
        )
        discovery = YouTubeDiscovery(cfg)

        assert discovery.language == "en-US"
        assert discovery.region == "US"
        assert discovery.proxy == ""

        captured_opts = _patch_yt_dlp_search(monkeypatch)
        discovery._search_youtube("GTA 6", max_results=5)

        assert len(captured_opts) == 1
        opts = captured_opts[0]
        assert opts["http_headers"] == {"Accept-Language": "en-US,en;q=0.9"}
        assert opts["geo_bypass_country"] == "US"
        assert opts["extractor_args"] == {"youtube": {"lang": ["en"]}}
        assert "proxy" not in opts

    def test_discovery_discover_integration_with_geo_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify full discover() flow preserves geo/lang/proxy parameters."""
        sample_entry = {
            "id": "vod123",
            "title": "GTA 6 Tokyo Night Stream",
            "uploader": "TokyoGamer",
            "duration": 7200,
            "view_count": 5000,
            "webpage_url": "https://www.youtube.com/watch?v=vod123",
        }
        cfg = _make_config(
            tmp_path,
            {
                "language": "de-DE",
                "region": "DE",
                "proxy": "socks5://127.0.0.1:1080",
                "min_viewers": 1000,
                "min_duration_hours": 1,
                "max_streams": 5,
            },
        )
        discovery = YouTubeDiscovery(cfg)
        captured_opts = _patch_yt_dlp_search(monkeypatch, default_entries=[sample_entry])

        streams = discovery.discover("GTA 6", max_streams=1)

        assert len(streams) == 1
        assert streams[0].video_id == "vod123"
        assert streams[0].streamer == "TokyoGamer"
        assert streams[0].platform == Platform.YOUTUBE
        assert len(captured_opts) >= 1
        opts = captured_opts[0]
        assert opts["http_headers"] == {"Accept-Language": "de-DE,en;q=0.9"}
        assert opts["geo_bypass_country"] == "DE"
        assert opts["extractor_args"] == {"youtube": {"lang": ["de"]}}
        assert opts["proxy"] == "socks5://127.0.0.1:1080"


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("en-US", "en"),
        ("en", "en"),
        ("en-GB", "en-GB"),
        ("en_gb", "en-GB"),
        ("es-US", "es-US"),
        ("ja-JP", "ja"),
        ("zh-cn", "zh-CN"),
        ("he-IL", "iw"),
        ("xx-YY", "en"),
        ("", "en"),
    ],
)
def test_to_ytdlp_youtube_lang_maps_to_supported_codes(language: str, expected: str) -> None:
    """Regression: yt-dlp rejects tags like 'en-US', which made discovery return 0 streams."""
    from streamclipper.discovery.youtube import _YTDLP_YOUTUBE_LANGS, _to_ytdlp_youtube_lang

    result = _to_ytdlp_youtube_lang(language)
    assert result == expected
    assert result in _YTDLP_YOUTUBE_LANGS
