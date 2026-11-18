"""Unit tests for YouTube live chat scraper and velocity analyzer (Package 2.5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yt_dlp

from streamclipper.analysis.chat_velocity import ChatVelocityAnalyzer
from streamclipper.config import Config, HardwareProfile
from streamclipper.download.chat import CHAT_COLUMNS, ChatScraper
from streamclipper.models import SignalType, TimeRange

SAMPLE_URL = "https://www.youtube.com/watch?v=chatTest123"


def _cpu_profile() -> HardwareProfile:
    """Deterministic HardwareProfile avoiding live CUDA/FFmpeg probes in unit tests."""
    return HardwareProfile(
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


def _make_config(tmp_path: Path, **chat_overrides: Any) -> Config:
    """Create a Config with optional analysis.chat overrides."""
    if not chat_overrides:
        return Config(user_config_path=tmp_path / "nonexistent.yaml", hardware_profile=_cpu_profile())

    import yaml

    cfg_path = tmp_path / "chat_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"analysis": {"chat": chat_overrides}}),
        encoding="utf-8",
    )
    return Config(user_config_path=cfg_path, hardware_profile=_cpu_profile())


def _chat_jsonl_entry(
    offset_ms: int | str,
    author: str,
    runs: list[dict[str, Any]],
    *,
    paid: bool = False,
) -> str:
    """Build one YouTube .live_chat.json JSONLines record."""
    renderer_key = "liveChatPaidMessageRenderer" if paid else "liveChatTextMessageRenderer"
    record = {
        "replayChatItemAction": {
            "videoOffsetTimeMsec": str(offset_ms),
            "actions": [
                {
                    "addChatItemAction": {
                        "item": {
                            renderer_key: {
                                "authorName": {"simpleText": author},
                                "message": {"runs": runs},
                            }
                        }
                    }
                }
            ],
        }
    }
    return json.dumps(record)


def _patch_yt_dlp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    video_id: str = "chatTest123",
    jsonl_lines: list[str] | None = None,
    extract_side_effect: Exception | None = None,
) -> list[dict[str, Any]]:
    """Patch yt_dlp.YoutubeDL and optionally materialize {video_id}.live_chat.json."""
    captured_opts: list[dict[str, Any]] = []

    def _factory(opts: dict[str, Any]) -> MagicMock:
        captured_opts.append(opts)
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.__exit__.return_value = False

        if extract_side_effect is not None:
            inst.extract_info.side_effect = extract_side_effect
        else:
            def _extract_info(_url: str, download: bool = True) -> dict[str, Any]:
                if download and jsonl_lines is not None:
                    out_dir = Path(opts["outtmpl"]).parent
                    chat_file = out_dir / f"{video_id}.live_chat.json"
                    chat_file.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
                return {"id": video_id, "title": "GTA 6 Live Stream"}

            inst.extract_info.side_effect = _extract_info

        return inst

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)
    return captured_opts


def _build_synthetic_chat(
    *,
    duration_seconds: int = 300,
    baseline_per_bin: int = 2,
    window_seconds: int = 10,
    bursts: list[tuple[float, int, str]] | None = None,
) -> pd.DataFrame:
    """Create a synthetic chat DataFrame with baseline chatter and injected hype bursts.

    Each burst tuple is ``(burst_start_second, message_count, message_prefix)``.
    """
    rows: list[dict[str, Any]] = []
    num_bins = duration_seconds // window_seconds
    for b in range(num_bins):
        base_t = b * window_seconds
        for k in range(baseline_per_bin):
            rows.append(
                {
                    "offset_seconds": base_t + 1.0 + k * 2.0,
                    "author": f"viewer_{b}_{k}",
                    "message": f"normal chat {b}-{k}",
                }
            )

    for burst_start, count, prefix in bursts or []:
        for i in range(count):
            rows.append(
                {
                    "offset_seconds": burst_start + (i / max(count, 1)) * (window_seconds - 0.5),
                    "author": f"hype_user_{i}",
                    "message": f"{prefix} #{i}",
                }
            )

    return pd.DataFrame(rows, columns=CHAT_COLUMNS)


# ---------------------------------------------------------------------------
# ChatScraper Tests
# ---------------------------------------------------------------------------


class TestChatScraper:
    def test_scrape_parses_text_paid_emoji_and_skips_negative_offsets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        lines = [
            # Pre-stream negative offsets MUST be skipped
            _chat_jsonl_entry(-15000, "EarlyBird", [{"text": "Waiting for stream"}]),
            _chat_jsonl_entry(-1, "AlmostZero", [{"text": "Starting soon"}]),
            # Zero and positive offsets MUST be kept
            _chat_jsonl_entry(0, "FirstViewer", [{"text": "Stream is live!"}]),
            _chat_jsonl_entry(
                12500,
                "HypeFan",
                [
                    {"text": "NO WAY "},
                    {"emoji": {"emojiId": "🔥", "isCustomEmoji": False, "shortcuts": [":fire:"]}},
                ],
            ),
            # Paid Super Chat message MUST be parsed
            _chat_jsonl_entry(
                45250,
                "BigTipper",
                [{"text": "Best GTA 6 clip ever!"}],
                paid=True,
            ),
            # Custom emoji-only message uses shortcut fallback
            _chat_jsonl_entry(
                46000,
                "EmoteSpammer",
                [{"emoji": {"emojiId": "UC123/custom", "isCustomEmoji": True, "shortcuts": [":kekw:"]}}],
            ),
            # Non-message renderer, empty message, blank line, and malformed JSON are ignored
            json.dumps(
                {
                    "replayChatItemAction": {
                        "videoOffsetTimeMsec": "50000",
                        "actions": [
                            {
                                "addChatItemAction": {
                                    "item": {"liveChatMembershipItemRenderer": {"id": "mem1"}}
                                }
                            }
                        ],
                    }
                }
            ),
            _chat_jsonl_entry(51000, "EmptyMsg", [{"text": "   "}]),
            "",
            "{not valid json",
        ]

        captured_opts = _patch_yt_dlp(monkeypatch, video_id="vid999", jsonl_lines=lines)
        scraper = ChatScraper()

        df = scraper.scrape(SAMPLE_URL, tmp_path / "chat_tmp")

        # Verify yt-dlp options (no pytchat)
        assert len(captured_opts) == 1
        opts = captured_opts[0]
        assert opts["skip_download"] is True
        assert opts["writesubtitles"] is True
        assert opts["subtitleslangs"] == ["live_chat"]
        assert opts["noplaylist"] is True

        # Verify parsed records
        assert list(df.columns) == CHAT_COLUMNS
        assert len(df) == 4
        assert (df["offset_seconds"] >= 0.0).all()

        assert df.iloc[0].to_dict() == {
            "offset_seconds": 0.0,
            "author": "FirstViewer",
            "message": "Stream is live!",
        }
        assert df.iloc[1].to_dict() == {
            "offset_seconds": 12.5,
            "author": "HypeFan",
            "message": "NO WAY 🔥",
        }
        assert df.iloc[2].to_dict() == {
            "offset_seconds": 45.25,
            "author": "BigTipper",
            "message": "Best GTA 6 clip ever!",
        }
        assert df.iloc[3].to_dict() == {
            "offset_seconds": 46.0,
            "author": "EmoteSpammer",
            "message": ":kekw:",
        }

    def test_scrape_missing_chat_file_returns_empty_dataframe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_yt_dlp(monkeypatch, video_id="nochat", jsonl_lines=None)
        scraper = ChatScraper()

        df = scraper.scrape(SAMPLE_URL, tmp_path)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == CHAT_COLUMNS
        assert df.empty

    def test_scrape_handles_yt_dlp_download_error_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_yt_dlp(
            monkeypatch,
            extract_side_effect=yt_dlp.utils.DownloadError("No live_chat available"),
        )
        scraper = ChatScraper()

        df = scraper.scrape(SAMPLE_URL, tmp_path)

        assert list(df.columns) == CHAT_COLUMNS
        assert df.empty

    @pytest.mark.parametrize("bad_url", ["", "   "])
    def test_scrape_empty_url_raises_value_error(
        self, bad_url: str, tmp_path: Path
    ) -> None:
        scraper = ChatScraper()
        with pytest.raises(ValueError, match="non-empty string"):
            scraper.scrape(bad_url, tmp_path)

    def test_save_and_load_cache_round_trip_preserves_strings_and_na_tokens(
        self, tmp_path: Path
    ) -> None:
        scraper = ChatScraper()
        cache_path = tmp_path / "nested" / "chat_cache.csv"

        original = pd.DataFrame(
            [
                {"offset_seconds": 1.5, "author": "User1", "message": "NA aim LUL"},
                {"offset_seconds": 2.0, "author": "", "message": "NA"},
                {"offset_seconds": 10.25, "author": "null", "message": "INSANE CLUTCH!"},
            ],
            columns=CHAT_COLUMNS,
        )

        scraper.save_cache(original, cache_path)
        loaded = scraper.load_cache(cache_path)

        assert loaded is not None
        assert list(loaded.columns) == CHAT_COLUMNS
        assert len(loaded) == 3
        assert loaded["offset_seconds"].tolist() == [1.5, 2.0, 10.25]
        assert loaded["author"].tolist() == ["User1", "", "null"]
        assert loaded["message"].tolist() == ["NA aim LUL", "NA", "INSANE CLUTCH!"]

    def test_load_cache_returns_none_when_missing_and_handles_empty_df(
        self, tmp_path: Path
    ) -> None:
        scraper = ChatScraper()
        missing = tmp_path / "does_not_exist.csv"
        assert scraper.load_cache(missing) is None

        empty_cache = tmp_path / "empty.csv"
        scraper.save_cache(pd.DataFrame(columns=CHAT_COLUMNS), empty_cache)
        loaded = scraper.load_cache(empty_cache)
        assert loaded is not None
        assert list(loaded.columns) == CHAT_COLUMNS
        assert loaded.empty


# ---------------------------------------------------------------------------
# ChatVelocityAnalyzer Tests
# ---------------------------------------------------------------------------


class TestChatVelocityAnalyzer:
    def test_detects_single_chat_burst_and_attaches_sample_messages(
        self, tmp_path: Path
    ) -> None:
        config = _make_config(tmp_path)
        analyzer = ChatVelocityAnalyzer(config)

        chat_df = _build_synthetic_chat(
            duration_seconds=300,
            baseline_per_bin=2,
            window_seconds=10,
            bursts=[(120.0, 35, "OH MY GOD")],
        )

        spikes = analyzer.analyze(chat_df)

        assert len(spikes) == 1
        spike = spikes[0]
        assert spike.signal_type == SignalType.CHAT_VELOCITY
        assert spike.start == 120.0
        assert spike.end == 130.0
        assert 0.0 < spike.score <= 10.0
        assert spike.score == 10.0

        # Verify YPP chat ticker sample_messages metadata
        assert "sample_messages" in spike.metadata
        samples = spike.metadata["sample_messages"]
        assert isinstance(samples, list)
        assert len(samples) == 5
        for msg in samples:
            assert set(msg.keys()) == {"author", "message", "offset_seconds"}
            assert isinstance(msg["author"], str)
            assert isinstance(msg["message"], str)
            assert isinstance(msg["offset_seconds"], float)
            assert spike.start <= msg["offset_seconds"] <= spike.end

        # Verify TimeRange JSON round-trip compatibility for StepCache
        round_tripped = TimeRange.from_dict(json.loads(spike.to_json()))
        assert round_tripped.signal_type == SignalType.CHAT_VELOCITY
        assert round_tripped.metadata["sample_messages"] == samples

    def test_detects_multiple_bursts_with_relative_scores(
        self, tmp_path: Path
    ) -> None:
        config = _make_config(tmp_path, prominence_factor=2.5)
        analyzer = ChatVelocityAnalyzer(config)

        chat_df = _build_synthetic_chat(
            duration_seconds=600,
            baseline_per_bin=2,
            window_seconds=10,
            bursts=[
                (80.0, 35, "WAIT WHAT"),
                (260.0, 65, "NO WAY INSANE"),
            ],
        )

        spikes = analyzer.analyze(chat_df)

        assert len(spikes) == 2
        first, second = spikes[0], spikes[1]
        assert first.start == 80.0 and first.end == 90.0
        assert second.start == 260.0 and second.end == 270.0
        assert 0.0 < first.score < second.score <= 10.0
        assert second.score == 10.0
        assert len(first.metadata["sample_messages"]) == 5
        assert len(second.metadata["sample_messages"]) == 5

    def test_merges_nearby_burst_bins_and_caps_sample_messages_at_five(
        self, tmp_path: Path
    ) -> None:
        config = _make_config(tmp_path, prominence_factor=2.0, merge_gap_seconds=10)
        analyzer = ChatVelocityAnalyzer(config)

        # Two nearby peak bins (bin 10: 100-110s, bin 12: 120-130s) separated by 10s gap
        chat_df = _build_synthetic_chat(
            duration_seconds=400,
            baseline_per_bin=2,
            window_seconds=10,
            bursts=[
                (100.0, 30, "SETUP HYPE"),
                (120.0, 45, "PAYOFF CHAOS"),
            ],
        )

        spikes = analyzer.analyze(chat_df)

        assert len(spikes) == 1
        merged = spikes[0]
        assert merged.start == 100.0
        assert merged.end == 130.0
        assert merged.score == 10.0
        assert len(merged.metadata["sample_messages"]) == 5
        assert all(
            merged.start <= m["offset_seconds"] <= merged.end
            for m in merged.metadata["sample_messages"]
        )

    def test_detects_boundary_burst_in_first_bin(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        analyzer = ChatVelocityAnalyzer(config)

        chat_df = _build_synthetic_chat(
            duration_seconds=300,
            baseline_per_bin=2,
            window_seconds=10,
            bursts=[(0.0, 40, "OPENING HYPE")],
        )

        spikes = analyzer.analyze(chat_df)

        assert len(spikes) == 1
        assert spikes[0].start == 0.0
        assert spikes[0].end == 10.0
        assert len(spikes[0].metadata["sample_messages"]) == 5

    def test_skips_when_below_min_messages(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, min_messages=50)
        analyzer = ChatVelocityAnalyzer(config)

        # 49 messages total (< 50) including a sharp burst
        rows = [
            {"offset_seconds": float(i * 10), "author": f"u{i}", "message": "hi"}
            for i in range(19)
        ] + [
            {"offset_seconds": 50.0 + i * 0.1, "author": f"b{i}", "message": "BOOM"}
            for i in range(30)
        ]
        chat_df = pd.DataFrame(rows, columns=CHAT_COLUMNS)
        assert len(chat_df) == 49

        assert analyzer.analyze(chat_df) == []

    def test_skips_empty_uniform_and_negative_offset_only_dataframes(
        self, tmp_path: Path
    ) -> None:
        config = _make_config(tmp_path, min_messages=10)
        analyzer = ChatVelocityAnalyzer(config)

        # Empty DataFrame
        assert analyzer.analyze(pd.DataFrame(columns=CHAT_COLUMNS)) == []

        # Uniform rate across bins (std == 0)
        uniform_df = _build_synthetic_chat(
            duration_seconds=100, baseline_per_bin=2, window_seconds=10
        )
        assert analyzer.analyze(uniform_df) == []

        # Negative timestamps filtered defensively before counting min_messages
        neg_rows = [
            {"offset_seconds": -5.0 - i, "author": f"pre{i}", "message": "waiting"}
            for i in range(60)
        ] + [
            {"offset_seconds": float(i), "author": f"live{i}", "message": "live"}
            for i in range(5)
        ]
        assert analyzer.analyze(pd.DataFrame(neg_rows, columns=CHAT_COLUMNS)) == []

    def test_invalid_config_values_raise_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="window_seconds"):
            ChatVelocityAnalyzer(_make_config(tmp_path, window_seconds=0))

        with pytest.raises(ValueError, match="min_messages"):
            ChatVelocityAnalyzer(_make_config(tmp_path, min_messages=-1))
