"""Unit tests for shared yt-dlp cookie and JS runtime options helper (Brief 01)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yt_dlp

import streamclipper.download._ytdlp_opts as ytdlp_opts_module
from streamclipper.download._ytdlp_opts import (
    apply_ytdlp_auth_opts,
    find_cookie_file,
    find_js_runtimes,
)
from streamclipper.download.audio import download_audio, extract_stream_info
from streamclipper.download.chat import ChatScraper
from streamclipper.download.video import _extract_video_id, download_segment

SAMPLE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# Cookie Lookup Order Tests
# ---------------------------------------------------------------------------


def test_A01_cookie_lookup_order_env_youtube(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f_yt_long = tmp_path / "yt_long_cookies.txt"
    f_yt_long.write_text("# Netscape HTTP Cookie File", encoding="utf-8")
    f_yt_short = tmp_path / "yt_short_cookies.txt"
    f_yt_short.write_text("# Netscape HTTP Cookie File", encoding="utf-8")
    cwd_cookie = tmp_path / "cookies.txt"
    cwd_cookie.write_text("# Netscape HTTP Cookie File", encoding="utf-8")

    monkeypatch.setenv("YOUTUBE_COOKIES_FILE", str(f_yt_long))
    monkeypatch.setenv("YT_COOKIES_FILE", str(f_yt_short))
    monkeypatch.chdir(tmp_path)

    found = find_cookie_file()
    assert found == str(f_yt_long.resolve())


def test_A01_cookie_lookup_order_env_yt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f_yt_short = tmp_path / "yt_short_cookies.txt"
    f_yt_short.write_text("# Netscape HTTP Cookie File", encoding="utf-8")
    cwd_cookie = tmp_path / "cookies.txt"
    cwd_cookie.write_text("# Netscape HTTP Cookie File", encoding="utf-8")

    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    monkeypatch.setenv("YT_COOKIES_FILE", str(f_yt_short))
    monkeypatch.chdir(tmp_path)

    found = find_cookie_file()
    assert found == str(f_yt_short.resolve())


def test_A01_cookie_lookup_order_cwd_cookies_txt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd_cookie = tmp_path / "cookies.txt"
    cwd_cookie.write_text("# Netscape HTTP Cookie File", encoding="utf-8")
    cwd_txt_txt = tmp_path / "cookies.txt.txt"
    cwd_txt_txt.write_text("# Netscape HTTP Cookie File", encoding="utf-8")

    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    (fake_repo / "cookies.txt").write_text("# Netscape", encoding="utf-8")

    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    monkeypatch.delenv("YT_COOKIES_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ytdlp_opts_module, "_REPO_ROOT", fake_repo)

    found = find_cookie_file()
    assert found == str(cwd_cookie.resolve())


def test_A01_cookie_lookup_order_cwd_cookies_txt_txt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd_txt_txt = tmp_path / "cookies.txt.txt"
    cwd_txt_txt.write_text("# Netscape HTTP Cookie File", encoding="utf-8")

    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    (fake_repo / "cookies.txt").write_text("# Netscape", encoding="utf-8")

    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    monkeypatch.delenv("YT_COOKIES_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ytdlp_opts_module, "_REPO_ROOT", fake_repo)

    found = find_cookie_file()
    assert found == str(cwd_txt_txt.resolve())


def test_A01_cookie_lookup_order_repo_root_cookies_txt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    repo_cookie = fake_repo / "cookies.txt"
    repo_cookie.write_text("# Netscape HTTP Cookie File", encoding="utf-8")
    (fake_repo / "cookies.txt.txt").write_text("# Netscape", encoding="utf-8")

    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    monkeypatch.delenv("YT_COOKIES_FILE", raising=False)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(ytdlp_opts_module, "_REPO_ROOT", fake_repo)

    found = find_cookie_file()
    assert found == str(repo_cookie.resolve())


def test_A01_cookie_lookup_order_repo_root_cookies_txt_txt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    repo_txt_txt = fake_repo / "cookies.txt.txt"
    repo_txt_txt.write_text("# Netscape HTTP Cookie File", encoding="utf-8")

    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    monkeypatch.delenv("YT_COOKIES_FILE", raising=False)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(ytdlp_opts_module, "_REPO_ROOT", fake_repo)

    found = find_cookie_file()
    assert found == str(repo_txt_txt.resolve())


def test_A01_no_cookie_found_no_cookiefile_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()

    monkeypatch.delenv("YOUTUBE_COOKIES_FILE", raising=False)
    monkeypatch.delenv("YT_COOKIES_FILE", raising=False)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(ytdlp_opts_module, "_REPO_ROOT", empty_repo)

    assert find_cookie_file() is None

    opts: dict[str, Any] = {"format": "best"}
    apply_ytdlp_auth_opts(opts)
    assert "cookiefile" not in opts


def test_A01_cookie_logging_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    secret_name = "secret_youtube_user_cookies_42.txt"
    cookie_file = tmp_path / secret_name
    cookie_file.write_text("SECRET_SESSION_TOKEN_12345", encoding="utf-8")

    monkeypatch.setenv("YOUTUBE_COOKIES_FILE", str(cookie_file))

    with caplog.at_level(logging.DEBUG):
        found = find_cookie_file()

    assert found == str(cookie_file.resolve())
    debug_messages = [rec.message for rec in caplog.records if rec.levelno == logging.DEBUG]
    assert "using cookie file" in debug_messages

    # Ensure neither file contents nor secret file name were logged
    for rec in caplog.records:
        assert "SECRET_SESSION_TOKEN_12345" not in rec.message
        assert secret_name not in rec.message


# ---------------------------------------------------------------------------
# Node Runtime Discovery Tests
# ---------------------------------------------------------------------------


def test_A01_node_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_node = tmp_path / "node.exe"
    fake_node.write_text("binary", encoding="utf-8")

    monkeypatch.setattr("shutil.which", lambda name: str(fake_node) if name == "node" else None)

    runtimes = find_js_runtimes()
    assert runtimes == {"node": {"path": str(fake_node.resolve())}}


def test_A01_node_fallback_program_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_node = Path("C:/Program Files/nodejs/node.exe")
    monkeypatch.setattr("shutil.which", lambda _name: None)

    orig_is_file = Path.is_file

    def _mock_is_file(self: Path) -> bool:
        if self == fallback_node:
            return True
        return orig_is_file(self)

    monkeypatch.setattr(Path, "is_file", _mock_is_file)

    runtimes = find_js_runtimes()
    assert runtimes == {"node": {"path": str(fallback_node.resolve())}}


def test_A01_node_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)

    orig_is_file = Path.is_file

    def _mock_is_file(self: Path) -> bool:
        if self == Path("C:/Program Files/nodejs/node.exe"):
            return False
        return orig_is_file(self)

    monkeypatch.setattr(Path, "is_file", _mock_is_file)

    assert find_js_runtimes() == {}

    opts: dict[str, Any] = {}
    apply_ytdlp_auth_opts(opts)
    assert "js_runtimes" not in opts
    assert "remote_components" not in opts


# ---------------------------------------------------------------------------
# Overwrite Protection Tests
# ---------------------------------------------------------------------------


def test_A01_apply_opts_does_not_overwrite_caller_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ytdlp_opts_module, "find_cookie_file", lambda: "discovered_cookie.txt")
    monkeypatch.setattr(
        ytdlp_opts_module, "find_js_runtimes", lambda: {"node": {"path": "discovered_node.exe"}}
    )

    caller_opts: dict[str, Any] = {
        "cookiefile": "caller_cookies.txt",
        "js_runtimes": {"caller_node": {"path": "custom_node.exe"}},
        "remote_components": ["custom:runtime"],
    }

    result = apply_ytdlp_auth_opts(caller_opts)
    assert result is caller_opts
    assert result["cookiefile"] == "caller_cookies.txt"
    assert result["js_runtimes"] == {"caller_node": {"path": "custom_node.exe"}}
    assert result["remote_components"] == ["custom:runtime"]


# ---------------------------------------------------------------------------
# Call Site Integration Tests (YoutubeDL Option Forwarding)
# ---------------------------------------------------------------------------


def _setup_mock_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[str, dict[str, Any]]:
    cookie_path = str((tmp_path / "mock_cookies.txt").resolve())
    node_runtime = {"node": {"path": str((tmp_path / "mock_node.exe").resolve())}}
    monkeypatch.setattr(ytdlp_opts_module, "find_cookie_file", lambda: cookie_path)
    monkeypatch.setattr(ytdlp_opts_module, "find_js_runtimes", lambda: node_runtime)
    return cookie_path, node_runtime


def test_A01_extract_stream_info_passes_auth_opts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cookie_path, node_runtime = _setup_mock_auth(monkeypatch, tmp_path)
    captured_opts: list[dict[str, Any]] = []

    def _factory(opts: dict[str, Any]) -> MagicMock:
        captured_opts.append(dict(opts))
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.__exit__.return_value = False
        inst.extract_info.return_value = {
            "id": "dQw4w9WgXcQ",
            "title": "Sample Title",
            "duration": 180,
            "uploader": "Sample Streamer",
        }
        return inst

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)

    info = extract_stream_info(SAMPLE_URL)
    assert info.video_id == "dQw4w9WgXcQ"
    assert len(captured_opts) == 1
    opts = captured_opts[0]
    assert opts["cookiefile"] == cookie_path
    assert opts["js_runtimes"] == node_runtime
    assert opts["remote_components"] == ["ejs:github"]


def test_A01_download_audio_passes_auth_opts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cookie_path, node_runtime = _setup_mock_auth(monkeypatch, tmp_path)
    captured_opts: list[dict[str, Any]] = []

    def _factory(opts: dict[str, Any]) -> MagicMock:
        captured_opts.append(dict(opts))
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.__exit__.return_value = False
        inst.extract_info.return_value = {
            "id": "dQw4w9WgXcQ",
            "title": "Sample Title",
            "duration": 180,
            "uploader": "Sample Streamer",
        }

        def _simulate_download(_urls: list[str]) -> None:
            # Materialize the WAV file yt-dlp would produce
            (tmp_path / "dQw4w9WgXcQ.wav").write_bytes(b"RIFFdummywav")

        inst.download.side_effect = _simulate_download
        return inst

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)

    audio_path, stream_info = download_audio(SAMPLE_URL, tmp_path)
    assert audio_path.is_file()
    assert stream_info.video_id == "dQw4w9WgXcQ"
    # download_audio calls extract_stream_info first, then download
    assert len(captured_opts) == 2
    for opts in captured_opts:
        assert opts["cookiefile"] == cookie_path
        assert opts["js_runtimes"] == node_runtime
        assert opts["remote_components"] == ["ejs:github"]

    # Verify 16 kHz mono WAV postprocessor args are preserved (AGENTS.md §4 rule 6)
    dl_opts = captured_opts[1]
    assert dl_opts["postprocessor_args"] == {"extractaudio": ["-ar", "16000", "-ac", "1"]}


def test_A01_video_extract_video_id_passes_auth_opts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cookie_path, node_runtime = _setup_mock_auth(monkeypatch, tmp_path)
    captured_opts: list[dict[str, Any]] = []

    def _factory(opts: dict[str, Any]) -> MagicMock:
        captured_opts.append(dict(opts))
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.__exit__.return_value = False
        inst.extract_info.return_value = {"id": "canonical_vid_123"}
        return inst

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)

    vid = _extract_video_id("https://example.com/live/some_stream")
    assert vid == "canonical_vid_123"
    assert len(captured_opts) == 1
    opts = captured_opts[0]
    assert opts["cookiefile"] == cookie_path
    assert opts["js_runtimes"] == node_runtime
    assert opts["remote_components"] == ["ejs:github"]


def test_A01_video_download_segment_passes_auth_opts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cookie_path, node_runtime = _setup_mock_auth(monkeypatch, tmp_path)
    captured_opts: list[dict[str, Any]] = []

    def _factory(opts: dict[str, Any]) -> MagicMock:
        captured_opts.append(dict(opts))
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.__exit__.return_value = False

        def _simulate_download(_urls: list[str]) -> None:
            expected_mp4 = (
                tmp_path / "testvid_10000_20000_b3000.mp4"
            )
            expected_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")

        inst.download.side_effect = _simulate_download
        return inst

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)

    seg_path = download_segment(
        SAMPLE_URL,
        start_seconds=10.0,
        end_seconds=20.0,
        output_dir=tmp_path,
        buffer_seconds=3.0,
        video_id="testvid",
    )
    assert seg_path.is_file()
    assert len(captured_opts) == 1
    opts = captured_opts[0]
    assert opts["cookiefile"] == cookie_path
    assert opts["js_runtimes"] == node_runtime
    assert opts["remote_components"] == ["ejs:github"]


def test_A01_chat_scraper_scrape_passes_auth_opts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cookie_path, node_runtime = _setup_mock_auth(monkeypatch, tmp_path)
    captured_opts: list[dict[str, Any]] = []

    def _factory(opts: dict[str, Any]) -> MagicMock:
        captured_opts.append(dict(opts))
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.__exit__.return_value = False
        inst.extract_info.return_value = {"id": "dQw4w9WgXcQ"}
        return inst

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)

    # Scrape without creating live_chat.json -> returns empty df gracefully
    df = ChatScraper().scrape(SAMPLE_URL, tmp_path)
    assert df.empty
    assert len(captured_opts) == 1
    opts = captured_opts[0]
    assert opts["cookiefile"] == cookie_path
    assert opts["js_runtimes"] == node_runtime
    assert opts["remote_components"] == ["ejs:github"]
