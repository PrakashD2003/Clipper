"""Unit tests for frame contact sheets, frame notes, FTS search, local vision, and MCP tools (T2)."""
from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from streamclipper.cli import cli
from streamclipper.config import Config
from streamclipper.library.frames import (
    ContactSheet,
    build_contact_sheets,
    download_proxy_video,
    extract_moment_contact_sheets,
)
from streamclipper.library.index import LibraryIndex
from streamclipper.library.models import LibrarySource
from streamclipper.library.vision import (
    VisionUnavailable,
    describe_sheet_ollama,
)
from streamclipper.mcp_server.server import (
    get_frame_sheets,
    index_frames,
    mcp,
    submit_frame_notes,
)


def _make_test_config(tmp_path: Path, **extra: Any) -> Config:
    """Create test configuration pointing cache and temp to tmp_path."""
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
                "sheet_seconds": 90,
                "grid": 3,
                "sheet_width": 1152,
                "proxy_height": 360,
                "note_weight": 0.8,
            },
            "vision": {
                "model": "qwen2.5vl:3b",
            },
        },
    }
    data.update(extra)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return Config(user_config_path=cfg_file)


def _create_synthetic_video(path: Path, duration: int = 200, static: bool = False) -> None:
    """Create a fast synthetic mp4 video using ffmpeg lavfi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if static:
        src = f"color=c=black:duration={duration}:size=160x90:rate=1"
    else:
        src = f"testsrc=duration={duration}:size=160x90:rate=1"

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        src,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _register_source(
    index: LibraryIndex,
    source_id: str,
    url: str = "https://youtube.com/watch?v=mock123",
    title: str = "Test Stream",
    channel: str = "StreamerOne",
    game: str = "GTA6",
    duration: int = 200,
) -> LibrarySource:
    """Insert a source record directly into the library SQLite database."""
    now_iso = datetime.now(timezone.utc).isoformat()
    source = LibrarySource(
        source_id=source_id,
        url=url,
        title=title,
        channel=channel,
        game=game,
        kind="stream",
        language="en",
        duration_seconds=duration,
        video_id=source_id,
        added_at=now_iso,
        indexed_at=now_iso,
        segment_count=0,
    )
    with index._get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sources
            (source_id, url, title, channel, game, kind, language, duration_seconds, video_id, added_at, indexed_at, segment_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                source.source_id,
                source.url,
                source.title,
                source.channel,
                source.game,
                source.kind,
                source.language,
                source.duration_seconds,
                source.video_id,
                source.added_at,
                source.indexed_at,
                source.segment_count,
            ),
        )
        conn.commit()
    return source


def test_T2_download_proxy_video(tmp_path: Path) -> None:
    """Verify download_proxy_video formats, options, and file caching."""
    out_dir = tmp_path / "frames" / "test_src"
    url = "https://youtube.com/watch?v=sample123"

    with patch("streamclipper.library.frames.apply_ytdlp_auth_opts") as mock_auth, patch(
        "yt_dlp.YoutubeDL"
    ) as mock_ydl_cls:
        mock_instance = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        # Simulate yt-dlp downloading proxy.mp4
        def _side_effect(*args: Any, **kwargs: Any) -> None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "proxy.mp4").write_bytes(b"dummy_video_bytes")

        mock_instance.download.side_effect = _side_effect

        proxy_path = download_proxy_video(url, out_dir, height=360)
        assert proxy_path == out_dir / "proxy.mp4"
        assert proxy_path.is_file()
        assert mock_auth.called
        assert mock_instance.download.called

        opts = mock_ydl_cls.call_args[0][0]
        assert opts["format"] == "bv*[height<=360]/b[height<=360]/worst"

        # Calling again should reuse cached file without invoking yt-dlp
        mock_instance.download.reset_mock()
        second_path = download_proxy_video(url, out_dir, height=360)
        assert second_path == proxy_path
        assert not mock_instance.download.called


def test_T2_download_proxy_video_ignores_sidecars_and_remuxes(tmp_path: Path) -> None:
    """Verify download_proxy_video ignores metadata sidecars like proxy.info.json and remuxes proxy.webm."""
    out_dir = tmp_path / "frames" / "test_sidecars"
    url = "https://youtube.com/watch?v=sidecars123"

    with patch("streamclipper.library.frames.apply_ytdlp_auth_opts"), patch(
        "yt_dlp.YoutubeDL"
    ) as mock_ydl_cls, patch("subprocess.run") as mock_subproc:
        mock_instance = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        # Simulate yt-dlp writing a sidecar JSON and a WebM video file
        def _side_effect(*args: Any, **kwargs: Any) -> None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "proxy.info.json").write_text('{"id": "sidecars123"}', encoding="utf-8")
            (out_dir / "proxy.webm").write_bytes(b"dummy_webm_bytes")

        mock_instance.download.side_effect = _side_effect

        def _subproc_side_effect(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
            # When ffmpeg is called to remux proxy.webm -> proxy.mp4, create proxy.mp4
            if cmd[0] == "ffmpeg":
                (out_dir / "proxy.mp4").write_bytes(b"dummy_mp4_bytes")
            return MagicMock(returncode=0)

        mock_subproc.side_effect = _subproc_side_effect

        proxy_path = download_proxy_video(url, out_dir, height=360)
        assert proxy_path == out_dir / "proxy.mp4"
        assert proxy_path.is_file()
        # Verify it didn't overwrite proxy.mp4 with proxy.info.json
        assert proxy_path.read_bytes() == b"dummy_mp4_bytes"
        # Verify info.json was not destroyed
        assert (out_dir / "proxy.info.json").is_file()


def test_T2_build_contact_sheets_count_timestamps_caching(tmp_path: Path) -> None:
    """Verify contact sheet count, timestamps, image generation, and disk caching."""
    config = _make_test_config(tmp_path)
    source_id = "test_source_200s"
    frames_dir = tmp_path / "temp" / "frames" / source_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    proxy_video = frames_dir / "proxy.mp4"

    _create_synthetic_video(proxy_video, duration=200, static=False)

    sheets = build_contact_sheets(config, source_id)
    # 200s video with 90s sheet interval -> 3 sheets (0-90s, 90-180s, 180-200s)
    assert len(sheets) == 3

    # Check Sheet 0
    s0 = sheets[0]
    assert s0.start == 0.0
    assert s0.end == 90.0
    assert len(s0.timestamps) == 9
    assert s0.timestamps == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    assert s0.path.is_file()

    # Check Sheet 1
    s1 = sheets[1]
    assert s1.start == 90.0
    assert s1.end == 180.0
    assert len(s1.timestamps) == 9
    assert s1.timestamps == [90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]
    assert s1.path.is_file()

    # Check Sheet 2 (tail)
    s2 = sheets[2]
    assert s2.start == 180.0
    assert s2.end == 200.0
    assert len(s2.timestamps) == 9
    assert s2.timestamps[0] == 180.0
    assert s2.path.is_file()

    # Verify PIL image size on Sheet 0
    from PIL import Image

    with Image.open(s0.path) as img:
        assert img.size[0] == 1152  # sheet_width

    # Verify idempotency / caching
    sheets_cached = build_contact_sheets(config, source_id)
    assert len(sheets_cached) == 3
    assert sheets_cached[0].path == s0.path


def test_T2_build_contact_sheets_static_skip(tmp_path: Path) -> None:
    """Verify static video sheets (loading screens) are skipped."""
    config = _make_test_config(tmp_path)
    source_id = "static_source_200s"
    frames_dir = tmp_path / "temp" / "frames" / source_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    proxy_video = frames_dir / "proxy.mp4"

    _create_synthetic_video(proxy_video, duration=200, static=True)

    sheets = build_contact_sheets(config, source_id)
    # Static video has diffs near 0.0 < static_threshold (2.0) -> all sheets skipped
    assert len(sheets) == 0

    # Calling a second time should load [] directly from sheets.json manifest without re-extracting frames
    with patch("streamclipper.library.frames._grab_tile_frame") as mock_grab:
        sheets_cached = build_contact_sheets(config, source_id)
        assert len(sheets_cached) == 0
        assert not mock_grab.called


def test_T2_add_frame_notes_replace_semantics_and_fts_sync(tmp_path: Path) -> None:
    """Verify add_frame_notes range replacement and FTS5 synchronization."""
    config = _make_test_config(tmp_path)
    index = LibraryIndex(config)
    source_id = "src_notes_replace"
    _register_source(index, source_id, duration=120)

    # Add initial batch of notes
    initial_notes = [
        {"t": 10.0, "note": "initial vault exterior"},
        {"t": 20.0, "note": "initial vault keypad"},
        {"t": 30.0, "note": "initial vault interior"},
    ]
    added = index.add_frame_notes(source_id, initial_notes)
    assert added == 3

    all_notes = index.list_frame_notes(source_id)
    assert len(all_notes) == 3
    assert all_notes[1]["note"] == "initial vault keypad"

    # FTS check for initial note
    hits = index.search("keypad")
    assert len(hits) == 1
    assert "keypad" in hits[0].snippet

    # Replace notes in range [15.0, 25.0]
    replacement = [{"t": 20.0, "note": "replaced laser security"}]
    added2 = index.add_frame_notes(source_id, replacement)
    assert added2 == 1

    updated_notes = index.list_frame_notes(source_id)
    assert len(updated_notes) == 3
    assert updated_notes[0]["t"] == 10.0
    assert updated_notes[1]["t"] == 20.0
    assert updated_notes[1]["note"] == "replaced laser security"
    assert updated_notes[2]["t"] == 30.0

    # Old note removed from FTS5
    hits_old = index.search("keypad")
    assert len(hits_old) == 0

    # New note searchable in FTS5
    hits_new = index.search("laser")
    assert len(hits_new) == 1
    assert "laser" in hits_new[0].snippet


def test_T2_search_frame_notes_and_speech_merge(tmp_path: Path) -> None:
    """Verify search returns frame-note hits and merges with overlapping speech hits."""
    config = _make_test_config(tmp_path)
    index = LibraryIndex(config)
    source_id = "src_speech_merge"
    _register_source(index, source_id, duration=100)

    # Insert a speech segment in [10.0, 40.0]
    with index._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO segments (source_id, start, end, text, chat_text, heat, loudness, chat_rate)
            VALUES (?, 10.0, 40.0, 'player approaching the diamond casino', '', 0.5, 2.0, 10.0);
            """,
            (source_id,),
        )
        conn.commit()

    # Insert a frame note at t=25.0
    index.add_frame_notes(source_id, [{"t": 25.0, "note": "neon casino signs illuminated"}])

    hits = index.search("casino")
    # Because speech [10.0, 40.0] and frame note [15.0, 35.0] overlap in same source, they merge!
    assert len(hits) == 1
    hit = hits[0]
    assert hit.source_id == source_id
    assert hit.start == 10.0
    assert hit.end == 40.0
    assert "casino" in hit.snippet.lower()


def test_T2_search_frame_notes_note_weight_ordering(tmp_path: Path) -> None:
    """Verify frame-note matches rank below speech matches of equal relative bm25."""
    config = _make_test_config(tmp_path)
    index = LibraryIndex(config)

    src_speech = "src_speech"
    src_frames = "src_frames"
    _register_source(index, src_speech, title="Speech Source", duration=100)
    _register_source(index, src_frames, title="Frames Source", duration=100)

    # Speech source has segment matching "submarine"
    with index._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO segments (source_id, start, end, text, chat_text, heat, loudness, chat_rate)
            VALUES (?, 50.0, 80.0, 'submarine diving deep underwater', '', 0.0, 0.0, 0.0);
            """,
            (src_speech,),
        )
        conn.commit()

    # Frame notes source has frame note matching "submarine" at t=50.0
    index.add_frame_notes(src_frames, [{"t": 50.0, "note": "submarine diving deep underwater"}])

    hits = index.search("submarine")
    assert len(hits) == 2
    # Speech hit has normalized bm25 = 1.0; frame hit has norm_bm25 * 0.8 = 0.8
    assert hits[0].source_id == src_speech
    assert hits[1].source_id == src_frames
    assert hits[0].score > hits[1].score


def test_T2_ollama_vision_parsing(tmp_path: Path) -> None:
    """Verify describe_sheet_ollama handles valid JSON, markdown code fences, and garbage."""
    config = _make_test_config(tmp_path)
    dummy_sheet = ContactSheet(
        path=tmp_path / "sheet_0.jpg",
        start=0.0,
        end=90.0,
        timestamps=[0.0, 10.0, 20.0],
    )
    dummy_sheet.path.write_bytes(b"dummy")

    with patch("ollama.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.list.return_value = {"models": [{"name": "qwen2.5vl:3b"}]}

        # 1. Valid JSON array
        client.chat.return_value = MagicMock(
            message=MagicMock(content='[{"t": 10.0, "note": "player driving red sports car"}]')
        )
        res1 = describe_sheet_ollama(config, dummy_sheet)
        assert len(res1) == 1
        assert res1[0]["t"] == 10.0
        assert res1[0]["note"] == "player driving red sports car"

        # 2. Markdown code fences
        client.chat.return_value = MagicMock(
            message=MagicMock(
                content="```json\n[{\"t\": 20.0, \"note\": \"helicopter landing on skyscraper roof\"}]\n```"
            )
        )
        res2 = describe_sheet_ollama(config, dummy_sheet)
        assert len(res2) == 1
        assert res2[0]["t"] == 20.0
        assert res2[0]["note"] == "helicopter landing on skyscraper roof"

        # 3. Garbage output -> returns empty list
        client.chat.return_value = MagicMock(
            message=MagicMock(content="I cannot understand this image properly.")
        )
        res3 = describe_sheet_ollama(config, dummy_sheet)
        assert res3 == []

        # 4. Empty notes dropped, word clamp to 15 words
        client.chat.return_value = MagicMock(
            message=MagicMock(
                content=json.dumps(
                    [
                        {"t": 10.0, "note": "   "},
                        {
                            "t": 30.0,
                            "note": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen",
                        },
                    ]
                )
            )
        )
        res4 = describe_sheet_ollama(config, dummy_sheet)
        assert len(res4) == 1
        assert res4[0]["t"] == 30.0
        assert len(res4[0]["note"].split()) == 15


def test_T2_ollama_vision_unavailable_when_missing(tmp_path: Path) -> None:
    """Verify VisionUnavailable raised when model is not installed."""
    config = _make_test_config(tmp_path)
    dummy_sheet = ContactSheet(
        path=tmp_path / "sheet_0.jpg",
        start=0.0,
        end=90.0,
        timestamps=[0.0, 10.0],
    )

    with patch("ollama.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.list.return_value = {"models": [{"name": "llama3:latest"}]}

        with pytest.raises(VisionUnavailable) as exc_info:
            describe_sheet_ollama(config, dummy_sheet)
        assert "ollama pull qwen2.5vl:3b" in str(exc_info.value)


def test_T2_resolve_installed_model() -> None:
    """Verify Ollama vision model name fuzzy resolution."""
    from streamclipper.library.vision import resolve_installed_model

    installed = ["qwen2.5-vl:3b-instruct", "llama3:latest", "mistral:7b"]

    # Normalized match (qwen2.5vl:3b -> qwen2.5-vl:3b-instruct prefix/norm)
    assert resolve_installed_model("qwen2.5vl:3b", installed) is not None
    # Latest variation
    assert resolve_installed_model("llama3", installed) == "llama3:latest"
    # Exact
    assert resolve_installed_model("mistral:7b", installed) == "mistral:7b"
    # Not installed
    assert resolve_installed_model("gemma:2b", installed) is None
    # Empty
    assert resolve_installed_model("", installed) is None


def test_T2_ollama_vision_disabled(tmp_path: Path) -> None:
    """Verify VisionUnavailable raised when model is configured to empty string (disabled)."""
    config = _make_test_config(tmp_path)
    config._data.setdefault("library", {}).setdefault("vision", {})["model"] = ""
    dummy_sheet = ContactSheet(
        path=tmp_path / "sheet_0.jpg",
        start=0.0,
        end=90.0,
        timestamps=[0.0, 10.0],
    )
    with pytest.raises(VisionUnavailable) as exc_info:
        describe_sheet_ollama(config, dummy_sheet)
    assert "disabled in configuration" in str(exc_info.value).lower()


def test_T2_mcp_submit_frame_notes_validation(tmp_path: Path) -> None:
    """Verify submit_frame_notes validates timestamp duration and note length."""
    config = _make_test_config(tmp_path)
    index = LibraryIndex(config)
    source_id = "mcp_source_val"
    _register_source(index, source_id, duration=100)

    # Patch Config inside server to use test config
    with patch("streamclipper.mcp_server.server.Config", return_value=config):
        # 1. Timestamp exceeds source duration (150 > 100)
        invalid_ts_res = json.loads(
            submit_frame_notes(source_id, [{"t": 150.0, "note": "player running"}])
        )
        assert "error" in invalid_ts_res
        assert "outside source duration" in invalid_ts_res["error"]

        # 2. Timestamp negative (-5 < 0)
        neg_ts_res = json.loads(
            submit_frame_notes(source_id, [{"t": -5.0, "note": "player running"}])
        )
        assert "error" in neg_ts_res

        # 3. Note exceeds 200 characters
        long_note = "A" * 250
        long_note_res = json.loads(
            submit_frame_notes(source_id, [{"t": 10.0, "note": long_note}])
        )
        assert "error" in long_note_res
        assert "exceeds 200 characters" in long_note_res["error"]

        # 4. Valid notes succeed (including timestamp within +1s integer truncation margin)
        valid_res = json.loads(
            submit_frame_notes(
                source_id,
                [
                    {"t": 10.0, "note": "speeding motorcycle"},
                    {"t": 100.5, "note": "police siren blaring at clip tail"},
                ],
            )
        )
        assert "error" not in valid_res
        assert valid_res["count"] == 2


def test_T2_mcp_get_frame_sheets_returns_images(tmp_path: Path) -> None:
    """Verify get_frame_sheets returns MCP Image helpers and metadata."""
    config = _make_test_config(tmp_path)
    source_id = "mcp_sheets_src"
    frames_dir = tmp_path / "temp" / "frames" / source_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    sheet_path = frames_dir / "sheet_0.jpg"
    sheet_path.write_bytes(b"fake_jpeg_bytes")

    # Write manifest
    sheet_meta = ContactSheet(path=sheet_path, start=0.0, end=90.0, timestamps=[0.0, 10.0])
    (frames_dir / "sheets.json").write_text(json.dumps([sheet_meta.to_dict()]), encoding="utf-8")

    with patch("streamclipper.mcp_server.server.Config", return_value=config):
        contents = get_frame_sheets(source_id, start=0.0, count=2)
        assert len(contents) == 2
        # First item is Image helper
        from mcp.server.mcpserver import Image

        assert isinstance(contents[0], Image)
        # Second item is JSON string with instructions
        meta = json.loads(contents[1])
        assert "instruction" in meta
        assert "write one short factual note per tile" in meta["instruction"]
        assert len(meta["sheets"]) == 1

        # count=0 should select 0 sheets
        c0 = get_frame_sheets(source_id, start=0.0, count=0)
        assert len(c0) == 1
        meta0 = json.loads(c0[0])
        assert len(meta0["sheets"]) == 0


@pytest.mark.asyncio
async def test_T2_mcp_index_frames_starts_job(tmp_path: Path) -> None:
    """Verify index_frames starts an asynchronous render job."""
    config = _make_test_config(tmp_path)
    index = LibraryIndex(config)
    source_id = "mcp_job_src"
    _register_source(index, source_id, url="https://youtube.com/watch?v=job123", duration=120)

    with patch("streamclipper.mcp_server.server.Config", return_value=config), patch(
        "streamclipper.mcp_server.server._start_render_job",
        return_value=json.dumps({"status": "started", "job_id": "job_123"}),
    ) as mock_start:
        res = await index_frames(source_id)
        data = json.loads(res)
        assert data["status"] == "started"
        assert data["job_id"] == "job_123"
        assert mock_start.called


def test_T2_cli_library_frames_and_describe(tmp_path: Path) -> None:
    """Verify CLI commands library frames and library describe."""
    config = _make_test_config(tmp_path)
    index = LibraryIndex(config)
    source_id = "cli_source_1"
    _register_source(index, source_id, duration=100)

    runner = CliRunner()
    cfg_file = tmp_path / "config.yaml"

    with patch("streamclipper.library.frames.download_proxy_video") as mock_dl, patch(
        "streamclipper.library.frames.build_contact_sheets"
    ) as mock_build:
        sheet_path = tmp_path / "sheet_0.jpg"
        sheet_path.write_bytes(b"fake")
        mock_build.return_value = [
            ContactSheet(path=sheet_path, start=0.0, end=90.0, timestamps=[0.0, 10.0])
        ]

        # 1. Test library frames
        res1 = runner.invoke(cli, ["--config", str(cfg_file), "library", "frames", source_id])
        assert res1.exit_code == 0
        assert "contact sheets" in res1.output.lower()
        assert mock_dl.called
        assert mock_build.called

    # 2. Test library describe with VisionUnavailable
    with patch("streamclipper.library.frames.build_contact_sheets") as mock_build, patch(
        "streamclipper.library.vision.describe_sheet_ollama",
        side_effect=VisionUnavailable("Model qwen2.5vl:3b missing"),
    ):
        mock_build.return_value = [
            ContactSheet(path=sheet_path, start=0.0, end=90.0, timestamps=[0.0, 10.0])
        ]
        res2 = runner.invoke(
            cli,
            ["--config", str(cfg_file), "library", "describe", source_id, "--limit", "1"],
        )
        assert res2.exit_code == 0
        assert "ollama pull" in res2.output.lower()


def test_extract_moment_contact_sheets_generates_dual_sheets(tmp_path: Path) -> None:
    """Verify extract_moment_contact_sheets builds 2 PNG 3x3 sheets with 9 tiles each."""
    from PIL import Image

    vid_path = tmp_path / "stream_source.mp4"
    _create_synthetic_video(vid_path, duration=240)

    sheets = extract_moment_contact_sheets(
        str(vid_path),
        center_time=150.0,
        temp_dir=tmp_path,
        window_before=120.0,
        window_after=60.0,
    )

    assert len(sheets) == 2
    sheet1, sheet2 = sheets[0], sheets[1]

    # Sheet 1: Buildup & Trigger [center - 120s to -30s] -> [30.0s to 120.0s]
    assert sheet1.start == 30.0
    assert sheet1.end == 120.0
    assert len(sheet1.timestamps) == 9
    assert sheet1.timestamps == [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0]
    assert sheet1.path.is_file()
    assert sheet1.path.suffix.lower() == ".png"

    with Image.open(sheet1.path) as im:
        assert im.size[0] == 1152
        assert im.size[1] > 0

    # Sheet 2: Climax & Reaction [center - 30s to +60s] -> [120.0s to 210.0s]
    assert sheet2.start == 120.0
    assert sheet2.end == 210.0
    assert len(sheet2.timestamps) == 9
    assert sheet2.timestamps == [120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]
    assert sheet2.path.is_file()
    assert sheet2.path.suffix.lower() == ".png"

    with Image.open(sheet2.path) as im:
        assert im.size[0] == 1152
        assert im.size[1] > 0


def test_extract_moment_contact_sheets_cached(tmp_path: Path) -> None:
    """Subsequent calls with identical parameters return cached ContactSheets immediately."""
    vid_path = tmp_path / "cached_source.mp4"
    _create_synthetic_video(vid_path, duration=240)

    sheets1 = extract_moment_contact_sheets(
        str(vid_path),
        center_time=150.0,
        temp_dir=tmp_path,
        window_before=120.0,
        window_after=60.0,
    )
    assert len(sheets1) == 2

    # Second call should load from manifest without calling ffmpeg
    with patch("subprocess.run") as mock_sub:
        sheets2 = extract_moment_contact_sheets(
            str(vid_path),
            center_time=150.0,
            temp_dir=tmp_path,
            window_before=120.0,
            window_after=60.0,
        )
        assert len(sheets2) == 2
        assert sheets2[0].path == sheets1[0].path
        assert sheets2[1].path == sheets1[1].path
        assert not mock_sub.called


def test_extract_moment_contact_sheets_download_failure(tmp_path: Path) -> None:
    """When proxy slice download encounters an error, return empty list gracefully."""
    import yt_dlp

    with patch.object(
        yt_dlp.YoutubeDL, "download", side_effect=yt_dlp.utils.DownloadError("Network error")
    ):
        res = extract_moment_contact_sheets(
            "https://www.youtube.com/watch?v=mock_fail",
            center_time=150.0,
            temp_dir=tmp_path,
        )
        assert res == []


def test_extract_moment_contact_sheets_validation(tmp_path: Path) -> None:
    """Empty URL or negative timestamp raises ValueError."""
    with pytest.raises(ValueError, match="URL cannot be empty"):
        extract_moment_contact_sheets("", 100.0, tmp_path)

    with pytest.raises(ValueError, match="center_time must be >= 0"):
        extract_moment_contact_sheets("https://example.com/stream", -5.0, tmp_path)


def test_extract_moment_contact_sheets_stream_isolation(tmp_path: Path) -> None:
    """Different streams at identical center_time must have isolated cache directories."""
    vid_a = tmp_path / "stream_a.mp4"
    vid_b = tmp_path / "stream_b.mp4"
    _create_synthetic_video(vid_a, duration=200)
    _create_synthetic_video(vid_b, duration=200)

    sheets_a = extract_moment_contact_sheets(str(vid_a), center_time=100.0, temp_dir=tmp_path)
    sheets_b = extract_moment_contact_sheets(str(vid_b), center_time=100.0, temp_dir=tmp_path)

    assert len(sheets_a) == 2
    assert len(sheets_b) == 2
    # Distinct paths — no cache collision
    assert sheets_a[0].path != sheets_b[0].path
    assert sheets_a[0].path.parent != sheets_b[0].path.parent


def test_format_timestamp_label_nan_inf_safety() -> None:
    """_format_timestamp_label handles NaN, Inf, negative, and hours gracefully."""
    from streamclipper.library.frames import _format_timestamp_label

    assert _format_timestamp_label(0.0) == "00:00"
    assert _format_timestamp_label(65.0) == "01:05"
    assert _format_timestamp_label(3665.0) == "01:01:05"
    assert _format_timestamp_label(-10.0) == "00:00"
    assert _format_timestamp_label(float("nan")) == "00:00"
    assert _format_timestamp_label(float("inf")) == "00:00"


