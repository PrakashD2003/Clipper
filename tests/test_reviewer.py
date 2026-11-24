"""Unit tests for post-render video review and automated QA (reviewer.py)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from streamclipper.output.reviewer import (
    _format_timestamp_label,
    _parse_fps,
    generate_review_contact_sheet,
    inspect_rendered_video,
)
from tests.test_assembly import _synth_test_video


def _synth_silent_video(path: Path, duration: float = 3.0) -> None:
    """Create a video with 3 seconds of total digital silence (-inf dB)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x180:d={duration}:r=15",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=48000:cl=stereo:d={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _synth_video_no_audio(path: Path, duration: float = 2.0) -> None:
    """Create a video with zero audio streams."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s=320x180:d={duration}:r=15",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_parse_fps() -> None:
    """Verify fps fraction parsing and edge case handling."""
    assert _parse_fps("30/1") == 30.0
    assert _parse_fps("60/1") == 60.0
    assert _parse_fps("30000/1001") == 29.97
    assert _parse_fps("24/1") == 24.0
    assert _parse_fps("0/0") == 0.0
    assert _parse_fps("") == 0.0
    assert _parse_fps("invalid") == 0.0


def test_format_timestamp_label() -> None:
    """Verify MM:SS timestamp formatting."""
    assert _format_timestamp_label(0.0) == "00:00"
    assert _format_timestamp_label(5.4) == "00:05"
    assert _format_timestamp_label(65.0) == "01:05"
    assert _format_timestamp_label(605.9) == "10:05"


def test_inspect_rendered_video_basic_contact_sheet(tmp_path: Path) -> None:
    """Verify contact sheet generation with 12 tiles in a 3x4 grid."""
    vid_path = tmp_path / "test_clip.mp4"
    _synth_test_video(vid_path, duration=3.0, freq=440)

    # Companion metadata
    meta = {
        "hook_title": "Epic Stunt",
        "clip_start": 10.0,
        "clip_end": 13.0,
        "narration_beats": [
            {"role": "setup", "text": "Watch this car jump", "at_second": 0.0, "duration": 1.2},
            {"role": "payoff", "text": "Clean landing", "at_second": 1.5, "duration": 1.0},
        ],
    }
    (tmp_path / "test_clip.json").write_text(json.dumps(meta), encoding="utf-8")

    res = inspect_rendered_video(vid_path)

    assert res["video_path"] == str(vid_path.resolve())
    assert Path(res["contact_sheet_path"]).is_file()
    assert res["contact_sheet_path"].endswith("test_clip_review_sheet.jpg")
    assert res["duration"] > 0.0
    assert res["resolution"] == [320, 180]
    assert res["audio_stats"]["has_audio"] is True
    assert len(res["narration_timeline"]) == 2
    assert res["narration_timeline"][0]["text"] == "Watch this car jump"
    assert res["narration_timeline"][1]["text"] == "Clean landing"

    # Verify contact sheet image dimensions (4 cols x 3 rows)
    with Image.open(res["contact_sheet_path"]) as sheet:
        # 16:9 tile_width=480, tile_height=270 -> 4*480=1920, 3*270=810
        assert sheet.width == 1920
        assert sheet.height == 810


def test_inspect_rendered_video_silence_detection(tmp_path: Path) -> None:
    """Verify dead silence gaps >= 1.5s are detected and flagged."""
    vid_path = tmp_path / "clip_silent.mp4"
    _synth_silent_video(vid_path, duration=3.0)

    res = inspect_rendered_video(vid_path)

    assert len(res["silence_gaps"]) >= 1
    assert any("Silence gap >= 1.5s" in flag for flag in res["red_flags"])
    # Full digital silence also triggers quiet audio warning
    assert any("Audio is very quiet" in flag for flag in res["red_flags"])


def test_inspect_rendered_video_no_audio(tmp_path: Path) -> None:
    """Verify video without audio is flagged as fatal."""
    vid_path = tmp_path / "clip_no_audio.mp4"
    _synth_video_no_audio(vid_path, duration=2.0)

    res = inspect_rendered_video(vid_path)

    assert res["audio_stats"]["has_audio"] is False
    assert any("no audio stream" in flag.lower() for flag in res["red_flags"])


def test_inspect_rendered_video_duration_bounds(tmp_path: Path) -> None:
    """Verify duration bounds flagging for Shorts (> 55s) and Long-form (< 420s or > 660s)."""
    # 1. Short exceeding 55s
    short_vid = tmp_path / "clip_01.mp4"
    _synth_test_video(short_vid, duration=2.0)

    # Mock ffprobe result to simulate 58s duration
    from unittest.mock import patch

    with patch("subprocess.run") as mock_run:
        # Provide real-looking ffprobe json
        probe_json = json.dumps({
            "streams": [
                {"codec_type": "video", "width": 1080, "height": 1920, "r_frame_rate": "30/1", "duration": "58.0"},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "58.0"},
        })
        # Audio run output
        audio_out = "[volumedetect @ 1] mean_volume: -20.0 dB\n[volumedetect @ 1] max_volume: -2.0 dB\n"

        def side_effect(cmd, *args, **kwargs):
            if "ffprobe" in cmd[0]:
                class Proc:
                    stdout = probe_json
                    returncode = 0
                return Proc()
            elif "-af" in cmd:
                class Proc:
                    stderr = audio_out
                    returncode = 0
                return Proc()
            else:
                return subprocess.run(cmd, *args, **kwargs)

        mock_run.side_effect = side_effect
        res = inspect_rendered_video(short_vid)
        assert any("Short duration 58.0s exceeds 55s" in f for f in res["red_flags"])

    # 2. Long-form video outside bounds (< 420s)
    long_vid = tmp_path / "long.mp4"
    _synth_test_video(long_vid, duration=2.0)

    with patch("subprocess.run") as mock_run:
        probe_json = json.dumps({
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30/1", "duration": "300.0"},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "300.0"},
        })
        audio_out = "[volumedetect @ 1] mean_volume: -18.0 dB\n[volumedetect @ 1] max_volume: -1.0 dB\n"

        def side_effect(cmd, *args, **kwargs):
            if "ffprobe" in cmd[0]:
                class Proc:
                    stdout = probe_json
                    returncode = 0
                return Proc()
            elif "-af" in cmd:
                class Proc:
                    stderr = audio_out
                    returncode = 0
                return Proc()
            else:
                return subprocess.run(cmd, *args, **kwargs)

        mock_run.side_effect = side_effect
        res_long = inspect_rendered_video(long_vid)
        assert any("Long-form duration 300.0s is below 420s" in f for f in res_long["red_flags"])


def test_inspect_rendered_video_timeline_from_topic_plan(tmp_path: Path) -> None:
    """Verify narration timeline extraction from plan.json for Long-form and Shorts."""
    long_vid = tmp_path / "long.mp4"
    _synth_test_video(long_vid, duration=2.0)

    plan_data = {
        "plan_id": "plan_test_123",
        "cold_open": "Welcome to Vice City mystery",
        "points": [
            {"index": 1, "heading": "Secret Tunnel", "narration": "Under the airport lies a tunnel"},
            {"index": 2, "heading": "Ghost Ship", "narration": "Off the coast floats an abandoned yacht"},
        ],
        "outro": "Which location surprised you most?",
        "shorts": [
            {
                "slot": "teaser",
                "narration_intro": "Crazy GTA 6 secret",
                "callout": "Look right here",
                "narration_outro": "Full video on channel",
            }
        ],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan_data), encoding="utf-8")
    (tmp_path / "chapters.txt").write_text("00:00 Intro\n00:30 Secret Tunnel\n01:15 Ghost Ship\n02:00 Outro\n", encoding="utf-8")

    # Long-form inspection
    res_long = inspect_rendered_video(long_vid)
    assert len(res_long["narration_timeline"]) == 4
    assert res_long["narration_timeline"][0]["role"] == "cold_open"
    assert res_long["narration_timeline"][1]["heading"] == "Secret Tunnel"
    assert res_long["narration_timeline"][1]["start"] == 30.0

    # Topic Short inspection
    short_vid = tmp_path / "short_teaser.mp4"
    _synth_test_video(short_vid, duration=2.0)
    res_short = inspect_rendered_video(short_vid)
    assert len(res_short["narration_timeline"]) == 3
    assert res_short["narration_timeline"][0]["role"] == "intro"
    assert res_short["narration_timeline"][0]["text"] == "Crazy GTA 6 secret"
    assert res_short["narration_timeline"][1]["role"] == "callout"


def test_grab_frame_fallback_when_opencv_unavailable(tmp_path: Path) -> None:
    """Verify _grab_frame_at_timestamp falls back to ffmpeg stdout pipe when cv2 fails."""
    from unittest.mock import patch
    from streamclipper.output.reviewer import _grab_frame_at_timestamp

    vid_path = tmp_path / "clip_fallback.mp4"
    _synth_test_video(vid_path, duration=2.0)

    # Force OpenCV to raise
    with patch.dict("sys.modules", {"cv2": None}):
        img = _grab_frame_at_timestamp(vid_path, 0.5, 270, 480)
        assert img is not None
        assert img.width == 270
        assert img.height == 480


def test_inspect_rendered_video_sub_clips_and_inf_volume(tmp_path: Path) -> None:
    """Verify sub_clips timeline extraction and safe serialization with -inf dB volume."""
    vid_path = tmp_path / "clip_sub_clips.mp4"
    _synth_test_video(vid_path, duration=2.5)

    meta = {
        "sub_clips": [
            {"start": 10.0, "end": 12.0, "label": "Setup"},
            {"start": 18.0, "end": 20.0, "label": "Reaction"},
        ]
    }
    (tmp_path / "clip_sub_clips.json").write_text(json.dumps(meta), encoding="utf-8")

    from unittest.mock import patch
    # Simulate -inf dB volume from volumedetect
    with patch("subprocess.run") as mock_run:
        probe_json = json.dumps({
            "streams": [
                {"codec_type": "video", "width": 1080, "height": 1920, "r_frame_rate": "30/1", "duration": "2.5"},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "2.5"},
        })
        audio_out = "[volumedetect @ 1] mean_volume: -inf dB\n[volumedetect @ 1] max_volume: -inf dB\n"

        def side_effect(cmd, *args, **kwargs):
            if "ffprobe" in cmd[0]:
                class Proc:
                    stdout = probe_json
                    returncode = 0
                return Proc()
            elif "-af" in cmd:
                class Proc:
                    stderr = audio_out
                    returncode = 0
                return Proc()
            else:
                return subprocess.run(cmd, *args, **kwargs)

        mock_run.side_effect = side_effect
        res = inspect_rendered_video(vid_path)

        assert len(res["narration_timeline"]) == 2
        assert res["narration_timeline"][0]["role"] == "sub_clip"
        assert res["narration_timeline"][0]["heading"] == "Setup"
        assert res["audio_stats"]["mean_volume_db"] == -99.0
        assert res["audio_stats"]["max_volume_db"] == -99.0
        # Ensure it serializes cleanly to standard JSON without NaN or Infinity
        dumped = json.dumps(res)
        assert "-99.0" in dumped
