"""Tests for output/variety.py, output/post_writer.py and organizer queued paths (v3 Batch E)."""
from __future__ import annotations

import json
from pathlib import Path

from streamclipper.config import Config, HardwareProfile
from streamclipper.models import ClipMetadata, Platform, StreamInfo
from streamclipper.output.organizer import FileOrganizer
from streamclipper.output.post_writer import write_post_file
from streamclipper.output.variety import VarietyState


def _meta() -> ClipMetadata:
    return ClipMetadata(
        hook_title="Nobody noticed this",
        caption="Crazy physics.",
        hashtags=["GTA6", "#Shorts"],
        streamer="Fleet",
        game="GTA 6",
        source_url="https://youtube.com/watch?v=abc",
        clip_start=1.0,
        clip_end=30.0,
        highlight_score=9.0,
    )


def _config(tmp_path: Path) -> Config:
    hw = HardwareProfile(
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
    cfg = Config(hardware_profile=hw)
    cfg._data.setdefault("general", {})["output_dir"] = str(tmp_path / "out")
    return cfg


def test_E8_next_format_rotates_least_recent(tmp_path: Path) -> None:
    vs = VarietyState(tmp_path / "variety_state.json")
    fmts = ["narrative", "story", "breakdown"]
    seen = []
    for _ in range(6):
        f = vs.next_format(fmts)
        seen.append(f)
        vs.record("Fleet", f, vs.next_hook_style(), "outro")
    assert seen[:3] == ["narrative", "story", "breakdown"]
    assert seen[3:] == seen[:3]


def test_E8_hook_style_rotation_and_recent_outros(tmp_path: Path) -> None:
    vs = VarietyState(tmp_path / "variety_state.json")
    styles = []
    for i in range(4):
        s = vs.next_hook_style()
        styles.append(s)
        vs.record("Fleet", "narrative", s, f"outro {i}")
    assert len(set(styles)) == 4
    assert vs.recent_outros(2) == ["outro 2", "outro 3"]


def test_E8_clips_today_counts_per_streamer_and_file_is_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "variety_state.json"
    vs = VarietyState(path)
    vs.record("Fleet", "story", "question", "o")
    vs.record("Fleet", "story", "question", "o")
    vs.record("Other", "story", "question", "o")
    assert vs.clips_today("Fleet") == 2
    assert vs.clips_today("Nobody") == 0
    assert len(json.loads(path.read_text(encoding="utf-8"))["entries"]) == 3
    assert not list(path.parent.glob("*.tmp"))


def test_E8_organizer_queued_path_under_queued_dir(tmp_path: Path) -> None:
    org = FileOrganizer(_config(tmp_path))
    normal = org.get_clip_path("GTA 6", "Fleet", 1, date_str="2026-01-01")
    queued = org.get_queued_path("GTA 6", "Fleet", 1, date_str="2026-01-01")
    assert queued.parent.parent.parent.parent.name == "queued"
    assert queued.name == normal.name
    assert queued != normal


def test_E11_post_file_contents(tmp_path: Path) -> None:
    video = tmp_path / "clip_01.mp4"
    video.write_bytes(b"x")
    info = StreamInfo(
        url="https://youtube.com/watch?v=abc",
        title="t",
        platform=Platform.YOUTUBE,
        streamer="Fleet",
        game="GTA 6",
        duration_seconds=10,
    )
    out = write_post_file(video, _meta(), info)
    assert out.name == "clip_01_post.txt"
    text = out.read_text(encoding="utf-8")
    assert "Nobody noticed this" in text
    assert "Original stream by Fleet: https://youtube.com/watch?v=abc" in text
    assert "#GTA6 #Shorts" in text
    assert "Tick YouTube's 'Altered or synthetic content' disclosure (AI voice)" in text
    assert "Post at most 2–3 Shorts per day" in text
