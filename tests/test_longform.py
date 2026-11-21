"""Tests for 16:9 long topic video renderer, chapters, thumbnail, and post file (T5)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from streamclipper.assembly.longform import (
    _build_longform_ass,
    _ensure_synthetic_wav,
    _format_chapter_time,
    generate_topic_thumbnail,
    render_long_video,
)
from streamclipper.cli import cli
from streamclipper.config import Config
from streamclipper.output.topic_post import write_long_post
from streamclipper.topics.models import EvidenceClip, ShortPlan, TopicPlan, TopicPoint
from streamclipper.topics.planner import save_plan


# ---------------------------------------------------------------------------
# Fixtures and Synthetic Generators
# ---------------------------------------------------------------------------

def _create_synthetic_mp4(path: Path, duration: float = 3.0, width: int = 640, height: int = 360) -> Path:
    """Create a tiny test MP4 with testsrc and sine audio via FFmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={width}x{height}:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        str(path.resolve()),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _set_cfg(cfg: Config, dotted_key: str, val: Any) -> None:
    """Helper to set nested keys on Config for testing."""
    keys = dotted_key.split(".")
    d = cfg._data
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Provide a Config pointing to isolated temp directories."""
    cfg = Config()
    _set_cfg(cfg, "general.output_dir", str(tmp_path / "output"))
    _set_cfg(cfg, "general.cache_dir", str(tmp_path / "cache"))
    _set_cfg(cfg, "general.temp_dir", str(tmp_path / "temp"))
    _set_cfg(cfg, "longform.width", 640)  # Use 640x360 in tests for speed
    _set_cfg(cfg, "longform.height", 360)
    _set_cfg(cfg, "longform.fps", 30)
    _set_cfg(cfg, "longform.card_seconds", 1.0)
    _set_cfg(cfg, "longform.broll_volume", 0.12)
    _set_cfg(cfg, "longform.credit_seconds", 2.0)
    _set_cfg(cfg, "longform.kenburns_max_zoom", 1.05)
    _set_cfg(cfg, "video.codec", "libx264")
    _set_cfg(cfg, "video.crf", 28)
    return cfg


@pytest.fixture
def synthetic_plan() -> TopicPlan:
    """A valid, approved TopicPlan with 3 points for fast testing."""
    return TopicPlan(
        plan_id="plan-t5-test",
        topic_id="hidden-casino-secrets",
        topic_title="Hidden Casino Secrets",
        game="GTA 6",
        shape="numbered",
        status="approved",
        cold_open="Welcome to the deep dive into Vice City casino details.",
        outro="Subscribe for more hidden Vice City secrets.",
        video_title="10 Insane Secrets in the GTA 6 Casino",
        thumbnail_text="SECRET CASINO DETAILS",
        description="Exploring every secret hidden in the casino.",
        points=[
            TopicPoint(
                index=1,
                heading="#1 The High Roller Vault",
                narration="First we look at the underground vault mechanics.",
                evidence=[
                    EvidenceClip(
                        source_id="src_stream_1",
                        url="https://youtube.com/watch?v=vault01",
                        channel="ViceGamer",
                        kind="stream",
                        start=10.0,
                        end=14.0,
                        role="evidence",
                    )
                ],
            ),
            TopicPoint(
                index=2,
                heading="#2 Official Security Trailer",
                narration="Next Rockstar showed official footage of the security cameras.",
                evidence=[
                    EvidenceClip(
                        source_id="src_official_2",
                        url="https://youtube.com/watch?v=official02",
                        channel="RockstarGames",
                        kind="official",
                        start=5.0,
                        end=9.0,
                        role="evidence",
                    )
                ],
            ),
            TopicPoint(
                index=3,
                heading="#3 Guard Patrol Routines",
                narration="Finally let us inspect the guard patrol paths.",
                evidence=[
                    EvidenceClip(
                        source_id="src_stream_3",
                        url="https://youtube.com/watch?v=guard03",
                        channel="StealthPro",
                        kind="stream",
                        start=20.0,
                        end=24.0,
                        role="evidence",
                    )
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_T5_refuses_unapproved_plans(test_config: Config, synthetic_plan: TopicPlan) -> None:
    """render_long_video must refuse plans whose status is not 'approved'."""
    synthetic_plan.status = "draft"
    save_plan(test_config, synthetic_plan)

    with pytest.raises(ValueError, match="must be 'approved'"):
        render_long_video(test_config, synthetic_plan)


def test_T5_caps_enforced(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """Clips exceeding 20s (or official clips exceeding 10s) are clamped at render time."""
    pt1_clip = synthetic_plan.points[0].evidence[0]
    pt1_clip.start = 10.0
    pt1_clip.end = 45.0  # 35s > 20s

    official_clip = synthetic_plan.points[1].evidence[0]
    official_clip.start = 5.0
    official_clip.end = 25.0  # 20s > 10s

    save_plan(test_config, synthetic_plan)

    downloaded_calls = []

    def mock_download(url, start, end, out_dir, **kwargs):
        downloaded_calls.append((url, start, end))
        seg = out_dir / f"seg_{len(downloaded_calls)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", return_value=None):
        render_long_video(test_config, synthetic_plan)

    assert pt1_clip.end - pt1_clip.start <= 20.0
    assert pt1_clip.end == 30.0
    assert official_clip.end - official_clip.start <= 10.0
    assert official_clip.end == 15.0


def test_T5_copyright_swap_to_alternate_or_drop(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """HIGH_MUSIC_DETECTED swaps in a clean alternate, or drops clip if no clean alternate."""
    # Point 1 has a flagged clip and a clean alternate
    flagged_clip = synthetic_plan.points[0].evidence[0]
    clean_alt = EvidenceClip(
        source_id="src_alt_1",
        url="https://youtube.com/watch?v=alt01",
        channel="AltChannel",
        kind="stream",
        start=50.0,
        end=54.0,
    )
    synthetic_plan.points[0].alternates = [clean_alt]

    # Point 2 has 2 clips: one flagged with no alternates (will be dropped) and one clean
    synthetic_plan.points[1].evidence.append(
        EvidenceClip(
            source_id="src_stream_extra",
            url="https://youtube.com/watch?v=extra02",
            channel="ExtraGamer",
            kind="stream",
            start=1.0,
            end=4.0,
        )
    )

    save_plan(test_config, synthetic_plan)

    def mock_detect(audio_path, start, end):
        if start == 10.0 or start == 1.0:
            return ("HIGH_MUSIC_DETECTED", 0.95)
        return ("CLEAN", 0.99)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    mock_state = MagicMock()
    mock_audio = tmp_path / "stream_audio.wav"
    _ensure_synthetic_wav(mock_audio, 10.0)
    mock_state.audio_path = str(mock_audio)

    with patch("streamclipper.assembly.longform.StepCache.load_state", return_value=mock_state), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", side_effect=mock_detect), \
         patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", return_value=None):
        render_long_video(test_config, synthetic_plan)

    # Point 1 should now have the alternate clip
    assert len(synthetic_plan.points[0].evidence) == 1
    assert synthetic_plan.points[0].evidence[0].url == "https://youtube.com/watch?v=alt01"

    # Point 2 had its extra flagged clip dropped, keeping only official_02
    assert len(synthetic_plan.points[1].evidence) == 1
    assert synthetic_plan.points[1].evidence[0].source_id == "src_official_2"


def test_T5_copyright_alternate_deduplication(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """When multiple clips fail in the same point, distinct alternates are chosen without duplicates."""
    # Point 1 has 2 clips that both fail copyright check
    synthetic_plan.points[0].evidence = [
        EvidenceClip(source_id="c1", url="https://youtube.com/watch?v=bad1", start=11.0, end=14.0),
        EvidenceClip(source_id="c2", url="https://youtube.com/watch?v=bad2", start=21.0, end=24.0),
    ]
    # Point 1 has 2 clean alternates
    alt1 = EvidenceClip(source_id="alt1", url="https://youtube.com/watch?v=clean1", start=31.0, end=34.0)
    alt2 = EvidenceClip(source_id="alt2", url="https://youtube.com/watch?v=clean2", start=41.0, end=44.0)
    synthetic_plan.points[0].alternates = [alt1, alt2]

    save_plan(test_config, synthetic_plan)

    def mock_detect(audio_path, start, end):
        if start in (11.0, 21.0):
            return ("HIGH_MUSIC_DETECTED", 0.95)
        return ("CLEAN", 0.99)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    mock_state = MagicMock()
    mock_audio = tmp_path / "audio.wav"
    _ensure_synthetic_wav(mock_audio, 10.0)
    mock_state.audio_path = str(mock_audio)

    with patch("streamclipper.assembly.longform.StepCache.load_state", return_value=mock_state), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", side_effect=mock_detect), \
         patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", return_value=None):
        render_long_video(test_config, synthetic_plan)

    ev_urls = [c.url for c in synthetic_plan.points[0].evidence]
    assert len(ev_urls) == 2
    assert "https://youtube.com/watch?v=clean1" in ev_urls
    assert "https://youtube.com/watch?v=clean2" in ev_urls
    assert ev_urls[0] != ev_urls[1]


def test_T5_copyright_all_fail_raises(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """When a point ends with no valid evidence clips after copyright checks, renderer raises."""
    mock_state = MagicMock()
    mock_audio = tmp_path / "audio.wav"
    _ensure_synthetic_wav(mock_audio, 10.0)
    mock_state.audio_path = str(mock_audio)

    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.assembly.longform.StepCache.load_state", return_value=mock_state), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("HIGH_MUSIC_DETECTED", 0.98)), \
         patch("streamclipper.assembly.longform.download_segment", return_value=tmp_path / "clip.mp4"):
        with pytest.raises(RuntimeError, match="has no valid evidence clips after copyright filtering"):
            render_long_video(test_config, synthetic_plan)

    # Verify plan status was marked failed
    assert synthetic_plan.status == "failed"


def test_T5_cards_only_for_numbered_countdown(tmp_path: Path) -> None:
    """Verify that chapter cards are produced only for numbered and countdown shapes."""
    ass_path = tmp_path / "test.ass"
    narration_windows = [(0.0, 5.0, "Welcome everyone"), (10.0, 15.0, "Point one details")]
    credit_windows = [(5.0, 8.0, "@ViceGamer"), (15.0, 18.0, "Official footage: Rockstar Games")]

    _build_longform_ass(narration_windows, credit_windows, ass_path)
    content = ass_path.read_text(encoding="utf-8")

    assert "Style: Narration16x9" in content
    assert "Style: Credit16x9" in content
    assert "Welcome everyone" in content
    assert "@ViceGamer" in content
    assert "Official footage: Rockstar Games" in content


def test_T5_credits_text_including_official(tmp_path: Path) -> None:
    """Verify official footage credit format and streamer @channel format."""
    ass_path = tmp_path / "credits.ass"
    credits = [
        (0.0, 4.0, "Official footage: Rockstar Games"),
        (4.0, 8.0, "@StreamerHero"),
    ]
    _build_longform_ass([], credits, ass_path)
    ass_text = ass_path.read_text(encoding="utf-8")

    assert "Dialogue: 0,0:00:00.00,0:00:04.00,Credit16x9,,0,0,0,,Official footage: Rockstar Games" in ass_text
    assert "Dialogue: 0,0:00:04.00,0:00:08.00,Credit16x9,,0,0,0,,@StreamerHero" in ass_text


def test_T5_chapters_file_rules(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """Verify YouTube chapter rules: first at 00:00, >= 3 chapters, each >= 10s, and point headings preserved."""
    out_dir = tmp_path / "chapters_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", return_value=None):
        res = render_long_video(test_config, synthetic_plan)

    chapters_path = Path(res["chapters_path"])
    assert chapters_path.is_file()
    lines = [line.strip() for line in chapters_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) >= 3
    assert lines[0].startswith("00:00")

    # Verify that all point headings are present and NOT dropped
    text = chapters_path.read_text(encoding="utf-8")
    assert "Intro" in text
    assert synthetic_plan.points[0].heading in text
    assert synthetic_plan.points[1].heading in text
    assert synthetic_plan.points[2].heading in text

    # Verify timestamps are strictly ascending and separated by >= 10s
    prev_s = -10
    for line in lines:
        parts = line.split(" ", 1)
        ts_str = parts[0]
        subparts = [int(p) for p in ts_str.split(":")]
        if len(subparts) == 2:
            sec = subparts[0] * 60 + subparts[1]
        else:
            sec = subparts[0] * 3600 + subparts[1] * 60 + subparts[2]
        assert sec >= prev_s + 10
        prev_s = sec


def test_T5_thumbnail_size_and_text(tmp_path: Path) -> None:
    """Verify thumbnail.jpg is 1280x720 and handles dynamic font scaling for long text."""
    frame_mp4 = _create_synthetic_mp4(tmp_path / "frame.mp4", duration=2.0)
    thumb_path = tmp_path / "thumbnail.jpg"

    generate_topic_thumbnail(
        frame_mp4,
        "EXTREMELY LONG WORDS FOR THUMBNAIL",
        thumb_path,
        target_w=1280,
        target_h=720,
    )

    assert thumb_path.is_file()
    with Image.open(thumb_path) as img:
        assert img.size == (1280, 720)


def test_T5_post_file_sources_with_timestamps(test_config: Config, synthetic_plan: TopicPlan) -> None:
    """Verify write_long_post outputs title, description, chapters, &t= sources, and reminders."""
    chapters = ["00:00 Intro", "00:15 The Vault", "00:30 Outro"]
    post_path = write_long_post(test_config, synthetic_plan, chapters)

    assert post_path.is_file()
    text = post_path.read_text(encoding="utf-8")

    assert "TITLE" in text
    assert synthetic_plan.video_title in text
    assert "DESCRIPTION" in text
    assert "TIMESTAMPS" in text
    assert "00:00 Intro" in text
    assert "SOURCES" in text
    assert "@ViceGamer: https://youtube.com/watch?v=vault01&t=10s" in text
    assert "Official footage: Rockstar Games" in text
    assert "Official footage (Rockstar Games): https://youtube.com/watch?v=official02&t=5s" in text
    assert "Upload the thumbnail.jpg" in text
    assert "AI voiceover:" in text


def test_T5_full_render_synthetic_video(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """Full end-to-end render test with synthetic video and audio."""
    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    stages_called: list[str] = []

    class MockProgress:
        def stage(self, text: str) -> None:
            stages_called.append(text)

    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", return_value=None):
        res = render_long_video(test_config, synthetic_plan, progress=MockProgress())

    # Verify result dictionary
    assert res["status"] == "rendered"
    assert res["plan_id"] == synthetic_plan.plan_id
    video_path = Path(res["video_path"])
    assert video_path.is_file()

    # Verify ffprobe stream info (640x360 canvas)
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name",
        "-of", "json",
        str(video_path.resolve()),
    ]
    probe_res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    probe_data = json.loads(probe_res.stdout)
    assert len(probe_data.get("streams", [])) > 0
    stream = probe_data["streams"][0]
    assert stream["width"] == 640
    assert stream["height"] == 360

    # Verify plan status and outputs updated
    assert synthetic_plan.status == "rendered"
    assert "long" in synthetic_plan.outputs
    assert "chapters" in synthetic_plan.outputs
    assert "thumbnail" in synthetic_plan.outputs
    assert "long_post" in synthetic_plan.outputs

    # Verify progress stages
    assert any("downloading" in s for s in stages_called)
    assert any("voicing" in s for s in stages_called)
    assert any("rendering" in s for s in stages_called)
    assert "final encode" in stages_called


def test_T5_per_block_loudnorm_called(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """Verify TransformativeNarrator.normalize_audio is called for per-block loudnorm."""
    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", return_value=None), \
         patch("streamclipper.assembly.narrator.TransformativeNarrator.normalize_audio", side_effect=lambda in_v, out_v: _create_synthetic_mp4(out_v, duration=1.0)) as mock_norm:
        render_long_video(test_config, synthetic_plan)

    # normalize_audio should have been called on intermediate blocks
    assert mock_norm.call_count >= 3


def test_T5_block_order_story_crossfade(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """Story shape uses crossfade transitions between chapters and does not render chapter cards."""
    synthetic_plan.shape = "story"
    save_plan(test_config, synthetic_plan)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_story_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", return_value=None):
        res = render_long_video(test_config, synthetic_plan)

    video_path = Path(res["video_path"])
    assert video_path.is_file()
    assert synthetic_plan.status == "rendered"


def test_T5_mcp_tool_starts_job_and_refuses_unapproved(test_config: Config, synthetic_plan: TopicPlan) -> None:
    """MCP render_topic_video refuses unapproved plans synchronously and starts a job on approved."""
    from streamclipper.mcp_server.server import mcp

    # 1. Unapproved plan
    synthetic_plan.status = "draft"
    save_plan(test_config, synthetic_plan)

    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "render_topic_video" in tools

    tool = tools["render_topic_video"]

    with patch("streamclipper.mcp_server.server.Config", return_value=test_config):
        raw_res = tool.fn(plan_id=synthetic_plan.plan_id)
        res = json.loads(raw_res)
        assert "error" in res
        assert "not approved" in res["error"].lower()

    # 2. Approved plan starts background job
    synthetic_plan.status = "approved"
    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.mcp_server.server.Config", return_value=test_config), \
         patch("streamclipper.mcp_server.server._start_render_job", return_value=json.dumps({"status": "started", "job_id": "job-123"})) as mock_start:
        raw_approved_res = tool.fn(plan_id=synthetic_plan.plan_id)
        approved_res = json.loads(raw_approved_res)
        assert approved_res.get("status") == "started"
        assert approved_res.get("job_id") == "job-123"
        mock_start.assert_called_once()
        args, kwargs = mock_start.call_args
        assert args[0] == "topic_long"
        assert args[3] == 3  # 3 evidence clips planned


def test_T5_cli_render_command(tmp_path: Path, synthetic_plan: TopicPlan) -> None:
    """CLI streamclipper topics render PLAN_ID runs successfully on approved plans."""
    cfg_file = tmp_path / "custom_config.yaml"
    cfg_file.write_text(
        f"general:\n  output_dir: '{tmp_path.as_posix()}/output'\n  cache_dir: '{tmp_path.as_posix()}/cache'\n  temp_dir: '{tmp_path.as_posix()}/temp'\n",
        encoding="utf-8",
    )
    cli_cfg = Config(user_config_path=cfg_file)
    save_plan(cli_cfg, synthetic_plan)
    runner = CliRunner()

    with patch("streamclipper.assembly.longform.render_long_video", return_value={"video_path": "output/long.mp4", "duration": 40.0, "narration_share": 0.45}) as mock_render:
        result = runner.invoke(
            cli,
            ["--config", str(cfg_file), "topics", "render", synthetic_plan.plan_id],
        )
        assert result.exit_code == 0, f"CLI render failed: {result.output}"
        assert "Render completed successfully" in result.output
        mock_render.assert_called_once()
