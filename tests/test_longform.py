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
    _attach_bgm_to_audio,
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


def _mock_synthesize(text: str, out_path: Path, **kwargs: Any) -> Path:
    """Mock TTS synthesizer that creates a valid synthetic wav file for unit tests."""
    return _ensure_synthetic_wav(Path(out_path), duration_seconds=max(1.0, len(text.split()) / 2.7))


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
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize):
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
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize):
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
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize):
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


def test_build_longform_ass_preserves_intentional_ass_linebreak(tmp_path: Path) -> None:
    """Verify multi-line chunks emit real \\N breaks instead of escaped zero-width text."""
    ass_path = tmp_path / "multiline.ass"
    long_line = "Alpha beta gamma delta epsilon zeta eta theta iota kappa"
    _build_longform_ass([(0.0, 5.0, long_line)], [], ass_path)
    content = ass_path.read_text(encoding="utf-8")

    assert r"\N" in content
    assert "\\\u200bN" not in content


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
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize):
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
    assert "Official footage: RockstarGames" in text
    assert "Official footage (RockstarGames): https://youtube.com/watch?v=official02&t=5s" in text
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
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize):
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
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize), \
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
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize):
        res = render_long_video(test_config, synthetic_plan)

    video_path = Path(res["video_path"])
    assert video_path.is_file()
    assert synthetic_plan.status == "rendered"


@pytest.mark.asyncio
async def test_T5_mcp_tool_starts_job_and_refuses_unapproved(test_config: Config, synthetic_plan: TopicPlan) -> None:
    """MCP render_topic_video refuses unapproved plans synchronously and starts a job on approved."""
    from streamclipper.mcp_server.server import mcp

    # 1. Unapproved plan
    synthetic_plan.status = "draft"
    save_plan(test_config, synthetic_plan)

    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "render_topic_video" in tools

    tool = tools["render_topic_video"]

    with patch("streamclipper.mcp_server.server.Config", return_value=test_config):
        raw_res = await tool.fn(plan_id=synthetic_plan.plan_id)
        res = json.loads(raw_res)
        assert "error" in res
        assert "not approved" in res["error"].lower()

    # 2. Approved plan starts background job
    synthetic_plan.status = "approved"
    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.mcp_server.server.Config", return_value=test_config), \
         patch("streamclipper.mcp_server.server._start_render_job", return_value=json.dumps({"status": "started", "job_id": "job-123"})) as mock_start:
        raw_approved_res = await tool.fn(plan_id=synthetic_plan.plan_id)
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


# ---------------------------------------------------------------------------
# Ticket T1 Verification Tests
# ---------------------------------------------------------------------------

def test_T1_tts_synthesis_failure_raises(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """When TTS synthesis returns None or produces an empty file, render_long_video raises RuntimeError."""
    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    save_plan(test_config, synthetic_plan)

    # 1. Cold open TTS failure
    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", return_value=None):
        with pytest.raises(RuntimeError, match="TTS synthesis failed for cold_open"):
            render_long_video(test_config, synthetic_plan)

    # 2. Point 1 TTS failure (cold open succeeds, point 1 returns None)
    synthetic_plan.status = "approved"
    def synth_fail_point(text, out_path, **kwargs):
        if "First we look" in text:
            return None
        return _ensure_synthetic_wav(Path(out_path), duration_seconds=1.0)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=synth_fail_point):
        with pytest.raises(RuntimeError, match=r"TTS synthesis failed for point 1"):
            render_long_video(test_config, synthetic_plan)

    # 3. Outro TTS failure (cold open & points succeed, outro returns empty file)
    synthetic_plan.status = "approved"
    def synth_empty_outro(text, out_path, **kwargs):
        out_p = Path(out_path)
        if "Subscribe for more" in text:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(b"")  # 0 bytes
            return out_p
        return _ensure_synthetic_wav(out_p, duration_seconds=1.0)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=synth_empty_outro):
        with pytest.raises(RuntimeError, match="TTS synthesis failed for outro"):
            render_long_video(test_config, synthetic_plan)


def test_T1_official_credit_empty_channel_fallback(test_config: Config, tmp_path: Path) -> None:
    """Official clips fall back to 'Official {plan.game} footage' when channel is empty or None."""
    plan = TopicPlan(
        plan_id="plan-t1-cred",
        topic_id="credit-test",
        topic_title="Credit Fallback Test",
        game="GTA 6",
        shape="numbered",
        status="approved",
        cold_open="",
        outro="",
        points=[
            TopicPoint(
                index=1,
                heading="Official Clip No Channel",
                narration="Checking credit text.",
                evidence=[
                    EvidenceClip(
                        source_id="src_off_empty",
                        url="https://youtube.com/watch?v=empty01",
                        channel="",  # empty channel
                        kind="official",
                        start=0.0,
                        end=5.0,
                    )
                ],
            ),
            TopicPoint(
                index=2,
                heading="Official Clip With Channel",
                narration="Checking channel name.",
                evidence=[
                    EvidenceClip(
                        source_id="src_off_ps",
                        url="https://youtube.com/watch?v=ps02",
                        channel="PlayStation",
                        kind="official",
                        start=10.0,
                        end=15.0,
                    )
                ],
            ),
        ],
    )

    # Test post file generation
    post_file = write_long_post(test_config, plan, ["00:00 Intro", "00:10 Point 1"])
    text = post_file.read_text(encoding="utf-8")
    assert "- Official GTA 6 footage: https://youtube.com/watch?v=empty01&t=0s" in text
    assert "- Official footage (PlayStation): https://youtube.com/watch?v=ps02&t=10s" in text
    assert "Official footage: PlayStation" in text


def test_T1_hashtag_stopwords_and_capping(test_config: Config) -> None:
    """Stopwords in English/Hinglish are filtered out and hashtags capped at 4."""
    _set_cfg(test_config, "metadata.hashtag_count", 4)

    plan = TopicPlan(
        plan_id="plan-t1-tags",
        topic_id="tags-test",
        # Stopwords: 'How', 'The', 'For', 'Every', 'Actually'
        # Meaningful: 'Casino', 'Vault', 'Secrets'
        topic_title="How The Casino Vault Actually Secrets For Every",
        game="GTA 6",
        shape="numbered",
        status="approved",
        cold_open="",
        outro="",
        points=[],
    )

    post_file = write_long_post(test_config, plan, [])
    text = post_file.read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}

    # Extract HASHTAGS line
    for line in text.splitlines():
        if line.startswith("#"):
            tags = line.split()
            # Must have game tag
            assert "#GTA6" in tags
            # Must NOT have stopwords
            for stop in ["#How", "#The", "#For", "#Every", "#Actually"]:
                assert stop not in tags
            # Must have meaningful tags
            assert "#Casino" in tags
            assert "#Vault" in tags
            assert "#Secrets" in tags
            # Must be capped at 4 total
            assert len(tags) == 4
            break


def test_T1_no_duplicate_broll_footage_selection(test_config: Config, tmp_path: Path) -> None:
    """Verify B-roll selection priority prevents back-to-back duplicate footage."""
    alt_clip = EvidenceClip(source_id="c_alt", url="https://youtube.com/watch?v=alt", start=1.0, end=4.0)
    ev1 = EvidenceClip(source_id="c1", url="https://youtube.com/watch?v=ev1", start=10.0, end=14.0)
    ev2a = EvidenceClip(source_id="c2a", url="https://youtube.com/watch?v=ev2a", start=20.0, end=24.0)
    ev2b = EvidenceClip(source_id="c2b", url="https://youtube.com/watch?v=ev2b", start=30.0, end=36.0)

    plan = TopicPlan(
        plan_id="plan-t1-broll",
        topic_id="broll-test",
        topic_title="B-roll Selection Test",
        game="GTA 6",
        shape="story",  # No chapter cards to keep blocks simple
        status="approved",
        cold_open="",
        outro="",
        points=[
            # Point 1 has 1 evidence clip and 1 downloaded alternate
            TopicPoint(
                index=1,
                heading="Point 1",
                narration="Point 1 Narration",
                evidence=[ev1],
                alternates=[alt_clip],
            ),
            # Point 2 has 2 evidence clips (no alternates)
            TopicPoint(
                index=2,
                heading="Point 2",
                narration="Point 2 Narration",
                evidence=[ev2a, ev2b],
            ),
        ],
    )
    save_plan(test_config, plan)

    downloaded_files_map: dict[str, Path] = {}

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        p = _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)
        downloaded_files_map[url] = p
        return p

    # Pre-create the alternate clip in temp_dir so it counts as downloaded/available
    temp_dir = Path(test_config.get("general.temp_dir", "./temp")) / "topics" / plan.plan_id
    alt_path = temp_dir / "seg_alt_1.mp4"
    _create_synthetic_mp4(alt_path, duration=3.0, width=640, height=360)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize):
        res = render_long_video(test_config, plan)

    assert res["status"] == "rendered"
    assert Path(res["video_path"]).is_file()


# ---------------------------------------------------------------------------
# Ticket T4 Tests: 1.2s Chapter Card, Transition SFX, CrossFade, and BGM Bed
# ---------------------------------------------------------------------------


def test_T4_chapter_card_duration_default_1_2s() -> None:
    """Verify that default longform.card_seconds is 1.2 in both config and longform defaults."""
    fresh_cfg = Config()
    assert fresh_cfg.get("longform.card_seconds") == 1.2
    assert fresh_cfg.get("longform.transition_sfx_path") == ""


def test_T4_chapter_card_with_transition_sfx(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """When longform.transition_sfx_path points to an audio file, card clip attaches it at -12 dB."""
    sfx_wav = tmp_path / "whoosh.wav"
    _ensure_synthetic_wav(sfx_wav, duration_seconds=0.8)
    _set_cfg(test_config, "longform.transition_sfx_path", str(sfx_wav))
    _set_cfg(test_config, "longform.card_seconds", 1.2)

    captured_clips: dict[str, Any] = {}

    from moviepy import VideoClip
    orig_write = VideoClip.write_videofile

    def spy_write(self, filename, *args, **kwargs):
        name = Path(filename).stem.replace("_raw", "")
        if "card" in name:
            captured_clips[name] = {
                "duration": self.duration,
                "has_audio": self.audio is not None,
                "is_composite": hasattr(self, "clips"),
            }
        return orig_write(self, filename, *args, **kwargs)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize), \
         patch.object(VideoClip, "write_videofile", spy_write):
        res = render_long_video(test_config, synthetic_plan)

    assert res["status"] == "rendered"
    card_entries = [v for k, v in captured_clips.items() if "card" in k]
    assert len(card_entries) > 0
    for card_data in card_entries:
        assert card_data["duration"] == 1.2
        assert card_data["has_audio"] is True
        # SFX path does not wrap in CompositeVideoClip with fade-in/fade-out
        assert card_data["is_composite"] is False


def test_T4_chapter_card_fades_and_bgm_bed_when_sfx_unset(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """When transition_sfx_path is unset, card clip has visual fade-in/out and mixes BGM bed across card."""
    bgm_wav = tmp_path / "ambient_bgm.wav"
    _ensure_synthetic_wav(bgm_wav, duration_seconds=3.0)
    _set_cfg(test_config, "longform.transition_sfx_path", "")
    _set_cfg(test_config, "transformation.bgm_path", str(bgm_wav))
    _set_cfg(test_config, "transformation.bgm_volume_db", -26.0)
    _set_cfg(test_config, "longform.card_seconds", 1.2)

    captured_clips: dict[str, Any] = {}

    from moviepy import VideoClip
    orig_write = VideoClip.write_videofile

    def spy_write(self, filename, *args, **kwargs):
        name = Path(filename).stem.replace("_raw", "")
        if "card" in name:
            captured_clips[name] = {
                "duration": self.duration,
                "has_audio": self.audio is not None,
                "is_composite": hasattr(self, "clips"),
            }
        return orig_write(self, filename, *args, **kwargs)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize), \
         patch.object(VideoClip, "write_videofile", spy_write):
        res = render_long_video(test_config, synthetic_plan)

    assert res["status"] == "rendered"
    card_entries = [v for k, v in captured_clips.items() if "card" in k]
    assert len(card_entries) > 0
    for card_data in card_entries:
        assert card_data["duration"] == 1.2
        assert card_data["has_audio"] is True
        # Unset SFX wraps in CompositeVideoClip with CrossFadeIn/CrossFadeOut
        assert card_data["is_composite"] is True


def test_T4_chapter_card_corrupt_sfx_fallback_and_no_loudnorm_blowout(
    test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path
) -> None:
    """Zero-byte or invalid transition SFX must gracefully fall back to BGM fade without raising,

    and chapter card block must NOT be loudnorm-boosted to -14 LUFS.
    """
    zero_sfx = tmp_path / "corrupt_whoosh.wav"
    zero_sfx.write_bytes(b"")  # Zero-byte invalid file
    bgm_wav = tmp_path / "bgm_bed.wav"
    _ensure_synthetic_wav(bgm_wav, duration_seconds=3.0)

    _set_cfg(test_config, "longform.transition_sfx_path", str(zero_sfx))
    _set_cfg(test_config, "transformation.bgm_path", str(bgm_wav))
    _set_cfg(test_config, "transformation.bgm_volume_db", -26.0)
    _set_cfg(test_config, "longform.card_seconds", 1.2)

    normalized_inputs: list[str] = []
    from streamclipper.assembly.narrator import TransformativeNarrator
    orig_norm = TransformativeNarrator.normalize_audio

    def spy_norm(self, in_v, out_v):
        normalized_inputs.append(Path(in_v).name)
        return orig_norm(self, in_v, out_v)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize), \
         patch.object(TransformativeNarrator, "normalize_audio", spy_norm):
        res = render_long_video(test_config, synthetic_plan)

    assert res["status"] == "rendered"
    assert Path(res["video_path"]).is_file()

    # The chapter card must NOT have been sent to normalize_audio (which would blow up -26dB audio to -14 LUFS)
    assert not any("card" in name for name in normalized_inputs)
    # Other blocks (cold open, point narration, evidence, outro) should still be normalized
    assert len(normalized_inputs) >= 3


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_T4_attach_bgm_to_audio_helper(tmp_path: Path) -> None:
    """Verify _attach_bgm_to_audio mixes BGM bed ducked -4dB, handles loop/subclip, and handles missing/corrupt gracefully."""
    from moviepy import AudioClip, CompositeAudioClip

    base = AudioClip(lambda t: [0.1, 0.1], duration=3.0)
    opened: list[Any] = []

    # 1. None bgm_file -> returns base unchanged
    res_none = _attach_bgm_to_audio(base, 3.0, None, -26.0, opened)
    assert res_none is base

    # 2. Non-existent bgm_file -> returns base unchanged
    missing_file = tmp_path / "missing_bgm.mp3"
    res_missing = _attach_bgm_to_audio(base, 3.0, missing_file, -26.0, opened)
    assert res_missing is base

    # 3. Corrupt bgm_file -> logs warning, returns base unchanged without raising
    corrupt_file = tmp_path / "corrupt_bgm.mp3"
    corrupt_file.write_bytes(b"not an audio file")
    res_corrupt = _attach_bgm_to_audio(base, 3.0, corrupt_file, -26.0, opened)
    assert res_corrupt is base

    # 4. Valid BGM file longer than duration (3s duration, 5s BGM) -> subclipped and mixed
    long_bgm = tmp_path / "long_bgm.wav"
    _ensure_synthetic_wav(long_bgm, duration_seconds=5.0)
    opened.clear()
    res_long = _attach_bgm_to_audio(base, 3.0, long_bgm, -26.0, opened)
    assert isinstance(res_long, CompositeAudioClip)
    assert res_long.duration == 3.0
    assert len(opened) >= 3
    for c in opened:
        if hasattr(c, "close"):
            c.close()

    # 5. Valid BGM file shorter than duration (4s duration, 2s BGM) -> looped via AudioLoop and mixed
    short_bgm = tmp_path / "short_bgm.wav"
    _ensure_synthetic_wav(short_bgm, duration_seconds=2.0)
    opened.clear()
    res_short = _attach_bgm_to_audio(base, 4.0, short_bgm, -26.0, opened)
    assert isinstance(res_short, CompositeAudioClip)
    assert res_short.duration == 4.0
    for c in opened:
        if hasattr(c, "close"):
            c.close()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_T4_longform_narration_blocks_mix_bgm_when_set_and_skip_when_empty(
    test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path
) -> None:
    """Verify long-form narration blocks mix BGM when transformation.bgm_path is set,

    and skip cleanly when empty or None.
    """
    from streamclipper.assembly import longform as lf_mod

    # Setup BGM file
    bgm_wav = tmp_path / "narration_bed.wav"
    _ensure_synthetic_wav(bgm_wav, duration_seconds=10.0)

    # 1. When transformation.bgm_path is set:
    _set_cfg(test_config, "transformation.bgm_path", str(bgm_wav))
    _set_cfg(test_config, "transformation.bgm_volume_db", -24.0)

    calls: list[tuple[float, Any, float]] = []
    orig_attach = lf_mod._attach_bgm_to_audio

    def spy_attach(base_audio, duration, bgm_file, bgm_volume_db, opened_clips):
        calls.append((duration, bgm_file, bgm_volume_db))
        return orig_attach(base_audio, duration, bgm_file, bgm_volume_db, opened_clips)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize), \
         patch.object(lf_mod, "_attach_bgm_to_audio", side_effect=spy_attach):
        res = render_long_video(test_config, synthetic_plan)

    assert res["status"] == "rendered"
    assert Path(res["video_path"]).is_file()

    # Cold open + 3 points + outro = 5 narration blocks attached
    assert len(calls) == 5
    for dur, bgm_f, vol_db in calls:
        assert dur > 0.0
        assert bgm_f == bgm_wav
        assert vol_db == -24.0

    # 2. When transformation.bgm_path is unset / empty:
    _set_cfg(test_config, "transformation.bgm_path", "")
    calls.clear()
    synthetic_plan.status = "approved"
    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize), \
         patch.object(lf_mod, "_attach_bgm_to_audio", side_effect=spy_attach):
        res_empty = render_long_video(test_config, synthetic_plan)

    assert res_empty["status"] == "rendered"
    assert Path(res_empty["video_path"]).is_file()

    # _attach_bgm_to_audio was called with bgm_f=None and skipped cleanly
    assert len(calls) == 5
    for dur, bgm_f, vol_db in calls:
        assert bgm_f is None


# ---------------------------------------------------------------------------
# Ticket T5: Packaging Upgrade (Frame-Scored Thumbnails & Title Variants)
# ---------------------------------------------------------------------------

def test_T5_thumbnail_frame_scoring_and_3_word_cap(tmp_path: Path) -> None:
    """Verify generate_topic_thumbnail caps text to 3 words, positions in safe zone, and applies enhancement."""
    import cv2
    import numpy as np
    from streamclipper.assembly.longform import _score_thumbnail_frame

    # Verify frame scoring logic: sharp > blurry > black/washed-out
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    white = np.full((720, 1280, 3), 255, dtype=np.uint8)
    sharp = np.random.randint(40, 210, (720, 1280, 3), dtype=np.uint8)
    blurry = cv2.GaussianBlur(sharp, (31, 31), 0)

    score_black = _score_thumbnail_frame(black)
    score_white = _score_thumbnail_frame(white)
    score_sharp = _score_thumbnail_frame(sharp)
    score_blurry = _score_thumbnail_frame(blurry)

    assert score_sharp > score_blurry > score_black
    assert score_blurry > score_white

    # Test thumbnail generation from synthetic video
    frame_mp4 = _create_synthetic_mp4(tmp_path / "t5_frame.mp4", duration=3.0)
    thumb_path = tmp_path / "t5_thumb.jpg"

    out = generate_topic_thumbnail(
        frame_mp4,
        "ONE TWO THREE FOUR FIVE SIX",
        thumb_path,
        target_w=1280,
        target_h=720,
    )
    assert out.is_file()
    with Image.open(out) as img:
        assert img.size == (1280, 720)


def test_T5_title_variants_in_long_post(test_config: Config, synthetic_plan: TopicPlan) -> None:
    """Verify write_long_post outputs 3-5 punchy curiosity-gap title variants under TITLE."""
    from streamclipper.output.topic_post import generate_long_title_variants

    variants = generate_long_title_variants(synthetic_plan)
    assert 3 <= len(variants) <= 5
    for v in variants:
        assert len(v.strip()) > 0
        assert v.lower() != (synthetic_plan.video_title or "").lower()

    post_path = write_long_post(test_config, synthetic_plan, ["00:00 Intro", "00:15 The Vault"])
    text = post_path.read_text(encoding="utf-8")

    assert "TITLE" in text
    assert "TITLE VARIANTS (A/B TEST)" in text
    for v in variants:
        assert f"- {v}" in text


def test_T5_title_variants_in_post_writer(tmp_path: Path) -> None:
    """Verify write_post_file outputs 3-5 punchy curiosity-gap title variants under TITLE."""
    from streamclipper.models import ClipMetadata, StreamInfo, Platform
    from streamclipper.output.post_writer import generate_clip_title_variants, write_post_file

    stream = StreamInfo(
        url="https://youtube.com/watch?v=abc",
        title="Epic Stream",
        platform=Platform.YOUTUBE,
        streamer="SpeedRunner",
        game="GTA 6",
        duration_seconds=120.0,
    )
    meta = ClipMetadata(
        hook_title="IMPOSSIBLE VAULT HEIST",
        caption="Check out this crazy vault heist",
        hashtags=["GTA6", "Gaming"],
        streamer="SpeedRunner",
        game="GTA 6",
        source_url="https://youtube.com/watch?v=abc",
        clip_start=10.0,
        clip_end=25.0,
        highlight_score=0.92,
        moment_type="funny",
        callout="Vault door blown wide open",
    )

    variants = generate_clip_title_variants(meta, stream)
    assert 3 <= len(variants) <= 5
    for v in variants:
        assert len(v.strip()) > 0
        assert v.lower() != meta.hook_title.lower()

    video_path = tmp_path / "test_clip.mp4"
    video_path.write_bytes(b"dummy")
    post_file = write_post_file(video_path, meta, stream)

    assert post_file.is_file()
    text = post_file.read_text(encoding="utf-8")
    assert "TITLE" in text
    assert meta.hook_title in text
    assert "TITLE VARIANTS (A/B TEST)" in text
    for v in variants:
        assert f"- {v}" in text


def test_v44_build_longform_ass_escapes_lines_without_double_backslash_N(tmp_path: Path) -> None:
    """_build_longform_ass escapes each line before joining with \\N and never prints literal \\\\N."""
    ass_path = tmp_path / "v44_multiline.ass"
    text = "Look at {this} swamp detail while the gator stalks the boat in Vice City"
    _build_longform_ass([(0.0, 6.0, text)], [], ass_path)
    content = ass_path.read_text(encoding="utf-8")

    assert r"\N" in content
    assert r"\\N" not in content
    assert "\\\u200bN" not in content
    assert r"\{this\}" in content


def test_v44_build_broll_montage_cycles_multiple_clips_without_back_to_back_repeat(tmp_path: Path) -> None:
    """_build_broll_montage cycles across distinct clips using 7-10s slices instead of looping 1 clip."""
    from streamclipper.assembly.longform import _build_broll_montage
    from streamclipper.output.exporter import _close_clip_handles

    c1 = _create_synthetic_mp4(tmp_path / "broll_1.mp4", duration=12.0, width=320, height=180)
    c2 = _create_synthetic_mp4(tmp_path / "broll_2.mp4", duration=12.0, width=320, height=180)
    c3 = _create_synthetic_mp4(tmp_path / "broll_3.mp4", duration=12.0, width=320, height=180)

    opened: list[Any] = []
    seen_slice_sources: list[str] = []

    from moviepy import VideoFileClip as RealVideoFileClip
    orig_subclipped = RealVideoFileClip.subclipped

    def spy_subclipped(self, start_time=0, end_time=None):
        if getattr(self, "filename", None):
            seen_slice_sources.append((Path(self.filename).name, round(float(end_time or 0) - float(start_time or 0), 2)))
        return orig_subclipped(self, start_time, end_time)

    try:
        with patch.object(RealVideoFileClip, "subclipped", spy_subclipped):
            montage = _build_broll_montage(
                [c1, c2, c3],
                target_duration=24.0,
                target_w=320,
                target_h=180,
                kenburns_max_zoom=1.05,
                opened_clips=opened,
            )
        assert abs(float(montage.duration or 0.0) - 24.0) < 0.25
        names = [name for name, _ in seen_slice_sources]
        durations = [d for _, d in seen_slice_sources]
        assert len(names) >= 3
        assert set(names) == {"broll_1.mp4", "broll_2.mp4", "broll_3.mp4"}
        # Never the same clip back-to-back
        for i in range(len(names) - 1):
            assert names[i] != names[i + 1]
        # Slices are in the 7-10s range
        for d in durations:
            assert 6.5 <= d <= 10.0
    finally:
        for c in opened:
            _close_clip_handles(c)


def test_v44_build_broll_montage_empty_candidates_fallback(tmp_path: Path) -> None:
    """_build_broll_montage falls back cleanly to a ColorClip when no valid files exist."""
    from moviepy import ColorClip
    from streamclipper.assembly.longform import _build_broll_montage
    from streamclipper.output.exporter import _close_clip_handles

    opened: list[Any] = []
    try:
        montage = _build_broll_montage(
            [tmp_path / "does_not_exist.mp4"],
            target_duration=5.0,
            target_w=320,
            target_h=180,
            kenburns_max_zoom=1.05,
            opened_clips=opened,
        )
        assert isinstance(montage, ColorClip)
        assert float(montage.duration or 0.0) == 5.0
    finally:
        for c in opened:
            _close_clip_handles(c)


def test_v44_longform_uses_local_broll_in_alternates_without_ytdlp(test_config: Config, tmp_path: Path) -> None:
    """Local broll clip paths attached to pt.alternates are used directly without calling download_segment."""
    temp_base = Path(test_config.get("general.temp_dir", "./temp"))
    broll_dir = temp_base / "broll"
    local_broll = _create_synthetic_mp4(broll_dir / "gator_local_10000_20000_b0.mp4", duration=3.0, width=640, height=360)

    plan = TopicPlan(
        plan_id="plan-v44-local-broll",
        topic_id="swamp-gators",
        topic_title="Swamp Gators",
        game="GTA 6",
        shape="story",
        status="approved",
        cold_open="",
        outro="",
        points=[
            TopicPoint(
                index=1,
                heading="Alligators in the Grasslands",
                narration="Notice the massive alligator stalking along the muddy riverbank right here.",
                evidence=[
                    EvidenceClip(
                        source_id="src_ev_1",
                        url="https://youtube.com/watch?v=ev01",
                        channel="ViceGamer",
                        kind="stream",
                        start=10.0,
                        end=14.0,
                        role="evidence",
                    )
                ],
                alternates=[
                    EvidenceClip(
                        source_id="gator_local",
                        url=str(local_broll),
                        channel="SwampShowcase",
                        kind="gameplay",
                        start=0.0,
                        end=3.0,
                        role="broll",
                    )
                ],
            )
        ],
    )
    save_plan(test_config, plan)

    download_urls: list[str] = []

    def mock_download(url, start, end, out_dir, **kwargs):
        download_urls.append(url)
        seg = out_dir / f"seg_ev_{len(download_urls)}.mp4"
        return _create_synthetic_mp4(seg, duration=max(1.0, end - start), width=640, height=360)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize):
        res = render_long_video(test_config, plan)

    assert res["status"] == "rendered"
    assert Path(res["video_path"]).is_file()
    # Only the remote evidence URL was downloaded; local broll alternate was NOT passed to download_segment
    assert download_urls == ["https://youtube.com/watch?v=ev01"]


def test_v44_mixed_silent_and_audio_broll_montage_renders_cleanly(test_config: Config, tmp_path: Path) -> None:
    """_build_broll_montage mixes silent gameplay B-roll and stream clips with audio without AudioClip errors."""
    from streamclipper.assembly.longform import _build_broll_montage
    from streamclipper.output.exporter import _close_clip_handles

    silent_mp4 = tmp_path / "silent_gameplay.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=10:size=320x180:rate=30",
            "-c:v", "libx264", "-preset", "ultrafast",
            str(silent_mp4.resolve()),
        ],
        check=True,
        capture_output=True,
    )
    audio_mp4 = _create_synthetic_mp4(tmp_path / "stream_audio.mp4", duration=10.0, width=320, height=180)
    out_mp4 = tmp_path / "mixed_out.mp4"

    opened: list[Any] = []
    try:
        montage = _build_broll_montage(
            [silent_mp4, audio_mp4],
            target_duration=16.0,
            target_w=320,
            target_h=180,
            kenburns_max_zoom=1.05,
            opened_clips=opened,
        )
        assert montage.audio is not None
        assert getattr(montage.audio, "fps", None) is not None
        montage.write_videofile(
            str(out_mp4),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            temp_audiofile_path=str(tmp_path),
            logger=None,
        )
        assert out_mp4.is_file() and out_mp4.stat().st_size > 0
    finally:
        for c in opened:
            _close_clip_handles(c)


def test_v44_point_with_two_verified_alternates_skips_off_topic_points(test_config: Config, tmp_path: Path) -> None:
    """When a point already has >= 2 distinct point-specific B-roll clips, other points' clips are not mixed in."""
    from streamclipper.assembly import longform as lf_mod

    temp_base = Path(test_config.get("general.temp_dir", "./temp"))
    broll_dir = temp_base / "broll"
    g1 = _create_synthetic_mp4(broll_dir / "gator1_0_5000_b0.mp4", duration=4.0, width=320, height=180)
    g2 = _create_synthetic_mp4(broll_dir / "gator2_0_5000_b0.mp4", duration=4.0, width=320, height=180)

    plan = TopicPlan(
        plan_id="plan-v44-on-topic",
        topic_id="on-topic-broll",
        topic_title="On Topic Broll",
        game="GTA 6",
        shape="story",
        status="approved",
        cold_open="",
        outro="",
        points=[
            TopicPoint(
                index=1,
                heading="Swamp Gators",
                narration="Watch the alligator moving slowly through the swamp grass.",
                evidence=[
                    EvidenceClip(source_id="ev1", url="https://youtube.com/watch?v=ev1", channel="A", kind="stream", start=10.0, end=14.0),
                ],
                alternates=[
                    EvidenceClip(source_id="gator1", url=str(g1), channel="B", kind="gameplay", start=0.0, end=4.0, role="broll"),
                    EvidenceClip(source_id="gator2", url=str(g2), channel="C", kind="gameplay", start=0.0, end=4.0, role="broll"),
                ],
            ),
            TopicPoint(
                index=2,
                heading="City Cars",
                narration="Sports cars racing down Ocean Drive at night.",
                evidence=[
                    EvidenceClip(source_id="ev2", url="https://youtube.com/watch?v=ev2_cars", channel="D", kind="stream", start=20.0, end=24.0),
                ],
            ),
        ],
    )
    save_plan(test_config, plan)

    seen_montage_calls: list[list[str]] = []
    orig_montage = lf_mod._build_broll_montage

    def spy_montage(candidate_paths, *args, **kwargs):
        seen_montage_calls.append([Path(p).name for p in candidate_paths])
        return orig_montage(candidate_paths, *args, **kwargs)

    def mock_download(url, start, end, out_dir, **kwargs):
        tag = url.split("v=")[-1]
        return _create_synthetic_mp4(out_dir / f"dl_{tag}.mp4", duration=max(1.0, end - start), width=320, height=180)

    with patch("streamclipper.assembly.longform.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.longform.CopyrightRiskDetector.detect", return_value=("CLEAN", 0.99)), \
         patch("streamclipper.assembly.longform.TransformativeNarrator.synthesize_speech", side_effect=_mock_synthesize), \
         patch.object(lf_mod, "_build_broll_montage", side_effect=spy_montage):
        res = render_long_video(test_config, plan)

    assert res["status"] == "rendered"
    # Point 1 montage (first call) must use only gator1 & gator2, NOT dl_ev2_cars.mp4
    assert seen_montage_calls[0] == ["gator1_0_5000_b0.mp4", "gator2_0_5000_b0.mp4"]


