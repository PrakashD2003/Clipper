"""Tests for 9:16 vertical topic Shorts assembly, post files, and MCP integration (T6)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from streamclipper.assembly.longform import _probe_duration
from streamclipper.assembly.topic_shorts import (
    _build_shorts_ass,
    _calculate_clip_trims,
    render_topic_shorts,
)
from streamclipper.cli import cli
from streamclipper.config import Config
from streamclipper.output.topic_post import write_short_post
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


def _create_synthetic_wav(path: Path, duration: float = 2.0) -> Path:
    """Create a tiny test WAV file with sine tone via FFmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-ar", "48000", "-ac", "1",
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
    _set_cfg(cfg, "output.require_narration", True)
    _set_cfg(cfg, "output.quarantine_dir", "_quarantine")
    _set_cfg(cfg, "video.codec", "libx264")
    _set_cfg(cfg, "video.crf", 28)
    _set_cfg(cfg, "video.fps", 30)
    return cfg


@pytest.fixture
def synthetic_plan() -> TopicPlan:
    """A valid, approved TopicPlan with 3 points and 3 shorts."""
    return TopicPlan(
        plan_id="plan-t6-test",
        topic_id="vice-city-secrets",
        topic_title="Vice City Secrets",
        game="GTA 6",
        shape="numbered",
        status="approved",
        video_title="10 Insane Secrets in GTA 6",
        points=[
            TopicPoint(
                index=1,
                heading="#1 The Hidden Casino Vault",
                narration="Look closely at the underground vault mechanics.",
                evidence=[
                    EvidenceClip(
                        source_id="src_stream_1",
                        url="https://youtube.com/watch?v=vault01",
                        channel="ViceGamer",
                        kind="stream",
                        start=10.0,
                        end=30.0,  # 20s
                        role="evidence",
                    )
                ],
            ),
            TopicPoint(
                index=2,
                heading="#2 Official Rockstar Trailer Secret",
                narration="Rockstar confirmed the security layout right here.",
                evidence=[
                    EvidenceClip(
                        source_id="src_official_2",
                        url="https://youtube.com/watch?v=official02",
                        channel="RockstarGames",
                        kind="official",
                        start=5.0,
                        end=25.0,  # 20s
                        role="evidence",
                    )
                ],
            ),
            TopicPoint(
                index=3,
                heading="#3 Nightclub Backroom Details",
                narration="The VIP lounge hides a subtle Easter egg.",
                evidence=[
                    EvidenceClip(
                        source_id="src_stream_3",
                        url="https://youtube.com/watch?v=club03",
                        channel="ClubMaster",
                        kind="stream",
                        start=50.0,
                        end=75.0,  # 25s
                        role="evidence",
                    )
                ],
            ),
        ],
        shorts=[
            ShortPlan(
                kind="teaser",
                slot="teaser",
                point_indexes=[1, 2],
                hook_title="Did Rockstar Just Leak This?",
                narration_intro="You won't believe what Rockstar hid in this trailer.",
                callout="Watch this exact moment right here!",
                narration_outro="Check out the full breakdown on the channel.",
            ),
            ShortPlan(
                kind="standalone",
                slot="short_a",
                point_indexes=[1],
                hook_title="The Secret Casino Vault in GTA 6",
                narration_intro="The high roller casino hides an underground secret.",
                callout="Look at this hidden door!",
                narration_outro="What do you think is inside?",
            ),
            ShortPlan(
                kind="standalone",
                slot="short_b",
                point_indexes=[3],
                hook_title="Insane Nightclub Easter Egg",
                narration_intro="This nightclub detail went completely unnoticed.",
                callout="Wait for the camera angle shift!",
                narration_outro="Did you catch that?",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_T6_refuses_unapproved_plans(test_config: Config, synthetic_plan: TopicPlan) -> None:
    """render_topic_shorts must reject unapproved plans synchronously."""
    synthetic_plan.status = "draft"
    save_plan(test_config, synthetic_plan)

    with pytest.raises(ValueError, match="must be 'approved'"):
        render_topic_shorts(test_config, synthetic_plan)


def test_T6_per_slot_clip_selection(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """Each short slot must gather evidence clips according to its point_indexes."""
    save_plan(test_config, synthetic_plan)
    downloaded_urls: list[str] = []

    def mock_download(url, start, end, out_dir, **kwargs):
        downloaded_urls.append(url)
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=1.0)

    def mock_write_videofile(filename, *args, **kwargs):
        _create_synthetic_mp4(filename, duration=1.0)

    with patch("streamclipper.assembly.topic_shorts.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.topic_shorts.CompositeVideoClip.write_videofile", side_effect=mock_write_videofile), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_narration_tracks", return_value=(None, None)), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_callout_track", return_value=None), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.mix_narration_with_ducking", return_value=None):

        # Render only short_b (point_indexes=[3])
        downloaded_urls.clear()
        render_topic_shorts(test_config, synthetic_plan, slots=("short_b",))
        assert downloaded_urls == ["https://youtube.com/watch?v=club03"]

        # Render only teaser (point_indexes=[1, 2])
        downloaded_urls.clear()
        render_topic_shorts(test_config, synthetic_plan, slots=("teaser",))
        assert "https://youtube.com/watch?v=vault01" in downloaded_urls
        assert "https://youtube.com/watch?v=official02" in downloaded_urls
        assert "https://youtube.com/watch?v=club03" not in downloaded_urls



def test_T6_duration_clamp_35_to_60(tmp_path: Path) -> None:
    """Clips exceeding 60s total are trimmed around their middle so Short lands in 35-60s with >= 4s each."""
    c1 = EvidenceClip(url="u1", start=0.0, end=30.0)
    c2 = EvidenceClip(url="u2", start=0.0, end=30.0)
    c3 = EvidenceClip(url="u3", start=0.0, end=30.0)
    # Total raw = 90s > 60s
    clips_with_dur = [(c1, tmp_path / "1.mp4", 30.0), (c2, tmp_path / "2.mp4", 30.0), (c3, tmp_path / "3.mp4", 30.0)]

    trims = _calculate_clip_trims(clips_with_dur, target_min=35.0, target_max=60.0, min_clip_dur=4.0)

    total_trimmed = sum(t_end - t_start for _, _, t_start, t_end in trims)
    assert 35.0 <= total_trimmed <= 60.0

    for _, _, t_start, t_end in trims:
        clip_dur = t_end - t_start
        assert clip_dur >= 4.0
        # Verify trimmed around middle: mid of 30.0 is 15.0
        mid = (t_start + t_end) / 2.0
        assert abs(mid - 15.0) < 0.1


def test_T6_credit_tag_and_hook_present_in_ass(tmp_path: Path) -> None:
    """Verify persistent hook title and @channel credit tags are generated into the ASS file."""
    ass_path = tmp_path / "test_short.ass"
    hook_title = "CAN YOU BELIEVE THIS DETAIL?"
    credit_windows = [
        (0.0, 4.0, "@ViceGamer"),
        (15.0, 19.0, "Rockstar Games"),
    ]
    narration_windows = [
        (0.0, 3.0, "Watch this secret."),
        (14.0, 16.5, "Right here!"),
        (25.0, 28.0, "Subscribe for more."),
    ]

    _build_shorts_ass(
        ass_path=ass_path,
        hook_title=hook_title,
        credit_windows=credit_windows,
        narration_windows=narration_windows,
        total_duration=30.0,
    )

    assert ass_path.is_file()
    content = ass_path.read_text(encoding="utf-8")

    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "Style: HookTitle" in content
    assert "Style: CreditTag" in content
    assert "Style: Hormozi" in content

    # Verify Hormozi style safe zone (MarginV >= 460) so captions never collide with Shorts bottom UI
    hormozi_style_line = next(line for line in content.splitlines() if line.startswith("Style: Hormozi"))
    margin_v = int(hormozi_style_line.split(",")[-2])
    assert margin_v >= 460

    # Hook must be present and persistent across full duration (0:00:00.00 to 0:00:30.00)
    assert hook_title in content
    assert "0:00:00.00,0:00:30.00,HookTitle" in content

    # Credits must be present
    assert "@ViceGamer" in content
    assert "Rockstar Games" in content
    assert "CreditTag" in content

    # Active-word highlighted Hormozi narration captions must be present
    assert "Hormozi" in content
    assert "\\c&H" in content
    assert "WATCH" in content
    assert "SECRET." in content
    assert "RIGHT" in content
    assert "HERE!" in content
    assert "SUBSCRIBE" in content
    assert "MORE." in content


def test_T6_post_files_teaser_vs_standalone(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """Teaser post has Related Video reminder; standalone posts never mention the long video."""
    teaser_short = synthetic_plan.shorts[0]
    standalone_short = synthetic_plan.shorts[1]

    teaser_mp4 = tmp_path / "short_teaser.mp4"
    standalone_mp4 = tmp_path / "short_a.mp4"

    teaser_post = write_short_post(test_config, synthetic_plan, teaser_short, teaser_mp4)
    standalone_post = write_short_post(test_config, synthetic_plan, standalone_short, standalone_mp4)

    assert teaser_post.is_file()
    assert standalone_post.is_file()

    teaser_text = teaser_post.read_text(encoding="utf-8")
    standalone_text = standalone_post.read_text(encoding="utf-8")

    # Teaser requirements
    assert "TITLE" in teaser_text
    assert teaser_short.hook_title in teaser_text
    assert "SOURCES" in teaser_text
    assert "@ViceGamer: https://youtube.com/watch?v=vault01&t=10s" in teaser_text
    assert "Official footage (RockstarGames): https://youtube.com/watch?v=official02&t=5s" in teaser_text
    assert "#Shorts" in teaser_text
    assert "AI voiceover:" in teaser_text
    assert "- Set YouTube's Related Video to the long video" in teaser_text

    # Standalone requirements
    assert "TITLE" in standalone_text
    assert standalone_short.hook_title in standalone_text
    assert "SOURCES" in standalone_text
    assert "@ViceGamer" in standalone_text
    assert "#Shorts" in standalone_text
    assert "AI voiceover:" in standalone_text
    # Must NOT mention long video anywhere
    assert "long video" not in standalone_text.lower()
    assert "related video" not in standalone_text.lower()


def test_T6_quarantine_when_intro_voice_fails(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """When expected intro voiceover is not mixed, the short is quarantined and reported."""
    save_plan(test_config, synthetic_plan)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=1.0)

    # Patch synthesize/mix so that intro voice fails
    with patch("streamclipper.assembly.topic_shorts.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_narration_tracks", return_value=(None, None)), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_callout_track", return_value=None), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.mix_narration_with_ducking", return_value=None):

        result = render_topic_shorts(test_config, synthetic_plan, slots=("teaser",))

    assert len(result["quarantined"]) == 1
    assert result["quarantined"][0]["slot"] == "teaser"
    assert "voiceover" in result["quarantined"][0]["quarantine_reason"]

    # Verify quarantined video exists in _quarantine sub-folder
    quarantine_path = Path(result["quarantined"][0]["video_path"])
    assert quarantine_path.is_file()
    assert "_quarantine" in quarantine_path.parts

    # Verify plan.outputs does not have approved teaser
    assert "teaser" not in synthetic_plan.outputs


def test_T6_1080x1920_output_with_audio(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """Full render produces a valid 1080x1920 video with an active audio stream."""
    save_plan(test_config, synthetic_plan)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=4.0)

    synth_wav = _create_synthetic_wav(tmp_path / "voice.wav", duration=2.5)

    def mock_mix(input_video, output_video, **kwargs):
        # Transcode intermediate video to output_video
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-i", str(synth_wav),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac",
            str(output_video),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_video

    with patch("streamclipper.assembly.topic_shorts.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_narration_tracks", return_value=(synth_wav, synth_wav)), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_callout_track", return_value=synth_wav), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.mix_narration_with_ducking", side_effect=mock_mix), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator._probe_audio_duration", return_value=2.5):

        result = render_topic_shorts(test_config, synthetic_plan, slots=("short_a",))

    assert len(result["rendered"]) == 1
    video_path = Path(result["rendered"][0]["video_path"])
    assert video_path.is_file()
    assert video_path.name == "short_a.mp4"

    # Verify dimensions are 1080x1920
    cmd_probe = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]
    res_v = subprocess.run(cmd_probe, check=True, capture_output=True, text=True)
    assert res_v.stdout.strip() == "1080x1920"

    # Verify audio stream is present
    cmd_audio = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "csv=p=0",
        str(video_path),
    ]
    res_a = subprocess.run(cmd_audio, check=True, capture_output=True, text=True)
    assert "aac" in res_a.stdout.strip().lower()

    # Verify post file was written
    post_path = Path(result["rendered"][0]["post_path"])
    assert post_path.is_file()


@pytest.mark.asyncio
async def test_T6_mcp_tool_starts_job(test_config: Config, synthetic_plan: TopicPlan) -> None:
    """MCP render_topic_shorts starts background job and refuses unapproved plans synchronously."""
    from streamclipper.mcp_server.server import mcp

    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "render_topic_shorts" in tools
    tool = tools["render_topic_shorts"]

    # Unapproved plan refusal
    synthetic_plan.status = "draft"
    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.mcp_server.server.Config", return_value=test_config):
        res_unapproved = await tool.fn(plan_id=synthetic_plan.plan_id)
        assert "is not approved" in res_unapproved

    # Approved plan starts job
    synthetic_plan.status = "approved"
    save_plan(test_config, synthetic_plan)

    with patch("streamclipper.mcp_server.server.Config", return_value=test_config), \
         patch("streamclipper.mcp_server.server._start_render_job", return_value=json.dumps({"status": "queued", "kind": "topic_shorts", "job_id": "job-123"})) as mock_start:
        res_approved = await tool.fn(plan_id=synthetic_plan.plan_id)
        payload = json.loads(res_approved)
        assert payload.get("status") == "queued"
        assert payload.get("kind") == "topic_shorts"
        assert payload.get("job_id") == "job-123"
        mock_start.assert_called_once()
        args, kwargs = mock_start.call_args
        assert args[0] == "topic_shorts"


def test_T6_cli_shorts(tmp_path: Path, synthetic_plan: TopicPlan) -> None:
    """CLI streamclipper topics shorts PLAN_ID invokes render_topic_shorts."""
    cfg_file = tmp_path / "custom_config.yaml"
    cfg_file.write_text(
        f"general:\n  output_dir: '{tmp_path.as_posix()}/output'\n  cache_dir: '{tmp_path.as_posix()}/cache'\n  temp_dir: '{tmp_path.as_posix()}/temp'\n",
        encoding="utf-8",
    )
    cli_cfg = Config(user_config_path=cfg_file)
    save_plan(cli_cfg, synthetic_plan)
    runner = CliRunner()

    with patch("streamclipper.assembly.topic_shorts.render_topic_shorts") as mock_render:
        mock_render.return_value = {
            "status": "complete",
            "plan_id": synthetic_plan.plan_id,
            "rendered": [{"slot": "teaser", "video_path": "short_teaser.mp4", "duration": 40.0, "post_path": "short_teaser_post.txt"}],
            "quarantined": [],
            "outputs": {},
        }

        result = runner.invoke(
            cli,
            ["--config", str(cfg_file), "topics", "shorts", synthetic_plan.plan_id, "--slot", "teaser"],
        )
        assert result.exit_code == 0, f"CLI shorts failed: {result.output}"
        assert "Rendered 1 Short(s) successfully" in result.output
        mock_render.assert_called_once()


def test_T6_empty_point_indexes_fallback(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """When short.point_indexes is empty, render_topic_shorts and write_short_post fall back to first evidence."""
    # Set point_indexes to empty for short_a
    synthetic_plan.shorts[1].point_indexes = []
    save_plan(test_config, synthetic_plan)

    downloaded_urls: list[str] = []

    def mock_download(url, start, end, out_dir, **kwargs):
        downloaded_urls.append(url)
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=1.0)

    def mock_write_videofile(filename, *args, **kwargs):
        _create_synthetic_mp4(filename, duration=1.0)

    with patch("streamclipper.assembly.topic_shorts.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.topic_shorts.CompositeVideoClip.write_videofile", side_effect=mock_write_videofile), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_narration_tracks", return_value=(None, None)), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_callout_track", return_value=None), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.mix_narration_with_ducking", return_value=None):

        render_topic_shorts(test_config, synthetic_plan, slots=("short_a",))
        # Point 1 evidence (vault01) is used as fallback
        assert downloaded_urls == ["https://youtube.com/watch?v=vault01"]

    # Also verify write_short_post only attributes point 1 fallback, not all points
    post_file = write_short_post(test_config, synthetic_plan, synthetic_plan.shorts[1], tmp_path / "short_a.mp4")
    post_text = post_file.read_text(encoding="utf-8")
    assert "@ViceGamer: https://youtube.com/watch?v=vault01&t=10s" in post_text
    assert "official02" not in post_text
    assert "club03" not in post_text


def test_T6_slot_alias_normalization(test_config: Config, synthetic_plan: TopicPlan) -> None:
    """Slot names like 'short_teaser', 'a', and 'b' are normalized to match short.slot."""
    save_plan(test_config, synthetic_plan)
    downloaded_urls: list[str] = []

    def mock_download(url, start, end, out_dir, **kwargs):
        downloaded_urls.append(url)
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=1.0)

    def mock_write_videofile(filename, *args, **kwargs):
        _create_synthetic_mp4(filename, duration=1.0)

    with patch("streamclipper.assembly.topic_shorts.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.topic_shorts.CompositeVideoClip.write_videofile", side_effect=mock_write_videofile), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_narration_tracks", return_value=(None, None)), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_callout_track", return_value=None), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.mix_narration_with_ducking", return_value=None):

        # "short_teaser" matches slot "teaser"
        downloaded_urls.clear()
        render_topic_shorts(test_config, synthetic_plan, slots="short_teaser")
        assert "https://youtube.com/watch?v=vault01" in downloaded_urls

        # "b" matches slot "short_b"
        downloaded_urls.clear()
        render_topic_shorts(test_config, synthetic_plan, slots="b")
        assert downloaded_urls == ["https://youtube.com/watch?v=club03"]


def test_T6_progress_tracking_and_cancellation(test_config: Config, synthetic_plan: TopicPlan) -> None:
    """render_topic_shorts notifies progress (start_clip, stage, clip_done) and honors cancellation."""
    save_plan(test_config, synthetic_plan)

    events: list[str] = []

    class MockProgress:
        def __init__(self):
            self.cancelled = False

        def start_clip(self, idx: int, total: int) -> None:
            events.append(f"start:{idx}/{total}")

        def stage(self, name: str) -> None:
            events.append(f"stage:{name}")

        def clip_done(self, item: Any) -> None:
            events.append("done")

    progress = MockProgress()

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=1.0)

    def mock_write_videofile(filename, *args, **kwargs):
        _create_synthetic_mp4(filename, duration=1.0)

    with patch("streamclipper.assembly.topic_shorts.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.topic_shorts.CompositeVideoClip.write_videofile", side_effect=mock_write_videofile), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_narration_tracks", return_value=(None, None)), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_callout_track", return_value=None), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.mix_narration_with_ducking", return_value=None):

        # Render with progress
        render_topic_shorts(test_config, synthetic_plan, slots=("teaser",), progress=progress)
        assert "start:1/1" in events
        assert "stage:rendering short teaser" in events
        assert "done" in events

        # Cancellation stops before subsequent clips
        progress.cancelled = True
        events.clear()
        render_topic_shorts(test_config, synthetic_plan, slots=("teaser", "short_a"), progress=progress)
        # Should exit immediately on check
        assert events == []


def test_T6_quarantine_cleans_stale_outputs(test_config: Config, synthetic_plan: TopicPlan) -> None:
    """When a short is quarantined, any pre-existing entry in plan.outputs for that slot is cleared."""
    synthetic_plan.outputs["teaser"] = "/old/path/short_teaser.mp4"
    synthetic_plan.outputs["teaser_post"] = "/old/path/short_teaser_post.txt"
    save_plan(test_config, synthetic_plan)

    def mock_download(url, start, end, out_dir, **kwargs):
        seg = out_dir / f"seg_{int(start)}.mp4"
        return _create_synthetic_mp4(seg, duration=1.0)

    with patch("streamclipper.assembly.topic_shorts.download_segment", side_effect=mock_download), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_narration_tracks", return_value=(None, None)), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.prepare_callout_track", return_value=None), \
         patch("streamclipper.assembly.topic_shorts.TransformativeNarrator.mix_narration_with_ducking", return_value=None):

        res = render_topic_shorts(test_config, synthetic_plan, slots=("teaser",))

    assert len(res["quarantined"]) == 1
    assert "teaser" not in synthetic_plan.outputs
    assert "teaser_post" not in synthetic_plan.outputs


# ---------------------------------------------------------------------------
# Ticket T1 Verification Tests for Shorts
# ---------------------------------------------------------------------------

def test_T1_shorts_official_credit_empty_fallback(test_config: Config, tmp_path: Path) -> None:
    """Topic Shorts official credit falls back to 'Official {plan.game} footage' when channel is empty."""
    plan = TopicPlan(
        plan_id="plan-t1-shorts-cred",
        topic_id="short-cred",
        topic_title="Short Credit Test",
        game="GTA 6",
        shape="numbered",
        status="approved",
        cold_open="",
        outro="",
        points=[
            TopicPoint(
                index=1,
                heading="Point 1",
                narration="Narration",
                evidence=[
                    EvidenceClip(
                        source_id="src_off_empty",
                        url="https://youtube.com/watch?v=empty",
                        channel="",  # empty channel
                        kind="official",
                        start=0.0,
                        end=10.0,
                    )
                ],
            )
        ],
        shorts=[
            ShortPlan(
                slot="short_a",
                kind="standalone",
                point_indexes=[1],
                hook_title="Official Leak Revealed",
                narration_intro="Intro",
                narration_outro="Outro",
            )
        ],
    )

    post_file = write_short_post(test_config, plan, plan.shorts[0], tmp_path / "short_a.mp4")
    text = post_file.read_text(encoding="utf-8")
    assert "- Official GTA 6 footage: https://youtube.com/watch?v=empty&t=0s" in text
    assert "Official GTA 6 footage" in text
    assert "Rockstar Games" not in text


def test_T1_shorts_hashtag_stopwords_and_capping(test_config: Config, tmp_path: Path) -> None:
    """Shorts hashtags filter stopwords and cap at 4 (#Shorts + #Game + up to 2 topic tags)."""
    plan = TopicPlan(
        plan_id="plan-t1-shorts-tags",
        topic_id="shorts-tags",
        # Stopwords: 'How', 'The', 'For', 'Every', 'Actually'
        # Meaningful: 'Casino', 'Vault', 'Secrets'
        topic_title="How The Casino Vault Actually Works For Every Secrets",
        game="GTA 6",
        shape="numbered",
        status="approved",
        cold_open="",
        outro="",
        points=[],
        shorts=[
            ShortPlan(
                slot="short_a",
                kind="standalone",
                hook_title="Casino Vault Hook",
            )
        ],
    )

    post_file = write_short_post(test_config, plan, plan.shorts[0], tmp_path / "short_a.mp4")
    text = post_file.read_text(encoding="utf-8")

    # Extract HASHTAGS line
    for line in text.splitlines():
        if line.startswith("#"):
            tags = line.split()
            # 1. Must start with #Shorts
            assert tags[0] == "#Shorts"
            # 2. Must contain game tag
            assert "#GTA6" in tags
            # 3. Must NOT contain stopwords
            for stop in ["#How", "#The", "#For", "#Every", "#Actually"]:
                assert stop not in tags
            # 4. Must contain up to 2 topic tags
            assert "#Casino" in tags
            assert "#Vault" in tags
            # 5. Must be strictly capped at 4
            assert len(tags) == 4
            break


def test_T2_calculate_clip_trims_preserves_off_center_peak_offset_near_end() -> None:
    """_calculate_clip_trims anchors around peak_offset near the end rather than midpoint."""
    from streamclipper.assembly.topic_shorts import _calculate_clip_trims

    clip = EvidenceClip(url="http://example.com/vid", start=0.0, end=20.0)
    path = Path("fake_path.mp4")
    # Total raw = 20.0s, allocate around target_min=10.0, target_max=10.0 so alloc_dur=10.0
    clips_with_durations = [(clip, path, 20.0)]
    peak_offsets = [18.0]  # Near the end of the 20s clip

    res = _calculate_clip_trims(
        clips_with_durations,
        target_min=10.0,
        target_max=10.0,
        min_clip_dur=4.0,
        peak_offsets=peak_offsets,
    )
    assert len(res) == 1
    c, p, t_start, t_end = res[0]
    assert t_end == 20.0
    assert t_start == 10.0
    # Peak at 18.0 must be inside [t_start, t_end]
    assert t_start <= 18.0 <= t_end
    # Midpoint trimming would have produced [5.0, 15.0], completely missing 18.0!
    assert t_start != 5.0


def test_T2_calculate_clip_trims_preserves_off_center_peak_offset_near_start() -> None:
    """_calculate_clip_trims anchors around peak_offset near start rather than midpoint."""
    from streamclipper.assembly.topic_shorts import _calculate_clip_trims

    clip = EvidenceClip(url="http://example.com/vid", start=0.0, end=20.0)
    path = Path("fake_path.mp4")
    clips_with_durations = [(clip, path, 20.0)]
    peak_offsets = [2.0]  # Near the start of the 20s clip

    res = _calculate_clip_trims(
        clips_with_durations,
        target_min=10.0,
        target_max=10.0,
        min_clip_dur=4.0,
        peak_offsets=peak_offsets,
    )
    assert len(res) == 1
    c, p, t_start, t_end = res[0]
    assert t_start == 0.0
    assert t_end == 10.0
    assert t_start <= 2.0 <= t_end


def test_T2_calculate_clip_trims_backward_compatibility() -> None:
    """_calculate_clip_trims falls back to midpoint trim when peak_offsets is None or invalid."""
    from streamclipper.assembly.topic_shorts import _calculate_clip_trims

    clip = EvidenceClip(url="http://example.com/vid", start=0.0, end=20.0)
    path = Path("fake_path.mp4")
    clips_with_durations = [(clip, path, 20.0)]

    # None passed
    res_none = _calculate_clip_trims(
        clips_with_durations,
        target_min=10.0,
        target_max=10.0,
        min_clip_dur=4.0,
    )
    _, _, s1, e1 = res_none[0]
    assert s1 == 5.0
    assert e1 == 15.0

    # Invalid peak_offset (<= 0 or >= raw_dur)
    res_invalid = _calculate_clip_trims(
        clips_with_durations,
        target_min=10.0,
        target_max=10.0,
        min_clip_dur=4.0,
        peak_offsets=[-5.0],
    )
    _, _, s2, e2 = res_invalid[0]
    assert s2 == 5.0
    assert e2 == 15.0

    # Boolean peak_offset (True or False) must NOT be treated as 1.0s
    res_bool = _calculate_clip_trims(
        clips_with_durations,
        target_min=10.0,
        target_max=10.0,
        min_clip_dur=4.0,
        peak_offsets=[True],
    )
    _, _, s3, e3 = res_bool[0]
    assert s3 == 5.0
    assert e3 == 15.0


def test_T3_shorts_ass_hormozi_active_word_highlighting_and_safe_zone(tmp_path: Path) -> None:
    """Verify Ticket T3: active-word highlighted Hormozi captions in safe zone (MarginV >= 460)."""
    ass_file = tmp_path / "t3_test.ass"
    narration_windows = [
        (0.0, 4.0, "Check this hidden garage entrance"),
        (10.0, 13.5, "Nobody ever noticed this detail"),
    ]
    credit_windows = [(0.0, 4.0, "@GamerChannel")]

    _build_shorts_ass(
        ass_path=ass_file,
        hook_title="SECRET LOCATION DISCOVERED",
        credit_windows=credit_windows,
        narration_windows=narration_windows,
        total_duration=15.0,
    )

    assert ass_file.is_file()
    text = ass_file.read_text(encoding="utf-8")

    # Safe zone verification: MarginV >= 460 strictly above bottom 420px YouTube Shorts UI
    hormozi_line = next(l for l in text.splitlines() if l.startswith("Style: Hormozi"))
    fields = hormozi_line.split(",")
    margin_v = int(fields[-2])
    alignment = int(fields[-5])
    assert alignment == 2  # Bottom-centered
    assert margin_v >= 460  # At or above bottom 460px safe threshold

    # HookTitle safe zone verification: MarginV == 220 below top status bar
    hook_line = next(l for l in text.splitlines() if l.startswith("Style: HookTitle"))
    assert int(hook_line.split(",")[-2]) == 220

    # Word-highlighting animation verification
    events = [l for l in text.splitlines() if l.startswith("Dialogue:") and "Hormozi" in l]
    assert len(events) >= 2
    # Ensure active word animation tags exist
    assert any(r"\fscx" in ev for ev in events)
    assert any(r"\c&H" in ev for ev in events)

    # All narration tokens uppercase and present in dialogue lines
    dialogue_full_text = " ".join(events)
    for word in ["CHECK", "HIDDEN", "GARAGE", "NOBODY", "DETAIL"]:
        assert word in dialogue_full_text


def test_T3_shorts_ass_empty_narration_and_safe_zone_floor(tmp_path: Path, test_config: Config) -> None:
    """_build_shorts_ass handles empty narration cleanly and enforces MarginV >= 460 floor."""
    # Custom config with low position_y_ratio (e.g. 0.85 -> would be MarginV=288 without floor)
    test_config._data.setdefault("captions", {})["position_y_ratio"] = 0.85

    ass_file = tmp_path / "t3_empty_test.ass"
    _build_shorts_ass(
        ass_path=ass_file,
        hook_title="EMPTY NARRATION SHORT",
        credit_windows=[(0.0, 5.0, "@SoloChannel")],
        narration_windows=[],
        total_duration=10.0,
        font_name="Impact",
        config=test_config,
    )
    assert ass_file.is_file()
    text = ass_file.read_text(encoding="utf-8")

    # MarginV must still be >= 460
    hormozi_line = next(l for l in text.splitlines() if l.startswith("Style: Hormozi"))
    margin_v = int(hormozi_line.split(",")[-2])
    assert margin_v >= 460

    # No Hormozi dialogue lines emitted when narration is empty
    dialogues = [l for l in text.splitlines() if l.startswith("Dialogue:")]
    assert not any("Hormozi" in l for l in dialogues)
    # HookTitle and CreditTag should still be present
    assert any("HookTitle" in l for l in dialogues)
    assert any("CreditTag" in l for l in dialogues)


# ---------------------------------------------------------------------------
# Ticket T5: Multi-Variant Titles in Short Posts
# ---------------------------------------------------------------------------

def test_T5_title_variants_in_short_post(test_config: Config, synthetic_plan: TopicPlan, tmp_path: Path) -> None:
    """Verify write_short_post outputs 3-5 punchy curiosity-gap title variants under TITLE."""
    from streamclipper.output.topic_post import generate_short_title_variants, write_short_post

    short = synthetic_plan.shorts[0]
    variants = generate_short_title_variants(synthetic_plan, short)
    assert 3 <= len(variants) <= 5
    for v in variants:
        assert len(v.strip()) > 0
        assert v.lower() != (short.hook_title or "").lower()

    short_mp4 = tmp_path / "short_t5.mp4"
    short_mp4.write_bytes(b"dummy")
    post_file = write_short_post(test_config, synthetic_plan, short, short_mp4)

    assert post_file.is_file()
    text = post_file.read_text(encoding="utf-8")
    assert "TITLE" in text
    assert short.hook_title in text
    assert "TITLE VARIANTS (A/B TEST)" in text
    for v in variants:
        assert f"- {v}" in text





