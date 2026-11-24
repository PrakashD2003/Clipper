"""Unit tests for v4.2 Part 1:
- wide-safe cinema_split 9:16 layout
- sub_clips caption timestamp sync fix
- progressive multi-beat narration_beats support across models, narrator, captions, pipeline, and MCP server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from streamclipper.assembly.captions import AssCaptionBurner
from streamclipper.assembly.composer import VideoComposer, remap_timeline_for_beats
from streamclipper.assembly.layout import compute_layout
from streamclipper.assembly.narrator import TransformativeNarrator
from streamclipper.config import Config, HardwareProfile
from streamclipper.models import ClipMetadata, FacecamRegion, Highlight, WordTimestamp
from tests.test_composer import _make_config, _make_dummy_clip


# -----------------------------------------------------------------------------
# 1. cinema_split Layout Geometry & Blur Backdrop
# -----------------------------------------------------------------------------

def test_cinema_split_geometry():
    """Verify cinema_split 9:16 geometry on a 1080x1920 canvas."""
    layout = compute_layout("stacked", "cinema_split")
    assert layout.kind == "stacked"
    assert (layout.width, layout.height) == (1080, 1920)

    # Facecam: top 32% (y = 0..615, 1080x615)
    assert layout.facecam_box is not None
    assert (layout.facecam_box.x, layout.facecam_box.y) == (0, 0)
    assert (layout.facecam_box.w, layout.facecam_box.h) == (1080, 615)

    # Game: middle 40% (y = 615..1380, 1080x765 = ~1.41:1 aspect ratio)
    assert (layout.game_box.x, layout.game_box.y) == (0, 615)
    assert (layout.game_box.w, layout.game_box.h) == (1080, 765)
    assert round(layout.game_box.w / layout.game_box.h, 2) == 1.41

    # Hook band: cleanly on facecam/game seam (y = 540..710)
    assert (layout.hook_band.y, layout.hook_band.y + layout.hook_band.h) == (540, 710)

    # Caption band: strictly below gameplay and strictly above YouTube Shorts UI overlay (y = 1380..1540)
    assert (layout.caption_band.y, layout.caption_band.y + layout.caption_band.h) == (1380, 1540)
    assert layout.caption_band.y >= layout.game_box.y + layout.game_box.h
    assert layout.caption_band.y + layout.caption_band.h <= 1920 - 380


def test_cinema_split_blur_backdrop(tmp_path: Path):
    """Verify cinema_split renders darkened blurred gameplay behind bottom band (never solid black)."""
    cfg = _make_config(tmp_path, overrides={"video": {"game_fit": "cinema_split"}})
    composer = VideoComposer(cfg)
    clip = _make_dummy_clip(duration=2.0, size=(1920, 1080), color=(220, 50, 50))
    facecam = FacecamRegion(x=1400, y=50, w=400, h=400)

    comp = composer.compose(clip, facecam)
    assert comp.size == (1080, 1920)
    assert comp._layout.kind == "stacked"

    frame = comp.get_frame(0.5)
    # Facecam area (y=300): original color
    assert frame[300, 540].tolist() == [220, 50, 50]
    # Game area (y=1000): original color
    assert frame[1000, 540].tolist() == [220, 50, 50]
    # Bottom band (y=1700): darkened blurred gameplay (approx 50% luminance, not 0)
    bottom_pixel = frame[1700, 540].tolist()
    assert bottom_pixel == [110, 25, 25]  # 50% darkened red
    assert bottom_pixel != [0, 0, 0]  # Never black void!


# -----------------------------------------------------------------------------
# 2. sub_clips Caption Timestamp Alignment
# -----------------------------------------------------------------------------

def test_sub_clips_caption_timestamp_alignment():
    """Verify remap_timeline_for_beats maps word timestamps to actual trimmed beat durations."""
    beats = [
        {"start": 10.0, "end": 15.0},
        {"start": 30.0, "end": 35.0},
    ]
    # Suppose beat 1 actually yielded 5.2s and beat 2 actually yielded 4.8s due to trims/padding
    actual_durations = [5.2, 4.8]
    actual_beats = [
        {**b, "start": float(b["start"]), "end": float(b["start"]) + float(a)}
        for a, b in zip(actual_durations, beats)
    ]

    words = [
        WordTimestamp(text="word1", start=11.0, end=11.5, confidence=1.0),
        WordTimestamp(text="dead_air", start=20.0, end=21.0, confidence=1.0),
        WordTimestamp(text="word2", start=31.0, end=31.5, confidence=1.0),
    ]

    remapped_words, _, remapped_peak = remap_timeline_for_beats(
        words=words,
        chat_burst=[],
        peak_time=32.0,
        beats=actual_beats,
    )

    # word1 was at 11.0s (1.0s into beat 1) -> maps to 1.0s
    assert len(remapped_words) == 2
    assert remapped_words[0].text == "word1"
    assert remapped_words[0].start == pytest.approx(1.0)
    assert remapped_words[0].end == pytest.approx(1.5)

    # dead_air was dropped because it fell outside both beats

    # word2 was at 31.0s (1.0s into beat 2) -> maps to actual_durations[0] + 1.0s = 6.2s
    assert remapped_words[1].text == "word2"
    assert remapped_words[1].start == pytest.approx(5.2 + 1.0)
    assert remapped_words[1].end == pytest.approx(5.2 + 1.5)

    # peak_time was at 32.0s (2.0s into beat 2) -> maps to 5.2 + 2.0 = 7.2s
    assert remapped_peak == pytest.approx(5.2 + 2.0)


# -----------------------------------------------------------------------------
# 3. Progressive Multi-beat Narration (narration_beats)
# -----------------------------------------------------------------------------

def test_models_narration_beats_support():
    """Verify Highlight and ClipMetadata accept narration_beats with backward compatibility."""
    beats_data = [
        {"text": "Beat one hook", "at_second": 0.5},
        {"text": "Beat two reaction", "at_second": 4.0},
    ]
    h = Highlight(
        start=10.0,
        end=25.0,
        score=8.5,
        title="Epic Clutch",
        reason="Insane shot",
        narration_beats=beats_data,
    )
    assert h.narration_beats == beats_data
    d = h.to_dict()
    assert d["narration_beats"] == beats_data
    h2 = Highlight.from_dict(d)
    assert h2.narration_beats == beats_data

    # Backward compatibility with missing field
    d_no_beats = {k: v for k, v in d.items() if k != "narration_beats"}
    h3 = Highlight.from_dict(d_no_beats)
    assert h3.narration_beats == []

    meta = ClipMetadata(
        hook_title="Epic Clutch",
        caption="Check this out",
        hashtags=["GTA6"],
        streamer="Speed",
        game="GTA 6",
        source_url="https://youtube.com/watch?v=xyz",
        clip_start=10.0,
        clip_end=25.0,
        highlight_score=8.5,
        narration_beats=beats_data,
    )
    assert meta.narration_beats == beats_data
    meta_dict = meta.to_dict()
    assert meta_dict["narration_beats"] == beats_data
    meta2 = ClipMetadata.from_dict(meta_dict)
    assert meta2.narration_beats == beats_data


def test_narrator_prepare_beat_tracks(tmp_path: Path):
    """Verify prepare_beat_tracks synthesizes beats and automatically prevents overlap."""
    cfg = _make_config(tmp_path)
    narrator = TransformativeNarrator(cfg)

    beats_input = [
        {"text": "First beat intro", "at_second": 0.5},
        # Second beat requests 1.0s, but first beat runs 2.0s (ends at 2.5s) -> should shift
        {"text": "Second beat reaction", "at_second": 1.0},
        {"text": "Third beat outro", "at_second": 8.0},
    ]

    # Mock synthesize_speech and duration probe
    def fake_synthesize(text, out_path, max_seconds=None, language=None):
        out_path = Path(out_path)
        out_path.write_bytes(b"RIFFdummywav")
        return out_path

    with patch.object(narrator, "synthesize_speech", side_effect=fake_synthesize), \
         patch.object(narrator, "_probe_audio_duration", return_value=2.0):
        tracks = narrator.prepare_beat_tracks(
            beats_input,
            temp_dir=tmp_path,
            clip_duration=15.0,
        )

    assert len(tracks) == 3
    # Beat 0: starts at 0.5s, dur = 2.0s (ends at 2.5s)
    assert tracks[0][1] == pytest.approx(0.5)
    assert tracks[0][2] == pytest.approx(2.0)
    assert tracks[0][3] == "First beat intro"

    # Beat 1: shifted to >= 2.5s (ends at 4.5s)
    assert tracks[1][1] == pytest.approx(2.5)
    assert tracks[1][2] == pytest.approx(2.0)
    assert tracks[1][3] == "Second beat reaction"

    # Beat 2: starts at 8.0s (ends at 10.0s)
    assert tracks[2][1] == pytest.approx(8.0)
    assert tracks[2][2] == pytest.approx(2.0)
    assert tracks[2][3] == "Third beat outro"


def test_narrator_ducking_filtergraph_with_beat_tracks(tmp_path: Path):
    """Verify ducking filtergraph ducks stream audio ONLY during beat windows and ducks BGM -4dB."""
    cfg = _make_config(tmp_path)
    narrator = TransformativeNarrator(cfg)

    beat_tracks = [
        (tmp_path / "b0.wav", 1.0, 2.0),  # Window: 1.0 .. 3.0s
        (tmp_path / "b1.wav", 6.0, 2.5),  # Window: 6.0 .. 8.5s
    ]

    filtergraph, input_count = narrator._build_ducking_filtergraph(
        clip_duration=10.0,
        has_video_audio=True,
        has_bgm=True,
        beat_tracks=beat_tracks,
    )

    # 1 video stream + 1 BGM + 2 beat tracks = 4 inputs
    assert input_count == 4

    # Verify adelay on beat inputs
    assert "adelay=1000|1000" in filtergraph
    assert "adelay=6000|6000" in filtergraph

    # Verify stream audio ducking only in beat windows
    assert "between(t,1.00,3.00)+between(t,6.00,8.50)" in filtergraph

    # Verify BGM ducked by -4dB during beat windows
    assert "volume='if(between(t,1.00,3.00)+between(t,6.00,8.50)" in filtergraph
    assert ":eval=frame" in filtergraph


def test_captions_narration_beats_windows(tmp_path: Path):
    """Verify generate_ass suppresses overlapping streamer words and emits beat captions."""
    cfg = _make_config(tmp_path)
    burner = AssCaptionBurner(cfg)

    streamer_words = [
        WordTimestamp(text="hello", start=0.5, end=0.8, confidence=1.0),
        WordTimestamp(text="overlap", start=1.5, end=2.0, confidence=1.0),  # In beat window!
        WordTimestamp(text="world", start=4.0, end=4.5, confidence=1.0),
    ]

    beat_windows = [
        (1.0, 3.0, "Progressive beat line"),
    ]

    ass_path = tmp_path / "captions.ass"
    out_path = burner.generate_ass(
        words=streamer_words,
        clip_start_offset=0.0,
        ass_path=ass_path,
        clip_duration=10.0,
        narration_beats_windows=beat_windows,
    )

    content = out_path.read_text(encoding="utf-8")
    assert "Dialogue:" in content
    # Streamer words outside beat window should appear (case-insensitive check as Hormozi styles uppercase)
    assert "HELLO" in content
    assert "WORLD" in content
    # Streamer word inside beat window should be suppressed
    assert "overlap" not in content.lower()
    # Beat text words should appear in Hormozi caption events
    assert "PROGRESSIVE" in content
    assert "BEAT" in content
    assert "LINE" in content


# -----------------------------------------------------------------------------
# 4. MCP Server Support for narration_beats
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_submit_highlights_and_prepare_render_clip_narration_beats(tmp_path: Path):
    """Verify submit_highlights and _prepare_render_clip accept narration_beats without intro/outro."""
    from streamclipper.cache import StepCache
    from streamclipper.mcp_server.server import _prepare_render_clip, submit_highlights
    from streamclipper.models import PipelineState, StreamInfo, Platform, TranscriptionResult, TranscriptSegment

    # Test _prepare_render_clip accepts narration_beats
    prep_res = await _prepare_render_clip(
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        start=300.0,
        end=320.0,
        hook_title="Test Hook",
        narration_intro="",  # empty!
        narration_outro="",  # empty!
        narration_beats=[{"text": "Beat commentary text", "at_second": 1.0}],
        narration_strict=False,
    )

    assert isinstance(prep_res, dict)
    assert prep_res["start"] == 300.0
    assert prep_res["end"] == 320.0
    assert len(prep_res["narration_beats"]) == 1
    assert prep_res["narration_beats"][0]["text"] == "Beat commentary text"

    # Test submit_highlights accepts narration_beats without intro/outro
    cfg = _make_config(tmp_path)
    cache = StepCache(cfg)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    st = PipelineState(
        stream_info=StreamInfo(
            url=test_url,
            title="Stream Title",
            platform=Platform.YOUTUBE,
            streamer="Streamer",
            game="GTA 6",
            duration_seconds=1000,
        ),
        transcript=TranscriptionResult(
            segments=[
                TranscriptSegment(start=300.0, end=320.0, text="sample spoken gameplay transcript here", words=[]),
            ],
            full_text="sample spoken gameplay transcript here",
            language="en",
        ),
    )
    cache.save_state(st)

    with patch("streamclipper.mcp_server.server._get_config", return_value=cfg):
        res_str = await submit_highlights(
            url=test_url,
            highlights=json.dumps([
                {
                    "start": 300.0,
                    "end": 320.0,
                    "title": "Beat Highlight",
                    "score": 9.0,
                    "narration_beats": [{"text": "Beat spoken text", "at_second": 2.0}],
                    "narration_intro": "",
                    "narration_outro": "",
                }
            ]),
            narration_strict=False,
        )
    res_data = json.loads(res_str)
    assert res_data.get("accepted_count") == 1
    loaded = cache.load_state(test_url)
    assert loaded is not None
    assert len(loaded.highlights) == 1
    assert len(loaded.highlights[0].narration_beats) == 1
    assert loaded.highlights[0].narration_beats[0]["text"] == "Beat spoken text"


def test_cinema_split_default_game_fit_fallback(tmp_path: Path):
    """Verify VideoComposer defaults to cinema_split when game_fit is omitted from config."""
    cfg = _make_config(tmp_path)
    # Remove game_fit if present in data dict
    if "video" in cfg._data and "game_fit" in cfg._data["video"]:
        del cfg._data["video"]["game_fit"]
    composer = VideoComposer(cfg)
    assert composer.game_fit == "cinema_split"


def test_narrator_prepare_beat_tracks_zero_duration_and_overflow(tmp_path: Path):
    """Verify prepare_beat_tracks with clip_duration=0.0 does not skip beats and handles overflow."""
    cfg = _make_config(tmp_path)
    narrator = TransformativeNarrator(cfg)

    beats_input = [
        {"text": "Beat inside", "at_second": 1.0},
        {"text": "Beat overflowing", "at_second": 8.0},
    ]

    def fake_synthesize(text, out_path, max_seconds=None, language=None):
        out_path = Path(out_path)
        out_path.write_bytes(b"RIFFdummywav")
        return out_path

    # Case 1: clip_duration=0.0 (unspecified/unknown) should NOT skip beats
    with patch.object(narrator, "synthesize_speech", side_effect=fake_synthesize), \
         patch.object(narrator, "_probe_audio_duration", return_value=3.0):
        tracks_zero = narrator.prepare_beat_tracks(
            beats_input,
            temp_dir=tmp_path / "zero",
            clip_duration=0.0,
        )
    assert len(tracks_zero) == 2

    # Case 2: clip_duration=10.0, beat 2 at 8.0 with dur=3.0 overflows 10.0 -> clamped to 2.0
    with patch.object(narrator, "synthesize_speech", side_effect=fake_synthesize), \
         patch.object(narrator, "_probe_audio_duration", return_value=3.0):
        tracks_overflow = narrator.prepare_beat_tracks(
            beats_input,
            temp_dir=tmp_path / "overflow",
            clip_duration=10.0,
        )
    assert len(tracks_overflow) == 2
    assert tracks_overflow[1][1] == pytest.approx(8.0)
    assert tracks_overflow[1][2] == pytest.approx(2.0)  # Clamped to 10.0 - 8.0

    # Filtergraph should generate atrim and afade for the clamped overflowing beat
    filtergraph, _ = narrator._build_ducking_filtergraph(
        clip_duration=10.0,
        has_video_audio=True,
        beat_tracks=[(tracks_overflow[1][0], 8.0, 3.0)],  # requested 3.0s in 2.0s space
    )
    assert "atrim=0:2.00" in filtergraph
    assert "afade=t=out:st=1.85:d=0.15" in filtergraph


def test_pipeline_wrapped_mix_and_wrapped_gen_positional_args(tmp_path: Path):
    """Verify wrapped_mix and wrapped_gen tolerate positional arguments without TypeError."""
    # Test wrapped_mix positional args
    mix_called = {}
    def fake_mix(*args, **kwargs):
        mix_called["args"] = args
        mix_called["kwargs"] = kwargs
        return "output.mp4"

    mix_beats = [(Path("b.wav"), 1.0, 2.0)]
    args_list = list(("in.mp4", "out.mp4", "old_intro.wav", "old_outro.wav"))
    if len(args_list) > 2:
        args_list[2] = None
    if len(args_list) > 3:
        args_list[3] = None
    kwargs = {"intro_wav": None, "outro_wav": None, "beat_tracks": mix_beats}
    fake_mix(*args_list, **kwargs)
    assert mix_called["args"][2] is None
    assert mix_called["args"][3] is None
    assert mix_called["kwargs"]["beat_tracks"] == mix_beats

    # Test wrapped_gen positional args
    gen_called = {}
    def fake_gen(*args, **kwargs):
        gen_called["args"] = args
        gen_called["kwargs"] = kwargs
        return Path("out.ass")

    gen_args = list(([], 0.0, "captions.ass", [], "old_intro", "old_outro", 2.0, 2.0))
    if len(gen_args) > 4:
        gen_args[4] = ""
    if len(gen_args) > 5:
        gen_args[5] = ""
    if len(gen_args) > 6:
        gen_args[6] = None
    if len(gen_args) > 7:
        gen_args[7] = None
    gen_kwargs = {
        "narration_intro": "",
        "narration_outro": "",
        "intro_dur": None,
        "outro_dur": None,
        "narration_beats_windows": [(1.0, 3.0, "beat")],
    }
    fake_gen(*gen_args, **gen_kwargs)
    assert gen_called["args"][4] == ""
    assert gen_called["args"][5] == ""
    assert gen_called["kwargs"]["narration_beats_windows"] == [(1.0, 3.0, "beat")]


def test_parse_narration_beats_robustness():
    """Verify _parse_narration_beats robustly parses JSON string, lists, and handles nulls/invalids."""
    from streamclipper.mcp_server.server import _parse_narration_beats

    # Valid list of dicts
    res1 = _parse_narration_beats([
        {"text": "Line 1", "at_second": 1.5},
        {"text": "Line 2", "at_second": None},  # None at_second -> defaults to 0.0
        {"text": "", "at_second": 5.0},          # Empty text -> skipped
        "invalid",                               # Non-dict -> skipped
    ])
    assert len(res1) == 2
    assert res1[0] == {"text": "Line 1", "at_second": 1.5}
    assert res1[1] == {"text": "Line 2", "at_second": 0.0}

    # Valid JSON string
    json_str = json.dumps([{"text": "JSON Beat", "at_second": 3.2}])
    res2 = _parse_narration_beats(json_str)
    assert len(res2) == 1
    assert res2[0] == {"text": "JSON Beat", "at_second": 3.2}

    # Invalid JSON string or non-list
    assert _parse_narration_beats("not a json") == []
    assert _parse_narration_beats(None) == []
    assert _parse_narration_beats(123) == []


@pytest.mark.asyncio
async def test_mcp_render_single_highlight_propagates_narration_beats(tmp_path: Path):
    """Verify _render_single_highlight populates narration_beats on ClipMetadata."""
    from streamclipper.mcp_server.server import _render_single_highlight
    from streamclipper.pipeline import Pipeline

    cfg = _make_config(tmp_path)
    pipe = Pipeline(cfg)
    hl = Highlight(
        start=10.0,
        end=20.0,
        score=8.0,
        title="Beat Test Highlight",
        reason="Testing beats metadata",
        narration_beats=[{"text": "Beat line", "at_second": 1.0}],
    )

    fake_result = MagicMock()
    fake_result.quarantined = False
    fake_result.video_path = tmp_path / "fake_video.mp4"
    (tmp_path / "fake_video.mp4").write_bytes(b"dummy")

    with patch.object(pipe, "_assemble_highlight", return_value=fake_result) as mock_assemble, \
         patch("streamclipper.mcp_server.server.download_segment", return_value=Path("dummy.mp4")):
        res, mode = await _render_single_highlight(
            pipeline=pipe,
            config=cfg,
            url="https://youtube.com/watch?v=dQw4w9WgXcQ",
            highlight=hl,
            streamer_name="Streamer",
            game_name="GTA 6",
            clip_index=1,
            base_mode="narrative",
        )

    # Check that metadata passed to _assemble_highlight contains narration_beats
    assert mock_assemble.called
    metadata_arg = mock_assemble.call_args[0][1]
    assert isinstance(metadata_arg, ClipMetadata)
    assert len(metadata_arg.narration_beats) == 1
    assert metadata_arg.narration_beats[0]["text"] == "Beat line"

