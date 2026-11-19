"""Regression tests for assembly review fixes (M11 M12 L8 L9 L10 L11 L23 G7 G8)."""

from __future__ import annotations

import logging
import subprocess
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from streamclipper.assembly import facecam as facecam_mod
from streamclipper.assembly.captions import AssCaptionBurner
from streamclipper.assembly.composer import VideoComposer
from streamclipper.assembly.facecam import FacecamDetector
from streamclipper.assembly.hooks import create_replay_badge
from streamclipper.assembly.narrator import TransformativeNarrator
from streamclipper.config import Config
from streamclipper.models import WordTimestamp

from tests.test_composer import _make_config, _make_dummy_clip


def _cfg(tmp_path: Path, body: str = "") -> Config:
    f = tmp_path / "cfg.yaml"
    f.write_text(body or "transformation:\n  enabled: true\n", encoding="utf-8")
    return Config(f)


def _wav(path: Path, seconds: float = 1.0) -> Path:
    sf.write(str(path), np.zeros(int(48000 * seconds), dtype=np.float32), 48000)
    return path


def test_M11_failing_ffmpeg_raises_runtime_error(tmp_path: Path) -> None:
    narrator = TransformativeNarrator(_cfg(tmp_path))
    vid = tmp_path / "in.mp4"
    vid.write_bytes(b"x")
    intro = _wav(tmp_path / "i.wav")
    err = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="A" * 800 + "BOOMTAIL")
    with patch.object(narrator, "_has_audio_stream", return_value=True), patch(
        "streamclipper.assembly.narrator.subprocess.run", side_effect=err
    ):
        with pytest.raises(RuntimeError) as ei:
            narrator.mix_narration_with_ducking(vid, tmp_path / "out.mp4", intro, None, 10.0)
    assert "BOOMTAIL" in str(ei.value)
    assert not (tmp_path / "out.mp4").exists()


def test_M12_explicit_kokoro_falls_back_to_edge_tts(tmp_path: Path, caplog: Any) -> None:
    narrator = TransformativeNarrator(
        _cfg(tmp_path, "transformation:\n  enabled: true\n  tts_engine: kokoro\n")
    )
    out = tmp_path / "o.wav"

    def fake_edge(text: str, wav: Path) -> Path:
        wav.write_bytes(b"RIFFdata")
        return wav

    with patch.object(narrator, "_synthesize_kokoro", return_value=None), patch.object(
        narrator, "_synthesize_edge_tts", side_effect=fake_edge
    ) as edge, caplog.at_level(logging.WARNING):
        res = narrator.synthesize_speech("hello there", out)
    assert res == out
    edge.assert_called_once()
    assert any("falling back to edge-tts" in r.message for r in caplog.records)


def test_L8_words_inside_breakdown_window_dropped(tmp_path: Path) -> None:
    burner = AssCaptionBurner(_cfg(tmp_path))
    words = [
        WordTimestamp(text="before", start=1.0, end=1.4, confidence=1.0),
        WordTimestamp(text="inside", start=5.0, end=5.4, confidence=1.0),
        WordTimestamp(text="after", start=9.0, end=9.4, confidence=1.0),
    ]
    ass = tmp_path / "b.ass"
    burner.generate_ass(
        words,
        clip_start_offset=0.0,
        ass_path=ass,
        breakdown_callout="Look",
        breakdown_window=(4.0, 7.0),
    )
    text = ass.read_text(encoding="utf-8")
    assert "BEFORE" in text and "AFTER" in text
    assert "INSIDE" not in text


@pytest.mark.parametrize("codec,expect_retry", [("h264_nvenc", True), ("libx264", False)])
def test_L9_captions_stderr_logged_and_retry_only_nvenc(
    tmp_path: Path, caplog: Any, codec: str, expect_retry: bool
) -> None:
    burner = AssCaptionBurner(_cfg(tmp_path))
    ass = tmp_path / "c.ass"
    ass.write_text("x", encoding="utf-8")
    err = subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"ENCODERFAIL")
    calls: list[Any] = []

    def fake_run(cmd: Any, **kw: Any) -> Any:
        calls.append(cmd)
        if len(calls) == 1:
            raise err
        return MagicMock()

    with patch(
        "streamclipper.assembly.captions.subprocess.run", side_effect=fake_run
    ), caplog.at_level(logging.WARNING):
        if expect_retry:
            burner.burn_captions(tmp_path / "in.mp4", ass, tmp_path / "out.mp4", codec=codec)
            assert len(calls) == 2
        else:
            with pytest.raises(subprocess.CalledProcessError):
                burner.burn_captions(tmp_path / "in.mp4", ass, tmp_path / "out.mp4", codec=codec)
            assert len(calls) == 1
    assert any("ENCODERFAIL" in r.message for r in caplog.records)


def test_L10_kpipeline_cached_and_release_clears(tmp_path: Path) -> None:
    constructed: list[str] = []

    class FakeKPipeline:
        def __init__(self, lang_code: str) -> None:
            constructed.append(lang_code)

        def __call__(self, text: str, **kw: Any) -> Any:
            yield None, None, np.zeros(2400, dtype=np.float32)

    fake_mod = types.ModuleType("kokoro")
    fake_mod.KPipeline = FakeKPipeline  # type: ignore[attr-defined]
    narrator = TransformativeNarrator(_cfg(tmp_path))
    with patch.dict(sys.modules, {"kokoro": fake_mod}), patch(
        "streamclipper.assembly.narrator._is_espeak_available", return_value=True
    ):
        assert narrator._synthesize_kokoro("one", tmp_path / "a.wav") is not None
        assert narrator._synthesize_kokoro("two", tmp_path / "b.wav") is not None
        assert len(constructed) == 1
        narrator.release()
        assert narrator._kpipelines == {}
        narrator._synthesize_kokoro("three", tmp_path / "c.wav")
        assert len(constructed) == 2


def test_L10_prepare_narration_tracks_releases(tmp_path: Path) -> None:
    narrator = TransformativeNarrator(_cfg(tmp_path))
    narrator._kpipelines["a"] = object()
    with patch.object(narrator, "synthesize_speech", return_value=None):
        narrator.prepare_narration_tracks("intro", "outro", temp_dir=tmp_path)
    assert narrator._kpipelines == {}


def test_L11_invalid_trim_warns_and_records_beat_durations(tmp_path: Path, caplog: Any) -> None:
    composer = VideoComposer(_make_config(tmp_path, {"video": {"resolution": [270, 480]}}))
    c1 = _make_dummy_clip(duration=4.0, size=(1920, 1080))
    c2 = _make_dummy_clip(duration=5.0, size=(1920, 1080))
    with caplog.at_level(logging.WARNING):
        res = composer.compose_story_beats(
            [c1, c2], facecam=None, beat_trim_ranges=[(0.5, 3.5), (6.0, 9.0)]
        )
    assert any("trim range invalid" in r.message for r in caplog.records)
    assert res._beat_durations == pytest.approx([3.0, 5.0])


def test_L23_replay_badge_text_from_config(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "transformation:\n  replay_badge_text: 'CUSTOM RE'\n")
    with patch("streamclipper.assembly.hooks.render_replay_badge") as r:
        r.return_value = np.zeros((10, 10, 4), dtype=np.uint8)
        create_replay_badge(config=cfg)
        assert r.call_args.kwargs["text"] == "CUSTOM RE"
        create_replay_badge()
        assert "INSTANT REPLAY" in r.call_args.kwargs["text"]
        create_replay_badge(text="EXPLICIT", config=cfg)
        assert r.call_args.kwargs["text"] == "EXPLICIT"


def test_G7_text_truncated_to_budget_at_sentence_boundary(tmp_path: Path, caplog: Any) -> None:
    narrator = TransformativeNarrator(
        _cfg(tmp_path, "transformation:\n  enabled: true\n  tts_engine: edge-tts\n")
    )
    text = "This is the first sentence. " + "word " * 40 + "end."
    seen: list[str] = []

    def fake_edge(t: str, wav: Path) -> Path:
        seen.append(t)
        wav.write_bytes(b"RIFF")
        return wav

    with patch.object(narrator, "_synthesize_edge_tts", side_effect=fake_edge), caplog.at_level(
        logging.INFO
    ):
        narrator.synthesize_speech(text, tmp_path / "o.wav", max_seconds=5.0)
        assert seen == ["This is the first sentence."]
        assert any("truncated" in r.message for r in caplog.records)
        narrator.synthesize_speech(text, tmp_path / "o2.wav")
        assert seen[-1] == text.strip()


def test_G8_profile_cascade_used_and_capture_released(tmp_path: Path) -> None:
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"x")
    det = FacecamDetector(min_area_ratio=0.001)
    frontal = MagicMock()
    frontal.empty.return_value = False
    frontal.detectMultiScale.return_value = []
    profile = MagicMock()
    profile.empty.return_value = False
    profile.detectMultiScale.return_value = [(30, 30, 100, 100)]
    det.cascade = frontal
    det.profile_cascade = profile

    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        facecam_mod.cv2.CAP_PROP_FRAME_COUNT: 30,
        facecam_mod.cv2.CAP_PROP_FRAME_WIDTH: 1280,
        facecam_mod.cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }.get(prop, 0)
    cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    with patch.object(facecam_mod.cv2, "VideoCapture", return_value=cap):
        region = det.detect(vid, num_samples=3)
    assert region is not None
    assert profile.detectMultiScale.called
    cap.release.assert_called_once()


def test_L23_instant_replay_badge_uses_composer_config(tmp_path: Path) -> None:
    """L23 integration: VideoComposer.compose_instant_replay forwards its config to the badge."""
    cfg = _cfg(tmp_path, "transformation:\n  replay_badge_text: 'CUSTOM RE'\n")
    composer = VideoComposer(cfg)
    with patch("streamclipper.assembly.composer.compose_instant_replay") as replay:
        replay.return_value = (MagicMock(), 0.0)
        composer.compose_instant_replay(MagicMock(), peak_offset=1.0)
    assert replay.call_args.kwargs["config"] is cfg
