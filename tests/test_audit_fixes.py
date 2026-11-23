"""Regression tests for the monetization / quality audit fixes (P1-P6)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from streamclipper.analysis.copyright_risk import CopyrightRiskDetector
from streamclipper.assembly import narrator as narrator_mod
from streamclipper.assembly.captions import AssCaptionBurner
from streamclipper.assembly.compilation import build_countdown_filtergraph
from streamclipper.assembly.narrator import TransformativeNarrator
from streamclipper.config import Config, HardwareProfile
from streamclipper.models import (
    ClipMetadata,
    ClipResult,
    Highlight,
    PipelineState,
    Platform,
    StreamInfo,
    WordTimestamp,
)
from streamclipper.output.exporter import Exporter
from streamclipper.output.post_writer import write_post_file
from streamclipper.output.variety import VarietyState
from streamclipper.pipeline import Pipeline

from tests.test_pipeline_breakdown import _make_config, _make_highlight, _v3_meta, _run_assemble


def _cfg(tmp_path: Path, **sections: dict[str, Any]) -> Config:
    cfg = _make_config(tmp_path)
    cfg._data.setdefault("general", {})["output_dir"] = str(tmp_path / "output")
    cfg._data["general"]["cache_dir"] = str(tmp_path / "cache")
    cfg._data["general"]["temp_dir"] = str(tmp_path / "temp")
    for name, values in sections.items():
        cfg._data.setdefault(name, {}).update(values)
    return cfg


def _info(**over: Any) -> dict[str, Any]:
    base = {
        "mix_ran": True,
        "narration_expected": True,
        "intro_seconds": 4.0,
        "outro_seconds": 4.0,
        "callout_seconds": 1.0,
        "breakdown_seconds": 0.0,
        "spoken_seconds": 9.0,
        "final_duration": 30.0,
    }
    base.update(over)
    return base


def _assemble(tmp_path: Path, info: dict[str, Any], *, cfg: Config | None = None) -> dict[str, Any]:
    (tmp_path / "out.json").write_text("{}", encoding="utf-8")
    with patch.object(Exporter, "_empty_export_info", staticmethod(lambda: dict(info))):
        return _run_assemble(tmp_path, config=cfg or _cfg(tmp_path))


# ---------------------------------------------------------------- P1


def test_P1_tts_chain_retried_after_total_failure(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, transformation={"tts_retries": 1})
    n = TransformativeNarrator(cfg)
    calls: list[str] = []

    def fake_run(name, text, wav, language=None):
        calls.append(name)
        if len(calls) < 2:
            return None
        Path(wav).write_bytes(b"RIFFxxxx")
        return Path(wav)

    n._engine_chain = lambda lang=None: ["edge-tts"]  # type: ignore[method-assign]
    n._run_engine = fake_run  # type: ignore[method-assign]
    out = n.synthesize_speech("hello there", tmp_path / "x.wav")
    assert out is not None and len(calls) == 2


def test_P1_tts_retries_zero_does_not_retry(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, transformation={"tts_retries": 0})
    n = TransformativeNarrator(cfg)
    calls: list[str] = []
    n._engine_chain = lambda lang=None: ["edge-tts"]  # type: ignore[method-assign]
    n._run_engine = lambda name, t, w, language=None: calls.append(name)  # type: ignore[method-assign]
    assert n.synthesize_speech("hello there", tmp_path / "x.wav") is None
    assert calls == ["edge-tts"]


def _write_wav(path: Path, seconds: float) -> Path:
    sf.write(str(path), np.zeros(int(16000 * seconds), dtype="float32"), 16000)
    return path


class _FakeClip:
    """Plain stand-in for a MoviePy clip (MagicMock recurses forever in _close_clip_handles)."""

    duration = 20.0
    audio = object()
    _source_clip = None
    _source_clips = None

    def write_videofile(self, path, **kwargs):
        Path(path).write_bytes(b"raw")

    def close(self):
        pass


def _fake_clip() -> _FakeClip:
    return _FakeClip()


def _export(tmp_path: Path, *, mode: str, tracks, **kwargs: Any) -> Exporter:
    cfg = _cfg(tmp_path)
    exporter = Exporter(cfg)
    out = tmp_path / "final.mp4"

    def fake_finalize(inp, outp):
        Path(outp).write_bytes(b"x")
        return Path(outp)

    mixed = tmp_path / "mixed.mp4"
    mixed.write_bytes(b"m")
    exporter.narrator.prepare_narration_tracks = lambda **kw: tracks  # type: ignore[method-assign]
    exporter.narrator.mix_narration_with_ducking = lambda **kw: mixed  # type: ignore[method-assign]
    exporter.narrator.normalize_audio = lambda a, b: a  # type: ignore[method-assign]
    exporter._finalize_without_captions = fake_finalize  # type: ignore[method-assign]
    exporter.caption_burner.burn_captions = lambda **kw: fake_finalize(kw["input_video"], kw["output_video"])  # type: ignore[method-assign]
    exporter.export(clip=_fake_clip(), words=[], clip_start_offset=0.0, output_path=out, mode=mode, **kwargs)
    return exporter


def test_P1_export_info_records_mix_and_probed_seconds(tmp_path: Path) -> None:
    intro = _write_wav(tmp_path / "i.wav", 3.0)
    outro = _write_wav(tmp_path / "o.wav", 2.0)
    exporter = _export(
        tmp_path,
        mode="narrative",
        tracks=(intro, outro),
        narration_intro="hello world",
        narration_outro="bye now",
    )
    info = exporter.last_export_info
    assert info["mix_ran"] is True and info["narration_expected"] is True
    assert info["intro_seconds"] == pytest.approx(3.0, abs=0.05)
    assert info["outro_seconds"] == pytest.approx(2.0, abs=0.05)
    assert info["callout_seconds"] == 0.0 and info["breakdown_seconds"] == 0.0
    assert info["spoken_seconds"] == pytest.approx(5.0, abs=0.1)
    assert set(info) >= {"final_duration"}


def test_P1_export_info_mix_not_run_when_tts_produced_nothing(tmp_path: Path) -> None:
    exporter = _export(
        tmp_path, mode="narrative", tracks=(None, None), narration_intro="hello world"
    )
    info = exporter.last_export_info
    assert info["narration_expected"] is True
    assert info["mix_ran"] is False
    assert info["spoken_seconds"] == 0.0


def test_P1_export_info_raw_mode_expects_no_narration_and_resets(tmp_path: Path) -> None:
    exporter = _export(tmp_path, mode="raw", tracks=(None, None), narration_intro="hello")
    assert exporter.last_export_info["narration_expected"] is False
    exporter.last_export_info["mix_ran"] = True  # stale value from a previous export
    exporter.last_export_info["spoken_seconds"] = 99.0
    exporter.narrator.prepare_narration_tracks = lambda **kw: (None, None)  # type: ignore[method-assign]
    exporter.export(
        clip=_fake_clip(), words=[], clip_start_offset=0.0,
        output_path=tmp_path / "again.mp4", mode="raw",
    )
    assert exporter.last_export_info["mix_ran"] is False
    assert exporter.last_export_info["spoken_seconds"] == 0.0


def test_P1_unnarrated_clip_is_quarantined_not_shipped(tmp_path: Path) -> None:
    info = _assemble(tmp_path, _info(mix_ran=False, spoken_seconds=0.0, intro_seconds=0.0))
    result = info["result"]
    assert result.quarantined is True
    assert "narration" in result.quarantine_reason
    qdir = tmp_path / "output" / "_quarantine" / "GTA 6" / "Fleet"
    assert Path(result.video_path).is_file() and qdir in Path(result.video_path).parents
    assert not (tmp_path / "out.mp4").exists()
    assert not (tmp_path / "out_post.txt").exists()
    assert not list(qdir.rglob("*_post.txt"))


def test_P1_gate_can_be_disabled(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, output={"require_narration": False})
    result = _assemble(tmp_path, _info(mix_ran=False), cfg=cfg)["result"]
    assert result.quarantined is False
    assert (tmp_path / "out_post.txt").is_file()


def test_P1_raw_clip_is_never_quarantined(tmp_path: Path) -> None:
    result = _assemble(tmp_path, _info(mix_ran=False, narration_expected=False))["result"]
    assert result.quarantined is False


def test_P1_clipresult_quarantine_fields_default_and_roundtrip(tmp_path: Path) -> None:
    d = ClipResult(
        video_path=tmp_path / "a.mp4", metadata_path=None, metadata=_v3_meta(), duration_seconds=1.0
    ).to_dict()
    d.pop("quarantined")
    d.pop("quarantine_reason")
    back = ClipResult.from_dict(d)
    assert back.quarantined is False and back.quarantine_reason == ""
    d["quarantined"] = True
    d["quarantine_reason"] = "why"
    again = ClipResult.from_dict(d)
    assert again.quarantined is True and again.quarantine_reason == "why"


def test_P1_pipeline_skips_quarantined_clip_in_state_and_variety(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    pipeline = Pipeline(cfg)
    stream = StreamInfo(
        url="https://youtube.com/watch?v=q1", title="t", platform=Platform.YOUTUBE,
        streamer="Fleet", game="GTA 6", duration_seconds=100,
    )
    state = PipelineState(stream_info=stream, selected_highlights=[_make_highlight()])
    seg = tmp_path / "seg.mp4"
    seg.write_bytes(b"x")
    quarantined = ClipResult(
        video_path=tmp_path / "q.mp4", metadata_path=None, metadata=_v3_meta(),
        duration_seconds=1.0, quarantined=True, quarantine_reason="no voice",
    )
    scorer = MagicMock()
    scorer.generate_metadata.return_value = _v3_meta()
    with (
        patch("streamclipper.pipeline.OllamaScorer", return_value=scorer),
        patch("streamclipper.pipeline.download_segment", return_value=seg),
        patch.object(Pipeline, "_assemble_highlight", return_value=quarantined),
        patch.object(Pipeline, "_assess_gameplay", return_value=False),
    ):
        out = asyncio.run(pipeline._step_assemble(state, mode="narrative"))
    assert out.clips == []
    entries = VarietyState(tmp_path / "cache" / "variety_state.json")._load()
    assert entries == []


def test_P1_mcp_render_skips_post_file_for_quarantined(tmp_path: Path, monkeypatch) -> None:
    from streamclipper.mcp_server import server as mcp_server

    cfg = _cfg(tmp_path)
    cfg._data.setdefault("output", {})["write_post_file"] = True
    seg = tmp_path / "seg.mp4"
    seg.write_bytes(b"x")
    video = tmp_path / "output" / "_quarantine" / "clip_01.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    def fake_assemble(self, highlight, metadata, *a, **kw):
        return ClipResult(
            video_path=video, metadata_path=None, metadata=metadata, duration_seconds=1.0,
            quarantined=True, quarantine_reason="no voice",
        )

    monkeypatch.setattr(mcp_server, "download_segment", lambda *a, **k: seg)
    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", fake_assemble)
    pipeline = Pipeline(cfg)
    h = _make_highlight(moment_type="funny")
    result, _mode = asyncio.run(
        mcp_server._render_single_highlight(
            pipeline, cfg, "https://youtube.com/watch?v=m1", h, "Fleet", "GTA 6", 1, "narrative"
        )
    )
    assert result.quarantined and result.to_dict()["quarantined"] is True
    assert not video.with_name("clip_01_post.txt").exists()


# ---------------------------------------------------------------- P2


def test_P2_replay_without_breakdown_voice_is_not_commentary(tmp_path: Path) -> None:
    pipeline = Pipeline(_cfg(tmp_path, output={"min_commentary_ratio": 0.0}))
    kw = dict(moment_dur=30.0, mode="breakdown", narration_seconds=6.0, replay_window=5.0, replay_speed=0.5)
    silent = pipeline._plan_budget(**kw)
    assert silent["coverage"] == pytest.approx(6.0 / silent["total"], abs=0.001)
    voiced = pipeline._plan_budget(**kw, breakdown_voiced=True, breakdown_seconds=4.0)
    assert voiced["coverage"] == pytest.approx((6.0 + 4.0) / voiced["total"], abs=0.001)
    long_speech = pipeline._plan_budget(**kw, breakdown_voiced=True, breakdown_seconds=60.0)
    assert long_speech["coverage"] == pytest.approx((6.0 + silent["replay_len"]) / long_speech["total"], abs=0.001)


def test_P2_measured_commentary_written_to_clip_json(tmp_path: Path) -> None:
    _assemble(tmp_path, _info(spoken_seconds=12.0, final_duration=30.0))
    data = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert data["commentary_seconds"] == 12.0
    assert data["commentary_coverage"] == 0.4
    assert "commentary_coverage_estimate" in data and "commentary_ratio" in data
    assert "low_commentary" not in data


def test_P2_low_measured_commentary_is_flagged_not_quarantined(tmp_path: Path, caplog) -> None:
    with caplog.at_level("WARNING"):
        info = _assemble(tmp_path, _info(spoken_seconds=3.0, final_duration=30.0))
    data = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert data["commentary_coverage"] == 0.1
    assert data["low_commentary"] is True
    assert info["result"].quarantined is False
    assert any("Low commentary" in r.message for r in caplog.records)


# ---------------------------------------------------------------- P3


def _top3_state(n: int = 4, **hl: Any) -> PipelineState:
    stream = StreamInfo(
        url="https://youtube.com/watch?v=top3", title="t", platform=Platform.YOUTUBE,
        streamer="Fleet", game="GTA 6", duration_seconds=4000,
    )
    hls = [
        Highlight(start=100.0 * (i + 1), end=100.0 * (i + 1) + 30, score=9.0 - i, title=f"h{i}",
                  reason="r", final_score=9.0 - i)
        for i in range(n)
    ]
    return PipelineState(stream_info=stream, selected_highlights=hls)


class _Scorer:
    def __init__(self, title: str | None = None) -> None:
        self.title = title
        self.meta_calls: list[dict[str, Any]] = []

    def generate_countdown_lines(self, hs, streamer, game):
        return [f"line {i}" for i in range(len(hs))]

    def generate_metadata(self, highlight, streamer, game, url, mode="narrative", hook_style="", language=""):
        self.meta_calls.append({"hook_style": hook_style, "language": language})
        if self.title is None:
            raise RuntimeError("llm down")
        return ClipMetadata(hook_title=self.title, caption="cap", hashtags=["GTA6"], streamer=streamer, game=game,
                            source_url=url, clip_start=0, clip_end=1, highlight_score=9.0)


def _run_top3(tmp_path: Path, state: PipelineState, scorer: Any, *, cfg: Config | None = None,
              info: dict[str, Any] | None = None, gameplay_flags: list[bool] | None = None):
    cfg = cfg or _cfg(tmp_path)
    pipeline = Pipeline(cfg)
    seg = tmp_path / "seg.mp4"
    seg.write_bytes(b"x")
    flags = list(gameplay_flags or [])

    def fake_assess(self, path):
        return flags.pop(0) if flags else False

    info_override = info
    seen: dict[str, Any] = {}

    def build(segs, peaks, cams, lines, out_path, config, narrator=None, info=None):
        seen["n"] = len(segs)
        seen["out"] = out_path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"top3")
        if info is not None:
            info.update(info_override if info_override is not None else _info())
        return Path(out_path)

    with (
        patch("streamclipper.pipeline.download_segment", return_value=seg),
        patch.object(Pipeline, "_assess_gameplay", fake_assess),
        patch("streamclipper.assembly.compilation.build_top3_compilation", build),
        patch("streamclipper.assembly.facecam.FacecamDetector.detect", return_value=None),
    ):
        result = asyncio.run(
            pipeline._build_top3(state, scorer, tmp_path / "segs", 3.0, "Fleet", "GTA 6")
        )
    return result, seen, pipeline


def test_P3_top3_skips_high_music_and_needs_three(tmp_path: Path) -> None:
    state = _top3_state(4)
    state.selected_highlights[0].music_risk = "HIGH_MUSIC_DETECTED"
    cfg = _cfg(tmp_path, analysis={"skip_high_music": True})
    result, seen, _ = _run_top3(tmp_path, state, _Scorer(), cfg=cfg)
    assert result is not None and seen["n"] == 3  # 3 left after dropping the HIGH one

    state = _top3_state(3)
    state.selected_highlights[0].music_risk = "HIGH_MUSIC_DETECTED"
    result, seen, _ = _run_top3(tmp_path, state, _Scorer(), cfg=cfg)
    assert result is None and not seen


def test_P3_top3_skips_non_gameplay_segments(tmp_path: Path) -> None:
    result, seen, _ = _run_top3(tmp_path, _top3_state(3), _Scorer(), gameplay_flags=[True, False, False])
    assert result is None  # only two gameplay segments left

    result, seen, _ = _run_top3(tmp_path, _top3_state(4), _Scorer(), gameplay_flags=[True, False, False, False])
    assert result is not None and seen["n"] == 3


def test_P3_top3_respects_daily_cap_and_records_variety(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, output={"max_clips_per_streamer_per_day": 1})
    VarietyState(tmp_path / "cache" / "variety_state.json").record("Fleet", "narrative", "question", "")
    queued_path = tmp_path / "output" / "queued" / "top3.mp4"
    with patch(
        "streamclipper.output.organizer.FileOrganizer.get_queued_path", return_value=queued_path
    ) as qp:
        result, seen, _ = _run_top3(tmp_path, _top3_state(3), _Scorer(), cfg=cfg)
    assert qp.called and seen["out"] == queued_path
    entries = VarietyState(tmp_path / "cache" / "variety_state.json")._load()
    assert entries[-1]["format"] == "top3" and entries[-1]["streamer"] == "Fleet"


def test_P3_top3_title_uses_llm_hook_with_rotated_style_and_language(tmp_path: Path) -> None:
    scorer = _Scorer(title="NOBODY EXPECTED THIS ENDING")
    result, _, _ = _run_top3(tmp_path, _top3_state(3), scorer)
    assert result.metadata.hook_title == "NOBODY EXPECTED THIS ENDING"
    assert scorer.meta_calls[0]["hook_style"] == "question"
    assert scorer.meta_calls[0]["language"]
    assert "Top3" in result.metadata.hashtags


def test_P3_top3_title_template_fallback_rotates(tmp_path: Path) -> None:
    titles = set()
    for i in range(4):
        result, _, _ = _run_top3(tmp_path, _top3_state(3), _Scorer(title=None))
        titles.add(result.metadata.hook_title)
    assert len(titles) >= 4


def test_P3_top3_quarantined_when_countdown_voice_missing(tmp_path: Path) -> None:
    result, _, _ = _run_top3(
        tmp_path, _top3_state(3), _Scorer(), info=_info(mix_ran=False, spoken_seconds=0.0)
    )
    assert result.quarantined is True
    assert "_quarantine" in Path(result.video_path).parts
    assert not list((tmp_path / "output").rglob("*_post.txt"))
    # not recorded in variety state
    assert VarietyState(tmp_path / "cache" / "variety_state.json")._load() == []


def test_P3_countdown_filtergraph_applies_final_loudnorm() -> None:
    graph = build_countdown_filtergraph([(0.2, 2.0)], 30.0, target_lufs=-14)
    assert "loudnorm=I=-14:TP=-1.5:LRA=11" in graph
    assert "loudnorm" not in build_countdown_filtergraph([(0.2, 2.0)], 30.0)


# ---------------------------------------------------------------- P4


def _wav_noise(path: Path, seconds: float) -> Path:
    rng = np.random.default_rng(0)
    sf.write(str(path), (rng.standard_normal(int(16000 * seconds)) * 0.1).astype("float32"), 16000)
    return path


def test_P4_talking_over_audio_is_unknown_not_clean(tmp_path: Path) -> None:
    wav = _wav_noise(tmp_path / "a.wav", 10.0)
    words = [WordTimestamp(text="w", start=float(i), end=float(i) + 0.9) for i in range(10)]
    assert CopyrightRiskDetector().detect(wav, 0.0, 10.0, words) == ("UNKNOWN", 0.0)


def test_P4_too_little_gap_audio_is_unknown_but_enough_is_clean(tmp_path: Path) -> None:
    wav = _wav_noise(tmp_path / "a.wav", 10.0)
    # 1.2s of gaps only (< copyright.min_gap_seconds = 2.0)
    words = [WordTimestamp(text="a", start=0.0, end=4.0), WordTimestamp(text="b", start=4.6, end=9.4)]
    assert CopyrightRiskDetector().detect(wav, 0.0, 10.0, words)[0] == "UNKNOWN"
    # 4s of gap audio actually tested
    words = [WordTimestamp(text="a", start=0.0, end=3.0), WordTimestamp(text="b", start=7.0, end=10.0)]
    assert CopyrightRiskDetector().detect(wav, 0.0, 10.0, words)[0] == "CLEAN"


def test_P4_min_gap_key_configurable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg._data.setdefault("analysis", {}).setdefault("copyright", {})["min_gap_seconds"] = 0.5
    wav = _wav_noise(tmp_path / "a.wav", 10.0)
    words = [WordTimestamp(text="a", start=0.0, end=4.0), WordTimestamp(text="b", start=4.6, end=9.4)]
    assert CopyrightRiskDetector(cfg).detect(wav, 0.0, 10.0, words)[0] == "CLEAN"


def test_P4_unknown_is_not_skipped_like_high(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, analysis={"skip_high_music": True})
    info = _run_assemble(tmp_path, config=cfg, highlight=_make_highlight(music_risk="UNKNOWN"))
    assert info["result"] is not None


def test_P4_post_file_reminds_on_unknown(tmp_path: Path) -> None:
    video = tmp_path / "c.mp4"
    video.write_bytes(b"x")
    stream = StreamInfo(url="u", title="t", platform=Platform.YOUTUBE, streamer="Fleet", game="g", duration_seconds=1)
    meta = _v3_meta(music_risk="UNKNOWN")
    text = write_post_file(video, meta, stream).read_text(encoding="utf-8")
    assert (
        "Music check inconclusive (streamer talks over the audio) - listen for background "
        "music before posting."
    ) in text
    clean = write_post_file(video, _v3_meta(music_risk="CLEAN"), stream).read_text(encoding="utf-8")
    assert "Music check inconclusive" not in clean


# ---------------------------------------------------------------- P5


def test_P5_disclosure_wording_is_accurate(tmp_path: Path) -> None:
    video = tmp_path / "c.mp4"
    video.write_bytes(b"x")
    stream = StreamInfo(url="u", title="t", platform=Platform.YOUTUBE, streamer="Fleet", game="g", duration_seconds=1)
    text = write_post_file(video, _v3_meta(), stream).read_text(encoding="utf-8")
    assert (
        "AI voiceover: YouTube requires the 'Altered or synthetic content' label when the voice is "
        "cloned from a real person or could pass as a real person; for an obviously synthetic "
        "narrator it's optional (ticking it is always safe). Never clone the streamer's voice."
    ) in text
    assert "Tick YouTube's" not in text
    assert "Post at most 2" in text and "Shorts per day" in text


# ---------------------------------------------------------------- P6


def test_P6_variety_rotates_caption_presets_per_streamer(tmp_path: Path) -> None:
    v = VarietyState(tmp_path / "v.json")
    seen = []
    for _ in range(4):
        idx = v.next_caption_preset(3, streamer="A")
        seen.append(idx)
        v.record("A", "narrative", "question", "", caption_preset=idx)
    assert seen == [0, 1, 2, 0]
    assert v.next_caption_preset(3, streamer="B") == 0  # independent per streamer
    v.record_caption_preset("B", 0)
    assert v.next_caption_preset(3, streamer="B") == 1
    assert v.clips_today("B") == 0  # caption-only entries are not clips
    assert VarietyState(tmp_path / "v.json").next_caption_preset(3, streamer="A") == 1  # persisted


def test_P6_default_presets_present_and_distinct(tmp_path: Path) -> None:
    presets = _cfg(tmp_path).get("captions.style_presets")
    assert len(presets) >= 3
    assert len({p["highlight_color"] for p in presets}) >= 3
    for p in presets:
        assert {"font", "primary_color", "highlight_color", "outline_color"} <= set(p)


def test_P6_generate_ass_preset_overrides_caption_styles_not_chat_ticker(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    burner = AssCaptionBurner(cfg)
    words = [WordTimestamp(text="hello", start=0.0, end=0.5), WordTimestamp(text="world", start=0.5, end=1.0)]
    base = burner.generate_ass(words, 0.0, tmp_path / "a.ass").read_text(encoding="utf-8")
    preset = {"font": "Segoe UI Black", "primary_color": "#FFEB3B", "highlight_color": "#FF4081",
              "outline_color": "#112233"}
    out = burner.generate_ass(words, 0.0, tmp_path / "b.ass", style_preset=preset).read_text(encoding="utf-8")
    hormozi = [l for l in out.splitlines() if l.startswith("Style: Hormozi")][0]
    assert "Segoe UI Black" in hormozi and "&H003BEBFF" in hormozi.upper()
    ticker_base = [l for l in base.splitlines() if l.startswith("Style: ChatTicker")][0]
    ticker = [l for l in out.splitlines() if l.startswith("Style: ChatTicker")][0]
    assert ticker == ticker_base  # ChatTicker untouched
    # the burner is restored afterwards: a later call without preset equals the baseline
    again = burner.generate_ass(words, 0.0, tmp_path / "c.ass").read_text(encoding="utf-8")
    assert again == base


def test_P6_pipeline_passes_rotated_preset_and_can_disable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    default_presets = cfg.get("captions.style_presets", [])
    info = _run_assemble(tmp_path, config=cfg)
    kwargs = info["export"].call_args[1]
    assert kwargs["style_preset"] == default_presets[0]  # first rotation slot
    second = _run_assemble(tmp_path, config=_cfg(tmp_path))  # MCP path: rotation persisted
    assert second["export"].call_args[1]["style_preset"] == default_presets[1]
    off = _run_assemble(tmp_path, config=_cfg(tmp_path, captions={"rotate_styles": False}))
    assert "style_preset" not in off["export"].call_args[1]


def test_P6_exporter_forwards_style_preset_to_generate_ass(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    exporter = Exporter(cfg)
    exporter.narrator.normalize_audio = lambda a, b: a  # type: ignore[method-assign]
    captured: dict[str, Any] = {}

    def fake_generate(**kw):
        captured.update(kw)
        return kw["ass_path"]

    exporter.caption_burner.generate_ass = fake_generate  # type: ignore[method-assign]
    exporter.caption_burner.burn_captions = lambda **kw: kw["output_video"]  # type: ignore[method-assign]
    preset = {"font": "Arial Black"}
    exporter.export(
        clip=_fake_clip(),
        words=[WordTimestamp(text="a", start=0.0, end=0.4)],
        clip_start_offset=0.0,
        output_path=tmp_path / "o.mp4",
        mode="raw",
        style_preset=preset,
    )
    assert captured["style_preset"] == preset
