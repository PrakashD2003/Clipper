"""Unit tests for Package 2.3 — WhisperX Transcriber."""
from __future__ import annotations

import builtins
import gc
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import streamclipper.analysis.transcriber as transcriber_mod
import streamclipper.config as config_mod
from streamclipper.analysis.transcriber import Transcriber
from streamclipper.config import Config, HardwareProfile
from streamclipper.models import TranscriptionResult, TranscriptSegment, WordTimestamp


def _make_rtx_4050_profile() -> HardwareProfile:
    return HardwareProfile(
        tier="cuda_mid_tensor",
        gpu_name="NVIDIA GeForce RTX 4050 Laptop GPU",
        vram_gb=6.44,
        compute_capability="8.9",
        has_tensor_cores=True,
        has_nvenc=True,
        device="cuda",
        model_size="large-v2",
        compute_type="int8_float16",
        batch_size=8,
        llm_model="qwen3:8b",
        video_codec="h264_nvenc",
    )


def _make_gtx_1660_super_profile() -> HardwareProfile:
    return HardwareProfile(
        tier="cuda_mid_no_tensor",
        gpu_name="NVIDIA GeForce GTX 1660 SUPER",
        vram_gb=6.44,
        compute_capability="7.5",
        has_tensor_cores=False,
        has_nvenc=True,
        device="cuda",
        model_size="medium",
        compute_type="int8",
        batch_size=4,
        llm_model="qwen3:4b",
        video_codec="h264_nvenc",
    )


def _make_cpu_profile() -> HardwareProfile:
    return HardwareProfile(
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


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool = True,
    events: list[str] | None = None,
) -> dict[str, int]:
    """Patch transcriber_mod.torch with a lightweight fake CUDA backend."""
    counters = {"empty_cache": 0}

    def _empty_cache() -> None:
        counters["empty_cache"] += 1
        if events is not None:
            events.append("empty_cache")

    fake_cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        empty_cache=_empty_cache,
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr(transcriber_mod, "torch", fake_torch)
    return counters


def _install_fake_whisperx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[str] | None = None,
    aligned_segments: list[dict[str, Any]] | None = None,
    detected_language: str = "en",
) -> dict[str, Any]:
    """Install a mock `whisperx` module in sys.modules and capture all call args."""
    calls: dict[str, Any] = {}

    if aligned_segments is None:
        aligned_segments = [
            {
                "text": " No way bro!",
                "start": 10.0,
                "end": 11.4,
                "words": [
                    {"word": "No", "start": 10.0, "end": 10.3, "score": 0.97},
                    {"word": "way", "start": 10.35, "end": 10.8, "score": 0.94},
                    {"word": "bro!", "start": 10.85, "end": 11.4, "score": 0.91},
                ],
            }
        ]

    class _FakeASRModel:
        def transcribe(self, audio: Any, batch_size: int = 4) -> dict[str, Any]:
            calls["transcribe"] = {"audio": audio, "batch_size": batch_size}
            if events is not None:
                events.append("transcribe")
            return {
                "language": detected_language,
                "segments": [{"text": "No way bro!", "start": 10.0, "end": 11.4}],
            }

    def _load_model(model_size: str, device: str, compute_type: str = "int8") -> Any:
        calls["load_model"] = {
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type,
        }
        if events is not None:
            events.append("load_model")
        return _FakeASRModel()

    def _load_audio(audio_path: str) -> str:
        calls["load_audio"] = audio_path
        if events is not None:
            events.append("load_audio")
        return f"audio_buffer:{audio_path}"

    def _load_align_model(language_code: str, device: str) -> tuple[object, dict[str, str]]:
        calls["load_align_model"] = {
            "language_code": language_code,
            "device": device,
        }
        if events is not None:
            events.append("load_align_model")
        return object(), {"language": language_code}

    def _align(
        segments: list[dict[str, Any]],
        model_a: Any,
        metadata: Any,
        audio: Any,
        device: str,
        return_char_alignments: bool = False,
    ) -> dict[str, Any]:
        calls["align"] = {
            "segments": segments,
            "metadata": metadata,
            "audio": audio,
            "device": device,
            "return_char_alignments": return_char_alignments,
        }
        if events is not None:
            events.append("align")
        return {"segments": aligned_segments}

    fake_whisperx = types.ModuleType("whisperx")
    fake_whisperx.load_model = _load_model  # type: ignore[attr-defined]
    fake_whisperx.load_audio = _load_audio  # type: ignore[attr-defined]
    fake_whisperx.load_align_model = _load_align_model  # type: ignore[attr-defined]
    fake_whisperx.align = _align  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    return calls


def _install_fake_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[str] | None = None,
) -> dict[str, Any]:
    """Install a mock `faster_whisper` module in sys.modules and capture calls."""
    calls: dict[str, Any] = {}

    class _FakeWhisperModel:
        def __init__(self, model_size: str, device: str = "cpu", compute_type: str = "int8") -> None:
            calls["init"] = {
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
            }
            if events is not None:
                events.append("fw_init")

        def transcribe(
            self,
            audio_path: str,
            language: str | None = None,
            word_timestamps: bool = False,
        ) -> tuple[Any, Any]:
            calls["transcribe"] = {
                "audio_path": audio_path,
                "language": language,
                "word_timestamps": word_timestamps,
            }
            if events is not None:
                events.append("fw_transcribe")

            def _gen() -> Any:
                yield SimpleNamespace(
                    text=" Clutch escape! ",
                    start=42.0,
                    end=43.5,
                    words=[
                        SimpleNamespace(word=" Clutch", start=42.0, end=42.6, probability=0.96),
                        SimpleNamespace(word=" escape!", start=42.65, end=43.5, probability=0.92),
                    ],
                )

            info = SimpleNamespace(language=language or "en")
            return _gen(), info

    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    return calls


@pytest.fixture
def dummy_wav(tmp_path: Path) -> Path:
    wav_path = tmp_path / "stream_16k.wav"
    wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    return wav_path


class TestTranscriberHardwareProfiles:
    def test_rtx_4050_profile_passes_resolved_params_and_logs_profile(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        dummy_wav: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", tmp_path / "missing.yaml")
        _install_fake_torch(monkeypatch, cuda_available=True)
        calls = _install_fake_whisperx(monkeypatch)

        cfg = Config(hardware_profile=_make_rtx_4050_profile())
        with caplog.at_level(logging.INFO, logger=transcriber_mod.__name__):
            transcriber = Transcriber(cfg)

        assert "tier=cuda_mid_tensor" in caplog.text
        assert "model=large-v2" in caplog.text
        assert "compute_type=int8_float16" in caplog.text
        assert "batch_size=8" in caplog.text

        result = transcriber.transcribe(dummy_wav)

        assert calls["load_model"] == {
            "model_size": "large-v2",
            "device": "cuda",
            "compute_type": "int8_float16",
        }
        assert calls["transcribe"]["batch_size"] == 8
        assert calls["load_align_model"] == {"language_code": "en", "device": "cuda"}
        assert calls["align"]["return_char_alignments"] is False
        assert isinstance(result, TranscriptionResult)

    def test_gtx_1660_super_profile_passes_resolved_params(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        dummy_wav: Path,
    ) -> None:
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", tmp_path / "missing.yaml")
        _install_fake_torch(monkeypatch, cuda_available=True)
        calls = _install_fake_whisperx(monkeypatch)

        cfg = Config(hardware_profile=_make_gtx_1660_super_profile())
        transcriber = Transcriber(cfg)
        transcriber.transcribe(dummy_wav)

        assert calls["load_model"] == {
            "model_size": "medium",
            "device": "cuda",
            "compute_type": "int8",
        }
        assert calls["transcribe"]["batch_size"] == 4

    def test_whisperx_forwards_configured_language_when_supported(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        dummy_wav: Path,
    ) -> None:
        user_cfg = tmp_path / "custom_lang.yaml"
        user_cfg.write_text("transcription:\n  language: en\n", encoding="utf-8")
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", user_cfg)
        _install_fake_torch(monkeypatch, cuda_available=False)

        captured_transcribe: dict[str, Any] = {}

        class _LangAwareASRModel:
            def transcribe(
                self, audio: Any, batch_size: int = 4, language: str | None = None
            ) -> dict[str, Any]:
                captured_transcribe["batch_size"] = batch_size
                captured_transcribe["language"] = language
                return {"language": language or "en", "segments": []}

        fake_wx = types.ModuleType("whisperx")
        fake_wx.load_model = lambda *a, **kw: _LangAwareASRModel()  # type: ignore[attr-defined]
        fake_wx.load_audio = lambda p: f"buf:{p}"  # type: ignore[attr-defined]
        fake_wx.load_align_model = lambda **kw: (object(), {})  # type: ignore[attr-defined]
        fake_wx.align = lambda *a, **kw: {"segments": []}  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "whisperx", fake_wx)

        cfg = Config(user_config_path=user_cfg, hardware_profile=_make_cpu_profile())
        Transcriber(cfg).transcribe(dummy_wav)
        assert captured_transcribe == {"batch_size": 4, "language": "en"}


class TestTranscriberWordConversion:
    def test_word_level_timestamp_conversion_and_unaligned_tokens(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        dummy_wav: Path,
    ) -> None:
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", tmp_path / "missing.yaml")
        _install_fake_torch(monkeypatch, cuda_available=True)
        aligned_segments = [
            {
                "text": " GTA 6 looks wild! ",
                "start": 1.0,
                "end": 2.8,
                "words": [
                    {"word": "GTA", "start": 1.0, "end": 1.4, "score": 0.99},
                    # Unaligned digit token missing start/end/score (common in wav2vec2)
                    {"word": "6"},
                    {"word": "looks", "start": 1.6, "end": 2.0, "score": 0.95},
                    {"word": "wild!", "start": 2.1, "end": 2.8, "score": 0.92},
                ],
            },
            {
                "text": " Let's go! ",
                "start": 3.0,
                "end": 3.9,
                "words": [
                    {"word": "Let's", "start": 3.0, "end": 3.4, "score": 0.88},
                    {"word": "go!", "start": 3.45, "end": 3.9, "score": 0.93},
                ],
            },
        ]
        _install_fake_whisperx(
            monkeypatch,
            aligned_segments=aligned_segments,
            detected_language="en",
        )

        cfg = Config(hardware_profile=_make_rtx_4050_profile())
        result = Transcriber(cfg).transcribe(dummy_wav)

        assert result.language == "en"
        assert result.full_text == "GTA 6 looks wild! Let's go!"
        assert len(result.segments) == 2

        seg0 = result.segments[0]
        assert seg0.text == "GTA 6 looks wild!"
        assert seg0.start == 1.0
        assert seg0.end == 2.8
        assert len(seg0.words) == 4
        assert seg0.words[0] == WordTimestamp(text="GTA", start=1.0, end=1.4, confidence=0.99)
        # Unaligned token falls back monotonically to previous word's end
        assert seg0.words[1] == WordTimestamp(text="6", start=1.4, end=1.4, confidence=0.0)
        assert seg0.words[3] == WordTimestamp(text="wild!", start=2.1, end=2.8, confidence=0.92)


class TestTranscriberSafetyAndVRAM:
    @pytest.mark.parametrize("bad_compute_type", ["float16", "int8_float16"])
    def test_cpu_compute_type_safety_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dummy_wav: Path,
        bad_compute_type: str,
    ) -> None:
        _install_fake_torch(monkeypatch, cuda_available=False)
        calls = _install_fake_whisperx(monkeypatch)

        # Simulate a config whose resolved values still report float16/int8_float16 on CPU
        class _FakeConfig:
            hardware_profile = _make_cpu_profile()

            def get(self, key: str, default: Any = None) -> Any:
                values = {
                    "transcription.engine": "whisperx",
                    "transcription.device": "cpu",
                    "transcription.model_size": "small",
                    "transcription.compute_type": bad_compute_type,
                    "transcription.batch_size": 4,
                    "transcription.language": "en",
                }
                return values.get(key, default)

        transcriber = Transcriber(_FakeConfig())  # type: ignore[arg-type]
        assert transcriber.compute_type == "int8"

        transcriber.transcribe(dummy_wav)
        assert calls["load_model"]["device"] == "cpu"
        assert calls["load_model"]["compute_type"] == "int8"

    def test_two_stage_vram_cleanup_order_and_count_in_whisperx(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        dummy_wav: Path,
    ) -> None:
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", tmp_path / "missing.yaml")
        events: list[str] = []
        counters = _install_fake_torch(monkeypatch, cuda_available=True, events=events)
        _install_fake_whisperx(monkeypatch, events=events)

        gc_calls = {"count": 0}
        real_collect = gc.collect

        def _tracked_gc() -> int:
            gc_calls["count"] += 1
            events.append("gc_collect")
            return real_collect()

        monkeypatch.setattr(transcriber_mod.gc, "collect", _tracked_gc)

        cfg = Config(hardware_profile=_make_gtx_1660_super_profile())
        Transcriber(cfg).transcribe(dummy_wav)

        assert counters["empty_cache"] == 2
        assert gc_calls["count"] == 2
        assert events == [
            "load_model",
            "load_audio",
            "transcribe",
            "gc_collect",
            "empty_cache",
            "load_align_model",
            "align",
            "gc_collect",
            "empty_cache",
        ]

    def test_vram_cleanup_skips_empty_cache_when_cuda_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        dummy_wav: Path,
    ) -> None:
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", tmp_path / "missing.yaml")
        counters = _install_fake_torch(monkeypatch, cuda_available=False)
        _install_fake_whisperx(monkeypatch)

        cfg = Config(hardware_profile=_make_cpu_profile())
        Transcriber(cfg).transcribe(dummy_wav)

        assert counters["empty_cache"] == 0


class TestTranscriberFallbackAndErrors:
    def test_faster_whisper_fallback_when_whisperx_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        dummy_wav: Path,
    ) -> None:
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", tmp_path / "missing.yaml")
        events: list[str] = []
        counters = _install_fake_torch(monkeypatch, cuda_available=True, events=events)
        fw_calls = _install_fake_faster_whisper(monkeypatch, events=events)

        # Force `import whisperx` to raise ImportError even if installed in .venv
        real_import = builtins.__import__

        def _block_whisperx(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "whisperx" or name.startswith("whisperx."):
                raise ImportError("No module named 'whisperx'")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "whisperx", raising=False)
        monkeypatch.setattr(builtins, "__import__", _block_whisperx)

        cfg = Config(hardware_profile=_make_rtx_4050_profile())
        result = Transcriber(cfg).transcribe(dummy_wav)

        assert fw_calls["init"] == {
            "model_size": "large-v2",
            "device": "cuda",
            "compute_type": "int8_float16",
        }
        assert fw_calls["transcribe"] == {
            "audio_path": str(dummy_wav),
            "language": None,
            "word_timestamps": True,
        }
        assert result.language == "en"
        assert result.full_text == "Clutch escape!"
        assert len(result.segments) == 1
        assert result.segments[0].words == [
            WordTimestamp(text="Clutch", start=42.0, end=42.6, confidence=0.96),
            WordTimestamp(text="escape!", start=42.65, end=43.5, confidence=0.92),
        ]
        assert counters["empty_cache"] == 1
        assert events == ["fw_init", "fw_transcribe", "empty_cache"]

    def test_explicit_faster_whisper_engine_bypasses_whisperx(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        dummy_wav: Path,
    ) -> None:
        user_cfg = tmp_path / "config.yaml"
        user_cfg.write_text("transcription:\n  engine: faster-whisper\n", encoding="utf-8")

        _install_fake_torch(monkeypatch, cuda_available=False)
        fw_calls = _install_fake_faster_whisper(monkeypatch)
        monkeypatch.delitem(sys.modules, "whisperx", raising=False)

        cfg = Config(user_config_path=user_cfg, hardware_profile=_make_cpu_profile())
        result = Transcriber(cfg).transcribe(dummy_wav)

        assert fw_calls["init"]["model_size"] == "base"
        assert fw_calls["transcribe"]["word_timestamps"] is True
        assert result.full_text == "Clutch escape!"

    def test_missing_audio_file_raises_file_not_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", tmp_path / "missing.yaml")
        cfg = Config(hardware_profile=_make_cpu_profile())
        transcriber = Transcriber(cfg)

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcriber.transcribe(tmp_path / "nonexistent.wav")


class TestTranscriptCaching:
    def test_save_and_load_transcript_round_trip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", tmp_path / "missing.yaml")
        cfg = Config(hardware_profile=_make_cpu_profile())
        transcriber = Transcriber(cfg)

        original = TranscriptionResult(
            segments=[
                TranscriptSegment(
                    text="That physics engine is insane!",
                    start=12.5,
                    end=14.8,
                    words=[
                        WordTimestamp(text="That", start=12.5, end=12.8, confidence=0.95),
                        WordTimestamp(text="physics", start=12.85, end=13.4, confidence=0.98),
                        WordTimestamp(text="engine", start=13.45, end=13.9, confidence=0.97),
                        WordTimestamp(text="is", start=13.95, end=14.1, confidence=0.99),
                        WordTimestamp(text="insane!", start=14.15, end=14.8, confidence=0.91),
                    ],
                )
            ],
            full_text="That physics engine is insane!",
            language="en",
        )

        cache_file = tmp_path / "nested" / "cache" / "stream_transcript.json"
        transcriber.save_transcript(original, cache_file)

        assert cache_file.exists()
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        assert raw["language"] == "en"
        assert len(raw["segments"][0]["words"]) == 5

        loaded = transcriber.load_transcript(cache_file)
        assert loaded == original

    def test_load_transcript_returns_none_for_missing_or_corrupt_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", tmp_path / "missing.yaml")
        cfg = Config(hardware_profile=_make_cpu_profile())
        transcriber = Transcriber(cfg)

        assert transcriber.load_transcript(tmp_path / "does_not_exist.json") is None

        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{invalid json", encoding="utf-8")
        assert transcriber.load_transcript(corrupt) is None

    def test_faster_whisper_maps_auto_language_to_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        user_cfg = tmp_path / "fw_auto.yaml"
        user_cfg.write_text(
            "transcription:\n  engine: faster-whisper\n  language: auto\n",
            encoding="utf-8",
        )
        cfg = Config(user_config_path=user_cfg, hardware_profile=_make_cpu_profile())
        audio_file = tmp_path / "sample.wav"
        audio_file.write_bytes(b"RIFF")

        captured_lang: list[object] = []

        class _FakeFWModel:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def transcribe(self, path: str, **kwargs: object) -> tuple[list[object], object]:
                captured_lang.append(kwargs.get("language"))
                return [], type("_Info", (), {"language": "en"})()

        monkeypatch.setitem(
            sys.modules,
            "faster_whisper",
            type("_Mod", (), {"WhisperModel": _FakeFWModel})(),
        )
        result = Transcriber(cfg).transcribe(audio_file)
        assert captured_lang == [None]
        assert result.language == "en"
