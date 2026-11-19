"""Unit tests for Pre-Flight Acoustic Copyright Music Risk Detector (Milestone 1)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import yaml

from streamclipper.analysis.copyright_risk import CopyrightRiskDetector
from streamclipper.config import Config, HardwareProfile
from streamclipper.models import Highlight, WordTimestamp

CPU_PROFILE = HardwareProfile(
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


def _make_config(tmp_path: Path, analysis_overrides: dict[str, Any] | None = None) -> Config:
    """Create a test Config with optional analysis settings."""
    if not analysis_overrides:
        return Config(user_config_path=tmp_path / "missing.yaml", hardware_profile=CPU_PROFILE)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"analysis": analysis_overrides}), encoding="utf-8")
    return Config(user_config_path=cfg_path, hardware_profile=CPU_PROFILE)


def _generate_sine_chord(duration: float, sr: int = 16000) -> np.ndarray:
    """Generate harmonic tri-tone chord (A4 440Hz, C#5 554.37Hz, E5 659.25Hz)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    chord = (
        0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.3 * np.sin(2 * np.pi * 554.37 * t)
        + 0.3 * np.sin(2 * np.pi * 659.25 * t)
    )
    return chord.astype(np.float32)


def _generate_white_noise(duration: float, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Generate Gaussian white noise with high spectral flatness."""
    rng = np.random.default_rng(seed)
    n_samples = int(sr * duration)
    return rng.normal(0, 0.1, n_samples).astype(np.float32)


def _create_wav_file(path: Path, data: np.ndarray, sr: int = 16000) -> Path:
    """Write float32 numpy array to a WAV file."""
    sf.write(str(path), data, sr, format="WAV", subtype="PCM_16")
    return path


class TestNonSpeechIntervals:
    """Test suite for non-speech pause interval extraction."""

    def test_no_words_returns_entire_clip_as_gap(self) -> None:
        detector = CopyrightRiskDetector()
        gaps = detector.find_non_speech_intervals(0.0, 10.0, [])
        assert gaps == [(0.0, 10.0)]

        gaps_none = detector.find_non_speech_intervals(5.0, 15.0, None)
        assert gaps_none == [(5.0, 15.0)]

    def test_clip_too_short_returns_no_gaps(self) -> None:
        detector = CopyrightRiskDetector()
        # default min_gap_seconds is 0.4
        gaps = detector.find_non_speech_intervals(1.0, 1.3, [])
        assert gaps == []

        gaps_negative = detector.find_non_speech_intervals(5.0, 4.0, [])
        assert gaps_negative == []

    def test_gaps_between_word_timestamps(self) -> None:
        detector = CopyrightRiskDetector()
        words = [
            WordTimestamp(text="hello", start=1.0, end=2.0),
            WordTimestamp(text="world", start=4.0, end=5.0),
        ]
        gaps = detector.find_non_speech_intervals(0.0, 7.0, words)
        # Expected gaps: [0.0, 1.0] (1.0s), [2.0, 4.0] (2.0s), [5.0, 7.0] (2.0s)
        assert gaps == [(0.0, 1.0), (2.0, 4.0), (5.0, 7.0)]

    def test_sub_threshold_gaps_are_filtered(self) -> None:
        detector = CopyrightRiskDetector()
        words = [
            WordTimestamp(text="word1", start=1.0, end=2.0),
            # Gap between word1 and word2 is 0.2s (< 0.4s)
            WordTimestamp(text="word2", start=2.2, end=3.0),
            # Gap between word2 and word3 is 1.0s (>= 0.4s)
            WordTimestamp(text="word3", start=4.0, end=5.0),
        ]
        gaps = detector.find_non_speech_intervals(1.0, 5.0, words)
        assert gaps == [(3.0, 4.0)]

    def test_overlapping_and_dict_timestamps(self) -> None:
        detector = CopyrightRiskDetector()
        # Dictionary format with overlapping words and words out of order
        raw_words: list[dict[str, Any]] = [
            {"text": "b", "start": 3.0, "end": 4.5},
            {"text": "a", "start": 2.0, "end": 3.5},
            {"text": "outside", "start": 20.0, "end": 25.0},
        ]
        gaps = detector.find_non_speech_intervals(0.0, 10.0, raw_words)
        # Merged speech: [2.0, 4.5]
        # Gaps: [0.0, 2.0] (2.0s), [4.5, 10.0] (5.5s)
        assert gaps == [(0.0, 2.0), (4.5, 10.0)]


class TestCopyrightRiskDetection:
    """Test suite verifying acoustic copyright music risk scoring."""

    def test_detect_pure_music_flags_high_music_detected(self, tmp_path: Path) -> None:
        detector = CopyrightRiskDetector()
        chord = _generate_sine_chord(duration=4.0)
        wav_path = _create_wav_file(tmp_path / "music.wav", chord)

        risk, conf = detector.detect(wav_path, clip_start=0.0, clip_end=4.0)
        assert risk == "HIGH_MUSIC_DETECTED"
        assert conf >= 0.65

    def test_detect_white_noise_flags_clean(self, tmp_path: Path) -> None:
        detector = CopyrightRiskDetector()
        noise = _generate_white_noise(duration=4.0)
        wav_path = _create_wav_file(tmp_path / "noise.wav", noise)

        risk, conf = detector.detect(wav_path, clip_start=0.0, clip_end=4.0)
        assert risk == "CLEAN"
        assert conf < 0.35

    def test_detect_silence_flags_clean_with_zero_confidence(self, tmp_path: Path) -> None:
        detector = CopyrightRiskDetector()
        silence = np.zeros(16000 * 3, dtype=np.float32)
        wav_path = _create_wav_file(tmp_path / "silence.wav", silence)

        risk, conf = detector.detect(wav_path, clip_start=0.0, clip_end=3.0)
        assert risk == "CLEAN"
        assert conf == 0.0

    def test_detect_continuous_speech_no_gaps_returns_clean(self, tmp_path: Path) -> None:
        """When speech covers the clip without gaps >= 0.4s, return CLEAN without reading file."""
        detector = CopyrightRiskDetector()
        chord = _generate_sine_chord(duration=5.0)
        wav_path = _create_wav_file(tmp_path / "speech_masked.wav", chord)

        # Words cover [0.0, 5.0] with gaps <= 0.2s
        words = [
            WordTimestamp(text="one", start=0.0, end=1.5),
            WordTimestamp(text="two", start=1.6, end=3.0),
            WordTimestamp(text="three", start=3.1, end=4.2),
            WordTimestamp(text="four", start=4.3, end=5.0),
        ]
        risk, conf = detector.detect(wav_path, clip_start=0.0, clip_end=5.0, word_timestamps=words)
        assert risk == "CLEAN"
        assert conf == 0.0

    def test_detect_short_clip_returns_clean(self, tmp_path: Path) -> None:
        detector = CopyrightRiskDetector()
        chord = _generate_sine_chord(duration=1.0)
        wav_path = _create_wav_file(tmp_path / "short.wav", chord)

        risk, conf = detector.detect(wav_path, clip_start=0.0, clip_end=0.3)
        assert risk == "CLEAN"
        assert conf == 0.0

    def test_detect_disabled_via_config_returns_clean(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, {"copyright_check": False})
        detector = CopyrightRiskDetector(cfg)
        chord = _generate_sine_chord(duration=4.0)
        wav_path = _create_wav_file(tmp_path / "music.wav", chord)

        risk, conf = detector.detect(wav_path, clip_start=0.0, clip_end=4.0)
        assert risk == "CLEAN"
        assert conf == 0.0

    def test_detect_missing_file_raises_filenotfound(self) -> None:
        detector = CopyrightRiskDetector()
        with pytest.raises(FileNotFoundError):
            detector.detect(Path("non_existent_audio_file.wav"), clip_start=0.0, clip_end=5.0)

    def test_seek_boundary_clamp_avoids_libsndfile_error(self, tmp_path: Path) -> None:
        """Seeking beyond audio file duration is safely clamped to EOF without LibsndfileError."""
        detector = CopyrightRiskDetector()
        # Audio is only 1.0s long
        chord = _generate_sine_chord(duration=1.0)
        wav_path = _create_wav_file(tmp_path / "one_sec.wav", chord)

        # Clip interval requests [0.0, 10.0]
        risk, conf = detector.detect(wav_path, clip_start=0.0, clip_end=10.0)
        assert risk == "HIGH_MUSIC_DETECTED"
        assert conf >= 0.65

    def test_detect_intermediate_music_flags_low(self, tmp_path: Path) -> None:
        """Test intermediate tonality or mixed signal yields LOW risk."""
        detector = CopyrightRiskDetector()
        sr = 16000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        rng = np.random.default_rng(42)
        # Moderate tone + noise giving tonality ~ 0.40
        tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)
        noise = rng.normal(0, 0.03, len(t)).astype(np.float32)
        mixed = (tone + noise).astype(np.float32)
        wav_path = _create_wav_file(tmp_path / "intermediate.wav", mixed)

        risk, conf = detector.detect(wav_path, clip_start=0.0, clip_end=2.0)
        assert risk == "LOW"
        assert 0.35 <= conf < 0.65

    def test_detect_stereo_audio_downmixes_and_detects(self, tmp_path: Path) -> None:
        """Multichannel (stereo) audio downmixes properly to 1D array."""
        detector = CopyrightRiskDetector()
        sr = 16000
        mono = _generate_sine_chord(duration=3.0, sr=sr)
        stereo = np.column_stack([mono, mono])
        wav_path = tmp_path / "stereo.wav"
        sf.write(str(wav_path), stereo, sr, format="WAV", subtype="PCM_16")

        risk, conf = detector.detect(wav_path, clip_start=0.0, clip_end=3.0)
        assert risk == "HIGH_MUSIC_DETECTED"
        assert conf >= 0.65

    def test_detect_highlight_dataclass(self, tmp_path: Path) -> None:
        """detect_highlight evaluates start, end, and word_timestamps from Highlight."""
        detector = CopyrightRiskDetector()
        chord = _generate_sine_chord(duration=5.0)
        wav_path = _create_wav_file(tmp_path / "highlight_test.wav", chord)

        hl = Highlight(
            start=1.0,
            end=4.0,
            score=8.5,
            title="Epic Moment",
            reason="Great play",
            word_timestamps=[WordTimestamp(text="hi", start=1.0, end=1.5)],
        )

        risk, conf = detector.detect_highlight(wav_path, hl)
        assert risk == "HIGH_MUSIC_DETECTED"
        assert conf >= 0.65

    def test_zero_gpu_vram_usage(self, tmp_path: Path) -> None:
        """Verify detector runs strictly on CPU with 0 MB GPU VRAM allocation."""
        import torch

        detector = CopyrightRiskDetector()
        chord = _generate_sine_chord(duration=2.0)
        wav_path = _create_wav_file(tmp_path / "vram_test.wav", chord)

        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated()
            detector.detect(wav_path, clip_start=0.0, clip_end=2.0)
            vram_after = torch.cuda.memory_allocated()
            assert vram_after == vram_before
        else:
            # CPU-only environment
            detector.detect(wav_path, clip_start=0.0, clip_end=2.0)


class TestReviewedDefects:
    def test_G2_all_gaps_failing_returns_unknown(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        wav = _create_wav_file(tmp_path / "m.wav", _generate_sine_chord(3.0))
        detector = CopyrightRiskDetector()
        with patch(
            "streamclipper.analysis.copyright_risk.librosa.effects.hpss",
            side_effect=RuntimeError("x"),
        ):
            assert detector.detect(wav, 0.0, 3.0) == ("UNKNOWN", 0.0)

    def test_L23_threshold_config_keys(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"copyright": {"high_risk_threshold": 0.8, "low_risk_threshold": 0.2}},
        )
        det = CopyrightRiskDetector(cfg)
        assert det.risk_threshold == 0.8 and det.low_risk_threshold == 0.2

        class _Cfg:
            def get(self, key, default=None):
                return {"analysis.music_risk_threshold": 0.7}.get(key, default)

        fb = CopyrightRiskDetector(_Cfg())  # type: ignore[arg-type]
        assert fb.risk_threshold == 0.7 and fb.low_risk_threshold == 0.35
