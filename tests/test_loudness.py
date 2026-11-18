"""Unit tests for Package 2.4 — Streamed Loudness Spike Detector."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import librosa
import numpy as np
import pytest
import soundfile as sf
import yaml

from streamclipper.analysis.loudness import LoudnessDetector
from streamclipper.config import Config, HardwareProfile
from streamclipper.models import SignalType, TimeRange


def _cpu_profile() -> HardwareProfile:
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


def _make_config(
    tmp_path: Path,
    *,
    threshold_std: float = 2.5,
    window_seconds: float = 2.0,
    merge_gap_seconds: float = 5.0,
) -> Config:
    cfg_path = tmp_path / "test_loudness_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "analysis": {
                    "loudness": {
                        "threshold_std": threshold_std,
                        "window_seconds": window_seconds,
                        "merge_gap_seconds": merge_gap_seconds,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return Config(user_config_path=cfg_path, hardware_profile=_cpu_profile())


def _write_synthetic_wav(
    path: Path,
    *,
    duration_seconds: float,
    bursts: list[tuple[float, float, float]],
    sr: int = 16000,
    noise_amp: float = 0.02,
    channels: int = 1,
    seed: int = 42,
) -> Path:
    """Create a synthetic WAV with quiet white noise and injected sine bursts.

    Each entry in ``bursts`` is ``(start_sec, end_sec, amplitude)``.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_seconds * sr))
    signal = (rng.standard_normal(n_samples) * noise_amp).astype(np.float32)

    t = np.arange(n_samples, dtype=np.float32) / float(sr)
    for start_sec, end_sec, amplitude in bursts:
        s_idx = max(0, int(round(start_sec * sr)))
        e_idx = min(n_samples, int(round(end_sec * sr)))
        if e_idx > s_idx:
            sine = amplitude * np.sin(2.0 * np.pi * 440.0 * t[s_idx:e_idx])
            signal[s_idx:e_idx] += sine.astype(np.float32)

    signal = np.clip(signal, -1.0, 1.0)
    if channels == 2:
        data = np.column_stack([signal, signal])
    else:
        data = signal

    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


class TestLoudnessDetectorSyntheticAudio:
    def test_detect_finds_two_bursts_on_odd_duration_wav(self, tmp_path: Path):
        """61.3s non-multiple duration exercises tail-block guard and detects 2 bursts."""
        wav_path = _write_synthetic_wav(
            tmp_path / "odd_61_3s.wav",
            duration_seconds=61.3,
            bursts=[
                (16.2, 17.8, 0.85),
                (44.2, 45.8, 0.85),
            ],
        )
        detector = LoudnessDetector(_make_config(tmp_path))

        spikes = detector.detect(wav_path)

        assert len(spikes) == 2
        for spike in spikes:
            assert isinstance(spike, TimeRange)
            assert spike.signal_type == SignalType.LOUDNESS
            assert 0.0 < spike.score <= 10.0
            assert "peak_time" in spike.metadata
            assert isinstance(spike.metadata["peak_time"], float)
            assert spike.start <= spike.metadata["peak_time"] <= spike.end

        # Verify burst 1 (~16.2-17.8s) and burst 2 (~44.2-45.8s) locations
        assert spikes[0].start <= 17.0 <= spikes[0].end
        assert 12.0 <= spikes[0].metadata["peak_time"] <= 20.0

        assert spikes[1].start <= 45.0 <= spikes[1].end
        assert 40.0 <= spikes[1].metadata["peak_time"] <= 48.0

    def test_detect_scores_scale_monotonically_with_burst_amplitude(self, tmp_path: Path):
        """Louder bursts receive strictly higher normalized 0-10 scores than quieter bursts."""
        wav_path = _write_synthetic_wav(
            tmp_path / "odd_121_3s.wav",
            duration_seconds=121.3,
            bursts=[
                (24.0, 26.5, 0.90),  # loudest scream
                (88.0, 90.5, 0.55),  # moderate yell
            ],
        )
        detector = LoudnessDetector(_make_config(tmp_path))

        spikes = detector.detect(wav_path)

        assert len(spikes) == 2
        assert spikes[0].score == 10.0
        assert 0.0 < spikes[1].score < spikes[0].score
        assert spikes[0].start <= 25.0 <= spikes[0].end
        assert spikes[1].start <= 89.0 <= spikes[1].end

    @pytest.mark.parametrize(
        "first_amp, second_amp, expected_peak_near",
        [
            (0.35, 0.95, 28.0),  # second burst louder
            (0.95, 0.35, 20.0),  # first burst louder
        ],
    )
    def test_detect_merges_nearby_bursts_and_keeps_loudest_peak_time(
        self,
        tmp_path: Path,
        first_amp: float,
        second_amp: float,
        expected_peak_near: float,
    ):
        """Nearby bursts within merge_gap_seconds merge into 1 TimeRange with loudest peak_time."""
        wav_path = _write_synthetic_wav(
            tmp_path / f"merge_{first_amp}_{second_amp}.wav",
            duration_seconds=90.7,
            bursts=[
                (19.0, 21.0, first_amp),
                (27.0, 29.0, second_amp),
            ],
        )
        # window=1.0s gives crisp separation between the two peaks at ~20s and ~28s,
        # while merge_gap=8.0s merges the two nearby spikes into a single cluster.
        detector = LoudnessDetector(
            _make_config(
                tmp_path,
                threshold_std=2.0,
                window_seconds=1.0,
                merge_gap_seconds=8.0,
            )
        )

        rms_db, hop_duration = detector._compute_rms_db(wav_path)
        raw_spikes = detector._find_spikes(rms_db, hop_duration)
        assert len(raw_spikes) == 2

        merged = detector._merge_nearby(raw_spikes)
        assert len(merged) == 1

        cluster = merged[0]
        assert cluster.signal_type == SignalType.LOUDNESS
        assert cluster.score == max(s.score for s in raw_spikes)
        assert cluster.start <= 20.0
        assert cluster.end >= 28.0
        assert abs(cluster.metadata["peak_time"] - expected_peak_near) <= 2.0

    def test_multi_block_streaming_uses_librosa_stream_not_load(self, tmp_path: Path):
        """Audio exceeding block_length=120 frames crosses block boundary safely via librosa.stream."""
        # window_seconds=0.5 -> 120 frames = 60.0s per block.
        # 75.3s audio -> Block 1 (0..60s) + Block 2 (60..75.3s odd tail block).
        wav_path = _write_synthetic_wav(
            tmp_path / "multiblock_75_3s.wav",
            duration_seconds=75.3,
            bursts=[
                (20.0, 21.5, 0.90),  # inside block 1
                (67.0, 68.5, 0.85),  # inside block 2 (tail block)
            ],
            sr=8000,
        )
        detector = LoudnessDetector(
            _make_config(tmp_path, threshold_std=2.5, window_seconds=0.5, merge_gap_seconds=5.0)
        )

        stream_calls: list[dict] = []
        original_stream = librosa.stream

        def spy_stream(*args, **kwargs):
            stream_calls.append(kwargs)
            return original_stream(*args, **kwargs)

        with (
            patch("streamclipper.analysis.loudness.librosa.stream", side_effect=spy_stream),
            patch(
                "librosa.load",
                side_effect=AssertionError("librosa.load must never be called"),
            ),
        ):
            spikes = detector.detect(wav_path)

        assert len(stream_calls) == 1
        assert stream_calls[0]["fill_value"] == 0.0
        assert stream_calls[0]["block_length"] == 120
        assert stream_calls[0]["mono"] is True

        assert len(spikes) == 2
        assert spikes[0].start <= 20.75 <= spikes[0].end
        assert spikes[1].start <= 67.75 <= spikes[1].end
        assert all(s.signal_type == SignalType.LOUDNESS for s in spikes)


class TestLoudnessMergeAndEdgeCases:
    def test_merge_nearby_unit_preserves_max_score_and_peak_time(self, tmp_path: Path):
        detector = LoudnessDetector(_make_config(tmp_path, merge_gap_seconds=5.0))

        # Unsorted input with a 3-spike chain (gaps 4.0s and 5.0s <= 5.0s) and 1 distant spike (gap 5.1s > 5.0s)
        s1 = TimeRange(start=10.0, end=14.0, score=6.2, signal_type=SignalType.LOUDNESS, metadata={"peak_time": 12.0})
        s2 = TimeRange(start=18.0, end=22.0, score=9.8, signal_type=SignalType.LOUDNESS, metadata={"peak_time": 20.0})
        s3 = TimeRange(start=27.0, end=31.0, score=7.4, signal_type=SignalType.LOUDNESS, metadata={"peak_time": 29.0})
        s4 = TimeRange(start=36.1, end=40.0, score=8.1, signal_type=SignalType.LOUDNESS, metadata={"peak_time": 38.0})

        merged = detector._merge_nearby([s3, s1, s4, s2])

        assert len(merged) == 2

        # First cluster merges s1 + s2 + s3; retains s2's max score and peak_time
        assert merged[0].start == 10.0
        assert merged[0].end == 31.0
        assert merged[0].score == 9.8
        assert merged[0].metadata["peak_time"] == 20.0
        assert merged[0].signal_type == SignalType.LOUDNESS

        # s4 is 5.1s away (> merge_gap=5.0s), stays separate
        assert merged[1].start == 36.1
        assert merged[1].end == 40.0
        assert merged[1].score == 8.1
        assert merged[1].metadata["peak_time"] == 38.0

        # Verify inputs were not mutated in place
        assert s1.end == 14.0
        assert s1.score == 6.2
        assert s1.metadata["peak_time"] == 12.0

    def test_merge_nearby_empty_returns_empty(self, tmp_path: Path):
        detector = LoudnessDetector(_make_config(tmp_path))
        assert detector._merge_nearby([]) == []

    def test_silent_and_constant_audio_return_no_spikes(self, tmp_path: Path):
        detector = LoudnessDetector(_make_config(tmp_path))

        silent_path = tmp_path / "silent.wav"
        sf.write(str(silent_path), np.zeros(16000 * 15, dtype=np.float32), 16000, subtype="PCM_16")
        assert detector.detect(silent_path) == []

        t = np.arange(16000 * 15, dtype=np.float32) / 16000.0
        tone_path = tmp_path / "constant_tone.wav"
        sf.write(
            str(tone_path),
            (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32),
            16000,
            subtype="PCM_16",
        )
        assert detector.detect(tone_path) == []

    def test_ultra_short_audio_does_not_crash(self, tmp_path: Path):
        """WAV shorter than frame_length (0.5s < 4.0s) safely returns [] without ParameterError."""
        short_path = tmp_path / "short_0_5s.wav"
        sf.write(str(short_path), np.full(8000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")

        detector = LoudnessDetector(_make_config(tmp_path))
        assert detector.detect(short_path) == []

    def test_stereo_wav_downmixes_and_detects_spikes(self, tmp_path: Path):
        wav_path = _write_synthetic_wav(
            tmp_path / "stereo_61_3s.wav",
            duration_seconds=61.3,
            bursts=[(24.0, 26.5, 0.90)],
            channels=2,
        )
        detector = LoudnessDetector(_make_config(tmp_path))
        spikes = detector.detect(wav_path)

        assert len(spikes) == 1
        assert spikes[0].signal_type == SignalType.LOUDNESS
        assert spikes[0].start <= 25.0 <= spikes[0].end
        assert "peak_time" in spikes[0].metadata

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        detector = LoudnessDetector(_make_config(tmp_path))
        with pytest.raises(FileNotFoundError):
            detector.detect(tmp_path / "does_not_exist.wav")

    def test_invalid_config_values_raise_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="window_seconds"):
            LoudnessDetector(_make_config(tmp_path, window_seconds=0))

        with pytest.raises(ValueError, match="merge_gap_seconds"):
            LoudnessDetector(_make_config(tmp_path, merge_gap_seconds=-1.0))

    def test_detected_spikes_serialize_round_trip(self, tmp_path: Path):
        wav_path = _write_synthetic_wav(
            tmp_path / "roundtrip.wav",
            duration_seconds=61.3,
            bursts=[(30.0, 32.5, 0.85)],
        )
        detector = LoudnessDetector(_make_config(tmp_path))
        spikes = detector.detect(wav_path)
        assert len(spikes) == 1

        restored = TimeRange.from_dict(spikes[0].to_dict())
        assert restored == spikes[0]
        assert restored.signal_type == SignalType.LOUDNESS
        assert restored.metadata["peak_time"] == spikes[0].metadata["peak_time"]

    def test_detects_boundary_burst_at_frame_zero_and_scales_secondary_spike(
        self, tmp_path: Path
    ):
        """Endpoint padding catches frame-0 bursts, and secondary spikes keep usable >=5.0 scores."""
        wav_path = _write_synthetic_wav(
            tmp_path / "boundary_and_scale.wav",
            duration_seconds=90.0,
            bursts=[
                (0.0, 1.8, 0.95),   # frame 0 boundary burst
                (50.0, 52.0, 0.45), # moderate secondary burst
            ],
        )
        detector = LoudnessDetector(_make_config(tmp_path, threshold_std=2.0))
        spikes = detector.detect(wav_path)

        assert len(spikes) == 2
        assert spikes[0].start == 0.0
        assert spikes[0].metadata["peak_time"] == 0.0
        assert spikes[0].score == 10.0
        assert 5.0 <= spikes[1].score < 10.0
