"""Constant-memory streamed loudness spike detection using librosa and soundfile."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from ..config import Config
from ..models import SignalType, TimeRange

logger = logging.getLogger(__name__)


class LoudnessDetector:
    """Detect high-energy loudness spikes in stream WAV audio via constant-memory streaming."""

    BLOCK_LENGTH_FRAMES = 120

    def __init__(self, config: Config):
        self.threshold_std = float(config.get("analysis.loudness.threshold_std", 2.5))
        self.window_seconds = float(config.get("analysis.loudness.window_seconds", 2))
        self.merge_gap = float(config.get("analysis.loudness.merge_gap_seconds", 5))

        if self.window_seconds <= 0:
            raise ValueError(
                f"analysis.loudness.window_seconds must be > 0, got {self.window_seconds}"
            )
        if self.merge_gap < 0:
            raise ValueError(
                f"analysis.loudness.merge_gap_seconds must be >= 0, got {self.merge_gap}"
            )

    def detect(self, audio_path: Path | str) -> list[TimeRange]:
        """Detect loudness spikes in audio using constant-memory streaming."""
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        rms_db, hop_duration = self._compute_rms_db(path)
        spikes = self._find_spikes(
            rms_db, hop_duration, audio_duration=float(sf.info(str(path)).duration)
        )
        return self._merge_nearby(spikes)

    def _compute_rms_db(self, audio_path: Path) -> tuple[np.ndarray, float]:
        """Stream audio in fixed-frame blocks to compute RMS energy in dB."""
        info = sf.info(str(audio_path))
        sr = info.samplerate
        hop_length = max(1, int(round(sr * self.window_seconds)))
        frame_length = hop_length * 2

        if info.frames == 0:
            return np.array([0.0], dtype=np.float32), float(self.window_seconds)

        rms_chunks: list[np.ndarray] = []
        stream = librosa.stream(
            str(audio_path),
            block_length=self.BLOCK_LENGTH_FRAMES,
            frame_length=frame_length,
            hop_length=hop_length,
            sr=sr,
            mono=True,
            fill_value=0.0,
        )
        for y_block in stream:
            if len(y_block) < frame_length:
                continue
            rms_block = librosa.feature.rms(
                y=y_block,
                frame_length=frame_length,
                hop_length=hop_length,
                center=False,
            )[0]
            rms_chunks.append(rms_block)

        rms = np.concatenate(rms_chunks) if rms_chunks else np.array([1e-10], dtype=np.float32)

        # Trim phantom frames created solely by fill_value=0.0 padding past EOF
        # so short recordings or tail blocks do not skew RMS mean/std with -80 dB silence.
        n_real_frames = max(1, math.ceil(info.frames / hop_length))
        rms = rms[:n_real_frames]

        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        return rms_db, float(self.window_seconds)

    def _find_spikes(
        self,
        rms_db: np.ndarray,
        hop_duration: float,
        audio_duration: float | None = None,
    ) -> list[TimeRange]:
        """Find peaks above mean + threshold_std * std_dev with 0-10 scores and metadata['peak_time']."""
        if rms_db.size < 2:
            logger.info(
                "Loudness spike detection skipped: audio too short (%d RMS frame(s), need >= 2).",
                rms_db.size,
            )
            return []

        mean_db = float(np.mean(rms_db))
        std_db = float(np.std(rms_db))
        if std_db < 1e-6:
            logger.info(
                "Loudness spike detection skipped: audio is flat (RMS dB std %.2e < 1e-6).",
                std_db,
            )
            return []

        threshold = mean_db + self.threshold_std * std_db
        sentinel = float(np.min(rms_db)) - 1.0
        padded = np.concatenate(([sentinel], rms_db, [sentinel]))
        raw_peaks, _ = find_peaks(padded, height=threshold)
        peaks = raw_peaks - 1
        if len(peaks) == 0:
            return []

        max_above = float(np.max(rms_db[peaks]) - threshold)
        total_duration = float(len(rms_db) * hop_duration)

        spikes: list[TimeRange] = []
        for peak_idx in peaks:
            idx = int(peak_idx)
            peak_db = float(rms_db[idx])
            above = peak_db - threshold
            if max_above > 1e-6:
                score = round(min(10.0, max(5.0, 5.0 + (above / max_above) * 5.0)), 2)
            else:
                score = 7.5

            left_idx = idx
            while left_idx > 0 and rms_db[left_idx - 1] >= threshold:
                left_idx -= 1

            right_idx = idx
            while right_idx + 1 < len(rms_db) and rms_db[right_idx + 1] >= threshold:
                right_idx += 1

            start = max(0.0, (left_idx - 1) * hop_duration)
            end = min(total_duration, (right_idx + 2) * hop_duration)
            # RMS frames are 2 * hop long (center=False): peak_time is the frame centre.
            peak_time = float(idx * hop_duration + hop_duration)
            peak_time = min(
                peak_time,
                audio_duration if audio_duration is not None else total_duration,
            )

            spikes.append(
                TimeRange(
                    start=round(start, 3),
                    end=round(end, 3),
                    score=score,
                    signal_type=SignalType.LOUDNESS,
                    metadata={
                        "peak_time": round(peak_time, 3),
                        "peak_db": round(peak_db, 3),
                    },
                )
            )

        return spikes

    def _merge_nearby(self, spikes: list[TimeRange]) -> list[TimeRange]:
        """Merge spikes within merge_gap seconds of each other, retaining max score and its peak_time."""
        if not spikes:
            return []

        ordered = sorted(spikes, key=lambda s: (s.start, s.end))
        first = ordered[0]
        merged: list[TimeRange] = [
            TimeRange(
                start=first.start,
                end=first.end,
                score=first.score,
                signal_type=SignalType.LOUDNESS,
                metadata=dict(first.metadata),
            )
        ]

        for spike in ordered[1:]:
            current = merged[-1]
            if spike.start - current.end <= self.merge_gap:
                new_end = max(current.end, spike.end)
                spike_is_louder = spike.score > current.score or (
                    spike.score == current.score
                    and float(spike.metadata.get("peak_db", float("-inf")))
                    > float(current.metadata.get("peak_db", float("-inf")))
                )
                best_score = spike.score if spike_is_louder else current.score
                best_metadata = dict(
                    spike.metadata if spike_is_louder else current.metadata
                )
                merged[-1] = TimeRange(
                    start=current.start,
                    end=new_end,
                    score=best_score,
                    signal_type=SignalType.LOUDNESS,
                    metadata=best_metadata,
                )
            else:
                merged.append(
                    TimeRange(
                        start=spike.start,
                        end=spike.end,
                        score=spike.score,
                        signal_type=SignalType.LOUDNESS,
                        metadata=dict(spike.metadata),
                    )
                )

        return merged
