"""Pre-flight acoustic copyright music risk detection on non-speech intervals."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from ..config import Config
from ..models import Highlight, WordTimestamp

logger = logging.getLogger(__name__)


class CopyrightRiskDetector:
    """Pre-flight acoustic copyright music risk detector analyzing non-speech intervals.

    Uses librosa Harmonic-Percussive Source Separation (HPSS) and spectral flatness
    on non-speech pauses (>0.4s) to identify sustained commercial background music
    (in-game car radio, Spotify streams) while ignoring speech, sound effects, and silence.
    Runs entirely on CPU with constant-memory soundfile block seeking (0 MB GPU VRAM).
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config
        if config is not None:
            self.enabled: bool = bool(config.get("analysis.copyright_check", True))
            self.risk_threshold: float = float(config.get("analysis.music_risk_threshold", 0.65))
            self.min_gap_seconds: float = float(config.get("analysis.music_min_gap_seconds", 0.4))
            self.flatness_threshold: float = float(
                config.get("analysis.music_flatness_threshold", 0.15)
            )
            self.min_rms: float = float(config.get("analysis.music_min_rms", 0.005))
        else:
            self.enabled = True
            self.risk_threshold = 0.65
            self.min_gap_seconds = 0.4
            self.flatness_threshold = 0.15
            self.min_rms = 0.005

    def find_non_speech_intervals(
        self,
        clip_start: float,
        clip_end: float,
        word_timestamps: list[WordTimestamp | dict[str, Any]] | None = None,
    ) -> list[tuple[float, float]]:
        """Identify intervals >= min_gap_seconds where no speech occurs within [clip_start, clip_end]."""
        start = float(clip_start)
        end = float(clip_end)
        if end <= start or (end - start) < self.min_gap_seconds:
            return []

        if not word_timestamps:
            return [(round(start, 3), round(end, 3))]

        # Extract and clamp speech intervals overlapping [start, end]
        speech_intervals: list[tuple[float, float]] = []
        for w in word_timestamps:
            if isinstance(w, WordTimestamp):
                w_start = float(w.start)
                w_end = float(w.end)
            elif isinstance(w, dict):
                w_start = float(w.get("start", 0.0))
                w_end = float(w.get("end", w_start))
            else:
                continue

            if w_end <= start or w_start >= end or w_end <= w_start:
                continue
            speech_intervals.append((max(start, w_start), min(end, w_end)))

        if not speech_intervals:
            return [(round(start, 3), round(end, 3))]

        speech_intervals.sort(key=lambda item: (item[0], item[1]))

        # Merge overlapping/adjacent speech intervals
        merged_speech: list[list[float]] = []
        for s_start, s_end in speech_intervals:
            if not merged_speech or s_start > merged_speech[-1][1]:
                merged_speech.append([s_start, s_end])
            else:
                merged_speech[-1][1] = max(merged_speech[-1][1], s_end)

        # Invert merged speech intervals to find non-speech gaps
        gaps: list[tuple[float, float]] = []
        curr = start
        for s_start, s_end in merged_speech:
            if (s_start - curr) >= self.min_gap_seconds:
                gaps.append((round(curr, 3), round(s_start, 3)))
            curr = max(curr, s_end)

        if (end - curr) >= self.min_gap_seconds:
            gaps.append((round(curr, 3), round(end, 3)))

        return gaps

    def detect(
        self,
        audio_path: Path | str,
        clip_start: float,
        clip_end: float,
        word_timestamps: list[WordTimestamp | dict[str, Any]] | None = None,
    ) -> tuple[str, float]:
        """Scan non-speech intervals for sustained harmonic music.

        Returns:
            (music_risk, music_confidence) where music_risk is one of:
            - "HIGH_MUSIC_DETECTED" (music_confidence >= 0.65 or peak_gap_conf >= 0.65 for >= 1.0s)
            - "LOW" (0.35 <= music_confidence < 0.65 or peak_gap_conf >= 0.50)
            - "CLEAN" (music_confidence < 0.35)
        """
        if not self.enabled:
            return ("CLEAN", 0.0)

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        start = float(clip_start)
        end = float(clip_end)
        if end <= start or (end - start) < self.min_gap_seconds:
            return ("CLEAN", 0.0)

        gaps = self.find_non_speech_intervals(start, end, word_timestamps)
        if not gaps:
            return ("CLEAN", 0.0)

        gap_confidences: list[tuple[float, float]] = []  # (confidence, duration)

        with sf.SoundFile(str(path), "r") as sf_file:
            sr = sf_file.samplerate
            total_frames = sf_file.frames

            for gap_start, gap_end in gaps:
                gap_dur = gap_end - gap_start
                if gap_dur < self.min_gap_seconds:
                    continue

                # Clamp seek bounds to prevent soundfile.LibsndfileError
                start_frame = max(0, min(total_frames, int(round(gap_start * sr))))
                end_frame = max(start_frame, min(total_frames, int(round(gap_end * sr))))
                frames_to_read = end_frame - start_frame
                if frames_to_read <= 0 or start_frame >= total_frames:
                    continue

                sf_file.seek(start_frame)
                chunk = sf_file.read(frames=frames_to_read, dtype="float32")
                if chunk.ndim > 1:
                    chunk = np.mean(chunk, axis=1)

                if len(chunk) < 512:
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms < self.min_rms:
                    gap_confidences.append((0.0, gap_dur))
                    continue

                try:
                    y_harm, y_perc = librosa.effects.hpss(chunk)
                    h_energy = float(np.sum(y_harm ** 2))
                    p_energy = float(np.sum(y_perc ** 2))
                    tot_energy = h_energy + p_energy
                    if tot_energy < 1e-9:
                        gap_confidences.append((0.0, gap_dur))
                        continue

                    h_ratio = h_energy / tot_energy
                    flatness = float(np.mean(librosa.feature.spectral_flatness(y=chunk)))
                    tonality = max(0.0, min(1.0, 1.0 - (flatness / self.flatness_threshold)))
                    conf = float(np.clip(h_ratio * tonality, 0.0, 1.0))
                    gap_confidences.append((conf, gap_dur))
                except Exception as exc:
                    logger.debug("HPSS analysis failed on gap [%.2f, %.2f]: %s", gap_start, gap_end, exc)
                    continue

        if not gap_confidences:
            return ("CLEAN", 0.0)

        total_gap_dur = sum(dur for _, dur in gap_confidences)
        if total_gap_dur <= 0.0:
            return ("CLEAN", 0.0)

        weighted_conf = sum(conf * dur for conf, dur in gap_confidences) / total_gap_dur
        peak_conf = max(conf for conf, _ in gap_confidences)
        peak_dur = max(dur for conf, dur in gap_confidences if conf == peak_conf)
        final_conf = round(float(np.clip(weighted_conf, 0.0, 1.0)), 3)

        if final_conf >= self.risk_threshold or (peak_conf >= self.risk_threshold and peak_dur >= 1.0):
            risk = "HIGH_MUSIC_DETECTED"
        elif final_conf >= 0.35 or peak_conf >= 0.50:
            risk = "LOW"
        else:
            risk = "CLEAN"

        return (risk, final_conf)

    def detect_highlight(self, audio_path: Path | str, highlight: Highlight) -> tuple[str, float]:
        """Convenience method to evaluate a Highlight dataclass directly."""
        return self.detect(audio_path, highlight.start, highlight.end, highlight.word_timestamps)
