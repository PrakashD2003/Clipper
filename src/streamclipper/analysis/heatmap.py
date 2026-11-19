"""Peak detection over YouTube's "Most replayed" heatmap segments.

Heatmap bins are coarse (~1/100 of the video, ~2.4 min for a 4 h VOD), so peaks are
regional boosts for fusion and candidate surfacing, not exact clip boundaries.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..config import Config
from ..models import SignalType, TimeRange


def detect_heatmap_peaks(
    segments: list[dict[str, Any]] | None,
    *,
    threshold_std: float = 1.0,
    min_value: float = 0.5,
    merge_gap: float = 1.0,
) -> list[TimeRange]:
    """Return merged peak ranges where ``value >= max(mean + threshold_std*std, min_value)``.

    ``score`` is ``value * 10`` (0..10, like loudness/chat spike scores); metadata carries
    ``source="heatmap"``, ``peak_time`` (centre of the hottest segment) and ``heat`` (0..1).
    Segments closer than ``merge_gap`` seconds are merged (adjacent bins touch).
    """
    clean: list[tuple[float, float, float]] = []
    for seg in segments or []:
        try:
            start, end, value = float(seg["start"]), float(seg["end"]), float(seg["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and math.isfinite(value) and end > start:
            clean.append((start, end, min(1.0, max(0.0, value))))
    if not clean:
        return []
    clean.sort(key=lambda t: t[0])

    values = np.array([c[2] for c in clean], dtype=float)
    threshold = max(float(values.mean() + threshold_std * values.std()), float(min_value))

    peaks = [c for c in clean if c[2] >= threshold]
    merged: list[list[Any]] = []  # [start, end, best_value, best_centre]
    for start, end, value in peaks:
        centre = (start + end) / 2.0
        if merged and start - merged[-1][1] <= merge_gap:
            cur = merged[-1]
            cur[1] = max(cur[1], end)
            if value > cur[2]:
                cur[2], cur[3] = value, centre
        else:
            merged.append([start, end, value, centre])

    return [
        TimeRange(
            start=round(m[0], 3),
            end=round(m[1], 3),
            score=round(m[2] * 10.0, 2),
            signal_type=SignalType.HEATMAP,
            metadata={"source": "heatmap", "peak_time": round(m[3], 3), "heat": round(m[2], 4)},
        )
        for m in merged
    ]


class HeatmapAnalyzer:
    """Config-driven wrapper around :func:`detect_heatmap_peaks`."""

    def __init__(self, config: Config):
        self.enabled = bool(config.get("analysis.heatmap.enabled", True))
        self.threshold_std = float(config.get("analysis.heatmap.threshold_std", 1.0))
        self.min_value = float(config.get("analysis.heatmap.min_value", 0.5))

    def detect(self, segments: list[dict[str, Any]] | None) -> list[TimeRange]:
        if not self.enabled:
            return []
        return detect_heatmap_peaks(
            segments, threshold_std=self.threshold_std, min_value=self.min_value
        )
