"""Content gate: reject segments that are letterboxed or static (not real gameplay)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_NEAR_BLACK = 16.0
_LETTERBOX_SAMPLE_SHARE = 0.60
DEFAULT_LETTERBOX_RATIO = 0.10
DEFAULT_STATIC_THRESHOLD = 2.0


@dataclass
class GateResult:
    is_gameplay: bool
    letterbox: bool
    static: bool
    reason: str


def _sample_frames(video_path: Path, num_samples: int) -> list:
    """Return up to *num_samples* grayscale frames evenly spread over the video."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return frames
        n = max(1, min(num_samples, total))
        for i in range(n):
            idx = int((i + 0.5) * total / n)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    return frames


def assess_segment(
    video_path: Path,
    num_samples: int = 12,
    *,
    letterbox_ratio: float = DEFAULT_LETTERBOX_RATIO,
    static_threshold: float = DEFAULT_STATIC_THRESHOLD,
) -> GateResult:
    """Assess whether *video_path* is real gameplay.

    letterbox: mean of the top and bottom ``letterbox_ratio`` bands is near-black (< 16) in at
    least 60% of samples. static: mean absolute difference between consecutive samples is below
    ``static_threshold``. ``is_gameplay = not (letterbox or static)``. Unreadable video fails open.
    """
    import numpy as np

    try:
        frames = _sample_frames(Path(video_path), int(num_samples))
    except Exception as exc:  # pragma: no cover - defensive (cv2 missing / codec errors)
        logger.warning("Content gate could not read %s: %s", video_path, exc)
        return GateResult(True, False, False, "unreadable; gate skipped")
    if not frames:
        return GateResult(True, False, False, "no frames; gate skipped")

    letterbox_hits = 0
    for gray in frames:
        h = gray.shape[0]
        band = max(1, int(round(h * float(letterbox_ratio))))
        top = float(np.mean(gray[:band]))
        bottom = float(np.mean(gray[h - band:]))
        if top < _NEAR_BLACK and bottom < _NEAR_BLACK:
            letterbox_hits += 1
    letterbox = letterbox_hits / len(frames) >= _LETTERBOX_SAMPLE_SHARE

    static = False
    if len(frames) >= 2:
        diffs = [
            float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
            for i in range(1, len(frames))
        ]
        static = float(np.mean(diffs)) < float(static_threshold)

    reasons = []
    if letterbox:
        reasons.append("letterboxed (black bars top and bottom)")
    if static:
        reasons.append("static (little frame-to-frame change)")
    return GateResult(
        is_gameplay=not (letterbox or static),
        letterbox=letterbox,
        static=static,
        reason="; ".join(reasons) if reasons else "gameplay",
    )
