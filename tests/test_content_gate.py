"""Tests for analysis/content_gate.py (spec D7)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from streamclipper.analysis.content_gate import assess_segment


def _write_video(path: Path, frame_fn, n: int = 40, size=(320, 180)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    for i in range(n):
        writer.write(frame_fn(i, size))
    writer.release()
    return path


def _noisy(i, size):
    rng = np.random.default_rng(i)
    return rng.integers(40, 255, (size[1], size[0], 3), dtype=np.uint8)


def test_D7_gameplay_passes(tmp_path):
    p = _write_video(tmp_path / "game.avi", _noisy)
    r = assess_segment(p, num_samples=8)
    assert r.is_gameplay and not r.letterbox and not r.static


def test_D7_letterbox_detected(tmp_path):
    def f(i, size):
        fr = _noisy(i, size)
        band = int(size[1] * 0.15)
        fr[:band] = 0
        fr[-band:] = 0
        return fr

    r = assess_segment(_write_video(tmp_path / "lb.avi", f), num_samples=8)
    assert r.letterbox and not r.is_gameplay


def test_D7_static_detected(tmp_path):
    def f(i, size):
        return np.full((size[1], size[0], 3), 120, dtype=np.uint8)

    r = assess_segment(_write_video(tmp_path / "st.avi", f), num_samples=8)
    assert r.static and not r.is_gameplay


def test_D7_unreadable_fails_open(tmp_path):
    r = assess_segment(tmp_path / "missing.mp4")
    assert r.is_gameplay
