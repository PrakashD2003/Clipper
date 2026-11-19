"""Tests for assembly/compilation.py (Top-3 countdown, v3 Batch E9)."""
from __future__ import annotations

from streamclipper.assembly.compilation import (
    MAX_COMPILATION_SECONDS,
    build_countdown_ass,
    build_countdown_filtergraph,
    plan_item_windows,
)


def test_E9_windows_capped_to_item_seconds_and_total() -> None:
    wins = plan_item_windows([20.0, 25.0, 30.0], [60.0, 60.0, 60.0], item_seconds=14.0)
    assert len(wins) == 3
    assert all(abs((e - s) - 14.0) < 1e-6 for s, e in wins)
    assert sum(e - s for s, e in wins) <= MAX_COMPILATION_SECONDS


def test_E9_windows_shrink_when_over_57s_and_stay_inside_source() -> None:
    wins = plan_item_windows([5.0, 58.0, None], [10.0, 60.0, 8.0], item_seconds=30.0)
    assert sum(e - s for s, e in wins) <= MAX_COMPILATION_SECONDS + 1e-6
    for (s, e), dur in zip(wins, [10.0, 60.0, 8.0]):
        assert 0.0 <= s < e <= dur + 1e-6


def test_E9_peak_is_inside_window() -> None:
    (s, e), = plan_item_windows([30.0], [60.0], item_seconds=14.0)
    assert s < 30.0 < e


def test_E9_filtergraph_ducks_bed_and_delays_each_countdown_line() -> None:
    graph = build_countdown_filtergraph([(0.2, 2.0), (14.2, 16.0), (28.2, 30.0)], total=42.0)
    assert "eval=frame" in graph
    assert "adelay=200|200" in graph and "adelay=14200|14200" in graph
    assert "amix=inputs=4" in graph
    assert "[3:a]" in graph


def test_E9_countdown_ass_has_one_card_per_line_in_order() -> None:
    ass = build_countdown_ass(["#3 Wild", "#2 Wilder", "#1 Wildest"], [(0, 3), (14, 17), (28, 31)])
    events = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert len(events) == 3
    assert events[0].endswith("#3 Wild") and events[2].endswith("#1 Wildest")
