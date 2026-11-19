"""Vertical (9:16) layout geometry shared by composer, captions and hook overlays."""

from __future__ import annotations

from dataclasses import dataclass

_BASE_W = 1080
_BASE_H = 1920


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class LayoutGeometry:
    kind: str  # "stacked" | "blur_fill" | "reaction"
    width: int
    height: int
    hook_band: Box
    facecam_box: Box | None
    game_box: Box
    caption_band: Box


# (hook y0,y1) / (facecam y0,y1 | None) / (game y0,y1) / (caption y0,y1) on a 1080x1920 canvas.
_TABLE: dict[
    tuple[str, str],
    tuple[tuple[int, int], tuple[int, int] | None, tuple[int, int], tuple[int, int]],
] = {
    ("stacked", "full"): ((0, 180), (180, 880), (880, 1488), (1488, 1920)),
    ("stacked", "balanced"): ((0, 180), (180, 680), (680, 1490), (1490, 1920)),
    ("reaction", "full"): ((0, 180), (0, 960), (960, 1568), (1568, 1920)),
    ("reaction", "balanced"): ((0, 180), (0, 960), (960, 1568), (1568, 1920)),
    ("blur_fill", "full"): ((380, 640), None, (656, 1264), (1264, 1920)),
    ("blur_fill", "balanced"): ((300, 540), None, (555, 1365), (1365, 1920)),
}


def compute_layout(
    kind: str, game_fit: str = "full", width: int = _BASE_W, height: int = _BASE_H
) -> LayoutGeometry:
    """Return the band/box geometry for a layout kind (see C1 in the output-quality-v3 spec)."""
    kind = str(kind or "stacked").strip().lower()
    if kind not in {"stacked", "blur_fill", "reaction"}:
        raise ValueError(f"Unknown layout kind: {kind!r}")
    fit = "balanced" if str(game_fit or "full").strip().lower() == "balanced" else "full"
    hook, face, game, cap = _TABLE[(kind, fit)]
    sy = height / _BASE_H

    def y_of(v: int) -> int:
        return int(round(v * sy))

    def band(y0: int, y1: int) -> Box:
        return Box(0, y_of(y0), width, y_of(y1) - y_of(y0))

    return LayoutGeometry(
        kind=kind,
        width=width,
        height=height,
        hook_band=band(*hook),
        facecam_box=band(*face) if face is not None else None,
        game_box=band(*game),
        caption_band=band(*cap),
    )
