"""Quality fixes: grounded intro fallback, narration gate, blur_fill geometry/hook/captions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from streamclipper.assembly.captions import AssCaptionBurner
from streamclipper.assembly.layout import compute_layout
from streamclipper.config import Config, HardwareProfile
from streamclipper.models import ClipMetadata, Highlight
from streamclipper.pipeline import Pipeline
from streamclipper.scoring.ollama_scorer import BANNED_PHRASES, grounded_fallback_intro


def _config() -> Config:
    hw = HardwareProfile(
        tier="cpu_only", gpu_name="CPU", vram_gb=0.0, compute_capability="N/A",
        has_tensor_cores=False, has_nvenc=False, device="cpu", model_size="base",
        compute_type="int8", batch_size=4, llm_model="qwen3:4b", video_codec="libx264",
    )
    return Config(hardware_profile=hw)


def _hl(**kw: Any) -> Highlight:
    base: dict[str, Any] = dict(
        start=600.0, end=640.0, score=8.0, title="First time seeing Vice City",
        reason="He reacts to the skyline. Then drives off.", best_quote="this looks unreal",
    )
    base.update(kw)
    return Highlight(**base)


def _meta() -> ClipMetadata:
    return ClipMetadata(
        hook_title="h", caption="c", hashtags=[], streamer="S", game="G",
        source_url="u", clip_start=0, clip_end=1, highlight_score=8.0,
    )


def test_grounded_fallback_intro_uses_highlight_text_and_avoids_banned() -> None:
    text = grounded_fallback_intro(_hl(), "CarryisLive", "GTA 6")
    assert "Vice City" in text and "this looks unreal" in text
    banned = grounded_fallback_intro(
        _hl(title="You won" + chr(39) + "t believe this", best_quote="", reason="Insane moment here"),
        "CarryisLive", "GTA 6",
    )
    assert not any(p in banned.lower() for p in BANNED_PHRASES)
    assert grounded_fallback_intro(_hl(title="", best_quote="", reason=""), "S", "G").strip()


def test_ensure_intro_falls_back_when_writer_fails() -> None:
    pipeline = Pipeline(_config())

    class _Boom:
        def generate_narrative(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("ollama down")

    meta = _meta()
    pipeline._ensure_intro_narration(meta, _hl(), _Boom(), "S", "G")
    assert "Vice City" in meta.narration_intro

    class _Ok:
        def generate_narrative(self, *a: Any, **k: Any) -> tuple[str, str, str]:
            return "Written intro.", "Written outro.", "badge"

    meta2 = _meta()
    pipeline._ensure_intro_narration(meta2, _hl(), _Ok(), "S", "G")
    assert meta2.narration_intro == "Written intro." and meta2.narration_outro == "Written outro."


def test_narration_gate_intro_and_breakdown() -> None:
    p = Pipeline(_config())
    base = {
        "narration_expected": True, "mix_ran": True, "intro_expected": True,
        "intro_seconds": 2.0,
    }
    assert p._narration_gate_reason(base) == ""
    assert "intro" in p._narration_gate_reason({**base, "intro_seconds": 0.0})
    assert p._narration_gate_reason({**base, "intro_expected": False, "intro_seconds": 0.0}) == ""
    assert "did not run" in p._narration_gate_reason({**base, "mix_ran": False})
    bd = {**base, "breakdown_expected": True, "breakdown_seconds": 3.0}
    assert p._narration_gate_reason(bd, mode="breakdown") == ""
    assert "breakdown" in p._narration_gate_reason({**bd, "breakdown_seconds": 0.0}, mode="breakdown")
    assert "breakdown" in p._narration_gate_reason({**base}, mode="breakdown")
    assert p._narration_gate_reason({**base}, mode="narrative") == ""


def test_breakdown_eligibility_by_moment_type() -> None:
    p = Pipeline(_config())
    assert p._breakdown_ineligible_reason(
        _hl(moment_type="hype", breakdown_commentary="x", peak_time=620.0)
    )
    assert not p._breakdown_ineligible_reason(
        _hl(moment_type="clutch", breakdown_commentary="x", peak_time=620.0)
    )


def test_blur_fill_geometry_and_scale() -> None:
    g = compute_layout("blur_fill", "full")
    assert (g.hook_band.y, g.hook_band.y2) == (180, 520)
    assert (g.game_box.y, g.game_box.y2, g.game_box.w, g.game_box.h) == (560, 1259, 1080, 699)
    assert (g.caption_band.y, g.caption_band.y2) == (1275, 1800)
    for b in (g.hook_band, g.game_box, g.caption_band):
        assert 0 <= b.y and b.y2 <= 1920 and b.x == 0 and b.x2 == 1080
    assert g.hook_band.y2 <= g.game_box.y and g.game_box.y2 <= g.caption_band.y
    big = compute_layout("blur_fill", "full", blur_game_scale=5.0)  # clamped to 1.30
    assert big.game_box.h == 790 and big.caption_band.y2 == 1800
    assert compute_layout("blur_fill", "full", blur_game_scale=1.0).game_box.h == 608


def test_blur_fill_captions_centered_in_band(tmp_path: Path) -> None:
    burner = AssCaptionBurner(_config())
    layout = compute_layout("blur_fill", "full")
    out = burner.generate_ass(
        words=[], clip_start_offset=0.0, ass_path=tmp_path / "c.ass",
        narration_intro="Hello there", intro_dur=2.0, clip_duration=20.0, layout=layout,
    )
    text = out.read_text(encoding="utf-8")
    cy = int(layout.caption_band.y + 0.5 * layout.caption_band.h)
    assert "{\\an5\\pos(540,%d)}" % cy in text
    stacked = burner.generate_ass(
        words=[], clip_start_offset=0.0, ass_path=tmp_path / "s.ass",
        narration_intro="Hello there", intro_dur=2.0, clip_duration=20.0,
        layout=compute_layout("stacked", "full"),
    ).read_text(encoding="utf-8")
    assert "\\an5" not in stacked


def test_hook_persistent_and_streamer_tag() -> None:
    from moviepy import ColorClip

    from streamclipper.assembly.hooks import HookTitleOverlay, render_streamer_tag

    arr = render_streamer_tag("@CarryisLive")
    assert arr.ndim == 3 and arr.shape[2] == 4 and arr[..., 3].max() < 255
    layout = compute_layout("blur_fill", "full")
    clip = ColorClip((1080, 1920), color=(10, 10, 10), duration=12.0)
    out = HookTitleOverlay(_config()).add_hook(
        clip, "First time in Vice City", layout=layout, streamer="CarryisLive"
    )
    assert len(out.clips) == 3  # video + hook + tag
    assert all(abs(c.duration - 12.0) < 1e-6 for c in out.clips[1:])
    ys = [c.pos(0)[1] for c in out.clips[1:]]
    assert all(layout.hook_band.y <= y < layout.hook_band.y2 for y in ys)

    cfg = _config()
    cfg._data.setdefault("hooks", {})["persistent_title"] = False
    cfg._data["hooks"]["streamer_tag"] = False
    short = HookTitleOverlay(cfg).add_hook(
        clip, "First time in Vice City", layout=layout, streamer="X"
    )
    assert len(short.clips) == 2
    assert abs(short.clips[1].duration - 4.0) < 1e-6


def test_blur_fill_game_layer_is_centre_cropped() -> None:
    from moviepy import ColorClip

    from streamclipper.assembly.composer import VideoComposer

    composer = VideoComposer(_config())
    layout = compute_layout("blur_fill", "full", blur_game_scale=composer.blur_fill_game_scale)
    src = ColorClip((1920, 1080), color=(50, 60, 70), duration=1.0)
    layer = composer._game_layer(src, layout.game_box, centre_crop=True)
    assert tuple(layer.size) == (1080, 699) and tuple(layer.pos(0)) == (0, 560)
    comp = composer._compose_layout(src, None, layout)
    assert comp.size == (1080, 1920) and comp._layout.kind == "blur_fill"
