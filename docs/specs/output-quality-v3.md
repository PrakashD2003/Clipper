# Output Quality v3 — Build Spec

Branch: `feat/output-quality-v3`. Source of decisions: design interview 2026-11-19 (Q1–Q28).
Every worker reads **§0 Rules**, **§1 Contracts**, and **only its own batch section**. Workers must not
edit files outside their batch's file list. Contracts in §1 are fixed: code against them even if the
other side is not merged yet (use `getattr(obj, "x", default)` / keyword arguments with defaults).

## 0. Rules (all batches)

- Guardrails from AGENTS.md still apply (IntEnum states, SerializableModel defaults + `.get` in `from_dict`,
  MoviePy v2 API, Windows-safe ffmpeg `cwd` ASS burn, close clips before deleting temps, `-c:v copy`
  in the narration mix, single final lossy encode, never hardcode one GPU).
- New config keys are read with `config.get("dotted.key", <default from §2>)`; only Batch F edits
  `config.default.yaml` (both copies, byte-identical).
- New public parameters are keyword arguments **with defaults**, so existing callers keep working.
- One regression test per spec ID, named `test_<ID>_<what>` (e.g. `test_A3_stacked_layout_bands`), in
  an owned test file. If a test fake's signature breaks, widen the fake to the real signature; no shims.
- Before testing (PowerShell): `$env:PYTHONPATH = "$PWD\src"`; verify
  `D:\GTA6-Clipper\.venv\Scripts\python.exe -c "import streamclipper; print(streamclipper.__file__)"`
  prints a path inside the worktree. Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -q`.
- Commit message `feat(<area>): <IDs>` ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## 1. Cross-batch contracts (fixed)

**C1 Layout geometry** — new module `assembly/layout.py` (Batch A):
```python
@dataclass(frozen=True)
class Box: x: int; y: int; w: int; h: int
@dataclass(frozen=True)
class LayoutGeometry:
    kind: str                  # "stacked" | "blur_fill" | "reaction"
    width: int; height: int    # 1080 x 1920
    hook_band: Box             # where hook title + context badge go
    facecam_box: Box | None    # None for blur_fill
    game_box: Box              # uncropped game area (16:9 full, or 4:3 when game_fit="balanced")
    caption_band: Box          # captions, narration cards, callout, chat ticker
def compute_layout(kind: str, game_fit: str = "full", width: int = 1080, height: int = 1920) -> LayoutGeometry
```
Geometry (1080x1920):

| kind / game_fit | hook_band | facecam_box | game_box | caption_band |
| :--- | :--- | :--- | :--- | :--- |
| stacked / full | y 0–180 | y 180–880 (1080x700) | y 880–1488 (1080x608) | y 1488–1920 |
| stacked / balanced | y 0–180 | y 180–680 (1080x500) | y 680–1490 (1080x810, 4:3 centre crop) | y 1490–1920 |
| reaction / any | y 0–180 (over facecam top) | y 0–960 (1080x960) | y 960–1568 (1080x608) | y 1568–1920 |
| blur_fill / full | y 380–640 | None | y 656–1264 (1080x608) | y 1264–1920 |
| blur_fill / balanced | y 300–540 | None | y 555–1365 (1080x810) | y 1365–1920 |

**C2 Composer** (Batch A): `VideoComposer.compose(..., layout_kind: str | None = None)` and
`compose_story_beats(..., layout_kind: str | None = None)`. `None` → `"stacked"` if a facecam region is
given else `"blur_fill"`. Game fit from `video.game_fit`. Every composed clip (incl. the outputs of
`compose_instant_replay`, `compose_freeze_breakdown`, `apply_spike_zoom`) carries attribute
`_layout: LayoutGeometry`. The 1.20x spike zoom applies **inside `game_box` only**.

**C3 Captions** (Batch A): `AssCaptionBurner.generate_ass(..., layout: LayoutGeometry | None = None,
callout: str = "", callout_window: tuple[float, float] | None = None)`. With a layout, captions,
narration cards and the callout render inside `caption_band` (upper part), chat ticker in the lower part
of `caption_band`; nothing renders over `game_box`. Without a layout, behaviour is unchanged.

**C4 Hooks** (Batch A): `HookTitleOverlay.add_hook(video_clip, hook_title, context_badge="",
layout: LayoutGeometry | None = None)` — hook and badge centred in `layout.hook_band`.
`create_replay_badge(..., layout=None)` → badge at top of `game_box`.

**C5 Facecam** (Batch B): `FacecamDetector.detect(video_path: Path | list[Path], num_samples=None, *,
streamer: str | None = None) -> FacecamRegion | None` (same return type).

**C6 Narrator / audio** (Batch C):
- `TransformativeNarrator.prepare_callout_track(callout: str, temp_dir: Path | None = None) -> Path | None`
- `mix_narration_with_ducking(..., callout_wav: Path | None = None, callout_start: float | None = None,
  duck_full: bool = False)` — callout voiced over the moment (stream ducked under it only);
  `duck_full=True` keeps the stream ducked for the whole clip (music-risk clips).
- `TransformativeNarrator.normalize_audio(input_video: Path, output_video: Path) -> Path` — `-c:v copy`,
  loudnorm to `transformation.target_lufs`; used when no narration mix ran.
- The mix itself ends with loudnorm to `target_lufs` and narration ~`narration_gain_db` above the
  ducked bed, with `crossfade_seconds` ramps at every duck transition.

**C7 Scoring / metadata** (Batch D):
- `Highlight.callout: str = ""` and `ClipMetadata.callout: str = ""` (models.py, with `.get` defaults).
- `moment_type` gains `"reaction"` (VALID_MOMENT_TYPES).
- `OllamaScorer.generate_metadata(..., mode=None, hook_style: str | None = None,
  avoid_phrases: list[str] | None = None)` fills `callout`.
- `OllamaScorer.generate_countdown_lines(highlights: list[Highlight], streamer: str, game: str)
  -> list[str]` — one short ranking line per item, ordered #3 → #1.
- `analysis/content_gate.py`: `assess_segment(video_path: Path, num_samples: int = 12) -> GateResult`
  with `GateResult(is_gameplay: bool, letterbox: bool, static: bool, reason: str)`.

**C8 Variety / publishing** (Batch E): `output/variety.py` `VarietyState(path: Path)` with
`next_format(available: list[str]) -> str`, `next_hook_style() -> str`,
`recent_outros(n: int = 20) -> list[str]`, `clips_today(streamer: str) -> int`,
`record(streamer: str, fmt: str, hook_style: str, outro: str) -> None` (JSON at
`<general.cache_dir>/variety_state.json`, atomic writes). `output/post_writer.py`
`write_post_file(video_path: Path, metadata: ClipMetadata, stream_info: StreamInfo) -> Path`.

## 2. New / changed config keys (defaults; Batch F writes them to YAML)

```yaml
video.game_fit: full                      # full | balanced
facecam.min_detection_ratio: 0.6          # was 0.30
facecam.max_center_jitter: 0.05           # max std of face centre / frame size
facecam.overrides: {}                     # {streamer: [x, y, w, h] normalized 0-1}
facecam.profiles_path: ""                 # default <general.cache_dir>/facecam_profiles.json
analysis.min_clip_seconds: 15
analysis.max_clip_seconds: 35
analysis.skip_intro_minutes: 5
analysis.skip_high_music: true
content_gate.enabled: true
content_gate.letterbox_ratio: 0.10        # black rows top+bottom as share of height
content_gate.static_threshold: 2.0        # mean abs frame diff (0-255) below = static
output.max_final_seconds: 59
output.max_clips_per_streamer_per_day: 3
output.queued_dir: queued
output.write_post_file: true
output.min_commentary_ratio: 0.35
transformation.tts_engines: [chatterbox, kokoro, edge-tts]   # fallback chain
transformation.tts_external_python: ".venv-tts/Scripts/python.exe"
transformation.tts_reference_wav: ""      # "" = engine default voice
transformation.tts_voice: am_puck         # energetic Kokoro voice (verify it exists; else am_fenrir)
transformation.tts_speed: 1.1
transformation.edge_tts_voice: en-US-AndrewMultilingualNeural
transformation.edge_tts_rate: "+10%"
transformation.target_lufs: -14
transformation.narration_gain_db: 3
transformation.crossfade_seconds: 0.25
transformation.callout_enabled: true
transformation.persona: "hyped, witty gaming commentator who reacts like a friend watching live"
transformation.enable_top3: true
transformation.top3_item_seconds: 14
transformation.format_rotation: [narrative, story, breakdown]
transformation.hook_styles: [question, hot_take, countdown, nobody_noticed]
discovery.blocked_streamers: []
```

## 3. Batches

### Batch A — Layout & on-screen text
Files: `assembly/layout.py` (new), `assembly/composer.py`, `assembly/captions.py`, `assembly/hooks.py`,
`tests/test_composer.py`, `tests/test_hooks.py`, `tests/test_layout_v3.py` (new).
- A1 Implement C1 exactly (table values; balanced = 4:3 centre crop of the game).
- A2 `stacked`: facecam region cropped (aspect-correct `_crop_to_aspect`) into `facecam_box`; the full
  16:9 frame scaled to `game_box` width with no cropping (`full`) or 4:3 centre crop (`balanced`);
  background black. Split line no longer fixed at y=672.
- A3 `blur_fill`: background = game frame scaled to cover 1080x1920, Gaussian-blurred and darkened
  (~0.5 brightness); sharp game in `game_box`. Replaces the old full-frame 9:16 centre crop
  (`_compose_gameplay_only`); `face_center_x` becomes unused (keep param).
- A4 `reaction`: facecam large in `facecam_box`, game in `game_box`.
- A5 Spike zoom inside `game_box` only, for all kinds and for story beats (C2).
- A6 Propagate `_layout` through replay / freeze / zoom outputs.
- A7 Captions per C3; callout rendered as a short Hormozi card during `callout_window`.
- A8 Hooks per C4; replay badge per C4.

### Batch B — Facecam detection
Files: `assembly/facecam.py`, `assets/` (new: YuNet ONNX), `tests/test_facecam.py`.
- B1 Detector: OpenCV `cv2.FaceDetectorYN` with `face_detection_yunet_2023mar.onnx` (opencv_zoo,
  MIT) stored at `src/streamclipper/assets/`; download it once from the official opencv_zoo GitHub
  raw URL during the build and commit it; also ensure `package-data` would include it only if
  pyproject already supports package data (do not edit pyproject otherwise — note it in the reply).
  Fall back to the existing Haar path if YuNet is unavailable.
- B2 Whole-frame search (drop the corner-only restriction). Sample across **all** given segment files
  (C5). Accept a candidate only if present in ≥ `facecam.min_detection_ratio` of samples and the face
  centre std ≤ `facecam.max_center_jitter` × frame size. Transient alerts/GIFs fail these tests.
- B3 Expand the stable face box to the surrounding webcam rectangle (edge/contour search around the
  face, bounded to ≤ 0.45 of frame width/height); fall back to the existing padded box.
- B4 Profiles: JSON store (path `facecam.profiles_path` or `<general.cache_dir>/facecam_profiles.json`)
  keyed by streamer, normalized coords. Order: `facecam.overrides[streamer]` → stored profile
  (re-validated: face present in ≥ 40% of samples inside it) → fresh detection (saved on success).

### Batch C — Voice engines & audio mix
Files: `assembly/narrator.py`, `assembly/tts_engines.py` (new), `tts_worker/__init__.py` +
`tts_worker/chatterbox_worker.py` (new), `tests/test_narrator.py`, `tests/test_tts_engines.py` (new).
- C1 `tts_engines.py`: `TTSEngine` protocol `name`, `available() -> bool`,
  `synthesize(text, out_wav) -> Path | None`; engines Kokoro (`tts_voice`, `tts_speed`, espeak check),
  EdgeTTS (`edge_tts_voice`, `edge_tts_rate`), ElevenLabs (existing), and `ExternalProcessTTS`
  ("chatterbox"): runs `transformation.tts_external_python` (relative to repo root if relative) with
  `-m streamclipper.tts_worker.chatterbox_worker --text ... --out ... [--ref WAV]`, timeout 120 s,
  success = WAV exists and non-empty; any failure → `None`.
- C2 `chatterbox_worker.py`: standalone CLI importing only `chatterbox` + stdlib + torchaudio; loads
  `ChatterboxTurboTTS` (cuda if available else cpu), `generate(text, audio_prompt_path=ref or None)`,
  saves WAV at `model.sr`. Must not import streamclipper modules that pull the main env's deps.
- C3 Narrator routes through the `transformation.tts_engines` chain (first engine returning audio
  wins); legacy `tts_engine` single value still honoured when set to a non-"auto" value.
- C4 C6 audio contract: callout track + `callout_start`; `duck_full`; per-narration-track loudnorm,
  narration +`narration_gain_db` over the ducked bed, `crossfade_seconds` duck ramps, final
  `loudnorm=I=<target_lufs>:TP=-1.5:LRA=11`; keep `-c:v copy` and `:eval=frame`.
- C5 `normalize_audio()` per C6.

### Batch D — Scoring, prompts, content gate, models
Files: `scoring/scorer.py`, `scoring/ollama_scorer.py`, `prompts/*.txt`, `models.py`,
`analysis/content_gate.py` (new), `tests/test_scorer.py`, `tests/test_content_gate.py` (new).
- D1 Models per C7 (`callout` on Highlight + ClipMetadata, `.get` defaults).
- D2 Skip transcript chunks/highlights starting before `analysis.skip_intro_minutes` × 60.
- D3 `highlight_scorer.txt`: reject pre-game intro, countdown screens, cutscenes and loading screens
  unless the streamer's on-cam reaction is the content (then `moment_type: "reaction"`); add
  `reaction` to allowed moment types.
- D4 `narrative_writer.txt` rewrite: persona from `transformation.persona`; every line must reference a
  concrete detail from the provided transcript excerpt or chat samples (new placeholders
  `{transcript_excerpt}`, `{chat_samples}`, `{avoid_phrases}`, `{hook_style}`); sentences ≤ 12 words;
  outro = a reaction/opinion, never a summary or "subscribe"; add `callout` (4–6 words, said during
  the build-up). Ollama path: reject + retry (max 2) when output contains a banned generic phrase
  (list in code: "the chat went wild", "chat went completely wild", "you won't believe",
  "watch what happens", "subscribe for more", "over a million people", "insane moment",
  "most viral") or any phrase in `avoid_phrases`.
- D5 `hook_generator.txt` uses `{hook_style}`; `generate_metadata` per C7.
- D6 `generate_countdown_lines` per C7.
- D7 `content_gate.assess_segment` per C7: letterbox = mean of the top and bottom
  `letterbox_ratio` bands near-black (< 16) in ≥ 60% of samples; static = mean abs diff between
  consecutive samples < `static_threshold`; `is_gameplay = not (letterbox or static)`.

### Batch E — Pipeline integration, budget, variety, compilation, publishing (hotspot)
Files: `pipeline.py`, `output/exporter.py`, `output/organizer.py`, `output/variety.py` (new),
`output/post_writer.py` (new), `assembly/compilation.py` (new), `tests/test_pipeline.py`,
`tests/test_pipeline_breakdown.py`, `tests/test_assembly.py`, `tests/test_breakdown_mode.py`,
`tests/test_variety.py` (new), `tests/test_compilation.py` (new).
Keep `_assemble_highlight` backward compatible (MCP server calls it): new params keyword-only with defaults.
- E1 Facecam call: `detect(segment_paths, streamer=metadata.streamer)` (C5).
- E2 Content gate after download: non-gameplay → render as `layout_kind="reaction"` only if a facecam is
  confirmed and `moment_type` in {reaction, funny, chat_chaos, hype}; otherwise skip the highlight
  with a warning.
- E3 Layout: `layout_kind="reaction"` when `moment_type == "reaction"`; pass `composed._layout` into
  the exporter → `generate_ass(layout=...)` and `add_hook(layout=...)`.
- E4 Structure: hook intro → moment → callout (if `transformation.callout_enabled` and metadata has
  one, voiced at ~40% of the moment, shown as on-screen card) → instant replay + breakdown →
  reaction outro. Freeze-frame only when the rotated format says `breakdown` and
  `transformation.breakdown_style == "freeze_frame"`.
- E5 59 s budget: before rendering, budget = `output.max_final_seconds`; shrink the replay window
  first, then trim the moment around `peak_time`; after export, probe and hard-trim (`-t`) if still
  over. Compute commentary coverage (narration + callout + replay seconds / total) into the clip JSON;
  if < `output.min_commentary_ratio` and mode != raw, trim the moment around the peak to raise it.
- E6 Music: `music_risk == "HIGH_MUSIC_DETECTED"` and skip disabled → `duck_full=True` (C6).
- E7 When no narration mix ran, call `narrator.normalize_audio` (C6) before captions.
- E8 Variety (C8): rotate format via `next_format`, hook style via `next_hook_style`, pass
  `recent_outros` as `avoid_phrases`; `record()` after each clip; per-streamer daily cap
  (`output.max_clips_per_streamer_per_day`): extra clips go to `<output_dir>/<queued_dir>/...`
  (organizer gains `get_queued_path(...)`).
- E9 Top 3 (`transformation.enable_top3`, once per stream, needs ≥ 3 highlights):
  `assembly/compilation.py` builds a single Short from the top 3 by score, each trimmed to
  `top3_item_seconds` around its peak, countdown lines from `generate_countdown_lines` voiced
  before each item, ≤ 57 s total; saved as its own clip with metadata `transformation_mode="top3"`.
- E10 `discovery.blocked_streamers`: `process_stream` refuses (logs + returns []) for a blocked
  streamer.
- E11 Post file (`output.write_post_file`): `write_post_file` writes `<clip_stem>_post.txt` with title,
  description (credit "Original stream by <streamer>" + source URL), hashtags, and the reminders
  "Tick YouTube's 'Altered or synthetic content' disclosure (AI voice)" and
  "Post at most 2–3 Shorts per day".

### Batch F — Config, MCP server, discovery
Files: `config.default.yaml` (repo root) + `src/streamclipper/config.default.yaml`, `config.py`,
`mcp_server/server.py`, `discovery/youtube.py`, `tests/test_config.py`, `tests/test_mcp_server.py`,
`tests/test_discovery.py`.
- F1 Add every §2 key to both YAML copies (identical), with comments; keep hardware keys "auto".
- F2 `config.py`: validate `video.game_fit` ∈ {full, balanced}; accept new mode `top3` where modes are
  validated; nothing else.
- F3 MCP: `submit_highlights` and `render_clip` accept an optional `callout` (string) and
  `moment_type: "reaction"`; `render_clip` gains `layout: str = ""` ("" = auto, else stacked |
  blur_fill | reaction) passed through to the pipeline render path as `layout_kind` keyword
  (use `inspect`-free keyword passing; the pipeline side adds the kwarg in Batch E).
- F4 MCP render path writes the post file via `output.post_writer.write_post_file` when
  `output.write_post_file` (import lazily; skip silently if the module is missing pre-merge).
- F5 `discovery/youtube.py`: drop results whose channel/uploader is in `discovery.blocked_streamers`
  (case-insensitive).
