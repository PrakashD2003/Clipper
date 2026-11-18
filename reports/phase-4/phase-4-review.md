# Phase 4 Lead Reviewer Audit Report — `phase-4-video-assembly`

## 1. Executive Verdict

| Field | Value |
| :--- | :--- |
| **Phase Audited** | Phase 4: Video Assembly & Output (`phase-4-video-assembly`) |
| **Specification Reference** | `docs/streamclipper-implementation-guide-v2.md` (v2.2 — Patched + Auto-Hardware + YPP Transformative Engine) & `AGENTS.md` |
| **Commits Reviewed** | `main` (`565893d`) `...` `phase-4-video-assembly` (`f02cff5`) — Packages `4.1`–`4.6` (`f55f633`/`52cbe11`, `ead1879`/`a1db797`, `a46b54d`, `9cfebdd`/`ee3224f`, `cf78d7d`/`cd7c62c`, `c5a1c81`), and Lead Review defect fixes (`f02cff5`), 21 files changed, `+6,839` / `-70` lines |
| **Package Reports Audited** | `reports/phase-4/package-4.1-report.md` through `reports/phase-4/package-4.6-report.md` |
| **Test Suite Verification** | `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` — **483 / 483 passed** (`61.97s`, Python 3.12.10, pytest 9.2.0, zero failures, zero skipped; 122 upstream MoviePy/NumPy 2.5 `DeprecationWarning`s) |
| **Audit Verdict** | ✅ **APPROVED (after Lead Review fixes in `f02cff5`) — Cleared for `--no-ff` merge into `main`** |
| **Review Timestamp** | 2026-11-18 20:56 IST |
| **Reviewer** | Lead Reviewer Audit (`4.R`) |

Phase 4 satisfies all specification requirements, acceptance criteria, and v2.2 architectural guardrails across Packages `4.1`–`4.6` plus four concrete robustness defects caught and fixed during Lead Review (`f02cff5`) with 5 new regression tests. All 483 unit, mocked integration, and real FFmpeg/MoviePy/NVENC/OpenCV end-to-end tests pass cleanly on Windows.

---

## 2. Verification Matrix (Lead Reviewer Checklist)

| # | Audit Requirement | Result | Evidence & Code Inspection |
| :-: | :--- | :---: | :--- |
| 1 | Package reports `4.1`–`4.6` audited & reconciled with git history | ✅ PASS | All 6 reports exist in `reports/phase-4/` and reconcile with `git log main..phase-4-video-assembly`: `4.1` (`f55f633`/`52cbe11`), `4.2` (`ead1879`/`a1db797`), `4.4` (`9cfebdd`/`ee3224f`), `4.5` (`cf78d7d`/`cd7c62c`), `4.3` (`a46b54d`), `4.6` (`c5a1c81`), plus Lead Review fix commit `f02cff5`. Earlier report test totals (`308`/`304`/`311`/`458`/`478`) reflect their branch snapshots; HEAD stands at **483 / 483**. |
| 2 | `composer.py` & `hooks.py`: MoviePy v2 imports/methods, aspect-preserving facecam crop, conditional multi-beat story stitching, `1.2x` spike zoom, and `_source_clip` / `_source_clips` tracking | ✅ PASS | Direct v2 imports (`from moviepy import CompositeVideoClip, VideoFileClip, concatenate_videoclips` in `composer.py:7`; `from moviepy import CompositeVideoClip, ImageClip` in `hooks.py:9`); zero deprecated v1 APIs (`moviepy.editor`, `.crop(`, `.resize(`, `.subclip(`, `.set_*(`) found across `assembly/` and `output/`. `VideoComposer._crop_to_aspect()` (`composer.py:155-190`) preserves target aspect ratio without stretching; `compose_story_beats()` (`composer.py:255-315`) clamps `(t_start, t_end)` to `[0.0, raw_clip.duration]` before `.subclipped()`, stitches beats via `concatenate_videoclips(..., method="compose")`, and falls back to `compose()` for 1 segment; `remap_timeline_for_beats()` (`composer.py:15-108`) shifts words/chat/peak onto `0.0`-based stitched timeline; `apply_spike_zoom()` (`composer.py:317-361`) applies `1.20x` center punch-in zoom preserving audio and `_source_clip` / `_source_clips`. `HookTitleOverlay.add_hook()` (`hooks.py:178-284`) uses Pillow RGBA arrays + MoviePy v2 `ImageClip` (no ImageMagick), `.with_duration()`, `.with_position()`, `CrossFadeIn`, `.with_audio()`, and propagates `_source_clip` / `_source_clips`. Hardened in `f02cff5`: leak-free close on partial failure, strict `[0, clip.w - 1]` / `[0, clip.h - 1]` clamping, and zero-duration short-circuit. |
| 3 | `captions.py`: dynamic `captions.highlight_color` → ASS BGR, `Hormozi` + `ChatTicker` styles, ASS delimiter escaping, Windows-safe `cwd=ass_path.parent` + `ass=<basename>`, `h264_nvenc` (`-cq`) with `libx264` fallback | ✅ PASS | `captions.py:108-125, 174-179` converts `#FFD700` → `&H00D7FF&` (with named color fallback and strict 6-hex regex). `generate_ass()` (`captions.py:195-316`) emits `Style: Hormozi` (honoring `style`, `color`, `stroke_color`, `stroke_width`, `font`, `font_size`, `margin_v`) and `Style: ChatTicker` (`BorderStyle=3`, semi-transparent `OutlineColour=&H99000000`, up to 5 messages). `_escape_ass_text()` (`captions.py:128-134`) + trailing-`\` guard (`captions.py:254`) neutralize `\r`/`\n`, `{`, `}`, and `\N`/`\n`/`\h`. `_sec_to_ass_time()` (`captions.py:96-105`) avoids `60.00` centisecond rollover. `burn_captions()` (`captions.py:347-396`) runs FFmpeg with `cwd=str(ass_file.resolve().parent)`, `-vf ass={_escape_ffmpeg_ass_filename(ass_file.name)}`, `-c:a copy`, `h264_nvenc` (`-preset p4 -cq <crf>`) with single-retry `libx264` (`-crf <crf>`) fallback. Verified by 67 tests in `tests/test_assembly.py` including live `libass` and GPU `h264_nvenc` burn-in. |
| 4 | `narrator.py`: pluggable TTS (`kokoro`, `edge-tts`, optional `elevenlabs` via `ELEVENLABS_API_KEY` + `voice_id`), custom `--voice-intro`/`--voice-outro`, dynamic 2-input / 3-input FFmpeg ducking (`:eval=frame`, 48kHz stereo) with short-clip overlap clamping | ✅ PASS | `TransformativeNarrator` (`narrator.py:24-569`) supports `kokoro`, `edge-tts` (using `self.edge_tts_voice`, never Kokoro voice ID), `elevenlabs` (`ELEVENLABS_API_KEY` + `elevenlabs_voice_id`, falling back to `auto` when unconfigured), `auto` (`kokoro` → `edge-tts`), and `none`; `resolve_narration_track()` prioritizes custom human mic files over TTS. `_build_ducking_filtergraph()` (`narrator.py:304-412`) dynamically emits 2-input (intro-only / outro-only) or 3-input (intro + outro) filtergraphs with `volume='...':eval=frame`, `aresample=48000,aformat=channel_layouts=stereo` on every stream, stereo `adelay={ms}|{ms}`, and `amix=inputs={n}:duration=first:normalize=0`. Fixed in `f02cff5`: when short-clip clamping scales `eff_intro` or `eff_outro` below raw track length, `atrim=0:{eff:.2f},asetpts=PTS-STARTPTS` is now applied to `[1:a]`/`[2:a]` so clamped narrator audio cannot bleed into the un-ducked climax or overlap outro audio. |
| 5 | `exporter.py`: closes all clip handles in `finally` before deleting temp files & avoids double-compression degradation | ✅ PASS | `Exporter.export()` (`exporter.py:143-277`) writes intermediate MP4 via MoviePy at `codec="libx264"`, `preset="ultrafast"`, `ffmpeg_params=["-crf", "18"]` inside an isolated per-export `work_dir`, immediately calls `_release_handles_once()` (`_close_clip_handles` walking `clip`, `_source_clip`, and `_source_clips` deduplicated by `id`) inside an inner `try/finally` before narration mixing (`-c:v copy`) and final single-pass H.264 encode (ASS burn-in or `_finalize_without_captions` with `h264_nvenc` → `libx264` fallback), and guarantees `_release_handles_once()` runs again before `shutil.rmtree(work_dir)` in outer `finally`. Proven on Windows by real `seg_mp4.unlink()` / `seg1.unlink(); seg2.unlink()` tests in `TestPackage46RealEndToEndIntegration`. |
| 6 | `pipeline.py` `_step_assemble` & `organizer.py`: full transformative assembly pipeline wired cleanly end-to-end | ✅ PASS | `pipeline.py:395-667` downloads single or multi-beat segments, resolves `clip_mode` (`auto` → `story` / `narrative`, or `raw`), runs `FacecamDetector`, `VideoComposer` (`compose` / `compose_story_beats` + `remap_timeline_for_beats` + spike zoom), `HookTitleOverlay` (suppressing `context_badge` in `raw`), `FileOrganizer.get_clip_path()`, `Exporter.export()` (suppressing narration & chat ticker in `raw`), defensive `_close_composed_handles()` on exceptions, and `FileOrganizer.save_clip()` persisting companion `.json` (all YPP fields) and `_transcript.txt`. Verified end-to-end with zero Phase 4 mocks in `TestPackage46RealEndToEndIntegration::test_real_e2e_pipeline_assemble_highlight_single_and_multi_beat` plus all 35 `tests/test_pipeline.py` tests. |

---

## 3. Code Audit Findings & Lead Review Fixes

### 3.1 Defects Found & Fixed During Lead Review (`f02cff5`)
1. **Short-clip clamping in `narrator.py` ducked `[0:a]` without trimming `[1:a]` / `[2:a]` (`src/streamclipper/assembly/narrator.py:349-410`)**: When `intro_dur + outro_dur > clip_duration * 0.75` (e.g., 4s intro + 4s outro on a 6s clip → `eff_intro=2.25s`, `outro_start=3.75s`), `[0:a]` stopped ducking at `2.25s`, but `[1:a]` lacked an `atrim` filter and continued speaking through the un-ducked climax (`2.25s–3.75s`) and overlapping `[2:a]` (`3.75s–4.0s`). **Fixed:** conditionally appends `,atrim=0:{eff:.2f},asetpts=PTS-STARTPTS` on clamped narrator tracks while keeping unclamped filtergraphs unchanged. Covered by `test_clamped_narrator_tracks_are_atrimmed_to_prevent_climax_bleed_and_overlap` (unit + real FFmpeg execution).
2. **`FacecamDetector.detect()` ignored `facecam.num_samples` config (`src/streamclipper/assembly/facecam.py:120-162`)**: `__init__` loaded `self.num_samples` from `config.get("facecam.num_samples", 50)`, but `detect()` defaulted `num_samples: int = 50`, silently overriding config whenever called as `detector.detect(segment_paths[0])` in `pipeline.py`. **Fixed:** changed signature to `num_samples: int | None = None`, defaulting to `self.num_samples` when omitted while preserving explicit call-site overrides. Covered by `test_detect_uses_configured_num_samples_by_default`.
3. **Windows file-handle leak on partial `VideoComposer` failure & out-of-bounds crop guard (`src/streamclipper/assembly/composer.py:129-165, 272-312`)**:
   - If beat 2 failed to open or compose in `compose_story_beats()`, or `_compose_with_facecam()` raised in `compose()`, previously opened `VideoFileClip` handles were never closed before re-raising. **Fixed:** tracked `opened_clips` and closed them inside `except Exception:` before re-raising. Covered by `test_compose_story_beats_closes_opened_handles_on_partial_failure` (proving `p1.unlink()` succeeds on Windows).
   - In `_crop_to_aspect()`, passing `x1 >= clip.w` clamped `x1 = clip.w` and then computed `x2 = max(x1 + 1, clip.w) = clip.w + 1`. **Fixed:** clamped `x1` to `[0, max(0, clip.w - 1)]` and `y1` to `[0, max(0, clip.h - 1)]`. Covered by `test_crop_to_aspect_out_of_bounds_box_stays_within_frame`.
4. **Zero-duration overlay wrapping (`src/streamclipper/assembly/hooks.py:203-204`)**: When `hook_title.duration_seconds == 0` (or `video_clip.duration == 0`), `add_hook()` still rasterized Pillow banners and wrapped `video_clip` in a new `CompositeVideoClip` with 0-second layers. **Fixed:** returns `video_clip` directly when `overlay_duration <= 0.0`. Covered by `test_zero_duration_overlay_returns_video_unchanged`.
5. **Milestone documentation sync (`AGENTS.md`)**: Updated repository milestone banner and Section 5.1 branch table to mark Phase 4 (`4.1–4.6` + Review `4.R`) complete and Phase 5 (`phase-5-mcp-server`) as the next branch.

### 3.2 Remaining Non-Blocking Observations
1. **Upstream MoviePy / NumPy 2.5 deprecation warning**: MoviePy `2.2.1` (`moviepy/video/io/ffmpeg_reader.py:226`) assigns `result.shape = ...`, emitting 122 `DeprecationWarning` lines during real MP4 reads under NumPy 2.5. Upstream MoviePy issue; zero functional impact.
2. **Spike zoom window duration convention (`composer.py:312-313`)**: `apply_spike_zoom` computes `z_start = max(0.0, peak_offset - 0.5)` and `z_end = min(clip.duration, peak_offset + self.zoom_duration)`, yielding a `zoom_duration + 0.5` total zoom window when `peak_offset >= 0.5` (locked by existing `tests/test_composer.py` tests). Benign visual behavior.
3. **External TTS / cloud services**: Kokoro local synthesis, Edge-TTS cloud synthesis, and ElevenLabs HTTP API calls are verified via mocks (optional packages/API keys not installed in `.venv`); real FFmpeg audio ducking, `libass` subtitle/chat burn-in, GPU `h264_nvenc`, OpenCV Haar cascade facecam detection, and MoviePy v2 rendering were all executed live on this Windows host.

---

## 4. Independent Verification Outputs

### 4.1 Full Test Suite (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`)

Executed on `phase-4-video-assembly` @ `f02cff5`:

```text
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collected 483 items

tests/test_assembly.py    (87 passed — Packages 4.3 & 4.6 + real FFmpeg/MoviePy/NVENC E2E)
tests/test_cache_cli.py   (16 passed — Phase 1 + audit hardening)
tests/test_chat.py        (14 passed — Package 2.5)
tests/test_composer.py    (26 passed — Package 4.2 + 4.R hardening)
tests/test_config.py      (39 passed — Phase 1 + validation hardening)
tests/test_discovery.py   (10 passed — Package 2.6 + hardening)
tests/test_download.py    (47 passed — Packages 2.1 & 2.2 + hardening)
tests/test_facecam.py     (31 passed — Package 4.1 + 4.R hardening)
tests/test_hooks.py       (27 passed — Package 4.4 + 4.R hardening)
tests/test_loudness.py    (14 passed — Package 2.4 + hardening)
tests/test_narrator.py    (34 passed — Package 4.5 + 4.R hardening)
tests/test_pipeline.py    (35 passed — Package 3.3 + Phase 4 real _assemble_highlight/FileOrganizer)
tests/test_scorer.py      (89 passed — Packages 3.1, 3.2 + audit & 3.R hardening)
tests/test_transcriber.py (14 passed — Package 2.3 + hardening)

================ 483 passed, 122 warnings in 61.97s (0:01:01) =================
```

### 4.2 Static & Environment Checks
- Deprecated MoviePy v1 API scan across `src/streamclipper/assembly/*.py` and `src/streamclipper/output/*.py` → **0 matches**.
- `Compare-Object (Get-Content config.default.yaml) (Get-Content src/streamclipper/config.default.yaml)` → **empty diff** (`CONFIG_PARITY_OK`).
- `D:\GTA6-Clipper\.venv\Scripts\streamclipper.exe --version` → `streamclipper, version 0.1.0`.

---

## 5. Sign-Off & Merge Action

- **Decision:** **APPROVE** Phase 4 (`phase-4-video-assembly`).
- **Merge Action:** Commit this audit report on `phase-4-video-assembly` and merge into `main` via:
  ```bash
  git checkout main && git merge --no-ff phase-4-video-assembly -m "chore: merge phase-4-video-assembly after review"
  ```
- **Gate Status:** Phase 5 (`phase-5-mcp-server`, starting with Package `5.1` — MCP Server `MCPServer` SDK v2) is cleared to branch from `main` after merge completion.
