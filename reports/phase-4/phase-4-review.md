# Phase 4 Lead Reviewer Audit Report — `phase-4-video-assembly`

## 1. Executive Verdict

| Field | Value |
| :--- | :--- |
| **Phase Audited** | Phase 4: Video Assembly & Output (`phase-4-video-assembly`) |
| **Specification Reference** | `docs/streamclipper-implementation-guide-v2.md` (v2.2 — Patched + Auto-Hardware + YPP Transformative Engine) & `AGENTS.md` |
| **Commits Reviewed** | `main` pre-Phase-4 (`565893d`) `...` `phase-4-video-assembly` (`e138317`) — Packages `4.1`–`4.6` (`f55f633`/`52cbe11`, `ead1879`/`a1db797`, `a46b54d`, `9cfebdd`/`ee3224f`, `cf78d7d`/`cd7c62c`, `c5a1c81`), Lead Review pass 1 (`f02cff5`), and Lead Review pass 2 hardening (`e138317`), 22 files changed, `+7,331` / `-71` lines |
| **Package Reports Audited** | `reports/phase-4/package-4.1-report.md` through `reports/phase-4/package-4.6-report.md` |
| **Test Suite Verification** | `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` — **496 / 496 passed** (`50.29s`, Python 3.12.10, pytest 9.2.0, zero failures, zero skipped; 122 upstream MoviePy/NumPy 2.5 `DeprecationWarning`s) |
| **Audit Verdict** | ✅ **APPROVED (after Lead Review fixes in `f02cff5` and `e138317`) — Cleared for `--no-ff` merge into `main`** |
| **Review Timestamp** | 2026-11-18 21:06 IST |
| **Reviewer** | Lead Reviewer Audit (`4.R`) |

Phase 4 satisfies all specification requirements, acceptance criteria, and v2.2 architectural guardrails across Packages `4.1`–`4.6` plus all Lead Review defect fixes (`f02cff5` and `e138317`, adding 18 new regression tests total). All 496 unit, mocked integration, and real FFmpeg/MoviePy/NVENC/OpenCV end-to-end tests pass cleanly on Windows.

---

## 2. Verification Matrix (Lead Reviewer Checklist)

| # | Audit Requirement | Result | Evidence & Code Inspection |
| :-: | :--- | :---: | :--- |
| 1 | Package reports `4.1`–`4.6` audited & reconciled with git history | ✅ PASS | All 6 reports exist in `reports/phase-4/` and reconcile with `git log 565893d..phase-4-video-assembly`: `4.1` (`f55f633`/`52cbe11`), `4.2` (`ead1879`/`a1db797`), `4.4` (`9cfebdd`/`ee3224f`), `4.5` (`cf78d7d`/`cd7c62c`), `4.3` (`a46b54d`), `4.6` (`c5a1c81`), plus Lead Review fixes `f02cff5` and `e138317`. Earlier report test totals (`308`/`304`/`311`/`458`/`478`/`483`) reflect their branch snapshots; HEAD stands at **496 / 496**. |
| 2 | `composer.py` & `hooks.py`: MoviePy v2 imports/methods, aspect-preserving facecam crop, conditional multi-beat story stitching, `1.2x` spike zoom, and `_source_clip` / `_source_clips` tracking | ✅ PASS | Direct v2 imports (`from moviepy import CompositeVideoClip, VideoFileClip, concatenate_videoclips` in `composer.py`; `from moviepy import CompositeVideoClip, ImageClip` in `hooks.py`); zero deprecated v1 APIs (`moviepy.editor`, `.crop(`, `.resize(`, `.subclip(`, `.set_*(`) found across `assembly/` and `output/`. `VideoComposer._crop_to_aspect()` preserves target aspect ratio without stretching and bounds `[0, max(0, w - 1)]`; `compose_story_beats()` clamps `(t_start, t_end)` to `[0.0, raw_clip.duration]` before `.subclipped()`, stitches beats via `concatenate_videoclips(..., method="compose")`, closes opened handles on partial failure, and falls back to `compose()` for 1 segment; `remap_timeline_for_beats()` shifts words/chat/peak onto `0.0`-based stitched timeline while safely ignoring malformed chat entries and trailing zero-duration beats; `apply_spike_zoom()` applies exact spec window `z_start = max(0.0, peak_offset - 0.5)`, `z_end = min(clip.duration, z_start + zoom_duration)` (`zoom_duration=0` skips zoom) preserving audio and `_source_clip` / `_source_clips`. `HookTitleOverlay.add_hook()` uses Pillow RGBA arrays + MoviePy v2 `ImageClip` (no ImageMagick), `.with_duration()`, `.with_position()`, `CrossFadeIn`, `.with_audio()`, propagates `_source_clip` / `_source_clips`, short-circuits zero-duration overlays, and wraps oversized single-token hooks and long context badges within 90% frame width. |
| 3 | `captions.py`: dynamic `captions.highlight_color` → ASS BGR, `Hormozi` + `ChatTicker` styles, ASS delimiter escaping, Windows-safe `cwd=ass_path.parent` + `ass=<basename>`, `h264_nvenc` (`-cq`) with `libx264` fallback | ✅ PASS | `captions.py:108-125, 174-179` converts `#FFD700` → `&H00D7FF&` (with named color fallback and strict 6-hex regex). `generate_ass()` emits `Style: Hormozi` (honoring `style`, `color`, `stroke_color`, `stroke_width`, `font`, `font_size`, `margin_v`) and `Style: ChatTicker` (`BorderStyle=3`, semi-transparent `OutlineColour=&H99000000`, up to 5 messages). `_escape_ass_text()` + trailing-`\` guard neutralize `\r`/`\n`, `{`, `}`, and `\N`/`\n`/`\h`. `_sec_to_ass_time()` avoids `60.00` centisecond rollover. `burn_captions()` runs FFmpeg with `cwd=str(ass_file.resolve().parent)`, `-vf ass={_escape_ffmpeg_ass_filename(ass_file.name)}`, `-c:a copy`, `h264_nvenc` (`-preset p4 -cq <crf>`) with single-retry `libx264` (`-crf <crf>`) fallback. Verified by 67 tests in `tests/test_assembly.py` including live `libass` and GPU `h264_nvenc` burn-in. |
| 4 | `narrator.py`: pluggable TTS (`kokoro`, `edge-tts`, optional `elevenlabs` via `ELEVENLABS_API_KEY` + `voice_id`), custom `--voice-intro`/`--voice-outro`, dynamic 2-input / 3-input FFmpeg ducking (`:eval=frame`, 48kHz stereo) with short-clip overlap clamping | ✅ PASS | `TransformativeNarrator` supports `kokoro`, `edge-tts` (using `self.edge_tts_voice`, never Kokoro voice ID), `elevenlabs` (`ELEVENLABS_API_KEY` + `elevenlabs_voice_id`, falling back to `auto` when unconfigured), `auto` (`kokoro` → `edge-tts`), and `none`; `resolve_narration_track()` prioritizes custom human mic files over TTS; `_probe_audio_duration()` probes via `soundfile` with `ffprobe` fallback so non-WAV custom inputs (`.m4a`, `.aac`, `.mp3`) are mixed cleanly. `_build_ducking_filtergraph()` dynamically emits 2-input / 3-input filtergraphs with `volume='...':eval=frame`, `aresample=48000,aformat=channel_layouts=stereo`, stereo `adelay={ms}|{ms}`, `amix=inputs={n}:duration=first:normalize=0`, conditional `atrim=0:{max(0.01, eff):.2f},asetpts=PTS-STARTPTS` on clamped tracks, and `anullsrc` base fallback when `input_video` has no audio stream. Case-insensitive `mode="RAW"` bypass verified. |
| 5 | `exporter.py`: closes all clip handles in `finally` before deleting temp files & avoids double-compression degradation | ✅ PASS | `Exporter.export()` writes intermediate MP4 via MoviePy at `codec="libx264"`, `preset="ultrafast"`, `ffmpeg_params=["-crf", "18"]` inside an isolated per-export `work_dir`, immediately calls `_release_handles_once()` (`_close_clip_handles` walking `clip`, `_source_clip`, and `_source_clips` deduplicated by `id`) inside an inner `try/finally` before narration mixing (`-c:v copy`) and final single-pass H.264 encode (ASS burn-in or `_finalize_without_captions` with `h264_nvenc` → `libx264` fallback), and guarantees `_release_handles_once()` runs again before `shutil.rmtree(work_dir)` in outer `finally`. Proven on Windows by real `seg_mp4.unlink()` / `seg1.unlink(); seg2.unlink()` tests in `TestPackage46RealEndToEndIntegration`. |
| 6 | `pipeline.py` `_step_assemble` & `organizer.py`: full transformative assembly pipeline wired cleanly end-to-end | ✅ PASS | `pipeline.py` downloads single or multi-beat segments, resolves `clip_mode` (`auto` → `story` / `narrative`, or `raw`), runs `FacecamDetector` (honoring configured `facecam.num_samples`), `VideoComposer` (`compose` / `compose_story_beats` + `remap_timeline_for_beats` + spike zoom), `HookTitleOverlay` (suppressing `context_badge` in `raw`), `FileOrganizer.get_clip_path()` (guarding invalid chars, path traversal, and Windows reserved device names `CON`/`PRN`/`AUX`/`NUL`/`COM1`–`COM9`/`LPT1`–`LPT9`), `Exporter.export()` (suppressing narration & chat ticker in `raw`), defensive `_close_composed_handles()` on exceptions, and `FileOrganizer.save_clip()` persisting companion `.json` (all YPP fields) and `_transcript.txt`. Verified end-to-end with zero Phase 4 mocks in `TestPackage46RealEndToEndIntegration::test_real_e2e_pipeline_assemble_highlight_single_and_multi_beat` plus all 35 `tests/test_pipeline.py` tests. |

---

## 3. Code Audit Findings & Lead Review Fixes

### 3.1 Defects Found & Fixed in Lead Review Pass 1 (`f02cff5`)
1. **Short-clip clamping in `narrator.py` ducked `[0:a]` without trimming `[1:a]` / `[2:a]`**: Added conditional `,atrim=0:{eff:.2f},asetpts=PTS-STARTPTS` on clamped narrator tracks so intro audio cannot bleed into un-ducked climax or overlap outro audio.
2. **`FacecamDetector.detect()` ignored `facecam.num_samples` config**: Changed default parameter to `num_samples: int | None = None`, resolving `self.num_samples` from `Config` when omitted.
3. **Windows file-handle leak on partial `VideoComposer` failure & out-of-bounds crop guard**: Tracked `opened_clips` in `compose()` / `compose_story_beats()` and closed them in `except Exception:`; tightened `_crop_to_aspect()` bounds to `[0, max(0, w - 1)]` / `[0, max(0, h - 1)]`.
4. **Zero-duration overlay wrapping in `hooks.py`**: Returned `video_clip` unchanged when `overlay_duration <= 0.0`.
5. **Milestone documentation sync (`AGENTS.md`)**: Updated repository milestone banner and Section 5.1 branch table to mark Phase 4 complete and Phase 5 (`phase-5-mcp-server`) next.

### 3.2 Defects Found & Fixed in Lead Review Pass 2 (`e138317`)
1. **Silent video clips dropped narrator audio (`src/streamclipper/assembly/narrator.py`)**: When `input_video` had no audio stream (`clip.audio is None`), `mix_narration_with_ducking()` referenced `[0:a]`, failed FFmpeg with exit code `4294967274`, and bypassed to a silent copy. **Fixed:** added `_has_audio_stream()` probe and `anullsrc=channel_layout=stereo:sample_rate=48000,atrim=0:{clip_dur:.2f},asetpts=PTS-STARTPTS[a0]` fallback when `has_video_audio=False`.
2. **Non-WAV custom voiceover containers (`.m4a`/`.aac`) silently skipped (`src/streamclipper/assembly/narrator.py`)**: `sf.info()` raised on `.m4a` inputs, leaving duration `None` and bypassing mixing. **Fixed:** added `_probe_audio_duration()` with `ffprobe` fallback.
3. **Case-sensitive `mode == "raw"` and tiny `atrim=0:0.00` edge case (`src/streamclipper/assembly/narrator.py`)**: Normalized `self.mode` with `.strip().lower()` and floored `atrim` endpoints at `max(0.01, eff)`.
4. **Oversized single-token hook titles & long `context_badge` overflowed 1080px frame (`src/streamclipper/assembly/hooks.py`)**: Single unspaced tokens rendered up to `1939px` wide and long context badges rendered up to `1509px` wide. **Fixed:** added character-chunk splitting (`_split_oversized_word`, opt-in via `break_long_words=True` on `_wrap_text` to preserve word-wrap unit contract) and optional `max_width` wrapping on `render_context_badge()` (wired to `int(video_w * 0.90)` in `add_hook()`).
5. **`apply_spike_zoom()` duration formula deviated from spec & failed on `zoom_duration == 0.0` (`src/streamclipper/assembly/composer.py`)**: Used `peak_offset + self.zoom_duration` instead of spec `z_start + self.zoom_duration` (`docs/streamclipper-implementation-guide-v2.md:2454`), adding 0.5s to every zoom window and still zooming 0.5s when `zoom_duration_seconds == 0.0`. **Fixed:** corrected to `z_end = min(float(clip.duration), z_start + self.zoom_duration)` with spy-verified slice duration test.
6. **`remap_timeline_for_beats()` crashed on malformed `chat_burst` entries & dropped boundary events with trailing zero-duration beats (`src/streamclipper/assembly/composer.py`)**: Pre-filtered valid beats before checking `is_last_beat` and defensively skipped non-dict, bool, non-numeric, or non-finite timestamps.
7. **Windows reserved device names crashed `FileOrganizer.get_clip_path()` (`src/streamclipper/output/organizer.py`, `src/streamclipper/pipeline.py`)**: Passing `CON`, `NUL`, `AUX`, `PRN`, `COM1`–`COM9`, or `LPT1`–`LPT9` raised `FileNotFoundError: [WinError 3]` on Windows. **Fixed:** guarded reserved device stems in both `FileOrganizer._sanitize_name()` and `pipeline._sanitize_path_segment()`.

### 3.3 Remaining Non-Blocking Observations
1. **Upstream MoviePy / NumPy 2.5 deprecation warning**: MoviePy `2.2.1` (`moviepy/video/io/ffmpeg_reader.py:226`) assigns `result.shape = ...`, emitting 122 `DeprecationWarning` lines during real MP4 reads under NumPy 2.5. Upstream MoviePy issue; zero functional impact.
2. **External TTS / cloud services**: Kokoro local synthesis, Edge-TTS cloud synthesis, and ElevenLabs HTTP API calls are verified via mocks (optional packages/API keys not installed in `.venv`); real FFmpeg audio ducking (including silent-video `anullsrc`, `.m4a` custom voiceovers, and clamped `atrim`), `libass` subtitle/chat burn-in, GPU `h264_nvenc`, OpenCV Haar cascade facecam detection, and MoviePy v2 rendering were all executed live on this Windows host.

---

## 4. Independent Verification Outputs

### 4.1 Full Test Suite (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`)

Executed on `phase-4-video-assembly` @ `e138317`:

```text
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collected 496 items

tests/test_assembly.py    (94 passed — Packages 4.3 & 4.6 + real FFmpeg/MoviePy/NVENC E2E + Windows reserved names)
tests/test_cache_cli.py   (16 passed — Phase 1 + audit hardening)
tests/test_chat.py        (14 passed — Package 2.5)
tests/test_composer.py    (28 passed — Package 4.2 + 4.R hardening)
tests/test_config.py      (39 passed — Phase 1 + validation hardening)
tests/test_discovery.py   (10 passed — Package 2.6 + hardening)
tests/test_download.py    (47 passed — Packages 2.1 & 2.2 + hardening)
tests/test_facecam.py     (31 passed — Package 4.1 + 4.R hardening)
tests/test_hooks.py       (28 passed — Package 4.4 + 4.R hardening)
tests/test_loudness.py    (14 passed — Package 2.4 + hardening)
tests/test_narrator.py    (37 passed — Package 4.5 + 4.R hardening)
tests/test_pipeline.py    (35 passed — Package 3.3 + Phase 4 real _assemble_highlight/FileOrganizer)
tests/test_scorer.py      (89 passed — Packages 3.1, 3.2 + audit & 3.R hardening)
tests/test_transcriber.py (14 passed — Package 2.3 + hardening)

===================== 496 passed, 122 warnings in 50.29s ======================
```

### 4.2 Static & Environment Checks
- Deprecated MoviePy v1 API scan across `src/streamclipper/assembly/*.py` and `src/streamclipper/output/*.py` → **0 matches**.
- `Compare-Object (Get-Content config.default.yaml) (Get-Content src/streamclipper/config.default.yaml)` → **empty diff** (`CONFIG_PARITY_OK`).
- `D:\GTA6-Clipper\.venv\Scripts\streamclipper.exe --version` → `streamclipper, version 0.1.0`.

---

## 5. Sign-Off & Merge Action

- **Decision:** **APPROVE** Phase 4 (`phase-4-video-assembly`).
- **Merge Action:** Commit this updated audit report on `phase-4-video-assembly` and merge into `main` via:
  ```bash
  git checkout main && git merge --no-ff phase-4-video-assembly -m "chore: merge phase-4-video-assembly after review"
  ```
- **Gate Status:** Phase 5 (`phase-5-mcp-server`, starting with Package `5.1` — MCP Server `MCPServer` SDK v2) is cleared to branch from `main` after merge completion.
