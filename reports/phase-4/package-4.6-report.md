# Package 4.6 — Exporter & File Organizer

## 1. Metadata

| Property | Value |
| :--- | :--- |
| **Package ID** | 4.6 |
| **Phase** | 4 — Video Assembly & Output |
| **Branch** | `phase-4-video-assembly` |
| **Base Commit Hash** | `a46b54d` |
| **Difficulty** | 🟡 Medium |
| **Model Used** | Gemini 3.8 (high) |
| **Timestamp** | 2026-11-18T20:48:00+05:30 |
| **Commit Message** | `feat: clip exporter with narrator ducking, ASS burn-in, NVENC support, and file organization` |

---

## 2. Files Created / Modified

| File Path | Action | Line Count | Summary of Classes / Functions |
| :--- | :--- | :---: | :--- |
| `src/streamclipper/output/organizer.py` | **Created** | 165 | `FileOrganizer(config)`: `_sanitize_name()` (strips `<>:"/\|?*`, control chars, and leading/trailing dots/spaces), `get_clip_path()` (honors `output.structure` and `output.filename_template` with path-traversal protection and safe fallbacks), `_probe_duration_seconds()` (`ffprobe` with metadata bounds fallback), and `save_clip()` (persists companion `.json` with YPP narrative fields and `_transcript.txt`, returning `ClipResult` with `metadata_path=None` when JSON saving is disabled). |
| `src/streamclipper/output/exporter.py` | **Created** | 277 | `_close_clip_handles()` (deduplicated recursive handle cleanup for `clip`, `_source_clip`, and `_source_clips`) and `Exporter(config)`: `export()` renders MoviePy clip to intermediate temp MP4 (`libx264`, `preset="ultrafast"`, `ffmpeg_params=["-crf", "18"]`), closes all MoviePy handles in `try/finally` before FFmpeg transforms/cleanup, mixes YPP narration via `TransformativeNarrator` (`-c:v copy`, `:eval=frame` ducking) when `effective_mode != "raw"`, burns ASS captions (`Hormozi` + `ChatTicker`) via `AssCaptionBurner` using resolved `video.codec` (`h264_nvenc` → `libx264` fallback) or finalizes directly when caption-less, and removes the isolated temp work directory in `finally`. |
| `src/streamclipper/pipeline.py` | **Modified** | 784 | Verified end-to-end `_step_assemble` / `_assemble_highlight` wiring across all Phase 4 modules and added defensive `_close_composed_handles()` cleanup if an exception occurs between composition and `Exporter.export()`. |
| `tests/test_assembly.py` | **Modified** | 1586 | Added 20 new Package 4.6 tests across `TestFileOrganizer` (11 tests), `TestExporter` (7 tests), and `TestPackage46RealEndToEndIntegration` (2 real FFmpeg + MoviePy Windows integration tests covering full Phase 4 chaining and `Pipeline._assemble_highlight` single-cut + multi-beat story mode with zero Phase 4 mocks). |
| `tests/test_pipeline.py` | **Modified** | 1680 | Updated `_install_mock_stage_deps()` to mock heavy video/FFmpeg Phase 4 classes (`FacecamDetector`, `VideoComposer`, `HookTitleOverlay`, `Exporter`) while keeping the real `FileOrganizer` active so existing mocked pipeline tests with dummy 10-byte segment stubs exercise real `_assemble_highlight()` + `FileOrganizer`. |
| `reports/phase-4/package-4.6-report.md` | **Created** | — | Mandatory Package 4.6 audit and completion report. |

---

## 3. Acceptance Criteria Verification

Mapped directly against **Package 4.6** in `docs/streamclipper-implementation-guide-v2.md`:

| # | Acceptance Criterion | Verified By | Result |
| :--- | :--- | :--- | :---: |
| 1 | Clips render to MP4 with H.264 (`h264_nvenc` or `libx264`), AAC audio, ducked narrator intro/outro (`-c:v copy` audio mix), and burned-in ASS captions + chat ticker without double-compression degradation | `TestExporter::test_export_ultrafast_crf18_narrator_ducking_ass_burn_and_handle_cleanup_order` (`preset="ultrafast"`, `ffmpeg_params=["-crf", "18"]`, narrator ducking, `AssCaptionBurner` with `h264_nvenc`), `test_export_without_captions_finalizes_nvenc_with_libx264_fallback`, `test_export_raw_mode_skips_narrator_and_chat_ticker`, `test_export_chat_burst_only_still_burns_ass_when_enabled`, and `TestPackage46RealEndToEndIntegration::test_real_e2e_compose_hook_narrator_export_and_organize` | ✅ PASSED |
| 2 | MoviePy clips and all underlying `VideoFileClip` handles (`_source_clip` / `_source_clips`) are closed in `finally` before deleting intermediate temp files (zero Windows `PermissionError`) | `TestExporter::test_export_ultrafast_crf18_narrator_ducking_ass_burn_and_handle_cleanup_order` (verifies `close:composite`, `close:src1`, `close:src2` all execute strictly before narrator mix / ASS burn / temp removal), `test_export_closes_all_handles_and_cleans_temp_on_write_error`, `test_close_clip_handles_deduplicates_nested_sources`, and real Windows `seg_mp4.unlink()` / `seg1.unlink(); seg2.unlink()` checks in `TestPackage46RealEndToEndIntegration` | ✅ PASSED |
| 3 | Directory structure follows `{game}/{streamer}/{date}/` pattern with sanitized folder names | `TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots` (7 parametrized cases), `test_get_clip_path_default_structure_and_sanitization`, `test_get_clip_path_custom_templates_and_traversal_protection` | ✅ PASSED |
| 4 | Companion JSON metadata (including `moment_type`, `narration_intro`, `narration_outro`, `context_badge`, `transformation_mode`) and transcript text files are saved beside each MP4 | `TestFileOrganizer::test_save_clip_writes_metadata_json_with_ypp_fields_and_transcript`, `test_save_clip_respects_disabled_json_and_transcript_flags` (`metadata_path=None` when disabled), and `TestPackage46RealEndToEndIntegration::test_real_e2e_compose_hook_narrator_export_and_organize` | ✅ PASSED |
| 5 | `pipeline.py` `_step_assemble` wires segment download (single & multi-beat), facecam detection, 9:16 composition (+ spike zoom), hook + context badge overlay, narrator ducking, export, and organization end-to-end | `TestPackage46RealEndToEndIntegration::test_real_e2e_pipeline_assemble_highlight_single_and_multi_beat` (zero Phase 4 mocks; real single-cut `narrative` and 2-segment multi-beat `story` runs) + all 35 tests in `tests/test_pipeline.py` (now executing real `_assemble_highlight()` + real `FileOrganizer`) | ✅ PASSED |

---

## 4. Verbatim Test Output

### 4.1 Package 4.6 Dedicated Tests (`pytest tests/test_assembly.py -k "TestFileOrganizer or TestExporter or TestPackage46RealEndToEndIntegration" -v`)

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 87 items / 67 deselected / 20 selected

tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[GTA 6-unknown-GTA 6] PASSED [  5%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[GTA: 6 / "Vice" *City?<>|-unknown-GTA 6 Vice City] PASSED [ 10%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[  ..TimTheTatman..  -unknown-TimTheTatman] PASSED [ 15%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[foo\\bar:baz-unknown-foobarbaz] PASSED [ 20%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[   ...   -UnknownGame-UnknownGame] PASSED [ 25%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[<>:"/\\|?*-fallback_name-fallback_name] PASSED [ 30%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[-unknown-unknown] PASSED [ 35%]
tests/test_assembly.py::TestFileOrganizer::test_get_clip_path_default_structure_and_sanitization PASSED [ 40%]
tests/test_assembly.py::TestFileOrganizer::test_get_clip_path_custom_templates_and_traversal_protection PASSED [ 45%]
tests/test_assembly.py::TestFileOrganizer::test_save_clip_writes_metadata_json_with_ypp_fields_and_transcript PASSED [ 50%]
tests/test_assembly.py::TestFileOrganizer::test_save_clip_respects_disabled_json_and_transcript_flags PASSED [ 55%]
tests/test_assembly.py::TestExporter::test_export_ultrafast_crf18_narrator_ducking_ass_burn_and_handle_cleanup_order PASSED [ 60%]
tests/test_assembly.py::TestExporter::test_export_closes_all_handles_and_cleans_temp_on_write_error PASSED [ 65%]
tests/test_assembly.py::TestExporter::test_export_raw_mode_skips_narrator_and_chat_ticker PASSED [ 70%]
tests/test_assembly.py::TestExporter::test_export_explicit_mode_overrides_disabled_config_and_restores_narrator PASSED [ 75%]
tests/test_assembly.py::TestExporter::test_export_chat_burst_only_still_burns_ass_when_enabled PASSED [ 80%]
tests/test_assembly.py::TestExporter::test_export_without_captions_finalizes_nvenc_with_libx264_fallback PASSED [ 85%]
tests/test_assembly.py::TestExporter::test_close_clip_handles_deduplicates_nested_sources PASSED [ 90%]
tests/test_assembly.py::TestPackage46RealEndToEndIntegration::test_real_e2e_compose_hook_narrator_export_and_organize PASSED [ 95%]
tests/test_assembly.py::TestPackage46RealEndToEndIntegration::test_real_e2e_pipeline_assemble_highlight_single_and_multi_beat PASSED [100%]

============================== warnings summary ===============================
tests/test_assembly.py: 116 warnings
  D:\GTA6-Clipper\.venv\Lib\site-packages\moviepy\video\io\ffmpeg_reader.py:226: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    result.shape = (h, w, len(s) // (w * h))  # reshape((h, w, len(s)//(w*h)))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=============== 20 passed, 67 deselected, 116 warnings in 9.17s ===============
```

### 4.2 Full Repository Test Suite (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`)

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 478 items
...
============================== warnings summary ===============================
tests/test_assembly.py: 116 warnings
tests/test_composer.py: 5 warnings
  D:\GTA6-Clipper\.venv\Lib\site-packages\moviepy\video\io\ffmpeg_reader.py:226: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    result.shape = (h, w, len(s) // (w * h))  # reshape((h, w, len(s)//(w*h)))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
===================== 478 passed, 121 warnings in 48.19s ======================
```

Zero failures, zero skipped tests, zero regressions across all 478 tests spanning Phases 1–4.

---

## 5. Architectural Notes, Edge Cases & Phase 4 Review Readiness

### 5.1 Key Implementation Details & Edge Cases Handled
1. **Isolated Per-Export Working Directory**: `Exporter.export()` creates a unique `streamclipper_export_*` temp directory under `general.temp_dir` for `intermediate_raw.mp4`, `temp_audio.m4a`, synthesized TTS WAVs, `intermediate_mixed.mp4`, and `captions.ass`. This prevents race conditions during parallel stream processing (`process_streams_parallel`) and guarantees user-supplied `--voice-intro` / `--voice-outro` files outside the work directory are never touched during cleanup.
2. **Deterministic Windows Handle Release**: `_close_clip_handles()` traverses `clip`, `_source_clip`, and `_source_clips` (deduplicated by object ID) inside an inner `try/finally` around `clip.write_videofile()` (and again idempotently in the outer `finally`), releasing all MoviePy/FFmpeg file locks before narration mixing, ASS burn-in, or `shutil.rmtree()`. Verified live on Windows by unlinking both source segment MP4s immediately after export in `TestPackage46RealEndToEndIntegration`.
3. **Per-Run `mode` Override Consistency**: When an explicit `mode` (e.g., `mode="narrative"`) overrides a config default (`transformation.enabled: false` or `transformation.mode: "raw"`), `Exporter` temporarily syncs `self.narrator.enabled` and `self.narrator.mode` in a `try/finally` block and restores the original attributes afterward — matching `Pipeline._resolve_base_mode()` without mutating persistent narrator state.
4. **Single-Pass Quality Preservation**: Intermediate MoviePy rendering uses near-lossless `libx264` (`preset="ultrafast"`, `-crf 18`), narration ducking uses `-c:v copy`, and final encoding occurs exactly once (via `AssCaptionBurner.burn_captions()` or `_finalize_without_captions()`) at the resolved hardware codec (`h264_nvenc` with automatic `libx264` fallback) and configured `video.crf`.
5. **Path Traversal & Template Safety**: `FileOrganizer.get_clip_path()` strips `..`/`.` path components and sanitizes every directory segment and filename stem so malformed or hostile `output.structure` / `output.filename_template` configs cannot escape `general.output_dir`.

### 5.2 Readiness Summary for Phase 4 Opus Review (`Review 4.R`)
All 6 packages in **Phase 4 (`phase-4-video-assembly`)** are implemented, documented, and passing with **478/478 tests green**:

| Package | Module(s) | Report | Status |
| :--- | :--- | :--- | :---: |
| **4.1** Facecam Detector | `src/streamclipper/assembly/facecam.py` | `reports/phase-4/package-4.1-report.md` | ✅ Complete |
| **4.2** 9:16 Video Composer (+ Story Beats & Spike Zoom) | `src/streamclipper/assembly/composer.py` | `reports/phase-4/package-4.2-report.md` | ✅ Complete |
| **4.3** Hormozi ASS Captions & Live Chat Ticker | `src/streamclipper/assembly/captions.py` | `reports/phase-4/package-4.3-report.md` | ✅ Complete |
| **4.4** Hook Title & Adaptive Context Badge Overlay | `src/streamclipper/assembly/hooks.py` | `reports/phase-4/package-4.4-report.md` | ✅ Complete |
| **4.5** YPP Transformative Narrator & Dynamic Audio Ducking | `src/streamclipper/assembly/narrator.py` | `reports/phase-4/package-4.5-report.md` | ✅ Complete |
| **4.6** Exporter & File Organizer (+ End-to-End Wiring) | `src/streamclipper/output/exporter.py`, `src/streamclipper/output/organizer.py` | `reports/phase-4/package-4.6-report.md` | ✅ Complete |

**Branch `phase-4-video-assembly` is ready for Phase 4 Review (`4.R`) by Claude Opus 4.6 and merge into `main`.**
