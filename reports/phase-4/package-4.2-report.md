# Package 4.2 — 9:16 Video Composer (MoviePy v2 + Conditional Story Beats + Spike Zoom)

## 1. Metadata

| Property | Value |
| :--- | :--- |
| **Package ID** | 4.2 |
| **Phase** | 4 — Video Assembly & Output |
| **Branch** | `subagent-Worker-Package-4-2-teamwork-worker-b297a34e` |
| **Difficulty** | 🟠 Hard |
| **Model** | Gemini 3.8 (high) |
| **Timestamp** | 2026-11-18T14:25:00Z |
| **Commit Message** | `feat: 9:16 MoviePy v2 composer with conditional story beats and spike punch-in zoom` |

## 2. Files Created / Modified

| File | Action | Lines | Summary |
| :--- | :--- | :--- | :--- |
| `src/streamclipper/assembly/composer.py` | **Created** | 300 | Implemented `VideoComposer(config)` with MoviePy v2 direct API (`VideoFileClip`, `CompositeVideoClip`, `concatenate_videoclips`), `_crop_to_aspect` (aspect-preserving center crop without horizontal distortion), `_compose_with_facecam` (Layout A: top 35% facecam, bottom 65% gameplay), `_compose_gameplay_only` (Layout B: 1080×1920 center-cropped gameplay), `compose_story_beats` (duration clamping to `[0.0, beat_clip.duration]` guarding MoviePy 2.2.1 `ValueError`, buffer trimming, multi-beat concatenation, and 1-segment fallback), `remap_timeline_for_beats` (stream-absolute to stitched `0.0`-based timeline remapping for words, chat burst, and peak offset while filtering dead air), `apply_spike_zoom` (1.20x center punch-in zoom at peak moment), and Windows file-handle tracking (`_source_clip` / `_source_clips`). |
| `tests/test_composer.py` | **Created** | 406 | Created 24 unit tests across 7 test classes (`TestVideoComposerInit`, `TestCropToAspect`, `TestLayoutComposition`, `TestApplySpikeZoom`, `TestComposeStoryBeats`, `TestRemapTimelineForBeats`, `TestRealFileHandlingAndWindowsCleanup`) covering all composition layouts, duration clamping, timeline remapping, spike zoom, audio preservation, and Windows file handle cleanup. |
| `reports/phase-4/package-4.2-report.md` | **Created** | — | This package audit and verification report. |

## 3. Acceptance Criteria Verification

| # | Criterion | Test(s) | Result |
| :--- | :--- | :--- | :--- |
| 1 | Layout A stacks aspect-preserving facecam on top (35%, zero stretching) and gameplay below (65%) using MoviePy v2 `.cropped()` and `.resized()` | `TestLayoutComposition::test_layout_a_with_facecam_stack`, `TestLayoutComposition::test_compose_routes_facecam_vs_gameplay` | ✅ 2/2 PASSED |
| 2 | Layout B center-crops gameplay to 1080×1920 without stretching | `TestLayoutComposition::test_layout_b_gameplay_only`, `TestLayoutComposition::test_compose_routes_facecam_vs_gameplay` | ✅ 2/2 PASSED |
| 3 | `compose_story_beats()` clamps per-beat trim ranges to `[0.0, beat_clip.duration]` before `.subclipped()`, stitches 2+ beat segments cleanly, and falls back safely to single-segment `compose()` when only 1 path is provided | `TestComposeStoryBeats::test_stitch_two_beats_with_clamping`, `TestComposeStoryBeats::test_stitch_three_beats_without_trim_ranges`, `TestComposeStoryBeats::test_invalid_trim_range_handled_safely`, `TestComposeStoryBeats::test_fallback_to_compose_on_single_segment`, `TestComposeStoryBeats::test_empty_segments_raises_value_error` | ✅ 5/5 PASSED |
| 4 | `remap_timeline_for_beats()` accurately remaps word timestamps, `chat_burst` messages, and `peak_offset` onto the stitched `0.0`-based timeline while dropping dead-air entries | `TestRemapTimelineForBeats::test_words_and_chat_shifted_and_dead_air_dropped`, `TestRemapTimelineForBeats::test_peak_in_dead_air_returns_none`, `TestRemapTimelineForBeats::test_peak_zero_or_negative_returns_none`, `TestRemapTimelineForBeats::test_empty_beats_returns_original`, `TestRemapTimelineForBeats::test_chat_timestamp_field_variations` | ✅ 5/5 PASSED |
| 5 | `apply_spike_zoom()` applies a `1.20x` center punch-in zoom around `peak_offset` when `transformation.punch_in_zoom > 1.0` without changing output resolution or total duration | `TestApplySpikeZoom::test_zoom_applied_around_peak_offset`, `TestApplySpikeZoom::test_zoom_near_beginning`, `TestApplySpikeZoom::test_zoom_disabled_when_factor_is_one`, `TestApplySpikeZoom::test_zoom_skipped_when_peak_offset_none_or_negative`, `TestApplySpikeZoom::test_zoom_skipped_when_peak_beyond_duration` | ✅ 5/5 PASSED |
| 6 | Original audio is preserved via `.with_audio(...)` | `TestLayoutComposition::test_layout_a_with_facecam_stack`, `TestLayoutComposition::test_layout_b_gameplay_only`, `TestApplySpikeZoom::test_zoom_applied_around_peak_offset`, `TestComposeStoryBeats::test_stitch_two_beats_with_clamping` | ✅ 4/4 PASSED |
| 7 | All source `VideoFileClip` references (`_source_clip` / `_source_clips`) are preserved for clean Windows handle release | `TestLayoutComposition::test_compose_routes_facecam_vs_gameplay`, `TestApplySpikeZoom::test_zoom_applied_around_peak_offset`, `TestComposeStoryBeats::test_fallback_to_compose_on_single_segment`, `TestComposeStoryBeats::test_stitch_two_beats_with_clamping`, `TestRealFileHandlingAndWindowsCleanup::test_compose_with_real_mp4_files` | ✅ 5/5 PASSED |

**Composer Test Suite:** `24/24 PASSED` (100% pass rate in `6.35s`)

## 4. Verbatim `pytest tests/test_composer.py -v` Output

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\praka\.gemini\antigravity\worktrees\bed06a0d-19d9-43a5-8a4e-7cf8178d0fcd\subagent-Worker-Package-4-2-teamwork-worker-b297a34e
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 24 items

tests/test_composer.py::TestVideoComposerInit::test_default_config_resolution_and_zoom PASSED [  4%]
tests/test_composer.py::TestVideoComposerInit::test_custom_config_resolution_and_zoom PASSED [  8%]
tests/test_composer.py::TestCropToAspect::test_crop_square_region_to_wide_aspect PASSED [ 12%]
tests/test_composer.py::TestCropToAspect::test_crop_wide_region_to_narrow_aspect PASSED [ 16%]
tests/test_composer.py::TestCropToAspect::test_crop_coordinates_boundary_clamped PASSED [ 20%]
tests/test_composer.py::TestLayoutComposition::test_layout_a_with_facecam_stack PASSED [ 25%]
tests/test_composer.py::TestLayoutComposition::test_layout_b_gameplay_only PASSED [ 29%]
tests/test_composer.py::TestLayoutComposition::test_compose_routes_facecam_vs_gameplay PASSED [ 33%]
tests/test_composer.py::TestApplySpikeZoom::test_zoom_applied_around_peak_offset PASSED [ 37%]
tests/test_composer.py::TestApplySpikeZoom::test_zoom_near_beginning PASSED [ 41%]
tests/test_composer.py::TestApplySpikeZoom::test_zoom_disabled_when_factor_is_one PASSED [ 45%]
tests/test_composer.py::TestApplySpikeZoom::test_zoom_skipped_when_peak_offset_none_or_negative PASSED [ 50%]
tests/test_composer.py::TestApplySpikeZoom::test_zoom_skipped_when_peak_beyond_duration PASSED [ 54%]
tests/test_composer.py::TestComposeStoryBeats::test_fallback_to_compose_on_single_segment PASSED [ 58%]
tests/test_composer.py::TestComposeStoryBeats::test_stitch_two_beats_with_clamping PASSED [ 62%]
tests/test_composer.py::TestComposeStoryBeats::test_stitch_three_beats_without_trim_ranges PASSED [ 66%]
tests/test_composer.py::TestComposeStoryBeats::test_invalid_trim_range_handled_safely PASSED [ 70%]
tests/test_composer.py::TestComposeStoryBeats::test_empty_segments_raises_value_error PASSED [ 75%]
tests/test_composer.py::TestRemapTimelineForBeats::test_words_and_chat_shifted_and_dead_air_dropped PASSED [ 79%]
tests/test_composer.py::TestRemapTimelineForBeats::test_peak_in_dead_air_returns_none PASSED [ 83%]
tests/test_composer.py::TestRemapTimelineForBeats::test_peak_zero_or_negative_returns_none PASSED [ 87%]
tests/test_composer.py::TestRemapTimelineForBeats::test_empty_beats_returns_original PASSED [ 91%]
tests/test_composer.py::TestRemapTimelineForBeats::test_chat_timestamp_field_variations PASSED [ 95%]
tests/test_composer.py::TestRealFileHandlingAndWindowsCleanup::test_compose_with_real_mp4_files PASSED [100%]

======================= 24 passed, 5 warnings in 6.35s ========================
```

## 5. Architectural Notes & Edge Cases

1. **MoviePy v2 Direct Imports & Modern API**:
   - Strictly imported `from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips` (never using deprecated `moviepy.editor`).
   - Used MoviePy v2 chaining API: `.cropped()`, `.resized()`, `.subclipped()`, `.with_position()`, and `.with_audio()`.
2. **Aspect-Correct Facecam Region Cropping**:
   - `_crop_to_aspect` calculates target aspect ratio `target_ratio = self.width / face_h` (~1.607 for 1080x672) and center-crops the detected bounding box before resizing. This guarantees square (1:1) or 4:3 webcams are never horizontally stretched into wide banners.
3. **MoviePy 2.2.1 Duration Clamping Guardrail**:
   - MoviePy 2.2.1 raises `ValueError: end_time should be smaller or equal to the clip's duration` if `t_end > clip.duration`.
   - `compose_story_beats` clamps `t_start = max(0.0, float(beat_trim_ranges[i][0]))` and `t_end = min(float(raw_clip.duration), float(beat_trim_ranges[i][1]))`, and verifies `t_end > t_start` before calling `.subclipped()`.
4. **Timeline Remapping Across Multi-Beat Jump Cuts**:
   - `remap_timeline_for_beats` shifts word timestamps, chat burst messages (`offset_seconds`, `timestamp`, `time`, `offset`), and `peak_time` onto the contiguous stitched timeline starting at `0.0`.
   - Word boundaries straddling cuts are safely trimmed, and events falling entirely in dead air are discarded. If `peak_time` occurred inside cut dead air, `peak_offset` returns `None`.
5. **Center Spike Zoom**:
   - `apply_spike_zoom` extracts the window `[max(0.0, peak_offset - 0.5), min(clip.duration, peak_offset + zoom_duration)]`, center-crops by `1 / punch_in_zoom`, resizes back to `(self.width, self.height)`, concatenates slices, and reattaches the clip's audio.
6. **Windows File Handle Tracking**:
   - Attached `_source_clip` and `_source_clips` references to every returned `CompositeVideoClip` so downstream consumers (`Exporter`) can safely close underlying file readers before temp file deletion on Windows.
