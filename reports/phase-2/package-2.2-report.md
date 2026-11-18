# Package 2.2 Implementation Report — Video Segment Downloader

## 1. Metadata

| Field | Value |
| :--- | :--- |
| **Package ID** | `2.2` — Video Segment Downloader |
| **Phase** | Phase 2: Core Pipeline — Download & Analysis |
| **Branch** | `phase-2-core-pipeline` (parent commit: `32c808e` — Package 2.1 complete) |
| **Commit Message** | `feat: video segment downloader for specific time ranges` |
| **Timestamp** | 2026-11-18 16:52 IST |
| **Model Used** | Gemini 3.8 Flash (High) |
| **Python Environment** | Python 3.12.10 (`D:\GTA6-Clipper\.venv\Scripts\python.exe`), `yt-dlp` 2026.10.26, `pytest` 9.2.0 |

---

## 2. Files Created / Modified

- Verified branch is `phase-2-core-pipeline` and confirmed Package 2.1 completion via `reports/phase-2/package-2.1-report.md` (commit `32c808e`).
- Strictly scoped all changes to Package 2.2 deliverables.

| File Path | Action | Lines | Summary of Classes / Functions |
| :--- | :---: | :---: | :--- |
| `src/streamclipper/download/video.py` | Created | 264 | `VIDEO_FORMAT_1080P`, `VideoDownloadError(RuntimeError)`, `_validate_url()`, `_validate_range()`, `_classify_video_error()`, `_extract_video_id()`, `_download_segment_with_id()`, `download_segment(url, start_seconds, end_seconds, output_dir, buffer_seconds=3.0) -> Path`, and `download_multiple_segments(url, segments, output_dir, buffer_seconds=3.0) -> list[Path]` using `yt_dlp.utils.download_range_func(None, [(actual_start, actual_end)])`, `force_keyframes_at_cuts: True`, 1080p MP4 merge, `0.0` start clamping, single-probe video ID reuse, and cached segment skip |
| `tests/test_download.py` | Modified | 575 (+223) | Expanded with `TestVideoSegmentDownloader` (17 new test cases, 41 total in `tests/test_download.py`) verifying `yt-dlp` format/merge/keyframe options, `download_range_func` spy + real callback output, `±buffer_seconds` padding and `0.0` clamping, cache skip when segment MP4 exists, multi-segment single-probe reuse and partial-cache skip, empty segment list handling, upfront input validation, and `VideoDownloadError` wrapping |
| `reports/phase-2/package-2.2-report.md` | Created | — | This implementation report |

---

## 3. Acceptance Criteria Verification

| # | Acceptance Criterion (Package 2.2) | Status | Verification Evidence |
| :-: | :--- | :---: | :--- |
| 1 | Downloads only the specified time range (not the full video) | ✅ PASS | `TestVideoSegmentDownloader::test_download_segment_options_and_default_buffer_padding` spies on `yt_dlp.utils.download_range_func(None, [(97.0, 148.0)])`, invokes the actual `download_ranges({}, None)` callback to assert `[{'start_time': 97.0, 'end_time': 148.0}]`, and confirms `force_keyframes_at_cuts: True` and `noplaylist: True` |
| 2 | Output file contains merged video + audio in MP4 format | ✅ PASS | `TestVideoSegmentDownloader::test_download_segment_options_and_default_buffer_padding` verifies `format == "bestvideo[height<=1080]+bestaudio/best[height<=1080]"`, `merge_output_format == "mp4"`, `outtmpl == "{output_dir}/{video_id}_{int(start)}_{int(end)}.%(ext)s"`, and returned path `{output_dir}/{video_id}_{int(start)}_{int(end)}.mp4` |
| 3 | Buffer padding works (±3 seconds, clamped at 0.0s start) | ✅ PASS | `test_download_segment_options_and_default_buffer_padding` (`100.0..145.0` → `97.0..148.0`), `test_download_segment_clamps_actual_start_at_zero` (`1.5..12.0` with `buffer_seconds=3.0` → `0.0..15.0` while filename remains `{video_id}_1_12.mp4`), and `test_download_segment_custom_buffer_seconds` (`0.0`, `5.0`, and `25.0` buffers) all pass |
| 4 | Multiple segments download reliably with configurable `buffer_seconds` | ✅ PASS | `test_download_multiple_segments_reuses_video_id_and_skips_cached` (3 beats with `buffer_seconds=4.0`, single `extract_info` probe, `0.0` clamping on beat 1, cached beat 2 skipped, uncached beat 3 downloaded), `test_download_multiple_segments_empty_list_skips_yt_dlp`, and `test_download_multiple_segments_validates_all_segments_before_network` all pass |

---

## 4. Test & Verification Output

### 4.1 Verbatim `pytest tests/test_download.py -v`

```text
D:\GTA6-Clipper\.venv\Lib\site-packages\pytest_asyncio\plugin.py:306: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to "function" scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session". Set it to "fixture-scope" to keep the current behavior.

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 41 items

tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_maps_fields_accurately PASSED [  2%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_preserves_caller_short_url PASSED [  4%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields[metadata0-ChannelName] PASSED [  7%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields[metadata1-CreatorName] PASSED [  9%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields[metadata2-Unknown] PASSED [ 12%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_detects_platform[Youtube-https://www.youtube.com/watch?v=abc-youtube] PASSED [ 14%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_detects_platform[TwitchVod-https://www.twitch.tv/videos/12345-twitch] PASSED [ 17%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_detects_platform[Kick-https://kick.com/streamer/videos/12345-kick] PASSED [ 19%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_raises_when_info_is_none PASSED [ 21%]
tests/test_download.py::TestDownloadAudio::test_download_audio_configures_16khz_mono_wav_and_returns_result PASSED [ 24%]
tests/test_download.py::TestDownloadAudio::test_download_audio_skips_download_when_cached_wav_exists PASSED [ 26%]
tests/test_download.py::TestDownloadAudio::test_download_audio_forwards_progress_callback PASSED [ 29%]
tests/test_download.py::TestDownloadAudio::test_download_audio_raises_when_video_id_missing PASSED [ 31%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: Private video. Sign in if you've been granted access-private] PASSED [ 34%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: Video unavailable. This video has been removed-deleted or unavailable] PASSED [ 36%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate-age-restricted] PASSED [ 39%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: The uploader has not made this video available in your country-region-locked] PASSED [ 41%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: 'not-a-url' is not a valid URL.-Invalid or unsupported stream URL] PASSED [ 43%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: Unsupported URL: https://example.invalid/stream-Invalid or unsupported stream URL] PASSED [ 46%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: Unexpected network timeout-Failed to download audio] PASSED [ 48%]
tests/test_download.py::TestDownloadErrorHandling::test_download_audio_translates_download_error_during_download_phase PASSED [ 51%]
tests/test_download.py::TestDownloadErrorHandling::test_download_audio_translates_postprocessing_error PASSED [ 53%]
tests/test_download.py::TestDownloadErrorHandling::test_empty_url_raises_descriptive_error[] PASSED [ 56%]
tests/test_download.py::TestDownloadErrorHandling::test_empty_url_raises_descriptive_error[   ] PASSED [ 58%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_options_and_default_buffer_padding PASSED [ 60%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_clamps_actual_start_at_zero PASSED [ 63%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_custom_buffer_seconds[0.0-20.0-50.0] PASSED [ 65%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_custom_buffer_seconds[5.0-15.0-55.0] PASSED [ 68%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_custom_buffer_seconds[25.0-0.0-75.0] PASSED [ 70%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_skips_when_cached_mp4_exists PASSED [ 73%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_multiple_segments_reuses_video_id_and_skips_cached PASSED [ 75%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_multiple_segments_empty_list_skips_yt_dlp PASSED [ 78%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[-10.0-20.0-3.0] PASSED [ 80%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[   -10.0-20.0-3.0] PASSED [ 82%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[https://www.youtube.com/watch?v=dQw4w9WgXcQ--1.0-20.0-3.0] PASSED [ 85%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[https://www.youtube.com/watch?v=dQw4w9WgXcQ-20.0-20.0-3.0] PASSED [ 87%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[https://www.youtube.com/watch?v=dQw4w9WgXcQ-25.0-20.0-3.0] PASSED [ 90%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[https://www.youtube.com/watch?v=dQw4w9WgXcQ-10.0-20.0--0.5] PASSED [ 92%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_multiple_segments_validates_all_segments_before_network PASSED [ 95%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_wraps_yt_dlp_errors PASSED [ 97%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_raises_when_video_id_missing PASSED [100%]

============================= 41 passed in 0.55s ==============================
```

### 4.2 Full Suite Regression Check (`pytest -v`)

- **Result:** `77 passed in 4.44s` (36 Phase 1 tests + 24 Package 2.1 tests + 17 Package 2.2 tests).

---

## 5. Architectural Notes & Edge Cases

1. **Single Metadata Probe for Multi-Beat Story Mode (`download_multiple_segments`)**:
   - Validates all `(start_seconds, end_seconds)` tuples upfront before making any network calls, resolves `video_id` once via `_extract_video_id(url)`, and skips any beat whose `{output_dir}/{video_id}_{int(start)}_{int(end)}.mp4` already exists on disk.
   - Directly supports Phase 4 conditional Multi-Beat Story Mode (`sub_clips`), where each downloaded beat retains predictable `±buffer_seconds` padding for clean trimming prior to `concatenate_videoclips`.
2. **Deterministic Cache Filenames vs. Padded Download Ranges**:
   - Output files are named using the unbuffered integer bounds `{video_id}_{int(start_seconds)}_{int(end_seconds)}.mp4` so callers can predict cached filenames independently of `buffer_seconds`, while `download_range_func(None, [(max(0.0, start - buffer), end + buffer)])` fetches the buffered range with `force_keyframes_at_cuts: True`.
3. **Real `yt-dlp` Callback Verification**:
   - Unit tests invoke the actual `yt_dlp.utils.download_range_func` callback (not a stub) on `yt-dlp 2026.10.26` to confirm the exact `[{'start_time': ..., 'end_time': ...}]` output alongside `format`, `merge_output_format`, `force_keyframes_at_cuts`, and `noplaylist`.
4. **Handoff to Package 2.3**:
   - Branch `phase-2-core-pipeline` is ready for Package 2.3 (`src/streamclipper/analysis/transcriber.py` + `tests/test_transcriber.py`).
