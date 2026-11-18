# Package 2.1 Implementation Report — Audio Downloader

## 1. Metadata

| Field | Value |
| :--- | :--- |
| **Package ID** | `2.1` — Audio Downloader |
| **Phase** | Phase 2: Core Pipeline — Download & Analysis |
| **Branch** | `phase-2-core-pipeline` (branched from `main` at `f576495`) |
| **Commit Message** | `feat: 16kHz mono audio downloader using yt-dlp` |
| **Timestamp** | 2026-11-18 16:48 IST |
| **Model Used** | Gemini 3.8 Flash (High) |
| **Python Environment** | Python 3.12.10 (`D:\GTA6-Clipper\.venv\Scripts\python.exe`), `yt-dlp` 2026.10.26, `pytest` 9.2.0 |

---

## 2. Files Created / Modified

| File Path | Action | Lines | Summary of Classes / Functions |
| :--- | :---: | :---: | :--- |
| `src/streamclipper/download/audio.py` | Created | 218 | `AudioDownloadError(RuntimeError)`, `_validate_url()`, `_classify_download_error()`, `_detect_platform()`, `_info_dict_to_stream_info()`, `extract_stream_info(url) -> StreamInfo`, and `download_audio(url, output_dir, progress_callback=None) -> tuple[Path, StreamInfo]` configuring `FFmpegExtractAudio` (`preferredcodec="wav"`) + `postprocessor_args: {"extractaudio": ["-ar", "16000", "-ac", "1"]}`, disk cache skip, and progress hook forwarding |
| `tests/test_download.py` | Created | 352 | 24 unit tests (`TestExtractStreamInfo`, `TestDownloadAudio`, `TestDownloadErrorHandling`) mocking `yt_dlp.YoutubeDL` to verify 16kHz mono `postprocessor_args`, `StreamInfo` mapping & fallbacks, caller URL preservation, WAV cache skip, `progress_callback` forwarding, and descriptive `AudioDownloadError` classification |
| `reports/phase-2/package-2.1-report.md` | Created | — | This implementation report |

---

## 3. Acceptance Criteria Verification

| # | Acceptance Criterion (Package 2.1) | Status | Verification Evidence |
| :-: | :--- | :---: | :--- |
| 1 | Downloads 16kHz mono WAV audio from a public YouTube VOD URL | ✅ PASS | `TestDownloadAudio::test_download_audio_configures_16khz_mono_wav_and_returns_result` verifies `yt_dlp.YoutubeDL` receives `format="bestaudio/best"`, `postprocessors=[{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]`, `postprocessor_args={"extractaudio": ["-ar", "16000", "-ac", "1"]}`, `outtmpl="{output_dir}/%(id)s.%(ext)s"`, and `noplaylist=True`, returning `{output_dir}/{video_id}.wav` |
| 2 | Returns accurate `StreamInfo` with title, streamer, duration, view count, and `video_id` | ✅ PASS | `TestExtractStreamInfo::test_extract_stream_info_maps_fields_accurately`, `test_extract_stream_info_preserves_caller_short_url`, `test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields`, and `test_extract_stream_info_detects_platform` verify all `StreamInfo` fields, `uploader`→`channel`→`creator`→`"Unknown"` fallbacks, `None`-safe integer coercion, and platform detection |
| 3 | Skips download if the WAV file already exists on disk | ✅ PASS | `TestDownloadAudio::test_download_audio_skips_download_when_cached_wav_exists` confirms that when `{output_dir}/{video_id}.wav` exists on disk, `download_audio()` skips the download pass (`ydl.download` not called) and returns `(cached_wav, stream_info)` |
| 4 | Raises descriptive errors for invalid or private URLs | ✅ PASS | `TestDownloadErrorHandling` (11 test cases) verifies `AudioDownloadError` messages for private, deleted/unavailable, age-restricted/auth-gated, region-locked, invalid/unsupported URLs, empty/whitespace URLs, missing `video_id`, and `PostProcessingError` |

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
collecting ... collected 24 items

tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_maps_fields_accurately PASSED [  4%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_preserves_caller_short_url PASSED [  8%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields[metadata0-ChannelName] PASSED [ 12%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields[metadata1-CreatorName] PASSED [ 16%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields[metadata2-Unknown] PASSED [ 20%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_detects_platform[Youtube-https://www.youtube.com/watch?v=abc-youtube] PASSED [ 25%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_detects_platform[TwitchVod-https://www.twitch.tv/videos/12345-twitch] PASSED [ 29%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_detects_platform[Kick-https://kick.com/streamer/videos/12345-kick] PASSED [ 33%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_raises_when_info_is_none PASSED [ 37%]
tests/test_download.py::TestDownloadAudio::test_download_audio_configures_16khz_mono_wav_and_returns_result PASSED [ 41%]
tests/test_download.py::TestDownloadAudio::test_download_audio_skips_download_when_cached_wav_exists PASSED [ 45%]
tests/test_download.py::TestDownloadAudio::test_download_audio_forwards_progress_callback PASSED [ 50%]
tests/test_download.py::TestDownloadAudio::test_download_audio_raises_when_video_id_missing PASSED [ 54%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: Private video. Sign in if you've been granted access-private] PASSED [ 58%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: Video unavailable. This video has been removed-deleted or unavailable] PASSED [ 62%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate-age-restricted] PASSED [ 66%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: The uploader has not made this video available in your country-region-locked] PASSED [ 70%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: 'not-a-url' is not a valid URL.-Invalid or unsupported stream URL] PASSED [ 75%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: Unsupported URL: https://example.invalid/stream-Invalid or unsupported stream URL] PASSED [ 79%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: Unexpected network timeout-Failed to download audio] PASSED [ 83%]
tests/test_download.py::TestDownloadErrorHandling::test_download_audio_translates_download_error_during_download_phase PASSED [ 87%]
tests/test_download.py::TestDownloadErrorHandling::test_download_audio_translates_postprocessing_error PASSED [ 91%]
tests/test_download.py::TestDownloadErrorHandling::test_empty_url_raises_descriptive_error[] PASSED [ 95%]
tests/test_download.py::TestDownloadErrorHandling::test_empty_url_raises_descriptive_error[   ] PASSED [100%]

============================= 24 passed in 0.30s ==============================
```

### 4.2 Full Suite Regression Check (`pytest -v --tb=short`)

- **Result:** `60 passed in 3.65s` (all 36 Phase 1 tests in `tests/test_config.py` + `tests/test_cache_cli.py`, plus all 24 Package 2.1 tests in `tests/test_download.py`).

---

## 5. Architectural Notes & Edge Cases

1. **`yt-dlp` 16kHz Mono PCM Configuration**:
   - Uses `format: "bestaudio/best"` with `postprocessors: [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]` and `postprocessor_args: {"extractaudio": ["-ar", "16000", "-ac", "1"]}`.
   - Forces single-video extraction via `"noplaylist": True` so passing a YouTube URL with `&list=...` never accidentally downloads an entire playlist.
   - Resulting 16-bit PCM WAV at 16kHz mono occupies ~460 MB per 4-hour VOD (~32 KB/s) instead of ~2.7 GB for 48kHz stereo, matching WhisperX's native 16kHz sample rate directly.
2. **`StepCache` URL Key Consistency**:
   - `_info_dict_to_stream_info()` preserves the caller-provided `url` (stripped) in `StreamInfo.url` instead of overwriting it with `info["webpage_url"]`, ensuring `StepCache.save_state()` and `StepCache.load_state()` always compute identical SHA-256 keys even when a user supplies short URLs like `https://youtu.be/<id>`.
3. **Metadata-First Cache Skip**:
   - `download_audio()` extracts metadata first via `extract_stream_info()` to obtain `video_id`, then checks `(output_dir / f"{video_id}.wav").is_file()` before instantiating the download `YoutubeDL` context, guaranteeing zero redundant network or FFmpeg work on cached streams.
4. **Error Classification**:
   - Catches `yt_dlp.utils.DownloadError`, `yt_dlp.utils.ExtractorError`, and `yt_dlp.utils.PostProcessingError`, mapping real YouTube error phrasings (including `"The uploader has not made this video available in your country"`, private/members-only, age-restricted/cookie-gated, removed/unavailable, and invalid/unsupported URLs) into `AudioDownloadError(RuntimeError)` while preserving the source URL and original message for debugging.
5. **Handoff to Package 2.2**:
   - Branch `phase-2-core-pipeline` is ready for Package 2.2 (`src/streamclipper/download/video.py` + expanded `tests/test_download.py`).
