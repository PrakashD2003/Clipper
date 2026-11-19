# Handoff Report — Brief 01: Shared yt-dlp Auth & Runtime Helper

- **Branch:** `handoff/01-ytdlp-auth-opts`
- **Base commit:** `8b80524` (branch `feat/async-render-heatmap`)
- **Worktree:** `D:\GTA6-Clipper-wt\handoff-01`
- **Commit:** `79c862f` (amended to include commit SHA)

## Pytest Summary

```text
tests/test_ytdlp_opts.py .................                               [ 19%]
tests/test_download.py ................................................. [ 75%]
tests/test_chat.py ......................                                [100%]
88 passed in 3.31s
```

## Files Changed

- `src/streamclipper/download/_ytdlp_opts.py`: New shared helper defining `find_cookie_file`, `find_js_runtimes`, and `apply_ytdlp_auth_opts`.
- `src/streamclipper/download/audio.py`: Removed private `_find_cookie_file` copy and wired `apply_ytdlp_auth_opts` into `extract_stream_info` and `download_audio` while keeping 16 kHz mono WAV extraction args intact.
- `src/streamclipper/download/video.py`: Removed private `_find_cookie_file` and `_find_js_runtimes` copies, removed back-to-back duplicated block, and wired `apply_ytdlp_auth_opts` into `_extract_video_id` and `_download_segment_with_id`.
- `src/streamclipper/download/chat.py`: Wired `apply_ytdlp_auth_opts` into `ChatScraper.scrape` before instantiating `yt_dlp.YoutubeDL`.
- `tests/test_ytdlp_opts.py`: 17 comprehensive unit tests (`test_A01_*`) covering cookie search order, debug logging safety, Node.js resolution & fallback, overwrite protection, and `YoutubeDL` argument forwarding for all five download call sites.
- `docs/handoff/01-ytdlp-auth-opts.REPORT.md`: This completion report.

## Decisions Made

1. **Non-overwriting caller keys:** `apply_ytdlp_auth_opts` sets `cookiefile`, `js_runtimes`, and `remote_components` only when missing from `ydl_opts`, respecting caller-specified overrides.
2. **Safe logging:** `find_cookie_file()` logs `"using cookie file"` strictly at `logging.DEBUG` level without logging the file's path or contents, preventing secret leakage in CI/CD and logs.
3. **Pure helper:** Kept `_ytdlp_opts.py` free of external config dependencies so callers invoke it cleanly.
4. **Preserved 16 kHz mono audio:** Retained `postprocessor_args: {"extractaudio": ["-ar", "16000", "-ac", "1"]}` in `download_audio()` per AGENTS.md §4 rule 6.

## Left Undone

None. All 88 tests pass cleanly with 0 failures.
