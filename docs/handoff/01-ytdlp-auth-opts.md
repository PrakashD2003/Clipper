# Brief 01: one shared yt-dlp auth/runtime helper for every download

**Base commit:** `8b80524` (branch `feat/async-render-heatmap`)
**Your branch:** `handoff/01-ytdlp-auth-opts`
**Worktree:** `D:\GTA6-Clipper-wt\handoff-01`

Read and follow `docs/handoff/README.md` first (own worktree, only listed files, no secrets,
one pytest at a time, never push/merge).

```powershell
git -C D:\GTA6-Clipper worktree add D:\GTA6-Clipper-wt\handoff-01 -b handoff/01-ytdlp-auth-opts 8b80524
```

## Why

Commit `479aa18` added YouTube cookie support (`_find_cookie_file`) and a Node.js runtime for
yt-dlp's JavaScript challenge (`_find_js_runtimes`, `js_runtimes` + `remote_components`). It was
pasted into two modules with differences, and two yt-dlp call sites are missing it:

- `download/audio.py` has its own `_find_cookie_file` (no `cookies.txt.txt` fallback) and no
  js_runtimes in `extract_stream_info` / `download_audio`.
- `download/video.py` has the fuller `_find_cookie_file` plus `_find_js_runtimes`, and the
  `js_rts = _find_js_runtimes(); if js_rts: ...` block is duplicated back-to-back in
  `_extract_video_id`.
- `download/chat.py` (`ChatScraper.scrape`, `ydl_opts` around line 73) has neither cookies nor
  js_runtimes, so live-chat download can fail on the same bot checks.

## Files you may change

- `src/streamclipper/download/_ytdlp_opts.py` (new)
- `src/streamclipper/download/audio.py`
- `src/streamclipper/download/video.py`
- `src/streamclipper/download/chat.py`
- `tests/test_ytdlp_opts.py` (new)
- existing download/chat test files only if a fake must accept the new options

## Do

1. Create `download/_ytdlp_opts.py` with:
   - `find_cookie_file() -> str | None`: env `YOUTUBE_COOKIES_FILE` or `YT_COOKIES_FILE` (if the
     file exists), then `./cookies.txt`, `./cookies.txt.txt`, `<repo root>/cookies.txt`,
     `<repo root>/cookies.txt.txt`. Never read or log the file contents; log only "using cookie
     file" at debug level without the path's contents.
   - `find_js_runtimes() -> dict`: `{"node": {"path": <resolved>}}` from `shutil.which("node")`,
     falling back to `C:/Program Files/nodejs/node.exe` only if that file exists; `{}` otherwise.
   - `apply_ytdlp_auth_opts(ydl_opts: dict) -> dict`: sets `cookiefile` when a cookie file is
     found, and `js_runtimes` + `remote_components: ["ejs:github"]` when a runtime is found.
     Never overwrites keys the caller already set. Returns the same dict.
   - Config switches read by callers are NOT needed; keep it a pure helper.
2. Replace both private copies in `audio.py` and `video.py` with calls to
   `apply_ytdlp_auth_opts(...)` at every `yt_dlp.YoutubeDL(...)` construction site
   (`extract_stream_info`, `download_audio`, `_extract_video_id`, `_download_segment_with_id`).
   Delete the duplicated block. Keep the `postprocessor_args` for 16 kHz mono WAV exactly as is
   (AGENTS.md §4 rule 6).
3. Call `apply_ytdlp_auth_opts(ydl_opts)` in `ChatScraper.scrape` before creating the YoutubeDL.
4. Tests (`tests/test_ytdlp_opts.py`, names `test_A01_*`): cookie lookup order with tmp files and
   monkeypatched env/cwd; no cookie → no `cookiefile` key; node found / not found; caller keys
   not overwritten; each of the four download call sites and the chat scraper pass the options
   to `yt_dlp.YoutubeDL` (patch `yt_dlp.YoutubeDL` and inspect the opts).

## Done when

- `pytest -q -p no:cacheprovider tests/test_ytdlp_opts.py tests/test_download.py tests/test_chat.py`
  is 0 failed (use the closest existing download/chat test files if names differ).
- Commit message: `refactor(download): shared yt-dlp cookie/JS-runtime helper for audio, video and chat downloads`
  followed by a blank line and `Co-Authored-By: Antigravity <noreply@google.com>`.
- `docs/handoff/01-ytdlp-auth-opts.REPORT.md` written and committed (format in README.md).
