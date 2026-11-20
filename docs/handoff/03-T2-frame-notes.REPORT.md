# Brief 03 (T2): Frame Contact Sheets + Frame Notes Report

## Metadata
- **Branch:** `handoff/03-T2-frame-notes`
- **Base Commit:** `c3ff807 docs(handoff): topic-video contracts and Antigravity briefs 03-07 (T2-T6)`
- **Commit Hash:** `6a899f0999dd07f085a20a585073fcc332c8a894`
- **Timestamp:** 2026-11-20T14:38:00Z

## Pytest Verification Summary
```
88 passed, 217 warnings in 51.39s
```
Command executed:
`pytest -q -p no:cacheprovider tests/test_frames.py tests/test_library.py tests/test_mcp_server.py`
All 88 tests passed with 0 failures.

## Byte-Identity Check
`config.default.yaml` and `src/streamclipper/config.default.yaml` are verified byte-identical via binary comparison (`fc.exe /b`).

## Files Changed
- `config.default.yaml`: Added `library.frames` (sheet_seconds: 90, grid: 3, sheet_width: 1152, proxy_height: 360, note_weight: 0.8) and `library.vision` (model: "qwen2.5vl:3b") configuration keys.
- `src/streamclipper/config.default.yaml`: Byte-identical copy of `config.default.yaml`.
- `src/streamclipper/library/frames.py`: New module defining `ContactSheet` model, `download_proxy_video` (360p yt-dlp format with cookie/auth and proxy.mp4 caching, filtering video-only extensions and ignoring sidecars), and `build_contact_sheets` (3x3 contact sheets, dynamic timestamp badges, static screen rejection reusing content_gate thresholds, and idempotent empty/non-empty disk caching via `sheets.json`).
- `src/streamclipper/library/vision.py`: New module defining `describe_sheet_ollama` and `VisionUnavailable` exception, handling Ollama vision chat with robust parsing of JSON, code fences, empty notes, and 15-word clamps.
- `src/streamclipper/library/index.py`: Added `add_frame_notes` (covered time range replacement + FTS trigger sync), `list_frame_notes`, and updated `search` to query `frame_notes_fts`, clamp hit span to `[t-10, t+10]`, rank with `note_weight` (0.8), and merge with speech hits on overlap with `snippet = the note`.
- `src/streamclipper/library/__init__.py`: Exported `ContactSheet`, `build_contact_sheets`, `download_proxy_video`, `describe_sheet_ollama`, and `VisionUnavailable`.
- `src/streamclipper/mcp_server/server.py`: Added MCP tools `index_frames` (background job), `get_frame_sheets` (returning MCP `Image` content and JSON instruction/metadata with count clamping), and `submit_frame_notes` (duration and length validation with +1s integer truncation margin), and updated `SERVER_INSTRUCTIONS`.
- `src/streamclipper/cli.py`: Added CLI commands `library frames` and `library describe` with disabled model messaging.
- `tests/test_frames.py`: New comprehensive test suite covering proxy video downloading, sidecar metadata isolation, contact sheet building, static skipping with idempotent caching, frame notes add/list/replace, FTS sync, search weighting/merging, Ollama vision parsing, disabled vision model handling, VisionUnavailable errors, MCP validation, and CLI commands.
- `tests/test_mcp_server.py`: Extended `EXPECTED_TOOLS` with `index_frames`, `get_frame_sheets`, and `submit_frame_notes`.
- `docs/handoff/03-T2-frame-notes.REPORT.md`: This report file.

## Key Decisions & Review Fixes
1. **Sidecar Video Isolation in Proxy Download**: yt-dlp produces `.info.json` or `.part` files alongside videos. `download_proxy_video` only considers recognized video extensions (`.webm`, `.mkv`, etc.) when remuxing to `proxy.mp4`, preventing metadata files from being moved into `proxy.mp4`.
2. **Idempotent Caching for Static Videos**: When a video's sheets are skipped as static, `sheets.json` caches `[]`. `build_contact_sheets` checks if `cached_data` is a valid list; if empty, it returns `[]` immediately without repeatedly re-probing or re-extracting frames.
3. **Stale Frame Prevention in ffmpeg Grabs**: `_grab_tile_frame` unlinks `temp_path` prior to invoking ffmpeg so previous sheet frames cannot be misread if ffmpeg encounters an error.
4. **Primary ffprobe Duration Probing**: `_get_video_duration` queries `ffprobe` format and stream durations first, avoiding OpenCV VFR and WebM/MKV duration inaccuracies.
5. **Integer Duration Margin in MCP Validation**: `submit_frame_notes` allows timestamps up to `duration + 1.0` seconds to avoid false rejections from integer truncation of `source.duration_seconds`.
6. **MCP SDK v2 Image Return**: Utilized `mcp.server.mcpserver.Image` helper directly, returning `[Image(path=s.path), ..., json_string]`, which the MCP SDK v2 automatically converts to `ImageContent` and `TextContent` blocks.
7. **FTS Merging & Weighting**: Frame-note hits generate synthetic spans `[max(0, t - 10), min(duration, t + 10)]` with bm25 scaled by `library.frames.note_weight` (0.8). Frame-note hits and speech segment hits are grouped by source and merged when their spans overlap.

## Undone Aspects
None. All acceptance criteria and tests pass cleanly without regressions.
