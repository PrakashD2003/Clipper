# Package 5.1 — MCP Server Implementation (MCP SDK v2)

## 1. Metadata

| Property | Value |
| :--- | :--- |
| **Package ID** | 5.1 |
| **Phase** | 5 — MCP Server & Integration |
| **Branch** | `phase-5-mcp-server` |
| **Commit Hash** | `68c2807` (initial), `0097ced` (hardening) |
| **Difficulty** | 🔴 Critical |
| **Model** | Claude Opus 4.6 |
| **Timestamp** | 2026-11-18T21:58:30+05:30 |
| **Commit Message** | `feat: MCP SDK v2 (MCPServer) with signal-filtered transcript chunking` |

## 2. Files Created / Modified

| File | Action | Lines | Summary |
| :--- | :--- | :---: | :--- |
| `src/streamclipper/mcp_server/server.py` | **Created & Hardened** | 1,463 | Implements the MCP SDK v2 `MCPServer("streamclipper")` server (`from mcp.server.mcpserver import MCPServer`) with all 11 `@mcp.tool()` functions (`discover_streams`, `download_stream_audio`, `transcribe_audio`, `analyze_signals`, `get_transcript_chunks`, `submit_highlights`, `render_clip`, `render_all_selected`, `get_pipeline_status`, `generate_post_metadata`, `cleanup_temp`), zero-Ollama `_MCPScorer(Scorer)` signal-filtered candidate chunking & dynamic tri-modal fusion, `StepCache` resolution by `url` OR `StepCache.find_by_audio_path(audio_path)`, per-loop shared GPU lock & per-URL cache locks with reload-before-save merge semantics, incremental `render_all_selected` persistence, output collision avoidance for uncached `render_clip`, glob-injection/path-traversal-safe temp cleanup, and structured JSON error handling (`{"error": "..."}`). |
| `src/streamclipper/mcp_server/__init__.py` | **Modified** | 11 | Replaces the Phase 1 placeholder with the `streamclipper-mcp` entry point exposing `main()` calling `mcp.run()`. |
| `tests/test_mcp_server.py` | **Created & Expanded** | 1,770 | 20 unit/async/integration tests covering `MCPServer` registration and `tool.input_schema` verification for all 11 tools, `main()` delegation to `mcp.run()`, `CallToolResult` invocation via `await mcp.call_tool(...)`, `StepCache.find_by_audio_path()` linkage, signal-filtered chunking (`only_with_signals=True` vs `False`), YPP `submit_highlights` parsing/fusion/cache persistence (including MM:SS timestamps, fenced JSON, explicit `peak_time`, and empty-list status reset), `render_clip` & `render_all_selected` across `auto`/`narrative`/`story`/`raw` + voiceover validation & collision avoidance, parallel `transcribe_audio` + `analyze_signals` cache-merge safety, shared GPU lock serialization, `cleanup_temp` glob-injection & path-traversal protection, mid-batch partial-progress persistence, real MoviePy/FFmpeg/ASS/Narrator end-to-end assembly via MCP, offline & opt-in Ollama `generate_post_metadata`, and structured JSON error handling across all 11 tools. |
| `reports/phase-5/package-5.1-report.md` | **Updated** | — | This report. |

## 3. Acceptance Criteria Verification

| # | Acceptance Criterion | Verified By | Result |
| :--- | :--- | :--- | :--- |
| 1 | `streamclipper-mcp` starts cleanly over stdio via `MCPServer.run()` (`mcp>=2.0.0`) | Live stdio JSON-RPC smoke test against `D:\GTA6-Clipper\.venv\Scripts\streamclipper-mcp.exe` (`initialize` + `notifications/initialized` + `tools/list` + live `tools/call` for `generate_post_metadata` returned exit code `0`, empty stderr, `serverInfo.name == "streamclipper"`, all 11 tools, and offline hashtag output) + `test_mcp_main_invokes_mcp_run` | ✅ PASSED |
| 2 | All 11 tools register with valid JSON schemas (`await mcp.list_tools()`, checking `tool.input_schema`) | `test_mcp_server_registration_and_schemas` (`isinstance(mcp, MCPServer)`, exact 11 tool names, `input_schema["type"] == "object"` and `properties` verified for every tool, including YPP parameters and defaults) | ✅ PASSED |
| 3 | `transcribe_audio`, `analyze_signals`, and `get_transcript_chunks` cleanly resolve `StepCache` by `url` OR `StepCache.find_by_audio_path(audio_path)` | `test_download_transcribe_analyze_via_audio_path_linkage` + `test_parallel_transcribe_and_analyze_preserve_both_cache_updates` | ✅ PASSED |
| 4 | `submit_highlights` and `render_clip` accept YPP transformative fields (`moment_type`, `narration_intro`, `narration_outro`, `context_badge`, `sub_clips`, `mode` in `auto\|narrative\|story\|raw`, `voice_intro_path`, `voice_outro_path`) | `test_submit_highlights_parses_ypp_and_persists_cache`, `test_render_clip_story_narrative_raw_and_voice_overrides`, `test_render_all_selected_renders_and_completes_cache`, `test_render_clip_real_assembly_end_to_end_narrative_and_story`, `test_submit_highlights_empty_resets_status_and_parses_string_timestamps` | ✅ PASSED |
| 5 | `get_transcript_chunks` filters to signal-active timestamped chunks by default (`only_with_signals=True`) | `test_mcp_server_registration_and_schemas` (schema default `True`) + `test_download_transcribe_analyze_via_audio_path_linkage` (`total_chunks=3`, `returned_chunks=2`, `filtered_out_chunks=1` with `only_with_signals=True`; `returned_chunks=3` with `only_with_signals=False`) | ✅ PASSED |
| 6 | `generate_post_metadata` works offline without requiring Ollama by default (`use_ollama=False`) | `test_generate_post_metadata_offline_default_and_ollama_opt_in` + live stdio `tools/call` smoke test | ✅ PASSED |
| 7 | Error handling returns structured JSON error messages instead of raw tracebacks | `test_unexpected_exception_returns_structured_json_error` (all 11 tools), `test_submit_highlights_invalid_json_returns_error`, `test_discover_streams_success_and_error`, `test_edge_cases_validation_and_cache_hydration`, `test_render_clip_nonexistent_voice_path_and_uncached_collision_avoidance`, `test_render_all_selected_persists_partial_progress_on_mid_batch_failure` | ✅ PASSED |

## 4. Test & CLI Output

### Verbatim `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_mcp_server.py -v`

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 20 items

tests/test_mcp_server.py::test_mcp_server_registration_and_schemas PASSED [  5%]
tests/test_mcp_server.py::test_mcp_main_invokes_mcp_run PASSED           [ 10%]
tests/test_mcp_server.py::test_discover_streams_success_and_error PASSED [ 15%]
tests/test_mcp_server.py::test_download_transcribe_analyze_via_audio_path_linkage PASSED [ 20%]
tests/test_mcp_server.py::test_submit_highlights_parses_ypp_and_persists_cache PASSED [ 25%]
tests/test_mcp_server.py::test_submit_highlights_invalid_json_returns_error PASSED [ 30%]
tests/test_mcp_server.py::test_render_clip_story_narrative_raw_and_voice_overrides PASSED [ 35%]
tests/test_mcp_server.py::test_render_all_selected_renders_and_completes_cache PASSED [ 40%]
tests/test_mcp_server.py::test_get_pipeline_status_empty_and_populated PASSED [ 45%]
tests/test_mcp_server.py::test_generate_post_metadata_offline_default_and_ollama_opt_in PASSED [ 50%]
tests/test_mcp_server.py::test_cleanup_temp_stream_and_all_completed_safety PASSED [ 55%]
tests/test_mcp_server.py::test_unexpected_exception_returns_structured_json_error PASSED [ 60%]
tests/test_mcp_server.py::test_edge_cases_validation_and_cache_hydration PASSED [ 65%]
tests/test_mcp_server.py::test_parallel_transcribe_and_analyze_preserve_both_cache_updates PASSED [ 70%]
tests/test_mcp_server.py::test_gpu_lock_serializes_concurrent_transcribe_and_render PASSED [ 75%]
tests/test_mcp_server.py::test_cleanup_temp_blocks_glob_injection_and_path_traversal PASSED [ 80%]
tests/test_mcp_server.py::test_render_all_selected_persists_partial_progress_on_mid_batch_failure PASSED [ 85%]
tests/test_mcp_server.py::test_submit_highlights_empty_resets_status_and_parses_string_timestamps PASSED [ 90%]
tests/test_mcp_server.py::test_render_clip_nonexistent_voice_path_and_uncached_collision_avoidance PASSED [ 95%]
tests/test_mcp_server.py::test_render_clip_real_assembly_end_to_end_narrative_and_story PASSED [100%]

====================== 20 passed, 229 warnings in 21.81s ======================
```

### Full Repository Test Suite (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`)

```text
================ 516 passed, 351 warnings in 65.32s (0:01:05) =================
```

### `streamclipper-mcp` Live Stdio JSON-RPC Handshake & Tool Call Smoke Test

```text
RETURNCODE: 0
STDERR: ''
RESPONSE LINES: 3
INIT SERVER: {'name': 'streamclipper', 'version': ''}
TOOLS COUNT: 11
STDIO CALL RESULT: offline STDIO TEST ['GTA6', 'Tim', 'GTA6Clips', 'GTA6Highlights']
LIVE STDIO CHECK PASSED
```

## 5. Architectural Notes & Edge Cases

- **Zero-Ollama MCP Rendering Path**: Both `render_clip` and `render_all_selected` construct `ClipMetadata` offline via `_build_offline_post_metadata()` and reuse `Pipeline._resolve_base_mode()`, `Pipeline._resolve_highlight_mode()`, and `Pipeline._assemble_highlight()` directly rather than calling `Pipeline._step_assemble()`, which invokes `OllamaScorer.generate_metadata()`. Verified end-to-end against real MoviePy v2, FFmpeg, `AssCaptionBurner`, and `TransformativeNarrator` in `test_render_clip_real_assembly_end_to_end_narrative_and_story`.
- **Concurrency & 6 GB GPU Serialization**: Per-event-loop `WeakKeyDictionary` locks (`_get_gpu_lock()` and `_get_url_lock(cache_key)`) serialize concurrent GPU-heavy operations (`transcribe_audio`, `_render_single_highlight`, and `generate_post_metadata(use_ollama=True)`) and wrap all `StepCache` read-modify-write updates with reload-before-save semantics so parallel `transcribe_audio` + `analyze_signals` tool calls from MCP hosts never clobber each other's cached fields.
- **Hardened `cleanup_temp` Containment**: `_collect_temp_files_for_stream()` rejects unsafe path prefixes (`/`, `\`, `:`, `..`), replaces `Path.glob()` wildcard interpolation with exact `iterdir()` prefix/suffix matching (preventing `**` crashes and `[a-z]*` cross-stream deletion), and enforces `_is_within_directory(p, temp_dir)` on every candidate path.
- **Partial Batch Persistence & Output Collision Avoidance**: `render_all_selected` persists `state.clips` incrementally after each clip (recording `PipelineStatus.FAILED` plus partial clips in both cache and JSON error output if a later clip fails). `render_clip` validates explicit `voice_intro_path`/`voice_outro_path` files exist and allocates the next non-colliding `clip_XX.mp4` index inside `_get_gpu_lock()` so repeated uncached renders never overwrite `clip_01.mp4`.
- **`analyze_signals` Optional `url` Default**: `analyze_signals(url: str = "", audio_path: str = "")` defaults `url` to `""` so MCP clients invoking `analyze_signals` with only `audio_path` pass MCP SDK v2 schema validation and resolve the linked stream state cleanly through `StepCache.find_by_audio_path()`.
