# Package 3.3 — Pipeline Orchestrator

## 1. Metadata

| Property | Value |
| :--- | :--- |
| **Package ID** | 3.3 |
| **Phase** | 3 — AI Scoring |
| **Branch** | `phase-3-ai-scoring` |
| **Difficulty** | 🔴 Critical |
| **Model** | Gemini 3.8 Flash (High) *(planned with Claude Opus 4.6)* |
| **Timestamp** | 2026-11-18T17:38:00+05:30 |
| **Commit Message** | `feat: pipeline orchestrator with IntEnum state caching, FAILED retry recovery, and parallel support` |

## 2. Files Created / Modified

| File | Action | Lines | Summary |
| :--- | :--- | :--- | :--- |
| `src/streamclipper/pipeline.py` | **Created** | 553 | Implemented `format_time()`, `_sanitize_path_segment()`, and `Pipeline(config)` with `_recover_failed_state(state)`, async `process_stream()` (`IntEnum` stage guards, per-step `StepCache` persistence, `FAILED (99)` exception capture, `--mode` / `--voice-intro` / `--voice-outro` forwarding), `process_streams_parallel()` (`asyncio.Semaphore` + forced `interactive=False` + `return_exceptions=True` fault isolation), async-safe `asyncio.to_thread()` stage helpers (`_step_download_audio`, `_step_transcribe`, `_step_analyze`, `_step_score`, `_step_assemble` with Phase 4 dynamic hook and Phase 3 fallback), and `_interactive_select()` displaying `moment_type`, `narration_intro`, and beat count. |
| `src/streamclipper/cli.py` | Modified | 336 (+129/-9) | Wired `streamclipper clip` and `streamclipper auto` commands to `Pipeline` and `YouTubeDiscovery` via `asyncio.run()`, adding `--mode [narrative\|story\|raw]`, `--voice-intro PATH`, and `--voice-outro PATH` options plus Rich completion/failure summaries. |
| `tests/test_pipeline.py` | **Created** | 1075 | Added 21 unit and mocked end-to-end tests across 6 test classes (`TestFormatTime`, `TestRecoverFailedState`, `TestPipelineEndToEnd`, `TestInteractiveSelection`, `TestParallelProcessing`, `TestCLIPipelineWiring`) covering full `PENDING (1) -> COMPLETE (7)` transitions, resume skipping from `ANALYZING (4)` / `SCORING (5)`, `FAILED (99)` retry recovery, YPP option forwarding, interactive vs. non-interactive selection, parallel concurrency/fault isolation, and CLI option wiring. |
| `tests/test_cache_cli.py` | Modified | 579 (+35/-1) | Updated the 2 obsolete Phase 1 CLI placeholder tests (`test_clip_command_accepts_all_options` and `test_auto_command_uses_config_and_game_override`) to stub `Pipeline` and `YouTubeDiscovery` instead of asserting the removed `"Coming in Phase 3."` placeholder text. All other 34 Phase 1 tests remain untouched. |
| `reports/phase-3/package-3.3-report.md` | **Created** | — | This package completion report. |

## 3. Acceptance Criteria Verification

| # | Criterion | Test(s) / Verification | Result |
| :--- | :--- | :--- | :--- |
| 1 | Full pipeline runs end-to-end for a single stream | `TestPipelineEndToEnd::test_full_pipeline_transitions_pending_to_complete`, `TestPipelineEndToEnd::test_empty_highlights_completes_cleanly` | ✅ 2/2 PASSED |
| 2 | `IntEnum` status comparisons accurately skip completed steps on resume | `TestPipelineEndToEnd::test_resume_from_cached_analyzing_skips_download_and_transcribe`, `TestPipelineEndToEnd::test_resume_from_cached_scoring_skips_download_transcribe_and_analyze` | ✅ 2/2 PASSED |
| 3 | Retrying a `FAILED (99)` cached state resets to the last completed artifact stage instead of skipping all steps | `TestRecoverFailedState` (7 tests covering every artifact tier) + `TestPipelineEndToEnd::test_failed_state_saves_99_and_retry_recovers_to_correct_stage` | ✅ 8/8 PASSED |
| 4 | Interactive selection works in CLI when `interactive=True` | `TestInteractiveSelection::test_process_stream_interactive_true_vs_false`, `TestInteractiveSelection::test_interactive_select_all_and_empty` | ✅ 2/2 PASSED |
| 5 | Parallel processing runs with `interactive=False` so `input()` never blocks the event loop | `TestParallelProcessing::test_parallel_forces_non_interactive_respects_concurrency_and_isolates_errors` (patches `builtins.input` to fail immediately if called) | ✅ 1/1 PASSED |
| 6 | Errors in one stream do not crash others in parallel mode | `TestParallelProcessing::test_parallel_forces_non_interactive_respects_concurrency_and_isolates_errors` (`return_exceptions=True` isolates `bad_stream` while 3 sibling streams succeed) | ✅ 1/1 PASSED |
| 7 | `cli.py` `clip` and `auto` commands wire into `Pipeline` and support `--mode [narrative\|story\|raw]`, `--voice-intro PATH`, and `--voice-outro PATH` | `TestCLIPipelineWiring::test_clip_command_forwards_mode_and_voice_options`, `TestCLIPipelineWiring::test_clip_command_rejects_invalid_mode`, `TestCLIPipelineWiring::test_auto_command_discovers_and_invokes_parallel_pipeline`, `TestPipelineEndToEnd::test_mode_and_voice_paths_forwarded_to_assembly`, plus `streamclipper clip --help` and `streamclipper auto --help` | ✅ 4/4 PASSED |
| 8 | `_interactive_select()` displays `moment_type`, proposed `narration_intro`, and beat count before rendering | `TestInteractiveSelection::test_interactive_select_displays_moment_type_narration_intro_and_beats` (verifies `clutch`/`funny`/`graphics_detail`, `2 cuts`/`single`, and `narration_intro` previews in rendered Rich table) | ✅ 1/1 PASSED |

**Pipeline Test Suite:** `21/21 PASSED` (`pytest tests/test_pipeline.py -v` in `5.03s`)  
**Full Repository Test Suite:** `213/213 PASSED` (`pytest -v` in `11.43s`)

## 4. Verbatim Test & CLI Output

### 4.1 `pytest tests/test_pipeline.py -v`

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 21 items

tests/test_pipeline.py::TestFormatTime::test_formats_hours_minutes_seconds_and_clamps_negative PASSED [  4%]
tests/test_pipeline.py::TestRecoverFailedState::test_non_failed_states_are_unchanged PASSED [  9%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_highlights_resets_to_assembling PASSED [ 14%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_spikes_only_resets_to_scoring PASSED [ 19%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_chat_spikes_only_resets_to_scoring PASSED [ 23%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_transcript_only_resets_to_analyzing PASSED [ 28%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_existing_audio_resets_to_transcribing PASSED [ 33%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_missing_audio_or_no_artifacts_resets_to_pending PASSED [ 38%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_full_pipeline_transitions_pending_to_complete PASSED [ 42%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_resume_from_cached_analyzing_skips_download_and_transcribe PASSED [ 47%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_resume_from_cached_scoring_skips_download_transcribe_and_analyze PASSED [ 52%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_failed_state_saves_99_and_retry_recovers_to_correct_stage PASSED [ 57%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_mode_and_voice_paths_forwarded_to_assembly PASSED [ 61%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_empty_highlights_completes_cleanly PASSED [ 66%]
tests/test_pipeline.py::TestInteractiveSelection::test_interactive_select_displays_moment_type_narration_intro_and_beats PASSED [ 71%]
tests/test_pipeline.py::TestInteractiveSelection::test_interactive_select_all_and_empty PASSED [ 76%]
tests/test_pipeline.py::TestInteractiveSelection::test_process_stream_interactive_true_vs_false PASSED [ 80%]
tests/test_pipeline.py::TestParallelProcessing::test_parallel_forces_non_interactive_respects_concurrency_and_isolates_errors PASSED [ 85%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_clip_command_forwards_mode_and_voice_options PASSED [ 90%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_clip_command_rejects_invalid_mode PASSED [ 95%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_auto_command_discovers_and_invokes_parallel_pipeline PASSED [100%]

============================= 21 passed in 5.03s ==============================
```

### 4.2 Full Repository Suite (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`)

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 213 items
...
tests/test_transcriber.py::TestTranscriptCaching::test_load_transcript_returns_none_for_missing_or_corrupt_file PASSED [100%]

============================ 213 passed in 11.43s =============================
```

### 4.3 CLI Help Verification (`streamclipper clip --help` & `streamclipper auto --help`)

```text
Usage: streamclipper clip [OPTIONS] URL

  Clip highlights from a stream URL.

  Example: streamclipper clip "https://youtube.com/watch?v=XXXXX" --mode
  narrative

Options:
  -n, --max-clips INTEGER         Max clips to generate
  -g, --game TEXT                 Game name (overrides config)
  -s, --streamer TEXT             Streamer name (auto-detected if omitted)
  --interactive / --no-interactive
                                  Prompt to select highlights
  --mode [narrative|story|raw]    YPP transformation mode (overrides config)
  --voice-intro PATH              Path to custom human intro voiceover WAV
  --voice-outro PATH              Path to custom human outro voiceover WAV
  --help                          Show this message and exit.

Usage: streamclipper auto [OPTIONS]

  Auto-discover top streams and clip them all.

  Example: streamclipper auto --game "GTA 6" --max-streams 5 --mode narrative

Options:
  -g, --game TEXT               Game to search for
  -n, --max-streams INTEGER     Number of streams
  --max-clips INTEGER           Clips per stream
  --mode [narrative|story|raw]  YPP transformation mode (overrides config)
  --voice-intro PATH            Path to custom human intro voiceover WAV
  --voice-outro PATH            Path to custom human outro voiceover WAV
  --help                        Show this message and exit.
```

## 5. Architectural Notes, Edge Cases & Phase 3 Review Readiness

- **True Async Non-Blocking Stage Execution**: All synchronous I/O and compute calls (`download_audio`, `Transcriber.transcribe`, `LoudnessDetector.detect`, `ChatScraper`, `ChatVelocityAnalyzer.analyze`, `OllamaScorer.check_connection`/`score_and_select`/`generate_metadata`, `download_segment`, `download_multiple_segments`, and assembly) execute inside `asyncio.to_thread(...)`, ensuring `process_streams_parallel()` runs concurrently without stalling the event loop.
- **`FAILED (99)` Recovery Invariant**: Because `PipelineStatus.FAILED == 99` is strictly greater than every stage enum (`1..7`), calling `_recover_failed_state(state)` before the `< PipelineStatus.*` guards guarantees a crashed stream resumes from the exact last persisted artifact (`ASSEMBLING`, `SCORING`, `ANALYZING`, `TRANSCRIBING`, or `PENDING`) rather than skipping all stages.
- **Forward-Compatible Phase 4 Assembly Hook**: `_assemble_highlight()` dynamically imports the upcoming Phase 4 modules (`FacecamDetector`, `VideoComposer`, `remap_timeline_for_beats`, `HookTitleOverlay`, `Exporter`, `FileOrganizer`) and wires `mode`, `voice_intro_path`, and `voice_outro_path` directly into `Exporter.export()`. While Phase 4 is not yet built, it gracefully falls back to `_fallback_save_segment()`, writing Windows-safe companion `.json` metadata and `_transcript.txt` files and transitioning cleanly to `PipelineStatus.COMPLETE (7)`.
- **Raw vs. Transformative Multi-Beat Dispatch**: When `effective_mode != "raw"` and a highlight contains 2+ validated `sub_clips`, `_step_assemble()` downloads only the individual beats via `download_multiple_segments()`. When `mode == "raw"` (or `sub_clips` is empty), it downloads a single contiguous cut via `download_segment()`.
- **Phase 3 Review Readiness (`3.R`)**: All three Phase 3 packages (`3.1` Scorer Interface + YPP Prompts, `3.2` Ollama Scorer, `3.3` Pipeline Orchestrator + CLI Wiring) are now implemented on `phase-3-ai-scoring` with `213/213` tests passing across Phases 1–3 (`68` scorer tests + `21` pipeline tests + `124` Phase 1/2 tests) and zero regressions. Ready for the Phase 3 Claude Opus 4.6 Lead Review and merge into `main`.
