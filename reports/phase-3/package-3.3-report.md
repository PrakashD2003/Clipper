# Package 3.3 — Pipeline Orchestrator

## 1. Metadata

| Property | Value |
| :--- | :--- |
| **Package ID** | 3.3 |
| **Phase** | 3 — AI Scoring |
| **Branch** | `phase-3-ai-scoring` |
| **Difficulty** | 🔴 Critical |
| **Model** | Gemini 3.8 Flash (High) *(planned with Claude Opus 4.6)* |
| **Timestamp** | 2026-11-18T17:46:00+05:30 |
| **Commit Message** | `feat: pipeline orchestrator with IntEnum state caching, FAILED retry recovery, and parallel support` + self-audit hardening patch |

## 2. Files Created / Modified

| File | Action | Lines | Summary |
| :--- | :--- | :--- | :--- |
| `src/streamclipper/pipeline.py` | **Created** | 596 | Implemented `format_time()`, `_sanitize_path_segment()`, and `Pipeline(config)` with `_recover_failed_state(state)`, async `process_stream()` (URL normalization/validation, `IntEnum` stage guards including `COMPLETE` guard on Step 6, per-step `StepCache` persistence, `FAILED (99)` capture, `--mode` / `--voice-intro` / `--voice-outro` forwarding), `process_streams_parallel()` (`asyncio.Semaphore` + forced `interactive=False` + `return_exceptions=True` fault isolation), async-safe `asyncio.to_thread()` stage helpers, `Scorer._validate_sub_clips()` validation before multi-beat downloads, Phase 4 dynamic hook (`compose_story_beats`, `beat_trim_ranges`, single-cut buffer-offset math) with Phase 3 fallback, and `_interactive_select()`. |
| `src/streamclipper/cli.py` | Modified | 336 (+129/-9) | Wired `streamclipper clip` and `streamclipper auto` commands to `Pipeline` and `YouTubeDiscovery` via `asyncio.run()`, adding `--mode [narrative\|story\|raw]`, `--voice-intro PATH` (`dir_okay=False`), and `--voice-outro PATH` (`dir_okay=False`) options plus Rich completion/failure summaries. |
| `tests/test_pipeline.py` | **Created** | 1336 | Added 26 unit and mocked end-to-end tests across 6 test classes (`TestFormatTime`, `TestRecoverFailedState`, `TestPipelineEndToEnd`, `TestInteractiveSelection`, `TestParallelProcessing`, `TestCLIPipelineWiring`) covering full `PENDING (1) -> COMPLETE (7)` transitions, resume skipping (`ANALYZING`, `SCORING`, `COMPLETE`), `FAILED (99)` recovery, URL normalization, malformed `sub_clips` single-cut fallback, Phase 4 `compose_story_beats` & buffer-offset integration contract, YPP option forwarding, interactive vs. non-interactive selection, parallel concurrency/fault isolation, and CLI option wiring. |
| `tests/test_cache_cli.py` | Modified | 579 (+35/-1) | Updated the 2 obsolete Phase 1 CLI placeholder tests (`test_clip_command_accepts_all_options` and `test_auto_command_uses_config_and_game_override`) to stub `Pipeline` and `YouTubeDiscovery` instead of asserting the removed `"Coming in Phase 3."` placeholder text. All other 34 Phase 1 tests remain untouched. |
| `reports/phase-3/package-3.3-report.md` | **Created** | — | This package completion report. |

## 3. Acceptance Criteria Verification

| # | Criterion | Test(s) / Verification | Result |
| :--- | :--- | :--- | :--- |
| 1 | Full pipeline runs end-to-end for a single stream | `TestPipelineEndToEnd::test_full_pipeline_transitions_pending_to_complete`, `test_empty_highlights_completes_cleanly`, `test_phase4_assembly_hook_single_and_multi_beat_contract` | ✅ 3/3 PASSED |
| 2 | `IntEnum` status comparisons accurately skip completed steps on resume | `TestPipelineEndToEnd::test_resume_from_cached_analyzing_skips_download_and_transcribe`, `test_resume_from_cached_scoring_skips_download_transcribe_and_analyze`, `test_complete_state_with_empty_selection_skips_reselection`, `test_url_whitespace_normalized_and_empty_url_rejected` | ✅ 4/4 PASSED |
| 3 | Retrying a `FAILED (99)` cached state resets to the last completed artifact stage instead of skipping all steps | `TestRecoverFailedState` (7 tests covering every artifact tier) + `TestPipelineEndToEnd::test_failed_state_saves_99_and_retry_recovers_to_correct_stage` | ✅ 8/8 PASSED |
| 4 | Interactive selection works in CLI when `interactive=True` | `TestInteractiveSelection::test_process_stream_interactive_true_vs_false`, `test_interactive_select_all_and_empty` | ✅ 2/2 PASSED |
| 5 | Parallel processing runs with `interactive=False` so `input()` never blocks the event loop | `TestParallelProcessing::test_parallel_forces_non_interactive_respects_concurrency_and_isolates_errors` (patches `builtins.input` to fail immediately if called) | ✅ 1/1 PASSED |
| 6 | Errors in one stream do not crash others in parallel mode | `TestParallelProcessing::test_parallel_forces_non_interactive_respects_concurrency_and_isolates_errors` (`return_exceptions=True` isolates `bad_stream` while 3 sibling streams succeed) | ✅ 1/1 PASSED |
| 7 | `cli.py` `clip` and `auto` commands wire into `Pipeline` and support `--mode [narrative\|story\|raw]`, `--voice-intro PATH`, and `--voice-outro PATH` | `TestCLIPipelineWiring` (4 tests) + `TestPipelineEndToEnd::test_mode_and_voice_paths_forwarded_to_assembly` + `streamclipper clip --help` and `streamclipper auto --help` | ✅ 5/5 PASSED |
| 8 | `_interactive_select()` displays `moment_type`, proposed `narration_intro`, and beat count before rendering | `TestInteractiveSelection::test_interactive_select_displays_moment_type_narration_intro_and_beats` (verifies `clutch`/`funny`/`graphics_detail`, `2 cuts`/`single`, and `narration_intro` previews in rendered Rich table) | ✅ 1/1 PASSED |

**Pipeline Test Suite:** `26/26 PASSED` (`pytest tests/test_pipeline.py -v` in `5.39s`)  
**Full Repository Test Suite:** `218/218 PASSED` (`pytest -v` in `11.21s`)

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
collecting ... collected 26 items

tests/test_pipeline.py::TestFormatTime::test_formats_hours_minutes_seconds_and_clamps_negative PASSED [  3%]
tests/test_pipeline.py::TestRecoverFailedState::test_non_failed_states_are_unchanged PASSED [  7%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_highlights_resets_to_assembling PASSED [ 11%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_spikes_only_resets_to_scoring PASSED [ 15%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_chat_spikes_only_resets_to_scoring PASSED [ 19%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_transcript_only_resets_to_analyzing PASSED [ 23%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_existing_audio_resets_to_transcribing PASSED [ 26%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_missing_audio_or_no_artifacts_resets_to_pending PASSED [ 30%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_full_pipeline_transitions_pending_to_complete PASSED [ 34%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_resume_from_cached_analyzing_skips_download_and_transcribe PASSED [ 38%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_resume_from_cached_scoring_skips_download_transcribe_and_analyze PASSED [ 42%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_failed_state_saves_99_and_retry_recovers_to_correct_stage PASSED [ 46%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_mode_and_voice_paths_forwarded_to_assembly PASSED [ 50%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_empty_highlights_completes_cleanly PASSED [ 53%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_url_whitespace_normalized_and_empty_url_rejected PASSED [ 57%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_complete_state_with_empty_selection_skips_reselection PASSED [ 61%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_invalid_sub_clips_falls_back_to_single_segment_download PASSED [ 65%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_phase4_assembly_hook_single_and_multi_beat_contract PASSED [ 69%]
tests/test_pipeline.py::TestInteractiveSelection::test_interactive_select_displays_moment_type_narration_intro_and_beats PASSED [ 73%]
tests/test_pipeline.py::TestInteractiveSelection::test_interactive_select_all_and_empty PASSED [ 76%]
tests/test_pipeline.py::TestInteractiveSelection::test_process_stream_interactive_true_vs_false PASSED [ 80%]
tests/test_pipeline.py::TestParallelProcessing::test_parallel_forces_non_interactive_respects_concurrency_and_isolates_errors PASSED [ 84%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_clip_command_forwards_mode_and_voice_options PASSED [ 88%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_clip_command_rejects_invalid_mode PASSED [ 92%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_clip_command_rejects_directory_for_voice_intro PASSED [ 96%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_auto_command_discovers_and_invokes_parallel_pipeline PASSED [100%]

============================= 26 passed in 5.39s ==============================
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
collecting ... collected 218 items
...
tests/test_transcriber.py::TestTranscriptCaching::test_load_transcript_returns_none_for_missing_or_corrupt_file PASSED [100%]

============================ 218 passed in 11.21s =============================
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
  --voice-intro FILE              Path to custom human intro voiceover WAV
  --voice-outro FILE              Path to custom human outro voiceover WAV
  --help                          Show this message and exit.

Usage: streamclipper auto [OPTIONS]

  Auto-discover top streams and clip them all.

  Example: streamclipper auto --game "GTA 6" --max-streams 5 --mode narrative

Options:
  -g, --game TEXT               Game to search for
  -n, --max-streams INTEGER     Number of streams
  --max-clips INTEGER           Clips per stream
  --mode [narrative|story|raw]  YPP transformation mode (overrides config)
  --voice-intro FILE            Path to custom human intro voiceover WAV
  --voice-outro FILE            Path to custom human outro voiceover WAV
  --help                        Show this message and exit.
```

## 5. Architectural Notes, Self-Audit Fixes & Phase 3 Review Readiness

- **True Async Non-Blocking Stage Execution**: All synchronous I/O and compute calls execute inside `asyncio.to_thread(...)`, ensuring `process_streams_parallel()` runs concurrently without stalling the event loop.
- **`FAILED (99)` Recovery Invariant**: Calling `_recover_failed_state(state)` before the `< PipelineStatus.*` guards guarantees a crashed stream resumes from the exact last persisted artifact (`ASSEMBLING`, `SCORING`, `ANALYZING`, `TRANSCRIBING`, or `PENDING`) rather than skipping all stages.
- **Self-Audit Hardening Applied**:
  1. Aligned Phase 4 assembly hook to `VideoComposer.compose_story_beats(segment_paths, facecam, beat_trim_ranges=..., peak_offset=...)` with exact per-beat trim windows and single-cut `actual_start = max(0.0, highlight.start - buffer_seconds)` offset math so captions and spike zoom never drift by `3.0s`.
  2. Validated `highlight.sub_clips` via `Scorer._validate_sub_clips()` before calling `download_multiple_segments()`, falling back safely to single-cut on invalid/overlapping beats.
  3. Guarded Step 6 highlight selection with `state.status < PipelineStatus.COMPLETE` so resuming an already-`COMPLETE` stream with empty `selected_highlights` never re-prompts or corrupts cached state.
  4. Normalized and validated `url` at the entry of `process_stream()` (and preserved canonical `state.stream_info.url` across `_step_download_audio`) so `StepCache` keys never drift on whitespace.
  5. Enforced `dir_okay=False` on `--voice-intro` and `--voice-outro` in `cli.py`.
- **Phase 3 Review Readiness (`3.R`)**: All three Phase 3 packages (`3.1` Scorer Interface + YPP Prompts, `3.2` Ollama Scorer, `3.3` Pipeline Orchestrator + CLI Wiring) are implemented and hardened on `phase-3-ai-scoring` with `218/218` tests passing across Phases 1–3 (`68` scorer tests + `26` pipeline tests + `124` Phase 1/2 tests) and zero regressions. Ready for the Phase 3 Claude Opus 4.6 Lead Review and merge into `main`.
