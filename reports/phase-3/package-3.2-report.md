# Package 3.2 — Ollama Scorer (Standalone Mode)

## 1. Metadata

| Property | Value |
| :--- | :--- |
| **Package ID** | 3.2 |
| **Phase** | 3 — AI Scoring |
| **Branch** | `phase-3-ai-scoring` |
| **Difficulty** | 🟡 Medium |
| **Model** | Gemini 3.8 (high) |
| **Timestamp** | 2026-11-18T17:28:00+05:30 |
| **Commit Message** | `feat: Ollama-based highlight scorer for standalone mode` |

## 2. Files Created / Modified

| File | Action | Lines | Summary |
| :--- | :--- | :--- | :--- |
| `src/streamclipper/scoring/ollama_scorer.py` | **Created** | 552 | Implemented `OllamaConnectionError` and `OllamaScorer(Scorer)` using resolved `llm.model` from `Config`/`HardwareProfile`, `ollama.Client(host, timeout)`, `_score_chunk()` with 3-attempt JSON retry logic, `_parse_response()` (raw list, wrapped `{"highlights": [...]}`, Qwen3 `<think>` / code-fence stripping, relative offset shifting, YPP fields + `sub_clips`), `generate_narrative()`, `generate_metadata()`, and `check_connection()`. |
| `tests/test_scorer.py` | Modified | 1226 (+501) | Expanded with 19 new unit tests across 5 test classes (`TestOllamaScorerInit`, `TestOllamaScorerResponseParsing`, `TestOllamaScorerScoreChunkAndRetries`, `TestOllamaScorerNarrativeAndMetadata`, `TestOllamaScorerCheckConnection`) mocking `ollama.Client.generate` and `ollama.Client.list`. |
| `reports/phase-3/package-3.2-report.md` | **Created** | — | This package completion report. |

## 3. Acceptance Criteria Verification

| # | Criterion | Test(s) | Result |
| :--- | :--- | :--- | :--- |
| 1 | Connects to Ollama via `ollama.Client(host=self.base_url, timeout=self.timeout)` using resolved `llm.model` | `TestOllamaScorerInit::test_uses_hardware_resolved_llm_model_and_client_args[cuda_mid_tensor-qwen3:8b]`, `TestOllamaScorerInit::test_uses_hardware_resolved_llm_model_and_client_args[cuda_mid_no_tensor-qwen3:4b]`, `TestOllamaScorerInit::test_respects_user_llm_override` | ✅ 3/3 PASSED |
| 2 | Parses JSON responses (both list and wrapped object) including `moment_type`, `narration_intro`, `narration_outro`, `context_badge`, and `sub_clips` | `TestOllamaScorerResponseParsing::test_parses_raw_json_list_with_ypp_fields_and_sub_clips`, `TestOllamaScorerResponseParsing::test_parses_wrapped_highlights_object_with_think_tags_and_fences`, `TestOllamaScorerResponseParsing::test_normalizes_invalid_moment_type_and_clamps_score`, `TestOllamaScorerResponseParsing::test_shifts_relative_timestamps_when_chunk_offset_positive`, `TestOllamaScorerResponseParsing::test_empty_responses_and_invalid_ranges` | ✅ 5/5 PASSED |
| 3 | Handles malformed JSON with up to 3 retries | `TestOllamaScorerScoreChunkAndRetries::test_retries_on_malformed_json_and_recovers`, `TestOllamaScorerScoreChunkAndRetries::test_returns_empty_after_three_malformed_attempts`, `TestOllamaScorerNarrativeAndMetadata::test_generate_narrative_uses_moment_type_and_retries` | ✅ 3/3 PASSED |
| 4 | `check_connection()` verifies Ollama server and model availability | `TestOllamaScorerCheckConnection::test_check_connection_succeeds_with_ollama_list_response`, `TestOllamaScorerCheckConnection::test_check_connection_succeeds_with_dict_response_and_latest_tag`, `TestOllamaScorerCheckConnection::test_check_connection_raises_when_model_missing`, `TestOllamaScorerCheckConnection::test_check_connection_raises_when_server_unreachable` | ✅ 4/4 PASSED |
| 5 | Generates tone-matched narrative scripts (`generate_narrative`), hook titles, and captions using prompt templates | `TestOllamaScorerNarrativeAndMetadata::test_generate_narrative_uses_moment_type_and_retries`, `TestOllamaScorerNarrativeAndMetadata::test_generate_metadata_backfills_missing_narrative_fields`, `TestOllamaScorerNarrativeAndMetadata::test_generate_metadata_preserves_existing_narrative_without_extra_call` | ✅ 3/3 PASSED |
| 6 | Raises descriptive error including the resolved `self.model` name if Ollama is unreachable or model is missing | `TestOllamaScorerCheckConnection::test_check_connection_raises_when_model_missing`, `TestOllamaScorerCheckConnection::test_check_connection_raises_when_server_unreachable`, `TestOllamaScorerScoreChunkAndRetries::test_connection_error_during_generate_propagates_with_model_hint` | ✅ 3/3 PASSED |

**Scorer Test Suite:** `68/68 PASSED` (`49` Phase 3.1 + `19` Phase 3.2)  
**Full Repository Test Suite:** `192/192 PASSED` (`pytest -v` in `18.16s`)

## 4. Verbatim `pytest tests/test_scorer.py -v` Output

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 68 items

tests/test_scorer.py::TestModelBackwardCompat::test_highlight_from_dict_without_ypp_fields PASSED [  1%]
tests/test_scorer.py::TestModelBackwardCompat::test_highlight_from_dict_with_ypp_fields PASSED [  2%]
tests/test_scorer.py::TestModelBackwardCompat::test_highlight_serialization_round_trip PASSED [  4%]
tests/test_scorer.py::TestModelBackwardCompat::test_clip_metadata_from_dict_without_ypp_fields PASSED [  5%]
tests/test_scorer.py::TestModelBackwardCompat::test_clip_metadata_with_ypp_fields_round_trip PASSED [  7%]
tests/test_scorer.py::TestTranscriptChunking::test_chunk_duration_and_overlap PASSED [  8%]
tests/test_scorer.py::TestTranscriptChunking::test_overlap_between_consecutive_chunks PASSED [ 10%]
tests/test_scorer.py::TestTranscriptChunking::test_custom_chunk_minutes PASSED [ 11%]
tests/test_scorer.py::TestTranscriptChunking::test_empty_transcript_returns_no_chunks PASSED [ 13%]
tests/test_scorer.py::TestTranscriptChunking::test_chunks_contain_text PASSED [ 14%]
tests/test_scorer.py::TestGetCandidateChunks::test_only_with_signals_filters_dead_chunks PASSED [ 16%]
tests/test_scorer.py::TestGetCandidateChunks::test_all_chunks_returned_when_signals_disabled PASSED [ 17%]
tests/test_scorer.py::TestSignalOverlapScore::test_full_overlap_returns_spike_score PASSED [ 19%]
tests/test_scorer.py::TestSignalOverlapScore::test_partial_overlap_proportional PASSED [ 20%]
tests/test_scorer.py::TestSignalOverlapScore::test_no_overlap_returns_zero PASSED [ 22%]
tests/test_scorer.py::TestSignalOverlapScore::test_multiple_spikes_takes_best PASSED [ 23%]
tests/test_scorer.py::TestSignalOverlapScore::test_clamped_to_ten PASSED [ 25%]
tests/test_scorer.py::TestSignalOverlapScore::test_empty_spikes_returns_zero PASSED [ 26%]
tests/test_scorer.py::TestWeightedScoreFusion::test_weighted_fusion_formula PASSED [ 27%]
tests/test_scorer.py::TestWeightedScoreFusion::test_weighted_score_with_no_signals PASSED [ 29%]
tests/test_scorer.py::TestWeightedScoreFusion::test_weighted_score_sets_loudness_and_chat_fields PASSED [ 30%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_peak_time_from_loudness_metadata PASSED [ 32%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_peak_time_fallback_to_midpoint PASSED [ 33%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_peak_time_selects_highest_scored_spike PASSED [ 35%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_chat_burst_extraction PASSED [ 36%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_chat_burst_max_five PASSED [ 38%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_chat_burst_empty_when_no_overlap PASSED [ 39%]
tests/test_scorer.py::TestSubClipsValidation::test_valid_two_beat_sub_clips PASSED [ 41%]
tests/test_scorer.py::TestSubClipsValidation::test_valid_three_beat_sub_clips PASSED [ 42%]
tests/test_scorer.py::TestSubClipsValidation::test_single_beat_returns_empty PASSED [ 44%]
tests/test_scorer.py::TestSubClipsValidation::test_empty_sub_clips_returns_empty PASSED [ 45%]
tests/test_scorer.py::TestSubClipsValidation::test_beat_outside_range_returns_empty PASSED [ 47%]
tests/test_scorer.py::TestSubClipsValidation::test_beat_too_short_returns_empty PASSED [ 48%]
tests/test_scorer.py::TestSubClipsValidation::test_overlapping_beats_return_empty PASSED [ 50%]
tests/test_scorer.py::TestSubClipsValidation::test_total_duration_too_short_returns_empty PASSED [ 51%]
tests/test_scorer.py::TestSubClipsValidation::test_missing_keys_returns_empty PASSED [ 52%]
tests/test_scorer.py::TestSubClipsValidation::test_unordered_beats_return_empty PASSED [ 54%]
tests/test_scorer.py::TestDeduplication::test_non_overlapping_kept PASSED [ 55%]
tests/test_scorer.py::TestDeduplication::test_overlapping_keeps_higher_score PASSED [ 57%]
tests/test_scorer.py::TestDeduplication::test_exact_duplicate_keeps_one PASSED [ 58%]
tests/test_scorer.py::TestDeduplication::test_empty_input PASSED         [ 60%]
tests/test_scorer.py::TestDeduplication::test_low_iou_keeps_both PASSED  [ 61%]
tests/test_scorer.py::TestWordExtraction::test_extract_words_in_range PASSED [ 63%]
tests/test_scorer.py::TestWordExtraction::test_extract_no_words_outside_range PASSED [ 64%]
tests/test_scorer.py::TestScoreAndSelect::test_pipeline_attaches_metadata PASSED [ 66%]
tests/test_scorer.py::TestPromptTemplates::test_highlight_scorer_template PASSED [ 67%]
tests/test_scorer.py::TestPromptTemplates::test_narrative_writer_template PASSED [ 69%]
tests/test_scorer.py::TestPromptTemplates::test_hook_generator_template PASSED [ 70%]
tests/test_scorer.py::TestPromptTemplates::test_caption_writer_template PASSED [ 72%]
tests/test_scorer.py::TestOllamaScorerInit::test_uses_hardware_resolved_llm_model_and_client_args[cuda_mid_tensor-qwen3:8b] PASSED [ 73%]
tests/test_scorer.py::TestOllamaScorerInit::test_uses_hardware_resolved_llm_model_and_client_args[cuda_mid_no_tensor-qwen3:4b] PASSED [ 75%]
tests/test_scorer.py::TestOllamaScorerInit::test_respects_user_llm_override PASSED [ 76%]
tests/test_scorer.py::TestOllamaScorerResponseParsing::test_parses_raw_json_list_with_ypp_fields_and_sub_clips PASSED [ 77%]
tests/test_scorer.py::TestOllamaScorerResponseParsing::test_parses_wrapped_highlights_object_with_think_tags_and_fences PASSED [ 79%]
tests/test_scorer.py::TestOllamaScorerResponseParsing::test_normalizes_invalid_moment_type_and_clamps_score PASSED [ 80%]
tests/test_scorer.py::TestOllamaScorerResponseParsing::test_shifts_relative_timestamps_when_chunk_offset_positive PASSED [ 82%]
tests/test_scorer.py::TestOllamaScorerResponseParsing::test_empty_responses_and_invalid_ranges PASSED [ 83%]
tests/test_scorer.py::TestOllamaScorerScoreChunkAndRetries::test_retries_on_malformed_json_and_recovers PASSED [ 85%]
tests/test_scorer.py::TestOllamaScorerScoreChunkAndRetries::test_returns_empty_after_three_malformed_attempts PASSED [ 86%]
tests/test_scorer.py::TestOllamaScorerScoreChunkAndRetries::test_connection_error_during_generate_propagates_with_model_hint PASSED [ 88%]
tests/test_scorer.py::TestOllamaScorerScoreChunkAndRetries::test_end_to_end_score_and_select_with_ollama PASSED [ 89%]
tests/test_scorer.py::TestOllamaScorerNarrativeAndMetadata::test_generate_narrative_uses_moment_type_and_retries PASSED [ 91%]
tests/test_scorer.py::TestOllamaScorerNarrativeAndMetadata::test_generate_metadata_backfills_missing_narrative_fields PASSED [ 92%]
tests/test_scorer.py::TestOllamaScorerNarrativeAndMetadata::test_generate_metadata_preserves_existing_narrative_without_extra_call PASSED [ 94%]
tests/test_scorer.py::TestOllamaScorerCheckConnection::test_check_connection_succeeds_with_ollama_list_response PASSED [ 95%]
tests/test_scorer.py::TestOllamaScorerCheckConnection::test_check_connection_succeeds_with_dict_response_and_latest_tag PASSED [ 97%]
tests/test_scorer.py::TestOllamaScorerCheckConnection::test_check_connection_raises_when_model_missing PASSED [ 98%]
tests/test_scorer.py::TestOllamaScorerCheckConnection::test_check_connection_raises_when_server_unreachable PASSED [100%]

============================= 68 passed in 5.32s ==============================
```

## 5. Architectural Notes & Edge Cases

- **Qwen3 Reasoning & Fence Resilience**: `_clean_llm_text()` strips `<think>...</think>` blocks and Markdown ` ```json ... ``` ` code fences before JSON decoding so both `qwen3:8b` and `qwen3:4b` outputs parse reliably while malformed JSON still triggers the 3-attempt retry loop.
- **Ollama SDK Compatibility**: `_extract_response_text()` and `_extract_model_names()` support both typed `ollama._types.GenerateResponse` / `ListResponse` Pydantic models (installed in `.venv`) and plain `dict` payloads, with `:latest` tag normalization that never falsely matches `qwen3:8b` against `qwen3:4b`.
- **Relative-Timestamp Safeguard**: When a smaller local model returns chunk-relative timestamps (`0 <= start < end <= chunk_offset`) on chunks after `0.0s`, `_parse_response()` automatically shifts both highlight boundaries and `sub_clips` beats by `+chunk_offset`, while leaving absolute stream timestamps untouched.
- **Selective Narrative Backfill**: `generate_metadata()` only invokes `generate_narrative()` when at least one of `narration_intro`, `narration_outro`, or `context_badge` is missing, avoiding redundant LLM calls when `_score_chunk()` already populated them.
- **Connection vs. Parse Failures**: `OllamaConnectionError` (subclass of `RuntimeError`) is raised immediately with the resolved `self.model` and `ollama pull <model>` guidance without wasting retry attempts, whereas malformed JSON retries up to 3 times and degrades gracefully per chunk.
- **Next Package**: 3.3 — Pipeline Orchestrator (Claude Opus 4.6, 🔴 Critical). Wires `OllamaScorer.check_connection()`, `score_and_select()`, and `generate_metadata()` into `Pipeline`.
