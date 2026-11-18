# Package 2.3 Implementation Report — WhisperX Transcriber

## 1. Metadata

| Field | Value |
| :--- | :--- |
| **Package ID** | `2.3` — WhisperX Transcriber |
| **Phase** | Phase 2: Core Pipeline (Audio, Video, Transcription, Signals & Discovery) |
| **Branch** | `phase-2-core-pipeline` (parent commit: `f576495`) |
| **Commit Message** | `feat: WhisperX transcription with hardware auto-tuning and VRAM cleanup` |
| **Timestamp** | 2026-11-18 16:55 IST |
| **Model Used** | Gemini 3.8 Flash (High) |
| **Python Environment** | Python 3.12.10 (`D:\GTA6-Clipper\.venv\Scripts\python.exe`), PyTorch `2.8.0+cu126`, `whisperx` + `faster-whisper` installed, pytest `9.2.0` |

---

## 2. Git Branch & Files Created / Modified

- Verified active branch is `phase-2-core-pipeline` (branched from `main` at `f576495`).
- Note: `reports/phase-2/package-2.2-report.md` was not present yet and untracked files (`src/streamclipper/download/audio.py`, `tests/test_download.py`) existed in the working tree; those files were left untouched and only Package 2.3 files were staged and committed.

| File Path | Action | Lines | Summary of Classes & Functions |
| :--- | :---: | :---: | :--- |
| `src/streamclipper/analysis/transcriber.py` | Created | 288 | `_clear_cuda_cache()` helper and `Transcriber` class (`__init__`, `transcribe`, `_transcribe_whisperx`, `_convert_aligned_segments`, `_transcribe_faster_whisper`, `save_transcript`, `load_transcript`) with hardware-resolved config integration, CPU `int8` safety guard, two-stage GPU VRAM cleanup, `faster-whisper` fallback, and JSON transcript caching |
| `tests/test_transcriber.py` | Created | 573 | 12 unit tests across `TestTranscriberHardwareProfiles`, `TestTranscriberWordConversion`, `TestTranscriberSafetyAndVRAM`, `TestTranscriberFallbackAndErrors`, and `TestTranscriptCaching` |
| `reports/phase-2/package-2.3-report.md` | Created | — | This Package 2.3 implementation report |

---

## 3. Acceptance Criteria Verification

| # | Acceptance Criterion (Package 2.3) | Status | Evidence |
| :-: | :--- | :---: | :--- |
| 1 | Reads resolved `HardwareProfile` settings (`device`, `model_size`, `compute_type`, `batch_size`) from `Config` and logs the active profile | ✅ PASS | Verified by `TestTranscriberHardwareProfiles::test_rtx_4050_profile_passes_resolved_params_and_logs_profile` (asserts `caplog` contains `tier=cuda_mid_tensor`, `model=large-v2`, `compute_type=int8_float16`, `batch_size=8` and checks `whisperx.load_model()`/`model.transcribe()` args) and `test_gtx_1660_super_profile_passes_resolved_params` (`medium`/`int8`/`batch_size=4`). |
| 2 | Transcribes a WAV file and returns `TranscriptionResult` with word-level timestamps (`WordTimestamp`) | ✅ PASS | Verified by `TestTranscriberWordConversion::test_word_level_timestamp_conversion_and_unaligned_tokens`, confirming `TranscriptionResult`, `TranscriptSegment`, `WordTimestamp` (`text`, `start`, `end`, `confidence`), `full_text`, `language`, and monotonic fallback for unaligned wav2vec2 tokens. |
| 3 | Frees GPU VRAM after ASR and alignment (`gc.collect()` + `torch.cuda.empty_cache()`) | ✅ PASS | Verified by `TestTranscriberSafetyAndVRAM::test_two_stage_vram_cleanup_order_and_count_in_whisperx`, proving exact event sequence `load_model → load_audio → transcribe → gc_collect → empty_cache → load_align_model → align → gc_collect → empty_cache` (`empty_cache` called exactly 2 times) and `test_vram_cleanup_skips_empty_cache_when_cuda_unavailable`. |
| 4 | Automatically forces `compute_type="int8"` on CPU | ✅ PASS | Verified by `TestTranscriberSafetyAndVRAM::test_cpu_compute_type_safety_fallback[float16]` and `[int8_float16]`, confirming automatic downgrade to `"int8"` in `Transcriber.__init__` and propagation to `whisperx.load_model()`. |
| 5 | Falls back to `faster-whisper` (`word_timestamps=True`) if WhisperX is not installed | ✅ PASS | Verified by `TestTranscriberFallbackAndErrors::test_faster_whisper_fallback_when_whisperx_missing` (forces `ImportError` on `whisperx`, asserts `WhisperModel(..., device=..., compute_type=...)`, `word_timestamps=True`, `TranscriptionResult` conversion, and VRAM cleanup) and `test_explicit_faster_whisper_engine_bypasses_whisperx`. |
| 6 | Saves and loads transcript cache as JSON | ✅ PASS | Verified by `TestTranscriptCaching::test_save_and_load_transcript_round_trip` (auto-creates parent dirs, validates JSON on disk, asserts exact round-trip equality) and `test_load_transcript_returns_none_for_missing_or_corrupt_file`. |

---

## 4. Verbatim Test Output

### 4.1 `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_transcriber.py -v`

```text
D:\GTA6-Clipper\.venv\Lib\site-packages\pytest_asyncio\plugin.py:306: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture scope. Future versions of pytest-asyncio will default to the loop scope for asynchronous fixtures to "function" scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session". Set it to "fixture-scope" to keep the current behavior.

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 12 items

tests/test_transcriber.py::TestTranscriberHardwareProfiles::test_rtx_4050_profile_passes_resolved_params_and_logs_profile PASSED [  8%]
tests/test_transcriber.py::TestTranscriberHardwareProfiles::test_gtx_1660_super_profile_passes_resolved_params PASSED [ 16%]
tests/test_transcriber.py::TestTranscriberWordConversion::test_word_level_timestamp_conversion_and_unaligned_tokens PASSED [ 25%]
tests/test_transcriber.py::TestTranscriberSafetyAndVRAM::test_cpu_compute_type_safety_fallback[float16] PASSED [ 33%]
tests/test_transcriber.py::TestTranscriberSafetyAndVRAM::test_cpu_compute_type_safety_fallback[int8_float16] PASSED [ 41%]
tests/test_transcriber.py::TestTranscriberSafetyAndVRAM::test_two_stage_vram_cleanup_order_and_count_in_whisperx PASSED [ 50%]
tests/test_transcriber.py::TestTranscriberSafetyAndVRAM::test_vram_cleanup_skips_empty_cache_when_cuda_unavailable PASSED [ 58%]
tests/test_transcriber.py::TestTranscriberFallbackAndErrors::test_faster_whisper_fallback_when_whisperx_missing PASSED [ 66%]
tests/test_transcriber.py::TestTranscriberFallbackAndErrors::test_explicit_faster_whisper_engine_bypasses_whisperx PASSED [ 75%]
tests/test_transcriber.py::TestTranscriberFallbackAndErrors::test_missing_audio_file_raises_file_not_found PASSED [ 83%]
tests/test_transcriber.py::TestTranscriptCaching::test_save_and_load_transcript_round_trip PASSED [ 91%]
tests/test_transcriber.py::TestTranscriptCaching::test_load_transcript_returns_none_for_missing_or_corrupt_file PASSED [100%]

============================= 12 passed in 3.61s ==============================
```

### 4.2 Regression Check (`tests/test_config.py tests/test_cache_cli.py tests/test_transcriber.py -v`)

- **48 passed in 5.76s** (all 36 Phase 1 tests + all 12 Package 2.3 tests passing with zero regressions).

---

## 5. Architectural Notes & Edge Cases

### 5.1 Hardware Profile Integration
- `Transcriber.__init__(config)` reads the already-resolved keys from `Config`: `transcription.engine`, `transcription.device`, `transcription.model_size`, `transcription.compute_type`, `transcription.batch_size`, and `transcription.language`.
- No GPU-specific settings are hardcoded:
  - **RTX 4050 (`cuda_mid_tensor`)** resolves to `device="cuda"`, `model_size="large-v2"`, `compute_type="int8_float16"`, `batch_size=8`.
  - **GTX 1660 SUPER (`cuda_mid_no_tensor`)** resolves to `device="cuda"`, `model_size="medium"`, `compute_type="int8"`, `batch_size=4`.
  - **CPU (`cpu_only` / `cpu_fallback`)** resolves to `device="cpu"`, `model_size="base"`, `compute_type="int8"`, `batch_size=4`.
- Even if a custom/programmatic `Config` supplies `device="cpu"` with `compute_type` set to `"float16"` or `"int8_float16"`, `Transcriber.__init__()` enforces a secondary safety guard that downgrades `compute_type` to `"int8"` and emits a warning.
- Active profile provenance (`tier`, `gpu_name`, `engine`, `device`, `model_size`, `compute_type`, `batch_size`) is logged at `INFO` level on initialization.

### 5.2 Two-Stage VRAM Cleanup Logic
On 6 GB VRAM GPUs (GTX 1660 SUPER and RTX 4050), holding the Whisper ASR model, wav2vec2 alignment model, and subsequent Phase 3 Ollama LLM concurrently causes CUDA out-of-memory errors. `_transcribe_whisperx()` prevents VRAM spikes via strict two-stage deallocation:
1. **Stage 1 (Post-ASR, Pre-Alignment)**: Immediately after `model.transcribe(audio, batch_size=self.batch_size)` finishes, `del model` releases the ASR model reference, `gc.collect()` collects cyclic references, and `torch.cuda.empty_cache()` returns cached GPU blocks to CUDA **before** `whisperx.load_align_model()` allocates wav2vec2.
2. **Stage 2 (Post-Alignment, Pre-Return)**: Immediately after `whisperx.align(...)` completes, `del model_a, audio` releases both the alignment model and the decoded audio buffer, followed by `gc.collect()` and `torch.cuda.empty_cache()` so full GPU VRAM is available for Phase 3 Ollama scoring.
3. **Failure & Fallback Hygiene**: An `except Exception:` block releases any still-allocated model/audio references if an intermediate WhisperX call fails, and `_transcribe_faster_whisper()` consumes the lazy segment iterator inside `try` before deleting `model` and clearing CUDA cache in `finally`.

### 5.3 Edge Cases & Handoff Notes
- **Unaligned wav2vec2 Tokens**: WhisperX alignment occasionally omits `start`, `end`, or `score` for numeric or punctuation tokens (e.g., `"6"`). `_convert_aligned_segments()` handles these gracefully with monotonic neighbor/segment timestamp fallback and `confidence=0.0` instead of raising `KeyError`/`TypeError`.
- **Input Validation & Corrupt Cache Resilience**: `transcribe()` raises `FileNotFoundError` when `audio_path` does not exist, and `load_transcript()` returns `None` (with a warning) on missing or corrupt JSON files.
- **Next Steps**: Ready for Packages 2.4–2.6 (and committing Packages 2.1–2.2 on `phase-2-core-pipeline` when ready).
