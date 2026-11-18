# Package 2.4 Implementation Report — Loudness Spike Detector

## 1. Metadata

| Field | Value |
| :--- | :--- |
| **Package ID** | `2.4` — Loudness Spike Detector |
| **Phase** | Phase 2: Core Pipeline — Download & Analysis |
| **Branch** | `phase-2-core-pipeline` (parent commit: `3a8f233`) |
| **Commit Message** | `feat: streamed loudness spike detection using librosa and soundfile` |
| **Timestamp** | 2026-11-18 17:03 IST |
| **Model Used** | Gemini 3.8 Flash (High) |
| **Python Environment** | Python 3.12.10 (`D:\GTA6-Clipper\.venv\Scripts\python.exe`), `librosa` 1.0.1, `soundfile` 0.14.0, `scipy` 1.18.1, `numpy` 2.5.4, `pytest` 9.2.0 |

---

## 2. Files Created / Modified

- Confirmed active branch is `phase-2-core-pipeline` and verified `reports/phase-2/package-2.3-report.md` (Package 2.3 complete at `c2e362b`, `12/12` tests passing).

| File Path | Action | Lines | Summary of Classes / Functions |
| :--- | :---: | :---: | :--- |
| `src/streamclipper/analysis/loudness.py` | Created | 189 | `LoudnessDetector(config)` with `detect(audio_path) -> list[TimeRange]`, `_compute_rms_db(audio_path)` (constant-memory `soundfile.info()` + `librosa.stream(..., block_length=120, sr=sr, mono=True, fill_value=0.0)` + `librosa.feature.rms(..., center=False)` + post-EOF zero-pad trimming), `_find_spikes(rms_db, hop_duration)` (`scipy.signal.find_peaks` above `mean + threshold_std * std`, monotonic `0.0–10.0` score normalization, contiguous window expansion, and `metadata["peak_time"]` / `metadata["peak_db"]`), and `_merge_nearby(spikes)` (merges within `merge_gap_seconds` while preserving max score and loudest `peak_time`) |
| `tests/test_loudness.py` | Created | 340 | 13 unit tests across `TestLoudnessDetectorSyntheticAudio` and `TestLoudnessMergeAndEdgeCases` covering synthetic `61.3s` and `121.3s` odd-duration WAVs, monotonic score scaling, nearby burst merging in both amplitude orders, multi-block streaming spy verifying `librosa.stream(..., fill_value=0.0)` (and asserting `librosa.load` is never called), unit merge boundary conditions, silence/constant/ultra-short/stereo/missing-file edge cases, invalid config validation, and `TimeRange` JSON round-trip serialization |
| `reports/phase-2/package-2.4-report.md` | Created | — | This Package 2.4 implementation report |

---

## 3. Acceptance Criteria Verification

| # | Acceptance Criterion (Package 2.4) | Status | Verification Evidence |
| :-: | :--- | :---: | :--- |
| 1 | Detects spikes in audio with screaming/yelling | ✅ PASS | `TestLoudnessDetectorSyntheticAudio::test_detect_finds_two_bursts_on_odd_duration_wav` and `test_detect_scores_scale_monotonically_with_burst_amplitude` inject high-amplitude sine bursts over quiet white noise in `61.3s` and `121.3s` synthetic WAVs and verify exact detection of both bursts with accurate `[start, end]` windows. |
| 2 | Uses `librosa.stream(..., fill_value=0.0)` so memory stays constant and short tail blocks never crash | ✅ PASS | `test_multi_block_streaming_uses_librosa_stream_not_load` spies on `librosa.stream` (confirming `block_length=120`, `fill_value=0.0`, `mono=True`), patches `librosa.load` to raise `AssertionError`, and processes a multi-block `75.3s` WAV with an odd tail block cleanly; `test_ultra_short_audio_does_not_crash` confirms a `0.5s` WAV shorter than `frame_length` returns `[]` without `ParameterError`. |
| 3 | Returns `TimeRange` objects with accurate timestamps, normalized 0–10 scores, and `metadata["peak_time"]` | ✅ PASS | `test_detect_finds_two_bursts_on_odd_duration_wav`, `test_detect_scores_scale_monotonically_with_burst_amplitude`, and `test_detected_spikes_serialize_round_trip` verify `TimeRange` output with `0.0 < score <= 10.0`, `start <= metadata["peak_time"] <= end`, monotonic amplitude-to-score ordering, and clean JSON serialization round-trip. |
| 4 | Merges nearby spikes into single ranges with `SignalType.LOUDNESS` while preserving the highest peak's `peak_time` | ✅ PASS | `test_detect_merges_nearby_bursts_and_keeps_loudest_peak_time` (parametrized for both quiet-then-loud `[0.35, 0.95]` and loud-then-quiet `[0.95, 0.35]` burst pairs) and `test_merge_nearby_unit_preserves_max_score_and_peak_time` (3-spike chain + boundary gap `5.0s` vs `5.1s` + input non-mutation) verify merging retains `SignalType.LOUDNESS`, maximum score, and the loudest spike's `peak_time`. |

---

## 4. Test & Verification Output

### 4.1 Verbatim `pytest tests/test_loudness.py -v`

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
collecting ... collected 13 items

tests/test_loudness.py::TestLoudnessDetectorSyntheticAudio::test_detect_finds_two_bursts_on_odd_duration_wav PASSED [  7%]
tests/test_loudness.py::TestLoudnessDetectorSyntheticAudio::test_detect_scores_scale_monotonically_with_burst_amplitude PASSED [ 15%]
tests/test_loudness.py::TestLoudnessDetectorSyntheticAudio::test_detect_merges_nearby_bursts_and_keeps_loudest_peak_time[0.35-0.95-28.0] PASSED [ 23%]
tests/test_loudness.py::TestLoudnessDetectorSyntheticAudio::test_detect_merges_nearby_bursts_and_keeps_loudest_peak_time[0.95-0.35-20.0] PASSED [ 30%]
tests/test_loudness.py::TestLoudnessDetectorSyntheticAudio::test_multi_block_streaming_uses_librosa_stream_not_load PASSED [ 38%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_merge_nearby_unit_preserves_max_score_and_peak_time PASSED [ 46%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_merge_nearby_empty_returns_empty PASSED [ 53%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_silent_and_constant_audio_return_no_spikes PASSED [ 61%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_ultra_short_audio_does_not_crash PASSED [ 69%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_stereo_wav_downmixes_and_detects_spikes PASSED [ 76%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_missing_file_raises_file_not_found PASSED [ 84%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_invalid_config_values_raise_value_error PASSED [ 92%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_detected_spikes_serialize_round_trip PASSED [100%]

============================= 13 passed in 3.00s ==============================
```

### 4.2 Full Suite Regression Check (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`)

- **Result:** `110 passed in 9.76s` (all Phase 1 and Phase 2 tests passing with zero regressions).

---

## 5. Architectural Notes & Edge Cases

1. **`librosa 1.0.1` `sr=sr` Guardrail**:
   - In `librosa 1.0.1`, `librosa.stream()` marks `sr` with `@future_default(old_default=None, new_default=22050, version="1.1.0")`. Passing `sr=info.samplerate` explicitly eliminates `FutureWarning` output and prevents future silent resampling that would desynchronize `hop_length` from the audio sample rate.
2. **Post-EOF Zero-Padding Trim (`n_real_frames`)**:
   - `librosa.stream(..., block_length=120, fill_value=0.0)` pads the final block all the way to `block_length` frames (240 seconds at `window_seconds=2`). On shorter recordings or partial tail blocks, those post-EOF zero-padded frames produce `-80 dB` RMS values that heavily skew `mean` and `std`. Trimming the concatenated RMS array to `n_real_frames = max(1, math.ceil(info.frames / hop_length))` retains every real frame (including partial tail frames) while excluding pure post-EOF padding.
3. **YPP Spike-Zoom Metadata (`peak_time`) & Merge Stability**:
   - `_find_spikes()` records `metadata["peak_time"] = round(float(peak_idx * hop_duration), 3)` and `metadata["peak_db"]` on every `TimeRange`, and expands `[start, end]` across contiguous above-threshold frames plus a 1-hop context margin.
   - `_merge_nearby()` creates fresh `TimeRange` copies (avoiding in-place mutation), merges clusters within `<= self.merge_gap`, and preserves both the highest score (breaking exact score ties via `peak_db`) and its `peak_time` for Phase 3 (`Scorer._extract_peak_time`) and Phase 4 (`composer.py` `1.20x` punch-in zoom).
4. **Handoff Notes**:
   - `LoudnessDetector` is ready for integration into Phase 3 (`scoring/base.py`) and `Pipeline.process_stream()` (`pipeline.py`).
