# Package 4.3 — Hormozi-Style Captions & Live Chat Burst Ticker (ASS + FFmpeg `libass` / Optional NVENC)

## 1. Metadata

| Property | Value |
| :--- | :--- |
| **Package ID** | 4.3 |
| **Phase** | 4 — Video Assembly & Export |
| **Branch** | `phase-4-video-assembly` |
| **Difficulty** | 🔴 Critical |
| **Model** | Claude Opus 4.6 |
| **Timestamp** | 2026-11-18T20:11:08+05:30 |
| **Commit Message** | `feat: Hormozi ASS captions and live chat burst ticker via FFmpeg libass with NVENC support` |

## 2. Files Created / Modified

| File | Action | Lines | Summary |
| :--- | :--- | :---: | :--- |
| `src/streamclipper/assembly/captions.py` | **Created** | 396 | Implements `_sec_to_ass_time()`, `_hex_to_ass_bgr()`, `_color_to_ass_style()`, `_escape_ass_text()`, `_escape_ffmpeg_ass_filename()`, and `AssCaptionBurner` (`generate_ass`, `_build_ffmpeg_cmd`, `burn_captions`) supporting `hormozi` / `single_word` / `sentence` caption styles, `Style: ChatTicker` live viewer burst overlay (`BorderStyle=3`, `OutlineColour=&H99000000`, up to 5 messages accepting both `"message"`/`"text"` and `"offset_seconds"`/`"timestamp"`/`"time"`/`"offset"` keys), defensive config coercion, Windows-safe `cwd` FFmpeg `libass` invocation with two-level filtergraph filename escaping, same-path overwrite guard, and `h264_nvenc` with automatic single-retry fallback to `libx264`. |
| `tests/test_assembly.py` | **Created** | 802 | 67 tests across 6 test classes covering centisecond rollover-safe timestamp formatting, hex/named color conversion (including double-hash rejection), ASS delimiter/control-code escaping, all 3 caption styles, ChatTicker capping/truncation/toggle/`"text"` & timestamp key variations, null/invalid config safety, Package 4.2 `remap_timeline_for_beats` integration, Windows-safe `cwd` FFmpeg invocation, filtergraph escaping, same-path guard, `h264_nvenc` flags & `libx264` fallback, and real FFmpeg `libass` + special-character filename + live GPU `h264_nvenc` integration tests. |
| `reports/phase-4/package-4.3-report.md` | **Created** | — | This report. |

## 3. Acceptance Criteria Verification

| # | Acceptance Criterion | Verified By | Result |
| :--- | :--- | :--- | :--- |
| 1 | Words appear in groups of 3–5 at a time with automatic line wrapping within 1080px | `TestAssGeneration::test_generate_ass_hormozi_style_and_chat_ticker` (`PlayResX: 1080`, `PlayResY: 1920`, `WrapStyle: 0`, 4-word grouping), `test_custom_colors_font_and_group_size` (3-word grouping), `test_single_word_style`, `test_sentence_style`, plus 1080×1920 visual frame inspection | ✅ PASSED |
| 2 | Currently-spoken word is highlighted using config `captions.highlight_color` converted to ASS BGR (`#FFD700` → `&H00D7FF&`) | `TestColorConversion::test_hex_to_ass_bgr_valid` (`#FFD700` → `&H00D7FF&`), `TestAssGeneration::test_generate_ass_hormozi_style_and_chat_ticker` (`{\c&H00D7FF&}WORD{\c&HFFFFFF&}`), `test_custom_colors_font_and_group_size` (`#FF00AA` → `&HAA00FF&`) | ✅ PASSED |
| 3 | Emits `Style: ChatTicker` dialogue events (up to 5 viewer messages from `chat_burst`) when `transformation.show_chat_ticker` is `True` | `TestAssGeneration::test_generate_ass_hormozi_style_and_chat_ticker` (`OutlineColour=&H99000000`, `BorderStyle=3`, Layer 1), `test_chat_ticker_caps_at_five_and_truncates_and_skips_blanks` (8 entries → 5 emitted), `test_chat_ticker_accepts_text_key_and_timestamp_variations`, `test_chat_ticker_disabled_by_config` | ✅ PASSED |
| 4 | ASS delimiters (`{`, `}`, `\`) in transcript words and chat messages are escaped safely | `TestEscapeAssText::*` (4 tests: `\r`/`\n`, `{`/`}`, `\N`/`\n`/`\h` neutralization without doubling literal `\`), `TestAssGeneration::test_word_escaping_and_edge_cases` (trailing `\` guard), `TestRealFfmpegLibassIntegration::test_end_to_end_real_ffmpeg_ass_burn_in`, plus pixel-level libass injection verification (0 red pixels on `{\c&H0000FF&}HACK`) | ✅ PASSED |
| 5 | Timing aligns with WhisperX word timestamps relative to `clip_start_offset` | `TestSecToAssTime::*` (15 parametrized cases including `59.999` → `0:01:00.00` and `3599.999` → `1:00:00.00`), `TestAssGeneration::test_generate_ass_hormozi_style_and_chat_ticker`, `test_integrates_with_composer_remap_timeline_for_beats` | ✅ PASSED |
| 6 | FFmpeg runs with `cwd=ass_path.parent` and `ass=<basename>` (Windows-safe) while copying AAC audio (`-c:a copy`) | `TestFfmpegBurnCaptions::test_burn_captions_uses_windows_safe_cwd_and_relative_ass_name`, `test_build_ffmpeg_cmd_normalizes_codec_case_and_escapes_special_ass_name`, `TestRealFfmpegLibassIntegration::test_end_to_end_real_ffmpeg_ass_burn_in`, `test_real_ffmpeg_handles_special_filtergraph_chars_in_ass_filename` | ✅ PASSED |
| 7 | Supports `h264_nvenc` (`-preset p4 -cq <crf>`) when resolved/configured and automatically falls back to `libx264` (`-crf <crf>`) if NVENC fails | `TestFfmpegBurnCaptions::test_build_ffmpeg_cmd_nvenc`, `test_burn_captions_nvenc_falls_back_to_libx264_on_failure`, `test_burn_captions_libx264_error_reraises_without_retry`, `test_burn_captions_codec_override_argument`, `TestRealFfmpegLibassIntegration::test_real_ffmpeg_nvenc_burn_in_when_gpu_available` (live GTX 1660 SUPER NVENC pass) | ✅ PASSED |

## 4. Verbatim Test Output

### `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_assembly.py -v`

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 67 items

tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[0.0-0:00:00.00] PASSED [  1%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[1.25-0:00:01.25] PASSED [  2%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[9.99-0:00:09.99] PASSED [  4%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[59.99-0:00:59.99] PASSED [  5%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[59.995-0:01:00.00] PASSED [  7%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[59.999-0:01:00.00] PASSED [  8%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[60.0-0:01:00.00] PASSED [ 10%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[65.5-0:01:05.50] PASSED [ 11%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[3599.999-1:00:00.00] PASSED [ 13%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[3661.07-1:01:01.07] PASSED [ 14%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[-5.0-0:00:00.00] PASSED [ 16%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[-0.001-0:00:00.00] PASSED [ 17%]
tests/test_assembly.py::TestSecToAssTime::test_non_finite_values_clamp_to_zero[nan] PASSED [ 19%]
tests/test_assembly.py::TestSecToAssTime::test_non_finite_values_clamp_to_zero[inf] PASSED [ 20%]
tests/test_assembly.py::TestSecToAssTime::test_non_finite_values_clamp_to_zero[-inf] PASSED [ 22%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[#FFD700-&H00D7FF&] PASSED [ 23%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[#ff0000-&H0000FF&] PASSED [ 25%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[#00FF00-&H00FF00&] PASSED [ 26%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[#0000FF-&HFF0000&] PASSED [ 28%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[1a2b3c-&H3C2B1A&] PASSED [ 29%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[  #ABCDEF  -&HEFCDAB&] PASSED [ 31%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[] PASSED [ 32%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[   ] PASSED [ 34%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[#FFF] PASSED [ 35%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[#12345] PASSED [ 37%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[#1234567] PASSED [ 38%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[##FFD700] PASSED [ 40%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[#GGGGGG] PASSED [ 41%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[-12345] PASSED [ 43%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[+12345] PASSED [ 44%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[12_345] PASSED [ 46%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[gold] PASSED [ 47%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[white-&H00FFFFFF] PASSED [ 49%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[  WHITE -&H00FFFFFF] PASSED [ 50%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[black-&H00000000] PASSED [ 52%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[yellow-&H0000FFFF] PASSED [ 53%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[gold-&H0000D7FF] PASSED [ 55%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[red-&H000000FF] PASSED [ 56%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[cyan-&H00FFFF00] PASSED [ 58%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[#FFD700-&H0000D7FF] PASSED [ 59%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[#112233-&H00332211] PASSED [ 61%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_unknown_returns_default PASSED [ 62%]
tests/test_assembly.py::TestEscapeAssText::test_replaces_all_newline_variants_with_spaces PASSED [ 64%]
tests/test_assembly.py::TestEscapeAssText::test_escapes_braces PASSED    [ 65%]
tests/test_assembly.py::TestEscapeAssText::test_neutralizes_ass_control_codes_without_doubling_literal_backslashes PASSED [ 67%]
tests/test_assembly.py::TestEscapeAssText::test_empty_string_returns_empty PASSED [ 68%]
tests/test_assembly.py::TestAssGeneration::test_generate_ass_hormozi_style_and_chat_ticker PASSED [ 70%]
tests/test_assembly.py::TestAssGeneration::test_custom_colors_font_and_group_size PASSED [ 71%]
tests/test_assembly.py::TestAssGeneration::test_single_word_style PASSED [ 73%]
tests/test_assembly.py::TestAssGeneration::test_sentence_style PASSED    [ 74%]
tests/test_assembly.py::TestAssGeneration::test_chat_ticker_caps_at_five_and_truncates_and_skips_blanks PASSED [ 76%]
tests/test_assembly.py::TestAssGeneration::test_chat_ticker_disabled_by_config PASSED [ 77%]
tests/test_assembly.py::TestAssGeneration::test_word_escaping_and_edge_cases PASSED [ 79%]
tests/test_assembly.py::TestAssGeneration::test_chat_ticker_accepts_text_key_and_timestamp_variations PASSED [ 80%]
tests/test_assembly.py::TestAssGeneration::test_defensive_init_handles_null_and_invalid_config_values PASSED [ 82%]
tests/test_assembly.py::TestAssGeneration::test_integrates_with_composer_remap_timeline_for_beats PASSED [ 83%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_build_ffmpeg_cmd_libx264 PASSED [ 85%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_build_ffmpeg_cmd_nvenc PASSED [ 86%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_burn_captions_uses_windows_safe_cwd_and_relative_ass_name PASSED [ 88%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_burn_captions_nvenc_falls_back_to_libx264_on_failure PASSED [ 89%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_burn_captions_libx264_error_reraises_without_retry PASSED [ 91%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_burn_captions_codec_override_argument PASSED [ 92%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_build_ffmpeg_cmd_normalizes_codec_case_and_escapes_special_ass_name PASSED [ 94%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_burn_captions_rejects_same_input_and_output_path PASSED [ 95%]
tests/test_assembly.py::TestRealFfmpegLibassIntegration::test_end_to_end_real_ffmpeg_ass_burn_in PASSED [ 97%]
tests/test_assembly.py::TestRealFfmpegLibassIntegration::test_real_ffmpeg_handles_special_filtergraph_chars_in_ass_filename PASSED [ 98%]
tests/test_assembly.py::TestRealFfmpegLibassIntegration::test_real_ffmpeg_nvenc_burn_in_when_gpu_available PASSED [100%]

============================= 67 passed in 3.36s ==============================
```

Full repository regression check (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`): **458/458 PASSED** (57.93s, zero `PYTHONPATH` override).

## 5. Architectural Notes & Edge Cases

- **Package 4.2 Integration Verified**: Confirmed `reports/phase-4/package-4.2-report.md` is complete on `phase-4-video-assembly`, and verified `composer.remap_timeline_for_beats()` → `AssCaptionBurner.generate_ass(..., clip_start_offset=0.0)` end-to-end (including `"text"` message fallback and `"timestamp"`/`"time"`/`"offset"` key preservation).
- **Hardening & Real-Hardware Verification**:
  - Added two-level FFmpeg filtergraph escaping (`_escape_ffmpeg_ass_filename`) in `_build_ffmpeg_cmd()` so ASS basenames containing commas, quotes, brackets, semicolons, `=`, `:`, `\`, or leading spaces render cleanly without breaking FFmpeg's filtergraph parser, while leaving standard names (`clip_01.ass`) untouched.
  - Aligned `ChatTicker` parsing with Phase 3 `HighlightScorer._extract_chat_burst()` (`"message"` / `"text"`) and Package 4.2 `remap_timeline_for_beats()` (`"offset_seconds"` / `"timestamp"` / `"time"` / `"offset"`).
  - Added defensive coercion (`_safe_int`, `_safe_float`, style validation) in `AssCaptionBurner.__init__` so `None`/invalid `captions.*` config entries fall back cleanly instead of crashing or emitting invalid ASS headers.
  - Tightened `_hex_to_ass_bgr()` to use `.removeprefix("#")` (rejecting malformed `##FFD700`) and strict 6-hex-digit regex matching.
  - Added same-path guard (`ValueError`) in `burn_captions()` and case-insensitive codec normalization in `_build_ffmpeg_cmd()`.
  - Visually inspected rendered 1080×1920 frames (5 simultaneous stacked `ChatTicker` lines + highlighted `Hormozi` captions with literal `\` and `\N`) and verified live `h264_nvenc` (`-preset p4 -cq 23`) burn-in on the host's NVIDIA GeForce GTX 1660 SUPER with `yuv420p` H.264 video and AAC audio stream copy.
- **`.venv` Editable Install Restored**: Re-installed the editable package against `D:\GTA6-Clipper` via `D:\GTA6-Clipper\.venv\Scripts\pip.exe install --no-deps --no-build-isolation -e .` so `D:\GTA6-Clipper\.venv\Lib\site-packages\__editable__.streamclipper-0.1.0.pth` points directly to `D:\GTA6-Clipper\src`. Both bare `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_assembly.py -v` (67/67) and bare `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` (458/458) pass cleanly with no `PYTHONPATH` workaround.
