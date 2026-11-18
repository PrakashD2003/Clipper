# Package 5.2 — README & Setup Guide

## 1. Metadata

| Property | Value |
| :--- | :--- |
| **Package ID** | 5.2 |
| **Phase** | 5 — MCP Server & Documentation |
| **Branch** | `phase-5-mcp-server` |
| **Base Commit Hash** | `68c2807` (Package 5.1 complete; updates initial Package 5.2 commit `6154c35`) |
| **Difficulty** | 🟡 Medium |
| **Model Used** | Gemini 3.8 (high) |
| **Timestamp** | 2026-11-18T21:47:30+05:30 |
| **Commit Message** | `docs: comprehensive README with setup, YPP transformative mode, hardware auto-tuning, and MCP config` |

---

## 2. Files Created / Modified

| File Path | Action | Line Count | Summary of Changes |
| :--- | :--- | :---: | :--- |
| `README.md` | **Modified** | 604 | Comprehensive end-to-end documentation covering project overview, Mermaid architecture flowchart, Audio-First + Tri-Modal Fusion + YPP Transformative Narrative & Story Engine (`auto` \| `narrative` \| `story` \| `raw`, adaptive `moment_type` commentary, conditional multi-beat `sub_clips`, `1.20x` spike zoom, Hormozi ASS captions + live chat ticker, `:eval=frame` 18% audio ducking), Windows/Linux system requirements, 5-tier `HardwareProfile` auto-configuration table & override guide, Windows/Linux `.venv` installation & cross-shell single-line Quick Start CLI examples (`--mode`, `--force`, `--voice-intro`, `--voice-outro`), Voiceover setup (`Kokoro-82M`, `edge-tts`, optional `ElevenLabs` API / personal voice clone, custom mic WAV/`.m4a`/`.mp3`/`.aac`), standalone Ollama setup (`qwen3:8b` / `qwen3:4b`), Claude Desktop / Antigravity MCP SDK v2 (`streamclipper-mcp`) configuration & finalized 11-tool table matching `src/streamclipper/mcp_server/server.py`, full 78-key `config.yaml` reference across all 11 sections, and Windows/Linux troubleshooting guide. |
| `reports/phase-5/package-5.2-report.md` | **Modified** | 694 | Mandatory Package 5.2 verification and audit report with live `streamclipper-mcp` stdio handshake, CLI smoke checks, automated README validation, and complete verbatim 509-test `pytest -v` output. |

---

## 3. Acceptance Criteria Verification

Mapped directly against **Package 5.2** in `docs/streamclipper-implementation-guide-v2.md` (with Package 5.1 confirmed complete via `reports/phase-5/package-5.1-report.md` at commit `68c2807`):

| # | Acceptance Criterion | Verified By | Result |
| :--- | :--- | :--- | :---: |
| 1 | README covers CLI, MCP, Ollama, YPP Transformative Mode (`--mode`, `--voice-intro`, free TTS + optional ElevenLabs voice clone), and dynamic `HardwareProfile` auto-tuning | Automated Python validation against live source code verified **78/78** `config.default.yaml` leaf keys (including all 11 `transformation.*` keys), **5/5** `HardwareProfile` tiers (`cuda_high_vram`, `cuda_mid_tensor`, `cuda_mid_no_tensor`, `cpu_fallback`, `cpu_only`), **4/4** `--mode` options (`auto`, `narrative`, `story`, `raw`), **13/13** CLI flags, **4/4** YAML config blocks merging cleanly through `streamclipper.config.Config()`, and exact fidelity to `hooks.py`, `captions.py`, `narrator.py`, `youtube.py`, `transcriber.py`, and `pipeline.py`. | ✅ PASSED |
| 2 | Setup instructions work on Windows and Linux | PowerShell (`py -3.12 -m venv .venv`, `winget install Gyan.FFmpeg`, `Set-ExecutionPolicy`, CUDA PyTorch wheel, `pip install -e ".[mcp,tts]"`) and portable Ubuntu 22.04+/24.04/Debian 12 bash (`apt install ffmpeg libsndfile1 libgl1 libglib2.0-0 python3 python3-venv`, `python3 -m venv .venv`, `ollama serve`) commands verified against `pyproject.toml`, with zero PowerShell-breaking backslash line continuations and live Windows `.venv` execution (`streamclipper setup`). | ✅ PASSED |
| 3 | MCP configuration example matches `streamclipper-mcp` entry point | Verified against `pyproject.toml` (`streamclipper-mcp = "streamclipper.mcp_server:main"`), finalized `src/streamclipper/mcp_server/server.py` (commit `68c2807`), `json.loads()` on both JSON blocks (confirming `"env": {}` does not clobber host `ELEVENLABS_API_KEY`), live `await mcp.list_tools()` matching **11/11** documented tools, and live stdio JSON-RPC handshake (`initialize` + `notifications/initialized` + `tools/list` on `D:\GTA6-Clipper\.venv\Scripts\streamclipper-mcp.exe` returning exit code `0`, empty stderr, and all 11 tools). | ✅ PASSED |

---

## 4. Test, MCP & CLI Verification Output

### 4.1 CLI Smoke Checks (`streamclipper --help`, `clip --help`, `auto --help`, `setup`, `status`)

```text
Usage: streamclipper [OPTIONS] COMMAND [ARGS]...

  🎮 StreamClipper — AI-powered stream clipping tool

Options:
  --version          Show the version and exit.
  -c, --config FILE  Path to config.yaml
  -v, --verbose      Verbose output
  --help             Show the message and exit.

Commands:
  auto    Auto-discover top streams and clip them all.
  clip    Clip highlights from a stream URL.
  setup   Check dependencies, detect hardware profile, and report system...
  status  Show status of cached/in-progress pipelines.
---CLIP---
Usage: streamclipper clip [OPTIONS] URL

  Clip highlights from a stream URL.

  Example: streamclipper clip "https://youtube.com/watch?v=XXXXX" --mode auto

Options:
  -n, --max-clips INTEGER         Max clips to generate
  -g, --game TEXT                 Game name (overrides config)
  -s, --streamer TEXT             Streamer name (auto-detected if omitted)
  --interactive / --no-interactive
                                  Prompt to select highlights
  --mode [auto|narrative|story|raw]
                                  YPP transformation mode (overrides config)
  --voice-intro FILE              Path to custom human intro voiceover WAV
  --voice-outro FILE              Path to custom human outro voiceover WAV
  --force                         Ignore cached pipeline state and re-run from
                                  scratch
  --help                          Show the message and exit.
---AUTO---
Usage: streamclipper auto [OPTIONS]

  Auto-discover top streams and clip them all.

  Example: streamclipper auto --game "GTA 6" --max-streams 5 --mode auto

Options:
  -g, --game TEXT                 Game to search for
  -n, --max-streams INTEGER       Number of streams
  --max-clips INTEGER             Clips per stream
  --mode [auto|narrative|story|raw]
                                  YPP transformation mode (overrides config)
  --voice-intro FILE              Path to custom human intro voiceover WAV
  --voice-outro FILE              Path to custom human outro voiceover WAV
  --force                         Ignore cached pipeline state and re-run from
                                  scratch
  --help                          Show the message and exit.
---SETUP---
🔧 Checking system dependencies...
  ✓ yt-dlp
  ✓ FFmpeg
  ✓ WhisperX
  ✓ SoundFile
  ✓ OpenCV
  ✓ MoviePy v2
  ✓ Ollama SDK
  ✓ MCP SDK v2 (MCPServer)
  ✓ GPU: NVIDIA GeForce GTX 1660 SUPER (6.4GB VRAM, CC 7.5)
               🖥️ Resolved Hardware Profile (cuda_mid_no_tensor)               
┌────────────────────────────┬─────────────────────────────────────┬──────────┐
│ Setting                    │ Active Value                        │ Source   │
├────────────────────────────┼─────────────────────────────────────┼──────────┤
│ GPU / VRAM / CC            │ NVIDIA GeForce GTX 1660 SUPER (6.4  │ detected │
│                            │ GB, CC 7.5)                         │          │
│ Tensor Cores / NVENC       │ Tensor=False, NVENC=True            │ detected │
│ transcription.device       │ cuda                                │ auto     │
│ transcription.model_size   │ medium                              │ auto     │
│ transcription.compute_type │ int8                                │ auto     │
│ transcription.batch_size   │ 4                                   │ auto     │
│ llm.model                  │ qwen3:4b                            │ auto     │
│ video.codec                │ h264_nvenc                          │ auto     │
└────────────────────────────┴─────────────────────────────────────┴──────────┘
---STATUS---
No cached pipelines found.
```

### 4.2 Automated `README.md` Coverage, JSON/YAML & Live `MCPServer` Validation

```text
README lines: 604
Config leaf keys: 78/78 (missing=[])
Hardware tiers: 5/5 (missing=[])
Modes: 4/4 (missing=[])
CLI flags: 13/13 (missing=[])
MCP tools (live MCPServer): 11/11 (missing=[])
JSON blocks parsed: 2 (clobbering_env=False)
YAML blocks Config-validated: [(1, 'cuda_mid_no_tensor'), (2, 'cuda_mid_no_tensor'), (3, 'cuda_mid_no_tensor'), (4, 'cuda_mid_no_tensor')]
Trailing backslash continuations: []
Broken Ubuntu 22.04 apt pkg: False
ALL README AUTOMATED CHECKS PASSED!
```

### 4.3 Live `streamclipper-mcp` Stdio JSON-RPC Smoke Test

```text
RETURNCODE: 0
STDERR: ''
INIT SERVER: {'name': 'streamclipper', 'version': ''}
TOOLS COUNT: 11
TOOL NAMES: ['discover_streams', 'download_stream_audio', 'transcribe_audio', 'analyze_signals', 'get_transcript_chunks', 'submit_highlights', 'render_clip', 'render_all_selected', 'get_pipeline_status', 'generate_post_metadata', 'cleanup_temp']
```

### 4.4 Verbatim Full Repository Test Suite (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`)

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 509 items

tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[0.0-0:00:00.00] PASSED [  0%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[1.25-0:00:01.25] PASSED [  0%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[9.99-0:00:09.99] PASSED [  0%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[59.99-0:00:59.99] PASSED [  0%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[59.995-0:01:00.00] PASSED [  0%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[59.999-0:01:00.00] PASSED [  1%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[60.0-0:01:00.00] PASSED [  1%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[65.5-0:01:05.50] PASSED [  1%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[3599.999-1:00:00.00] PASSED [  1%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[3661.07-1:01:01.07] PASSED [  1%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[-5.0-0:00:00.00] PASSED [  2%]
tests/test_assembly.py::TestSecToAssTime::test_formats_and_avoids_sixty_rollover[-0.001-0:00:00.00] PASSED [  2%]
tests/test_assembly.py::TestSecToAssTime::test_non_finite_values_clamp_to_zero[nan] PASSED [  2%]
tests/test_assembly.py::TestSecToAssTime::test_non_finite_values_clamp_to_zero[inf] PASSED [  2%]
tests/test_assembly.py::TestSecToAssTime::test_non_finite_values_clamp_to_zero[-inf] PASSED [  2%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[#FFD700-&H00D7FF&] PASSED [  3%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[#ff0000-&H0000FF&] PASSED [  3%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[#00FF00-&H00FF00&] PASSED [  3%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[#0000FF-&HFF0000&] PASSED [  3%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[1a2b3c-&H3C2B1A&] PASSED [  3%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_valid[  #ABCDEF  -&HEFCDAB&] PASSED [  4%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[] PASSED [  4%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[   ] PASSED [  4%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[#FFF] PASSED [  4%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[#12345] PASSED [  4%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[#1234567] PASSED [  5%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[##FFD700] PASSED [  5%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[#GGGGGG] PASSED [  5%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[-12345] PASSED [  5%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[+12345] PASSED [  5%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[12_345] PASSED [  6%]
tests/test_assembly.py::TestColorConversion::test_hex_to_ass_bgr_invalid_returns_default[gold] PASSED [  6%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[white-&H00FFFFFF] PASSED [  6%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[  WHITE -&H00FFFFFF] PASSED [  6%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[black-&H00000000] PASSED [  6%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[yellow-&H0000FFFF] PASSED [  7%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[gold-&H0000D7FF] PASSED [  7%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[red-&H000000FF] PASSED [  7%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[cyan-&H00FFFF00] PASSED [  7%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[#FFD700-&H0000D7FF] PASSED [  7%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_named_and_hex[#112233-&H00332211] PASSED [  8%]
tests/test_assembly.py::TestColorConversion::test_color_to_ass_style_unknown_returns_default PASSED [  8%]
tests/test_assembly.py::TestEscapeAssText::test_replaces_all_newline_variants_with_spaces PASSED [  8%]
tests/test_assembly.py::TestEscapeAssText::test_escapes_braces PASSED    [  8%]
tests/test_assembly.py::TestEscapeAssText::test_neutralizes_ass_control_codes_without_doubling_literal_backslashes PASSED [  8%]
tests/test_assembly.py::TestEscapeAssText::test_empty_string_returns_empty PASSED [  9%]
tests/test_assembly.py::TestAssGeneration::test_generate_ass_hormozi_style_and_chat_ticker PASSED [  9%]
tests/test_assembly.py::TestAssGeneration::test_custom_colors_font_and_group_size PASSED [  9%]
tests/test_assembly.py::TestAssGeneration::test_single_word_style PASSED [  9%]
tests/test_assembly.py::TestAssGeneration::test_sentence_style PASSED    [  9%]
tests/test_assembly.py::TestAssGeneration::test_chat_ticker_caps_at_five_and_truncates_and_skips_blanks PASSED [ 10%]
tests/test_assembly.py::TestAssGeneration::test_chat_ticker_disabled_by_config PASSED [ 10%]
tests/test_assembly.py::TestAssGeneration::test_word_escaping_and_edge_cases PASSED [ 10%]
tests/test_assembly.py::TestAssGeneration::test_chat_ticker_accepts_text_key_and_timestamp_variations PASSED [ 10%]
tests/test_assembly.py::TestAssGeneration::test_defensive_init_handles_null_and_invalid_config_values PASSED [ 10%]
tests/test_assembly.py::TestAssGeneration::test_integrates_with_composer_remap_timeline_for_beats PASSED [ 11%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_build_ffmpeg_cmd_libx264 PASSED [ 11%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_build_ffmpeg_cmd_nvenc PASSED [ 11%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_burn_captions_uses_windows_safe_cwd_and_relative_ass_name PASSED [ 11%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_burn_captions_nvenc_falls_back_to_libx264_on_failure PASSED [ 11%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_burn_captions_libx264_error_reraises_without_retry PASSED [ 11%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_burn_captions_codec_override_argument PASSED [ 12%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_build_ffmpeg_cmd_normalizes_codec_case_and_escapes_special_ass_name PASSED [ 12%]
tests/test_assembly.py::TestFfmpegBurnCaptions::test_burn_captions_rejects_same_input_and_output_path PASSED [ 12%]
tests/test_assembly.py::TestRealFfmpegLibassIntegration::test_end_to_end_real_ffmpeg_ass_burn_in PASSED [ 12%]
tests/test_assembly.py::TestRealFfmpegLibassIntegration::test_real_ffmpeg_handles_special_filtergraph_chars_in_ass_filename PASSED [ 12%]
tests/test_assembly.py::TestRealFfmpegLibassIntegration::test_real_ffmpeg_nvenc_burn_in_when_gpu_available PASSED [ 13%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[GTA 6-unknown-GTA 6] PASSED [ 13%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[GTA: 6 / "Vice" *City?<>|-unknown-GTA 6 Vice City] PASSED [ 13%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[  ..TimTheTatman..  -unknown-TimTheTatman] PASSED [ 13%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[foo\\bar:baz-unknown-foobarbaz] PASSED [ 13%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[   ...   -UnknownGame-UnknownGame] PASSED [ 14%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[<>:"/\\|?*-fallback_name-fallback_name] PASSED [ 14%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_strips_forbidden_chars_and_dots[-unknown-unknown] PASSED [ 14%]
tests/test_assembly.py::TestFileOrganizer::test_get_clip_path_default_structure_and_sanitization PASSED [ 14%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_and_get_clip_path_guard_windows_reserved_names[CON-CON_] PASSED [ 14%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_and_get_clip_path_guard_windows_reserved_names[nul-nul_] PASSED [ 15%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_and_get_clip_path_guard_windows_reserved_names[AUX-AUX_] PASSED [ 15%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_and_get_clip_path_guard_windows_reserved_names[prn-prn_] PASSED [ 15%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_and_get_clip_path_guard_windows_reserved_names[COM1-COM1_] PASSED [ 15%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_and_get_clip_path_guard_windows_reserved_names[lpt9-lpt9_] PASSED [ 15%]
tests/test_assembly.py::TestFileOrganizer::test_sanitize_name_and_get_clip_path_guard_windows_reserved_names[con.txt-con.txt_] PASSED [ 16%]
tests/test_assembly.py::TestFileOrganizer::test_get_clip_path_custom_templates_and_traversal_protection PASSED [ 16%]
tests/test_assembly.py::TestFileOrganizer::test_save_clip_writes_metadata_json_with_ypp_fields_and_transcript PASSED [ 16%]
tests/test_assembly.py::TestFileOrganizer::test_save_clip_respects_disabled_json_and_transcript_flags PASSED [ 16%]
tests/test_assembly.py::TestExporter::test_export_ultrafast_crf18_narrator_ducking_ass_burn_and_handle_cleanup_order PASSED [ 16%]
tests/test_assembly.py::TestExporter::test_export_closes_all_handles_and_cleans_temp_on_write_error PASSED [ 17%]
tests/test_assembly.py::TestExporter::test_export_raw_mode_skips_narrator_and_chat_ticker PASSED [ 17%]
tests/test_assembly.py::TestExporter::test_export_explicit_mode_overrides_disabled_config_and_restores_narrator PASSED [ 17%]
tests/test_assembly.py::TestExporter::test_export_chat_burst_only_still_burns_ass_when_enabled PASSED [ 17%]
tests/test_assembly.py::TestExporter::test_export_without_captions_finalizes_nvenc_with_libx264_fallback PASSED [ 17%]
tests/test_assembly.py::TestExporter::test_close_clip_handles_deduplicates_nested_sources PASSED [ 18%]
tests/test_assembly.py::TestPackage46RealEndToEndIntegration::test_real_e2e_compose_hook_narrator_export_and_organize PASSED [ 18%]
tests/test_assembly.py::TestPackage46RealEndToEndIntegration::test_real_e2e_pipeline_assemble_highlight_single_and_multi_beat PASSED [ 18%]
tests/test_cache_cli.py::TestStepCache::test_cache_key_and_path_are_deterministic PASSED [ 18%]
tests/test_cache_cli.py::TestStepCache::test_load_state_returns_none_when_missing PASSED [ 18%]
tests/test_cache_cli.py::TestStepCache::test_save_and_load_full_state_with_nested_transcript PASSED [ 19%]
tests/test_cache_cli.py::TestStepCache::test_save_and_load_state_without_transcript PASSED [ 19%]
tests/test_cache_cli.py::TestStepCache::test_list_all_ignores_transcript_and_corrupt_files PASSED [ 19%]
tests/test_cache_cli.py::TestStepCache::test_clear_removes_state_and_transcript_cleanly PASSED [ 19%]
tests/test_cache_cli.py::TestStepCache::test_find_by_audio_path_resolves_full_state_and_analysis_complete PASSED [ 19%]
tests/test_cache_cli.py::TestCLI::test_help_lists_all_commands PASSED    [ 20%]
tests/test_cache_cli.py::TestCLI::test_version_option PASSED             [ 20%]
tests/test_cache_cli.py::TestCLI::test_clip_command_accepts_all_options PASSED [ 20%]
tests/test_cache_cli.py::TestCLI::test_auto_command_uses_config_and_game_override PASSED [ 20%]
tests/test_cache_cli.py::TestCLI::test_setup_renders_dependencies_gpu_and_hardware_profile_with_overrides PASSED [ 20%]
tests/test_cache_cli.py::TestCLI::test_setup_without_cuda_reports_cpu_fallback PASSED [ 21%]
tests/test_cache_cli.py::TestCLI::test_check_dependency_handles_missing_module_and_missing_cmd PASSED [ 21%]
tests/test_cache_cli.py::TestCLI::test_status_empty_and_populated_cache PASSED [ 21%]
tests/test_cache_cli.py::TestCLI::test_missing_config_path_fails_cleanly_and_verbose_sets_debug_logging PASSED [ 21%]
tests/test_chat.py::TestChatScraper::test_scrape_parses_text_paid_emoji_and_skips_negative_offsets PASSED [ 21%]
tests/test_chat.py::TestChatScraper::test_scrape_missing_chat_file_returns_empty_dataframe PASSED [ 22%]
tests/test_chat.py::TestChatScraper::test_scrape_handles_yt_dlp_download_error_gracefully PASSED [ 22%]
tests/test_chat.py::TestChatScraper::test_scrape_empty_url_raises_value_error[] PASSED [ 22%]
tests/test_chat.py::TestChatScraper::test_scrape_empty_url_raises_value_error[   ] PASSED [ 22%]
tests/test_chat.py::TestChatScraper::test_save_and_load_cache_round_trip_preserves_strings_and_na_tokens PASSED [ 22%]
tests/test_chat.py::TestChatScraper::test_load_cache_returns_none_when_missing_and_handles_empty_df PASSED [ 22%]
tests/test_chat.py::TestChatVelocityAnalyzer::test_detects_single_chat_burst_and_attaches_sample_messages PASSED [ 23%]
tests/test_chat.py::TestChatVelocityAnalyzer::test_detects_multiple_bursts_with_relative_scores PASSED [ 23%]
tests/test_chat.py::TestChatVelocityAnalyzer::test_merges_nearby_burst_bins_and_caps_sample_messages_at_five PASSED [ 23%]
tests/test_chat.py::TestChatVelocityAnalyzer::test_detects_boundary_burst_in_first_bin PASSED [ 23%]
tests/test_chat.py::TestChatVelocityAnalyzer::test_skips_when_below_min_messages PASSED [ 23%]
tests/test_chat.py::TestChatVelocityAnalyzer::test_skips_empty_uniform_and_negative_offset_only_dataframes PASSED [ 24%]
tests/test_chat.py::TestChatVelocityAnalyzer::test_invalid_config_values_raise_value_error PASSED [ 24%]
tests/test_composer.py::TestVideoComposerInit::test_default_config_resolution_and_zoom PASSED [ 24%]
tests/test_composer.py::TestVideoComposerInit::test_custom_config_resolution_and_zoom PASSED [ 24%]
tests/test_composer.py::TestCropToAspect::test_crop_square_region_to_wide_aspect PASSED [ 24%]
tests/test_composer.py::TestCropToAspect::test_crop_wide_region_to_narrow_aspect PASSED [ 25%]
tests/test_composer.py::TestCropToAspect::test_crop_coordinates_boundary_clamped PASSED [ 25%]
tests/test_composer.py::TestLayoutComposition::test_layout_a_with_facecam_stack PASSED [ 25%]
tests/test_composer.py::TestLayoutComposition::test_layout_b_gameplay_only PASSED [ 25%]
tests/test_composer.py::TestLayoutComposition::test_compose_routes_facecam_vs_gameplay PASSED [ 25%]
tests/test_composer.py::TestApplySpikeZoom::test_zoom_applied_around_peak_offset PASSED [ 26%]
tests/test_composer.py::TestApplySpikeZoom::test_zoom_near_beginning PASSED [ 26%]
tests/test_composer.py::TestApplySpikeZoom::test_zoom_disabled_when_factor_is_one PASSED [ 26%]
tests/test_composer.py::TestApplySpikeZoom::test_zoom_skipped_when_peak_offset_none_or_negative PASSED [ 26%]
tests/test_composer.py::TestApplySpikeZoom::test_zoom_skipped_when_peak_beyond_duration PASSED [ 26%]
tests/test_composer.py::TestComposeStoryBeats::test_fallback_to_compose_on_single_segment PASSED [ 27%]
tests/test_composer.py::TestComposeStoryBeats::test_stitch_two_beats_with_clamping PASSED [ 27%]
tests/test_composer.py::TestComposeStoryBeats::test_stitch_three_beats_without_trim_ranges PASSED [ 27%]
tests/test_composer.py::TestComposeStoryBeats::test_invalid_trim_range_handled_safely PASSED [ 27%]
tests/test_composer.py::TestComposeStoryBeats::test_empty_segments_raises_value_error PASSED [ 27%]
tests/test_composer.py::TestRemapTimelineForBeats::test_words_and_chat_shifted_and_dead_air_dropped PASSED [ 28%]
tests/test_composer.py::TestRemapTimelineForBeats::test_peak_in_dead_air_returns_none PASSED [ 28%]
tests/test_composer.py::TestRemapTimelineForBeats::test_peak_zero_or_negative_returns_none PASSED [ 28%]
tests/test_composer.py::TestRemapTimelineForBeats::test_empty_beats_returns_original PASSED [ 28%]
tests/test_composer.py::TestRemapTimelineForBeats::test_chat_timestamp_field_variations PASSED [ 28%]
tests/test_composer.py::TestRealFileHandlingAndWindowsCleanup::test_compose_with_real_mp4_files PASSED [ 29%]
tests/test_composer.py::TestRealFileHandlingAndWindowsCleanup::test_compose_story_beats_closes_opened_handles_on_partial_failure PASSED [ 29%]
tests/test_composer.py::TestRealFileHandlingAndWindowsCleanup::test_crop_to_aspect_out_of_bounds_box_stays_within_frame PASSED [ 29%]
tests/test_composer.py::TestRealFileHandlingAndWindowsCleanup::test_apply_spike_zoom_uses_spec_window_and_skips_zero_duration PASSED [ 29%]
tests/test_composer.py::TestRealFileHandlingAndWindowsCleanup::test_remap_timeline_handles_malformed_chat_and_trailing_zero_duration_beat PASSED [ 29%]
tests/test_config.py::TestHardwareDetection::test_tier_high_vram_rtx_4070_12gb[True-h264_nvenc] PASSED [ 30%]
tests/test_config.py::TestHardwareDetection::test_tier_high_vram_rtx_4070_12gb[False-libx264] PASSED [ 30%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_tensor_rtx_4050_6gb[True-h264_nvenc] PASSED [ 30%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_tensor_rtx_4050_6gb[False-libx264] PASSED [ 30%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_rtx_turing_cc75_detected_as_tensor PASSED [ 30%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_no_tensor_gtx_1660_super_6gb[True-h264_nvenc] PASSED [ 31%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_no_tensor_gtx_1660_super_6gb[False-libx264] PASSED [ 31%]
tests/test_config.py::TestHardwareDetection::test_tier_low_vram_4gb_fallback_to_cpu PASSED [ 31%]
tests/test_config.py::TestHardwareDetection::test_tier_cpu_only_when_cuda_unavailable PASSED [ 31%]
tests/test_config.py::TestHardwareDetection::test_detect_nvenc_support_present_absent_and_missing_ffmpeg PASSED [ 31%]
tests/test_config.py::TestConfigSystem::test_defaults_and_dot_notation_lookup PASSED [ 32%]
tests/test_config.py::TestConfigSystem::test_user_overrides_preserved_over_hardware_auto_config PASSED [ 32%]
tests/test_config.py::TestConfigSystem::test_resolve_hardware_config_cpu_float16_safety_guard PASSED [ 32%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_weight_sum PASSED [ 32%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_resolution PASSED [ 32%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[0] PASSED [ 33%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[-2] PASSED [ 33%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[fast] PASSED [ 33%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[True] PASSED [ 33%]
tests/test_config.py::TestConfigSystem::test_packaged_default_config_matches_repo_root PASSED [ 33%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_malformed_or_negative_weights[None] PASSED [ 33%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_malformed_or_negative_weights[bad_weights1] PASSED [ 34%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_malformed_or_negative_weights[bad_weights2] PASSED [ 34%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_malformed_or_negative_weights[bad_weights3] PASSED [ 34%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_malformed_or_negative_weights[bad_weights4] PASSED [ 34%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_non_positive_or_non_int_resolution[bad_res0] PASSED [ 34%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_non_positive_or_non_int_resolution[bad_res1] PASSED [ 35%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_non_positive_or_non_int_resolution[bad_res2] PASSED [ 35%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_non_positive_or_non_int_resolution[bad_res3] PASSED [ 35%]
tests/test_config.py::TestConfigSystem::test_validation_accepts_all_four_transformation_modes[auto] PASSED [ 35%]
tests/test_config.py::TestConfigSystem::test_validation_accepts_all_four_transformation_modes[narrative] PASSED [ 35%]
tests/test_config.py::TestConfigSystem::test_validation_accepts_all_four_transformation_modes[story] PASSED [ 36%]
tests/test_config.py::TestConfigSystem::test_validation_accepts_all_four_transformation_modes[raw] PASSED [ 36%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_transformation_mode PASSED [ 36%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_non_mapping_user_yaml PASSED [ 36%]
tests/test_config.py::TestDataModels::test_wildcard_import_exports_all_models PASSED [ 36%]
tests/test_config.py::TestDataModels::test_pipeline_status_is_int_enum_with_numeric_ordering PASSED [ 37%]
tests/test_config.py::TestDataModels::test_all_models_json_round_trip PASSED [ 37%]
tests/test_config.py::TestDataModels::test_from_dict_tolerates_explicit_json_null_fields PASSED [ 37%]
tests/test_discovery.py::TestYouTubeSearch::test_search_youtube_uses_ytsearch_and_extract_flat_false PASSED [ 37%]
tests/test_discovery.py::TestYouTubeSearch::test_discover_uses_default_game_and_configured_keywords PASSED [ 37%]
tests/test_discovery.py::TestYouTubeSearch::test_discover_for_custom_game_ignores_default_keywords PASSED [ 38%]
tests/test_discovery.py::TestYouTubeSearch::test_search_youtube_handles_download_error_gracefully PASSED [ 38%]
tests/test_discovery.py::TestFilterStreams::test_filters_short_videos_and_low_view_videos PASSED [ 38%]
tests/test_discovery.py::TestFilterStreams::test_deduplicates_by_video_id PASSED [ 38%]
tests/test_discovery.py::TestFilterStreams::test_sorts_by_view_count_descending_and_respects_max_streams PASSED [ 38%]
tests/test_discovery.py::TestFilterStreams::test_metadata_fallbacks_for_url_and_streamer PASSED [ 39%]
tests/test_discovery.py::TestFilterStreams::test_discover_max_streams_override_exceeds_config_default PASSED [ 39%]
tests/test_discovery.py::TestFilterStreams::test_filter_streams_handles_non_finite_and_float_string_metadata PASSED [ 39%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_maps_fields_accurately PASSED [ 39%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_preserves_caller_short_url PASSED [ 39%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields[metadata0-ChannelName] PASSED [ 40%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields[metadata1-CreatorName] PASSED [ 40%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields[metadata2-Unknown] PASSED [ 40%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_detects_platform[Youtube-https://www.youtube.com/watch?v=abc-youtube] PASSED [ 40%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_detects_platform[TwitchVod-https://www.twitch.tv/videos/12345-twitch] PASSED [ 40%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_detects_platform[Kick-https://kick.com/streamer/videos/12345-kick] PASSED [ 41%]
tests/test_download.py::TestExtractStreamInfo::test_extract_stream_info_raises_when_info_is_none PASSED [ 41%]
tests/test_download.py::TestDownloadAudio::test_download_audio_configures_16khz_mono_wav_and_returns_result PASSED [ 41%]
tests/test_download.py::TestDownloadAudio::test_download_audio_skips_download_when_cached_wav_exists PASSED [ 41%]
tests/test_download.py::TestDownloadAudio::test_download_audio_forwards_progress_callback PASSED [ 41%]
tests/test_download.py::TestDownloadAudio::test_download_audio_raises_when_video_id_missing PASSED [ 42%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: Private video. Sign in if you've been granted access-private] PASSED [ 42%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: Video unavailable. This video has been removed-deleted or unavailable] PASSED [ 42%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate-age-restricted] PASSED [ 42%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: [youtube] abc: The uploader has not made this video available in your country-region-locked] PASSED [ 42%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: 'not-a-url' is not a valid URL.-Invalid or unsupported stream URL] PASSED [ 43%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: Unsupported URL: https://example.invalid/stream-Invalid or unsupported stream URL] PASSED [ 43%]
tests/test_download.py::TestDownloadErrorHandling::test_extract_stream_info_translates_download_errors[ERROR: Unexpected network timeout-Failed to download audio] PASSED [ 43%]
tests/test_download.py::TestDownloadErrorHandling::test_download_audio_translates_download_error_during_download_phase PASSED [ 43%]
tests/test_download.py::TestDownloadErrorHandling::test_download_audio_translates_postprocessing_error PASSED [ 43%]
tests/test_download.py::TestDownloadErrorHandling::test_empty_url_raises_descriptive_error[] PASSED [ 44%]
tests/test_download.py::TestDownloadErrorHandling::test_empty_url_raises_descriptive_error[   ] PASSED [ 44%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_options_and_default_buffer_padding PASSED [ 44%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_clamps_actual_start_at_zero PASSED [ 44%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_custom_buffer_seconds[0.0-20.0-50.0] PASSED [ 44%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_custom_buffer_seconds[5.0-15.0-55.0] PASSED [ 44%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_custom_buffer_seconds[25.0-0.0-75.0] PASSED [ 45%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_skips_when_cached_mp4_exists PASSED [ 45%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_multiple_segments_reuses_video_id_and_skips_cached PASSED [ 45%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_multiple_segments_empty_list_skips_yt_dlp PASSED [ 45%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[-10.0-20.0-3.0] PASSED [ 45%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[   -10.0-20.0-3.0] PASSED [ 46%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[https://www.youtube.com/watch?v=dQw4w9WgXcQ--1.0-20.0-3.0] PASSED [ 46%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[https://www.youtube.com/watch?v=dQw4w9WgXcQ-20.0-20.0-3.0] PASSED [ 46%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[https://www.youtube.com/watch?v=dQw4w9WgXcQ-25.0-20.0-3.0] PASSED [ 46%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_validation_rejects_invalid_arguments[https://www.youtube.com/watch?v=dQw4w9WgXcQ-10.0-20.0--0.5] PASSED [ 46%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_multiple_segments_validates_all_segments_before_network PASSED [ 47%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_wraps_yt_dlp_errors PASSED [ 47%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_raises_when_video_id_missing PASSED [ 47%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_audio_handles_na_metadata_and_offline_sidecar_cache PASSED [ 47%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_rejects_non_finite_and_bool_ranges[nan-20.0-3.0] PASSED [ 47%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_rejects_non_finite_and_bool_ranges[10.0-inf-3.0] PASSED [ 48%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_rejects_non_finite_and_bool_ranges[10.0-20.0-nan] PASSED [ 48%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_rejects_non_finite_and_bool_ranges[True-20.0-3.0] PASSED [ 48%]
tests/test_download.py::TestVideoSegmentDownloader::test_download_segment_invalidates_cache_on_buffer_change_and_detects_alt_container PASSED [ 48%]
tests/test_facecam.py::TestCornerDetection::test_is_in_corner_all_four_corners PASSED [ 48%]
tests/test_facecam.py::TestCornerDetection::test_is_in_corner_center_rejected PASSED [ 49%]
tests/test_facecam.py::TestCornerDetection::test_is_in_corner_edge_midpoints_rejected PASSED [ 49%]
tests/test_facecam.py::TestCornerDetection::test_is_in_corner_boundary_conditions PASSED [ 49%]
tests/test_facecam.py::TestCornerDetection::test_is_in_corner_zero_or_negative_dimensions PASSED [ 49%]
tests/test_facecam.py::TestCornerDetection::test_is_in_corner_callable_via_class_and_instance PASSED [ 49%]
tests/test_facecam.py::TestAreaFiltering::test_valid_area_within_range PASSED [ 50%]
tests/test_facecam.py::TestAreaFiltering::test_area_below_minimum_rejected PASSED [ 50%]
tests/test_facecam.py::TestAreaFiltering::test_area_above_maximum_rejected PASSED [ 50%]
tests/test_facecam.py::TestAreaFiltering::test_area_exact_boundaries PASSED [ 50%]
tests/test_facecam.py::TestAreaFiltering::test_area_zero_or_negative_dimensions PASSED [ 50%]
tests/test_facecam.py::TestMedianAndPaddingClamping::test_padded_box_centered_and_scaled PASSED [ 51%]
tests/test_facecam.py::TestMedianAndPaddingClamping::test_padded_box_clamps_to_top_left_boundaries PASSED [ 51%]
tests/test_facecam.py::TestMedianAndPaddingClamping::test_padded_box_clamps_to_bottom_right_boundaries PASSED [ 51%]
tests/test_facecam.py::TestMedianAndPaddingClamping::test_custom_padding_ratios PASSED [ 51%]
tests/test_facecam.py::TestThresholdAndConsistency::test_detection_below_30_percent_returns_none PASSED [ 51%]
tests/test_facecam.py::TestThresholdAndConsistency::test_detection_at_or_above_30_percent_succeeds PASSED [ 52%]
tests/test_facecam.py::TestThresholdAndConsistency::test_scattered_corner_detections_return_none PASSED [ 52%]
tests/test_facecam.py::TestEndToEndSyntheticVideo::test_detect_facecam_top_left PASSED [ 52%]
tests/test_facecam.py::TestEndToEndSyntheticVideo::test_detect_facecam_top_right PASSED [ 52%]
tests/test_facecam.py::TestEndToEndSyntheticVideo::test_detect_facecam_bottom_left PASSED [ 52%]
tests/test_facecam.py::TestEndToEndSyntheticVideo::test_detect_facecam_bottom_right PASSED [ 53%]
tests/test_facecam.py::TestEndToEndSyntheticVideo::test_detect_stream_without_facecam_returns_none PASSED [ 53%]
tests/test_facecam.py::TestEndToEndSyntheticVideo::test_detect_in_game_center_face_returns_none PASSED [ 53%]
tests/test_facecam.py::TestEndToEndSyntheticVideo::test_60s_video_processes_under_15_seconds PASSED [ 53%]
tests/test_facecam.py::TestRobustnessAndEdgeCases::test_non_existent_file_returns_none PASSED [ 53%]
tests/test_facecam.py::TestRobustnessAndEdgeCases::test_zero_or_negative_sample_count_returns_none PASSED [ 54%]
tests/test_facecam.py::TestRobustnessAndEdgeCases::test_corrupt_video_file_returns_none PASSED [ 54%]
tests/test_facecam.py::TestRobustnessAndEdgeCases::test_config_overrides_respected PASSED [ 54%]
tests/test_facecam.py::TestRobustnessAndEdgeCases::test_detect_uses_configured_num_samples_by_default PASSED [ 54%]
tests/test_facecam.py::TestRobustnessAndEdgeCases::test_facecam_region_serialization_round_trip PASSED [ 54%]
tests/test_hooks.py::TestHookTitleOverlayConfig::test_init_defaults PASSED [ 55%]
tests/test_hooks.py::TestHookTitleOverlayConfig::test_init_custom_config PASSED [ 55%]
tests/test_hooks.py::TestPillowRendering::test_render_hook_banner_shape_and_channels PASSED [ 55%]
tests/test_hooks.py::TestPillowRendering::test_render_hook_banner_text_wrapping PASSED [ 55%]
tests/test_hooks.py::TestPillowRendering::test_render_context_badge_shape_and_channels PASSED [ 55%]
tests/test_hooks.py::TestPillowRendering::test_render_context_badge_emojis[\U0001f480 Instant Karma] PASSED [ 55%]
tests/test_hooks.py::TestPillowRendering::test_render_context_badge_emojis[\U0001f3a8 Insane Detail] PASSED [ 56%]
tests/test_hooks.py::TestPillowRendering::test_render_context_badge_emojis[\U0001f525 Clutch Escape] PASSED [ 56%]
tests/test_hooks.py::TestPillowRendering::test_render_context_badge_emojis[\U0001f9e0 Hidden Detail] PASSED [ 56%]
tests/test_hooks.py::TestPillowRendering::test_render_context_badge_emojis[\u26a1 High Energy] PASSED [ 56%]
tests/test_hooks.py::TestPillowRendering::test_render_context_badge_emojis[\U0001f602 NPC Chaos] PASSED [ 56%]
tests/test_hooks.py::TestPillowRendering::test_wrap_text_single_word_and_empty PASSED [ 57%]
tests/test_hooks.py::TestAddHookOverlay::test_add_hook_both_enabled PASSED [ 57%]
tests/test_hooks.py::TestAddHookOverlay::test_add_hook_duration_clamped_to_shorter_video PASSED [ 57%]
tests/test_hooks.py::TestAddHookOverlay::test_add_hook_disappears_after_duration PASSED [ 57%]
tests/test_hooks.py::TestAddHookOverlay::test_add_hook_disabled_with_context_badge PASSED [ 57%]
tests/test_hooks.py::TestAddHookOverlay::test_add_hook_disabled_with_empty_badge_returns_unchanged PASSED [ 58%]
tests/test_hooks.py::TestAddHookOverlay::test_add_hook_enabled_empty_strings_returns_unchanged PASSED [ 58%]
tests/test_hooks.py::TestAddHookOverlay::test_add_hook_enabled_empty_badge_only_hook_layer PASSED [ 58%]
tests/test_hooks.py::TestAddHookOverlay::test_add_hook_position_center PASSED [ 58%]
tests/test_hooks.py::TestAudioAndHandlePreservation::test_audio_preservation PASSED [ 58%]
tests/test_hooks.py::TestAudioAndHandlePreservation::test_audio_none_when_absent PASSED [ 59%]
tests/test_hooks.py::TestAudioAndHandlePreservation::test_source_clip_references_propagated PASSED [ 59%]
tests/test_hooks.py::TestAudioAndHandlePreservation::test_source_clip_defaults_when_missing PASSED [ 59%]
tests/test_hooks.py::TestFadeInEffect::test_fade_in_enabled PASSED       [ 59%]
tests/test_hooks.py::TestFadeInEffect::test_fade_in_disabled PASSED      [ 59%]
tests/test_hooks.py::TestFadeInEffect::test_zero_duration_overlay_returns_video_unchanged PASSED [ 60%]
tests/test_hooks.py::TestFadeInEffect::test_oversized_single_token_hook_and_long_badge_fit_within_frame_width PASSED [ 60%]
tests/test_loudness.py::TestLoudnessDetectorSyntheticAudio::test_detect_finds_two_bursts_on_odd_duration_wav PASSED [ 60%]
tests/test_loudness.py::TestLoudnessDetectorSyntheticAudio::test_detect_scores_scale_monotonically_with_burst_amplitude PASSED [ 60%]
tests/test_loudness.py::TestLoudnessDetectorSyntheticAudio::test_detect_merges_nearby_bursts_and_keeps_loudest_peak_time[0.35-0.95-28.0] PASSED [ 60%]
tests/test_loudness.py::TestLoudnessDetectorSyntheticAudio::test_detect_merges_nearby_bursts_and_keeps_loudest_peak_time[0.95-0.35-20.0] PASSED [ 61%]
tests/test_loudness.py::TestLoudnessDetectorSyntheticAudio::test_multi_block_streaming_uses_librosa_stream_not_load PASSED [ 61%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_merge_nearby_unit_preserves_max_score_and_peak_time PASSED [ 61%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_merge_nearby_empty_returns_empty PASSED [ 61%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_silent_and_constant_audio_return_no_spikes PASSED [ 61%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_ultra_short_audio_does_not_crash PASSED [ 62%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_stereo_wav_downmixes_and_detects_spikes PASSED [ 62%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_missing_file_raises_file_not_found PASSED [ 62%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_invalid_config_values_raise_value_error PASSED [ 62%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_detected_spikes_serialize_round_trip PASSED [ 62%]
tests/test_loudness.py::TestLoudnessMergeAndEdgeCases::test_detects_boundary_burst_at_frame_zero_and_scales_secondary_spike PASSED [ 63%]
tests/test_mcp_server.py::test_mcp_server_registration_and_schemas PASSED [ 63%]
tests/test_mcp_server.py::test_mcp_main_invokes_mcp_run PASSED           [ 63%]
tests/test_mcp_server.py::test_discover_streams_success_and_error PASSED [ 63%]
tests/test_mcp_server.py::test_download_transcribe_analyze_via_audio_path_linkage PASSED [ 63%]
tests/test_mcp_server.py::test_submit_highlights_parses_ypp_and_persists_cache PASSED [ 64%]
tests/test_mcp_server.py::test_submit_highlights_invalid_json_returns_error PASSED [ 64%]
tests/test_mcp_server.py::test_render_clip_story_narrative_raw_and_voice_overrides PASSED [ 64%]
tests/test_mcp_server.py::test_render_all_selected_renders_and_completes_cache PASSED [ 64%]
tests/test_mcp_server.py::test_get_pipeline_status_empty_and_populated PASSED [ 64%]
tests/test_mcp_server.py::test_generate_post_metadata_offline_default_and_ollama_opt_in PASSED [ 65%]
tests/test_mcp_server.py::test_cleanup_temp_stream_and_all_completed_safety PASSED [ 65%]
tests/test_mcp_server.py::test_unexpected_exception_returns_structured_json_error PASSED [ 65%]
tests/test_mcp_server.py::test_edge_cases_validation_and_cache_hydration PASSED [ 65%]
tests/test_narrator.py::TestTransformativeNarratorConfig::test_init_defaults PASSED [ 65%]
tests/test_narrator.py::TestTransformativeNarratorConfig::test_init_custom_overrides PASSED [ 66%]
tests/test_narrator.py::TestSynthesizeSpeechRoutingAndFallbacks::test_empty_or_whitespace_text_returns_none PASSED [ 66%]
tests/test_narrator.py::TestSynthesizeSpeechRoutingAndFallbacks::test_disabled_transformation_returns_none PASSED [ 66%]
tests/test_narrator.py::TestSynthesizeSpeechRoutingAndFallbacks::test_engine_none_returns_none PASSED [ 66%]
tests/test_narrator.py::TestSynthesizeSpeechRoutingAndFallbacks::test_provider_fallback_kokoro_import_failure_to_edge_tts PASSED [ 66%]
tests/test_narrator.py::TestSynthesizeSpeechRoutingAndFallbacks::test_auto_kokoro_success_bypasses_edge_tts PASSED [ 66%]
tests/test_narrator.py::TestSynthesizeSpeechRoutingAndFallbacks::test_explicit_edge_tts_uses_edge_tts_voice_never_kokoro_voice PASSED [ 67%]
tests/test_narrator.py::TestSynthesizeSpeechRoutingAndFallbacks::test_elevenlabs_unconfigured_falls_back_to_auto PASSED [ 67%]
tests/test_narrator.py::TestSynthesizeSpeechRoutingAndFallbacks::test_mocked_elevenlabs_api_call_with_api_key_and_voice_id PASSED [ 67%]
tests/test_narrator.py::TestSynthesizeSpeechRoutingAndFallbacks::test_unknown_engine_falls_back_to_auto PASSED [ 67%]
tests/test_narrator.py::TestCustomVoiceoverPriority::test_custom_voice_intro_takes_priority_over_tts PASSED [ 67%]
tests/test_narrator.py::TestCustomVoiceoverPriority::test_custom_voice_outro_takes_priority_over_tts PASSED [ 68%]
tests/test_narrator.py::TestCustomVoiceoverPriority::test_missing_custom_voice_falls_back_to_tts PASSED [ 68%]
tests/test_narrator.py::TestCustomVoiceoverPriority::test_prepare_narration_tracks_respects_both_priorities PASSED [ 68%]
tests/test_narrator.py::TestCustomVoiceoverPriority::test_prepare_narration_tracks_disabled_or_raw PASSED [ 68%]
tests/test_narrator.py::TestDuckingFiltergraph::test_dynamic_filtergraph_intro_only PASSED [ 68%]
tests/test_narrator.py::TestDuckingFiltergraph::test_dynamic_filtergraph_outro_only PASSED [ 69%]
tests/test_narrator.py::TestDuckingFiltergraph::test_dynamic_filtergraph_both_intro_and_outro PASSED [ 69%]
tests/test_narrator.py::TestDuckingFiltergraph::test_eval_frame_guardrail_present_in_all_filtergraphs PASSED [ 69%]
tests/test_narrator.py::TestDuckingFiltergraph::test_stereo_sample_rate_normalization_present PASSED [ 69%]
tests/test_narrator.py::TestShortClipOverlapClamping::test_both_windows_scaled_when_exceeding_75_percent_limit PASSED [ 69%]
tests/test_narrator.py::TestShortClipOverlapClamping::test_intro_only_clamping PASSED [ 70%]
tests/test_narrator.py::TestShortClipOverlapClamping::test_outro_only_clamping PASSED [ 70%]
tests/test_narrator.py::TestShortClipOverlapClamping::test_unclamped_when_within_budget PASSED [ 70%]
tests/test_narrator.py::TestShortClipOverlapClamping::test_clamped_narrator_tracks_are_atrimmed_to_prevent_climax_bleed_and_overlap PASSED [ 70%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_bypass_when_disabled PASSED [ 70%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_bypass_when_raw_mode PASSED [ 71%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_bypass_when_no_audio_tracks_provided PASSED [ 71%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_real_e2e_ducking_intro_and_outro PASSED [ 71%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_real_e2e_ducking_intro_only PASSED [ 71%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_real_e2e_ducking_outro_only PASSED [ 71%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_in_place_mixing_safe_replacement PASSED [ 72%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_ffmpeg_failure_falls_back_cleanly_to_copy PASSED [ 72%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_silent_video_still_receives_narrator_audio_track PASSED [ 72%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_non_wav_m4a_custom_voiceover_is_probed_and_mixed PASSED [ 72%]
tests/test_narrator.py::TestMixNarrationWithDucking::test_case_insensitive_raw_mode_and_tiny_atrim_floor PASSED [ 72%]
tests/test_pipeline.py::TestFormatTime::test_formats_hours_minutes_seconds_and_clamps_negative PASSED [ 73%]
tests/test_pipeline.py::TestRecoverFailedState::test_non_failed_states_are_unchanged PASSED [ 73%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_highlights_resets_to_assembling PASSED [ 73%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_spikes_only_resets_to_scoring PASSED [ 73%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_chat_spikes_only_resets_to_scoring PASSED [ 73%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_transcript_only_resets_to_analyzing PASSED [ 74%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_existing_audio_resets_to_transcribing PASSED [ 74%]
tests/test_pipeline.py::TestRecoverFailedState::test_failed_with_missing_audio_or_no_artifacts_resets_to_pending PASSED [ 74%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_full_pipeline_transitions_pending_to_complete PASSED [ 74%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_resume_from_cached_analyzing_skips_download_and_transcribe PASSED [ 74%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_resume_from_cached_scoring_skips_download_transcribe_and_analyze PASSED [ 75%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_failed_state_saves_99_and_retry_recovers_to_correct_stage PASSED [ 75%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_mode_and_voice_paths_forwarded_to_assembly PASSED [ 75%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_empty_highlights_completes_cleanly PASSED [ 75%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_url_whitespace_normalized_and_empty_url_rejected PASSED [ 75%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_complete_state_with_empty_selection_skips_reselection PASSED [ 76%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_invalid_sub_clips_falls_back_to_single_segment_download PASSED [ 76%]
tests/test_pipeline.py::TestPipelineEndToEnd::test_phase4_assembly_hook_single_and_multi_beat_contract PASSED [ 76%]
tests/test_pipeline.py::TestInteractiveSelection::test_interactive_select_displays_moment_type_narration_intro_and_beats PASSED [ 76%]
tests/test_pipeline.py::TestInteractiveSelection::test_interactive_select_all_and_empty PASSED [ 76%]
tests/test_pipeline.py::TestInteractiveSelection::test_process_stream_interactive_true_vs_false PASSED [ 77%]
tests/test_pipeline.py::TestParallelProcessing::test_parallel_forces_non_interactive_respects_concurrency_and_isolates_errors PASSED [ 77%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_clip_command_forwards_mode_and_voice_options PASSED [ 77%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_clip_command_rejects_invalid_mode PASSED [ 77%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_clip_command_rejects_directory_for_voice_intro PASSED [ 77%]
tests/test_pipeline.py::TestCLIPipelineWiring::test_auto_command_discovers_and_invokes_parallel_pipeline PASSED [ 77%]
tests/test_pipeline.py::TestAuditHardeningAndModes::test_narrative_mode_forces_single_cut_even_when_sub_clips_exist PASSED [ 78%]
tests/test_pipeline.py::TestAuditHardeningAndModes::test_transformation_disabled_defaults_to_raw_mode PASSED [ 78%]
tests/test_pipeline.py::TestAuditHardeningAndModes::test_analyze_failure_during_chat_does_not_skip_chat_on_retry PASSED [ 78%]
tests/test_pipeline.py::TestAuditHardeningAndModes::test_recover_failed_state_handles_zero_spike_analysis_and_missing_audio PASSED [ 78%]
tests/test_pipeline.py::TestAuditHardeningAndModes::test_assemble_highlight_reraises_non_phase4_import_error PASSED [ 78%]
tests/test_pipeline.py::TestAuditHardeningAndModes::test_parallel_streams_serialize_gpu_stages_and_preserve_detected_game_and_force PASSED [ 79%]
tests/test_pipeline.py::TestAuditHardeningAndModes::test_process_stream_redownloads_missing_audio_at_analyzing_without_retranscribing PASSED [ 79%]
tests/test_pipeline.py::TestAuditHardeningAndModes::test_invalid_mode_override_raises_and_fallback_sanitizes_filename_template PASSED [ 79%]
tests/test_pipeline.py::TestAuditHardeningAndModes::test_process_stream_fails_fast_on_invalid_mode_before_cache_clear_or_download PASSED [ 79%]
tests/test_scorer.py::TestModelBackwardCompat::test_highlight_from_dict_without_ypp_fields PASSED [ 79%]
tests/test_scorer.py::TestModelBackwardCompat::test_highlight_from_dict_with_ypp_fields PASSED [ 80%]
tests/test_scorer.py::TestModelBackwardCompat::test_highlight_serialization_round_trip PASSED [ 80%]
tests/test_scorer.py::TestModelBackwardCompat::test_clip_metadata_from_dict_without_ypp_fields PASSED [ 80%]
tests/test_scorer.py::TestModelBackwardCompat::test_clip_metadata_with_ypp_fields_round_trip PASSED [ 80%]
tests/test_scorer.py::TestTranscriptChunking::test_chunk_duration_and_overlap PASSED [ 80%]
tests/test_scorer.py::TestTranscriptChunking::test_overlap_between_consecutive_chunks PASSED [ 81%]
tests/test_scorer.py::TestTranscriptChunking::test_custom_chunk_minutes PASSED [ 81%]
tests/test_scorer.py::TestTranscriptChunking::test_empty_transcript_returns_no_chunks PASSED [ 81%]
tests/test_scorer.py::TestTranscriptChunking::test_chunks_contain_text PASSED [ 81%]
tests/test_scorer.py::TestGetCandidateChunks::test_only_with_signals_filters_dead_chunks PASSED [ 81%]
tests/test_scorer.py::TestGetCandidateChunks::test_all_chunks_returned_when_signals_disabled PASSED [ 82%]
tests/test_scorer.py::TestSignalOverlapScore::test_full_overlap_returns_spike_score PASSED [ 82%]
tests/test_scorer.py::TestSignalOverlapScore::test_partial_overlap_proportional PASSED [ 82%]
tests/test_scorer.py::TestSignalOverlapScore::test_no_overlap_returns_zero PASSED [ 82%]
tests/test_scorer.py::TestSignalOverlapScore::test_multiple_spikes_takes_best PASSED [ 82%]
tests/test_scorer.py::TestSignalOverlapScore::test_clamped_to_ten PASSED [ 83%]
tests/test_scorer.py::TestSignalOverlapScore::test_empty_spikes_returns_zero PASSED [ 83%]
tests/test_scorer.py::TestWeightedScoreFusion::test_weighted_fusion_formula PASSED [ 83%]
tests/test_scorer.py::TestWeightedScoreFusion::test_weighted_score_with_no_signals PASSED [ 83%]
tests/test_scorer.py::TestWeightedScoreFusion::test_weighted_score_renormalizes_when_chat_unavailable PASSED [ 83%]
tests/test_scorer.py::TestWeightedScoreFusion::test_weighted_score_quiet_top_tier_ai_pick_passes_while_signal_backed_ranks_higher PASSED [ 84%]
tests/test_scorer.py::TestWeightedScoreFusion::test_weighted_score_sets_loudness_and_chat_fields PASSED [ 84%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_peak_time_from_loudness_metadata PASSED [ 84%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_peak_time_fallback_to_midpoint PASSED [ 84%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_peak_time_selects_highest_scored_spike PASSED [ 84%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_chat_burst_extraction PASSED [ 85%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_chat_burst_max_five PASSED [ 85%]
tests/test_scorer.py::TestPeakTimeAndChatBurst::test_chat_burst_empty_when_no_overlap PASSED [ 85%]
tests/test_scorer.py::TestSubClipsValidation::test_valid_two_beat_sub_clips PASSED [ 85%]
tests/test_scorer.py::TestSubClipsValidation::test_valid_three_beat_sub_clips PASSED [ 85%]
tests/test_scorer.py::TestSubClipsValidation::test_single_beat_returns_empty PASSED [ 86%]
tests/test_scorer.py::TestSubClipsValidation::test_empty_sub_clips_returns_empty PASSED [ 86%]
tests/test_scorer.py::TestSubClipsValidation::test_beat_outside_range_returns_empty PASSED [ 86%]
tests/test_scorer.py::TestSubClipsValidation::test_beat_too_short_returns_empty PASSED [ 86%]
tests/test_scorer.py::TestSubClipsValidation::test_overlapping_beats_return_empty PASSED [ 86%]
tests/test_scorer.py::TestSubClipsValidation::test_total_duration_too_short_returns_empty PASSED [ 87%]
tests/test_scorer.py::TestSubClipsValidation::test_missing_keys_returns_empty PASSED [ 87%]
tests/test_scorer.py::TestSubClipsValidation::test_unordered_beats_return_empty PASSED [ 87%]
tests/test_scorer.py::TestDeduplication::test_non_overlapping_kept PASSED [ 87%]
tests/test_scorer.py::TestDeduplication::test_overlapping_keeps_higher_score PASSED [ 87%]
tests/test_scorer.py::TestDeduplication::test_exact_duplicate_keeps_one PASSED [ 88%]
tests/test_scorer.py::TestDeduplication::test_empty_input PASSED         [ 88%]
tests/test_scorer.py::TestDeduplication::test_low_iou_keeps_both PASSED  [ 88%]
tests/test_scorer.py::TestWordExtraction::test_extract_words_in_range PASSED [ 88%]
tests/test_scorer.py::TestWordExtraction::test_extract_no_words_outside_range PASSED [ 88%]
tests/test_scorer.py::TestScoreAndSelect::test_pipeline_attaches_metadata PASSED [ 88%]
tests/test_scorer.py::TestPromptTemplates::test_highlight_scorer_template PASSED [ 89%]
tests/test_scorer.py::TestPromptTemplates::test_narrative_writer_template PASSED [ 89%]
tests/test_scorer.py::TestPromptTemplates::test_hook_generator_template PASSED [ 89%]
tests/test_scorer.py::TestPromptTemplates::test_caption_writer_template PASSED [ 89%]
tests/test_scorer.py::TestOllamaScorerInit::test_uses_hardware_resolved_llm_model_and_client_args[cuda_mid_tensor-qwen3:8b] PASSED [ 89%]
tests/test_scorer.py::TestOllamaScorerInit::test_uses_hardware_resolved_llm_model_and_client_args[cuda_mid_no_tensor-qwen3:4b] PASSED [ 90%]
tests/test_scorer.py::TestOllamaScorerInit::test_respects_user_llm_override PASSED [ 90%]
tests/test_scorer.py::TestOllamaScorerResponseParsing::test_parses_raw_json_list_with_ypp_fields_and_sub_clips PASSED [ 90%]
tests/test_scorer.py::TestOllamaScorerResponseParsing::test_parses_wrapped_highlights_object_with_think_tags_and_fences PASSED [ 90%]
tests/test_scorer.py::TestOllamaScorerResponseParsing::test_normalizes_invalid_moment_type_and_clamps_score PASSED [ 90%]
tests/test_scorer.py::TestOllamaScorerResponseParsing::test_shifts_relative_timestamps_when_chunk_offset_positive PASSED [ 91%]
tests/test_scorer.py::TestOllamaScorerResponseParsing::test_empty_responses_and_invalid_ranges PASSED [ 91%]
tests/test_scorer.py::TestOllamaScorerScoreChunkAndRetries::test_retries_on_malformed_json_and_recovers PASSED [ 91%]
tests/test_scorer.py::TestOllamaScorerScoreChunkAndRetries::test_returns_empty_after_three_malformed_attempts PASSED [ 91%]
tests/test_scorer.py::TestOllamaScorerScoreChunkAndRetries::test_connection_error_during_generate_propagates_with_model_hint PASSED [ 91%]
tests/test_scorer.py::TestOllamaScorerScoreChunkAndRetries::test_end_to_end_score_and_select_with_ollama PASSED [ 92%]
tests/test_scorer.py::TestOllamaScorerNarrativeAndMetadata::test_generate_narrative_uses_moment_type_and_retries PASSED [ 92%]
tests/test_scorer.py::TestOllamaScorerNarrativeAndMetadata::test_generate_metadata_backfills_missing_narrative_fields PASSED [ 92%]
tests/test_scorer.py::TestOllamaScorerNarrativeAndMetadata::test_generate_metadata_preserves_existing_narrative_without_extra_call PASSED [ 92%]
tests/test_scorer.py::TestOllamaScorerCheckConnection::test_check_connection_succeeds_with_ollama_list_response PASSED [ 92%]
tests/test_scorer.py::TestOllamaScorerCheckConnection::test_check_connection_succeeds_with_dict_response_and_latest_tag PASSED [ 93%]
tests/test_scorer.py::TestOllamaScorerCheckConnection::test_check_connection_raises_when_model_missing PASSED [ 93%]
tests/test_scorer.py::TestOllamaScorerCheckConnection::test_check_connection_raises_when_server_unreachable PASSED [ 93%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_null_json_fields_never_stringify_to_none PASSED [ 93%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_null_metadata_in_loudness_and_chat_formatters PASSED [ 93%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_relative_offset_shift_does_not_overshift_out_of_window_timestamps PASSED [ 94%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_rejects_nan_infinity_bool_and_negative_timestamps PASSED [ 94%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_parses_mm_ss_and_hh_mm_ss_timestamp_strings PASSED [ 94%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_hashtags_honor_game_and_streamer_flags_and_deduplicate PASSED [ 94%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_raw_or_disabled_transformation_skips_narrative_backfill PASSED [ 94%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_narrative_and_caption_retries_reinforce_json_instruction PASSED [ 95%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_chunk_transcript_includes_timestamps_no_explosion_and_no_tail_dup PASSED [ 95%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_score_and_select_enforces_clip_bounds_scopes_transcript_and_max_clips_override PASSED [ 95%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_validate_sub_clips_rejects_nan_bool_and_oversized_duration PASSED [ 95%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_peak_time_clamped_and_boundary_straddling_words_included PASSED [ 95%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_ollama_think_false_late_chunk1_offset_hashtag_cap_and_explicit_raw_mode PASSED [ 96%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_zero_llm_weight_nan_chunk_minutes_transcript_range_and_subclip_normalization PASSED [ 96%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_ollama_chunk1_relative_subclips_all_shift_and_invalid_mode_raises PASSED [ 96%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_extract_chat_burst_handles_null_non_dict_and_blank_messages PASSED [ 96%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_extract_peak_time_ignores_bool_and_overlap_ignores_inf_score PASSED [ 96%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_score_and_select_short_sub_clips_falls_back_to_single_cut PASSED [ 97%]
tests/test_scorer.py::TestOllamaScorerEdgeCasesAndHardening::test_ollama_parse_response_clears_sub_clips_on_any_malformed_beat PASSED [ 97%]
tests/test_transcriber.py::TestTranscriberHardwareProfiles::test_rtx_4050_profile_passes_resolved_params_and_logs_profile PASSED [ 97%]
tests/test_transcriber.py::TestTranscriberHardwareProfiles::test_gtx_1660_super_profile_passes_resolved_params PASSED [ 97%]
tests/test_transcriber.py::TestTranscriberHardwareProfiles::test_whisperx_forwards_configured_language_when_supported PASSED [ 97%]
tests/test_transcriber.py::TestTranscriberWordConversion::test_word_level_timestamp_conversion_and_unaligned_tokens PASSED [ 98%]
tests/test_transcriber.py::TestTranscriberSafetyAndVRAM::test_cpu_compute_type_safety_fallback[float16] PASSED [ 98%]
tests/test_transcriber.py::TestTranscriberSafetyAndVRAM::test_cpu_compute_type_safety_fallback[int8_float16] PASSED [ 98%]
tests/test_transcriber.py::TestTranscriberSafetyAndVRAM::test_two_stage_vram_cleanup_order_and_count_in_whisperx PASSED [ 98%]
tests/test_transcriber.py::TestTranscriberSafetyAndVRAM::test_vram_cleanup_skips_empty_cache_when_cuda_unavailable PASSED [ 98%]
tests/test_transcriber.py::TestTranscriberFallbackAndErrors::test_faster_whisper_fallback_when_whisperx_missing PASSED [ 99%]
tests/test_transcriber.py::TestTranscriberFallbackAndErrors::test_explicit_faster_whisper_engine_bypasses_whisperx PASSED [ 99%]
tests/test_transcriber.py::TestTranscriberFallbackAndErrors::test_missing_audio_file_raises_file_not_found PASSED [ 99%]
tests/test_transcriber.py::TestTranscriptCaching::test_save_and_load_transcript_round_trip PASSED [ 99%]
tests/test_transcriber.py::TestTranscriptCaching::test_load_transcript_returns_none_for_missing_or_corrupt_file PASSED [ 99%]
tests/test_transcriber.py::TestTranscriptCaching::test_faster_whisper_maps_auto_language_to_none PASSED [100%]

============================== warnings summary ===============================
tests/test_assembly.py: 116 warnings
tests/test_composer.py: 6 warnings
  D:\GTA6-Clipper\.venv\Lib\site-packages\moviepy\video\io\ffmpeg_reader.py:226: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    result.shape = (h, w, len(s) // (w * h))  # reshape((h, w, len(s)//(w*h)))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
===================== 509 passed, 122 warnings in 59.25s ======================
```

---

## 5. Architectural Notes & Hardening Summary

1. **Package 5.1 Integration & Live MCP Verification**: Confirmed `reports/phase-5/package-5.1-report.md` and commit `68c2807` on `phase-5-mcp-server`, validated all 11 live `MCPServer` tools via `streamclipper-mcp.exe` over stdio, and aligned the README's MCP tool reference table with the finalized signatures (`url` / `audio_path` `StepCache` resolution across `transcribe_audio`, `analyze_signals`, and `get_transcript_chunks`; `sub_clips_json` and zero-Ollama rendering on `render_clip` / `render_all_selected`; directory-bounded `COMPLETE`-only default on `cleanup_temp`).
2. **Cross-Platform & Config Fidelity Hardening**:
   - Replaced Ubuntu 22.04 / Debian 12-incompatible `python3.12 python3.12-venv` apt packages with portable `python3 python3-venv` + `libglib2.0-0` matching `python3 -m venv .venv`.
   - Removed `"ELEVENLABS_API_KEY": ""` from the Windows `.venv` MCP JSON config so host environment variables are never silently overwritten.
   - Replaced bash-only `\` multi-line continuations in CLI examples with cross-shell single-line commands valid in PowerShell, cmd.exe, and bash.
   - Corrected config reference accuracy for `discovery.game_keywords`, `transcription.language` (`"auto"` support), `captions.color`/`highlight_color`/`stroke_color` (exact 6 named colors + `#RRGGBB`), `hook_title.fade_in` (`CrossFadeIn` on hook + context badge), `raw` mode `hook_title.enabled` behavior, custom voiceover format support (`.wav`/`.m4a`/`.mp3`/`.aac`), and `prompts/` in the project tree.
