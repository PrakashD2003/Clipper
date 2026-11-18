# Phase 1 Final Implementation Report — Foundation & Scaffolding

## 1. Executive Summary & Phase Status

**Phase 1 implementation is complete.** All three Phase 1 packages (`1.1`, `1.2`, and `1.3`) defined in `streamclipper-implementation-guide-v2.md` have been implemented, verified, and committed on branch `phase-1-foundation`.

| Field | Value |
| :--- | :--- |
| **Phase** | Phase 1: Foundation & Scaffolding |
| **Branch** | `phase-1-foundation` (branched from `main` @ `9093a8f`) |
| **Packages Completed** | `1.1` (Project Scaffolding), `1.2` (Data Models & Hardware-Aware Config), `1.3` (CLI Skeleton & Step Cache) |
| **Implementation Status** | ✅ 3 / 3 Packages Complete (15 / 15 Acceptance Criteria Passed) |
| **Automated Tests** | ✅ 36 / 36 Passing (`tests/test_config.py`: 22, `tests/test_cache_cli.py`: 14) |
| **Remaining Gate** | Pending **Prompt 1.R** Lead Reviewer audit (`reports/phase-1/phase-1-review.md`) and `--no-ff` merge into `main` |
| **Python Environment** | Python 3.14.10 (`C:\Users\praka\AppData\Local\Python\pythoncore-3.14-64\python.exe`), pip 26.2.2, setuptools 82.0.1, pytest 9.2.0, Click 8.5.1, Rich 15.0.0, PyYAML 6.0.3 |
| **Source Reports Combined** | `reports/phase-1/package-1.1-report.md`, `reports/phase-1/package-1.2-report.md`, `reports/phase-1/package-1.3-report.md` |

---

## 2. Git Commit History (`main..phase-1-foundation`)

| Order | Commit | Package | Commit Message | Summary |
| :---: | :---: | :---: | :--- | :--- |
| 1 | `9306655` | `1.1` | `feat: project scaffolding, pyproject.toml, directory structure` | 15 files created (`+249` lines) |
| 2 | `5f1b854` | `1.2` | `feat: data models, config system with dynamic HardwareProfile auto-tuning` | 5 files created (`+1,331` lines) |
| 3 | `09d4807` | `1.3` | `feat: CLI skeleton with hardware profile display and step cache for resume` | 4 files changed (`+1,043 / -3` lines) |

---

## 3. Complete Deliverable Inventory

### 3.1 Directories Created
- `src/streamclipper/` (`discovery/`, `download/`, `analysis/`, `scoring/`, `assembly/`, `output/`, `mcp_server/`, `prompts/`)
- `tests/`
- `reports/phase-1/`
- `output/` *(created on disk; gitignored via `/output/`)*

### 3.2 Files Created / Updated Across Phase 1

| Package | File Path | Lines | Purpose |
| :---: | :--- | :---: | :--- |
| `1.1` | `pyproject.toml` | 42 | PEP 517 build config (`setuptools.build_meta`), 15 v2 dependencies (no `pytchat`), optional extras (`whisperx`, `mcp`, `dev`), console entry points (`streamclipper`, `streamclipper-mcp`) |
| `1.1` | `README.md` | 5 | Minimal placeholder required by `pyproject.toml` (`readme = "README.md"`); full documentation in Package 5.2 |
| `1.1` | `.gitignore` | 28 | Ignores Python build/cache artifacts, root `/output/`, `temp/`, media/subtitle/chat artifacts (`*.wav`, `*.mp3`, `*.mp4`, `*.ass`, `*.live_chat.json`), `config.yaml`, IDE files, and `.cache/` |
| `1.1` | `src/streamclipper/__init__.py` | 3 | Package docstring and `__version__ = "0.1.0"` |
| `1.1` | `src/streamclipper/{discovery,download,analysis,scoring,assembly,output}/__init__.py` | 1 each | Subpackage initializers (6 files) |
| `1.1` | `src/streamclipper/mcp_server/__init__.py` | 6 | Minimal MCP server placeholder exposing `main()` for `streamclipper-mcp` (expanded in Package 5.1) |
| `1.1` | `src/streamclipper/prompts/.gitkeep` | 0 | Preserves `prompts/` directory in Git until Phase 3 prompt templates are added |
| `1.1` | `tests/__init__.py` | 1 | Test suite package initializer |
| `1.2` | `config.default.yaml` | 98 | Default YAML configuration with `"auto"` defaults for `transcription.{model_size,device,compute_type,batch_size}`, `llm.model`, and `video.codec` |
| `1.2` | `src/streamclipper/models.py` | 285 | Shared enums (`Platform`, `SignalType`, `PipelineStatus(IntEnum)`), recursive `SerializableModel` (`_serialize_value`), and 10 dataclasses (`StreamInfo`, `TimeRange`, `WordTimestamp`, `TranscriptSegment`, `TranscriptionResult`, `Highlight`, `FacecamRegion`, `ClipMetadata`, `ClipResult`, `PipelineState`) with full `from_dict()` classmethods |
| `1.2` | `src/streamclipper/config.py` | 255 | `HardwareProfile`, `_detect_nvenc_support()`, `detect_hardware_profile()`, `resolve_hardware_config()`, and `Config` (defaults, deep merge, `is_auto_resolved()`, dot-notation `.get()`, validation) |
| `1.2` | `tests/test_config.py` | 577 | 22 unit tests covering all 4 hardware tiers + NVENC present/absent, config defaults/merging/validation, user override precedence, CPU `float16` safety guard, `PipelineStatus` `IntEnum` ordering, and full JSON round-trip serialization |
| `1.1 → 1.3` | `src/streamclipper/cli.py` | 215 | Click CLI group (`--version`, `--config`/`-c`, `--verbose`/`-v`), commands (`clip`, `auto`, `setup`, `status`), `_check_dependency()`, `_check_gpu()` (`props.total_memory / 1e9`, `major.minor`), and `_render_hardware_profile()` Rich table |
| `1.3` | `src/streamclipper/cache.py` | 127 | `StepCache` (`get_cache_key`, `get_cache_path`, `save_state`, `load_state`, `clear`, `list_all`) persisting `int(state.status)` and `{key}_transcript.json` alongside spikes, highlights, selected highlights, and clips |
| `1.3` | `tests/test_cache_cli.py` | 545 | 14 unit tests covering `StepCache` save/load/list/clear/corrupt-file resilience and Click `CliRunner` tests for `--help`, `--version`, `clip`, `auto`, `setup`, `_check_dependency`, and `status` |
| `1.1–1.3` | `reports/phase-1/package-1.{1,2,3}-report.md` | 427 total | Individual package completion reports (`143` + `116` + `168` lines) |

---

## 4. Consolidated Acceptance Criteria Checklist (15 / 15 PASS)

### Package 1.1 — Project Scaffolding
| # | Acceptance Criterion | Status | Evidence |
| :-: | :--- | :---: | :--- |
| 1.1.1 | `pip install -e .` succeeds with `setuptools.build_meta` | ✅ PASS | `pyproject.toml` configures `build-backend = "setuptools.build_meta"`; editable install exited `0` with all 15 v2 dependencies (no `pytchat`) |
| 1.1.2 | `streamclipper --help` prints the CLI help text | ✅ PASS | Verified in Package 1.1 and expanded in Package 1.3 |
| 1.1.3 | All `__init__.py` files exist | ✅ PASS | Created across `src/streamclipper/`, all 7 subpackages, and `tests/`; all import cleanly |
| 1.1.4 | `.gitignore` covers outputs, `.ass` subtitles, `.live_chat.json`, cache, and Python artifacts | ✅ PASS | Verified via `git check-ignore -v` (with `src/streamclipper/output/` remaining tracked) |

### Package 1.2 — Data Models & Dynamic Hardware-Aware Config System
| # | Acceptance Criterion | Status | Evidence |
| :-: | :--- | :---: | :--- |
| 1.2.1 | `from streamclipper.models import *` imports all dataclasses without error | ✅ PASS | Verified in `TestDataModels::test_wildcard_import_exports_all_models` (all 14 public symbols) |
| 1.2.2 | `PipelineStatus.PENDING < PipelineStatus.ANALYZING` evaluates to `True` | ✅ PASS | `PipelineStatus` subclasses `IntEnum` (`PENDING=1 .. COMPLETE=7, FAILED=99`); verified in `TestDataModels::test_pipeline_status_is_int_enum_with_numeric_ordering` |
| 1.2.3 | `Config()` loads defaults, merges user overrides, and resolves all `"auto"` hardware fields dynamically | ✅ PASS | Verified in `TestConfigSystem::test_defaults_and_dot_notation_lookup` and `test_user_overrides_preserved_over_hardware_auto_config` |
| 1.2.4 | `detect_hardware_profile()` correctly classifies all 4 tiers (`>= 10 GB` VRAM, `5.5–9.9 GB` + Tensor Cores, `5.5–9.9 GB` without Tensor Cores, CPU / `< 5.5 GB` VRAM) + NVENC present/absent | ✅ PASS | Verified across 10 test cases in `TestHardwareDetection` mocking `torch.cuda.get_device_properties(0)` and `_detect_nvenc_support()` |
| 1.2.5 | Explicit user overrides in `config.yaml` (e.g., forcing `transcription.model_size: "large-v2"` on GTX 1660 SUPER) are preserved untouched | ✅ PASS | Verified in `TestConfigSystem::test_user_overrides_preserved_over_hardware_auto_config`; `is_auto_resolved()` returns `False` for overridden keys and `True` for untouched `"auto"` keys |
| 1.2.6 | `config.get('analysis.weights.loudness')` returns `0.25` and validation rejects weights not summing to `1.0` | ✅ PASS | Verified in `TestConfigSystem::test_defaults_and_dot_notation_lookup` and `test_validation_rejects_invalid_weight_sum` (plus resolution and positive-int `batch_size` checks) |
| 1.2.7 | Nested dataclasses (`TranscriptionResult` → `TranscriptSegment` → `WordTimestamp`) round-trip through JSON without `TypeError` | ✅ PASS | Verified in `TestDataModels::test_all_models_json_round_trip` across all 10 dataclasses including `Path` and `Enum` fields |

### Package 1.3 — CLI Skeleton & Step Cache
| # | Acceptance Criterion | Status | Evidence |
| :-: | :--- | :---: | :--- |
| 1.3.1 | `streamclipper --help` shows all commands: `clip`, `auto`, `setup`, `status` | ✅ PASS | Verified via live `streamclipper --help` and `TestCLI::test_help_lists_all_commands` |
| 1.3.2 | `streamclipper setup` checks each dependency, reports GPU VRAM/CC via `total_memory`, and displays the resolved `HardwareProfile` table (including auto vs. user override provenance) | ✅ PASS | Verified via live `streamclipper setup` and `TestCLI::test_setup_renders_dependencies_gpu_and_hardware_profile_with_overrides` |
| 1.3.3 | `StepCache` saves and restores full `PipelineState` including nested `TranscriptionResult`, `loudness_spikes`, `chat_spikes`, and `highlights` | ✅ PASS | Verified in `TestStepCache::test_save_and_load_full_state_with_nested_transcript` (`int(status)` persisted and restored via `PipelineStatus(int(...))`) |
| 1.3.4 | `StepCache` lists and clears cached states cleanly | ✅ PASS | Verified in `TestStepCache::test_list_all_ignores_transcript_and_corrupt_files` and `test_clear_removes_state_and_transcript_cleanly` |

---

## 5. Consolidated Verification Outputs

### 5.1 Editable Install & `.gitignore` Checks (Package 1.1)
- `pip install -e .` and `pip install -e ".[dev]"` succeeded cleanly, generating `streamclipper.exe` and `streamclipper-mcp.exe` in `C:\Users\praka\AppData\Local\Python\pythoncore-3.14-64\Scripts`.
- `git check-ignore -v` verified:
  - Ignored: `/output/`, `temp/`, `*.ass`, `*.live_chat.json`, `config.yaml`, `.cache/`, `__pycache__/`, `*.pyc`, `src/streamclipper.egg-info/`.
  - Tracked: `src/streamclipper/output/__init__.py`.

### 5.2 Full Automated Test Suite (`pytest -v`)

```text
============================= test session starts =============================
platform win32 -- Python 3.14.10, pytest-9.2.0, pluggy-1.7.0 -- C:\Users\praka\AppData\Local\Python\pythoncore-3.14-64\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.13.0, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 36 items

tests/test_cache_cli.py::TestStepCache::test_cache_key_and_path_are_deterministic PASSED [  2%]
tests/test_cache_cli.py::TestStepCache::test_load_state_returns_none_when_missing PASSED [  5%]
tests/test_cache_cli.py::TestStepCache::test_save_and_load_full_state_with_nested_transcript PASSED [  8%]
tests/test_cache_cli.py::TestStepCache::test_save_and_load_state_without_transcript PASSED [ 11%]
tests/test_cache_cli.py::TestStepCache::test_list_all_ignores_transcript_and_corrupt_files PASSED [ 13%]
tests/test_cache_cli.py::TestStepCache::test_clear_removes_state_and_transcript_cleanly PASSED [ 16%]
tests/test_cache_cli.py::TestCLI::test_help_lists_all_commands PASSED    [ 19%]
tests/test_cache_cli.py::TestCLI::test_version_option PASSED             [ 22%]
tests/test_cache_cli.py::TestCLI::test_clip_command_accepts_all_options PASSED [ 25%]
tests/test_cache_cli.py::TestCLI::test_auto_command_uses_config_and_game_override PASSED [ 27%]
tests/test_cache_cli.py::TestCLI::test_setup_renders_dependencies_gpu_and_hardware_profile_with_overrides PASSED [ 30%]
tests/test_cache_cli.py::TestCLI::test_setup_without_cuda_reports_cpu_fallback PASSED [ 33%]
tests/test_cache_cli.py::TestCLI::test_check_dependency_handles_missing_module_and_missing_cmd PASSED [ 36%]
tests/test_cache_cli.py::TestCLI::test_status_empty_and_populated_cache PASSED [ 38%]
tests/test_config.py::TestHardwareDetection::test_tier_high_vram_rtx_4070_12gb[True-h264_nvenc] PASSED [ 41%]
tests/test_config.py::TestHardwareDetection::test_tier_high_vram_rtx_4070_12gb[False-libx264] PASSED [ 44%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_tensor_rtx_4050_6gb[True-h264_nvenc] PASSED [ 47%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_tensor_rtx_4050_6gb[False-libx264] PASSED [ 50%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_rtx_turing_cc75_detected_as_tensor PASSED [ 52%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_no_tensor_gtx_1660_super_6gb[True-h264_nvenc] PASSED [ 55%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_no_tensor_gtx_1660_super_6gb[False-libx264] PASSED [ 58%]
tests/test_config.py::TestHardwareDetection::test_tier_low_vram_4gb_fallback_to_cpu PASSED [ 61%]
tests/test_config.py::TestHardwareDetection::test_tier_cpu_only_when_cuda_unavailable PASSED [ 63%]
tests/test_config.py::TestHardwareDetection::test_detect_nvenc_support_present_absent_and_missing_ffmpeg PASSED [ 66%]
tests/test_config.py::TestConfigSystem::test_defaults_and_dot_notation_lookup PASSED [ 69%]
tests/test_config.py::TestConfigSystem::test_user_overrides_preserved_over_hardware_auto_config PASSED [ 72%]
tests/test_config.py::TestConfigSystem::test_resolve_hardware_config_cpu_float16_safety_guard PASSED [ 75%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_weight_sum PASSED [ 77%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_resolution PASSED [ 80%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[0] PASSED [ 83%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[-2] PASSED [ 86%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[fast] PASSED [ 88%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[True] PASSED [ 91%]
tests/test_config.py::TestDataModels::test_wildcard_import_exports_all_models PASSED [ 94%]
tests/test_config.py::TestDataModels::test_pipeline_status_is_int_enum_with_numeric_ordering PASSED [ 97%]
tests/test_config.py::TestDataModels::test_all_models_json_round_trip PASSED [ 100%]

======================== 36 passed, 1 warning in 2.99s ========================
```

### 5.3 Live CLI Output (`streamclipper --help`, `setup`, `status`)

```text
=== streamclipper --help ===
Usage: streamclipper [OPTIONS] COMMAND [ARGS]...

  🎮 StreamClipper — AI-powered stream clipping tool

Options:
  --version          Show the version and exit.
  -c, --config PATH  Path to config.yaml
  -v, --verbose      Verbose output
  --help             Show this message and exit.

Commands:
  auto    Auto-discover top streams and clip them all.
  clip    Clip highlights from a stream URL.
  setup   Check dependencies, detect hardware profile, and report system...
  status  Show status of cached/in-progress pipelines.

=== streamclipper setup ===
🔧 Checking system dependencies...
  ✓ yt-dlp
  ✗ FFmpeg — not installed
  ✗ WhisperX — not installed
  ✓ SoundFile
  ✓ OpenCV
  ✓ MoviePy v2
  ✓ Ollama SDK
  ⚠ PyTorch not installed — GPU check skipped
               🖥️ Resolved Hardware Profile (cpu_only)               
┌────────────────────────────┬───────────────────────────┬──────────┐
│ Setting                    │ Active Value              │ Source   │
├────────────────────────────┼───────────────────────────┼──────────┤
│ GPU / VRAM / CC            │ CPU (0.0 GB, CC N/A)      │ detected │
│ Tensor Cores / NVENC       │ Tensor=False, NVENC=False │ detected │
│ transcription.device       │ cpu                       │ auto     │
│ transcription.model_size   │ base                      │ auto     │
│ transcription.compute_type │ int8                      │ auto     │
│ transcription.batch_size   │ 4                         │ auto     │
│ llm.model                  │ qwen3:4b                  │ auto     │
│ video.codec                │ libx264                   │ auto     │
└────────────────────────────┴───────────────────────────┴──────────┘

=== streamclipper status ===
No cached pipelines found.
```

---

## 6. Consolidated Architectural Notes, Proactive Fixes & Reviewer Handoff

### 6.1 Proactive Bug & Edge-Case Fixes vs. Literal Guide Snippets
1. **`.gitignore` Anchoring & Comment Placement (1.1)**: Anchored `/output/` so `src/streamclipper/output/` remains tracked in Git, and placed `# user config (not default)` on its own line above `config.yaml` since `.gitignore` does not support inline comments.
2. **Validation & Serialization Hardening (1.2)**:
   - Rejected `bool` values for `transcription.batch_size` (since `isinstance(True, int)` is `True`) and verified `video.resolution` is a `list`/`tuple` of length 2.
   - Added `PipelineState.from_dict()` so all 10 dataclasses support symmetric JSON round-trips.
   - Preserved the CPU safety guard coercing `float16`/`int8_float16` → `int8` when `transcription.device == "cpu"`.
3. **`_check_dependency()` `UnboundLocalError` Fix (1.3)**: Moved `import subprocess` to module scope in `src/streamclipper/cli.py` (with `timeout=10` and `TimeoutExpired`/`OSError` handling) so missing Python modules with `cmd=None` (e.g., optional `whisperx`) do not trigger `UnboundLocalError` in the `except` clause.
4. **GPU VRAM API Safety (1.2 & 1.3)**: Verified zero occurrences of the invalid `.total_mem` attribute across `src/streamclipper/`; both `config.py` and `cli.py` exclusively read `props.total_memory / 1e9` and `props.major`/`props.minor`.
5. **CLI `--version` Continuity & Resilient `StepCache` (1.3)**: Retained `@click.version_option` from Package 1.1 alongside `--config`/`--verbose`; `StepCache.list_all()` iterates deterministically via `sorted()` and skips malformed JSON files cleanly.

### 6.2 Local Environment Notes for Phase 2+
- **Windows PATH**: `git.exe` is located at `C:\Program Files\Git\cmd`, and Python 3.14.10 + installed console scripts (`streamclipper.exe`, `pytest.exe`) live under `C:\Users\praka\AppData\Local\Python\pythoncore-3.14-64\` and its `Scripts\` directory. Neither was on the default non-interactive shell `PATH`.
- **Optional / External Runtime Dependencies**: `FFmpeg`, `PyTorch` (CUDA), and optional `WhisperX` are not yet installed in this environment (correctly surfaced by `streamclipper setup`). `setuptools>=77` also emits a benign deprecation warning for `license = {text = "MIT"}` in `pyproject.toml`.

### 6.3 Next Step
Run **Prompt 1.R** (Phase 1 Lead Reviewer audit) to produce `reports/phase-1/phase-1-review.md` and merge `phase-1-foundation` into `main` before creating branch `phase-2-core-pipeline`.


---

## 7. Environment & CUDA Setup Verification

### 7.1 Installed Runtime Components (Windows)
| Component | Version / Path | Verification |
| :--- | :--- | :--- |
| **FFmpeg** | `9.0.2-full_build-www.gyan.dev` (`Gyan.FFmpeg` via WinGet, added to User `PATH`: `C:\Users\praka\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin`) | `ass` (`libass`) filter and `h264_nvenc` / `hevc_nvenc` / `av1_nvenc` encoders confirmed |
| **Python `.venv`** | Python `3.12.10` (`D:\GTA6-Clipper\.venv`, `.venv/` and `venv/` added to `.gitignore`) | Dedicated venv for CUDA & WhisperX Windows wheel compatibility |
| **PyTorch (CUDA)** | `torch 2.8.0+cu126`, `torchvision 0.23.0+cu126`, `torchaudio 2.8.0+cu126` | `torch.cuda.is_available() == True`; CUDA tensor op verified on `cuda:0` (`NVIDIA GeForce GTX 1660 SUPER`, `6.44 GB`, CC `7.5`, Driver `610.88`) |
| **WhisperX & Extras** | `whisperx 3.8.6`, `streamclipper 0.1.0` (`-e ".[dev,mcp]"`) | `pip check` reported `No broken requirements found.` *(Note: `whisperx 3.8.6` pins `torch==2.8.0`, so PyTorch `cu126` wheels were installed instead of `cu124` to preserve CUDA support without dependency conflicts.)* |

### 7.2 Verbatim `streamclipper setup` Output (`.venv`)
```text
🔧 Checking system dependencies...
  ✓ yt-dlp
  ✓ FFmpeg
  ✓ WhisperX
  ✓ SoundFile
  ✓ OpenCV
  ✓ MoviePy v2
  ✓ Ollama SDK
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
```

### 7.3 Verbatim `pytest -v` Output (`.venv`)
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\GTA6-Clipper\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\GTA6-Clipper
configfile: pyproject.toml
plugins: anyio-4.16.0, platformdirs-4.14.7, asyncio-1.5.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 36 items

tests/test_cache_cli.py::TestStepCache::test_cache_key_and_path_are_deterministic PASSED [  2%]
tests/test_cache_cli.py::TestStepCache::test_load_state_returns_none_when_missing PASSED [  5%]
tests/test_cache_cli.py::TestStepCache::test_save_and_load_full_state_with_nested_transcript PASSED [  8%]
tests/test_cache_cli.py::TestStepCache::test_save_and_load_state_without_transcript PASSED [ 11%]
tests/test_cache_cli.py::TestStepCache::test_list_all_ignores_transcript_and_corrupt_files PASSED [ 13%]
tests/test_cache_cli.py::TestStepCache::test_clear_removes_state_and_transcript_cleanly PASSED [ 16%]
tests/test_cache_cli.py::TestCLI::test_help_lists_all_commands PASSED    [ 19%]
tests/test_cache_cli.py::TestCLI::test_version_option PASSED             [ 22%]
tests/test_cache_cli.py::TestCLI::test_clip_command_accepts_all_options PASSED [ 25%]
tests/test_cache_cli.py::TestCLI::test_auto_command_uses_config_and_game_override PASSED [ 27%]
tests/test_cache_cli.py::TestCLI::test_setup_renders_dependencies_gpu_and_hardware_profile_with_overrides PASSED [ 30%]
tests/test_cache_cli.py::TestCLI::test_setup_without_cuda_reports_cpu_fallback PASSED [ 33%]
tests/test_cache_cli.py::TestCLI::test_check_dependency_handles_missing_module_and_missing_cmd PASSED [ 36%]
tests/test_cache_cli.py::TestCLI::test_status_empty_and_populated_cache PASSED [ 38%]
tests/test_config.py::TestHardwareDetection::test_tier_high_vram_rtx_4070_12gb[True-h264_nvenc] PASSED [ 41%]
tests/test_config.py::TestHardwareDetection::test_tier_high_vram_rtx_4070_12gb[False-libx264] PASSED [ 44%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_tensor_rtx_4050_6gb[True-h264_nvenc] PASSED [ 47%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_tensor_rtx_4050_6gb[False-libx264] PASSED [ 50%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_rtx_turing_cc75_detected_as_tensor PASSED [ 52%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_no_tensor_gtx_1660_super_6gb[True-h264_nvenc] PASSED [ 55%]
tests/test_config.py::TestHardwareDetection::test_tier_mid_vram_no_tensor_gtx_1660_super_6gb[False-libx264] PASSED [ 58%]
tests/test_config.py::TestHardwareDetection::test_tier_low_vram_4gb_fallback_to_cpu PASSED [ 61%]
tests/test_config.py::TestHardwareDetection::test_tier_cpu_only_when_cuda_unavailable PASSED [ 63%]
tests/test_config.py::TestHardwareDetection::test_detect_nvenc_support_present_absent_and_missing_ffmpeg PASSED [ 66%]
tests/test_config.py::TestConfigSystem::test_defaults_and_dot_notation_lookup PASSED [ 69%]
tests/test_config.py::TestConfigSystem::test_user_overrides_preserved_over_hardware_auto_config PASSED [ 72%]
tests/test_config.py::TestConfigSystem::test_resolve_hardware_config_cpu_float16_safety_guard PASSED [ 75%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_weight_sum PASSED [ 77%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_resolution PASSED [ 80%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[0] PASSED [ 83%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[-2] PASSED [ 86%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[fast] PASSED [ 88%]
tests/test_config.py::TestConfigSystem::test_validation_rejects_invalid_batch_size[True] PASSED [ 91%]
tests/test_config.py::TestDataModels::test_wildcard_import_exports_all_models PASSED [ 94%]
tests/test_config.py::TestDataModels::test_pipeline_status_is_int_enum_with_numeric_ordering PASSED [ 97%]
tests/test_config.py::TestDataModels::test_all_models_json_round_trip PASSED [ 100%]

============================= 36 passed in 4.93s ==============================
```
