# StreamClipper — Agent Execution Prompts & Audit Logging Runbook (v2.2 — Patched + Auto-Hardware + YPP Transformative Engine)

> **Current Milestone Status:**  
> **Phase 1 (`phase-1-foundation`: Packages `1.1`, `1.2`, `1.3` + Review `1.R`) is ✅ COMPLETE (`36/36` unit tests passing) and MERGED into `main`.**  
> **Start execution at Prompt 2.1 (`phase-2-core-pipeline`).** Every prompt below enforces `AGENTS.md`, `.venv\Scripts\pytest.exe`, branch discipline, all v2.2 Windows & auto-hardware guardrails, and the YPP Transformative Narrative & Story Engine (adaptive `moment_type` commentary, conditional multi-beat story cuts, spike punch-in zoom, live chat ASS ticker, and pluggable Kokoro/Edge-TTS/ElevenLabs/custom-mic audio ducking).

## 1. Quick Reference Matrix: Phases, Packages, Branches & Models

| Phase | Branch Name | Package | Package Name | Status | Assigned Model | Report Path |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **Phase 1** | `phase-1-foundation` | **1.1** | Project Scaffolding | ✅ Merged | Gemini 3.8 (high) | `reports/phase-1/package-1.1-report.md` |
| **Phase 1** | `phase-1-foundation` | **1.2** | Data Models & 5-Tier Hardware-Aware Config | ✅ Merged | Gemini 3.8 (high) | `reports/phase-1/package-1.2-report.md` |
| **Phase 1** | `phase-1-foundation` | **1.3** | CLI Skeleton (Hardware Profile) & Step Cache | ✅ Merged | Gemini 3.8 (high) | `reports/phase-1/package-1.3-report.md` |
| **Phase 1** | `phase-1-foundation` -> `main` | **1.R** | Phase 1 Review & Merge to `main` | ✅ Merged | Claude Opus 4.6 | `reports/phase-1/phase-1-review.md` |
| **Phase 2** | `phase-2-core-pipeline` | **2.1** | Audio Downloader (`yt-dlp` 16kHz Mono) | ⏳ Next | Gemini 3.8 (high) | `reports/phase-2/package-2.1-report.md` |
| **Phase 2** | `phase-2-core-pipeline` | **2.2** | Video Segment Downloader (Single + Multi-Beat) | Pending | Gemini 3.8 (high) | `reports/phase-2/package-2.2-report.md` |
| **Phase 2** | `phase-2-core-pipeline` | **2.3** | WhisperX Transcriber (Auto-Hardware + VRAM Cleanup) | Pending | Gemini 3.8 (high) | `reports/phase-2/package-2.3-report.md` |
| **Phase 2** | `phase-2-core-pipeline` | **2.4** | Loudness Spike Detector (Streamed RMS + `peak_time`) | Pending | Gemini 3.8 (high) | `reports/phase-2/package-2.4-report.md` |
| **Phase 2** | `phase-2-core-pipeline` | **2.5** | Chat Scraper (`yt-dlp`) & Velocity (`sample_messages`) | Pending | Gemini 3.8 (high) | `reports/phase-2/package-2.5-report.md` |
| **Phase 2** | `phase-2-core-pipeline` | **2.6** | Stream Discovery | Pending | Gemini 3.8 (high) | `reports/phase-2/package-2.6-report.md` |
| **Phase 3** | `phase-3-ai-scoring` | **3.1** | Scorer Interface & YPP Adaptive Narrative Prompts | Pending | Claude Opus 4.6 | `reports/phase-3/package-3.1-report.md` |
| **Phase 3** | `phase-3-ai-scoring` | **3.2** | Ollama Scorer (+ Adaptive Narrative Generator) | Pending | Gemini 3.8 (high) | `reports/phase-3/package-3.2-report.md` |
| **Phase 3** | `phase-3-ai-scoring` | **3.3** | Pipeline Orchestrator (`FAILED` Recovery + `--mode`/`--voice-intro`) | Pending | Claude Opus 4.6 | `reports/phase-3/package-3.3-report.md` |
| **Phase 4** | `phase-4-video-assembly` | **4.1** | Facecam Detector (OpenCV) | Pending | Gemini 3.8 (high) | `reports/phase-4/package-4.1-report.md` |
| **Phase 4** | `phase-4-video-assembly` | **4.2** | 9:16 Composer (MoviePy v2 + Story Beats + 1.2x Zoom) | Pending | Gemini 3.8 (high) | `reports/phase-4/package-4.2-report.md` |
| **Phase 4** | `phase-4-video-assembly` | **4.3** | Hormozi Captions + Chat Ticker (`.ass` + Windows `cwd` FFmpeg / NVENC) | Pending | Claude Opus 4.6 | `reports/phase-4/package-4.3-report.md` |
| **Phase 4** | `phase-4-video-assembly` | **4.4** | Hook Title & Adaptive Context Badge Overlay | Pending | Gemini 3.8 (high) | `reports/phase-4/package-4.4-report.md` |
| **Phase 4** | `phase-4-video-assembly` | **4.5** | YPP Narrator (Kokoro/Edge-TTS/ElevenLabs/Mic) & Dynamic Audio Ducking | Pending | Gemini 3.8 (high) | `reports/phase-4/package-4.5-report.md` |
| **Phase 4** | `phase-4-video-assembly` | **4.6** | Exporter (`crf=18` Temp + Handle Cleanup) & File Organizer | Pending | Gemini 3.8 (high) | `reports/phase-4/package-4.6-report.md` |
| **Phase 5** | `phase-5-mcp-server` | **5.1** | FastMCP Server Implementation (incl. YPP Transformative Params) | Pending | Claude Opus 4.6 | `reports/phase-5/package-5.1-report.md` |
| **Phase 5** | `phase-5-mcp-server` | **5.2** | README & Setup Guide | Pending | Gemini 3.8 (high) | `reports/phase-5/package-5.2-report.md` |

## 2. Universal Boilerplate Prompts (Parameterized Templates)

### 2A. Universal Package Implementation Prompt Template

Use this template if you ever want to run a custom package or re-run a single package by changing only the 4 variables at the top:

```markdown
You are implementing **Package {PACKAGE_ID}** of **Phase {PHASE_NUM}** from `streamclipper-implementation-guide-v2.md`.

### 1. Context & Source of Truth
- Read `AGENTS.md` and `streamclipper-implementation-guide-v2.md` first, specifically the **Dynamic Hardware Auto-Configuration Policy**, **Mandatory v2.2 Guardrails**, and **Package {PACKAGE_ID}**.
- Always execute Python and tests via `D:\GTA6-Clipper\.venv\Scripts\python.exe` and `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`.
- Remember that **Phase 1 (`1.1–1.3`) is already complete and merged into `main`** — never break the 36 existing Phase 1 unit tests.
- If previous package reports exist in `reports/`, review the latest report to understand current repository state.

### 2. Git Branch & Workspace Discipline
- Target branch: `{BRANCH_NAME}`
- If this is the first package of Phase {PHASE_NUM} (Package {PHASE_NUM}.1): run `git checkout main && git checkout -b {BRANCH_NAME}`.
- Otherwise: run `git checkout {BRANCH_NAME}` and verify `git status` is clean before modifying files.
- Implement ONLY the files and tests scoped to **Package {PACKAGE_ID}**. Do not jump ahead to future packages.

### 3. Implementation & Verification
- Implement all files specified in **Package {PACKAGE_ID}** following all v2.2 architectural & YPP guardrails in `AGENTS.md` (`IntEnum` `PipelineStatus`, `FAILED` retry recovery, recursive `SerializableModel`, 5-tier `HardwareProfile` auto-tuning, `setuptools.build_meta`, MoviePy v2 API with aspect-preserving crop, conditional story beats, spike zoom, and handle cleanup, `yt-dlp` 16kHz mono WAV + non-negative `live_chat` + `sample_messages`, `librosa.stream(..., fill_value=0.0)` + `peak_time`, WhisperX 2-stage VRAM cleanup, Windows-safe FFmpeg `cwd` `ass` filter with `Hormozi` + `ChatTicker` and `h264_nvenc` → `libx264` fallback, pluggable TTS/ElevenLabs/custom-mic narrator with dynamic audio ducking, and `FastMCP`).
- Write or update unit tests in `tests/` covering every acceptance criterion for **Package {PACKAGE_ID}** and run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` to confirm 100% pass rate.

### 4. Mandatory Audit Report & Commit
Before finishing your turn, create a detailed implementation report at `reports/phase-{PHASE_NUM}/package-{PACKAGE_ID}-report.md` (and also output it in chat) containing:
1. **Metadata**: Package ID, Phase, Branch name, Commit hash, Timestamp, and Model used.
2. **Files Created / Modified**: Exact file paths, line counts, and summary of functions/classes implemented.
3. **Acceptance Criteria Verification**: A checklist table mapping every acceptance criterion from `streamclipper-implementation-guide-v2.md` for Package {PACKAGE_ID} to the exact test or command output that proved it passed.
4. **Test & CLI Output**: Verbatim terminal output of `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` and any CLI verification commands.
5. **Architectural Notes / Edge Cases / Reviewer Handoff**: Any edge cases handled, assumptions made, or items for the reviewer to inspect.

Finally, stage all implementation files, tests, and the report file, and commit them to `{BRANCH_NAME}` using the exact commit message from `streamclipper-implementation-guide-v2.md`.
```

### 2B. Universal Phase Review & Merge Prompt Template (For Opus / Reviewer)

Use this prompt at the end of each phase before merging `{BRANCH_NAME}` into `main`:

```markdown
You are the Lead Architect & Code Reviewer reviewing **Phase {PHASE_NUM}** on branch `{BRANCH_NAME}` against `AGENTS.md` and `streamclipper-implementation-guide-v2.md`.

1. Checkout `{BRANCH_NAME}` and read all package reports in `reports/phase-{PHASE_NUM}/`.
2. Run `git log main..{BRANCH_NAME} --stat` and `git diff main...{BRANCH_NAME}` to inspect all code changes across Phase {PHASE_NUM}.
3. Run the full test suite (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`) and verify zero regressions.
4. Audit the implementation against all v2.2 guardrails in `AGENTS.md` and `streamclipper-implementation-guide-v2.md`:
   - Confirm no v1 regressions exist (no string `PipelineStatus`, no `pytchat`, no MoviePy v1 `.crop()`/`.resize()`, no `total_mem` typo, no low-level `mcp.server.Server`).
   - Confirm Windows/edge-case patches, 5-tier `HardwareProfile` / NVENC rules, and YPP Transformative Engine features are implemented cleanly.
   - Confirm every acceptance criterion across all packages in Phase {PHASE_NUM} is met.
5. Write a Phase Review Report to `reports/phase-{PHASE_NUM}/phase-{PHASE_NUM}-review.md` summarizing the audit results, test coverage, and approval status.
6. If approved, commit the review report, switch to `main` (`git checkout main`), merge `{BRANCH_NAME}` (`git merge --no-ff {BRANCH_NAME} -m "chore: merge {BRANCH_NAME} after review"`), and report readiness for the next phase.
```

## 3. Phase 1 Prompts — Foundation & Scaffolding (`phase-1-foundation`) — ✅ COMPLETED & MERGED INTO `main`

> **Status:** Packages `1.1`, `1.2`, `1.3`, and Lead Reviewer audit `1.R` (`reports/phase-1/phase-1-review.md`) are **✅ COMPLETE (`36/36` tests passing)** and **already merged into `main`**. Do **NOT** re-run Phase 1 prompts; start directly at **Section 4 (Prompt 2.1)** below. *(Historical Phase 1 prompts kept collapsed/archived below for reference.)*

### Prompt 1.1 — Package 1.1: Project Scaffolding
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-1-foundation` (Create from `main`)

```markdown
You are implementing **Package 1.1 — Project Scaffolding** (Phase 1) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Setup
- Check your current git state. Ensure `main` exists (initialize git with `git init -b main` and make an initial commit if the repository is not yet initialized).
- Create and switch to the Phase 1 branch:
  `git checkout -b phase-1-foundation`

### 2. What to Build (Strictly Scoped to Package 1.1)
Read **Package 1.1** in `streamclipper-implementation-guide-v2.md` and create:
1. `pyproject.toml`:
   - MUST use `build-backend = "setuptools.build_meta"` (never `setuptools.backends._legacy:_Backend`).
   - Include dependencies from the v2 guide (`yt-dlp>=2024.8.0`, `faster-whisper>=1.0.0`, `librosa>=0.10.0`, `soundfile>=0.12.1`, `opencv-python>=4.8.0`, `moviepy>=2.0.0`, `ollama>=0.3.0`, `numpy>=1.24.0`, `scipy>=1.11.0`, `pandas>=2.0.0`, `Pillow>=10.0.0`, `pyyaml>=6.0`, `click>=8.1.0`, `rich>=13.0.0`, `tqdm>=4.65.0`). Do NOT include `pytchat`.
   - Include both entry points under `[project.scripts]`: `streamclipper = "streamclipper.cli:cli"` and `streamclipper-mcp = "streamclipper.mcp_server:main"`.
2. Full directory structure under `src/streamclipper/` (`discovery/`, `download/`, `analysis/`, `scoring/`, `assembly/`, `output/`, `mcp_server/`, `prompts/`), `tests/`, `reports/phase-1/`, and `output/` with all `__init__.py` files.
3. `src/streamclipper/__init__.py` with `__version__ = "0.1.0"`.
4. Minimal placeholder `src/streamclipper/cli.py` and `src/streamclipper/mcp_server/__init__.py` so `pip install -e .` and `streamclipper --help` succeed immediately.
5. `.gitignore` covering Python artifacts, `output/`, `temp/`, `*.wav`, `*.mp3`, `*.mp4`, `*.ass`, `*.live_chat.json`, `config.yaml`, and `.cache/`.

### 3. Verification
- Run `pip install -e .` (or `pip install --no-build-isolation -e .` if offline) and verify installation succeeds.
- Run `streamclipper --help` and verify it prints the CLI help text.

### 4. Mandatory Completion Report & Commit
Write a comprehensive implementation report to `reports/phase-1/package-1.1-report.md` (and display a summary in chat) documenting:
- Branch name (`phase-1-foundation`) and git status
- Complete list of created files and directories
- Verification output for `pip install -e .` and `streamclipper --help`
- Acceptance criteria checklist (all 4 items from Package 1.1 verified)
- Any environment notes for the reviewer

Stage all files (including `reports/phase-1/package-1.1-report.md`) and commit with:
`git commit -m "feat: project scaffolding, pyproject.toml, directory structure"`
```

### Prompt 1.2 — Package 1.2: Data Models & Dynamic Hardware-Aware Config System
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-1-foundation`

```markdown
You are implementing **Package 1.2 — Data Models & Dynamic Hardware-Aware Config System** (Phase 1) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-1-foundation` (`git checkout phase-1-foundation`).
- Read `reports/phase-1/package-1.1-report.md` to confirm Package 1.1 is complete.

### 2. What to Build (Strictly Scoped to Package 1.2)
Read the **Dynamic Hardware Auto-Configuration Policy**, **Data Models** section, and **Package 1.2** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/models.py`:
   - Implement `Platform`, `SignalType`, and `PipelineStatus(IntEnum)` (`PENDING = 1` through `COMPLETE = 7`, `FAILED = 99`). `PipelineStatus` MUST be an `IntEnum` so numeric stage comparisons work properly.
   - Implement `SerializableModel` with recursive `_serialize_value()` (iterating `dataclasses.fields(val)` with `is_dataclass(val) and not isinstance(val, type)`), handling `Path`, `Enum`, `list`, and `dict`, plus `from_dict()` classmethods on `StreamInfo`, `TimeRange`, `WordTimestamp`, `TranscriptSegment`, `TranscriptionResult`, `Highlight`, `FacecamRegion`, `ClipMetadata`, `ClipResult`, and `PipelineState`.
2. `config.default.yaml`:
   - Exact default YAML configuration from Package 1.2 in `streamclipper-implementation-guide-v2.md`, with `"auto"` defaults for `transcription.model_size`, `transcription.device`, `transcription.compute_type`, `transcription.batch_size`, `llm.model`, and `video.codec`.
3. `src/streamclipper/config.py`:
   - Implement `HardwareProfile`, `_detect_nvenc_support()`, `detect_hardware_profile()`, and `resolve_hardware_config()` using `torch.cuda.get_device_properties(0)` (`total_memory`, `major`, `minor`, `name`):
     - `>= 10 GB` VRAM → `cuda`, `large-v2`, `float16`, `batch_size=16`, `llm.model="qwen3:8b"`, `video.codec="h264_nvenc"` (else `libx264`)
     - `5.5–9.9 GB` VRAM + Tensor Cores (`major >= 8` or `"RTX"` in name, e.g., RTX 4050 6GB CC 8.9) → `cuda`, `large-v2`, `int8_float16`, `batch_size=8`, `llm.model="qwen3:8b"`, `video.codec="h264_nvenc"` (else `libx264`)
     - `5.5–9.9 GB` VRAM without Tensor Cores (`major < 8` and non-RTX, e.g., GTX 1660 SUPER 6GB CC 7.5) → `cuda`, `medium`, `int8`, `batch_size=4`, `llm.model="qwen3:4b"`, `video.codec="h264_nvenc"` (else `libx264`)
     - CPU / `< 5.5 GB` VRAM → `cpu`, `base`, `int8`, `batch_size=4`, `llm.model="qwen3:4b"`, `video.codec="libx264"`
   - Implement `Config` with default loading, deep merging of user `config.yaml` (preserving any explicit non-`"auto"` overrides), `is_auto_resolved(dotted_key)`, `hardware_profile` attribute, dot-notation `.get(...)`, and validation (signal weights sum to 1.0, `video.resolution` length 2, positive int `batch_size`).
4. `tests/test_config.py`:
   - Unit tests verifying:
     - `Config()` defaults, user override merging, dot-notation lookup, and invalid weight sum rejection.
     - All 4 `detect_hardware_profile()` tiers via mocked `torch.cuda.get_device_properties(0)` (12GB RTX 3060/4070, 6GB RTX 4050 CC 8.9, 6GB GTX 1660 SUPER CC 7.5, and CPU / 4GB fallback) plus NVENC present vs. absent.
     - Explicit user overrides in `config.yaml` are NOT overwritten by `resolve_hardware_config()`.
     - `IntEnum` `PipelineStatus` comparisons (`PipelineStatus.PENDING < PipelineStatus.ANALYZING`) and full JSON round-trip serialization/deserialization for all dataclasses (including nested `TranscriptionResult` → `TranscriptSegment` → `WordTimestamp`).

### 3. Verification
- Run `pytest tests/test_config.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-1/package-1.2-report.md` (and show it in chat) covering:
- Branch (`phase-1-foundation`), files created/modified, and line counts
- Acceptance criteria verification table for all 7 criteria in Package 1.2
- Verbatim `pytest tests/test_config.py -v` output
- Reviewer notes on hardware auto-tuning tiers, override precedence, and serialization edge cases

Stage all files (including `reports/phase-1/package-1.2-report.md`) and commit with:
`git commit -m "feat: data models, config system with dynamic HardwareProfile auto-tuning"`
```

### Prompt 1.3 — Package 1.3: CLI Skeleton & Step Cache
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-1-foundation`

```markdown
You are implementing **Package 1.3 — CLI Skeleton & Step Cache** (Phase 1) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-1-foundation` (`git checkout phase-1-foundation`).
- Read `reports/phase-1/package-1.2-report.md` to confirm Package 1.2 is complete.

### 2. What to Build (Strictly Scoped to Package 1.3)
Read **Package 1.3** in `streamclipper-implementation-guide-v2.md` and implement:
1. `src/streamclipper/cli.py`:
   - Implement `cli` group (`--config`, `--verbose`) and commands: `clip` (with `--max-clips`, `--game`, `--streamer`, `--interactive/--no-interactive`), `auto`, `setup`, and `status`.
   - In `_check_gpu()`, use `torch.cuda.get_device_properties(0).total_memory / 1e9` and compute capability (`major.minor`) (NEVER `.total_mem`).
   - In `setup`, check `yt-dlp`, `FFmpeg`, `WhisperX`, `SoundFile`, `OpenCV`, `MoviePy v2`, `Ollama SDK`, GPU status, AND render the resolved `HardwareProfile` table via `_render_hardware_profile(config)` showing each active value (`device`, `model_size`, `compute_type`, `batch_size`, `llm.model`, `video.codec`) and whether it was `auto`-resolved or a `user override`.
2. `src/streamclipper/cache.py`:
   - Implement `StepCache` with `get_cache_key`, `get_cache_path`, `save_state`, `load_state`, `clear`, and `list_all`.
   - Persist `status` as `int(state.status)` and restore it via `PipelineStatus(int(data["status"]))`.
   - Save and restore full `TranscriptionResult` (`{key}_transcript.json`), `loudness_spikes`, `chat_spikes`, `highlights`, `selected_highlights`, and `clips`.
3. Add unit tests in `tests/test_config.py` (or `tests/test_cache_cli.py`) testing `StepCache` save/load/list/clear with nested `TranscriptionResult` and `Click` `CliRunner` tests for `--help`, `setup` (verifying `HardwareProfile` table output), `status`, and `clip`.

### 3. Verification
- Run `pytest -v` and verify all tests pass.
- Run `streamclipper --help`, `streamclipper setup`, and `streamclipper status`.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-1/package-1.3-report.md` (and show it in chat) covering:
- Branch (`phase-1-foundation`), files created/modified, and line counts
- Acceptance criteria verification table for all criteria in Package 1.3
- Verbatim `pytest -v` and CLI command outputs
- Readiness summary for the Phase 1 Opus review

Stage all files (including `reports/phase-1/package-1.3-report.md`) and commit with:
`git commit -m "feat: CLI skeleton with hardware profile display and step cache for resume"`
```

### Prompt 1.R — Phase 1 Review & Merge (`phase-1-foundation` -> `main`)
**Assigned Model:** Claude Opus 4.6 | **Branch:** `phase-1-foundation` -> `main`

```markdown
You are the Lead Reviewer auditing **Phase 1 (`phase-1-foundation`)** against `streamclipper-implementation-guide-v2.md`.

1. Read `reports/phase-1/package-1.1-report.md`, `package-1.2-report.md`, and `package-1.3-report.md`.
2. Inspect `git diff main...phase-1-foundation` and run `pytest -v`.
3. Verify:
   - `pyproject.toml` uses `setuptools.build_meta`, `moviepy>=2.0.0`, `soundfile>=0.12.1`, and omits `pytchat`.
   - `PipelineStatus` is an `IntEnum` (`PENDING = 1` through `COMPLETE = 7`, `FAILED = 99`).
   - `config.py` implements `HardwareProfile` and `resolve_hardware_config()` across all 4 tiers (including RTX 4050 6GB vs. GTX 1660 SUPER 6GB) while preserving explicit user overrides.
   - `cli.py` uses `.total_memory`, displays the resolved `HardwareProfile` table in `setup`, and `StepCache` round-trips `TranscriptionResult` and spikes without error.
4. Write the phase audit report to `reports/phase-1/phase-1-review.md`, commit it on `phase-1-foundation`, then merge into `main`:
   `git checkout main && git merge --no-ff phase-1-foundation -m "chore: merge phase-1-foundation after review"`
```

## 4. Phase 2 Prompts — Core Pipeline: Download & Analysis (`phase-2-core-pipeline`)

### Prompt 2.1 — Package 2.1: Audio Downloader
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-2-core-pipeline` (Create from `main`)

```markdown
You are implementing **Package 2.1 — Audio Downloader** (Phase 2) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Setup
- Ensure Phase 1 is merged into `main`, then create and switch to the Phase 2 branch:
  `git checkout main && git checkout -b phase-2-core-pipeline`

### 2. What to Build (Strictly Scoped to Package 2.1)
Read **Package 2.1** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/download/audio.py`:
   - Implement `download_audio(url, output_dir, progress_callback=None) -> tuple[Path, StreamInfo]` and `extract_stream_info(url) -> StreamInfo` using the `yt_dlp.YoutubeDL` Python API.
   - Configure `FFmpegExtractAudio` with `preferredcodec: "wav"` AND `postprocessor_args: {"extractaudio": ["-ar", "16000", "-ac", "1"]}` so extracted WAV files are **16kHz mono PCM** (~460 MB for 4 hours, matching WhisperX's native sample rate).
   - Skip downloading if `{output_dir}/{video_id}.wav` already exists on disk (still returning `(audio_path, stream_info)`).
   - Catch `yt_dlp.utils.DownloadError` and raise descriptive exceptions for private, deleted, age-restricted, or invalid URLs.
2. `tests/test_download.py`:
   - Unit tests mocking `yt_dlp.YoutubeDL` to verify: 16kHz mono `postprocessor_args` configuration, metadata mapping to `StreamInfo`, cache skip when WAV file exists, progress hook forwarding, and descriptive error handling on `DownloadError`.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_download.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-2/package-2.1-report.md` (and show it in chat) covering:
- Branch (`phase-2-core-pipeline`), files created/modified, and line counts
- Acceptance criteria verification table for all 4 criteria in Package 2.1
- Verbatim `pytest tests/test_download.py -v` output
- Notes on `yt-dlp` options and error handling

Stage all files (including `reports/phase-2/package-2.1-report.md`) and commit with:
`git commit -m "feat: 16kHz mono audio downloader using yt-dlp"`
```

### Prompt 2.2 — Package 2.2: Video Segment Downloader
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-2-core-pipeline`

```markdown
You are implementing **Package 2.2 — Video Segment Downloader** (Phase 2) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-2-core-pipeline` (`git checkout phase-2-core-pipeline`).
- Read `reports/phase-2/package-2.1-report.md` to confirm Package 2.1 is complete.

### 2. What to Build (Strictly Scoped to Package 2.2)
Read **Package 2.2** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/download/video.py`:
   - Implement `download_segment(url, start_seconds, end_seconds, output_dir, buffer_seconds=3.0) -> Path` and `download_multiple_segments(url, segments, output_dir, buffer_seconds=3.0) -> list[Path]`.
   - Clamp `actual_start = max(0.0, start_seconds - buffer_seconds)` and `actual_end = end_seconds + buffer_seconds`.
   - Use `yt_dlp.utils.download_range_func(None, [(actual_start, actual_end)])` with `force_keyframes_at_cuts: True` and `format: "bestvideo[height<=1080]+bestaudio/best[height<=1080]"`, `merge_output_format: "mp4"`.
   - Save to `{output_dir}/{video_id}_{int(start_seconds)}_{int(end_seconds)}.mp4` and skip download if the file already exists.
2. Expand `tests/test_download.py`:
   - Add unit tests verifying `download_segment` buffer padding (including clamping at `0.0`), `download_ranges` and `force_keyframes_at_cuts` options, cache skip when segment MP4 exists, and `download_multiple_segments`.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_download.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-2/package-2.2-report.md` (and show it in chat) covering:
- Branch (`phase-2-core-pipeline`), files created/modified, and line counts
- Acceptance criteria verification table for all 4 criteria in Package 2.2
- Verbatim `pytest tests/test_download.py -v` output

Stage all files (including `reports/phase-2/package-2.2-report.md`) and commit with:
`git commit -m "feat: video segment downloader for specific time ranges"`
```

### Prompt 2.3 — Package 2.3: WhisperX Transcriber
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-2-core-pipeline`

```markdown
You are implementing **Package 2.3 — WhisperX Transcriber** (Phase 2) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-2-core-pipeline` (`git checkout phase-2-core-pipeline`).
- Read `reports/phase-2/package-2.2-report.md` to confirm Package 2.2 is complete.

### 2. What to Build (Strictly Scoped to Package 2.3)
Read **Package 2.3** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/analysis/transcriber.py`:
   - Implement `Transcriber(config)` reading the hardware-resolved `transcription.device`, `transcription.model_size`, `transcription.compute_type`, and `transcription.batch_size` from `Config` (logging the active `config.hardware_profile` tier and resolved parameters).
   - Enforce safety guard: if `device == "cpu"` and `compute_type in ("float16", "int8_float16")`, automatically downgrade `compute_type` to `"int8"`.
   - Implement `transcribe(audio_path) -> TranscriptionResult`, `_transcribe_whisperx(audio_path)`, `_transcribe_faster_whisper(audio_path)`, `save_transcript(result, output_path)`, and `load_transcript(path)`.
   - **CRITICAL VRAM CLEANUP**: In `_transcribe_whisperx()`, explicitly `del model`, run `gc.collect()`, and call `torch.cuda.empty_cache()` BEFORE loading the wav2vec2 alignment model `model_a`, and again `del model_a, audio`, `gc.collect()`, `torch.cuda.empty_cache()` before returning. Do the same model cleanup in `_transcribe_faster_whisper()`.
   - Gracefully fall back to `_transcribe_faster_whisper()` (using `faster_whisper.WhisperModel` with `word_timestamps=True`) if `whisperx` is not installed.
2. `tests/test_transcriber.py`:
   - Unit tests with mocked `whisperx` and `faster_whisper` verifying:
     - Resolved `HardwareProfile` settings (e.g., RTX 4050 `large-v2`/`int8_float16`/`batch_size=8` vs. GTX 1660 SUPER `medium`/`int8`/`batch_size=4`) are passed cleanly to `whisperx.load_model()` and `model.transcribe()`.
     - Word-level timestamp conversion to `TranscriptionResult`.
     - Automatic CPU `int8` compute_type safety fallback.
     - `faster-whisper` fallback when `whisperx` raises `ImportError`.
     - Two-stage VRAM cleanup (`torch.cuda.empty_cache()` called twice in WhisperX path).
     - JSON transcript caching (`save_transcript` / `load_transcript`).

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_transcriber.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-2/package-2.3-report.md` (and show it in chat) covering:
- Branch (`phase-2-core-pipeline`), files created/modified, and line counts
- Acceptance criteria verification table for all 6 criteria in Package 2.3
- Verbatim `pytest tests/test_transcriber.py -v` output
- Explanation of hardware profile integration and two-stage VRAM cleanup logic

Stage all files (including `reports/phase-2/package-2.3-report.md`) and commit with:
`git commit -m "feat: WhisperX transcription with hardware auto-tuning and VRAM cleanup"`
```

### Prompt 2.4 — Package 2.4: Loudness Spike Detector
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-2-core-pipeline`

```markdown
You are implementing **Package 2.4 — Loudness Spike Detector** (Phase 2) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-2-core-pipeline` (`git checkout phase-2-core-pipeline`).
- Read `reports/phase-2/package-2.3-report.md` to confirm Package 2.3 is complete.

### 2. What to Build (Strictly Scoped to Package 2.4)
Read **Package 2.4** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/analysis/loudness.py`:
   - Implement `LoudnessDetector(config)` with `detect(audio_path) -> list[TimeRange]`, `_compute_rms_db(audio_path)`, `_find_spikes(rms_db, hop_duration)`, and `_merge_nearby(spikes)`.
   - **CRITICAL MEMORY & TAIL-BLOCK GUARDRAIL**: Use `soundfile.info()` and `librosa.stream(..., block_length=120, frame_length=hop_length*2, hop_length=hop_length, mono=True, fill_value=0.0)` with `center=False` instead of `librosa.load()`. Always pass `fill_value=0.0` (and skip any block shorter than `frame_length`) so non-aligned tail blocks never raise `ParameterError`.
   - **YPP SPIKE-ZOOM METADATA**: Record the exact peak timestamp inside `TimeRange.metadata["peak_time"]` (e.g., `float(peak_idx * hop_duration)`). When `_merge_nearby()` merges ranges separated by `<= self.merge_gap` seconds, retain both the maximum score AND the `peak_time` of the highest-scoring spike so Phase 4 (`composer.py`) can trigger a `1.20x` punch-in zoom!
   - Normalize spike scores to a `0.0 - 10.0` scale based on peak height above threshold and expand each spike window appropriately.
2. `tests/test_loudness.py`:
   - Generate a synthetic WAV file in `tmp_path` (including an odd/non-multiple duration such as 61.3 seconds to exercise the tail-block guard, with quiet white noise and 2 injected high-amplitude sine-wave bursts) and verify `LoudnessDetector.detect()` accurately finds the bursts, populates `metadata["peak_time"]`, merges nearby bursts within `merge_gap_seconds` while keeping the loudest peak's `peak_time`, and assigns `SignalType.LOUDNESS`.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_loudness.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-2/package-2.4-report.md` (and show it in chat) covering:
- Branch (`phase-2-core-pipeline`), files created/modified, and line counts
- Acceptance criteria verification table for all criteria in Package 2.4
- Verbatim `pytest tests/test_loudness.py -v` output

Stage all files (including `reports/phase-2/package-2.4-report.md`) and commit with:
`git commit -m "feat: streamed loudness spike detection using librosa and soundfile"`
```

### Prompt 2.5 — Package 2.5: Chat Scraper & Velocity Analyzer
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-2-core-pipeline`

```markdown
You are implementing **Package 2.5 — Chat Scraper & Velocity Analyzer** (Phase 2) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-2-core-pipeline` (`git checkout phase-2-core-pipeline`).
- Read `reports/phase-2/package-2.4-report.md` to confirm Package 2.4 is complete.

### 2. What to Build (Strictly Scoped to Package 2.5)
Read **Package 2.5** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/download/chat.py`:
   - Implement `ChatScraper` using `yt_dlp.YoutubeDL` (`skip_download=True`, `writesubtitles=True`, `subtitleslangs=["live_chat"]`) to download `{video_id}.live_chat.json` and parse YouTube's JSONLines `replayChatItemAction` records (supporting both `liveChatTextMessageRenderer` and `liveChatPaidMessageRenderer`) into a `pandas.DataFrame` with columns `["offset_seconds", "author", "message"]`. Do NOT use `pytchat`.
   - **NON-NEGATIVE OFFSET GUARD**: Skip pre-stream messages where `videoOffsetTimeMsec < 0` so binning never creates negative time bins.
   - Return an empty DataFrame with those columns if the stream has no chat replay, and implement `save_cache(df, path)` and `load_cache(path)`.
2. `src/streamclipper/analysis/chat_velocity.py`:
   - Implement `ChatVelocityAnalyzer(config)` with `analyze(chat_df) -> list[TimeRange]`.
   - Skip analysis and return `[]` if `len(chat_df) < self.min_messages`.
   - Bin messages into `window_seconds` intervals, detect peaks via `scipy.signal.find_peaks` above `mean + prominence_factor * std`, normalize scores to `0.0 - 10.0`, and return merged `TimeRange` objects with `signal_type=SignalType.CHAT_VELOCITY`.
   - **YPP CHAT TICKER METADATA**: For each detected spike `TimeRange`, attach up to 5 representative viewer messages from `chat_df` within `[start, end]` into `TimeRange.metadata["sample_messages"]` (`[{"author": str, "message": str, "offset_seconds": float}]`) so Phase 4 (`captions.py`) can render a real live-chat burst ticker on the video.
3. `tests/test_chat.py`:
   - Unit tests verifying `.live_chat.json` JSONLines parsing (including paid messages and skipping negative `videoOffsetTimeMsec`), missing chat file graceful fallback, CSV caching, velocity spike detection on synthetic chat bursts, `metadata["sample_messages"]` attachment, and skipping when fewer than `min_messages` exist.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_chat.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-2/package-2.5-report.md` (and show it in chat) covering:
- Branch (`phase-2-core-pipeline`), files created/modified, and line counts
- Acceptance criteria verification table for all 5 criteria in Package 2.5
- Verbatim `pytest tests/test_chat.py -v` output

Stage all files (including `reports/phase-2/package-2.5-report.md`) and commit with:
`git commit -m "feat: YouTube live_chat extraction via yt-dlp and velocity spike detection"`
```

### Prompt 2.6 — Package 2.6: Stream Discovery
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-2-core-pipeline`

```markdown
You are implementing **Package 2.6 — Stream Discovery** (Phase 2) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-2-core-pipeline` (`git checkout phase-2-core-pipeline`).
- Read `reports/phase-2/package-2.5-report.md` to confirm Package 2.5 is complete.

### 2. What to Build (Strictly Scoped to Package 2.6)
Read **Package 2.6** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/discovery/youtube.py`:
   - Implement `YouTubeDiscovery(config)` with `discover(game=None) -> list[StreamInfo]`, `_search_youtube(query, max_results=20)`, and `_filter_streams(results, game)`.
   - Search via `yt_dlp.YoutubeDL` (`f"ytsearch{max_results}:{query} live stream"`), filter out entries with `duration < self.min_duration` or `view_count < self.min_viewers`, deduplicate by `video_id`, sort by `view_count` descending, and limit to `self.max_streams`.
2. Add discovery unit tests to `tests/test_download.py` (or `tests/test_discovery.py`):
   - Test filtering of short videos (<1 hour) and low-view videos, deduplication, sorting by view count descending, and respecting `max_streams`.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` across the entire test suite and confirm all Phase 1 and Phase 2 tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-2/package-2.6-report.md` (and show it in chat) covering:
- Branch (`phase-2-core-pipeline`), files created/modified, and line counts
- Acceptance criteria verification table for all 4 criteria in Package 2.6
- Verbatim `pytest -v` output across all Phase 1 and Phase 2 tests
- Readiness summary for the Phase 2 Opus review

Stage all files (including `reports/phase-2/package-2.6-report.md`) and commit with:
`git commit -m "feat: YouTube stream discovery for any game"`
```

### Prompt 2.R — Phase 2 Review & Merge (`phase-2-core-pipeline` -> `main`)
**Assigned Model:** Claude Opus 4.6 | **Branch:** `phase-2-core-pipeline` -> `main`

```markdown
You are the Lead Reviewer auditing **Phase 2 (`phase-2-core-pipeline`)** against `streamclipper-implementation-guide-v2.md`.

1. Read all reports in `reports/phase-2/` (`package-2.1-report.md` through `package-2.6-report.md`).
2. Inspect `git diff main...phase-2-core-pipeline` and run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`.
3. Verify:
   - `audio.py` extracts 16kHz mono WAV (`-ar 16000 -ac 1`).
   - `transcriber.py` uses resolved `HardwareProfile` settings from `Config`, deletes ASR and alignment models, and runs `gc.collect()` + `torch.cuda.empty_cache()`.
   - `loudness.py` streams audio via `librosa.stream(..., fill_value=0.0)` rather than `librosa.load()`, and records `metadata["peak_time"]` (preserving the loudest peak on merge).
   - `chat.py` and `chat_velocity.py` use `yt-dlp` `live_chat` JSONLines extraction with non-negative offset filtering, attach `metadata["sample_messages"]`, and have zero `pytchat` references.
4. Write the phase audit report to `reports/phase-2/phase-2-review.md`, commit it on `phase-2-core-pipeline`, then merge into `main`:
   `git checkout main && git merge --no-ff phase-2-core-pipeline -m "chore: merge phase-2-core-pipeline after review"`
```

## 5. Phase 3 Prompts — AI Brain: Highlight Scoring & YPP Adaptive Narrative (`phase-3-ai-scoring`)

### Prompt 3.1 — Package 3.1: Scorer Interface, YPP Adaptive Narrative Models & Prompt Templates
**Assigned Model:** Claude Opus 4.6 | **Branch:** `phase-3-ai-scoring` (Create from `main`)

```markdown
You are implementing **Package 3.1 — Scorer Interface, YPP Adaptive Narrative Models & Prompt Templates** (Phase 3, 🔴 Critical) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Setup
- Ensure Phase 2 is merged into `main`, then create and switch to the Phase 3 branch:
  `git checkout main && git checkout -b phase-3-ai-scoring`

### 2. What to Build (Strictly Scoped to Package 3.1)
Read `AGENTS.md` and **Package 3.1** in `streamclipper-implementation-guide-v2.md` and implement:
1. **Backward-Compatible Model & Config Extensions (`src/streamclipper/models.py` & `config.default.yaml`)**:
   - Add default-valued YPP fields to `Highlight` (`moment_type: str = "hype"`, `narration_intro: str = ""`, `narration_outro: str = ""`, `context_badge: str = ""`, `peak_time: float = 0.0`, `chat_burst: list[dict[str, Any]] = field(default_factory=list)`, `sub_clips: list[dict[str, Any]] = field(default_factory=list)`) and `ClipMetadata` (`moment_type`, `narration_intro`, `narration_outro`, `context_badge`, `transformation_mode: str = "narrative"`), updating `from_dict()` with `.get(...)` defaults so all 36 Phase 1 tests remain 100% green without modification.
   - Add the `transformation:` block (`enabled`, `mode`, `tts_engine`, `tts_voice`, `elevenlabs_voice_id`, `elevenlabs_model_id`, `duck_stream_volume`, `punch_in_zoom`, `zoom_duration_seconds`, `show_chat_ticker`) to `config.default.yaml`.
2. `src/streamclipper/scoring/scorer.py`:
   - Implement `TranscriptChunk` and the abstract `Scorer(ABC)` class with:
     - `score_and_select(transcript, loudness_spikes, chat_spikes) -> list[Highlight]`
     - `_chunk_transcript(transcript, chunk_minutes=None, overlap_seconds=60.0) -> list[TranscriptChunk]` (12-minute chunks with 1-minute overlap)
     - `get_candidate_chunks(transcript, loudness_spikes, chat_spikes, chunk_minutes=None, only_with_signals=True)` (shared helper used by both Scorer and Phase 5 MCP server)
     - `_find_overlapping(spikes, chunk)` and `_signal_overlap_score(highlight, spikes) -> float` (`0.0 - 10.0`)
     - `_compute_weighted_score(highlight, loudness_spikes, chat_spikes) -> float`
     - `_deduplicate(highlights, iou_threshold=0.3) -> list[Highlight]` (keeping higher `final_score`)
     - `_extract_words_for_range(transcript, start, end) -> list[WordTimestamp]`
     - `_extract_peak_time(highlight, loudness_spikes) -> float` (from overlapping `metadata["peak_time"]`, fallback to midpoint), `_extract_chat_burst(highlight, chat_spikes) -> list[dict]`, and `_validate_sub_clips(sub_clips, start, end) -> list[dict]` (ordered, non-overlapping, each beat `>= 3.0s`, else fallback to `[]`).
3. Prompt templates in `src/streamclipper/prompts/`:
   - `highlight_scorer.txt` (requesting `moment_type`, `narration_intro`, `narration_outro`, `context_badge`, and conditional `sub_clips` only when `>= 5s` dead air exists)
   - `narrative_writer.txt` (adaptive tone matching `moment_type` — funny/fail, graphics_detail, hype/clutch, chat_chaos, lore_secret/analysis — NEVER forcing dry analysis onto fun or visual moments)
   - `hook_generator.txt`
   - `caption_writer.txt`
4. `tests/test_scorer.py`:
   - Unit tests verifying 12-minute chunking with 1-minute overlap, `get_candidate_chunks(only_with_signals=True)`, signal overlap calculation, weighted fusion (`0.50 * llm + 0.25 * loudness + 0.25 * chat`), `peak_time` & `chat_burst` attachment, `_validate_sub_clips()` valid vs. invalid fallback, deduplication, and all 4 prompt templates.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` and confirm all Phase 1, Phase 2, and Package 3.1 tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-3/package-3.1-report.md` (and show it in chat) covering:
- Branch (`phase-3-ai-scoring`), files created/modified, and line counts
- Acceptance criteria verification table for all 7 criteria in Package 3.1
- Verbatim `pytest -v` output
- Mathematical details of signal overlap scoring, `sub_clips` validation, and deduplication

Stage all files (including `reports/phase-3/package-3.1-report.md`) and commit with:
`git commit -m "feat: scorer interface, YPP adaptive narrative prompts, and multi-beat signal fusion"`
```

### Prompt 3.2 — Package 3.2: Ollama Scorer (Standalone Mode)
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-3-ai-scoring`

```markdown
You are implementing **Package 3.2 — Ollama Scorer (Standalone Mode)** (Phase 3) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-3-ai-scoring` (`git checkout phase-3-ai-scoring`).
- Read `reports/phase-3/package-3.1-report.md` to confirm Package 3.1 is complete.

### 2. What to Build (Strictly Scoped to Package 3.2)
Read **Package 3.2** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/scoring/ollama_scorer.py`:
   - Implement `OllamaScorer(Scorer)` using resolved `llm.model` from `Config` (auto-tuned to `qwen3:8b` or `qwen3:4b` based on `HardwareProfile`, or user override) and `ollama.Client(host=self.base_url, timeout=self.timeout)`.
   - Implement `_score_chunk(chunk, loudness, chat) -> list[Highlight]` with up to 3 retries on malformed JSON (supporting both raw list `[...]` and wrapper `{"highlights": [...]}`), parsing `moment_type`, `narration_intro`, `narration_outro`, `context_badge`, and `sub_clips`.
   - Implement `generate_narrative(highlight, streamer, game) -> tuple[str, str, str]` using `narrative_writer.txt` to backfill missing intro/outro/badge fields with tone matched to `moment_type`.
   - Implement `generate_metadata(highlight, streamer, game, source_url) -> ClipMetadata` using `hook_generator.txt`, `caption_writer.txt`, and highlight narrative fields.
   - Implement `check_connection() -> bool` verifying Ollama is reachable and the resolved model is available (raising actionable `ollama pull <model>` error messages when missing).
2. Expand `tests/test_scorer.py`:
   - Add unit tests mocking `ollama.Client.generate` and `ollama.Client.list` to test JSON parsing of `moment_type`, narrative fields, and `sub_clips`, retry recovery on malformed JSON, `generate_narrative()`, `generate_metadata()`, and `check_connection()`.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_scorer.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-3/package-3.2-report.md` (and show it in chat) covering:
- Branch (`phase-3-ai-scoring`), files created/modified, and line counts
- Acceptance criteria verification table for all 6 criteria in Package 3.2
- Verbatim `pytest tests/test_scorer.py -v` output

Stage all files (including `reports/phase-3/package-3.2-report.md`) and commit with:
`git commit -m "feat: Ollama-based highlight scorer for standalone mode"`
```

### Prompt 3.3 — Package 3.3: Pipeline Orchestrator
**Assigned Model:** Claude Opus 4.6 | **Branch:** `phase-3-ai-scoring`

```markdown
You are implementing **Package 3.3 — Pipeline Orchestrator** (Phase 3, 🔴 Critical) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-3-ai-scoring` (`git checkout phase-3-ai-scoring`).
- Read `reports/phase-3/package-3.2-report.md` to confirm Package 3.2 is complete.

### 2. What to Build (Strictly Scoped to Package 3.3)
Read **Package 3.3** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/pipeline.py`:
   - Implement `Pipeline(config)` with `_recover_failed_state(state)`, `process_stream(url, max_clips=None, game=None, streamer=None, interactive=False, mode=None, voice_intro_path=None, voice_outro_path=None) -> list[ClipResult]`, and `process_streams_parallel(urls, max_concurrent=2)`.
   - **CRITICAL STATE & RETRY GUARDRAIL**: Compare `IntEnum` status ranks directly (`if state.status < PipelineStatus.TRANSCRIBING:`, etc.), save state to `StepCache` after every step, mark `state.status = PipelineStatus.FAILED` on exception, AND call `_recover_failed_state(state)` on resume so a `FAILED (99)` cached state resets to the last completed artifact stage instead of skipping every step!
   - **CRITICAL ASYNC SAFETY**: Only call `_interactive_select()` when `interactive=True`. In `process_streams_parallel()` and default headless runs (`interactive=False`), automatically select the top `max_clips` highlights so `input()` never blocks the `asyncio` event loop.
   - In `_interactive_select()`, display `moment_type`, proposed `narration_intro`, and beat count (`len(h.sub_clips)`) so the user can review highlights before rendering.
   - Wire `cli.py` `clip` and `auto` commands to accept `--mode [narrative|story|raw]`, `--voice-intro PATH`, and `--voice-outro PATH` and invoke `Pipeline` via `asyncio.run()`.
2. `tests/test_pipeline.py`:
   - End-to-end mocked pipeline tests verifying: full state transitions from `PENDING (1)` to `COMPLETE (7)`, step skipping when resuming from cached `ANALYZING (4)` or `SCORING (5)` states, retry recovery from `FAILED (99)` back to the right stage, `--mode` / `--voice-intro` forwarding, interactive vs. non-interactive highlight selection, and fault isolation in `process_streams_parallel()`.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` across the entire test suite and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-3/package-3.3-report.md` (and show it in chat) covering:
- Branch (`phase-3-ai-scoring`), files created/modified, and line counts
- Acceptance criteria verification table for all 8 criteria in Package 3.3
- Verbatim `pytest -v` output
- Readiness summary for the Phase 3 Opus review

Stage all files (including `reports/phase-3/package-3.3-report.md`) and commit with:
`git commit -m "feat: pipeline orchestrator with IntEnum state caching, FAILED retry recovery, and parallel support"`
```

### Prompt 3.R — Phase 3 Review & Merge (`phase-3-ai-scoring` -> `main`)
**Assigned Model:** Claude Opus 4.6 | **Branch:** `phase-3-ai-scoring` -> `main`

```markdown
You are the Lead Reviewer auditing **Phase 3 (`phase-3-ai-scoring`)** against `AGENTS.md` and `streamclipper-implementation-guide-v2.md`.

1. Read all reports in `reports/phase-3/` (`package-3.1-report.md` through `package-3.3-report.md`).
2. Inspect `git diff main...phase-3-ai-scoring` and run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`.
3. Verify:
   - All 36 Phase 1 unit tests still pass untouched after adding default-valued YPP fields (`moment_type`, `narration_intro`, `narration_outro`, `context_badge`, `peak_time`, `chat_burst`, `sub_clips`, `transformation_mode`) to `models.py` and `transformation:` to `config.default.yaml`.
   - `highlight_scorer.txt` and `narrative_writer.txt` adapt commentary tone and badge to `moment_type` (never forcing dry analysis onto funny/graphics/hype clips) and `_validate_sub_clips()` enforces `len >= 2`, bounds inside `[start, end]`, non-overlapping `>= 3.0s` beats, and valid total duration.
   - `Pipeline.process_stream` compares `IntEnum` values (`state.status < PipelineStatus.TRANSCRIBING`), saves `FAILED (99)` on error, recovers cleanly via `_recover_failed_state()` on retry, forwards `mode`/`voice_intro_path`/`voice_outro_path` to `_step_assemble()`, and exposes `--mode`, `--voice-intro`, `--voice-outro` in `cli.py`.
   - `process_streams_parallel` runs non-interactively without blocking on `input()`.
   - `Scorer.get_candidate_chunks` and `OllamaScorer` retry malformed JSON responses and use the hardware-resolved `llm.model`.
4. Write the phase audit report to `reports/phase-3/phase-3-review.md`, commit it on `phase-3-ai-scoring`, then merge into `main`:
   `git checkout main && git merge --no-ff phase-3-ai-scoring -m "chore: merge phase-3-ai-scoring after review"`
```

## 6. Phase 4 Prompts — Video Assembly & Output (`phase-4-video-assembly`)

### Prompt 4.1 — Package 4.1: Facecam Detector
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-4-video-assembly` (Create from `main`)

```markdown
You are implementing **Package 4.1 — Facecam Detector** (Phase 4) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Setup
- Ensure Phase 3 is merged into `main`, then create and switch to the Phase 4 branch:
  `git checkout main && git checkout -b phase-4-video-assembly`

### 2. What to Build (Strictly Scoped to Package 4.1)
Read **Package 4.1** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/assembly/facecam.py`:
   - Implement `FacecamDetector(config)` with `detect(video_path, num_samples=50) -> FacecamRegion | None` and `_is_in_corner(x, y, w, h, frame_w, frame_h) -> bool`.
   - Sample `num_samples` frames evenly across the video using `cv2.VideoCapture`.
   - Filter detections by size (between 1.5% and 25% of frame area) and corner placement (`center_x < 0.3*W or center_x > 0.7*W` and `center_y < 0.35*H or center_y > 0.65*H`).
   - Require consistent corner detections in at least 30% of sampled frames; compute median bounding box `(x, y, w, h)` and add webcam frame padding clamped to video dimensions.
2. `tests/test_assembly.py`:
   - Unit tests verifying `_is_in_corner` for all 4 corners vs. center of frame, area filtering, median bounding box calculation with padding, and returning `None` when fewer than 30% of frames contain a corner face.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_assembly.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-4/package-4.1-report.md` (and show it in chat) covering:
- Branch (`phase-4-video-assembly`), files created/modified, and line counts
- Acceptance criteria verification table for all 5 criteria in Package 4.1
- Verbatim `pytest tests/test_assembly.py -v` output

Stage all files (including `reports/phase-4/package-4.1-report.md`) and commit with:
`git commit -m "feat: auto facecam detection using OpenCV"`
```

### Prompt 4.2 — Package 4.2: 9:16 Video Composer (MoviePy v2 + Conditional Story Beats + Spike Zoom)
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-4-video-assembly`

```markdown
You are implementing **Package 4.2 — 9:16 Video Composer** (Phase 4) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-4-video-assembly` (`git checkout phase-4-video-assembly`).
- Read `reports/phase-4/package-4.1-report.md` to confirm Package 4.1 is complete.

### 2. What to Build (Strictly Scoped to Package 4.2)
Read `AGENTS.md` and **Package 4.2** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/assembly/composer.py`:
   - **CRITICAL MOVIEPY V2 API GUARDRAIL**: Import directly from `moviepy` (`from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips`, NEVER `from moviepy.editor`) and use `.cropped()`, `.resized()`, `.subclipped()`, `.with_position()`, and `.with_audio()` (NEVER v1 `.crop()` or `.resize()`).
   - Implement `VideoComposer(config)` with `compose(video_path, facecam, peak_offset=None) -> CompositeVideoClip`, `_crop_to_aspect(...)` (aspect-preserving center crop so square/4:3 facecams are NEVER stretched horizontally!), `_compose_with_facecam(clip, facecam)` (Layout A: top 35% aspect-correct facecam, bottom 65% gameplay), and `_compose_gameplay_only(clip)` (Layout B: full 1080×1920 center-cropped gameplay).
   - Implement `compose_story_beats(segment_paths, facecam, beat_trim_ranges=None, peak_offset=None)` trimming per-beat download buffer padding (`subclipped`) before stitching validated beats via `concatenate_videoclips(..., method="compose")` (falling back cleanly to `compose()` when only 1 path is provided).
   - Implement `remap_timeline_for_beats(words, chat_burst, peak_time, beats) -> tuple[list[WordTimestamp], list[dict], float | None]` so captions, live chat ticker messages, and spike zoom `peak_offset` stay 100% synced across multi-beat cuts (dropping dead-air entries and shifting Beat 2+ timestamps onto the `0.0`-based stitched timeline).
   - Implement `apply_spike_zoom(clip, peak_offset)` applying a `1.20x` center punch-in zoom during `[peak_offset - 0.5, peak_offset + zoom_duration]` when `transformation.punch_in_zoom > 1.0`.
   - Attach `result._source_clip = clip` and `result._source_clips = [...]` so `Exporter` can cleanly close ALL source file handles on Windows before temp cleanup.
2. Expand `tests/test_assembly.py`:
   - Generate synthetic 1920×1080 clips and test Layout A (incl. square `FacecamRegion` aspect-preserving crop), Layout B (`facecam=None`), `compose_story_beats()` buffer trimming + concatenation, `remap_timeline_for_beats()` timestamp shifting + dead-air filtering, `apply_spike_zoom()` output dimensions & duration, audio preservation, and `_source_clip` / `_source_clips` tracking.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_assembly.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-4/package-4.2-report.md` (and show it in chat) covering:
- Branch (`phase-4-video-assembly`), files created/modified, and line counts
- Acceptance criteria verification table for all 7 criteria in Package 4.2
- Verbatim `pytest tests/test_assembly.py -v` output

Stage all files (including `reports/phase-4/package-4.2-report.md`) and commit with:
`git commit -m "feat: 9:16 MoviePy v2 composer with conditional story beats and spike punch-in zoom"`
```

### Prompt 4.3 — Package 4.3: Hormozi-Style Captions & Live Chat Burst Ticker (ASS + FFmpeg `libass` / Optional NVENC)
**Assigned Model:** Claude Opus 4.6 | **Branch:** `phase-4-video-assembly`

```markdown
You are implementing **Package 4.3 — Hormozi-Style Captions & Live Chat Burst Ticker** (Phase 4, 🔴 Critical) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-4-video-assembly` (`git checkout phase-4-video-assembly`).
- Read `reports/phase-4/package-4.2-report.md` to confirm Package 4.2 is complete.

### 2. What to Build (Strictly Scoped to Package 4.3)
Read `AGENTS.md` and **Package 4.3** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/assembly/captions.py`:
   - Implement `_sec_to_ass_time(seconds: float) -> str`, `_hex_to_ass_bgr(hex_color: str) -> str` (converting config `captions.highlight_color` e.g. `#FFD700` → `&H00D7FF&`), `_escape_ass_text(text: str) -> str` (escaping `\`, `{`, `}`), and `AssCaptionBurner(config)` with:
     - `generate_ass(words: list[WordTimestamp], clip_start_offset: float, ass_path: Path, chat_burst: list[dict] | None = None) -> Path`
     - `_build_ffmpeg_cmd(input_video, ass_filename, output_video, codec, crf) -> list[str]`
     - `burn_captions(input_video: Path, ass_path: Path, output_video: Path, crf: int = 23, codec: str | None = None) -> Path`
   - Emit both `Style: Hormozi` (active word highlighted via `self.highlight_bgr`) AND `Style: ChatTicker` (rendering up to 5 real viewer messages from `chat_burst` when `transformation.show_chat_ticker` is `True`).
   - **CRITICAL WINDOWS & NVENC GUARDRAILS**:
     - Run `subprocess.run(..., cwd=str(ass_path.resolve().parent))` with `-vf f"ass={ass_path.name}"` and resolved absolute input/output paths so Windows drive letters/quotes never break FFmpeg's filtergraph.
     - Use resolved `video.codec` from `Config`: when `h264_nvenc`, pass `-c:v h264_nvenc -preset p4 -cq <crf>`; when `libx264`, pass `-c:v libx264 -crf <crf>`; always copy audio (`-c:a copy`).
     - If `h264_nvenc` fails with `CalledProcessError`, automatically retry once with `libx264`.
2. Expand `tests/test_assembly.py`:
   - Test `_sec_to_ass_time()`, `_hex_to_ass_bgr()`, `_escape_ass_text()`, `.ass` file generation (both `Hormozi` and `ChatTicker` dialogue events), Windows-safe `cwd` FFmpeg invocation, `h264_nvenc` command flags (`-cq`), and automatic fallback to `libx264` on NVENC failure.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_assembly.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-4/package-4.3-report.md` (and show it in chat) covering:
- Branch (`phase-4-video-assembly`), files created/modified, and line counts
- Acceptance criteria verification table for all 7 criteria in Package 4.3
- Verbatim `pytest tests/test_assembly.py -v` output

Stage all files (including `reports/phase-4/package-4.3-report.md`) and commit with:
`git commit -m "feat: Hormozi ASS captions and live chat burst ticker via FFmpeg libass with NVENC support"`
```

### Prompt 4.4 — Package 4.4: Hook Title & Adaptive Context Badge Overlay
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-4-video-assembly`

```markdown
You are implementing **Package 4.4 — Hook Title & Adaptive Context Badge Overlay** (Phase 4) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-4-video-assembly` (`git checkout phase-4-video-assembly`).
- Read `reports/phase-4/package-4.3-report.md` to confirm Package 4.3 is complete.

### 2. What to Build (Strictly Scoped to Package 4.4)
Read **Package 4.4** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/assembly/hooks.py`:
   - Implement `HookTitleOverlay(config)` with `add_hook(video_clip: CompositeVideoClip, hook_title: str, context_badge: str = "") -> CompositeVideoClip`.
   - Render the hook banner and optional adaptive `context_badge` pill (e.g. `"💀 Instant Karma"`, `"🎨 Insane Detail"`, `"🔥 Clutch Escape"`) into RGBA numpy arrays using Pillow and wrap them in MoviePy v2 `ImageClip` layers without ImageMagick.
   - Preserve `video_clip.audio` and propagate `_source_clip` / `_source_clips` onto the returned `CompositeVideoClip`.
   - Return `video_clip` unchanged if `self.enabled` is `False` and both `hook_title` and `context_badge` are empty.
2. Expand `tests/test_assembly.py`:
   - Test `HookTitleOverlay.add_hook()` with `hook_title` + `context_badge` enabled vs. disabled, verifying layer count, duration, and `_source_clip` / `_source_clips` propagation.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_assembly.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-4/package-4.4-report.md` (and show it in chat) covering:
- Branch (`phase-4-video-assembly`), files created/modified, and line counts
- Acceptance criteria verification table for all 4 criteria in Package 4.4
- Verbatim `pytest tests/test_assembly.py -v` output

Stage all files (including `reports/phase-4/package-4.4-report.md`) and commit with:
`git commit -m "feat: hook title and adaptive context badge overlay with MoviePy v2 API"`
```

### Prompt 4.5 — Package 4.5: YPP Transformative Narrator & Dynamic FFmpeg Audio Ducking
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-4-video-assembly`

```markdown
You are implementing **Package 4.5 — YPP Transformative Narrator & Dynamic FFmpeg Audio Ducking** (Phase 4) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-4-video-assembly` (`git checkout phase-4-video-assembly`).
- Read `reports/phase-4/package-4.4-report.md` to confirm Package 4.4 is complete.

### 2. What to Build (Strictly Scoped to Package 4.5)
Read `AGENTS.md` and **Package 4.5** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/assembly/narrator.py`:
   - Implement `TransformativeNarrator(config)` with pluggable speech synthesis (`synthesize_speech(text, output_wav) -> Path | None`) supporting:
     - `kokoro` (local free primary in `auto` mode if installed)
     - `edge-tts` (cloud free fallback in `auto` mode)
     - `elevenlabs` (optional API + personal voice clone via `ELEVENLABS_API_KEY` env var and `transformation.elevenlabs_voice_id`, falling back gracefully to `auto` if unconfigured)
   - Support custom human-recorded `--voice-intro` / `--voice-outro` WAV paths with higher priority than TTS.
   - Implement `_build_ducking_filtergraph(intro_dur, outro_dur, clip_duration)` and `mix_narration_with_ducking(input_video, output_video, intro_wav, outro_wav, clip_duration) -> Path`:
     - Dynamically build a **2-input** filtergraph for intro-only or outro-only, and a **3-input** filtergraph when both exist (NEVER hardcode `amix=inputs=3`!).
     - Normalize all inputs to `48000 Hz` stereo (`aresample=48000,aformat=channel_layouts=stereo`) and use stereo `adelay={delay_ms}|{delay_ms}` before `amix=inputs=N:duration=first:normalize=0` so 24kHz mono Kokoro/Edge-TTS/ElevenLabs WAVs mix cleanly with 48kHz stereo stream audio.
     - Clamp narration windows safely on short clips so intro and outro never overlap.
     - Duck `[0:a]` to `self.duck_volume` (`0.18`) during narration windows and preserve `1.0` (`100%`) original stream audio during the reaction climax, copying video (`-c:v copy`) so no extra video compression occurs.
2. Expand `tests/test_assembly.py`:
   - Add unit tests verifying provider fallback (`kokoro` -> `edge-tts`), mocked `elevenlabs` API call with `ELEVENLABS_API_KEY` + `voice_id`, custom `voice_intro_path` priority, dynamic FFmpeg filtergraph generation for intro-only (2 inputs), outro-only (2 inputs), both (3 inputs), and short-clip overlap clamping.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_assembly.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-4/package-4.5-report.md` (and show it in chat) covering:
- Branch (`phase-4-video-assembly`), files created/modified, and line counts
- Acceptance criteria verification table for all 5 criteria in Package 4.5
- Verbatim `pytest tests/test_assembly.py -v` output

Stage all files (including `reports/phase-4/package-4.5-report.md`) and commit with:
`git commit -m "feat: YPP transformative narrator (Kokoro/Edge-TTS/ElevenLabs/mic) and dynamic FFmpeg audio ducking"`
```

### Prompt 4.6 — Package 4.6: Exporter & File Organizer
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-4-video-assembly`

```markdown
You are implementing **Package 4.6 — Exporter & File Organizer** (Phase 4) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-4-video-assembly` (`git checkout phase-4-video-assembly`).
- Read `reports/phase-4/package-4.5-report.md` to confirm Package 4.5 is complete.

### 2. What to Build (Strictly Scoped to Package 4.6)
Read `AGENTS.md` and **Package 4.6** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/output/organizer.py`:
   - Implement `OutputOrganizer(config)` with `get_clip_path(game, streamer, index, date_str=None) -> Path`, `save_metadata(clip_path, metadata) -> Path` (including `moment_type`, `narration_intro`, `narration_outro`, `context_badge`, `transformation_mode`), `save_transcript(clip_path, words) -> Path`, and `_sanitize_name(name) -> str` (stripping `<>:"/\|?*`).
2. `src/streamclipper/output/exporter.py`:
   - Implement `Exporter(config)` with `export(clip, words, clip_start_offset, output_path, progress_callback=None, chat_burst=None, narration_intro="", narration_outro="", voice_intro_path=None, voice_outro_path=None, mode=None) -> Path`.
   - Render composed MoviePy clip to intermediate temp MP4 at high quality (`codec="libx264"`, `preset="ultrafast"`, `ffmpeg_params=["-crf", "18"]`), release MoviePy handles immediately in `try/finally`, run `TransformativeNarrator.mix_narration_with_ducking()` when `self.narrator.enabled` and `(mode or self.narrator.mode) != "raw"`, then burn `.ass` (`Hormozi` + `ChatTicker`) via `AssCaptionBurner` whenever `words` OR (`show_chat_ticker and chat_burst`) is non-empty using resolved `video.codec` (`h264_nvenc` → `libx264` fallback).
   - **CRITICAL WINDOWS FILE-LOCK CLEANUP**: Always close `clip`, `getattr(clip, "_source_clip", None)`, and all clips in `getattr(clip, "_source_clips", [])` inside a `try/finally` block BEFORE unlinking intermediate temp MP4 / `.ass` / WAV files.
   - Wire `_step_assemble` in `src/streamclipper/pipeline.py` end-to-end across `VideoSegmentDownloader` (single & multi-beat), `FacecamDetector`, `VideoComposer` (+ beat trimming, `remap_timeline_for_beats()`, & spike zoom), `HookTitleOverlay` (+ context badge), `TransformativeNarrator`, `Exporter`, and `OutputOrganizer`.
3. Expand `tests/test_assembly.py`:
   - Add unit tests for `OutputOrganizer` (directory template, sanitization, companion JSON with narrative fields, transcript file) and `Exporter` (ultrafast `crf=18` temp encode, narrator ducking integration, `AssCaptionBurner` call with `chat_burst`, and full handle `.close()` before temp cleanup).

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` across the entire test suite and confirm all Phase 1–4 tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-4/package-4.6-report.md` (and show it in chat) covering:
- Branch (`phase-4-video-assembly`), files created/modified, and line counts
- Acceptance criteria verification table for all 5 criteria in Package 4.6
- Verbatim `pytest -v` output
- Readiness summary for the Phase 4 Opus review

Stage all files (including `reports/phase-4/package-4.6-report.md`) and commit with:
`git commit -m "feat: clip exporter with narrator ducking, ASS burn-in, NVENC support, and file organization"`
```

### Prompt 4.R — Phase 4 Review & Merge (`phase-4-video-assembly` -> `main`)
**Assigned Model:** Claude Opus 4.6 | **Branch:** `phase-4-video-assembly` -> `main`

```markdown
You are the Lead Reviewer auditing **Phase 4 (`phase-4-video-assembly`)** against `AGENTS.md` and `streamclipper-implementation-guide-v2.md`.

1. Read all reports in `reports/phase-4/` (`package-4.1-report.md` through `package-4.6-report.md`).
2. Inspect `git diff main...phase-4-video-assembly` and run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`.
3. Verify:
   - `composer.py` and `hooks.py` use MoviePy v2 imports and methods (`.cropped()`, `.resized()`, `.subclipped()`, `.with_position()`, `.with_duration()`, `.with_audio()`), aspect-preserving facecam cropping, conditional multi-beat story stitching, `1.2x` spike zoom, and `_source_clip` / `_source_clips` tracking.
   - `captions.py` dynamically converts `captions.highlight_color` to ASS BGR, emits both `Hormozi` and `ChatTicker` styles, escapes ASS delimiters, runs FFmpeg with `cwd=ass_path.parent` and `ass=<basename>`, and supports `h264_nvenc` (`-cq`) with `libx264` fallback.
   - `narrator.py` supports pluggable TTS (`kokoro`, `edge-tts`, optional `elevenlabs` via `ELEVENLABS_API_KEY` + `voice_id`), custom `--voice-intro`/`--voice-outro`, and dynamic 2-input / 3-input FFmpeg audio ducking with short-clip overlap clamping.
   - `exporter.py` closes all clip handles in `finally` before deleting temp files and avoids double-compression degradation.
   - `pipeline.py` `_step_assemble` wires the full transformative assembly pipeline cleanly.
4. Write the phase audit report to `reports/phase-4/phase-4-review.md`, commit it on `phase-4-video-assembly`, then merge into `main`:
   `git checkout main && git merge --no-ff phase-4-video-assembly -m "chore: merge phase-4-video-assembly after review"`
```

## 7. Phase 5 Prompts — MCP Server & Integration (`phase-5-mcp-server`)

### Prompt 5.1 — Package 5.1: FastMCP Server Implementation
**Assigned Model:** Claude Opus 4.6 | **Branch:** `phase-5-mcp-server` (Create from `main`)

```markdown
You are implementing **Package 5.1 — MCP Server Implementation** (Phase 5, 🔴 Critical) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Setup
- Ensure Phase 4 is merged into `main`, then create and switch to the Phase 5 branch:
  `git checkout main && git checkout -b phase-5-mcp-server`

### 2. What to Build (Strictly Scoped to Package 5.1)
Read **Package 5.1** in `streamclipper-implementation-guide-v2.md` and create:
1. `src/streamclipper/mcp_server/server.py`:
   - **CRITICAL MCP SDK GUARDRAIL**: Use `from mcp.server.fastmcp import FastMCP` and instantiate `mcp = FastMCP("streamclipper")` (NEVER `from mcp.server import Server`).
   - Implement all 11 `@mcp.tool()` functions returning structured JSON strings (catching exceptions and returning `{"error": str(e)}` rather than raw tracebacks):
     1. `discover_streams(game: str = "GTA 6", max_results: int = 10) -> str`
     2. `download_stream_audio(url: str) -> str`
     3. `transcribe_audio(audio_path: str) -> str`
     4. `analyze_signals(audio_path: str, video_url: str) -> str`
     5. `get_transcript_chunks(audio_path: str, chunk_minutes: int = 12, only_with_signals: bool = True) -> str` (using `Scorer.get_candidate_chunks` to filter out dead-air chunks with zero signal spikes when `only_with_signals=True`)
     6. `submit_highlights(url: str, highlights: str) -> str` (parsing YPP fields `moment_type`, `narration_intro`, `narration_outro`, `context_badge`, `sub_clips`, computing weighted `final_score`, and saving selected highlights to `StepCache`)
     7. `render_clip(video_url: str, start: float, end: float, hook_title: str, game: str = "", streamer: str = "", moment_type: str = "hype", narration_intro: str = "", narration_outro: str = "", context_badge: str = "", sub_clips_json: str = "[]", mode: str = "narrative", voice_intro_path: str = "", voice_outro_path: str = "") -> str`
     8. `render_all_selected(video_url: str) -> str`
     9. `get_pipeline_status() -> str`
     10. `generate_post_metadata(hook_title: str, streamer: str, game: str, description: str) -> str`
     11. `cleanup_temp(url: str = "") -> str`
2. `src/streamclipper/mcp_server/__init__.py`:
   - Expose `main()` calling `mcp.run()`.
3. `tests/test_mcp_server.py`:
   - Unit tests verifying `FastMCP` tool registration (all 11 tools), `get_transcript_chunks(only_with_signals=True)` filtering, `submit_highlights` JSON parsing and cache persistence, `get_pipeline_status`, and structured JSON error handling.

### 3. Verification
- Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_mcp_server.py -v` and confirm all tests pass.

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-5/package-5.1-report.md` (and show it in chat) covering:
- Branch (`phase-5-mcp-server`), files created/modified, and line counts
- Acceptance criteria verification table for all criteria in Package 5.1
- Verbatim `pytest tests/test_mcp_server.py -v` output

Stage all files (including `reports/phase-5/package-5.1-report.md`) and commit with:
`git commit -m "feat: FastMCP server with signal-filtered transcript chunking"`
```

### Prompt 5.2 — Package 5.2: README & Setup Guide
**Assigned Model:** Gemini 3.8 (high) | **Branch:** `phase-5-mcp-server`

```markdown
You are implementing **Package 5.2 — README & Setup Guide** (Phase 5) from `streamclipper-implementation-guide-v2.md`.

### 1. Git Branch Verification
- Verify you are on branch `phase-5-mcp-server` (`git checkout phase-5-mcp-server`).
- Read `reports/phase-5/package-5.1-report.md` to confirm Package 5.1 is complete.

### 2. What to Build (Strictly Scoped to Package 5.2)
Read **Package 5.2** in `streamclipper-implementation-guide-v2.md` and create:
1. `README.md`:
   - Comprehensive project overview, architecture diagram, feature list (incl. YPP Transformative Narrative & Story Engine: adaptive `moment_type` commentary, conditional multi-beat cuts, spike punch-in zoom, live chat ASS ticker, and dynamic audio ducking), system requirements (Python 3.12 `.venv` / 3.10+, FFmpeg with `libass` and optional `h264_nvenc`, NVIDIA GPU or CPU), 5-tier Dynamic Hardware Auto-Configuration table (`cuda_high_vram`, `cuda_mid_tensor`, `cuda_mid_no_tensor`, `cpu_fallback`, `cpu_only`), quick start guide (`--mode narrative`, `--mode story`, `--voice-intro`, `--voice-outro`), voiceover setup (free `Kokoro-82M` / `edge-tts`, optional **ElevenLabs API / personal voice clone** via `ELEVENLABS_API_KEY` + `elevenlabs_voice_id`, and custom mic WAVs), Claude Desktop / Antigravity MCP configuration (`streamclipper-mcp`), Ollama standalone setup, full `config.yaml` reference table, and Windows/Linux troubleshooting guide.

### 3. Verification
- Run the entire test suite (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`) and CLI smoke checks (`streamclipper --help`, `streamclipper setup`, `streamclipper status`).

### 4. Mandatory Completion Report & Commit
Write a detailed report to `reports/phase-5/package-5.2-report.md` (and show it in chat) covering:
- Branch (`phase-5-mcp-server`), files created/modified, and line counts
- Acceptance criteria verification table for all criteria in Package 5.2
- Verbatim `pytest -v` output across the full repository

Stage all files (including `reports/phase-5/package-5.2-report.md`) and commit with:
`git commit -m "docs: comprehensive README with setup, YPP transformative mode, hardware auto-tuning, and MCP config"`
```

### Prompt 5.R — Phase 5 Final Review & Merge (`phase-5-mcp-server` -> `main`)
**Assigned Model:** Claude Opus 4.6 | **Branch:** `phase-5-mcp-server` -> `main`

```markdown
You are the Lead Reviewer performing the final audit of **Phase 5 (`phase-5-mcp-server`)** and the complete **StreamClipper v2.2** repository.

1. Read `reports/phase-5/package-5.1-report.md` and `package-5.2-report.md`, plus the prior phase review reports in `reports/phase-1/` through `reports/phase-4/`.
2. Inspect `git diff main...phase-5-mcp-server` and run the full test suite (`D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`).
3. Run the CLI smoke checks (`streamclipper --help`, `streamclipper setup`, `streamclipper status`).
4. Write the final project audit report to `reports/phase-5/phase-5-review.md`, commit it on `phase-5-mcp-server`, merge into `main`:
   `git checkout main && git merge --no-ff phase-5-mcp-server -m "chore: merge phase-5-mcp-server and complete StreamClipper v2.2"`
   and provide a final release summary.
```
