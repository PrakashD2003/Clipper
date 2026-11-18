# AGENTS.md — StreamClipper AI Agent Instructions

> **Repository:** `D:\GTA6-Clipper` (`streamclipper`)  
> **Primary Blueprint:** `streamclipper-implementation-guide-v2.md` (v2.2 — Patched + Auto-Hardware + YPP Transformative Engine)  
> **Prompt Runbook:** `streamclipper-agent-prompts.md`  
> **Current Repository Milestone:** **Phase 1 (`phase-1-foundation`, Packages `1.1–1.3` + Review `1.R`) is ✅ COMPLETE (`36/36` tests passing) and MERGED into `main`.** Do NOT redo Phase 1; branch `phase-2-core-pipeline` from `main` and proceed from **Package 2.1**.

This file defines the mandatory workspace conventions, multi-device hardware auto-configuration rules, YouTube Partner Program (YPP) transformative content guardrails, technical requirements, git workflow, and reporting standards for all AI coding agents (**Gemini 3.8 (high)** and **Claude Opus 4.6**) working in this repository.

## 1. Project Overview & Architecture

**StreamClipper** is a free, open-source, game-agnostic AI stream clipping pipeline with dual **CLI (`click` + `rich`)** and **MCP (`FastMCP`)** execution modes, engineered to satisfy the [YouTube Channel Monetization Policies](https://support.google.com/youtube/answer/1311392?hl=en) and [YouTube Reused Content FAQ](https://support.google.com/youtube/community-guide/271248162/%F0%9F%94%8E-faq-reused-content-youtube%E2%80%99s-partner-program?hl=en):

- **Audio-First Ingestion**: Downloads 16kHz mono WAV audio first via [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) (~460 MB for a 4-hour VOD) and fetches only ±3s padded 1080p video cuts (`download_ranges`) for selected highlights.
- **Tri-Modal Signal Fusion**: Combines streamed audio RMS loudness spikes (25%), YouTube `.live_chat.json` velocity spikes (25%), and LLM semantic scoring (50% via local Ollama or host MCP client).
- **YPP Transformative Narrative & Story Engine**: Turns raw stream slices into engaging, original Shorts by combining:
  1. **Adaptive Commentary Intro & Outro (`assembly/narrator.py`)**: Classifies each highlight's `moment_type` (`funny`, `fail`, `graphics_detail`, `hype`, `clutch`, `chat_chaos`, `lore_secret`, `analysis`) so commentary tone matches the clip naturally (witty setup for funny/fails, detail spotlight for graphics/physics, stakes-building for clutches — never forcing dry analysis onto fun moments). Supports free local [`Kokoro-82M`](https://github.com/hexgrad/kokoro) or [`edge-tts`](https://github.com/rany2/edge-tts), optional **ElevenLabs API / personal voice clone** (`ELEVENLABS_API_KEY` + `voice_id`), and real human mic overrides (`--voice-intro` / `--voice-outro` WAV), mixed over `18%` ducked stream audio while preserving `100%` original stream audio during the climax.
  2. **Conditional Multi-Beat Jump-Cut Story Mode (`sub_clips`)**: When a moment contains ≥5s of dead air between setup and payoff, stitches 2–3 validated beats (*Setup → Escalation → Payoff*); otherwise renders a clean single cut.
  3. **Spike-Synced Punch-In Zoom, Adaptive Context Badge & Live Chat Ticker**: Applies a `1.20x` center punch-in zoom at `peak_time`, renders an adaptive `context_badge` pill, and burns real `.live_chat.json` viewer bursts via the Windows-safe [FFmpeg `ass` filter](https://trac.ffmpeg.org/wiki/HowToBurnSubtitlesIntoVideo).

## 2. Windows Environment & Toolchain Rules

Always execute Python, `pytest`, `pip`, and CLI entry points using the repository's dedicated **Python 3.12 virtual environment** (`.venv`) so prebuilt Windows CUDA wheels (`torch`, `ctranslate2`, `faster-whisper`, [`whisperx`](https://github.com/m-bain/whisperx)) resolve cleanly:

| Tool | Canonical Executable Path / Command | Notes |
| :--- | :--- | :--- |
| **Python (`.venv`)** | `D:\GTA6-Clipper\.venv\Scripts\python.exe` | Preferred Python 3.12 interpreter with CUDA & WhisperX wheels |
| **Pytest** | `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` | Run full test suite before every commit |
| **CLI Entry Point** | `D:\GTA6-Clipper\.venv\Scripts\streamclipper.exe` | Installed in editable mode (`pip install -e ".[dev,mcp]"`) |
| **MCP Entry Point** | `D:\GTA6-Clipper\.venv\Scripts\streamclipper-mcp.exe` | Starts the `FastMCP` stdio server |
| **Git** | `git` (fallback: `"C:\Program Files\Git\cmd\git.exe"`) | Add `C:\Program Files\Git\cmd` to session `PATH` if missing |
| **FFmpeg** | `ffmpeg` (Gyan build via WinGet) | Must include `libass` (`ass` filter) and optional `h264_nvenc` encoder |

## 3. Multi-Device Hardware Auto-Configuration

This codebase runs across heterogeneous machines — specifically a **Desktop with an NVIDIA GeForce GTX 1660 SUPER (6 GB VRAM, Turing TU116, Compute Capability 7.5, no Tensor Cores)** and a **Laptop with an NVIDIA GeForce RTX 4050 (6 GB VRAM, Ada Lovelace AD107, Compute Capability 8.9, Tensor Cores)**, as well as high-VRAM and CPU-only systems.

- **Never hardcode settings for a single GPU** in `config.default.yaml` or pipeline modules.
- Keep `transcription.device`, `transcription.model_size`, `transcription.compute_type`, `transcription.batch_size`, `llm.model`, and `video.codec` set to `"auto"` in `config.default.yaml`.
- Route all `"auto"` resolution through `detect_hardware_profile()` and `resolve_hardware_config()` in `src/streamclipper/config.py` using [`torch.cuda.get_device_properties`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_device_properties.html):

| Hardware Tier (`HardwareProfile.tier`) | Detection Rule | `device` | `model_size` | `compute_type` | `batch_size` | `llm.model` | `video.codec` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`cuda_high_vram`** | CUDA `>= 10.0 GB` VRAM | `cuda` | `large-v2` | `float16` | `16` | `qwen3:8b` | `h264_nvenc` (else `libx264`) |
| **`cuda_mid_tensor`** (e.g. RTX 4050 6GB) | CUDA `5.5–9.9 GB` + (`major >= 8` or `"RTX"` in name) | `cuda` | `large-v2` | `int8_float16` | `8` | `qwen3:8b` | `h264_nvenc` (else `libx264`) |
| **`cuda_mid_no_tensor`** (e.g. GTX 1660 SUPER 6GB) | CUDA `5.5–9.9 GB` + (`major < 8` and no `"RTX"`) | `cuda` | `medium` | `int8` | `4` | `qwen3:4b` | `h264_nvenc` (else `libx264`) |
| **`cpu_fallback`** | CUDA present but `< 5.5 GB` VRAM | `cpu` | `base` | `int8` | `4` | `qwen3:4b` | `h264_nvenc` (else `libx264`) |
| **`cpu_only`** | No CUDA available | `cpu` | `base` | `int8` | `4` | `qwen3:4b` | `libx264` |

- Always preserve explicit user overrides from `config.yaml` over `"auto"` defaults and track override provenance via `Config.is_auto_resolved(dotted_key)`.

## 4. Mandatory v2.2 Technical & YPP Transformation Guardrails

Every agent writing or reviewing code in this repository must enforce these 12 architectural rules:

1. **Preserve Completed Phase 1 Foundation**: Phase 1 (`1.1–1.3`) is merged on `main` (`36/36` passing). Any new fields added to `models.py` (`moment_type`, `narration_intro`, `narration_outro`, `context_badge`, `peak_time`, `chat_burst`, `sub_clips`, `transformation_mode`) or `config.default.yaml` (`transformation:`) in Phase 3+ MUST have default values and `.get(...)` defaults in `from_dict()` so all 36 Phase 1 tests pass without modification.
2. **PEP 517 Packaging (`pyproject.toml`)**: Always use `build-backend = "setuptools.build_meta"`. Never use `setuptools.backends._legacy:_Backend` and never add `pytchat`.
3. **`IntEnum` Pipeline States & `FAILED` Retry Recovery (`models.py`, `pipeline.py`)**: `PipelineStatus` MUST subclass `IntEnum` (`PENDING = 1` through `COMPLETE = 7`, `FAILED = 99`). In `Pipeline.process_stream()`, call `_recover_failed_state(state)` before `<` stage comparisons so a `FAILED (99)` cached state resets to the last completed artifact stage instead of skipping steps.
4. **Recursive JSON Serialization (`models.py`, `cache.py`)**: All dataclasses inherit from `SerializableModel` with `_serialize_value()` and symmetric `from_dict()` classmethods. Never dump raw `__dict__`.
5. **PyTorch VRAM Attribute (`config.py`, `cli.py`)**: Always read `torch.cuda.get_device_properties(0).total_memory` — never `.total_mem`.
6. **16kHz Mono WAV Extraction (`download/audio.py`)**: Always pass `postprocessor_args: {"extractaudio": ["-ar", "16000", "-ac", "1"]}` to `yt_dlp.YoutubeDL` so extracted WAV files are 16kHz mono PCM (~460 MB per 4h stream).
7. **Two-Stage GPU VRAM Cleanup (`analysis/transcriber.py`)**: Explicitly delete the WhisperX ASR model (`del model; gc.collect(); torch.cuda.empty_cache()`) before loading the wav2vec2 alignment model, and delete `model_a, audio` (`gc.collect(); torch.cuda.empty_cache()`) before returning.
8. **Constant-Memory Streamed RMS + `peak_time` Metadata (`analysis/loudness.py`)**: Never call `librosa.load()` on full stream WAVs. Use `soundfile.info()` + `librosa.stream(..., fill_value=0.0)` so tail blocks never crash, and record exact peak timestamps in `TimeRange.metadata["peak_time"]` (preserving the loudest peak's `peak_time` when merging).
9. **Native `yt-dlp` Live Chat Scraper + `sample_messages` Metadata (`download/chat.py`, `analysis/chat_velocity.py`)**: Extract live chat via `yt_dlp` (`skip_download=True`, `writesubtitles=True`, `subtitleslangs=["live_chat"]`), skip pre-stream entries with `offset_ms < 0`, and attach up to 5 representative viewer messages in `TimeRange.metadata["sample_messages"]` for each detected chat spike.
10. **Non-Blocking Async Pipeline & Adaptive YPP Prompts (`scoring/`, `pipeline.py`)**: Keep `interactive: bool = False` as default in `Pipeline.process_stream()` and force `interactive=False` in `process_streams_parallel()`. Adapt `narration_intro`, `narration_outro`, and `context_badge` tone to `moment_type` (do not force dry analysis onto funny/graphics/hype clips).
11. **MoviePy v2 API, Windows-Safe FFmpeg `cwd` ASS Burn-In, Handle Cleanup & Dynamic Audio Ducking (`assembly/`, `output/`)**:
    - Import directly from `moviepy` (`from moviepy import VideoFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips`) and use `.cropped()`, `.resized()`, `.subclipped()`, `.with_position()`, `.with_duration()`, `.with_audio()`.
    - Preserve aspect-correct facecam cropping (`_crop_to_aspect`), validate `sub_clips` before multi-beat stitching (falling back to single-cut if invalid), trim per-beat download buffer padding before concatenation, remap word timestamps / `chat_burst` / `peak_offset` onto the stitched timeline via `remap_timeline_for_beats()`, and apply `1.20x` center punch-in zoom around `peak_offset` when enabled.
    - Keep **Windows-safe `cwd` FFmpeg ASS burn-in**: write & close the `.ass` file (with dynamic `_hex_to_ass_bgr`, `_escape_ass_text`, `Style: Hormozi`, and `Style: ChatTicker`), then invoke `ffmpeg` with `cwd=str(ass_path.resolve().parent)` and `-vf ass={ass_path.name}` (avoiding Windows drive-letter colon escaping bugs), supporting `h264_nvenc` with automatic `libx264` fallback.
    - Write intermediate composed temp MP4s at high quality (`preset="ultrafast"`, `-crf 18`) and close `clip` + `_source_clip` / `_source_clips` in a `try/finally` block BEFORE deleting temp files on Windows.
    - In `assembly/narrator.py`, support pluggable TTS engines (`kokoro`, `edge-tts`, optional `elevenlabs` via `ELEVENLABS_API_KEY` + `voice_id`, and custom `--voice-intro` / `--voice-outro` WAVs), normalize narrator tracks to `48000 Hz` stereo (`aresample=48000,aformat=channel_layouts=stereo` + stereo `adelay=ms|ms`) before `amix`, and dynamically construct the FFmpeg `volume` + `adelay` + `amix` filtergraph for intro-only (2 inputs), outro-only (2 inputs), or both (3 inputs) with overlap clamping on short clips.
12. **FastMCP Server & Signal-Filtered Chunks (`mcp_server/server.py`)**: Use `from mcp.server.fastmcp import FastMCP` with `@mcp.tool()` and `mcp.run()`. Keep `only_with_signals: bool = True` default on `get_transcript_chunks`, and accept YPP narrative/story/voiceover parameters on `submit_highlights` and `render_clip`.

## 5. Git Branching, Model Routing & Reporting Protocol

### 5.1 Branching Strategy
- `main`: Protected integration branch; receives merges only via `git merge --no-ff <phase-branch>` after Phase Review approval.
- Phase branches:
  1. `phase-1-foundation` (Packages `1.1` – `1.3`) — **✅ COMPLETE & MERGED INTO `main`**
  2. `phase-2-core-pipeline` (Packages `2.1` – `2.6`) — **⏳ NEXT**
  3. `phase-3-ai-scoring` (Packages `3.1` – `3.3`)
  4. `phase-4-video-assembly` (Packages `4.1` – `4.6`, including `4.5` YPP Narrator & Audio Ducking)
  5. `phase-5-mcp-server` (Packages `5.1` – `5.2`)

### 5.2 Model Responsibilities
- **Gemini 3.8 (high)**: Implements all 🟢 Easy, 🟡 Medium, and 🟠 Hard packages (`1.1`, `1.2`, `1.3`, `2.1`, `2.2`, `2.3`, `2.4`, `2.5`, `2.6`, `3.2`, `4.1`, `4.2`, `4.4`, `4.5`, `4.6`, `5.2`).
- **Claude Opus 4.6**: Implements all 🔴 Critical packages (`3.1`, `3.3`, `4.3`, `5.1`) and performs Phase Reviews (`2.R` – `5.R`).

### 5.3 Mandatory Package & Phase Audit Reports
Every package implementation turn MUST produce a structured Markdown report before committing:
- **Package Report Path**: `reports/phase-{N}/package-{X.Y}-report.md`
- **Phase Summary Path**: `reports/phase-{N}/phase-{N}-final-report.md`
- **Required Report Sections**:
  1. **Metadata**: Package ID, Phase, Branch name, Commit hash, Timestamp, and Model used.
  2. **Files Created / Modified**: File paths, line counts, and summary of classes/functions.
  3. **Acceptance Criteria Verification**: Table mapping every acceptance criterion from `streamclipper-implementation-guide-v2.md` to the exact test or command output that proved it passed.
  4. **Test & CLI Output**: Verbatim terminal output from `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` and relevant CLI checks.
  5. **Architectural Notes & Edge Cases**: Proactive fixes, Windows/CUDA considerations, and handoff notes for the next package.
