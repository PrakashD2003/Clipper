# AGENTS.md — StreamClipper v4 Agent Instructions

> **Repo:** `D:\GTA6-Clipper` (`streamclipper`)
> **Active Branch:** `v4-quality` (branched from `main`; never commit directly to `main`)
> **Active Plan:** `docs/v4/v4-implementation-plan.md`
> **Prompt Runbook:** `docs/v4/v4-antigravity-prompts.md`

This file is the single source of truth for AI coding agents (including **Antigravity / Gemini 3.8 Flash High**) working on StreamClipper v4. Keep edits strictly scoped to the active ticket.

---

## 1. Environment & Commands (Windows)

Always run Python, pytest, and CLI commands via `.venv`:

| Tool | Command |
| :--- | :--- |
| Python | `D:\GTA6-Clipper\.venv\Scripts\python.exe` |
| Targeted tests | `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_<module>.py -v` |
| Full test suite | `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v` |
| CLI | `D:\GTA6-Clipper\.venv\Scripts\streamclipper.exe` |
| FFmpeg | `ffmpeg` (must include `libass` / `ass` filter and optional `h264_nvenc`) |

---

## 2. Hard Rules (Never Violate)

1. **Scope strictly to the current ticket:** Only modify the files explicitly listed in the ticket prompt. Do not refactor unrelated modules, rename public functions, or "clean up" other files.
2. **Keep both `config.default.yaml` copies identical:** `config.default.yaml` (repo root) and `src/streamclipper/config.default.yaml` MUST remain byte-for-byte identical (`test_G12_default_config_copies_are_identical` enforces this).
3. **Preserve backward compatibility:** Any new fields in `models.py`, `topics/models.py`, or `config.default.yaml` MUST have sensible defaults and `.get(...)` fallbacks in `from_dict()` so all 710+ existing tests keep passing.
4. **Never hardcode GPU settings:** Keep `transcription.device`, `model_size`, `compute_type`, `batch_size`, `llm.model`, and `video.codec` set to `"auto"` in `config.default.yaml`. Respect `HardwareProfile` (GTX 1660 SUPER 6 GB / RTX 4050 6 GB / CPU).
5. **MoviePy v2 API only:** Import directly from `moviepy` and use `.cropped()`, `.resized()`, `.subclipped()`, `.with_position()`, `.with_duration()`, `.with_audio()`. Always close opened clips in `try/finally` before deleting temp files on Windows.
6. **Windows-safe FFmpeg `ass` burn-in:** Always write/close the `.ass` file first, then run `ffmpeg` with `cwd=str(ass_path.resolve().parent)` and `-vf ass={escaped_ass_name}`. Always fall back from `h264_nvenc` to `libx264` on encoder failure, and log `exc.stderr[-500:]`.
7. **Single lossy video encode:** Write intermediate video at `preset="ultrafast"`, `-crf 18`, use `-c:v copy` during audio mixing, and do the single final lossy encode during final ASS burn-in.
8. **Audio ducking:** Always append `:eval=frame` to FFmpeg time-varying `volume` filters and normalize audio streams to `48000 Hz` stereo before `amix`.
9. **No silent test-tone fallbacks in production:** Never substitute a synthetic sine wave for missing TTS audio in production renders. If required narration fails, raise or quarantine the clip cleanly.
10. **Channel-language isolation:** Never mix `english` and `hinglish` outputs into the same channel folder or topic ledger. Channel profiles (`en` vs `hi`) keep separate output roots, voices, and topic ledgers while sharing the same ingested source `library.sqlite` and `.cache/`.

---

## 3. Per-Ticket Workflow

For every ticket in `docs/v4/v4-antigravity-prompts.md`:
1. Read only the target files named in the prompt.
2. Implement the exact change and add/update unit tests covering the new behavior.
3. Run targeted `pytest` first, then run the full `pytest -v` suite.
4. Do NOT invent extra features outside the ticket.
5. Report back with:
   - Modified files and summary of changes
   - Exact `pytest` pass/fail counts
   - Any edge cases or manual config needed before the user runs the real-video render test.
