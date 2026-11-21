# Handoff Report: Brief 06 (T5) — Long-Form 16:9 Topic Video Renderer

**Branch:** `handoff/06-T5-longform-render`  
**Base commit:** `ff8b644` (feat(topics): topic plans with validation, approval and MCP/CLI flow (T4))  
**Commit hash:** `950a5bc`  
**Date:** 2026-11-21  
**Model:** Antigravity (Gemini 3.8 high)

---

## 1. Test Summary Line

```
102 passed, 7562 warnings in 135.39s (0:02:15)
```
Command executed:
```powershell
$env:PYTHONPATH="D:\GTA6-Clipper-wt\handoff-06\src"; $env:STREAMCLIPPER_NO_DOTENV="1"; D:\GTA6-Clipper\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_longform.py tests/test_topics_planner.py tests/test_mcp_server.py
```
- `tests/test_longform.py`: 15 passed (added per-block loudnorm verification and alternate deduplication)
- `tests/test_topics_planner.py`: 33 passed
- `tests/test_mcp_server.py`: 54 passed

Configuration consistency:
```powershell
fc.exe /b config.default.yaml src/streamclipper/config.default.yaml
# Result: FC: no differences encountered (100% byte-identical)
```

---

## 2. Files Changed

- `config.default.yaml`: added `longform:` section with 1920x1080 canvas, 30 fps, card_seconds (2.5), broll_volume (0.12), credit_seconds (4.0), and kenburns_max_zoom (1.08).
- `src/streamclipper/config.default.yaml`: byte-identical addition of `longform:` section.
- `src/streamclipper/output/topic_post.py`: implemented `write_long_post` creating `long_post.txt` with title, description, timestamps/chapters, sources with robust `&t=` parameters via `urllib.parse` and official footage attribution, hashtags, and AI-voice disclosure reminder.
- `src/streamclipper/assembly/longform.py`: implemented `render_long_video`, thumbnail generation (`generate_topic_thumbnail`) with dynamic font scaling, chapter card generation on blurred dark frame, Ken-Burns zoom, ASS subtitle generation with 16:9 narration and lower-third credit styles, per-block loudnorm via `TransformativeNarrator.normalize_audio`, chapter-based 0.4s crossfades with frame-accurate timeline synchronization in story mode, and single final lossy encode with loudnorm (-14 LUFS).
- `src/streamclipper/mcp_server/server.py`: registered `render_topic_video` tool with synchronous approval checks, background job scheduling (`topic_long`), and docstrings.
- `src/streamclipper/cli.py`: implemented `streamclipper topics render PLAN_ID` subcommand with rich status progress and result summary, wrapping execution with user-friendly error handling.
- `tests/test_mcp_server.py`: updated `EXPECTED_TOOLS` to include `render_topic_video` (28 tools total).
- `tests/test_longform.py`: 15 comprehensive tests covering unapproved plan rejection, clip caps enforcement, copyright detection and alternate swapping without duplicate reuse, non-dropping chapter spacing rules, thumbnail dimensions and dynamic font scaling, post file formatting, full synthetic video rendering, per-block loudnorm verification, story crossfade timeline synchronization, MCP job lifecycle, and CLI commands.
- `docs/handoff/06-T5-longform-render.REPORT.md`: this handoff audit report.

---

## 3. Decisions & Architecture

1. **Chapter Preservation Without Dropping Headings**: Enforced YouTube chapter rules (starts at `00:00`, >= 3 chapters, each >= 10s apart) by adjusting and clamping chapter timestamps (`target_t = max(t_actual, prev_t + 10.0)`) instead of dropping valid point headings whose actual start time was under 10 seconds.
2. **Per-Block Loudnorm & Single Final Encode**: Encoded intermediate blocks at high quality (`-crf 18`, `ultrafast`), normalized each intermediate block's audio via `TransformativeNarrator.normalize_audio` (`-c:v copy`), concatenated them, and executed a single final encode with `video.codec` and `video.crf`, applying ASS subtitles (narration captions + lower-third credits) and final loudness normalization (`loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000`) with Windows-safe `cwd` execution.
3. **Synchronized Story Crossfade & MoviePy EOF Protection**: In `story` mode, chapter transitions are marked with 0.4s crossfades between chapter groups, and each crossfade overlap is deducted from the running timeline so burned ASS subtitles, lower-third credits, and chapter markers remain 100% synchronized with the video. Intermediate clips are clamped to probed video stream durations, eliminating MoviePy frame index over-reading warnings.
4. **Safe Windows Reader Handle Cleanup**: Explicitly tracked and closed all underlying MoviePy readers (`broll_video`, `nar_audio`, etc.) using `_close_clip_handles` in `try/finally` blocks before temporary file movement or deletion, avoiding Windows file lock `PermissionError`s.
5. **Dynamic Thumbnail Text Scaling**: Handled thumbnail text fitting with dynamic font size reduction (from 86 down to 36) in Pillow, ensuring text never overflows the 1280x720 canvas.

---

## 4. Left Undone / Handoff to T6

- None. All acceptance criteria for T5 are complete. Ready for Brief 07 (T6: Topic Shorts).
