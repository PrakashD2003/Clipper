# Handoff Report: Brief 07 (T6) — Topic Shorts (Teaser + 2 Standalone) + Docs

**Branch:** `handoff/07-T6-topic-shorts`  
**Base commit:** `8fde732` (feat(topics): long-form 16:9 topic video renderer with chapters, thumbnail and post file (T5))  
**Commit hash:** `af4fdbf` (will match HEAD)  
**Date:** 2026-11-21  
**Model:** Antigravity (Gemini 3.8 high)

---

## 1. Test Summary Line

```
82 passed, 7811 warnings in 205.07s (0:03:25)
```
Command executed:
```powershell
$env:PYTHONPATH="D:\GTA6-Clipper-wt\handoff-07\src"; $env:STREAMCLIPPER_NO_DOTENV="1"; D:\GTA6-Clipper\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_topic_shorts.py tests/test_longform.py tests/test_mcp_server.py
```
- `tests/test_topic_shorts.py`: 13 passed
- `tests/test_longform.py`: 15 passed
- `tests/test_mcp_server.py`: 54 passed

Topic Planner suite verification:
```powershell
$env:PYTHONPATH="D:\GTA6-Clipper-wt\handoff-07\src"; $env:STREAMCLIPPER_NO_DOTENV="1"; D:\GTA6-Clipper\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_topics_planner.py
# Result: 33 passed in 10.16s
```

Configuration consistency:
```powershell
fc.exe /b D:\GTA6-Clipper-wt\handoff-07\config.default.yaml D:\GTA6-Clipper-wt\handoff-07\src\streamclipper\config.default.yaml
# Result: FC: no differences encountered (100% byte-identical)
```

---

## 2. Files Changed

- `src/streamclipper/assembly/topic_shorts.py`: implemented `render_topic_shorts` assembling 9:16 vertical Shorts (`short_teaser.mp4`, `short_a.mp4`, `short_b.mp4`) with `VideoComposer.compose(..., layout_kind="blur_fill")`, clip duration clamping (35–60 s total, ≥ 4 s per clip trimmed around middle), slot alias normalization, audio ducking and voicing via `TransformativeNarrator` (`narration_intro`, callout before strongest clip, `narration_outro`), silent audio fallback on audioclipless segments, narration gate quarantine for unvoiced intros with stale output eviction, job cancellation & progress reporting hooks (`start_clip`, `clip_done`), persistent hook title and `@channel`/official credit tags burned via 9:16 ASS overlay, and final loudness normalization (-14 LUFS).
- `src/streamclipper/output/topic_post.py`: implemented `write_short_post` generating `<name>_post.txt` with title, description, timestamped sources (`&t=`), hashtags (`#Shorts`, game, topic), AI-voice disclosure reminder, empty-point fallback safety, and the YouTube Related Video reminder strictly for teaser shorts (standalone posts omit any mention of the long video).
- `src/streamclipper/mcp_server/server.py`: registered `render_topic_shorts` tool with synchronous approval gating, background job lifecycle (`topic_shorts`), and updated final `SERVER_INSTRUCTIONS` workflow summary line.
- `src/streamclipper/cli.py`: implemented `streamclipper topics shorts PLAN_ID [--slot teaser|short_a|short_b]` CLI subcommand with rich progress and result table.
- `docs/topic-videos-guide.md`: comprehensive end-to-end user and integration guide covering CLI (Ollama) and MCP (any host LLM) workflows, file output hierarchy, official footage & streamer fair-use rules, and the complete YouTube upload checklist. Corrected CLI `--channel` parameter and MCP `library_add_source` tool signature.
- `README.md`: added "Topic Videos (16:9 Long-Form & 9:16 Shorts)" overview section linking directly to `docs/topic-videos-guide.md`, with accurate CLI `--channel` command.
- `tests/test_topic_shorts.py`: 13 comprehensive tests covering unapproved plan refusal, per-slot evidence selection, 35–60 s duration clamp with midpoint trimming, persistent hook and credit tag ASS burn-in, teaser vs standalone post reminders, narration gate quarantine upon intro voice failure, 1080x1920 video with AAC audio stream verification, MCP job dispatch, CLI invocation, empty point_indexes fallback, slot alias normalization, progress tracking with cancellation, and stale output eviction upon quarantine.
- `tests/test_mcp_server.py`: updated `EXPECTED_TOOLS` to include `render_topic_shorts` (29 tools total).
- `docs/handoff/07-T6-topic-shorts.REPORT.md`: this handoff report.

---

## 3. Decisions & Architecture

1. **Midpoint Trimming & Clamping**: When total evidence clip duration exceeds 60 s, `_calculate_clip_trims` calculates proportional allocation targeting 45–50 s, trimming each clip around its exact midpoint while guaranteeing each clip retains ≥ 4.0 s.
2. **Persistent Hook & Lower-Third Credits via 9:16 ASS**: Implemented custom 9:16 vertical ASS styles (`HookTitle` aligned top-center at `Alignment=8`, `MarginV=180` persistent across entire duration; `CreditTag` aligned bottom-left at `Alignment=1`, `MarginV=280` active for each clip's duration; `NarrationCard` aligned bottom-center at `Alignment=2`, `MarginV=400`). Both hook title and credit tags are rendered in the ASS filtergraph during the single final lossy encode, preserving pristine quality and avoiding multi-pass re-encodes.
3. **Hard Narration Gate with Quarantine**: If `narration_intro` was expected on a Short but the narration mix failed or produced zero audio, the short is quarantined to `<output_dir>/_quarantine/<game>/topics/<topic_id>/` with normalized audio for inspection, reported in the result dictionary under `"quarantined"`, and any stale entry in `plan.outputs` for that slot is purged.
4. **Teaser vs Standalone Post Distinction**: `write_short_post` strictly checks `short.slot == "teaser" or short.kind == "teaser"`. Only teaser posts include "- Set YouTube's Related Video to the long video" and refer to the full breakdown. Standalone posts never mention the long video.
5. **Windows-Safe Resource Management**: Wrapped MoviePy composed clips, beat clips, and opened source clip handles with `_close_clip_handles` in `try/finally` blocks before deleting or moving intermediate files, avoiding Windows file locking `PermissionError`s.
6. **Robust Slot Name Normalization & Job Progress Integration**: Support aliases like `"short_teaser"`, `"a"`, and `"b"`, and bridge `progress.start_clip`, `progress.clip_done`, and `progress.cancelled` into `render_topic_shorts` for accurate progress reporting to background MCP callers.

---

## 4. Left Undone / Handoff

- None. All acceptance criteria for Brief 07 (T6) are complete and tested.
