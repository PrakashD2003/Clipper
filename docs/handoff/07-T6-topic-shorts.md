# Brief 07 (T6): topic Shorts (teaser + 2 standalone) + docs

**Start after:** brief 06 is merged into `main` (the lead confirms). Use the latest `main`.
**Your branch:** `handoff/07-T6-topic-shorts` · **Worktree:** `D:\GTA6-Clipper-wt\handoff-07`

Read `docs/handoff/README.md`, `docs/specs/topic-videos.md` (D1, D12, D15) and
`docs/specs/topic-videos-contracts.md` (`ShortPlan`, paths). AGENTS.md §4 rule 11 applies.

```powershell
git -C D:\GTA6-Clipper fetch origin
git -C D:\GTA6-Clipper worktree add D:\GTA6-Clipper-wt\handoff-07 -b handoff/07-T6-topic-shorts origin/main
```

## Files you may change

- `src/streamclipper/assembly/topic_shorts.py` (new)
- `src/streamclipper/output/topic_post.py` — add `write_short_post`
- `src/streamclipper/mcp_server/server.py` — only `render_topic_shorts` + docstring + final `SERVER_INSTRUCTIONS` topic workflow line
- `src/streamclipper/cli.py` — only `topics shorts PLAN_ID`
- `README.md` — a "Topic videos" section; `docs/topic-videos-guide.md` (new user guide)
- `tests/test_topic_shorts.py` (new); `tests/test_mcp_server.py` only `EXPECTED_TOOLS`

## Do

1. `render_topic_shorts(config, plan, *, slots=("teaser","short_a","short_b"), progress=None) -> dict`
   in `topic_shorts.py`. For each `ShortPlan` (plan must be `approved`):
   - Clips = the evidence of its `point_indexes` (reuse the segments T5 downloaded under
     `<temp_dir>/topics/<plan_id>/`; download if missing), trimmed so the Short lands in
     35–60 s including narration (trim each clip around its middle; keep ≥ 4 s per clip).
   - Compose each clip into 9:16 with the existing layout system (`VideoComposer.compose(..., layout_kind="blur_fill")`
     or the same functions it uses) — persistent hook title (`ShortPlan.hook_title`), small
     `@channel` credit tag (official: `Rockstar Games`), then concatenate the clips.
   - Narration: `narration_intro` at the start, `callout` before the strongest clip,
     `narration_outro` at the end, voiced with `plan.language` through the existing narrator and
     mixed with ducking (`mix_narration_with_ducking`), captions/cards via `Exporter`/`AssCaptionBurner`,
     final loudnorm −14 LUFS, one final encode. Reuse `Exporter.export(...)` if it can take the
     concatenated clip directly; otherwise mirror its steps without duplicating helpers.
   - Apply the existing narration gate: if the intro voice didn't mix, quarantine that Short
     (same rule as single clips) and report it.
   - Outputs `short_teaser.mp4`, `short_a.mp4`, `short_b.mp4` in the topic folder; update
     `plan.outputs`.
2. `write_short_post(config, plan, short, video_path) -> Path`: title, description with credits
   (each channel + URL + timestamp), hashtags from the game/topic, AI-voice wording, and for the
   teaser the reminder "Set YouTube's Related Video to the long video". Standalone posts don't
   mention the long video.
3. MCP `render_topic_shorts(plan_id, slots="")` → background job (kind `topic_shorts`,
   clips_planned = number of Shorts); refuses unapproved plans synchronously. Final
   `SERVER_INSTRUCTIONS` line summarising the whole topic workflow (add sources → frame notes for
   no-commentary → material → candidates → user picks → plan → user approves →
   `render_topic_video` → `render_topic_shorts` → poll `get_render_status`).
4. CLI `topics shorts PLAN_ID [--slot teaser|short_a|short_b]`.
5. Docs: `docs/topic-videos-guide.md` — the end-to-end workflow for CLI (Ollama) and MCP (any
   LLM), commands, where outputs go, approval step, credits/official-footage rules, upload
   checklist (chapters, thumbnail, Related Video, AI disclosure). Short "Topic videos" section in
   `README.md` linking to it.

## Tests (`tests/test_topic_shorts.py`, names `test_T6_*`, no network/GPU/TTS)

Synthetic segments and patched TTS as in T5. Cover: refuses unapproved plans; per-slot clip
selection from point_indexes; duration clamp 35–60 s; 1080×1920 output with audio; credit tag and
hook present in the ASS/overlay inputs; teaser post has the Related Video reminder, standalone
posts don't mention the long video; quarantine when intro voice fails; MCP tool starts a job.

## Done when

- `pytest -q -p no:cacheprovider tests/test_topic_shorts.py tests/test_longform.py tests/test_mcp_server.py` → 0 failed.
- Commit: `feat(topics): topic Shorts (teaser + standalone), post files and guide (T6)` + blank line + `Co-Authored-By: Antigravity <noreply@google.com>`.
- `docs/handoff/07-T6-topic-shorts.REPORT.md` committed (final hash in your chat reply).
