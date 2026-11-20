# Brief 06 (T5): long 16:9 topic video renderer + chapters + thumbnail + post file

**Start after:** brief 05 is merged into `main` (the lead confirms). Use the latest `main`.
**Your branch:** `handoff/06-T5-longform-render` · **Worktree:** `D:\GTA6-Clipper-wt\handoff-06`

Read `docs/handoff/README.md`, `docs/specs/topic-videos.md` (D5, D8, D11, D15, policy
safeguards) and `docs/specs/topic-videos-contracts.md`. Also AGENTS.md §4 rule 11 (MoviePy v2
API, Windows-safe `cwd` ASS burn, close clips before deleting temps, single final lossy encode,
`h264_nvenc` → `libx264` fallback).

```powershell
git -C D:\GTA6-Clipper fetch origin
git -C D:\GTA6-Clipper worktree add D:\GTA6-Clipper-wt\handoff-06 -b handoff/06-T5-longform-render origin/main
```

## Files you may change

- `src/streamclipper/assembly/longform.py` (new), `src/streamclipper/output/topic_post.py` (new)
- `src/streamclipper/mcp_server/server.py` — only `render_topic_video` + docstring
- `src/streamclipper/cli.py` — only `topics render PLAN_ID`
- both `config.default.yaml` copies — new `longform:` section (byte-identical)
- `tests/test_longform.py` (new); `tests/test_mcp_server.py` only `EXPECTED_TOOLS`

## Video structure (1920×1080, 30 fps)

`cold open` → for each point: [`chapter card` (numbered/countdown only, `longform.card_seconds` 2.5 s:
heading text on a dark blurred frame from the point's first clip)] → `narration block` (point
narration voiced by `TransformativeNarrator.synthesize_speech(language=plan.language)` over
B-roll = that point's footage with original audio ducked to `longform.broll_volume` 0.12, slow
Ken-Burns zoom ≤ 1.08×) → `evidence clips` (each at full volume, original audio, with a **small**
lower-third credit `@channel` — `Official footage: Rockstar Games` for `kind == "official"` — bottom-left,
~28 px text, semi-transparent pill, shown for the first 4 s) → next point … → `outro` narration over the
best B-roll. Story shape: no cards; a 0.4 s crossfade marks chapters instead.

- Narration captions (the spoken narration text, 2-line max, bottom-centre, white with dark
  box) burned in the same single final encode as the credits (ASS file, Windows-safe `cwd`
  burn — reuse `AssCaptionBurner` helpers where possible; add a 16:9 style).
- Loudness: per-block loudnorm, final `loudnorm=I=-14:TP=-1.5:LRA=11` (reuse narrator helpers).
- Encode: intermediate blocks at high quality (`-crf 18`, `ultrafast`), concatenate, then ONE
  final encode with `video.codec`/`video.crf` (nvenc → libx264 fallback like `Exporter`).

## Do

1. `render_long_video(config, plan, *, progress=None) -> dict` in `longform.py`:
   - refuse unless `plan.status == "approved"`; set `rendering`, then `rendered`/`failed` in plan.json.
   - Download each evidence/B-roll range with `download_segment(url, start, end, temp, buffer_seconds=0.0)`
     into `<temp_dir>/topics/<plan_id>/`; reuse files already present.
   - Before use, run `CopyrightRiskDetector(config).detect(audio_path, start, end)` on each
     evidence clip using the source's cached stream audio (`StepCache.load_state(url).audio_path`;
     skip the check with a warning if missing). `HIGH_MUSIC_DETECTED` → swap in the point's
     `alternates` (first that passes) or drop the clip; if a point ends with no evidence, fail with a clear message.
   - Enforce caps again at render time (≤ 20 s, official ≤ 10 s).
   - Build blocks, mix, burn, encode → `long.mp4` in the topic folder; probe final duration;
     measured narration share; warn (not fail) outside 480–600 s or below 40 %.
   - `chapters.txt`: YouTube chapter lines (`00:00 Intro`, then each point heading at its card or
     narration start, `mm:ss` or `h:mm:ss`); YouTube needs ≥ 3 chapters, first at 00:00, each ≥ 10 s.
   - `thumbnail.jpg` 1280×720: frame from the highest-heat evidence clip's midpoint, darkened
     gradient, `plan.thumbnail_text` (≤ 5 words, very large bold, yellow/white with black stroke) — Pillow.
   - Update `plan.outputs` (`long`, `chapters`, `thumbnail`, `long_post`) and save the plan.
   - Report progress through `progress.stage(...)` ("downloading 4/12", "voicing point 3/6",
     "rendering block 5/14", "final encode") when given a `JobProgress`.
2. `topic_post.py`: `write_long_post(config, plan, chapters) -> Path` → `long_post.txt` with title,
   description (`plan.description` + chapters + a **Sources** section listing every channel with
   its video URL and `&t=` timestamp of each clip used; official footage line), tags, the
   AI-voice disclosure wording from `output/post_writer.py`, and "Upload the thumbnail.jpg".
3. MCP `render_topic_video(plan_id)` → background job via `_start_render_job` (kind `topic_long`,
   clips_planned = number of evidence clips); refuses unapproved plans synchronously.
4. CLI `topics render PLAN_ID` (runs `render_long_video` with a rich progress line).
5. Config (both copies): `longform: {width: 1920, height: 1080, fps: 30, card_seconds: 2.5, broll_volume: 0.12, credit_seconds: 4.0, kenburns_max_zoom: 1.08}`.

## Tests (`tests/test_longform.py`, names `test_T5_*`, no network/GPU/TTS)

Use tiny synthetic videos (ffmpeg `testsrc` + `sine` audio) as "downloaded segments" (patch
`download_segment`), patch TTS to write short sine WAVs, patch the copyright detector. Cover:
refuses unapproved plans; block order for numbered vs story; cards only for numbered/countdown;
credits text incl. official; copyright swap to alternate / drop / fail; caps enforced; chapters
file rules (first 00:00, ≥3, each ≥10 s); thumbnail size; post file Sources with timestamps; final
file exists with an audio stream and 1920×1080; plan status/outputs updated; MCP tool starts a job
and refuses unapproved plans. Keep the synthetic plan short (e.g. 3 points, ~40 s total) and allow
`longform` length warnings in tests.

## Done when

- `pytest -q -p no:cacheprovider tests/test_longform.py tests/test_topics_planner.py tests/test_mcp_server.py` → 0 failed; config copies identical.
- Commit: `feat(topics): long-form 16:9 topic video renderer with chapters, thumbnail and post file (T5)` + blank line + `Co-Authored-By: Antigravity <noreply@google.com>`.
- `docs/handoff/06-T5-longform-render.REPORT.md` committed (final hash in your chat reply).
