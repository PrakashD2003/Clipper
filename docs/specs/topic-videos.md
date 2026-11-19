# Topic Videos — Design Spec (v1)

Source of decisions: design Q&A 2026-11-20. Status: approved design, not built.

## Goal

Turn many streams (many streamers + no-commentary gameplay of the same game) into **original,
narrator-led videos about one topic** ("every hidden detail in the docks", "how the wanted
system works"). One approved script produces:

- **1 long 16:9 video, 8–10 min** (mid-roll ads, watch hours), and
- **3 Shorts (9:16):** 1 teaser that points to the long video + 2 standalone Shorts, each built
  around one strong point, with no footage repeated between the two standalone Shorts.

## Decisions

| # | Topic | Decision |
| :--- | :--- | :--- |
| D1 | Formats | Long 16:9 + Shorts from one script |
| D2 | Topic ideas | AI proposes topics with evidence counts; user approves (or types own) |
| D3 | Sources | All indexed streams of the same game, any streamer, plus no-commentary gameplay videos |
| D4 | Official footage | Rockstar trailers/reveals allowed only as clips **under 10 s**, flagged `official` and credited |
| D5 | Structure | Narrator-led: the ElevenLabs voice explains point by point; between points short source clips play at full volume as evidence/reaction beats |
| D6 | Driver | Any LLM attached to the MCP (host-agnostic tools); locally the CLI uses Ollama |
| D7 | Visual search | No-commentary videos: the MCP host LLM reads frame contact sheets; locally an Ollama vision model |
| D8 | Length | Long video 8–10 min |
| D9 | Story shapes | Rotate: numbered explainer, countdown ranking, mystery → evidence → verdict |
| D10 | Approval | User approves outline + script + clip list before the long render |
| D11 | 16:9 look | Chapter title cards **only for numbered/countdown shapes** (story shape gets chapter markers without cards); small lower-third credit; narration captions; auto thumbnail |
| D12 | Shorts | 1 teaser (ends with "full breakdown on the channel"; post file reminds to set YouTube's Related Video) + 2 standalone |
| D13 | Search | Local searchable index of all transcribed streams (speech, chat, heatmap, frame notes) |
| D14 | Output | `output/<game>/topics/<topic-slug>/` holds long video, Shorts, thumbnail, chapters, post files |
| D15 | Credits | Small on-screen credit while a source plays + full "Sources" list with links in description/post file |

## Architecture

### 1. Source library and index (`library/`)
- `library/sources.py`: register a source video (URL, streamer or channel, game, kind:
  `stream` | `gameplay` | `official`). Reuses download (audio first), transcribe, chat,
  heatmap and loudness steps; no clips rendered.
- `library/index.py`: SQLite at `<cache_dir>/library.sqlite` with FTS5 tables:
  `segments(source_id, start, end, text, chat_text, heat, loudness, kind)` (~30 s windows) and
  `frame_notes(source_id, t, note)`. Keyword search via FTS5; optional embedding re-rank later.
- `library/frames.py`: contact sheets (e.g. 3×3 frames per 90 s) for visual tagging; notes come
  from the MCP host LLM (`submit_frame_notes`) or a local Ollama vision model.

### 2. Topic discovery
- `topics/discover.py`: cluster indexed segments (keywords + LLM) into candidate topics with
  evidence counts per source; avoid topics already produced (`VarietyState`-like ledger).
- MCP `suggest_topics(game, limit)` → candidates; user approves one (or names their own).

### 3. Outline, script and clip selection
- `topics/planner.py`: pick the story shape (rotation), build an outline of 5–7 points for the
  long video, search the index per point, select 1–2 evidence clips per point (prefer distinct
  sources; official clips ≤ 10 s), write the narration with the shared `narration_style` guide
  (energetic, grounded, interactive), and mark which points feed the teaser / standalone Shorts.
- Output: `TopicPlan` (serializable) saved under the topic folder; status `draft`.
- MCP `plan_topic_video(topic, game)` returns the plan; `update_topic_plan(...)` applies edits;
  `approve_topic_plan(plan_id)` locks it. CLI: `streamclipper topic plan|approve`.

### 4. Rendering
- `assembly/longform.py`: 1920×1080 timeline: cold open → per point [chapter card (numbered
  shapes only) → narration over B-roll → evidence clip at full volume with small lower-third
  credit] → outro. Narration captions burned; loudnorm −14 LUFS; ElevenLabs voice per language;
  YouTube chapters file; thumbnail from strongest frame + title.
- Shorts reuse the 9:16 layouts and narration pipeline for 3 Shorts from the approved plan.
- Rendering runs as async MCP jobs (`render_topic_video(plan_id)` → job_id; poll
  `get_render_status`).

### 5. Policy safeguards
- Narration must cover ≥ 40 % of the long video; each evidence clip ≤ 20 s (official ≤ 10 s).
- Same music/copyright check per clip; flagged clips are swapped or dropped.
- Post files: title, description with chapters and Sources list (every streamer/channel link),
  AI-voice disclosure wording, Related Video reminder for the teaser.

## Build plan (briefs)

| Brief | Scope | Depends on |
| :--- | :--- | :--- |
| T1 | Source library + SQLite FTS index + CLI `library add/search` | — |
| T2 | Frame contact sheets + frame-note tools (MCP + Ollama vision) | T1 |
| T3 | Topic discovery + MCP `suggest_topics` | T1 |
| T4 | Planner (`TopicPlan`, shapes, clip selection, script) + approve flow | T1, T3 |
| T5 | Long-form 16:9 renderer + chapters + thumbnail | T4 |
| T6 | Topic Shorts (teaser + 2 standalone) + post files | T4, T5 |
