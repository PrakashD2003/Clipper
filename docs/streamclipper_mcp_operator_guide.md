# StreamClipper v4 — MCP Operator Guide (Host LLM Runbook)

> **Purpose:** Load this guide at the start of every content-production session using `streamclipper-mcp`. It defines the exact tool order, dual-channel isolation (`en` vs `hi`), and editorial quality rules required to produce high-retention, YPP-safe GTA 6 videos.

---

## 1. Golden Rules (Read Before Calling Any Tool)

1. **Always confirm the target channel (`channel="en"` or `channel="hi"`):**
   - **Never mix English and Hinglish on the same channel.**
   - `channel="en"` → English narration, US/Tier-1 tone, writes to `output/en/`, uses `topics_ledger_en.json`.
   - `channel="hi"` → Roman Hinglish narration (natural Indian gamer tone), writes to `output/hi/`, uses `topics_ledger_hi.json`.
   - Both channels **share** the same `.cache/` and `library.sqlite` — never re-download or re-transcribe a VOD just to produce the other language.
2. **Never skip visual boundary refinement (`refine_highlight_bounds`):**
   - Streamers react 15–90 seconds *after* an event; chat reacts 5–25 seconds late.
   - Always run `refine_highlight_bounds` (with contact sheets) before `submit_highlights` so clips start right before the **visual trigger**, never mid-scream.
3. **Quality over volume:**
   - Reject boring filler, menus, donation reads, or context-dependent moments. Returning 1–2 great clips from a 4-hour VOD is better than 5 mediocre ones.
4. **YPP originality & copyright safety:**
   - Never render `mode="raw"` for channel uploads. Always write specific, grounded commentary (`narration_intro`, `callout`, `narration_outro`, and `breakdown_commentary` for `clutch`/`fail`/`graphics_detail`/`analysis`).
   - Respect `HIGH_MUSIC_DETECTED` flags — swap to an alternate clip or drop the highlight.
   - Cap hashtags at **3–4 relevant tags** (no stopword tags, no 15-tag spam).

---

## 2. Workflow A — Standalone 9:16 Shorts (Single Stream)

Follow this exact sequence:

1. **Discover or accept URL:**
   - Optional: `discover_streams(game="GTA 6", max_streams=5)` to pick a recent high-view VOD.
2. **Ingest & analyze (shared across `en` and `hi`):**
   - `download_stream_audio(url)`
   - `transcribe_audio(url)`
   - `analyze_signals(url)`
3. **Inspect signal-filtered chunks:**
   - `get_transcript_chunks(url, only_with_signals=True)`
   - Pick only the top 1–3 genuinely viral candidates (clear setup → trigger → reaction payoff).
4. **Refine exact boundaries (mandatory):**
   - Call `refine_highlight_bounds(url, start, end)` for each candidate to snap `start`/`end`/`peak_time` to the actual visual trigger and detect multi-beat `sub_clips` when dead air >= 5s exists.
5. **Write sharp, grounded scripts & submit:**
   - Call `submit_highlights(url, highlights=[...])` with:
     - `hook_title`: <= 8 words, present tense, curiosity gap, no ALL CAPS, never spoils the payoff.
     - `narration_intro`: 1–2 punchy sentences (<= 12 words each) hooking the viewer into the specific moment.
     - `callout`: 4–6 hype words right before the climax.
     - `narration_outro`: 1–2 sentences giving a real opinion or comment-bait question (never "like and subscribe", never a recap).
     - `breakdown_commentary` + `breakdown_callout` + `breakdown_kind` (`zoom_spotlight` for screens/clues/details; `slow_mo` for physics/stunts/clutches).
6. **Render per channel:**
   - English channel: `render_all_selected(url, channel="en")` (or `render_clip(..., channel="en")`)
   - Hinglish channel (optional second pass from same cache, after rewriting scripts in natural Roman Hinglish): `render_all_selected(url, channel="hi")`
7. **Report to user:**
   - List each rendered `.mp4` path, duration, resolved mode, title variants from `*_post.txt`, and any quarantined clips.

---

## 3. Workflow B — 8–10 Min Long-Form Topic Video + 3 Companion Shorts

This is the **primary monetization workflow**. Follow this exact sequence:

1. **Build the shared GTA 6 source library (once for both channels):**
   - `library_add_source(url, channel="StreamerName", game="GTA 6")` for 2–5 VODs / trailers.
   - Use `list_library(game="GTA 6")` or `search_library(...)` to verify indexed sources.
2. **Discover a high-CTR topic:**
   - `suggest_topics(game="GTA 6", channel="en")` (checks that channel's ledger so topics never repeat).
   - Present the top 3–5 concrete, curiosity-driven topic ideas to the user and let them pick one.
3. **Draft the blueprint:**
   - `start_topic_plan(game="GTA 6", topic="...", channel="en")`
   - Verify the plan before asking for approval:
     - Every point has distinct evidence + alternate clips (so narration B-roll never repeats the evidence clip).
     - Sentences are <= 14 words, grounded in real transcript/frame details, zero invented facts.
     - `thumbnail_text` is **<= 3 punchy words** (not a full sentence).
     - Exactly 3 Shorts: `teaser` (points to full video), `short_a` and `short_b` (standalone, zero overlap, no mention of long video).
4. **Human Approval Gate (mandatory):**
   - Show the user the plan summary (title, shape, chapter headings, estimated duration, narration ratio, and thumbnail text).
   - **Stop and wait for explicit user approval** before calling `approve_topic_plan(plan_id, channel="en")`.
5. **Render long-form + companion Shorts:**
   - `render_topic_video(plan_id, channel="en")`
   - `render_topic_shorts(plan_id, channel="en")`
6. **Optional Hinglish adaptation:**
   - If the user wants the same topic for the Hinglish channel, draft a separate Hinglish plan (`start_topic_plan(..., channel="hi")`) using the **already-ingested library** — with culturally natural Roman Hinglish scripts, not literal robotic translation — get approval, and render with `channel="hi"`.
7. **Final Handoff to User:**
   - Provide paths to `long.mp4`, `thumbnail.jpg`, `long_post.txt`, `short_teaser.mp4`, `short_a.mp4`, `short_b.mp4`, and remind the user to:
     1. Watch the first 15 seconds + one chapter transition to verify audio/pacing.
     2. Review `thumbnail.jpg` at mobile size (and polish manually if needed).
     3. Link `short_teaser.mp4` to the long video via YouTube Studio's **Related Video** setting.
     4. Check `assets/LICENSES.md` if any CC-BY background music credit is needed in the description.

---

## 4. Scriptwriting Quality Bar (Anti-"AI Slop" Rules)

- **Never use banned clichés:** *"you won't believe"*, *"watch what happens next"*, *"the chat went wild"*, *"insane moment"*, *"in today's video"*, *"like and subscribe"*.
- **Specifics beat hype:** Reference the exact object, quote, mistake, or chat message from the transcript/contact sheet.
- **English vs. Hinglish voice:**
  - **English (`en`):** Tight, witty US/global gaming commentary—conversational, direct, zero formal essay phrasing.
  - **Hinglish (`hi`):** Natural Roman-script spoken Hindi+English mix (how Indian gaming creators actually talk—e.g. *"Bhai ne bina dekhe bridge se full send kar diya"*), never textbook Hindi or awkward word-for-word English translations.
- **Maintenance:** Only run `cleanup_temp(url)` after the user confirms they are happy with the rendered clips for both channels.
