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
   - Target **30–40 seconds** total Short duration.
4. **Refine exact boundaries & jump-cuts (mandatory):**
   - Call `refine_highlight_bounds(url, start, end)` for each candidate to snap `start`/`end`/`peak_time` to the actual visual trigger.
   - **Target 30–40 seconds** total Short duration.
   - **Always use `sub_clips`** to jump-cut out loading screens, menus, walking filler, or >3s dead air—even when using `breakdown` mode.
   - **Layout selection:** Note that `cinema_split` preserves wide 16:9 screens/text and places captions cleanly below the gameplay without occluding HUD or in-game details.
5. **Write sharp, grounded scripts & select modes:**
   - Call `submit_highlights(url, highlights=[...])` making **progressive `narration_beats`** the default recommended style:
     - Always craft a **30–40s jump-cut Short** (`sub_clips`) paired with **4–5 connected `narration_beats`** (targeting 55–70% narration coverage) that weave around 2–3 second 100%-volume streamer reaction pops:
       - **Beat 1 (0s, Hook):** Open with an intriguing twist or high-stakes claim in 1 punchy line (<= 10 words) using natural pauses (`...`) and exclamation (`!`) so TTS delivers energetic delivery—**NEVER narrate play-by-play what is already obvious on screen**.
       - **Beat 2 (Escalation):** Explain the hidden context or buildup while the stream audio is ducked.
       - **Gap (2–3s):** Let the streamer's raw peak reaction pop at 100% volume without narrator talk-over.
       - **Beat 3 (Payoff / Breakdown):** Reveal why the moment is wild, funny, or mechanically impressive using concrete details from the transcript/screen.
       - **Beat 4–5 (Sharp Closer):** Land a witty 1-line punchline or bold verdict—**NO generic filler questions** (such as *"kya tumne kabhi dekha tha?"* or *"what would you do?"*).
       - Every beat's `text` must match what is visibly on screen at `at_second` (verify against contact sheets/transcript; never praise live gameplay/controls while an intro text or menu screen is showing).
     - `hook_title`: <= 8 words, present tense, curiosity gap, no ALL CAPS, never spoils the payoff.
     - Legacy fields (`narration_intro`, `callout`, `narration_outro`) remain fully supported for backward compatibility, with progressive `narration_beats` as the standard.
     - **Mode selection & Breakdown rules:** Only pick `mode="breakdown"` when there is a genuine tactical/visual moment AND always pass exact `breakdown_focus` + `breakdown_target_time` (the exact second the object/stunt is visible on screen, never `clip_start + 1s`); otherwise use `story` or `narrative`.
     - Only populate `breakdown_kind: "zoom_spotlight"` when a specific visible object (screen, HUD detail, Easter egg) is clearly shown on screen AND `breakdown_focus` names that exact object.
     - `breakdown_commentary` + `breakdown_callout`: grounded directly in the visual stunt or spotlighted detail.
6. **Mandatory Automated Draft & Self-Review Loop (per channel):**
   - **Step 1 (Free Draft):** Always render first with `draft=True` (free `edge-tts`, $0 ElevenLabs):
     `render_all_selected(url, channel="en", draft=True)` (or `render_clip(..., channel="en", draft=True)`)
   - **Step 2 (Self-Review):** Immediately call `review_rendered_video(draft_video_path)` and visually inspect `contact_sheet_path` + `red_flags` + `narration_timeline`. Check:
     (a) Does every spoken beat match what's visibly on screen at that `MM:SS` tile?
     (b) Are captions readable, unobstructed, and free of `□` tofu glyphs?
     (c) Are there any silence gaps >= 1.5s, black loading screens, or awkward crops?
   - **Step 3 (Auto-Refine, max 2 loops):** If any issue is found, fix the script/timestamps/`sub_clips` and re-render `draft=True` (cap at 2 self-refine passes so it never loops endlessly).
   - **Step 4 (Final ElevenLabs Master):** Only after self-review passes cleanly AND the user approves the draft/storyboard, render the final master with `draft=False` (ElevenLabs):
     `render_all_selected(url, channel="en", draft=False)`
   - Hinglish channel (optional second pass from same cache, after rewriting scripts in natural Roman Hinglish): repeat Steps 1–4 with `channel="hi"`.
7. **Report to user:**
   - List each rendered `.mp4` path, duration (verifying 30–40s target), resolved mode, title variants from `*_post.txt`, and any quarantined clips.

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
5. **Mandatory Automated Draft & Self-Review Loop (Long-Form & Companion Shorts):**
   - **Step 1 (Free Draft):** Always render first with `draft=True` (free `edge-tts`, $0 ElevenLabs):
     `render_topic_video(plan_id, channel="en", draft=True)`
     `render_topic_shorts(plan_id, channel="en", draft=True)`
   - **Step 2 (Self-Review):** Immediately call `review_rendered_video(draft_video_path)` for `long_draft.mp4` and each companion short draft (e.g. `short_teaser_draft.mp4`), inspecting `contact_sheet_path` + `red_flags` + `narration_timeline`. Check:
     (a) Does every spoken beat match what's visibly on screen at that `MM:SS` tile?
     (b) Are captions readable, unobstructed, and free of `□` tofu glyphs?
     (c) Are there any silence gaps >= 1.5s, black loading screens, or awkward crops?
   - **Step 3 (Auto-Refine, max 2 loops):** If any issue is found, update the plan outline/evidence/narration via `update_topic_plan` and re-render with `draft=True` (cap at 2 self-refine passes so it never loops endlessly).
   - **Step 4 (Final ElevenLabs Master):** Only after self-review passes cleanly AND the user approves the draft/storyboard, render the final master with `draft=False` (ElevenLabs):
     `render_topic_video(plan_id, channel="en", draft=False)`
     `render_topic_shorts(plan_id, channel="en", draft=False)`
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

- **Progressive Story Arc (Default):** Always write 4–5 progressive beats across 30–40s jump-cut Shorts (55–70% narration coverage), leaving 2–3s reaction gaps for raw streamer screams/laughter at 100% volume without narrator talk-over.
- **Visual preservation:** `cinema_split` preserves wide 16:9 screens/text and places captions cleanly below the gameplay without occluding HUD or in-game details.
- **Never use banned clichés:** *"you won't believe"*, *"watch what happens next"*, *"the chat went wild"*, *"insane moment"*, *"in today's video"*, *"like and subscribe"*, *"watch till end"*, *"kya tumne kabhi dekha"*, *"kya tum ise khelna"*.
- **NEVER narrate play-by-play what is already obvious on screen:** Spoken narration must add stakes, opinions, or curiosity gaps; let the video carry the action. Avoid filler questions like *"kya tumne kabhi dekha tha?"* or *"what would you do?"*.
- **High-energy delivery for ElevenLabs:** Use natural spoken pauses (`...`) and punctuation (`!`, `?`) with punchy rhythm so TTS delivers energetic gamer delivery instead of flat newsreader cadence.
- **Specifics beat hype:** Reference the exact object, quote, mistake, or chat message from the transcript/contact sheet. Match `callout` strictly to what is on screen right before `peak_time`.
- **No blind breakdowns:** Only pick `mode="breakdown"` when there is a genuine tactical/visual moment AND always pass exact `breakdown_focus` + `breakdown_target_time` (the exact second the object/stunt is visible on screen, never `clip_start + 1s`); otherwise use `story` or `narrative`. Only populate `breakdown_kind: "zoom_spotlight"` when a specific visible object (screen, HUD detail, Easter egg) is clearly shown on screen AND `breakdown_focus` names that exact object.
- **English vs. Hinglish voice:**
  - **English (`en`):** Tight, witty US/global gaming commentary—conversational, direct, zero formal essay phrasing.
  - **Hinglish (`hi`):** Casual, punchy Indian gamer speech (short bursts, rhetorical punch, natural Hinglish rhythm), never formal Hindi translation or robotic textbook Hindi.
- **Maintenance:** Only run `cleanup_temp(url)` after the user confirms they are happy with the rendered clips for both channels.
