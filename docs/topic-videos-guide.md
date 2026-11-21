# StreamClipper Topic Videos — User & Integration Guide

> **Transform multiple streams and gameplay videos into narrator-led 16:9 deep dives and 9:16 vertical Shorts.**

StreamClipper's **Topic Video System** turns dozens of streams and no-commentary gameplay videos of the same game into original, transformative content. One approved topic plan produces:
1. **1 Long 16:9 Video** (8–10 minutes) for YouTube mid-roll monetization, watch hours, and chaptered deep dives.
2. **3 Vertical 9:16 Shorts** (35–60 seconds):
   - **1 Teaser Short (`short_teaser.mp4`)**: Hooks the audience and drives traffic to the long video using YouTube's "Related Video" feature.
   - **2 Standalone Shorts (`short_a.mp4`, `short_b.mp4`)**: Independent spotlight videos built around distinct strong beats, with zero footage repetition between them.

---

## 1. End-to-End Workflow

Whether executing via the **CLI (with local Ollama)** or through the **MCP Server (Claude Desktop, Antigravity, or any host LLM)**, the workflow follows 7 deterministic stages:

```
+--------------------+
| 1. Ingest Sources  |  Transcribes audio, extracts chat, indexes 30s FTS5 segments
+---------+----------+
          |
          v
+--------------------+
| 2. Tag Visuals     |  Contact sheets for no-commentary sources via vision models
+---------+----------+
          |
          v
+--------------------+
| 3. Discover Topics |  Clusters indexed evidence into fresh topic candidates
+---------+----------+
          |
          v
+--------------------+
| 4. Draft Plan      |  Picks story shape, builds outline, selects clips, writes narration
+---------+----------+
          |
          v
+--------------------+
| 5. User Approval   |  MANDATORY GATE: User reviews and locks plan before rendering
+---------+----------+
          |
          v
+--------------------+
| 6. Render Long Form|  16:9 video + YouTube chapters + thumbnail + post file
+---------+----------+
          |
          v
+--------------------+
| 7. Render Shorts   |  9:16 vertical videos (teaser + 2 standalone) + post files
+--------------------+
```

---

## 2. CLI Workflow (Local Ollama)

### Step 1: Add Sources to the Library
Ingest stream VODs and official gameplay trailers into your local SQLite FTS5 library:
```powershell
# Add streamer VODs
streamclipper library add "https://www.youtube.com/watch?v=STREAM_VOD_1" --channel "ViceGamer" --game "GTA 6" --kind stream

# Add official Rockstar footage (trailers/reveals)
streamclipper library add "https://www.youtube.com/watch?v=OFFICIAL_TRAILER" --channel "Rockstar Games" --game "GTA 6" --kind official
```

### Step 2: Suggest Topics
Discover candidate topics based on indexed transcript keywords, chat heat, and frame notes:
```powershell
streamclipper topics suggest "GTA 6" --limit 5
```

### Step 3: Create Outline and Script Plan
Draft a plan for a chosen topic:
```powershell
streamclipper topics plan "GTA 6" "hidden-casino-secrets"
```
Inspect the drafted points, evidence clips, and Shorts:
```powershell
streamclipper topics show plan-20261121-001
```

### Step 4: Validate and Approve the Plan
Check against guardrails (narration share, clip duration limits, copyright checks):
```powershell
streamclipper topics validate plan-20261121-001
```
When satisfied with the script and clip selection, approve the plan:
```powershell
streamclipper topics approve plan-20261121-001
```

### Step 5: Render Long 16:9 Video
Render the long-form video with chapter cards, Ken-Burns ducked B-roll, and auto thumbnail:
```powershell
streamclipper topics render plan-20261121-001
```

### Step 6: Render 9:16 Topic Shorts
Render all 3 Shorts or a specific slot:
```powershell
# Render all 3 Shorts (teaser, short_a, short_b)
streamclipper topics shorts plan-20261121-001

# Or render a single slot
streamclipper topics shorts plan-20261121-001 --slot teaser
streamclipper topics shorts plan-20261121-001 --slot short_a
streamclipper topics shorts plan-20261121-001 --slot short_b
```

---

## 3. MCP Server Workflow (Claude / Antigravity / Any Host LLM)

When running StreamClipper as an MCP server (`streamclipper-mcp`), the host LLM drives the pipeline via native tools:

| Stage | MCP Tool Call | Description |
| :--- | :--- | :--- |
| **Ingest** | `library_add_source(url, game, kind="stream", channel="")` | Audio-first download and FTS5 indexing. |
| **Vision** | `index_frames(source_id)` → `get_frame_sheets` → `submit_frame_notes` | Visual tagging for no-commentary gameplay. |
| **Discover** | `get_topic_material(game)` → `record_topic_candidates` | Scans index, checks ledger, proposes topics. |
| **Plan** | `start_topic_plan(game, topic_id)` → `submit_topic_plan(plan_json)` | Synthesizes outline, selects clips, drafts narration. |
| **Approval** | `get_topic_plan(plan_id)` → `approve_topic_plan(plan_id)` | **Synchronous gate**: user reviews and approves plan. |
| **Render Long** | `render_topic_video(plan_id)` | Starts async background job (`topic_long`). |
| **Render Shorts** | `render_topic_shorts(plan_id, slots="")` | Starts async background job (`topic_shorts`). |
| **Monitor** | `get_render_status(job_id)` | Polls progress and retrieves output file paths. |

---

## 4. Where Outputs Go

All rendered artifacts for a topic are organized under:
```
<output_dir>/<game>/topics/<topic_id>/
```

### File Layout:
```
output/
└── GTA 6/
    └── topics/
        └── hidden-casino-secrets/
            ├── plan.json                 # Single source of truth (TopicPlan)
            ├── long.mp4                  # 1920x1080 30fps long-form video
            ├── chapters.txt              # YouTube-formatted chapters
            ├── thumbnail.jpg             # 1280x720 video thumbnail with headline
            ├── long_post.txt             # Ready-to-paste long video description & sources
            ├── short_teaser.mp4          # 1080x1920 9:16 vertical teaser short
            ├── short_teaser_post.txt     # Teaser description (includes Related Video reminder)
            ├── short_a.mp4               # 1080x1920 9:16 standalone short A
            ├── short_a_post.txt          # Standalone short A description
            ├── short_b.mp4               # 1080x1920 9:16 standalone short B
            └── short_b_post.txt          # Standalone short B description
```

---

## 5. Policy Safeguards & Fair Use Rules

To ensure 100% compliance with YouTube Partner Program monetization policies and avoid reused content strikes:

### 1. Narrator-Led Transformative Commentary
- Spoken narration must cover **≥ 40%** of the long video duration.
- Narration is synthesized in the plan's language (`english` or `hinglish`) using ElevenLabs, Edge-TTS, or Kokoro, and mixed over ducked background audio (`18%` volume).
- **Narration Gate**: If a Short's intro commentary fails to mix, the Short is automatically moved to `<output_dir>/_quarantine/` and never shipped silent.

### 2. Clip Duration Caps
- **Streamer Evidence Clips**: Capped at **≤ 20 seconds** per beat.
- **Official Footage (Rockstar Games)**: Capped at **≤ 10 seconds** per beat and flagged with `kind="official"`.
- **Shorts Trimming**: Multiple clips in a Short are trimmed around their midpoints so the final video lands in **35–60 seconds** while keeping each clip **≥ 4 seconds**.

### 3. Attribution & On-Screen Credits
- Every evidence clip shows a lower-third credit tag during playback:
  - Streamers: `@ChannelName`
  - Official footage: `Rockstar Games`
- Every post file (`long_post.txt`, `short_*_post.txt`) contains a full **SOURCES** section with channel names and clickable timestamped links (`&t=`).

---

## 6. YouTube Upload Checklist

When publishing your rendered topic package to YouTube Studio:

### For the Long 16:9 Video:
- [ ] **Title**: Copy from `TITLE` in `long_post.txt`.
- [ ] **Description**: Copy `DESCRIPTION` block from `long_post.txt`.
- [ ] **Timestamps**: Verify YouTube chapters formatted in description (starts at `00:00`, ≥ 3 chapters, each ≥ 10s apart).
- [ ] **Thumbnail**: Upload the generated `thumbnail.jpg` (1280×720).
- [ ] **Altered Content (AI Disclosure)**: In YouTube Studio Advanced Settings, check **Yes** for altered/synthetic content (reminded by `long_post.txt`).

### For the Teaser Short (`short_teaser.mp4`):
- [ ] **Video Upload**: Upload `short_teaser.mp4` as a YouTube Short.
- [ ] **Related Video Link**: In YouTube Studio Video Details, under **Related video**, select your newly uploaded long-form topic video.
- [ ] **Post Metadata**: Copy Title, Description, and Hashtags (`#Shorts`) from `short_teaser_post.txt`.
- [ ] **Altered Content (AI Disclosure)**: Check **Yes** for AI voiceover.

### For Standalone Shorts (`short_a.mp4`, `short_b.mp4`):
- [ ] **Upload**: Upload each standalone MP4 independently.
- [ ] **Post Metadata**: Copy Title, Description, and Hashtags from the respective `short_*_post.txt`.
- [ ] **Altered Content (AI Disclosure)**: Check **Yes** for AI voiceover.
- [ ] **No Long Video Link**: Do not link Related Video on standalone shorts (they stand on their own).
