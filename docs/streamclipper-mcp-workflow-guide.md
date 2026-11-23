# StreamClipper MCP Operator Runbook & Workflow Guide

> **Target Audience:** Human Operators, MCP Client LLMs (Claude Desktop, Antigravity, Cursor, Goose, Custom Agents)  
> **Resource URI:** `streamclipper://workflow-guide`  
> **Repository:** `D:\GTA6-Clipper` (`streamclipper` v2.2)  
> **Core Mission:** Convert raw, multi-hour gaming live streams into highly engaging, viral 9:16 vertical YouTube Shorts and TikToks that strictly satisfy the [YouTube Channel Monetization Policies](https://support.google.com/youtube/answer/1311392?hl=en) and [YouTube Reused Content FAQ](https://support.google.com/youtube/community-guide/271248162/%F0%9F%94%8E-faq-reused-content-youtube%E2%80%99s-partner-program?hl=en).

---

## Table of Contents

1. [Executive Summary & YPP Compliance](#1-executive-summary--ypp-compliance)
2. [Dual Execution Models: CLI vs MCP](#2-dual-execution-models-cli-vs-mcp)
3. [End-to-End Tool Sequence (Step-by-Step Operator Flow)](#3-end-to-end-tool-sequence-step-by-step-operator-flow)
   - [Step 1: Download Stream Audio](#step-1-download-stream-audio-download_stream_audio)
   - [Step 2: Speech Transcription (Asynchronous)](#step-2-speech-transcription-asynchronous-transcribe_audio)
   - [Step 3: Multi-Modal Signal Fusion](#step-3-multi-modal-signal-fusion-analyze_signals)
   - [Step 4: Scout Candidate Chunks](#step-4-scout-candidate-chunks-get_transcript_chunks)
   - [Step 5: COMPULSORY Visual Verification (Contact Sheets)](#step-5-compulsory-visual-verification-inspect_moment_context--refine_highlight_bounds)
   - [Step 6: Submit Formatted Highlights](#step-6-submit-formatted-highlights-submit_highlights)
   - [Step 7: Vertical Video Assembly (Asynchronous)](#step-7-vertical-video-assembly-asynchronous-render_all_selected--render_clip)
   - [Step 8: Generate Post-Production Metadata](#step-8-generate-post-production-metadata-generate_post_metadata)
   - [Step 9: Workspace Hygiene & Temp Cleanup](#step-9-workspace-hygiene--temp-cleanup-cleanup_temp)
4. [Tool JSON Specifications & Payloads](#4-tool-json-specifications--payloads)
5. [Async Polling Rules & Background Jobs](#5-async-polling-rules--background-jobs)
6. [Visual Inspection Masterclass: Sheet 1 & Sheet 2 Analysis](#6-visual-inspection-masterclass-sheet-1--sheet-2-analysis)
   - [The "Delayed Reaction Trap"](#the-delayed-reaction-trap)
   - [Anatomy of Sheet 1: Buildup & Trigger](#anatomy-of-sheet-1-buildup--trigger-t-120s-to-t-30s)
   - [Anatomy of Sheet 2: Climax & Reaction](#anatomy-of-sheet-2-climax--reaction-t-30s-to-t60s)
   - [Multi-Beat Jump-Cut Story Detection](#multi-beat-jump-cut-story-detection)
7. [Transcript Grounding & Commentary Guardrails](#7-transcript-grounding--commentary-guardrails)
   - [Moment Types & Adaptive Tone](#moment-types--adaptive-tone)
   - [Transformation Modes](#transformation-modes)
   - [The Narration Quality Checker](#the-narration-quality-checker)
8. [Ready-to-Copy Starter Prompt for New Chats](#8-ready-to-copy-starter-prompt-for-new-chats)

---

## 1. Executive Summary & YPP Compliance

Raw video game re-uploads or simple uncut stream clips face instant demonetization or rejection under YouTube's **Reused Content** policy. To qualify as transformative content under the YouTube Partner Program (YPP), each clip must include:
1. **Meaningful Human-like Commentary**: Contextual introduction (`narration_intro`) that hooks viewers within the first 1.5 seconds, and a closing punchline or takeaway (`narration_outro`).
2. **Audio Ducking**: Streamer audio ducked to 18% during narration, returning to 100% full volume during the climax/payoff.
3. **Adaptive Visual Context**: On-screen context pill badges (`context_badge`), animated Hormozi/bounce ASS subtitles, and spike-synced punch-in zoom (1.20x center zoom at peak excitement).
4. **Authentic Community Reaction**: Real viewer chat velocity bursts formatted into live-stream chat ticker overlays.

StreamClipper automates this entire pipeline while giving human and AI operators full precision control over moment curation.

---

## 2. Dual Execution Models: CLI vs MCP

StreamClipper operates in two primary modes:
- **CLI Mode (`streamclipper`)**: Terminal-driven command-line tool with rich progress bars, interactive curses-like candidate selection, and headless batch operations.
- **MCP Mode (`streamclipper-mcp`)**: Model Context Protocol stdio server (`MCPServer` from MCP SDK v2). Exposes 30+ tools and resources allowing client LLMs (Claude Desktop, Cursor, Antigravity) to inspect transcripts, view visual contact sheets, refine timestamps, and trigger background render jobs.

---

## 3. End-to-End Tool Sequence (Step-by-Step Operator Flow)

The StreamClipper MCP pipeline follows a strictly ordered, 9-step progression. Never skip steps or reorder them.

```
+---------------------------------------------------------------------------------------------+
|                                STREAMCLIPPER TOOL SEQUENCE                                  |
+---------------------------------------------------------------------------------------------+
| 1. download_stream_audio  --> Download 16kHz mono WAV + live chat (.live_chat.json)        |
| 2. transcribe_audio       --> [ASYNC] WhisperX word-level alignment (poll get_render_status)|
| 3. analyze_signals        --> RMS loudness spikes + chat velocity spikes                    |
| 4. get_transcript_chunks  --> Scout candidate highlight moments + style guide & language   |
| 5. inspect_moment_context --> [MANDATORY] Inspect visual Sheet 1 & Sheet 2 + chat bursts   |
|    / refine_highlight_bounds                                                                |
| 6. submit_highlights      --> Submit validated highlights with grounded intro/outro & badge |
| 7. render_all_selected    --> [ASYNC] Render 9:16 clips with ducking & ASS (poll status)   |
| 8. generate_post_metadata --> Generate titles, descriptions, hashtags, and pinned comment   |
| 9. cleanup_temp           --> Safe removal of intermediate segment cuts & temporary frames |
+---------------------------------------------------------------------------------------------+
```

---

### Step 1: Download Stream Audio (`download_stream_audio`)
- **Action**: Extracts lightweight 16kHz mono WAV audio (~460 MB for a 4-hour stream) and live chat replay (`.live_chat.json`).
- **Why**: Video download is postponed until top highlights are chosen; this saves ~95% network bandwidth and disk space.
- **Tool**: `download_stream_audio(url="https://www.youtube.com/watch?v=...")`

### Step 2: Speech Transcription (Asynchronous) (`transcribe_audio`)
- **Action**: Runs WhisperX with wav2vec2 forced phoneme alignment for word-level timestamps.
- **Execution**: **ASYNCHRONOUS**. Validates input, starts a background job, and returns `{"job_id": "...", "status": "started"}` immediately.
- **Rule**: Operator must poll `get_render_status(job_id=...)` every ~20 seconds until `status == "done"`. Never abort or fall back to local ffmpeg while running.

### Step 3: Multi-Modal Signal Fusion (`analyze_signals`)
- **Action**: Computes continuous RMS audio loudness spikes (constant-memory librosa streaming) and detects chat velocity spikes (rolling messages/second bursts).
- **Tool**: `analyze_signals(url="...")`
- **Output**: Returns detected loudness peaks and chat velocity bursts.

### Step 4: Scout Candidate Chunks (`get_transcript_chunks`)
- **Action**: Retrieves candidate transcript chunks fused with loudness and chat spikes.
- **Output**: Provides candidate timestamps, spoken text, audio/chat scores, active narration language (`english` or `hinglish`), and the `narration_guide`.
- **Tool**: `get_transcript_chunks(url="...", only_with_signals=True)`

### Step 5: COMPULSORY Visual Verification (`inspect_moment_context` & `refine_highlight_bounds`)
> [!IMPORTANT]
> **COMPULSORY STEP**: Never submit highlights directly from transcript chunks without calling `inspect_moment_context`.
- **Why**: Streamers frequently scream or react **5 to 15 seconds AFTER** the in-game action happened on screen. Setting clip bounds based solely on audio screams cuts off the actual trigger (the vehicle flip, sniper shot, or glitch).
- **Tool**: `inspect_moment_context(url="...", timestamp=1420.5, window_before=120.0, window_after=60.0)`
- **What it returns**:
  - Two high-resolution 3x3 contact sheets (`Sheet 1` and `Sheet 2`) as MCP `Image` content.
  - Interleaved chronological timeline of speech and live chat velocity.
  - `[BURST]` tags indicating where viewers first saw the event.
- **Refinement**: Call `refine_highlight_bounds(url="...", start=..., end=..., hint="...")` to automatically snap start/end times to natural speech and action boundaries.

### Step 6: Submit Formatted Highlights (`submit_highlights`)
- **Action**: Submits the final 3 to 5 curated highlights.
- **Required fields per highlight**:
  - `start`, `end`: Boundary seconds verified via contact sheets.
  - `hook_title`: High-impact on-screen title (3-5 words).
  - `moment_type`: `funny` | `fail` | `graphics_detail` | `hype` | `clutch` | `chat_chaos` | `lore_secret` | `analysis`.
  - `narration_intro`: 1-2 punchy sentences opening the clip (grounded in the stream).
  - `narration_outro`: 1-2 punchy sentences closing the clip (call-to-action or payoff reflection).
  - `context_badge`: Short 2-4 word pill badge (e.g. `Physics Fail`, `World Record Pace`).
  - `mode`: `auto` (recommended), `narrative`, `story`, `breakdown`, or `raw`.
  - `sub_clips`: JSON array of beats for multi-beat jump-cut story mode (e.g. `Setup`, `Payoff`).
- **Breakdown Mode Parameters (when `mode="breakdown"` or `moment_type` is fail/clutch/analysis/graphics_detail)**:
  - `breakdown_commentary`: Full transformative analysis explaining the key moment, plot hook, or hidden clue.
  - `breakdown_callout`: Short on-screen callout card (e.g. `SECRET HARD DRIVE TEASER`).
  - `breakdown_kind`: `auto` (default) | `slow_mo` | `zoom_spotlight`.
    - Choose `zoom_spotlight` for computer/phone screens, text clues, classified files, secret documents, or easter eggs.
    - Choose `slow_mo` for physical combat, vehicle crashes, stunts, or headshots.
  - `breakdown_target_time`: (Optional) Exact stream second where the visual detail is visible facing the camera. If omitted, the pipeline automatically scans candidate frames or setup phase to avoid reverse-angle blind spots.
  - `breakdown_roi`: (Optional) Normalized bounding box `[ymin, xmin, ymax, xmax]` (0.0 to 1.0) targeting the detail (e.g. `[0.45, 0.25, 0.78, 0.72]` for a laptop display).
  - `breakdown_focus`: (Optional) Description of the target visual element (e.g. `"laptop screen showing encrypted Coast Guard files"`).
  - `breakdown_zoom`: (Optional) Magnification factor (default `2.2`).
- **Tool**: `submit_highlights(url="...", highlights="[...]")`

### Step 7: Vertical Video Assembly (Asynchronous) (`render_all_selected` / `render_clip`)
- **Action**: Downloads ±3s padded 1080p video cuts only for the selected moments, applies aspect-correct 9:16 layout (Layout A with compact facecam & centered hero gameplay or Layout B full gameplay), mixes Kokoro/edge-tts/ElevenLabs voiceover over 18% ducked stream audio, burns ASS styled subtitles and chat ticker, and produces final MP4s up to 180 seconds.
- **Execution**: **ASYNCHRONOUS**. Returns `job_id`. Poll `get_render_status(job_id=...)` every ~20 seconds.
- **Tool**: `render_all_selected(url="...")` or `render_clip(url="...", highlight_index=0)`

### Step 8: Generate Post-Production Metadata (`generate_post_metadata`)
- **Action**: Generates SEO-optimized YouTube Shorts and TikTok publishing copy:
  - High-CTR Title options with emoji.
  - Clean video description.
  - Viral game and streamer hashtags.
  - Pinned engagement comment asking an open-ended viewer question.
- **Tool**: `generate_post_metadata(url="...")`

### Step 9: Workspace Hygiene & Temp Cleanup (`cleanup_temp`)
- **Action**: Deletes intermediate video slices, downloaded proxy frames, and chat JSON caches for the stream while strictly preserving final rendered MP4s and sidecar metadata.
- **Tool**: `cleanup_temp(url="...")`

---

## 4. Tool JSON Specifications & Payloads

### 1. `download_stream_audio`
**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```
**Response:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "title": "GTA 6 Next-Gen Gameplay Stream",
  "streamer": "TimTheTatman",
  "game": "Grand Theft Auto VI",
  "duration_seconds": 14400.0,
  "audio_path": "temp/audio/dQw4w9WgXcQ.wav",
  "status": "downloaded"
}
```

### 2. `transcribe_audio`
**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```
**Response (Immediate Job Dispatch):**
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "tool": "transcribe_audio",
  "status": "started",
  "message": "Transcription started in background. Call get_render_status('job_a1b2c3d4e5f6') to monitor."
}
```

### 3. `get_render_status`
**Request:**
```json
{
  "job_id": "job_a1b2c3d4e5f6"
}
```
**Response (In Progress):**
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "tool": "transcribe_audio",
  "status": "running",
  "progress": 0.45,
  "elapsed_seconds": 42.5,
  "eta_seconds": 51.0,
  "current_step": "WhisperX transcription passing segment 12/28"
}
```
**Response (Done):**
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "tool": "transcribe_audio",
  "status": "done",
  "progress": 1.0,
  "elapsed_seconds": 88.2,
  "result": {
    "segments_count": 842,
    "language": "en",
    "status": "scoring"
  }
}
```

### 4. `analyze_signals`
**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```
**Response:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "loudness_spikes_count": 8,
  "chat_spikes_count": 5,
  "total_signals": 13,
  "status": "scoring"
}
```

### 5. `get_transcript_chunks`
**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "only_with_signals": true
}
```
**Response:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "chunks_count": 4,
  "total_chunks_in_stream": 45,
  "narration_language": "english",
  "narration_guide": "Write energetic, interactive gamer commentary grounded in the transcript...",
  "chunks": [
    {
      "index": 0,
      "start": 1230.0,
      "end": 1290.0,
      "text": "Tim: Watch this bridge jump right now. If I hit the ramp at top speed... OH NO!",
      "loudness_spikes": [{"start": 1245.0, "end": 1255.0, "peak_time": 1250.0, "score": 9.2}],
      "chat_spikes": [{"start": 1244.0, "end": 1254.0, "score": 8.8, "sample_messages": ["OMG", "LMAO", "RIP"]}]
    }
  ]
}
```

### 6. `inspect_moment_context`
**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "timestamp": 1245.0,
  "window_before": 120.0,
  "window_after": 60.0
}
```
**Response:**
Returns dual MCP `Image` objects (Sheet 1 & Sheet 2) followed by JSON:
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "timestamp": 1245.0,
  "window_start": 1125.0,
  "window_end": 1305.0,
  "total_chat_messages": 384,
  "avg_chat_rate": 2.13,
  "burst_timestamps": [1238.0, 1244.0],
  "contact_sheets": [
    "temp/moment_frames/.../sheet1_buildup.jpg",
    "temp/moment_frames/.../sheet2_climax.jpg"
  ],
  "sheet_timestamps": [
    [1125.0, 1135.0, 1145.0, 1155.0, 1165.0, 1175.0, 1185.0, 1195.0, 1205.0],
    [1215.0, 1225.0, 1235.0, 1245.0, 1255.0, 1265.0, 1275.0, 1285.0, 1295.0]
  ],
  "timeline": "=== STREAM CONTEXT WINDOW [1125.0s - 1305.0s] ===\n...",
  "hint": "Chat bursts marked [BURST] indicate where viewers saw the on-screen event..."
}
```

### 7. `refine_highlight_bounds`
**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "start": 1245.0,
  "end": 1280.0,
  "hint": "Streamer screams at 1248s but vehicle takes off at 1238s in Sheet 1 tile 3"
}
```
**Response:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "original_start": 1245.0,
  "original_end": 1280.0,
  "refined_start": 1238.5,
  "refined_end": 1276.2,
  "hook_title": "BRIDGE JUMP DISASTER",
  "reasoning": "Extended start back 6.5s to capture vehicle acceleration trigger seen before streamer reaction."
}
```

### 8. `submit_highlights`
**Request (Narrative / Story Mode):**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "highlights": "[{\"start\": 1238.5, \"end\": 1276.2, \"hook_title\": \"BRIDGE JUMP DISASTER\", \"moment_type\": \"fail\", \"narration_intro\": \"Tim thought his muscle car could clear the Vice City drawbridge.\", \"narration_outro\": \"Spoiler alert: gravity always wins. Drop a follow if you saw that landing!\", \"context_badge\": \"Physics Fail\", \"mode\": \"auto\", \"score\": 9.5}]"
}
```

**Request (Evidence Breakdown / Zoom Spotlight Mode):**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "highlights": "[{\"start\": 19169.0, \"end\": 19204.0, \"hook_title\": \"SECRET CUTSCENE TEASER\", \"moment_type\": \"analysis\", \"narration_intro\": \"Did you catch this secret Coast Guard cutscene after the credits?\", \"narration_outro\": \"Drop your theories on what this file means in the comments below!\", \"context_badge\": \"Secret File\", \"mode\": \"breakdown\", \"breakdown_commentary\": \"Look closely at the laptop screen! As the hacker opens the classified data, the decryption window reveals project names tied directly to the story DLC.\", \"breakdown_callout\": \"SECRET HARD DRIVE TEASER\", \"breakdown_kind\": \"zoom_spotlight\", \"breakdown_target_time\": 19171.0, \"breakdown_roi\": [0.45, 0.25, 0.78, 0.72], \"breakdown_focus\": \"laptop screen showing encrypted files\", \"breakdown_zoom\": 2.2, \"score\": 9.8}]"
}
```

**Response:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "accepted_count": 1,
  "rejected_count": 0,
  "status": "rendering",
  "highlights": [
    {
      "index": 0,
      "start": 19169.0,
      "end": 19204.0,
      "hook_title": "SECRET CUTSCENE TEASER",
      "moment_type": "analysis",
      "mode": "breakdown",
      "breakdown_kind": "zoom_spotlight",
      "score": 9.8
    }
  ]
}
```

### 9. `render_all_selected` & `render_clip`
**Request (`render_all_selected`):**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```
**Response (Immediate Job Dispatch):**
```json
{
  "job_id": "render_b2c3d4e5f6a1",
  "tool": "render_all_selected",
  "status": "started",
  "total_highlights": 1,
  "message": "Rendering 1 highlights started in background. Call get_render_status('render_b2c3d4e5f6a1') to monitor."
}
```

**Request (`render_clip` - Single Highlight with Optional Override):**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "highlight_index": 0,
  "mode": "breakdown",
  "breakdown_kind": "zoom_spotlight",
  "breakdown_target_time": 19171.0,
  "breakdown_roi": [0.45, 0.25, 0.78, 0.72],
  "breakdown_focus": "laptop screen",
  "breakdown_zoom": 2.2
}
```

### 10. `generate_post_metadata`
**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```
**Response:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "clips_count": 1,
  "metadata_path": "output/Grand Theft Auto VI/TimTheTatman/post_metadata.json",
  "posts": [
    {
      "index": 0,
      "title_options": [
        "Bro Really Thought He Could Make This Jump?! 💀 #shorts",
        "When Physics Says NO in GTA 6 🚗💨"
      ],
      "description": "TimTheTatman tries the impossible Vice City drawbridge jump...\n\n#GTA6 #GrandTheftAuto #Gaming #StreamerHighlights",
      "hashtags": ["#GTA6", "#Gaming", "#Shorts", "#TimTheTatman"],
      "pinned_comment": "Could you have landed this jump? Rate Tim's driving 1-10 👇"
    }
  ]
}
```

### 11. `cleanup_temp`
**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```
**Response:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "scope": "stream",
  "deleted_count": 14,
  "errors": null
}
```

---

## 5. Async Polling Rules & Background Jobs

1. **Which tools are asynchronous?**
   - `transcribe_audio`: Speech recognition and phoneme alignment.
   - `render_clip`: Video slicing, vertical cropping, TTS synthesis, audio ducking, ASS burning.
   - `render_all_selected`: Batch rendering of all queued highlights.
   - `render_topic_video` & `render_topic_shorts`: Long-form topic assembly and short teaser generation.
2. **Polling Frequency**:
   - Always call `get_render_status(job_id)` every **15 to 25 seconds** (e.g. ~20 seconds).
   - Do NOT poll in a rapid spin loop (e.g. every 1 second); this wastes CPU cycles and floods server logs.
3. **Patience is Normal**:
   - Background rendering taking **2 to 5 minutes** per clip is completely normal (video decoding, 1080x1920 scaling, ASS filter burn-in, NVENC/libx264 encoding).
4. **Never Fall Back to Manual Workarounds**:
   - **STRICT PROHIBITION**: NEVER attempt to assemble clips yourself using raw `ffmpeg` commands, local Python scripts, or system TTS voices. Manual workarounds bypass audio ducking, ASS formatting, split-line zoom, loudness checks, and copyright metadata. Keep polling until `get_render_status` reports `done` or `failed`.

---

## 6. Visual Inspection Masterclass: Sheet 1 & Sheet 2 Analysis

### The "Delayed Reaction Trap"
In competitive gaming and livestreaming, human reaction latency introduces a significant offset:
1. **T = 0s**: In-game event occurs (player drives off a cliff or enemy snipes from a crane).
2. **T + 2s**: Live chat viewer burst triggers (`[BURST]`, "OMEGALUL", "NOOOO", "RIP").
3. **T + 6s**: Streamer finishes processing and begins screaming ("WHAT JUST HAPPENED?!").
4. **T + 8s**: Audio RMS loudness spike registers.

If an AI clipper relies only on the audio loudness spike, it will place the highlight start at `T + 7s`, completely missing the stunt, cliff jump, or sniper shot!

### Anatomy of Sheet 1: Buildup & Trigger (T - 120s to T - 30s)
- **Grid**: 3x3 layout containing 9 chronological tiles spaced evenly across the 90-second buildup.
- **Labels**: High-contrast yellow timestamp in `MM:SS` format on each tile.
- **Key Inspection Points**:
  - **Objective / Mission Markers**: Look for mission titles appearing on the top HUD.
  - **Dialogue Subtitles**: Check if NPCs or teammates are planning a heist or strategy.
  - **Vehicle / Equipment Setup**: Check if the streamer is gearing up, tuning a car, or flying an aircraft.
  - **Stealth / Silent Tensions**: Identify moments where streamer dialogue drops silent right before an ambush.

### Anatomy of Sheet 2: Climax & Reaction (T - 30s to T + 60s)
- **Grid**: 3x3 layout containing 9 chronological tiles covering the climax and immediate aftermath.
- **Labels**: High-contrast yellow timestamp in `MM:SS` format on each tile.
- **Key Inspection Points**:
  - **Exact Event Point**: Identify the exact tile showing the explosion, death screen ("WASTED"), kill banner, or crash.
  - **Peak Impact Offset**: Note the tile timestamp where the visual climax peaks (`peak_time`).
  - **Streamer Facecam vs Gameplay**: Verify the facecam reaction correlates with the action on screen.
  - **Clean Outro Boundary**: Ensure the clip extends 5-8 seconds past the climax so the streamer's reaction and your `narration_outro` conclude naturally without an abrupt cutoff.

### Multi-Beat Jump-Cut Story Detection
When reviewing candidate chunks, look for dead air (≥ 5 seconds of uneventful driving or silence between action moments):
- If the moment has dead air, define 2 or 3 distinct beats in `sub_clips`:
  - **Beat 1 (Setup)**: E.g., `100.0s - 105.0s` ("Streamer spots helicopter").
  - **Beat 2 (Escalation)**: E.g., `112.0s - 118.0s` ("Aiming rocket launcher").
  - **Beat 3 (Payoff)**: E.g., `124.0s - 130.0s` ("Mid-air explosion and streamer reaction").
- Set `mode: "story"` or `mode: "auto"`. StreamClipper will stitch the beats seamlessly with jump-cuts and remap all captions and zooms onto the stitched timeline.

---

## 7. Transcript Grounding & Commentary Guardrails

### Moment Types & Adaptive Tone
Every highlight must specify an accurate `moment_type` so voiceover and caption animations match the content:

| `moment_type` | Suitable Content | Commentary Tone | Recommended Caption Style |
| :--- | :--- | :--- | :--- |
| **`funny`** | Jokes, physics glitches, absurd NPC encounters | Witty, sarcastic, dry humor | `bounce` |
| **`fail`** | Wasted screens, botched stunts, accidental deaths | Mocking, comedic disbelief | `bounce` |
| **`hype`** | High-speed chases, epic escapes, clutch wins | High energy, rapid-fire, enthusiastic | `pop` or `hormozi` |
| **`clutch`** | 1v3 outplays, low HP survival, buzzer beaters | Tense, dramatic, stakes-building | `pop` |
| **`graphics_detail`**| Water physics, lighting, vehicle deformation, easter eggs | Spotlighting, informative, observant | `box` |
| **`chat_chaos`** | Chat trolling streamer, text-to-speech donation chaos | Interactive, amused, community-centric | `bounce` |
| **`lore_secret`** | Hidden map secrets, lore theories, backstory | Mysterious, intriguing, conspiratorial | `sentence` |
| **`analysis`** | Weapon breakdown, mission strategy, map analysis | Expert breakdown, tactical, direct | `box` |

### Transformation Modes
- **`auto` (Default)**: Automatically resolves to `story` if validated multi-beat `sub_clips` exist; otherwise resolves to `narrative`.
- **`narrative`**: Single continuous cut with commentary intro/outro, context badge, spike zoom, and chat ticker.
- **`story`**: Multi-beat jump-cut stitching (2-3 beats) with commentary and overlays.
- **`breakdown`**: Transformative replay mode for clutches, fails, easter eggs, and analysis. Operates in two distinct archetypes:
  1. **Action Replay (`slow_mo`)**:
     - *When to use*: Combat headshots, vehicle stunts, explosions, physics glitches.
     - *Behavior*: Takes a 5.0s window around the climax peak and slows it down to 0.5x speed with an `🔴 INSTANT REPLAY` badge and breakdown commentary.
  2. **Visual Evidence Spotlight (`zoom_spotlight`)**:
     - *When to use*: In-game computer/phone screens, text clues, classified files, secret documents, easter eggs, or UI modals.
     - *Behavior*: Takes over the full 9:16 vertical canvas (1080×1920) for mobile legibility. Performs a smooth Ken Burns pan & zoom (1.2s ease-in to 2.2x zoom $\rightarrow$ 3.5s hold $\rightarrow$ 1.0s ease-out) with sub-pixel Lanczos interpolation and a `🔍 EVIDENCE BREAKDOWN` badge.
     - *Reverse-Angle Detection & Setup Fallback*: In cutscenes where the camera cuts $180^\circ$ behind the subject (rendering screens invisible), the pipeline automatically falls back to setup-phase frames ($t = \text{clip\_start} + \min(2.0, 0.20 \times \text{clip\_dur})$) where screens face the camera.
     - *Direct-Source Cropping*: Crops directly from raw 16:9 frames without canvas padding bars.
  3. **Dynamic Spotlight Duration (Zero Speech Cutoff)**:
     - The spotlight duration automatically scales to fit the exact TTS speech length (`words / 2.5 + buffer`), ensuring voiceover commentary and outro are **never truncated mid-sentence**.
- **`raw`**: Uncut video slice with subtitles only (no voiceover, no badges, no zoom). *Use sparingly.*

### Video Duration Limits & Modern Shorts Specs
- **180-Second Video Cap**: StreamClipper fully supports modern 3-minute YouTube Shorts (`output.max_final_seconds: 180`). Highlights up to 90 seconds (`analysis.max_clip_seconds: 90`) can be curated with full intros, outros, and breakdowns without being cut in half.
- **Centered Hero Gameplay**: In stacked layout, facecam is compact (480px, 25% height) while 16:9 gameplay sits dead-center ($y = 964 \approx 960$) with captions in the lower third.
- **Broadcast-Grade Video Quality**: High-motion gameplay is encoded at 12–16 Mbps VBR (`-b:v 12M -maxrate 16M -bufsize 24M`) with CRF/CQ 19 and NVENC spatial/temporal adaptive quantization (`-spatial-aq 1 -temporal-aq 1`), eliminating macroblocking and pixelation.
- **Natural Studio Voiceover**: ElevenLabs narration runs on `eleven_v4` with calibrated settings (`stability: 0.40`, `similarity_boost: 0.85`, native 1.0x speed) and a clean bypass into the ducking mix, eliminating AGC volume pumping and digital clipping.

### The Narration Quality Checker
StreamClipper enforces strict validation on `narration_intro` and `narration_outro`:
- **Must be Grounded**: Must refer to specific game events, character names, or streamer statements found in the transcript or visual sheets.
- **Prohibited Clichés**: Lines like *"Watch till the end!"*, *"You won't believe what happens next!"*, *"Look at this crazy play!"*, or *"OMG this is insane!"* will be **REJECTED** by the server with a list of violations.
- **Language Matching**: Must match the stream's detected language (`english` or `hinglish`).

---

## 8. Ready-to-Copy Starter Prompt for New Chats

Copy and paste this prompt when starting a new session with an MCP-enabled agent to ensure flawless execution:

```markdown
You are an expert gaming video editor and viral Short curator operating StreamClipper MCP.
Your mission is to find, verify, and render the top viral highlights from a gaming live stream.

Follow the mandatory StreamClipper workflow in exact sequence:
1. Call download_stream_audio(url="<STREAM_URL>") to extract audio and live chat.
2. Call transcribe_audio(url="<STREAM_URL>") to begin speech transcription.
   - ASYNCHRONOUS: Poll get_render_status(job_id) every 20 seconds until done. Do NOT proceed until done.
3. Call analyze_signals(url="<STREAM_URL>") to compute audio loudness and chat velocity spikes.
4. Call get_transcript_chunks(url="<STREAM_URL>", only_with_signals=True) to scout candidate moments.
5. COMPULSORY VISUAL VERIFICATION:
   - For every candidate timestamp, call inspect_moment_context(url="<STREAM_URL>", timestamp=...).
   - Carefully inspect visual Sheet 1 (buildup & trigger [-120s to -30s]) and Sheet 2 (climax & reaction [-30s to +60s]).
   - Check the [BURST] chat tags and yellow MM:SS tile timestamps to identify the TRUE on-screen trigger.
   - Never start the clip at the streamer's delayed audio scream; start BEFORE the trigger event.
   - Use refine_highlight_bounds if needed to snap boundaries to natural speech pauses.
6. Call submit_highlights(url="<STREAM_URL>", highlights="[...]") with 3-5 curated highlights:
   - Provide punchy narration_intro and narration_outro strictly grounded in the transcript/visuals.
   - Specify accurate moment_type, hook_title, and context_badge.
   - For moments with computer/phone screens, text clues, secret files, or easter eggs:
     - Set mode="breakdown", breakdown_kind="zoom_spotlight", and breakdown_focus="...".
     - Set breakdown_target_time to the exact second where the screen/file faces the camera in the visual sheets.
     - Specify breakdown_roi=[ymin, xmin, ymax, xmax] around the target text/screen.
   - For physical action/stunts/crashes, use breakdown_kind="slow_mo" or mode="auto".
7. Call render_all_selected(url="<STREAM_URL>").
   - ASYNCHRONOUS: Poll get_render_status(job_id) every 20 seconds until done.
   - NEVER fall back to manual ffmpeg, local scripts, or manual assembly outside MCP.
8. Call generate_post_metadata(url="<STREAM_URL>") to produce viral titles, tags, and pinned comments.
9. Call cleanup_temp(url="<STREAM_URL>") to clean intermediate slices.

Begin by confirming the stream URL and downloading the audio.
```

---
*StreamClipper v2.2 Documentation & Operator Runbook — Maintained under `docs/`.*
