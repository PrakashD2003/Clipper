# Topic Videos — Fixed Contracts (T2–T6)

Every handoff brief for topic videos codes against these names and shapes. Do not rename or
reshape them; add fields only with defaults. Design: `docs/specs/topic-videos.md`.

## Existing pieces to reuse (do not reimplement)

| Need | Use |
| :--- | :--- |
| Library index/search | `streamclipper.library.index.LibraryIndex` (`search`, `list_sources`, `get_source`, `index_state`), `SearchHit`, `LibrarySource` |
| Ingest a source | `streamclipper.library.ingest.add_source`, `index_cached_states`; `Pipeline.prepare_stream` |
| yt-dlp auth/runtime opts | `streamclipper.download._ytdlp_opts.apply_ytdlp_auth_opts(ydl_opts)` on every new `YoutubeDL(...)` |
| 1080p segment download | `streamclipper.download.video.download_segment(url, start, end, output_dir, buffer_seconds=0.0, video_id=...)` |
| Narration style + checks | `streamclipper.narration_style.narration_guide(language, hook_style, avoid_phrases)`, `check_narration(fields, transcript_excerpt=..., chat_samples=..., avoid_phrases=...)`, `BANNED_PHRASES` |
| Language | `streamclipper.language.resolve_narration_language`, `language_instruction` |
| TTS | `TransformativeNarrator(config).synthesize_speech(text, out_wav, language=...)` (ElevenLabs first when configured), `normalize_audio(in, out)` |
| 9:16 layouts | `assembly.layout.compute_layout`, `VideoComposer.compose(..., layout_kind=...)`, `Exporter.export(...)` |
| Music/copyright check | `analysis.copyright_risk.CopyrightRiskDetector(config).detect(audio_path, start, end)` → `(risk, confidence)` (`HIGH_MUSIC_DETECTED` / `UNKNOWN` / `CLEAN`) |
| Background jobs (MCP) | `mcp_server/jobs.py` (`create_job`, `set_stage`, `clip_done`, `finish_job`, `JobProgress`) and `server._start_render_job(kind, config, url, clips_planned, factory)`; clients poll `get_render_status(job_id)` |
| Hardware/codec | `config.get("video.codec")` (auto-resolved), `video.crf`; never hardcode a GPU |

## New package layout

```
src/streamclipper/library/frames.py        (T2)
src/streamclipper/library/vision.py        (T2)
src/streamclipper/topics/__init__.py       (T3)
src/streamclipper/topics/models.py         (T3, extended in T4)
src/streamclipper/topics/ledger.py         (T3)
src/streamclipper/topics/discover.py       (T3)
src/streamclipper/topics/planner.py        (T4)
src/streamclipper/topics/validate.py       (T4)
src/streamclipper/prompts/topic_planner.txt (T4)
src/streamclipper/assembly/longform.py     (T5)
src/streamclipper/output/topic_post.py     (T5)
src/streamclipper/assembly/topic_shorts.py (T6)
```

## Models (`topics/models.py`, all `SerializableModel`, defaults + `.get` in `from_dict`)

```python
@dataclass
class EvidenceRef(SerializableModel):          # T3
    source_id: str; url: str; channel: str = ""; kind: str = "stream"
    start: float = 0.0; end: float = 0.0; snippet: str = ""; heat: float = 0.0; score: float = 0.0

@dataclass
class TopicCandidate(SerializableModel):       # T3
    topic_id: str                               # slug, e.g. "hidden-casino-details"
    title: str; game: str; description: str = ""
    keywords: list[str] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    source_count: int = 0; channel_count: int = 0
    already_made: bool = False; similar_to: str = ""   # ledger topic_id when already_made

@dataclass
class EvidenceClip(SerializableModel):         # T4
    source_id: str; url: str; channel: str = ""; kind: str = "stream"
    start: float = 0.0; end: float = 0.0       # stream seconds; end-start <= 20 (official <= 10)
    role: str = "evidence"                      # evidence | reaction | broll
    reason: str = ""

@dataclass
class TopicPoint(SerializableModel):           # T4
    index: int; heading: str                    # "#3 The hidden casino" style for numbered shapes
    narration: str                              # spoken before/over B-roll for this point
    evidence: list[EvidenceClip] = field(default_factory=list)   # 1-2 played at full volume
    alternates: list[EvidenceClip] = field(default_factory=list) # swap-ins if a clip fails checks
    use_in: list[str] = field(default_factory=lambda: ["long"])  # long | teaser | short_a | short_b

@dataclass
class ShortPlan(SerializableModel):            # T4
    kind: str                                   # teaser | standalone
    slot: str                                   # teaser | short_a | short_b
    point_indexes: list[int] = field(default_factory=list)
    hook_title: str = ""; narration_intro: str = ""; callout: str = ""; narration_outro: str = ""

@dataclass
class TopicPlan(SerializableModel):            # T4
    plan_id: str; topic_id: str; topic_title: str; game: str
    language: str = "english"; shape: str = "numbered"   # numbered | countdown | story
    status: str = "draft"                                 # draft | approved | rendering | rendered | failed
    cold_open: str = ""; outro: str = ""
    points: list[TopicPoint] = field(default_factory=list)
    shorts: list[ShortPlan] = field(default_factory=list)
    video_title: str = ""; thumbnail_text: str = ""; description: str = ""
    estimated_seconds: float = 0.0; narration_seconds: float = 0.0
    created_at: str = ""; approved_at: str = ""
    outputs: dict[str, str] = field(default_factory=dict) # "long", "teaser", "short_a", "short_b", "thumbnail", "chapters", post files
```

## Paths

- Topic folder: `<general.output_dir>/<game>/topics/<topic_id>/` (sanitized segments).
- Plan file: `<topic folder>/plan.json` (the single source of truth; written atomically).
- Long video: `long.mp4`, `chapters.txt`, `thumbnail.jpg`, `long_post.txt`.
- Shorts: `short_teaser.mp4`, `short_a.mp4`, `short_b.mp4` + `<name>_post.txt`.
- Ledger: `<general.cache_dir>/topics_ledger.json`.
- Frames: `<general.temp_dir>/frames/<source_id>/sheet_<start>.jpg`.

## Plan rules (enforced by `topics/validate.py`, T4)

- Long: 5–7 points; estimated length 8–10 min (`topics.long_min_seconds` 480, `topics.long_max_seconds` 600).
  Estimate = narration words / 2.7 + chapter cards (2.5 s each, numbered/countdown only) + evidence clip lengths.
- Narration share ≥ `topics.min_narration_ratio` (0.40) of the estimate.
- Every evidence clip exists in the library (source_id known, inside source duration), ≤ 20 s; `kind == "official"` ≤ 10 s.
- `short_a` and `short_b` use no overlapping footage (same source_id with overlapping ranges).
- All narration fields pass `check_narration` (grounded against the library snippets of the clips used).
- Shape rotation: avoid the shape used by the previous ledger entry for the same game.

## MCP tool names (all return JSON strings; long work returns `{"status":"started","job_id":...}`)

| Tool | Brief |
| :--- | :--- |
| `library_add_source(url, game, kind="stream", channel="")` (job) | T3 |
| `search_library(query, game="", kinds="", channels="", limit=20)` | T3 |
| `list_library(game="")` | T3 |
| `get_frame_sheets(source_id, start=0.0, count=6)` (returns images) / `submit_frame_notes(source_id, notes)` | T2 |
| `index_frames(source_id)` (job: low-res download + sheets) | T2 |
| `get_topic_material(game, max_terms=40)` / `suggest_topics(game, limit=10)` / `record_topic_candidates(game, candidates)` | T3 |
| `start_topic_plan(game, topic)` / `submit_topic_plan(plan_json)` / `get_topic_plan(plan_id)` / `update_topic_plan(plan_id, patch_json)` / `approve_topic_plan(plan_id)` | T4 |
| `render_topic_video(plan_id)` (job) | T5 |
| `render_topic_shorts(plan_id)` (job) | T6 |
