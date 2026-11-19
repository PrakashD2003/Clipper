# Brief 02 (T1): source library + searchable index for topic videos

**Base commit:** `2f206ca` (`main`)
**Your branch:** `handoff/02-T1-source-library`
**Worktree:** `D:\GTA6-Clipper-wt\handoff-02`

Read and follow `docs/handoff/README.md` first (own worktree, only listed files, no secrets,
one pytest at a time, never push/merge). Design context: `docs/specs/topic-videos.md`
(this brief is build step **T1**; decisions D3, D4, D13).

```powershell
git -C D:\GTA6-Clipper worktree add D:\GTA6-Clipper-wt\handoff-02 -b handoff/02-T1-source-library 2f206ca
```

## Goal

A local library of analysed source videos (streams, no-commentary gameplay, short official
clips) for one or more games, plus a fast full-text index over ~30 s windows of speech, chat
and interest signals, so later steps can ask "find every moment about the casino" across all
videos at once. **No rendering in this brief.**

## Files you may change

- `src/streamclipper/library/__init__.py` (new)
- `src/streamclipper/library/models.py` (new)
- `src/streamclipper/library/index.py` (new)
- `src/streamclipper/library/ingest.py` (new)
- `src/streamclipper/pipeline.py` — only to add the public `prepare_stream` method below
- `src/streamclipper/cli.py` — only to add the `library` command group
- `config.default.yaml` and `src/streamclipper/config.default.yaml` — only a new `library:` section (byte-identical copies)
- `tests/test_library.py` (new)

## Existing code you build on (read these first)

- `Pipeline.process_stream(...)` in `pipeline.py` runs `_step_download_audio` →
  `_step_transcribe` → `_step_analyze` → `_step_score` → `_step_assemble`, saving
  `PipelineState` through `StepCache` (`cache.py`) after each step; `_recover_failed_state`
  must run before status comparisons (AGENTS.md §4 rule 3).
- `PipelineState` (`models.py`): `stream_info: StreamInfo` (url, title, streamer, game,
  duration_seconds, video_id), `transcript: TranscriptionResult | None` (segments with
  `text/start/end/words`, plus `language`), `loudness_spikes`, `chat_spikes`,
  `heatmap_spikes` (`TimeRange` with `score` 0–10 and `metadata`; heatmap metadata has `heat` 0–1).
- Raw live chat is cached by `_step_analyze` as `<general.temp_dir>/<video_id>_chat.csv`, columns
  `offset_seconds, author, message` (`download/chat.py` `ChatScraper.load_cache`).
- `StepCache(config).load_state(url)`; cache files are `<general.cache_dir>/<key>.json`.

## Do

### 1. `Pipeline.prepare_stream`
Add `async def prepare_stream(self, url: str, *, game: str | None = None, streamer: str | None = None, force: bool = False) -> PipelineState`
that runs exactly the download-audio, transcribe and analyze steps (resuming from cache like
`process_stream`, including `_recover_failed_state`), saves state, and stops before scoring.
It must not change `process_stream` behaviour. Reuse the existing step methods; no copy-paste
of their bodies.

### 2. Data model (`library/models.py`)
`SerializableModel` dataclasses with defaults and `.get` in `from_dict`:
- `LibrarySource`: `source_id` (StepCache key of the URL), `url`, `title`, `channel` (uploader /
  streamer), `game`, `kind` (`stream` | `gameplay` | `official`), `language`,
  `duration_seconds`, `video_id`, `added_at`, `indexed_at` (ISO strings), `segment_count`.
- `SearchHit`: `source_id`, `url`, `channel`, `kind`, `start`, `end`, `snippet`, `score`
  (final rank), `bm25`, `heat`, `loudness`, `chat_rate`.

### 3. Index (`library/index.py`)
`LibraryIndex(config)` backed by SQLite at `library.db_path` (default
`<general.cache_dir>/library.sqlite`), WAL mode, created on first use, schema version table.
- Tables: `sources` (columns of `LibrarySource`), `segments(id INTEGER PK, source_id, start,
  end, text, chat_text, heat, loudness, chat_rate)`, FTS5 table `segments_fts(text, chat_text)`
  with `content='segments'`, `content_rowid='id'`, tokenizer `unicode61 remove_diacritics 2`
  (works for English, Roman Hinglish and Devanagari), kept in sync on insert/delete. Also
  create (empty for now) `frame_notes(id, source_id, t, note)` + `frame_notes_fts(note)` for T2.
- `index_state(state: PipelineState, *, kind="stream", chat_df=None) -> LibrarySource`:
  builds windows of `library.window_seconds` (default 30) stepping by
  `library.window_step_seconds` (default 20) across the stream; each window gets the
  transcript text whose segment midpoint falls inside it, the chat messages inside it
  (joined, capped at `library.max_chat_chars_per_window`, default 600), `heat` = max heatmap
  heat overlapping (0–1), `loudness` = max loudness spike score overlapping (0–10),
  `chat_rate` = chat messages per minute in the window. Windows with no text, no chat and no
  signal are skipped. Re-indexing a source replaces its rows (idempotent). Language from the
  transcript.
- `search(query, *, game=None, kinds=None, channels=None, limit=20) -> list[SearchHit]`:
  FTS5 `MATCH` (escape user input safely: quote each term; never build SQL by string
  concatenation of user text), rank = `-bm25` normalised + `library.heat_boost` (default 0.3)
  × heat + small chat_rate boost; snippet via FTS5 `snippet()`. Merge hits that overlap within
  the same source into one span.
- `list_sources(game=None)`, `get_source(source_id)`, `remove_source(source_id)`.

### 4. Ingest (`library/ingest.py`)
- `async add_source(config, url, *, game, kind="stream", channel=None, force=False) -> LibrarySource`:
  `Pipeline.prepare_stream(...)` then load the cached chat CSV (if any) and `index_state`.
- `index_cached_states(config, *, game=None) -> list[LibrarySource]`: index every analysed
  `PipelineState` already in the cache dir (transcript present), so existing streams join the
  library without re-downloading.
- `kind="official"` sources are indexed normally but flagged; later steps cap their clips at 10 s.

### 5. CLI (`cli.py`): new group `streamclipper library`
- `library add URL --game TEXT [--kind stream|gameplay|official] [--channel TEXT] [--force]`
- `library import-cache [--game TEXT]` (runs `index_cached_states`)
- `library list [--game TEXT]` (rich table: channel, kind, title, duration, segments, indexed)
- `library search QUERY [--game] [--kind ...] [--channel ...] [--limit 20]` (rich table:
  score, channel, kind, mm:ss–mm:ss, snippet, heat)
- `library remove SOURCE_ID`

### 6. Config (both YAML copies, byte-identical, with comments)
```yaml
library:
  db_path: ""                      # "" = <general.cache_dir>/library.sqlite
  window_seconds: 30
  window_step_seconds: 20
  max_chat_chars_per_window: 600
  heat_boost: 0.3
```

### 7. Tests (`tests/test_library.py`, names `test_T1_*`, no network, no GPU)
Build `PipelineState` objects in code (transcript segments, chat DataFrame, heatmap and
loudness spikes). Cover: schema creation and version; windowing (text assignment by
midpoint, chat capping, heat/loudness/chat_rate values, empty windows skipped); idempotent
re-index; search ranking (term match beats non-match; heat boost reorders ties); Hinglish and
Devanagari terms are searchable; filters by game/kind/channel; overlapping hits merged; SQL
injection-ish input (`"casino' OR 1=1 --"`) is safe and returns normally;
`index_cached_states` picks up states saved with `StepCache`; `prepare_stream` stops before
scoring (patch the step methods and assert call order, scoring/assembly not called);
CLI `library search` and `library list` via `click.testing.CliRunner`.

## Done when

- `pytest -q -p no:cacheprovider tests/test_library.py tests/test_pipeline.py tests/test_cache_cli.py`
  is 0 failed (use the closest existing CLI test file if the name differs).
- `cmp` shows both config.default.yaml copies identical.
- Commit message: `feat(library): source library and FTS5 search index for topic videos (T1)`
  followed by a blank line and `Co-Authored-By: Antigravity <noreply@google.com>`.
- `docs/handoff/02-T1-source-library.REPORT.md` written and committed (format in README.md;
  put the real final commit hash in it, after any amend).
