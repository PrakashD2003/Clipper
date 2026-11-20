# Brief 03 (T2): frame contact sheets + frame notes (find moments in no-commentary videos)

**Start after:** brief 02 is merged (it is: `main` at `631052c` or later). Use the latest `main`.
**Your branch:** `handoff/03-T2-frame-notes` · **Worktree:** `D:\GTA6-Clipper-wt\handoff-03`

Read `docs/handoff/README.md` (rules), `docs/specs/topic-videos.md` (design, D7) and
`docs/specs/topic-videos-contracts.md` (fixed names). Run only the tests listed under "Done when".

```powershell
git -C D:\GTA6-Clipper fetch origin
git -C D:\GTA6-Clipper worktree add D:\GTA6-Clipper-wt\handoff-03 -b handoff/03-T2-frame-notes origin/main
```

## Goal

No-commentary gameplay has no speech to search. Let an LLM look at sampled frames and write
short notes ("player enters underground casino vault, neon signs") that are indexed and
searchable next to speech and chat. Two drivers: the MCP host LLM (any vision-capable LLM attached
to the MCP) or, locally, an Ollama vision model.

## Files you may change

- `src/streamclipper/library/frames.py` (new), `src/streamclipper/library/vision.py` (new)
- `src/streamclipper/library/index.py` — only: `add_frame_notes`, `list_frame_notes`, and making `search` also match `frame_notes_fts` (tables already exist from T1)
- `src/streamclipper/library/__init__.py` — exports only
- `src/streamclipper/mcp_server/server.py` — only the three new tools below + their docstrings + one line in `SERVER_INSTRUCTIONS`
- `src/streamclipper/cli.py` — only new `library frames` / `library describe` commands
- `config.default.yaml` + `src/streamclipper/config.default.yaml` — new `library.frames` / `library.vision` keys (byte-identical)
- `tests/test_frames.py` (new); `tests/test_mcp_server.py` only to extend `EXPECTED_TOOLS`

## Do

1. **Low-res proxy download** (`frames.py`): `download_proxy_video(url, out_dir, *, height=360) -> Path`
   via yt-dlp format `bv*[height<=360]/b[height<=360]/worst` (video only is fine), calling
   `apply_ytdlp_auth_opts`; cached at `<temp_dir>/frames/<source_id>/proxy.mp4`, reused if present.
2. **Contact sheets** (`frames.py`): `build_contact_sheets(config, source_id, *, interval_seconds=None) -> list[ContactSheet]`
   where `ContactSheet(path: Path, start: float, end: float, timestamps: list[float])` (SerializableModel).
   One sheet per `library.frames.sheet_seconds` (default 90 s) with a `library.frames.grid` (default 3 → 3×3)
   grid of frames, each tile labelled with its `mm:ss` timestamp, sheet width
   `library.frames.sheet_width` (default 1152). Use ffmpeg (`select`/`tile` filters or one `-ss`
   grab per tile) — no MoviePy needed. Skip sheets whose frames are nearly identical
   (static/loading screens; reuse `analysis.content_gate` thresholds if convenient). Cache sheets
   on disk; idempotent.
3. **Notes storage** (`index.py`): `add_frame_notes(source_id, notes: list[dict(t: float, note: str)]) -> int`
   (replace notes within the covered time range, keep FTS in sync), `list_frame_notes(source_id)`.
   `search(...)` also searches `frame_notes_fts`; a frame-note hit becomes a `SearchHit` spanning
   `[t-10, t+10]` (clamped) with `snippet` = the note, merged with speech hits from the same
   source when they overlap. Frame-note matches rank slightly below speech matches of equal bm25
   (`library.frames.note_weight`, default 0.8).
4. **Local vision** (`vision.py`): `describe_sheet_ollama(config, sheet) -> list[dict]` calling the
   Ollama chat API with the image (model `library.vision.model`, default `"qwen2.5vl:3b"`;
   `""` disables) and a prompt asking for one ≤ 15-word factual note per tile in JSON
   `[{"t": seconds, "note": "..."}]` (location, action, notable objects; no guessing names). Parse
   robustly, drop empty notes. If Ollama or the model is missing, raise a clear
   `VisionUnavailable` error (CLI prints how to `ollama pull` it).
5. **MCP tools** (contracts table): `index_frames(source_id)` → background job (download proxy +
   build sheets) via `_start_render_job`; `get_frame_sheets(source_id, start=0.0, count=6)` →
   returns the sheets as **MCP image content** (check how MCP SDK v2 returns images from a tool:
   `mcp.server.mcpserver` `Image` helper or `ImageContent`; fall back to base64 JPEG in JSON) plus
   JSON with each sheet's `start`, `end`, `timestamps` and the instruction text "write one short
   factual note per tile: location, action, notable objects"; `submit_frame_notes(source_id, notes)`
   (notes = JSON list of `{t, note}`) → validates t within source duration, note ≤ 200 chars,
   stores via `add_frame_notes`, returns counts. Add one sentence to `SERVER_INSTRUCTIONS`:
   no-commentary sources need `index_frames` → `get_frame_sheets` → `submit_frame_notes` before
   topic planning.
6. **CLI**: `library frames SOURCE_ID` (download proxy + sheets, prints count/folder);
   `library describe SOURCE_ID [--limit N]` (runs local vision over sheets, stores notes).
7. **Config** (both copies): `library.frames: {sheet_seconds: 90, grid: 3, sheet_width: 1152, proxy_height: 360, note_weight: 0.8}`,
   `library.vision: {model: "qwen2.5vl:3b"}`.

## Tests (`tests/test_frames.py`, names `test_T2_*`, no network, no GPU, no Ollama)

Make a tiny synthetic video with ffmpeg `testsrc` (e.g. 200 s at 160×90) for sheet building;
patch yt-dlp for the proxy download. Cover: sheet count/timestamps/caching/static-skip;
`add_frame_notes` replace semantics + FTS sync; search returns frame-note hits, merges with
speech hits, note_weight ordering; Ollama response parsing (valid JSON, JSON in code fences,
garbage → empty) with a mocked client; `VisionUnavailable` when the model is missing; MCP
`submit_frame_notes` validation; `get_frame_sheets` returns images; `index_frames` starts a job.

## Done when

- `pytest -q -p no:cacheprovider tests/test_frames.py tests/test_library.py tests/test_mcp_server.py` → 0 failed.
- `cmp` of both config.default.yaml copies → identical.
- Commit: `feat(library): frame contact sheets and searchable frame notes for no-commentary videos (T2)` + blank line + `Co-Authored-By: Antigravity <noreply@google.com>`.
- `docs/handoff/03-T2-frame-notes.REPORT.md` committed, with the **final** commit hash (write it after any amend, then amend once more without changing code if needed, or put the hash in your chat reply).
