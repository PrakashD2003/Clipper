# StreamClipper Full Review — 2026-11-19

| Field | Value |
| :--- | :--- |
| Scope | `src/streamclipper/**` (33 modules), `config.default.yaml`, `tests/` fakes, `docs/streamclipper-implementation-guide-v2.md` |
| Branch / HEAD | `main` @ `60d4b1f` |
| Method | 6 read-only slice reviewers (Sonnet), 1 except/config sweep (Haiku), 1 spec-gap agent (Sonnet); lead-architect merge, verification, and cross-module pass (Opus) |
| Code changes | None. This is a read-only review. |
| Excluded as already fixed | except-TypeError retry shims (pipeline/exporter/MCP), breakdown windows, `_trim_start` offset, music-penalty ordering |

Verdict key: **CONFIRMED** means the lead read the cited code. **PLAUSIBLE** means the lead checked the code partially, or the finding is a design-level risk. `agent-confirmed` / `agent-likely` means a slice agent read the code and the lead did not re-verify it (medium and low findings only, per the review rules).

---

## Critical

No critical findings.

## High

| ID | File:line | Defect | Failure | Fix | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| H1 | `scoring/scorer.py:141` (words set at `:153`); `mcp_server/server.py:1065` (words set at `:1075`) | `CopyrightRiskDetector.detect_highlight()` runs **before** `highlight.word_timestamps` is populated in both callers. With an empty list, `copyright_risk.py:56-57` returns the entire `[start,end]` as one non-speech gap. | Any highlight gets HPSS/flatness run over streamer speech + game audio, so `music_risk` / `music_confidence` are wrong. Speech-heavy clips can be flagged `HIGH_MUSIC_DETECTED` and take the 0.80 penalty (and drop below `min_score`), while real music is diluted. | Populate `word_timestamps` from the transcript before calling the detector, in both callers (or pass the transcript to `detect()`). | CONFIRMED (both call orders read; empty-list branch read) |
| H2 | `pipeline.py:500,745-749` → `output/organizer.py:103-120` | The CLI/`Pipeline` path names clips `clip_{idx:02d}` by `game/streamer/date` with `idx` from `enumerate(...,1)` and no existence check. The MCP path uses `_next_available_clip_index` (`server.py:404`). | Two VODs by the same streamer/game on the same day (e.g. `streamclipper auto`, or a second `clip` run) both write `clip_01.mp4`. The second run silently overwrites the first clip plus its `.json` and `_transcript.txt`. | Move the next-free-index logic into `FileOrganizer` and use it from `Pipeline._step_assemble`. | CONFIRMED (`filename_template: "clip_{index:02d}"`, no exists check in the pipeline path) |

## Medium / Low

| ID | Sev | File:line | Defect | Failure | Fix | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M1 | med | `scoring/scorer.py:464-492` → `assembly/captions.py:383` | `_extract_chat_burst` takes `sample_messages` from any overlapping chat spike, with no per-message `[h_start,h_end]` filter and no time sort. | A spike spanning 100–140 s and a highlight at 120–170 s: the message at 101 s is clamped by `max(0, offset - clip_start_offset)` and pinned to t=0. Late messages fall past the clip end. | Filter by message offset, sort by time, then dedupe and cap at 5. | CONFIRMED |
| M2 | med | `scoring/ollama_scorer.py:442` | The chunk-relative→absolute heuristic shifts by `chunk_offset` when `start < chunk_offset and end <= chunk_duration`. | Chunk at 660 s (duration 720): the model correctly returns absolute 655–700 (from the overlap region), which is shifted to 1315–1360 → the wrong moment is clipped. Beats may diverge from the parent time base. | Shift only when the whole range is below `chunk_offset`, or drop the heuristic (the prompt is already absolute). | agent-likely |
| M3 | med | `scoring/ollama_scorer.py:210-275,375` | Every `client.generate` exception (including `httpx.ReadTimeout`) becomes `OllamaConnectionError`, and `_score_chunk` does not catch it. | One slow chunk on `qwen3:4b`/GTX 1660 SUPER (120 s timeout) aborts scoring of a 4 h VOD. All prior chunks are lost and the error says "cannot connect". | Retry/skip per chunk on timeout; raise only on real connection/404 errors. | agent-confirmed |
| M4 | med | `download/chat.py:77-91`; consumers `pipeline.py:422-438`, `server.py:797-808` | A transient yt-dlp failure (429, bot-check, network) returns the same empty DataFrame as "no chat replay". Callers then persist `chat_spikes=[]` + analysis complete; the empty DataFrame itself is never cached (MCP re-scrapes on every call). | One rate-limit permanently strips the chat modality from that state; fusion silently renormalizes and resume never re-fetches. | Raise `ChatScrapeError` on extractor failure (the step then goes FAILED and retries); cache an explicit "no chat" sentinel only on extractor success. | agent-confirmed |
| M5 | med | `cache.py:55-73,134-150` | State and transcript JSON are written in place (no tmp + `os.replace`); `load_state` does not handle `JSONDecodeError`; `find_by_audio_path` hides corrupt files via `list_all`. | A crash or Ctrl-C mid-write → every later `process_stream(url)` raises at `pipeline.py:220`, outside the FAILED handler → the URL is stuck until `--force`. | Write atomically; return `None`/quarantine on decode error; log corrupt files. | agent-confirmed |
| M6 | med | `pipeline.py:270-285` (and the `:235-245` warning) | Highlights are cached under the first run's `max_clips`; the "options ignored" warning covers only COMPLETE. | Run with `--max-clips 1`, fail in assembly, rerun with `--max-clips 5` → still 1 clip, no warning. | Persist `max_clips` in state and re-score or warn on mismatch. | agent-confirmed |
| M7 | med | `mcp_server/server.py:1346-1363` | `render_all_selected` re-renders every selected highlight on each call; it dedupes only by `video_path`, and `_next_available_clip_index` always advances. | A batch of 4 fails on #3, then the client retries → 4 more renders as clip_03..06, with duplicate MP4s and ClipResults. | Skip highlights already present in `state.clips` (match start/end/url) unless `force=True`. | agent-confirmed |
| M8 | med | `mcp_server/server.py:1156,1314` vs `pipeline.py:117-135` | The MCP tools default `mode="auto"`, and `_resolve_base_mode` treats any non-empty string as explicit. | `transformation.mode: raw` or `transformation.enabled: false` in config is ignored for every MCP render that omits `mode`. | Default `mode=None` so config applies. | agent-confirmed |
| M9 | med | `mcp_server/server.py:1010-1058` | `submit_highlights` checks only `end > start`, not the `min_clip_seconds` / `max_clip_seconds` limits that `scorer.py:120` enforces. | A client submits 0–7200 → a 2 h 1080p segment download and a 2 h composite render. | Reject or clamp out-of-range durations (also in `render_clip`). | agent-confirmed |
| M10 | med | `mcp_server/server.py:1059-1069`; `scorer.py:147-148`; `copyright_risk.py:177` | The copyright detector runs synchronously inside an async MCP tool while holding the URL lock. Failures are swallowed (`pass` / `debug`), and per-gap HPSS errors are also only debug-logged. | 5 highlights on a 4 h WAV block the MCP event loop; a detector crash silently leaves `music_risk` at its default (looks "clean"), so the flags are non-uniform across highlights. | Use `asyncio.to_thread`; log at warning; report an `UNKNOWN` risk state. | agent-confirmed |
| M11 | med | `assembly/narrator.py:806-828` → `output/exporter.py:277-320` | When the ducking mix fails, the narrator returns a plain video copy, but the exporter still passes `eff_intro`/`eff_outro` to `generate_ass`. | An ffmpeg filtergraph error → the clip has no voiceover, yet streamer captions are suppressed during the narration windows and narration cards are burned for audio nobody hears. | Return a failure flag; on failure, pass `intro_dur`/`outro_dur=None`. | agent-confirmed |
| M12 | med | `assembly/narrator.py:97-101` | An explicit `tts_engine: kokoro` returns `_synthesize_kokoro()` directly, with no edge-tts fallback (guardrail 13 violation). | Windows without espeak-ng → Kokoro returns `None` → no narration at all (the log claims it falls back). | Fall through to edge-tts when the Kokoro result is `None`, as the `auto` branch does. | agent-confirmed |
| M13 | med | `output/exporter.py:300-305` | `generate_ass` runs only `if has_words or has_chat`. | A quiet clip (the exact case the `0.85*score` quiet-pick floor exists for) with no streamer words and no chat → `narration_intro/outro` and `breakdown_callout` cards are never burned. | Also generate the ASS file when narration or callout text is present. | CONFIRMED (promoted from assembly BOUNDARY) |
| M14 | med | `analysis/loudness.py:~125` | `peak_time = idx * hop_duration` is the RMS frame **start** (`frame_length = 2*hop`, `center=False`). | `peak_time` lands up to ~1–2 s early, so the 1.20× zoom and `peak_offset` fire before the actual spike. | Use the frame centre `(idx+1)*hop_duration`, or argmax the samples within the frame. | agent-likely |
| M15 | med | `analysis/transcriber.py:~112-123` | `load_align_model`/`align` share the ASR try block; only `ImportError` falls back. | A detected language with no wav2vec2 aligner → ValueError → the whole transcription fails (ASR output discarded); FAILED retry repeats the same failure. | Catch alignment errors and return the unaligned segments (or use faster-whisper word timestamps). | agent-confirmed |
| M16 | med | `download/video.py:304-306,340` | `download_segment(s)` calls `_extract_video_id(url)` (a full network extraction) before the cache check, on every call. | Re-rendering cached segments offline or while throttled raises `VideoDownloadError`; every highlight pays an extra extraction. | Accept an optional `video_id` (callers already have `state.stream_info.video_id`); hit the network only on a cache miss. | agent-confirmed |
| L1 | low | `download/video.py:163-189,301` | Segment cache key `{id}_{int(start)}_{int(end)}`, and a missing sidecar counts as a match. | 100.2 s and 100.7 s highlights share one file; concurrent renders can delete or swap it; a crash before the sidecar write leaves the wrong buffer reused. | Key on milliseconds plus buffer; write the sidecar before treating the file as valid. | agent-confirmed |
| L2 | low | `download/audio.py:57-93` (and the copy in `video.py`) | The error classifier matches over-broad substrings ("sign in", "region"). | A bot-check is reported as age-restricted, sending the user toward the wrong remedy. | Match specific phrases; check bot/429 first. | agent-confirmed |
| L3 | low | `download/audio.py:166-185,222` | No `live_status` check. | A currently-live URL → unbounded recording, and `duration_seconds=0`. | Reject `is_live` / `is_upcoming`. | agent-likely |
| L4 | low | `download/chat.py:170-171` | The try/except wraps the whole replay line; superchats without text runs are dropped. | One malformed action drops the rest of that burst. | Move the try inside the per-action loop. | agent-confirmed |
| L5 | low | `discovery/youtube.py:150-227` | No `was_live`/recency/title-relevance filter; results are sorted by views only. | Long non-stream videos pass, and `StreamInfo.game` is stamped with `target_game` regardless. | Require `was_live` and a recency window. | agent-confirmed |
| L6 | low | `analysis/chat_velocity.py` (`_extract_sample_messages`, `<= end`) | The sample window is inclusive on both ends. | A boundary message is counted in two adjacent spikes. | Use `< end` except for the final bin. | agent-confirmed |
| L7 | low | `scoring/scorer.py:309-310` | Modality weights are zeroed only when the whole stream has none of that signal. | One loudness spike anywhere → a quiet LLM-8.0 pick scores 5.33 < 7.0 and is dropped. | Document the intent, or renormalize per highlight. | agent-likely |
| L8 | low | `assembly/captions.py:326-333` + `pipeline.py:723-729` | In freeze-frame breakdown mode, post-freeze words are shifted by `freeze_dur`, but the callout window spans the full TTS length. | TTS 7 s > freeze 4 s → streamer words overlap the Hormozi callout. | Drop words inside `breakdown_window`. | agent-likely |
| L9 | low | `assembly/captions.py:455-476`; `output/exporter.py:132-140,296-331` | ffmpeg `CalledProcessError` stderr is never logged; the NVENC retry fires on any error. | A libass/encoder failure surfaces as a bare "exit status 1". | Log `exc.stderr[-500:]` before retry or re-raise. | agent-confirmed |
| L10 | low | `assembly/narrator.py:139` | `KPipeline` is rebuilt for every synthesis call and its VRAM is never released. | On a 6 GB card, Kokoro VRAM is left behind for the next Whisper/Ollama step. | Cache one pipeline per voice and release it after the render (or run it on CPU). | agent-likely |
| L11 | low | `assembly/composer.py:440-446`; `pipeline.py:638-643` | An invalid beat trim silently uses the raw clip; `beat_trim_ranges` uses nominal, not actual, segment durations. | A short beat download → `cum_offset` drift → captions, chat and zoom misaligned for later beats. | Remap from actual beat durations or fall back to single-cut. | agent-confirmed |
| L12 | low | `pipeline.py:526-562` | `_gpu_lock` is held across the whole `_assemble_highlight` (MoviePy, ffmpeg, TTS). | `auto` with `max_concurrent_streams: 2` serialises scoring behind full renders. | Lock only around the GPU sub-steps. | agent-confirmed |
| L13 | low | `pipeline.py:849-854` | `_close_composed_handles` uses `except Exception: pass`. | A failed Windows handle close is invisible until a later PermissionError. | `logger.debug` the exception. | agent-confirmed |
| L14 | low | `cli.py:135,246-268` | `auto` exits 0 when every stream failed; a ValueError (bad mode / missing model) prints a raw traceback. | Schedulers see success after 5/5 failures. | `ctx.exit(1)` on failures; wrap ValueError in `ClickException`. | agent-confirmed |
| L15 | low | `config.py:54-68,227` | `detect_hardware_profile` runs `ffmpeg -encoders` and imports torch on every `Config()`. | Repeated cost in the MCP server and tests, and `--help` is slow. | `functools.lru_cache` the profile. | agent-likely |
| L16 | low | `config.py:134` | `except Exception: pass` in hardware detection silently falls through to `cpu_only`. | A transient torch/CUDA query error on an RTX 4050 → the whole run is on CPU with no log. | Log a warning with the exception. | CONFIRMED (sweep) |
| L17 | low | `mcp_server/server.py:1522-1552` | `cleanup_temp` does not take the URL lock. | Deletes in-use `temp/segments/*.mp4` during a render. | Wrap it in `_get_url_lock(cache_key)`. | agent-confirmed |
| L18 | low | `mcp_server/server.py:607-616,714-721,840-848` | The MCP tools' own FAILED handling ignores existing highlights and does not call `Pipeline._recover_failed_state`. | A FAILED state with `selected_highlights` → status regresses to SCORING/TRANSCRIBING. | Call `_recover_failed_state` before the `<` comparisons. | agent-confirmed |
| L19 | low | `mcp_server/server.py:1342-1358` | `render_all_selected` reports `"auto"` instead of the resolved per-clip mode (the tuple element is discarded). | The client cannot tell which clips are story, narrative or breakdown. | Return the resolved mode per clip. | agent-confirmed |
| L20 | low | `pipeline.py:415-420` vs `mcp_server/server.py:315-345` | The chat CSV cache filename is built with different sanitizers in the CLI and MCP paths. | IDs with unusual characters → split or duplicate chat caches between entry points. | Share one helper. | agent-likely (promoted BOUNDARY) |
| L21 | low | `tests/test_pipeline.py:193,285,910-927` | Test fakes are narrower than the real functions: `fake_download_audio` has no `progress_callback`; `FakeOllamaScorer.generate_metadata` has no `mode`; `StubVideoComposer.compose`/`compose_story_beats` have no `face_center_x`. | Tests never exercise `mode=` / breakdown metadata or a future progress callback; the `inspect.signature` shim hides the drift. | Widen the fakes to the real signatures. | agent-confirmed (promoted BOUNDARY) |
| L22 | low | `pipeline.py:584-594` | Non-story modes ignore `valid_beats` and use `segment_paths[0]`. | `mode=breakdown` with beats silently drops the beats (may be by design; undocumented). | Document this or log when beats are dropped. | agent-likely (promoted BOUNDARY) |
| L23 | low | `config.default.yaml:44-45,64,122,128` | Dead config keys: `analysis.copyright.high_risk_threshold`/`low_risk_threshold` (the detector reads `analysis.music_risk_threshold`), `llm.provider`, `transformation.replay_badge_text`, `transformation.narration_style`. | Users who tune these keys see no effect. | Wire them up or remove them (keep both `config.default.yaml` copies in sync; they are currently identical). | CONFIRMED (grep) |
| L24 | low | `assembly/narrator.py:155,212,286,669,687,857`; `assembly/hooks.py:378,408`; `assembly/composer.py:210,346,360,484` | Sweep: `except Exception` handlers that only pass/return/log; no slice agent explained them. | Possible silent degradation in TTS, hooks and composition. | Audit each site; raise the log level where a feature silently turns off. | sweep only (unreviewed) |

---

## Architecture (cross-module contract pass)

Data flow: url → audio → transcript → signals → highlights → clips.

| ID | Handoff | Mismatch | Consequence | Fix | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A1 | scoring → assembly (chat time base) | Scoring emits stream-absolute `chat_burst` that is not bounded to the highlight; captions assume every message lies inside `[clip_start_offset, clip_end]` and clamp with `max(0, …)`. | Out-of-window chat is pinned to frame 0 or burned past the clip end (= M1). | Enforce the window at the producer. | CONFIRMED |
| A2 | analysis → assembly (`peak_time`) | Guardrail 7 promises an "exact" `peak_time`; the producer emits the frame start. The composer converts it to a clip-relative `peak_offset` for the zoom. | The zoom fires early (= M14). | Producer emits the frame centre. | PLAUSIBLE |
| A3 | analysis (copyright) ← scoring / MCP | The detector's contract needs absolute `word_timestamps`; both producers fill them after the call. | Wrong music risk (= H1). | Fix the call order. | CONFIRMED |
| A4 | Pipeline (CLI) vs MCP → organizer | Two clip-index allocators: `enumerate` vs `_next_available_clip_index`. | CLI overwrites (= H2). | One allocator in `FileOrganizer`. | CONFIRMED |
| A5 | cache key vs artifact key | `StepCache.get_cache_key` = `sha256(raw url)[:16]` with no URL normalisation (`cache.py:26-29`), while the WAV, `_chat.csv` and segments are keyed by `video_id`. | `youtu.be/X`, `watch?v=X` and `watch?v=X&t=1s` create three independent states sharing the same `{vid}.wav` / segments. Parallel `auto` runs or MCP calls race on those files, and the per-URL MCP lock does not cover the aliases. | Key state on the canonical `video_id` (fall back to the normalised URL). | CONFIRMED |
| A6 | FAILED-resume semantics (all stages) | `Pipeline._recover_failed_state` vs per-tool MCP recovery (L18). Resume is keyed on status only, not on inputs (`max_clips` M6, chat-fetch success M4). Neither path checkpoints per clip (`state.clips` is written at the end; M7). | A retry can regress stage, ignore new options, persist a missing modality, or duplicate or redo renders. | One shared recovery function, an input fingerprint in state, and per-clip checkpoints. | PLAUSIBLE (components agent-confirmed) |
| A7 | MCP → Pipeline (mode resolution) | The MCP default `"auto"` is treated as explicit by `_resolve_base_mode`. | Config `transformation.*` is bypassed (= M8). | Pass `None`. | agent-confirmed |
| A8 | narrator → exporter → captions (narration windows) | The narrator does not signal a mix failure (M11), and the exporter skips ASS generation without words or chat (M13). | Caption/narration state disagrees with the audio actually present. | Explicit narration result object `{mixed: bool, intro_dur, outro_dur}`. | CONFIRMED (M13) / agent-confirmed (M11) |
| A9 | GPU lock / VRAM | `Pipeline._gpu_lock` is per instance; MCP builds a new `Pipeline` per call (`server.py:1182,1325`) and relies on its own global lock, which is consistent within one process. Kokoro VRAM is never freed (L10), the lock spans whole renders (L12), and a concurrent CLI process plus MCP server share no lock. | On 6 GB cards (GTX 1660 SUPER / RTX 4050), CLI + MCP or Kokoro + WhisperX can run out of VRAM. | Release Kokoro, narrow the lock, and optionally use a cross-process file lock for GPU stages. | PLAUSIBLE |
| A10 | download → pipeline → composer (beat durations) | Segment files are keyed by truncated ints (L1); `beat_trim_ranges` assume nominal lengths (L11). | Beat timeline drift in story mode. | Key on ms plus buffer; remap from actual durations. | agent-confirmed |

---

## Feature gaps

| # | Gap | Source |
| :--- | :--- | :--- |
| G1 | No alignment-failure fallback to unaligned ASR; no VAD in the faster-whisper path | analysis |
| G2 | No `UNKNOWN`/unchecked copyright state when the detector cannot run | analysis, scoring |
| G3 | No per-chunk scoring checkpoint (a failed score step re-scores every chunk) | scoring |
| G4 | Blank or silent transcript chunks are still sent to the LLM | scoring |
| G5 | No live/upcoming-stream guard; no retry/backoff for yt-dlp throttling (download + chat) | download |
| G6 | Discovery has no recency window or title/game relevance filter | download/discovery |
| G7 | No TTS duration-budget check before synthesis (long narration is cut mid-sentence by the filtergraph) | assembly |
| G8 | Facecam detection uses Haar frontal-face only, on the first segment (VTuber/stylised cams fall back to Layout B) | assembly |
| G9 | No per-clip checkpoint (CLI) and no single-clip re-render / delete / `force` (MCP) | core, MCP |
| G10 | `HIGH_MUSIC_DETECTED` highlights are still rendered in non-interactive runs (no skip/config gate) | core |
| G11 | `submit_highlights` has no 3–5 count cap and no overlap dedupe (unlike `score_and_select`) | MCP |
| G12 | Spec check: all guide packages 1.x–5.x present by identifier (existence only, not behaviour). Deviations: the CLI `--mode` also accepts `auto`/`breakdown` beyond the guide's list; `config.default.yaml` is duplicated at the repo root and in `src/streamclipper/` (currently identical; drift risk). | spec |

---

## Prioritised fix plan

Batches have **no overlapping files** and can run in parallel. The hotspot files (`pipeline.py`, `exporter.py`, `tests/test_pipeline.py`) are all in Batch A. Effort: S < ½ day, M ≈ 1 day, L > 1 day.

| Order | Batch | Fix IDs (in order) | Files touched | Effort |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **B — Scoring** | H1 (scorer side) → M1 → M3 → M2 → M10 (scorer + detector logging, `UNKNOWN` state) → L7 → L23 | `scoring/scorer.py`, `scoring/ollama_scorer.py`, `analysis/copyright_risk.py`, `config.default.yaml` (both copies), `tests/test_scorer.py`, `tests/test_copyright_risk.py` | H1 S, M1 S, M3 M, M2 S, M10 S, L7 S, L23 S → **M** |
| 2 | **A — Pipeline / output / assembly hotspot** | H2 → M13 → M11 → M12 → M6 → L11 → L8 → L9 → L10 → L12 → L13 → L21 → L22 → L24 | `pipeline.py`, `output/exporter.py`, `output/organizer.py`, `assembly/narrator.py`, `assembly/captions.py`, `assembly/composer.py`, `assembly/hooks.py`, `tests/test_pipeline.py`, `tests/test_pipeline_breakdown.py`, `tests/test_assembly.py`, `tests/test_composer.py`, `tests/test_narrator.py`, `tests/test_breakdown_mode.py`, `tests/test_hooks.py` | H2 S, M13 S, M11 M, M12 S, M6 S, L11 M, others S → **L** |
| 3 | **C — MCP server** | H1 (server side) → M8 → M9 → M7 → M10 (`to_thread`) → L18 → L17 → L19 → L20 | `mcp_server/server.py`, `tests/test_mcp_server.py` | H1 S, M8 S, M9 S, M7 M, M10 S, L17–L20 S → **M** |
| 4 | **F — Cache / config / CLI** | A5 → M5 → L16 → L15 → L14 | `cache.py`, `config.py`, `cli.py`, `tests/test_cache_cli.py`, `tests/test_config.py` | A5 M, M5 M, L14–L16 S → **M** |
| 5 | **D — Download / chat / discovery** | M4 → M16 → L1 → L3 → L2 → L4 → L6 → L5 | `download/chat.py`, `download/video.py`, `download/audio.py`, `discovery/youtube.py`, `analysis/chat_velocity.py`, `tests/test_chat.py`, `tests/test_download.py`, `tests/test_discovery.py`, `tests/test_discovery_geo.py` | M4 S, M16 S, L1 M, others S → **M** |
| 6 | **E — Audio analysis** | M15 → M14 | `analysis/transcriber.py`, `analysis/loudness.py`, `tests/test_transcriber.py`, `tests/test_loudness.py` | M15 M, M14 S → **S–M** |

Sequencing notes:
- H1 spans B (scorer) and C (server); land both before the next release.
- M16 adds an optional `video_id` parameter in D. Callers in `pipeline.py` (A) and `server.py` (C) can adopt it in a later pass without file overlap.
- M4 is designed to need no caller changes: raising on extractor failure sends the analyze step to FAILED.
- A6 (unified recovery plus input fingerprint) is a follow-up design item after A, C and F land.

---

## Footer — counts

| Agent | Findings reported | BOUNDARY lines | Kept in report | Merged / deduped | Excluded |
| :--- | :--- | :--- | :--- | :--- | :--- |
| (a) download + discovery | 7 | 3 | 7 (M4, M16, L1–L5) | 3 boundaries → L1/A10, L20, L21 | 0 |
| (b) analysis | 6 | 3 | 5 (H1, M14, M15, L6, + M10 part) | 3 boundaries → M4, A2 | 1 (short-audio no-spike: non-defect) |
| (c) scoring + prompts | 6 | 3 | 4 (M1, M2, M3, L7) | 1 (detector swallow → M10); 3 boundaries → L21, 2 dropped (not read by agent) | 1 (dedupe-before-penalty: on the already-fixed list; note that `scorer.py:126` still dedupes before the penalty at `:131`, so confirm this is intended) |
| (d) assembly | 6 | 3 | 6 (M11, M12, L8–L11) | 3 boundaries → M13, L21, L11 | 0 |
| (e) core (pipeline/cache/models/config/cli/output) | 8 | 3 | 8 (H2, M5, M6, L9, L12–L15) | 3 boundaries → L1, 2 dropped (unsubstantiated) | 0 |
| (f) mcp_server | 8 | 3 | 8 (M7–M10, L17–L19, + M4 part) | 3 boundaries → L1, L22, M5 | 0 |
| Sweep (Haiku) | 4 `except TypeError` + 20 swallow sites + 5 dead config keys | — | L16, L23, L24 | 4 TypeError hits benign (2 known shims, `ollama_scorer.py:223` think-kwarg compat, `hooks.py:36` PIL compat); 7 swallow sites explained by slice agents or logged | 0 |
| Spec | 0 missing / 0 partial | — | G12 notes | — | 0 |
| Lead cross-module pass | — | — | A1–A10 (1 new: A5) | 9 cross-refs to slice findings | 0 |

**Totals:** Critical 0 · High 2 · Medium 16 · Low 24 · Architecture 10 (1 new) · Feature gaps 12.
**Verification:** 2/2 high findings CONFIRMED; 6 further items CONFIRMED by the lead (M1, M13, L16, L23, A5, A3/A4 cross-refs). **REJECTED: 0.** Excluded: 2 (1 already-fixed, 1 non-defect).
