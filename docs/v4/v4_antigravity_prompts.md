# StreamClipper v4 — Step-by-Step Antigravity Prompts (`3.8 Flash High`)

> **Before Ticket 1:**
> 1. Create a new branch in your `D:\GTA6-Clipper` repo: `git checkout -b v4-quality`
> 2. Replace the root `AGENTS.md` with the new v4 `AGENTS.md`, and place `v4-implementation-plan.md` and `v4-antigravity-prompts.md` inside `docs/v4/`.
> 3. Feed **one prompt at a time** to Antigravity. After each ticket passes `pytest`, render one real video/clip and inspect it before moving on.

---

## Prompt 1 — Ticket T1: Fix Critical Long-Form Bugs, Hardcoded Credits & Hashtag Spam

```text
Read AGENTS.md first, then implement ONLY Ticket T1 from docs/v4/v4-implementation-plan.md.

Scope — touch ONLY these files (+ their corresponding unit tests):
- src/streamclipper/assembly/longform.py
- src/streamclipper/assembly/topic_shorts.py
- src/streamclipper/output/topic_post.py
- config.default.yaml AND src/streamclipper/config.default.yaml (keep both byte-for-byte identical)

Exact changes required:
1. Remove the 440 Hz synthetic sine-wave fallback in production (`longform.py`):
   - In `render_long_video()`, if `narrator.synthesize_speech(...)` returns `None` or an empty file for cold_open, any point, or outro (when the text is non-empty), raise a clear `RuntimeError("TTS synthesis failed for ...")` instead of calling `_ensure_synthetic_wav()`.
   - Keep `_ensure_synthetic_wav()` only if existing unit tests import it directly, or update the tests cleanly so production renders never emit a 440 Hz tone.
2. Stop back-to-back duplicate footage in `longform.py`:
   - For each point's narration block (lines ~746–777), do NOT play the exact same `pt.evidence[0]` segment that immediately plays next at full volume.
   - Selection priority for a point's narration B-roll: (a) an unused clip from `pt.alternates` if downloaded/available, (b) `pt.evidence[1]` if the point has >= 2 evidence clips (trimmed to the last half), or (c) if only 1 clip exists, play a visibly distinct sub-range or dimmed/blurred Ken-Burns background from a *different* point's clip so the same footage never plays twice in a row.
3. Remove hardcoded `"Rockstar Games"` credits:
   - In `longform.py:805`, `topic_shorts.py:423`, and `topic_post.py:77,97,174,191`, replace hardcoded `"Rockstar Games"` with the clip's actual `clip.channel` when present, falling back to `f"Official {plan.game} footage"` when empty.
4. Fix hashtag stuffing:
   - Set `metadata.hashtag_count: 4` in both `config.default.yaml` files.
   - In `topic_post.py` (`write_long_post` and `write_short_post`), filter out common English/Hinglish stopwords (e.g., `the`, `and`, `for`, `how`, `every`, `what`, `this`, `that`, `with`, `from`, `into`, `actually`, `really`) when generating hashtags and cap total hashtags at 4 (`#Shorts` + game tag + up to 2 meaningful topic tags for Shorts; game tag + up to 3 for long-form).

Verification:
1. Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_longform.py tests/test_topic_shorts.py tests/test_config.py -v`
2. Run full suite `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`
3. Summarize exact changes and test results. Do not modify any other files.
```

---

## Prompt 2 — Ticket T2: Wire `ContextRefiner` into CLI & Fix Topic Shorts Midpoint Trims

```text
Read AGENTS.md first, then implement ONLY Ticket T2 from docs/v4/v4-implementation-plan.md.

Scope — touch ONLY these files (+ their corresponding unit tests):
- src/streamclipper/pipeline.py
- src/streamclipper/assembly/topic_shorts.py

Exact changes required:
1. Wire `ContextRefiner` into CLI `Pipeline._step_score()` (`pipeline.py`):
   - When `config.get("refinement.enabled", True)` is True, run `ContextRefiner(self.config)` on the top selected highlights (after signal fusion & deduplication, before assembly).
   - Build the 5-minute `ContextWindow` around each highlight (using existing `analysis/context_window.py` helpers, matching how `mcp_server/server.py:refine_highlight_bounds` constructs it).
   - If `refiner.refine(...)` succeeds with `confidence >= 0.5` and valid bounds (`exact_end > exact_start`), update `highlight.start`, `highlight.end`, `highlight.peak_time`, and `highlight.sub_clips` (if returned), then pass through `fit_clip_range()` so bounds stay within `[min_clip_seconds, max_clip_seconds]`.
   - Wrap each highlight's refinement in `try/except Exception` so a Gemini/Ollama failure logs a warning and keeps the original highlight bounds cleanly (never crashes the pipeline).
2. Fix `_calculate_clip_trims()` in `topic_shorts.py`:
   - Add an optional `peak_offsets: list[float | None] | None = None` parameter (default `None` for backward compatibility).
   - When a clip has a valid `peak_offset` inside `(0, raw_dur)`, anchor the trim window around `peak_offset` (60% before peak for setup, 40% after peak for reaction/payoff), clamped cleanly to `[0.0, raw_dur]`, instead of always trimming around `raw_dur / 2.0`.

Verification:
1. Add unit tests verifying (a) `Pipeline` calls `ContextRefiner` when `refinement.enabled=True` and falls back cleanly on error, and (b) `_calculate_clip_trims` preserves an off-center `peak_offset` near the end of a clip.
2. Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_pipeline.py tests/test_refiner.py tests/test_topic_shorts.py -v`
3. Run full suite `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`.
```

---

## Prompt 3 — Ticket T3: Full-Bleed 9:16 Shorts Layout (Zero Black Bars) & Animated Topic Shorts Captions

```text
Read AGENTS.md first, then implement ONLY Ticket T3 from docs/v4/v4-implementation-plan.md.

Scope — touch ONLY these files (+ their corresponding unit tests):
- src/streamclipper/assembly/layout.py
- src/streamclipper/assembly/composer.py
- src/streamclipper/assembly/topic_shorts.py
- config.default.yaml AND src/streamclipper/config.default.yaml (keep both identical)

Exact changes required:
1. Eliminate dead black bars in `stacked` Shorts layout (`layout.py` & `composer.py`):
   - Add a `"full_bleed"` `game_fit` option (and set `video.game_fit: "full_bleed"` as the new default in both `config.default.yaml` files; keep `"full"` and `"balanced"` working for existing tests).
   - For `("stacked", "full_bleed")` on a 1080x1920 canvas:
     - `facecam_box`: `y = 0..640` (top 33.3%, full width)
     - `game_box`: `y = 640..1920` (bottom 66.7%, center-cropped cleanly to fill 1080x1280 with zero black void)
     - `hook_band`: `y = 220..460` (inside YouTube Shorts top safe zone, below status/search bar)
     - `caption_band`: `y = 1120..1460` (strictly above YouTube's bottom ~420px title/channel UI zone)
   - In `composer.py:_compose_layout()`, ensure `game_fit == "full_bleed"` center-crops the 16:9 gameplay to fill `layout.game_box` without stretching, and for `"stacked"/"full"` render `_blur_background(clip)` behind the canvas so there are never solid black voids in any mode.
2. Upgrade `topic_shorts.py` captions:
   - Replace the static `NarrationCard` Arial boxes in `_build_shorts_ass()` with the active-word highlighted Hormozi style from `AssCaptionBurner` (or reuse `AssCaptionBurner.generate_ass()`), positioned cleanly in the safe zone (`MarginV >= 460`) so captions never collide with YouTube Shorts bottom UI.

Verification:
1. Update/add tests in `tests/test_layout_v3.py`, `tests/test_composer.py`, and `tests/test_topic_shorts.py`.
2. Run full suite `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`.
```

---

## Prompt 4 — Ticket T4: Audio Polish — BGM Bed, Transition SFX, Narration Gain & No Silent Chapter Cards

```text
Read AGENTS.md first, then implement ONLY Ticket T4 from docs/v4/v4-implementation-plan.md.

Scope — touch ONLY these files (+ their corresponding unit tests):
- src/streamclipper/assembly/narrator.py
- src/streamclipper/assembly/longform.py
- config.default.yaml AND src/streamclipper/config.default.yaml (keep both identical)

Exact changes required:
1. Wire `narration_gain_db` in `narrator.py`:
   - In `_build_ducking_filtergraph()`, if `abs(self.narration_gain_db) > 0.05`, apply `,volume={self.narration_gain_db:+.1f}dB` to each narrator track before `adelay` (instead of hardcoded `narr_tail = ""`).
2. Add optional royalty-free BGM bed support (`narrator.py` + `config.default.yaml`):
   - Add config keys under `transformation:`:
     - `bgm_path: ""` (path to an `.mp3`/`.wav` file or directory of tracks; `""` = disabled)
     - `bgm_volume_db: -26.0` (quiet background level under speech)
   - When `bgm_path` points to an existing audio file (or folder with `.mp3`/`.wav` files), loop it seamlessly (`-stream_loop -1`) trimmed to `clip_duration`, apply `volume={bgm_volume_db}dB` (ducked an extra `-4dB` during narration windows), and mix it cleanly into `amix`. If unset or missing, skip silently without failing.
3. Eliminate 2.5-second dead-silent chapter cards in `longform.py`:
   - Lower default `longform.card_seconds` from `2.5` to `1.2` in both `config.default.yaml` files.
   - Add optional `longform.transition_sfx_path: ""` config key. When set to an existing `.wav`/`.mp3` (e.g. a soft whoosh), attach it to the chapter card clip at `-12 dB` instead of `AudioClip(lambda t: [0.0, 0.0], ...)`. When unset, add a subtle fade-in/fade-out on the chapter card visual (`CrossFadeIn(0.2)` / `CrossFadeOut(0.2)`) and mix the configured BGM bed across the card so there is never abrupt dead silence.

Verification:
1. Add unit tests for `narration_gain_db` filtergraph emission, BGM mixing when present vs. skipped when empty, and 1.2s chapter card behavior.
2. Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_narrator.py tests/test_longform.py tests/test_config.py -v`
3. Run full suite `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`.
```

---

## Prompt 5 — Ticket T5: Packaging Upgrade — Frame-Scored Thumbnails & 5 Title Variants

```text
Read AGENTS.md first, then implement ONLY Ticket T5 from docs/v4/v4-implementation-plan.md.

Scope — touch ONLY these files (+ their corresponding unit tests):
- src/streamclipper/assembly/longform.py
- src/streamclipper/output/topic_post.py
- src/streamclipper/output/post_writer.py

Exact changes required:
1. Upgrade `generate_topic_thumbnail()` in `longform.py`:
   - Instead of grabbing a single arbitrary midpoint frame, sample 5 evenly spaced frames across the clip (20%, 35%, 50%, 65%, 80%) and score each frame by Laplacian sharpness (`cv2.Laplacian(..., cv2.CV_64F).var()`) + brightness balance (penalize near-black/washed-out frames). Pick the highest-scoring frame.
   - Apply subtle Pillow `ImageEnhance.Contrast(1.18)` and `ImageEnhance.Color(1.22)` so the frame pops at mobile thumbnail size.
   - Cap thumbnail text to **3 words max**, render in bold white/yellow with thick black stroke + subtle drop shadow in the **upper-left or upper-center safe zone** (never bottom-right where YouTube puts the duration badge).
2. Add multi-variant title suggestions in `topic_post.py` and `post_writer.py`:
   - Under `TITLE`, output the primary title plus a `TITLE VARIANTS (A/B TEST)` section listing 3–5 punchy curiosity-gap alternatives (shortened/reframed from the topic/hook metadata) ready for YouTube Studio's Test & Compare feature.

Verification:
1. Run `D:\GTA6-Clipper\.venv\Scripts\pytest.exe tests/test_longform.py tests/test_topic_shorts.py tests/test_assembly.py -v`
2. Run full suite `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`.
```

---

## Prompt 6 — Ticket T6: Dual-Channel Profiles (`en` / `hi`) Sharing One GTA 6 Library

```text
Read AGENTS.md first, then implement ONLY Ticket T6 from docs/v4/v4-implementation-plan.md.

Scope — touch ONLY these files (+ their corresponding unit tests):
- config.default.yaml AND src/streamclipper/config.default.yaml (keep both identical)
- src/streamclipper/config.py
- src/streamclipper/cli.py
- src/streamclipper/topics/ledger.py

Exact changes required:
1. Add a `channels:` block to both `config.default.yaml` files:
   ```yaml
   channels:
     en:
       output_dir: "./output/en"
       narration_language: "english"
       ledger_name: "topics_ledger_en.json"
     hi:
       output_dir: "./output/hi"
       narration_language: "hinglish"
       ledger_name: "topics_ledger_hi.json"
   ```
2. Add `Config.apply_channel_profile(channel: str | None)` in `config.py`:
   - When `channel` is provided (e.g. `"en"` or `"hi"`), override `general.output_dir`, `transformation.narration_language`, and `topics.ledger_name` from `channels.<channel>` while keeping `general.cache_dir` and `library.db_path` shared!
   - When `channel` is `None`, preserve exact existing default behavior so all existing tests pass untouched.
3. Add optional `--channel` CLI option (`en` | `hi`, default `None`) to `clip`, `auto`, `topics plan`, `topics render`, and `topics shorts` in `cli.py`, calling `cfg.apply_channel_profile(channel)`.
4. Update `TopicLedger` (`topics/ledger.py`) to read `config.get("topics.ledger_name", "topics_ledger.json")` so English and Hinglish channels track produced topics independently without blocking each other.

Verification:
1. Add unit tests verifying that `--channel en` and `--channel hi` write to separate output dirs and ledgers while sharing the same `cache_dir` and `library.db_path`.
2. Run full suite `D:\GTA6-Clipper\.venv\Scripts\pytest.exe -v`.
```
