# Brief 04 (T3): topic discovery + library MCP tools

**Start after:** brief 03 is merged into `main` (the lead confirms). Use the latest `main`.
**Your branch:** `handoff/04-T3-topic-discovery` · **Worktree:** `D:\GTA6-Clipper-wt\handoff-04`

Read `docs/handoff/README.md`, `docs/specs/topic-videos.md` (D2, D3, D13) and
`docs/specs/topic-videos-contracts.md`.

```powershell
git -C D:\GTA6-Clipper fetch origin
git -C D:\GTA6-Clipper worktree add D:\GTA6-Clipper-wt\handoff-04 -b handoff/04-T3-topic-discovery origin/main
```

## Goal

Propose topic-video ideas that the library has enough evidence for, so the user can pick one.
Works with any LLM attached to the MCP (host proposes topics from material we send) and
locally with Ollama. Also expose the library itself through MCP.

## Files you may change

- `src/streamclipper/topics/__init__.py`, `topics/models.py` (only `EvidenceRef`, `TopicCandidate`), `topics/ledger.py`, `topics/discover.py` (all new)
- `src/streamclipper/prompts/topic_discovery.txt` (new)
- `src/streamclipper/mcp_server/server.py` — only the new tools below + docstrings + `SERVER_INSTRUCTIONS` workflow sentence
- `src/streamclipper/cli.py` — only a new `topics` group with `suggest` and `ledger` commands
- both `config.default.yaml` copies — new `topics:` section (byte-identical)
- `tests/test_topics_discovery.py` (new); `tests/test_mcp_server.py` only to extend `EXPECTED_TOOLS`

## Do

1. **Ledger** (`topics/ledger.py`): `TopicLedger(config)` over `<cache_dir>/topics_ledger.json`
   (atomic writes): `record(topic_id, title, game, keywords, shape, plan_id="")`, `entries(game)`,
   `find_similar(title, keywords, game) -> entry | None` (token Jaccard on title+keywords ≥
   `topics.similarity_threshold`, default 0.5), `last_shape(game)`.
2. **Material** (`discover.py`): `topic_material(config, game, *, max_terms=40) -> dict` with:
   top salient terms/bigrams across the game's library segments + frame notes (TF-IDF style:
   frequent across many windows/sources, rare in generic gaming talk; drop EN + Hinglish
   stopwords and filler like "bro", "bhai", "chat", "guys", "lol"); for each term up to 3 example
   snippets with channel + mm:ss; library stats (sources, channels, kinds, total hours); and the
   titles already in the ledger for this game.
3. **Candidates** (`discover.py`): `evaluate_topic(config, game, title, keywords) -> TopicCandidate`
   runs `LibraryIndex.search` for the title and each keyword, keeps the best distinct spans as
   `EvidenceRef`s (max `topics.max_evidence` 12), counts distinct sources/channels, sets
   `already_made`/`similar_to` from the ledger, `topic_id` = slug of the title.
   A candidate is **viable** when it has ≥ `topics.min_evidence` (6) evidence spans from ≥
   `topics.min_sources` (2) sources.
   `suggest_topics_local(config, game, *, limit=10) -> list[TopicCandidate]`: send the material to
   Ollama with `prompts/topic_discovery.txt` (ask for JSON list of `{title, keywords[3-8], angle}`
   — specific, curiosity-driven, e.g. "Every hidden detail in the Vice City docks", not
   "GTA 6 gameplay"), evaluate each, return viable ones sorted by evidence strength, already-made
   ones last and flagged. If Ollama is unavailable, fall back to term clusters (top terms grouped by
   co-occurrence) evaluated the same way.
4. **MCP tools** (names fixed in contracts):
   - `library_add_source(url, game, kind="stream", channel="")` → background job running
     `add_source` (download + transcribe + analyze + index); kinds validated.
   - `search_library(query, game="", kinds="", channels="", limit=20)` → hits as JSON.
   - `list_library(game="")` → sources.
   - `get_topic_material(game, max_terms=40)` → the material dict plus `instructions`: "Propose 5–10
     specific topic videos this evidence supports. Return them with record_topic_candidates."
   - `record_topic_candidates(game, candidates)` (JSON list of `{title, keywords}`) → evaluates each
     and returns `TopicCandidate`s (viable first, already-made flagged).
   - `suggest_topics(game, limit=10)` → `suggest_topics_local` (for clients that want the server's own Ollama to propose).
   Extend `SERVER_INSTRUCTIONS` with the topic workflow: add sources → (no-commentary: frame notes) →
   `get_topic_material` → `record_topic_candidates` → user picks → `start_topic_plan` (T4).
5. **CLI**: `topics suggest --game TEXT [--limit 10]` (rich table: title, evidence, sources,
   channels, already made); `topics ledger [--game]`.
6. **Config** (both copies): `topics: {min_evidence: 6, min_sources: 2, max_evidence: 12, similarity_threshold: 0.5}`.

## Tests (`tests/test_topics_discovery.py`, names `test_T3_*`, no network/Ollama)

Build a small library in a tmp SQLite from synthetic `PipelineState`s (2–3 sources, English +
Hinglish text, some frame notes). Cover: ledger record/similar/last_shape + atomic file; material
terms exclude stopwords/filler and include snippets; `evaluate_topic` viability thresholds and
distinct-source counting; already-made flagging; Ollama path with mocked client (valid JSON,
fenced JSON, garbage → fallback clusters); MCP tools: search/list, record_topic_candidates,
get_topic_material, library_add_source starts a job (patch `add_source`).

## Done when

- `pytest -q -p no:cacheprovider tests/test_topics_discovery.py tests/test_library.py tests/test_mcp_server.py` → 0 failed; config copies identical.
- Commit: `feat(topics): topic discovery, ledger and library MCP tools (T3)` + blank line + `Co-Authored-By: Antigravity <noreply@google.com>`.
- `docs/handoff/04-T3-topic-discovery.REPORT.md` committed (final hash in your chat reply).
