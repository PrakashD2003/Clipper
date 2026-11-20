# Brief 04 (T3): Topic Discovery + Library MCP Tools Report

## Metadata
- **Branch:** `handoff/04-T3-topic-discovery`
- **Base Commit:** `a22215b` (`handoff/03-T2-frame-notes`, branched from T2 because T3 topic material and MCP tool workflow explicitly depend on T2 frame notes and brief 03 had not yet been merged into `main` at kickoff)
- **Commit Hash:** `82e73c0362bd438fc8d9a780df80aa67256b7363`
- **Timestamp:** 2026-11-20T15:20:00Z

## Pytest Verification Summary
```
88 passed, 217 warnings in 43.27s
```
Command executed:
`pytest -q -p no:cacheprovider tests/test_topics_discovery.py tests/test_library.py tests/test_mcp_server.py`
All 88 tests passed with 0 failures.

In addition, the T2 frame suite was re-verified:
`pytest -q -p no:cacheprovider tests/test_frames.py`
`14 passed in 9.22s` (0 failures).

Topic discovery suite isolated run:
`pytest -q -p no:cacheprovider tests/test_topics_discovery.py`
`14 passed in 7.00s` (0 failures).

## Byte-Identity Check
`config.default.yaml` and `src/streamclipper/config.default.yaml` are verified byte-identical via binary comparison (`fc.exe /b`).

## Files Changed
- `config.default.yaml`: Added `topics:` section (`min_evidence: 6`, `min_sources: 2`, `max_evidence: 12`, `similarity_threshold: 0.5`).
- `src/streamclipper/config.default.yaml`: Byte-identical copy of `config.default.yaml`.
- `src/streamclipper/topics/models.py`: Defined `EvidenceRef` and `TopicCandidate` models inheriting from `SerializableModel` with symmetric `from_dict` and `.is_viable` property.
- `src/streamclipper/topics/ledger.py`: Implemented `TopicLedger` managing `<cache_dir>/topics_ledger.json` with atomic writes, token Jaccard similarity checking, `record`, `entries`, `find_similar`, and `last_shape`.
- `src/streamclipper/prompts/topic_discovery.txt`: Ollama prompt template for curiosity-driven topic video proposal generation.
- `src/streamclipper/topics/discover.py`: Implemented `topic_material` (TF-IDF term and bigram extraction dropping EN/Hinglish stopwords and gaming filler, formatted snippets, stats), `evaluate_topic` (evidence deduplication, viability threshold, distinct source/channel counting, ledger similarity checking), `suggest_topics_local` (Ollama chat with code fence/think tag cleaning and fallback term clustering), and helper routines.
- `src/streamclipper/topics/__init__.py`: Package exports for `EvidenceRef`, `TopicCandidate`, `TopicLedger`, `topic_material`, `evaluate_topic`, and `suggest_topics_local`.
- `src/streamclipper/mcp_server/server.py`: Extended `SERVER_INSTRUCTIONS` with the topic workflow and added 6 new MCP tools: `library_add_source` (async background job), `search_library`, `list_library`, `get_topic_material`, `record_topic_candidates`, and `suggest_topics`.
- `src/streamclipper/cli.py`: Added `topics` command group with `topics suggest` and `topics ledger` subcommands rendering rich tables.
- `tests/test_mcp_server.py`: Extended `EXPECTED_TOOLS` with the 6 new MCP tools.
- `tests/test_topics_discovery.py`: New test suite covering ledger persistence/similarity, material term extraction/filtering/snippets, topic viability/source counting, ledger already-made flagging, mocked Ollama pathways (valid, fenced, garbage fallback), all 6 new MCP tools, and CLI commands.
- `docs/handoff/04-T3-topic-discovery.REPORT.md`: This report file.

## Key Decisions & Architectural Notes
1. **Base Commit Selection**: Brief 03 had not been merged into `main` before kickoff. Because T3 (`topic_material`, search merging, and workflow instructions) explicitly requires frame notes from T2, the branch was based on `a22215b` (`handoff/03-T2-frame-notes`). This ensures all 6 frame tools/schemas exist and merge cleanly once T2 lands on `main`.
2. **Stopwords and Filler Filtering**: Comprehensive stopword sets for English, Hinglish ("bhai", "yaar", "pe", "hai", etc.), and generic gaming talk ("chat", "guys", "lol", "look", "video", etc.) prevent generic conversational words from dominating salient topic terms in `topic_material`.
3. **Robust Prompt Template Placeholder Formatting**: Replaced `str.format()` with targeted `.replace("{key}", ...)` in `suggest_topics_local` and aligned `{total_hours}` in `topic_discovery.txt` with formatted float values.
4. **Resilient Multi-Format JSON Parsing**: `_extract_json` supports raw JSON strings, markdown code-fenced strings, conversational prefixes with bracketed JSON arrays/objects, direct list payloads, and dictionary-wrapped candidate lists (`{"candidates": [...]}` or `{"topics": [...]}`).
5. **Background Ingestion Tool**: `library_add_source` validates `kind` against allowed values (`"stream"`, `"gameplay"`, `"official"`) before registering a background job via `_start_render_job`, matching the pipeline's async job architecture.
6. **Multi-Source Evidence Diversity**: `evaluate_topic` prevents single-source evidence starvation by ensuring each source with matching hits gets evidence representation before filling `max_evidence`, correctly achieving `>= min_sources` when multiple sources have hits.
7. **SQL JOIN Ingestion & Parameter Safety**: `topic_material` joins `segments`/`frame_notes` with `sources` via SQL `JOIN sources src ON ... WHERE LOWER(src.game) = LOWER(?)` instead of large `IN (...)` parameter lists, scaling to large libraries without hitting SQLite parameter count limits.
8. **Co-Occurrence Fallback Clustering**: Salient terms are clustered using pairwise document co-occurrences so fallback topic candidates represent coherent scene topics.
9. **Ledger Freshness & Re-entrant Locking**: `TopicLedger` uses `threading.RLock()` and positions re-recorded entries at the latest position so `last_shape(game)` accurately reflects the most recent shape recorded.
10. **Configurable Viability**: `TopicCandidate` and `evaluate_topic` respect `topics.min_evidence` and `topics.min_sources` overrides from `config`.

## Undone Aspects
None. All requirements, contracts, MCP tools, CLI commands, edge cases, and tests are complete and passing.
