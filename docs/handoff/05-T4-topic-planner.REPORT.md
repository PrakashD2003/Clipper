# Handoff Report: Brief 05 (T4) — Topic Plan: Outline, Script, Clip Selection, Validation, Approval

**Branch:** `handoff/05-T4-topic-planner`  
**Base commit:** `426e7e9` (feat(topics): topic discovery, ledger and library MCP tools (T3))  
**Date:** 2026-11-20  
**Model:** Antigravity (Gemini 3.8 high)

---

## 1. Test Summary Line

```
101 passed, 217 warnings in 45.45s
```
Command executed:
```powershell
$env:PYTHONPATH="D:\GTA6-Clipper-wt\handoff-05\src"; $env:STREAMCLIPPER_NO_DOTENV="1"; D:\GTA6-Clipper\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_topics_planner.py tests/test_topics_discovery.py tests/test_mcp_server.py
```
- `tests/test_topics_planner.py`: 33 passed
- `tests/test_topics_discovery.py`: 14 passed
- `tests/test_mcp_server.py`: 54 passed

Configuration consistency:
```powershell
fc.exe /b config.default.yaml src/streamclipper/config.default.yaml
# Result: FC: no differences encountered (100% byte-identical)
```

---

## 2. Files Changed

- `config.default.yaml`: extended `topics:` with long/short duration boundaries, narration ratio (0.40), and clip length caps.
- `src/streamclipper/config.default.yaml`: byte-identical extension of `topics:`.
- `src/streamclipper/topics/models.py`: added T4 models (`EvidenceClip`, `TopicPoint`, `ShortPlan`, `TopicPlan`) with complete `from_dict` / `to_dict` serialization and 0-value preservation.
- `src/streamclipper/topics/planner.py`: implemented plan storage (`save_plan`, `load_plan`, `plan_dir`), shape rotation (`choose_shape`), planning material assembly with overridable language (`planning_material`), RFC 7396 merge patch, clip normalizer, rich outline formatter with narration %, and local Ollama planner with retry feedback (`plan_topic_local`).
- `src/streamclipper/topics/validate.py`: implemented `validate_plan` verifying all duration math, shape rules, point counts, chapter headings, countdown descending order, official clip caps (10s), clip caps (20s), Shorts 35–60s bounds, non-overlap between short_a and short_b, teaser pointer checks, standalone video reference bans in body and hook_title, and active `check_narration` style/grounding on all narration fields.
- `src/streamclipper/prompts/topic_planner.txt`: system and user prompt template for the local Ollama explainer planner.
- `src/streamclipper/mcp_server/server.py`: registered 5 new MCP tools (`start_topic_plan`, `submit_topic_plan`, `get_topic_plan`, `update_topic_plan`, `approve_topic_plan`) with user-approval docstrings and updated `SERVER_INSTRUCTIONS` topic workflow sentence.
- `src/streamclipper/cli.py`: implemented CLI subcommands under `topics` (`plan`, `show`, `validate`, `approve`).
- `tests/test_mcp_server.py`: updated `EXPECTED_TOOLS` to include the 5 new topic plan tools (total 27 tools).
- `tests/test_topics_planner.py`: 33 unit tests covering serialization, shape rotation, all validation failure paths, sentence length limits, ungrounded rejection, estimate math, official 10s cap, short footage overlap detection, Ollama planner retry, MCP lifecycle, and CLI flows.
- `docs/handoff/05-T4-topic-planner.REPORT.md`: this handoff audit report.

---

## 3. Decisions & Architecture

1. **Worktree & Continuation from T3**: Branched `handoff/05-T4-topic-planner` from commit `426e7e9` (the HEAD of `handoff/04-T3-topic-discovery`), which contains all prior topic discovery and frame notes contracts.
2. **Narration Grounding & Style Enforcement**: Connected `check_narration` to point narration fields using `narration_intro` mapping, ensuring sentence length (max 14 words), banned phrases, and grounding against segment/clip transcript & chat are actively enforced and reported under point context.
3. **Library Clip Lookup by source_id or URL**: In `validate_plan` and `normalize_plan_clips`, lookup matches `WHERE source_id = ? OR url = ?`, allowing draft plans referencing original source URLs to be normalized to database source IDs seamlessly.
4. **Atomic Plan Storage with Cache Index**: `save_plan` atomically writes `<output_dir>/<game>/topics/<topic_id>/plan.json` and updates `<cache_dir>/topic_plans.json`. `load_plan` queries the index first, falling back to globbing `<output_dir>/*/topics/*/plan.json` if the index is cleared.
5. **Language Resolution & Overrides**: `planning_material` reuses `streamclipper.language.resolve_narration_language` and accepts an explicit `language` parameter override, falling back to majority evidence source language or config settings.
6. **Teaser & Standalone Shorts Guardrails**: Enforced strict validation preventing standalone shorts from mentioning long videos/breakdowns in body or hook_title and requiring teaser outros to contain pointers (e.g. "Full breakdown on the channel!"). Also validated that short_a and short_b share no overlapping source footage intervals.
7. **Estimation Math**: Accurately computes narration time (`narration_words / 2.7`), chapter card duration (2.5s per point for numbered/countdown shapes, 0s for story), and evidence clip durations, ensuring minimum 40% narration ratio and 480–600s total long-form duration. Outline output displays duration and narration % clearly.
8. **Scope Guardrails Compliance**: Reverted `src/streamclipper/topics/__init__.py` to base version so only files explicitly listed in the brief's "Files you may change" are modified.

---

## 4. Left Undone

None. All T4 requirements, validation rules, MCP tools, CLI commands, and test suites are complete and passing.
