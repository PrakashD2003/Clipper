# Brief 05 (T4): topic plan — outline, script, clip selection, validation, approval

**Start after:** brief 04 is merged into `main` (the lead confirms). Use the latest `main`.
**Your branch:** `handoff/05-T4-topic-planner` · **Worktree:** `D:\GTA6-Clipper-wt\handoff-05`

Read `docs/handoff/README.md`, `docs/specs/topic-videos.md` (D1, D5, D8–D12, D15) and
`docs/specs/topic-videos-contracts.md` (models `EvidenceClip`, `TopicPoint`, `ShortPlan`,
`TopicPlan`, the plan rules, paths). The plan is what the user approves before any rendering.

```powershell
git -C D:\GTA6-Clipper fetch origin
git -C D:\GTA6-Clipper worktree add D:\GTA6-Clipper-wt\handoff-05 -b handoff/05-T4-topic-planner origin/main
```

## Files you may change

- `src/streamclipper/topics/models.py` — add the T4 models exactly as in the contracts
- `src/streamclipper/topics/planner.py`, `topics/validate.py` (new)
- `src/streamclipper/prompts/topic_planner.txt` (new)
- `src/streamclipper/mcp_server/server.py` — only the five plan tools + docstrings + `SERVER_INSTRUCTIONS` sentence
- `src/streamclipper/cli.py` — only `topics plan|show|validate|approve`
- both `config.default.yaml` copies — extend `topics:` (byte-identical)
- `tests/test_topics_planner.py` (new); `tests/test_mcp_server.py` only `EXPECTED_TOOLS`

## Do

1. **Plan storage** (`planner.py`): `plan_dir(config, game, topic_id)`, `save_plan(config, plan)`
   (atomic write of `plan.json`), `load_plan(config, plan_id)` (find by id across
   `<output_dir>/*/topics/*/plan.json`; keep a small id→path index in `<cache_dir>/topic_plans.json`).
2. **Shape** : `choose_shape(config, game)` rotates `numbered → countdown → story`, skipping the
   ledger's `last_shape(game)`.
3. **Material for writing** (`planner.py`): `planning_material(config, game, topic) -> dict`:
   evaluate the topic (T3 `evaluate_topic`), then for the best ~20 evidence spans include
   channel, kind, start/end, transcript text, chat sample, frame notes and heat; the chosen
   shape; resolved narration language (majority language of the evidence sources, overridable);
   `narration_guide(...)` from `narration_style`; the plan rules from the contracts (lengths,
   clip caps, narration ≥ 40 %, 5–7 points, Shorts slots) and the JSON schema of `TopicPlan`.
   Narration must sound like the energetic gamer friend from `narration_style` **but** for an
   explainer: each point's narration states the fact/claim in the first sentence, then hypes the
   evidence ("watch what Carry finds when he opens the vault") — still grounded, no invention.
4. **Validation** (`validate.py`): `validate_plan(config, plan) -> list[str]` implementing every
   rule in the contracts plus: chapter headings present for numbered/countdown (and
   countdown points ordered high→low index), story shape has 4–6 points with no `#` headings,
   exactly 3 `ShortPlan`s (`teaser`, `short_a`, `short_b`) whose `point_indexes` exist, teaser
   outro ends with a pointer to the full video (e.g. "Full breakdown on the channel" or Hinglish
   equivalent), standalone Shorts don't reference the long video, each Short estimated 35–60 s.
   Also fills `plan.estimated_seconds` and `plan.narration_seconds`.
5. **Local planner** (`planner.py`): `plan_topic_local(config, game, topic) -> TopicPlan` sends the
   material with `prompts/topic_planner.txt` to Ollama, parses JSON into `TopicPlan`, runs
   `validate_plan`, and retries up to 3 times feeding back the problems; keeps the attempt with
   the fewest problems, status `draft`, problems saved in the response.
6. **MCP tools** (names fixed in contracts):
   - `start_topic_plan(game, topic)` → `planning_material` + `instructions` ("write a TopicPlan JSON
     following the rules; submit it with submit_topic_plan") + a fresh `plan_id`.
   - `submit_topic_plan(plan_json)` → parse, normalize (fill channel/url/kind from the library for each clip),
     validate; on problems return them (status `draft`, saved so the client can fix with
     `update_topic_plan`); on success save and return the plan + a human-readable outline.
   - `get_topic_plan(plan_id)`; `update_topic_plan(plan_id, patch_json)` (JSON merge-patch on the
     plan; re-validate; never allowed once `approved`); `approve_topic_plan(plan_id)` → only when
     `validate_plan` is empty; sets `approved`, `approved_at`, records it in the ledger.
   Tool docstrings must say the **user approves** the plan: the client must show the outline,
   script and clip list to the user and call `approve_topic_plan` only after the user agrees.
7. **CLI**: `topics plan --game TEXT --topic TEXT` (local planner; prints outline + problems),
   `topics show PLAN_ID` (rich outline: shape, points with headings, narration, clips with
   channel and mm:ss, Shorts slots, estimated length, narration %), `topics validate PLAN_ID`
   (after hand-editing plan.json), `topics approve PLAN_ID`.
8. **Config**: add to `topics:` → `long_min_seconds: 480, long_max_seconds: 600, min_narration_ratio: 0.40, max_clip_seconds: 20, max_official_clip_seconds: 10, short_min_seconds: 35, short_max_seconds: 60`.

## Tests (`tests/test_topics_planner.py`, names `test_T4_*`, no network/Ollama)

Fixture library as in T3. Cover: model round-trips; shape rotation; every validation rule
(one failing plan per rule + one fully valid plan); estimate math; official 10 s cap;
short_a/short_b overlap detection; Ollama planner retry/keep-best with a mocked client; MCP
start → submit (problems) → update → submit OK → approve; approve refused while invalid;
update refused after approval; ledger records approved topics.

## Done when

- `pytest -q -p no:cacheprovider tests/test_topics_planner.py tests/test_topics_discovery.py tests/test_mcp_server.py` → 0 failed; config copies identical.
- Commit: `feat(topics): topic plans with validation, approval and MCP/CLI flow (T4)` + blank line + `Co-Authored-By: Antigravity <noreply@google.com>`.
- `docs/handoff/05-T4-topic-planner.REPORT.md` committed (final hash in your chat reply).
