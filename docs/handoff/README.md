# Handoff briefs for external agents (Antigravity / Gemini)

The lead session (Claude Code) writes one brief per task in this folder as
`docs/handoff/<NN>-<task>.md`. Paste the whole brief into Antigravity. Every brief
assumes these rules; they are repeated at the top of each brief.

## Rules every external agent must follow

1. **Never edit the main checkout `D:\GTA6-Clipper` directly.** Create your own worktree
   and branch first, from the base commit the brief names:
   ```powershell
   git -C D:\GTA6-Clipper worktree add D:\GTA6-Clipper-wt\<branch> -b <branch> <base-commit>
   ```
   Work only inside `D:\GTA6-Clipper-wt\<branch>`.
2. **Touch only the files the brief lists.** If you think another file must change, stop
   and say so in your report instead of editing it.
3. **Guardrails:** read `AGENTS.md` §4 before coding. New config keys go into BOTH
   `config.default.yaml` (repo root) and `src/streamclipper/config.default.yaml`, byte-identical.
   New parameters are keyword arguments with defaults. Dataclass fields get defaults and
   `.get(...)` in `from_dict`.
4. **Python and tests:** use `D:\GTA6-Clipper\.venv\Scripts\python.exe`. Point imports at
   your worktree and run one pytest process at a time (the machine is memory-limited):
   ```powershell
   $env:PYTHONPATH = "D:\GTA6-Clipper-wt\<branch>\src"
   $env:STREAMCLIPPER_NO_DOTENV = "1"
   D:\GTA6-Clipper\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider <test files>
   ```
   Check the import path first:
   `D:\GTA6-Clipper\.venv\Scripts\python.exe -c "import streamclipper; print(streamclipper.__file__)"`
   must print a path inside your worktree.
5. **Secrets:** never open, print, copy or commit `.env`, `cookies*.txt`, or `config.yaml`.
6. **Never** push, merge into `main`, rebase other branches, delete branches, or run
   `git stash` in the main checkout.
7. **Commit** on your branch with the message the brief gives, ending with your agent name.
8. **Report** by writing `docs/handoff/<NN>-<task>.REPORT.md` in your worktree (and include
   it in the commit) with: branch, commit hash, the exact pytest summary line(s), files
   changed (one line each), decisions you made, and anything left undone.

The lead session reviews the diff, runs the full suite, and merges.

## Queue (run in order; start each only after the lead confirms the previous merge)

| # | Brief | Builds |
| :--- | :--- | :--- |
| 01 | `01-ytdlp-auth-opts.md` | shared yt-dlp cookie/JS helper — merged |
| 02 | `02-T1-source-library.md` | source library + FTS5 index — merged |
| 03 | `03-T2-frame-notes.md` | frame contact sheets + frame notes |
| 04 | `04-T3-topic-discovery.md` | topic discovery + library MCP tools |
| 05 | `05-T4-topic-planner.md` | topic plans, validation, approval |
| 06 | `06-T5-longform-render.md` | long 16:9 renderer, chapters, thumbnail |
| 07 | `07-T6-topic-shorts.md` | topic Shorts + user guide |

Report hygiene: if you amend after writing the REPORT, the hash inside it goes stale — give the
final `git rev-parse HEAD` in your chat reply as well.
