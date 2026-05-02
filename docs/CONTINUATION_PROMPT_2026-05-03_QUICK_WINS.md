# Continuation prompt — Quick-wins batch (T-50, T-32, T-14, T-29, T-30)

**Filed:** 2026-05-02 (post-T-19 implementation, awaiting T-19 PR)
**For:** the next agent picking up the small-bug-fix sweep
**Branch state at filing:** main currently at the latest commit on origin/main; `fix/T19-expose-model-native-imbalance` (4 commits, tip `9cea07c`) is awaiting user-opened PR. Do NOT open the T-19 PR yourself — the user opens PRs. Wait until the user has merged T-19 (or rebases the quick-wins branches off it post-merge) before starting work that touches `code_generator.py` or related export-code paths to avoid merge conflicts. Tickets in this prompt do NOT touch the export-code surface so they can branch off latest main now.

---

## You are starting fresh — read these BEFORE doing anything else

1. **`docs/PROJECT_STATUS.md` top section** for current state (what's working, what's deferred, what's queued). The "T-19 reframed — IMPLEMENTED" entry at the top is the most recent ground truth.
2. **`CLAUDE.md` Session Protocol** for mandatory session-end discipline (PROJECT_STATUS update + SESSION_LOG append + commit + push).
3. **Your auto-memory at `C:\Users\mspon\.claude\projects\C--Users-mspon-git-dasp\memory\MEMORY.md`.** Read every entry before deciding scope. Critical entries this round:
   - `feedback_chemometrics_relevance_per_ticket.md` — bug fixes are fine; pipeline methodology changes need user confirmation; pure polish defers.
   - `feedback_chemometrics_conventions.md` — SNV / SG derivatives / baseline / autoscale-pre-CV / variable-selection-on-full-training are NOT leakage. Reviewers who flag these generate false positives.
   - `feedback_deepseek_routing.md` — DeepSeek alias on opencode-call works; retry on pre-emptive refusals.
   - `feedback_glm_routing.md` (just updated) — GLM 5.1 via z.ai for solo calls; via opencode-call when paired with DeepSeek so they share repo access.
   - `feedback_parallel_review_reroute.md` (new) — when the user corrects routing for one reviewer mid-session, only re-dispatch the affected model; leave others running.
   - `feedback_tests.md` — targeted pytest, not full suite. Use the consolidated 11-file regression sweep as the gate.
   - `project_venv.md` — `.venv312/Scripts/python.exe`. `.venv311` is removed.

---

## Initial state

- **Branch tip:** main + 1 doc commit + 1 ticket-50 doc commit (filed during overnight session). Main is the right base for these tickets — none touch the T-19 surface.
- **Working tree:** clean except `.review-tmp/` (ignore — git-untracked review artifacts).
- **Test sweep state:** 282 passed + 1 skipped across the consolidated 11-file regression sweep (per T-19 baseline). Same 11 files: `test_t41_bayesian_sqlite_auto_calculator`, `test_t42_write_path_plumbing`, `test_t43_resume_auto_restore`, `test_run_state`, `test_run_logging`, `test_cv_strategy`, `test_unified_bayesian_baseline`, `test_autoscale_bayesian`, `test_cv_pls_clamp`, `test_ensemble_preprocessing`, `test_ensemble_integration`.
- **Pre-existing test flakes (deselect):** `tests/test_export_code.py::test_python_script_execution` and `test_full_workflow_python` — known environmental issue.
- **SESSION_LOG status:** active log (`docs/SESSION_LOG.md`) is over the 200-line archive threshold but already flagged for separate-ticket archive cut. Do NOT piggyback the archive on these PRs.

---

## The work — 5 quick-win tickets, in priority order

Critical framing rule: these are all **bug-fix-class** or trivial hygiene. Apply `feedback_chemometrics_relevance_per_ticket.md`. None should require user confirmation on methodology — but classify each explicitly in your message before starting, and **pause if your classification lands in pipeline-methodology territory**.

### 1. T-50 — Auto-cleanup old optuna SQLite files

**Plan doc:** `docs/plans/2026-05-02-T50-auto-cleanup-old-optuna-sqlite-files.md` (complete; ~30 LOC + 6 tests sketched).

**Classification:** infrastructure bug fix (silent disk leak, ~50 MB/month grows on user systems). Pipeline behavior unchanged.

**Scope:** new `cleanup_old_sqlite_files()` function in `run_state.py`; called from GUI `main()` and `cli.main()` startup (mirrors T-45 wiring). Policy: keep last 5 by mtime, delete anything >30 days, skip active run, skip files with fresh `*-wal` siblings. 6 regression tests in `tests/test_run_state.py`.

**Estimated:** 2-3 hours.

### 2. T-32 — `sample_weight` length-mismatch in resampling path

**Roadmap entry:** `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` § T-32. **No plan doc** — small enough to scope inline.

**Classification:** real bug fix. Pipeline behavior unchanged for users not using resampler + sample_weight together.

**Scope:** `search.py:3869-3883` — `sample_weight_train` is computed from pre-resampling `y_train` but passed to `final_model.fit(X_train_transformed, y_train, sample_weight=sample_weight_train)` against `y_train` (not `y_train_for_model`, the resampled y). When a resampler runs, lengths mismatch → silent wrong weights or array-length error. Fix: pass `y_train_for_model` and recompute `sample_weight_train` from it. Add a regression test that exercises the resampler + sample_weight path.

**Direct sibling of T-19** — same `sample_weight` thread, same chemometrics review brief applies. Cheap to leverage T-19 reviewer context.

**Estimated:** 1-3 hours.

### 3. T-14 — Hardcoded version-string mismatch

**Roadmap entry:** `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` § T-14. No plan doc; trivial.

**Classification:** hygiene bug. Three sites disagree:
- `__init__.py:3` says `0.5.0b1`
- `report.py:139` says `v0.4.0`
- `code_generator.py:373` says `3.9.0`

Single version constant should drive all three. Verify pyproject.toml's authoritative version and propagate.

**Estimated:** 30 min - 1 hour.

### 4. T-29 — Replace bare `except:` in scoring.py with NaN+warning

**Roadmap entry:** `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` § T-29. No plan doc.

**Classification:** silent-failure bug. Bare `except:` swallows everything including KeyboardInterrupt and SystemExit. Replace with specific exception types or `except Exception:` + `logger.warning(...)` + return NaN. Read the existing scoring.py paths to identify which exceptions are expected.

**Estimated:** 1-2 hours.

### 5. T-30 — Remove debug prints

**Roadmap entry:** `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` § T-30. No plan doc.

**Classification:** hygiene. **CAUTION** — there are legitimate user-facing `print()` calls (resampler progress at `imbalance.py:307-313`, fold metrics, CV results in exported scripts). Triage carefully. The roadmap's T-30 specifically targets debug prints left over from development; not all `print()` calls are debug. If unsure, ask the user before removing.

**Estimated:** 1-3 hours depending on how aggressive the cleanup is.

---

## Process rules

- **Use `.venv312/Scripts/python.exe`.** Project is Python 3.12 only.
- **Targeted pytest, not full suite.** For each ticket run the matching test file plus the consolidated 11-file regression sweep as the gate.
- **Branch per ticket: `fix/T<NN>-<short-name>`** off latest main. Each ticket gets its own branch and PR. Don't stack.
- **Comments default to none.** Only write a comment if WHY is non-obvious. NEVER include review-trail prefixes (`# T-NN fix:`, `# DeepSeek HIGH:`, etc.).
- **Black line-length 100, type hints with `from __future__ import annotations`, snake_case** per `.claude/rules/code-style.md`.
- **Don't add features beyond what the ticket requires.** No premature abstractions.
- **Don't add backwards-compat shims unless explicitly required.**
- **Open the PR yourself only if the user authorizes it.** Default: stop and ask after the branch is pushed and tests are green. The user opens PRs.

## Review protocol

Per ticket (after full implementation + tests passing on the branch):

1. **DeepSeek V4 Pro Max review** via `opencode-call` agent (model: `deepseek`). Read-only; brief should include the chemometrics-convention guardrails verbatim. Apply HIGH findings as fix-of-fixes commit; defer MEDIUM/LOW unless surgical.
2. **GLM 5.1 review in parallel** via `opencode-call` (NOT llm-call when DeepSeek is also reviewing — they need to share repo access; per `feedback_parallel_review_reroute.md` and `feedback_glm_routing.md` updated entries). For solo reviews, GLM via `llm-call` is fine.
3. Skip Codex review unless one of these tickets surfaces something complicated. The quick-wins are small enough that DeepSeek + GLM coverage is sufficient.

If `opencode-call`'s `deepseek` alias refuses on routing-policy grounds, retry with explicit instruction that the alias is known-working and to report the actual error verbatim. Do not accept pre-emptive refusals.

**Branch-state hygiene:** opencode-call agents leave the working tree on the branch they checked out. After each agent returns, run `git branch` to verify you're on the right branch before resuming edits. Lesson learned during T-19 — an MLP edit landed on `main` instead of the feature branch.

## Ticket-evaluation rule (mandatory)

Before starting each ticket, **explicitly classify** in your message to the user:

- **Bug fix** (silently broken, metadata wrong, code path doesn't do what it claims, edge case crashes) → ship after review.
- **Pipeline methodology change** (SNV / autoscale / SG / baseline / variable-selection refactored to address ML "leakage" objections) → STOP. Confirm with user. Default to NO.
- **Pure engineering polish** (test infrastructure for already-covered surfaces, defensive-only changes) → defer unless trivial.

T-50, T-32, T-14, T-29 are bug-fix-class. T-30 is hygiene-class with caveat (don't accidentally remove user-facing prints).

## Deliverables when you stop

1. List of branches pushed with tip commit hashes.
2. Per-ticket: tests passing count, review verdicts, deviations from plan.
3. Updated `PROJECT_STATUS.md` reflecting what shipped + what's queued.
4. Updated `SESSION_LOG.md` with any architectural traps surfaced.
5. If stopped on BLOCKER: a fresh `docs/CONTINUATION_PROMPT_<YYYY-MM-DD>.md` for the next agent.

## After this batch

Suggested next: **T-20** (saved-model ↔ exported-script reproducibility test) — pairs with T-19 freshness. Then start the bigger arcs (T-31 multi-class SIMCA, T-16 reframed model comparison, T-17 PLS-2). See `docs/PROJECT_STATUS.md` top section for the full open-ticket roster.

## Quality bar

Production-ready, but don't gold-plate beyond plan + HIGH review findings. Stick to scope. **Bug fixes / observability / behavioral defaults that materially affect what most users get** earn their slots. Engineering polish defers.
