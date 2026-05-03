# Continuation prompt — T-20 + quick-wins-batch follow-ups (T-29b, T-14b, T-30b)

**Filed:** 2026-05-03 (at the close of the quick-wins batch session — PRs #16-#20 all merged to main at `1d3eba0`)
**For:** the next agent picking up where the quick-wins batch left off
**Branch state at filing:** `main` at `1247389` (or later — pull origin/main first). Working tree should be clean. All 5 quick-wins branches deleted from origin + locally.

---

## You are starting fresh — read these BEFORE doing anything else

1. **`docs/PROJECT_STATUS.md` top section** for current state. The quick-wins batch entry at the top is the most recent ground truth. Note the deferred-follow-up callouts under T-29 / T-14 / T-30 / T-32 / T-50 — those map directly to the tickets in this prompt.
2. **`CLAUDE.md` Session Protocol** for mandatory session-end discipline (PROJECT_STATUS update + SESSION_LOG append + commit + push).
3. **Your auto-memory at `C:\Users\mspon\.claude\projects\C--Users-mspon-git-dasp\memory\MEMORY.md`.** Read every entry before deciding scope. Critical entries this round, with the newest at the bottom:
   - `feedback_chemometrics_relevance_per_ticket.md` — bug fixes are fine; pipeline methodology changes need user confirmation; pure polish defers.
   - `feedback_chemometrics_conventions.md` — SNV / SG derivatives / baseline / autoscale-pre-CV / variable-selection-on-full-training are NOT leakage by chemometrics convention. Reviewers who flag these generate false positives.
   - `feedback_deepseek_routing.md` — DeepSeek alias on opencode-call works; retry on pre-emptive refusals.
   - `feedback_glm_routing.md` — GLM 5.1 via z.ai for solo calls; via opencode-call when paired with DeepSeek so they share repo access.
   - `feedback_parallel_review_reroute.md` — when the user corrects routing for one reviewer mid-session, only re-dispatch the affected model.
   - `feedback_tests.md` — targeted pytest, not full suite. Use the consolidated 13-file regression sweep as the gate (now includes `test_t32_sample_weight_resampling.py` and `test_scoring.py`).
   - `project_venv.md` — `.venv312/Scripts/python.exe`. `.venv311` is removed.
   - **`feedback_commit_before_yielding_control.md` (NEW from prior session) — long pytest / agent dispatch / `git push` can collide with parallel sessions or hooks. Commit ASAP after implementation, before any operation that yields control. T-50 lost ~30 min to this in the prior batch.**

4. **`docs/SESSION_LOG.md` top entry (2026-05-03)** for three non-obvious lessons earned in the prior batch:
   - `pytest | tail` silently masks pytest's non-zero exit code (T-29 burned this — fix-of-fixes commit landed despite a failing test).
   - Cross-family review (DeepSeek + GLM via opencode-call) earned its slot — **5/5 tickets in the prior batch had at least one HIGH/MEDIUM finding I missed, despite confidence the fixes were complete**. Don't skip review on non-trivial surfaces.
   - DeepSeek had 1 false positive in the prior batch (T-50 `bytes_freed` ordering claim). Verify findings against actual code before applying.

---

## Initial state

- **Branch tip:** `main` post-quick-wins-batch merge. All 5 prior PR branches (`fix/T50-cleanup-stale-optuna-sqlite`, `fix/T14-version-string-single-source`, `fix/T29-scoring-no-bare-except`, `fix/T32-sample-weight-resampling-mismatch`, `fix/T30-remove-debug-prints`) deleted from origin + locally.
- **Working tree:** should be clean except `.review-tmp/` (ignore — git-untracked review artifacts).
- **Test sweep state:** **327 passed + 1 skipped** across the consolidated 13-file regression sweep:
  ```
  test_t41_bayesian_sqlite_auto_calculator test_t42_write_path_plumbing
  test_t43_resume_auto_restore             test_run_state
  test_run_logging                         test_cv_strategy
  test_unified_bayesian_baseline           test_autoscale_bayesian
  test_cv_pls_clamp                        test_ensemble_preprocessing
  test_ensemble_integration                test_t32_sample_weight_resampling
  test_scoring
  ```
- **Pre-existing test flakes (deselect):** `tests/test_export_code.py::test_python_script_execution` and `test_full_workflow_python` — known environmental issue.
- **SESSION_LOG status:** active log is now ~325 lines (over the 200-line archive threshold). DO NOT piggyback the archive cut on a fix PR — file as a separate doc-only commit if you trim it.

---

## The work — 4 tickets, in priority order

Critical framing rule: classify each ticket as **bug-fix** vs **pipeline-methodology change** vs **pure polish** before implementing. Apply `feedback_chemometrics_relevance_per_ticket.md`. **Pause and confirm with the user if your classification lands in pipeline-methodology territory.**

### 1. T-20 — Saved-model ↔ exported-script reproducibility test (PRIORITY)

**Status:** queued; pairs naturally with T-19 (now merged at `1d3eba0`) which closed the export-code class_weight bugs.
**Classification:** test infrastructure for already-shipped surfaces — borderline polish vs. bug-prevention. **Genuinely useful** because T-19 + T-32 both fixed bugs in the export/runtime parity surface; without a reproducibility test pinning the parity contract, the next regression silently slips.

**Scope:**
- New test file (e.g., `tests/test_t20_saved_model_export_parity.py`) that:
  - Trains a model end-to-end (PLS, Ridge, RandomForest, XGBoost — at minimum the boosting models because of the T-19 sample_weight surface).
  - Saves it via `model_io.save_dasp_model()`.
  - Generates the equivalent exported Python script via `code_generator.CodeGenerator.generate_script()`.
  - Loads the saved model AND `exec()`s the generated script on the same data.
  - Asserts predictions match within numerical tolerance.
- Cover: regression + binary classification + multi-class classification. With and without imbalance handling (T-19 paths). With `imbalance_method='auto'` (T-19 Auto mode runtime resolution).
- The test_export_code.py:test_python_script_execution and test_full_workflow_python flakes are environmental — if they're related to your test surface, document the flake separately, don't try to fix it here.

**Estimated:** 2-3 days. Most of the work is fixture + harness, not the assertions themselves.

**Branch:** `fix/T20-saved-model-export-parity-test`

### 2. T-29b — Per-run-log visibility for `spectral_predict.*` warnings (architectural)

**Status:** deferred from T-29 fix-of-fixes (DeepSeek HIGH-2 finding). NOT folded into T-29 because it's broader than T-29 alone — affects ALL `spectral_predict.*` module warnings (T-50 cleanup, T-46 WAL rejection, T-45 sidecar corruption, T-29 metric failures).
**Classification:** observability bug fix. Pipeline behavior unchanged.

**Scope:**
- Current state: `setup_run_logger()` (in `run_logging.py`) creates a handler on `dasp.run` with `propagate=False`. Module warnings under `spectral_predict.*` propagate to the `spectral_predict` handler (which `setup_app_logger` installed) → land in `<user_data_dir>/dasp.log`. **But** they do NOT land in the per-run log (`run_TIMESTAMP.log`) the GUI's "View log" button surfaces.
- Result: a user investigating a strange leaderboard checks the per-run log → finds nothing. The actual warning is in `dasp.log`, which is the rotating app-lifetime log, not the per-run forensic log.
- Fix: extend `setup_run_logger` so that during a run, WARNING-level messages from any `spectral_predict.*` child logger also land in the per-run file. Two viable approaches:
  - (a) Add a second handler attached to the `spectral_predict` logger with the per-run file path; remove it on `tear_down_run_logger`. Cleanest.
  - (b) Set `propagate=True` on `dasp.run` so child warnings reach the run's handler. Riskier — could double-log if `dasp.run` has its own children.
- Pick (a). Test that the warning lands in BOTH `dasp.log` AND `run_TIMESTAMP.log` during a run, and only in `dasp.log` outside a run.

**Estimated:** 3-5 hours including tests. Cross-family review WORTH IT (architectural change — the surface area is small but the failure mode is silent visibility loss across ALL future warnings).

**Branch:** `fix/T29b-per-run-log-visibility-for-module-warnings`

### 3. T-14b — GUI title bar version drift

**Status:** deferred from T-14 fix-of-fixes (DeepSeek LOW). Same bug class as T-14's 5 closed sites.
**Classification:** hygiene bug. Same class as T-14, just out of T-14's PR scope.

**Scope:** `spectral_predict_gui_optimized.py:2556` and `:4663` hardcode `0.5.0b1` in the GUI title bar. Replace with `from spectral_predict import __version__` reference (or import-aware string). Add a regression test (or extend an existing GUI-string test) that pins absence of stale literals.

**Estimated:** 30 min - 1 hour. Skip cross-family review — too small.

**Branch:** `fix/T14b-gui-title-bar-version-drift`

### 4. T-30b — `calibration_transfer.py` + `nsga2_search.py` print() triage

**Status:** deferred from T-30 fix-of-fixes. Both T-30 reviewers (DeepSeek + GLM) flagged these as same-class follow-ups requiring per-file user judgment. **Bigger triage surface than T-30 itself** — 66 prints in calibration_transfer.py and 42 in nsga2_search.py.
**Classification:** hygiene with caveat. **CAUTION** — there are legitimate user-facing print() calls in both files (DS/PDS calibration progress; NSGA-II generation summaries). **Triage carefully.** If unsure on any specific print, default to KEEPING it and ask the user.

**Scope:**
- Read both files end-to-end. Per print():
  - If it's `print(f"[DEBUG]...")` or has an adjacent `# DEBUG:` comment → **DELETE**.
  - If it's a per-iteration / per-fold spam print with no clear user value → **CONVERT TO `logger.debug(...)`**.
  - If it's a user-facing progress message ("Computing DS transfer matrix...", "NSGA-II generation 5/20...", etc.) → **KEEP**.
- The `logger.debug` conversion path (vs delete) preserves the diagnostic value while suppressing it in normal use — gated by Python logging level.
- File a follow-up ticket if either file's print volume after triage is still uncomfortably high — don't gold-plate a single PR with another sweep ticket.

**Estimated:** 2-4 hours including triage time. Cross-family review WORTH IT for the two-file surface (~110 prints reviewed) — both reviewers caught a missed block in T-30, so a fresh pair of eyes will likely catch missed user-facing-vs-debug calls.

**Branch:** `fix/T30b-calibration-transfer-and-nsga2-print-triage`

---

## Process rules

- **Use `.venv312/Scripts/python.exe`.** Project is Python 3.12 only.
- **Targeted pytest, not full suite.** For each ticket run the matching test file plus the consolidated 13-file regression sweep as the gate.
- **Branch per ticket: `fix/T<NN>[-<short-name>]`** off latest main. Each ticket gets its own branch and PR. Don't stack.
- **Commit ASAP after implementation, before any operation that yields control.** Long pytest, agent dispatch, `git push`, anything > a few seconds. Memory `feedback_commit_before_yielding_control.md`.
- **Don't use `pytest | tail` in chain-commit patterns.** The pipe inherits `tail`'s exit code (always 0), so a failing test silently propagates to `git commit && git push`. Use `2>&1 | grep -E "passed|failed"` if you need length-limited output (grep returns 0 on match — preserves "passed" success vs "failed" failure correctly). Or split the steps.
- **Comments default to none.** Only write a comment if WHY is non-obvious. NEVER include review-trail prefixes (`# T-NN fix:`, `# DeepSeek HIGH:`, etc.) in production code — those belong in commit messages and PR descriptions. (Exception: complex fix-of-fixes commits citing prior reviewers in code comments is acceptable when the reasoning is non-obvious from context — e.g., `feedback_commit_before_yielding_control.md` lessons.)
- **Black line-length 100, type hints with `from __future__ import annotations`, snake_case** per `.claude/rules/code-style.md`.
- **Don't add features beyond what the ticket requires.** No premature abstractions. Bug fixes that preserve pipeline behavior are fine; pipeline methodology changes need user confirmation.
- **Don't add backwards-compat shims unless explicitly required.**
- **Open the PR yourself only if the user authorizes it.** Default: stop and ask after the branch is pushed and tests are green.

## Review protocol

Cross-family review pattern proved its value in the prior batch — **5/5 quick-win tickets had at least one HIGH/MEDIUM finding the reviewers caught**. Recommended dispatch policy:

- **T-20** (test infra, ~150-300 LOC): cross-family review WORTH IT. Test-infrastructure parity is exactly the kind of surface where reviewers spot missed coverage paths.
- **T-29b** (architectural, ~50-100 LOC): cross-family review WORTH IT. Logger-handler architecture changes have non-obvious failure modes (handler propagation, double-attach, dedup) — both reviewers will likely catch different things.
- **T-14b** (hygiene, ~5-10 LOC): SKIP review. Trivial mechanical change. Test will catch any regression.
- **T-30b** (triage, ~50-100 LOC of deletions + conversions): cross-family review WORTH IT. The whole point is a second pair of eyes on triage decisions.

Per ticket where review IS dispatched (after full implementation + tests passing on the branch):

1. **DeepSeek V4 Pro Max review** via `opencode-call` agent (model: `deepseek`). Read-only; brief should include the chemometrics-convention guardrails verbatim. Apply HIGH findings as fix-of-fixes commit; defer MEDIUM/LOW unless surgical.
2. **GLM 5.1 review in parallel** via `opencode-call` (NOT llm-call when DeepSeek is also reviewing — they need to share repo access; per `feedback_glm_routing.md` and `feedback_parallel_review_reroute.md`).
3. **For the bigger or more architectural tickets (T-20, T-29b)**: consider a final consolidated GLM-only sanity-recheck of the fix-of-fixes commit (single dispatch, all ticket fix-of-fixes if you batch them — the prior batch did this for 4 fix-of-fixes commits and it caught zero issues). Optional but cheap reassurance.
4. Skip Codex review unless one of these tickets surfaces something complicated. The 4-ticket batch is small enough that DeepSeek + GLM coverage is sufficient.

If `opencode-call`'s `deepseek` alias refuses on routing-policy grounds, retry with explicit instruction that the alias is known-working and to report the actual error verbatim. Do not accept pre-emptive refusals.

**Verify reviewer findings against actual code before applying.** DeepSeek had 1 false positive in the prior batch (T-50 `bytes_freed` ordering — claimed `bytes_freed += sibling_size` ran before `sibling.unlink()`; actually ran AFTER, so OSError correctly skipped via except). Cross-family review is high-leverage but not infallible.

**Branch-state hygiene:** opencode-call agents may end on a different branch than they checked out (or never check out at all — they often use `git show <sha>:<path>` for read-only review, leaving the worktree on whichever branch was active at dispatch time). After each agent returns, run `git branch` to verify you're on the right branch before resuming edits.

## Ticket-evaluation rule (mandatory)

Before starting each ticket, **explicitly classify** in your message to the user:

- **Bug fix** (silently broken, metadata wrong, code path doesn't do what it claims, edge case crashes) → ship after review.
- **Pipeline methodology change** (SNV / autoscale / SG / baseline / variable-selection refactored to address ML "leakage" objections) → STOP. Confirm with user. Default to NO.
- **Pure engineering polish** (test infrastructure for already-covered surfaces, defensive-only changes) → defer unless trivial.

T-20 is **test-infrastructure-borderline-polish** but justified because T-19/T-32 just fixed bugs in the export/runtime parity surface — without a parity test the next regression silently slips. Worth shipping.

T-29b, T-14b, T-30b are bug-fix-class or hygiene-class. None should require user confirmation on methodology — but classify each explicitly in your message before starting, and **pause if your classification lands in pipeline-methodology territory**.

## Deliverables when you stop

1. List of branches pushed with tip commit hashes.
2. Per-ticket: tests passing count, review verdicts (if dispatched), deviations from plan.
3. Updated `PROJECT_STATUS.md` reflecting what shipped + what's queued.
4. Updated `SESSION_LOG.md` with any architectural traps surfaced.
5. If stopped on BLOCKER: a fresh `docs/CONTINUATION_PROMPT_<YYYY-MM-DD>.md` for the next agent.
6. If `SESSION_LOG.md` exceeds ~500 lines, file a separate archive-cut PR moving older entries to `docs/SESSION_LOG_ARCHIVE.md` (currently ~325 lines — borderline).

## After this batch

The bigger arcs queued in `PROJECT_STATUS.md`:
- **T-31** multi-class SIMCA (1-2w) — confirmed real per `project_t31_simca_confirmed.md`. Implement as true multi-class class-modeling per Oliveri & Downey 2012, NOT as one-class extension of `PCASIMCA`.
- **T-16 reframed** — competitive model-comparison machinery survey (what Unscrambler/SIMCA/OPUS/PLS_Toolbox expose), then scope.
- **T-17** PLS-2 (2-3w) — multi-target PLS.
- **T-01 reframed** (~2-3d).

Plus the deferred follow-ups from the quick-wins batch that DON'T appear in this prompt:
- **T-32b** — regression-side imbalance latent bug (DeepSeek LOW). Pre-existing, abnormal combination — file as separate ticket if a user hits it.
- **T-29 dead-code investigation** — `compute_imbalance_metrics` has zero callers in the repo (verified via grep). Either wire it into `search.py` somewhere it's missing OR deprecate. Cross-family confirmed in the prior batch; not urgent.
- **T-50b** — configurable `_KEEP_LAST_N_RUNS` / `_DELETE_AFTER_DAYS` (currently module-level constants). Both reviewers acked the trade-off; gold-plating unless a user asks.

## Quality bar

Production-ready, but don't gold-plate beyond plan + HIGH review findings. Stick to scope. **Bug fixes / observability / behavioral defaults that materially affect what most users get** earn their slots. Engineering polish defers.
