# Continuation prompt — overnight run after the 2026-05-04 five-PR session

**Filed:** 2026-05-04 (at the close of the T-14b / T-20 / T-20b / codegen-fix / T-29b session — five PRs in flight, none yet merged)
**For:** the next agent picking up overnight, autonomously, while the user is asleep
**Branch state at filing:** five branches pushed to origin, plus `main` at `1247389` (or later — the user may have merged some PRs by the time you wake up).

---

## First command on wake-up — get to a clean baseline

**Run these in order:**

```
git fetch --all --prune
git checkout main
git pull origin main
git status
git branch -a
```

Then check whether the user merged any of the five PRs while you were asleep:

```
gh pr list --state merged --search "merged:>=2026-05-04" --json number,title,mergedAt
gh pr list --state open --json number,title,headRefName,baseRefName
```

**You should always START every new ticket from a fresh `main` branch.** The five session-2026-05-04 branches (`fix/T14b-...`, `fix/T20-...`, `fix/T20b-...`, `fix/Tcv-...`, `fix/T29b-...`) are NOT bases for new work — they're waiting for the user to merge them. New tickets go on fresh `fix/<ticket-name>` branches off whatever `main` looks like when you start.

**If you need to apply fix-of-fixes to one of the existing PRs** (e.g., closing review findings on T-29b), check out THAT PR's branch:

```
git checkout fix/T29b-per-run-log-visibility-spectral-predict-warnings
git pull origin fix/T29b-per-run-log-visibility-spectral-predict-warnings
```

Then commit + push back. Don't rebase those branches unless their parent merged (see "Branch parentage and rebasing" below).

---

## Read these BEFORE doing anything else

1. **`docs/PROJECT_STATUS.md` top section.** The 2026-05-04 entry has the exact tip SHA and review state of every PR. **Critical**: check whether the user has merged any of the 5 PRs while you were asleep. If yes, the chain rebases need different handling (see "Branch parentage and rebasing" below).
2. **`CLAUDE.md` Session Protocol** — same as always. Mandatory PROJECT_STATUS.md update, SESSION_LOG.md append for non-obvious lessons, commit-and-push at session end.
3. **Auto-memory at `C:\Users\mspon\.claude\projects\C--Users-mspon-git-dasp\memory\MEMORY.md`.** Critical entries this round, with the newest at the bottom:
   - `feedback_commit_before_yielding_control.md` — commit ASAP before any operation that yields control.
   - `feedback_chemometrics_relevance_per_ticket.md` — bug fixes fine; pipeline methodology changes need user confirmation; engineering polish defers.
   - `feedback_chemometrics_conventions.md` — SNV / SG derivatives / autoscale-pre-CV / variable-selection-on-full-training are NOT leakage by chemometrics convention. Reviewers who flag these generate false positives.
   - `feedback_deepseek_routing.md` — DeepSeek alias on opencode-call works; retry on routing-policy refusals.
   - `feedback_glm_routing.md` — GLM 5.1 via z.ai for solo calls; via opencode-call when paired with DeepSeek so they share repo access.
   - `feedback_parallel_review_reroute.md` — when the user corrects routing for one reviewer mid-session, only re-dispatch the affected model.
   - `feedback_tests.md` — targeted pytest, not full suite. Use the consolidated 13-file regression sweep as the gate.
   - `project_venv.md` — `.venv312/Scripts/python.exe`. `.venv311` is removed.
4. **`docs/SESSION_LOG.md` top entry (2026-05-04)** — five non-obvious lessons earned this session:
   - CatBoost multiclass `predict()` returns shape `(n, 1)` not `(n,)` — caught by T-20b's parity test, fixed in PR #24.
   - `predict()` vs `predict_proba()` matters: hard labels mask probability drift up to 0.296 (DeepSeek empirical).
   - DEFAULT_PARAMS merge pattern in `code_generator._render_model` is the parity contract — in-process tests must mirror runtime's param assembly.
   - XGBoost has no `class_weight` constructor kwarg; codebase uses model-native imbalance handling per library, with uniform user-facing interface.
   - Stacked PR pattern with chained bases kept review cost linear.

---

## Initial state (at filing)

**Five PRs pushed to origin, all open:**

| PR | Branch | Tip SHA | Base | Reviews | Status |
|----|--------|---------|------|---------|--------|
| #21 | `fix/T14b-gui-title-bar-version-drift` | `1ef393f` | `main` | DeepSeek + GLM cross-family complete | CLEAN, awaiting merge |
| #22 | `fix/T20-saved-model-export-parity-test` | `ec7f8a7` | `main` | DeepSeek + GLM + pr-review-toolkit suite + GLM sanity-recheck | CLEAN, awaiting merge |
| #23 | `fix/T20b-saved-model-export-parity-additional-models` | `a12e843` | `fix/T20-saved-model-export-parity-test` | DeepSeek + GLM cross-family complete | CLEAN, awaiting merge |
| #24 | `fix/Tcv-catboost-multiclass-cv-pooling-scalarise` | `685553e` | `fix/T20b-saved-model-export-parity-additional-models` | DeepSeek + GLM cross-family complete | CLEAN, awaiting merge |
| #25 | `fix/T29b-per-run-log-visibility-spectral-predict-warnings` | `3b8b748` | `main` | DeepSeek + GLM cross-family **dispatched in background at session close** — agent IDs `af4b5eedbde4c5b37` (DeepSeek) and `a4d710be76004ca03` (GLM). They may have completed by the time you wake up; check `~/AppData/Local/Temp/claude/.../tasks/<agentId>.output` for transcripts, or just dispatch fresh reviews. | Awaiting reviews |

**Working tree at session close:** clean on `main` (after the doc commits below). Untracked artifacts to ignore: `.review-tmp/` and the weirdly-named `C\357\200\272UsersmsponAppDataLocalTempopencodesearch_t30.py` left over from a previous session.

**Test sweep state at filing:** consolidated 13-file regression sweep + each PR's branch tested green.

---

## Priority order

### Priority 1 — finish T-29b review and ship if clean

**Why first:** T-29b (PR #25) was the only PR without completed cross-family review at session close. The reviews were dispatched in background with agent IDs above. By the time you wake up they're either complete (transcripts in `~/AppData/Local/Temp/claude/.../tasks/<agentId>.output`) or ready to be re-dispatched.

**Steps:**

1. Check the dispatched-review transcripts. If clean (no HIGH findings), proceed to step 3.
2. If HIGH findings exist, apply as a fix-of-fixes commit on `fix/T29b-per-run-log-visibility-spectral-predict-warnings`, push, and re-dispatch a single GLM solo recheck.
3. Mark the PR ready for the user to merge — do NOT merge yourself unless explicitly instructed.

If reviews never returned (background agents failed), dispatch fresh DeepSeek + GLM cross-family review in parallel via `opencode-call`. The brief from the original session is in `Agent` invocations early in this session's history; you can reconstruct it from the diff and the PR description.

**Estimated:** 30 min - 1 hour.

### Priority 2 — T-30b (calibration_transfer + nsga2_search print() triage)

**Why second:** Continuation prompt's queued ticket. Hygiene class with caveat — both files have legitimate user-facing print() calls (DS/PDS calibration progress, NSGA-II generation summaries). When in doubt, KEEP. Do NOT delete prints that look like user-facing progress messages.

**Branch:** `fix/T30b-calibration-transfer-and-nsga2-print-triage` off latest `main`.

**Scope** (per the original 2026-05-03 continuation prompt):

- Read `src/spectral_predict/calibration_transfer.py` end-to-end (66 prints surveyed at filing time).
- Read `src/spectral_predict/nsga2_search.py` end-to-end (42 prints).
- For each print():
  - `print(f"[DEBUG]...")` or has adjacent `# DEBUG:` comment → **DELETE**.
  - Per-iteration / per-fold spam print with no clear user value → **CONVERT TO `logger.debug(...)`**.
  - User-facing progress message ("Computing DS transfer matrix...", "NSGA-II generation 5/20...") → **KEEP**.
- Cross-family review WORTH IT for the two-file surface (~110 prints reviewed) — both reviewers caught a missed block in T-30, so a fresh pair of eyes will likely catch missed user-facing-vs-debug calls.

**Estimated:** 2-4 hours including triage time + cross-family review + fix-of-fixes.

**Important:** the user explicitly said *"defer engineering polish unless trivial."* T-30b is hygiene-class but **the user has already approved the scope** (it's in the original continuation prompt). Just exercise judgment per print — when in doubt, keep.

### Priority 3 — Ridge classifier export investigation (small ticket)

**Why third:** Pre-existing codegen bug surfaced during T-20b. `code_generator._resolve_model_ctor_class` returns plain `'Ridge'` (sklearn regressor) for `task_type='classification'` task instead of `'RidgeClassifier'`. The runtime in `models.py:472-473` correctly uses `RidgeClassifier`, but the export path falls through. Either Ridge classification has never been exercised through the export surface (likely), or this is an unreported bug.

**Scope:**

1. Trace whether Ridge classification is reachable from the GUI search flow at all. If yes, the user has been exporting broken scripts for Ridge classification — probably a HIGH bug.
2. If reachable, fix `_resolve_model_ctor_class` to return `'RidgeClassifier'` for classification (mirroring what's already done for RandomForest, LightGBM, etc.). Add `'RidgeClassifier'` to `templates/models.MODEL_IMPORTS` and `DEFAULT_PARAMS` if not present.
3. Add a parity row to `tests/test_t20_saved_model_export_parity.py` covering Ridge classification.
4. If NOT reachable from GUI, file as a no-op cleanup ticket.

**Branch:** `fix/Tridge-classifier-export-resolution` off latest `main`.

**Estimated:** 1-3 hours depending on reachability. Cross-family review WORTH IT if you change the codegen.

### Priority 4 — defensive `.ravel()` on per-fold metrics (GLM MEDIUM from PR #24, deferred)

**Why fourth:** GLM raised a hygiene MEDIUM during the codegen-fix review: `templates/validation.py:167-168`'s `accuracy_score`/`f1_score` calls pass raw `y_pred_fold` which is `(n, 1)` for CatBoost multiclass. Today sklearn auto-broadcasts; future shape changes could silently produce wrong answers. Mirror what the regression template already does at line 103 (`fold_model.predict(X_test).ravel()`).

**Scope:** one line in `templates/validation.py:160` — change `y_pred_fold = fold_model.predict(X_test)` to `y_pred_fold = fold_model.predict(X_test).ravel()`. Run the consolidated sweep + T-20 file to confirm no regressions.

**Branch:** `fix/Tval-classification-cv-fold-ravel` off latest `main`.

**Estimated:** 30 min including review. Cross-family review optional (one-line hygiene change, low risk).

### Priority 5 (only if all above are done) — PLS-DA parity row for T-20

**Why last:** T-20b skipped PLS-DA because the codegen path (`_render_pls_da_pipeline`) emits a Pipeline of PLS scores → StandardScaler → LogisticRegression that needs careful in-process construction. Non-trivial scaffolding work.

**Scope:**

1. Read `code_generator._render_pls_da_pipeline` to understand exactly what Pipeline shape the script constructs.
2. Build the same Pipeline in-process for the test using sklearn primitives (PLSRegression as transformer + StandardScaler + LogisticRegression).
3. Add a parity row to `tests/test_t20_saved_model_export_parity.py`.

**Branch:** `fix/T20c-pls-da-parity-row` off latest `main`.

**Estimated:** 2-3 hours. Cross-family review WORTH IT given the Pipeline complexity.

---

## Branch parentage and rebasing

If the user merges one or more of the stacked PRs (#22, #23, #24) while you sleep:

- **#22 merged → main:** rebase `fix/T20b-saved-model-export-parity-additional-models` onto `origin/main`. Force-push (`git push --force-with-lease`). PR #23's base auto-updates from GitHub's perspective.
- **#23 merged → main:** rebase `fix/Tcv-catboost-multiclass-cv-pooling-scalarise` onto `origin/main`. Force-push. PR #24's base auto-updates.
- **#24 merged:** chain done.

If the user ALREADY merged #22 + #23 + #24 by the time you wake up, all stacked branches are gone — the T-20 family is fully on main. Nothing to rebase.

`#21` (T-14b) and `#25` (T-29b) are independent of the chain — no rebase needed regardless.

**Always pull `origin/main` before starting any new work**: `git checkout main && git pull origin main`.

---

## Process rules (mandatory)

- **Use `.venv312/Scripts/python.exe`.** Project is Python 3.12 only. `.venv311` is removed.
- **Targeted pytest, not full suite.** For each ticket: matching test file + the consolidated 13-file regression sweep as the gate. Files in the sweep:
  ```
  test_t41_bayesian_sqlite_auto_calculator    test_t42_write_path_plumbing
  test_t43_resume_auto_restore                test_run_state
  test_run_logging                            test_cv_strategy
  test_unified_bayesian_baseline              test_autoscale_bayesian
  test_cv_pls_clamp                           test_ensemble_preprocessing
  test_ensemble_integration                   test_t32_sample_weight_resampling
  test_scoring
  ```
- **Pre-existing test flakes (deselect):** `tests/test_export_code.py::test_python_script_execution` and `test_full_workflow_python` — known environmental issue.
- **Commit ASAP after implementation, before any operation that yields control.** Long pytest, agent dispatch, `git push`, anything > a few seconds. Memory `feedback_commit_before_yielding_control.md`.
- **Don't use `pytest | tail` in chain-commit patterns.** The pipe inherits `tail`'s exit code (always 0), masking failing tests. Use `2>&1 | grep -E "passed|failed"` if you need length-limited output (grep returns 0 on match — preserves passed-success vs failed-failure correctly).
- **Comments default to none.** Only when WHY is non-obvious. NEVER include review-trail prefixes (`# T-NN fix:`, `# DeepSeek HIGH:`) in production code — those belong in commit messages and PR descriptions.
- **Black line-length 100, type hints with `from __future__ import annotations`, snake_case** per `.claude/rules/code-style.md`.
- **Don't add features beyond what the ticket requires.** No premature abstractions. Bug fixes that preserve pipeline behavior are fine; pipeline methodology changes need user confirmation.
- **Don't open PRs without authorization.** Default: stop and ask after the branch is pushed and tests are green. *Exception this overnight run:* the user has explicitly authorized you to operate autonomously, so opening PRs after green-and-clean state is fine. But don't merge — leave that to the user.
- **Don't merge any PRs.** This is the user's decision regardless of how clean reviews look.

---

## Cross-family review protocol (when dispatching)

Per ticket where review is dispatched (after full implementation + tests passing on the branch):

1. **DeepSeek V4 Pro Max review** via `opencode-call` agent (model: `deepseek`). Read-only; brief should include the chemometrics-convention guardrails verbatim. Apply HIGH findings as fix-of-fixes commit; defer MEDIUM/LOW unless surgical.
2. **GLM 5.1 review in parallel** via `opencode-call` (NOT llm-call when DeepSeek is also reviewing — they need to share repo access; per `feedback_glm_routing.md` and `feedback_parallel_review_reroute.md`).
3. **For larger / architectural tickets**: consider a final consolidated GLM-only sanity-recheck of the fix-of-fixes commit. Optional but cheap reassurance.
4. Skip Codex review unless something complicated comes up.

If `opencode-call`'s `deepseek` alias refuses on routing-policy grounds, retry with explicit instruction that the alias is known-working and to report the actual error verbatim. Do not accept pre-emptive refusals.

**Verify reviewer findings against actual code before applying.** DeepSeek had 1 false positive in the prior batch (T-50 `bytes_freed` ordering claim). Cross-family review is high-leverage but not infallible.

**Branch-state hygiene:** opencode-call agents may end on a different branch than they checked out (or never check out at all — they often use `git show <sha>:<path>` for read-only review, leaving the worktree on whichever branch was active at dispatch time). After each agent returns, run `git branch` to verify you're on the right branch before resuming edits.

---

## Ticket-evaluation rule (mandatory)

Before starting each ticket, **explicitly classify** in your message to the user:

- **Bug fix** (silently broken, metadata wrong, code path doesn't do what it claims, edge case crashes) → ship after review.
- **Pipeline methodology change** (SNV / autoscale / SG / baseline / variable-selection refactored to address ML "leakage" objections) → STOP. Confirm with user. Default to NO.
- **Pure engineering polish** (test infrastructure for already-covered surfaces, defensive-only changes) → defer unless trivial.

T-30b is hygiene-class with the user-pre-approved scope. Ship after review.
Ridge classifier export is bug-fix-class IF reachable from GUI; pure cleanup IF not. Investigate first, classify second.
PLS-DA parity row is test-infrastructure-class — defer unless priorities 1-4 are all done and there's time left.

---

## Deliverables when you stop

1. List of branches pushed with tip commit hashes.
2. Per-ticket: tests passing count, review verdicts (if dispatched), deviations from plan.
3. Updated `PROJECT_STATUS.md` reflecting what shipped + what's queued.
4. Updated `SESSION_LOG.md` with any architectural traps surfaced.
5. If stopped on BLOCKER: a fresh `docs/CONTINUATION_PROMPT_<YYYY-MM-DD>.md` for the next agent.
6. **`SESSION_LOG.md` is now ~370 lines (over the 200-line archive threshold).** If you trim, file as a separate doc-only commit, not piggybacked on a fix PR.

## After this overnight batch

The bigger arcs queued in `PROJECT_STATUS.md`:

- **T-31** multi-class SIMCA (1-2w) — confirmed real per memory. Implement as true multi-class class-modeling per Oliveri & Downey 2012, NOT as one-class extension of `PCASIMCA`.
- **T-16 reframed** — competitive model-comparison machinery survey.
- **T-17** PLS-2 (2-3w) — multi-target PLS.
- **T-01 reframed** (~2-3d).

Plus session-2026-05-04 deferred follow-ups (in priority order above): T-30b, Ridge classifier export, defensive `.ravel()` hygiene, PLS-DA parity, MLP imbalance no-op marker test.

## Quality bar

Production-ready, but don't gold-plate beyond plan + HIGH review findings. Stick to scope. **Bug fixes / observability / behavioral defaults that materially affect what most users get** earn their slots. Engineering polish defers.
