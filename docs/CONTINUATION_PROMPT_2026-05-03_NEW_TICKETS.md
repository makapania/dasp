# Continuation prompt — start work on the next round of tickets

**Filed:** 2026-05-02 (end of overnight queue session)
**For:** the next agent picking up the next batch of tickets
**Branch state at filing:** main at `6fe4393`, working tree clean except a parallel-session unstaged edit to `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` (six amendment entries — see "Initial state" below).

---

## You are starting fresh — read these BEFORE doing anything else

1. **`docs/PROJECT_STATUS.md` top section.** This is the ground truth for what's currently shipped, what's working, what's deferred, and what's queued. The "What's working now" + "Things to note for future sessions" subsections answer most "is X already done?" questions.
2. **`CLAUDE.md` Session Protocol.** Mandatory reads on session start; mandatory `SESSION_LOG.md` appends when you discover non-obvious things; mandatory `PROJECT_STATUS.md` updates after each commit; mandatory commit + push at session end.
3. **Your auto-memory at `C:\Users\mspon\.claude\projects\C--Users-mspon-git-dasp\memory\MEMORY.md`.** Read every entry before deciding how to approach the work. Critical entries this session reinforced:
   - `feedback_chemometrics_relevance_per_ticket.md` — bugs vs methodology vs polish; default to NO on pipeline-methodology changes; defer pure polish.
   - `feedback_chemometrics_conventions.md` — SNV / autoscale-pre-CV / variable-selection-on-full-training are NOT leakage in this domain; chemometrics community convention overrides ML orthodoxy here.
   - `feedback_deepseek_routing.md` — opencode-call's `deepseek` alias works; retry on pre-emptive refusal.
   - `feedback_tests.md` — don't run the full test suite for small changes; targeted `pytest tests/test_*.py` for the modules each ticket touches.
   - `project_venv.md` — `.venv312/Scripts/python.exe`. `.venv311` is removed.
   - `project_t31_simca_confirmed.md` — T-31 is multi-class SIMCA per Oliveri & Downey 2012, NOT an extension of the one-class `PCASIMCA` in `contamination.py`.
   - `project_t19_user_framing.md` — T-19 is "expose model-native imbalance abilities", not "reproduce a publication's recipe."
   - `project_t15_dropped_t16_reframed.md` — T-15 is dropped; T-16 is reframed as "competitive model-comparison machinery survey + implementation."

---

## Initial state

- **Branch tip:** main at `6fe4393` (post-doc-commit). All overnight-queue PRs merged: #10, #11, #12, #13, #14.
- **Working tree:** clean except `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` has unstaged amendment-block + inline `> **Amended 2026-05-02:** ...` markers from a parallel session. Six dispositions confirmed: T-12 SUBSUMED by T-45 (so T-12 closed), T-15 DROP, T-16 REFRAMED, T-19 REFRAMED, T-22 REFRAMED-DEFERRED, T-31 KEEP. **Decide whether to commit those changes early in your session** — they're additive (don't delete original audit trail), they reference real memory entries, and the corresponding SESSION_LOG entry is already on main referencing them. Safest path: read the diff, confirm it matches the SESSION_LOG entry's description, commit + push as a doc-only commit before starting ticket work.
- **Test sweep state:** 282 passed + 1 skipped across the consolidated regression sweep. Same 10 files: `test_t41_bayesian_sqlite_auto_calculator`, `test_t42_write_path_plumbing`, `test_t43_resume_auto_restore`, `test_run_state`, `test_run_logging`, `test_cv_strategy`, `test_unified_bayesian_baseline`, `test_autoscale_bayesian`, `test_cv_pls_clamp`, `test_ensemble_preprocessing`, `test_ensemble_integration`.
- **Pre-existing test flakes (deselect):** `tests/test_export_code.py::test_python_script_execution` and `test_full_workflow_python` spawn `python` instead of `.venv312` — known environmental issue, not a regression.
- **SESSION_LOG is 1800+ lines past the 200-line archive threshold.** Flagged in the log itself; file a separate maintenance ticket if it bites grep performance. **Do not piggyback the archive cut on a feature PR.**

---

## The work — top 5 actionable tickets, in priority order

**Critical framing rule:** these tickets are larger and more methodology-touching than what shipped overnight. Apply `feedback_chemometrics_relevance_per_ticket.md` rigorously. **Before starting any ticket, classify it as bug / methodology change / polish in your message to the user, and confirm with them if methodology change.** The chemometrics-convention guardrails are load-bearing.

### 1. T-19 (REFRAMED) — expose model-native imbalance abilities

**Memory:** `project_t19_user_framing.md`. **Plan/design doc:** check `docs/plans/` for a T-19-prefixed file; if the original plan is still in publication-reproducibility framing, the REFRAMED scope is "expose / auto-detect model-native imbalance hyperparameters" — `scale_pos_weight` (XGBoost), `is_unbalance` (LightGBM), `auto_class_weights` (CatBoost), `class_weight` (LogisticRegression / SVC / RandomForest).

**Estimated:** 1–2 days. Bug-fix-class (some classifiers can be tuned for imbalance but the GUI doesn't expose it). Pipeline behavior unchanged for users who don't enable imbalance handling.

### 2. T-16 (REFRAMED) — competitive model-comparison machinery survey + implementation

**Memory:** `project_t15_dropped_t16_reframed.md`. **Survey first**, then scope. Look at what Unscrambler / SIMCA / OPUS / PLS_Toolbox expose for two-method comparison: bootstrap CIs, paired permutation tests, Wilcoxon, paired t, jackknife, Bayesian, AIC/BIC. Pick the smallest defensible shape that matches what the chemometrics community actually uses.

**Estimated:** survey + scope first (a few hours), implementation depends on scope. Methodology-adjacent — the *output* is new statistical machinery, but it's machinery the community already uses. Confirm scope with user before implementing.

### 3. T-31 — Multi-class SIMCA (KEEP confirmed)

**Memory:** `project_t31_simca_confirmed.md`. **Implementation must be true multi-class SIMCA per Oliveri & Downey 2012** (one PCA model per class + independent membership decisions per specimen producing "belongs to class A", "belongs to class B", "both", or "neither"). **NOT** an extension of the existing one-class `PCASIMCA` in `src/spectral_predict/contamination.py` — that is a single-class membership detector, structurally wrong base for class-modeling.

User confirmed the use case: bone-FTIR / diagenesis specimens can legitimately belong to multiple classes or none; discriminant classifiers (PLS-DA etc.) force every specimen into one trained class with no "none of the above" output, which produces wrong inference for these data.

**Estimated:** 1–2 weeks. New infrastructure ticket. Real chemometrics value (not polish, not pure methodology). Touches scoring, scoring metrics, and possibly the result-display surface.

### 4. T-17 — Multi-Y / PLS-2

**Estimated:** 2–3 weeks. Largest single-ticket effort but unlocks correlated-target modeling. Read the existing plan doc carefully before starting. Methodology-relevant — get user buy-in on the scope before implementing.

### 5. T-01 (REFRAMED) — external-test-set workflow + RMSEP labeling

The canonical chemometrics solution to varsel-on-full-cal bias is "use an external test set", not "rerun varsel inside CV". Check if there's a REFRAMED plan; if not, sketch one and confirm with user before implementing. The original T-01 plan was in ML-orthodoxy framing (per-fold variable selection inside CV) which is exactly the kind of pipeline-methodology change the user has explicitly said NO to.

**Estimated:** 2–3 days for the reframed scope (UI surface + RMSEP labeling). The reframe itself may merit a fresh plan doc.

---

## Process rules from the overnight session (still apply)

- **Use `.venv312/Scripts/python.exe`.** Project is Python 3.12 only.
- **Targeted pytest, not full suite.** The 10-file consolidated sweep is the regression gate for changes that touch run state, Bayesian, preprocessing, CV, or autoscale. For single-module changes, run the matching test file plus any obviously-related ones.
- **Branch per ticket: `fix/T<NN>-<short-name>`** off latest main.
- **Comments default to none.** Only write a comment if the WHY is non-obvious. NEVER include review-trail prefixes (`# T-NN fix:`, `# DeepSeek HIGH:`, `# Codex review:`) — those rot. Keep the architectural fact, drop the review tag.
- **Black line-length 100, type hints with `from __future__ import annotations`, snake_case** per `.claude/rules/code-style.md`.
- **Don't add features beyond what the task requires.** Three similar lines is better than a premature abstraction.
- **Don't add backwards-compat shims unless explicitly required.**
- **Open the PR yourself only if the user authorizes it.** Default: stop and ask after the branch is pushed and tests are green. The user opens PRs.

## Review protocol

Per ticket (after full implementation + tests passing on the branch):

1. **DeepSeek V4 Pro Max review via `opencode-call` agent (model: deepseek).** Read-only review of the just-pushed branch. Brief should include the chemometrics-convention guardrails verbatim: SNV / SG derivatives / baseline applied per-spectrum or pre-CV global is correct per chemometrics community convention; autoscale fit pre-CV global is PLS_Toolbox / Unscrambler / SIMCA-P default; variable selection on full training set is the chemometrics workflow per Li 2009 / Centner 1996 / Norgaard 2000. **Reviewers who flag these generate false positives.** Apply HIGH findings as a fix-of-fixes commit on the same branch; defer MEDIUM/LOW unless the fix is one-line and surgical.
2. **Every 2 tickets, additionally run a Codex review via the codex-reviewer agent.** Codex catches things DeepSeek misses (and vice versa) — orthogonal-RLHF blind spots. Same chemometrics guardrails. Apply HIGHs only if surgical.
3. **If a review verdict is BLOCKERS or there are unfixable HIGHs:** stop the queue, push what you have, and write a fresh `CONTINUATION_PROMPT_<date>.md` capturing the state and what's left.

If `opencode-call`'s `deepseek` alias refuses on routing-policy grounds, retry with explicit instruction that the alias is known-working and to report the actual error verbatim. Do not accept pre-emptive refusals (memory `feedback_deepseek_routing.md`).

## Ticket-evaluation rule (mandatory)

Before starting each ticket, **explicitly classify** in your message to the user:

- **Bug fix** (something is silently broken, metadata is wrong, code path doesn't do what it claims, edge case crashes) → ship after review.
- **Pipeline methodology change** (SNV / autoscale / SG / baseline / variable-selection refactored to address ML "leakage" objections) → STOP. Confirm with user. Default to NO. **The chemometrics-conventions memory + the original T-01/T-02/T-03 leakage-panic retirement entries make clear that the user does not want these refactors.**
- **Pure engineering polish** (test infrastructure for already-covered surfaces, defensive-only changes, low-probability-regression pinning) → defer unless trivial.

T-19 / T-31 are bug-fix / new-infrastructure (real chemometrics value). T-16 / T-17 are methodology-adjacent — get explicit user buy-in on scope before implementing. T-01 reframed is bug-fix-class for the reframed scope (UI + RMSEP labeling).

## Deliverables when you stop

1. List of branches pushed with tip commit hashes.
2. Per-ticket: tests passing count, review verdicts, deviations from plan.
3. Updated `PROJECT_STATUS.md` reflecting what shipped + what's deferred + what's queued.
4. Updated `SESSION_LOG.md` with any architectural traps surfaced.
5. If stopped on BLOCKER: a fresh `docs/CONTINUATION_PROMPT_<YYYY-MM-DD>.md` for the next agent.

## Quality bar

Production-ready, but don't gold-plate beyond plan + HIGH review findings. Stick to scope. **Bug fixes / observability / behavioral defaults that materially affect what most users get** are the kinds of changes that earned their slots overnight. Keep that bar.
