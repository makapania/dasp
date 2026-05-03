# Continuation prompt — morning 2026-05-05 ticket-closing run

**Filed:** 2026-05-04 late evening, after the 8-PR merge cascade landed (main at `a31682b`).
**For:** the next agent picking up Tuesday morning to close deferred tickets and verify GUI wording.
**Branch state at filing:** clean main, no PRs in flight, working tree clean apart from the persistent `.review-tmp/` and `C\357\200\272Users...search_t30.py` artifacts (ignore both).

---

## First commands on wake-up — get oriented

```
git fetch --all --prune
git checkout main
git pull origin main
git status
gh pr list --state open
```

`gh pr list` should return **empty** — all 8 PRs from the 2026-05-04 batch were merged tonight. If that's not the case, something in the cascade didn't land cleanly; check `docs/PROJECT_STATUS.md` first.

---

## MUST READ before doing any work

1. **`docs/PROJECT_STATUS.md`** — top section captures what just merged and the current backlog. The "Session 2026-05-04 — 8 PRs MERGED to main" table lists every squash-merge SHA + ticket scope. The **deferred-followup tickets** section lists what's queued.
2. **`docs/SESSION_LOG.md`** — top three entries are the most recent and most relevant:
   - `2026-05-04 stacked-PR merge cascade` — GitHub gotcha for stacked-PR rebase-and-reopen
   - `2026-05-04 codex round 2` — codex's call-graph reasoning catches cross-file bugs (PR #24 was a real production bug)
   - `2026-05-04 overnight (T-30b + P3 + T-20c)` — review-method asymmetry findings
3. **`CLAUDE.md`** Session Protocol — same as always: read PROJECT_STATUS first; append to SESSION_LOG immediately on non-obvious findings; commit doc updates before session end.
4. **`C:\Users\mspon\.claude\projects\C--Users-mspon-git-dasp\memory\MEMORY.md`** — the auto-memory index. Most relevant entries this morning:
   - `feedback_review_method_signal.md` — per-reviewer track record + stage-assignment matrix. **Use this to decide which reviewers to dispatch on each ticket.** Codex is mandatory for codegen / dispatcher / runtime↔export work; cross-family for everything else; pr-test-analyzer when test files change.
   - `feedback_chemometrics_relevance_per_ticket.md` — bug fixes fine; pipeline methodology changes need user confirmation; engineering polish defers.
   - `feedback_chemometrics_conventions.md` — SNV / SG / autoscale-pre-CV / variable-selection-on-full-training are NOT leakage by chemometrics convention.
   - `feedback_commit_before_yielding_control.md` — commit ASAP after implementation, before any operation that yields control.
   - `feedback_glm_routing.md` + `feedback_deepseek_routing.md` + `feedback_opencode_call_heredoc_hang.md` — routing rules. Default to opencode-call for full repo access.
   - `project_venv.md` — `.venv312/Scripts/python.exe`. `.venv311` is removed.
   - `feedback_tests.md` — targeted pytest, not full suite. Consolidated 13-file regression sweep is the gate.

---

## CHEMOMETRICS GUARDRAILS — read these verbatim, include them in every reviewer brief

This is a chemometrics / spectral-modeling project. ML conventions don't always apply. **Most outside reviewers (and most LLMs) will flag normal chemometrics patterns as bugs unless explicitly told not to.** Embed these guardrails verbatim in every cross-family / pr-review-toolkit / Codex / Kimi review brief.

### What is NOT a leakage bug (despite ML reviewer instinct)

The following patterns are **chemometrics convention** and are **NOT** cross-sample leakage:

1. **SNV (Standard Normal Variate)** applied to full dataset before CV split. SNV is a per-spectrum operation (mean and std computed within a single spectrum across wavelengths, not across samples). Spectrum *i* knows nothing about spectrum *j* after SNV.
2. **Savitzky-Golay derivatives** (1st, 2nd) applied to full dataset before CV split. Same reasoning — per-spectrum convolution along the wavelength axis.
3. **Baseline correction** (asymmetric least squares, polynomial detrending) applied to full dataset before CV split. Per-spectrum.
4. **Autoscale (UV-scale)** applied within the search loop, before CV — convention is to pre-process the full training set then CV. ASTM E1655 and Pomerantsev et al. 2025 LOVE both treat this as standard practice.
5. **Variable selection (UVE, SPA, iPLS, CARS, GA-PLS) on full training data** — same. Not leakage when the test/validation set is genuinely held out from training.
6. **`scale=False` on `PLSRegression`** — codebase-wide convention (24+ runtime sites). Chemometrics expects SNV/derivative preprocessing has already centred/standardised the spectra; letting sklearn auto-scale on top would double-process. Don't change to `scale=True`.

If a reviewer flags any of these as "leakage," it's a false positive. Defuse and continue.

### Ticket-classification rule (mandatory before starting each ticket)

For every ticket, **explicitly classify** which of these three buckets it falls into:

- **Bug fix** (silently broken, metadata wrong, code path doesn't do what it claims, edge case crashes) → **ship after review.**
- **Pipeline methodology change** (SNV / autoscale / SG / baseline / variable-selection refactored to address ML "leakage" objections, or any change that alters what the chemometrics pipeline actually computes) → **STOP. Confirm with user. Default to NO.**
- **Pure engineering polish** (test infrastructure for already-covered surfaces, defensive-only changes, comment polish) → **defer unless trivial.**

Examples from this morning's queue:
- GUI wording fix → polish (trivial, ship after one round of confirmation on the wording itself).
- Imbalanced auto parity row → bug fix (closes a real coverage gap on a known bug class).
- `n_jobs` strip fix → bug fix (multi-threaded determinism drift potential — real, even if not currently observed).
- Ridge dead-code deletion → polish (cleans up confusion, no behavior change).
- PLS-DA `ndim>2` codegen port → defensive polish (currently-unreachable for 1-D y; defer unless shipping standalone).

### Specific T-19 / T-31 / chemometrics scope-correction history

User has corrected scope multiple times when reviewers (or I) drifted into ML-publication-reproducibility framing instead of "expose model-native chemometrics abilities":
- **T-19 reframed** (memory `project_t19_user_framing.md`): expose model-native imbalance abilities (or auto-detect), NOT a publication-reproducibility framework. Auto mode is the priority; per-model UI dropdowns deferred.
- **T-15 dropped** (memory `project_t15_dropped_t16_reframed.md`): LOGO is footgun in user's data regime; not field-mandated. Don't re-implement.
- **T-16 reframed**: standalone "competitive model-comparison machinery survey + implementation."
- **T-31 multi-class SIMCA confirmed** (memory `project_t31_simca_confirmed.md`): user confirmed 2026-05-02 — true multi-class class-modeling per Oliveri & Downey 2012, NOT one-class extension. Don't re-litigate.
- **CARS-Tree** (memory `project_cars_tree_origin.md`): is dasp's invention; not in Li 2009. Don't search for canonical reference.

If you're tempted to extend a ticket's scope toward "ML best practice" or "publication reproducibility," **stop and confirm with user first**. Default answer is no.

### Reviewer brief template (use for every dispatch)

When dispatching DeepSeek / GLM / Codex / Kimi / pr-review-toolkit, include this verbatim block in the prompt:

```
## Project chemometrics guardrails (read before flagging)

This is a chemometrics / spectral-modeling project. Reviewer guardrails:
1. Chemometrics conventions are NOT ML "leakage" bugs. SNV / SG derivatives /
   baseline correction / autoscale-pre-CV / variable-selection-on-full-training
   are per-spectrum or per-sample operations and are NOT cross-sample leakage
   by community convention (Pomerantsev et al. 2025 LOVE; ASTM E1655). Don't
   flag them.
2. `scale=False` on PLSRegression is the codebase-wide convention (24+ runtime
   sites). Don't suggest changing to scale=True.
3. Bug fixes fine; pipeline methodology changes need user confirmation; pure
   engineering polish defers.
4. Black line-length 100, type hints with `from __future__ import annotations`,
   snake_case. No review-trail prefixes (`# T-NN fix:`) in production code.
```

Reviewers without this brief generated false positives in the past (T-50 false-positive on `bytes_freed`; multiple "leakage" flags on SNV/autoscale across earlier sessions). The brief defuses ~90% of those.

---

## Priority order for this morning

### Priority 0 (USER ASK — blocking everything else): GUI wording sanity check on auto-resume / Auto mode speed claims

The user noted: *"check the gui wording about speed in the auto-resume section. i think we might have made it a bit faster so what the user sees in the gui may not be quite right anymore."*

The user is referring to one of two areas (their phrasing was slightly ambiguous):
- **Auto-resume of interrupted Bayesian searches** (T-43 / T-50 territory — Optuna sqlite resumption, stale-archive cleanup)
- **T-19 Auto mode for imbalance handling** (auto-resolves to `class_weight` or `None` based on imbalance ratio; could affect perceived search speed because balanced data skips correction)

Recent changes that could have affected speed in either area:
- T-19 Auto mode (commit `0b1d4a2`) — added auto-resolution at `run_search` / `run_nsga2_search` / `run_unified_bayesian` entry points. On balanced data, skips `class_weight` injection entirely → faster than always-on `'class_weight'` mode for balanced data.
- T-50 (PR #16, SHA `348bdd8`) — auto-cleanup of stale Optuna SQLite archives at app startup. Doesn't affect runtime speed, only startup file-IO.
- T-43 (PR #4 historical) — resume auto-restore for interrupted Bayesian searches. Speed claims here date back to the original implementation.

**What to do:**

1. `rg -n "auto" spectral_predict_gui_optimized.py | rg -i "speed|fast|faster|slow|second|minute|hour"` — find any GUI label/tooltip that mentions speed/timing in auto-* contexts.
2. For each hit, check what the actual current behavior is — read the underlying implementation (`search.py` / `unified_bayesian.py` / `run_state.py`) and confirm whether the GUI claim is still accurate.
3. Specifically check:
   - The Auto mode dropdown tooltip in the imbalance-method selector (somewhere in the GUI — grep for `'auto'` in the GUI file's imbalance-method selector code).
   - The auto-resume confirmation dialog or status message (grep for `resume` or `Resume` in the GUI).
   - Any "estimated time" / "speed" hint near these features.
4. If any wording is now stale, **propose** the corrected wording first (don't just edit). Confirm with the user before pushing — wording changes are user-facing and should match the user's mental model.

Branch this as `fix/Tgui-auto-mode-speed-wording` off main.

**Expected effort**: 30-60 min including grep, code-reading verification, and one round of confirmation with the user.

### Priority 1: Pre-existing T-22 / T-28 / T-31 / similar real tickets

Read `docs/PROJECT_STATUS.md`'s deferred-followup section. The current queue:

- **Codex MEDIUM from PR #22 (now `0e7eeb6` on main)**: `_PIPELINE_PARAMS` strips bare `n_jobs` from test params, after which codegen re-injects `n_jobs=-1` for XGB/LGBM via `setdefault`. Multi-threaded determinism could drift. **Tractability: medium.** Fix is to remove `n_jobs` from `_PIPELINE_PARAMS` (only strip true pipeline params and prefixed pipeline keys). Run T-20 parity suite to confirm no breaks. Cross-family review WORTH IT (production code change).
- **Codex MEDIUM from PR #22**: missing imbalanced auto parity row. The existing auto-mode T-20 row uses balanced data and expects no-correction; no row pins the auto-resolves-to-class_weight imbalanced path. **Tractability: easy.** Add `test_xgboost_binary_classification_parity_auto_with_correction` to `tests/test_t20_saved_model_export_parity.py` mirroring the pattern at `test_xgboost_binary_classification_parity_class_weight` but with `imbalance_method='auto'` and `expect_stdout_marker="applying class_weight"`. pr-test-analyzer review WORTH IT.
- **Codex MEDIUM/LOW from PR #28**: `scaler__*` keys silently dropped by `_split_pls_da_params`; MLP marker stdout-only. **Tractability: low (scope-creep).** Both are production-codegen changes. Defer until user confirms scope.
- **Ridge classifier export cleanup** — runtime branch in `models.py:471-473` is dead code. Either delete it (smallest change) or implement Ridge classification end-to-end (feature, not bug fix). **Tractability: trivial-to-delete; substantial-to-implement.** Default: delete the dead code with a 1-line PR.
- **DeepSeek M2 from PR #28**: port the `ndim>2` fallback from `models.PLSTransformer` to the codegen's inline `PLSTransformer`. Future-proofs against PLS-2 (T-17). **Tractability: easy** (4-line addition). Naturally bundled with T-17 if it ships next; can ship standalone otherwise.

**Recommended pickup order** (highest leverage first):
1. **GUI auto-mode/auto-resume wording verification** (P0 user ask)
2. **Imbalanced auto parity row** (easy + closes a real coverage gap on the most-leverage T-19 path)
3. **`n_jobs` strip fix** (medium effort + closes a determinism-drift concern)
4. **Ridge classifier dead-code deletion** (trivial + cleans up confusion)
5. **PLS-DA `ndim>2` codegen port** (only if shipping standalone — otherwise let T-17 absorb it)

Skip the bigger arcs (T-31 multi-class SIMCA at 1-2w, T-17 PLS-2 at 2-3w, T-16 reframed model-comparison, T-01 reframed) unless the user asks for them — those need scoping conversations first.

### Priority 2 (only if P0 + several from P1 are done before noon): T-01 reframed scoping

T-01 reframed (~2-3d) is the next real arc. If the morning runs faster than expected, read what's filed about T-01 reframed scope and propose a one-page scoping doc to the user before lunch — don't start implementing.

---

## Process rules (mandatory)

- **Use `.venv312/Scripts/python.exe`.** Project is Python 3.12 only.
- **Targeted pytest, not full suite.** Per-ticket: matching test file + the consolidated 13-file regression sweep:
  ```
  test_t41_bayesian_sqlite_auto_calculator    test_t42_write_path_plumbing
  test_t43_resume_auto_restore                test_run_state
  test_run_logging                            test_cv_strategy
  test_unified_bayesian_baseline              test_autoscale_bayesian
  test_cv_pls_clamp                           test_ensemble_preprocessing
  test_ensemble_integration                   test_t32_sample_weight_resampling
  test_scoring
  ```
- **Pre-existing flakes (deselect):** `tests/test_export_code.py::test_python_script_execution` and `test_full_workflow_python` — known environmental.
- **Commit ASAP after implementation.** Long pytest runs / agent dispatch / `git push` can collide with parallel sessions. See `feedback_commit_before_yielding_control.md`.
- **Don't use `pytest | tail`.** Use `2>&1 | grep -E "passed|failed"` for length-limited output.
- **Comments default to none.** Only when WHY is non-obvious. NEVER include review-trail prefixes (`# T-NN fix:`, `# DeepSeek HIGH:`) in production code.
- **Black line-length 100, type hints with `from __future__ import annotations`, snake_case.**
- **Don't add features beyond what the ticket requires.** Bug fixes / hygiene fine; methodology changes need user confirmation.
- **Don't open PRs without authorization in normal mode.** If user has set auto-mode for the morning session, opening PRs after green-and-clean state is fine. Don't merge — that's the user's call.

---

## GLM 5.1 for non-review reasoning (cheap-option calls)

Per `~/.claude/skills/llm-call/SKILL.md` line 85: *"When the user asks for a 'second opinion' on something Claude or Codex already touched, **route to a non-Anthropic, non-OpenAI model** unless they explicitly name otherwise."* This means GLM 5.1 (via opencode-call alias `glm` or `/llm-call glm`) bills against the z.ai subscription and is available for **second opinions, scope checks, draft review, and rubber-duck reasoning on anything Claude has already touched** — not just formal code review. The 2026-05-04 session under-used this; specific calls that should have been made:

- **Merge / rebase / git-state decisions** where you're reasoning from first principles (e.g., GitHub auto-closing a stacked PR after base deletion — 30-second GLM consult would have surfaced the recovery pattern faster than trial-and-error).
- **Scope-expansion checks** when you're tempted to add a test row / fix-of-fixes / extra commit beyond the user-approved ticket. Ask GLM: "is this in scope for the original ask?" — if it says scope creep, file as a separate ticket and confirm with user.
- **Doc / prompt drafts** when the audience is another autonomous agent. Ask GLM to review the draft for clarity-to-an-agent-with-no-prior-context.
- **xfail / test-marker resolution** when a rebase exposes a stale marker. Ask GLM: "marker X went XPASS-strict after rebase; should I remove it or keep it scoped to the original fix?"

Cheap to dispatch (one-shot llm-call, no agent dispatch); skip when you're 100% confident or the decision is reversible. Default-ON for non-trivial git operations and scope decisions.

---

## Review protocol (use the new memory)

Per `feedback_review_method_signal.md`:

| Change class | Mandatory | Optional |
|---|---|---|
| Bug fix in single file, no cross-file dispatch | DeepSeek+GLM cross-family | — |
| Codegen / search-loop dispatch / runtime↔export parity | DeepSeek+GLM **+ Codex** | Kimi sister-site sweep on fix-of-fixes |
| Test additions | DeepSeek+GLM **+ pr-test-analyzer** | — |
| Pure docs / hygiene / one-line | Skip if confidence high | — |

Specifically for the morning's likely tickets:
- **GUI wording fix**: cross-family if any. Surgical hygiene; could skip review.
- **Imbalanced auto parity row** (test addition): cross-family + pr-test-analyzer.
- **`n_jobs` strip fix** (production code, codegen-adjacent): cross-family + Codex (mandatory, this is exactly where Codex earns its slot).
- **Ridge dead-code deletion**: cross-family if any; could skip review on a 1-line deletion.
- **PLS-DA `ndim>2` codegen port**: cross-family + Codex.

Dispatch reviewers in parallel via `Agent` calls in a single message when possible.

---

## Branch / PR cadence

- One ticket per branch, one PR per branch. Don't bundle.
- Branch name: `fix/<ticket-id>-<short-slug>` (`fix/Tgui-auto-mode-wording`, `fix/T20-imbalanced-auto-parity-row`, etc.).
- Push after tests are green AND commits are clean. Open PR with descriptive body referencing the relevant ticket / SESSION_LOG entry.
- Squash-merge from the user; you don't merge.

---

## Deliverables when you stop

1. List of branches pushed with tip commit hashes.
2. Per-ticket: tests passing count, review verdicts, deviations from plan.
3. Updated `PROJECT_STATUS.md` reflecting what shipped + what's queued.
4. Updated `SESSION_LOG.md` with any architectural traps surfaced.
5. If stopped on a BLOCKER, file `docs/CONTINUATION_PROMPT_2026-05-05_AFTERNOON.md` for the next agent.
6. **`SESSION_LOG.md`** is now ~410 lines (over the 200-line archive threshold). If you trim, file as a separate doc-only commit, not piggybacked onto a fix PR.

---

## Quality bar

Production-ready, but don't gold-plate. Stick to ticket scope + HIGH review findings. Bug fixes / observability / behavioral defaults that materially affect what most users get earn their slots. Engineering polish defers.

The user explicitly cares about **GUI wording accuracy** (P0) and **closing the deferred backlog before T-31 / T-17 starts**. That's the morning's mandate.
