# Project Status

> **Last updated:** 2026-05-05 late evening — Validation-rebuild MEDIUM closed (`fix/Tclass-weight-validation-rebuild`); ensemble-rebuild LOW dropped as moot (ensemble methods are regression-only — class_weight discriminator can't reach that path). All other 6 morning/afternoon PRs (#32, #33, #35, #36, #38, #39, #40) merged earlier in this session. PR #34 closed (Codex caught false-premise). PR #37 closed (redundant docs). Queue empty pre-validation-rebuild PR.
>
> ## Session 2026-05-05 late evening — validation-rebuild MEDIUM (fix/Tclass-weight-validation-rebuild)
>
> ### What shipped
>
> | File | Change |
> |---|---|
> | `src/spectral_predict/search.py` | New helper `_apply_class_weight_discriminator_for_rebuilt_model` mirrors the canonical PR #38 pattern but uses Codex's `get_params(deep=True)` priority-order probe (`lr__class_weight`, `model__class_weight`, `class_weight`) to handle Pipeline step naming without dispatching on `model_name`. Plus `imbalance_method` parameter added to `compute_validation_metrics_for_top_models`; helper applied at the rebuild fit site. |
> | `src/spectral_predict/search.py` (callers) | `run_search` line 3346 + `run_bayesian_search` line 3987 — pass `imbalance_method` through. |
> | `spectral_predict_gui_optimized.py` (callers) | gui:27914 + gui:28069 — pass `imbalance_method` through. The two GUI direct callers were caught by Codex's design pass; the continuation prompt only flagged the 2 backend callers. |
> | `tests/test_class_weight_validation_rebuild.py` (new) | 12 tests: short-circuit invariants, per-model class_weight application (PLS-DA Pipeline, scale-sensitive Pipeline, sklearn bare), XGBoost sample_weight fallback (the headline behavioral test), CatBoost auto_class_weights branch, caller-threading structural pins. |
>
> ### Ensemble-rebuild LOW dropped
>
> Continuation prompt's LOW (`_reconstruct_models_from_results` at gui:23642+) is **not reachable for classification** — ensemble methods are regression-only via two gates: `_update_ensemble_controls_state` greys out UI controls (gui:16669-16670) AND auto-flow at gui:28323+ explicitly skips. class_weight discriminator is classification-only → cannot apply to this path → no fix needed. Same transferred-justification fallacy as the previous-session entry. Per "double-check fixes are really needed" rule, dropped not patched.
>
> ### Test coverage
>
> 190 passed + 1 skipped across `test_class_weight_validation_rebuild` (12 new) + `test_class_weight_sister_sites` + `test_imbalance` + `test_t19_class_weight_per_library` + `test_t32_sample_weight_resampling` + `test_unified_bayesian_baseline` + `test_cv_strategy`. No regressions.
>
> ### Lessons
>
> Detailed writeup in SESSION_LOG.md — three lessons: continuation prompts are subject to the transferred-justification fallacy; Codex's `get_params(deep=True)` priority-order probe is a cleaner Pipeline-discriminator pattern; continuation prompts under-specify caller surface (the GUI sister sites were missing).
>
> ## Session 2026-05-05 evening — factory + needs_resampling_pipeline asymmetry fix (PR #39)
>
> ### What happened
>
> 1. A prior agent's "trivial defense-in-depth" patch for `RegressionResampler.fit_resample` (`f6ccfd1`) was committed locally, then reset before push when the user surfaced doubts about silent-failure shape.
> 2. A four-way cross-family investigation was run on the contested no-op: silent-failure-hunter (Claude) + Codex GPT-5.5 + GLM 5.1 (z.ai) + DeepSeek V4 Pro Max (max-thinking, DeepSeek API). All four converged on "no-op is wrong" — the shape-pattern-match transferred without the safety-justification across the classification/regression boundary. Vote on remedy: 2-1 toward `raise ValueError` (Codex+GLM) over warn-and-no-op (DeepSeek). User directive: scientific accuracy over user convenience → raise.
> 3. The same misanalysis was identified at *two* layers, not one: the inner `RegressionResampler.fit_resample` patch (already reverted) AND the factory `build_imbalance_transformer` (lines 947-956 pre-fix, silently routing regression+sentinel to ClassificationResampler's no-op regardless of task_type — same shape, same broken safety justification). The factory layer is reachable today via the GUI ensemble-reload path (`spectral_predict_gui_optimized.py:37587`) loading a saved classification config into a regression context.
>
> ### Sites fixed
>
> | # | File:line | Function | Change |
> |---|---|---|---|
> | 1 | `imbalance.py:947-967` | `build_imbalance_transformer` factory | Split sentinel guard by task_type. Classification: keep route-to-no-op (compensation is real). Regression: raise `ValueError` with diagnostic message naming the classification-only nature of the sentinel and listing valid regression methods. |
> | 2 | `unified_bayesian.py:156` | `_needs_resampling_pipeline` | Guard both `'class_weight'` AND `'auto'` (was `'class_weight'` only). Brings into line with `search.py:289`. Defense-in-depth against future `'auto'`-resolution-delay refactors. |
> | 3 | `nsga2_search.py:142` | `_needs_resampling_pipeline` | Same as #2. |
>
> ### Sites NOT touched (deliberate)
>
> - `ClassificationResampler.fit:244` no-op stays as-is. Architecturally fragile (a future bypass-the-router caller would silently train an unweighted classifier) but produces correct scientific output today because every integrated caller compensates at construction. Filed in SESSION_LOG as a future hardening item; not expanded into present scope per "fix what's wrong, don't redesign around it."
>
> ### Test contract update
>
> - `tests/test_imbalance.py`: parametrized regression-task no-op test split into two — `test_classification_sentinels_no_op` (4 cases including `CLASS_WEIGHT`/`Auto` to exercise the `.lower()` in the factory guard) and `test_regression_sentinels_raise` (4 cases, `pytest.raises(ValueError, match="classification-only")`).
> - 165 passed + 1 skipped across `test_imbalance.py` + `test_t19_class_weight_per_library.py` + `test_t32_sample_weight_resampling.py` + `test_unified_bayesian_baseline.py` + `test_cv_strategy.py`. No regressions.
>
> ### Pre-merge verification (Codex GPT-5.5 + MiMo 2.5 Pro)
>
> - **Codex GPT-5.5**: NEEDS_CHANGES on a documentation LOW (test parametrize claimed 4 cases, was 2 — fixed by expanding to 4 cases as above). Code itself verified correct.
> - **MiMo 2.5 Pro**: READY_TO_MERGE. Notably traced the `ValueError` through the GUI exception handlers and confirmed it surfaces to the user — the inner `try/except` at `spectral_predict_gui_optimized.py:37595,37722` catches only `ImportError`, so the `ValueError` propagates to the outer `except Exception` at `:38614` and renders via `messagebox.showerror`. Same for both reachable GUI call sites. Loud-failure end-to-end verified, not just at the API layer.
> - **MiMo non-blocking informational** (future awareness): caller guards at `search.py:4374`, `unified_bayesian.py:1231`, `nsga2_search.py:1394+` check `!= 'class_weight'` but not `!= 'auto'`. Currently safe because `'auto'` is resolved at run-entry; for regression `'auto'` reaches the factory and now raises. The factory is the single enforcement point; the `_needs_resampling_pipeline` dual-sentinel guard is defense-in-depth only. Acceptable architecture, flagged for future awareness.
>
> ### Lessons
>
> Detailed writeup in SESSION_LOG.md — four lessons captured: transferred-justification fallacy in defense-in-depth; cross-family panels reveal values disagreements that single-reviewer panels can't; "verification" reviews check reachability not remedy-fitness; the "skips the misleading print()" argument is circular self-justification.
>
> ### Deferred follow-ups added this session
>
> - **T-resume-y-variable-persist (state-restoration gap, user-clarified 2026-05-05 evening; GLM 5.1 review 2026-05-05 late evening)** — Y variable selection is not persisted/restored as part of the run-state sidecar. On data load the first column is always defaulted as Y, so when a user resumes a disrupted run, whatever-was-first-in-the-file shows up as Y rather than the variable the run was actually using. If the run was a classification analysis but the file's first column is a regression variable, resume effectively can't restart against the original target. **Not a bug** — defaulting to first-column on load is correct behavior. Just a missing piece of state that needs to round-trip alongside the other run-state already persisted (preprocessing config, model selection, validation set, etc.).
>
>   **Implementation specifics (verified by GLM 5.1 against current code):**
>
>   - Add `"target_column"` to `CAPTURABLE_SETTINGS` in `src/spectral_predict/run_gui_settings.py` (the whitelist at lines 45-211). The Tk variable to capture is `self.target_column` (a `StringVar` at `spectral_predict_gui_optimized.py:2837`).
>   - **No change to `run_state.py` needed** — it already stores `gui_settings: dict[str, Any] | None` generically (line 148), populated by `capture_gui_settings`.
>   - **`restore_gui_settings` works generically** — calls `gui_obj.<key>.set(value)` (line 259), which works for `StringVar`.
>   - **Edge case (don't skip)**: if the restored column name doesn't exist in the newly-loaded dataset (renamed column, different file), the Combobox will display a stale string that isn't in its option list. The existing readback check at `run_gui_settings.py:299-309` only verifies `.set()`/`.get()` round-trip — it does NOT validate against the current data's columns. Implementation must validate the restored value against `_get_available_target_columns()` (defined at `spectral_predict_gui_optimized.py:16849`) and fall back to default if it doesn't match. Without this, the user sees a "stuck" combobox they can't interact with cleanly.
>   - **Related-state observation**: `dataset_fingerprint` (`run_state.py:138`) already detects when the user loaded different data on resume. A stale-column-name scenario would likely co-occur with the existing fingerprint-mismatch warning the GUI already shows. The validation-against-current-columns is still required for cases where the fingerprint matches but the column ordering/naming changed (e.g., user added a column, deleted a column, etc.).
>   - **No schema migration concern**: `gui_settings` is a flat dict and `from_dict` already ignores unknown keys (line 183). Adding `target_column` is forward-compatible.
>
>   Detailed continuation prompt at `docs/CONTINUATION_PROMPT_2026-05-07_resume_y_variable_persist.md`.
>
> ## Session 2026-05-05 afternoon — class_weight sister-site bug class fully closed (PR #38)
>
> ### What shipped
>
> | PR | Branch | Scope | Reviewers | Verdict |
> |---|---|---|---|---|
> | #38 | `fix/Tclass-weight-discriminator-sister-sites` | Bayesian + NSGA-II classification: 4 sister sites of the GUI dispatcher bug (4dcedbc/c395317) had the same defective `hasattr(model, 'class_weight')`-only check; pre-fix, every Bayesian trial AND NSGA-II evaluation for CatBoost/XGBoost classifier under `imbalance_method='class_weight'` (or `'auto'` → it) trained UNWEIGHTED. User-visible calibration metrics (Accuracy / F1 / AUC etc. shown in Bayesian Results panel and NSGA-II Pareto results) reflected the wrong model. Plus 4 cv_utils CV helpers extended to thread sample_weight per-fold; sklearn 1.8 metadata-routing compat for the non-boosting fallback; def-in-depth `'auto'` guards in `_needs_resampling_pipeline`; explicit CatBoost branch added to canonical `search.py:4418-4435` for codebase-wide consistency. | DeepSeek + GLM + Codex (round 1, all caught the same 4th sister site = convergent HIGH); Kimi K2.6 (round 2, exhaustive sister-site sweep confirmed exhaustive). 4-way convergent verdict: **READY_TO_MERGE**. | Ready, NOT MERGED |
>
> ### Sites fixed (all 4 confirmed exhaustive by Kimi)
>
> | # | File:line | Function | Symptom |
> |---|---|---|---|
> | 1 | `unified_bayesian.py:1259` | `objective()` discriminator | Trial-time CV scores AND calibration refit at line 1532 produced unweighted metrics for CatBoost/XGBoost |
> | 2 | `nsga2_search.py:1420` | `SearchProblem._evaluate` | Pareto front built on unweighted scores |
> | 3 | `nsga2_search.py:3175` | `_compute_classification_cv_metrics` | NSGA-II Results-panel `*_cv` columns on unweighted models |
> | 4 | `nsga2_search.py:3596` | `_compute_calibration_metrics` | NSGA-II Results-panel non-`*_cv` calibration columns on unweighted models (4th site, caught by Codex+GLM+DeepSeek convergence) |
> | def-in-depth | `unified_bayesian.py:156` + `nsga2_search.py:142` | `_needs_resampling_pipeline` | Missing `'auto'` guard (currently unreachable) |
> | consistency | `search.py:4424` | `_run_single_fold` setup | Was using sample_weight fallback for CatBoost; now explicit `auto_class_weights` branch (mechanism-aligned with codebase convention) |
>
> ### Test coverage
>
> - **437 passed + 2 skipped** across the consolidated 13-file regression sweep + `test_class_weight_sister_sites.py` (9 new tests) + `test_imbalance.py` (62) + `test_nsga2_search.py` (34). Pre-PR baseline was 333 + 1 skipped.
> - End-to-end: parametrized over CatBoost + XGBoost, Bayesian search runs twice (with/without `class_weight`), assert calibration metrics differ. Pre-fix the assertion would FAIL because both calls produced identical (unweighted) calibration metrics.
> - Plumbing: sample_weight propagates through `cross_val_predict_pooled` and `cross_val_predict_with_early_stopping` on imbalanced data.
> - Defense-in-depth: `_needs_resampling_pipeline('auto', 'classification') is False` in both modules.
>
> ### Lessons captured
>
> Detailed writeup in SESSION_LOG.md — three high-leverage lessons:
> 1. **Four-way convergent reviews on a fix-of-fixes commit is the strongest signal possible.** Codex + GLM + DeepSeek + Kimi all independently confirmed READY_TO_MERGE on round-2 commit `724847d` (the `_compute_calibration_metrics` 4th sister site fix + `set_fit_request` clone-before-mutate + search.py CatBoost branch + test `importorskip` fix).
> 2. **The 4th sister site (`_compute_calibration_metrics`) was the most insidious of the four** because it computes the CALIBRATION metrics shown right next to the `_cv` metrics in the Results panel. Pre-fix you'd see `Accuracy=0.85 / Accuracycv=0.78` where Accuracy came from an unweighted refit and Accuracycv came from a weighted CV — visually inconsistent but only a red flag if you knew to look.
> 3. **`set_fit_request` mutation is sticky** on the estimator instance even after `sklearn.config_context(enable_metadata_routing=True)` exits. Cloning the model first via `sklearn.base.clone` before set_fit_request is the right pattern — caught by GLM's grep-then-read sweep, not flagged by Codex's call-graph reasoning despite the same trace going through both.
>
> ### Deferred follow-ups (filed for next agent)
>
> Continuation prompt at `docs/CONTINUATION_PROMPT_2026-05-06_validation_rebuild.md` covers:
>
> - **MEDIUM (Kimi)** — `compute_validation_metrics_for_top_models` at `search.py:408`: validation-set metric computation rebuilds models via `_rebuild_model_from_row` but doesn't thread `imbalance_method`. XGBoost retrains unweighted there; PLS-DA's `class_weight='balanced'` is baked into the LR constructor at search.py:4488 but never serialized into the captured params dict, so reconstruction loses it. Affects only `val_*` columns. Reached when `validation_count > 0` in `run_search` / `run_bayesian_search`.
> - **LOW (Kimi)** — `_reconstruct_models_from_results` at `gui:23643`: ensemble-reconstruction path fits rebuilt pipelines at 4 sites without applying class_weight / auto_class_weights / sample_weight. Affects only "Train Ensemble" feature.
> - **PR #33 deferred** — pr-test-analyzer's 2 HIGHs (behavioral `assert "'n_jobs': 1" in script` test + architectural module-walk test). Pin tests are structural only.
> - **PR #33 LOW** — `code_generator.py` KNOWN ISSUE comment incorrectly claims LightGBM gets a `verbose=0` re-injection (only CatBoost does). 1-line comment fix.
> - **PR #32 MEDIUM** — pr-test-analyzer flagged missing LightGBM + CatBoost auto-with-correction parity rows for symmetry with the XGBoost row. ~80 LOC.
>
> ## Session 2026-05-05 morning — 6 PRs opened (#32–#37), all reviewed
>
> ### Morning PRs (still open, all reviewed clean)
>
> | PR | Branch | Scope | Reviewers | Verdict |
> |---|---|---|---|---|
> | #36 | `fix/Tgui-auto-resume-tooltip-speed-wording` | P0 user ask: refresh Bayesian crash-resume tooltip — drop stale T-41 planning-era speed claims (`PLS 8x slower`, `XGBoost 1.4x slower`) that became wrong after T-42's per-trial-write hoist. Replace with `~1.06x of in-memory in benchmarks`. Adjacent stale code comment in `unified_bayesian.py:2232` updated. | None — user-approved wording | READY_TO_MERGE |
> | #32 | `fix/T20-imbalanced-auto-parity-row` | P1a: closes deferred Codex MEDIUM from PR #22. Adds `test_xgboost_binary_classification_parity_auto_with_correction` mirroring the explicit `class_weight` row but with `imbalance_method='auto'` and `expect_stdout_marker='applying class_weight'`. | DeepSeek + GLM cross-family + pr-test-analyzer | All 3 READY_TO_MERGE / Approve, zero HIGH (1 deferred MEDIUM: LightGBM/CatBoost auto-with-correction symmetry rows) |
> | #33 | `fix/T20-pipeline-params-strip-bare-njobs` | P1b: closes deferred Codex MEDIUM. `_PIPELINE_PARAMS` was incorrectly stripping bare `n_jobs` from user model params; downstream `setdefault('n_jobs', -1)` then silently overrode user-set determinism (`n_jobs=1` → `-1`). Sister site at `unified_bayesian.PIPELINE_PARAMS:88`. KNOWN ISSUE comment for `verbose` deferred. Round-2 fix-of-fixes (`b3c5994`) added 2 structural pin tests. | DeepSeek + GLM + Codex (round 1: convergent HIGH on sister site → fixed); pr-review-toolkit suite (round 2: code-reviewer clean, silent-failure-hunter HIGH on `search.py:4827 PIPELINE_META_PARAMS` verbose, pr-test-analyzer 2 HIGHs deferred) | READY_TO_MERGE post fix-of-fixes; 2 pr-test-analyzer HIGHs (behavioral test + architectural test) deferred |
> | #34 | `fix/Tridge-classifier-dead-code-delete` | P1c: deletes unreachable `RidgeClassifier` branch at `models.py:471-473`. No entry point can select Ridge for classification (registry / tier / GUI / `is_valid_model` all exclude it). Defensive scaffolding for a feature that never shipped. | code-reviewer | Clean (3 LOW informational) |
> | #35 | `fix/Tplsda-ndim-fallback-codegen-port` | P1d: ports the `ndim>2` fallback from `models.PLSTransformer.transform` to the codegen's inline `PLSTransformer.transform`. Currently unreachable (PLS-DA always fits 1D y); future-proofs against T-17 (PLS-2). | code-reviewer | Clean (1 LOW about pre-existing 111-char line that mirrors models.py) |
> | #37 | `docs/morning-2026-05-05-session-end` | Morning docs: PROJECT_STATUS + SESSION_LOG with the 5 morning PRs and three process lessons. | None (pure docs) | n/a |
>
> ---
>
> ## Session 2026-05-05 — refined-model imbalance dispatcher (PUSHED)
>
> **Three commits to `origin/main`:**
> - `4dcedbc` round-1: stops the crash, but had silent CatBoost/XGBoost gap
> - `4bd5d30` docs: round-1 root-cause writeup in SESSION_LOG
> - `c395317` round-2: closes silent gap; `auto_class_weights='Balanced'` for CatBoost; `compute_sample_weight('balanced', y)` threaded for XGBoost at 3 fit sites; MLP message via `_log_progress` (visible in PyInstaller bundle); defense-in-depth at 4 layers
>
> **User report**: running any classifier from Model Development → Refine raised `ValueError: Unknown resampling method: class_weight` from `imbalance.py:252`. Broken for all classifier model types since T-19 (`1d2bf6d`) made the auto→class_weight resolution path more reachable in saved-and-reloaded model configs.
>
> **Round-1 fix (`4dcedbc`)**: GUI dispatcher routes `class_weight` via `model.set_params(class_weight='balanced')` with `hasattr` guard + clear `loaded_imbalance_method=None`; `ClassificationResampler` no-ops on sentinels as defense in depth.
>
> **Round-1 review (DeepSeek V4 Pro Max + Codex, parallel)**: convergent HIGH — `hasattr(model, 'class_weight')` returns False for CatBoostClassifier (uses `auto_class_weights`) and XGBClassifier (only `sample_weight` at fit). Both classifiers silently trained UNWEIGHTED — wrong numbers, no crash. Violated the "Refine reproduces Results" contract.
>
> **Round-2 fix-of-fixes (`c395317`)**: discriminator now branches on `model_name` first (PLS-DA / CatBoost / hasattr / sample_weight fallback / MLP / no-support); `use_sample_weight_for_classification` flag drives `compute_sample_weight('balanced', y)` at 3 fit sites; `warnings.warn` → `self._log_progress` for PyInstaller-bundle visibility; `_fit_with_early_stopping` extended with optional `sample_weight=None` kwarg; explicit `'auto'` guard added to `_needs_resampling_pipeline`; `build_imbalance_transformer` sentinel guard extended to regression task.
>
> **Round-2 review (GLM 5.1 max + Codex, convergent)**: both READY_TO_PUSH / CONFIRM_READY_TO_PUSH. Codex's call-graph trace verified `'model'` step name consistent across all 3 GUI pipeline-build paths, sample_weight reaches XGBClassifier.fit() through `**fit_kwargs`, codegen export-path already handles CatBoost/XGBoost analogously (no parallel bug), and saved-config backward compatibility is intact. One stale-comment nit (RidgeClassifier mentioned under sample_weight fallback though code routes via class_weight branch) deferred per engineering-polish rule.
>
> **Test sweep**: `tests/test_imbalance.py` 62 passed + 1 skipped (was 54 + 1 pre-session; +4 round-1 sentinel no-op + +4 round-2 regression-task sentinel). py_compile clean.
>
> **Generalisable lesson** (filed in SESSION_LOG): `hasattr(model, '<flag>')` alone is NOT a sufficient discriminator for model-parameter flags across the sklearn ecosystem. CatBoost / XGBoost / MLP all need different routing. Canonical pattern: branch on model_name first for known per-library kwargs → hasattr → inspect.signature(model.fit) for sample_weight-only → warn for no-support. Mirrored in `search.py:4411-4448` and now in `_run_refined_model_thread`.
>
> ---
>
> ## Session 2026-05-04 (post-merge — all 8 outstanding PRs squash-merged to main; post-merge regression sweep **360 passed, 1 skipped**, no regressions).
>
> ## Session 2026-05-04 — 8 PRs MERGED to main
>
> | PR | Squash-merge SHA | Ticket / scope |
> |---|---|---|
> | #21 | `35c3fc9` | T-14b: PyInstaller + GUI version-drift sites; cut 0.5.0b2 |
> | #25 | `1d18a50` | T-29b: per-run-log visibility for `spectral_predict.*` warnings |
> | #26 | `cc8c63a` | P3: defensive `.ravel()` on classification CV-fold predict |
> | #27 | `9eabce0` | T-30b: print triage in `calibration_transfer.py` |
> | #22 | `0e7eeb6` | T-20: saved-model ↔ exported-script parity test foundation |
> | #29 | `9ff30f7` | T-20b: extend parity matrix to LightGBM + CatBoost (replaces auto-closed #23) |
> | #30 | `94f57ba` | codegen-fix: scalarise predictions across plain + imbalance-aware CV pooling (replaces auto-closed #24; closed the codex-caught bug class fully) |
> | #31 | `52ad273` | T-20c: PLS-DA parity row + MLP imbalance no-op marker (replaces auto-closed #28) |
>
> **Stack-rebase note**: GitHub auto-closed PRs #23 / #24 / #28 when their stacked base branches were deleted by parent squash-merges. Each was rebased locally onto post-parent main and reopened as a new PR (#29 / #30 / #31) targeting main. All squashed-merge clean.
>
> **Post-merge regression sweep**: **360 passed, 1 skipped** across the consolidated 13-file regression sweep + `test_t14b_pyinstaller_and_gui_version_drift` + `test_t20_saved_model_export_parity`. Math: 327 base + 6 T-29b + 8 T-14b + 19 T-20-family = 360. Zero failures, zero integration issues. All branches deleted from origin + locally.
>
> **Merge-cascade lesson** (filed for future stacked-merge sessions): when GitHub auto-closes a stacked PR after its base branch is deleted, the PR cannot be reopened or have its base changed via `gh pr edit --base`. The clean recovery is: rebase the head branch locally onto post-parent main with `git rebase --onto origin/main <last-parent-commit> <head>`, force-push, then `gh pr create --base main --head <head>` to open a new PR. The new PR number replaces the old one in the merge queue.
>
> ---
>
> ## Session 2026-05-04 codex-review round (pre-merge)
>
> ## Session 2026-05-04 codex-review round
>
> Dispatched 8 codex-reviewer agents in parallel against PRs #21–#28. Findings + dispositions:
>
> | PR | Codex verdict | Fix-of-fixes |
> |---|---|---|
> | #21 T-14b | 2 MEDIUM | **Both applied** (`ca79f10`): build-guide doc-drift on `BUNDLED_APP_BUILD_GUIDE_PY312.md:22`, GUI version-drift test rigor (added bare `0.5.0` to ban-list, replaced repo-wide `_DASP_VERSION` substring check with two specific f-string-interpolation assertions on each display site). |
> | #22 T-20 | 1 HIGH + 2 MEDIUM + 1 LOW | HIGH addressed via docstring (`250559d`) — the file claimed T-32 coverage but no row exercises a resampler+sample_weight path; T-32 is actually pinned by `tests/test_t32_sample_weight_resampling.py`. Two MEDIUMs deferred (note in PR #22 description): `_PIPELINE_PARAMS` strips bare `n_jobs` then re-injects `-1` for XGB/LGBM (multi-threaded determinism drift potential); auto-resolves-to-class_weight imbalanced row missing. |
> | #23 T-20b | CLEAN | None. Convergent with prior DeepSeek + GLM verdicts. |
> | #24 codegen-fix | **1 MEDIUM — real production bug** | **Applied** (`dfeb27d`). Codex traced the call graph and found that the imbalance-aware classification CV path (`code_generator._render_cross_validation_with_imbalance` at `:1761-1769`) bypasses `templates/validation.py` entirely and had the SAME unfixed pooling bug. **CatBoost multiclass + class_weight / auto / resampling exports were ALL silently broken** before this catch (the codegen routes via `code_generator.py:1435-1436` whenever `imbalance_method` is set). Cross-family DeepSeek + GLM both missed this because they reviewed only `templates/validation.py`. The same `.ravel()` fix mirrored to the imbalance-aware path closes the bug class fully. |
> | #25 T-29b | CLEAN (partial) | None. Codex's local PowerShell sandbox tripped Windows ConstrainedLanguage-mode errors during investigation — the verdict is real but coverage on the specific concerns I asked about (file-handle leak, attach/detach thread-safety, log-record propagation) was incomplete. Prior DeepSeek + GLM cross-family review covers these more thoroughly. |
> | #26 P3 .ravel() | CLEAN | None. Verified `.ravel()` semantics against every supported classifier (PLS-DA / RF / LightGBM / XGBoost / CatBoost / SVC / MLP / RidgeClassifier / NeuralBoostedClassifier). LOW residual risk only. |
> | #27 T-30b | READY_TO_MERGE | None. Codex confirmed `setup_run_logger()` tees backend `print()` output into the per-run log via stdout capture (`run_logging.py:240-263`), so the kept user-facing CTAI/NS-PFCE prints are visible in run logs. Logger setup correct; package-tree `WARNING` floor at `run_logging.py:359/368` keeps the new `logger.debug` calls silent in normal app logs (intentional). |
> | #28 T-20c | 1 MEDIUM + 1 LOW | Both **deferred**. MEDIUM: `_split_pls_da_params` silently drops `scaler__*` keys (codegen drops at `code_generator.py:1270`); fixing requires the codegen to actually route scaler params, a production change beyond test-PR scope. LOW: MLP stdout marker assumes stdout emission, would not catch a future shift to `warnings.warn` / `logger.warning`. |
>
> **Tally**: 1 HIGH (defused via docstring), 5 MEDIUM (3 applied as fix-of-fixes, 2 deferred), 2 LOW (deferred). One real production bug closed on PR #24 — codex's call-graph trace was the highest-leverage finding of the round.
>
> **Why this matters**: cross-family review (DeepSeek + GLM) is great at within-file logic, but codex's call-graph reasoning caught a bug that lived across two file boundaries — the `_render_cross_validation` dispatcher routing to a different inline template under `imbalance_method`. **Cross-family + codex are complementary review structures**; either one alone would have shipped a broken-for-CatBoost-multiclass-imbalance regression.
>
> **Branch tips after fix-of-fixes (round 2)**:
> - PR #21: `ca79f10` (codex MEDIUM × 2)
> - PR #22: `250559d` (docstring T-32 clarification)
> - PR #24: `dfeb27d` (real bug closure on imbalance-aware CV path)
> - PR #28: `b6960a1` (unchanged from earlier MLP warning marker addition)
> - PR #23, #25, #26, #27: unchanged (codex CLEAN)
>
> ## Session 2026-05-04 overnight — 3 additional PRs (NOT MERGED)
>
> ## Session 2026-05-04 overnight — 3 additional PRs (NOT MERGED)
>
> | PR | Branch tip | Base | What it ships | Reviews |
> |---|---|---|---|---|
> | #27 | `4ddffe6` | `main` | T-30b: print() triage in `calibration_transfer.py` (66→41 prints kept; CTAI's debug-instrumentation block demoted to `logger.debug`, explicit "=== Debug Information ===" header deleted, user-facing CTAI summary + NS-PFCE progress messages preserved). `nsga2_search.py` untouched — all 43 prints already properly verbosity-gated. | DeepSeek + GLM cross-family complete. **Both READY_TO_MERGE — no findings.** GLM noted one LOW about pre-existing dual-emission pattern at `nsga2_search.py:1804-1805` (out of T-30b scope). |
> | #26 | `ec7d90f` | `main` | P3 defensive `.ravel()` on classification CV-fold predict in `templates/validation.py:160`. Mirrors the regression template's existing pattern (line 103). Closes the GLM MEDIUM hygiene finding from PR #24's review. | No formal cross-family review (one-line hygiene change; DeepSeek's broader codebase logger-pattern verification on T-30b indirectly validated the defensive style). |
> | #28 | `18548d2` | `fix/Tcv-...` (PR #24) | T-20c: PLS-DA parity row + MLP imbalance-is-no-op marker. PLS-DA pins the three-step Pipeline contract (PLS scores → StandardScaler → LogisticRegression) end-to-end; MLP marker pins that `imbalance_method='class_weight'` is a no-op for MLPClassifier. Tip `18548d2` applied DeepSeek L1 (explicit `lr__random_state: 42` in test params, decouples test from codegen's internal `lr_defaults`). | DeepSeek + GLM cross-family complete. Both **READY_TO_MERGE.** GLM 1 LOW (pre-existing `_lr_filtered` inspection-filter pattern, out of scope). DeepSeek 2 MEDIUM + 2 LOW: L1 applied as fix-of-fixes; M1 (stdout marker for codegen-path proof) and M2 (port `ndim>2` fallback to inline `PLSTransformer` for future PLS-2) deferred per standing engineering-polish rule; L2 withdrawn by reviewer. |
>
> **Session timing:** all overnight branches pushed by ~01:00 local. Cross-family reviews dispatched in parallel for T-30b (returned READY_TO_MERGE × 2) and T-20c (GLM returned, DeepSeek pending).
>
> **Working tree at session-end snapshot:** clean on `main` after this doc commit. Untracked artifacts to ignore: `.review-tmp/` and the weirdly-named `C\357\200\272UsersmsponAppDataLocalTempopencodesearch_t30.py` left over from the previous session.
>
> **Test sweep state at session-end:** all branches green at `327 passed + 1 skipped` on the consolidated 13-file regression sweep. T-20 parity matrix at **19 rows** with PLS-DA + MLP marker added (was 17 on the T-20b branch).
>
> ### P2 — Ridge classifier export investigation (filed; deferred)
>
> Outcome: **dead code, NOT a bug fix.** `code_generator._resolve_model_ctor_class` returns plain `'Ridge'` for `task_type='classification'` (not `'RidgeClassifier'`), and `_resolve_default_param_key` likewise has no Ridge classification mapping. Initially looked like a real export bug.
>
> But the runtime branch in `models.py:471-473` that builds `RidgeClassifier(random_state=42, **params)` is **unreachable**: `model_registry.CLASSIFICATION_MODELS` excludes Ridge; `model_config.CLASSIFICATION_TIERS` excludes Ridge from every tier; the GUI's `valid_models_classification` list at `spectral_predict_gui_optimized.py:32479` and `:33425` explicitly excludes Ridge; and `is_valid_model('Ridge', 'classification')` returns False. The runtime code path is defensive scaffolding that no entry point selects.
>
> Per the user's "engineering polish defers" rule, adding Ridge classification as a real feature requires registry + tier + export-resolver + DEFAULT_PARAMS + test additions and is a feature, not a bug. Filed here as a future cleanup ticket: either remove the `models.py:471-473` runtime dead code (smallest change, eliminates inconsistency) or add Ridge classification end-to-end as a real feature.
>
> ---
>
> ## Session 2026-05-04 — five PRs in flight (NOT MERGED — user opens/merges)
>
> | PR | Branch tip | Base | What it ships | Reviews |
> |---|---|---|---|---|
> | #21 | `70c55b7..1ef393f` | `main` | T-14b: PyInstaller + GUI version-drift sites closed (`build_installer_py312.py`, `installer/spectral_predict_py312.iss`, `version_info.txt`, GUI title bar at `:2556`/`:4663`); cut `0.5.0b2`. CHANGELOG backfill. 8 regression tests in `tests/test_t14b_pyinstaller_and_gui_version_drift.py`. | DeepSeek + GLM cross-family, 2 convergent MEDIUMs applied as fix-of-fixes. Verdict CLEAN. |
> | #22 | `94beab1..ec7f8a7` | `main` | T-20: saved-model ↔ exported-script reproducibility test foundation. 9 parity tests in `tests/test_t20_saved_model_export_parity.py` covering PLS / RandomForest / XGBoost × regression/classification × imbalance variants (no-imbalance, class_weight via sample_weight at fit, auto-resolves-to-None on balanced data). | DeepSeek + GLM cross-family + pr-review-toolkit suite (test-analyzer + code-reviewer + comment-analyzer + silent-failure-hunter) + GLM sanity-recheck. 3 HIGH + 6 MEDIUM applied as 2 fix-of-fixes commits. Verdict CLEAN. |
> | #23 | `2d1e283..a12e843` | `fix/T20-...` | T-20b: extends matrix to LightGBM (4 rows) + CatBoost (4 rows). 1 row was xfail(strict=True) documenting a real codegen bug T-20b surfaced — fixed in PR #24 below; xfail flipped to passing. | DeepSeek + GLM cross-family, 1 convergent MEDIUM applied (misleading `n_jobs=-1` codegen-injection comment). Verdict CLEAN. |
> | #24 | `464c5ef..685553e` | `fix/T20b-...` | Codegen fix: `templates/validation.py` CV pooling block scalarises predictions via `np.ravel(...)[0]` so `Counter` can hash CatBoost multiclass's `(n, 1)`-shape `predict()` return. T-20b's xfail flipped to passing. | DeepSeek + GLM cross-family. 0 HIGH, 0 MEDIUM, 1 LOW polish applied (comment accuracy — `np.ravel` on 0-d scalar isn't technically "no-op", just behaves correctly). Verdict CLEAN. |
> | #25 | `3b8b748..c5b2c66` | `main` | T-29b: per-run-log visibility for `spectral_predict.*` warnings. `setup_run_logger` now also attaches the per-run `_SafeRotatingFileHandler` to the `spectral_predict` logger (sharing the same instance — no double file-open on Windows). Defense against module-reload double-attach mirrors `setup_app_logger`'s existing pattern. 6 regression tests added (5 initial + 1 fix-of-fixes module-reload test). | DeepSeek + GLM cross-family complete. GLM CLEAN. DeepSeek caught 2 MEDIUMs (collapsing to one finding: existing double-attach test exercised the wrong guard, dedup scan was effectively dead code under that scenario) — fix-of-fixes `c5b2c66` added a module-reload test mirroring `setup_app_logger`'s pattern. Verdict CLEAN. |
>
> **Test-count progression** (consolidated 13-file regression sweep + per-PR additions, no regressions at any layer):
> - Pre-session baseline: 327 passed + 1 skipped.
> - After T-14b (PR #21): 335 passed (+8).
> - After T-20 (PR #22): 336 passed + 1 skipped (+9).
> - After T-20b (PR #23): 343 passed + 1 skipped + 1 xfailed (+7 +1xfail).
> - After codegen fix (PR #24): 344 passed + 1 skipped (xfail flipped to passing).
> - After T-29b (PR #25, independent of T-20 chain): consolidated sweep alone is 332 + 1 skipped (5 new T-29b tests + 327 baseline).
> - Combined when all 5 PRs land + rebase: ~349 passed total.
>
> **PR ordering** when reviewers/CI clear: #21, #22, #25 are independent (any order). Then #23 rebases onto post-#22 main. Then #24 rebases onto post-#23 main.
>
> **Codegen bug surfaced + fixed in same session:** T-20b's CatBoost multiclass parity test caught a real bug — `CatBoostClassifier.predict()` returns shape `(n, 1)` for multiclass, and `templates/validation.py:163-176`'s `Counter(preds_per_sample[i])` couldn't hash 1-element ndarrays. PR #24 ships the one-line scalarisation fix. **The regression-net loop (T-20 → T-20b → codegen-fix) closed within the same session.** This is exactly the value the user elevated T-20 coverage to highest priority for.
>
> **Deferred follow-up tickets (next-session priority):**
>
> - **T-30b** ✅ shipped overnight as PR #27 — `calibration_transfer.py` triaged (66 → 41 prints kept; 1 deleted, 20 demoted to `logger.debug`). `nsga2_search.py` confirmed already-clean (all 43 prints properly verbosity-gated).
> - **Ridge classifier export investigation** ✅ investigated overnight — dead code, NOT a bug fix. See P2 section above. Future cleanup ticket: either remove `models.py:471-473` runtime dead code or add Ridge classification as a real feature.
> - **PLS-DA parity row** ✅ shipped overnight on the T-20c branch (`fix/T20c-pls-da-parity-row`, tip `6bc2b20`). Pipeline-matching scaffolding done; runs at `rtol=1e-3, atol=1e-6` like the rest of the classification matrix.
> - **MLP parity marker test** ✅ shipped overnight on the same T-20c branch. Pins MLP+`class_weight`=no-op contract via predict_proba parity (catches the same bug class as T-19's source-inspection tests via numerical equivalence).
> - **Per-fold metric defensive .ravel** ✅ shipped overnight as PR #26.
>
> **Process lessons captured this session:**
> 1. **Stacked-PR pattern with chained bases successfully kept review cost linear.** PR #22 → #23 → #24 form a foundation → extension → bug-fix-from-extension chain. Reviewers see only the new content per layer (`git diff base..HEAD`); after each merge, child PRs rebase onto post-merge main and their bases auto-update. Without stacking, T-20b would have to wait for T-20 to merge before being reviewable; same cascade for the codegen fix.
> 2. **`predict()` vs `predict_proba()` in classification parity tests is load-bearing.** Hard labels via `predict()` are quantised — they hide probability drift up to whichever class boundary is nearest. DeepSeek empirically demonstrated 0.296 probability drift while hard labels remained identical on T-20's class_weight rows. Switched to `predict_proba` with rtol=1e-3, atol=1e-6 for classification (regression keeps `predict` with rtol=1e-5, atol=1e-8).
> 3. **The DEFAULT_PARAMS merge in `code_generator._render_model` matters for parity tests.** PLS gets `scale=False` injected (chemometrics convention — 24+ runtime sites hardcode this); RandomForest gets `random_state=42`; XGBoost/LightGBM get `n_jobs=-1` via setdefault. In-process test models must mirror the runtime's param assembly or parity assertions fail-for-the-wrong-reason. The fix-of-fixes for T-20 had to thread `fit_kwargs` through `_run_parity` to match the codegen's `sample_weight=compute_sample_weight('balanced', y)` for XGBoost class_weight (XGBoost has no native class_weight kwarg).
> 4. **`@pytest.mark.xfail(strict=True)` is the right marker when a test catches a real bug that's out of the current PR's scope.** A regular test would block the suite. xfail-non-strict silently ignores XPASS and lets the marker rot. xfail-strict turns the failure into a tracking marker that's actively forced to be removed when the underlying bug is fixed (XPASS triggers test failure). T-20b's CatBoost multiclass row used this pattern; PR #24's codegen fix flipped it.
>
> ---
>
> ## Earlier session — 2026-05-03 (quick-wins batch — 5 PRs MERGED to main, post-merge tip `1d3eba0`)
>
> | PR | Ticket | Squash-merge SHA on main |
> |---|---|---|
> | #16 | T-50 | `348bdd8` |
> | #17 | T-14 | `134fabb` |
> | #18 | T-29 | `d5c6900` |
> | #19 | T-32 | `34d52fe` |
> | #20 | T-30 | `1d3eba0` |
>
> Post-merge regression sweep: **327 passed + 1 skipped** across the 13-file consolidated sweep (added `test_t32_sample_weight_resampling.py` + `test_scoring.py` to the original 11-file). Zero failures, zero integration issues. Branches deleted from origin + locally.
>
> ## Quick-wins batch — pre-merge state (preserved for audit trail)
>
> All branched off main (post-PR #15 merge), all bug-fix-class except T-30 (hygiene-class with CAUTION caveat applied). All tested. Test-merge with main: clean for all five. **Cross-family review trail per ticket: DeepSeek V4 Pro Max + GLM 5.1 in parallel via opencode-call → fix-of-fixes commit closing all HIGH + surgical MEDIUM findings → consolidated GLM 5.1 sanity-pass over the 4 fix-of-fixes commits → all 4 returned READY_TO_MERGE with no HIGH or MEDIUM findings.** Then squash-merged via PRs #16-#20.
>
> | Ticket | Branch | Tip (post fix-of-fixes) | LOC (cumulative) | Review verdict | Surface |
> |---|---|---|---|---|---|
> | T-50 | `fix/T50-cleanup-stale-optuna-sqlite` | `d09b241` | +362/-5 | DeepSeek READY_TO_MERGE / GLM READY_WITH_REVISIONS → fix-of-fixes `d09b241` → GLM recheck READY_TO_MERGE | Auto-cleanup stale Optuna SQLite trial archives at app startup. `cleanup_old_sqlite_files()` in `run_state.py` + GUI/CLI wiring. 6 regression tests. Fix-of-fixes closed 3 surgical MEDIUMs (`.sqlite3-journal` sibling, `is_relative_to` defense-in-depth, `logger.debug(exc_info=True)` in wirings). DeepSeek had one false positive on `bytes_freed` ordering — verified safe. |
> | T-14 | `fix/T14-version-string-single-source` | `4366165` | +118/-10 | DeepSeek READY_WITH_REVISIONS (3 HIGH) / GLM READY_WITH_REVISIONS → fix-of-fixes `4366165` → GLM recheck READY_TO_MERGE | Initial fix closed 2 stale version-string sites (`report.py:139` + `code_generator.py:373`). DeepSeek's review caught **3 additional HIGH version-drift sites in the same bug class**: `model_io.py:48` (local `__version__` redefine baked into `.dasp` metadata), `templates/header.py:8` (`Generated by Spectral Predict v3` in every exported script — major-version-wrong claim), `export_bundle.py:243` (`Version: 1.0` in every README). Fix-of-fixes closed all 3 + GLM's empty-results-footer MEDIUM + strengthened tests with negative assertions against re-hardcoding. |
> | T-29 | `fix/T29-scoring-no-bare-except` | `3de1f25` | +187/-55 | DeepSeek "ship with minor changes" (2 HIGH) / GLM "ship-with-nits" (3 MEDIUM) → fix-of-fixes `3de1f25` → GLM recheck READY_TO_MERGE | Initial fix replaced 3 bare `except:` in `compute_imbalance_metrics` with `except Exception:` + `logger.warning()` (kept 0.0/None sentinels per chemometrics-relevance rule). Fix-of-fixes closed DeepSeek HIGH-1 (wrapped `balanced_accuracy_score` for symmetry), enriched all 5 warning messages with `n=`, `n_classes=` diagnostic context, parametrized test across all 5 metrics, switched KeyboardInterrupt test to `monkeypatch.setattr`. **DEFERRED follow-ups**: DeepSeek HIGH-2 (per-run-log visibility — affects ALL `spectral_predict.*` warnings, not T-29-specific; separate ticket); dead-code finding (`compute_imbalance_metrics` has zero callers — fix correct but user-visible impact is theoretical). |
> | T-32 | `fix/T32-sample-weight-resampling-mismatch` | `e51c89c` | +215/-9 | DeepSeek "ship-ready" / GLM "ship" → fix-of-fixes `e51c89c` → GLM recheck READY_TO_MERGE | Initial fix threaded `y_train_for_model` through `_run_single_fold`'s classification-sample-weight branch (search.py:~4109) so X/y/sample_weight stay length-consistent post-resample (was crashing with sklearn ValueError on every SMOTE+sample_weight-supporting-classifier combination). Fix-of-fixes extended the same pattern to the early-stopping branch (architectural-landmine closure), added `RandomUnderSampler` test (covers reduction direction), fixed wrong numbers in test docstring. **DEFERRED follow-up**: regression-side imbalance path may have similar latent bug for unusual combinations (DeepSeek LOW). |
> | T-30 | `fix/T30-remove-debug-prints` | `fd56111` | +0/-36 | DeepSeek NEEDS_FIXES (1 HIGH) / GLM 1 MEDIUM → fix-of-fixes `fd56111` → GLM recheck READY_TO_MERGE | Initial sweep removed 7 `[DEBUG]`-prefixed prints in `search.py`. Both reviewers caught a missed `[PLS-DA DEBUG]` block (search.py:4192-4208) — same class, slipped my regex due to bracket-text variation. Fix-of-fixes removed it. **DEFERRED follow-up**: `calibration_transfer.py` (66 prints) and `nsga2_search.py` (42 prints) are same class but bigger triage surfaces requiring user judgment per file. |
>
> **Process lessons captured this batch:**
> 1. **Commit before yielding control.** Long pytest runs / agent dispatch / `git push` can collide with parallel sessions or hooks that swap branches and clobber uncommitted working-tree work. Memory entry `feedback_commit_before_yielding_control.md` filed. T-50 lost ~30 min to this; T-14/T-29/T-32/T-30 followed an "edit → compile → commit IMMEDIATELY → only then run long sweeps" pattern with no further loss.
> 2. **`pytest | tail` masks pytest's exit code** because the pipe inherits `tail`'s exit. T-29's "atomic burst" (compile && pytest && commit && push) silently committed despite a failing test because the `| tail -10` made pytest always exit 0. Workaround: drop the `| tail`, use `2>&1 | grep -E "passed|failed"` if length matters, or split into separate steps. T-32 onwards used the corrected pattern.
> 3. **DeepSeek false positive on bytes_freed ordering** (T-50 review): claimed `bytes_freed += sibling_size` ran before `sibling.unlink()`. Code reading shows it ran after — the increment is at the line *after* unlink in the same try block, so unlink-OSError correctly skips the increment via except. Cross-family review caught one real bug class GLM missed and one false positive — net positive but reviewers aren't infallible.
>
> **What's next (open-ticket roster after this batch):** T-20 (saved-model ↔ exported-script reproducibility test, pairs with T-19 freshness) per the original continuation prompt suggestion. Then the bigger arcs: T-31 multi-class SIMCA (1-2w), T-16 reframed model-comparison machinery survey, T-17 PLS-2 (2-3w), T-01 reframed (~2-3d).
>
> ---
>
> ## T-19 reframed — MERGED via PR #15 at `1d2bf6d` (2026-05-03)
>
> Tip: `0b1d4a2` (commits: `3c63ffe` initial bug fix → `765a82f` GLM MEDIUMs closed → `bdeb735` DeepSeek residual multiclass test → `9cea07c` PROJECT_STATUS+SESSION_LOG → `0b1d4a2` **Auto mode**). Pushed to origin. **NOT MERGED — user opens PRs.**
>
> **Two parts shipped together:**
> 1. **Bug fix** (commits `3c63ffe`/`765a82f`/`bdeb735`): exported scripts under `imbalance_method='class_weight'` were broken for CatBoost/XGBoost/MLP. Library-aware dispatch in `code_generator._render_model()`.
> 2. **Auto mode** (commit `0b1d4a2`): new `'auto'` dropdown option + helper + runtime-resolution at all three search entry points + codegen-time emission of post-data-load resolution block. Resolves to `class_weight` or `None` based on `detect_class_imbalance(y, threshold=3.0)`. The user's framing memo `project_t19_user_framing.md` calls Auto mode the priority — without it T-19 isn't done. Per-model UI dropdowns (separate selectors for `is_unbalance` / `scale_pos_weight` / `auto_class_weights`) remain deferred — explicit user direction that those add UI surface without correctness/speed improvement.
>
> **Bug class:** exported scripts under `imbalance_method='class_weight'` were broken in three ways:
> - **CatBoost**: `class_weight='balanced'` injected into `__init__` → `TypeError: unexpected keyword argument 'class_weight'`. Hard crash on instantiation.
> - **XGBoost**: `class_weight='balanced'` accepted by `__init__` but **silently ignored at fit time** (XGBoost has no class_weight kwarg in its loss). User runs the export and gets unweighted predictions while believing imbalance was handled.
> - **MLP** (StandardScaler-wrapped path, GLM finding): same `TypeError` as CatBoost — MLPClassifier has no `class_weight` kwarg.
>
> **Fix:** library-aware dispatch in `code_generator._render_model()` catch-all else branch + StandardScaler-wrapped path:
> - CatBoost → `auto_class_weights='Balanced'` (canonical CatBoost balanced-loss kwarg)
> - XGBoost → `sample_weight=compute_sample_weight('balanced', y)` threaded into per-fold and final fit() calls (mirrors `search.py:4418-4435` runtime path; uniformly correct for binary AND multiclass)
> - MLP → no class_weight injection (mirrors runtime fallback at `search.py:4439-4444`; sklearn≥1.7 sample_weight at fit() is above the pyproject 1.5 floor)
> - Other libraries (LightGBM/RandomForest/sklearn LR/SVC/NeuralBoosted) keep `class_weight='balanced'` — works natively.
>
> **Conditional fit_kwargs emission** (GLM finding): non-XGBoost paths under any imbalance method no longer emit `fit_kwargs = {}` and `**fit_kwargs` noise into generated code. Plain `.fit(X, y)` returns when sample_weight isn't being threaded.
>
> **Indentation discipline noted in code:** `sample_weight_block_cv` prefixes 4 spaces (lands inside for-loop body); `sample_weight_block_final` prefixes 0 spaces (module-level). Comment in `_render_cross_validation()` warns against "normalizing" prefixes during refactor.
>
> **Auto mode mechanics (commit `0b1d4a2`):**
> - GUI: `'auto'` is the 9th classification option in the imbalance dropdown. Tooltip explains the 3:1 trigger.
> - Helper `imbalance.resolve_auto_imbalance(y, task_type, threshold=3.0)` returns `('class_weight', info)` on imbalanced data, `(None, info)` on balanced data, `(None, {})` for regression. Single source of truth.
> - Runtime resolution at the entry of `run_search`, `run_nsga2_search`, `run_unified_bayesian`. Mutates `imbalance_method` to either `'class_weight'` or `None` so downstream code paths fire correctly without needing to know about `'auto'`. Audit print on either path: `[Auto imbalance] ratio X.X:1; applying class_weight` or `[Auto imbalance] ratio X.X:1 (below 3:1); no correction`.
> - Code generator: emits a runtime-resolution block at the end of `_render_imbalance_handling()`. Generated scripts mutate `IMBALANCE_METHOD` post-data-load. Per-library kwargs (`auto_class_weights` for CatBoost, `sample_weight` for XGBoost via `fit_kwargs`) are baked at codegen time as if `'auto'` were `'class_weight'`; on balanced data the runtime resolution prevents the `sample_weight` block from firing and the baked `class_weight='balanced'` is mathematically a no-op (uniform per-class weights).
> - `validate_classification_config` accepts `'auto'` the same as `'class_weight'`.
> - **Resolution scope: once at run-entry, against global y (not per-fold).** Stratified CV preserves global ratios tightly enough that per-fold drift rarely flips the decision; per-fold check is feasible follow-up if needed.
>
> **Test coverage:**
> - **17 Auto-mode tests** in `tests/test_t19_auto_mode.py`: resolver contract (imbalanced/balanced/regression/custom-threshold) + codegen emission per library (CatBoost auto_class_weights, XGBoost sample_weight, RF/LightGBM class_weight, MLP no-injection) + end-to-end `exec()` on imbalanced AND balanced synthetic data for each library.
> - **18 class_weight tests** in `tests/test_t19_class_weight_per_library.py` — per-library kwarg emission contracts + end-to-end exec (binary + 3-class for XGBoost) + MLP no-injection + SVC keeps class_weight + non-XGBoost noise-free + negative controls.
> - **Consolidated regression sweep**: 336 passed + 2 skipped (12-file sweep including `test_imbalance.py`). Pre-Auto baseline was 282+1; +54 from imbalance suite when included. No regressions.
>
> **Cross-family review trail (review of bug-fix portion completed; Auto mode under review):**
> 1. **GLM 5.1 via opencode-call** (bug-fix review): READY_WITH_REVISIONS — 1 MEDIUM (MLP gap I had wrongly asserted was untouched-correct) + 1 MEDIUM (fit_kwargs noise) closed in `765a82f`. 1 LOW deferred — harmless.
> 2. **DeepSeek V4 Pro Max via opencode-call** (bug-fix review): no HIGH or MEDIUM. 3 LOW/Informational findings — 2 already addressed by `765a82f`; 1 (duplicate `compute_sample_weight` import) deferred per "defer LOW unless surgical." Residual-risk multiclass test added in `bdeb735`.
> 3. **GLM 5.1 via llm-call/z.ai (initial bundle review, before routing correction)**: same MEDIUMs caught — confirms GLM's RLHF brings consistent value across both routing paths.
> 4. **DeepSeek V4 Pro Max review of Auto mode (commit `0b1d4a2`)**: dispatched, in flight at filing time. Findings will be applied as fix-of-fixes commits on the same branch before user opens PR.
>
> **Out of scope (deliberately deferred to follow-up tickets if user wants):**
> - **Per-model imbalance dropdowns** (separate selectors for `is_unbalance` / `scale_pos_weight` / `auto_class_weights`) — UI surgery without correctness/speed gain. User explicitly walked this back: "if we don't need to expose and do as good with just 2 then that si fine." Memory: `feedback_t19_auto_mode_deferred.md`.
> - Custom resampling-aware estimator wrapper for the SMOTE × class_weight combo — fragile under imblearn fit_params; defer.
> - MLP sample_weight threading via sklearn ≥1.7 floor bump — separate version-policy decision.
> - Per-fold Auto-mode resolution (vs the current run-entry global resolution) — feasible if stratified CV class drift becomes a concern; not currently a real failure mode.
>
> **Lessons documented in SESSION_LOG / memory (this date):**
> - GLM caught MLP path that I had explicitly asserted was correct in my review prompt. Cross-family RLHF orthogonality earned its slot.
> - Routing correction for parallel reviewers is surgical, not full-panel re-launch. (Memory: `feedback_parallel_review_reroute.md`.)
> - opencode-call agents leave the working tree on the branch they checked out. Verify branch state after each agent returns — an MLP edit landed on `main` instead of `fix/T19` once.
> - Don't conflate "if we don't need to expose" with "defer Auto mode." The user's walkback was about per-model UI dropdowns specifically; Auto mode itself is the priority. (Memory: `feedback_t19_auto_mode_deferred.md`.)
> - Commit before yielding control to long pytest runs / agent dispatch / git push — parallel sessions can collide and clobber uncommitted work. (Memory: `feedback_commit_before_yielding_control.md`.)
>
> ---
>
> ## Open-ticket re-validation pass (2026-05-02)
>
> User-driven check that all currently-open tickets are "real" chemometrics work, not phantom problems imported from sklearn-pipeline-purity instincts (the recurring failure mode that produced the original T-01/02/03 leakage panic and T-08 CARS framing). Six dispositions amended in `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md` (top-of-doc Amendments block, with inline `> **Amended 2026-05-02:**` markers preserving original verdicts as audit trail):
>
> | Ticket | Original | Amended | Notes |
> |---|---|---|---|
> | T-12 | KEEP | **SUBSUMED by T-45** | T-45 already merged via PR #13 at `2333fae`. T-12 closed without separate work. |
> | T-15 | KEEP | **DROP** | LOGO is a footgun in user's data regime; not field-mandated. Memory: `project_t15_dropped_t16_reframed.md`. |
> | T-16 | KEEP (block-bootstrap) | **REFRAMED — competitive model-comparison machinery survey** | Survey what Unscrambler/SIMCA/OPUS/PLS_Toolbox expose, then scope. |
> | T-19 | KEEP (publication-reproducibility) | **REFRAMED — expose model-native abilities** | Just expose `scale_pos_weight`/`is_unbalance`/`auto_class_weights`/`class_weight`. Simpler scope; ~1-2d. |
> | T-22 | REFRAME | **REFRAMED, DEFERRED** | Varsel stability bootstrap is additive, not bug-class. Defer until concrete need. |
> | T-31 | NEEDS_USER_DECISION | **KEEP** | Multi-class SIMCA confirmed real. Implementation must be true multi-class per Oliveri & Downey 2012, NOT extension of one-class `PCASIMCA`. Memory: `project_t31_simca_confirmed.md`. |
>
> **Net effect on currently-open tickets:** zero phantom chemometrics tickets remain. T-33 through T-49 are all GUI/infrastructure (resume lifecycle, logging, type cleanup) — no chemometrics framing at risk. Updated Top-5 actionable order under master rule + post-amendment dispositions: T-19 reframed (~1-2d) → T-16 reframed survey → T-31 SIMCA (1-2w) → T-17 PLS-2 (2-3w) → T-01 reframed (~2-3d). Detail and lessons in SESSION_LOG entry of same date.
>
> ---
>
> ## Session arc (2026-05-02 overnight)
>
> Five PRs merged onto `main`, one filed ticket deferred as polish-only. Main is at `3870b29`. Working tree clean; no orphan branches.
>
> | Order | Ticket | PR | Merge commit | Verdict |
> |---|---|---|---|---|
> | 1 | Consolidated T-43 + T-49 + T-42 + T-38 (resume reliability + Bayesian write-path cleanup) | #10 | `f063d9e` | DeepSeek READY_TO_MERGE; Codex READY_WITH_REVISIONS — 1 MEDIUM closed |
> | 2 | T-47 (default flip `'never'` → `'auto'`) | #11 | `a29ad2d` | DeepSeek READY_WITH_REVISIONS — 1 MEDIUM + 2 LOW closed |
> | 3 | T-46 (surface `_apply_wal_pragmas` return value) | #12 | `08b20ea` | DeepSeek READY_WITH_REVISIONS — 1 cross-cutting MEDIUM (auto-migration path under T-47's new default) closed; Codex READY_WITH_REVISIONS — 1 LOW closed by same fix |
> | 4 | T-45 (file-handler logger for bundled GUI) | #13 | `2333fae` | Codex READY_WITH_REVISIONS — 1 HIGH (cli.main bypass) + 1 MEDIUM (module-reload dedup) closed; DeepSeek retro confirmed nothing missed |
> | 5 | T-44 (`n_trials_var` typo + sibling `task_type_var`) | #14 | `3870b29` | DeepSeek + Codex paired — DeepSeek surveyed all hasattr-phantom sites and found 1 sibling MEDIUM (`task_type_var` at gui:30859), folded into same PR |
> | — | T-48 (real-Tk integration tests) | — | deferred | Polish, not a bug — existing `_FakeGUI` shim tests already pin all three contracts; per "bugs vs methodology vs polish" rule, defer |
>
> **Test coverage:** 282 passed + 1 skipped across the consolidated regression sweep (`test_t41_bayesian_sqlite_auto_calculator`, `test_t42_write_path_plumbing`, `test_t43_resume_auto_restore`, `test_run_state`, `test_run_logging`, `test_cv_strategy`, `test_unified_bayesian_baseline`, `test_autoscale_bayesian`, `test_cv_pls_clamp`, `test_ensemble_preprocessing`, `test_ensemble_integration`).
>
> ## What's working now (post-session)
>
> - **Resume from sidecar restores ~91 GUI settings + validation indices.** Non-deterministic algorithms (Random, Manual) and hand-picked partitions resume without silent leakage. T-43 + T-49 closed both halves.
> - **Bayesian persistence default is `'auto'`** — most users get crash-resume without thinking about it; benchmark numbers (post-T-42) show all models within 1.06× of in-memory throughput, well under the 1.15× target. Users wanting pre-T-11 in-memory speed set the radio to "Never on".
> - **WAL rejection surfaces visibly** at both SQLite call sites (always-on + auto-migration), via both progress_callback (GUI) and logger.warning (file log). OneDrive/Dropbox-synced data dirs no longer silently halve Bayesian throughput.
> - **Module warnings land on disk.** `<user_data_dir>/dasp.log` (1 MB × 3 backups) catches sidecar corruption, capture-time Tk failures, WAL rejection, T-42 hoist mismatches, etc. — for the bundled GUI AND the `spectral-predict` console script. Pre-T-45, all of these vanished into PyInstaller's `nul` stderr.
> - **Sidecar metadata is honest.** `n_trials_per_model` and `task_type` now reflect actual GUI state instead of always being None / silently inferred-from-y.
>
> ## Things to note for future sessions
>
> - **DeepSeek routing through `opencode-call` IS working** — the agent's earlier pre-emptive refusal claiming opencode-go routing was a misjudgment. When the agent refuses on routing grounds, retry with explicit instruction to attempt the dispatch and report the actual error verbatim. Memory updated at `feedback_deepseek_routing.md`.
> - **`hasattr(self, ...)` phantom-typo class is latent** — DeepSeek's T-44 sibling-survey audited 18 inline `hasattr(self, X) + self.X.get()` sites and found 2 phantoms (`n_trials_var`, `task_type_var`). The other ~230 multi-line `hasattr` patterns in the GUI weren't fully audited — same class of bug could exist there. **Recommended follow-up:** ruff AST visitor that flags `hasattr(obj, name)` where `name` is a string literal that never appears in any `obj.name = ...` assignment in the same class. Closes the class permanently; out of scope for T-44.
> - **The ticket-evaluation rule** (memory at `feedback_chemometrics_relevance_per_ticket.md`): bug fixes that preserve pipeline behavior are fine; pipeline methodology changes (refactoring SNV/autoscale/SG/baseline/variable-selection to chase ML "leakage" objections) need user confirmation; pure engineering polish defers. T-48 was deferred under this rule.
> - **Cross-ticket interactions matter at review time** — T-46's plan was filed when `'never'` was the default, but T-47 (which flipped to `'auto'`) merged earlier in the same session. That reclassified T-46's auto-migration code path from "rare edge case" to "most-traveled WAL surface". DeepSeek caught this in cross-family review by integrating session context. Per-ticket reviewers in isolation would have missed it. Logged in SESSION_LOG.
> - **Two-pair contracts in default-related code** — T-47 surfaced four textual default sites for one field (dataclass default + function default + legacy-sidecar fallback + corruption coercion). The first two are "default for unspecified input" (flip together); the latter two are "fallback for malformed input" (decide separately). Conflating them is a common refactor mistake. Tests pin both halves of the asymmetry. Logged in SESSION_LOG.
>
> ---
>
> ## Detailed entries (most recent first)
>
> **2026-05-02 — T-44 MERGED via PR #14 at `3870b29`.** One-line typo fix at `spectral_predict_gui_optimized.py:25264` — read site was `self.n_trials_var.get()` guarded by `hasattr(self, "n_trials_var")`, but the actual Tk var is `self.n_unified_trials`. The hasattr guard silently short-circuited to None on every Bayesian `start_run` call, so `RunMetadata.n_trials_per_model` was always null in every sidecar and the resume banner showed "Trials per model (target): ?" instead of the actual count. T-43 captures `n_unified_trials` correctly via the whitelist so resume restoration was unaffected; only the audit-trail metadata field was broken. DeepSeek post-PR review surveyed sibling `hasattr(self, ...)` sites and surfaced one phantom MEDIUM at `gui:30859` (`task_type_var` read but actual var is `task_type`); folded into same PR rather than filing a separate ticket. Pinned with a class-of-bug source-text regression test (`test_t44_no_phantom_hasattr_typos_in_gui`) covering both phantom names. 282 + 1 skipped. Pipeline behavior unchanged.
>
> **2026-05-02 — T-45 MERGED via PR #13 at `2333fae`.** App-lifetime file handler at `<user_data_dir>/dasp.log` now catches module warnings from all three startup paths (bundled GUI, console-script, future entry points). Codex caught a HIGH (cli.main bypass) + MEDIUM (module-reload double-attach) post-PR; both closed in `66e1da1`. DeepSeek pair attempted but routing-blocked at first attempt (subagent misjudgment); retroactive review on T-44 dispatch confirmed nothing Codex missed at HIGH/MEDIUM.
>
> ---
>
> **Previously:** 2026-05-02 (T-46 MERGED + T-45 IMPLEMENTED) —
>
> **T-46 MERGED via PR #12 at merge commit `08b20ea`.** Branch deleted from origin and locally. Both call sites of `_apply_wal_pragmas` now surface WAL rejection via the right channel; OneDrive/Dropbox/network-share users get a structured warning event instead of silent throughput halving.
>
> **T-45 IMPLEMENTED on `fix/T45-logger-file-handler` (off post-T-46 main).** Wired a `RotatingFileHandler` to the `spectral_predict` logger at app startup (1 MB per file, 3 backups, 4 MB total ceiling). Module-level `logger.warning` calls in `run_state` (corrupt-sidecar coercion, malformed `gui_settings`, empty-SQLite cleanup), `run_gui_settings` (capture-time Tk failures), and `unified_bayesian` (WAL rejection, T-42 hoist mismatch, migration failures) now land in `<user_data_dir>/dasp.log` instead of vanishing into PyInstaller's null stderr. New `setup_app_logger()` function in `src/spectral_predict/run_logging.py` — distinct from the existing `setup_run_logger()` (per-run, keyed to Run Analysis click): app-lifetime log catches startup-path warnings BEFORE the user clicks Run, plus all `spectral_predict.*` propagated warnings during the run. 4 new regression tests (`test_setup_app_logger_creates_dasp_log_at_user_data_dir`, `test_setup_app_logger_captures_module_warnings`, `test_setup_app_logger_idempotent`, `test_setup_app_logger_swallows_setup_failure`) + 11 existing `test_run_logging.py` tests joined the consolidated sweep. 279 + 1 skipped. **Pipeline behavior unchanged** — pure observability fix; warnings that already existed but were going to /dev/null now go to a file the user can find.
>
> **T-47 MERGED via PR #11 at merge commit `a29ad2d`.** Default-flip + DeepSeek follow-ups all rebased onto main; both T-47 commits + branch deleted from origin and locally. The dataclass-vs-function default symmetry + the deliberate "never"-kept legacy-sidecar/corruption-coercion fallbacks are now load-bearing in 3 regression tests.
>
> **T-46 IMPLEMENTED on `fix/T46-wal-pragma-surface-return` (off post-T-47 main).** Surface `_apply_wal_pragmas` return value at both call sites — pre-T-46 silent fallback to DELETE journal mode now signals through the right channel. Site 1 (`run_unified_bayesian:2117`, always-on path): captured return + `progress_callback({"t41_decision": "wal_rejected", ...})`. Site 2 (`_migrate_study_to_sqlite:1693`, auto-migration path): captured return + contextual `logger.warning("T-46: WAL rejected during auto-migration to ...")` AND (post-DeepSeek-MEDIUM follow-up) progress_callback emission with `t41_decision: "wal_rejected_at_migration"` — under the post-T-47 'auto' default, auto-migration is the COMMON WAL-touching path, so logger-only surfacing was inadequate (invisible in bundled GUI per T-45). 3 new regression tests (`test_wal_rejection_surfaces_via_progress_callback`, `test_wal_rejection_surfaces_at_auto_migration_site`, `test_migrate_without_callback_still_logs_on_wal_rejection`) — last one pins the headless-caller log surface so a future refactor can't silently drop the unconditional log. 264 + 1 skipped across the consolidated sweep. **Cross-family review:** DeepSeek V4 Pro Max READY_WITH_REVISIONS (1 MEDIUM caught the auto-migration site gap under T-47's new default + 3 LOW deferred); Codex CLI READY_WITH_REVISIONS (1 LOW — same site-2 test gap, closed by the same fix). DeepSeek's MEDIUM was effectively load-bearing because of T-47's recent merge — auto-migration changed from edge case to common case the day before. **OneDrive/Dropbox-synced data dirs no longer silently halve Bayesian throughput on either persistence mode** (the failure mode the ticket was filed against).
>
> ---
>
> **Previously:** 2026-05-02 (PR #10 MERGED + T-47 IMPLEMENTED) —
>
> **PR #10 (consolidated T-43 + T-42 + T-38 + T-49) MERGED via merge commit `f063d9e`.** 17 commits flattened onto main. Final fix-of-fixes commit `30fa673` closed the Codex post-PR MEDIUM (partial-mutation in `_apply_pending_validation_indices` when X/y indices disagreed on a captured label — silent skip of validation metrics). Cross-family review verdicts on the PR: DeepSeek V4 Pro Max READY_TO_MERGE (5 MEDIUM, 4 LOW deferred); Codex CLI READY_WITH_REVISIONS (1 MEDIUM closed). Per-ticket branches deleted on origin + locally: `fix/T43-resume-auto-restore-settings`, `fix/T42-write-path-plumbing-approach-c`, `fix/T38-dead-preprocessing-cleanup`, `fix/bayesian-resume-and-cleanup`.
>
> **T-47 IMPLEMENTED on `fix/T47-bayesian-persistence-default-auto` (off post-PR-10 main).** One-line behavioral flip: `bayesian_persistence_mode` field default flips from `'never'` → `'auto'` at both surfaces — `RunMetadata` dataclass field default + `start_run()` function default — keeping the two consistent. Legacy-sidecar fallback in `from_dict` (line 162) and corruption-coercion path (line 168) deliberately stay at `'never'` — those are safety nets for malformed input, not "the field default for unspecified input". Justified by the post-T-42 baseline benchmark: PLS 0.69×, Ridge 1.06×, RandomForest 0.25× (caching), LightGBM 0.93×, XGBoost 1.01× — all well below the plan's 1.15×/1.30× targets. Crash-resume now on by default; users who want the prior in-memory-only behavior set the radio to "Never on". 3 new regression tests (`test_default_persistence_mode_is_auto`, `test_dataclass_default_persistence_mode_matches_start_run`, `test_legacy_sidecar_without_persistence_mode_defaults_to_never` — last one added post-DeepSeek MEDIUM, pinning the deliberate kept-fallback half of the two-pair contract); 261 + 1 skipped across the consolidated sweep. DeepSeek V4 Pro Max review (READY_WITH_REVISIONS): 1 MEDIUM (missing absent-key fallback test) closed; 2 LOWs (plan-doc drift at step 3, GUI hasattr safety-net comment) closed in the same commit.
>
> ---
>
> **Previously:** 2026-05-02 (T-43 + T-42 + T-38 + T-49 CONSOLIDATED — single PR shipped) —
>
> **All four tickets (T-43 + T-42 + T-38 + T-49) land on `fix/bayesian-resume-and-cleanup` (tip `6465306`, 14 commits off main).** User decision after the parallel pr-review-toolkit pass: ship as one PR rather than three. T-49 was originally filed as a follow-up but the user (correctly) flagged it as a correctness blocker for the resume flow and folded it into the same PR. The per-ticket branches (`fix/T43-resume-auto-restore-settings`, `fix/T42-write-path-plumbing-approach-c`, `fix/T38-dead-preprocessing-cleanup`) remain on origin as reference and can be deleted once the consolidated PR merges.
>
> **What's in the PR:**
> 1. **T-43 (auto-restore GUI settings on resume):** ~91 analysis-defining Tk vars captured at `start_run` time and restored when the user accepts the resume banner — preprocessing toggles, autoscale, baseline, smoothing, variable selection, model selection, n_trials, tier, CV strategy, imbalance method, one-class hyperparameters, the 6 `bayes_*` Bayesian-only controls, and the 5 external-validation-set Tk vars. New `src/spectral_predict/run_gui_settings.py` module + extended `RunMetadata.gui_settings: dict | None` with `dataclasses.fields()` filter + type-guard for malformed sidecars + `RestoreReport` with `restored / skipped_unknown / skipped_no_var / errors` buckets and `fully_succeeded` property. **Plus T-49 (folded in):** validation set indices persisted alongside `gui_settings` in the sidecar so non-deterministic algorithms (Random, Manual) and hand-picked partitions resume without silent leakage. New `_apply_pending_validation_indices` GUI helper re-slices `self.X` / `self.y` after the fingerprint check passes; respects the user's manual re-creation if they did one before clicking Run Analysis; defense-in-depth on missing labels.
> 2. **T-42 (Bayesian write-path Approach C):** hoisted three constant keys (`cv_strategy`, `cv_n_repeats`, `early_stopping_rounds`) from per-trial to per-study scope. Two were dead writes (read by nobody); the third is now read once outside the trial loop with a trial-level fallback for legacy/migrated studies. Hoist guarded against silent overwrite on resume — disagreeing values keep the stored value and log a warning. `_supports_early_stopping(model_name)` helper de-duplicates the trial-time and study-hoist gates. Per-trial set_user_attr count: PLS regression 30→27, Ridge regression 30→27, PLS-DA classification 42→39.
> 3. **T-38 (dead preprocessing cleanup):** deleted `learned_preprocessing.py` (775 LOC, zero callers) and `ensemble_preprocessing.py` (701 LOC, only referenced by a dead `HAS_ENSEMBLE_PREPROCESSING` flag). Did NOT delete `preprocessing_wrapper.py` (still imported by `ensemble.py:18`) or `tests/test_ensemble_preprocessing.py` (despite the matching name — actually tests `preprocessing_wrapper` and `ensemble`, both alive).
>
> **Surprise finding documented during T-42:** post-T-41 baseline benchmark already met T-42's "definition of done" target — PLS 0.69×, Ridge 1.06×, LightGBM 0.93×, XGBoost 1.01× SQLite-WAL ratios, all well below the plan's 1.15× / 1.30× targets. T-41's WAL pragmas + 30s `busy_timeout` collapsed per-trial overhead to near-noise; the plan's 1.36× / 1.89× ratios were PRE-T-41. Flipping T-41's default from `'never'` to `'auto'` is now justified by bench data alone — separate trivial follow-up ticket.
>
> **Review trail (7 review passes + 1 user catch, cross-family):**
> 1. **T-43** post-`f934c81` — DeepSeek V4 Pro Max: READY_WITH_REVISIONS, 2 HIGH + 1 MEDIUM closed in `3f071b3`.
> 2. **T-43** post-`487b1d9` — Codex CLI: BLOCKERS, 1 HIGH (`bayes_*` whitelist gap) + 1 MEDIUM (`skipped_no_var` not surfaced) + 1 LOW (docstring drift) closed in `2568be8`.
> 3. **T-42** post-`ab9d00f` — DeepSeek V4 Pro Max: READY_WITH_REVISIONS, 1 HIGH (stale `test_cv_strategy.py` assertion) + 1 MEDIUM (copy_study migration test gap) + 1 LOW closed in `e1daaa6`.
> 4. **T-42** post-`ab9d00f` — Codex CLI: READY_TO_MERGE; verified empirically that `optuna.copy_study` preserves user_attrs.
> 5. **T-38** post-`2eaef4f` — DeepSeek V4 Pro Max: READY_TO_MERGE, auto-applied 2 doc fixes in `c0b4c20`.
> 6. **T-38** post-`2eaef4f` — Codex CLI: READY_WITH_REVISIONS, 1 LOW (stale PROJECT_STATUS sentence) closed in `de55f32`.
> 7. **Stacked diff** post-`de55f32` — pr-review-toolkit parallel-5 (code-reviewer / silent-failure-hunter / type-design-analyzer / pr-test-analyzer / comment-analyzer): 3 HIGH + 3 MEDIUM closed in `fe5e25d`.
> 8. **User catch** during review wrap-up: validation partition not maintained on resume — silent leakage on Random/Manual algorithms, silent skip if user forgets the button on deterministic algorithms. Originally filed as T-49 follow-up; user (correctly) called it a correctness blocker. Folded in at `6465306` with 8 new tests.
>
> **Test coverage:** **257/257 + 1 skipped** across `test_t41_bayesian_sqlite_auto_calculator`, `test_t42_write_path_plumbing` (8 tests), `test_t43_resume_auto_restore` (33 tests including the 8 new T-49 cases), `test_run_state`, `test_cv_strategy`, `test_unified_bayesian_baseline`, `test_autoscale_bayesian`, `test_cv_pls_clamp`, `test_ensemble_preprocessing`, `test_ensemble_integration`. Sanity-imported every `spectral_predict.*` submodule via `pkgutil.walk_packages` post-T-38 deletion — clean.
>
> **Follow-up tickets filed (NOT in this PR):**
> - **T-44** (`docs/plans/2026-05-02-T44-n-trials-var-typo.md`): `n_trials_var` typo at `gui:25123`; hasattr-guarded so silent-no-op. T-43 captures `n_unified_trials` correctly so resume-restore is unaffected; only the sidecar's `n_trials_per_model` field is wrong. ~5 LOC.
> - **T-45** (`docs/plans/2026-05-02-T45-logger-warning-visibility-bundled-gui.md`): `logger.warning` calls invisible in the bundled GUI (no `logging.basicConfig` anywhere). Silent in production for sidecar corruption, WAL rejection, capture-time Tk failures, and the new T-42 hoist mismatch warning. File-handler at app startup is the cheap fix.
> - **T-46** (`docs/plans/2026-05-02-T46-wal-pragma-return-not-surfaced.md`): `_apply_wal_pragmas` return value discarded at both call sites despite "caller should surface that" docstring. OneDrive/Dropbox-synced data dirs silently fall back to DELETE journal mode.
> - **T-47** (`docs/plans/2026-05-02-T47-flip-bayesian-persistence-default-to-auto.md`): one-liner default flip from `'never'` to `'auto'`. Justified by the T-42 baseline benchmark numbers (XGBoost 1.01×, LightGBM 0.93× — well below 1.15× target). Gated on this PR merging first.
> - **T-48** (`docs/plans/2026-05-02-T48-real-tk-integration-tests-for-resume.md`): real-Tk integration tests for the resume flow. Pins Tcl set/get-verify behavior + order-dependent restore-then-override contract on the actual method instead of `_FakeGUI` shims. Belt-and-braces; current shim coverage is sound at the unit level.
>
> **Audit trail:**
> - T-42: `docs/bugfix_validation/T42_write_path_plumbing_approach_c.md`.
> - T-43 + T-38 + T-49: code + tests + this PROJECT_STATUS entry serve as audit trail. T-49 plan doc (`docs/plans/2026-05-02-T49-...md`) is marked IMPLEMENTED with implementation summary.
>
> ---
>
> **Previously: per-ticket framing (now consolidated):**
>
> **T-38 dead preprocessing module cleanup — IMPLEMENTED on `fix/T38-dead-preprocessing-cleanup` (1 commit, branched off T-42 tip).** Deleted `src/spectral_predict/learned_preprocessing.py` (775 LOC, zero callers — imports torch but never wired into the GUI) and `src/spectral_predict/ensemble_preprocessing.py` (701 LOC, only referenced by a dead `HAS_ENSEMBLE_PREPROCESSING` flag in the GUI try/except block at `gui:236-241`). Removed the dead flag too. **Did NOT delete `preprocessing_wrapper.py`** (still imported by `ensemble.py:18`) — plan was corrected on this point during T-37 review. Also did NOT delete `tests/test_ensemble_preprocessing.py` despite its matching name — that file actually tests `preprocessing_wrapper` and `ensemble`, both alive (the test name is misleading). Updated build-time torch-exclusion comments + `BUNDLED_APP_BUILD_GUIDE_PY312.md` Step 3 rationale. 244/244 across the regression sweep + clean `pkgutil.walk_packages` sanity-import of all `spectral_predict.*` submodules.
>
> ---
>
> **T-42 Bayesian write-path Approach C — IMPLEMENTED on `fix/T42-write-path-plumbing-approach-c` (tip `5670c26`, 3 commits, branched off T-43 tip).** Hoisted three constant `set_user_attr` keys (`cv_strategy`, `cv_n_repeats`, `early_stopping_rounds`) from per-trial to per-study scope in `unified_bayesian.py`. Two were dead writes (read by nobody — `convert_study_to_dataframe` takes them as function parameters); the third is now read once outside the trial loop with a trial-level fallback for legacy/migrated studies. Per-trial set_user_attr count: PLS regression 30→27, Ridge regression 30→27, PLS-DA classification 42→39 (each model saves 3 writes/trial).
>
> **Surprise empirical finding (changes T-42's framing):** the post-T-41 baseline benchmark (`tests/_bench_bayesian_per_model.py`, n_trials=10, synthetic n=100, n_features=200) **already met T-42's "definition of done" target**: PLS 0.69×, Ridge 1.06×, RandomForest 0.25× (caching), LightGBM 0.93×, XGBoost 1.01× — all well below the plan's 1.15× / 1.30× targets. T-41's WAL pragmas + 30s `busy_timeout` collapsed per-trial overhead to near-noise; the plan's 1.36× / 1.89× ratios were PRE-T-41 measurements. **Flipping T-41's default from `'never'` to `'auto'` is now justified by bench data alone — file as a separate trivial follow-up ticket.** Approaches A and D no longer needed.
>
> **T-42 review trail (2 passes, cross-family):**
> 1. DeepSeek V4 Pro Max post-`ab9d00f`: READY_WITH_REVISIONS — 1 HIGH (test_cv_strategy.py still asserted old per-trial contract; audit doc's "132/132 green" claim was wrong because that file was excluded from the curated sweep) + 1 MEDIUM (no test that user_attrs survive copy_study migration — same trap class as T-41's `load_if_exists+sampler` silent-ignore) + 1 LOW (XGBoost trial-absence assertion missing) all closed in `e1daaa6`. Wider sweep now 225/225 green.
> 2. Codex CLI post-`ab9d00f` (paired with T-43 review per overnight protocol): READY_TO_MERGE. Verified empirically against installed Optuna source that `copy_study()` does preserve user_attrs (study.py:1567-1568); confirmed ordering safety (study.set_user_attr lands before progress_wrapper closure runs); confirmed no other readers of the three keys remain in `trial.user_attrs`.
>
> **T-43 resume auto-restore GUI settings — IMPLEMENTED on `fix/T43-resume-auto-restore-settings` (tip `2568be8`, 3 commits).** Closes the user-reported gap from T-41 verification: previously the user had to manually recreate every preprocessing/model setting before the resumed SQLite study would actually pick up trial-level state. Now ~86 analysis-defining Tk vars (preprocessing toggles, autoscale, baseline, smoothing, variable selection, model selection, n_trials, tier, CV strategy, imbalance method, one-class hyperparameters, **plus the 6 `bayes_*` Bayesian-only baseline/smoothing/region/UVE controls Codex flagged as missing**) are auto-restored from the sidecar.
>
> **Architecture:** `RunMetadata.gui_settings: dict | None` (extended schema, backward-compat via `dataclasses.fields()` filter + type-guard). New `src/spectral_predict/run_gui_settings.py` module with `CAPTURABLE_SETTINGS` whitelist + `capture_gui_settings(gui)` + `restore_gui_settings(gui, dict) -> RestoreReport` + `summarize_gui_settings()`. GUI captures at `start_run` time and restores BEFORE the existing 'always' persistence override (order matters — comment + new test document this).
>
> **T-43 review trail (2 passes, cross-family):**
> 1. DeepSeek V4 Pro Max post-`f934c81`: READY_WITH_REVISIONS — 2 HIGH (banner-lies-on-errors + missing type-guard on gui_settings) + 1 MEDIUM (Tcl quirk: `IntVar.set("abc")` doesn't raise) all closed in `3f071b3`.
> 2. Codex CLI post-`487b1d9`: BLOCKERS — 1 HIGH (the 6 `bayes_*` Tk vars at gui:27554-27570 weren't in the whitelist; resumed Bayesian runs would silently use defaults despite the banner claiming "settings restored" — the exact silent-failure trap T-43 was meant to close) + 1 MEDIUM (skipped_no_var bucket not surfaced in resume log) + 1 LOW (docstring drift) all closed in `2568be8`.
>
> **Pre-existing bug surfaced for follow-up (NOT a T-43 regression):** `spectral_predict_gui_optimized.py:25123` reads `self.n_trials_var.get()` guarded by `hasattr`; the actual var is `self.n_unified_trials`, so `n_trials_per_model` is always None in `start_run` metadata. T-43 itself captures `n_unified_trials` correctly via the whitelist. To file as a separate ticket.
>
> **Test coverage:** 23/23 T-43 tests + 6/6 T-42 tests + the 225/225 wider regression sweep = comprehensive coverage of the resume-auto-restore + write-path-cleanup surface.
>
> **Audit trail:**
> - T-42: `docs/bugfix_validation/T42_write_path_plumbing_approach_c.md`.
> - T-43: code + tests speak for themselves; no separate audit doc.
>
> ---
>
> **Previously:** 2026-05-02 (T-41 MERGED to main after 5-pass review trail) —
>
> **T-43 resume auto-restore GUI settings — IMPLEMENTED on `fix/T43-resume-auto-restore-settings` (tip `3f071b3`, 2 commits).** Closes the user-reported gap from T-41 verification: previously the user had to manually recreate every preprocessing/model setting before the resumed SQLite study would actually pick up trial-level state. Now, when the user accepts the resume banner, ~80 analysis-defining Tk vars (preprocessing toggles, autoscale, baseline, smoothing, variable selection, model selection, n_trials, tier, CV strategy, imbalance method, one-class hyperparameters) are auto-restored from the sidecar — they just click Run Analysis.
>
> **Architecture:** `RunMetadata.gui_settings: dict | None` (extended schema, backward-compat via `dataclasses.fields()` filter). New `src/spectral_predict/run_gui_settings.py` module with `CAPTURABLE_SETTINGS` whitelist + `capture_gui_settings(gui)` + `restore_gui_settings(gui, dict) -> RestoreReport` + `summarize_gui_settings()` for the resume prompt. GUI captures at `start_run` time and restores BEFORE the existing 'always' persistence override (order matters — comment in code).
>
> **Review trail (1 pass):**
> 1. DeepSeek V4 Pro Max post-`f934c81`: READY_WITH_REVISIONS — 2 HIGH (banner-lies-on-errors + missing type-guard on `gui_settings`) + 1 MEDIUM (Tcl quirk: `IntVar.set("abc")` doesn't raise) all closed in `3f071b3`. 5 LOW deferred (whitelist debt for variable_penalty/gap_penalty, pre-existing `n_trials_var` typo at gui:25123, ga_preprocess settings on the T-38 deletion path).
>
> **Pre-existing bug surfaced for follow-up (NOT a T-43 regression):** `spectral_predict_gui_optimized.py:25123` reads `self.n_trials_var.get()` guarded by `hasattr`; the actual var is `self.n_unified_trials`, so `n_trials_per_model` is always None in `start_run` metadata. T-43 itself captures `n_unified_trials` correctly via the whitelist. To file as a separate ticket.
>
> **Test coverage:** 19/19 T-43 tests + 29 run_state + 23 T-41 = 71/71 passing (~18s on `.venv312`).
>
> **Audit trail:** code + tests speak for themselves; no `bugfix_validation/` doc filed for this small ticket.
>
> ---
>
> **Previously:** 2026-05-02 (T-41 MERGED to main after 5-pass review trail) —
>
> **T-41 Bayesian SQLite auto-calculator + WAL mode — MERGED via PR #9.** Five rebase commits on main: `cd406f0` (feat) → `e99d35a` (DeepSeek HIGH/MEDIUM + resume bug + default 'auto') → `2087134` (docs) → `081ad6a` (pr-review-toolkit findings: silent failures + Literal validation + cleanup gates) → `f745d37` (DeepSeek Finding 1: multi-model SQLite collateral deletion via `optuna.delete_study` instead of `Path.unlink`).
>
> **Default flipped from plan's `'auto'`→`'never'`→`'auto'` during review:** User originally requested `'never'` ("99.99% of the time this would not be used"), then flipped back to `'auto'` for resume-path troubleshooting. Resume reliability now confirmed end-to-end.
>
> **Review trail (8 passes total):**
> 1. DeepSeek V4 Pro Max pre-implementation: 6 plan findings, all applied
> 2. DeepSeek V4 Pro Max post-`cd406f0`: 2 HIGH (study_ref doesn't redirect Optuna writes; cb_study.stop crash) + 1 MEDIUM + 1 LOW, all closed
> 3. GLM 5.1 post-`e99d35a`: READY_TO_MERGE; 1 MEDIUM (integration test gap) deferred
> 4. Codex post-`e99d35a`: READY_WITH_REVISIONS; 1 MEDIUM (phantom resume prompt for in-memory crashes) folded into T-43, 1 LOW fixed
> 5. pr-review-toolkit (5 specialist agents in parallel): 1 type-design HIGH + 4 silent-failure HIGHs + 4 MEDIUMs + comment-quality polish — all closed in `081ad6a`
> 6. DeepSeek V4 Pro Max post-`081ad6a`: 1 MEDIUM (multi-model collateral deletion) + 2 LOWs, MEDIUM closed in `f745d37`
> 7. User runtime verification: resume banner → recreate settings → click Run Analysis → SQLite study picks up trial-level state ✓
>
> **Type design:** `PersistenceMode = Literal["auto","always","never"]` shared between `RunMetadata` and `run_unified_bayesian`; validated at all three call boundaries (`__post_init__`, `start_run`, `run_unified_bayesian` entry); `from_dict` coerces corrupted sidecar values to `'never'` with warning instead of crashing the resume flow.
>
> **Architecture (final):** auto-decision callback aborts in-memory `optimize()` via `cb_study.stop()`; outer scope detects `_auto_migrated` and restarts `optimize()` on the migrated SQLite-backed study so trials 11..N persist directly. Multi-model migration failures use `optuna.delete_study` (study-scoped) not `Path.unlink` (file-scoped) so prior models' trials survive a later model's failed migration on the shared SQLite.
>
> **Test coverage:** 127/127 passing across `test_t41_bayesian_sqlite_auto_calculator` (23) + `test_run_state` (29) + `test_unified_bayesian_baseline` (36) + `test_autoscale_bayesian` (14) + `test_cv_pls_clamp` (25). New test classes added in `081ad6a`: `TestPersistenceModeValidation`, `TestCleanupByTrialCount`, `TestWALPragmaReturnValue`, `TestMigrationOrphanCleanup`.
>
> **Performance:**
> - `'never'` mode: pre-T-11 in-memory speed (~1.0× ratio). Default for users who don't need crash-resume.
> - `'auto'` mode: PLS/Ridge stay in-memory (median fit < 1s); LightGBM/XGBoost migrate at trial 11. XGBoost runs at ~1.36× WAL overhead; that's the residual `set_user_attr` cost which T-42 (filed) targets to push toward ~1.15×.
> - `'always'` mode: SQLite+WAL from trial 0; PLS pays ~8× cost in exchange for universal crash-resume.
>
> **Audit trail:** `docs/bugfix_validation/T41_bayesian_sqlite_auto_calculator.md`.
>
> **Open follow-ups (filed):**
> - **T-42** (`docs/plans/2026-05-01-T42-bayesian-write-path-plumbing.md`): hoist constant `set_user_attr` keys to `study.set_user_attr` and batch per-trial keys; brings XGBoost overhead toward 1.15× as prerequisite for promoting `'auto'` from troubleshooting-default to no-regret-default.
> - **T-43** (`docs/plans/2026-05-02-T43-resume-auto-restore-settings.md`): auto-restore GUI settings on resume so users don't have to manually recreate preprocessing/model settings before clicking Run Analysis.
>
> **Previously:** 2026-05-01 (T-36 + T-37 BOTH MERGED to main after final cross-family adversarial sweep) —
>
> **PR #8 (T-36) MERGED** at `ff03193` (rebase merge: `bc989f2` + `ffd6451` + `ff03193`). Final adversarial sweep verdict: READY_TO_MERGE with zero new findings from both reviewers.
>
> **PR #7 (T-37) MERGED** at `9430d5c` (rebase merge — single squashed commit since the local merge-commit-of-T-36 had to be flattened after T-36 rebased onto main with new commit hashes).
>
> **Cross-family review trail:** Two reviewers via `opencode-call` with full repo access — DeepSeek V4 Pro Max (DeepSeek API direct, max thinking) + MiMo 2.5 Pro Max (opencode-go subscription) — dispatched in parallel. **Both converged independently on the same structural finding:** T-37 was branched off T-36 at `1f49d73` BEFORE T-36's two post-merge review fix commits landed, so T-37 inherited the silent-mismatch bugs T-36 already fixed (3 HIGH + 4 MEDIUM + 2 LOW per MiMo's tally; equivalent ratings from DeepSeek). Resolution applied via merge of T-36 fixes into T-37 before final ship. 123 T-37+autoscale tests + 27 code-export tests all green post-merge.
>
> **Lesson logged in SESSION_LOG.md:** in-band PR review tools (Codex, gemini-code-assist, CodeRabbit, pr-review-toolkit) operate on diff-against-main and are structurally blind to sister-branch inheritance gaps. A final cross-family sweep with full repo access is the right gate for this class of bug. Convergence between two family-orthogonal models is high-signal enough to act on without further verification beyond the initial sanity check.
>
> **Architectural correction logged this session:** I initially told the user (incorrectly) that the grid-search path applied autoscale per-fold via the sklearn Pipeline mechanic, while the Bayesian path applied it pre-CV global — claiming this was a real divergence. **That was wrong.** The grid path uses `skip_spectral_preprocessing=True` at every call site of `_run_single_config`, which means `pipe_steps = []` for the inner CV loop (search.py:4243-4259); spectral preprocessing including the autoscale `StandardScaler` is `fit_transform`'d once on full training data at search.py:2061, and CV runs on the already-preprocessed `X_for_models`. So **both paths apply autoscale pre-CV global, by design, matching PLS_Toolbox / Unscrambler chemometrics convention.** No grid/Bayesian divergence exists. The misleading line 4270 path (`else` branch with full Pipeline construction) is reachable in principle but not exercised by the live grid call sites. See SESSION_LOG.md for the trace details so this doesn't get re-discovered.
>
> **Deferred deliberately (chemometrics-convention calls, NOT bugs):**
> - **Gemini "Bayesian autoscale leakage" finding** — autoscale fits on full training data before CV in both paths. By ML orthodoxy this is mild leakage; by chemometrics convention (PLS_Toolbox / Unscrambler / SIMCA-P / Pirouette default behavior) it is correct. Project follows chemometrics convention. Bias is small for stable NIR data and dominated by sample-population shift risks that KS-style train/test splits address.
> - **Gemini "code_generator export uses fit_transform not transform" finding** — entangled with the Bayesian leakage architectural rewrite above; not applicable until/unless that decision changes.
>
> **Doc nits + plan corrections done this commit (T-37 branch):**
> - `docs/bugfix_validation/T37_tpe_preprocessing_discovery.md:97` — Phase 7 row was marked `_pending_` after commit `03d95cb` already landed; updated to landed-commit reference. Also added Phase 8 row for `6a0eb28`.
> - `docs/plans/2026-05-01-T41-bayesian-sqlite-auto-calculator.md:74-82, 176, 235, 250` — internal contradiction between architecture (10 trials, median fit time) and examples / tasks / acceptance criteria (5 trials, mean fit time). Architecture is authoritative per the DeepSeek review. All inconsistent references updated to 10 trials / median to match.
> - `docs/plans/2026-05-01-T36-autoscale-toggle.md:210` — fixed nonexistent `T34_autoscale_toggle.md` reference → `T36_autoscale_toggle.md`.
> - `docs/plans/2026-05-01-T38-dead-preprocessing-cleanup.md:27-32` — corrected the false claim that `preprocessing_wrapper.py` is doubly orphaned. It is actively imported by `ensemble.py:18` and instantiated at `ensemble.py:1309`, exercised by 2 test files. Plan changed to NOT delete it.
>
> **Open follow-ups (NOT done — not "definitely broken"):**
> - pr-review-toolkit T-37 M1: missing regression test for the new mutual-exclusion `ValueError`. Coverage gap, not a bug.
> - pr-review-toolkit T-37 M2: bare `except Exception` in `_quick_evaluate` could be narrowed to `(ImportError, ModuleNotFoundError)`. Defensive over-catch, not a bug.
> - pr-review-toolkit T-36 M2: `refined_config` doesn't carry an `'autoscale'` key, so the GUI fallback chain always falls through to `selected_model_config`. Works in practice today (selected_model_config always has 'Autoscale').
> - pr-review-toolkit T-36 L1: `contamination.py` cache key omits `baseline_params`. Latent — no current producer emits per-row baseline_params.
> - `tests/test_autoscale_bayesian.py:169` cache-key assertion uses bare substring `'apply_autoscale' in src` rather than asserting it appears specifically inside the `cache_key` tuple. Test passes today; would not catch a regression that drops `apply_autoscale` from the tuple but keeps it elsewhere in the function. Test-quality issue, not a bug.
>
> **Previously:** 2026-05-01 (T-37 Phase 7 fix commit `03d95cb` — F2, F6, F9, F10, F12 all closed) —
>
> **DeepSeek V4 Pro Max review trail (this session):**
>
> - **T-41 plan pre-implementation review** (read-only) — verdict: READY_WITH_PLAN_REVISIONS, 6 findings (2 HIGH, 2 MEDIUM, 4 LOW). All applied in commit `7861a81`. Critical empirical catch: DeepSeek tested `optuna.copy_study` + `optuna.load_study(sampler=TPESampler(...))` mid-run on Optuna 4.8 and confirmed TPE state is preserved. Key trap caught: `create_study(load_if_exists=True, sampler=...)` SILENTLY IGNORES the sampler kwarg on existing studies — the migration helper now uses `load_study(sampler=...)` explicitly. Plan also revised to use 10-trial median window (was 5-trial mean), corrected WAL durability claim, moved WAL pragma application from `start_run` to migration time (file doesn't exist until then), added stale-sidecar cleanup, added 4 missing tests.
>
> - **T-37 Phase 1 review of commit `ef5f61e`** by **GLM 5.1** (read-only, attribution corrected by user 2026-05-01 PM) — verdict: READY_WITH_REVISIONS. Re-prioritized findings against initial draft:
>   - **MEDIUM F2:** `_tpe_baseline_params` never attached to output config dicts. Currently safe by coincidence (TPE defaults match `preprocess.py` defaults), fragile if either changes.
>   - **MEDIUM F6:** `_quick_evaluate` PLS exception fallback uses `neg_root_mean_squared_error` regardless of task_type — wrong metric for classification/one_class when LightGBM is unavailable.
>   - **MEDIUM F9:** Mutual exclusion test only checks `results is not None`, doesn't verify TPE actually ran vs smart_preprocess fallback. False confidence.
>   - **MEDIUM F12:** Test coverage gaps — no `_apply_full_preprocessing` unit tests, no `_quick_evaluate` direct tests, no empty-TPE-result fallback test, no roundtrip-through-`build_preprocessing_pipeline` verification.
>   - **LOW F1/F10:** `_resolve_window_choices` is dead code AND **GLM corrected a critical earlier misdiagnosis:** the apparent "fix" (varying search space per trial) **would break Optuna** — `trial.suggest_categorical` requires CONSTANT choices across trials per the ask/tell contract. The current "union + return-inf-on-invalid" approach is the correct workaround. Status: leave as-is; remove or comment the unused helper.
>   - LOW F7, F8 (style/hygiene), several "NOT AN ISSUE" items (F4, F5, F11).
>
>   **Recommendation:** Phase 7 fix commit needed before T-37 ships, but smaller scope than original draft suggested. Fix F2 (~5 lines), F6 (~10 lines), F9 (test strengthening, ~15 lines), F12 (new tests, ~80 lines). Original "fix C1 by varying search space" approach explicitly rejected per GLM analysis.
>
> **T-41 plan ready for implementation** at `docs/plans/2026-05-01-T41-bayesian-sqlite-auto-calculator.md` (revised to 10-trial median window, explicit `load_study(sampler=...)` mechanism, WAL durability honest, stale sidecar cleanup, 4 added tests). User has the prompt for dispatching implementation when ready.
>
> **Previously:** 2026-05-01 (post-T-37 ship — TPE preprocessing discovery implemented on `feature/T37-tpe-preprocessing-discovery`) —
>
> **T-37 TPE preprocessing discovery IMPLEMENTED.** Branch `feature/T37-tpe-preprocessing-discovery` at `03d95cb` (7 commits past T-36 tip `1f49d73`).
>
> Key wins:
> - **New module `tpe_preprocessing_discovery.py`** — Optuna TPESampler(multivariate=True) over 5-D space (preproc x window x autoscale x baseline x smoothing), 75-trial default budget, model-agnostic LightGBM surrogate (Architecture A — preserves model diversity).
> - **Derivative-aware windows** ported from `ga_preprocessing.py` DERIVATIVE_WINDOW_RANGES.
> - **GUI checkbox + n_trials dropdown** in Preprocessing tab, mutual exclusion with smart/GA preprocessing.
> - **Self-review caught two silent-mismatch bugs** before they shipped: (a) preprocess_cfg name field used pipeline name instead of display name with prefix cascades, (b) baseline/smoothing/autoscale doubling blocks re-doubled TPE-discovered per-config settings — guarded by nulling global flags inside TPE block.
> - **Phase 7 fix commit (`03d95cb`)** closes GLM 5.1 review findings: F2 (_tpe_baseline_params now attached), F6 (_quick_evaluate branches on task_type), F9 (tpe_score propagated to result rows + test strengthening), F10 (_resolve_window_choices commented as RESERVED), F12 (12 new tests covering _apply_full_preprocessing, _quick_evaluate, empty-TPE fallback, pipeline roundtrip).
>
> Test coverage: **33 T-37 tests all green** (21 unit + 5 integration + 4 direct + 3 roundtrip/fallback). Module: 5 files changed, ~700 LOC new.
>
> Audit trail: `docs/bugfix_validation/T37_tpe_preprocessing_discovery.md`.
>
> **T-38 (dead preprocessing module cleanup) is now unblocked** — `ga_preprocessing.py` can be retired after T-37 merge.
>
> **Previously:** 2026-05-01 (post-T-36 ship — T-41 Bayesian SQLite auto-calculator plan filed in response to user-reported 5-28x Bayesian slowdown) —
>
> **T-41 Bayesian SQLite auto-calculator plan filed at `docs/plans/2026-05-01-T41-bayesian-sqlite-auto-calculator.md`.** User reported Bayesian 5x slower than pre-T-11; investigation traced to T-11's per-trial Optuna SQLite writes. Benchmarks (`tests/_bench_bayesian_per_model.py`) show overhead is ~200ms/trial constant — fast models pay 8-10x, heavy models 1.36-1.89x. Plan: per-model auto-calculator (first 5 trials in-memory + decide based on mean fit time + migrate via `optuna.copy_study` if heavy), 3-way GUI override (Auto/Always-on/Always-off), WAL mode bundled in. ~180 LOC + tests. Branch `fix/T41-bayesian-sqlite-auto-calculator` reserved (off post-T-36 main).
>
> **Previously:** 2026-05-01 (overnight T-36 implementation — Codex-final READY_TO_MERGE_WITH_NITS_CLOSED, awaiting user PR open) —
>
> **T-36 autoscale (UV scaling) preprocessing toggle — IMPLEMENTED, READY FOR PR.**
> Branch `feature/T36-autoscale-toggle` at `2351c3c` (13 commits past `da51f60`).
> Multi-pass review trail (echoing T-11's 7-pass pattern):
> Phase 2 DeepSeek V4 Pro Max → Phase 3 → Phase 4 → Phase 5 → Phase 6 (all READY_TO_PROCEED) → Codex round-1 final review (3 BLOCKERS — Model Dev parser, 3 refinement rebuild paths, code export — all closed in `98fb80f`) → Codex round-2 (READY_TO_MERGE_WITH_NITS — both nits closed in `2351c3c`).
>
> Key wins:
> - **Three pre-existing bugs fixed** (uncovered during plan review, would have shipped silently if autoscale were added naively): `apply_preprocessing` early-return killing autoscale for every preprocessing name, Bayesian cache key omitting autoscale, contamination cache key omitting autoscale.
> - **Bundled one-class metadata-write fix** — `run_one_class_search` result rows now emit `baseline_method`, `smoothing`, `smoothing_window`, `smoothing_polyorder` so contamination.py validation rebuild reads real values rather than silent defaults.
> - **Codex caught three downstream silent-mismatch paths DeepSeek missed** — Model Development tab parser dropped `+autoscale`, three refinement rebuild paths didn't pass autoscale, code export emitted no StandardScaler. All three closed.
>
> Test coverage: **62 new T-36 tests + 203 in the targeted regression sweep — all green.** 10 pre-existing I/O test failures (jcamp/spc/opus/perkinelmer) and 2 environmental flakes (subprocess `python` vs `.venv312`) are unrelated.
>
> Audit trail: `docs/bugfix_validation/T36_autoscale_toggle.md`.
>
> Branch is ready for `gh pr create`. NOT pushed yet (per cross-machine doc rule); user should review + push from the originating machine.
>
> **Previously:** 2026-05-01 (early hours — T-11 MERGED via PR #6 after seven reviewer passes) —
>
> **T-11 MERGED to main via PR #6** at `50057af` (rebase merge, linear history).
> Three independent reviewer families converged on READY_TO_MERGE:
> Codex (initial, 4 HIGH + 3 MEDIUM all fixed pre-merge) → Kimi K2.6 (initial,
> 2 MAJOR + 4 MINOR all fixed pre-merge) → DeepSeek V4 Pro pass 1 (24h, 2 HIGH +
> 3 MEDIUM + 9 LOW/INFO all closed) → DeepSeek V4 Pro pass 2 (recheck, 2 NEW HIGH
> closed) → 5 specialist agents post-recheck (code-reviewer, pr-test-analyzer,
> silent-failure-hunter, comment-analyzer, type-design-analyzer; 13 findings,
> 12 closed + 1 deferred) → Codex meta-review (1 NEW critical closed, 1 NEW
> deferred as architectural cost) → DeepSeek V4 Pro pass 3 (high-effort, 0
> blockers + 1 cosmetic). Final: **47/47 T-11 tests passing**, NEW BUG #2
> (per-model Bayesian failure → mark_complete still runs) deferred for a
> follow-up ticket since the fix requires per-model sidecars (architectural
> change). Full audit trail at `docs/reviews/deepseek_v4pro_24h_review_2026-04-30.md`.
>
> **T-33 Gaussian Process Regression** — rough plan filed at
> `docs/plans/2026-04-30-T33-gaussian-process-regression.md`. Not yet
> implemented; awaits prioritization.
>
> **T-36 / T-37 / T-38 preprocessing-discovery roadmap filed (2026-05-01)** —
> three tickets surfaced from a conversation about adding autoscale (UV
> scaling) as a preprocessing option, which led to a deeper audit of the
> preprocessing-discovery surface (basic, GA, exhaustive, learned, ensemble
> modules). Branch `feature/T36-autoscale-toggle` reserved.
>
> | Ticket | Title | Status | Plan |
> |---|---|---|---|
> | **T-36** | Autoscale (UV scaling) preprocessing toggle | IMPLEMENTED on `feature/T36-autoscale-toggle` (tip `2351c3c`); 13 commits, full review trail, awaiting PR | `docs/plans/2026-05-01-T36-autoscale-toggle.md` + `docs/bugfix_validation/T36_autoscale_toggle.md` |
> | **T-37** | TPE quick preprocessing discovery (replaces basic + GA + exhaustive) | **READY_TO_MERGE_WITH_NITS_CLOSED** on `feature/T37-tpe-preprocessing-discovery` (tip `03d95cb`); Phase 7 fix commit closes F2, F6, F9, F10, F12; 33 tests, all green | `docs/plans/2026-05-01-T37-tpe-quick-preprocessing-discovery.md` + `docs/bugfix_validation/T37_tpe_preprocessing_discovery.md` |
> | **T-38** | Dead preprocessing module cleanup | ROUGH_PLAN — three modules deletable now (zero callers); `ga_preprocessing.py` retires after T-37 | `docs/plans/2026-05-01-T38-dead-preprocessing-cleanup.md` |
>
> **T-36 plan caught three pre-existing implementation bugs in a draft autoscale plan (DeepSeek V4 Pro draft, audited):**
> 1. `unified_bayesian.py:apply_preprocessing()` returns early in every named-preprocessing branch — naive autoscale step at end of function would be unreachable. Fix: restructure to assign-not-return.
> 2. `unified_bayesian.py:857-864` preprocessing cache key omits `apply_autoscale` — would cause two trials with same prep but different autoscale to collide on cache. Fix: add to tuple.
> 3. `contamination.py:1109-1112` validation cache key omits `autoscale` — same class of bug. Fix: add to tuple.
>
> **T-36 also bundles a pre-existing one-class metadata gap:** `run_one_class_search` result rows omit `baseline_method`, `smoothing`, `smoothing_window`, `smoothing_polyorder` columns; validation rebuild reads them and falls back to defaults silently. Same code surface as the autoscale row writes — low-cost to fix together.
>
> **T-37 background:** Independent audits (Explore agent + DeepSeek V4 Pro via opencode-call) confirmed `ga_preprocessing.py`'s GA mode evaluates ~6× the entire 238-point search space, providing zero benefit over exhaustive on a discrete categorical space with no neighborhood structure. Basic and GA paths cover the same operation space with different search strategies; merge into one TPE-based path covering a richer 5-D space (preproc × window × autoscale × baseline × smoothing) at 75-100 trials.
>
> **T-38 deletes:** `learned_preprocessing.py` (775 LOC, zero callers) and `ensemble_preprocessing.py` (701 LOC, only referenced by the dead `HAS_ENSEMBLE_PREPROCESSING` flag at `gui:236-241`). **Note: this earlier note's claim that T-38 also deletes `preprocessing_wrapper.py` and retires `ga_preprocessing.py` was wrong** — `preprocessing_wrapper.py` is still imported by `ensemble.py:18` (CodeRabbit caught this during T-37 review), and `ga_preprocessing.py` retirement is a separate post-T-37-merge step kept out of T-38's scope. See the top-of-file status entry for the actual T-38 outcome.
>
> **Previously:** 2026-04-30 (evening — T-08 dropped, T-11 staged, T-15 dropped, T-16 reframed, T-19 user-framed) —
>
> **Session close-out:**
>
> | Ticket | Disposition | Branch / commit |
> |---|---|---|
> | T-06 SPA canonical Araújo 2001 enumeration | MERGED + T-06b parallelization | `af44ad4` + `85063b8` |
> | T-08 CARS tree-mode bug | DROPPED — false alarm; algorithm converges; CARS-Tree confirmed dasp-invented | n/a |
> | T-11 Pause/resume + disk logging + Optuna SQLite | **MERGED 2026-05-01 via PR #6** at `50057af` (rebase, 13 commits) — 7-pass review trail | PR #6 |
> | T-33 Gaussian Process Regression | ROUGH_PLAN filed; awaits prioritization | `docs/plans/2026-04-30-T33-gaussian-process-regression.md` |
> | T-12 disk-mirrored logging | CLOSED via T-11 Unit A | (subsumed) |
> | T-15 LeaveOneGroupOut / GroupKFold | DROPPED — user decision; LOGO is footgun in user's data regime (5-100 N per site, 20× ratio); chemometrics literature recommends external test sets, not LOGO | n/a |
> | T-16 model-comparison machinery | REFRAMED — strategic split surfaced. Chemometrics canon (jackknife + Y-permutation = competitor parity) vs ML canon (paired t-test/Wilcoxon = "comparing between models"). Awaiting user decision on Shape A vs B vs hybrid | (investigation only) |
> | T-19 model-native imbalance handling | APPROVED with smaller scope (~2-3 days) — "extend single dropdown to dispatch native kwargs internally + fix 5 PLS-DA sites + close T-32." Math is already statistically equivalent today; gap is audit-trail labels + 5 unwired PLS-DA sites | (awaiting implementation) |
> | T-32 sample_weight length mismatch | DEFERRED into T-19 scope (unchanged) | n/a |
>
> **T-11 detail:** Three sub-units (A: disk-mirrored logging with `RotatingFileHandler` + tee-stdout proxy; B: thread-alive check + UI pause-state honesty with `_actually_paused` event; D: Optuna SQLite persistence + resume-on-startup dialog). Cross-family review pattern caught real bugs that would have shipped: trial-count overrun on resume (silently doubles work), non-atomic sidecar writes, dataset fingerprint stored but never enforced, study names too coarse (load_if_exists would mix incompatible trials), `_TeeStream` thread-unsafe, no log rotation, run-state firing for grid/NSGA-II falsely. All fixed. Pre-existing Nuitka `is_frozen()` bug also fixed (`__compiled__ in dir()` → `__compiled__ in globals()`).
>
> **Memory rules added this session:**
> - GLM 5.1 must NEVER be dispatched through opencode-go (bills against z.ai subscription)
> - CARS-Tree is dasp's invention, not in Li 2009 — don't search for canonical "tree-mode CARS"
> - T-19's user framing is "expose model-native abilities OR auto-detect," not paper-reproducibility
> - T-15 dropped + T-16 reframed (competitive-framework survey)
>
> **Pending user decisions still blocking bigger tickets:**
> - T-31 multi-class SIMCA (yes/no on usefulness)
> - T-22 bootstrap stability diagnostic (yes/no on investment)
> - T-01 reframe (external-test-set workflow yes/no)
> - **T-16 (NEW): pick Shape A (chemometrics canon) vs Shape B (ML canon) vs hybrid**
>
> See `docs/bugfix_validation/` for per-ticket validation notes. See `docs/CONTINUATION_PROMPT_2026-04-30_v2.md` for the next-session pickup prompt with all session lessons applied.
>
> **Previously:** 2026-04-30 (later — T-06 canonical SPA merged + T-06b parallelization follow-up) —
>
> **T-06 SPA canonical seed enumeration MERGED + T-06b parallelization MERGED.** Replaced non-functional `n_random_starts` knob with canonical Araújo 2001 deterministic enumeration over all J variables as candidate first variable. Branch `fix/T06-spa-canonical-seeds` ff-merged into `main`. Validation note: `docs/bugfix_validation/T06_spa_canonical_seeds.md`. Follow-up branch `fix/T06b-spa-parallel` ff-merged immediately after — parallelizes the J-seed loop via `joblib.Parallel(backend='threading')`, capped at min(cpu_count, 8) workers, PyInstaller-bundle-safe. Empirical FTIR-scale benchmark (J=800, n=50, n_features=20): 10.92 sec parallelized vs. ~70-80 sec sequential — ~7-8× speedup as expected for GIL-releasing numpy/sklearn work.
>
> Key gate-finding: the roadmap's proposed `rng.choice()` fix (Option A) was a sklearn-instinct slippage — canonical SPA per Araújo 2001 + auswahl + Galvão 2012 SPA-GUI is deterministic enumeration, not random restarts. No verified chemometrics implementation exposes a "random restart count" parameter. The gate's step-3 field-alignment lookup caught this; without it the fix would have invented a non-canonical pattern.
>
> Cross-family review (Codex + Kimi K2.6) caught: a missed call-site (`tests/gui/test_comprehensive.py:387`), template ↔ in-app divergence on small-sample CV-fold handling, dead `y_norm` code, and a Python-loop projection in the export template that was 100-1000× slower than the vectorized production path. All fixed before merge. Five new T-06 tests + 226-test regression sweep all green (post-parallelization too).
>
> **Previously:** 2026-04-30 (validation gate session — bugfix branches resolved) —
>
> **Bugfix branch validation gate complete.** Five branches that came out of the T-01 roadmap each went through a strict validation gate (literature check + commercial-software comparison + reachability verification + regression test sweep). Final dispositions:
>
> | Branch / Ticket | Disposition | Commit |
> |---|---|---|
> | T-26 SNV near-zero std | DROP / WONT_FIX (current dasp behavior matches PLS_Toolbox default at offset=0; bundled-app distribution makes a backend-only knob useless) | n/a |
> | T-10 PLS components clamp (LOO) | MERGED | `fbeb50c` |
> | T-05 VIP central formula | MERGED | `2c068cd` |
> | T-07 PDS even-window | MERGED | `1b91d93` |
> | T-24 Lin's CCC metric addition | MERGED | `0087cad` |
> | T-05a duplicate VIP formulas (templates + nsga2) | MERGED | `1eb6c06` |
> | T-04 one-class UVE prefilter | MERGED (GUI grey-out matching iPLS pattern) | `6beb5e8` |
> | T-21 SG uniformity guard | PENDING USER DECISION (DROP / DEFER / APPROVE_MINIMAL — radio-button path is safe; only the rarely-used Convert button creates the bug surface) | n/a |
> | T-32 sample_weight length mismatch | DEFERRED to T-19 (current path is unreachable; bug fires only after T-19's planned scope) | n/a |
> | `bayesian_utils.py random_state=42` hardcoding | MERGED (refactored to use the shared `RANDOM_STATE` constant) | `50d5d05` |
> | `search.py:2855 top_n_vars=30` hardcoding | DROPPED (display economy, not a bug — `all_vars` already preserves the full wavelength list for replication) | n/a |
> | Worktree cleanup (5 stale worktrees) | DONE — branch refs preserved locally | n/a |
>
> Per-branch validation notes at `docs/bugfix_validation/`. The gate itself caught two false alarms (T-26 + the two re-evaluation flags) where the underlying claim was real-but-unreachable or really a code-style issue. The recurring pattern: verify reachability + match-the-field BEFORE drafting verdicts. See `docs/bugfix_validation/README.md` "Lessons" section.
>
> **One pending user decision:** T-21 disposition. Investigation findings at `docs/bugfix_validation/T21_findings.md` flagged that the plan as written has citation errors (OpenSpecy `is_evenly_spaced()` is fabricated; PLS_Toolbox `gridcheck` is unverified) and a scope gap (~half of SG paths bypass the proposed guard). The user's typical workflow uses the radio-button relabel (which is safe — verified `_on_x_unit_override` does NOT call `convert_x_axis`); only the explicit "Convert" button creates the non-uniform-grid bug surface, and that button is rarely used in routine analysis.
>
> **Three pending user decisions still from the re-evaluation:** (1) T-31 multi-class SIMCA — confirm "none of the above" output is useful. (2) T-01 reframe scope — confirm external-test-set approach over per-fold varsel. (3) T-22 reframe — confirm bootstrap stability diagnostic investment.

> **Previously:** 2026-04-30 (full roadmap re-evaluation under chemometrics master rule) —
>
> **Roadmap re-evaluation complete.** All 32 tickets + deferred items re-evaluated against chemometrics literature + bone-FTIR application domain. Full results at `docs/RECONCILED_ROADMAP_2026-04-30_REEVALUATED.md`. Summary: **27 KEEP, 2 REFRAME (T-01, T-22), 2 DROP (T-02, T-03), 2 DEFER (T-05a, T-10b), 2 NEEDS_USER_DECISION (T-31).** Key findings: T-02 (ensemble OOF preprocessor) and T-03 (preprocessing-discovery full-data) are false alarms — same pattern as T-01 (sklearn-pipeline-purity misapplied to per-spectrum chemometrics operations). T-04 (one-class UVE prefilter on outlier-contaminated labels) is a REAL issue distinct from the leakage question. T-21 (SG uniformity guard) is chemometrics-correct (Savitzky & Golay 1964 assumes uniform sampling; PLS_Toolbox's `gridcheck` is precedent). T-22 reframed as bootstrap stability diagnostic. Top 5 next actions: T-15 (LOGO), T-16 (bootstrap CIs), T-11 (Optuna SQLite), T-01 reframe (external-test-set workflow), T-19 (loss reweighting).
>
> **Worktree disposition:** `fix/varsel-leakage` branch artifacts evaluated. `varsel_transformer.py` → MERGE_AS_OPTIONAL_TOOL (useful for T-22 stability diagnostic, expert mode, paper reproduction). `test_varsel_transformer.py` → MERGE. Audit doc → CHERRY_PICK (rename to VARSEL_ARCHITECTURE_AUDIT.md, fix labels). Plan doc + Codex reviews → DROP. See `docs/varsel_leakage_worktree_disposition_2026-04-30.md`.

> **Last updated:** 2026-04-30 (T-01 framing reconsidered + 5 small tickets implemented) —
>
> **Critical update:** the T-01 audit's "leakage" framing has been **reconsidered after literature validation by both Codex and GLM 5.1** (full analyses at `docs/analysis_vs_chemometrics_lit/`). Verdict: dasp's "varsel-on-full-calibration + CV" pattern is the **standard chemometrics workflow** documented in Li 2009 (CARS), Centner 1996 (UVE), Araujo 2001 (SPA), Norgaard 2000 (iPLS), Wold 2001 (PLS/VIP). The audit was technically correct as a *description of the code* but rhetorically wrong to label it "LEAKY" — that framing applied sklearn-pipeline-purity convention to a domain with its own established methodology. Furthermore, the user flagged that **per-fold varsel actively defeats the scientific purpose** of these methods (which is identifying wavelengths corresponding to real chemistry — stable selection across resamples is *evidence* the chemistry is real; per-fold instability destroys interpretability). The audit doc has been annotated with a banner pointing here. Read `docs/T01_VARSEL_LEAKAGE_AUDIT.md` (banner) + the two validation files for full reasoning.
>
> **Master rule (now memory-pinned):** dasp methodology decisions must be validated against chemometrics literature + the applied domain (bone FTIR / isotopes / paleoanthropology), NOT against sklearn / generic ML / genomics ML conventions. When literatures conflict, chemometrics wins. See `feedback_validate_against_chemometrics_and_application_lit.md` in user's auto-memory.
>
> **Status of in-flight work:**
> - `fix/varsel-leakage` Phase 1 (additive `VarselTransformer` infrastructure) is implemented but **paused — should NOT be merged**. Phases 2–7 paused indefinitely.
> - **T-02, T-03, T-04 in the roadmap need re-evaluation** under the same literature lens before any work. T-04 (one-class UVE prefilter on outlier-contaminated labels) may be a real issue distinct from the selection-on-full question and should be evaluated separately. T-02 / T-03 framing is suspect.
> - **5 implemented branches that came out of the T-01 roadmap (T-05 VIP, T-07 PDS, T-10 PLS clamp, T-24 CCC, T-26 SNV) are unaffected** — those are real bugs verified independently. All on `fix/T<X>-*` branches, none pushed, ready for review/merge:
>   - `fix/T05-vip-formula-fix` — canonical Wold (2001) VIP, 5+2+33+5 tests pass. T-05a deferred (duplicate VIP formulas in `templates/variable_selection.py:54` + `nsga2_search.py:627`).
>   - `fix/T07-pds-even-window` — odd-window enforcement + `apply_pds` hardening, 10+44 tests pass.
>   - `fix/T10-pls-components-clamp` — CV-aware clamp + new `compute_min_train_fold_size` helper, 16+5+1+113 tests pass. T-10b (NSGA-II audit) deferred.
>   - `fix/T24-lins-ccc` — Lin's CCC in 4 result paths + exported template, 14 unit tests pass.
>   - `fix/T26-snv-near-zero-std` — 1e-12 floor + warning, 5+14+31 tests pass.
> - **T-21 SG uniformity guard** has a plan committed on main (`d8c46f0`) but no implementation yet (codex review wedged). Plan applies a uniformity check to SG derivatives — **needs literature check before implementation** (the user flagged this in conversation: applying lessons from other literatures is the recurring failure mode).
>
> **Empty `opencode draft snapshot` commits** (2 in T-05, 1 in T-07, 3 in T-24) auto-created by the dispatcher wrapper after killed/snapshotted opencode sessions. Drop or squash on merge — they contain no diffs.
>
> **Next action recommended:** dispatch a fresh agent to re-evaluate the entire roadmap (`docs/RECONCILED_ROADMAP_2026-04-29.md`) under the master rule. Each ticket must be checked against chemometrics + bone-FTIR application literature before being treated as actionable.
>
> **Previously:** 2026-04-29 — T-10 PLS components clamp (cv_strategy-aware) — IMPLEMENTED on branch `fix/T10-pls-components-clamp` (not yet merged). New helper `compute_min_train_fold_size(cv_strategy, n_samples, n_folds)` in `cv_utils.py` returns the smallest training-fold size for kfold, repeated_kfold, loo, and repeated_loo strategies. `run_search` and `run_bayesian_search` now use it to clamp `max_n_components` before passing to `get_model_grids`. `models.py` contract documented via docstring only (no assert — Codex suggestion #3). Raises `ValueError` when `n_folds > n_samples` for kfold/repeated_kfold (Codex suggestion #1). Deferred NSGA-II audit ticket (T-10b): `_get_constrained_pls_components` clamps by n_samples but some evaluation paths already use min_train_samples from cv_folds; full audit of decode/result-row/reporting consistency needed before declaring same bug present. **Tests added:** 16 helper unit tests (`test_cv_strategy.py`), 5 integration tests (`test_cv_pls_clamp.py` for run_search), 1 Bayesian test (`test_cv_pls_clamp.py` for run_bayesian_search). Full sweep: 113 passed, 0 failed. **Commits:** `64c16d0` (task 1), `8de2cf4` (tasks 2+3), `1384625` (task 4), `0921e77` (task 5).

> **Previously:** 2026-04-29 (T-05 VIP formula fix) — Fixed `compute_vip()` in `models.py` to use canonical Wold (2001) formula (`SSY_a = q_a**2 * sum(T_a**2)` via `y_loadings_`) instead of the buggy `np.var(y)` scalar. Old formula collapsed all components to the same Y-weighting, skewing VIP rankings. 7 new regression tests in `tests/test_vip_formula.py`. All existing tests pass. Deferred follow-up T-05a: apply same fix to duplicated formulas in `templates/variable_selection.py:54` and `nsga2_search.py:627`.
> **Previously:** 2026-04-29 (T-24 Lin's CCC merged on `fix/T24-lins-ccc`) — Lin (1989) Concordance Correlation Coefficient added as an always-computed regression metric. New `lins_ccc()` helper in `scoring.py`; `CCC` (calibration) and `CCCcv` (CV-pooled) columns now emitted by all four regression result paths: grid (`search.py:4744-4748`), Bayesian-path-1 (`unified_bayesian.py:1437-1443` user_attrs), Bayesian-path-2 (`unified_bayesian.py:2056-2059` row dict), NSGA-II (`nsga2_search.py:3655,3690`). Exported validation template (`templates/validation.py`) gets an inline `_lins_ccc()` helper since templates can't import internal scoring. `ddof=0` (population variance) per Lin (1989) closed-form. Display convention: 0.0 for one-sided degeneracy, 1.0 for both-equal-constants. 14 TestLinsCCC tests pass, including `test_ccc_finite_sample_ddof_zero_exact` (CCC=4/7 for `y=[0,1,2], pred=[1,2,3]`) per Codex suggestion. GUI: CCC/CCCcv added to `higher_is_better_cols` for descending default sort + tooltip dict for column header hover.
>
> **Previously:** 2026-04-29 (T-01 audit complete) — Completed T-01 per-fold variable-selection leakage audit. Full report at `docs/T01_VARSEL_LEAKAGE_AUDIT.md`. Result: **60 LEAKY / 0 CLEAN / 35 N/A** across 19 methods x 6 search paths. Every y-using varsel method in every search path computes importances once on full calibration data and passes a frozen `top_indices` into CV — no method refits per-fold. Top 3: (1) grid CARS at `search.py:2552` (inflates R2cv 0.05-0.15), (2) grid `importance` mode at `search.py:2476` (0.02-0.05), (3) one-class UVE prefilter at `search.py:5649` (compounds with T-04). NSGA-II CARS guidance (`nsga2_search.py:1872`) is leaky but low-priority (guidance only, not per-fold varsel). Refinement path is clean (reuses saved wavelength list, no varsel re-run). Recommended fix: `VarselTransformer` sklearn wrapper for grid search, per-fold varsel inside Optuna objectives for Bayesian paths.
>
> **Previously:** 2026-04-29 (latest, post-Codex T-19 review) — Codex did a focused review of T-19 (the imbalance-handling ticket) and found three substantive issues: (1) **PLS-DA inner LR is split across ~10 sites, not 3** — five active sites still missing `class_weight` handling: `search.py:380`, `bayesian_utils.py:400`, `nsga2_search.py:822`, `nsga2_search.py:3453`, `ga_preprocessing.py:428`; (2) **the originally-prescribed "route sample_weight via imblearn fit_params" approach is broken** because imblearn's Pipeline does NOT rewrite fit_params after resampling — the right design is a custom estimator wrapper that recomputes sample_weight from the y received (post-resample). (3) **An active length-mismatch bug at `search.py:3883`** today — passes `sample_weight_train` (computed from pre-resampling `y_train`) against `y_train` instead of `y_train_for_model`, so when a resampler runs the lengths mismatch. New ticket **T-32** added to roadmap. T-19 effort revised up to 5-7 days. MLP support is now conditional on sklearn ≥1.7 PyInstaller bundle compatibility — if incompatible, skip MLP rather than block T-19. Roadmap, design doc, and SESSION_LOG all updated. Total tickets now 32 (P0/P1/P2) plus drop list.
>
> **Previously:** 2026-04-29 (latest) — **Reconciled roadmap from three independent reviews of the FTIR-paper + Unscrambler analyses**: see `docs/RECONCILED_ROADMAP_2026-04-29.md`. Three reviewers (Claude / Codex / DeepSeek V4 Pro, intentionally cross-family) deduplicated ~240 raw findings to 30 tickets across P0/P1/P2, plus 1 PENDING (multi-class SIMCA — needs user decision on whether "none of the above" outputs are scientifically useful) and 1 elevated per user request (T-17: multi-Y / PLS-2 workflow — substantial 2-3 week effort but high-leverage). Top-5 first: T-01 (per-fold varsel leakage audit nobody actually did), T-15 (LeaveOneGroupOut by site), T-11 (Optuna SQLite for crash-survivable resume), T-16 (bootstrap CIs + permutation tests for two-method comparison), T-12 (disk-mirrored logging). Key corrections during reconciliation: (a) "binary specificity bug" in LESSER_FINDINGS 1.1 is a false alarm (current code is correct under standard sklearn convention); (b) FTIR analysis claim that PLS-DA inner LR is stuck at `class_weight=None` is wrong — `search.py:4257-4261` already applies balanced weights when `imbalance_method='class_weight'`, so loss-reweighting design doc scope shrinks to boosting models only; (c) "509 except blocks" stat is unreproducible / scope-inflated; (d) HMAC signing of `.dasp` is technically insufficient (key extractable from app), use safe `extractall` + UX warning instead. New auto-memory: `feedback_chemometrics_conventions.md` — per-spectrum preprocessing (SNV / SG derivatives / baseline) is NOT data leakage by chemometrics community convention, so reviewers should not flag it; cross-sample operations (PCA fit on full data, varsel using y, hyperparameter tuning on full data) ARE real leakage and should be flagged.
>
> **Previously:** 2026-04-29 (later, continued) — Completed quick-wins audit of core modules. **`docs/QUICK_WINS.md`** — 14-item prioritized matrix (P0–P3) with exact line numbers, estimated fix times, and commercial-impact ratings. Highest-payoff items: (1) delete 8 debug prints in `search.py` (5 min), (2) replace ~66 prints with logger in `calibration_transfer.py` (15 min), (3) fix hardcoded version mismatch (`0.5.0b1` backend vs `v3` GUI/templates, 10 min), (4) fix VIP formula bug in `models.py` (15 min), (5) clean 42 prints + 29 bare excepts in `nsga2_search.py` (20 min). Total P0–P2 effort ~80 minutes for measurable commercial polish. See QUICK_WINS.md for full execution order.
>
> **Previously:** 2026-04-29 (later) — Completed gap analysis of dasp vs. the Sponheimer FTIR Bone PLS paper (in prep, working dir `C:\Users\mspon\Desktop\_DeskSync\FTIR Bone PLS\`). Full analysis in **`docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md`** — 10 confirmed-missing items, 5 partially-supported items, ranked priority recommendations. Top 3 gaps for contemplation: (1) **LOGO / GroupKFold CV** by site (paper's transferability claim depends on it; dasp has none), (2) **bootstrap CIs + paired permutation tests** for two-method comparison (paper's only inferential machinery; dasp has none), (3) **model-native loss reweighting for boosting models + PLS-DA inner LR** (resampling-only today; design doc at `docs/plans/2026-04-29-model-native-loss-reweighting.md`). Also flagged: ENLR is genuinely missing as a classifier (the existing `ElasticNet` card is the regression flavor), LDA-on-PLS-scores is missing as a model card, Lin's CCC is missing as a metric, stratified Kennard-Stone is missing for cal/test split. Use the gap analysis as the planning reference when prioritizing feature work.
>
> **Previously:** 2026-04-29 — Completed deep analysis vs. CAMO Unscrambler across all subsystems (8 parallel agents + 4 validation agents + 3 lesser-findings agents). Reports written to `docs/analysis_vs_unscrambler/` (6 documents: MASTER_ANALYSIS.md, TECHNICAL_GAPS.md, CODE_QUALITY.md, COMMERCIAL_READINESS.md, VALIDATION_SUPPLEMENT.md, LESSER_FINDINGS.md). Key findings: analytical engine exceeds Unscrambler in 9/17 categories, but commercial release is blocked by 12 issues. Validation supplement corrected understatements (error swallowing is 509 blocks, not 130; additional security issues: Zip Slip, ReDoS, GUI pickle.load). Lesser findings document catalogs ~200+ non-blocker issues including 11 silent correctness bugs, ~150 UX polish issues, 36 code hygiene items, 8 performance issues, and 14 robustness gaps. See analysis documents for phased release plan.
>
> **Previously:** 2026-04-21 (late evening) — Merged `fix/duplicate-target-col`: Data Management viewer no longer shows the target column twice when loaded from a combined-format CSV/Excel. Four view/export sites updated to filter target out of combined `metadata_cols` iteration (mirroring the already-correct separate-ref branch), plus a target-preservation guard in `_apply_data_viewer_edits` so the target-switch dropdown survives cell edits. Investigation by Codex + Kimi K2.6 in parallel; implementation by GLM-5.1 via z.ai direct (opencode build); independent verification by Codex + Claude; user manual verification with Valley CSV. Also: `9bd8751` handles RepeatedKFold in `compute_pls_complexity_curve` (sklearn's `cross_val_predict` rejects non-partition splitters — now falls back to manual per-fold aggregation) and surfaces validation-curve failures to the Results text instead of console-only. All regression tests still pass (144/144). Pending push.
>
> **Earlier today (same date, evening):** Reverted `_infer_task_type_from_y` to Fix-B1 heuristic (binary OR non-numeric → classification, else regression) after `0ef91e5`'s switch to `sklearn.utils.multiclass.type_of_target` caused integer-valued numeric targets (burn temperatures, 3-value proportion columns) to auto-flip back to classification — the exact bug `41371e0`/Fix-B1 was meant to prevent. Three stale auto-detect sites (`_run_analysis_thread`, `_update_task_type_label`, imbalance thread) that `0ef91e5` missed are now routed through the helper for consistency; user-reported crash on `group.cov.heath` (3 int-valued target, GUI auto-checked PLS-DA, backend resolved regression → `run_search`: "No valid models found") is resolved. 12/12 task-type preservation tests pass (was 10/12 against HEAD). Also: CV strategy tooltip overhaul — consolidated 3 duplicated inline texts into `TOOLTIP_CONTENT['cross_validation']['strategy_detail']`, added Kohavi (1995) / Krstajic et al. (2014) / Hastie et al. / Ezenarro et al. (2025) citations, widened Repeated-K-Fold inline-hint range to n=30–200.
>
> **Earlier today (same date):** Long session of imbalance-handling + task-type refinements. Session commits (in order): `a6955c5`/`b8a70cb`/`f8ab20b`/`ac43289` (imbalance task-type-aware dropdown + banner + logger fallback + banner-ordering fix), `f9fa206` (docs), `8d05dc0` (prediction-tab export duplicate-method fix), `aa73edd`/`8f14511`/`b9d2b39` (CV guard + type_of_target + refine preflight before model creation), `5dd00a0` (imbalance dropdown refreshes on task-type change), `8b2cd74` (don't auto-clear banner on redundant refresh), `18b2244` (classification y_range debug-print guard), `10d97a6` (target variable authoritatively drives task type, overrides sticky Import-tab radio except for one_class), `0ef91e5` (`_infer_task_type_from_y` helper + banner suppressed when imbalance disabled), `0f4a1e6` (Bayesian cost estimate no longer multiplies by phantom 10× preprocessing dimension), `0da5bd4` (LOO warning thresholds raised 10× to match small-spectral-dataset reality). Earlier: Mixed-type target Validation-Wide fix, `7407140` (one-class export support), `de6b618` (2x2 confusion matrix + NaN-safety), PLS-regression hotfix, phantom-class label normalization. Pending push.

> **Previously:** 2026-04-19 by GLM-5.1 — Analysis Subset V1 implemented on branch `glm/analysis-subset-v1` (NOT yet merged). New pure-logic module `src/spectral_predict/analysis_subset.py` with 41 tests. Analysis tab has "Analysis Subset" card above Holdout Validation. Metadata-only dialog with categorical multi-select. Subset provenance stored in `last_training_config`. Mismatch warnings in Model Development. One-class guardrail blocks when subset removes all inlier samples. C2 missing-column safe handling via `_refresh_active_group_indices`. All changes uncommitted in worktree. 41/41 analysis_subset tests + 61/61 OC regression tests pass. See `docs/SESSION_LOG.md` 2026-04-19 entry for architecture decisions and gotchas.

---

## Distribution path — bundle-only as of 0.5.0b1 (updated 2026-04-21)

**Decision:** the PyInstaller 3.12 bundle is now the only supported distribution path. Nobody is expected to clone and `pip install -e .`; the source-install scaffolding (`install.bat` / `install.sh` / `INSTALL.md`) stays in-repo as a developer convenience but is no longer marketed to end users. Beta version `0.5.0b1` ships exclusively as the bundled installer.

**Implication for parallelism:** the 3.12 bundle still uses the threading-backend fallback (see `src/spectral_predict/search.py:_frozen_needs_threading_fallback` — frozen-state-only, NOT version-gated; the original 3.12 plan to recover loky was wrong). Practical impact: numpy/sklearn/lightgbm/xgboost get thread-parallel speedup (those C extensions release the GIL), but pure-Python parallel loops (pymoo NSGA-II, GA-PLS evaluation) are single-core in the bundle. There is no longer a "use the source install for full multiprocessing" escape hatch for users — what the bundle does is what they get.

**Still in-repo from the source-install era (kept, not deleted):**
- `install.bat` / `install.sh`: detects Python 3.12, creates `.venv312`, runs `pip install -e . --upgrade`. Idempotent. Useful for developer setup.
- `INSTALL.md`: GUI-focused walkthrough — now developer-facing, not user-facing.
- `pyproject.toml` deps audited via AST scan. Added: `Pillow>=10.0.0`, `shap>=0.44.0`. Re-enabled: `jcamp>=1.2.1`. Floors bumped: `numpy>=2.0`, `pandas>=2.0`, `scikit-learn>=1.5`, `scipy>=1.11`.

**Intentionally NOT declared as required:**
- **`torch`** — no in-tree module imports it. T-38 deleted the last importer (`learned_preprocessing.py`); the dead `HAS_ENSEMBLE_PREPROCESSING` flag was removed at the same time. The build still excludes torch defensively in case a transitive PyInstaller import sneaks it in.
- **`agilent-ir-formats`** — no Python 3.12 wheel on PyPI. Stays in optional `[agilent]` extra. .seq file loading raises a clear ImportError from `agilent_reader.py:62` until upstream ships a 3.12 wheel.

**Open follow-up if bundle parallelism becomes a real bottleneck:** fix the PyInstaller spawned-child runtime hook (the argv-parse crash in `multiprocessing.freeze_support()`) so loky can be used in the bundle. Tractable but non-trivial — would need a custom runtime hook + verification across the supported Windows targets.

---

## 3.11 build retirement — DONE 2026-04-21

The 3.11 build path was archived to `archive/build_3_11/` on 2026-04-21 as part of the `0.5.0b1` beta cut. The 3.12 PyInstaller bundle is now the only build path. Files moved (with git history preserved via `git mv`):

- `spectral_predict.spec` → `archive/build_3_11/spectral_predict.spec`
- `spectral_predict_mac.spec` → `archive/build_3_11/spectral_predict_mac.spec`
- `build_installer.py` → `archive/build_3_11/build_installer.py`
- `installer/spectral_predict.iss` → `archive/build_3_11/installer/spectral_predict.iss`
- `docs/BUNDLED_APP_BUILD_GUIDE.md` → `archive/build_3_11/BUNDLED_APP_BUILD_GUIDE.md`
- `run_gui.bat` → `archive/build_3_11/run_gui.bat` (was `.venv311`-based)
- `run_v3.bat` → `archive/build_3_11/run_v3.bat` (deprecated Dear PyGui V3 prototype)

**Kept in repo root** (current `.venv312` dev launchers, NOT 3.11-specific): `RUN_SPECTRAL_PREDICT.bat`, `run_gui.sh`.

**Still TODO** (deferred — cosmetic, not blocking):
- Rename `spectral_predict_py312.spec` → `spectral_predict.spec`, `build_installer_py312.py` → `build_installer.py`, `installer/spectral_predict_py312.iss` → `installer/spectral_predict.iss`, `docs/BUNDLED_APP_BUILD_GUIDE_PY312.md` → `docs/BUNDLED_APP_BUILD_GUIDE.md`. Drop the `_py312` / `-py312` suffixes everywhere they appear (APP_NAME constants, .iss `OutputBaseFilename`, etc.). Save for after `0.5.0b1` ships so installer filenames stay stable through the beta.
- Delete `.venv311/` (or leave for occasional 3.11 reproduction if needed).

**Restore one file from archive:** `git mv archive/build_3_11/<file> <original-path>`.

---

## Tooltip coverage — RESOLVED 2026-04-17 (GLM 5.1 implementation)

Originally an audit-only finding (6 missing-tooltip items + 1 codebase discrepancy); GLM then implemented the fixes in the same session. **All 16 missing `CreateToolTip()` calls added** + new `TOOLTIP_CONTENT['one_class']` section (10 entries) + new `cross_validation['repeats']` key. Tooltips are research-backed (sklearn docs + chemometrics literature with citations: Bouckaert & Frank 2004, Schölkopf et al. 2001, Liu/Ting/Zhou 2008, Breunig et al. 2000, Rousseeuw & Van Driessen 1999, Wold 1976, Oliveri & Downey 2012).

**Coverage now includes:**
- CV repeats label + spinbox (main tab + refine tab)
- CV strategy combobox (main tab + refine tab)
- All 5 one-class model checkboxes (PCA-SIMCA, OneClassSVM, IsolationForest, EllipticEnvelope, LOF)
- Inlier class label + combobox
- All 4 one-class hyperparameter controls (`nu`, `contamination`, `alpha`, `n_components`)

**Verification: passed.** GUI imports cleanly, TOOLTIP_CONTENT['one_class'] dict has all 10 expected keys, no key shorter than 50 chars (sanity guard against placeholder text).

### `repeated_stratified_kfold` — clarified (was a documentation issue, not a bug)
Earlier audit said `RepeatedStratifiedKFold` doesn't appear in the codebase. **Refined finding:** the *string value* `'repeated_stratified_kfold'` is not in the dropdown — but the *class* `RepeatedStratifiedKFold` IS imported and used in `cv_utils.py`. The CV factory `build_cv_splitter` auto-selects between `RepeatedKFold` and `RepeatedStratifiedKFold` based on task type (classification → stratified, regression → unstratified) when the user selects `repeated_kfold`. So users see one dropdown option, get the right splitter under the hood. No user action needed; status doc PR #4 description was technically accurate, just misleading without that context.

---

## Recently resolved

### T-10: PLS component grid CV-strategy-aware clamp — FIXED 2026-04-30

`search.py:1109` and `search.py:3312` previously used `n_samples * (folds - 1) // folds`
to bound the PLS `n_components` grid. Correct for K-fold/RepeatedKFold, under-counts
for LOO (true train-fold size is `n - 1`, not `4n/5`). Added
`cv_utils.compute_min_train_fold_size(cv_strategy, n_samples, n_folds)` and routed
both grid clamps through it. LOO grids now reach the proper `n - 1` ceiling.
Also validates `n_folds > n_samples` geometry (Codex suggestion #1). Group splitters
(`group_kfold`, `leave_one_group_out`) deliberately raise `NotImplementedError`;
T-15 will plumb group-aware sizing.

Test coverage: `tests/test_cv_pls_clamp.py` — 16 unit tests on the helper +
4 grid integration tests on N=10 / N=80 synthetic data + 1 Bayesian integration test.
### T-05: VIP formula corrected to canonical Wold (2001) — FIXED 2026-04-30

Replaced `np.var(y)` per-component weight with canonical `q_a**2 * sum(T_a**2)` (Wold 2001, Mehmood et al. 2012 Eq. 1) in `src/spectral_predict/models.py:compute_vip`. Old formula collapsed all components to the same Y-weighting scalar, skewing VIP rankings whenever components had similar X-score energy but different Y-loading. Added `ssy_total <= 0` guard for degenerate fits. 7 new tests in `tests/test_vip_formula.py`. All 33 preprocessing_discovery tests + 5 PLS-DA importance tests pass unchanged.

**Deferred to T-05a:** Two additional copies of the buggy formula at `templates/variable_selection.py:54` and `nsga2_search.py:627` still need the same correction. See plan at `docs/plans/2026-04-29-T05-vip-formula-fix.md`.
### T-07: PDS even-window arithmetic — FIXED 2026-04-30

`estimate_pds` crashed on even window sizes due to B-allocation overflow (slice has window+1 columns for even window but B has window columns). Fix: reject even windows with clear ValueError. `apply_pds` signature changed to `window: int | None = None`, deriving geometry from `B.shape[1]` with FutureWarning on mismatch. 10 new tests in `tests/test_pds_window_arithmetic.py`. Commits: `d3d1606` (tests), `f438083` (fix).

### PLS regression hotfix: refine tab PLS-DA instead of PLS — FIXED 2026-04-21

Two bugs latent since commit `057d9f6` (2026-04-11), triggered by Burned-temperature data with ≤9 unique values:

1. **Fix A:** `_load_model_for_refinement` now trusts the saved Task type for regression/classification (was only trusting one_class). Extracted pure helper `_detect_refine_task_type(config, y)` for testability. Prevents the PLS → PLS-DA cascade when loading regression results into Model Development.
2. **Fix B1:** Removed `nunique() < 10` heuristic from auto-detect at all 9 sites. Only `nunique() == 2` and non-numeric dtype remain as classification triggers.

Tests: `tests/test_refine_task_type_preservation.py` (12 tests). 376/376 broader suite pass.

### OC validation helper silent-wrong-result paths + PCA-SIMCA default — FIXED 2026-04-18 (9e9ca11 + follow-up)

Three silent-wrong-result paths in `compute_validation_metrics_for_top_one_class_models` (`src/spectral_predict/contamination.py`) eliminated — all now warn and skip instead of producing believable-but-wrong `val_*` metrics:
- Malformed / missing / empty `Params` string → skip (was: default to `{}`, validate with estimator defaults)
- Partial wavelength-map miss → skip (was: silently drop missing wavelengths, validate on truncated feature set)
- Removed fallback to `selected_wavelengths` user-attr (for regression/classification Bayesian, only first 50 are stored there; `all_vars` always holds the full trained subset for all OC code paths)
- Follow-up: missing/empty/NaN `all_vars` also now warns and skips (was: silently validated on full spectrum) — both OC Bayesian (`unified_bayesian.py:1056-1062`) and OC grid (`search.py:5332`) always populate it, so a missing value means row metadata is corrupt

Also bumped PCA-SIMCA `n_components` default from 5 to 10 at `gui:2892` (more sensible starting point for typical NIR datasets). Regression test suite: 6 new parametrized cases in `TestGridSearchValidationMetricsParity::test_corrupt_metadata_row_is_skipped_not_silently_wrong` covering all warn+skip paths. 54/54 `test_contamination_detection.py` pass. Closes three "Follow-Ups (unclaimed)" items flagged MAJOR by Codex in the pre-merge review.

### LOO + Repeated K-Fold cross-validation strategies — MERGED 2026-04-17 (PR #4)

Added `LeaveOneOut`, `RepeatedKFold`, and `RepeatedStratifiedKFold` as first-class CV options across regression / classification / one-class. Backend: `src/spectral_predict/cv_utils.py` — factory `build_cv_splitter`, `validate_cv_strategy_for_task`, `cross_val_predict_pooled`, `reduce_repeated_cv_predictions`. Pooled RMSEcv (regression — fixes the LOO degeneracy where mean-of-fold-RMSE = MAE); pooled sensitivity/specificity (one-class); BER consistency under repeated CV; AUC kept as mean-of-fold-AUCs (scores not comparable across independently-fit folds). GUI: Analysis-tab + Model-Development controls, cost estimator with LOO/Repeated warnings. `training_config` stores `cv_strategy` for save/load. Includes a code-export fallback in `code_generator.py:108-115` so exported scripts/notebooks honor the user's CV regime when callers pass `training_config` rather than top-level keys (caught in pre-merge codex review). 1456-line test suite at `tests/test_cv_strategy.py`, 5 prior review rounds visible in commit history.

### LightGBM "parameter capture" warning + NaN / crash on `.venv311` — FIXED 2026-04-16 (PR #5)

Fix: `clone(model)` at four pipe-construction sites in `search.py` (`:2191`, `:4161`, `:4163` for the regression bug, plus `:4139` PLS-DA defensive). Root cause: shared estimator instance being fit first on full preprocessed X (setting `n_features_in_=2151`), then reused in subset-feature fits; sklearn 1.5.2's pre-fit `_check_n_features(reset=False)` fires before the refit would reset the count, raising. sklearn 1.7.2's pre-fit check is more lenient and doesn't raise on the same pattern. The PLS-DA `:4139` site was added after Claude pr-reviewer flagged it as the same pattern; classification baseline run on `.venv311` with PLS-DA un-cloned proved PLS-DA doesn't actually hit the bug (importance-capture at `:2191` doesn't fire for PLS-DA), so the clone there is purely defensive/symmetric. (The separate sklearn 1.7.2 `"X does not have valid feature names"` UserWarning flood seen on `.venv312` comes from a DataFrame-fit / ndarray-predict transition in the pipeline; it's unrelated to the shared-state bug and is NOT changed by this fix — see COMPARISON.md.)

Verification: harness `scripts/verify_shared_model_fix.py` run with GUI defaults on BoneCollagen — 4-run before/after matrix on both venvs. Post-fix `.venv311` produces 9408 LightGBM rows with `best_cv_rmse=0.9702327793086989`, bit-identical to `.venv312`. See `docs/plans/2026-04-16-lightgbm-shared-model-fix.md` for the plan and `docs/plans/artifacts/2026-04-16/COMPARISON.md` for the metrics matrix.

## Known follow-ups (deferred from PR #4 reviews, non-blocking)

- **`run_bayesian_search()` does not call `validate_cv_strategy_for_task()` upfront.** Located in `src/spectral_predict/search.py:3096`. Unlike `run_search()` and `run_unified_bayesian()` which validate the CV-strategy / task-type / class-count combination before any trials run, `run_bayesian_search()` accepts `cv_strategy` / `cv_n_repeats` and proceeds directly. Result: a bad CV config (e.g. LOO requested with task='classification' and a class with <2 samples) would only fail mid-trial inside the Bayesian objective rather than producing a clean preflight error. Codex flagged this as non-blocking in the PR #4 review. Fix is a one-liner: add `validate_cv_strategy_for_task(...)` at the top of the function, mirroring the other two entry points.
- **Style nits in CV code**: `cv_utils.py` duplicates the `RepeatedKFold` import (module-level + inside `build_cv_splitter`); `templates/validation.py` prints `({cv_folds}-fold)` even for `loo`/`repeated_kfold` (misleading once strategy-specific exports are working); `_majority_vote` should default to NaN in the empty-votes else-branch for safety; LOO `splits = list(...)` materialization at `search.py:~4228` is O(n²) memory for very large datasets (fine for n<5000 spectral data but worth a comment).
- **Suppress sklearn 1.7.2 `"X does not have valid feature names"` UserWarning flood** on `.venv312` — cosmetic noise, ~12MB stderr per grid run, unrelated to any bug. Consider `warnings.filterwarnings(...)` in the GUI entry point.

---

## Active Branch context (secondary)

## Active Branch: `claude/contamination-detection-model-PrntN`

**Goal:** Add one-class contamination detection models as first-class citizens alongside regression and classification. One-class models learn from clean/inlier samples only and flag anything outside that distribution as suspect.

**Models:** OneClassSVM, IsolationForest, LOF (LocalOutlierFactor), EllipticEnvelope, PCA-SIMCA

**Prediction tab manual verification: PASSED 2026-04-11.** User confirmed one-class predictions correctly show inliers when they should after the 2026-04-10 order-of-operations fix. See SESSION_LOG.md 2026-04-10 entry for the original bug trace.

---

## What Works

- [x] **Analysis Subset V1** (branch `glm/analysis-subset-v1`, uncommitted): Pure-logic module `src/spectral_predict/analysis_subset.py` with 41 tests. Analysis tab card, metadata-only dialog with categorical multi-select, subset provenance in training config, mismatch warnings, one-class guardrail, C2 missing-column safe handling.
- [x] One-class model implementations in `src/spectral_predict/contamination.py`
- [x] Bayesian optimization for one-class (`unified_bayesian.py` handles `task_type='one_class'`)
- [x] Grid search for one-class (`run_one_class_search` in `search.py`)
- [x] GUI task type selection: "One-Class" radio button shows one-class model checkboxes
- [x] Inlier class selection UI with auto-detection
- [x] Model save/load with scaler + PCA reducer persistence (`model_io.py`)
- [x] Prediction tab: save→load→predict round-trip verified manually 2026-04-11. Training inliers come back as "Inlier (X)" as expected. Regression test: `tests/test_contamination_detection.py::TestOneClassRefinementRoundtrip`.
- [x] External validation: label mapping, confusion matrix, balanced accuracy/sensitivity/specificity
- [x] External validation metrics in Results tab — top N respects `validation_top_n` (default 700), matching classification/regression
- [x] External validation produces full 7-metric set (Sensitivity, Specificity, Precision, F1, Accuracy, BalancedAcc, AUC) via `compute_validation_metrics_for_top_one_class_models()` in `contamination.py` — parity with cal/CV
- [x] External validation metrics in Model Development results text
- [x] Validation checkbox preserved when loading results into Model Development
- [x] Dancing man animation stops on completion
- [x] Model Development: runs complete, buttons re-enable, cursor resets
- [x] Wavelength importance: surrogate LightGBM via `compute_one_class_importances()`
- [x] Code export: one-class models export as standalone Python scripts and Jupyter notebooks with full CV reproduction
- [x] SHAP: permutation importance for most models, TreeExplainer for IsolationForest
- [x] Diagnostics: decision score distribution + sample classification plots
- [x] Residual/leverage diagnostics route to one-class-specific plots (not regression fallback)
- [x] Basic preprocessing discovery works for one-class grid search (callback wrapper fixed)
- [x] Variable selection: 'importance' method works for one-class
- [x] Results Treeview tooltips: all one-class metrics (Sensitivity/AUC/cv) + validation metrics (RMSEP/R2pred/val_*) now covered; jargon-heavy tooltips (ROC_AUC, Kappa, MCC, BER, LogLoss) rewritten for non-technical audience
- [x] CV strategy support: LOO, Repeated K-Fold, and standard K-Fold via `build_cv_splitter()` factory in `cv_utils.py`. Pooled RMSEcv (regression) and pooled sensitivity/specificity (one-class). GUI controls in Analysis tab + Model Development. Cost estimator with LOO/Repeated warnings. training_config stores cv_strategy for model save/load. Post-review fixes (2026-04-12): RepeatedKFold crash via `cross_val_predict_pooled`, one-class Bayesian cv_strategy forwarding, GUI LOO folds metadata, LOO classification minority-class guard, differentiated mixed-regime warnings.

## Known Issues

- [x] ~~**Bayesian one-class results NaN on Results tab**~~ — Fixed: enabled `compute_calibration=True` in Bayesian objective
- [x] ~~**Scaler/PCA not persisted in grid search results**~~ — Fixed: `cal_scaler`, `cal_pca_reducer`, `oc_score_stats` now stored in result dicts
- [x] ~~**LightGBM label encoding bug**~~ — Fixed: `compute_one_class_importances()` now converts +1/-1 to 0/1
- [x] ~~**"Contamination" naming**~~ — Fixed: comments/docstrings updated where "contamination" referred to task type
- [x] ~~**Performance: IsolationForest/LOF grid bloat**~~ — Fixed: reduced IF n_estimators (300→100), removed redundant LOF configs, skipped unused importance in discovery
- [x] ~~**PCA-SIMCA silently produces 0 rows on small training sets**~~ — Fixed 2026-04-10: `PCASIMCA.fit()` had a hardcoded `n_samples < 10` guard that tripped every CV fold when training inliers were scarce (e.g. ~7 inliers + 5-fold CV → 5-6 per training fold). All 5 folds failed → skip guard → `+inf` every trial → 0 rows in final DataFrame. Other one-class models (OCSVM/IF/LOF/EllipticEnvelope) all have explicit small-sample fallbacks. Relaxed SIMCA floor to `n_samples < 3` (the true mathematical minimum for PCA + chi² moment fit). Also added `skip_reason` surfacing through `run_one_class_cv` → `unified_bayesian` progress wrapper so any remaining skips show "SKIPPED (reason)" in the GUI progress tab instead of a silent `-inf`.
- [x] ~~**Every specimen labeled "Outlier" at predict time, including training specimens**~~ — Fixed 2026-04-10 (second session). Root cause: one-class refinement thread in `_run_refined_model_thread` trained with full-spectrum preprocessing first then subset to selected wavelengths, but hardcoded `'use_full_spectrum_preprocessing': False` in the saved metadata and never populated `full_wavelengths`. At predict time `model_io.py` took the subset-first branch, which for SNV / SG derivatives / baseline correction produces feature values mathematically different from training, so the StandardScaler fit inside `run_one_class_cv` received distributionally alien inputs and `model.predict()` returned -1 on everything. Fix: assign `self.refined_full_wavelengths = list(original_wavelengths)` and flip the flag to `True`, matching the regression/classification refinement path. Regression test added: `tests/test_contamination_detection.py::TestOneClassRefinementRoundtrip`. See `docs/SESSION_LOG.md 2026-04-10 "One-Class Prediction Disaster"` for full trace.
- [x] ~~**Uncertainty subtab always shows "Outlier" for one-class**~~ — Fixed 2026-04-10 (second session). `_display_uncertainty()` at line ~38318 read already-stringified predictions (`"Inlier (…)"` / `"Outlier"`) from `predictions_df` but compared them to the integer `1`, which always evaluated False. Would have silently contradicted the Results tab once the primary bug above was fixed. Replaced with an `isinstance(raw, str)` branch.
- [x] ~~**Bayesian trials did not persist `cal_scaler`, `cal_pca_reducer`, `oc_score_stats`**~~ — Closed 2026-04-10 (second session). Latent bug: `unified_bayesian.py` stored metrics and preprocessing config but not the fitted calibration artifacts, so any future direct-save-from-results path would silently save `None`. Added `set_user_attr` calls mirroring `search.py:5171-5173` and propagated the three keys through `convert_study_to_dataframe`. Does not change current user-visible behavior because saves go through refinement, which re-runs CV from scratch.
- [x] ~~**Grid-search one-class validation columns silently empty**~~ — Fixed 2026-04-11. Grid search wrote display name (`'snv_deriv1_w11'`) in `Preprocess` but never set `PreprocessBase`, so the validation helper passed an unparseable name to `build_preprocessing_pipeline`, the resulting `ValueError("Unknown preprocess: …")` was swallowed by the row-level except, and val_* dropped to NaN for every derivative grid-search row. Bayesian was unaffected because it already wrote `PreprocessBase`. Two-layer fix: search.py now writes `PreprocessBase` (mirrors classification at line 4506), and `compute_validation_metrics_for_top_one_class_models` now normalizes display names as a fallback. Regression test: `tests/test_contamination_detection.py::TestGridSearchValidationMetricsParity`. See `docs/SESSION_LOG.md 2026-04-11`.
- [ ] **One-class grid search inherently slower than classification** — Expected due to two-phase CV+calibration design and LOF O(n²) complexity. Optimized but still slower by nature.
- [ ] **Preprocessing importance dropdown has no effect for one-class** — All methods resolve to LightGBM. Per-model refinement never triggers because `models_to_test` isn't passed.
- [ ] **Variable selection limited** — Only 'importance' method works. UVE/SPA/iPLS/CARS are PLS-specific and incompatible. This is by design but could use a UI hint.
- [ ] **Residual correlation overlay** — Disabled for one-class (no continuous residuals). Shows zeros.
- [x] **One-class hyperparameter grids now exposed in Model Config subtab** — Fixed 2026-04-19. Five collapsible cards in Tab 4C (OneClassSVM, IsolationForest, EllipticEnvelope, LOF, PCA-SIMCA) using old-style BooleanVar + Checkbutton pattern. Untouched cards preserve curated default grids. Customized cards use per-model Cartesian product. Bayesian one-class path unchanged.
- [ ] **LOO / Repeated K-Fold results may not auto-populate the Results tab** — observed 2026-04-12 during manual testing of PR #4 (cv-strategy-overhaul). User ran LOO on a 49-sample regression dataset; analysis completed without errors, but the Results tab did not display the completed runs as expected. Repeated K-Fold uncertain — not explicitly tested for the same behavior. Save/load/predict round-trip from a refined LOO model works. Possible causes to investigate: (a) Results tab's Treeview populate path is gated on `Folds` matching a specific integer range or on `training_config['cv_strategy'] == 'kfold'`; (b) progress callback's final "completion" signal isn't wired for non-kfold strategies; (c) sort/filter logic in `_populate_results_treeview` (or equivalent) drops rows where `Folds == n_samples` (LOO case). Needs repro + trace before fixing.

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Surrogate LightGBM for wavelength importance | Model-agnostic, fast, already implemented in `compute_one_class_importances()` |
| Permutation importance instead of SHAP KernelExplainer | KernelExplainer is impossibly slow for spectral data (hundreds of wavelengths × thousands of model evals per sample) |
| TreeExplainer only for IsolationForest without scaler/PCA | Only case where SHAP TreeExplainer works directly |
| Validation metrics: balanced accuracy, sensitivity, specificity | Standard one-class metrics. Sensitivity = outlier detection rate, Specificity = inlier retention rate |
| String comparison for inlier labels | Both Bayesian and grid search convert y and inlier_label to strings before comparison to handle numeric/text label mismatches |
| Early return pattern in GUI threads | One-class has separate pipeline from regression/classification. Every early return MUST include cleanup (animation stop, button reset, etc.) |

## Key Files

| File | Role |
|------|------|
| `spectral_predict_gui_optimized.py` | Main GUI (~45K lines). One-class paths scattered throughout. |
| `src/spectral_predict/contamination.py` | One-class model implementations, `run_one_class_cv()`, `compute_one_class_importances()` |
| `src/spectral_predict/search.py` | `run_one_class_search()` for grid search |
| `src/spectral_predict/unified_bayesian.py` | Bayesian optimization, handles `task_type='one_class'` |
| `src/spectral_predict/scoring.py` | Scoring functions with one-class metrics |
| `src/spectral_predict/model_io.py` | Save/load with scaler/PCA persistence |
| `src/spectral_predict/preprocessing_discovery.py` | Smart preprocessing, has one-class path at line ~680 |

## Next Steps

1. ~~End-to-end manual test of save→load→predict~~ ✅ done 2026-04-11
2. ~~Re-run grid-search OC validation in the GUI to confirm `val_*` columns now populate~~ ✅ done 2026-04-11 — user confirmed "the validation stats now appear in grid search"
3. ~~Merge PR #3 to main~~ ✅ pending merge in this session
4. Follow-up PRs (post-merge): see "Follow-Ups (unclaimed)" section below

## Follow-Ups (unclaimed)

- **Model-native loss reweighting for class imbalance — not exposed for boosting models or PLS-DA's inner LR (flagged 2026-04-29).** Today the Data Quality tab handles imbalance via resampling (SMOTE family) and `class_weight='balanced'` is injected only into `RandomForest` / `LogisticRegression` / `SVC`. The boosting model construction sites in `models.py` (XGBoost line 285, LightGBM 307, CatBoost 327) never receive `scale_pos_weight`, `is_unbalance`, `class_weight`, or `auto_class_weights` — verified by grep, zero matches in `src/`. PLS-DA's inner `LogisticRegression` (`search.py:380`, `:4261`, `bayesian_utils.py:400`) is also stuck at default `class_weight=None`. This means standard chemometrics configurations like the FTIR Bone PLS paper's "XGBoost (scale_pos_weight)", "LightGBM (balanced)", and "Logistic regression (EN, balanced)" are not reproducible in dasp without source edits or routing through resampling (which is a different statistical intervention). Proposed fix: per-model "Imbalance handling" dropdown with four options (Equal sample weights / Auto / Class-balanced loss reweighting / scale_pos_weight custom). Auto mode reuses `detect_class_imbalance(y, threshold=3.0)` from `imbalance.py:61` to decide per-fold at fit time. Important UX: must clearly distinguish "Equal sample weights" (current default — does NOT mean balanced; means majority class dominates the loss) from "Class-balanced loss reweighting" (corrects the imbalance), since the standard library defaults are widely misunderstood. Full proposal in `docs/plans/2026-04-29-model-native-loss-reweighting.md`. Effort estimate ~1-2 days, low risk because default preserves all existing behavior.

- **Expose remaining one-class hyperparameters (partial exposure shipped 2026-04-19).** The 2026-04-19 parity work added Tab 4C cards covering the parameters already in the curated `get_one_class_model_grids()` defaults — OCSVM `kernel`/`gamma`/`nu`/`degree`, IF `n_estimators`/`contamination`/`max_features`, EllipticEnvelope `contamination`, LOF `n_neighbors`/`contamination`, PCA-SIMCA `n_components`/`alpha`. Deferred parameters that sklearn supports but the GUI still does not surface: **LOF `metric`**, **IsolationForest `max_samples`**, **OneClassSVM `coef0`** (and additional kernels like `sigmoid`/`linear`), **EllipticEnvelope `support_fraction`**. Also not yet exposed: additional predefined values beyond the shipped curated grid (e.g. OCSVM `nu=0.2`, IF `n_estimators=200/500`, LOF `n_neighbors=50`, PCA-SIMCA `alpha=0.10`) — users can enter these as custom values today, but there's no predefined checkbox. Finally, the one-class **Bayesian** path was deliberately out of scope for 2026-04-19 and still uses the hardcoded parameter ranges in `unified_bayesian.py::suggest_one_class_params()` rather than anything from the new cards; any future work to honor user-customized grids in Bayesian search would need a separate design. Exposing the deferred parameters requires: (a) extending the curated default grid in `get_one_class_model_grids()`, (b) adding checkbox rows to the corresponding Tab 4C card in `spectral_predict_gui_optimized.py::_create_tab4c_model_configuration`, (c) extending the per-model builder in `_resolve_one_class_model_grids()` in `src/spectral_predict/search.py`, and (d) adding default-state comparison entries in `_collect_one_class_model_param_overrides()` so the curated-default preservation invariant still holds. The 2026-04-19 plan's "Non-Goals" section lists these explicitly as future follow-ups.

- **`predict_with_uncertainty` swallows one-class decision_function failures silently.** `model_io.py:854-860`: after `predict_with_model()` returns, the bare `except Exception: decision_scores = None` catches any score-extraction error and returns predictions with empty `uncertainty`/`applicability_domain` payloads. The GUI has no way to surface "predictions worked, uncertainty broke" — exactly the failure mode this PR has already hit once with the order-of-operations bug. Fix: log the exception with `logger.warning(...)` at minimum; ideally return a structured `decision_score_error` flag that the GUI can display alongside the predictions.

- **OC validation helper can't recover the right `polyorder` for 2nd-derivative grid-search rows.** Grid search writes `polyorder=None` (delegating to `polyorder_map` inside `SavgolDerivative` which picks 3 for `deriv=2`), but the validation helper at `contamination.py:922-927` falls back to `min(2, window-1) if window > 2 else 0`, which gives `poly=2` for `deriv=2`. The pipeline then runs with the wrong polyorder and the resulting `val_*` metrics are slightly off vs. what training actually used. Fix: either store the resolved polyorder in the grid-search result dict (most direct), or import the same `polyorder_map` lookup into the validation helper, or have `_maybe_int` fall back to `polyorder_map[deriv]` when poly isn't set.

- **CV Strategy Phase 2: Propagate outer CV strategy into variable selection inner loops.** Currently inner loops (UVE, SPA, iPLS, CARS, GA-PLS) always use hardcoded 5-fold K-fold regardless of the outer CV strategy. A GUI warning is displayed when outer != kfold. Phase 2 would thread `cv_strategy`/`cv_n_repeats` into the variable selection internals.

- **CV Strategy Phase 2: Propagate CV strategy into predictor screening and smart preprocessing.** These internal CV sites are currently hardcoded 5-fold and don't respect the user's chosen strategy.

- **CV Strategy Phase 2: NSGA-II multi-objective search CV strategy support.** Currently falls back to K-fold with a logged warning when non-kfold strategy is selected. Needs native LOO/Repeated K-fold support.

- **One-class search: Add `training_config` to result rows.** Regression/classification grid search writes `training_config` (with `cv_strategy`, `folds`, `cv_n_repeats`) but one-class search does not. This means one-class saved models don't preserve the CV strategy used during training.

- **One-class export lacks automated regression tests.** The existing 33 export tests (`tests/test_code_generator.py`, etc.) do not exercise the `task_type='one_class'` path. Manual verification confirms all 5 models compile, but a future refactor of `templates/validation.py` or `code_generator.py` could silently break one-class export. A single parameterized test generating scripts for all 5 OC models would be cheap insurance.

- **Template string formatting fragility in one-class CV block.** `CROSS_VALIDATION_ONE_CLASS_TEMPLATE` uses `.format()` with `{model_name}` and `{x_var}` variables inside a large multi-line template string. This is correct today but slightly fragile: a future edit that introduces an additional brace pair (e.g. a dictionary literal `{` `}`) would cause a `KeyError` at generation time. This is pre-existing technical debt in the template system, not introduced by this commit, but worth noting if the template engine is ever refactored.

- **One-class export lacks per-fold CV statistics printout (parity bug).** Regression export prints pooled metrics **and** per-fold breakdown (`print(f"\\nPer-fold RMSE: {{[f'{{x:.4f}}' for x in fold_rmse]}}")`). One-class CV template computes `fold_metrics` list but `METRICS_ONE_CLASS_TEMPLATE` prints only pooled metrics, not the per-fold details. This is a **parity mismatch** with the regression export experience. Classification also lacks per-fold printout but gives detailed confusion matrix + classification report instead, which one-class already does via its own confusion matrix plot. Still, for parity with regression's explicit per-fold reporting, one-class should print per-fold sensitivity/specificity/AUC if available.

- **Pause/resume hardening — worth real design work (flagged 2026-04-23).** User lost hours on a CatBoost run after Pause → Resume appeared to do nothing; log was gone by the time we could inspect it (progress pane is widget-only, no disk log). Diagnosis surfaced four real gaps in the current `SearchController` implementation (`src/spectral_predict/search_controller.py` — two `threading.Event`s; worker calls `check_and_wait()` at checkpoints):
  1. **Coarse checkpoint granularity in the Bayesian path.** `unified_bayesian.py:1826` only checks pause in Optuna's `progress_wrapper` callback, which fires *between* trials. A single CatBoost trial on a 19-hour search can be 10–30+ minutes. Pause's UI flips to "paused" instantly but the thread is still burning CPU until the current trial finishes — and if the user Resumes before that happens, pause/resume is a silent no-op.
  2. **UI lies about pause state.** `_pause_search` (gui:22842) calls `_update_search_buttons('paused')` the instant it's clicked, regardless of whether `check_and_wait()` has actually been reached. Should display an intermediate "pausing — waiting for current trial to finish…" until the thread actually blocks.
  3. **No thread-alive check on Resume.** If the worker thread silently errored during the paused window (exception caught by the `except Exception` at gui:27664 which logs `[X] Error:` and exits the thread), `_resume_search` (gui:22849) just sets the pause event — nobody is waiting on it, no feedback to the user, UI returns to "running" state with a dead thread. Resume should call `self.analysis_thread.is_alive()` and surface a clear error if the thread has died.
  4. **Zero persistence.** Pause is purely in-memory. Any crash, OOM, Windows update, or process exit while paused loses everything. Optuna has built-in SQLite persistence (`storage="sqlite:///..."`) — adding this would make resume survive process death and make a 19-hour run recoverable; likely the single highest-value change.
  Bonus gap: the progress log is Tkinter-only (`_log_progress` at gui:27770 only writes to `self.progress_text`), capped at 2000 lines (gui:27782), never mirrored to disk. Post-mortem of any long run is impossible once the app closes. A simple `logging.FileHandler` writing to `outputs/logs/<run-timestamp>.log` would have let us diagnose this incident definitively.

## Analysis Subset V1 — Known Limitations / Risk (2026-04-19)

Blocking bugs (stale `active_indices` after dataset replacement or row deletion) fixed in this session. Remaining deferrals:

- **Integration-level GUI tests not yet written:** reload-with-subset, row-delete-with-subset, revert-after-column-delete, and mismatch-warning text. Pure-logic coverage (41 tests in `test_analysis_subset.py`) is solid, but no automated test drives the GUI through these multi-step scenarios.
- **Manual GUI verification checklist still pending:** end-to-end validation that the Analysis-tab card, data-viewer highlighting, provenance in training config, and one-class guardrail all behave correctly after each dataset-replacement path.
- **`_use_for_analysis()` does not carry `combined_metadata_df`** from data sources. If a user loads data via Data Management → Use for Analysis, metadata columns may be absent, causing the subset dialog to show no columns. This is a pre-existing limitation of the data-source path, not introduced by Analysis Subset V1.
