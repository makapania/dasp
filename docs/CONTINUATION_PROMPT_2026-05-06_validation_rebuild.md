# Continuation prompt — validation-rebuild + ensemble-rebuild class_weight gap

**Filed:** 2026-05-05 afternoon, after PR #38 (class_weight discriminator sister-site fix) closed the optimization-loop training paths. Kimi K2.6's exhaustive sister-site sweep on PR #38 round-2 surfaced TWO PRE-EXISTING gaps in DOWNSTREAM rebuild paths that share the same bug-class shape but are different consumers.

**For:** the next agent picking up the validation-rebuild + ensemble-rebuild fix.

**Branch state at filing:** all 7 PRs from the 2026-05-05 morning + afternoon are open, none merged. `origin/main` at `0576358`. Working tree clean.

---

## What's already done (do NOT reopen)

PR #38 (`fix/Tclass-weight-discriminator-sister-sites`) is **READY_TO_MERGE** with four-way convergent reviews. It closes the bug class for FOUR sister sites in the optimization-loop training paths:

1. `unified_bayesian.objective` — Bayesian trial CV + calibration refit
2. `nsga2_search.SearchProblem._evaluate` — NSGA-II main eval CV
3. `nsga2_search._compute_classification_cv_metrics` — NSGA-II Results-panel `*_cv` columns
4. `nsga2_search._compute_calibration_metrics` — NSGA-II Results-panel non-`*_cv` calibration columns

Plus `search.py:4424` had its CatBoost branch made explicit (was using sample_weight fallback — functionally weighted but mechanism-divergent). Plus 2 def-in-depth `'auto'` guards in `_needs_resampling_pipeline`.

**Do not re-litigate any of these.** All four sites verified by Codex + GLM + DeepSeek + Kimi.

---

## What this prompt is for

Two NEW gaps surfaced by Kimi during the PR #38 round-2 sister-site sweep. **Same bug class** (model rebuilt without applying the imbalance discriminator), **different consumers** (downstream re-fit paths, not the optimization-loop CV/calibration paths PR #38 fixed):

### MEDIUM — `compute_validation_metrics_for_top_models` (`src/spectral_predict/search.py:408`)

This function computes the `val_*` columns shown in the Results panel when the user enables a held-out validation set (`validation_count > 0`). It rebuilds each top-N model via `_rebuild_model_from_row` (line ~739) and fits at line 747:

```python
model.fit(X_train_final, y_train)  # line 747 — imbalance_method NOT threaded
```

**Two failure modes**:

1. **XGBoost retrains UNWEIGHTED** for the validation set. XGBoost's class_weight handling lives at fit-time via `sample_weight` (no constructor kwarg), so it's not in the `Params` string captured on the result row. `_rebuild_model_from_row` reconstructs from that string, losing the weighting.
2. **PLS-DA retrains UNWEIGHTED** for the validation set. PLS-DA's class_weight lives in the LR constructor at `search.py:4488` (`LogisticRegression(class_weight='balanced', ...)`) but the LR sub-step's params are NOT serialized into the result-row `Params` dict — only the PLS step's `n_components` ends up there. `_rebuild_model_from_row:388` reconstructs LR without `class_weight`.

**Reached when**: `validation_count > 0` in `run_search` (Grid) or `run_bayesian_search` (Bayesian). Hit on every classification job with `imbalance_method='class_weight'` (or `'auto'` resolving to it) where the user has enabled validation. Core CV metrics + calibration metrics are correct (PR #38 fixed those); only the `val_*` columns are wrong.

### LOW — `_reconstruct_models_from_results` (`spectral_predict_gui_optimized.py:23643`)

Ensemble-reconstruction path used by the "Train Ensemble" feature. Fits rebuilt pipelines at four sites (lines 24086, 24106, 24112, 24120) without applying class_weight / auto_class_weights / sample_weight. Same shape as the validation-rebuild bug. Affects only the ensemble feature, not primary search results.

---

## Fix shape (mirror PR #38, applied to rebuild paths)

The canonical discriminator pattern, established across `search.py:4418-4435` (post-PR-#38), `spectral_predict_gui_optimized.py:_run_refined_model_thread` (c395317), `unified_bayesian.objective` (PR #38), `nsga2_search._evaluate` (PR #38), `nsga2_search._compute_classification_cv_metrics` (PR #38), `nsga2_search._compute_calibration_metrics` (PR #38):

```python
use_sample_weight_for_classification = False
if imbalance_method == 'class_weight' and task_type == 'classification':
    if model_name == 'PLS-DA':
        # Set class_weight='balanced' on the LR constructor when building
        # the Pipeline. Bake it into _lr_kwargs at the construction site.
        pass
    elif model_name == 'CatBoost':
        try:
            model.set_params(auto_class_weights='Balanced')
        except Exception:
            pass
    elif hasattr(model, 'class_weight'):
        try:
            model.set_params(class_weight='balanced')
        except Exception:
            pass
    else:
        import inspect
        sig = inspect.signature(model.fit) if hasattr(model, 'fit') else None
        if sig and 'sample_weight' in sig.parameters:
            use_sample_weight_for_classification = True

# At fit site:
if use_sample_weight_for_classification:
    from sklearn.utils.class_weight import compute_sample_weight
    model.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))
else:
    model.fit(X_train, y_train)
```

For Pipeline-wrapped models, sample_weight goes via `model__sample_weight=` kwarg.

### Specific changes for `compute_validation_metrics_for_top_models`

1. Add `imbalance_method: Optional[str] = None` parameter to the signature.
2. After `_rebuild_model_from_row` returns the model, apply the discriminator before `model.fit(X_train_final, y_train)` at line 747.
3. Update all callers in `run_search` and `run_bayesian_search` to pass `imbalance_method`.
4. For PLS-DA specifically: `_rebuild_model_from_row` at line ~388 needs to also accept `imbalance_method` and bake `class_weight='balanced'` into the LR constructor. Alternative: serialize the LR's class_weight setting into the result-row `Params` dict at `search.py:4488` so it round-trips through reconstruction. Either approach works; the latter is cleaner architecturally.

### Specific changes for `_reconstruct_models_from_results` (gui)

1. Read `imbalance_method` from the saved-config or active GUI state before reconstruction.
2. Apply the discriminator at each of the 4 fit sites (lines 24086, 24106, 24112, 24120).
3. Test that ensemble training under class_weight produces correctly-weighted constituent models.

---

## Test plan

Mirror PR #38's parametrized end-to-end test pattern for these new sites:

1. **Validation-rebuild test** (`tests/test_class_weight_validation_rebuild.py`):
   - Parametrized over CatBoost + XGBoost (use `_HAS_CATBOOST` / `_HAS_XGBOOST` from `importlib.util.find_spec` — NOT `pytest.importorskip` inside parametrize marks; that over-skips per GLM's PR #38 finding).
   - Run grid-search OR Bayesian search twice on imbalanced data: once with `imbalance_method='class_weight'`, once with `imbalance_method=None`. Pass `validation_count=10` (or similar) to exercise the validation-rebuild path.
   - Assert that the `val_*` columns differ between the two runs. Pre-fix the assertion FAILS because both runs produce identical (unweighted) validation metrics.
2. **Ensemble-rebuild test** — defer if "Train Ensemble" doesn't have an existing test scaffold; structural source-level inspection like the NSGA-II tests in `test_class_weight_sister_sites.py::TestNSGA2DisplayMetricsDiscriminatorStructural` is acceptable for the GUI ensemble path since end-to-end GUI tests are out-of-scope.

---

## Recommended pickup order

1. Read `docs/PROJECT_STATUS.md` top section — captures all 7 morning + afternoon PRs and the deferred items.
2. Read `docs/SESSION_LOG.md` top entry (2026-05-05 afternoon) — captures the bug-class lessons and the `set_fit_request` clone pattern.
3. Read PR #38's commits (`git log fix/Tclass-weight-discriminator-sister-sites --oneline`) — the canonical fix shape lives there.
4. Branch `fix/Tclass-weight-validation-rebuild` from `origin/main`. Note: if PR #38 has been merged by the time you start, branch from the new `origin/main`. If not, branching from `origin/main` is still correct — your fix doesn't depend on PR #38's optimization-loop changes; it's a parallel fix on rebuild paths.
5. Implement validation-rebuild fix first (MEDIUM, higher leverage).
6. Test green.
7. Implement ensemble-rebuild fix (LOW, smaller surface).
8. Commit each independently.
9. Push and open PR. Dispatch DeepSeek + GLM + Codex (canonical cross-family + Codex for codegen/dispatcher work).

Estimated effort: **2-3 hours implementation + tests + 1 round of cross-family review + any fix-of-fixes**. Roughly 100-200 LOC code + 80-120 LOC tests.

---

## Smaller deferred items from PR #33 + PR #32 (fold in or separate)

These are smaller LOWs/MEDIUMs from earlier 2026-05-05 PRs. The reviewers flagged them but they were deferred. Pick up if time permits.

### PR #33 deferred (medium leverage, ~30 LOC)

- **pr-test-analyzer HIGH #1**: behavioral test for `n_jobs` survival through codegen. Add `assert "'n_jobs': 1" in script` (substring check on generated script content) for both XGBoost and RandomForest. Tests structural pin tests do NOT subsume because a refactor that moves the bug to a different mechanism would pass the structural pins but fail the behavioral test.
- **pr-test-analyzer HIGH #2**: architectural module-walk test that asserts no module-level `*PIPELINE_PARAMS*` set in the codebase contains `n_jobs`. Prevents a future third-PIPELINE_PARAMS sister site from being silently born.

### PR #33 LOW (1-line fix)

- `code_generator.py:1023-1041` KNOWN ISSUE comment incorrectly claims LightGBM gets a `verbose=0` re-injection. Only CatBoost does (lines 974-975). Reword the comment: `CatBoost gets a verbose=0 re-injection at lines 974-975 that masks the issue there; RandomForest / SVM / LightGBM users would lose their setting.`

### PR #32 MEDIUM (~80 LOC)

- pr-test-analyzer flagged that PR #32 added the XGBoost auto-with-correction parity row but the LightGBM and CatBoost rows are still missing the symmetric coverage. Add `test_lightgbm_binary_classification_parity_auto_with_correction` and `test_catboost_binary_classification_parity_auto_with_correction` mirroring the XGBoost row pattern.

---

## Process rules (mandatory)

- **Use `.venv312/Scripts/python.exe`.** Project is Python 3.12 only.
- **Targeted pytest, not full suite.** Per-fix: matching test file + the consolidated 13-file regression sweep:
  ```
  test_t41_bayesian_sqlite_auto_calculator    test_t42_write_path_plumbing
  test_t43_resume_auto_restore                test_run_state
  test_run_logging                            test_cv_strategy
  test_unified_bayesian_baseline              test_autoscale_bayesian
  test_cv_pls_clamp                           test_ensemble_preprocessing
  test_ensemble_integration                   test_t32_sample_weight_resampling
  test_scoring                                test_class_weight_sister_sites (new)
  ```
- **Pre-existing flakes (deselect):** `tests/test_export_code.py::test_python_script_execution` and `test_full_workflow_python` — known environmental.
- **Commit ASAP after implementation.** Long pytest runs / agent dispatch / `git push` can collide with parallel sessions. See memory `feedback_commit_before_yielding_control.md`.
- **Don't use `git add -A` for code commits.** Stage explicitly by file. The afternoon-2026-05-05 session had a `.review-tmp/` accident that polluted PR #38 with 39k+ LOC of scratch artifacts. The `.gitignore` now covers `.review-tmp/` but explicit staging is the belt-and-suspenders.
- **Don't use `pytest | tail`.** Use `2>&1 | grep -E "passed|failed"` for length-limited output.
- **Comments default to none.** Only when WHY is non-obvious. NEVER include review-trail prefixes (`# T-NN fix:`, `# DeepSeek HIGH:`) in production code.
- **Black line-length 100, type hints with `from __future__ import annotations`, snake_case.**
- **Don't add features beyond what the ticket requires.** Bug fixes / hygiene fine; methodology changes need user confirmation.

---

## Deliverables when you stop

1. List of branches pushed with tip commit hashes.
2. Per-fix: tests passing count, review verdicts, deviations from plan.
3. Updated `PROJECT_STATUS.md` reflecting what shipped + what's queued.
4. Updated `SESSION_LOG.md` with any architectural traps surfaced.
5. If stopped on a BLOCKER, file `docs/CONTINUATION_PROMPT_<date>_<topic>.md` for the next agent.
