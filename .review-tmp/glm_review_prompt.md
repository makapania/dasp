# Review request: fix-of-fixes commit c395317 — chemometrics class_weight dispatcher

You are reviewing a fix-of-fixes commit on a Python chemometrics tool (Spectral Predict, sklearn-based, Tkinter GUI). Your verdict matters: the user explicitly said "have your fixes checked before pushing" and you are the gatekeeper. Be skeptical. The user's chemometrics tool runs in a chemistry lab and silently-wrong numbers are worse than a crash. Convergent HIGH from two prior reviewers caught the previous round's silent failure — your job is to catch the next one.

## Background

### Original bug

User reported: running ANY classifier from Model Development → Refine raised:
```
ValueError: Unknown resampling method: class_weight. Available: ['smote', 'adasyn', ...]
File 'spectral_predict_gui_optimized.py', line 37849, in _run_refined_model_thread
  X_train_transformed, y_train_fold = step.fit_resample(X_train_transformed, y_train_fold)
File 'src/spectral_predict/imbalance.py', line 281, in fit_resample
  self.fit(X, y)
```

Root cause: `class_weight` is a model-parameter sentinel (per `_needs_resampling_pipeline` at search.py:286 and `validate_classification_config` at imbalance.py:1259), not an actual resampler method. The refined-model thread was constructing `ClassificationResampler('class_weight')` and inserting it as a Pipeline step. Canonical pattern that should have been mirrored: search.py:4411-4470, which applies `model.set_params(class_weight='balanced')` directly and then never passes the sentinel to the resampler factory.

Why visible recently: T-19 (commit 1d2bf6d) added runtime resolution where `imbalance_method='auto'` mutates to `'class_weight'` or None at run-entry inside the search modules. Saved-then-reloaded model configs whose original search resolved auto→class_weight feed the sentinel into the broken refined-model path.

### Round-1 fix (commit 4dcedbc)

Two-layer fix:

1. **imbalance.py**: ClassificationResampler.fit / fit_resample no-op on 'class_weight' / 'auto' (case-insensitive). Defense in depth.
2. **_run_refined_model_thread in the GUI**: resolve 'auto' via `resolve_auto_imbalance`, apply `class_weight='balanced'` to the model with `hasattr` guard + sample_weight fallback warning + MLP warning, route PLS-DA's LR tail via shared `_lr_kwargs` dict consumed at three pipeline-build sites (GA / derivative+subset / raw), clear `loaded_imbalance_method=None` so no resampler step is appended downstream.

### Convergent HIGH from DeepSeek V4 Pro Max + Codex (independent reviewers)

Both flagged the same bug in 4dcedbc:

- **CatBoostClassifier** has no `class_weight` attribute — uses `class_weights` (plural) and `auto_class_weights`. `hasattr(model, 'class_weight')` returns False. The 4dcedbc fix fell into the warning branch and silently cleared `loaded_imbalance_method`, so CatBoost trained UNWEIGHTED.
- **XGBClassifier** has no `class_weight` kwarg — only `sample_weight` at fit time. Same hasattr-False path, same silent-unweighted result.

Failure mode: "Refined metrics mismatch Results tab; class weighting silently dropped" — violates the foundational "Refine reproduces Results" contract for chemometrics. NO crash, just wrong numbers.

## Fix-of-fixes (commit c395317) — what you are reviewing

5 files changed, 140 insertions, 30 deletions:

- **spectral_predict_gui_optimized.py**: discriminator now branches on model name (PLS-DA / CatBoost / hasattr / sample_weight fallback / MLP / no-support); MLP message via `self._log_progress` is now directive ("switch to SMOTE / ADASYN / SMOTE-Tomek, or pick RandomForest/SVC/LightGBM/CatBoost/XGBoost/NeuralBoosted/PLS-DA"); `use_sample_weight_for_classification` flag drives `compute_sample_weight('balanced', y)` at 3 fit sites:
  1. early-stopping fold via `_fit_with_early_stopping(..., sample_weight=...)` (new kwarg)
  2. non-early-stopping fold via `pipe_fold.fit(X, y, model__sample_weight=...)` (sklearn fit_param routing)
  3. final-pipeline fit on full data via `final_pipe.fit(X, y, model__sample_weight=...)`

- **src/spectral_predict/cv_utils.py**: `_fit_with_early_stopping` accepts optional `sample_weight=None`, threads to all 3 booster fits (XGBoost, LightGBM, CatBoost) and the non-booster fallback.

- **src/spectral_predict/search.py**: `_needs_resampling_pipeline` early-return now explicitly guards 'auto' (was correct-by-accident before; defense-in-depth from DeepSeek MEDIUM).

- **src/spectral_predict/imbalance.py**: `build_imbalance_transformer` routes 'class_weight'/'auto' sentinels through `ClassificationResampler` regardless of task_type so the no-op short-circuits (defense-in-depth from DeepSeek MEDIUM).

- **tests/test_imbalance.py**: parametrized regression-task no-op coverage for both sentinels (4 new test cases; total now 62 passed + 1 skipped).

## What the user wants you to verify

User's exact framing: **"the key is to make sure this works for all model types and analysis types"**. So your review must check the fix-of-fixes is correct across:

- **Model types**: PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, SVM (SVC/SVR), MLP, NeuralBoosted, PLS-DA. One-class out of scope.
- **Task types**: regression, classification.
- **Imbalance method values flowing through this codepath**: None, 'auto', 'class_weight', 'smote', 'adasyn', 'borderline_smote', 'random_undersampler', 'tomek_links', 'smote_tomek', 'smote_enn'.

## Specific questions to answer

**(a) CatBoost auto_class_weights routing**: `model.set_params(auto_class_weights='Balanced')` — does CatBoostClassifier accept this kwarg? Will the set_params call succeed (no TypeError)? Compare to code_generator.py:946 which already uses this pattern in the export codegen.

**(b) XGBoost sample_weight threading**: Does the threading at all 3 fit sites actually reach XGBClassifier.fit() correctly?
- Early-stopping via cv_utils._fit_with_early_stopping with new sample_weight kwarg
- Non-early-stopping via `model__sample_weight` fit_param routing through sklearn Pipeline
- Final-pipeline fit on full data via `model__sample_weight`
Are there any pipeline shapes (e.g., the GA path with subsetting + scaler + model) where the routing breaks?

**(c) MLP directive messaging**: `self._log_progress` is the GUI's run-log helper. Is this message visible to users in the bundled .exe (PyInstaller)? The prior MEDIUM finding was that `warnings.warn` goes to nul stderr — does `_log_progress` have the same problem?

**(d) Defense-in-depth in _needs_resampling_pipeline + build_imbalance_transformer**: Does the new 'auto' guard in `_needs_resampling_pipeline` interact correctly with all callers (search.py / nsga2_search.py / unified_bayesian.py / GUI)? Does the new sentinel guard in `build_imbalance_transformer` break any existing test?

**(e) New gaps**: Did the fix-of-fixes introduce any new bugs?
- `model__sample_weight` fit_param assumes the final pipeline step is named 'model' — verify this for all 3 GUI pipeline-build paths (GA / derivative+subset / raw).
- For PLS-DA the final step is 'lr' not 'model' — but PLS-DA goes via `apply_class_weight_to_lr`, not sample_weight, so this should be fine. Confirm.
- Edge case: what if the user has XGBoost in the GA path with subsetting? Step name still 'model'?

**(f) MLP+SMOTE: validate user can actually do this**: User reads MLP message saying "switch to SMOTE". Is SMOTE actually compatible with MLP in this codebase? (i.e., the dropdown allows it, and the pipeline works.) If not, the directive is misleading.

## Output format

Return:
- **VERDICT**: READY_TO_PUSH or BLOCK_PUSH
- For each blocking issue: file:line, failure mode, suggested fix
- For each non-blocking observation: file:line, what to consider

The diff included below contains BOTH commits (4dcedbc original fix + c395317 fix-of-fixes) so you can see the evolution.

---

# Diffs of both commits (4dcedbc then c395317)

