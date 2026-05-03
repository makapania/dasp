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

commit 4dcedbca35c5a834df14074ceb9f4bbd191795e7
Author: Makapania <matt.sponheimer@gmail.com>
Date:   Sun May 3 10:17:29 2026 -0600

    fix: refined-model thread crashed on imbalance_method='class_weight' / 'auto'
    
    The Model Development "refined model" path constructed
    ClassificationResampler('class_weight') and inserted it as a Pipeline step.
    'class_weight' is a model-parameter sentinel (per _needs_resampling_pipeline
    and validate_classification_config), not a resampler — so when CV called
    step.fit_resample() the resampler exploded with "Unknown resampling method:
    class_weight". This regressed all classifier model types in Model Dev once
    T-19 (commit 1d2bf6d) made auto-mode resolution route 'class_weight' values
    into the same path more frequently.
    
    Two-layer fix mirroring search.py:4411-4470:
    
    1. spectral_predict_gui_optimized.py (_run_refined_model_thread): after
       reading loaded_imbalance_method, resolve 'auto' via resolve_auto_imbalance,
       then for 'class_weight' apply class_weight='balanced' via model.set_params
       (with hasattr guard + MLP/no-support warnings) and clear the variable so
       no resampler step is appended downstream. PLS-DA routes the kwarg to its
       LogisticRegression tail via a shared _lr_kwargs dict consumed at all three
       pipeline-build sites (GA / derivative+subset / raw). Without this the no-op
       in (2) would silently drop class_weight handling.
    
    2. src/spectral_predict/imbalance.py (ClassificationResampler.fit /
       fit_resample): defense-in-depth no-op for 'class_weight' / 'auto' sentinels
       so any future caller that misroutes them doesn't crash mid-CV.
    
    Regression test: tests/test_imbalance.py adds parametrized assertions that
    both sentinels (case-insensitive) return input unchanged from fit_resample.
    
    Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

diff --git a/spectral_predict_gui_optimized.py b/spectral_predict_gui_optimized.py
index 91af575..715c3bf 100644
--- a/spectral_predict_gui_optimized.py
+++ b/spectral_predict_gui_optimized.py
@@ -37390,6 +37390,63 @@ F1 Score:  {f1:.4f}
                     print(f"  imbalance_method: {loaded_imbalance_method}")
                     print(f"  imbalance_params: {loaded_imbalance_params}")
 
+            # Resolve 'auto' (T-19) to a concrete method based on y, mirroring
+            # search.py:1068-1074. Then apply 'class_weight' as a model kwarg —
+            # NOT as a pipeline step — mirroring search.py:4411-4448. This keeps
+            # the refined-model path semantically aligned with the search path
+            # and prevents ClassificationResampler from receiving non-resampler
+            # sentinels (which would crash at fit_resample).
+            if loaded_imbalance_method == 'auto' and task_type == 'classification':
+                from spectral_predict.imbalance import (
+                    resolve_auto_imbalance,
+                    format_auto_imbalance_message,
+                )
+                resolved_method, resolved_info = resolve_auto_imbalance(
+                    y_array, task_type=task_type
+                )
+                print(format_auto_imbalance_message(resolved_info))
+                loaded_imbalance_method = resolved_method  # 'class_weight' or None
+
+            # Routed to the LogisticRegression tail for PLS-DA (mirrors search.py:4469-4470),
+            # consumed at the three LR-construction sites below.
+            apply_class_weight_to_lr = False
+            if loaded_imbalance_method == 'class_weight' and task_type == 'classification':
+                import warnings as _warnings
+                if model_name == 'PLS-DA':
+                    # The 'classifier' is the LogisticRegression tail, not the PLS transformer.
+                    apply_class_weight_to_lr = True
+                    print(f"DEBUG: Will apply class_weight='balanced' to PLS-DA LogisticRegression tail")
+                elif hasattr(model, 'class_weight'):
+                    try:
+                        model.set_params(class_weight='balanced')
+                        print(f"DEBUG: Applied class_weight='balanced' to {model_name}")
+                    except Exception as e:
+                        _warnings.warn(
+                            f"{model_name} has class_weight attribute but set_params failed: {e}. "
+                            f"Consider using SMOTE or other resampling method.",
+                            UserWarning,
+                        )
+                else:
+                    if model_name in ('MLP', 'MLPClassifier'):
+                        _warnings.warn(
+                            f"{model_name} does not support class_weight. For imbalanced "
+                            f"classification with MLP, use SMOTE or other resampling methods instead.",
+                            UserWarning,
+                        )
+                    else:
+                        _warnings.warn(
+                            f"{model_name} does not support class_weight. Consider using "
+                            f"SMOTE or other resampling methods for imbalanced data.",
+                            UserWarning,
+                        )
+                # Strip from imbalance_method so no resampler step is appended downstream.
+                loaded_imbalance_method = None
+
+            # Shared LR kwargs for the three PLS-DA tail-construction sites below.
+            _lr_kwargs = {'max_iter': 1000, 'random_state': 42}
+            if apply_class_weight_to_lr:
+                _lr_kwargs['class_weight'] = 'balanced'
+
             # Initialize variables that may only be set in certain paths
             # (needed later for validation prediction)
             prep_pipeline = None
@@ -37514,7 +37571,7 @@ F1 Score:  {f1:.4f}
                     pipe_steps.extend([
                         ('pls', model),
                         ('scaler', StandardScaler()),  # Scale PLS scores for LogisticRegression
-                        ('lr', LogisticRegression(max_iter=1000, random_state=42))
+                        ('lr', LogisticRegression(**_lr_kwargs))
                     ])
                 # For scale-sensitive models (SVC/SVR, MLP, NeuralBoosted), add StandardScaler
                 # These use gradient descent or kernel methods that are sensitive to feature scale
@@ -37642,7 +37699,7 @@ F1 Score:  {f1:.4f}
                     pipe_steps.extend([
                         ('pls', model),
                         ('scaler', StandardScaler()),  # Scale PLS scores for LogisticRegression
-                        ('lr', LogisticRegression(max_iter=1000, random_state=42))
+                        ('lr', LogisticRegression(**_lr_kwargs))
                     ])
                 # For scale-sensitive models (SVC/SVR, MLP, NeuralBoosted), add StandardScaler
                 # These use gradient descent or kernel methods that are sensitive to feature scale
@@ -37698,7 +37755,7 @@ F1 Score:  {f1:.4f}
                     from sklearn.preprocessing import StandardScaler
                     pipe_steps.append(('pls', model))
                     pipe_steps.append(('scaler', StandardScaler()))  # Scale PLS scores for LogisticRegression
-                    pipe_steps.append(('lr', LogisticRegression(max_iter=1000, random_state=42)))
+                    pipe_steps.append(('lr', LogisticRegression(**_lr_kwargs)))
                 # For scale-sensitive models (SVC/SVR, MLP, NeuralBoosted), add StandardScaler
                 # These use gradient descent or kernel methods that are sensitive to feature scale
                 elif model_name in ('SVC', 'SVR', 'MLP', 'NeuralBoosted', 'Ridge', 'Lasso', 'ElasticNet'):
diff --git a/src/spectral_predict/imbalance.py b/src/spectral_predict/imbalance.py
index bf5e5e1..dfc0fae 100644
--- a/src/spectral_predict/imbalance.py
+++ b/src/spectral_predict/imbalance.py
@@ -236,6 +236,14 @@ class ClassificationResampler(BaseEstimator):
 
     def fit(self, X, y=None):
         """Fit the resampler (creates internal resampler object)."""
+        # 'class_weight' and 'auto' are model-parameter sentinels, not resampler
+        # methods — see _needs_resampling_pipeline() / validate_classification_config().
+        # If a caller still routes them through this factory, no-op so the pipeline
+        # doesn't crash; the model receives class_weight='balanced' at construction
+        # time in the actual call site.
+        if isinstance(self.method, str) and self.method.lower() in ('class_weight', 'auto'):
+            self.resampler_ = None
+            return self
         # Create resampler based on method name
         if isinstance(self.method, str):
             method_map = {
@@ -280,6 +288,10 @@ class ClassificationResampler(BaseEstimator):
         """
         self.fit(X, y)
 
+        # No-op for 'class_weight' / 'auto' sentinels — see fit() for rationale.
+        if self.resampler_ is None:
+            return X, y
+
         original_size = len(y)
         original_class_counts = Counter(y)
 
diff --git a/tests/test_imbalance.py b/tests/test_imbalance.py
index 3f08903..e4e0113 100644
--- a/tests/test_imbalance.py
+++ b/tests/test_imbalance.py
@@ -351,6 +351,19 @@ class TestClassificationResampler:
         X_res, y_res = resampler.fit_resample(X, y)
         assert len(X_res) > 0
 
+    @pytest.mark.parametrize("sentinel", ["class_weight", "auto", "CLASS_WEIGHT", "Auto"])
+    def test_class_weight_and_auto_are_no_ops(self, sentinel, imbalanced_data):
+        """Regression: GUI's refined-model path may route 'class_weight' or 'auto'
+        sentinels through the resampler factory (these are model-parameter flags,
+        not actual resamplers). Pre-fix this raised ValueError mid-CV. Post-fix
+        the resampler returns input unchanged so the pipeline can proceed; the
+        actual class_weight='balanced' kwarg is applied to the model elsewhere."""
+        X, y = imbalanced_data
+        resampler = ClassificationResampler(method=sentinel, random_state=42)
+        X_res, y_res = resampler.fit_resample(X, y)
+        assert X_res is X
+        assert y_res is y
+
 
 # =============================================================================
 # Regression Resampler Tests
commit c395317a60ec3d5b2fb5016fbfc1388aeda10517
Author: Makapania <matt.sponheimer@gmail.com>
Date:   Sun May 3 10:43:57 2026 -0600

    fix(round 2): close HIGH from cross-family review of 4dcedbc
    
    Both DeepSeek V4 Pro Max and Codex independently flagged the same HIGH:
    the round-1 fix at 4dcedbc only routed class_weight via the hasattr check,
    which silently returns False for CatBoostClassifier (uses auto_class_weights)
    and XGBClassifier (no class_weight kwarg, only sample_weight at fit time).
    Result: every CatBoost or XGBoost classifier loaded into Model Dev → Refine
    with imbalance_method='class_weight' (or 'auto' resolved to class_weight)
    was training UNWEIGHTED — silent violation of the "Refine reproduces
    Results" contract. No crash, just wrong numbers — the worst kind of bug for
    a chemometrics tool.
    
    This fix-of-fixes mirrors search.py:4411-4448 fully:
    
    1. spectral_predict_gui_optimized.py — discriminator now branches on
       model name:
         PLS-DA → class_weight='balanced' on the LR tail (already done)
         CatBoost → auto_class_weights='Balanced' (parity with code_generator.py:946)
         hasattr(model, 'class_weight') → set_params(class_weight='balanced')
         sample_weight in fit() signature → use_sample_weight_for_classification flag
         MLP / no-support → loud directive log message naming SMOTE alternatives
       The flag drives compute_sample_weight('balanced', y) at three fit sites:
       non-early-stopping fold fit (via model__sample_weight fit_param), early-
       stopping fold fit (via _fit_with_early_stopping's new sample_weight kwarg),
       and the final-pipeline fit on full data.
    
       Also: warnings.warn → self._log_progress so messages reach users in the
       bundled .exe (warnings.warn goes to nul stderr in PyInstaller). The MLP
       branch is now explicitly directive about which methods to switch to.
    
    2. src/spectral_predict/cv_utils.py — _fit_with_early_stopping accepts
       optional sample_weight kwarg, threads to the booster fit.
    
    3. src/spectral_predict/search.py — _needs_resampling_pipeline early-return
       now explicitly guards 'auto' (was correct-by-accident; defense-in-depth).
    
    4. src/spectral_predict/imbalance.py — build_imbalance_transformer routes
       'class_weight'/'auto' sentinels through ClassificationResampler regardless
       of task_type so the no-op short-circuits. Pre-fix, regression callers
       leaking these would hit the regression else-clause and raise.
    
    5. tests/test_imbalance.py — added parametrized regression-task no-op
       coverage for both sentinels.
    
    Verified correct (called out in DeepSeek's review):
      - PLS-DA _lr_kwargs scope at all 3 sites (no UnboundLocalError)
      - y_array shape contract (1D from y_series.values at line 37100)
      - One-class unreachable from this code path
      - Real resamplers (SMOTE, ADASYN, etc.) unaffected
      - RidgeClassifier still not in GUI (would now be handled if added)
    
    Test sweep: tests/test_imbalance.py 62 passed + 1 skipped (was 58 + 1; +4
    from regression-sentinel parametrize). py_compile clean across all 4 src
    files + GUI.
    
    Deferred per engineering-polish rule (LOW from DeepSeek):
      - Integration test exercising _run_refined_model_thread end-to-end
      - import warnings hoisted to top-level (functionally fine; cosmetic)
    
    Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

diff --git a/spectral_predict_gui_optimized.py b/spectral_predict_gui_optimized.py
index 715c3bf..797508c 100644
--- a/spectral_predict_gui_optimized.py
+++ b/spectral_predict_gui_optimized.py
@@ -37407,37 +37407,67 @@ F1 Score:  {f1:.4f}
                 print(format_auto_imbalance_message(resolved_info))
                 loaded_imbalance_method = resolved_method  # 'class_weight' or None
 
-            # Routed to the LogisticRegression tail for PLS-DA (mirrors search.py:4469-4470),
-            # consumed at the three LR-construction sites below.
-            apply_class_weight_to_lr = False
+            # Apply class_weight to the model directly (NOT as a resampler step) —
+            # mirrors search.py:4411-4448. Different model families use different
+            # kwargs for class weighting:
+            #   - LogisticRegression (PLS-DA tail), RF, SVC, LightGBM, NeuralBoosted: class_weight='balanced'
+            #   - CatBoost: auto_class_weights='Balanced'  (no class_weight kwarg; mirrors code_generator.py:946)
+            #   - XGBoost, RidgeClassifier: per-fold sample_weight=compute_sample_weight('balanced', y) at fit time
+            #   - MLP / NeuralBoosted-MLP / others with neither: cannot apply — must use a resampler instead
+            apply_class_weight_to_lr = False              # PLS-DA → LR tail
+            use_sample_weight_for_classification = False  # XGBoost / sample_weight-only fallback
             if loaded_imbalance_method == 'class_weight' and task_type == 'classification':
-                import warnings as _warnings
                 if model_name == 'PLS-DA':
-                    # The 'classifier' is the LogisticRegression tail, not the PLS transformer.
                     apply_class_weight_to_lr = True
-                    print(f"DEBUG: Will apply class_weight='balanced' to PLS-DA LogisticRegression tail")
+                    self._log_progress(
+                        f"Imbalance: applying class_weight='balanced' to PLS-DA LogisticRegression tail"
+                    )
+                elif model_name == 'CatBoost':
+                    # CatBoost classifier exposes auto_class_weights, not class_weight.
+                    try:
+                        model.set_params(auto_class_weights='Balanced')
+                        self._log_progress(
+                            f"Imbalance: applied auto_class_weights='Balanced' to CatBoost"
+                        )
+                    except Exception as e:
+                        self._log_progress(
+                            f"Imbalance WARNING: CatBoost set_params(auto_class_weights='Balanced') failed: {e}. "
+                            f"Model will train UNWEIGHTED. Consider switching to SMOTE/ADASYN."
+                        )
                 elif hasattr(model, 'class_weight'):
                     try:
                         model.set_params(class_weight='balanced')
-                        print(f"DEBUG: Applied class_weight='balanced' to {model_name}")
+                        self._log_progress(
+                            f"Imbalance: applied class_weight='balanced' to {model_name}"
+                        )
                     except Exception as e:
-                        _warnings.warn(
-                            f"{model_name} has class_weight attribute but set_params failed: {e}. "
-                            f"Consider using SMOTE or other resampling method.",
-                            UserWarning,
+                        self._log_progress(
+                            f"Imbalance WARNING: {model_name} has class_weight attribute but set_params failed: {e}. "
+                            f"Model will train UNWEIGHTED. Consider switching to SMOTE/ADASYN."
                         )
                 else:
-                    if model_name in ('MLP', 'MLPClassifier'):
-                        _warnings.warn(
-                            f"{model_name} does not support class_weight. For imbalanced "
-                            f"classification with MLP, use SMOTE or other resampling methods instead.",
-                            UserWarning,
+                    # Sample_weight fallback for XGBoost / RidgeClassifier-style models
+                    # (mirrors search.py:4427-4432). Detection via fit() signature.
+                    import inspect as _inspect
+                    _model_fit_sig = _inspect.signature(model.fit) if hasattr(model, 'fit') else None
+                    if _model_fit_sig and 'sample_weight' in _model_fit_sig.parameters:
+                        use_sample_weight_for_classification = True
+                        self._log_progress(
+                            f"Imbalance: {model_name} supports sample_weight; will compute "
+                            f"compute_sample_weight('balanced', y_train) per fold and pass to fit()"
+                        )
+                    elif model_name in ('MLP', 'MLPClassifier'):
+                        self._log_progress(
+                            f"Imbalance WARNING: {model_name} does NOT support class_weight or sample_weight. "
+                            f"Class-weighting was requested but cannot be applied — model will train UNWEIGHTED. "
+                            f"For imbalanced classification with MLP, switch the imbalance method to "
+                            f"SMOTE / ADASYN / SMOTE-Tomek, or pick a model that supports class_weight "
+                            f"(RandomForest, SVC, LightGBM, CatBoost, XGBoost, NeuralBoosted, PLS-DA)."
                         )
                     else:
-                        _warnings.warn(
-                            f"{model_name} does not support class_weight. Consider using "
-                            f"SMOTE or other resampling methods for imbalanced data.",
-                            UserWarning,
+                        self._log_progress(
+                            f"Imbalance WARNING: {model_name} does not support class_weight or sample_weight. "
+                            f"Model will train UNWEIGHTED. Consider switching to SMOTE / ADASYN / SMOTE-Tomek."
                         )
                 # Strip from imbalance_method so no resampler step is appended downstream.
                 loaded_imbalance_method = None
@@ -37926,11 +37956,22 @@ F1 Score:  {f1:.4f}
                             else:
                                 y_test_es = y_test
 
+                            # Per-fold balanced sample weights for sample_weight-only models
+                            # (XGBoost class_weight path — mirrors search.py:4068, 4088).
+                            # Computed AFTER any in-fold resampler so weights match resampled y.
+                            _es_sample_weight = None
+                            if use_sample_weight_for_classification:
+                                from sklearn.utils.class_weight import compute_sample_weight
+                                _es_sample_weight = compute_sample_weight(
+                                    'balanced', y_train_fold
+                                )
+
                             _fit_with_early_stopping(
                                 final_model_fold,
                                 X_train_transformed, y_train_fold,
                                 X_test_transformed, y_test_es,
-                                early_stopping_rounds
+                                early_stopping_rounds,
+                                sample_weight=_es_sample_weight,
                             )
                             y_pred = final_model_fold.predict(X_test_transformed)
 
@@ -37954,7 +37995,16 @@ F1 Score:  {f1:.4f}
                                 y_proba = pipe_fold.predict_proba(X_test)
                                 all_y_proba.append(y_proba)
                     else:
-                        pipe_fold.fit(X_train, y_train)
+                        # Per-fold balanced sample weights threaded via the 'model'
+                        # step name (sklearn fit_params convention) for sample_weight-only
+                        # classifiers like XGBoost — mirrors search.py:4068, 4098.
+                        _fit_kwargs = {}
+                        if use_sample_weight_for_classification:
+                            from sklearn.utils.class_weight import compute_sample_weight
+                            _fit_kwargs['model__sample_weight'] = compute_sample_weight(
+                                'balanced', y_train
+                            )
+                        pipe_fold.fit(X_train, y_train, **_fit_kwargs)
                         y_pred = pipe_fold.predict(X_test)
                 except ValueError as e:
                     if y_transform_active and any(kw in str(e).lower() for kw in ('log', 'positive', 'negative', 'sqrt', 'domain')):
@@ -38231,7 +38281,18 @@ Configuration:
             # Fit final pipeline on full dataset for model persistence
             # Clone the pipeline and fit on all data
             final_pipe = clone(pipe)
-            final_pipe.fit(X_raw, y_array)
+            # Mirror the per-fold sample_weight wiring: full-data balanced weights
+            # for sample_weight-only classifiers (XGBoost). PLS-DA / CatBoost / RF /
+            # SVC / LightGBM / NeuralBoosted carry their class_weight or
+            # auto_class_weights kwarg on the model itself, so they don't need
+            # anything extra here.
+            _final_fit_kwargs = {}
+            if use_sample_weight_for_classification:
+                from sklearn.utils.class_weight import compute_sample_weight
+                _final_fit_kwargs['model__sample_weight'] = compute_sample_weight(
+                    'balanced', y_array
+                )
+            final_pipe.fit(X_raw, y_array, **_final_fit_kwargs)
 
             # --- Calibration metrics (predict on training data) ---
             cal_text = ""
diff --git a/src/spectral_predict/cv_utils.py b/src/spectral_predict/cv_utils.py
index 355191b..cdeae0a 100644
--- a/src/spectral_predict/cv_utils.py
+++ b/src/spectral_predict/cv_utils.py
@@ -563,7 +563,8 @@ def _fit_with_early_stopping(
     y_train: np.ndarray,
     X_val: np.ndarray,
     y_val: np.ndarray,
-    early_stopping_rounds: int = 40
+    early_stopping_rounds: int = 40,
+    sample_weight: np.ndarray | None = None,
 ) -> None:
     """Fit a boosting model with early stopping.
 
@@ -581,14 +582,24 @@ def _fit_with_early_stopping(
         Validation targets for early stopping
     early_stopping_rounds : int, default=50
         Number of rounds without improvement before stopping
+    sample_weight : ndarray or None, default=None
+        Optional per-sample weights threaded into the booster's ``fit``.
+        XGBoost has no native ``class_weight`` kwarg; per-sample balanced
+        weights via ``compute_sample_weight('balanced', y)`` is the canonical
+        path for imbalanced XGBoost classification (mirrors search.py:4427).
     """
+    fit_kwargs: dict = {}
+    if sample_weight is not None:
+        fit_kwargs['sample_weight'] = sample_weight
+
     if isinstance(model, XGBOOST_MODELS):
         # Set early_stopping_rounds before fitting
         model.set_params(early_stopping_rounds=early_stopping_rounds)
         model.fit(
             X_train, y_train,
             eval_set=[(X_val, y_val)],
-            verbose=False
+            verbose=False,
+            **fit_kwargs,
         )
 
     elif isinstance(model, LIGHTGBM_MODELS):
@@ -598,7 +609,8 @@ def _fit_with_early_stopping(
             callbacks=[
                 lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                 lgb.log_evaluation(period=0)  # Suppress output
-            ]
+            ],
+            **fit_kwargs,
         )
 
     elif CATBOOST_AVAILABLE and isinstance(model, CATBOOST_MODELS):
@@ -611,11 +623,12 @@ def _fit_with_early_stopping(
             X_train, y_train,
             eval_set=(X_val, y_val),  # CatBoost uses tuple, not list
             early_stopping_rounds=early_stopping_rounds,
-            verbose=0  # CatBoost prefers int over bool
+            verbose=0,  # CatBoost prefers int over bool
+            **fit_kwargs,
         )
     else:
         # Not a boosting model - standard fit
-        model.fit(X_train, y_train)
+        model.fit(X_train, y_train, **fit_kwargs)
 
 
 def _compute_score(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
diff --git a/src/spectral_predict/imbalance.py b/src/spectral_predict/imbalance.py
index dfc0fae..62f6ff4 100644
--- a/src/spectral_predict/imbalance.py
+++ b/src/spectral_predict/imbalance.py
@@ -944,6 +944,17 @@ def build_imbalance_transformer(method, task_type='classification', random_state
     ...     'binning', task_type='regression', random_state=42, n_bins=5
     ... )
     """
+    # Sentinel guard: 'class_weight' / 'auto' are model-parameter flags, not
+    # resamplers. Route through ClassificationResampler regardless of task_type
+    # so its built-in no-op short-circuit kicks in. Pre-fix, regression callers
+    # passing these sentinels (corrupted metadata or future feature) would hit
+    # the regression else-clause and raise ValueError. (Reachability today: low,
+    # since resolve_auto_imbalance returns None for non-classification and
+    # class_weight isn't a regression dropdown option — but defense-in-depth
+    # symmetry with the classification path costs nothing.)
+    if isinstance(method, str) and method.lower() in ('class_weight', 'auto'):
+        return ClassificationResampler(method=method, random_state=random_state, **params)
+
     if task_type == 'classification':
         return ClassificationResampler(method=method, random_state=random_state, **params)
 
diff --git a/src/spectral_predict/search.py b/src/spectral_predict/search.py
index a0e24db..3e63ff8 100644
--- a/src/spectral_predict/search.py
+++ b/src/spectral_predict/search.py
@@ -282,8 +282,11 @@ def _needs_resampling_pipeline(imbalance_method, task_type):
     if imbalance_method is None:
         return False
 
-    # class_weight doesn't need resampling pipeline (it's a model parameter)
-    if imbalance_method == 'class_weight':
+    # class_weight / auto don't need resampling pipeline — they're model-parameter
+    # sentinels (auto is resolved into class_weight or None at run-entry; both
+    # short-circuit here as defense-in-depth so a future refactor that delays
+    # resolution can't accidentally wrap them in ImbPipeline).
+    if imbalance_method in ('class_weight', 'auto'):
         return False
 
     # Classification resampling methods need imblearn Pipeline
diff --git a/tests/test_imbalance.py b/tests/test_imbalance.py
index e4e0113..e68edaf 100644
--- a/tests/test_imbalance.py
+++ b/tests/test_imbalance.py
@@ -753,6 +753,28 @@ class TestBuildImbalanceTransformer:
         with pytest.raises(ValueError, match="Unknown"):
             transformer.fit_resample(X, y)
 
+    @pytest.mark.parametrize("sentinel", ["class_weight", "auto"])
+    @pytest.mark.parametrize("task_type", ["classification", "regression"])
+    def test_class_weight_auto_sentinels_no_op_both_task_types(self, sentinel, task_type):
+        """Regression: 'class_weight' / 'auto' are model-parameter sentinels and
+        must no-op regardless of task_type. Pre-fix-of-fixes the regression path
+        raised ValueError on these (sentinel guard was classification-only).
+        Defense-in-depth: both branches now route through ClassificationResampler
+        which short-circuits at fit_resample."""
+        import numpy as np
+        import pandas as pd
+
+        transformer = build_imbalance_transformer(
+            sentinel, task_type=task_type, random_state=42
+        )
+        # Build doesn't raise; fit_resample no-ops and returns input unchanged.
+        X = pd.DataFrame(np.random.randn(20, 5)).values
+        y = pd.Series(np.arange(20) % 2 if task_type == "classification"
+                      else np.linspace(0.0, 1.0, 20)).values
+        X_res, y_res = transformer.fit_resample(X, y)
+        assert X_res is X
+        assert y_res is y
+
 
 # =============================================================================
 # Recommendation Tests
