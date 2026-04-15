# Session Log

Non-obvious discoveries, bug root causes, and failed approaches. Prevents re-discovery across sessions and machines.

---

## 2026-04-14 — PR #4 round-5 review: ship-with-followup (Claude Opus 4.6)

### Outcome

Commit `4248c83` reviewed by codex + peer-review panel (MiniMax M2.7 + DeepSeek V3.2 + MiniMax M2.5 free). **Codex verdict: ship-with-followup, no BLOCKERS, no MAJORs.** Only two MINOR test-coverage suggestions (not ship-blockers):

1. `test_one_class_repeated_kfold_auc_is_mean_of_fold_aucs` only exercises `OneClassSVM(kernel='linear')`. Parametrize across IsolationForest / EllipticEnvelope / LOF / PCA-SIMCA.
2. `test_one_class_plain_kfold_still_works_after_loop_restructure` is a smoke test only. Add a plain-K-Fold AUC parity test that asserts non-repeated AUC follows pooled-score semantics (guards against accidental routing through the repeated-CV override).

### Codex confirmations that rebutted peer-review panel speculation

Peer-review panel (all 3 models) flagged "CRITICAL: same incomparable-scores bug likely affects other pooled metrics (accuracy, precision, recall, F1, specificity)." Codex verified the claim is false by reading `one_class_metrics` at `contamination.py:442`:

- `one_class_metrics` uses `scores` ONLY for AUC (line 506-516). Every other metric (sensitivity/specificity/precision/f1/accuracy/balanced_accuracy) is label-based — computed from `y_true` and `y_pred`. Passing `scores=None` is safe: no crash, `auc = np.nan`, other metrics unaffected.
- No other score-based metric exists in the codebase's one_class path (no `average_precision_score`, no `log_loss`, etc.).
- Downstream consumers (`unified_bayesian.py:1016`, `search.py:5296`) read `mean_metrics['auc']` and trial `auc_cv` attribute — they see the override correctly.
- Calibration path (`contamination.py:775`) fits ONE model on ALL inliers, so scores there are self-consistent — repeated-CV comparability issue does not apply.
- `fold_metrics` is appended only for successful folds → AUC override and pooled outputs consistently exclude fold failures.
- All-NaN fold-AUC edge case correctly handled: filtered list empties → override returns `float('nan')`.
- Tie-break-to-outlier bias exists only in repeated-CV per-sample majority vote; plain K-Fold has no per-sample reduction so no semantic mismatch.

### Lesson: peer-review default mode lacks code access

The peer-review panel ran in default mode (no librarian) and worked from a text summary doc. All three models correctly flagged "I cannot see the actual diff" but still produced speculative CRITICAL findings without that verification. Codex, running with file-read tools on the actual code, resolved all speculation correctly. For high-stakes correctness reviews, either pair peer-review with a tool-enabled reviewer (codex, or peer-review deep mode) or expect the panel's findings to need secondary verification. Default peer-review is best as a cheap breadth sweep, not a ship-gate.

### Non-fixed (explicitly decided)

- Parametrize AUC test across model families — worth doing but not a ship-blocker. Listed as Follow-Up.
- Plain-K-Fold AUC parity test — same.
- Tie-break bias toward outlier — documented in inline comment, conservative default for contamination detection. Callers who need different semantics should use odd `cv_n_repeats`.

### Results

- **71/71 cv-strategy tests pass** (unchanged from round-4).
- **138/138 adjacent suites pass**.
- Commit `4248c83` on `claude/cv-strategy-overhaul`, **PR #4 ready to merge**.

---

## 2026-04-14 — PR #4 round-4 review fix: one-class repeated-CV AUC stays as mean-of-folds (Claude Opus 4.6)

### Bug: Pooled AUC under repeated CV uses decision scores not comparable across folds

**Root cause (caught by codex in round-4 review of commit `5ca0886`):** The round-3 per-sample reduction in `run_one_class_cv` averaged `decision_function` / `score_samples` outputs across folds before passing to `roc_auc_score`. For `OneClassSVM`, `IsolationForest`, `EllipticEnvelope`, `LOF` — scores from independently-fitted models are NOT on a common scale (SVM margins depend on support vectors selected; IF path lengths depend on tree ensemble; EllipticEnvelope Mahalanobis depends on covariance fit). Averaging them produces a meaningless ranking and destroys AUC semantics. Labels-based metrics (Sensitivity/Specificity/Accuracy/BalancedAcc/F1/Precision) pool correctly via majority vote because each label is a self-consistent binary decision per fold; AUC is not.

**Fix:** Under `_is_repeated_cv(kf)`:
1. Drop `pooled_scores` entirely (pass `None` to `one_class_metrics`, which returns NaN for AUC).
2. Override `mean_metrics['auc']` with `np.mean([m['auc'] for m in fold_metrics])` — per-fold AUCs are self-contained and comparable across folds.

Plain K-Fold and LOO paths unchanged (each sample has 1 prediction/score per fold; scores within a fold are self-consistent).

Regression test: `test_one_class_repeated_kfold_auc_is_mean_of_fold_aucs` — replays the exact splits manually, computes per-fold AUCs, asserts backend AUC equals mean-of-fold-AUCs within 1e-10.

### Gotcha: majority vote tie-break policy is implicit "favor outlier"

`np.unique([-1, 1])` returns `[-1, 1]` sorted, and `np.argmax(counts)` picks the first max → ties go to `-1` (outlier). Under even `cv_n_repeats`, exact ties become a silent "flag as outlier" policy. Flagged by codex + all three peer-review models. Kept as-is because it's a sensible conservative default for contamination detection (when ambiguous, screen); documented in an inline comment. Callers who want a different tie-break can pick odd `cv_n_repeats` to eliminate ties.

### Non-issues caught by peer-review but wrong

- MiniMax claimed `np.unique(y)` in binary-specificity fix could return >2 labels for multi-class. Wrong — the fix is inside `if is_binary_classification:` which by definition means exactly 2 classes.
- DeepSeek flagged BER semantic change as potential regression. Already addressed by the round-3 correlation-argument design.
- All three models flagged the unknown-`cv_strategy` → `ValueError` change as "breaking." Not really — typos previously silently defaulted to K-fold, producing wrong numbers without warning. Raising is correct.

### Results

- **71/71 `tests/test_cv_strategy.py` tests pass** (added `test_one_class_repeated_kfold_auc_is_mean_of_fold_aucs`).
- **138/138 adjacent suites** pass.

---

## 2026-04-14 — PR #4 round-3 review fixes: BER pool consistency + one-class repeated-CV per-sample reduction (Claude Opus 4.6)

### Context

Round-2 commit (`75a1eb5`) fixed bugs from the prior review cleanly. Round-3 review by 4 specialized agents (code-reviewer, silent-failure-hunter, test-analyzer, comment-analyzer) + codex + peer-review panel (MiniMax + DeepSeek) found two real bugs, one defensive fix, and a test-coverage gap. Plan: `~/.claude/plans/swirling-zooming-frost.md`.

### Bug: Backend BER inconsistent with BalancedAcc under Repeated K-Fold

**Root cause:** Round-2 commit pooled per-sample for Accuracy/F1/Precision/Recall/BalancedAcc/Kappa/MCC/Specificity under repeated CV but kept BER as mean-of-folds with a comment claiming "BER requires probabilities." BER is actually `1 - balanced_accuracy_score(y_test, y_pred)` at `search.py:4012-4015` — label-based, not probability-based. Under repeated CV the two metrics diverged (BER was mean-of-fold-BERs, BalancedAcc was from pooled labels), so the invariant `BER = 1 - BalancedAcc` broke.

**Fix:** Move BER into the pooled branch with `mean_ber = 1.0 - mean_balanced_acc`. Keep mean-of-folds for AUC and LogLoss (those genuinely need probabilities). Regression test `test_ber_pools_under_repeated_kfold` asserts the invariant numerically on a repeated-K-Fold classification run.

### Bug: One-class `run_one_class_cv` pools correlated observations under repeated CV

**Root cause:** Every fold's test set includes ALL outliers (`contamination.py:633`: `test_idx = np.concatenate([test_inlier, outlier_indices])`). Under Repeated K-Fold with `k` folds × `r` repeats, inliers appear r times in the pool, outliers appear k*r times. The metric EXPECTED VALUES don't change (TP/FP/TN/FN all scale together so ratios are preserved — codex was correct that the original "double-counts → inflates metrics" framing was wrong). The REAL issue is statistical: pooled metrics come from CORRELATED observations (the same sample's r predictions come from related models). Inconsistent with the regression/classification path which now per-sample-reduces under repeated CV.

**Fix — scope-limited to repeated CV:** After the fold loop, if `_is_repeated_cv(kf)`, reduce pool to one prediction per original sample index (majority vote via `np.unique + np.argmax` for preds; mean for scores). Added `all_test_idx` instrumentation alongside existing `all_test_labels`/`all_y_pred`/`all_scores`. Plain K-Fold and LOO unchanged.

**Plain K-Fold preserved by design:** codex correctly noted plain K-Fold one-class has the same correlated-prediction structure. NOT fixed here because:
1. Plain K-Fold one-class behavior baked since commit `057d9f6` — changing it rebaselines all existing user models.
2. Repeated K-Fold is NEW in this PR — no downstream deps.
3. Follow-up: migrate plain K-Fold one-class to per-sample-pooled in a separate PR with explicit changelog entry.

Regression test: `test_one_class_repeated_kfold_matches_reference_reduction` replays the same splits manually, reduces per-sample, computes reference Sensitivity/Specificity, and asserts backend matches to 1e-10. Sanity test: `test_one_class_plain_kfold_still_works_after_loop_restructure` pins that the `all_test_idx` instrumentation and `_is_repeated_cv` branch don't break non-repeated paths (catches missing imports).

### Defensive: binary specificity `labels=` kwarg

**Root cause:** The round-2 pooled-specificity block at `search.py:~4341-4353` unpacks `tn, fp, fn, tp = confusion_matrix(all_y_test, all_y_pred).ravel()`. If both y_true and y_pred are single-class, the matrix is 1×1 and unpack crashes with `ValueError: not enough values to unpack`. In practice the upstream classification validator rejects single-class y, so y_true always has both classes and the unpack works (cm is 2×2 via sklearn's union-of-labels). Bug is theoretical but a future refactor could break that invariant.

**Fix:** Pass `labels=np.unique(y)` to `confusion_matrix` so the shape is always 2×2. Codex corrected my original plan: `y_np` is not in scope inside `_run_single_config` — use the function-arg `y`. Regression test `test_pooled_binary_specificity_survives_degenerate_predictions` pins the sklearn contract.

### Gotcha: `run_search` returns a tuple, not a DataFrame

While writing tests I hit `AttributeError: 'tuple' object has no attribute 'iloc'`. `run_search` returns `(df_ranked, label_encoder)` (line 3090). My initial tests did `df = run_search(...)`. Pattern across existing tests is either `df, _ = run_search(...)` or a test that doesn't need the return (e.g. `pytest.raises`). Fixed all new tests to unpack.

### Gotcha: `confusion_matrix` shape depends on BOTH y_true and y_pred

My initial binary-specificity sklearn test asserted `cm.shape == (1, 1)` for `y_true=[0,0,0,0,1,1], y_pred=[0,0,0,0,0,0]` — wrong. sklearn uses the union of unique labels from BOTH inputs, so the matrix is 2×2 when either array has multiple classes. The 1×1 case requires BOTH arrays to be single-class — which is precisely the (theoretical) degenerate case the `labels=` kwarg protects against.

### Gotcha: editable install roots at main repo, pytest respects worktree

`python -c 'from spectral_predict.search import run_search; inspect.getsourcefile(...)'` from the worktree returned the MAIN repo's `search.py`, not the worktree's. Why: `pip install -e` was run against the main repo, so ad-hoc imports use the installed path. pytest, however, respects the worktree via `pyproject.toml`'s `pythonpath=["src"]`. Not a bug, just a surprise when debugging imports — always route verification through pytest in worktrees.

### Minor cleanups (same commit)

- `contamination.py:583` — tightened fall-through `else: min_inliers = n_folds` to `raise ValueError(f"Unknown cv_strategy: {cv_strategy!r}")`. Codex noted `build_cv_splitter` already rejects unknown strategies (`cv_utils.py:188`) — belt-and-suspenders against future additions that bypass the constructor path.
- `cv_utils.py` — `validate_cv_strategy_for_task` one-class branch now str-coerces both sides of `inlier_label` comparison (matches the convention in `search.py:~4878`). Prevents silent "too few inliers" errors on dtype mismatches.
- `search.py:~4262` — guard `if not cv_metrics: raise ValueError("All CV folds failed")`. Prevents silent `accuracy_score([], [])` → 0.0. Used `ValueError` not `RuntimeError` because no caller catches `RuntimeError` specially (codex grep).
- `cv_utils.py` — deleted unused `sample_truth` variable in `reduce_repeated_cv_predictions`.
- Tightened 3 stale-prone comments (dropped "codex" refs, dropped line-number refs).
- `reduce_repeated_cv_predictions` docstring — added explicit "ORDER MUST MATCH" line warning about silent miscorrespondence.

### What still doesn't have a test

- Direct unit test of `reduce_repeated_cv_predictions` on regression is strong; classification only tested at integration level. The 3 existing `TestExportScriptMatchesBackend` tests cover the parity contract at the template/reducer level, which is sufficient.
- Plain K-Fold one-class K-fold-bias is preserved-by-design; no test asserts this. Documented in Follow-Ups instead.

### Follow-Ups

- **Plain K-Fold one-class metric rebaseline** — plain K-Fold one-class has the same correlated-prediction structure as repeated K-Fold (outliers appear k times, inliers once). Fixing would rebaseline all existing user one-class models. Separate PR with changelog entry. `contamination.py:633,703-720`. Proposed fix: always per-sample-reduce for one-class regardless of `_is_repeated_cv`.
- **More aggressive e2e classification parity test** — current test only asserts `BER = 1 - BalancedAcc`. A parity test that runs `run_search` with known model + compares to `cross_val_predict_pooled` directly (same model) would catch wire-up bugs more reliably. PLS-DA is a custom pipeline that can't trivially be swapped for sklearn's LogisticRegression.

### Results

- 60 (from round-2) + 10 (round-3) = **70/70 tests in `tests/test_cv_strategy.py` pass**.
- **138/138 adjacent suites** pass (contamination_detection, bayesian_utils, search_comprehensive, unified_bayesian_baseline, nsga2_search).
- Net diff: ~+200 / -30 across 5 code files + 1 test file.

---

## 2026-04-14 — PR #4 round-2 review fixes: Repeated K-Fold backend pooling + exported-script correctness (Claude Opus 4.6)

### Context

PR #4 (`claude/cv-strategy-overhaul`) went through a second pre-merge review with codex + code-reviewer agent. Both flagged significant correctness bugs that survived the first-round fixes. CI was red but from pre-existing infrastructure (no `$DISPLAY` for tkinter on Ubuntu runners), not from this PR. The findings below were the actual merge blockers.

### Bug: Backend Repeated-K-Fold regression RMSEcv double-counts duplicated samples

**Root cause:** `search.py:4260` aggregated CV predictions by `np.concatenate([m["y_test"] for m in cv_metrics])` + concat of `y_pred`, then computed `RMSE = sqrt(mean_squared_error(all_y_test, all_y_pred))`. Under `RepeatedKFold` (the whole point of this PR), each sample appears in `n_repeats` different test folds, so flat concatenation duplicates every sample row `n_repeats` times. The resulting RMSE scores duplicates as independent observations and diverges from the per-sample-pooled RMSE that `cv_utils.cross_val_predict_pooled` computes — the number shown in the Results tab was quietly incorrect whenever the user picked Repeated K-Fold.

Same class of bug for classification (`search.py:4312`): headline metrics were `np.mean([m["Accuracy"] ...])` across folds (duplicated samples each contribute to `n_repeats` fold accuracies), and the per-class `classification_report` ran on the concatenated-duplicated predictions. The majority-vote semantics in `cv_utils._majority_vote` were never reached from the grid-search path.

The helper the PR already shipped (`cross_val_predict_pooled`) had the correct per-sample reduction, but `search.py` had its own fold-aggregation path and never routed through it.

**Fix:** Added `reduce_repeated_cv_predictions(cv_metrics, splits, n_samples, task_type)` in `cv_utils.py`. Post-hoc reduction from the already-collected fold dicts — no re-fitting. Regression averages per-sample predictions; classification majority-votes per-sample via `Counter.most_common(1)` (deterministic for fixed split order + random state). `search.py` now realises `splits = list(cv_splitter.split(X, y))` once, passes to both serial/parallel fold execution, and under repeated CV pools per-sample before computing RMSE / R² / regional RMSE (regression) and Accuracy / F1 / Precision / Recall / BalancedAcc / Kappa / MCC / Specificity-for-binary (classification). AUC / LogLoss / BER still come from mean-of-folds because those need probabilities, not labels. Non-repeated paths (KFold, LOO) are unchanged — concat is per-sample-natural when each sample appears in exactly one fold.

Regression tests added: `TestPostMergeReviewFixes::test_backend_regression_pools_under_repeated_kfold`, `test_reduce_repeated_cv_regression_matches_manual_pool`, `test_reduce_repeated_cv_classification_majority_vote`.

### Bug: Exported classification script raised `NameError: name 'all_y_true' is not defined`

**Root cause:** `templates/validation.py:190,193` (`METRICS_CLASSIFICATION_TEMPLATE`) printed `confusion_matrix(np.array(all_y_true), y_pred_cv)` and `classification_report(np.array(all_y_true), y_pred_cv)`, but the new per-sample-pooling CV block at lines 151-156 defines `all_y_true_arr` and `all_y_pred_arr`. Any exported classification script crashed on the first metric print — never triggered by the existing test suite because `tests/test_cv_strategy.py::TestExportScriptMatchesBackend` reimplemented the pooling logic inline rather than rendering + executing the template.

**Fix:** changed both references to `all_y_true_arr` / `all_y_pred_arr`. Added `TestPostMergeReviewFixes::test_classification_metrics_template_has_no_nameerror` that renders the template and `exec()`s it against a synthetic dataset.

### Bug: `code_generator.py` imbalance path hardcoded K-Fold and flat-extended predictions

**Root cause:** `_render_cross_validation_with_imbalance` (`code_generator.py:1413-1540`) had two hardcoded literal blocks (one per task type) that emitted `cv = StratifiedKFold(n_splits={cv_folds}, ...)` / `cv = KFold(...)` regardless of `self.cv_strategy`. When the user selected LOO or Repeated K-Fold AND enabled any imbalance method (SMOTE / RegressionUndersampler / class_weight / etc.), the exported script silently ran plain K-Fold while the backend ran the chosen strategy. Also extended predictions flat (`all_y_true.extend(y_test); all_y_pred.extend(y_pred_fold)`) instead of per-sample reduction, so the classification accuracy would scale duplicates under Repeated K-Fold.

**Fix:** both branches now import `_cv_splitter_code` from `templates.validation` (single source of truth for CV emission across normal and imbalance paths) and mirror the per-sample reduction pattern (`preds_per_sample` / `truth_per_sample` dicts → sorted keys → mean or majority-vote). Regression test: `test_imbalance_code_generator_honors_cv_strategy`.

### Bug: Pre-existing indentation bug in imbalance-regression final-model template

**Root cause:** `_render_final_model_with_imbalance` had `fit_kwargs = {{}}` at column 0 followed by `    if sample_weight is not None:` at column 4 — Python `IndentationError: unexpected indent`. Every imbalance-regression export script failed to compile. Pre-existed since commit `44af5545` (2026-01-23), so not actually introduced by this PR — but the compile-check in the new test caught it. Fixed while touching adjacent code: de-indented the `if sample_weight is not None` block to match the flat layout.

### Bug: PCA-SIMCA + LOO with tiny inlier sets still failed silently

**Root cause:** `contamination.py:578-583` set `min_inliers = 2` for LOO regardless of model. PCA-SIMCA requires 3 training samples (`PCASIMCA.fit` hard floor at `:128`). With 3 inliers and LOO, every training fold has only 2 samples → every fold fails → skip guard fires → trial returns +inf silently (the same class of bug the prior PCA-SIMCA small-sample fix tried to kill).

**Fix:** model-aware minimum: `SIMCA_FIT_FLOOR = 3`. Under LOO, require `min_inliers = 4` for PCA-SIMCA (so every leave-one-out training fold has ≥ 3 samples); under K-fold / Repeated K-fold, require `n_inliers >= ceil(3 * n_folds / (n_folds - 1))`. Emits a clear skip_reason ("PCA-SIMCA needs N+ so every training fold has >= 3 samples"). Regression test: `test_one_class_loo_rejects_pca_simca_with_three_inliers`.

### Bug: `validate_cv_strategy_for_task` didn't reject `n_repeats <= 0` or validate one-class inlier counts

**Root cause:** The validator ran before training for classification but was a no-op for one-class, and never checked `n_repeats` at all. `RepeatedKFold(n_repeats=0)` would construct happily and the fold loop would yield zero folds, leading to empty `cv_metrics` and downstream division-by-zero or all-NaN metrics.

**Fix:** `validate_cv_strategy_for_task` now accepts `n_repeats` and `inlier_label` parameters. Rejects `n_repeats < 1` upfront. For `task_type='one_class'`, validates inlier count against the strategy's minimum (LOO: 2, K-fold: `n_folds`). Wired into all three callers: `search.run_search`, `search.run_one_class_search`, and `unified_bayesian.run_unified_bayesian`. Regression tests: `test_validate_cv_strategy_rejects_zero_repeats`, `test_validate_cv_strategy_rejects_one_class_too_few_inliers`.

### Bug: `cv_strategy` / `cv_n_repeats` not persisted on Bayesian `trial.user_attrs`

**Root cause:** Both the non-one-class Bayesian objective (`unified_bayesian.py:1404-1414`) and the one-class objective (`:1037-1056`) wrote trial user_attrs for preprocessing config, n_vars, model_params, subset_tag — but never for `cv_strategy` / `cv_n_repeats`. Converted result rows carried them (via `convert_study_to_dataframe`), but any raw-study consumer reading `study.trials_dataframe()` or `trial.user_attrs` directly would lose the CV configuration.

**Fix:** two-line addition to both objective function user-attr blocks. Noticed by code-reviewer; not a ship-blocker on its own but closes a silent landmine.

### Minor: `cv_utils._model_is_classifier` had bare `except Exception`

**Root cause:** `try: return is_classifier(inner) except Exception: return False`. If a custom estimator's `_estimator_type` tag raised for any reason, the repeated-CV predict path would silently fall back to numeric averaging of integer class labels — which is exactly the bug the majority-vote code exists to prevent.

**Fix:** narrowed to `except (AttributeError, TypeError)` and added a `warnings.warn` with the class name + error so the silent fallback becomes visible. Low risk under current sklearn but closes a known-silent-failure path.

### Gotcha: splits generator must be realised before pooling

`KFold.split()` / `RepeatedKFold.split()` return generators. The existing code in `search.py` called `cv_splitter.split(X, y)` once inside the list comprehension that built `cv_metrics`. To pool per-sample after the fact, we need the test indices from each split. Solution: `splits = list(cv_splitter.split(X, y))` once before the fold loop, pass the same realised list to both serial and parallel fold executors and to `reduce_repeated_cv_predictions`. A second call to `cv_splitter.split()` would produce a fresh generator with the same seed — but would not align with the `cv_metrics` order under any parallel joblib backend that reorders futures.

### Results

- 60/60 tests in `tests/test_cv_strategy.py` pass (8 new regression tests added under `TestPostMergeReviewFixes`).
- 138/138 adjacent tests pass: `test_contamination_detection`, `test_bayesian_utils`, `test_search_comprehensive`, `test_unified_bayesian_baseline`, `test_nsga2_search`.
- Net diff: +383 / -65 across 7 files.

---

## 2026-04-11 — Grid-search OC validation silent NaN + final-merge cleanup (Claude Opus 4.6)

### Bug: Grid-search one-class results show no `val_*` metrics, Bayesian works fine

**Symptom:** User ran the final pre-merge verification of PR #3. Bayesian one-class displayed validation columns correctly, but grid-search one-class results had empty `val_*` columns even with the same validation set, same inlier label, same `validation_top_n`.

**Root cause:** `compute_validation_metrics_for_top_one_class_models` (`contamination.py:909`) reads the preprocessing name from `row.get('PreprocessBase', row.get('Preprocess', 'raw'))` and passes it to `build_preprocessing_pipeline`. Bayesian writes a clean base name in `PreprocessBase` (`unified_bayesian.py:1995` — e.g. `'snv_deriv'`), but the one-class grid path at `search.py:5136-5174` and `:5640-5710` only wrote `Preprocess` (the *display* name like `'snv_deriv1_w11'`, including window suffix) and never set `PreprocessBase`. `build_preprocessing_pipeline` doesn't accept display names — it raises `ValueError("Unknown preprocess: snv_deriv1_w11")` at `preprocess.py:548`. The row-level `except Exception: continue` in the validation helper at `contamination.py:1058` swallowed every ValueError, logged a warning that never reached the GUI progress tab, and dropped val_* to NaN for every derivative-based grid-search row. The classification/regression grid path at `search.py:4506-4507` had the right contract all along — both `Preprocess` (display) and `PreprocessBase` (clean) — but nobody copied that to the one-class result-dict construction when it was added.

**Fix:** Two layers, defense in depth:
1. **Producer side** (`search.py:5136-5174` and `:5640-5710`) — added `"PreprocessBase": preprocess_cfg.get("method", preprocess_cfg["name"])` to both the full-spectrum and variable-selected one-class result dicts. Mirrors the classification grid contract at line 4506-4507. The key in the one-class `preprocess_cfg` is `'method'` rather than `'base_name'`, so the lookup uses the right name.
2. **Consumer side** (`contamination.py`) — added `_normalize_preprocess_for_pipeline()` helper that strips `_w<digits>` suffixes and `deriv\d+` digits from a Preprocess display name, falling back to `'raw'` if it can't be normalized. The validation helper now invokes this on whatever name it ends up with so any caller that forgets `PreprocessBase` (older results CSV reload, third-party scripting) still produces metrics rather than silent NaN.

Regression test: `tests/test_contamination_detection.py::TestGridSearchValidationMetricsParity` covers both the fixed-row case and the missing-PreprocessBase fallback. Before the fix the second test failed with the exact ValueError above, captured in pytest's log: `[OC Validation] Row 0 failed: Unknown preprocess: snv_deriv1_w11`.

**Why it slipped past three rounds of CodeRabbit/Gemini review:** the validation helper had no test coverage at all (`grep compute_validation_metrics_for_top_one_class_models tests/` → no matches). The producer-side fix is two lines per call site, but no test wired the producer's output through the consumer's input, so the schema mismatch was invisible until manual GUI testing.

### Bundled minimum-to-merge cleanup (same commit)

Six other findings from the multi-reviewer pre-merge audit (Codex + Qwen3-235B + CodeRabbit/Gemini triage), all small and localized:

1. **`search.py:5698-5703` — `all_vars`/`top_vars` swap for grid-search varsel rows.** Was storing the pre-subset working set (`wavelengths_current`) in `all_vars` and the selected subset (`wavelengths_subset`) in `top_vars`. Downstream consumers (`spectral_predict_gui_optimized.py:30556-30573` Model-Dev reload, `contamination.py:972-989` validation rebuild) read `all_vars` as "the trained wavelength list", so a variable-selected grid-search one-class model would be reconstructed on the full spectrum. Fixed: both fields now hold the selected subset, matching the Bayesian contract at `unified_bayesian.py:1046-1050`.

2. **`spectral_predict_gui_optimized.py:34467-34478` — refined OC external validation bypassed `one_class_metrics()`.** Inline `balanced_accuracy_score(y_val_oc, val_preds)` and `roc_auc_score(y_val_oc, val_scores)` got the AUC sign convention wrong (rest of the codebase treats outliers as the positive class and negates `decision_function`) and didn't NaN-out clean-only validation sets. Same model could display different validation quality in Results tab vs. Model Development. Fixed: replaced with `metrics = one_class_metrics(y_val_oc, val_preds, val_scores)` and a small `_fmt` helper for the display string.

3. **`spectral_predict_gui_optimized.py:35838` — `inlier_class_label` save wrote raw combobox value.** When the user accepted auto-detect via the confirm dialog at `21852`, `self.inlier_class_label.get()` was still the empty string and round-tripped that into the saved model. Fixed: prefer `self.refined_config.get('inlier_class_label')` (set by the refinement thread at line 34373 with the actually-trained label), fall back to combobox.

4. **`models.py:379-390` — `build_model()` silent fallback to classification for unknown one-class names.** Typo'd or registry-mismatched one-class model names silently routed through the classification branch, eventually raising a confusing error. Fixed: raise `ValueError(f"Unknown one-class model: {model_name!r}")` immediately. No live caller relied on the surrogate fallback (verified by grep — `compute_one_class_importances` imports LightGBM directly).

5. **`scoring.py:254-260` — `_compute_unified_complexity` lost PCA-SIMCA `n_components` when `Params` was a dict.** `isinstance(params_str, str)` was False for in-memory result rows where Params is already a dict, so `params_dict={}` and `n_components` defaulted to NaN, collapsing `lv_complexity` to the median fallback (50). Fixed: branch on `dict` vs `str` before calling `ast.literal_eval`.

6. **`contamination.py:86` — `PCASIMCA.__init__` accepted out-of-range `alpha`.** `alpha=0` or `alpha=1` would feed `stats.chi2.ppf(1-alpha, ...)` undefined inputs, silently producing a model that flags everything (or nothing). Not reachable from any current GUI/grid/Bayesian entry point but a public API footgun. Fixed: one-line `if not 0 < alpha < 1: raise ValueError(...)` in `__init__`.

### Follow-up flagged but NOT addressed in this PR — one-class hyperparameter exposure gap

User noted during the same session: **none of the one-class model hyperparameter grids are exposed in the Model Config subtab**, unlike every regression/classification model. The Model Config tab (`_create_tab4c_model_configuration` at `spectral_predict_gui_optimized.py:11373`) has individual collapsible hyperparameter cards for each regression/classification model (Random Forest at line 11726, Ridge at 11854, Lasso at 11918, ElasticNet at 11978, PLS at 12053, etc.) where users edit min/max ranges, num steps, and per-model knobs. For one-class there is only a tiny `oc_hyperparams_frame` at `gui:6040-6054` with four single-value spinboxes (`oc_nu`, `oc_contamination`, `oc_alpha`, `oc_n_components`).

Everything else is hardcoded in `ONE_CLASS_PARAM_GRIDS` inside `src/spectral_predict/contamination.py`:
- OneClassSVM: `kernel`, `gamma`, additional `nu` values
- IsolationForest: `n_estimators`, `max_samples`, `max_features`, additional `contamination` values
- EllipticEnvelope: `support_fraction`, additional `contamination` values
- LocalOutlierFactor: `n_neighbors`, `metric`, additional `contamination` values
- PCA-SIMCA: additional `n_components` values, additional `alpha` values

This violates the project rule from `CLAUDE.md`: *"All hyperparameters are exposed and user-editable."* It also means a user who wants to widen the search space for one-class (e.g. test n_neighbors=[10, 20, 35, 50] for LOF) has to edit Python source instead of clicking checkboxes/spinboxes.

**Not blocking this PR's merge** (the four exposed scalars cover the most-tuned parameters and the hardcoded grids are sane defaults), but it's a design debt to pay down before claiming feature parity with regression/classification. Likely scope: add five collapsible hyperparameter cards to `_create_tab4c_model_configuration` (one per OC model family), thread the resulting per-model dicts through `run_one_class_search` and `run_unified_bayesian` instead of `ONE_CLASS_PARAM_GRIDS`, and gate the existing scalars on Custom tier the same way classification/regression do.

---

## 2026-04-10 — One-Class Prediction Disaster: Order-of-Operations Bug (Claude Opus 4.6)

### Bug: Every specimen labeled "Outlier" at predict time, including training data

**Symptom reported:** User trained one-class models via Bayesian optimization, loaded top results into Model Development, ran refinement, saved models, then predicted on novel specimens. *Every* specimen — including specimens that were literally in the training set — was labeled "Outlier" in the main Prediction Results table. All one-class model families affected (OneClassSVM/IF/LOF/EllipticEnvelope/PCA-SIMCA). Multi-class models on the same data worked correctly.

**Root cause:** Order-of-operations mismatch between training and predict. The one-class early-exit branch in `_run_refined_model_thread` (`spectral_predict_gui_optimized.py:34260-34283`) applies `prep_pipeline_oc.fit_transform(X_full)` to the **full** spectrum *then* subsets to selected wavelengths:

```python
X_full_preprocessed = prep_pipeline_oc.fit_transform(X_full)   # full-spectrum fit
X_work = X_full_preprocessed[:, wavelength_indices]            # subset AFTER
```

The `StandardScaler` inside `run_one_class_cv` is then fit on `X_work`, so it expects data in the (preprocess-then-subset) feature space. BUT `self.refined_config` at line 34342 hardcodes `'use_full_spectrum_preprocessing': False`, and `self.refined_full_wavelengths` is never assigned in the one-class branch. These propagate into model metadata at lines 35726-35727. At predict time, `model_io.py:640-677` sees both flags False/None and takes Mode B (subset-first → preprocess), which for SNV / SG derivatives / baseline correction produces mathematically *different* feature values than training. The scaler then receives distributionally alien inputs, `model.predict()` returns -1 across the board, including on samples the model was trained on.

**Why the 2026-04-10 "scaler persistence" fix did not catch this:** that fix ensured `cal_scaler` / `cal_pca_reducer` / `oc_score_stats` were persisted for grid-search results. Orthogonal to this bug. The scaler IS being saved correctly; it is just receiving inputs in the wrong feature space at predict time.

**Why the multi-class path works on the same data:** `spectral_predict_gui_optimized.py:33815` sets `use_full_spectrum_preprocessing = True` for regression/classification, and line 35515 assigns `self.refined_full_wavelengths`. The one-class early-exit branch was a copy-paste that missed both lines. This is the Nth instance of the pattern flagged in SESSION_LOG `2026-04-09` ("One-class early returns are the #1 bug pattern").

**Severity note on Savitzky–Golay:** for `deriv=1` or `deriv=2` preprocessing the subset-first path is not merely slightly different — the SG window at subset boundaries lands on indices outside the subset, producing artificial discontinuities and derivative values off by orders of magnitude. This is the dominant contributor when SG is enabled (flagged by qwen3-235B review).

**Fix:** Two lines in `_run_refined_model_thread`:
- Assign `self.refined_full_wavelengths = list(original_wavelengths)` right after `X_work` is built.
- Flip `'use_full_spectrum_preprocessing'` from `False` to `True` in `self.refined_config`.

The metadata save path at `:35726-35727` already reads these correctly — no save-side change needed.

### Secondary bug: Uncertainty-tab display always shows "Outlier"

`spectral_predict_gui_optimized.py:38245`:

```python
predictions = self.predictions_df[model_name].values   # STRINGS at this point
pred_label = "Inlier" if predictions[i] == 1 else "Outlier"   # str == int → False
```

Line 37770-37780 converts numeric +1/-1 to `"Inlier ({label})"` / `"Outlier"` strings before storing in `predictions_df`. The Uncertainty display then compares those strings to the integer `1`, which is unconditionally False, so every row shows "Outlier" regardless of the real prediction. Not what the user was seeing in *this* report (they confirmed they were on the Results tab), but would silently contradict the Results tab once the primary bug is fixed. Fix: replace with `isinstance(raw, str)` branch that accepts already-stringified predictions.

### Latent bug: Bayesian path never stores cal_scaler/cal_pca_reducer/oc_score_stats on trials

`src/spectral_predict/unified_bayesian.py:1009-1037` stores metrics and preprocessing config but never calls `trial.set_user_attr('cal_scaler', ...)`, `'cal_pca_reducer'`, or `'oc_score_stats'`. The 2026-04-10 fix only covered `search.py` (grid search). Not biting the user *today* because all saves flow through refinement (which re-runs CV), but a latent landmine for any future "save direct from results" code path. Close it, but audit `convert_study_to_dataframe` consumers first — storing sklearn objects as DataFrame columns could break `to_csv` / Treeview inserts / sort handlers.

### Architecture insight: test suite does not exercise save→load→predict-on-novel-specimens for one-class with preprocessing

All 44 existing contamination tests passed. None of them round-trip through `save_model` → `load_model` → `predict_with_model` against training data with an actual preprocessing pipeline (SNV/SG). This is exactly why the bug was invisible and PROJECT_STATUS listed prediction as working. Added `test_one_class_save_load_predict_roundtrip_with_preprocessing` to cover it.

### Architecture insight: `use_full_spectrum_preprocessing=False` is effectively dead for regression/classification

Searching the codebase, the flag is only ever set to `True` in the non-one-class paths (lines 28371, 33815). It is only ever set to `False` in the one-class early-exit branch (34342 — the bug) and in a couple of load/default contexts. If no path actually wants Mode B anymore, Step 1 here is a point fix but the underlying dichotomy in `model_io.py:640-677` could be simplified in a future cleanup. Out of scope for this bugfix.

---

## 2026-04-10 — Tooltip Gap Fixes for Metric Column Headers (Claude Opus 4.6)

### Gap: Results Treeview had no tooltip for Sensitivity, Sensitivitycv, AUC, AUCcv columns

**Root cause:** `TreeviewHeaderTooltip._get_tooltip_for_column()` at `spectral_predict_gui_optimized.py:354` does exact-match lookup against `TOOLTIP_CONTENT['metrics']`. The one-class Bayesian/grid result dict emits keys `Sensitivity`/`Sensitivitycv`/`AUC`/`AUCcv` (see `:34324-34330`), but those keys were never added to the tooltip dict — so hover silently returned `None`. Same gap existed for classification validation columns `val_Accuracy`/`val_Precision`/`val_Recall`/`val_F1` (populated at `:35832-35833`) and regression validation columns `RMSEP`/`R2pred`.

### Gotcha: `R²pred` superscript vs. `R2pred` ASCII

The main results Treeview uses the **ASCII** key `R2pred` (defined authoritatively in `src/spectral_predict/search.py:416, 647`). The separate ensemble comparison Treeview in `_populate_ensemble_comparison_table` around `:27977` uses the **superscript** key `R²pred`. The tooltip dict needs only `R2pred` because the ensemble comparison Treeview has no `TreeviewHeaderTooltip` attached at all — that's a separate pre-existing gap for a future fix.

### Gotcha: Some existing metric tooltips were too jargony for the target audience

`ROC_AUC`, `Kappa`, `MCC`, `BER`, `LogLoss` assumed ML background (e.g., "agreement beyond chance", "Balances all confusion matrix categories", "Standard metric in PLS-DA"). Rewrote all five with concrete, audience-appropriate wording plus "rule of thumb" numeric guidance where available. `ROC_AUCcv` was a one-liner placeholder — expanded to match.

### Gotcha: `LogLoss` is not a calibration metric

Initial rewrite said "measures how well-calibrated the model's probability estimates are". Code review flagged this: LogLoss captures both calibration AND discrimination. Changed to "measures how good the model's probability estimates are" to avoid the technically-incorrect calibration framing.

### `TOOLTIP_CONTENT` requires GUI restart

The dict is populated at import time (`:1253`), so tooltip edits only take effect after a full Python/GUI restart. Not a hot-reloadable change.

---

## 2026-04-10 — One-Class External Validation Parity (Claude Opus 4.6)

### Bug: One-class external validation only populated on top 10 models regardless of validation_top_n

**Root cause:** `spectral_predict_gui_optimized.py:25133` hardcoded `for idx in range(min(10, len(results_df))):` in the one-class validation block — ignored `self.validation_top_n` (default 700 that the classification/regression path honors). User running Bayesian one-class with 700-trial validation saw only the first 10 rows populated; rest were NaN.

### Bug: One-class external validation had 3 metrics (BalancedAcc/Sensitivity/Specificity) while cal/CV had 7

**Root cause:** Same inline GUI block hand-computed only `balanced_accuracy_score`/`recall_score` (for pos_label=-1 and +1). Never called `decision_function` to get scores for AUC, and never computed Precision/F1/Accuracy. Meanwhile `run_one_class_cv()` stored 7 metrics for cal AND CV via `one_class_metrics()` — so the Results tab showed the column mismatch.

**Fix:** Added `compute_validation_metrics_for_top_one_class_models()` in `contamination.py` — mirrors `compute_validation_metrics_for_top_models()` in `search.py` (the classification/regression template). Reuses existing `one_class_metrics()`, `build_one_class_model()`, and `build_preprocessing_pipeline()` rather than duplicating logic. Replaced the inline GUI block (spectral_predict_gui_optimized.py ~25115-25227) with a single call passing `self.validation_top_n.get()`.

Adds all 7 val_* columns in canonical order: val_Sensitivity, val_Specificity, val_Precision, val_F1, val_Accuracy, val_BalancedAcc, val_AUC. Preprocessing cache keyed by config so models sharing preprocessing only pay the transform cost once.

### Bug: Progress tab flooded with blank lines during one-class validation metrics step

**Root cause:** First cut of `compute_validation_metrics_for_top_one_class_models` called `progress_callback` every row with `{'stage': 'validation', 'current': ..., 'total': ...}` — no `'message'` key. `_progress_callback` at `spectral_predict_gui_optimized.py:25958` does `msg = info.get('message', '')` then `self._log_progress(msg)`, so every invocation appended an empty line. 80 rows → 80 blank lines.

**Fix:** Throttled callback to every 10 rows and included a `'message'` key, matching the classification helper's pattern at `search.py:738-744`.

### Bug: val_* columns appeared at far-right of Results tab instead of adjacent to cal/CV columns

**Root cause:** Helper only did `df_results[col] = np.nan` which appends new columns at the end of the DataFrame. Classification helper explicitly reorders at `search.py:748-789` to slot validation columns right after their calibration counterparts.

**Fix:** After populating metrics, helper now removes val_* columns and re-inserts them immediately after the last cv metric present in the frame (prefers `AUCcv`, falls back through the cv metric chain, then to `Imbalance`/`SubsetTag`). Final layout now reads: cal metrics → cv metrics → val metrics → top_vars/all_vars/CompositeScore/Rank.

**Verified:** Synthetic 25-model DataFrame — progress_callback invoked exactly 2 times (at i=10, i=20) with proper messages; val_Sensitivity lands at index `AUCcv + 1`; existing 44 contamination tests still pass.

### Bug: PCA-SIMCA silently produces 0 rows in Bayesian one-class runs on small training sets

**Symptom reported:** User selected 5 one-class models × 20 Bayesian trials and expected 100 rows in the Results tab. Got 80 — PCA-SIMCA was completely missing. Running again with 4 models INCLUDING SIMCA (implying a different split/inlier count) produced SIMCA results. User asked: "is there a reason small validation set would impact SIMCA more than other models?"

**Root cause:** `PCASIMCA.fit()` at `contamination.py:108-111` had a hardcoded guard:

```python
if n_samples < 10:
    raise ValueError(f"Need at least 10 clean samples to fit DD-SIMCA, got {n_samples}")
```

User's dataset had ~7 training inliers after the validation split. 5-fold CV produces training folds of size `4N/5` = 5–6 samples. Every single fold tripped the guard → `run_one_class_cv` skip guard (`contamination.py:599`) fired → Bayesian objective returned `+inf` → `convert_study_to_dataframe` dropped all 20 SIMCA trials (`unified_bayesian.py:1960-1961`) → 0 SIMCA rows in final DataFrame.

**Answer to the user's question: yes, SIMCA alone is vulnerable to small training sets.** The other 4 one-class models each have explicit small-sample fallbacks:

| Model | Small-sample accommodation |
|---|---|
| OneClassSVM | Kernel-based — fits 2+ points |
| IsolationForest | Random split trees — fits 1+ points |
| LOF | `n_neighbors` auto-clamped to `n_samples - 1` |
| EllipticEnvelope | PCA fallback at `contamination.py:568-572` when `n_features > n_samples` |
| PCA-SIMCA | **None — hard floor at 10** |

So the validation set size doesn't affect SIMCA "more" in some abstract sense; it's that carving any fixed amount off an already-small training set pushes SIMCA over its unique cliff while the others coast.

**Fix (`contamination.py:105-128`):** Replaced the 10-sample floor with the true mathematical minimum:

```python
if n_samples < 3:
    raise ValueError(f"Need at least 3 clean samples to fit DD-SIMCA, got {n_samples}")
# ...
max_components = min(n_samples - 1, n_features)
if max_components < 1:
    raise ValueError(...)
```

`n_components` was already clamped to `n_samples - 1` at line 114, and `_fit_chi2` already has a method-of-moments fallback (`:187-210`) that handles low-sample instability gracefully. The 10-sample floor was conservative, not mathematical.

**Skip-reason surfacing:** Augmented `run_one_class_cv` to collect `fold_errors` and return `skip_reason` in the result dict (`contamination.py:540-610`). The Bayesian objective now stores this as `trial.set_user_attr('skip_reason', ...)` when a trial bails (`unified_bayesian.py:1003-1009`), and the progress wrapper surfaces it in the GUI as `"Trial N/20 - SKIPPED (<reason>)"` instead of the previous useless `-inf` (`unified_bayesian.py:1810-1821`). Future silent-drop failure modes will now be visible.

**Verified:** Direct unit test with user's exact scenario (7 inliers, 30 features, 5-fold CV):
- 5 samples: fit succeeds (previously failed)
- 3 samples: fit succeeds (new floor)
- 2 samples: correctly rejected with new error message
- `run_one_class_cv` with 7 inliers / 5-fold: **all 5 folds succeed**, BalancedAcc=0.60, sensitivity=1.00
- 46 existing contamination tests still pass

**Follow-up requested by user (unclaimed):** Add Leave-One-Out CV as an optional cv_strategy for all task types. LOO uses every sample and eliminates the fold-too-small failure mode entirely for tiny datasets — noted in PROJECT_STATUS.md Follow-Ups section.

---

## 2026-04-09 — Pre-Merge Review Fixes (Claude Opus 4.6)

### Bug: Bayesian one-class results show NaN on Results tab
**Root cause:** `unified_bayesian.py:1000` explicitly set `compute_calibration=False`. Cal metrics never computed → trial user_attrs empty → DataFrame columns all NaN. Model Development worked because it recomputes metrics directly from trained model.
**Fix:** Changed to `compute_calibration=True`. Adds ~2x per trial but Bayesian runs 50-100 trials vs grid's thousands.

### Bug: Scaler/PCA not persisted in grid search result dicts
**Root cause (found by Codex gpt-5.4):** `run_one_class_search()` result dicts never stored `cal_scaler`, `cal_pca_reducer`, or `oc_score_stats` from `run_one_class_cv()`. Saved OneClassSVM/LOF/EllipticEnvelope models would predict on raw features instead of scaled space. Also caused batch-dependent uncertainty thresholds.
**Fix:** Added `scaler`, `pca_reducer`, `oc_score_stats` to both result dict blocks (full-spectrum and varsel).

### Bug: LightGBM label encoding in importance computation
**Root cause:** `compute_one_class_importances()` passed +1/-1 labels to `LGBMClassifier` which expects 0/1. Auto-remapped but semantically ambiguous and may fail in strict builds.
**Fix:** Convert `y_oc` to binary before fit: `(y_oc == -1).astype(int)`.

### Performance: Reduced IsolationForest/LOF grid bloat
- IF: replaced `n_estimators=200/300` entries with `n_estimators=100` + `max_features` variants (~50% faster)
- LOF: removed 2 redundant configs that only varied `contamination` with identical `n_neighbors=20` (~40% faster)
- Discovery: skipped unused importance computation during initial scan (~17% faster)

---

## 2026-04-09 — One-Class Bug Fixes (Claude Opus 4.6)

### Bug: One-class models "hang" on Model Development page
**Root cause:** `_run_refined_model_thread()` one-class path returns at ~line 34075 and ~line 33997, bypassing `_update_refined_results()` which re-enables buttons, resets cursor, enables save. Not a real hang — the model completes but the UI never updates.
**Fix:** Call `_update_refined_results(results_text)` before returning.

### Bug: Dancing man animation never stops for one-class
**Root cause:** `_run_analysis_thread()` one-class path returns at ~line 25087, bypassing cleanup code at lines 25808-25822 (stop animation, play chime, reset buttons). Same pattern as Model Dev — early return skips shared cleanup.
**Fix:** Added cleanup calls before all 4 one-class early returns. Also fixed 2 pre-existing issues: `_update_search_buttons('stopped')` (invalid state, should be `'idle'`) and non-one-class Bayesian missing-module error path.

### Bug: External validation checkbox unchecks for one-class
**Root cause:** One-class early return skips `self.last_training_config` storage (line 25716). When user double-clicks a result to load in Model Dev, `_load_model_for_refinement` sees no training config → clears `validation_enabled` (line 30349).
**Fix:** Store `last_training_config` before the one-class early return.

### Bug: No validation metrics for one-class
**Root cause:** One-class early return at ~line 25099 exits before the validation computation block at ~line 25256. `run_one_class_search` also has no validation parameters. Validation metrics simply never computed.
**Fix:** Added validation computation in both the Results path (rebuild full pipeline per result row) and Model Development path (use calibration model's scaler/PCA).

### Bug: Grid search inlier label type mismatch
**Root cause:** `search.py` line 4796 does `y_np == inlier_class_label` without string conversion. Bayesian path (line 831-832) converts both to strings. If y has strings and label is numeric (or vice versa), all samples marked as outliers.
**Fix:** Added string conversion to match Bayesian path.

### Bug: SHAP hangs for one-class models
**Root cause:** KernelExplainer is O(n_features × n_samples × n_background) model evaluations. With spectral data (hundreds of wavelengths), this takes hours, not seconds.
**Fix:** Replaced with permutation importance (fast, model-agnostic). TreeExplainer used only for IsolationForest without scaler/PCA.

### Bug: Wavelength importance silent failure for one-class
**Root cause:** `get_feature_importances()` in `models.py` raises ValueError for one-class model names. Exception caught silently at line 31393 — no plot, no error message.
**Fix:** Added one-class branch in `_plot_wavelength_importance()` that calls existing `compute_one_class_importances()` from `contamination.py`.

### Bug: Preprocessing discovery callback mismatch
**Root cause:** `discover_preprocessing` calls `progress_callback(current, total, msg)` with 3 positional args. GUI's `_progress_callback` expects a dict. Regular grid search had a wrapper (line 1204); one-class grid search passed the raw callback.
**Fix:** Added same wrapper pattern for one-class.

### Bug: Residual correlation zeroed out for ALL models
**Root cause:** Indentation error — `self.wavelength_residual_corr_data = np.zeros(len(wavelengths))` at line 31453 was at the `else` block level, not inside the `except` block. Always overwrote the successfully computed correlation.
**Fix:** Indented into the `except` block.

### Architecture insight: One-class early returns are the #1 bug pattern
The one-class pipeline was added as an early-exit branch (`if task_type == "one_class": ... return`) in both `_run_analysis_thread` and `_run_refined_model_thread`. Every piece of shared cleanup code after the return is skipped. There are 6+ locations where this caused bugs. When adding any new post-analysis feature, check that the one-class path also reaches it.

### Architecture insight: Preprocessing discovery dropdown is cosmetic for one-class
All importance methods (model_specific, lightgbm, cars_tree, vip) resolve to LightGBM for one-class. The per-model refinement at line 935 of `preprocessing_discovery.py` never triggers because `models_to_test` isn't passed from `run_one_class_search`.

### Performance: First-time one-class preprocessing discovery is slower than classification
User reported ~20-100x slower. Likely because one-class has fewer outliers → stratified CV with very small minority class → more variance → potentially more work. Needs profiling. Applies to grid search, not just discovery.

### Bug: Validation metrics fail with "Unknown preprocess: None"
**Root cause:** Grid search results use column names `'Preprocess'`/`'PreprocessBase'`/`'Deriv'`/`'Window'`/`'Poly'`. Validation code used `'Preprocessing'`/`'Derivative'` — wrong column names, so all values were None.
**Fix:** Use `row.get('PreprocessBase', row.get('Preprocess', row.get('preprocessing', 'raw')))` chain that handles both grid search and Bayesian column naming.

---

## 2026-04-11 — CV Strategy Overhaul (LOO + Repeated K-Fold)

**Branch:** claude/cv-strategy-overhaul

### Changes
- Added `build_cv_splitter()` factory in `cv_utils.py` supporting kfold, repeated_kfold, loo
- Fixed RMSEcv to use pooled predictions (was per-fold average; K-fold shift: ~1-4% on equal folds, up to ~12% on unequal folds; LOO was silently reporting MAE)
- Fixed one-class CV metrics to use pooled computation (averaging-of-ratios != ratio-of-sums; K-fold shift: 0-3% typically)
- Propagated cv_strategy through grid search, Bayesian search, and one-class search
- Added CV strategy combobox + conditional controls to Analysis tab + Model Development
- Refinement path fully wired (restore, validation, thread dispatch, sample-count guards, one-class)
- Cost estimator + LOO/Repeated warnings + stochastic model warning
- NSGA-II fallback: logs warning and uses K-fold when non-kfold strategy selected
- Variable selection uses mixed regime (inner 5-fold K-fold regardless of outer strategy) with prominent GUI warning

### Design decisions
- **Option C for variable selection:** inner loops stay 5-fold K-fold, visible GUI warning when outer strategy != kfold. Phase 2 will propagate strategy into inner loops.
- **training_config schema:** `cv_strategy` is source of truth; `folds` stored as integer (n_samples for LOO) for cross-machine backwards compatibility.
- **Pooled metrics:** both RMSEcv (regression) and sensitivity/specificity/etc (one-class) now computed from pooled predictions, matching Unscrambler/IUPAC convention. This is a numerical shift for existing K-fold results -- intentional correctness fix.

### Known limitations
- LOO computationally infeasible for n > ~50 with full Bayesian search matrix
- Variable selection inner CV doesn't respect outer strategy (Phase 2)
- NSGA-II search always falls back to K-fold
- Predictor screening / smart preprocessing internal CV sites left as hardcoded 5-fold

---

## 2026-04-12 — CV Strategy Bugfixes (Qwen Max + Codex review)

**Branch:** claude/cv-strategy-overhaul

### Critical bug found by Codex: RepeatedKFold + cross_val_predict crash
sklearn's `cross_val_predict` raises `ValueError: cross_val_predict only works for partitions` when passed `RepeatedKFold` (overlapping test sets). Affected all Bayesian search trials using repeated K-fold. Grid search was unaffected (uses own manual fold loop).

**Fix:** Added `cross_val_predict_pooled()` in `cv_utils.py` — for repeated CV, runs a manual loop that accumulates predictions per sample and averages across repeats. Updated `cross_val_predict_with_early_stopping` with the same accumulation logic. Replaced all `cross_val_predict` calls in `unified_bayesian.py` with `cross_val_predict_pooled`.

### High bug found by Codex: One-class Bayesian ignored cv_strategy
`unified_bayesian.py:1000` called `run_one_class_cv(...)` without forwarding `cv_strategy`/`cv_n_repeats`, so one-class Bayesian always used default K-fold regardless of GUI selection. Fixed by threading both params through.

### High bug found by Codex: GUI last_training_config wrong folds under LOO
Two sites in GUI stored `self.folds.get()` (spinbox value, typically 5) instead of computing effective folds (`len(X)` for LOO). This overwrote the correct per-result training_config from the search layer. Fixed both sites.

### Other fixes
- Differentiated mixed-regime warning text for LOO vs Repeated K-Fold
- Added LOO + classification minority-class preflight guard (blocks LOO when any class has < 2 samples)
- Removed unnecessary `.copy()` calls in early-stopping CV helper (saves memory per fold)
- Added `training_config` to one-class grid search results (was missing, unlike regression/classification)
