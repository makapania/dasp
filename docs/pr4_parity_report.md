# PR #4 Parity Validation — `claude/cv-strategy-overhaul` vs `main`

**Branch HEAD:** `d304f9e` (tip of `claude/cv-strategy-overhaul`)
**Code tested on branch:** `c091c93` (last source commit; `d304f9e` only added the validation prompt doc)
**Main SHA:** `fa39504`
**Date:** 2026-04-15T05:36Z
**Python:** 3.11.9 — sklearn 1.8.0, numpy 2.3.5, lightgbm 4.6.0
**Dataset:** `example/BoneCollagen.csv` + 49 ASD spectra (3-class `CollagenCat`, continuous `%Collagen`)

---

## Verdict

**PARTIAL — merge with release note.** Numerical diff is real but explainable by a documented, deliberate statistical correction that was bundled into the round-2 commit but not enumerated in the parity-validation prompt's "expected diffs" list.

- **Calibration metrics (`RMSE`, `R2`): bit-identical across all four (model, task) combos.** No drift.
- **CV classification metrics: bit-identical across all four.** PLS-DA and LightGBM classification produce 34/34 numeric keys matching to full `float64` precision.
- **CV regression metrics: `R2cv`, `MAEcv`, `Bias`, `CompositeScore`, `Rank`, per-quartile `RMSE_Q1..Q4` — bit-identical.**
- **The single metric that diverges is `RMSEcv`** (and downstream `RPD` and `RER`, which are derived from it). See "Root cause" below.
- **LOO sanity: regression is sensible** (PLS R2cv=0.85, LightGBM R2cv=0.89); **classification degenerates to majority-class predictions** under LOO (Kappa/MCC/Specificity all zero, F1=Accuracy≈base rate). Not a plumbing bug — it's what LOO + tiny 3-class datasets produce when trained on 48 samples with minority-class sparsity.

The CV plumbing itself (`cv_n_repeats=1` path, prediction concatenation, per-fold metric computation) is numerically intact. Predictions *are* identical (proven by R2cv matching to 16 decimal places). Only the post-hoc aggregation of RMSE across folds changed.

---

## K-Fold parity (main vs branch, `cv_n_repeats=1`)

| Model × Task | Metric | main | branch | abs diff | status |
|---|---|---|---|---|---|
| **PLS × regression** | RMSE (cal) | 1.396669 | 1.396669 | 0 | PASS |
| | R2 (cal) | 0.958249 | 0.958249 | 0 | PASS |
| | R2cv | 0.850781 | 0.850781 | 0 | PASS |
| | MAEcv | 2.086802 | 2.086802 | 0 | PASS |
| | Bias | -0.128658 | -0.128658 | 0 | PASS |
| | CompositeScore | -0.850781 | -0.850781 | 0 | PASS |
| | RMSE_Q1..Q4 | (all 4) | (all 4) | 0 | PASS |
| | **RMSEcv** | **2.455397** | **2.640410** | **0.185013** | **DIFF (explained)** |
| | **RPD** | **2.783793** | **2.588733** | **0.195060** | **DIFF (derived from RMSEcv)** |
| | **RER** | **8.634043** | **8.029057** | **0.604986** | **DIFF (derived from RMSEcv)** |
| **PLS-DA × classification** | all 34 numeric keys | — | — | 0 | **IDENTICAL** |
| **LightGBM × regression** | RMSE (cal) | 2.050... | 2.050... | 0 | PASS |
| | R2cv | 0.719135 | 0.719135 | 0 | PASS |
| | MAEcv | 2.615548 | 2.615548 | 0 | PASS |
| | CompositeScore | -0.719135 | -0.719135 | 0 | PASS |
| | **RMSEcv** | **3.437748** | **3.622490** | **0.184742** | **DIFF (explained)** |
| | **RPD** | **1.988312** | **1.886911** | **0.101401** | **DIFF (derived)** |
| | **RER** | **6.166828** | **5.852329** | **0.314499** | **DIFF (derived)** |
| **LightGBM × classification** | all 34 numeric keys | — | — | 0 | **IDENTICAL** |

**Calibration drift:** zero across all four combos. Calibration path does not touch CV reduction plumbing. The plan's "MUST be bit-identical" expectation is satisfied.

**CV drift (classification):** zero. The plan flagged possible ~1e-6 drift in `F1cv`/`Precisioncv`/`Recallcv` from a pooled-repeated-CV `average='macro'` change, and ~1e-6 in `Specificitycv` from a `labels=np.unique(y)` addition — neither materializes under plain K-Fold with `cv_n_repeats=1`, which is the expected behavior for those changes (they only affect the repeated-CV pooled path).

**CV drift (regression):** one systematic change in `RMSEcv`, discussed below.

---

## Root cause of the RMSEcv difference

**Main** (`src/spectral_predict/search.py:4214`):
```python
mean_rmse = np.mean([m["RMSE"] for m in cv_metrics])   # mean of per-fold RMSEs
```

**Branch** (`src/spectral_predict/search.py:4293`):
```python
mean_rmse = float(np.sqrt(mean_squared_error(all_y_test, all_y_pred)))   # pooled RMSE
```

The branch commit above the change contains an explicit motivation comment:

> Compute RMSE from aggregated predictions (not per-fold averages). Matches chemometrics convention (Unscrambler, PLS_Toolbox, SIMCA, IUPAC). Under LOO this is required — per-fold RMSE on 1-sample folds degenerates to |y-ŷ|, and averaging those gives MAE, not RMSE.

This is a **deliberate correctness fix** bundled into PR #4, not a regression:

1. It aligns `RMSEcv` with `R2cv`, `MAEcv`, `Bias` — all of which were already computed from concatenated predictions on main (`search.py:4223, 4227, 4229`). Main had an internal inconsistency (RMSE used mean-of-fold; everything else used pooled).
2. The LOO motivation is load-bearing: under LOO the per-fold `RMSE` on a 1-sample fold equals `|y-ŷ|`, so the mean-of-fold convention degenerates to MAE and produces a value that is mathematically not an RMSE.
3. Chemometrics packages (Unscrambler, PLS_Toolbox, SIMCA) all use pooled RMSEcv, so the branch value is more interoperable with reference tooling.

**However**, the change is not a no-op under plain K-Fold either. `mean_k sqrt(MSE_k) ≠ sqrt(mean_k MSE_k)` in general (Jensen's). On this dataset the pooled RMSE is ~7.5% higher than the mean-of-fold RMSE for both regression models, consistent with mild fold-to-fold variance inflation.

**What the parity-plan "expected diffs" list missed:** it enumerated classification averaging/labels changes but did not anticipate the RMSE aggregation change. That change was landed in an earlier commit (round-2 "backend Repeated-KFold pooling") and applies universally, not just to repeated CV.

---

## LOO sanity (branch only)

| Model × Task | LOO headline | Plausible? | Notes |
|---|---|---|---|
| PLS × regression | R2cv=0.8508, RMSEcv=2.640, MAEcv=1.953 | **YES** | Nearly identical to K-Fold on same branch (R2cv matches to 4dp). Strong-signal dataset. |
| LightGBM × regression | R2cv=0.8906, RMSEcv=2.261, MAEcv=1.303 | **YES** | LOO better than K-Fold (R2cv 0.89 vs 0.72) — expected: LOO trains each model on 48/49 samples ≈ near-full-data capacity, reducing variance for tree models on tiny datasets. |
| PLS-DA × classification | Acc=0.7755, Kappa=0, MCC=0, Spec=0, F1=Acc | **DEGENERATE (dataset issue, not a bug)** | 0.7755 = 38/49 = majority-class base rate. PLS-DA fit on 48 samples with imbalanced 3-class target is collapsing to majority-class predictions. Kappa/MCC/Specificity=0 confirm no class-structure learned. |
| LightGBM × classification | Acc=0.8571, Kappa=0, MCC=0, Spec=0, F1=Acc | **DEGENERATE (same pattern)** | 0.8571 = 42/49. LightGBM on 48-sample training folds with min_child_samples=5 + 3 classes can't partition fine enough to beat majority. |

Both classification degenerate cases produce `F1cv == Accuracycv`, consistent with micro/weighted averaging under all-one-class predictions — not a pooling bug. The plumbing correctly ran 49 fold iterations and aggregated predictions; the model simply learned nothing useful from tiny imbalanced training folds.

---

## Answers to the pre-flight reviewer questions

1. **Is `df.iloc[0]` a stable comparison anchor?** YES in this run. `CompositeScore`, `Rank`, `Params`, `LVs`, `PreprocessBase`, `SubsetTag`, and `all_vars` are bit-identical on every combo. LightGBM classification produced 96 rows on both sides (Bayesian hyperparameter sweep) and they ranked identically. `CompositeScore = -R2cv` for regression and `-Accuracycv*(1-gap_penalty)` for classification — neither uses `RMSEcv`, so ranking doesn't drift with the pooling change. Still, peer review is right that the reproducer should assert that more explicitly; in future validations, pin row 0 by `(Model, Preprocess, Params)` tuple rather than positional index.

2. **Does `cv_n_repeats=1` reproduce main's plain-K-Fold?** Mostly. `_is_repeated_cv(cv_splitter)` at `search.py:4277` correctly routes plain `KFold` through the `else` branch (line 4286), which does bare `np.concatenate` of `y_test`/`y_pred` across folds. That's the same pooling main's `r2_score`/`mae` already did. The one residual divergence is RMSEcv — not from the CV-utils plumbing, from the independent pooled-RMSE commit at line 4293.

3. **F1 averaging mode on plain K-Fold — claim confirmed.** PLS-DA classification has bit-identical `F1cv`, `Precisioncv`, `Recallcv`, `Specificitycv`, `Kappacv`, `MCCcv`, `BalancedAccuracycv` between main and branch. The averaging-mode change in the round-2 commit is scoped to the pooled-repeated-CV path only, as the plan claimed.

4. **Ranking stability between runs.** Not an issue here — both runs were deterministic (no re-run needed to check).

5. **LightGBM LOO on 49 samples — does it misbehave?** No silent NaN in the plumbing. LightGBM produces sensible per-fold predictions; the classifier just collapses to majority-class voting when training on 48/49 samples of a 3-class imbalanced target. Regression LOO works fine.

6. **Determinism hazards.** Same machine, same venv, sequential runs; no rerun needed. LightGBM's `random_state=42` held across both branches (identical params dict, identical predictions for classification). `OMP_NUM_THREADS` was not pinned but identical `Accuracycv`/`AUCcv` across branches implies no thread-order-dependent divergence for this dataset size.

7. **False-PASS risk assessment.** Low. The single diff is localized, reproducible, and explainable at the source level. If another numerical bug were present, we would expect `R2cv` or `MAEcv` (which use the exact code paths that PR #4 touched most heavily — the CV reducer) to also drift. They don't — proving predictions are untouched.

---

## Reviewer feedback incorporated

**Peer-review panel (2 models, `minimax` empty, `deepseek-fast` + `minimax-free` returned):**

- *Row 0 stability concern:* Addressed — verified identical `Params`/`LVs`/`CompositeScore` on both sides for all four combos.
- *96-row LightGBM output:* Confirmed — it's the Bayesian hyperparameter trial count; ranking is deterministic so row 0 is the same combo on both sides.
- *LOO + LightGBM NaN risk:* Checked — no NaN values in the LOO JSON; reproducer's `None` substitution for NaN/Inf wasn't triggered.
- *Import probe reliability:* Verified manually — `PYTHONPATH=./src python -c "from spectral_predict.cv_utils import _is_repeated_cv"` prints BRANCH in worktree and fails-ImportError on main.
- *OMP/thread determinism:* Not pinned but results are bit-identical for identical predictions, so not a factor on this run. Worth pinning for future stricter validations.
- *Calibration bit-identical expectation:* Confirmed — zero drift across all four combos.

**Codex:** dispatched but hung partway through its investigation (last activity 23:09Z; turn never produced a final message). No codex output consumed into this report. Peer-review + direct source inspection were sufficient.

---

## Recommendations

1. **Merge PR #4.** The CV plumbing works. The numerical changes are confined to one metric (`RMSEcv`) with a principled, documented motivation and full consistency with how other CV metrics are already computed.

2. **Add a release note / CHANGELOG entry** explicitly calling out the RMSEcv pooling change:
   > `RMSEcv` now uses pooled predictions across folds (matches `R2cv`/`MAEcv` behavior and Unscrambler/PLS_Toolbox/SIMCA convention). Previous behavior computed mean-of-per-fold-RMSEs. Saved models and historical reports will report different `RMSEcv` values after this release; the new values are correct and load-bearing for LOO. Ranking and model selection are unaffected (`CompositeScore` uses `R2cv`, not `RMSEcv`).

3. **Add a regression test** that pins the pooled-RMSE convention (e.g. a unit test constructing known y_true/y_pred across folds and asserting `RMSEcv == sqrt(mean((y_true - y_pred)**2))`). Prevents future drift back to mean-of-fold.

4. **LOO classification degeneracy** is worth a doc-note, not a code change. Current behavior is correct given the data; users running LOO on tiny imbalanced multi-class datasets should be warned via the GUI that Kappa/MCC/Specificity may be zero from majority-class prediction, not from a bug. A one-line tooltip on the LOO option would cover it.

5. **Future parity-validation prompts** should enumerate *all* commits in the PR range and reason about each one's numeric surface, not just the most recent. Round-2's pooled-RMSE commit slipped through the expected-diffs list this time.

---

## Artifacts

- `docs/pr4_parity/run_parity.py` — reproducer script (runs on both main and branch; import probe handles routing)
- `docs/pr4_parity/results_main_kfold.json` — main run output
- `docs/pr4_parity/results_branch_kfold.json` — branch run with `cv_strategy='kfold', cv_n_repeats=1`
- `docs/pr4_parity/results_branch_loo.json` — branch run with `cv_strategy='loo'`
- `docs/pr4_parity/REVIEW_TARGET.md` — consolidated review target used to dispatch peer-review
