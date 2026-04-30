# Model-Native Loss Reweighting for Class Imbalance

**Status:** Proposal — superseded in scope by REVISIONS section below (post-Codex review)
**Date:** 2026-04-29 (original); revised same day after Codex code-verification review
**Origin:** Session analyzing the FTIR Bone PLS paper revealed dasp does not expose model-native imbalance handling for the boosting models (XGBoost / LightGBM / CatBoost), nor for SVC's gamma-C combination, nor for elastic-net logistic regression (which itself isn't a model card yet — see `2026-04-29-enlr-and-lda-on-pls-models.md` if/when that doc is written).

> **READ THIS FIRST:** the REVISIONS section directly below supersedes specific claims in the original body. When the REVISIONS section conflicts with the body, REVISIONS wins. Implementation reference is `docs/RECONCILED_ROADMAP_2026-04-29.md` T-19.

---

## REVISIONS — 2026-04-29 (post-Codex code-verification review)

After this design doc was first written, Codex performed a code-verification review against the actual `src/spectral_predict/` codebase. Six points in the original body are superseded.

### R1. PLS-DA inner LR scope is broader than originally documented

The original body's "Implementation surface" section claims PLS-DA inner LR sites are `search.py:380`, `:4261`, and `bayesian_utils.py:400`, all stuck at `class_weight=None`. **The actual codebase has ~10 PLS-DA inner LR construction sites.** Codex's audit:

| Site | Status |
|------|--------|
| `search.py:4257-4261` | ✅ already honors `imbalance_method='class_weight'` |
| `unified_bayesian.py:1219-1222` | ✅ already honors |
| `nsga2_search.py:1417-1426` | ✅ already honors |
| `nsga2_search.py:3077-3080` | ✅ already honors |
| `code_generator.py:880-884, 1248` | ✅ injects `class_weight='balanced'` in exported code |
| `search.py:380` | ❌ silent default — fix |
| `bayesian_utils.py:400` | ❌ silent default — fix |
| `nsga2_search.py:822` | ❌ silent default — fix |
| `nsga2_search.py:3453` | ❌ silent default — fix |
| `ga_preprocessing.py:428` | ❌ silent default — fix |

So **5 sites** need wiring (auxiliary paths like rebuild-from-row and importance-fit utilities), not 2.

### R2. The original "sample_weight via fit_params" approach is broken when a resampler is in the Pipeline

The original Auto-mode implementation note #4 prescribes: "Auto mode for multiclass should fall back to `sample_weight=compute_sample_weight('balanced', y_train)` regardless of which boosting model is used. This works for all three boosting libraries via their `fit(..., sample_weight=...)` interface."

**This is mechanically correct ONLY if no resampler is in the Pipeline.** Codex verified that imblearn's `Pipeline._fit()` does NOT rewrite or recompute `fit_params` after a resampler step (`imblearn/pipeline.py:435-455`; helper returns only `X_res, y_res, sampler` at `pipeline.py:1330-1334`). If you precompute `sample_weight` from pre-resampling y and pass it via `fit_params`, the resampler will change y length and the `sample_weight` array will mismatch.

**The right approach is a custom estimator wrapper** (originally dismissed in implementation note #1 as "more complex"; now the recommended path). The wrapper intercepts `fit(X, y)` inside the Pipeline and computes `sample_weight = compute_sample_weight('balanced', y)` from the y the wrapper RECEIVES (post-resampling). The wrapper then calls `inner.fit(X, y, sample_weight=sample_weight)`. This bypasses imblearn's fit_params problem and is automatically resampling-aware.

**Where each library lands:**

| Library | Mechanism | Resampling-aware? |
|---------|-----------|-------------------|
| sklearn LR / RF / SVC | `class_weight='balanced'` | ✅ lazy at fit |
| LightGBM | `class_weight='balanced'` | ✅ lazy at fit (verified `lightgbm/sklearn.py:969-976`) |
| CatBoost | `auto_class_weights='Balanced'` / `'SqrtBalanced'` | ✅ computed on train pool (verified `catboost/core.py:5020-5023`) |
| XGBoost binary | `scale_pos_weight` | ❌ frozen at __init__ — wrapper needed under resampling |
| XGBoost multiclass / MLP | `sample_weight` at fit | ⚠ wrapper needed (cannot route through imblearn fit_params) |

### R3. MLP sample_weight requires sklearn ≥1.7 — and bundle compatibility is the gating question

The original body says "sklearn MLP doesn't accept `class_weight`; reweighting requires routing per-sample weights through `Pipeline.fit(..., classifier__sample_weight=w)`." The first half is correct (no `class_weight` kwarg). The second half is **doubly wrong**: (a) `MLPClassifier.fit()` only accepts `sample_weight` as of sklearn 1.7 (`sklearn/neural_network/_multilayer_perceptron.py:842-845`), and (b) routing through imblearn's `fit_params` is broken anyway (see R2).

The current `pyproject.toml:29` floor is `>=1.5.0`. **Decision required before bumping the floor:** test whether sklearn ≥1.7 builds cleanly in the PyInstaller bundle (`spectral_predict_py312.spec`). If incompatible, **skip MLP from the imbalance dropdown** rather than block T-19 on a sklearn version bump — MLP is among this user's least-used models, so omitting balanced loss reweighting for MLP specifically is an acceptable tradeoff. The user's local venv is sklearn 1.8, so the design works locally regardless.

When MLP is included, it uses the same custom wrapper approach as XGBoost multiclass — wrapper intercepts fit(), computes sample_weight from received y, calls `inner.fit(X, y, sample_weight=sw)`.

### R4. Active bug found while verifying T-19 (separate ticket T-32 in the roadmap)

`search.py:3883` in the existing resampling path computes `sample_weight_train` from `y_train` (pre-resampling) but passes it to `final_model.fit(X_train_transformed, y_train, sample_weight=sample_weight_train)` against `y_train` (rather than `y_train_for_model`, the resampled y). When a resampler runs, this is a length-mismatch bug TODAY — not a hypothetical risk. Tracked as **T-32** in the roadmap.

### R5. Stacking ensemble does not route sample_weight to base models

`StackingEnsemble.fit()` (`ensemble.py:796-825`) calls `fold_model.fit(X_train_proc, y_train)` without `sample_weight`. Any auto-mode sample-weight in the outer pipeline will NOT propagate into stacking-internal models. **Document as a known limitation.** Users wanting balanced loss reweighting in stacked models should pick `class_weight`-style modes for base models that support them (sklearn / LightGBM / CatBoost native kwargs work; XGBoost binary scale_pos_weight is frozen as discussed in R2).

### R6. Effort estimate revised

Original: "1-2 days." Revised: **5-7 days** after counting:
- 5 PLS-DA sites needing wiring
- Custom wrapper design + tests
- sklearn floor bump compatibility check (or MLP skip)
- Tests for resampling × balanced paths
- Code export updates (boosting kwargs + wrapper-based sample_weight path)
- GUI dropdown + tooltip + per-fold logging

The original body of this doc is preserved below for context but should not be implemented as written. Read sections 1-6 of REVISIONS as authoritative.

---

## Problem

Today dasp handles class imbalance through exactly two routes:

1. **Resampling** via the Data Quality tab — SMOTE, ADASYN, SMOTEENN, random over/undersampler. These mutate the training data (synthesize new minority samples or discard majority samples).
2. **`class_weight='balanced'` injection** via the imbalance system — but per `code_generator.py:866`, this is only applied to `RandomForest`, `LogisticRegression`, and `SVC`. The boosting model construction in `models.py` (XGBoost at line 285, LightGBM at 307, CatBoost at 327) never receives any imbalance-aware kwarg.

Verified: `scale_pos_weight`, `is_unbalance`, and `auto_class_weights` appear nowhere in `src/`. Zero matches via grep.

This means a chemometrics user trying to follow standard practice — e.g., the FTIR Bone PLS paper's "XGBoost (scale_pos_weight)", "LightGBM (balanced)", "Logistic regression (EN, balanced)" configurations — cannot reproduce them in dasp without either editing source or routing through SMOTE-style resampling. **Resampling is a different statistical intervention from loss reweighting**, and the chemometrics literature defaults to loss reweighting for small spectroscopic datasets where SMOTE can synthesize unrealistic interpolated spectra and undersampling discards real measurements.

## The "nothing vs balanced" misconception

Worth stating explicitly because it's the most common point of confusion:

**Default `class_weight=None` (or omitting `scale_pos_weight` for boosting models) does NOT mean "assume the data is balanced."** It means *all samples receive equal weight in the loss function*. With 100 positives and 10 negatives, the loss is dominated 10:1 by positives. The model learns to predict the majority class well and the minority class poorly. This is the opposite of what most users intuitively expect from the default.

**`class_weight='balanced'` (or the per-library equivalent) is what corrects for imbalance.** It computes weights so each class contributes equally to the loss regardless of sample count.

The GUI labelling and tooltips for this feature must make that distinction clear. Suggested radio labels:

- ❌ Bad: `none` vs `balanced` — implies "none" might mean balanced
- ✅ Good: `Equal sample weights (no correction)` vs `Class-balanced loss reweighting`

## Native imbalance kwargs by model

| Model | Native kwarg(s) | Mechanism |
|---|---|---|
| **sklearn LogisticRegression** | `class_weight='balanced'` | Per-class weight `n_samples / (n_classes * n_samples_in_class)` applied to loss |
| **sklearn SVC** | `class_weight='balanced'` | Same formula, applied to dual-form C |
| **sklearn RandomForestClassifier** | `class_weight='balanced'` or `'balanced_subsample'` | Weights computed per-bootstrap for `'balanced_subsample'` |
| **XGBoost** (`XGBClassifier`) | `scale_pos_weight=N` (binary only) | Multiplies positive-class loss by N. Standard recipe: `N = n_neg / n_pos`. Multiclass requires `sample_weight` at fit time instead |
| **LightGBM** (`LGBMClassifier`) | `class_weight='balanced'`, `is_unbalance=True`, or `scale_pos_weight=N` | All three exist; `class_weight` is most general (works for multiclass), `is_unbalance` is a binary boolean shortcut, `scale_pos_weight` mirrors XGBoost |
| **CatBoost** (`CatBoostClassifier`) | `auto_class_weights='Balanced'` or `'SqrtBalanced'`; or explicit `class_weights=[...]`; or `scale_pos_weight=N` | `'Balanced'` matches sklearn formula; `'SqrtBalanced'` is a softer correction (sqrt of class frequencies) |
| **MLPClassifier** | None natively — needs `sample_weight` at fit time | sklearn MLP doesn't accept `class_weight`; reweighting requires routing per-sample weights through `Pipeline.fit(..., classifier__sample_weight=w)` |

## Proposed UI

A per-model **Imbalance handling** dropdown in each classifier's hyperparameter card on the Model Config tab. Options:

1. **`Equal sample weights (no correction)`** — default; current behavior. Tooltip: "All samples contribute equally to the loss. With imbalanced classes, the loss is dominated by the majority class and the model learns the minority poorly. This is the library default but is rarely the right choice for imbalanced data."

2. **`Auto (decide at fit time)`** — call `detect_class_imbalance(y, threshold=3.0)` inside the pipeline's `fit`. If severity ≥ moderate (ratio ≥ 3:1), apply class-balanced loss reweighting using the model's native kwarg. Otherwise apply nothing. Log/report the decision. Tooltip: "At fit time, checks the class ratio in the training data. If imbalanced (ratio ≥ 3:1), applies class-balanced loss reweighting; otherwise leaves the loss unweighted. The decision is made per CV fold on each fold's training subset, so it is correct even when global ratios drift from per-fold ratios."

3. **`Class-balanced loss reweighting`** — always apply the model-native balanced-loss kwarg regardless of observed ratio. Tooltip: "Reweights the loss so each class contributes equally regardless of sample count. Standard chemometrics practice for imbalanced classification. Maps to `class_weight='balanced'` (sklearn / LightGBM), `scale_pos_weight=n_neg/n_pos` (XGBoost binary), or `auto_class_weights='Balanced'` (CatBoost)."

4. **`scale_pos_weight (custom)`** — XGBoost / LightGBM / CatBoost binary only. Numeric entry. Greys out for multiclass tasks. Tooltip: "Manually set the positive-class loss multiplier. The class-balanced default is `n_negative / n_positive`. Use higher values to penalize false negatives more heavily; use lower values when you want to be more conservative about positive predictions."

**Default = option 1 (`Equal sample weights`)** for backward compatibility. Existing saved configurations and reproducibility-critical workflows are unchanged.

## Auto-mode implementation notes

Auto mode is the most attractive option for typical users and has subtle correctness requirements:

1. **Per-fold computation.** `class_weight='balanced'` (sklearn / LightGBM) computes weights at `fit` time on whatever data is passed in, so it's automatically per-CV-fold correct. `scale_pos_weight` is a constructor argument and *cannot* be made per-fold-aware unless we either (a) wrap the estimator in a custom transformer that recomputes it inside `fit`, or (b) restrict auto mode to use only the `class_weight`-style kwargs (LightGBM `class_weight='balanced'`, CatBoost `auto_class_weights='Balanced'`, XGBoost via `sample_weight=compute_sample_weight('balanced', y_train)` passed at fit time).

   **Recommendation:** option (b). It's simpler, has no custom wrapper to maintain, and gives identical statistical behavior. The user-visible distinction between auto-mode and explicit-balanced-mode then becomes "auto = check ratio first, only apply if imbalanced; balanced = always apply."

2. **Severity threshold.** `detect_class_imbalance` defaults to `threshold=3.0`. That's the right starting point — anything below 3:1 is statistically modest enough that loss reweighting often makes little difference. Expose this as an advanced setting, but don't put it in the main UI.

3. **Logging the decision.** Auto mode must report what it decided per fold (or at least per run). Surface this in the Results tab as an "Imbalance handling: auto → balanced (ratio 4.2:1)" annotation alongside other training config. Without this, "auto" is a black box and the user can't audit what actually happened.

4. **Multiclass handling.** `scale_pos_weight` is binary-only for XGBoost and LightGBM. Auto mode for multiclass should fall back to `sample_weight=compute_sample_weight('balanced', y_train)` regardless of which boosting model is used. This works for all three boosting libraries via their `fit(..., sample_weight=...)` interface.

5. **Interaction with the Data Quality resampling pipeline.** Loss reweighting and resampling are different statistical interventions. Stacking them is not invalid but is rarely intended. When the user enables both, show a non-blocking banner: *"Heads up — you have both resampling (SMOTE/etc.) on the Data Quality tab AND class-balanced loss reweighting on this model. These are different interventions and stacking them may double-correct for imbalance."*

## Implementation surface

Files that need changes:

- **`src/spectral_predict/models.py`** — extend each classifier-construction site to accept and pass through the imbalance kwarg. Approximately:
  - `XGBClassifier(...)` at lines 285, 506, 1573 — add `scale_pos_weight` (binary) or route `sample_weight` at fit (multiclass)
  - `LGBMClassifier(...)` at line 307 — add `class_weight`, `is_unbalance`, or `scale_pos_weight`
  - `CatBoostClassifier(...)` at line 327 — add `auto_class_weights` or `class_weights` or `scale_pos_weight`
  - `SVC(...)` at 282 — already supports `class_weight` but currently isn't wired to the per-model card; just exposure work
  - PLS-DA's inner `LogisticRegression` at `search.py:380`, `:4261`, and `bayesian_utils.py:400` — currently always uses default `class_weight=None`; needs to honor the PLS-DA card's imbalance setting
  - `MLPClassifier` — needs sample-weight routing through Pipeline (`fit_params` mechanism)

- **`spectral_predict_gui_optimized.py`** — add the per-model "Imbalance handling" dropdown to each classifier's hyperparameter card on the Model Config tab. Tooltips must distinguish "Equal sample weights" from "Auto" from "Balanced" using the language above.

- **`src/spectral_predict/imbalance.py`** — add a small helper `resolve_loss_reweighting(model_name, y_train, mode)` that returns `{'class_weight': ...}`, `{'scale_pos_weight': N}`, `{'auto_class_weights': ...}`, or `{'sample_weight': array}` as appropriate per model and per task (binary vs multiclass) given the user's chosen mode (`none`, `auto`, `balanced`, `custom`). Centralizes the per-library kwarg mapping in one place.

- **`src/spectral_predict/code_generator.py`** — extend `_render_model_instantiation` (around line 860) so exported scripts/notebooks include the chosen loss-reweighting kwarg. Currently only handles `class_weight='balanced'` for `RandomForest`/`LogisticRegression`/`SVC`; needs the boosting-model variants too.

- **Bayesian search paths** (`unified_bayesian.py`, `bayesian_utils.py`) — pass the user's loss-reweighting choice through to model construction. Do NOT search over imbalance handling as a hyperparameter — it's a methodological choice, not a tunable. (A future extension could include it as a search dimension for power users, but that's out of scope here.)

- **Tests** — `tests/test_imbalance_loss_reweighting.py` (new file). Cases:
  - Auto mode applies balanced loss when ratio > 3:1, leaves it alone when < 3:1
  - Auto mode is per-fold correct (CV with class drift across folds)
  - `scale_pos_weight` for XGBoost binary, sample_weight fallback for multiclass
  - LightGBM `class_weight='balanced'` ≡ XGBoost `scale_pos_weight=n_neg/n_pos` on identical synthetic binary data (within reasonable tolerance — they're not exactly equivalent algorithms, but should give similar headline metrics)
  - Saved config round-trips: load a model trained with auto mode, predictions match
  - Code export includes the right kwarg per model

## Out of scope (future follow-ups)

- **Regression-target imbalance.** `detect_regression_imbalance` exists in `imbalance.py:133`, but routing per-bin sample weights into regression models is a separate design problem with different ergonomics. Worth its own doc.
- **Cost-sensitive learning beyond balanced**. Asymmetric misclassification costs (false-positive vs false-negative) are a generalization of `scale_pos_weight` but require a different UI affordance — leave for later.
- **One-class models.** Inlier-vs-outlier imbalance is the *defining* property of one-class methods; the existing `class_weight='balanced'` plumbing in `contamination.py:1344` handles the surrogate LightGBM importance computation. No changes needed in this scope.

## Risk and rollout

Low risk because:

- Default = `Equal sample weights` preserves all existing behavior bit-for-bit. No saved configs change behavior.
- Each model's kwarg is well-documented in its respective library and has stable behavior across versions.
- The auto-mode threshold and the per-fold timing are isolated to a single helper (`resolve_loss_reweighting`), so any future refinement is localized.

Medium-effort estimate: **~1-2 days of focused work** for backend + GUI exposure + tests + code-export support. Could be split into a backend-only PR first (helpers + per-model kwarg plumbing + tests) and a GUI-exposure PR second.
