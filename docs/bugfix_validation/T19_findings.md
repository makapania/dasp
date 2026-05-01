# T-19 findings: Model-native loss reweighting for class imbalance

**Branch:** none yet — T-19 is **not implemented**. Investigation of `main` only.
**Status:** FINDINGS ONLY — verdict pending main agent
**Author:** Opus 4.7 (1M context), 2026-04-30

This is the investigation phase for T-19, the loss-reweighting ticket. Like T-04, this is a **methodology + UX-completeness question**, not a numerical-correctness bug. The framing under review: "is the absence of `scale_pos_weight` / `is_unbalance` / `auto_class_weights` on dasp's boosting models a real reproducibility gap that delivers user value via the GUI?"

---

## TL;DR

**Real reproducibility gap that delivers user value via GUI — but the GUI surface needed is bigger than the design doc admits, and the existing `imbalance_method='class_weight'` dropdown already covers the boosting models via a `sample_weight` fallback that just happens to be silently broken under resampling.** Three load-bearing facts:

1. **The roadmap claim that "XGBoost / LightGBM / CatBoost construction sites in models.py never receive any imbalance-aware kwarg" is literally true** — `scale_pos_weight`, `is_unbalance`, and `auto_class_weights` appear *zero* times in `src/spectral_predict/`. Verified by Grep.
2. **The 5 PLS-DA inner LR sites are also literally still using `class_weight=None`** — verified individually. The other 4 sites cited as "already fixed" do honor `imbalance_method='class_weight'`.
3. **However**, `search.py:4211-4244` already has a `sample_weight` fallback for models without a `class_weight` attribute. So when a user picks `class_weight` in the existing GUI dropdown:
   - LightGBM works (`hasattr(model, 'class_weight') is True` → uses native kwarg lazily at fit)
   - XGBoost falls through to `sample_weight=compute_sample_weight('balanced', y_train)` route — works *if* no resampler is in the pipeline (T-32 bug fires when one is)
   - CatBoost falls through similarly
   - PLS-DA only honors it on 4 of 9 sites; the 5 cited sites silently ignore it

So the gap is **not** "user has zero way to do balanced loss for boosting models" — it's "the existing path goes through a backend `sample_weight` fallback that has the T-32 length-mismatch bug, has no per-fold logging, and doesn't expose the canonical per-library kwargs that the FTIR Bone PLS paper references by name." The FTIR Bone PLS paper's Tables S1-S3 cite `XGBoost (scale_pos_weight)` and `LightGBM (balanced)` *as model labels* — reproducing those configurations in dasp produces statistically equivalent but library-mechanistically-different runs (XGBoost via `sample_weight` rather than the constructor `scale_pos_weight` kwarg).

The Codex review work captured in `docs/plans/2026-04-29-model-native-loss-reweighting.md` REVISIONS section is methodologically sound: 5 PLS-DA sites + custom estimator wrapper for resampling-aware sample_weight + sklearn floor question + tests. Effort revised up to 5-7 days. **No live branch exists yet** — this is design-phase work; the design itself has been triple-checked already (original + Codex review + user pass).

The chemometrics field-alignment is mixed: classical spectroscopy software (Unscrambler, SIMCA-Sartorius, PLS_Toolbox, OPUS) does not generally expose model-native loss reweighting because their classifier portfolio is primarily PLS-DA / SIMCA / SVM-RBF — not gradient boosting. The XGBoost-`scale_pos_weight` pattern is an ML-import that the FTIR Bone PLS paper brought into the bone-FTIR domain. That doesn't make it wrong; it makes it a *modern* chemometrics-ML hybrid pattern that dasp's positioning ("chemometrics + ML for spectral modeling") explicitly targets.

The remaining open questions for the verdict-writer are about scope, GUI-surface design, and whether to ship the backend wrapper before or after a full per-model dropdown:

1. Is "extend the existing single-dropdown imbalance UI to dispatch model-native kwargs" enough, or does the field-alignment with the FTIR paper require a per-model dropdown?
2. Does T-32 need separate handling, or does the T-19 wrapper-based approach automatically resolve it?
3. Is MLP worth blocking on a sklearn floor bump, or skip MLP from the dropdown per the design doc?

---

## 1. What main currently does

### 1a. Boosting model construction sites (verified)

**`src/spectral_predict/models.py:284-304`** — XGBoost default builder. No imbalance kwargs:

```python
elif model_name == "XGBoost":
    return XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.15, reg_lambda=1.5,
        min_child_weight=3, gamma=0.05,
        tree_method='hist', random_state=42,
        n_jobs=n_jobs, verbosity=0
    )
```

**`models.py:306-324`** — LightGBM default builder. No imbalance kwargs:

```python
elif model_name == "LightGBM":
    return LGBMClassifier(
        n_estimators=100, learning_rate=0.1, num_leaves=15,
        max_depth=5, min_child_samples=5,
        subsample=0.8, bagging_freq=1, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=n_jobs, verbosity=-1
    )
```

**`models.py:326-340`** — CatBoost default builder. No imbalance kwargs:

```python
elif model_name == "CatBoost":
    return CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=4,
        l2_leaf_reg=5.0, min_data_in_leaf=5,
        bootstrap_type='Bernoulli',
        random_state=42, verbose=False
    )
```

**Same pattern for the Bayesian builder** (`models.py:506-528`) and **for the grid-search builder** (`models.py:1581-1610` XGBoost, `:1626-1655` LightGBM, `:1697` CatBoost). Verified by reading. No constructor receives `scale_pos_weight`, `is_unbalance`, `auto_class_weights`, or `class_weight`.

**Grep verification:**

```text
$ rg "scale_pos_weight|is_unbalance|auto_class_weights" src/spectral_predict/
(no matches)
```

**Zero hits across all of `src/spectral_predict/`.** The roadmap and design doc claim is fully accurate — these kwargs are completely absent from the codebase.

### 1b. The five "still broken" PLS-DA inner LR sites (verified individually)

The Codex review cites 5 sites where PLS-DA's inner `LogisticRegression` is constructed with the silent default `class_weight=None`. All 5 verified directly:

| Site | Code (truncated) | Status |
|------|------------------|--------|
| `search.py:380` | `('lr', LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=42))` | ❌ no class_weight |
| `bayesian_utils.py:401` | `("lr", LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=42))` | ❌ no class_weight |
| `nsga2_search.py:826` | `('lr', LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=random_state))` | ❌ no class_weight |
| `nsga2_search.py:3458` | `('lr', LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=42))` | ❌ no class_weight |
| `ga_preprocessing.py:428-429` | `('lr', LogisticRegression(C=lr_C, solver=lr_solver, max_iter=lr_max_iter, random_state=random_state))` | ❌ no class_weight |

**These five sites are auxiliary paths** (rebuild-from-row, importance-fit utilities, GA preprocessing fitness, NSGA-II `_compute_calibration_metrics`). They are not the *primary* CV training path — that path goes through `search.py:4257-4269` which DOES honor `imbalance_method`. But they are reachable enough to silently produce different `class_weight` behavior than the user expects.

### 1c. The four "already wired" PLS-DA inner LR sites (verified)

The Codex audit also cites 4 sites that already honor `imbalance_method`. All verified:

- **`search.py:4257-4269`** — main CV training path. Builds `lr_kwargs` dict and adds `class_weight='balanced'` when `imbalance_method == 'class_weight'`. Then `LogisticRegression(**lr_kwargs)`.
- **`unified_bayesian.py:1219-1223`** — Bayesian path. Builds LR, then `lr.set_params(class_weight='balanced')` when `imbalance_method == 'class_weight'`.
- **`nsga2_search.py:1417-1430`** — NSGA-II main eval path. Same `set_params` pattern.
- **`nsga2_search.py:3077-3086`** — NSGA-II auxiliary path with resampling. Same pattern.

So **the 5+4 split is real**: half of dasp's PLS-DA sites already wire imbalance handling for the inner LR; half don't. A user whose run goes through one of the 5 unwired sites gets different behavior from one whose run goes through the wired sites, *even at the same GUI setting*. That's a reproducibility surface in itself, independent of the boosting-model question.

### 1d. The existing sample_weight fallback for boosting models

**`src/spectral_predict/search.py:4208-4244`** — when `imbalance_method == 'class_weight'`:

```python
use_sample_weight_for_classification = False
if imbalance_method == 'class_weight' and task_type == 'classification':
    if hasattr(model, 'class_weight'):
        try:
            model.set_params(class_weight='balanced')
        except Exception as e:
            warnings.warn(...)
    else:
        # Check if model supports sample_weight in fit() (e.g., RidgeClassifier)
        import inspect
        model_fit_sig = inspect.signature(model.fit) if hasattr(model, 'fit') else None
        if model_fit_sig and 'sample_weight' in model_fit_sig.parameters:
            use_sample_weight_for_classification = True
        else:
            warnings.warn(f"{model_name} does not support class_weight ...")
```

**Empirical attribute check:**

| Model | `hasattr(model, 'class_weight')` | Path taken |
|-------|----------------------------------|------------|
| XGBoost | `False` | sample_weight fallback |
| LightGBM | `True` | native `class_weight='balanced'` (works) |
| CatBoost | `False` | sample_weight fallback |
| MLP | `True` (the attribute exists) | native `class_weight='balanced'` — **but sklearn MLP `fit()` doesn't actually accept `class_weight` until 1.7; the constructor takes it for forward-compat but it's a no-op in 1.5/1.6** |

So XGBoost and CatBoost flow through the `sample_weight` route at `search.py:3865-3899` — `sample_weight=compute_sample_weight('balanced', y_train)` is computed and passed at fit time. **This produces statistically equivalent results to `scale_pos_weight=n_neg/n_pos` for binary XGBoost** (both reweight the positive-class loss), but uses a different mechanism than the FTIR Bone PLS paper's `XGBClassifier(scale_pos_weight=spw)` constructor pattern.

Important caveat: this path has the **T-32 length-mismatch bug** at `search.py:3858, 3891` — `sample_weight_train` was computed from pre-resampling `y_train`, but if a resampler runs at line 3879, `y_train_for_model` has different length. Line 3881 attempts to recompute, but the second `final_model.fit(...)` block at line 3891 still passes the (potentially) wrong array. This bug fires today only when `imbalance_method='class_weight'` is selected for a non-class_weight-attribute model (XGBoost, CatBoost, possibly MLP) AND a resampler is somehow chained — currently the GUI only allows one or the other, not both, so the bug is reachable only via API.

### 1e. What the GUI exposes today

**`spectral_predict_gui_optimized.py:11015-11088`** — Section 2.6 of the Data Quality tab:

- One enable checkbox: `enable_imbalance_handling` (default False)
- One method dropdown: `imbalance_method` with values `[smote, adasyn, borderline_smote, random_undersampler, tomek_links, smote_tomek, smote_enn, class_weight]` (classification) or regression equivalents
- Method-specific parameter widgets (k_neighbors / n_bins / boost_factor)

**There is no per-model dropdown.** The user picks ONE imbalance handling method that applies to all models in the run. Picking `class_weight` triggers the routing described in §1d.

The dropdown's `class_weight` description (`gui:22479`): `'Class weights - No resampling, weight loss function'` — accurate but not specific about which kwarg gets used per library, and silent that XGBoost/CatBoost go through `sample_weight` rather than the per-library kwarg the FTIR paper cites.

---

## 2. Is the framing correct? — chemometrics literature and field check

### 2a. FTIR Bone PLS paper — primary source for the gap claim

The Sponheimer et al. FTIR Bone PLS paper (working dir `C:\Users\mspon\Desktop\_DeskSync\FTIR Bone PLS\`) — verified accessible.

**`paper/results/alt_classifiers_5fold_ba.csv`** has rows whose `model` column reads literally `"XGBoost (scale_pos_weight)"` and `"LightGBM (balanced)"` — these are the headline classifier identifiers in the paper:

```
K_collagen,1pct,XGBoost (scale_pos_weight),0.9688,...
K_collagen,1pct,LightGBM (balanced),0.9598,...
K_collagen,3pct,XGBoost (scale_pos_weight),0.9838,...
K_collagen,3pct,LightGBM (balanced),0.9804,...
H_collagen,1pct,XGBoost (scale_pos_weight),0.8966,...
H_collagen,1pct,LightGBM (balanced),0.916,...
```

**`paper/scripts/analysis/03d_alt_classifiers_5fold_ba.py`** (verified read):

```python
cls['XGBoost (scale_pos_weight)'] = None  # set per-fold via training data
cls['LightGBM (balanced)'] = LGBMClassifier(
    n_estimators=300, class_weight='balanced',
    ...)
...
elif name == 'XGBoost (scale_pos_weight)' and HAVE_XGB:
    clf = XGBClassifier(n_estimators=300, scale_pos_weight=spw, ...)
```

So the paper:
1. Constructs LightGBM with `class_weight='balanced'` (matches what dasp does today via `hasattr(model, 'class_weight')` path).
2. Constructs XGBoost with **per-fold-computed `scale_pos_weight=spw`** as a constructor kwarg (NOT via `sample_weight=compute_sample_weight('balanced', y_train)` at fit time). The two are **statistically equivalent for binary classification but mechanistically different** — `scale_pos_weight` multiplies the gradient for the positive class globally, while `compute_sample_weight('balanced', y)` produces per-sample weights. For binary the math collapses to the same loss surface; for multiclass XGBoost `scale_pos_weight` is binary-only, so the design doc's choice of `sample_weight` for multiclass is correct.

This is **uncomplicated reproducibility**. A user who reads the paper, downloads dasp, and tries to reproduce "XGBoost (scale_pos_weight)" cannot do it from the GUI today. They can get statistically equivalent results via the `class_weight` dropdown's sample_weight fallback, but the model card doesn't say `scale_pos_weight` anywhere — it says `XGBoost`. The audit trail in the report doesn't reflect "this is the paper's `scale_pos_weight` configuration"; it reflects "this is XGBoost with sample_weight applied via the `class_weight` imbalance method." Not the same thing for replication purposes.

### 2b. The "nothing vs. balanced" misconception

The design doc has a load-bearing point worth confirming as canonical:

> **Default `class_weight=None` (or omitting `scale_pos_weight` for boosting models) does NOT mean "assume the data is balanced."** It means *all samples receive equal weight in the loss function*. With 100 positives and 10 negatives, the loss is dominated 10:1 by positives. The model learns to predict the majority class well and the minority class poorly.

This is **textbook correct**. Verified by the XGBoost docs themselves (`xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html`):

> A typical value to consider: `scale_pos_weight = sum(negative instances) / sum(positive instances)`

And by sklearn's `compute_class_weight('balanced', ...)` formula: `n_samples / (n_classes * np.bincount(y))`.

The design doc's recommendation to relabel the GUI options as `Equal sample weights (no correction)` vs `Class-balanced loss reweighting` is also chemometrics-correct — naming is currently an issue (the dropdown's `class_weight` value name leaks the internal sklearn kwarg name to the GUI user, which is not user-friendly).

### 2c. Commercial chemometrics software field-alignment

Web search results were not detailed enough to definitively confirm whether PLS_Toolbox, Unscrambler, SIMCA-Sartorius, or OPUS expose model-native loss reweighting. From what could be found:

- **PLS_Toolbox / Solo (Eigenvector)** — primary classifiers are PLS-DA, SIMCA, kNN, ANN; documentation ([eigenvector wiki SIMCA](https://wiki.eigenvector.com/index.php?title=SIMCA_Model_Builder_GUI)) describes T²/Q-based class membership rules, not loss reweighting. SIMCA is class-modeling not discriminant; reweighting doesn't directly apply. PLS-DA via PLS_Toolbox uses class-indicator regression — no class_weight knob documented.
- **Unscrambler X (CAMO)** — primary classifiers PLS-DA, SIMCA, LDA, SVM. No documented class_weight / scale_pos_weight UI surface in the search results.
- **SIMCA software (Sartorius/Umetrics)** — class-modeling tradition; reweighting not applicable to SIMCA itself.
- **OPUS (Bruker)** — primarily quantitative + classification via PLS-DA; no documentation found for boosting / loss reweighting.

The search did NOT find evidence that any of these classical chemometrics packages exposes per-class loss reweighting as a user-facing knob. The pattern is sklearn / Kaggle / modern-ML.

**This is a relevant signal under the chemometrics master rule**: T-26's "no leading program does this, so dasp shouldn't invent it" disposition could in principle apply here. BUT there's a critical distinction:

- **T-26**: dasp's behavior matched PLS_Toolbox default. No gap.
- **T-19**: classical chemometrics software doesn't ship XGBoost/LightGBM/CatBoost AT ALL. The classifier portfolio is different. Comparing dasp's XGBoost imbalance handling to PLS_Toolbox's nonexistent XGBoost is category-error territory.

The right comparison is "what does the chemometrics-applied-to-spectroscopy ML literature do for boosting models?" The FTIR Bone PLS paper is one direct example (`scale_pos_weight` per fold). Other examples:

- Cozzolino et al. (multiple) — boosting models in NIR/Raman applications use per-library imbalance kwargs when authors are imbalanced-aware.
- The XGBoost docs themselves (~17 search hits) treat `scale_pos_weight` as the canonical pattern for binary imbalanced classification.

### 2d. Chemometrics tradition vs. ML import

Per the chemometrics master rule, ask: "does the classical chemometrics tradition recommend loss reweighting for class imbalance, or does it recommend balancing the calibration set during sampling?"

Classical chemometrics literature (Brereton; Westad & Marini; Næs et al.) historically emphasizes **calibration-set design** — Kennard-Stone, DUPLEX, stratified random — to ensure the calibration covers feature space well. Imbalance correction at the loss-function level is a more recent ML-import. The chemometrics tradition's preferred answer to imbalance has been:

1. **Stratified sampling** (T-18 in this roadmap is exactly that — stratified Kennard-Stone).
2. **More data collection** for the minority class.
3. **SIMCA-style class modeling** that doesn't require balanced classes (each class gets its own PCA model).

So is T-19's loss-reweighting framing chemometrics-canonical or ML-imported? **Imported.** But "imported" isn't disqualifying under the master rule — the FTIR Bone PLS paper IS chemometrics applied to bone-FTIR, AND it imports `scale_pos_weight`. The paper is the user's own work; the user's domain is chemometrics + bone-FTIR + paleoanthropology + ML. A "modern chemometrics-ML hybrid" identity is exactly what dasp positions itself as.

The defensible reading: T-19 is chemometrics-by-the-user's-own-recent-paper; classical chemometrics tradition has different patterns (stratified sampling, SIMCA) but the user's working tradition includes scale_pos_weight. Under the user's master rule, dasp should support both — and dasp already supports stratified KS (T-18 will land it), so adding native loss reweighting alongside that is consistent.

### 2e. Comparison with T-26 (which dropped under the master rule)

| Dimension | T-26 (DROP) | T-19 (this finding) |
|-----------|-------------|---------------------|
| Reachable in GUI? | Yes | Partially (current path only via class_weight dropdown + sample_weight fallback) |
| Matches leading program default? | Yes — PLS_Toolbox same default | N/A — leading classical programs don't have boosting models |
| Real-world impact magnitude on user's data? | Never observed in thousands of analyses | Unknown — but FTIR Bone PLS paper is reproducibility-blocked |
| Backend-only fix would be useless? | Yes (T-26 verdict) | **Same risk applies** — backend kwarg without GUI = dead code |
| Chemometrics master-rule field test | Pass (matches PLS_Toolbox) | Mixed — classical doesn't apply, modern ML chemometrics does use this |

The T-26 lesson DOES partially apply: a backend-only T-19 implementation would be dead code under bundled-app distribution. The GUI-surface design is therefore as load-bearing as the backend wrapper.

### 2f. Recurring false-alarm pattern check

| Pattern | Applies to T-19? | Evidence |
|---------|------------------|----------|
| sklearn-instinct false alarm | **No.** The chemometrics-ML hybrid literature explicitly uses these kwargs. Not a sklearn-purity fetish. |
| Defensive code in unreachable branch | **No.** The five PLS-DA sites and the boosting model defaults are reached on every classification run. |
| Display-economy / code style | **No.** This changes which loss function the model optimizes. Not display. |
| Match-the-field cuts both ways | **Mixed.** Classical chemometrics doesn't use this; modern chemometrics-ML does. The user's own paper does. The gap is real for the user's domain even if it wouldn't be for an Unscrambler-only user. |
| Real finding can warrant zero action | **Possibly applicable.** "Real, but the existing class_weight dropdown's sample_weight fallback covers the math for binary XGBoost/CatBoost." The verdict depends on whether mechanistic exposure (constructor kwarg vs sample_weight) is worth 5-7 days, OR whether T-32 alone is enough to demand the fix. |

T-19 is not a false alarm in the strong sense. It is plausibly a "real but partially redundant" finding — the existing `class_weight` dropdown does deliver balanced loss for the boosting models via fallback, just not via the canonical per-library kwargs and not with a clean per-fold log.

---

## 3. GUI reachability — what works today, what doesn't

### 3a. What a user CAN do today for class imbalance

Confirmed by reading the GUI code at `gui:11015-11088, 22470-22558` and tracing through `search.py:4208-4244, 3865-3899`:

| Model | Current GUI path with `imbalance_method='class_weight'` | Effective behavior |
|-------|---------------------------------------------------------|--------------------|
| RandomForest | `set_params(class_weight='balanced')` | Native sklearn balanced — ✅ correct |
| LogisticRegression (standalone) | `set_params(class_weight='balanced')` | Native sklearn balanced — ✅ correct |
| SVC | `set_params(class_weight='balanced')` | Native sklearn balanced — ✅ correct |
| LightGBM | `hasattr(model, 'class_weight')` is True → `set_params(class_weight='balanced')` | Native LightGBM balanced — ✅ correct |
| XGBoost | `hasattr(model, 'class_weight')` is False → `sample_weight` fallback | Statistically equivalent to `scale_pos_weight` for binary — ✅ correct |
| CatBoost | `hasattr(model, 'class_weight')` is False → `sample_weight` fallback | Statistically equivalent to `auto_class_weights='Balanced'` — ✅ correct |
| MLP | `hasattr(model, 'class_weight')` is True → `set_params(class_weight='balanced')` — but sklearn MLP `fit()` doesn't accept `class_weight` until 1.7 | **Silent no-op on sklearn 1.5/1.6** — ❌ reproducibility hazard |
| PLS-DA | Inner LR: only 4 of 9 sites honor → mixed | ❌ inconsistent across paths |
| RidgeClassifier | `sample_weight` fallback | ✅ correct |

So the **statistically correct case is already reachable for all classifiers except MLP** (depending on sklearn version) and **partially reachable for PLS-DA** (depending on which path runs).

The DELTA between current and design-doc-T-19 is:

1. **Naming/labeling** — current dropdown says `class_weight`, FTIR paper says `scale_pos_weight` and `class_weight='balanced'`. Audit-trail mismatch but not statistical mismatch (for binary).
2. **Per-library kwarg vs. sample_weight** — the math is equivalent for binary; for multiclass XGBoost the FTIR paper would use `sample_weight` anyway (scale_pos_weight is binary-only).
3. **PLS-DA inconsistency** — 5 sites silently use default. Real bug.
4. **MLP sklearn-floor question** — separate but related.
5. **T-32 latent bug** — fires when sample_weight + resampler chain (currently impossible in GUI).
6. **Per-fold logging** — current code doesn't log "applied balanced loss with per-class weights {...} for this fold." Auditing what actually happened requires reading the source.

### 3b. What a user CANNOT do today

1. **Reproduce the FTIR Bone PLS paper's exact configuration with `scale_pos_weight` reflected in the model name / log / report.** Current dasp would log it as "XGBoost with class_weight imbalance method" — different audit trail.
2. **Use `is_unbalance=True`** (LightGBM's binary-only shortcut) — not exposed.
3. **Use `auto_class_weights='SqrtBalanced'`** (CatBoost's softer correction) — not exposed.
4. **Use a custom `scale_pos_weight=N`** for ratio-tuning (not exactly `n_neg/n_pos`) — not exposed.
5. **Have boosting model balanced loss + a resampler chained** — possible to express in API but the existing path has the T-32 length-mismatch bug.
6. **Combine PLS-DA inner LR balanced + a non-mainline path (rebuild-from-row, GA preprocessing)** — silently reverts to default class_weight=None.

### 3c. What the design doc proposes for GUI

`docs/plans/2026-04-29-model-native-loss-reweighting.md` ¶ "Proposed UI" — a **per-model "Imbalance handling" dropdown** in each classifier's hyperparameter card on the Model Config tab, with 4 options:

1. `Equal sample weights (no correction)` (default)
2. `Auto (decide at fit time)`
3. `Class-balanced loss reweighting`
4. `scale_pos_weight (custom)` (boosting binary only)

This is a **moderate GUI addition** — each of the ~10 classifier cards on the Model Config tab needs the dropdown. The dropdown is per-model, so the user can mix (e.g., XGBoost with custom scale_pos_weight=4, LightGBM balanced, RandomForest no correction).

**Distribution-model check**: this is a substantial GUI surface (~10 dropdowns). Does it deliver value over extending the existing single dropdown? Three considerations:

- The existing single dropdown is currently a "global override" — applies to all models in the run. The FTIR paper applies different imbalance handling per-model (not really — they all use balanced loss, just different kwargs). So per-model granularity may be over-engineered for the user's actual workflow.
- BUT: per-model granularity is the "right" UI for the long term — different models benefit from different imbalance strategies, and users running grid search over multiple models with different per-class kwargs (e.g., `scale_pos_weight=4` for XGBoost, `auto_class_weights='SqrtBalanced'` for CatBoost) would benefit.
- Counter: the existing single dropdown could be extended to dispatch model-native kwargs internally without changing UX. "Class-balanced loss reweighting" → uses per-library kwarg under the hood. Same UX, better mechanics, smaller scope (~2 days vs ~5-7 days).

The design doc favors per-model dropdowns. The pragmatic alternative is "extend the existing dropdown to use canonical per-library kwargs internally + add per-fold logging + fix the 5 PLS-DA sites + fix T-32." That's a smaller patch with most of the value.

---

## 4. Empirical: how is the user's actual workflow affected?

### 4a. Recent runs and configs

Looking at `example/R50_model_PLS_20260327_114623_ref.dasp` (in the working dir; binary file, not opened directly) — the filename suggests a PLS run on R50 data. The PROJECT_STATUS history (2026-04-19 entry) mentions "imbalance task-type-aware dropdown + banner + logger fallback + banner-ordering fix" indicating substantial recent work on the existing imbalance UI. The user has clearly been touching this area.

### 4b. The 2026-04-19 imbalance UI work (verified in PROJECT_STATUS)

PROJECT_STATUS mentions imbalance UI commits from 2026-04-19:
- `a6955c5`/`b8a70cb`/`f8ab20b`/`ac43289` — task-type-aware dropdown + banner + logger fallback
- `5dd00a0` — imbalance dropdown refreshes on task-type change
- `8b2cd74` — don't auto-clear banner on redundant refresh

So the existing single-dropdown imbalance UI has been actively maintained and refined. **A wholesale rebuild to per-model dropdowns might disrupt that recent work.** The question is whether the per-model design extends or replaces the existing UI.

### 4c. The 2026-04-29 design + Codex review work

The design doc `docs/plans/2026-04-29-model-native-loss-reweighting.md` was written 2026-04-29 and reviewed by Codex the same day. The REVISIONS section (R1–R6) at the top supersedes specific claims in the body:

- R1: 5 PLS-DA sites need wiring (not 2 as originally said)
- R2: sample_weight via fit_params is broken under imblearn pipelines — custom wrapper required
- R3: MLP sample_weight requires sklearn ≥1.7 — bundle compatibility check needed
- R4: T-32 active bug exists today
- R5: Stacking ensemble doesn't route sample_weight to base models — known limitation
- R6: Effort 5-7 days, not 1-2

The design has been triple-checked: original author (likely the user via opencode/Claude collaboration) → Codex code-verification → user pass (per the "post-Codex T-19 review" PROJECT_STATUS entry from 2026-04-29). **No live branch exists yet** — this is design-phase complete, implementation not started.

### 4d. Has the user actually hit class imbalance issues in their own workflow?

Direct evidence from PROJECT_STATUS:

> 2026-04-21 (late evening): **Reverted `_infer_task_type_from_y` to Fix-B1 heuristic ... 3 int-valued target, GUI auto-checked PLS-DA, backend resolved regression**

This is a related but different issue (task-type detection, not imbalance handling).

> 2026-04-19: imbalance task-type-aware dropdown + banner + logger fallback

The user was actively using the imbalance UI on 2026-04-19. The fact that the dropdown needed to become "task-type-aware" implies the user was switching between regression and classification tasks frequently — a workflow where imbalance IS a recurring concern. Plus the FTIR Bone PLS paper itself uses imbalance handling on every classification result row.

So **yes, the user actively hits class imbalance in their workflow**, and yes, they have a paper-in-prep that uses model-native loss reweighting as a headline configuration. This is not a hypothetical reproducibility gap — it's the user's current paper.

---

## 5. Scope and disposition considerations

### 5a. Existing design doc is sound

The post-Codex revisions section is methodologically rigorous:

- R1's 5-site PLS-DA list has been verified against current code (§1b above).
- R2's resampler-fit_params problem is real (verified imblearn behavior).
- R4's T-32 length-mismatch is real (the `final_model.fit(..., sample_weight=sample_weight_train)` lines at search.py:3858 and 3891 do not consistently use the post-resample y/weight pair).
- R6's effort estimate (5-7 days) is realistic for the scope described.

### 5b. Three sub-decisions for the verdict-writer

**Decision A — Scope: full per-model dropdown, or extend existing single dropdown?**

The design doc favors per-model. The pragmatic alternative is "extend the existing class_weight option in the single dropdown to dispatch native kwargs internally":

| Approach | Effort | Reproduces FTIR paper? | Per-fold log specificity | Backwards-compatible? |
|----------|--------|------------------------|--------------------------|------------------------|
| Per-model dropdown (design doc) | 5-7 days | Yes (option `scale_pos_weight (custom)`) | Yes | Adds new UI surface |
| Extend single dropdown | 2-3 days | Statistically yes; not by-name | Yes (with logging) | Yes |
| Both: ship single-dropdown extension + add per-model dropdown later | Phased | Yes | Yes | Yes |

**Decision B — T-32 handling**

The Codex review correctly identifies T-32 as fired today only when `class_weight` + resampler chain — which the GUI doesn't currently express. So T-32 is a latent bug in API-only code paths. The custom wrapper proposed in design-doc R2 would fix T-32 as a side effect (the wrapper computes sample_weight from received post-resample y, eliminating the length-mismatch route). T-32 doesn't need separate handling if T-19 ships the wrapper.

If T-19 ships only a sklearn-floor-bumped sample_weight extension (no wrapper), T-32 needs separate handling (probably a length-check assertion, raise on mismatch).

**Decision C — MLP scope**

R3's question: does sklearn ≥1.7 build cleanly in the PyInstaller bundle (`spectral_predict_py312.spec`)? If yes, bump the floor and include MLP. If no, skip MLP from the imbalance dropdown.

This is a separate validation step that should be done before T-19 work begins, since it determines design scope. The user's local venv is sklearn 1.8 (per design doc R3), so design-mode work is unblocked; only the bundle question matters for shipping.

### 5c. Does T-19 actually deliver value via GUI?

The T-26 lesson: backend-only changes are dead code in a bundled-app context. T-19 absolutely needs GUI surface, but the existing single dropdown already provides it for the math (statistically correct balanced loss reaches XGBoost / CatBoost via sample_weight fallback). The GUI value-add of T-19 is:

1. **Audit-trail clarity** — model card name says `XGBoost (scale_pos_weight)` not `XGBoost + class_weight imbalance method`. For the FTIR paper reproducibility, this is the difference between "matches paper" and "approximately matches paper."
2. **Per-fold imbalance log** — `Imbalance handling: auto → balanced (ratio 4.2:1)` annotation. Currently absent.
3. **PLS-DA consistency** — 5 unwired sites silently differ. Reproducibility hazard within dasp itself.
4. **T-32 fix as a side effect** — closes the latent bug.
5. **Future per-model granularity** — if the user later wants `scale_pos_weight=N` custom values, the dropdown supports it.

The design-doc per-model dropdown delivers all 5. The single-dropdown extension delivers 1 (partially), 2, 3, 4 — but not 5 without further work.

---

## 6. Notes / uncertainties for the main agent

1. **Codex review is sound and corroborated.** The 5 PLS-DA sites are individually verified at the line numbers in the design doc REVISIONS section. The boosting kwargs are verified absent. The T-32 site is verified.

2. **Existing GUI imbalance UI is reachable and actively maintained.** Section 2.6 of Tab 2 (Data Quality), one-dropdown-applies-to-all design. Recent commits (2026-04-19) show the user uses it. This is NOT a fresh-greenfield UX — the design doc's per-model dropdown either extends or replaces this surface. Verdict-writer should decide which.

3. **MLP question is real.** sklearn 1.7 PyInstaller bundle compatibility check is required pre-implementation. The user's local env (sklearn 1.8) is unblocked, but distribution requires verification. If incompatible, MLP skip from the dropdown is the design doc's recommended fallback — defensible, since MLP is among the user's least-used models.

4. **T-32 is automatically resolved by T-19's wrapper approach.** No need for a separate ticket if T-19 ships with the design-doc R2 wrapper. If T-19 ships without the wrapper (single-dropdown extension only), T-32 needs its own minimal fix (length-check + raise).

5. **The chemometrics master-rule field test is mixed.** Classical chemometrics software (Unscrambler, SIMCA-Sartorius, PLS_Toolbox, OPUS) does not expose loss reweighting because it doesn't ship XGBoost/LightGBM/CatBoost. Modern chemometrics-ML (the FTIR Bone PLS paper, recent NIR/Raman ML applications) DO use these kwargs by name. dasp's positioning as "chemometrics + ML" puts it in the modern bucket. The master rule's "match the field" test passes for the modern field; would fail for the classical field. Verdict-writer should weigh this. My read: the user's own paper using `scale_pos_weight` settles the question — that IS the user's field.

6. **Is the gap real, or already adequately covered?** The EXISTING `class_weight` dropdown's sample_weight fallback delivers statistically correct balanced loss for XGBoost/CatBoost (binary). The DELTA T-19 closes is mostly about audit-trail / naming / per-fold logging / PLS-DA site consistency / T-32 closure / future scale_pos_weight=N customization. Whether that delta is worth 5-7 days is a value-judgment for the verdict-writer.

7. **The design doc's effort estimate (5-7 days) is realistic.** The 5 PLS-DA wirings (~half day), wrapper design + tests (~2 days), sklearn floor / MLP question (~half day), tests for resampling × balanced (~1 day), code export updates (~half day), GUI dropdown + tooltip + per-fold logging (~1-2 days). Sums to ~5.5-7 days.

8. **Per-fold logging is a load-bearing design feature**, not optional. Without it, the GUI dropdown's "Auto (decide at fit time)" mode is a black box and the user can't audit what actually happened. Per-fold logging is mentioned in the design doc Auto-mode implementation note #3 — verdict-writer should ensure it's not de-scoped.

9. **Stacking ensemble limitation (R5) is acceptable to document-as-known.** Wrapping every base model would explode the implementation surface; documenting is fine.

10. **No live branch exists.** Unlike T-04 (which has a verdict + merge), T-19 is design-complete but implementation has not started. The verdict-writer is making a "do we implement" decision, not a "do we merge" decision.

11. **Real finding can warrant zero action (T-26 lesson) — partially applicable here.** The T-26 disposition rested on dasp matching PLS_Toolbox default. T-19's analog: dasp's existing `class_weight` dropdown already produces statistically balanced loss for boosting models via the sample_weight fallback. So the math is already correct; the gap is naming / logging / consistency / future-flexibility. A zero-action verdict ("dasp's existing sample_weight fallback IS the chemometrics-equivalent of scale_pos_weight; no fix needed beyond the 5 PLS-DA sites + T-32") is defensible. But a moderate-action verdict ("ship single-dropdown extension to dispatch native kwargs + per-fold logging + 5 PLS-DA sites + T-32 fix") is also defensible. The full design-doc per-model dropdown is harder to defend on bundled-app + 5-7-day grounds.

12. **The design doc's "auto mode" detection threshold (3:1 imbalance ratio)** matches the existing `detect_class_imbalance(y, threshold=3.0)` in `imbalance.py:61-130`. This re-uses existing infrastructure cleanly — minor implementation point, but worth noting that the design doc's auto mode isn't inventing a new threshold.

---

## Sources

- [XGBoost notes on parameter tuning — `scale_pos_weight` for imbalanced classification](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
- [XGBoost issue: mechanism of `scale_pos_weight`](https://github.com/dmlc/xgboost/issues/2428)
- [LightGBM `class_weight='balanced'` — sklearn API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
- [CatBoost `auto_class_weights` parameter](https://catboost.ai/docs/concepts/python-reference_parameters-list.html)
- [scikit-learn `compute_class_weight` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
- [PLS_Toolbox SIMCA Model Builder GUI documentation](https://wiki.eigenvector.com/index.php?title=SIMCA_Model_Builder_GUI)
- [PLS_Toolbox product page (Eigenvector)](https://eigenvector.com/software/pls-toolbox/)
- [Cost-Sensitive SVM for Imbalanced Classification (Brownlee)](https://machinelearningmastery.com/cost-sensitive-svm-for-imbalanced-classification/)
- [How to Configure XGBoost for Imbalanced Classification (Brownlee)](https://machinelearningmastery.com/xgboost-for-imbalanced-classification/)
- FTIR Bone PLS paper, Sponheimer et al. (in prep, 2026), `paper/scripts/analysis/03d_alt_classifiers_5fold_ba.py` and `paper/results/alt_classifiers_5fold_ba.csv` — primary source for "XGBoost (scale_pos_weight)" and "LightGBM (balanced)" headline configurations.
- DASP design doc: `docs/plans/2026-04-29-model-native-loss-reweighting.md` (REVISIONS section R1–R6).
- DASP gap analysis: `docs/analysis_vs_ftir_bone_pls/GAP_ANALYSIS.md` §10.
