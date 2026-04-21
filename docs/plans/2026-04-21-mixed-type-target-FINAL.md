# Final plan: Mixed-type target coercion — Validation-Wide (Codex-approved)

**Status:** Approved for implementation 2026-04-21
**Scope:** Fix three reported crashes when the classification target column is object-dtype with heterogeneous Python types (e.g., `[1, 2, "3", "4", NaN]` from mixed Excel cells).
**Option:** Validation-Wide (coerce **locally**, after NaN filter, at every consumer that sorts or encodes validation labels).

## Why not Centralized / Narrow

- **Narrow** (only the Bayesian `np.unique` site) leaves Bugs #2 and #3 broken.
- **Centralized** (coerce `self.y` at assignment time, or migrate all ~162 `self.y` reads to a helper) has a **blocking NaN regression**: `astype(str)` converts `np.nan` to the literal string `"nan"`, which silently breaks `self.y.isna()` at `gui:16487` plus downstream NaN filters at `search.py:419, 3070, 3701` and `gui:26836, 27060, 37101`.
- **Validation-Wide** coerces on **local copies**, strictly **after** NaN filtering, leaving `self.y` and `self.validation_y` as originally loaded.

## Rule for every coercion site

```python
# 1. Drop NaN first (existing code already does this in most sites)
mask = pd.isna(y_local)
y_local = y_local[~mask]

# 2. Then coerce mixed-type object arrays
if getattr(y_local, 'dtype', None) == object:
    types_present = {type(v).__name__ for v in y_local}
    if len(types_present) > 1:
        y_local = y_local.astype(str)
```

Never mutate `self.y` or `self.validation_y`. Only local copies.

## Sites to fix

### 1. `_create_validation_set` (fixes Bugs #2 and #3)

**File:** `spectral_predict_gui_optimized.py`
**Function:** `_create_validation_set`
**Change:**
- Around `gui:19648` (automatic validation NaN drop) — unchanged, keep as-is.
- Around `gui:19682` (after `y_available` is computed, after auto NaN drop, before method dispatch to `_validation_stratified` / `_validation_spxy` / `_validation_random` / `_validation_kennard_stone`): apply the rule above to `y_available` when `is_classification` is True.
- Fix `gui:19609`: replace `if y.dtype in ['object', 'category'] or len(y.unique()) < 10:` with `if y.dtype in ['object', 'category'] or y.nunique() == 2:` (matches the classification-detection rule used elsewhere after commit `41371e0`).
- Around `gui:19871` (`_create_manual_validation_set`): add an explicit NaN filter + coercion step for `y_available` when `is_classification`, since this path previously did not drop NaN.

### 2. Bayesian validation metrics (fixes Bug #1)

**File:** `spectral_predict_gui_optimized.py`
**Location:** after the existing NaN-filter block at `gui:26836-26841` and before `np.unique(y_np_clean)` at `gui:26848`.
**Change:** apply the coercion rule to **both** `y_np_clean` and `y_val_np` (local copies already in scope). Do this **only** when `task_type == 'classification'`, so regression is untouched.

### 3. NSGA-II validation metrics

**File:** `spectral_predict_gui_optimized.py`
**Location:** after NaN filter at `gui:27060`, before `np.unique` / `LabelEncoder`.
**Change:** same rule applied to `y_np_clean` and `y_val_np` when classification.

### 4. Refine-tab validation

**File:** `spectral_predict_gui_optimized.py`
**Location:** `_run_refined_model_thread`, after NaN filter at `gui:37101`, before `LabelEncoder.transform(val_y)`.
**Change:** same rule applied to `val_y` and training labels before encoding.

### 5. Backend library caller

**File:** `src/spectral_predict/search.py`
**Function:** `compute_validation_metrics_for_top_models`
**Location:** top of the classification branch (after NaN filter at `search.py:3701`).
**Change:** same rule applied to `y_val_for_val` and `y_train_for_val`. Protects callers that bypass the GUI.

### 6. Prediction/display paths (Codex-discovered)

**File:** `spectral_predict_gui_optimized.py`
**Locations:** `gui:40267`, `gui:40453`, `gui:40644` — these call `np.unique` on raw validation labels for prediction plots/statistics.
**Change:** insert the coercion rule on local copies at each site.

### 7. Task-type-switch UX (WARN, not reset)

**File:** `spectral_predict_gui_optimized.py`
**Function:** `_on_task_type_changed` (~`gui:16242`)
**Change:** when the new task type crosses the regression ↔ classification/one-class boundary AND `self.validation_y is not None`, emit a progress-log warning:
```
[!] Task type changed from {old} to {new}. Existing validation set was created under {old}; its stratification may no longer be representative. Consider recreating.
```
**Do NOT auto-clear** — holdout indices are still valid; user may intentionally reuse the split.

### 8. Stale heuristic cleanup (from commit `41371e0` sweep)

`gui:19609` is the last site that still uses `len(y.unique()) < 10` for classification detection. Fixed as part of Site 1 above.

## Explicitly NOT doing

- No mutation of `self.y` or `self.validation_y`.
- No `_get_y_for_task` helper migration.
- No auto-clear of the validation set on task-type change.
- No coverage of `search.py:975` and `search.py:3232` (`LabelEncoder.fit_transform(y_np)` in regression/classification internals). These are training-path sites outside the validation-metric scope; any fix there belongs in a follow-up. Noted in known follow-ups.

## Tests (new file: `tests/test_mixed_type_target_coercion.py`)

1. `test_validation_stratified_accepts_mixed_type_labels` — call `_validation_stratified` (via a test harness or module-level copy) with `pd.Series([1, "1", 2, "2", 1, "2"], dtype=object)` paired with a synthetic X. Assert no TypeError, returned indices are a valid subset.

2. `test_validation_spxy_accepts_mixed_type_labels` — same for `_validation_spxy`.

3. `test_compute_validation_metrics_accepts_mixed_type_y_val` — call `compute_validation_metrics_for_top_models` with mixed `y_val` and a minimal fitted classifier fixture. Assert no TypeError and `val_Accuracy` column appears.

4. `test_nan_is_dropped_not_stringified` — pass `pd.Series([1, np.nan, "A", np.nan, 2], dtype=object)`. Assert post-processing array contains no `"nan"` string and NaN rows were dropped.

5. `test_task_type_change_warns_but_does_not_clear_validation` — GUI integration stub: simulate `validation_y is not None`, fire `_on_task_type_changed`, assert `self.validation_y is not None` and a warning was logged.

## Implementation sequencing (two commits)

**Commit 1:** Sites 1–6 (the seven coercion/NaN-filter sites) + all four coercion/null-preservation tests (tests 1–4).

**Commit 2:** Site 7 (task-type-switch warning) + test 5 + remove stale `nunique()<10` at site 8 (if not already handled by Site 1's rewrite).

Optional **Commit 3:** Known follow-up — migrate `search.py:975` / `3232` to a shared helper. Skip unless the user requests it.

## Known follow-ups (defer)

- `search.py:975` and `search.py:3232` — `LabelEncoder.fit_transform(y_np)` in training-path, would crash on mixed-type targets invoked via library API. Out of scope for this plan.
- Migrate to a real `_get_y_for_task()` helper once the validation-wide fix is battle-tested. The helper must preserve NaN; `astype(str)` semantics must only fire after NaN filtering.
