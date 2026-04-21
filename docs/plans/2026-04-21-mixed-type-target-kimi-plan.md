# Kimi 2.6 diagnosis + plan: Mixed-type target column bugs (three failing paths)

**Status:** Draft for Codex review. Not yet implemented.

## Bug symptoms reported by user (2026-04-21)

When the classification target column has mixed Python types in its object-dtype values (e.g. `int + str` or `float + str` from heterogeneous Excel cells), three independent code paths crash with the same class of error:

1. **Bayesian Optimization validation metrics**
   ```
   [Warning] Failed to compute validation metrics: '<' not supported between instances of 'str' and 'int'
   ...
   File "numpy/lib/_arraysetops_impl.py", line 382, in _unique1d
     ar.sort()
   TypeError: '<' not supported between instances of 'str' and 'int'
   ```
   Triggered inside the classification branch of the validation-metrics block in `spectral_predict_gui_optimized.py` around line 26848 (`np.unique(y_np_clean)` / `np.unique(y_val_np)`).

2. **Stratified validation-set creation** — `'<' not supported between instances of 'str' and 'float'`. Fires inside `train_test_split(stratify=...)` → `StratifiedShuffleSplit`.

3. **SPXY validation-set creation** — same class of error. Fires inside SPXY's `LabelEncoder` / distance-matrix calculation for non-numeric y.

**User confirmed manually resetting the validation set does NOT fix the Bayesian-metrics crash.** So the root cause is NOT stale validation state from a prior regression run; the mixed-type array is produced during the classification run itself.

## Root cause (precise — from Kimi's final investigation)

`self.y` is stored as a pandas Series with `dtype=object` containing heterogeneous Python types when the source has mixed cells (e.g. `[1, 2, "3", "4", NaN]`). Six `self.y = ...` assignment sites exist in `spectral_predict_gui_optimized.py` (lines 17863, 17912, 17260, 46532, 17659, 16487); none coerce to a consistent dtype at assignment time.

Commit `2a1c77c` coerces the **training** target `y_filtered` at `_run_analysis_thread` lines 25986–26001:

```python
if (task_type in ('classification', 'one_class')
        and y_filtered is not None
        and hasattr(y_filtered, 'dtype')
        and y_filtered.dtype == object):
    types_present = {type(v).__name__ for v in y_filtered.dropna().values}
    if len(types_present) > 1:
        y_filtered = y_filtered.astype(str)
```

But `self.validation_y` is **never coerced anywhere** — `_on_target_column_changed:16511`, `_create_validation_set:19719`, `_create_manual_validation_set:19871` all set `self.validation_y` from the raw `self.y` without applying the same rule.

**Bug #1 crash site is line 26849** (not 26848):
```python
train_labels = np.unique(y_np_clean)   # line 26848 — OK, coerced upstream
val_labels = np.unique(y_val_np)       # line 26849 — CRASHES, y_val_np is uncoerced
```
Training y was already coerced to all-strings at line 25991 before being passed in as `y_np`. But `y_val_np` (line 26822: `y_val_np = self.validation_y.values`) reads the uncoerced validation Series. One side is strings, the other is mixed int/str — hence the failure.

**Bug #2 and #3 root cause is earlier**: the `_create_validation_set` path has no coercion at all. When the user invokes stratified/SPXY split on a mixed-type target, the split algorithm hits the mixed array before any analysis thread ever starts. This is independent of Bug #1 and needs its own fix.

**Additional finding**: line 19609 still contains the `len(y.unique()) < 10` classification-detection heuristic that was removed at 9 other sites in commit `41371e0`. Likely missed by that sweep.

### Scope map (which paths are affected)

| Path | Affected | Evidence |
|------|----------|----------|
| Bayesian validation metrics | **Yes (Bug #1)** | `gui:26849` crashes on `np.unique(y_val_np)` |
| Grid search validation metrics | **Yes** | `gui:27253` passes uncoerced `self.validation_y.values` → `search.py:707` |
| NSGA-II validation metrics | **Yes** | `gui:27046` → `search.py:707` same path |
| Stratified validation creation | **Yes (Bug #2)** | `_create_validation_set` / `_validation_stratified` has no coercion |
| SPXY validation creation | **Yes (Bug #3)** | same function, different algorithm branch |
| One-class validation | **No** | `contamination.py:1018` already casts `y_val` to str |
| Refine tab validation | **Partially** | `gui:37098` reads uncoerced `self.validation_y.values`; would crash LabelEncoder, not np.unique |
| Non-validation classification runs | **No** | validation-metrics block is gated on `validation_y is not None` |

## Proposed fix — three options, in order of increasing scope

### Option Narrow (fixes Bug #1 only, smallest diff)

Copy the existing coercion block from `_run_analysis_thread` lines 25986–26001 and insert an identical block below it for `self.validation_y`:

```python
if (task_type in ('classification', 'one_class')
        and self.validation_y is not None
        and hasattr(self.validation_y, 'dtype')
        and self.validation_y.dtype == object):
    val_types_present = {type(v).__name__ for v in self.validation_y.dropna().values}
    if len(val_types_present) > 1:
        self._log_progress(
            f"  [i] Validation target has mixed Python types "
            f"({sorted(val_types_present)}) — coercing to strings for {task_type}."
        )
        self.validation_y = self.validation_y.astype(str)
```

Plus the same insertion in `_run_refined_model_thread` (near line 35238) for Refine-tab parity. Plus a `y_val.astype(str)` guard at the top of the classification branch inside `compute_validation_metrics_for_top_models` (`src/spectral_predict/search.py:707`) as cheap defense-in-depth.

**This fixes Bug #1, Grid, NSGA-II, and Refine.** Does NOT fix Bug #2 or Bug #3 (those fire before any analysis thread runs).

### Option Validation-Wide (fixes all three bugs, still narrow)

Add Option Narrow, PLUS coerce at the entry of `_create_validation_set`, before any split algorithm dispatches:

```python
# In _create_validation_set, after y_available is computed:
if is_classification and y_available is not None and hasattr(y_available, 'dtype'):
    if y_available.dtype == object:
        types_present = {type(v).__name__ for v in y_available.dropna().values}
        if len(types_present) > 1:
            self._log_progress(
                f"  [i] Target column has mixed Python types "
                f"({sorted(types_present)}) — coercing to strings for validation split."
            )
            y_available = y_available.astype(str)
```

Also: fix line 19609's stale `len(y.unique()) < 10` heuristic (commit `41371e0` missed this site).

**This fixes all three user-reported bugs.**

### Option Centralized (Kimi's preferred, largest diff)

Add a new helper method `_get_y_for_task(self, task_type, y_series=None)` and migrate the ~20 critical `self.y` call sites to use it. Preserves the original data; each consumer explicitly requests the format it needs.

```python
def _get_y_for_task(self, task_type, y_series=None):
    if y_series is None:
        y_series = self.y
    if y_series is None or y_series.dtype != object:
        return y_series
    if task_type == 'regression':
        return y_series
    if task_type in ('classification', 'one_class'):
        types_present = {type(v).__name__ for v in y_series.dropna().values}
        if len(types_present) > 1:
            return y_series.astype(str)
    return y_series
```

**Prevents future instances of this bug class** but migrating all ~20 call sites is a wider refactor than strictly necessary.

### Step 1: Centralized helper (primary fix)

Add new method `_get_y_for_task(self, task_type, y_series=None)` to the GUI class, inserted after `_on_target_column_changed` (~line 16540):

```python
def _get_y_for_task(self, task_type, y_series=None):
    """Return target values coerced appropriately for the task type."""
    if y_series is None:
        y_series = self.y
    if y_series is None:
        return None
    if y_series.dtype != object:
        return y_series
    if task_type == 'regression':
        return y_series  # preserve numeric or numeric-string
    if task_type in ('classification', 'one_class'):
        types_present = {type(v).__name__ for v in y_series.dropna().values}
        if len(types_present) > 1:
            return y_series.astype(str)
        return y_series
    return y_series
```

Plus: coerce at target-column assignment (line 16487) when task_type is already known:
```python
self.y = new_y.reindex(self.X_original.index)
current_task = self.task_type.get() if hasattr(self, 'task_type') else 'auto'
if current_task in ('classification', 'one_class'):
    self.y = self._get_y_for_task(current_task, self.y)
```

### Step 2: Defense-in-depth at the three failing sites

**2a. Bayesian validation metrics (`~line 26848`):**
```python
if task_type == 'classification' and y_np_clean.dtype == object:
    types_present = {type(v).__name__ for v in y_np_clean if pd.notna(v)}
    if len(types_present) > 1:
        y_np_clean = y_np_clean.astype(str)
# also coerce y_val_np the same way
train_labels = np.unique(y_np_clean)
val_labels = np.unique(y_val_np)
```

**2b. Stratified split (`~line 19609`):**
```python
if y.dtype in ['object', 'category'] or len(y.unique()) < 10:
    if y.dtype == object:
        types_present = {type(v).__name__ for v in y.dropna().values}
        if len(types_present) > 1:
            y = y.astype(str)
    stratify = y
```

**2c. SPXY split** (location: `_validation_spxy` in GUI + possibly `src/spectral_predict/sample_selection.py`):
```python
if not pd.api.types.is_numeric_dtype(y_values.dtype):
    if y_values.dtype == object:
        types_present = {type(v).__name__ for v in y_values if pd.notna(v)}
        if len(types_present) > 1:
            y_values = y_values.astype(str)
    le = LabelEncoder()
    y_array = le.fit_transform(y_values).reshape(-1, 1).astype(float)
```

### Step 3: Task-type-switch validation reset (secondary defense)

In `_on_task_type_changed` (line 16242), clear `validation_X`/`validation_y` when the user crosses the regression ↔ classification/one-class boundary, since the validation split algorithm's stratification assumptions no longer hold. Warn in the progress log.

### Step 4: Regression test

New file `tests/test_mixed_type_target_coercion.py` with parametrized tests that feed `pd.Series([1, 2, "3", "4", np.nan], dtype=object)` into each of the three failing paths and assert no TypeError.

## Known risks and open questions

1. **NaN → "nan" string gotcha.** `y_series.astype(str)` converts `np.nan` to the literal string `"nan"`. Any downstream code using `pd.isna()` on the coerced Series will return False for those rows. Kimi flagged this but did not fully resolve — the plan suggests `.where(pd.notna(y), y.astype(str))` if a consumer relies on post-coercion NaN detection. **Needs verification per consumer.**

2. **"Single upstream" fix is still narrow.** The helper lives in the GUI class. The six `self.y = ...` assignment sites are only protected if they all go through the helper. Kimi listed 162 direct `self.y` accesses; the plan only migrates the critical ones (validation creation, analysis/refine threads). Non-critical ones (visualization, export) are left as-is. **Acceptable tradeoff or too narrow?**

3. **Line-number accuracy.** Kimi worked from grep results, not exhaustive re-verification against current `main` (tip `9ea4267`). Line numbers should be spot-checked before editing.

4. **Line 19609 `nunique()<10` heuristic.** Commit `41371e0` removed this heuristic at 9 sites but `_create_validation_set` appears to have been missed. Should the fix also remove this heuristic, or just coerce around it?

5. **Regression guard on existing coercion.** Kimi said "verify regression guard is present at 25986/35224" but did not actually verify. **Codex should confirm.**

6. **Test realism.** The proposed `test_bayesian_validation_metrics_with_mixed_types` stub calls `run_unified_bayesian` with only 2 trials and 2-fold CV — may not actually exercise the failing `np.unique` code path. Test may pass even if fix is wrong.

## Three commits proposed

1. Centralized helper + defense-in-depth at three sites + tests
2. Validation-set reset on task-type change
3. (optional) Migrate remaining `self.y` access to helper
