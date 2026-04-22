# 2026-04-21 Refine Task-Type Root Cause Investigation

## Verdict

Strongly suspect: the pasted crash was from a pre-`aa73edd` runtime, or from a stale GUI process/imported file that did not include the `aa73edd` / `8f14511` guard.

I did **not** find a current code path in `HEAD` (`8f14511`) that resets a regression Bayesian result to `classification` between loading a result row and clicking Run. The traceback line evidence points more strongly to version skew than to a still-present Tkinter state mutation.

## Evidence

### The pasted line number matches pre-guard code

The user reported:

```text
spectral_predict_gui_optimized.py line 36790 in _run_refined_model_thread
    _final = pipe.steps[-1][1]
```

In the parent of `aa73edd`, line 36790 is exactly:

```python
_final = pipe.steps[-1][1]
```

In current `HEAD` (`8f14511`), that statement has moved to `spectral_predict_gui_optimized.py:36816`, and the defensive task/y consistency block now appears before the CV splitter at `spectral_predict_gui_optimized.py:36119-36134`.

Current relevant code:

```python
if task_type == 'classification':
    from sklearn.utils.multiclass import type_of_target
    try:
        y_kind = type_of_target(y_array)
    except (TypeError, ValueError):
        y_kind = None
    if y_kind == 'continuous':
        logger.warning(...)
        task_type = 'regression'
        self.root.after(0, lambda: self.refine_task_type.set('regression'))
```

Then the only Model Development splitter is built at `spectral_predict_gui_optimized.py:36393-36400`:

```python
cv = build_cv_splitter(
    strategy=cv_strategy,
    n_folds=n_folds,
    task_type=task_type,
    n_repeats=cv_n_repeats,
    random_state=42,
    y=y_array,
)
```

And `src/spectral_predict/cv_utils.py:227-240` rejects `task_type='classification'` with a continuous target before sklearn's `StratifiedKFold` can raise the cryptic error.

### BoneCollagen target is continuous

Local check:

```text
%Collagen float64, 43 unique values, sklearn type_of_target = continuous
CollagenCat str, 3 unique values, sklearn type_of_target = multiclass
```

So if current `HEAD` reaches the defensive block with `task_type == 'classification'`, it should auto-correct to regression before `cv.split(...)`.

## Suspect Classes Checked

### (a) Tkinter trace / event ordering

`self.refine_task_type.trace_add(...)` is registered at `spectral_predict_gui_optimized.py:14725-14726`.

The callback `_on_refine_task_type_changed()` at `spectral_predict_gui_optimized.py:16630-16657` only:

- reads `self.refine_task_type.get()`
- updates the supported model dropdown
- changes `self.refine_model_type` if the current model is unsupported

It does **not** write `self.refine_task_type`.

No evidence found that the trace resets `refine_task_type` during load.

### (b) Bayesian imbalance auto-substitution

`src/spectral_predict/unified_bayesian.py:1711-1723` substitutes regression methods:

```python
UNSUPPORTED_REGRESSION_METHODS = {'binning', 'rare_boost', 'balanced'}
if task_type == 'regression' and imbalance_method in UNSUPPORTED_REGRESSION_METHODS:
    original_method = imbalance_method
    imbalance_method = 'smogn'
```

This does not modify `task_type`. The substituted `imbalance_method` is then passed into `create_unified_objective(...)` at `unified_bayesian.py:1778-1803` and into `convert_study_to_dataframe(...)` at `unified_bayesian.py:1897-1908`.

`convert_study_to_dataframe()` writes the result row task directly at `unified_bayesian.py:2005-2007`:

```python
row = {
    'Task': task_type,
    'Model': model_name,
    ...
}
```

No classification-flavored row write was found in this substitution path.

### (c) Multiple `build_cv_splitter` call sites

There are three relevant `build_cv_splitter` references near the reported area:

- `spectral_predict_gui_optimized.py:34992-35013`: learning-curve thread, not `_run_refined_model_thread`.
- `spectral_predict_gui_optimized.py:35132-35141`: import inside `_run_refined_model_thread`, not a splitter call.
- `spectral_predict_gui_optimized.py:36393-36400`: the actual Model Development splitter call.

The actual call is **after** the `type_of_target` guard in current `HEAD`.

The later `cv.split(X_raw, y_array)` at `spectral_predict_gui_optimized.py:36855` uses the already-built splitter. There is no earlier refinement CV split before the guard in current `HEAD`.

### (d) `self.task_type` vs `self.refine_task_type`

The refinement runner reads:

```python
task_type = self.refine_task_type.get()
```

at `spectral_predict_gui_optimized.py:35711-35712`.

I did not find `_run_refined_model_thread` reading the Analysis-tab `self.task_type` for the main refinement task decision.

### (e) `self.y` mutation

There are many dataset-load paths assigning `self.y`, but no Bayesian-completion path was found that reassigns `self.y` to classification labels after a regression Bayesian run.

For the reported dataset, `%Collagen` remains a continuous float target. Even if some state marked the refinement as classification, current `HEAD` should detect `type_of_target(y_array) == 'continuous'` and correct before CV.

### (f) Race condition / queued `after`

`_load_model_for_refinement()` sets `self.refine_task_type` synchronously at `spectral_predict_gui_optimized.py:32502-32503`:

```python
detected_task_type, inlier_label_saved = _detect_refine_task_type(config, self.y)
self.refine_task_type.set(detected_task_type)
```

No `root.after(...)` is involved in setting the loaded result's task type. I did not find a queued task-type update that could run after the user clicks Run.

## Why GLM's Defense-In-Depth Is Mostly Sufficient

`aa73edd` and `8f14511` are sufficient for the exact sklearn crash mode shown in the report when current code is actually running:

- `_run_refined_model_thread` auto-corrects `classification + continuous y` to regression before building CV.
- `build_cv_splitter(..., y=y_array)` now validates the same mismatch and raises a clearer error if an unguarded call path passes bad state.
- `8f14511` improved the guard from "numeric with >2 uniques" to sklearn's `type_of_target`, avoiding false correction of integer-encoded multiclass labels.

The defense is not a complete root-cause fix for why `refine_task_type` could ever become `classification` for a regression row. It is a robust crash blocker around the observed failure.

One remaining weakness: the model object is constructed before the auto-correction (`spectral_predict_gui_optimized.py:35750-35761`), so if a stale/missing `Task` causes `task_type='classification'`, the model may be created in classification mode before `task_type` is corrected for CV. For PLS-DA this is especially awkward because `model_name == "PLS-DA" and task_type == "classification"` creates a `PLSTransformer`. The later guard prevents `StratifiedKFold`, but it does not rebuild the estimator after changing `task_type`.

## Suggested Real Fix

1. Add a preflight helper immediately after `y_series` is finalized and **before** model creation:

```python
task_type = _coerce_refine_task_type_against_y(
    task_type=self.refine_task_type.get(),
    y=y_series,
    config=self.selected_model_config,
)
```

This should run before `get_model(...)`, before PLS-DA special handling, and before any pipeline construction.

2. If `selected_model_config['Task']` is `regression` or `classification`, prefer it over the radio value at Run time and log when they disagree. `_load_model_for_refinement()` already does this on load; Run should reassert it because the radio is user-editable and can drift.

3. If the guard corrects `classification -> regression`, also revalidate/remap `model_name`. For example, `PLS-DA` should become `PLS` or the run should stop with a clear message.

4. Add a regression test for this exact stale-state case:

```python
config = {'Task': 'regression', 'Model': 'PLS', ...}
app.refine_task_type.set('classification')  # simulate stale UI/radio drift
app.y = BoneCollagen['%Collagen']
run preflight
assert task_type == 'regression'
assert model_name is regression-compatible
assert build_cv_splitter(...).split(...) uses KFold, not StratifiedKFold
```

5. Operationally, rerun the smoke test from a fresh Python process after confirming:

```text
git rev-parse --short HEAD
```

prints `8f14511` or later, and that the traceback line numbers no longer match the pre-`aa73edd` file.
