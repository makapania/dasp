# One-Class Model Config Parity — Safe Implementation Spec

**Goal:** Expose one-class grid-search hyperparameters to the user in the same UI style already used for regression/classification model configuration, while preserving the current default one-class search behavior exactly unless the user explicitly changes a model's settings.

**Architecture:** Add one-class model config cards to Tab 4C using the existing old-style `BooleanVar` + `ttk.Checkbutton` + single custom `Entry` pattern already used by Ridge/RandomForest. Extend `run_one_class_search()` with an optional per-model override payload. For each one-class model, if the GUI remains at the shipped default widget state, the backend must use the existing curated grid from `get_one_class_model_grids()`. Only models whose card state differs from the shipped default should use a user-defined grid. Bayesian one-class tuning is out of scope and must remain unchanged.

**Tech Stack:** Python, Tkinter, pandas, sklearn, existing GUI patterns in `spectral_predict_gui_optimized.py`, existing one-class grid search path in `src/spectral_predict/search.py`, curated one-class defaults in `src/spectral_predict/contamination.py`.

---

## Workflow Requirement

Before starting implementation, create a dedicated git worktree for this change.

Reason:
- this repo is actively used across sessions/machines
- the GUI file is large and high-churn
- isolating this work reduces risk of accidental overlap with unrelated changes
- it makes it easier to compare default-behavior preservation before/after

Recommended flow:

```bash
git status
git branch --show-current
git worktree add "../dasp-oc-model-config-parity" -b "feature/one-class-model-config-parity"
```

Then perform all implementation and testing from the new worktree, not the main checkout.

Requirements:
- the worktree should start from the intended base branch
- do not reuse a dirty worktree
- verify the new worktree is on the new branch before editing
- keep this feature isolated until default-preservation tests pass

---

## Critical Invariants

These are mandatory. Do not violate them.

1. **Default one-class grid-search behavior must stay unchanged.**
   If the user opens the new one-class config cards and does not change anything, `run_one_class_search()` must test the same curated parameter combinations it tests today.

2. **Bayesian one-class behavior must stay unchanged.**
   Do not modify `suggest_one_class_params()` or any one-class Bayesian UI/backend behavior.

3. **Do not convert the default one-class grid into a full Cartesian product.**
   The current one-class grids are curated. The new UI is list-based, but untouched models must still use the curated defaults.

4. **Only grid search is being opened up here.**
   This task is not about adding new one-class model families, not about Bayesian ranges, and not about introducing extra one-class parameters that were not already in the current grid.

5. **Use the same proven widget style as the older regression/classification cards.**
   Do not use `_create_parameter_grid_control()` for this task. That path already has a wiring bug in the codebase and is not the safe template.

6. **Keep the change minimal and reversible.**
   Preserve backward compatibility in `run_one_class_search()` for any existing callers that still use `oc_hyperparams` or no override payload at all.

---

## Non-Goals

Do not include any of the following in this implementation:

- No new one-class parameters beyond the ones already used by the current grid.
- No changes to one-class Bayesian search.
- No broad refactor of all existing model-config cards.
- No attempt to fix unrelated GUI model-config bugs outside the one-class path.
- No changes to `get_one_class_model_grids()` default contents.
- No changes to model tier logic beyond making the one-class cards available in the same tab.

Possible future follow-ups, but explicitly out of scope here:
- LOF `metric`
- IsolationForest `max_samples`
- OneClassSVM `coef0`
- EllipticEnvelope `support_fraction`
- Any redesign of one-class Bayesian parameter suggestions

---

## Existing Default Behavior To Preserve

Current curated grid source:
`src/spectral_predict/contamination.py::get_one_class_model_grids()`

At the time of writing, the default curated one-class grid is:

### OneClassSVM
```python
[
    {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.01},
    {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.05},
    {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.1},
    {'kernel': 'rbf', 'gamma': 'auto', 'nu': 0.05},
    {'kernel': 'poly', 'gamma': 'scale', 'nu': 0.05, 'degree': 2},
]
```

### IsolationForest
```python
[
    {'n_estimators': 100, 'contamination': 0.01, 'max_features': 1.0},
    {'n_estimators': 100, 'contamination': 0.05, 'max_features': 1.0},
    {'n_estimators': 100, 'contamination': 0.05, 'max_features': 0.5},
    {'n_estimators': 100, 'contamination': 0.1, 'max_features': 1.0},
    {'n_estimators': 100, 'contamination': 0.1, 'max_features': 0.5},
]
```

### EllipticEnvelope
```python
[
    {'contamination': 0.01},
    {'contamination': 0.05},
    {'contamination': 0.1},
]
```

### LOF
```python
[
    {'n_neighbors': 10, 'contamination': 0.05},
    {'n_neighbors': 20, 'contamination': 0.05},
    {'n_neighbors': 30, 'contamination': 0.05},
]
```

### PCA-SIMCA
```python
[
    {'n_components': 3, 'alpha': 0.05},
    {'n_components': 5, 'alpha': 0.05},
    {'n_components': 7, 'alpha': 0.05},
    {'n_components': 5, 'alpha': 0.01},
    {'n_components': 0.95, 'alpha': 0.05},
]
```

These exact defaults must still apply whenever the user has not customized a model's card.

---

## High-Level Design

### Core idea

The new one-class cards will visually expose parameter values the same way regression/classification cards do, but the backend must distinguish between:

1. **Untouched/default model card**
   Use the existing curated grid from `get_one_class_model_grids()` for that model.

2. **Customized model card**
   Build a grid for that specific model from the user-selected values.

This keeps the current shipped behavior intact while finally allowing users to change the grid.

### Why this design is required

If we simply translate the default checked boxes into a Cartesian product, the default one-class search space changes immediately. That would silently change runtime, search coverage, and possibly results. That is explicitly not allowed.

---

## Files To Modify

### Primary files
- `spectral_predict_gui_optimized.py`
- `src/spectral_predict/search.py`
- `tests/test_contamination_detection.py`

### Documentation files to update after implementation is complete
- `docs/SESSION_LOG.md`
- `docs/PROJECT_STATUS.md`

Do not modify `unified_bayesian.py` for this task.

---

## Task 1: Add A Small Backend Helper To Resolve One-Class Grids Safely

**File:** `src/spectral_predict/search.py`

### Objective
Create a small private helper that resolves the one-class model grids in one place, with explicit logic for:
- default curated behavior
- optional per-model override behavior
- backward compatibility with the existing `oc_hyperparams` path

### Recommended helper
Add a private helper near `run_one_class_search()`:

```python
def _resolve_one_class_model_grids(
    enabled_models,
    oc_model_param_overrides=None,
    oc_hyperparams=None,
):
    ...
```

This should not be a public API. Keep it private.

### Required behavior

1. Start from:
```python
oc_grids = get_one_class_model_grids()
```

2. If `oc_model_param_overrides` is falsy:
- preserve current behavior exactly
- if `oc_hyperparams` is provided, apply the existing legacy override behavior
- otherwise just return filtered curated grids

3. If `oc_model_param_overrides` is provided:
- treat it as a **per-model override**
- only models included in the override dict should switch to a user-defined grid
- all other enabled models must keep their curated default grids unchanged

4. Filter final output to `enabled_models`

### Override payload shape

Use this structure:

```python
{
    'OneClassSVM': {
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale'],
        'nu': [0.05],
        'degree': [2],
    },
    'IsolationForest': {
        'n_estimators': [100, 200],
        'contamination': [0.05],
        'max_features': [0.5, 1.0],
    },
}
```

Only customized models should appear in this dict.

### Important efficiency rule

For `OneClassSVM`, do **not** blindly include `degree` on non-`poly` kernels.

When building custom combinations:
- build `rbf` combos without `degree`
- build `poly` combos with `degree`
- if `poly` is not selected, ignore `degree`
- if `degree` list ends up empty, fall back to the default list `[2]`

This avoids duplicate/equivalent configs and keeps the search efficient.

### Combination behavior for customized models

For a customized model, use a Cartesian product across that model's selected parameter values.

That is acceptable only for customized models, because the user explicitly asked for custom control.

### Fallback behavior for empty selections inside a customized model

If a customized model has an empty list for a required parameter after parsing:
- fall back to that parameter's default shipped value list, not an empty grid

Examples:
- customized OneClassSVM but no kernels checked: fall back to `['rbf', 'poly']`
- customized LOF but no contamination values selected: fall back to `[0.05]`

This is consistent with the rest of the GUI's forgiving behavior and avoids zero-result runs caused by accidental unchecking.

### Backward compatibility

Keep `oc_hyperparams` support in `run_one_class_search()` for now.
Do not remove it in this task.
The new GUI path should stop depending on it, but old callers should still work.

---

## Task 2: Extend `run_one_class_search()` With A New Optional Override Argument

**File:** `src/spectral_predict/search.py`

### Add new optional argument

Extend the signature of `run_one_class_search()` with:

```python
oc_model_param_overrides=None,
```

Place it close to the existing `oc_hyperparams=None` argument so the one-class configuration arguments remain grouped.

### Required behavior inside `run_one_class_search()`

Replace the current inline grid resolution block with a call to the helper from Task 1.

Current behavior location is around:
- import of `get_one_class_model_grids`
- the "Get model grids" section
- the legacy `oc_hyperparams` overwrite logic

Refactor that block to:

1. determine `enabled_models`
2. call `_resolve_one_class_model_grids(...)`
3. continue the existing search loop unchanged

### Important constraint

Do not change the main search loop if you do not have to.
The goal is to change only how the grids are assembled, not how the models are evaluated.

---

## Task 3: Add One-Class Widget Variables Using The Old Proven Pattern

**File:** `spectral_predict_gui_optimized.py`

### Objective
Add one-class model-config widget variables using the same style as the old Ridge/RF sections:
- `tk.BooleanVar` for predefined values
- `tk.StringVar` for a single custom value entry where appropriate

### Location
Near the existing variable initialization section where:
- model tier vars are defined
- existing one-class vars currently exist
- RF/Ridge vars are defined

Current one-class flat vars are around:
- `self.oc_nu`
- `self.oc_contamination`
- `self.oc_alpha`
- `self.oc_n_components`

Do not delete those immediately. Leave them in place until the new path is fully wired and verified.

### New widget vars to add

#### OneClassSVM
- kernels:
  - `self.ocsvm_kernel_rbf = tk.BooleanVar(value=True)`
  - `self.ocsvm_kernel_poly = tk.BooleanVar(value=True)`
- gamma:
  - `self.ocsvm_gamma_scale = tk.BooleanVar(value=True)`
  - `self.ocsvm_gamma_auto = tk.BooleanVar(value=True)`
- nu:
  - `self.ocsvm_nu_001 = tk.BooleanVar(value=True)`   for `0.01`
  - `self.ocsvm_nu_005 = tk.BooleanVar(value=True)`   for `0.05`
  - `self.ocsvm_nu_01 = tk.BooleanVar(value=True)`    for `0.1`
  - `self.ocsvm_nu_custom = tk.StringVar(value="")`
- degree:
  - `self.ocsvm_degree_2 = tk.BooleanVar(value=True)`
  - `self.ocsvm_degree_custom = tk.StringVar(value="")`

Do not add `sigmoid`, `linear`, or extra `nu` defaults in this task.

#### IsolationForest
- n_estimators:
  - `self.if_n_estimators_100 = tk.BooleanVar(value=True)`
  - `self.if_n_estimators_custom = tk.StringVar(value="")`
- contamination:
  - `self.if_contamination_001 = tk.BooleanVar(value=True)` for `0.01`
  - `self.if_contamination_005 = tk.BooleanVar(value=True)` for `0.05`
  - `self.if_contamination_01 = tk.BooleanVar(value=True)`  for `0.1`
  - `self.if_contamination_custom = tk.StringVar(value="")`
- max_features:
  - `self.if_max_features_05 = tk.BooleanVar(value=True)` for `0.5`
  - `self.if_max_features_10 = tk.BooleanVar(value=True)` for `1.0`
  - `self.if_max_features_custom = tk.StringVar(value="")`

Do not add extra max_features defaults beyond what is already in the grid.

#### EllipticEnvelope
- contamination:
  - `self.ee_contamination_001 = tk.BooleanVar(value=True)`
  - `self.ee_contamination_005 = tk.BooleanVar(value=True)`
  - `self.ee_contamination_01 = tk.BooleanVar(value=True)`
  - `self.ee_contamination_custom = tk.StringVar(value="")`

No additional parameters in this task.

#### LOF
- n_neighbors:
  - `self.lof_n_neighbors_10 = tk.BooleanVar(value=True)`
  - `self.lof_n_neighbors_20 = tk.BooleanVar(value=True)`
  - `self.lof_n_neighbors_30 = tk.BooleanVar(value=True)`
  - `self.lof_n_neighbors_custom = tk.StringVar(value="")`
- contamination:
  - `self.lof_contamination_005 = tk.BooleanVar(value=True)`
  - `self.lof_contamination_custom = tk.StringVar(value="")`

Do not add metric or extra contamination defaults in this task.

#### PCA-SIMCA
- n_components:
  - `self.simca_n_components_3 = tk.BooleanVar(value=True)`
  - `self.simca_n_components_5 = tk.BooleanVar(value=True)`
  - `self.simca_n_components_7 = tk.BooleanVar(value=True)`
  - `self.simca_n_components_095 = tk.BooleanVar(value=True)` for `0.95`
  - `self.simca_n_components_custom = tk.StringVar(value="")`
- alpha:
  - `self.simca_alpha_001 = tk.BooleanVar(value=True)` for `0.01`
  - `self.simca_alpha_005 = tk.BooleanVar(value=True)` for `0.05`
  - `self.simca_alpha_custom = tk.StringVar(value="")`

Do not add `10` or other new defaults in this task.

---

## Task 4: Add One-Class Model Config Cards In Tab 4C

**File:** `spectral_predict_gui_optimized.py`

### Location
Inside:
```python
def _create_tab4c_model_configuration(self):
```

This function starts around line `11578` in the current file.

### Objective
Add a dedicated container for one-class model-config cards, using the same visual and widget style as the older regression/classification sections.

### Required structure

Create a parent container dedicated to one-class cards, for example:
```python
self.oc_model_config_container = tk.Frame(content_frame, bg=self.colors['bg'])
```

Grid this container into the same tab layout, and place all new one-class collapsible sections inside it.

This gives a single frame that can be shown/hidden in `_update_one_class_controls_visibility()` without refactoring the old cards.

### Important scope constraint

Do not attempt a large refactor of all existing regression/classification card visibility in this task.

It is sufficient to:
- add a dedicated one-class config container
- show it only for task type `one_class`
- hide it for regression/classification

### Card style

Use the same pattern as the old working cards:

1. `_create_collapsible_section(...)`
2. `_create_card(...)`
3. inner `ttk.Frame(...)`
4. explicit `ttk.Label`, `ttk.Checkbutton`, `ttk.Entry`
5. explicit widget-variable wiring

Do not use `_create_parameter_grid_control()`.

### Cards to add

#### Card 1: OneClassSVM Hyperparameters
Rows:
- Kernel:
  - `rbf`
  - `poly`
- Gamma:
  - `scale`
  - `auto`
- Nu:
  - `0.01`
  - `0.05`
  - `0.1`
  - `Custom: [entry]`
- Degree:
  - `2`
  - `Custom: [entry]`

Add a short caption label:
- degree only applies to `poly`
- default untouched behavior still uses the current shipped curated grid

#### Card 2: IsolationForest Hyperparameters
Rows:
- Number of Trees (`n_estimators`)
  - `100`
  - `Custom: [entry]`
- Contamination
  - `0.01`
  - `0.05`
  - `0.1`
  - `Custom: [entry]`
- Max Features
  - `0.5`
  - `1.0`
  - `Custom: [entry]`

#### Card 3: EllipticEnvelope Hyperparameters
Rows:
- Contamination
  - `0.01`
  - `0.05`
  - `0.1`
  - `Custom: [entry]`

#### Card 4: LOF Hyperparameters
Rows:
- Number of Neighbors (`n_neighbors`)
  - `10`
  - `20`
  - `30`
  - `Custom: [entry]`
- Contamination
  - `0.05`
  - `Custom: [entry]`

#### Card 5: PCA-SIMCA Hyperparameters
Rows:
- Number of Components (`n_components`)
  - `3`
  - `5`
  - `7`
  - `0.95 (variance)`
  - `Custom: [entry]`
- Alpha
  - `0.01`
  - `0.05`
  - `Custom: [entry]`

### Important UI behavior

The cards should feel like the existing model config cards, not like a separate mini-feature.

That means:
- same checkbutton style
- same labels
- same "Custom:" entry pattern
- same forgiving behavior if a custom value is invalid: ignore it with a warning instead of crashing

---

## Task 5: Show/Hide The One-Class Card Container Safely

**File:** `spectral_predict_gui_optimized.py`

### Location
Inside:
```python
def _update_one_class_controls_visibility(self):
```

Current function is around line `15692`.

### Required changes

When `task_type == "one_class"`:
- show `self.oc_model_config_container`

When not one-class:
- hide `self.oc_model_config_container`

### Important note on legacy flat controls

The old `oc_hyperparams_frame` currently appears in this function as well.

For safety:
- stop relying on it for one-class grid search
- it is acceptable to leave the old widget construction code in place temporarily
- update visibility so the new Tab 4C cards are the active configuration surface
- do not aggressively delete old code in the same pass unless it is obviously isolated and safe

The goal is to avoid breaking any unrelated path accidentally.

---

## Task 6: Add A GUI Helper To Collect One-Class Grid Overrides

**File:** `spectral_predict_gui_optimized.py`

### Objective
Keep `_run_analysis_thread()` from becoming more monolithic.

Add a small helper method, for example:

```python
def _collect_one_class_model_param_overrides(self):
    ...
```

This helper should return:

- `None` if no one-class model card differs from its shipped default widget state
- otherwise a dict shaped like:
```python
{
    'OneClassSVM': {'kernel': [...], 'gamma': [...], 'nu': [...], 'degree': [...]},
    ...
}
```

### Required design

For each one-class model:

1. Read the current widget values
2. Compare them to the **shipped default widget state**
3. If the model's current state still exactly matches the shipped default widget state:
   - do not include it in the override dict
4. If the model differs from the default widget state:
   - include it in the override dict

This is the key mechanism that preserves the curated default grids while still allowing customization.

### Do not use a one-way "dirty flag" only

A model should be treated as default again if the user changes it and later restores the exact shipped default widget state.

That allows the backend to switch back to the curated shipped grid automatically.

### Required parsing behavior

Follow the same style as the older cards:

- checkboxes contribute predefined literal values
- the custom entry contributes at most one extra literal value
- invalid custom values are ignored with a warning
- duplicate values are ignored
- preserve deterministic ordering

### Recommended default-state helpers

Add private helpers if useful, for example:

```python
def _one_class_model_card_defaults(self):
    ...
```

and/or

```python
def _one_class_model_card_matches_default(self, model_name):
    ...
```

Keep them private and small.

### Default widget states to compare against

#### OneClassSVM default widget state
- kernel: `rbf=True`, `poly=True`
- gamma: `scale=True`, `auto=True`
- nu: `0.01=True`, `0.05=True`, `0.1=True`
- degree: `2=True`
- custom entries empty

#### IsolationForest default widget state
- `n_estimators=100`
- contamination: `0.01`, `0.05`, `0.1`
- max_features: `0.5`, `1.0`
- custom entries empty

#### EllipticEnvelope default widget state
- contamination: `0.01`, `0.05`, `0.1`
- custom entry empty

#### LOF default widget state
- `n_neighbors`: `10`, `20`, `30`
- contamination: `0.05`
- custom entries empty

#### PCA-SIMCA default widget state
- `n_components`: `3`, `5`, `7`, `0.95`
- alpha: `0.01`, `0.05`
- custom entries empty

### Parameter-level empty-selection fallback behavior

For customized models, if a row ends up empty after parsing:
- fall back to the shipped default values for that row

Examples:
- no LOF neighbors selected => use `[10, 20, 30]`
- no PCA-SIMCA alpha selected => use `[0.01, 0.05]`

This prevents accidental zero-combo grids.

### Parsing rules

#### Floats constrained to `(0, 1]`
Applies to:
- `nu`
- `contamination`
- `alpha`
- `max_features`
- `0.95`-style `n_components` variance fractions

#### Positive integers only
Applies to:
- `degree`
- `n_estimators`
- `n_neighbors`
- integer `n_components`

#### PCA-SIMCA `n_components` custom parsing
Accept:
- positive integer `>= 1`
- or float in `(0, 1)` for variance-fraction mode

Reject everything else.

---

## Task 7: Wire The New Override Helper Into One-Class Grid Search Only

**File:** `spectral_predict_gui_optimized.py`

### Location
Inside `_run_analysis_thread()` in the one-class grid-search branch, around the current call to:

```python
results_df = run_one_class_search(...)
```

### Required change

Replace the old flat `oc_hyperparams={...}` payload from the GUI path with:

```python
oc_model_param_overrides=self._collect_one_class_model_param_overrides(),
```

### Important compatibility behavior

It is acceptable to leave the old flat `oc_hyperparams` argument as `None` or omit it entirely from the new GUI path.

However:
- do not remove `oc_hyperparams` support from the backend
- do not touch the Bayesian one-class path

### Important scope constraint

Only the one-class grid-search GUI call path should change here.

Do not alter regression/classification calls.
Do not alter one-class Bayesian calls.

---

## Task 8: Preserve Tier Semantics

**Files:**
- `spectral_predict_gui_optimized.py`
- `src/spectral_predict/model_config.py` should not require changes

### Required behavior

Keep current tier behavior intact:
- tier still determines which one-class models are enabled
- the new one-class config cards simply expose the grid parameters for those models
- no new special one-class tier logic is needed

This matches regression/classification behavior, where tiers primarily affect model inclusion and not a separate hidden backend mode.

---

## Task 9: Tests — Add Focused Coverage For Default Preservation And Per-Model Overrides

**File:** `tests/test_contamination_detection.py`

### Add a new test class
Recommended name:
```python
class TestOneClassModelConfigParity:
```

### Required tests

#### Test 1: default resolution returns the existing curated grids
Target the new helper in `search.py` directly if practical.

Verify:
- with `enabled_models` set and no overrides, the resolved grid equals `get_one_class_model_grids()` filtered to enabled models
- counts and exact dict contents match current defaults

This is the most important safety test.

#### Test 2: override on one model leaves other models untouched
Example:
- enable `['OneClassSVM', 'IsolationForest']`
- provide override only for `OneClassSVM`
- verify:
  - `OneClassSVM` grid uses the user-defined combinations
  - `IsolationForest` grid still equals the curated default grid

#### Test 3: OneClassSVM override prunes non-poly degree
Example override:
```python
{
    'OneClassSVM': {
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale'],
        'nu': [0.05],
        'degree': [2],
    }
}
```

Verify:
- `rbf` combo has no `degree` key
- `poly` combo includes `degree=2`

This protects the efficiency rule.

#### Test 4: customized model with empty row falls back to shipped row defaults
Example:
- customized LOF override provides contamination but empty neighbors list
- resolved grid should use default neighbors `[10, 20, 30]`

#### Test 5: untouched `run_one_class_search()` still works end-to-end
Add or extend an integration-style test calling `run_one_class_search()` with no override payload.
Goal:
- prove no regression in the default path

### Optional test if cheap
If the helper in the GUI is made small and reasonably testable, add a GUI-level unit test only if there is already a low-friction pattern for doing so. Do not sink time into brittle Tkinter tests for this task.

### Testing philosophy for this change

The critical thing to test is **default preservation**, not flashy UI behavior.

---

## Task 10: Manual Verification Checklist

After code changes are done, run this checklist manually.

### GUI checks
1. Launch the GUI.
2. Load a dataset with classes.
3. Switch task type to `One-Class`.
4. Open Tab 4C / Model Configuration.
5. Confirm the new one-class cards appear.
6. Confirm the checked defaults visually match the current shipped one-class grid values.
7. Confirm regression/classification workflows still launch normally.

### One-class default behavior checks
1. Run one-class grid search without changing any new card values.
2. Confirm the search completes.
3. Confirm result counts and general runtime are consistent with the prior behavior.
4. Spot-check that `Params` values match the old curated combinations.

### One-class customization checks
1. Change only `LOF n_neighbors` to `10` only.
2. Run one-class grid search with LOF enabled.
3. Confirm LOF uses only `n_neighbors=10`.
4. Confirm untouched models still use their shipped curated defaults.

### Reversion-to-default check
1. Change a one-class model card.
2. Restore it exactly to the shipped default widget state.
3. Run again.
4. Confirm that model uses the curated default grid again, not a Cartesian expansion.

This check is important because the implementation should compare current widget state to the shipped defaults, not rely on a one-way dirty flag.

---

## Acceptance Criteria

The implementation is complete only if all of the following are true:

1. The one-class model configuration is visible in Tab 4C using the same working widget style as existing regression/classification cards.
2. Default one-class grid-search behavior is unchanged.
3. Untouched models keep using `get_one_class_model_grids()` curated defaults.
4. Customized models use user-selected values.
5. Customization is per-model, not global.
6. Restoring a model card to its shipped default widget state restores curated-grid behavior for that model.
7. One-class Bayesian behavior is unchanged.
8. Existing one-class tests still pass.
9. New tests cover default preservation and per-model override behavior.

---

## Risks To Watch

### Risk 1: silent default-grid expansion
If the implementation directly turns default checkbox selections into combinations, it will silently expand the default one-class search space. This is the main risk and must be explicitly prevented.

### Risk 2: over-refactoring the GUI
Do not attempt a large model-config UI cleanup while doing this task. Keep the change surgical.

### Risk 3: accidental Bayesian coupling
Do not reuse the new GUI grid override path in Bayesian code.

### Risk 4: repeating the `_create_parameter_grid_control()` bug
Do not use that helper. Wire widgets directly to variables like the old working cards do.

### Risk 5: removing legacy one-class flat controls too aggressively
The safe path is:
- stop using them for the active one-class grid-search path
- keep backend compatibility
- clean up only what is clearly isolated and safe

---

## Suggested Commit Shape

Keep this as one focused implementation commit if possible, or two commits if needed:

1. backend + tests
2. GUI wiring

Do not mix in unrelated cleanup.

---

## Post-Implementation Documentation

After implementation and verification:

1. Append `docs/SESSION_LOG.md` with:
   - the default-preservation design
   - the reason untouched cards must map back to curated grids
   - any gotchas discovered during implementation

2. Update `docs/PROJECT_STATUS.md` with:
   - one-class model-config parity status
   - what is now user-editable in grid search
   - explicit note that Bayesian one-class remains separate and unchanged

3. If a code-review agent is used afterward, ask it to focus specifically on:
   - default-behavior preservation
   - accidental search-space expansion
   - one-class/Bayesian isolation
   - UI/back-end mismatch risk

---

## Final Instruction To Implementing Agent

This is a preservation-first task.

Create and use a dedicated git worktree before editing anything.

The success condition is **not** "more one-class flexibility at any cost."
The success condition is:

- the same shipped one-class defaults still run the same way,
- the user can now edit them in the same style as the rest of the app,
- and nothing else regresses.
