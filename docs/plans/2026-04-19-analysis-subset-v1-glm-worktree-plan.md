# Analysis Subset V1 - GLM 5.1 Worktree Implementation Plan

**Audience:** GLM 5.1 working in a dedicated git worktree.

**Goal:** Implement a first usable metadata-driven `Analysis Subset` feature so the user can restrict analysis to a cohort such as grasses-only or trees-only using existing metadata. The subset must affect main analysis, validation-set creation, and Model Development/refinement, while also being recorded in training provenance and checked for mismatch later.

**Architecture:** Build this on the existing `active_group_filter` / `active_indices` path already used by analysis, validation, and refinement. Keep those internal names for V1 to minimize blast radius. Change the user-facing language to `Analysis Subset`. Do not build this on Explore `sample_sets`.

**Scope:**
- one active subset at a time
- new user-facing controls in the Analysis tab
- metadata columns only in the new Analysis Subset UI
- exact-match categorical multi-select for manageable categorical metadata columns
- additive provenance in `training_config`
- mismatch warnings in Model Development
- one-class guardrails when the active subset removes all inlier samples

**Out of scope for V1:**
- no saved library of named subsets
- no "create one subset per unique value"
- no Results-table regrouping/greying by subset
- no migration to a brand-new subset subsystem
- no Explore `sample_sets` analysis wiring
- no broad backend refactor of `search.py` / `unified_bayesian.py`

---

## Workflow Requirement

Before implementation, create a dedicated git worktree.

Reason:
- the GUI file is large and high churn
- this feature touches multiple stateful GUI paths
- the user explicitly asked for a new worktree for GLM
- isolating the change makes it easier to test subset-sensitive behavior without unrelated local edits

Use this exact flow from the main repo root:

```bash
git status
git branch --show-current
git worktree add "C:\Users\sponheim\git\dasp-glm-analysis-subset" -b "glm/analysis-subset-v1"
```

Then work only inside:

```text
C:\Users\sponheim\git\dasp-glm-analysis-subset
```

Before editing anything in that worktree:
1. Read `docs/PROJECT_STATUS.md`.
2. Grep `docs/SESSION_LOG.md` for `active group`, `subset`, `validation`, and `one-class`.
3. Verify the new worktree is on branch `glm/analysis-subset-v1`.
4. Do not commit or push unless the user explicitly asks.

---

## Critical Invariants

These are mandatory.

1. **Use the existing active-group runtime path.**
   Preserve the current filtering behavior where `active_indices` is applied before exclusions and validation-set removal.

2. **Do not use Explore `sample_sets` for training cohort logic.**
   `sample_sets` are exploratory/manual labels. They are not the correct architecture for this feature.

3. **Do not expose the target column in the new Analysis Subset UI.**
   V1 subset creation from the Analysis tab must be metadata-only.

4. **Keep one active subset at a time.**
   Do not introduce multiple simultaneously active subset definitions.

5. **Treat the filter definition as truth.**
   `active_group_filter` is the durable definition. `active_indices` is a cache that must be recomputed when data or metadata changes.

6. **Categorical multi-select must use exact membership matching.**
   No regex. No substring semantics for multi-select.

7. **Do not silently change the user's active subset when loading results into Model Development.**
   Only warn on mismatch.

8. **One-class must fail fast when the subset removes all inlier samples.**
   Do not let the run proceed to a confusing downstream failure.

9. **Keep the change minimal.**
   Prefer small helpers and careful reuse over a new abstraction layer.

---

## Key Existing Code Paths

Inspect these first and keep them aligned:

### GUI state and current active-group feature
- `spectral_predict_gui_optimized.py:2727-2728`
  Current state fields:
  - `self.active_group_filter = None`
  - `self.active_indices = None`

- `spectral_predict_gui_optimized.py:10268-10282`
  Current Data Viewer buttons/label for active group.

- `spectral_predict_gui_optimized.py:29696-29967`
  Current helper methods:
  - `_get_active_group_columns()`
  - `_get_column_series()`
  - `_compute_active_group_matches()`
  - `_apply_active_group()`
  - `_clear_active_group()`
  - `_update_active_group_ui()`
  - `_format_active_group_condition()`
  - `_show_active_group_dialog()`

### Filtering order that must remain consistent
- `spectral_predict_gui_optimized.py:18977-19003`
  Validation-set creation uses active subset first.

- `spectral_predict_gui_optimized.py:19189-19218`
  Manual validation-set creation also uses active subset first.

- `spectral_predict_gui_optimized.py:24913-24942`
  Main analysis applies active subset, then exclusions, then validation removal.

- `spectral_predict_gui_optimized.py:33863-33918`
  Refinement path applies active subset, then exclusions, then validation removal.

### Training metadata and model transfer
- `spectral_predict_gui_optimized.py:25475-25486`
  One-class `last_training_config` storage.

- `spectral_predict_gui_optimized.py:26181-26193`
  Regression/classification/Bayesian `last_training_config` storage.

- `spectral_predict_gui_optimized.py:28295-28314`
  Results to Model Development transfer.

- `spectral_predict_gui_optimized.py:29988-30083`
  Current mismatch checks in `_validate_training_configuration()`.

- `spectral_predict_gui_optimized.py:30822-30869`
  Model load for refinement.

### Data-viewer state handling
- `spectral_predict_gui_optimized.py:29322-29357`
  Current snapshot/revert behavior for `active_group_filter` and `active_indices`.

### Metadata loading/alignment behavior
- `spectral_predict_gui_optimized.py:17181-17191`
  Combined-format metadata path using `combined_metadata_df`.

- `src/spectral_predict/io.py:395-437`
  Reference-file loading path.

- `src/spectral_predict/io.py:471-571`
  `align_xy()` behavior for separate reference metadata.

### One-class inlier controls
- `spectral_predict_gui_optimized.py:6123-6135`
  Inlier-class combobox UI.

- `spectral_predict_gui_optimized.py:22104`
  Inlier label resolved in analysis path.

---

## High-Level Design

### User-facing behavior

The user should be able to:
1. Open the Analysis tab.
2. See an `Analysis Subset` card.
3. Click `Define Subset...`.
4. Pick a metadata column.
5. If the column is categorical, choose one or more exact values such as `grass` and `sedge`.
6. Apply the subset.
7. Run analysis, create validation sets, and refine models against only that cohort.

### Internal behavior

V1 continues using:
- `active_group_filter` for the selected subset definition
- `active_indices` for the currently resolved sample IDs

But:
- the UI terminology becomes `Analysis Subset`
- the filter definition gains support for categorical multi-select via a new `condition: "in"`
- `active_indices` must be recomputed from the filter definition after data/metadata edits or reloads

### New filter dict shape for multi-select

Existing single-rule shape stays valid:

```python
{
    "column": "Species",
    "condition": "==",
    "value": "tree",
    "value2": "",
}
```

New categorical multi-select shape for V1:

```python
{
    "column": "PlantType",
    "condition": "in",
    "values": ["grass", "sedge"],
}
```

Do not rename the state field. Keep `active_group_filter`.

---

## Files To Modify

### Primary implementation files
- `spectral_predict_gui_optimized.py`

### Optional tiny helper file if testing pressure becomes too high
- `src/spectral_predict/analysis_subset.py`

Only create this helper module if needed to unit test matching/summary logic cleanly. Keep it very small.

### Tests
- update existing test files if a good home already exists
- otherwise create a focused new file such as:
  - `tests/test_analysis_subset.py`

### Documentation to update after implementation
- `docs/SESSION_LOG.md`
- `docs/PROJECT_STATUS.md`

---

## Task 1 - Create Worktree And Baseline

**Files:** none modified yet.

1. From `C:\Users\sponheim\git\dasp`, create the new worktree exactly as specified above.
2. In the new worktree, run:

```bash
git status
git branch --show-current
```

3. Confirm the branch is `glm/analysis-subset-v1`.
4. Read `docs/PROJECT_STATUS.md`.
5. Grep `docs/SESSION_LOG.md` for:

```text
active group
subset
validation
one-class
```

6. Before code edits, read the key code locations listed in this plan.

---

## Task 2 - Add Analysis Subset Card To Analysis Configuration

**File:** `spectral_predict_gui_optimized.py`

### Objective
Add a first-class Analysis Subset control block to the Analysis tab.

### Placement
Insert the card above the existing `Holdout Validation Set` card.

### Required controls
Add a new card with:
1. Title: `Analysis Subset`
2. Subtitle: something short such as `Restrict analysis to a metadata-defined cohort`
3. A status label showing either:
   - `All samples`
   - or a formatted subset summary
4. A `Define Subset...` button
5. A `Clear Subset` button
6. A caption explaining that the subset affects:
   - analysis
   - validation-set creation
   - model refinement

### Implementation notes
1. Reuse the existing `_create_card(...)` pattern in the Analysis tab.
2. Add instance fields for the new Analysis-tab status label and buttons.
3. Wire `Define Subset...` to the new/refactored subset dialog method.
4. Wire `Clear Subset` to the existing clear logic or its refactored equivalent.
5. `Clear Subset` must be disabled when no subset is active.

---

## Task 3 - Make User-Facing Terminology Consistent

**File:** `spectral_predict_gui_optimized.py`

### Objective
Rename visible `active group` terminology to `Analysis Subset` where the user sees it.

### Required updates
1. In the Data Viewer toolbar, rename:
   - `Set Active Group...` -> `Set Analysis Subset...`
   - `Clear Group` -> `Clear Subset`
2. Update the Data Viewer label text so it says `Analysis Subset:` instead of `Active Group:`.
3. Update warnings/dialog titles/messages that mention active group if they are user-facing.

### Important
Do not rename internal variable names or helper names unless absolutely necessary.
Internal names can stay as `active_group_*` for V1.

---

## Task 4 - Refactor Matching Logic To Accept Filter Dicts

**File:** `spectral_predict_gui_optimized.py`

### Objective
Make filter evaluation explicit and extensible, with one source of truth.

### Required helper changes
Add a new helper near the current active-group methods:

```python
def _compute_active_group_matches_from_filter(self, filter_def: dict) -> set:
    ...
```

Keep the current 4-argument helper for compatibility if needed, but implement it as a thin adapter that builds a filter dict and delegates to the new helper.

### Required behavior
Support these conditions:
1. `has value`
2. `==`
3. `!=`
4. `>`
5. `<`
6. `>=`
7. `<=`
8. `between`
9. `contains`
10. `in`  (new categorical multi-select)

### Exact semantics
1. `has value`
   - matches `notna()`
   - for strings, also require `str.strip() != ''`
2. `contains`
   - use literal substring matching, not regex
   - use `regex=False`
   - case-insensitive is fine if the current behavior is case-insensitive
3. `in`
   - exact membership only
   - convert comparison values to strings for string/categorical columns
   - do not use regex
4. numeric comparisons
   - continue using numeric coercion with `pd.to_numeric(errors='coerce')`
   - preserve current fallback behavior for string equality if numeric conversion fails

### Zero-match behavior
The apply path must warn and refuse to activate a subset if it matches zero samples.

---

## Task 5 - Add A Recompute Helper And Use It Consistently

**File:** `spectral_predict_gui_optimized.py`

### Objective
Prevent stale `active_indices` after data or metadata changes.

### Required helper
Add a helper such as:

```python
def _refresh_active_group_indices(self):
    ...
```

### Required behavior
1. If `self.active_group_filter` is falsy:
   - set `self.active_indices = None`
   - update UI
   - return
2. If a filter exists:
   - recompute `self.active_indices` from the filter definition
   - if the recomputed set is empty, keep the filter definition but show the UI clearly reflects zero matches
   - do not silently clear the definition unless there is a strong reason

### Minimum call sites
Call this helper at least in these places:
1. after applying a subset
2. after clearing a subset
3. after Data Viewer edits are applied
4. after Data Viewer revert restores state
5. after data load/reload completes and `X`, `y`, `ref`, and `combined_metadata_df` are finalized

If additional edit paths modify metadata columns, hook them too.

### Important note
When reverting Data Viewer state, if a saved `active_group_filter` exists, do not trust a stale saved `active_indices` blindly. Recompute it.

---

## Task 6 - Create A Dedicated Analysis Subset Dialog For The Analysis Tab

**File:** `spectral_predict_gui_optimized.py`

### Objective
Provide a user-friendly subset-definition dialog from the Analysis tab.

### Recommended approach
Refactor the current `_show_active_group_dialog()` carefully so the matching/apply logic is shared, but the new Analysis-tab experience is metadata-focused.

### Required column source for the new Analysis Subset dialog
Use metadata columns only:
1. If `combined_metadata_df` exists and is not empty, use `combined_metadata_df.columns`.
2. Else if `ref` exists, use reference columns excluding the current target column.
3. Do not include the target column in the new Analysis Subset dialog.

### Required UI behavior
1. Column picker.
2. A preview count label.
3. If the selected column is a manageable categorical column, show exact-value multi-select UI.
4. Otherwise, show the current condition + value/value2 controls.
5. Buttons:
   - `Preview`
   - `Apply`
   - `Cancel`

### Categorical detection rule for V1
Treat a column as categorical if:
1. it is non-numeric, or
2. after dropping nulls it has a small unique set of values that is reasonable to render

Use a hard cap of 20 unique values for the multi-select widget in V1.

If unique non-null values exceed 20, fall back to the existing condition/value UI.

### Multi-select implementation notes
Use a Tkinter widget that is simple and stable in this codebase, e.g. a `Listbox` in `selectmode='multiple'` inside a frame with a scrollbar if needed.

### Filter dict for multi-select apply
When the user chooses values from the categorical list, build:

```python
{
    "column": selected_column,
    "condition": "in",
    "values": selected_values,
}
```

### Empty selection behavior
If the user opens a categorical multi-select column but selects no values, warn and refuse apply.

---

## Task 7 - Preserve And Reuse Existing Apply/Clear Logic

**File:** `spectral_predict_gui_optimized.py`

### Objective
Avoid duplicating how active subsets are stored and propagated.

### Required changes
1. Update `_apply_active_group(...)` or add a thin new method that can accept a filter dict directly.
2. Ensure applying a subset still:
   - warns that current validation set will be reset
   - clears validation set if the user confirms
   - stores the filter definition in `self.active_group_filter`
   - refreshes `self.active_indices`
   - updates both Analysis-tab and Data Viewer subset UI
   - repopulates the Data Viewer
3. Ensure clearing a subset still:
   - warns before resetting validation if a validation set exists
   - clears `self.active_group_filter`
   - refreshes indices/UI
   - repopulates the Data Viewer

---

## Task 8 - Add Shared Formatting For Subset Summary

**File:** `spectral_predict_gui_optimized.py`

### Objective
Use one consistent human-readable subset description everywhere.

### Required helper behavior
Extend or replace `_format_active_group_condition()` so it can format all supported filter types, including `in`.

### Example outputs
- `All samples`
- `PlantType in [grass, sedge]`
- `Species == tree`
- `HarvestYear between 2021 and 2023`
- `Collector contains smith`

### Required UI consumers
Use the same summary text in:
1. the Analysis Subset card status
2. the Data Viewer label/status text
3. training metadata summary
4. refinement mismatch warning text
5. Model Development info text when available

---

## Task 9 - Keep Filtering Order Identical In Analysis, Validation, And Refinement

**File:** `spectral_predict_gui_optimized.py`

### Objective
Preserve the current semantics.

### Required order
For all major paths, the effective calibration pool must continue to be:
1. all loaded samples
2. restricted by active subset (`active_indices`) if present
3. restricted by `excluded_spectra`
4. restricted by `validation_indices` when validation is enabled

### Do not change these existing sections except as needed for subset metadata/validation
1. `_create_validation_set()`
2. `_create_manual_validation_set()`
3. `_run_analysis_thread()` filtering section
4. `_run_refined_model_thread()` filtering section

The job here is to preserve behavior while making the subset definition more robust and more visible.

---

## Task 10 - Add Subset Provenance To last_training_config

**File:** `spectral_predict_gui_optimized.py`

### Objective
Record the active subset used for training so later refinement can warn correctly.

### Required helper
Add a helper that returns current subset provenance, for example:

```python
def _get_analysis_subset_training_metadata(self) -> dict:
    ...
```

### Required keys
Add these keys to `self.last_training_config` in both existing storage sites:
1. `analysis_subset_active`
2. `analysis_subset_filter`
3. `analysis_subset_summary`
4. `analysis_subset_n_samples`

### Required semantics
1. If no subset is active:
   - `analysis_subset_active = False`
   - `analysis_subset_filter = None`
   - `analysis_subset_summary = "All samples"`
   - `analysis_subset_n_samples = total loaded samples` or `None` if that is cleaner; choose one and be consistent
2. If a subset is active:
   - `analysis_subset_active = True`
   - `analysis_subset_filter` is a copy of the filter dict
   - `analysis_subset_summary` is the formatted summary string
   - `analysis_subset_n_samples` is the number of samples matched by the subset before exclusions and before validation removal

### Important
This metadata is additive. Do not remove or rename existing training-config keys.

---

## Task 11 - Carry Subset Provenance Into Model Development

**File:** `spectral_predict_gui_optimized.py`

### Objective
Make the saved subset visible and available during refinement.

### Required changes
1. In the Results -> Model Development transfer path, ensure the `training_config` attached to `model_config` includes the new subset metadata.
2. Update the Model Development info text to display the saved subset summary if present.
3. Do not make the display dependent on current active subset state; show what the model/result was originally trained with.

---

## Task 12 - Extend Training-Configuration Mismatch Warnings

**File:** `spectral_predict_gui_optimized.py`

### Objective
Warn the user when the current active subset differs from the one used to generate the loaded model/result.

### Required changes
Extend `_validate_training_configuration()` with subset checks.

### Required comparison behavior
Compare:
1. saved `analysis_subset_active`
2. saved `analysis_subset_summary`
3. optionally saved `analysis_subset_filter`

against the current UI/runtime subset state.

### Warning behavior
If the saved subset differs from the current subset, append a warning similar to existing mismatch warnings.

The warning should explicitly show:
1. original subset summary
2. current subset summary

Example:

```text
Analysis subset: trained with PlantType in [grass], current state is PlantType in [tree]
```

### Important
Do not automatically apply the saved subset when loading a model.
Warn only.

---

## Task 13 - Add One-Class Guardrails

**File:** `spectral_predict_gui_optimized.py`

### Objective
Fail fast when the active subset removes all inlier samples.

### Main analysis path
In the one-class analysis path, after filtering the calibration pool by:
1. active subset
2. excluded spectra
3. validation holdout removal

and before launching one-class search:
1. resolve the intended inlier label
2. count how many calibration samples remain for that inlier label
3. if zero, block the run with a clear error message

### Refinement path
Do the same in the one-class refinement / Model Development path.

### Error message requirement
The message must explain that the active Analysis Subset excludes all samples for the selected inlier class.

Example:

```text
The active Analysis Subset excludes all samples for inlier class 'grass'. Clear or change the subset, or choose a different inlier class.
```

---

## Task 14 - Keep Data-Viewer Snapshot/Revert Safe

**File:** `spectral_predict_gui_optimized.py`

### Objective
Ensure reverting Data Viewer changes restores a consistent subset definition and recomputed membership.

### Required behavior
1. It is fine to continue snapshotting `active_group_filter`.
2. If `active_indices` is snapshotted, treat it as informational only.
3. After revert restores `active_group_filter`, recompute `active_indices` from the filter.
4. Update the Analysis-tab and Data Viewer subset UI afterward.

---

## Task 15 - Add Tests

**Files:**
- update existing tests if there is an obvious home
- otherwise create `tests/test_analysis_subset.py`

### Objective
Cover the new matching logic, provenance, and one-class guardrails.

### Minimum required tests

#### A. Pure matching tests
Add tests for:
1. categorical `in` exact matching
2. `contains` uses literal substring matching, not regex semantics
3. `has value` excludes blank strings and whitespace-only strings
4. numeric `between`

#### B. Summary-format test
Add a test that the summary formatter handles the new `in` condition correctly.

#### C. Training-config provenance test
Add a test that after applying a subset, the generated training metadata includes:
1. `analysis_subset_active`
2. `analysis_subset_filter`
3. `analysis_subset_summary`
4. `analysis_subset_n_samples`

#### D. One-class guard test
Add a test where:
1. an active subset is set
2. the chosen one-class inlier label has zero remaining samples in that subset
3. the run is blocked with the expected error path/message

### Testing strategy guidance
If GUI-level tests become too heavy, extract only the pure matching and summary helpers into a tiny helper module and unit test those directly.

Do not create a large new architecture just for testing.

---

## Task 16 - Manual Verification Checklist

Run this manually after tests pass.

1. Load a dataset with a metadata column like `PlantType` containing at least `grass` and `tree`.
2. Open the Analysis tab and verify the new `Analysis Subset` card appears.
3. Define subset `PlantType in [grass]`.
4. Confirm the Analysis-tab status text updates to the correct summary.
5. Confirm the Data Viewer label also reflects the subset.
6. Create a validation set and confirm the selected pool comes only from grass samples.
7. Run a regression analysis and confirm the progress log shows the reduced sample count after subset filtering.
8. Load a result into Model Development and confirm the saved subset summary is shown.
9. Change the current subset to `PlantType in [tree]`.
10. Load the older grass-trained result again and confirm a mismatch warning appears.
11. Run a one-class analysis where the active subset excludes all chosen inlier samples and confirm the run is blocked with a clear message.
12. Edit relevant metadata in the Data Viewer, then confirm the active subset membership is recomputed instead of silently staying stale.

---

## Task 17 - Documentation Updates

### During implementation
Append to `docs/SESSION_LOG.md` immediately when you confirm any of these or discover something similar:
- a non-obvious subset/recompute gotcha
- a Model Development mismatch edge case
- a one-class interaction detail that could surprise future sessions

### At the end
Update `docs/PROJECT_STATUS.md` with:
1. what now works in Analysis Subset V1
2. known limitations deferred from this plan
3. any follow-up work still recommended

---

## Suggested Minimal Helper Inventory

The implementation should probably end up with helpers close to this size/scope:

1. `_compute_active_group_matches_from_filter(self, filter_def)`
2. `_refresh_active_group_indices(self)`
3. `_get_analysis_subset_training_metadata(self)`
4. one small formatter extension for subset summary

Optional only if testing becomes difficult:
5. tiny helper module for pure subset matching/formatting logic

Do not create a larger subsystem than this for V1.

---

## Acceptance Criteria

The task is done when all of the following are true:

1. The Analysis tab has a visible `Analysis Subset` card.
2. The user can define a metadata-based subset from that card.
3. Categorical multi-select uses exact membership matching.
4. Main analysis honors the subset.
5. Validation-set creation honors the subset.
6. Model Development/refinement honors the subset.
7. `last_training_config` stores subset provenance.
8. Model Development warns when the current subset differs from the saved subset.
9. One-class runs fail fast if the active subset removes all inlier samples.
10. Tests covering the new logic pass.
11. Docs are updated per project protocol.

---

## Explicit Non-Goals To Recheck Before Finishing

Before considering the task complete, verify you did not accidentally add any of these:

- named saved subset library
- bulk subset generation by unique values
- target-column filtering in the new Analysis tab UI
- Explore `sample_sets` training behavior
- automatic application of saved subset state during Model Development load
- large backend refactor unrelated to this GUI-layer cohort filter

If any of those slipped in, scale the change back before presenting it.
