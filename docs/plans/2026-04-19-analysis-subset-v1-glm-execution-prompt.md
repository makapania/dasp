# Analysis Subset V1 - GLM Execution Prompt

Paste the prompt below directly into GLM 5.1.

---

You are working on the repo at:

`C:\Users\sponheim\git\dasp`

Your task is to implement a safe V1 of a metadata-driven `Analysis Subset` feature in a **new git worktree**.

## First: create and use a new worktree

Run exactly:

```bash
git status
git branch --show-current
git worktree add "C:\Users\sponheim\git\dasp-glm-analysis-subset" -b "glm/analysis-subset-v1"
```

Then do all work inside:

`C:\Users\sponheim\git\dasp-glm-analysis-subset`

Do not commit or push unless explicitly asked.

## Required repo context

Before editing:
1. Read `docs/PROJECT_STATUS.md`
2. Grep `docs/SESSION_LOG.md` for:
   - `active group`
   - `subset`
   - `validation`
   - `one-class`

## Feature goal

Implement a first usable metadata-driven `Analysis Subset` feature so a user can do things like:
- only use grasses in analysis
- only use trees in analysis

This subset must affect:
1. main analysis runs
2. validation-set creation
3. Model Development / refinement
4. training provenance and mismatch warnings

## Critical architecture decision

Build this on the existing internal runtime path:
- `active_group_filter`
- `active_indices`

Do **not** build this on Explore `sample_sets`.

For V1:
- keep the internal names `active_group_filter` and `active_indices`
- change the **user-facing terminology** to `Analysis Subset`
- support **one active subset at a time**

## Scope for this V1

In scope:
1. New `Analysis Subset` card in the Analysis tab
2. Metadata-only subset definition UI in the Analysis tab
3. Exact-match categorical multi-select for manageable metadata columns
4. Recompute active subset membership after relevant data/metadata changes
5. Persist subset provenance into `last_training_config`
6. Warn on subset mismatch in Model Development
7. Fail fast for one-class when the active subset removes all inlier samples
8. Add tests

Out of scope:
1. No named saved subset library
2. No “create one subset per unique value”
3. No Results-table grouping/greying by subset
4. No broad backend refactor
5. No Explore `sample_sets` training logic
6. No target-column filtering in the new Analysis-tab subset UI

## Key code locations to inspect first

Read these before changing code:

### Current active-group/runtime state
- `spectral_predict_gui_optimized.py:2727-2728`
- `spectral_predict_gui_optimized.py:10268-10282`
- `spectral_predict_gui_optimized.py:29696-29967`

### Current filtering order that must stay intact
- validation creation: `spectral_predict_gui_optimized.py:18977-19003`
- manual validation creation: `spectral_predict_gui_optimized.py:19189-19218`
- main analysis filtering: `spectral_predict_gui_optimized.py:24913-24942`
- refinement filtering: `spectral_predict_gui_optimized.py:33863-33918`

### Training metadata / mismatch / model transfer
- `spectral_predict_gui_optimized.py:25475-25486`
- `spectral_predict_gui_optimized.py:26181-26193`
- `spectral_predict_gui_optimized.py:28295-28314`
- `spectral_predict_gui_optimized.py:29988-30083`
- `spectral_predict_gui_optimized.py:30822-30869`

### Data viewer state / metadata alignment
- `spectral_predict_gui_optimized.py:29322-29357`
- `spectral_predict_gui_optimized.py:17181-17191`
- `src/spectral_predict/io.py:395-437`
- `src/spectral_predict/io.py:471-571`

### One-class inlier controls
- `spectral_predict_gui_optimized.py:6123-6135`
- `spectral_predict_gui_optimized.py:22104`

## Exact implementation requirements

### 1. Add Analysis Subset card to Analysis tab

In `spectral_predict_gui_optimized.py`, add a new card above the existing holdout validation card.

The card must contain:
1. Title: `Analysis Subset`
2. A short subtitle/caption explaining it restricts analysis to a metadata-defined cohort
3. A status label showing either:
   - `All samples`
   - or a human-readable subset summary
4. A `Define Subset...` button
5. A `Clear Subset` button
6. A short note that it affects:
   - analysis
   - validation-set creation
   - model refinement

`Clear Subset` must be disabled when no subset is active.

### 2. Rename user-facing “active group” terminology

Keep internal names, but rename visible UI text from active-group wording to subset wording.

At minimum update Data Viewer controls:
- `Set Active Group...` -> `Set Analysis Subset...`
- `Clear Group` -> `Clear Subset`
- label text should say `Analysis Subset:` rather than `Active Group:`

### 3. Add/Refactor a subset-definition dialog

Use the current active-group dialog as the starting point, but the Analysis-tab version must be **metadata-only**.

Allowed columns in the new Analysis Subset UI:
1. if `combined_metadata_df` exists, use those columns
2. else if `ref` exists, use reference columns excluding the current target column

Do **not** expose the target column in the new Analysis-tab subset UI.

The dialog must support:
1. column picker
2. preview count
3. Apply / Cancel
4. categorical multi-select when appropriate
5. fallback to existing condition/value UI otherwise

### 4. Add exact-match categorical multi-select

For manageable categorical columns, show a multi-select widget.

Rules:
1. Use a hard cap of 20 unique non-null values for V1
2. If the column exceeds that, fall back to the existing condition/value UI
3. Multi-select matching must use exact membership only
4. No regex
5. No substring semantics

Store the filter as:

```python
{
    "column": "PlantType",
    "condition": "in",
    "values": ["grass", "sedge"],
}
```

If no values are selected, warn and refuse apply.

### 5. Refactor matching logic so filter dicts are the source of truth

Add a helper like:

```python
def _compute_active_group_matches_from_filter(self, filter_def: dict) -> set:
    ...
```

Keep the old 4-arg helper only as a compatibility wrapper if needed.

Supported conditions must be:
1. `has value`
2. `==`
3. `!=`
4. `>`
5. `<`
6. `>=`
7. `<=`
8. `between`
9. `contains`
10. `in`

Behavior rules:
1. `has value` = non-null and, for strings, non-blank after strip
2. `contains` must use literal substring matching, not regex
3. `in` must use exact membership only
4. numeric comparisons should continue using numeric coercion
5. applying a filter that matches zero samples must warn and refuse activation

### 6. Treat `active_group_filter` as truth and `active_indices` as a cache

Add a helper like:

```python
def _refresh_active_group_indices(self):
    ...
```

Required behavior:
1. if no filter exists, set `active_indices = None`
2. if a filter exists, recompute `active_indices` from the filter definition
3. update subset UI after recompute

You must call this recompute helper at least:
1. after applying a subset
2. after clearing a subset
3. after Data Viewer edits are applied
4. after Data Viewer revert
5. after data load/reload once metadata is finalized

Important:
if a saved filter exists, do not trust a stale saved `active_indices` set blindly.

### 7. Preserve filtering order everywhere

Do not change the effective order of sample filtering.

Keep this order in all paths:
1. start with loaded data
2. apply active subset / `active_indices`
3. apply excluded spectra
4. apply validation holdout exclusion

Preserve this in:
1. validation creation
2. manual validation creation
3. main analysis
4. refinement

### 8. Add shared subset summary formatting

Add or extend a formatter so all UI and metadata uses the same subset summary.

Examples:
- `All samples`
- `PlantType in [grass, sedge]`
- `Species == tree`
- `HarvestYear between 2021 and 2023`

Use this same summary in:
1. Analysis-tab status label
2. Data Viewer subset label
3. training metadata
4. mismatch warnings
5. Model Development info text

### 9. Add subset provenance into training metadata

Add a helper like:

```python
def _get_analysis_subset_training_metadata(self) -> dict:
    ...
```

Add these keys to `self.last_training_config` in both existing storage sites:
1. `analysis_subset_active`
2. `analysis_subset_filter`
3. `analysis_subset_summary`
4. `analysis_subset_n_samples`

Semantics:
1. If no subset is active:
   - active = False
   - filter = None
   - summary = `All samples`
2. If subset is active:
   - active = True
   - filter = copy of filter dict
   - summary = formatted subset summary
   - n_samples = number matched by subset before exclusions and before validation removal

Do not remove existing training-config keys.

### 10. Extend Model Development mismatch warnings

Update `_validate_training_configuration()` so it compares the saved subset metadata against the current active subset.

If different, warn clearly using both summaries.

Example:

```text
Analysis subset: trained with PlantType in [grass], current state is PlantType in [tree]
```

Do not automatically apply the saved subset when loading a model.
Warn only.

### 11. Show saved subset summary in Model Development

When loading a result/model into Model Development, show the saved subset summary in the config/info text if available.

### 12. Add one-class guardrails

In both the main analysis path and refinement path:
1. after subset/exclusion/validation filtering
2. before launching one-class search/refinement

resolve the effective inlier label and verify that the filtered calibration data still contains at least one inlier sample.

If zero remain, block the run with a clear message like:

```text
The active Analysis Subset excludes all samples for inlier class 'grass'. Clear or change the subset, or choose a different inlier class.
```

### 13. Keep Data Viewer revert/snapshot safe

After restoring `active_group_filter` from Data Viewer state, recompute `active_indices` from the filter rather than trusting an old cached set.

## Testing requirements

Add tests. Use existing files if they fit well; otherwise create something focused like:

`tests/test_analysis_subset.py`

Minimum test coverage:
1. categorical `in` exact matching
2. `contains` uses literal substring matching, not regex semantics
3. `has value` excludes blank/whitespace-only strings
4. numeric `between`
5. summary formatter handles `in`
6. training metadata includes:
   - `analysis_subset_active`
   - `analysis_subset_filter`
   - `analysis_subset_summary`
   - `analysis_subset_n_samples`
7. one-class guard test where the active subset excludes all selected inlier samples and the run is blocked

If GUI-heavy tests are painful, extract only the pure matching/summary logic into a **tiny** helper module and unit test that. Do not create a big new abstraction layer.

## Manual verification checklist

After tests pass, manually verify:
1. load a dataset with metadata column `PlantType` containing `grass` and `tree`
2. open Analysis tab and confirm `Analysis Subset` card exists
3. define `PlantType in [grass]`
4. confirm status updates correctly
5. create validation set and confirm it uses only grass samples
6. run regression analysis and confirm reduced sample count is reflected
7. load a result into Model Development and confirm saved subset summary is shown
8. change current subset to `PlantType in [tree]`
9. load the older grass-trained result and confirm subset mismatch warning appears
10. run one-class with an inlier class excluded by the subset and confirm it blocks clearly
11. edit metadata in Data Viewer and confirm subset membership recomputes rather than staying stale

## Documentation requirements

Follow repo protocol:
1. append to `docs/SESSION_LOG.md` immediately if you confirm a non-obvious architecture gotcha or failure mode
2. update `docs/PROJECT_STATUS.md` at the end with what now works, known limitations, and deferred follow-ups

## Code quality constraints

1. prefer the smallest correct change
2. keep this as a GUI-layer cohort filter for V1
3. do not wire analysis logic to Explore `sample_sets`
4. keep user-facing terminology as `Analysis Subset`
5. use `apply_patch` for edits
6. run targeted tests after changes

## Definition of done

The task is done when:
1. Analysis tab has a visible Analysis Subset card
2. user can define a metadata-based subset there
3. categorical multi-select uses exact membership matching
4. analysis honors the subset
5. validation creation honors the subset
6. refinement honors the subset
7. subset provenance is stored in training metadata
8. refinement warns on subset mismatch
9. one-class runs fail fast if subset removes all inlier samples
10. relevant tests pass

Reference plan file in the repo:

`docs/plans/2026-04-19-analysis-subset-v1-glm-worktree-plan.md`

Use that plan as the detailed spec if you need more context while implementing.

---

When you finish, report:
1. files changed
2. key implementation decisions
3. tests run and results
4. any deferred limitations
