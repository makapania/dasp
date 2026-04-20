# Analysis Subset V1 - GLM Execution Prompt (v2, opencode edition)

You are working on the repo at:

`C:\Users\sponheim\git\dasp-glm-analysis-subset`

This is a git worktree that has **already been created for you** on branch `glm/analysis-subset-v1`. Do **not** run `git worktree add` — it exists. `cd` into it and do all work there.

Do not commit or push unless explicitly asked.

## Required repo context

Before editing:
1. Read `docs/PROJECT_STATUS.md`
2. Grep `docs/SESSION_LOG.md` for:
   - `active group`
   - `subset`
   - `validation`
   - `one-class`
3. Read the detailed design plan at `docs/plans/2026-04-19-analysis-subset-v1-glm-worktree-plan.md`. That plan is the authoritative spec; this document restates the essentials and adds five clarifications.

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

## Scope

In scope:
1. New `Analysis Subset` card in the Analysis tab (above the Holdout Validation card)
2. Metadata-only subset definition UI in the Analysis tab
3. Exact-match categorical multi-select for manageable metadata columns (≤ 20 unique non-null values)
4. Recompute active subset membership after relevant data/metadata changes
5. Persist subset provenance into `last_training_config`
6. Warn on subset mismatch in Model Development
7. Fail fast for one-class when the active subset removes all inlier samples
8. Add tests

Out of scope:
1. No named saved subset library
2. No "create one subset per unique value"
3. No Results-table grouping/greying by subset
4. No broad backend refactor
5. No Explore `sample_sets` training logic
6. No target-column filtering in the new Analysis-tab subset UI

## Key code locations to inspect first

(Line numbers were captured when the plan was written; re-verify before editing — the file may have shifted.)

### Current active-group/runtime state
- `spectral_predict_gui_optimized.py:2727-2728` — state fields `active_group_filter`, `active_indices`
- `spectral_predict_gui_optimized.py:10268-10282` — Data Viewer buttons/label
- `spectral_predict_gui_optimized.py:29696-29967` — existing active-group helper methods

### Filtering order that must remain consistent
- validation creation: `spectral_predict_gui_optimized.py:18977-19003`
- manual validation creation: `spectral_predict_gui_optimized.py:19189-19218`
- main analysis filtering: `spectral_predict_gui_optimized.py:24913-24942`
- refinement filtering: `spectral_predict_gui_optimized.py:33863-33918`

### Training metadata / mismatch / model transfer
- `spectral_predict_gui_optimized.py:25475-25486` — one-class `last_training_config` storage
- `spectral_predict_gui_optimized.py:26181-26193` — regression/classification/Bayesian storage
- `spectral_predict_gui_optimized.py:28295-28314` — Results → Model Development transfer
- `spectral_predict_gui_optimized.py:29988-30083` — mismatch checks in `_validate_training_configuration()`
- `spectral_predict_gui_optimized.py:30822-30869` — model load for refinement

### Data viewer state / metadata alignment
- `spectral_predict_gui_optimized.py:29322-29357` — snapshot/revert
- `spectral_predict_gui_optimized.py:17181-17191` — combined-format metadata path
- `src/spectral_predict/io.py:395-437` — reference-file loading
- `src/spectral_predict/io.py:471-571` — `align_xy` for separate reference metadata

### One-class inlier controls
- `spectral_predict_gui_optimized.py:6123-6135` — inlier-class combobox UI
- `spectral_predict_gui_optimized.py:22104` — inlier label resolved in analysis path

---

## FIVE REQUIRED CLARIFICATIONS (changes from the base plan)

These five items override or tighten the base plan. Follow them.

### C1. Prescribed helper module — NOT optional

Create **`src/spectral_predict/analysis_subset.py`** as a new module containing the pure matching and summary logic. This is **not optional**; it is required for V1.

The module must expose:

```python
def compute_matches(df, filter_def: dict) -> set:
    """Return set of df index values that match filter_def.
    df may be combined_metadata_df or a reference DataFrame.
    filter_def is the dict stored at self.active_group_filter.
    Returns an empty set if the column is missing, or the filter dict is malformed."""

def format_summary(filter_def: dict | None) -> str:
    """Return a human-readable one-liner.
    None or falsy -> 'All samples'.
    Otherwise format per the examples in the design plan."""

def is_categorical_column(series, max_unique: int = 20) -> bool:
    """True if the column should be shown as exact-value multi-select in the UI.
    Returns True when: non-numeric dtype, OR numeric dtype with <= max_unique
    unique non-null values. Otherwise False."""
```

The GUI's existing `_compute_active_group_matches(...)` and `_format_active_group_condition(...)` should become thin wrappers that delegate to these pure functions.

**Why:** `spectral_predict_gui_optimized.py` is ~45,000 lines. Testing matching/summary logic through Tk is painful and slow; testing pure functions against small DataFrames is trivial. It also tightly bounds the GUI diff.

All new tests for matching/summary behavior live in `tests/test_analysis_subset.py` and import from this module — no Tk required.

### C2. Missing-column safe handling in `_refresh_active_group_indices`

If a saved `active_group_filter` references a column that is no longer present in the current metadata (Data Viewer deletion, reload with a different file, etc.), the recompute helper must:

1. Set `self.active_indices = None`
2. Clear `self.active_group_filter = None`
3. Update UI to show `All samples` with a one-time warning log / status message like: `Analysis Subset cleared: column 'X' no longer present in metadata.`

Do **not** crash. Do **not** leave the filter active while silently matching zero.

The other "empty-after-recompute" case — column still present, filter still valid, but matches zero rows — should keep the filter definition and reflect zero matches in the UI (per the base plan). Only the missing-column case clears the filter.

### C3. `analysis_subset_n_samples` semantics when inactive

When no subset is active, set:

```python
{
    "analysis_subset_active": False,
    "analysis_subset_filter": None,
    "analysis_subset_summary": "All samples",
    "analysis_subset_n_samples": None,
}
```

Always `None`. Do not substitute total sample count. Keeps saved metadata clean and avoids bogus "mismatch" triggers on load.

### C4. No `apply_patch` requirement

Use whatever edit tool opencode provides. Ignore any reference to `apply_patch` in the base plan or this prompt. What matters is: smallest correct diffs, preserved indentation, no unrelated reformatting of surrounding code.

### C5. Verify line numbers before editing

The key code locations listed above were captured when the plan was written. Before editing any specific region, `grep` or re-read to confirm the anchor still matches. If it doesn't, find the current location via the surrounding symbol name (e.g., `_show_active_group_dialog`, `_validate_training_configuration`). Do not blindly trust a stale line number.

---

## Exact implementation requirements

### 1. Add Analysis Subset card to Analysis tab

Insert above the existing `Holdout Validation Set` card.

Card contents:
1. Title: `Analysis Subset`
2. Short subtitle/caption: restricts analysis to a metadata-defined cohort
3. Status label showing either `All samples` or the summary from `format_summary()`
4. `Define Subset...` button
5. `Clear Subset` button (disabled when no subset is active)
6. Short note that it affects: analysis, validation-set creation, model refinement

### 2. Rename user-facing "active group" terminology

Internal names stay. User-visible text changes:
- `Set Active Group...` → `Set Analysis Subset...`
- `Clear Group` → `Clear Subset`
- `Active Group:` → `Analysis Subset:`
- Any user-facing dialog titles, warnings, or messages mentioning "active group".

### 3. Subset-definition dialog (Analysis tab)

Refactor the current `_show_active_group_dialog()` so matching/apply logic is shared, but the Analysis-tab entry point is **metadata-only**:

Column source (in priority order):
1. If `combined_metadata_df` exists and is non-empty, use its columns.
2. Else if `ref` exists, use reference columns **excluding** the current target column.

Do **not** expose the target column in the Analysis-tab dialog.

Dialog must include: column picker, preview count, Apply, Cancel.
For columns passing `is_categorical_column(series)`: exact-value multi-select (Tk `Listbox` with `selectmode='multiple'` or similar).
Otherwise: fall back to the existing condition/value UI.

Empty selection in multi-select → warn and refuse apply.

### 4. Exact-match categorical multi-select

Store as:

```python
{
    "column": "PlantType",
    "condition": "in",
    "values": ["grass", "sedge"],
}
```

### 5. Filter dicts as source of truth

Supported conditions: `has value`, `==`, `!=`, `>`, `<`, `>=`, `<=`, `between`, `contains`, `in`.

Semantics:
- `has value`: non-null AND for strings, non-blank after `.strip()`
- `contains`: literal substring (`regex=False`), current case-sensitivity preserved
- `in`: exact membership (string-coerced for categorical columns), no regex, no substring
- numeric comparisons: continue using `pd.to_numeric(errors='coerce')` with existing string-equality fallback
- apply-path zero matches → warn and refuse activation

### 6. Treat `active_group_filter` as truth, `active_indices` as cache

Add `_refresh_active_group_indices(self)`. Required behavior:

1. If filter is falsy → `active_indices = None`, update UI, return.
2. If filter's `column` is missing from current metadata → clear the filter per C2, warn, update UI.
3. Otherwise recompute `active_indices` via `analysis_subset.compute_matches(...)`, keep the filter even if zero matches (per base plan), update UI.

Call this helper from at least:
1. After applying a subset
2. After clearing a subset
3. After Data Viewer edits are applied
4. After Data Viewer revert restores state
5. After data load/reload once `X`, `y`, `ref`, and `combined_metadata_df` are finalized

### 7. Preserve filtering order

Keep the existing order in all paths — validation creation, manual validation creation, main analysis, refinement:

1. start with loaded data
2. apply active subset (`active_indices`)
3. apply excluded spectra
4. apply validation holdout exclusion

### 8. Shared subset summary formatting

All five consumers use `analysis_subset.format_summary(...)`:
1. Analysis-tab status label
2. Data Viewer subset label
3. Training metadata `analysis_subset_summary`
4. Refinement/Model-Development mismatch warnings
5. Model Development info text

Examples:
- `All samples`
- `PlantType in [grass, sedge]`
- `Species == tree`
- `HarvestYear between 2021 and 2023`
- `Collector contains smith`

### 9. Subset provenance in `last_training_config`

Add helper `_get_analysis_subset_training_metadata(self) -> dict`. Add these keys in both existing storage sites (one-class path at ~25475, regression/classification/Bayesian path at ~26181):

- `analysis_subset_active`
- `analysis_subset_filter`
- `analysis_subset_summary`
- `analysis_subset_n_samples`

Semantics per C3 when inactive; when active: `active=True`, `filter=copy of dict`, `summary=format_summary(...)`, `n_samples=count matched by subset before exclusions and before validation removal`.

Additive only. Do not remove or rename existing training-config keys.

### 10. Model Development mismatch warnings

Extend `_validate_training_configuration()` to compare saved subset metadata against current runtime subset. On mismatch, append a warning like:

```text
Analysis subset: trained with PlantType in [grass], current state is PlantType in [tree]
```

Warn only. Do not auto-apply the saved subset when loading a model.

### 11. Show saved subset in Model Development

When loading a result/model into Model Development, display `analysis_subset_summary` in the config/info text if present.

### 12. One-class guardrails

In both main analysis and refinement one-class paths, after subset/exclusion/validation filtering and before launching the one-class search:

1. Resolve the intended inlier label.
2. Count remaining calibration samples for that label.
3. If zero, block the run with a clear message:

```text
The active Analysis Subset excludes all samples for inlier class 'grass'. Clear or change the subset, or choose a different inlier class.
```

### 13. Data Viewer revert safety

After restoring `active_group_filter` from snapshot, recompute `active_indices` from the filter via `_refresh_active_group_indices` rather than trusting a cached set.

---

## Testing requirements

Create `tests/test_analysis_subset.py`. Import from `src/spectral_predict/analysis_subset.py` — no Tk fixtures.

Minimum coverage:
1. categorical `in` exact matching
2. `contains` uses literal substring (not regex): confirm a value like `a.b` matches literal `a.b` and does not regex-match `axb`
3. `has value` excludes blank/whitespace-only strings
4. numeric `between` (inclusive on both ends, per current behavior — verify against existing implementation)
5. `format_summary` handles `in`, `==`, `between`, `contains`, and `None`/empty → `All samples`
6. `is_categorical_column` returns True for object dtype, True for numeric-with-few-uniques, False for numeric-with-many-uniques
7. missing-column case: `compute_matches` returns empty set when `filter_def['column']` is not in `df.columns`
8. training metadata includes all four `analysis_subset_*` keys with correct semantics (active and inactive). This one can be a small focused test that constructs the dict via the helper without spinning up the full GUI — if that's impossible, skip the GUI construction and leave a TODO comment noting the coverage gap.
9. one-class guard: a unit-level test that the filtered calibration pool's inlier-count check blocks when zero inliers remain for the chosen label. Again, prefer a pure-logic extraction if the existing GUI path makes direct testing painful.

Run targeted tests after changes:

```bash
pytest tests/test_analysis_subset.py -v
pytest tests/test_contamination_detection.py -v  # regression check — must still pass
```

---

## Manual verification checklist

(Run after tests pass, from inside the worktree.)

1. Load a dataset with metadata column `PlantType` containing `grass` and `tree`
2. Open Analysis tab — confirm `Analysis Subset` card exists
3. Define `PlantType in [grass]`
4. Status updates correctly
5. Create validation set — uses only grass samples
6. Run regression analysis — reduced sample count reflected in progress log
7. Load a result into Model Development — saved subset summary shown
8. Change current subset to `PlantType in [tree]`
9. Load the older grass-trained result — mismatch warning appears
10. Run one-class with an inlier class excluded by the subset — blocks clearly
11. Edit metadata in Data Viewer — subset membership recomputes, doesn't stay stale
12. In Data Viewer, delete the column referenced by the active subset — filter clears safely, status shows `All samples`, warning visible (this is the C2 case)

---

## Documentation

Follow repo protocol:
1. Append to `docs/SESSION_LOG.md` **immediately** when you confirm a non-obvious architecture gotcha or failure mode (do not wait until the end).
2. Update `docs/PROJECT_STATUS.md` at the end with what now works, known limitations, and deferred follow-ups.

## Code quality constraints

1. Prefer the smallest correct change.
2. Keep this as a GUI-layer cohort filter for V1.
3. Do not wire analysis logic to Explore `sample_sets`.
4. Keep user-facing terminology as `Analysis Subset`.
5. Run targeted tests after changes.
6. No reformatting of unrelated surrounding code.

## Definition of done

1. Analysis tab has a visible `Analysis Subset` card
2. User can define a metadata-based subset from that card
3. Categorical multi-select uses exact membership matching
4. Main analysis honors the subset
5. Validation creation honors the subset
6. Refinement honors the subset
7. Subset provenance is stored in training metadata
8. Refinement warns on subset mismatch
9. One-class runs fail fast if subset removes all inlier samples
10. Missing-column case (C2) clears safely
11. `tests/test_analysis_subset.py` passes; one-class regression suite still passes
12. Docs updated per project protocol

## When you finish, report

1. Files changed (full list with short description)
2. Key implementation decisions (especially any place you deviated from this prompt)
3. Tests run and results (command + pass/fail/counts)
4. Any deferred limitations
5. Anything ambiguous in the plan that you made a judgment call on
