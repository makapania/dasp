# Analysis Subset V1.1 — UX Improvements Plan

**Status:** plan, not yet dispatched
**Date:** 2026-04-19
**Target implementer:** GLM 5.1 via opencode, in the existing worktree `C:\Users\sponheim\git\dasp-glm-analysis-subset` (branch `glm/analysis-subset-v1`) — V1 is still uncommitted there.

---

## Goal

Fix three concrete UX problems with the current (uncommitted) V1 Analysis Subset feature. Backend semantics, provenance, one-class guardrails, filter order — all stay the same. This is a GUI-only refactor plus one tiny pure-helper addition.

## The three issues

1. **Placement.** The card currently lives in Tab 4E Validation at `spectral_predict_gui_optimized.py:13533`. That's buried. Move it to the **top of Tab 4A Basic Settings**, so it's visible when the user starts configuring a run. Not inside Data Management (the user doesn't use that tab, and it may be removed later). Not inside Tab 4E Validation (its current home — too buried; "Validation" is a confusing name for a global subset).

2. **Continuous-variable support.** Users need to express conditions like `Carbon < 7.0` or `HarvestYear between 2021 and 2023`. The backend already supports `<`, `<=`, `>`, `>=`, `between`, and numeric-coerced `==`/`!=`. The current dialog hides or half-hides these behind the `>20-unique` fallback. They must become first-class and obvious.

3. **Categorical dropdowns.** The current dialog forces users to remember exact values. Replace the bare text entry for categorical `==`/`!=`/`contains` with a **ttk.Combobox** populated by the column's actual unique values — editable, so the user can pick *or* type. The existing Listbox multi-select stays for the `in` condition.

## Out of scope

- No backend changes to filter provenance, training-config metadata, or mismatch warnings.
- No change to filter order (data → subset → excluded → validation) in analysis / validation / refinement paths.
- No change to one-class guardrail logic.
- No tests for the Tk rendering itself — pure helper coverage only.
- No support for `OR` or `AND` across multiple columns — V1 still has one filter at a time.
- No saved-subset library, no bulk subset generation, no target-column exposure.

## Architecture

Backend (`src/spectral_predict/analysis_subset.py`) gets one small pure addition; GUI dialog gets rewritten; one new subtab method is added; the existing Tab 4E placement of the card is removed. Internal state fields (`active_group_filter`, `active_indices`) and all helper method names are unchanged — existing call sites continue to work without modification.

---

## Tasks

### T1. Relocate the Analysis Subset card

**File:** `spectral_predict_gui_optimized.py`

1. **Remove** the card creation block at roughly line 13533–13553 inside `_create_tab4e_validation`. That block is: the section header, the `_create_card(title="Analysis Subset", ...)` call, the status label assignment, and the define/clear button wiring that currently lives inside Tab 4E. Leave the rest of Tab 4E untouched.

2. **Insert** the Analysis Subset card as the **first** section inside `_create_tab4a_basic_settings` (starting near line 10740). Put it above whatever is currently the first content block of that subtab. Use the same `_create_section_header` + `_create_card` pattern as the V1 implementation — same state field names so every downstream reference keeps working:
   - `self.analysis_subset_status_label` (ttk.Label) — feed it only the `format_summary(...)` output, no count prefix
   - `self.analysis_subset_clear_btn` (ttk.Button, disabled by default)
   - A "Define Subset..." button wired to `self._show_active_group_dialog`
   - A short caption noting that the subset affects analysis, validation-set creation, and model refinement

   Card title: `Analysis Subset`. Card subtitle: `Restricts analysis to a metadata-defined cohort` (match V1 copy style — "Restricts" with the "s").

3. Do **not** create a new subtab. Do not rename any existing subtab methods. Internal subtab structure of Tab 4 (4A Basic Settings, 4B Variable Selection, 4C Model Config, 4D Ensemble Methods, 4E Validation) stays the same; only content within 4A and 4E changes.

4. Confirm via targeted grep after the move that no orphan references to the card's widgets or helpers remain inside `_create_tab4e_validation`, and that Basic Settings visually renders the card on top with sensible padding.

**Alternatives the user mentioned but not chosen in this plan:** (a) new dedicated subtab after Basic Settings, (b) new dedicated subtab after Validation. If the embedded-in-4A layout feels cramped when the card grows, we can spin it out into its own subtab in a follow-up — purely a re-parenting change.

### T2. Type-aware subset dialog

**File:** `spectral_predict_gui_optimized.py` — rewrite `_show_active_group_dialog(self)` at line 30481.

Replace the current body with a dialog that adapts its value-entry widgets to the picked column's type. High-level flow:

1. Build the modal dialog with title `Set Analysis Subset` and the existing size/modality behavior (preserve the parent-centered, grab-set, wait-window pattern).

2. **Column picker.** A `ttk.Combobox` listing metadata columns from (in priority order) `self.combined_metadata_df.columns` else `self.ref.columns` with the current target column excluded. Bind its `<<ComboboxSelected>>` event to a render callback.

3. **Render callback `_render_condition_controls(column_name)`.** Wipes and re-populates a child frame. Detects the column's type by calling `classify_column_type(series)` (the new pure helper in T3a — do NOT reimplement the logic inline).

   Widget lifecycle: destroy only children of the condition-controls sub-frame; keep modality on the `Toplevel`. Use **fresh `StringVar`s on each render** — do NOT reuse variables across renders, and do NOT cache widget references in dialog-level closures. Apply/Preview handlers should read values from the currently-live widgets via the sub-frame's `winfo_children()` walk or from freshly-bound StringVars captured in a per-render dict. This prevents stale-reference bugs when the user switches columns/operators repeatedly.

4. **Numeric branch.** Operator combobox with values `['==', '!=', '<', '<=', '>', '>=', 'between', 'has value']`. Single value `ttk.Entry` for all operators except `between` (which shows two entries labelled "Min" and "Max") and `has value` (which shows no value field). Show the column's observed numeric range ("min: 0.12, max: 9.87") as a small caption hint under the entry — builds on `series.min()` / `series.max()` from `pd.to_numeric(series, errors='coerce').dropna()`.

5. **Categorical branch.** Operator combobox with values `['==', '!=', 'in', 'contains', 'has value']`. Value widget depends on operator:

   - `==`, `!=`: `ttk.Combobox` with `values=get_unique_non_null_values(series)`, `state='normal'` (editable, so user can pick OR type).
   - `in`: Listbox with `selectmode='extended'` populated with `get_unique_non_null_values(series)`, inside a frame with a vertical scrollbar. Same widget as V1.
   - `contains`: plain `ttk.Entry` (free-text literal substring).
   - `has value`: no value field.

6. **Preview count label + Preview button.** Below the operator/value widgets: `ttk.Label` showing `Matches: N of M samples` (or `Select a value to preview` before inputs are valid). Recompute only when the explicit `Preview` button is clicked. Do NOT auto-recompute on every keystroke or operator change — that adds event-handler surface. The current V1 dialog already uses an explicit Preview button pattern; V1.1 preserves that specifically.

   Important: typed text in a `ttk.Combobox(state='normal')` does NOT emit `<<ComboboxSelected>>` — only dropdown-pick emits it. That's fine because we're using a Preview button rather than event-driven live preview. No `FocusOut` binding needed.

7. **Apply / Cancel.** Apply calls `self._apply_active_group(...)` with the filter dict produced by `build_filter_dict(...)` (T3c). Apply still refuses when selected values resolve to zero matches. Cancel closes without state change.

8. **Filter dict shape.** Built by `build_filter_dict(column, column_kind, operator, raw_values)` (T3c). Shapes:
   - `in` → `{"column": col, "condition": "in", "values": [...]}`
   - `between` → `{"column": col, "condition": "between", "value": min, "value2": max}`
   - `has value` → `{"column": col, "condition": "has value"}`
   - everything else → `{"column": col, "condition": cond, "value": v, "value2": ""}`

   (These match what `compute_matches` already consumes — no backend change needed.)

9. **Keyboard + modal behavior (explicitly new requirements, not "existing convention").** V1's dialog uses `grab_set()` without `wait_window()` and has no Return-key binding. V1.1 adds: `grab_set()` + `wait_window()` to make the dialog properly modal-blocking; Enter in the single-line numeric entry triggers Preview; Enter in Combobox does NOT trigger anything while the dropdown is open (default Combobox behavior), but triggers Preview when the dropdown is closed. Initial focus lands on the column picker.

### T3. Two new pure helpers in `analysis_subset.py`

**File:** `src/spectral_predict/analysis_subset.py`

Codex specifically called out that type detection is the riskiest UI-logic fork and must live in the pure module so it is testable and stable. Add **both** of these helpers:

#### T3a. `classify_column_type`

```python
def classify_column_type(series, numeric_threshold: float = 0.9) -> str:
    """Return 'numeric' or 'categorical' for subset-UI purposes.

    Rule (explicit — do not substitute anything else):
    1. Drop nulls. If the remaining series is empty, return 'categorical'.
    2. If dtype is already numeric (pd.api.types.is_numeric_dtype), return 'numeric'.
    3. Otherwise, for each non-null value: strip surrounding whitespace, then if
       the value ends with a single trailing '%' strip that too. Attempt
       pd.to_numeric on the cleaned string.
    4. If >= numeric_threshold (default 0.9) of non-null values parse
       cleanly as numeric, return 'numeric'. Otherwise return 'categorical'.

    Examples:
    - [1, 2, 3] -> 'numeric' (native dtype)
    - ['1', '2', '3'] -> 'numeric' (all strings parse)
    - ['12.5%', '7.1%', '3.0%'] -> 'numeric' (trailing %, all parse)
    - ['2021', '2022', 'unknown'] -> 'categorical' (only 2/3 = 0.67 < 0.9)
    - ['grass', 'tree', 'sedge'] -> 'categorical' (none parse)
    """
```

The `numeric_threshold` is tunable but the default of 0.9 is the real rule; don't expose the threshold in the GUI.

#### T3b. `get_unique_non_null_values`

```python
def get_unique_non_null_values(series, max_count: int = 200) -> list:
    """Return sorted unique non-null values from a pandas Series.

    For the subset dialog's categorical dropdown / multi-select population.
    Caps at max_count for UI responsiveness; if there are more unique values,
    returns the first max_count in sort order.

    Sort order:
    - If all surviving values are numeric-coercible (pd.to_numeric succeeds),
      sort numerically.
    - Otherwise, case-insensitive string sort on the string representation.

    This avoids the "2021 sorts before 999" surprise for year-like object
    columns.
    """
```

The GUI dialog's type-branch selection MUST call `classify_column_type(series)` — no inline `is_numeric_dtype` logic scattered in the Tk code.

Keep the existing `is_categorical_column` helper intact for backwards compatibility with pure-logic tests; it can stay as a thin shim or be left alone. The new `classify_column_type` is the source of truth going forward.

#### T3c. Filter-dict construction helper (integration-style seam)

Codex also recommended a small pure helper that maps `(column_kind, operator, raw_inputs)` → filter dict, so that the dialog's construction logic is testable without Tk. Add:

```python
def build_filter_dict(column: str, column_kind: str, operator: str,
                     raw_values: dict) -> dict:
    """Construct the filter dict that compute_matches consumes.

    column_kind: 'numeric' or 'categorical' (from classify_column_type).
    operator: one of the 10 supported conditions.
    raw_values: a dict with keys 'value', 'value2', 'values' (list),
                depending on operator — dialog populates only the relevant keys.

    Returns a dict matching the shape compute_matches already handles:
    - 'in': {'column', 'condition': 'in', 'values': [...]}
    - 'between': {'column', 'condition': 'between', 'value': min, 'value2': max}
    - 'has value': {'column', 'condition': 'has value'}
    - everything else: {'column', 'condition', 'value', 'value2': ''}

    Raises ValueError if operator is unknown or raw_values is missing
    required keys for the operator.
    """
```

### T4. Update `_apply_active_group` callers (minor)

**File:** `spectral_predict_gui_optimized.py`

The method `_apply_active_group(self, col, cond, val, val2, ...)` already accepts the existing 4-arg signature. The dialog continues to call it through the same signature for non-`in` conditions. For `in`, the dialog already goes through a filter-dict-first path (per V1). No signature change needed.

If the rewritten dialog finds any site that was relying on the old Listbox-for-everything-≤20 behavior and that behavior is no longer reachable, that site can be removed. Do not delete anything speculatively.

### T5. Tests

**File:** `tests/test_analysis_subset.py`

Extend with tests for:

1. **`classify_column_type`** — the riskiest new logic; test all of:
   - native numeric dtype (`[1, 2, 3]`) → `'numeric'`
   - all-string-numeric (`['1', '2', '3']`) → `'numeric'`
   - percentages (`['12.5%', '7.1%', '3.0%']`) → `'numeric'`
   - mixed numeric/garbage (`['2021', '2022', 'unknown']`, 2/3 numeric) → `'categorical'` (below 0.9 threshold)
   - plain text (`['grass', 'tree', 'sedge']`) → `'categorical'`
   - empty / all-null series → `'categorical'`
   - `numeric_threshold=0.5` override → verify the threshold is actually honored
2. **`get_unique_non_null_values`** — basic sort, duplicate dedup, NaN exclusion, `max_count` cap, **numeric-sort-when-all-coercible** (e.g. `['999', '2021', '7']` returns `[7, 999, 2021]` stringified or in numeric order), case-insensitive string sort for mixed.
3. **`build_filter_dict`** (T3c) — for each (column_kind, operator) combination, assert the output dict shape matches `compute_matches` expectations. Include: numeric `<`, numeric `between`, numeric `has value`, categorical `==`, categorical `in`, categorical `contains`, categorical `has value`. Include at least one failure case (unknown operator → ValueError).
4. `compute_matches` with `<`, `<=`, `>`, `>=`, `between` on numeric column (if not already covered — verify current coverage first, add only missing cases).

No Tk-level tests. Pure helpers only.

Run after changes:

```
python -m pytest tests/test_analysis_subset.py -v
python -m pytest tests/test_contamination_detection.py -v
```

Both must pass. The contamination suite is the regression canary.

### T6. Docs

- Append to `docs/SESSION_LOG.md` if any non-obvious gotcha is discovered during implementation (Tk focus issues, Combobox text-entry quirks, column-type detection edge cases).
- Update `docs/PROJECT_STATUS.md`: add V1.1 entry noting the three UX improvements; update the "Known Limitations / Risk" section to remove resolved items.

### T7. Manual verification checklist

(User will run this after tests pass.)

1. Load a dataset with a mix of numeric columns (e.g. `Carbon`, `HarvestYear`) and categorical columns (e.g. `PlantType`, `Collector`).
2. Tab 4 Configuration → Tab 4A Basic Settings → confirm the Analysis Subset card is visible at the top.
3. Confirm the Tab 4E Validation subtab no longer shows the subset card.
4. Open the Set dialog. Verify:
   - Column picker shows metadata columns, excludes the current target column.
   - Numeric column (`Carbon`): operator dropdown shows `<`, `<=`, `>`, `>=`, `between`, `==`, `!=`, `has value`. Numeric range caption shows observed min/max.
   - Numeric: set `Carbon < 7.0` → preview count is correct.
   - Numeric: set `HarvestYear between 2021 and 2023` → preview count is correct.
   - Categorical column (`PlantType`): operator dropdown shows `==`, `!=`, `in`, `contains`, `has value`.
   - Categorical `==`: combobox dropdown shows actual column values; can pick OR type.
   - Categorical `in`: multi-select listbox shows values; can pick multiple.
   - Categorical `contains`: free-text entry; literal substring (not regex).
5. Apply each of the above in turn. Verify:
   - Card status label in Tab 4 shows just the summary (e.g. `Carbon < 7.0`).
   - Data Viewer's label still shows full `Analysis Subset: N of total (…)` text.
   - Creating a validation set honors the subset.
   - Running regression analysis honors the subset.
   - Loading a result into Model Development shows the saved subset summary.
6. Clear subset → status returns to `All samples`, clear button greys out.
7. Edge cases still work: delete the subset's column from Data Viewer → filter safely clears (V1 behavior); reload different dataset → subset recomputes (V1 behavior).

---

## Dispatch plan

1. **Dispatch GLM 5.1 via opencode** in write mode, same pattern as the two prior successful V1 dispatches. Critical: the orchestrator prompt must contain ONLY task instructions — wrapper flags (`--dangerously-skip-permissions`, etc.) stay entirely outside the prompt text. The harness classifier inspects the prompt; if those flag names appear in the subagent's prompt, the wrapper's internal Bash call gets blocked before opencode is ever spawned.
2. **No commits, no pushes.** The worktree stays uncommitted for user inspection.
3. **After GLM completes**, run `codex exec --full-auto` in the background with an open-ended review prompt pointing at the new helpers, the rewritten dialog, and the new tests. Same pattern that caught the V1 blockers.
4. If codex finds issues, iterate until clean or user decides to merge with known limitations.

## Code quality constraints

- Smallest correct diff. No reformatting of unrelated code.
- No commits or pushes unless explicitly asked.
- Tk widgets created on the main thread only (existing pattern).
- No backwards-compat shims for V1 — V1 is uncommitted; this plan directly replaces those UX bits.
- Do not rename internal fields (`active_group_filter`, `active_indices`, `analysis_subset_status_label`, etc.).
- Preserve the zero-match refuse-to-apply behavior from V1.
- Preserve the `has value` / `contains` / `between` semantics already tested in `tests/test_analysis_subset.py`.

## Definition of done

1. Analysis Subset card is the first section inside Tab 4A Basic Settings. Tab 4E Validation no longer contains the card.
2. Dialog adapts to column type via `classify_column_type(series)` — no inline type logic in the GUI file.
3. Numeric columns: `<`, `<=`, `>`, `>=`, `between`, `==`, `!=`, `has value` all usable; observed min/max caption visible.
4. Categorical `==` / `!=` / `contains` use an editable `ttk.Combobox` populated from `get_unique_non_null_values(series)` — user can pick from the dropdown AND type a value not in the list, and either way the Preview button produces the correct count.
5. Categorical `in` keeps the multi-select listbox.
6. Year-like object columns (e.g. `pd.Series(["2021", "2022", "2023"], dtype=object)`) classify as `'numeric'` — verify explicitly in the manual checklist.
7. Mostly-categorical-with-some-numbers columns (e.g. `pd.Series(["2021", "2022", "unknown"])`) classify as `'categorical'` — verify explicitly.
8. Dialog is modal-blocking (`grab_set()` + `wait_window()`), focus lands on column picker, Enter in numeric entry triggers Preview. (These are new behaviors for V1.1, not V1 preservation.)
9. Changing columns/operators multiple times in one dialog session does NOT leak stale widget state or stale StringVars.
10. All V1 behavior preserved: filter order, provenance, mismatch warnings, one-class guardrail, missing-column safe clear, data-replacement refresh wiring.
11. Tests pass: `test_analysis_subset.py` (expanded per T5) and `test_contamination_detection.py`.
12. Docs updated only if a non-obvious gotcha surfaces (SESSION_LOG.md) and with a brief V1.1 entry in PROJECT_STATUS.md.
