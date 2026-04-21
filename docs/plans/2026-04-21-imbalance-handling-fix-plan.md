# Imbalance Handling Fix Plan

**Date:** 2026-04-21  
**Scope:** GUI dropdown filtering, one-class visibility, substitution banners, backend logging.  
**Out of scope:** Grid-search crash guard (4.6), any new models or preprocessing methods.

---

## 1. Overview

The imbalance-handling dropdown in Spectral Predict silently resets the user\'s selected method when the task type changes (e.g. ADASYN -> SMOGN for regression) and the backend silently substitutes binning/rare_boost/balanced -> smogn during Bayesian/NSGA-II searches. Neither substitution is visible to the user. This plan makes the GUI task-type-aware (hide imbalance for one-class, filter dropdown per task type), surfaces substitution warnings in the GUI, and replaces backend print() fallbacks with logger.warning() + progress_callback() so messages appear in the GUI progress log.

---

## 2. Pre-flight context

Read these files **before** touching code:

- `docs/plans/2026-04-21-imbalance-handling-investigation.md` — root-cause analysis and gap table.
- `spectral_predict_gui_optimized.py:10875-10915` — imbalance UI creation block (local `imbalance_frame`, heading label, dropdown).
- `spectral_predict_gui_optimized.py:16243-16320` — `_on_task_type_changed()` handler.
- `spectral_predict_gui_optimized.py:16322-16376` — `_update_one_class_controls_visibility()` (existing show/hide pattern).
- `spectral_predict_gui_optimized.py:22172-22301` — `_detect_and_display_imbalance()` (dropdown reset site).
- `spectral_predict_gui_optimized.py:27475-27500` — `_progress_callback()` (already reads `info.get('message', '')`).
- `src/spectral_predict/unified_bayesian.py:1708-1716` — Bayesian fallback block.
- `src/spectral_predict/nsga2_search.py:1810-1818` — NSGA-II fallback block.

---

## 3. Valid methods per task type (verified against code)

| Task type | Valid methods | Source |
|-----------|---------------|--------|
| **Classification** | `smote`, `adasyn`, `borderline_smote`, `random_undersampler`, `tomek_links`, `smote_tomek`, `smote_enn`, `class_weight` | `spectral_predict_gui_optimized.py:22208-22210` |
| **Regression** | `smogn`, `oversample`, `smotetomek`, `undersample`, `binning`, `rare_boost`, `balanced` | `spectral_predict_gui_optimized.py:22217` |
| **One-class** | *(none)* — `run_one_class_search()` at `search.py:4996-5021` accepts no `imbalance_method` parameter. | `search.py:4996-5021` |

Backend limitation (verified):
- `binning`, `rare_boost`, `balanced` are **regression-only** but **unsupported in Bayesian/NSGA-II** because they require manual `sample_weight_` extraction that only grid search implements.  
  Sources: `unified_bayesian.py:1708-1716`, `nsga2_search.py:1810-1818`.

---

## 4. Task breakdown

### Task 4.1 — Task-type-aware dropdown filtering

**Files touched:**
- `spectral_predict_gui_optimized.py` (add helpers near existing imbalance methods, modify `_on_task_type_changed()` and `_detect_and_display_imbalance()`)

**Current code (verbatim):**

In `_on_task_type_changed()` around line 16268 there is **no** imbalance dropdown refresh at all.

In `_detect_and_display_imbalance()` at `22206-22221`:
```python
            # Update method dropdown based on task type
            if task_type == 'classification':
                classification_methods = [
                    'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
                    'tomek_links', 'smote_tomek', 'smote_enn', 'class_weight'
                ]
                self.imbalance_method_combo['values'] = classification_methods
                # Set default if current selection is not valid
                if self.imbalance_method.get() not in classification_methods:
                    self.imbalance_method.set('smote')
            else:  # regression
                regression_methods = ['smogn', 'oversample', 'smotetomek', 'undersample', 'binning', 'rare_boost', 'balanced']
                self.imbalance_method_combo['values'] = regression_methods
                # Set default if current selection is not valid
                if self.imbalance_method.get() not in regression_methods:
                    self.imbalance_method.set('smogn')
```

**Desired replacement:**

1. Add two helper methods to the `SpectralPredictApp` class (place them near `_get_imbalance_params` or `_update_imbalance_method_description`):

```python
    def _get_imbalance_methods(self, task_type):
        """Return valid imbalance methods for a task type."""
        if task_type == 'classification':
            return [
                'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
                'tomek_links', 'smote_tomek', 'smote_enn', 'class_weight'
            ]
        elif task_type == 'regression':
            return [
                'smogn', 'oversample', 'smotetomek', 'undersample',
                'binning', 'rare_boost', 'balanced'
            ]
        else:
            return []

    def _refresh_imbalance_methods(self, task_type):
        """Update dropdown values for task type; notify if current selection becomes invalid."""
        methods = self._get_imbalance_methods(task_type)
        self.imbalance_method_combo['values'] = methods
        current = self.imbalance_method.get()
        if current not in methods:
            old = current
            default = 'smote' if task_type == 'classification' else 'smogn' if task_type == 'regression' else ''
            if default and methods:
                self.imbalance_method.set(default)
            else:
                self.imbalance_method.set('')
            if old:
                self._set_imbalance_banner(
                    f"Method changed: {old} -> {self.imbalance_method.get()} (not applicable for {task_type})"
                )
            self._update_imbalance_method_description(None)
        else:
            self._clear_imbalance_banner()
```

2. In `_on_task_type_changed()` (`16243`), after `actual_task` is determined, call the refresh.
   - In the `auto` + data branch (`16252-16255`), insert after `actual_task = "regression"`:
     ```python
     self._refresh_imbalance_methods(actual_task)
     ```
   - In the `auto` + no-data branch (`16256-16267`), insert before `return`:
     ```python
     self._refresh_imbalance_methods('classification')
     ```
   - In the explicit-task-type branch (`16268-16270`), insert after `actual_task = task_type`:
     ```python
     self._refresh_imbalance_methods(actual_task)
     ```

3. In `_detect_and_display_imbalance()` (`22172`), replace lines `22206-22221` with:
   ```python
   # Update method dropdown based on task type
   self._refresh_imbalance_methods(task_type)
   ```

**Verification:**
- `python -m py_compile spectral_predict_gui_optimized.py`
- `grep -n "_refresh_imbalance_methods" spectral_predict_gui_optimized.py` — should appear in `_on_task_type_changed` and `_detect_and_display_imbalance`.
- GUI smoke test: load data, run outlier detection, switch task type between Classification/Regression/One-Class and observe dropdown values change immediately.

**Dependencies:** none (first task).

---

### Task 4.2 — Hide imbalance UI for one-class

**Files touched:**
- `spectral_predict_gui_optimized.py:10875-10915` (creation block)
- `spectral_predict_gui_optimized.py:16322-16376` (`_update_one_class_controls_visibility`)

**Current code (verbatim):**

Creation block at `10875-10880`:
```python
        # === SECTION 2.6: Imbalance Handling (Optional) ===
        ttk.Label(content_frame, text="2.6 Imbalance Handling (Optional)", style='Heading.TLabel').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(25, 15))
        row += 1

        imbalance_frame = ttk.LabelFrame(content_frame, text="Imbalance Handling Settings", padding="20")
        imbalance_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
```

Note: `imbalance_frame` is a **local variable** (line 10879). It is never stored on `self`, so it cannot be hidden later.

**Desired replacement:**

1. Capture the heading and frame as instance variables in the creation block (`10875-10880`):
```python
        # === SECTION 2.6: Imbalance Handling (Optional) ===
        self.imbalance_section_heading = ttk.Label(content_frame, text="2.6 Imbalance Handling (Optional)", style='Heading.TLabel')
        self.imbalance_section_heading.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(25, 15))
        row += 1

        self.imbalance_frame = ttk.LabelFrame(content_frame, text="Imbalance Handling Settings", padding="20")
        self.imbalance_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
```

2. Update **all** remaining references to the local `imbalance_frame` in the same block to `self.imbalance_frame`. The exact lines to change are:
   - `10885`: `ttk.Checkbutton(imbalance_frame, ...)` → `ttk.Checkbutton(self.imbalance_frame, ...)`
   - `10890`: `ttk.Label(imbalance_frame, ...)` → `ttk.Label(self.imbalance_frame, ...)`
   - `10892`: `ttk.Combobox(imbalance_frame, ...)` → `ttk.Combobox(self.imbalance_frame, ...)`
   - `10903`: `ttk.Label(imbalance_frame, ...)` → `ttk.Label(self.imbalance_frame, ...)`
   - `10908`: `ttk.Frame(imbalance_frame)` → `ttk.Frame(self.imbalance_frame)`

3. In `_update_one_class_controls_visibility()` (`16322`), add hide/show logic following the existing pattern (`.grid()` / `.grid_remove()`).

Inside the `if task_type == "one_class":` block (after line 16354), add:
```python
            if hasattr(self, 'imbalance_frame'):
                self.imbalance_frame.grid_remove()
            if hasattr(self, 'imbalance_section_heading'):
                self.imbalance_section_heading.grid_remove()
```

Inside the `else:` block (after line 16376), add:
```python
            if hasattr(self, 'imbalance_frame'):
                self.imbalance_frame.grid()
            if hasattr(self, 'imbalance_section_heading'):
                self.imbalance_section_heading.grid()
```

**Verification:**
- `python -m py_compile spectral_predict_gui_optimized.py`
- `grep -n "imbalance_frame = " spectral_predict_gui_optimized.py` — should return **zero** results (no more local variable).
- GUI smoke test: select "One-Class" task type → the "2.6 Imbalance Handling" heading and frame disappear. Switch back to Classification/Regression → they reappear.

**Dependencies:** none (can be done in parallel with 4.1, but touches adjacent lines in the creation block; best done in the same commit as 4.1/4.5).

---

### Task 4.3 — GUI notification for auto-substitution

**Files touched:**
- `spectral_predict_gui_optimized.py:10905-10915` (banner creation)
- `spectral_predict_gui_optimized.py:22303-22318` (`_toggle_imbalance_controls`)
- `spectral_predict_gui_optimized.py:22319-22353` (`_update_imbalance_method_description`)
- `spectral_predict_gui_optimized.py:22172-22301` (`_detect_and_display_imbalance`)

**Current code (verbatim):**

There is **no** banner widget in the imbalance frame.

**Desired replacement:**

1. Add a banner label inside the imbalance frame, immediately after the method description label (`10902-10905`). Insert after `10905`:
```python
        # Substitution notification banner
        self.imbalance_banner_label = ttk.Label(
            self.imbalance_frame, text="", foreground='#ff9800',
            font=('Segoe UI', 10, 'bold'), wraplength=600, justify='left'
        )
        self.imbalance_banner_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(5, 0), padx=(20, 0))
```
*(Row 3 is free because the parameters subframe occupies row 2.)*

2. Add helper methods near the other imbalance helpers:
```python
    def _set_imbalance_banner(self, text):
        """Show a substitution warning in the imbalance section."""
        if hasattr(self, 'imbalance_banner_label'):
            self.imbalance_banner_label.config(text=text)

    def _clear_imbalance_banner(self):
        """Clear the imbalance substitution warning."""
        if hasattr(self, 'imbalance_banner_label'):
            self.imbalance_banner_label.config(text="")
```

3. In `_toggle_imbalance_controls()` (`22303`), when disabling, clear the banner. Replace:
```python
        else:
            # Disable all parameter controls
            for widget_name in ['k_neighbors_spin', 'n_bins_spin', 'boost_factor_spin']:
                self.imbalance_widgets[widget_name].config(state='disabled')
```
with:
```python
        else:
            self._clear_imbalance_banner()
            # Disable all parameter controls
            for widget_name in ['k_neighbors_spin', 'n_bins_spin', 'boost_factor_spin']:
                self.imbalance_widgets[widget_name].config(state='disabled')
```

4. In `_update_imbalance_method_description()` (`22319`), clear the banner at the top because a manual valid selection should remove any stale warning:
```python
    def _update_imbalance_method_description(self, event):
        """Update method description and show/hide relevant parameters."""
        self._clear_imbalance_banner()
        method = self.imbalance_method.get()
```

5. In `_detect_and_display_imbalance()`, after each auto-select of the recommended method (`22256-22257` and `22289-22290`), clear the banner so the intentional recommendation does not show a substitution warning. After each pair insert `self._clear_imbalance_banner()`:
```python
                        self.imbalance_method.set(recommendation['recommended_method'])
                        self._update_imbalance_method_description(None)
                        self._clear_imbalance_banner()
```

**Verification:**
- `python -m py_compile spectral_predict_gui_optimized.py`
- GUI smoke test: enable imbalance, select a method, then switch task type so the method becomes invalid → an orange banner appears under the dropdown stating the change.
- Disable imbalance checkbox → banner disappears.
- Select a valid method from the dropdown → banner disappears.

**Dependencies:** depends on Task 4.1 (`_refresh_imbalance_methods` calls the banner helpers) and Task 4.2 (`self.imbalance_frame` must exist before placing the banner inside it).

---

### Task 4.4 — Backend `logger.warning` + `progress_callback` for fallbacks

**Files touched:**
- `src/spectral_predict/unified_bayesian.py:34` (add logger) and `:1708-1716` (replace fallback)
- `src/spectral_predict/nsga2_search.py:1810-1818` (replace fallback)

**Current code (verbatim):**

`unified_bayesian.py:1708-1716`:
```python
    # Substitute regression sample weighting methods that don't work with cross_val_score
    # These methods (binning, rare_boost, balanced) require manual sample_weight extraction
    # which is only implemented in Grid Search. Substitute with smogn (SMOTE-style for regression).
    UNSUPPORTED_REGRESSION_METHODS = {'binning', 'rare_boost', 'balanced'}
    if task_type == 'regression' and imbalance_method in UNSUPPORTED_REGRESSION_METHODS:
        original_method = imbalance_method
        imbalance_method = 'smogn'  # SMOTE-style synthetic oversampling for regression
        if verbose:
            print(f"Note: '{original_method}' requires Grid Search. Using 'smogn' instead for Bayesian optimization.")
```

`nsga2_search.py:1810-1818`:
```python
    # Substitute regression sample weighting methods that don't work with cross_val_score
    # These methods (binning, rare_boost, balanced) require manual sample_weight extraction
    # which is only implemented in Grid Search. Substitute with smogn (SMOTE-style for regression).
    UNSUPPORTED_REGRESSION_METHODS = {'binning', 'rare_boost', 'balanced'}
    if task_type == 'regression' and imbalance_method in UNSUPPORTED_REGRESSION_METHODS:
        original_method = imbalance_method
        imbalance_method = 'smogn'  # SMOTE-style synthetic oversampling for regression
        if verbose >= 1:
            print(f"  Note: '{original_method}' requires Grid Search. Using 'smogn' instead for NSGA-II.")
```

**Desired replacement:**

1. In `unified_bayesian.py`, add a logger right after the existing `import logging` at line 34:
```python
import logging

logger = logging.getLogger(__name__)
```

2. Replace the fallback block (`1708-1716`) with:
```python
    # Substitute regression sample weighting methods that don't work with cross_val_score
    # These methods (binning, rare_boost, balanced) require manual sample_weight extraction
    # which is only implemented in Grid Search. Substitute with smogn (SMOTE-style for regression).
    UNSUPPORTED_REGRESSION_METHODS = {'binning', 'rare_boost', 'balanced'}
    if task_type == 'regression' and imbalance_method in UNSUPPORTED_REGRESSION_METHODS:
        original_method = imbalance_method
        imbalance_method = 'smogn'  # SMOTE-style synthetic oversampling for regression
        warn_msg = f"'{original_method}' requires Grid Search. Using 'smogn' instead for Bayesian optimization."
        logger.warning(warn_msg)
        if progress_callback:
            progress_callback({'message': f"[Warning] {warn_msg}"})
        elif verbose:
            print(f"Note: {warn_msg}")
```

3. In `nsga2_search.py`, replace the fallback block (`1810-1818`) with:
```python
    # Substitute regression sample weighting methods that don't work with cross_val_score
    # These methods (binning, rare_boost, balanced) require manual sample_weight extraction
    # which is only implemented in Grid Search. Substitute with smogn (SMOTE-style for regression).
    UNSUPPORTED_REGRESSION_METHODS = {'binning', 'rare_boost', 'balanced'}
    if task_type == 'regression' and imbalance_method in UNSUPPORTED_REGRESSION_METHODS:
        original_method = imbalance_method
        imbalance_method = 'smogn'  # SMOTE-style synthetic oversampling for regression
        warn_msg = f"'{original_method}' requires Grid Search. Using 'smogn' instead for NSGA-II."
        logger.warning(warn_msg)
        if progress_callback:
            progress_callback({'message': f"[Warning] {warn_msg}"})
        elif verbose >= 1:
            print(f"  Note: {warn_msg}")
```

**Why this works:** the GUI’s `_progress_callback()` at `spectral_predict_gui_optimized.py:27475` already reads `info.get('message', '')` and forwards it to `self._log_progress()`, which appends it to the live progress text area. No extra GUI wiring is required.

**Verification:**
- `python -m py_compile src/spectral_predict/unified_bayesian.py`
- `python -m py_compile src/spectral_predict/nsga2_search.py`
- Run a regression Bayesian search with `imbalance_method='binning'` via a script and confirm the warning appears in the log output.
- In the GUI, run a regression Bayesian search with `binning` selected → the progress text area should show: `[Warning] 'binning' requires Grid Search. Using 'smogn' instead for Bayesian optimization.`

**Dependencies:** none (backend-only change).

---

### Task 4.5 — Clean up initial dropdown

**Files touched:**
- `spectral_predict_gui_optimized.py:10893-10898`

**Current code (verbatim):**
```python
        self.imbalance_method_combo['values'] = [
            'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
            'tomek_links', 'smote_tomek', 'smote_enn', 'class_weight',
            'binning', 'rare_boost', 'balanced'
        ]
```

**Desired replacement:**
```python
        self.imbalance_method_combo['values'] = self._get_imbalance_methods('classification')
```

Rationale: the default `self.imbalance_method` is `"smote"` (`10891`), which is a classification method. The default task type at startup is `"auto"`, but until data is loaded the safest default list is the classification set.

**Verification:**
- `python -m py_compile spectral_predict_gui_optimized.py`
- Launch GUI → open the imbalance method dropdown before loading data → it should show exactly the 8 classification methods (no regression methods mixed in).

**Dependencies:** depends on Task 4.1 (`_get_imbalance_methods` must exist).

---

## 5. Ordering

| Order | Task | Rationale |
|-------|------|-----------|
| 1 | **4.1** — Dropdown filtering | Introduces `_get_imbalance_methods` and `_refresh_imbalance_methods`, which every other GUI task depends on. |
| 2 | **4.2** — Hide UI for one-class | Must happen before 4.3 because the banner label is placed **inside** `self.imbalance_frame`, which 4.2 creates. Also touches the same creation block as 4.5; do them together. |
| 3 | **4.5** — Clean up initial dropdown | One-line change that depends on 4.1’s helper. Safe to do immediately after 4.2 in the same commit. |
| 4 | **4.3** — GUI notification banner | Depends on both 4.1 (banner helpers called from `_refresh_imbalance_methods`) and 4.2 (banner widget placed inside `self.imbalance_frame`). |
| 5 | **4.4** — Backend logging | Independent; can be done at any time. |

**Recommended commit grouping:**
- Commit A (GUI init + wiring): tasks 4.1, 4.2, 4.5 together.
- Commit B (GUI banner): task 4.3.
- Commit C (Backend): task 4.4.

---

## 6. Out of scope

- **Grid-search crash guard (4.6)** — defense-in-depth validation inside `run_search()`; not required for the user-facing silent-substitution fix.
- **Changing `_detect_and_display_imbalance()` behavior for one-class** — when task type is `one_class`, that method currently falls through to the regression branch and shows target-distribution text in the *Distribution Analysis* section (not the imbalance frame). Since 4.2 hides the imbalance frame entirely, this is acceptable for now.
- **Modifying `imbalance.py` method implementations** — no backend resampler logic changes.
- **Adding new imbalance methods** — not requested.

---

## 7. Post-implementation smoke test

Run this short manual test after all commits are applied:

1. **Launch the GUI** (`python spectral_predict_gui_optimized.py`).
2. **Initial dropdown check** — before loading any data, open Tab 3 → Section 2.6 → Method dropdown.  
   - Expected: exactly 8 classification methods (`smote` … `class_weight`). No `smogn`, no `binning`.
3. **Task-type switch check** — load `example/BoneCollagen.csv`, run outlier detection, then change Task Type:
   - **Classification** → dropdown shows 8 classification methods.
   - **Regression** → dropdown shows 7 regression methods (`smogn` … `balanced`).
   - **One-Class** → the entire “2.6 Imbalance Handling” heading and frame disappear.
   - Switch back to **Classification** → heading/frame reappear; dropdown reverts to classification list.
4. **Invalid-method banner check** — enable imbalance handling, select `adasyn`, then switch Task Type to **Regression**.  
   - Expected: dropdown changes to `smogn`; an orange banner appears under the dropdown: `Method changed: adasyn → smogn (not applicable for regression)`.
5. **Banner clear check** — select a valid regression method from the dropdown.  
   - Expected: banner disappears.
6. **Backend fallback check** — set Task Type = Regression, enable imbalance, select `binning`, run a **Bayesian** search (small trial count).  
   - Expected: the progress text area shows `[Warning] 'binning' requires Grid Search. Using 'smogn' instead for Bayesian optimization.`
7. **Backend fallback check (NSGA-II)** — repeat step 6 with NSGA-II search.  
   - Expected: progress text area shows `[Warning] 'binning' requires Grid Search. Using 'smogn' instead for NSGA-II.`

If all 7 steps pass, the fix is complete.
