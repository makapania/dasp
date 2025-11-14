# Derivative Calculation Architecture - Future Enhancement

**Status**: Deferred (Major Refactoring Required)
**Date**: 2025-01-13
**Priority**: Medium (Quality of Life improvement, not blocking)

---

## Problem Statement

### Current Issue
Derivatives (Savitzky-Golay, gap-segment) are currently calculated on **already-filtered wavelength ranges**, which can cause **edge effects** at the boundaries of the filtered range.

### Why This Matters
- **Savitzky-Golay derivatives** use a sliding window (e.g., 7, 11, 17 points)
- When the window reaches the edge of the filtered wavelength range, it has incomplete data
- This causes artifacts/distortions at the boundaries
- **Industry standard practice**: Calculate derivatives on full spectrum, then filter for analysis

### Mathematical Benefit
Calculating derivatives on a larger wavelength range (full spectrum or NIR range) and then filtering to a subset for analysis prevents boundary artifacts and produces more accurate derivative spectra.

---

## Current Architecture

### Data Flow
```
Import Tab
  ↓
User sets wavelength range (e.g., 1800-2400 nm)
  ↓
self._apply_wavelength_filter()
  ↓
self.X = filtered DataFrame (only 1800-2400 nm)
  ↓
Analysis Config Tab → "Run Analysis"
  ↓
_run_analysis_thread()
  ↓
run_search(X_filtered, y, preprocessing_methods={'raw': True, 'sg1': True, ...})
  ↓
Inside run_search():
  for each preprocessing method:
    pipeline = build_preprocessing_pipeline(method, window) + model
    cross_validate(pipeline, X_filtered, y)
    └─> Derivatives calculated on FILTERED data (edge effects!)
```

### Key Code Locations

**Wavelength Filtering** (Import Tab):
- File: `spectral_predict_gui_optimized.py`
- Lines: ~5095-5115 (`_apply_wavelength_filter()`)
- Lines: ~1796-1810 (Wavelength Range UI controls)

**Analysis Execution**:
- File: `spectral_predict_gui_optimized.py`
- Lines: ~6779-7860 (`_run_analysis_thread()`)
- Line: ~7801 (call to `run_search()`)

**Preprocessing Pipeline**:
- File: `src/spectral_predict/search.py`
- Line: 13 (import: `from .preprocess import build_preprocessing_pipeline`)
- Lines: 371, 481, 800 (calls to `build_preprocessing_pipeline()`)

**Derivative Implementation**:
- File: `src/spectral_predict/preprocess.py`
- Lines: 43-105 (`SavgolDerivative` class)
- Lines: 107-147 (`build_preprocessing_pipeline()` function)

---

## Desired Architecture

### Proposed Data Flow
```
Import Tab
  ↓
Keep FULL spectrum or NIR range (e.g., 400-2500 nm or 800-2500 nm)
  ↓
self.X = full DataFrame (all imported wavelengths)
  ↓
Analysis Config Tab → User specifies analysis wavelength subset (e.g., 1800-2400)
  ↓
_run_analysis_thread()
  ↓
STEP 1: Apply preprocessing to FULL spectrum
  for each method in ['raw', 'snv', 'sg1', 'sg2', 'deriv_snv']:
    prep_pipeline = build_preprocessing_pipeline(method, window)
    preprocessed_data[method] = prep_pipeline.fit_transform(X_full)
    └─> Derivatives calculated on FULL spectrum (no edge effects!)

STEP 2: Filter wavelengths AFTER preprocessing
  for method, X_prep in preprocessed_data.items():
    filtered_data[method] = filter_wavelengths(X_prep, analysis_range)

STEP 3: Pass pre-processed data to run_search
  run_search(filtered_data, y, ...)
  └─> No preprocessing happens inside anymore
      Just trains models on pre-filtered derivative data
```

### Key Changes Required

1. **Import Tab Defaults**:
   - Change default wavelength range to "Full Spectrum" or "NIR (800-2500)"
   - Update UI labels to clarify purpose: "Import Range (for derivatives)"
   - Add help text explaining this is for derivative calculation

2. **Analysis Config Tab**:
   - **Wire up existing wavelength selection UI** (already exists at lines 3796-3845!)
   - Parse wavelength spec from `self.refine_wl_spec` text widget
   - Use existing `_parse_wavelength_spec()` method
   - Apply filtering AFTER preprocessing, BEFORE run_search()

3. **Preprocessing Extraction** (MAJOR CHANGE):
   - Move preprocessing OUT of `run_search()`
   - Create new function in GUI: `_apply_all_preprocessing(X_full)`
   - Returns dict: `{'raw': X_raw, 'snv': X_snv, 'sg1_w7': X_sg1, ...}`
   - Filter each preprocessed variant using wavelength selection
   - Pass pre-processed dict to modified `run_search()`

4. **run_search() Refactoring** (MAJOR CHANGE):
   - Currently expects raw X and applies preprocessing internally
   - Would need to accept pre-processed data dict instead
   - Remove preprocessing loops (lines 371, 481, 800)
   - Update Pipeline construction to skip preprocessing steps
   - Significant changes to internal logic

---

## Why This Is Deferred

### Complexity Assessment: **HIGH**

1. **Deep Integration**:
   - Preprocessing is embedded in sklearn Pipelines throughout `run_search()`
   - Pipelines are used for cross-validation, model training, feature selection
   - Extracting preprocessing breaks the Pipeline paradigm

2. **Widespread Changes**:
   - Modify `run_search()` signature and internal logic
   - Update all calls to `run_search()`
   - Change how cross-validation works
   - Update feature importance calculation
   - Modify subset selection logic (variable/region subsets)

3. **Testing Requirements**:
   - Regression testing of ALL model types
   - Verify cross-validation still works correctly
   - Ensure feature selection methods still work
   - Validate subset analysis functionality
   - Test with classification AND regression tasks

4. **Risk of Breaking Changes**:
   - High risk of introducing bugs in core analysis
   - May break existing saved models/pipelines
   - Could affect reproducibility of results

5. **Lower Priority**:
   - Edge effects are likely minimal in practice
   - Current workaround: Import wider wavelength range
   - Not blocking critical functionality

### Alternative Workaround

**Quick Fix** (can implement now without refactoring):
- Change Import Tab defaults to suggest wider ranges (NIR: 800-2500 nm)
- Add tooltip: "Import wider range for better derivative quality"
- Users can still filter in Import tab if desired
- **No code changes to core analysis logic**

---

## Implementation Plan (When Pursued)

### Phase 1: Preparation
1. Write comprehensive tests for current `run_search()` behavior
2. Document current preprocessing pipeline structure
3. Identify all code that depends on current architecture

### Phase 2: Create Preprocessing Extraction Function
**File**: `spectral_predict_gui_optimized.py`

```python
def _apply_all_preprocessing(self, X_full, preprocessing_methods, window_sizes):
    """
    Apply all preprocessing methods to full spectrum data.

    Parameters
    ----------
    X_full : pd.DataFrame
        Full spectrum data (not wavelength-filtered)
    preprocessing_methods : dict
        {'raw': True, 'snv': True, 'sg1': True, ...}
    window_sizes : list
        [7, 11, 17, ...]

    Returns
    -------
    preprocessed_data : dict
        {'raw': X_raw, 'snv': X_snv, 'sg1_w7': X_sg1_w7, ...}
    """
    from spectral_predict.preprocess import build_preprocessing_pipeline

    preprocessed_data = {}

    for method, enabled in preprocessing_methods.items():
        if not enabled:
            continue

        if method == 'raw':
            preprocessed_data['raw'] = X_full
        elif method in ['sg1', 'sg2', 'deriv_snv']:
            for window in window_sizes:
                pipeline = build_preprocessing_pipeline(method, window)
                X_prep = pipeline.fit_transform(X_full)
                # Convert back to DataFrame with wavelength columns
                X_prep_df = pd.DataFrame(X_prep,
                                        index=X_full.index,
                                        columns=X_full.columns)
                key = f"{method}_w{window}"
                preprocessed_data[key] = X_prep_df
        else:  # snv, etc.
            pipeline = build_preprocessing_pipeline(method, window=None)
            X_prep = pipeline.fit_transform(X_full)
            X_prep_df = pd.DataFrame(X_prep,
                                    index=X_full.index,
                                    columns=X_full.columns)
            preprocessed_data[method] = X_prep_df

    return preprocessed_data
```

### Phase 3: Add Wavelength Filtering
**File**: `spectral_predict_gui_optimized.py`

```python
def _filter_wavelengths_for_analysis(self, X):
    """
    Filter wavelengths based on Analysis Config selection.

    Parameters
    ----------
    X : pd.DataFrame
        Preprocessed spectral data

    Returns
    -------
    X_filtered : pd.DataFrame
        Data with only selected wavelengths
    """
    # Get wavelength specification from Analysis Config tab
    spec_text = self.refine_wl_spec.get('1.0', 'end').strip()

    # If empty, return all wavelengths (backward compatible)
    if not spec_text:
        return X

    # Parse wavelength specification
    available_wl = X.columns.astype(float).values
    selected_wl = self._parse_wavelength_spec(spec_text, available_wl)

    # Filter to selected wavelengths
    selected_wl_str = [str(wl) for wl in selected_wl]
    X_filtered = X[selected_wl_str]

    return X_filtered
```

### Phase 4: Modify _run_analysis_thread()
**File**: `spectral_predict_gui_optimized.py` (~line 6779)

**Current**:
```python
X_filtered = self.X  # Already filtered from import
run_search(X_filtered, y, preprocessing_methods=preprocessing_methods, ...)
```

**New**:
```python
# STEP 1: Preprocessing on FULL spectrum
X_full = self.X  # Should be full/NIR spectrum from import
preprocessed_data = self._apply_all_preprocessing(
    X_full, preprocessing_methods, window_sizes
)

# STEP 2: Filter wavelengths AFTER preprocessing
filtered_data = {}
for method, X_prep in preprocessed_data.items():
    filtered_data[method] = self._filter_wavelengths_for_analysis(X_prep)

# Log what happened
self._log_progress(f"Derivatives calculated on {X_full.shape[1]} wavelengths")
first_filtered = next(iter(filtered_data.values()))
self._log_progress(f"Filtered to {first_filtered.shape[1]} wavelengths for analysis")

# STEP 3: Pass to run_search (needs modification!)
run_search(filtered_data, y, ...)  # Different signature!
```

### Phase 5: Refactor run_search()
**File**: `src/spectral_predict/search.py`

**Major Changes**:
1. Change signature to accept `preprocessed_data` dict instead of raw `X`
2. Remove all preprocessing loops (lines 371, 481, 800)
3. Iterate over preprocessed variants instead of building pipelines
4. Update Pipeline construction to skip preprocessing steps
5. Fix feature importance/selection to work with pre-processed data

**This is the HARD part** - lots of internal logic to update.

### Phase 6: Update Import Tab UI
**File**: `spectral_predict_gui_optimized.py` (~lines 1796-1810)

1. Change default wavelength range to "" (empty = all) or "800-2500"
2. Update label: "Import Wavelength Range (for derivative calculation)"
3. Add help text explaining purpose
4. Add preset buttons: "Full Spectrum", "NIR (800-2500)", "VIS-NIR (400-2500)"

### Phase 7: Testing
1. Test all model types (PLS, Ridge, RandomForest, XGBoost, LightGBM, etc.)
2. Test all preprocessing methods (raw, SNV, SG1, SG2, deriv+SNV)
3. Test classification AND regression
4. Test subset analysis (variable subsets, region subsets)
5. Test with different wavelength selections
6. Verify backward compatibility (empty selection = all wavelengths)
7. Check that results are reproducible

---

## Risks & Considerations

### High Risk Items
1. **Breaking sklearn Pipelines**: Preprocessing is part of Pipeline for cross-validation
2. **Feature Selection**: Currently happens on preprocessed data within Pipeline
3. **Reproducibility**: Changing when preprocessing happens may affect random seeds
4. **Saved Models**: May break loading of previously saved Pipeline objects

### Medium Risk Items
1. **Performance**: Preprocessing full spectrum then filtering may be slower
2. **Memory**: Temporarily storing full-spectrum derivatives
3. **Testing Coverage**: Need comprehensive tests to catch regressions

### Low Risk Items
1. **UI Changes**: Import/Analysis Config tab modifications are straightforward
2. **Wavelength Filtering**: Already have working implementation

---

## Decision

**DEFER** this enhancement until:
1. There's a critical need (users reporting derivative edge effect issues)
2. Major refactoring window (e.g., version 2.0 release)
3. Time for comprehensive testing and validation
4. Lower-risk items are completed first

**Current Workaround**:
- Users can import wider wavelength ranges in Import tab
- Document best practice: Import full NIR range (800-2500 nm)
- Edge effects are likely minimal for most use cases

---

## Future Considerations

### Simpler Alternative (Lower Risk)
Instead of extracting preprocessing from `run_search()`, could:
1. Add wavelength filtering INSIDE `run_search()` that happens AFTER preprocessing
2. Pass wavelength selection as parameter to `run_search()`
3. Modify Pipeline to include wavelength filtering step AFTER preprocessing
4. Less refactoring, but still requires changes to Pipeline structure

### Even Simpler (Minimal Change)
1. Just change Import Tab defaults to wider range
2. Add UI hints about derivative quality
3. **No code changes to analysis logic**
4. Accept minor edge effects as acceptable trade-off

---

## References

**Related Code**:
- Wavelength filtering: `spectral_predict_gui_optimized.py:5095-5115`
- Preprocessing pipeline: `src/spectral_predict/preprocess.py:43-147`
- Analysis execution: `spectral_predict_gui_optimized.py:6779-7860`
- Search function: `src/spectral_predict/search.py:21-900`

**Discussion Date**: 2025-01-13
**Participants**: User, Claude Code Agent
**Outcome**: Document and defer - not worth the refactoring risk at this time
