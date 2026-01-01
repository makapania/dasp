# V1 Codebase - Comprehensive Bug Fix Handoff

**Date**: 2025-12-21
**Context**: Rogue agent damaged codebase. Deep functional analysis completed.
**Analysis Method**: Traced actual execution paths, not just structure comparison.

---

## SUMMARY: 17 Issues Found

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 4 | Features completely broken or UI disconnected |
| HIGH | 6 | Silent failures, data loss, missing options |
| MEDIUM | 7 | Silent degradation, misleading messages |

---

# CRITICAL ISSUES (Fix First)

## CRITICAL 1: GA Checkbox Not Connected to Backend

**File**: `spectral_predict_gui_optimized.py`
**Lines**: 5377-5378 (checkbox), 15633+ (search button)

**Problem**: User can check "GA Preprocessing Optimization" checkbox in the GUI, but the checkbox state is NEVER read or passed to the backend. GA preprocessing only works if explicitly set in code.

**How to Find**:
1. Search for where checkbox is created: `self.ga_preproc_checkbox`
2. Search for where search button collects parameters
3. Note: GA checkbox is never referenced when building search parameters

**Fix Required**: Read checkbox state and pass `ga_preprocess=True` to `run_search()` when enabled.

---

## CRITICAL 2: Baseline/Smoothing Never Applied in Grid Search

**File**: `src/spectral_predict/search.py`
**Lines**: 892-902

**Problem**: `build_preprocessing_pipeline()` accepts these parameters:
- `baseline_method`, `baseline_params`
- `smoothing`, `smoothing_window`, `smoothing_polyorder`

But search.py NEVER passes them. They're not even in the `preprocess_cfg` dict.

**Current Code** (lines 892-902):
```python
prep_pipe_steps = build_preprocessing_pipeline(
    preprocess_cfg["name"],
    preprocess_cfg["deriv"],
    preprocess_cfg["window"],
    preprocess_cfg["polyorder"],
    imbalance_method=None,
    imbalance_params=None,
    task_type=task_type,
    interference=preprocess_cfg.get("interference"),
    wavelengths=wavelengths
    # MISSING: baseline_method, baseline_params, smoothing, smoothing_window, smoothing_polyorder
)
```

**Impact**:
- GUI collects baseline/smoothing settings from user
- GA preprocessing optimizes over baseline methods
- But regular grid search NEVER tests baseline or smoothing
- Features exist but are unreachable

**Fix Required**:
1. Add baseline/smoothing to `preprocess_cfg` dict (lines 662-823)
2. Pass them to `build_preprocessing_pipeline()` (lines 892-902, 1864-1874, 2234-2244)

---

## CRITICAL 3: GA Preprocessing Lost in Model Development

**File**: `spectral_predict_gui_optimized.py`
**Multiple Locations**:
- Lines 18399-18418: Loading logic
- Lines 19855-19868: `preprocess_name_map`
- Lines 20204-20280: PATH A/B logic

**Problem**: When loading a GA preprocessing result into Model Development:

1. **Line 18415**: No handler for GA preprocessing - falls through to `gui_preprocess = 'raw'`
```python
# Current code (buggy):
elif preprocess in ['raw', 'snv']:
    gui_preprocess = preprocess
else:
    gui_preprocess = 'raw'  # GA PREPROCESSING FALLS HERE
```

2. **Lines 19855-19868**: `preprocess_name_map` has no GA entry
```python
preprocess_name_map = {
    'raw': 'raw', 'snv': 'snv', 'sg1': 'deriv', ...
    # NO GA ENTRY
}
```

3. **Lines 20204-20280**: PATH A/B logic never checks for GA config

**Result**: User sees "Loaded: XGBoost | GA_Linear_SNV" but model actually runs with RAW preprocessing. R² is dramatically lower. NO WARNING.

**Root Cause**: GA config fields (`ga_genes`, `ga_config`, `ga_transform`) are stored in results but lost when loading because the GUI doesn't know how to reconstruct them.

**Fix Required**:
1. Add GA detection in loading logic (line 18415)
2. Store `ga_genes`, `ga_config` when loading GA result
3. Add reconstruction logic using `chromosome_to_transform()` from `ga_preprocessing.py`
4. Apply GA transform instead of standard preprocessing

---

## CRITICAL 4: Mixed Model Selection Breaks GA Preprocessing

**File**: `src/spectral_predict/search.py`
**Lines**: 995-1013

**Problem**: GA preprocessing creates SEPARATE configs for:
- Linear models (PLS): `ga_model_type = "linear"`
- Tree models (RandomForest, LightGBM): `ga_model_type = "tree"`

When user selects BOTH types (e.g., PLS + RandomForest):
- Lines 995-1013 filter configs by model type
- Each model only sees its matching GA config
- Non-matching models may get wrong or no GA preprocessing

**Example**: User selects PLS + RandomForest with GA preprocessing
- PLS gets `linear` GA config ✓
- RandomForest gets `tree` GA config (if generated) OR nothing ✗

**Fix Required**: Ensure all selected models get appropriate GA preprocessing, or warn user about incompatibility.

---

# HIGH SEVERITY ISSUES

## HIGH 1: Edge Masking Missing for Top Variables Display

**File**: `src/spectral_predict/search.py`
**Lines**: 2520-2527

**Problem**: `_apply_edge_mask()` is called at line 1352 for variable selection, but NOT at line 2522 when computing `top_vars` for display.

**Current Code** (lines 2520-2527):
```python
importances = get_feature_importances(
    fitted_model, model_name, X_transformed, y
)
# MISSING: importances = _apply_edge_mask(importances, preprocess_cfg)

n_to_select = min(top_n_vars, len(importances))
top_indices = np.argsort(importances, kind='stable')[-n_to_select:][::-1]
```

**Impact**: Edge wavelengths (350, 2490, etc.) appear in "Top Variables" even when using SG derivatives.

**Fix**: Add one line after line 2522:
```python
importances = _apply_edge_mask(importances, preprocess_cfg)
```

---

## HIGH 2: All-Zero Importances Silently Replaced

**File**: `src/spectral_predict/search.py`
**Lines**: 1324-1326

**Problem**: If variable selection returns all zeros, code silently replaces with uniform `np.ones()`:
```python
if np.allclose(importances, 0):
    importances = np.ones(len(importances))
```

**Impact**: User gets wrong variable ranking with no warning. CARS with outliers might return all zeros.

**Fix**: At minimum, add a warning. Better: try alternative method or raise error.

---

## HIGH 3: VCPA Edge Cases Lose Information

**File**: `src/spectral_predict/search.py`
**Lines**: 1225-1233

**Problem**: VCPA edge cases default to binary mask (all 1.0), losing actual importance scores.

---

## HIGH 4: Dropdown Missing Preprocessing Options

**File**: `spectral_predict_gui_optimized.py`
**Lines**: 7593, 7644

**Current Options**: `['raw', 'snv', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv']`

**Missing**:
- `'snv_deriv'` - backend generates this but GUI can't select it
- ALL GA preprocessing options

**Fix**: Add missing options to dropdown, or dynamically populate based on what's in results.

---

## HIGH 5: Ensemble Preprocessor Explicitly None

**File**: `spectral_predict_gui_optimized.py`
**Line**: 22026

**Problem**: For ensemble models, `preprocessor` field is explicitly set to `None`:
```python
'preprocessor': None,  # Ensembles have preprocessing in base models
```

**Impact**: Can't reconstruct preprocessing for ensemble predictions in Prediction tab.

---

## HIGH 6: Silent Variable Selection Failure

**File**: `src/spectral_predict/search.py`
**Lines**: 1451-1454

**Problem**: If variable selection fails, error is printed to console but execution continues:
```python
except Exception as e:
    print(f"Warning: Could not compute importances...")
    # CONTINUES TO NEXT METHOD
```

**Impact**: No results generated for failed methods, user only sees console warning (not GUI).

---

# MEDIUM SEVERITY ISSUES

## MEDIUM 1: Hyperparameter Parse Errors Caught Silently
**File**: `spectral_predict_gui_optimized.py`, lines 20007-20036

## MEDIUM 2: Bare Except Clause
**File**: `src/spectral_predict/search.py`, lines 2407-2411
```python
except:  # Should be except (TypeError, ValueError, AttributeError):
    continue
```

## MEDIUM 3: Silent top_vars Degradation
**File**: `src/spectral_predict/search.py`, lines 2427-2429

## MEDIUM 4: Status Messages Don't Warn About Fallback
**File**: `spectral_predict_gui_optimized.py`, line 18423

## MEDIUM 5: Misleading Parameter Warnings
**File**: `src/spectral_predict/search.py`, lines 391-394
Warnings say "parameter is noted but not yet applied" for params that ARE used.

## MEDIUM 6: Polyorder Inconsistency
**Files**: `preprocess.py` lines 87-91, `search.py` lines 757-788

---

# PARAMETER FLOW TABLE

| Parameter | GUI Collects | search.py Config | search.py Passes | Results Store |
|-----------|-------------|------------------|------------------|---------------|
| deriv | ✓ | ✓ | ✓ | ✓ |
| window | ✓ | ✓ | ✓ | ✓ |
| polyorder | ✓ | ✓ | ✓ | ✗ |
| baseline_method | ✓ | ✗ | ✗ | ✗ |
| baseline_params | ✓ | ✗ | ✗ | ✗ |
| smoothing | ✓ | ✗ | ✗ | ✗ |
| ga_genes | N/A | ✓ | ✓ | ✓ (lost on load) |
| ga_config | N/A | ✓ | ✓ | ✓ (lost on load) |
| ga_transform | N/A | ✓ | ✓ | ✗ (can't serialize) |

---

# FIX ORDER

## Phase 1: Critical (Broken Features)
1. Connect GA checkbox to search
2. Add GA preprocessing handler in Model Development
3. Add GA entry to preprocess_name_map
4. Pass baseline/smoothing to build_preprocessing_pipeline

## Phase 2: High (Silent Failures)
5. Add edge masking for top_vars display
6. Fix all-zero importances handling
7. Add missing dropdown options
8. Fix mixed model GA preprocessing

## Phase 3: Medium (Polish)
9. Replace bare except
10. Add warnings for preprocessing fallback
11. Fix misleading parameter warnings

---

# FILES TO MODIFY

| File | # Issues | Key Lines |
|------|----------|-----------|
| `src/spectral_predict/search.py` | 8 | 892-902, 995-1013, 1324-1326, 1225-1233, 1451-1454, 2407-2411, 2427-2429, 2520-2527 |
| `spectral_predict_gui_optimized.py` | 8 | 5377-5378, 7593-7644, 15633+, 18399-18418, 18423, 19855-19868, 20007-20036, 20204-20280, 22026 |
| `src/spectral_predict/preprocess.py` | 1 | 87-91 |

---

# VALIDATION CHECKLIST

After fixes:
- [ ] GA checkbox enables GA preprocessing in search
- [ ] GA preprocessing results load correctly in Model Development
- [ ] Model Development can RUN GA preprocessing models
- [ ] Baseline correction is applied when selected in GUI
- [ ] Smoothing is applied when selected in GUI
- [ ] Edge wavelengths excluded from top_vars with derivatives
- [ ] Mixed model selections work with GA preprocessing
- [ ] All-zero importances raise warning
- [ ] snv_deriv option available in dropdown
- [ ] Ensemble models work in Prediction tab
- [ ] Status messages warn when preprocessing falls back

---

# RECOVERY RESOURCES

If code is too damaged:
- `recovery_blobs/blob_833d05d.txt` - Full search.py
- `recovery_blobs/blob_57e4f52.txt` - Full GUI
- `backup_2025-12-20/` - Dec 16-20 working code
