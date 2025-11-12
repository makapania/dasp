# Quick Reference: Critical Issues to Fix

## The 7 CRITICAL Issues That Cause R² Failures

### 1. ⭐ Parameter Serialization (10 min fix)
**File**: `src/spectral_predict/search.py:777`
```python
# CURRENT (WRONG)
"Params": str(params),  # Loses type info!

# FIX
import json
from spectral_predict.model_io import _json_serializer
"Params": json.dumps(params, default=_json_serializer),
```
**Impact**: 0.05-0.2+ R² drop  
**Risk**: LOW

---

### 2. 🔴 Model Config Not Stored (30 min fix)
**File**: `src/spectral_predict/search.py:820`
```python
# CURRENT (INCOMPLETE)
result = {
    "Model": model_name,
    "Params": str(params),  # Only grid params!
    ...
}

# FIX
result = {
    "Model": model_name,
    "ModelClass": model.__class__.__name__,
    "ModelParams": model.get_params(),  # ALL params
    "Params": json.dumps(params, default=_json_serializer),
    ...
}
```
**Impact**: 0.05-0.15+ R² drop for NeuralBoosted/MLP  
**Risk**: MEDIUM

---

### 3. 🔴 CV ≠ Full Data Fit (1-2 hour fix)
**File**: `src/spectral_predict/search.py:246-249, 403-406, 730-735`
```python
# PROBLEM:
# - R² computed from CV folds
# - Feature importances from full-data fit
# = DIFFERENT MODEL INSTANCES!

# FIX OPTION A: Average importances across CV folds
importances_cv = []
for train_idx, test_idx in cv_splitter.split(X, y):
    pipe.fit(X[train_idx], y[train_idx])
    importances_cv.append(get_feature_importances(...))
importances = np.mean(importances_cv, axis=0)

# FIX OPTION B: Document the trade-off (faster)
# Add comment: "R² from CV (reliable), importances from full data (faster)"
```
**Impact**: 0.02-0.2 R² variance  
**Risk**: MEDIUM (Option A) / LOW (Option B)

---

### 4. 🔴 Feature Index Mismatch (20 min fix)
**File**: `src/spectral_predict/search.py:503-541`
```python
# PROBLEM: Indices into transformed space, wavelengths are original
# Both have same length (Savitzky-Golay doesn't reduce dims)
# But must track which space we're in!

# FIX: Add explicit tracking
result['feature_space'] = 'preprocessed' if skip_preprocessing else 'original'
result['feature_indices'] = subset_indices.tolist()
result['n_features_input'] = X.shape[1]
```
**Impact**: Catastrophic if features mismatch  
**Risk**: LOW

---

### 5. 🔴 Random State Inconsistency (20 min fix)
**File**: `models.py`, `neural_boosted.py`, `model_io.py`
```python
# PROBLEM: NeuralBoosted early_stopping uses different data split
# If not documented, refinement produces different R²

# FIX: Document and store
metadata['training_config'] = {
    'random_state': 42,
    'validation_fraction': getattr(model, 'validation_fraction', None),
    'early_stopping': getattr(model, 'early_stopping', None),
    'n_iter_no_change': getattr(model, 'n_iter_no_change', None)
}
```
**Impact**: 0.02-0.08 R² variance  
**Risk**: LOW

---

### 6. 🔴 No Model Validation on Load (20 min fix)
**File**: `src/spectral_predict/model_io.py:48-146`
```python
# PROBLEM: save_model() doesn't validate model state matches metadata

# FIX: Add validation
def save_model(model, preprocessor, metadata, filepath):
    # Existing validation...
    required_fields = ['model_name', 'task_type', 'wavelengths', 'n_vars']
    
    # ADD THIS:
    if hasattr(model, 'n_features_in_'):
        if model.n_features_in_ != len(metadata['wavelengths']):
            raise ValueError(
                f"Model has {model.n_features_in_} features "
                f"but metadata has {len(metadata['wavelengths'])} wavelengths"
            )
```
**Impact**: Prevents catastrophic mismatches  
**Risk**: LOW

---

### 7. 🔴 Preprocessing Inconsistency (already partially fixed)
**File**: `src/spectral_predict/search.py:508-525`
```python
# CURRENT (MOSTLY OK, but add documentation)
if preprocess_cfg["deriv"] is not None:
    subset_result = _run_single_config(
        X_transformed,  # Already preprocessed ✓
        y_np,
        wavelengths,
        model,
        model_name,
        params,
        preprocess_cfg,
        cv_splitter,
        task_type,
        is_binary_classification,
        subset_indices=top_indices,
        subset_tag=f"top{n_top}_{varsel_method}",
        skip_preprocessing=True,  # ✓ Correct
    )

# ADD: Documentation comment
# "For derivatives: X_transformed already preprocessed, skip re-applying
#  For non-derivatives: X_np passes through pipeline in _run_single_config"
```
**Impact**: 0.02-0.05 R² if inconsistent  
**Risk**: LOW (documentation only)

---

## The 10 HIGH Priority Issues

### Priority Order for Implementation:

1. **HIGH #1**: Hyperparameter types → Use JSON serialization (same fix as CRITICAL #1)
2. **HIGH #3**: SNV edge case → Add warning for zero variance
3. **HIGH #4**: CV seeds → Make fold indices independent  
4. **HIGH #5**: Feature importance → Weight by CV uncertainty
5. **HIGH #6**: Wavelength precision → Use explicit mapping
6. **HIGH #7**: Region subsets → Standardize computation
7. **HIGH #8**: PLS validation → Check per-fold constraints
8. **HIGH #9**: Model clone → Verify independence in loky
9. **HIGH #10**: Metadata staleness → Flag on refinement

---

## Implementation Checklist

### Phase 1 (2-3 hours):
- [ ] Fix parameter serialization (CRITICAL #1)
- [ ] Store complete model config (CRITICAL #2)
- [ ] Add feature index tracking (CRITICAL #4)
- [ ] Add model validation (CRITICAL #6)
- [ ] Document preprocessing strategy (CRITICAL #7)

### Phase 2 (2-3 hours):
- [ ] Fix CV vs full-data fit (CRITICAL #3) - OPTION A (complex) or OPTION B (simple)
- [ ] Fix SNV edge cases (HIGH #3)
- [ ] Make CV fold seeds independent (HIGH #4)
- [ ] Standardize region computation (HIGH #7)
- [ ] Validate PLS components (HIGH #8)

### Phase 3 (1-2 hours):
- [ ] Parameter validation (MEDIUM #1)
- [ ] Data integrity checks (MEDIUM #3)
- [ ] Better error messages (MEDIUM #2)

---

## Estimated R² Improvement

- **Before fixes**: R² discrepancies 0.05-0.5 (catastrophic)
- **After Phase 1**: R² discrepancies 0.01-0.1 (40-60% improvement)
- **After Phase 2**: R² discrepancies < 0.01 (95% improvement)
- **After Phase 3**: Production quality

---

## Start Here

1. Read `COMPREHENSIVE_CODE_REVIEW.md` for full details
2. Read `ISSUE_SUMMARY_AND_FIXES.txt` for implementation guide
3. Fix in order: CRITICAL #1 → #4 → #2 → #6 → #7 → #3
4. Run tests after each fix
5. Move to HIGH priority issues

---

## File Locations

- Main search logic: `/home/user/dasp/src/spectral_predict/search.py`
- Model I/O: `/home/user/dasp/src/spectral_predict/model_io.py`
- Model definitions: `/home/user/dasp/src/spectral_predict/models.py`
- Preprocessing: `/home/user/dasp/src/spectral_predict/preprocess.py`
- NeuralBoosted: `/home/user/dasp/src/spectral_predict/neural_boosted.py`

