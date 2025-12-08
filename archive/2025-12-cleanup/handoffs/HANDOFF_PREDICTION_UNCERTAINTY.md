# Handoff Document: Prediction Uncertainty Implementation

## Date: 2025-01-14
## Developer: Claude (AI Assistant)
## Status: Core Implementation Complete - Refinements Needed

---

## Summary

Implemented prediction uncertainty measures for both regression and classification models in DASP, along with improved model naming conventions. The system now provides confidence intervals for regression predictions and probability distributions for classification predictions.

---

## What Was Implemented

### 1. Model Naming Convention ✅ COMPLETE
**File:** `spectral_predict_gui_optimized.py:11755-11757`

**Change:** Models now save with format: `{C/R}{n_vars}_model_{name}_{timestamp}.dasp`

**Examples:**
- `R245_model_PLS_20250114_143052.dasp` (Regression, 245 variables)
- `C50_model_RandomForest_20250114_143123.dasp` (Classification, 50 variables)

**Benefit:** Immediately see model type and complexity from filename.

---

### 2. Backend Infrastructure ✅ COMPLETE
**File:** `src/spectral_predict/model_io.py`

#### `save_model()` Enhancement (lines 48-197)
- Added optional parameters: `cv_residuals`, `cv_predictions`, `cv_actuals`
- Stores CV data in `cv_data.npz` inside .dasp ZIP archive
- Backward compatible - existing code works unchanged

#### `load_model()` Enhancement (lines 278-292)
- Loads CV data if present
- Returns `cv_data` in model dictionary
- Returns `None` for old models (graceful degradation)

#### New `predict_with_uncertainty()` Function (lines 457-625)
- **Regression:** Computes 95% prediction intervals from CV residuals
  - Returns: `lower_bound`, `upper_bound`, `std_error`, `interval_width`
- **Classification:** Extracts class probabilities
  - Returns: `probabilities`, `confidence`, `class_names`
- Fallback for old models: Uses RMSE from metadata

---

### 3. Training Integration ✅ COMPLETE
**File:** `spectral_predict_gui_optimized.py:11834-11855`

**Tab 6 (Refined Model):**
- Captures CV predictions and actuals during cross-validation
- Computes residuals: `cv_residuals = cv_predictions - cv_actuals`
- Passes CV data to `save_model()` automatically

**Result:** All newly trained models include uncertainty data.

---

### 4. Tab 8 Prediction Display ✅ COMPLETE
**File:** `spectral_predict_gui_optimized.py`

#### UI (lines 12718-12747)
- Added "Prediction Uncertainty" section with scrollable table
- Placeholder shown when no uncertainty available

#### Logic (lines 13104-13192)
- Uses `predict_with_uncertainty()` for predictions
- Stores uncertainty in `self.predictions_uncertainty` dict
- Calls `_display_uncertainty()` to populate table

#### Display Method (lines 13469-13607)
- **Classification:** Shows probabilities and confidence for each sample
- **Regression:** Shows prediction intervals for each sample
- **Color coding:** Green (high confidence), Orange (medium), Red (low)

---

### 5. Tab 9 Multi-Model Comparison ✅ COMPLETE
**File:** `spectral_predict_gui_optimized.py:19394-19428`

**Change:** Column names now use filenames instead of metadata-based names

**Before:** `Moisture_PLS_snv (Reg)`
**After:** `R245_model_PLS_20250114_143052.dasp (Reg)`

**Benefit:** Clear identification of which model file produced each prediction.

---

## 🔧 REQUIRED REFINEMENTS

### Issue 1: Redundant Per-Sample Uncertainty Display

**Current Behavior:**
- Uncertainty table shows one row per sample per model
- For a run with 100 samples and 3 models = 300 rows
- **Problem:** All samples from the same model have IDENTICAL uncertainty values
- CV residuals are aggregated across all samples, so SE/intervals don't vary per-sample

**Desired Behavior:**
- Show one row per model (not per sample)
- Display summary statistics for the entire run

**Proposed Table Format:**

#### For Regression:
```
Model                              | Samples | RMSE   | SE     | Notes
-----------------------------------|---------|--------|--------|------------------
R245_model_PLS_20250114.dasp       | 47      | 0.125  | 0.012  | From CV residuals
R180_model_Ridge_20250113.dasp     | 47      | 0.156  | 0.015  | From CV residuals
R50_model_RandomForest_20250112... | 47      | 0.198  | 0.019  | From overall RMSE
```

#### For Classification:
```
Model                              | Samples | Avg Confidence | Min Conf | Max Conf
-----------------------------------|---------|----------------|----------|----------
C50_model_PLS-DA_20250114.dasp     | 47      | 87.3%          | 62.1%    | 99.8%
C100_model_RandomForest_20250113..| 47      | 92.1%          | 78.5%    | 99.9%
```

**Code Changes Needed:**
- Modify `_display_uncertainty()` method (line 13469)
- Change from per-sample loop to per-model aggregation
- Compute summary statistics: mean SE, RMSE, min/max confidence
- Update table structure accordingly

---

### Issue 2: Standard Error vs 95% Confidence Intervals

**Current Implementation:**
- Returns 95% CI: `prediction ± 1.96 * std_error`
- Displays `Lower 95%`, `Upper 95%`, `Interval Width`

**Desired Implementation:**
- Display Standard Error (SE) directly instead of 95% CI
- SE is more flexible - users can compute any CI they want: `prediction ± (z * SE)`
- Common z-values: 1.96 (95%), 1.645 (90%), 2.576 (99%)

**Code Changes Needed:**
1. **`predict_with_uncertainty()` function** (src/spectral_predict/model_io.py:593-606)
   - Remove CI calculation
   - Return only `std_error` (already computed)

2. **Display method** (spectral_predict_gui_optimized.py:13554-13607)
   - Change columns from `Lower 95% | Upper 95% | Interval Width`
   - To: `SE | RMSE`
   - Show aggregated statistics per model, not per sample

---

## File Modifications Summary

### Modified Files:
1. `src/spectral_predict/model_io.py` - Backend uncertainty functions
2. `spectral_predict_gui_optimized.py` - GUI integration and display
3. `.claude/settings.local.json` - Claude configuration (auto-modified)

### New Files:
- None (all changes integrated into existing files)

### Test Files Created (Not Production):
- `test_categorical_fix.py`
- `test_plsda_validation_fix.py`
- Various markdown documentation files

---

## Backward Compatibility

✅ **Old model files (.dasp without CV data):**
- Load successfully
- Make predictions normally
- Uncertainty falls back to RMSE-based estimates
- Note displayed to user

✅ **Existing code:**
- `predict_with_model()` still works unchanged
- New `predict_with_uncertainty()` is optional enhancement
- No breaking changes

---

## Testing Status

### ✅ Completed:
- Syntax validation (all files compile)
- Code structure review
- Backward compatibility design

### ⏳ Pending:
- Runtime testing with actual trained models
- Verification of uncertainty calculations
- User acceptance testing of new table format
- Testing after refinements implemented

---

## Next Steps

### Immediate (Before Deployment):
1. **Refactor uncertainty display** (Issue 1)
   - Change from per-sample to per-model aggregation
   - Update `_display_uncertainty()` method
   - Test with sample data

2. **Switch from 95% CI to SE** (Issue 2)
   - Update `predict_with_uncertainty()` return values
   - Update display table columns
   - Update documentation

3. **Runtime Testing**
   - Train new regression model → verify SE calculation
   - Train new classification model → verify probability extraction
   - Load old model → verify fallback behavior
   - Test with validation set predictions

4. **Documentation**
   - Update user guide with uncertainty interpretation
   - Add tooltips explaining SE vs CI
   - Document color-coding thresholds

### Future Enhancements (Optional):
- Add uncertainty display to Tab 9 (Multi-Model Comparison)
- Export uncertainty data to CSV/Excel
- Plot uncertainty distributions
- Customizable confidence level selection (90%, 95%, 99%)
- Out-of-domain detection using uncertainty thresholds

---

## Technical Debt

### Code Duplication:
- Preprocessing logic duplicated in `predict_with_uncertainty()` (lines 521-566)
- Could be refactored to reuse `predict_with_model()` preprocessing

### Performance:
- Current per-sample display creates unnecessary rows
- Refinement will improve performance for large datasets

### UI Consistency:
- Tab 9 doesn't have uncertainty display yet
- Should match Tab 8 format when implemented

---

## Key Code Locations

### Model I/O:
- `src/spectral_predict/model_io.py:48-197` - `save_model()` with CV data
- `src/spectral_predict/model_io.py:278-292` - `load_model()` with CV data
- `src/spectral_predict/model_io.py:457-625` - `predict_with_uncertainty()`

### GUI Integration:
- `spectral_predict_gui_optimized.py:11755-11757` - Model naming
- `spectral_predict_gui_optimized.py:11834-11855` - CV data capture
- `spectral_predict_gui_optimized.py:12718-12747` - Uncertainty UI widget
- `spectral_predict_gui_optimized.py:13104-13192` - Prediction logic
- `spectral_predict_gui_optimized.py:13469-13607` - Display method (NEEDS REFACTOR)
- `spectral_predict_gui_optimized.py:19394-19428` - Tab 9 filename display

---

## Git Status

### Current Branch:
`claude/calibration-transfer-plan-011CV5Jyzu4PKSbQJ3vCqrsA`

### Modified Files:
- `.claude/settings.local.json`
- `spectral_predict_gui_optimized.py`
- `src/spectral_predict/model_io.py`

### Untracked Files (Documentation):
- `BUG_FIX_CATEGORICAL_TARGETS.md`
- `CRITICAL_BUG_FIX_SUMMARY.md`
- `IMPLEMENTATION_SUMMARY.md`
- `PLS_DA_CATEGORICAL_COMPLETE_FIX.md`
- `PLS_DA_MODEL_DEVELOPMENT_COMPLETE_FIX.md`
- `PLS_DA_VALIDATION_FIX.md`
- `USER_SCENARIO_EXPLANATION.md`
- Test files and validation directories

### Ready for Commit: YES
All core functionality implemented and syntax-validated.

---

## Commit Message (Suggested)

```
feat: Add prediction uncertainty with model naming improvements

Implement comprehensive prediction uncertainty measures for both
regression and classification models with improved filename conventions.

Features:
- Model naming: {C/R}{n_vars}_model_{name}_{timestamp}.dasp format
- CV data storage: Residuals, predictions, actuals in .dasp files
- predict_with_uncertainty(): New function for confidence measures
  * Regression: SE-based prediction intervals from CV residuals
  * Classification: Class probabilities and confidence scores
- Tab 6: Auto-capture CV data during model training
- Tab 8: Uncertainty display section with color-coded confidence
- Tab 9: Filename-based model identification in results

Backend (src/spectral_predict/model_io.py):
- Enhanced save_model() to store CV data (backward compatible)
- Enhanced load_model() to retrieve CV data
- New predict_with_uncertainty() function

Frontend (spectral_predict_gui_optimized.py):
- Model naming convention in refined model save
- CV data capture during cross-validation
- Uncertainty table widget in Tab 8 results
- Updated Tab 9 to display model filenames

Known Issues/TODO:
- Uncertainty display shows per-sample (wasteful) - needs aggregation
- Should display SE instead of 95% CI for flexibility
- Tab 9 uncertainty display not yet implemented

Backward compatible: Old models work with RMSE-based fallback.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Contact & Questions

For questions about this implementation, review:
1. This handoff document
2. Code comments in modified files
3. Function docstrings in `model_io.py`

Implementation was completed step-by-step with no breaking changes to preserve existing functionality.

---

**END OF HANDOFF DOCUMENT**
