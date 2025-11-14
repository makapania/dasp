# Calibration Transfer Redesign - Handoff Document

**Date:** 2025-11-13
**Branch:** `claude/calibration-transfer-plan-011CV5Jyzu4PKSbQJ3vCqrsA`
**Status:** Phase 1 Complete, Phase 2 In Progress

---

## ✅ COMPLETED (All Committed)

### Commit 1: Algorithm Fixes
**Hash:** 5d90a14
**File:** `src/spectral_predict/calibration_transfer.py`

#### CTAI Complete Rewrite
- **Problem:** Used broken covariance-space approach (M = C_slave^-1 @ C_cross where C_cross was actually master self-covariance!)
- **Solution:** Rewrote as PCA-regularized regression in data space
- **Results:** 95%+ RMSE improvement, transformation values now reasonable (~1x instead of 500x)
- **Key Changes:**
  - Lines 756-830: New algorithm using SVD on slave data, project to PC space, solve in reduced space
  - Lines 712-743: Comprehensive input validation (NaN/inf checks)
  - Lines 783-830: Step-by-step debug logging
  - Now requires paired samples (n_master == n_slave)

#### NS-PFCE Dictionary Key Fixes
- **Problem:** Implementation returned wrong keys ('convergence_iterations', 'transformation_matrix', 'objective_history') but GUI expected ('n_iterations', 'T', 'convergence_history', 'converged')
- **Solution:** Added all expected keys as aliases while maintaining backward compatibility
- **Key Changes:**
  - Lines 1144-1165: Updated params dictionary with dual naming
  - Line 1159: Added NEW 'converged' boolean flag: `(convergence_iterations < max_iterations)`

### Commit 2: GUI Improvements
**Hash:** 82233f8
**File:** `spectral_predict_gui_optimized.py`

#### Tabbed Derivative Views
- **Location:** Lines 14917-15045
- **Feature:** 3-tab notebook (Raw / 1st Deriv / 2nd Deriv) in calibration transfer plots
- **Implementation:**
  - Uses Savitzky-Golay filter for derivatives (window=11, polyorder=2)
  - Helper function `compute_derivative()` at line 14925
  - Helper function `create_comparison_figure()` at line 14943
  - Each tab has export button and same 3-subplot layout

#### Improved Error Messages
- **Location:** Lines 13606-13650
- **Feature:** Clearer diagnostics when instrument spectral data not loaded
- **Explains:** Registry JSON contains metadata only, not actual spectra
- **Provides:** Step-by-step instructions to fix (go to Instrument Lab, re-characterize)

---

## 🔄 IN PROGRESS (Not Committed)

### Current State
- Line 1663: Instrument Lab tab creation **commented out**
- File `spectral_predict_gui_optimized.py.backup` created as restore point
- No other changes made yet

### Remaining Work
**Goal:** Complete 3-step wizard redesign of Calibration Transfer tab

---

## 📋 PHASE 2: COMPLETE REDESIGN PLAN

### The Problem
**Current workflow is broken:**
1. User loads data in Instrument Lab → characterizes (computes 8 metrics)
2. Metrics are NOT used by any calibration transfer method
3. Registry JSON saves metadata but NOT spectral data
4. Loading registry requires re-characterizing to get data back
5. "Import from Instrument Lab" errors out if data not loaded
6. Confusing, indirect, broken

### The Solution: 3-Step Wizard

#### STEP 1: Get/Build Transfer Model

**UI Structure:**
```
Section A: Transfer Model
  Radio buttons or tabs:

  [ ] Load Existing Transfer Model
      - Browse button → .pkl file
      - Display: method, master/slave IDs, date, wavelengths
      - Show preview plots if possible

  [ ] Build New Transfer Model
      Sub-section A1: Load Data
        - "Load Master Spectra" button → Browse folder/file
        - "Load Slave Spectra" button → Browse folder/file
        - Display: wavelength ranges, sample counts, overlap %
        - Preview plots (Master vs Slave)

      Sub-section A2: Configure Method
        - Method dropdown: DS, PDS, TSR, CTAI, NS-PFCE, JYPLS-inv
        - Method-specific parameters (reuse existing UI)
        - "Build Transfer Model" button
        - "Save Transfer Model" button (after built) → .pkl
```

**Data Structures:**
```python
# In __init__:
self.current_transfer_model = None  # TransferModel object
self.current_master_data = None  # (wavelengths, X) tuple
self.current_slave_data = None  # (wavelengths, X) tuple
self.master_data_format = None  # 'csv', 'npy', 'folder', etc.
self.slave_data_format = None
```

**Key Functions to Implement:**
- `_load_master_spectra()` - Browse and load master data, store format
- `_load_slave_spectra()` - Browse and load slave data, store format
- `_build_transfer_model_new()` - Build model from loaded data
- `_save_transfer_model()` - Save as .pkl with metadata
- `_load_transfer_model()` - Load from .pkl
- `_detect_data_format()` - Determine CSV vs NPY vs folder structure

#### STEP 2: Choose Application Mode

**UI Structure:**
```
Section B: How to Use Transfer Model

  Radio buttons:
  [ ] Mode A: Predict Properties
      Icon: 📊
      "Apply transfer to predict properties (e.g., % nitrogen) from new slave data"
      Requires: Master prediction model + new slave data

  [ ] Mode B: Transform & Export
      Icon: 📁
      "Transform slave spectra to master domain and export files"
      Requires: Only new slave data (NO prediction model needed)
```

**Data Structure:**
```python
self.application_mode = None  # 'predict' or 'export'
```

**Key Functions:**
- `_on_mode_selected()` - Update UI based on mode selection

#### STEP 3A: Prediction Workflow (if Mode A)

**UI Structure:**
```
Section C: Apply Transfer & Predict

  C1: Load Master Prediction Model
      - Dropdown: "Select from registry" (current session models)
      - OR "Load from file" button → .pkl
      - Display: model type, target, training metrics

  C2: Load New Slave Data
      - "Load Slave Spectra" button
      - Display: sample count, wavelengths
      - Validate: matches transfer model grid

  C3: Run Prediction
      - "Run Prediction" button
      - Results table: Sample ID | Predicted Value | Confidence
      - Plots: distribution, spectra comparison
      - "Export Predictions" button → CSV
```

**Key Functions:**
- `_load_prediction_model()` - Load sklearn model from file or registry
- `_load_new_slave_data_predict()` - Load data for prediction
- `_run_prediction_workflow()` - Apply transfer → predict → display
- `_export_predictions()` - Save to CSV

#### STEP 3B: Export Workflow (if Mode B)

**UI Structure:**
```
Section C: Transform & Export Spectra

  C1: Load New Slave Data
      - "Load Slave Spectra to Transform" button
      - Display: sample count, wavelengths, detected format
      - Validate: compatible with transfer model

  C2: Transform
      - "Transform Spectra" button
      - Preview: Before/After plots with tabbed derivatives
      - Statistics: RMSE, coverage

  C3: Export
      - "Export Transformed Spectra" button
      - Format: **Match input format automatically**
      - Location picker
      - Preserve file naming/structure from input
```

**Key Functions:**
- `_load_new_slave_data_export()` - Load data for transformation
- `_transform_spectra()` - Apply transfer, show preview
- `_export_transformed_spectra()` - Save in original format with preserved names

**Format Preservation Logic:**
```python
def _export_transformed_spectra(self):
    if self.slave_data_format == 'csv':
        # Export as CSV with same structure
    elif self.slave_data_format == 'npy':
        # Export as .npy files
    elif self.slave_data_format == 'folder':
        # Recreate folder structure with transformed data
```

---

## 📂 FILE STRUCTURE

### Files to Modify
1. **spectral_predict_gui_optimized.py** (~560 lines to rewrite)
   - Lines 15373-15930: Complete `_create_tab10_calibration_transfer()` rewrite
   - Lines 13197-13592: Update/remove helper methods that reference Instrument Lab
   - Lines 169-172: Remove `instrument_profiles`, `instrument_spectral_data` dicts

2. **src/spectral_predict/calibration_transfer.py** (✅ Already updated, committed)
   - No changes needed

### Files to Delete (Later)
- Lines 12868-13192: `_create_tab9_instrument_lab()` and all helper methods
- Most of `src/spectral_predict/instrument_profiles.py` (keep basic I/O if needed)

### Backup Created
- `spectral_predict_gui_optimized.py.backup` - Restore point before redesign

---

## 🔧 IMPLEMENTATION GUIDE

### Step-by-Step Implementation

#### Phase 1: Remove Dependencies (NEXT)
1. Comment out all Instrument Lab helper methods (lines 12868-13192)
2. Remove from `__init__`:
   ```python
   # DELETE THESE:
   self.instrument_profiles = {}
   self.instrument_spectral_data = {}
   ```
3. Add new data structures:
   ```python
   # ADD THESE:
   self.current_transfer_model = None
   self.current_master_data = None
   self.current_slave_data = None
   self.current_prediction_model = None
   self.application_mode = None
   self.master_data_format = None
   self.slave_data_format = None
   ```

#### Phase 2: Build Step 1 UI
1. Replace `_create_tab10_calibration_transfer()` starting at line 15373
2. Create UI with Load/Build radio buttons
3. Implement data loading functions
4. Implement transfer model save/load
5. Reuse existing model building code (lines 15500-15900)

#### Phase 3: Build Step 2 UI
1. Add mode selection radio buttons
2. Connect to state variable
3. Implement dynamic UI updates

#### Phase 4: Build Step 3A (Prediction)
1. Implement prediction model loading
2. Wire up prediction workflow
3. Add results display and export

#### Phase 5: Build Step 3B (Export)
1. Implement format detection
2. Wire up transformation workflow
3. Add export with format preservation

#### Phase 6: Testing
1. Test each workflow end-to-end
2. Verify file format preservation
3. Test with real data

#### Phase 7: Cleanup
1. Delete commented Instrument Lab code
2. Remove backup file
3. Final commit

---

## 🚨 CRITICAL DETAILS

### Transfer Model Save Format
```python
# When saving:
transfer_model_data = {
    'model': transfer_model,  # TransferModel object
    'method': 'ctai',
    'master_id': 'optional_label',
    'slave_id': 'optional_label',
    'date_created': datetime.now().isoformat(),
    'wavelengths_common': wavelengths_array,
    'n_samples': n,
    'metadata': {...}
}
pickle.dump(transfer_model_data, file)
```

### Format Detection Logic
```python
def _detect_data_format(self, filepath):
    """Detect input data format for export matching."""
    if filepath.endswith('.csv'):
        return 'csv'
    elif filepath.endswith('.npy'):
        return 'npy'
    elif os.path.isdir(filepath):
        # Check folder structure
        return 'folder_structure'
    # etc.
```

### Existing Code to Reuse
- **Method selection UI:** Lines 15640-15790 (dropdowns, parameters)
- **Transfer model building:** Lines 15793-15900 (_build_ct_transfer_model logic)
- **Plot generation:** Lines 14878-15045 (_plot_transfer_quality with tabs)
- **Registry table:** Lines 15451-15480 (ct_registry_tree)

---

## 💾 GIT STATUS

```
Branch: claude/calibration-transfer-plan-011CV5Jyzu4PKSbQJ3vCqrsA
Commits ahead: 2

Committed:
  5d90a14 - CTAI rewrite + NS-PFCE fixes
  82233f8 - Tabbed derivative views + error messages

Uncommitted changes:
  M spectral_predict_gui_optimized.py (line 1663 commented out)

Untracked:
  CALIBRATION_TRANSFER_FIXES.md
  test_calibration_fixes.py
  test_ctai_fix.py
  spectral_predict_gui_optimized.py.backup
```

---

## 🎯 SUCCESS CRITERIA

The redesign is complete when:

1. ✅ Instrument Lab tab removed completely
2. ✅ Calibration Transfer has clear 3-step wizard
3. ✅ Users can load/build transfer models in Step 1
4. ✅ Users can save/load transfer models as .pkl
5. ✅ Users can choose Predict vs Export in Step 2
6. ✅ Prediction workflow works (Step 3A)
7. ✅ Export workflow preserves file format (Step 3B)
8. ✅ No dependency on instrument_profiles or instrument_spectral_data
9. ✅ All workflows tested end-to-end
10. ✅ Code committed and documented

---

## 📞 NEXT SESSION PROMPT

"I'm continuing the Calibration Transfer redesign from the handoff document. I need to:

1. Complete the 3-step wizard implementation in `_create_tab10_calibration_transfer()`
2. The branch is `claude/calibration-transfer-plan-011CV5Jyzu4PKSbQJ3vCqrsA`
3. CTAI and NS-PFCE fixes are already committed
4. Instrument Lab tab is commented out (line 1663)
5. See CALIBRATION_TRANSFER_REDESIGN_HANDOFF.md for complete plan

Start with Phase 1: Remove dependencies from __init__ and comment out Instrument Lab methods."

---

## 📚 REFERENCE LINKS

- Handoff doc: `CALIBRATION_TRANSFER_REDESIGN_HANDOFF.md`
- Algorithm fixes doc: `CALIBRATION_TRANSFER_FIXES.md`
- Test scripts: `test_calibration_fixes.py`, `test_ctai_fix.py`
- Backup: `spectral_predict_gui_optimized.py.backup`

---

**End of Handoff Document**
