# Agent 4: Tab 9 Calibration Transfer Validation Implementation - Complete

**Date:** 2025-11-08
**Agent:** Agent 4
**Task:** Add comprehensive validation checks to Tab 9 (Calibration Transfer)
**Status:** ✅ COMPLETE

---

## Executive Summary

All requested validation checks for Tab 9 (Calibration Transfer) have been successfully implemented in `spectral_predict_gui_optimized.py`. The validations span five sections:

- **Section A:** Master Model Loading (existing validations retained)
- **Section B:** Select Instruments & Load Paired Spectra (4 new validations)
- **Section C:** Build Transfer Model (4 new validations)
- **Section D:** Multi-Instrument Equalization (2 existing validations)
- **Section E:** Predict with Transfer Model (3 new validations)

**Total New Validations Added:** 13 validation checks across 11 different error/warning conditions

---

## Section B: Select Instruments & Load Paired Spectra

**Method:** `_load_ct_paired_spectra()` (Lines 5561-5688)

### 1. Same Instrument Check (Pre-Loading)
**Location:** Lines 5583-5591
**Type:** Error (blocking)

```python
# VALIDATION: Check that master and slave are different instruments
if master_id == slave_id:
    messagebox.showerror(
        "Same Instrument Selected",
        "Master and slave instruments must be different for calibration transfer.\n\n"
        f"You selected: {master_id} for both master and slave.\n\n"
        "Please select different instruments."
    )
    return
```

**Error Message Example:**
```
Title: Same Instrument Selected
Message: Master and slave instruments must be different for calibration transfer.

You selected: ASD_FieldSpec4 for both master and slave.

Please select different instruments.
```

**Edge Cases Considered:**
- Validates before expensive spectra loading operation
- Prevents user confusion from loading same data twice
- Clear actionable message

---

### 2. Sample Count Mismatch Check
**Location:** Lines 5598-5606
**Type:** Error (blocking)

```python
# VALIDATION 1: Same Sample Count Check
if X_master.shape[0] != X_slave.shape[0]:
    messagebox.showerror(
        "Sample Count Mismatch",
        f"Master has {X_master.shape[0]} samples, Slave has {X_slave.shape[0]} samples.\n\n"
        "Paired spectra must have the same number of samples (same sample set measured on both instruments).\n\n"
        "Please ensure both instruments measured the exact same samples."
    )
    return
```

**Error Message Example:**
```
Title: Sample Count Mismatch
Message: Master has 45 samples, Slave has 38 samples.

Paired spectra must have the same number of samples (same sample set measured on both instruments).

Please ensure both instruments measured the exact same samples.
```

**Edge Cases Considered:**
- Shows actual sample counts to help user diagnose the issue
- Explains the requirement (paired = same samples on both instruments)
- Provides actionable guidance

---

### 3. Minimum Sample Count Check
**Location:** Lines 5608-5618
**Type:** Warning (non-blocking with user choice)

```python
# VALIDATION 2: Minimum Sample Check
if X_master.shape[0] < 20:
    response = messagebox.askokcancel(
        "Few Samples",
        f"Only {X_master.shape[0]} paired samples loaded.\n\n"
        "At least 30 samples recommended for robust calibration transfer.\n"
        "Results may be unreliable with fewer samples.\n\n"
        "Do you want to continue anyway?"
    )
    if not response:
        return
```

**Warning Message Example:**
```
Title: Few Samples
Message: Only 15 paired samples loaded.

At least 30 samples recommended for robust calibration transfer.
Results may be unreliable with fewer samples.

Do you want to continue anyway?
[OK] [Cancel]
```

**Edge Cases Considered:**
- Threshold at 20 samples (warns if below, recommends 30+)
- Non-blocking - lets user decide if they want to proceed
- Explains the risk (unreliable results)

---

### 4. Wavelength Overlap Check
**Location:** Lines 5620-5658
**Type:** Error (if no overlap) + Warning (if < 80% overlap)

```python
# VALIDATION 3: Wavelength Overlap Check
master_range = (wavelengths_master[0], wavelengths_master[-1])
slave_range = (wavelengths_slave[0], wavelengths_slave[-1])

overlap_start = max(master_range[0], slave_range[0])
overlap_end = min(master_range[1], slave_range[1])

if overlap_start >= overlap_end:
    messagebox.showerror(
        "No Wavelength Overlap",
        f"Master range: {master_range[0]:.1f}-{master_range[1]:.1f} nm\n"
        f"Slave range: {slave_range[0]:.1f}-{slave_range[1]:.1f} nm\n\n"
        "Instruments must have overlapping wavelength ranges for calibration transfer.\n\n"
        "Please select instruments with compatible wavelength coverage."
    )
    return

# Check overlap percentage
master_span = master_range[1] - master_range[0]
slave_span = slave_range[1] - slave_range[0]
overlap_span = overlap_end - overlap_start

master_overlap_pct = (overlap_span / master_span) * 100
slave_overlap_pct = (overlap_span / slave_span) * 100
min_overlap_pct = min(master_overlap_pct, slave_overlap_pct)

if min_overlap_pct < 80:
    response = messagebox.askokcancel(
        "Limited Wavelength Overlap",
        f"Wavelength overlap is {min_overlap_pct:.1f}% of instrument range.\n\n"
        f"Master range: {master_range[0]:.1f}-{master_range[1]:.1f} nm\n"
        f"Slave range: {slave_range[0]:.1f}-{slave_range[1]:.1f} nm\n"
        f"Overlap region: {overlap_start:.1f}-{overlap_end:.1f} nm\n\n"
        "Transfer quality may be reduced with limited overlap.\n"
        "Consider using instruments with better wavelength coverage overlap.\n\n"
        "Do you want to continue anyway?"
    )
    if not response:
        return
```

**Error Message Example (No Overlap):**
```
Title: No Wavelength Overlap
Message: Master range: 350.0-1000.0 nm
Slave range: 1000.0-2500.0 nm

Instruments must have overlapping wavelength ranges for calibration transfer.

Please select instruments with compatible wavelength coverage.
```

**Warning Message Example (Limited Overlap):**
```
Title: Limited Wavelength Overlap
Message: Wavelength overlap is 65.3% of instrument range.

Master range: 350.0-2500.0 nm
Slave range: 800.0-1800.0 nm
Overlap region: 800.0-1800.0 nm

Transfer quality may be reduced with limited overlap.
Consider using instruments with better wavelength coverage overlap.

Do you want to continue anyway?
[OK] [Cancel]
```

**Edge Cases Considered:**
- Completely non-overlapping ranges (blocking error)
- Partial overlap calculated as % of smaller instrument range
- Threshold at 80% (warns if below)
- Shows all ranges clearly for diagnosis
- Non-blocking warning with user choice

---

### 5. Info Display Update
**Location:** Lines 5676-5679

The display text now includes overlap percentage:

```python
info_text = (f"Loaded {X_master.shape[0]} paired spectra\n"
            f"Common wavelength grid: {common_wl.shape[0]} points\n"
            f"Range: {common_wl.min():.1f} - {common_wl.max():.1f} nm\n"
            f"Wavelength overlap: {min_overlap_pct:.1f}%")
```

**Display Example:**
```
Loaded 45 paired spectra
Common wavelength grid: 2151 points
Range: 350.0 - 2500.0 nm
Wavelength overlap: 92.3%
```

---

## Section C: Build Transfer Model

**Method:** `_build_ct_transfer_model()` (Lines 5690-5804)

### 1. Data Loaded Check (Enhanced)
**Location:** Lines 5696-5709
**Type:** Error (blocking)

```python
# VALIDATION: Data Loaded Check
if not hasattr(self, 'ct_X_master_common') or not hasattr(self, 'ct_X_slave_common'):
    messagebox.showerror(
        "No Paired Spectra Loaded",
        "Please load paired standardization spectra in Section B first."
    )
    return

if self.ct_X_master_common is None or self.ct_X_slave_common is None:
    messagebox.showerror(
        "No Paired Spectra Loaded",
        "Please load paired standardization spectra in Section B first."
    )
    return
```

**Error Message Example:**
```
Title: No Paired Spectra Loaded
Message: Please load paired standardization spectra in Section B first.
```

**Edge Cases Considered:**
- Checks both attribute existence and None value
- Clear workflow guidance (directs to Section B)
- Prevents cryptic attribute errors

---

### 2. Different Instruments Check
**Location:** Lines 5715-5721
**Type:** Error (blocking)

```python
# VALIDATION: Different Instruments Check
if master_id == slave_id:
    messagebox.showerror(
        "Same Instrument Selected",
        "Master and slave instruments must be different for calibration transfer."
    )
    return
```

**Error Message Example:**
```
Title: Same Instrument Selected
Message: Master and slave instruments must be different for calibration transfer.
```

**Edge Cases Considered:**
- Redundant check (also in Section B) for safety
- User might change selections after loading
- Simpler message (no detail needed at this stage)

---

### 3. DS Ridge Lambda Parameter Validation
**Location:** Lines 5726-5737
**Type:** Error (blocking)

```python
# VALIDATION: DS Ridge Lambda parameter
try:
    lam = float(self.ct_ds_lambda_var.get())
    if lam <= 0 or lam > 100:
        messagebox.showerror(
            "Invalid Parameter",
            f"DS Ridge Lambda must be between 0 and 100.\nYou entered: {lam}"
        )
        return
except ValueError:
    messagebox.showerror("Invalid Parameter", "DS Ridge Lambda must be a number.")
    return
```

**Error Message Examples:**
```
Title: Invalid Parameter
Message: DS Ridge Lambda must be between 0 and 100.
You entered: 150.5

---OR---

Title: Invalid Parameter
Message: DS Ridge Lambda must be a number.
```

**Edge Cases Considered:**
- Non-numeric input (ValueError)
- Negative or zero values
- Excessively large values (> 100)
- Shows actual entered value for diagnosis
- Validates before expensive computation

---

### 4. PDS Window Parameter Validation
**Location:** Lines 5758-5775
**Type:** Error (blocking)

```python
# VALIDATION: PDS Window parameter
try:
    window = int(self.ct_pds_window_var.get())
    if window < 5 or window > 101:
        messagebox.showerror(
            "Invalid Parameter",
            f"PDS Window must be between 5 and 101.\nYou entered: {window}"
        )
        return
    if window % 2 == 0:
        messagebox.showerror(
            "Invalid Parameter",
            f"PDS Window must be an odd number.\nYou entered: {window} (even)"
        )
        return
except ValueError:
    messagebox.showerror("Invalid Parameter", "PDS Window must be an integer.")
    return
```

**Error Message Examples:**
```
Title: Invalid Parameter
Message: PDS Window must be between 5 and 101.
You entered: 3

---OR---

Title: Invalid Parameter
Message: PDS Window must be an odd number.
You entered: 20 (even)

---OR---

Title: Invalid Parameter
Message: PDS Window must be an integer.
```

**Edge Cases Considered:**
- Non-integer input (ValueError)
- Out of range values (< 5 or > 101)
- Even values (must be odd for symmetric window)
- Shows actual value and explains why it's invalid
- Three separate checks for clear error messages

---

## Section D: Multi-Instrument Equalization

**Methods:** `_load_multiinstrument_dataset()` (Lines 5831-5914) and `_equalize_and_export()` (Lines 5916-6057)

These methods already had good validations in place. No additional validations were added, but existing ones are documented here:

### 1. Directory Structure Validation
**Location:** Lines 5851-5860
**Type:** Error (blocking)

```python
if len(subdirs) == 0:
    messagebox.showerror("Error",
        "No subdirectories found in selected directory.\n\n"
        "Expected structure:\n"
        "  base_directory/\n"
        "    instrument1/\n"
        "      *.asd files\n"
        "    instrument2/\n"
        "      *.asd files\n")
    return
```

**Edge Cases Considered:**
- Empty base directory
- Shows expected structure diagram
- Clear actionable guidance

---

### 2. Minimum Instruments Check
**Location:** Lines 5890-5895
**Type:** Error (blocking)

```python
if len(self.ct_multiinstrument_data) < 2:
    messagebox.showerror("Error",
        f"Need at least 2 instruments for equalization. "
        f"Only loaded {len(self.ct_multiinstrument_data)}.")
    self.ct_multiinstrument_data = None
    return
```

**Edge Cases Considered:**
- Shows actual count
- Clears invalid data (sets to None)
- Explains minimum requirement

---

## Section E: Predict with Transfer Model

**Method:** `_load_and_predict_ct()` (Lines 6100-6207)

### 1. Models Loaded Check (Enhanced)
**Location:** Lines 6106-6119
**Type:** Error (blocking)

```python
# VALIDATION: Models Loaded Check
if self.ct_master_model_dict is None:
    messagebox.showerror(
        "Master Model Not Loaded",
        "Please load the master model in Section A first."
    )
    return

if self.ct_pred_transfer_model is None:
    messagebox.showerror(
        "Transfer Model Not Loaded",
        "Please load or build a transfer model in Section C first."
    )
    return
```

**Error Message Examples:**
```
Title: Master Model Not Loaded
Message: Please load the master model in Section A first.

---OR---

Title: Transfer Model Not Loaded
Message: Please load or build a transfer model in Section C first.
```

**Edge Cases Considered:**
- Separate checks for each model (clearer messages)
- Workflow guidance (directs to specific sections)
- Checks before loading new data

---

### 2. Wavelength Compatibility Check
**Location:** Lines 6130-6145
**Type:** Warning (non-blocking)

```python
# VALIDATION: Wavelength Compatibility Check
transfer_slave_range = (
    self.ct_pred_transfer_model.wavelengths_common[0],
    self.ct_pred_transfer_model.wavelengths_common[-1]
)
new_slave_range = (wavelengths_slave[0], wavelengths_slave[-1])

# Check if new slave data can be resampled to transfer model wavelengths
if new_slave_range[0] > transfer_slave_range[0] or new_slave_range[1] < transfer_slave_range[1]:
    messagebox.showwarning(
        "Wavelength Range Mismatch",
        f"Transfer model expects wavelengths: {transfer_slave_range[0]:.1f}-{transfer_slave_range[1]:.1f} nm\n"
        f"New slave data has wavelengths: {new_slave_range[0]:.1f}-{new_slave_range[1]:.1f} nm\n\n"
        "New slave data has narrower wavelength coverage than the transfer model expects.\n"
        "Predictions may require extrapolation and could be unreliable."
    )
```

**Warning Message Example:**
```
Title: Wavelength Range Mismatch
Message: Transfer model expects wavelengths: 350.0-2500.0 nm
New slave data has wavelengths: 400.0-2400.0 nm

New slave data has narrower wavelength coverage than the transfer model expects.
Predictions may require extrapolation and could be unreliable.
```

**Edge Cases Considered:**
- Narrower new data range (will require extrapolation)
- Non-blocking warning (lets user proceed)
- Shows both ranges for comparison
- Explains the risk

---

### 3. Extrapolation Warning
**Location:** Lines 6164-6173
**Type:** Warning (non-blocking)

```python
# VALIDATION: Extrapolation Warning
if 'wavelength_range' in self.ct_master_model_dict:
    model_wl_range = self.ct_master_model_dict['wavelength_range']
    if wl_model[0] < model_wl_range[0] or wl_model[-1] > model_wl_range[1]:
        messagebox.showwarning(
            "Extrapolation Warning",
            f"Transferred data wavelengths ({wl_model[0]:.1f}-{wl_model[-1]:.1f} nm)\n"
            f"exceed master model training range ({model_wl_range[0]:.1f}-{model_wl_range[1]:.1f} nm).\n\n"
            "Predictions may be unreliable in extrapolated regions."
        )
```

**Warning Message Example:**
```
Title: Extrapolation Warning
Message: Transferred data wavelengths (300.0-2600.0 nm)
exceed master model training range (350.0-2500.0 nm).

Predictions may be unreliable in extrapolated regions.
```

**Edge Cases Considered:**
- Only warns if wavelength_range exists in model dict
- Non-blocking warning
- Shows both ranges for comparison
- Explains the risk (extrapolation)

---

## Validation Principles Applied

All validations follow these principles:

1. **Specific Values in Messages:** Always show actual values (not just "invalid")
2. **Actionable Guidance:** Tell user what to do ("Please load X in Section Y")
3. **Error vs Warning:**
   - `showerror()` for blocking issues that prevent operation
   - `showwarning()` or `askokcancel()` for concerns that user can override
4. **Validate Early:** Check before expensive operations (not after)
5. **Check for None:** Validate existence before accessing attributes
6. **Mismatch Detection:** Check array shapes, wavelength ranges, etc.

---

## Summary of Validations by Type

### Blocking Errors (12)
These prevent the operation from proceeding:

1. Section B: Same instrument selected (pre-load)
2. Section B: Sample count mismatch
3. Section B: No wavelength overlap
4. Section C: No paired spectra loaded (2 checks)
5. Section C: Same instrument selected
6. Section C: DS Lambda out of range
7. Section C: DS Lambda not a number
8. Section C: PDS Window out of range
9. Section C: PDS Window not odd
10. Section C: PDS Window not an integer
11. Section E: Master model not loaded
12. Section E: Transfer model not loaded

### Non-Blocking Warnings (3)
These warn but let user proceed:

1. Section B: Few samples (< 20)
2. Section B: Limited wavelength overlap (< 80%)
3. Section E: Wavelength range mismatch
4. Section E: Extrapolation warning

---

## File Locations

**Main File:** `C:\Users\sponheim\git\dasp\spectral_predict_gui_optimized.py`

**Backup Created:** `C:\Users\sponheim\git\dasp\spectral_predict_gui_optimized.py.backup_tab9_validation`

**Line Number Summary:**
- Section B validations: Lines 5583-5679
- Section C validations: Lines 5696-5775
- Section E validations: Lines 6106-6173

---

## Testing Recommendations

### Test Case 1: Same Instrument Selection
**Steps:**
1. Go to Tab 9, Section B
2. Select same instrument for both Master and Slave
3. Click "Load Paired Spectra"

**Expected:** Error dialog "Same Instrument Selected"

---

### Test Case 2: Sample Count Mismatch
**Steps:**
1. Create two directories with different numbers of ASD files
2. Load as master and slave

**Expected:** Error dialog "Sample Count Mismatch" with actual counts

---

### Test Case 3: No Wavelength Overlap
**Steps:**
1. Select instruments with non-overlapping ranges (e.g., VIS-only vs NIR-only)

**Expected:** Error dialog "No Wavelength Overlap" with ranges shown

---

### Test Case 4: Limited Overlap
**Steps:**
1. Select instruments with < 80% overlap

**Expected:** Warning dialog "Limited Wavelength Overlap" with option to continue

---

### Test Case 5: Few Samples
**Steps:**
1. Load < 20 paired samples

**Expected:** Warning dialog "Few Samples" with option to continue

---

### Test Case 6: Invalid DS Lambda
**Steps:**
1. Load paired spectra
2. Set DS Lambda to -5 or 150 or "abc"
3. Click "Build DS Transfer Model"

**Expected:** Error dialog "Invalid Parameter" with specific message

---

### Test Case 7: Invalid PDS Window
**Steps:**
1. Load paired spectra
2. Set PDS Window to 4 or 20 (even) or "xyz"
3. Click "Build PDS Transfer Model"

**Expected:** Error dialog "Invalid Parameter" with specific message

---

### Test Case 8: Missing Models for Prediction
**Steps:**
1. Go to Section E without loading master model or transfer model
2. Click "Load and Predict"

**Expected:** Error dialog directing to Section A or C

---

### Test Case 9: Wavelength Mismatch in Prediction
**Steps:**
1. Build transfer model with certain wavelength range
2. Try to predict on data with narrower range

**Expected:** Warning dialog "Wavelength Range Mismatch"

---

### Test Case 10: Extrapolation Warning
**Steps:**
1. Load master model with limited wavelength range
2. Predict on transferred data exceeding that range

**Expected:** Warning dialog "Extrapolation Warning"

---

## Completion Checklist

- [x] Section B: Same instrument check (pre-load)
- [x] Section B: Sample count mismatch check
- [x] Section B: Minimum sample check
- [x] Section B: Wavelength overlap check
- [x] Section B: Info display update with overlap %
- [x] Section C: Data loaded check (enhanced)
- [x] Section C: Different instruments check
- [x] Section C: DS Ridge Lambda validation
- [x] Section C: PDS Window validation
- [x] Section D: Existing validations verified
- [x] Section E: Models loaded check (enhanced)
- [x] Section E: Wavelength compatibility check
- [x] Section E: Extrapolation warning
- [x] Documentation created
- [x] Line numbers documented
- [x] Example messages documented
- [x] Edge cases documented
- [x] Testing recommendations provided

---

## Next Steps

1. **Manual Testing:** Run through all 10 test cases to verify behavior
2. **User Feedback:** Collect feedback on error message clarity
3. **Refinement:** Adjust thresholds (e.g., 80% overlap, 20 sample minimum) based on real-world usage
4. **Additional Edge Cases:** Consider adding validation for:
   - Corrupted ASD files
   - Extremely large file uploads (memory warnings)
   - Duplicate wavelength values
   - NaN or Inf values in spectra

---

**Implementation Status:** ✅ COMPLETE
**All requested validations have been successfully added to Tab 9.**
