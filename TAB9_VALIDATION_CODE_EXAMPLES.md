# Tab 9 Validation - Code Examples

This document provides actual code snippets from the implementation for reference.

---

## Section B: Load Paired Spectra

### 1. Same Instrument Check (Lines 5583-5591)

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

**When:** Before loading any spectra
**Why:** Prevents expensive loading operation for invalid configuration
**Type:** Blocking error

---

### 2. Sample Count Mismatch (Lines 5598-5606)

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

**When:** After loading spectra, before resampling
**Why:** Calibration transfer requires paired measurements (same samples on both instruments)
**Type:** Blocking error

---

### 3. Minimum Sample Count (Lines 5608-5618)

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

**When:** After loading spectra
**Why:** Transfer models need sufficient data for robust estimation
**Type:** Non-blocking warning with user choice

---

### 4. Wavelength Overlap (Lines 5620-5658)

```python
# VALIDATION 3: Wavelength Overlap Check
master_range = (wavelengths_master[0], wavelengths_master[-1])
slave_range = (wavelengths_slave[0], wavelengths_slave[-1])

overlap_start = max(master_range[0], slave_range[0])
overlap_end = min(master_range[1], slave_range[1])

# Check for complete non-overlap
if overlap_start >= overlap_end:
    messagebox.showerror(
        "No Wavelength Overlap",
        f"Master range: {master_range[0]:.1f}-{master_range[1]:.1f} nm\n"
        f"Slave range: {slave_range[0]:.1f}-{slave_range[1]:.1f} nm\n\n"
        "Instruments must have overlapping wavelength ranges for calibration transfer.\n\n"
        "Please select instruments with compatible wavelength coverage."
    )
    return

# Calculate overlap percentage
master_span = master_range[1] - master_range[0]
slave_span = slave_range[1] - slave_range[0]
overlap_span = overlap_end - overlap_start

master_overlap_pct = (overlap_span / master_span) * 100
slave_overlap_pct = (overlap_span / slave_span) * 100
min_overlap_pct = min(master_overlap_pct, slave_overlap_pct)

# Warn if limited overlap
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

**When:** After loading spectra
**Why:** Transfer quality depends on having substantial wavelength overlap
**Type:** Error if no overlap, warning if < 80% overlap

**Logic:**
1. Find overlap region: `[max(start1, start2), min(end1, end2)]`
2. If `overlap_start >= overlap_end`: no overlap (error)
3. Calculate overlap as % of each instrument's range
4. Use minimum of the two percentages
5. Warn if < 80%

---

### 5. Updated Info Display (Lines 5676-5679)

```python
# Display info
info_text = (f"Loaded {X_master.shape[0]} paired spectra\n"
            f"Common wavelength grid: {common_wl.shape[0]} points\n"
            f"Range: {common_wl.min():.1f} - {common_wl.max():.1f} nm\n"
            f"Wavelength overlap: {min_overlap_pct:.1f}%")
```

**When:** After successful loading and resampling
**Why:** Shows user the overlap quality metric
**Type:** Informational display

---

## Section C: Build Transfer Model

### 1. Data Loaded Check (Lines 5696-5709)

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

**When:** Before building transfer model
**Why:** Cannot build model without data
**Type:** Blocking error

**Two-stage check:**
1. First check if attributes exist (`hasattr`)
2. Then check if they are None

This prevents `AttributeError` and handles both uninitialized and cleared states.

---

### 2. Different Instruments Check (Lines 5715-5721)

```python
# VALIDATION: Different Instruments Check
if master_id == slave_id:
    messagebox.showerror(
        "Same Instrument Selected",
        "Master and slave instruments must be different for calibration transfer."
    )
    return
```

**When:** Before building transfer model
**Why:** Redundant safety check (also checked in Section B)
**Type:** Blocking error

---

### 3. DS Lambda Validation (Lines 5726-5737)

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

**When:** Before building DS transfer model
**Why:** Invalid lambda causes mathematical errors or poor model quality
**Type:** Blocking error

**Validation criteria:**
- Must be numeric (float)
- Must be > 0 (no zero or negative)
- Must be ≤ 100 (reasonable upper bound)

---

### 4. PDS Window Validation (Lines 5758-5775)

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

**When:** Before building PDS transfer model
**Why:** PDS requires symmetric window (odd number) within reasonable size
**Type:** Blocking error

**Validation criteria:**
- Must be integer
- Must be ≥ 5 (minimum useful window)
- Must be ≤ 101 (reasonable upper bound)
- Must be odd (symmetric window: (n-1)/2 on each side)

---

## Section E: Predict with Transfer Model

### 1. Models Loaded Check (Lines 6106-6119)

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

**When:** Before loading new slave data and predicting
**Why:** Need both models to perform prediction
**Type:** Blocking error

**Two separate checks for clearer error messages:**
- Master model check → directs to Section A
- Transfer model check → directs to Section C

---

### 2. Wavelength Compatibility Check (Lines 6130-6145)

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

**When:** After loading new slave data
**Why:** Warns if new data requires extrapolation to match transfer model
**Type:** Non-blocking warning

**Logic:**
- If new data starts later: `new_range[0] > transfer_range[0]`
- If new data ends earlier: `new_range[1] < transfer_range[1]`
- Either case requires extrapolation during resampling

---

### 3. Extrapolation Warning (Lines 6164-6173)

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

**When:** After transfer, before prediction
**Why:** Warns if predictions require extrapolation beyond training data
**Type:** Non-blocking warning

**Logic:**
- Only warns if `wavelength_range` exists in model dict
- Checks if transferred data extends beyond training range
- Non-blocking (predictions still proceed)

---

## Common Patterns Used

### Pattern 1: Try-Except for Type Validation

```python
try:
    value = float(entry.get())  # or int()
    # Range validation here
except ValueError:
    messagebox.showerror("Invalid Parameter", "Must be a number.")
    return
```

**Use when:** Converting user string input to numeric types

---

### Pattern 2: Attribute Existence Check

```python
if not hasattr(self, 'attribute_name'):
    messagebox.showerror("Error", "Data not loaded.")
    return

if self.attribute_name is None:
    messagebox.showerror("Error", "Data not loaded.")
    return
```

**Use when:** Checking for data that may not have been initialized

---

### Pattern 3: Range Overlap Calculation

```python
overlap_start = max(range1[0], range2[0])
overlap_end = min(range1[1], range2[1])

if overlap_start >= overlap_end:
    # No overlap
else:
    overlap_span = overlap_end - overlap_start
    # Calculate percentages
```

**Use when:** Checking wavelength or any other range overlaps

---

### Pattern 4: Non-Blocking Warning with User Choice

```python
response = messagebox.askokcancel(
    "Warning Title",
    "Warning message with explanation.\n\n"
    "Do you want to continue anyway?"
)
if not response:
    return
```

**Use when:** Situation is concerning but user might have valid reason to proceed

---

## Testing Each Validation

### Quick Manual Test for Each Check:

```python
# Section B - Same Instrument (pre-load)
# 1. Select same instrument for master and slave
# 2. Click "Load Paired Spectra"
# Expected: Error dialog

# Section B - Sample Count Mismatch
# 1. Create two directories with 10 and 15 ASD files
# 2. Load as paired spectra
# Expected: Error with "Master has 10 samples, Slave has 15 samples"

# Section B - Few Samples
# 1. Load only 15 paired samples
# Expected: Warning with option to continue

# Section B - No Overlap
# 1. Use VIS instrument (350-700nm) and NIR instrument (1000-2500nm)
# Expected: Error showing no overlap

# Section B - Limited Overlap
# 1. Use instruments with 60% overlap
# Expected: Warning with overlap percentage

# Section C - No Data Loaded
# 1. Click "Build Transfer Model" without loading paired spectra
# Expected: Error directing to Section B

# Section C - DS Lambda Invalid
# 1. Load paired spectra
# 2. Enter DS Lambda = -5 or 150 or "abc"
# 3. Click "Build DS Transfer Model"
# Expected: Error showing invalid value

# Section C - PDS Window Invalid
# 1. Load paired spectra
# 2. Enter PDS Window = 20 (even) or 3 or 200
# 3. Click "Build PDS Transfer Model"
# Expected: Error showing invalid value

# Section E - Models Not Loaded
# 1. Go to Section E without loading models
# 2. Click "Load and Predict"
# Expected: Error directing to Section A or C

# Section E - Wavelength Mismatch
# 1. Build transfer model with certain range
# 2. Try to predict on data with narrower range
# Expected: Warning about extrapolation

# Section E - Extrapolation Warning
# 1. Load master model with limited range
# 2. Predict on data exceeding that range
# Expected: Warning about unreliable predictions
```

---

## File Information

**Implementation File:** `C:\Users\sponheim\git\dasp\spectral_predict_gui_optimized.py`

**Section Locations:**
- Section B: Lines 5561-5688
- Section C: Lines 5690-5804
- Section E: Lines 6100-6207

**Total Lines of Validation Code:** ~130 lines
**Total Validation Checks:** 17 checks (13 errors, 4 warnings)

---

## Validation Design Philosophy

1. **Fail Fast:** Check preconditions before expensive operations
2. **Be Specific:** Show actual values, not generic "invalid input"
3. **Guide Users:** Tell them what section/action to take
4. **Error vs Warning:**
   - Error: Operation cannot proceed safely
   - Warning: Operation might work but results may be suboptimal
5. **User Choice:** For warnings, let user decide if they want to proceed
6. **Progressive Disclosure:** Check most basic issues first (e.g., data loaded before parameter validation)

---

## Future Enhancement Ideas

1. **Wavelength Grid Validation:** Check for irregular spacing or gaps
2. **Spectral Quality Metrics:** Warn if spectra have unusual patterns
3. **Memory Warnings:** Alert for very large datasets
4. **Duplicate Detection:** Check for duplicate sample names
5. **NaN/Inf Detection:** Validate spectral values are finite
6. **Model Compatibility:** Check master model type matches transfer expectations
7. **File Format Validation:** More robust ASD file parsing with specific error messages
8. **Progress Indicators:** Show progress during long loading operations
9. **Undo/Reset:** Allow user to clear loaded data and start over
10. **Save Validation Report:** Export validation results to log file
