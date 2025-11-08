# Tab 9 Validation Quick Reference

## Validation Checks at a Glance

### Section B: Load Paired Spectra (`_load_ct_paired_spectra`)

| Check | Type | Line | Trigger | Message Title |
|-------|------|------|---------|---------------|
| Same instrument (pre-load) | Error | 5583-5591 | `master_id == slave_id` | "Same Instrument Selected" |
| Sample count mismatch | Error | 5598-5606 | `X_master.shape[0] != X_slave.shape[0]` | "Sample Count Mismatch" |
| Few samples | Warning | 5608-5618 | `X_master.shape[0] < 20` | "Few Samples" |
| No wavelength overlap | Error | 5627-5635 | `overlap_start >= overlap_end` | "No Wavelength Overlap" |
| Limited overlap | Warning | 5646-5658 | `min_overlap_pct < 80` | "Limited Wavelength Overlap" |

### Section C: Build Transfer Model (`_build_ct_transfer_model`)

| Check | Type | Line | Trigger | Message Title |
|-------|------|------|---------|---------------|
| Data not loaded (hasattr) | Error | 5697-5702 | No `ct_X_master_common` attribute | "No Paired Spectra Loaded" |
| Data not loaded (None) | Error | 5704-5709 | `ct_X_master_common is None` | "No Paired Spectra Loaded" |
| Same instrument | Error | 5716-5721 | `master_id == slave_id` | "Same Instrument Selected" |
| DS Lambda range | Error | 5729-5734 | `lam <= 0 or lam > 100` | "Invalid Parameter" |
| DS Lambda type | Error | 5735-5737 | `ValueError` from float() | "Invalid Parameter" |
| PDS Window range | Error | 5761-5766 | `window < 5 or window > 101` | "Invalid Parameter" |
| PDS Window odd | Error | 5767-5772 | `window % 2 == 0` | "Invalid Parameter" |
| PDS Window type | Error | 5773-5775 | `ValueError` from int() | "Invalid Parameter" |

### Section E: Predict with Transfer Model (`_load_and_predict_ct`)

| Check | Type | Line | Trigger | Message Title |
|-------|------|------|---------|---------------|
| Master model missing | Error | 6107-6112 | `ct_master_model_dict is None` | "Master Model Not Loaded" |
| Transfer model missing | Error | 6114-6119 | `ct_pred_transfer_model is None` | "Transfer Model Not Loaded" |
| Wavelength mismatch | Warning | 6138-6145 | New data has narrower range | "Wavelength Range Mismatch" |
| Extrapolation | Warning | 6167-6173 | Data exceeds model training range | "Extrapolation Warning" |

---

## Parameter Validation Rules

### DS Ridge Lambda
- **Type:** Float
- **Range:** (0, 100]
- **Errors:**
  - Not a number → "DS Ridge Lambda must be a number."
  - Out of range → "DS Ridge Lambda must be between 0 and 100. You entered: {value}"

### PDS Window
- **Type:** Integer
- **Range:** [5, 101]
- **Constraint:** Must be odd
- **Errors:**
  - Not an integer → "PDS Window must be an integer."
  - Out of range → "PDS Window must be between 5 and 101. You entered: {value}"
  - Even number → "PDS Window must be an odd number. You entered: {value} (even)"

---

## Wavelength Validation Logic

### Sample Count Check
```python
if X_master.shape[0] != X_slave.shape[0]:
    # ERROR: Must have exactly same number of samples
```

### Overlap Calculation
```python
overlap_start = max(master_range[0], slave_range[0])
overlap_end = min(master_range[1], slave_range[1])

if overlap_start >= overlap_end:
    # ERROR: No overlap at all

overlap_pct = (overlap_span / min_span) * 100
if overlap_pct < 80:
    # WARNING: Limited overlap
```

### Compatibility Check (Prediction)
```python
if new_range[0] > expected_range[0] or new_range[1] < expected_range[1]:
    # WARNING: Narrower range, may need extrapolation
```

---

## Error Message Format

All error messages follow this pattern:

```
[SPECIFIC VALUES]
Shows actual measurements/counts/ranges

[EXPLANATION]
Why this is a problem

[GUIDANCE]
What the user should do next
```

**Example:**
```
Master has 45 samples, Slave has 38 samples.
↑ SPECIFIC VALUES

Paired spectra must have the same number of samples.
↑ EXPLANATION

Please ensure both instruments measured the exact same samples.
↑ GUIDANCE
```

---

## Validation Flow Diagram

```
Section B: Load Paired Spectra
├─ Pre-checks:
│  ├─ Same instrument? → ERROR
│  └─ Instruments in registry? → ERROR
├─ Load data
├─ Post-load checks:
│  ├─ Sample count match? → ERROR if mismatch
│  ├─ Enough samples? → WARNING if < 20
│  └─ Wavelength overlap?
│     ├─ No overlap → ERROR
│     └─ < 80% overlap → WARNING
└─ Display info (with overlap %)

Section C: Build Transfer Model
├─ Pre-checks:
│  ├─ Data loaded? → ERROR
│  └─ Different instruments? → ERROR
├─ Parameter validation:
│  ├─ DS: Lambda range? → ERROR
│  └─ PDS: Window range, odd? → ERROR
└─ Build model

Section E: Predict
├─ Pre-checks:
│  ├─ Master model loaded? → ERROR
│  └─ Transfer model loaded? → ERROR
├─ Load new slave data
├─ Wavelength checks:
│  ├─ Compatibility → WARNING if narrow
│  └─ Extrapolation → WARNING if exceeds
└─ Predict
```

---

## Quick Debugging Guide

### User sees: "Same Instrument Selected"
**Where:** Section B or C
**Fix:** Select different instruments for master and slave

### User sees: "Sample Count Mismatch"
**Where:** Section B
**Fix:** Ensure both directories contain same sample set

### User sees: "Few Samples"
**Where:** Section B
**Fix:** Add more paired samples (30+ recommended)

### User sees: "No Wavelength Overlap"
**Where:** Section B
**Fix:** Select instruments with compatible wavelength ranges

### User sees: "Limited Wavelength Overlap"
**Where:** Section B
**Fix:** Use instruments with better overlap, or proceed with caution

### User sees: "No Paired Spectra Loaded"
**Where:** Section C
**Fix:** Go to Section B and load paired spectra first

### User sees: "Invalid Parameter" (DS Lambda)
**Where:** Section C
**Fix:** Enter a number between 0 and 100

### User sees: "Invalid Parameter" (PDS Window)
**Where:** Section C
**Fix:** Enter an odd integer between 5 and 101

### User sees: "Master Model Not Loaded"
**Where:** Section E
**Fix:** Go to Section A and load master model

### User sees: "Transfer Model Not Loaded"
**Where:** Section E
**Fix:** Go to Section C and build or load transfer model

### User sees: "Wavelength Range Mismatch"
**Where:** Section E
**Fix:** Use new slave data with same wavelength coverage, or proceed with caution

### User sees: "Extrapolation Warning"
**Where:** Section E
**Fix:** Understand predictions may be unreliable outside training range

---

## File Locations

**Main implementation:** `spectral_predict_gui_optimized.py`
- Lines 5561-5688: Section B validations
- Lines 5690-5804: Section C validations
- Lines 6100-6207: Section E validations

**Documentation:**
- `AGENT4_TAB9_VALIDATION_COMPLETE.md` - Full detailed report
- `TAB9_VALIDATION_QUICK_REFERENCE.md` - This file

**Backup:** `spectral_predict_gui_optimized.py.backup_tab9_validation`
