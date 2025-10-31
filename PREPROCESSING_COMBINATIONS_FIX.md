# Preprocessing Combinations Fix

**Date:** 2025-10-29
**Issue:** SNV + derivative combinations not running
**Status:** FIXED

---

## Problem

When you selected both SNV and derivatives (sg1 or sg2), the analysis was running them **separately** but not creating the **combination** preprocessing methods:

### Before Fix (BROKEN):
If you selected:
- ✅ SNV
- ✅ SG2 (2nd derivative)
- Window: 17

You would get:
1. `snv` (SNV only)
2. `deriv` (2nd derivative only, window=17)

**Missing:**
- `snv_deriv` (SNV → 2nd derivative)
- `deriv_snv` (2nd derivative → SNV) [if deriv_snv checked]

### After Fix (WORKING):
Now you get ALL combinations:
1. `snv` (SNV only)
2. `deriv` (2nd derivative only, window=17)
3. `snv_deriv` (SNV → 2nd derivative, window=17) ✅ **NEW**
4. `deriv_snv` (2nd derivative → SNV, window=17) ✅ **NEW** [if deriv_snv checked]

---

## How It Works Now

### Rule 1: Pure Methods Always Created
If you check a preprocessing box, the pure method is always created:
- `raw` → raw spectra
- `snv` → SNV only
- `sg1` → 1st derivative only
- `sg2` → 2nd derivative only

### Rule 2: Auto-Combination with SNV
**If you select BOTH SNV and a derivative,** the combination is **automatically** created:

| You Select | Auto-Created Combinations |
|------------|---------------------------|
| SNV + SG1 | `snv_deriv` (SNV → 1st deriv) |
| SNV + SG2 | `snv_deriv` (SNV → 2nd deriv) |
| SNV + SG1 + SG2 | Both combinations |

### Rule 3: deriv_snv Checkbox
The `deriv_snv` checkbox adds **reverse order** combinations:

| You Select | Created If deriv_snv Checked |
|------------|------------------------------|
| SG1 | `deriv_snv` (1st deriv → SNV) |
| SG2 | `deriv_snv` (2nd deriv → SNV) |
| SG1 + SG2 | Both combinations |

### Complete Example

**Selections:**
- ✅ SNV
- ✅ SG1 (1st derivative)
- ✅ SG2 (2nd derivative)
- ✅ deriv_snv
- ✅ Window: 11, 17

**Preprocessing configs created:**
1. `snv`
2. `deriv` (d=1, w=11)
3. `deriv` (d=1, w=17)
4. `snv_deriv` (d=1, w=11) ← SNV + 1st deriv
5. `snv_deriv` (d=1, w=17) ← SNV + 1st deriv
6. `deriv_snv` (d=1, w=11) ← 1st deriv + SNV
7. `deriv_snv` (d=1, w=17) ← 1st deriv + SNV
8. `deriv` (d=2, w=11)
9. `deriv` (d=2, w=17)
10. `snv_deriv` (d=2, w=11) ← SNV + 2nd deriv
11. `snv_deriv` (d=2, w=17) ← SNV + 2nd deriv
12. `deriv_snv` (d=2, w=11) ← 2nd deriv + SNV
13. `deriv_snv` (d=2, w=17) ← 2nd deriv + SNV

**Total:** 13 preprocessing configs

---

## What Was Fixed in Code

### File: `src/spectral_predict/search.py`
**Lines:** 123-169

### Before (Broken Logic):
```python
if preprocessing_methods.get('sg1', False):
    for window in window_sizes:
        preprocess_configs.append({"name": "deriv", "deriv": 1, ...})

if preprocessing_methods.get('sg2', False):
    for window in window_sizes:
        preprocess_configs.append({"name": "deriv", "deriv": 2, ...})

if preprocessing_methods.get('deriv_snv', False):
    for window in window_sizes:
        preprocess_configs.append({"name": "deriv_snv", "deriv": 1, ...})
```

**Problems:**
1. ❌ Only created pure derivative configs
2. ❌ Never created `snv_deriv` (SNV → derivative)
3. ❌ Only created `deriv_snv` for 1st derivative, not 2nd

### After (Fixed Logic):
```python
if preprocessing_methods.get('sg1', False):
    # 1st derivative only
    for window in window_sizes:
        preprocess_configs.append({"name": "deriv", "deriv": 1, ...})

    # If SNV is also selected, add SNV → derivative combination
    if preprocessing_methods.get('snv', False):
        for window in window_sizes:
            preprocess_configs.append({"name": "snv_deriv", "deriv": 1, ...})

    # If deriv_snv is selected, add derivative → SNV combination
    if preprocessing_methods.get('deriv_snv', False):
        for window in window_sizes:
            preprocess_configs.append({"name": "deriv_snv", "deriv": 1, ...})

if preprocessing_methods.get('sg2', False):
    # 2nd derivative only
    for window in window_sizes:
        preprocess_configs.append({"name": "deriv", "deriv": 2, ...})

    # If SNV is also selected, add SNV → derivative combination
    if preprocessing_methods.get('snv', False):
        for window in window_sizes:
            preprocess_configs.append({"name": "snv_deriv", "deriv": 2, ...})

    # If deriv_snv is selected, add derivative → SNV combination
    if preprocessing_methods.get('deriv_snv', False):
        for window in window_sizes:
            preprocess_configs.append({"name": "deriv_snv", "deriv": 2, ...})
```

**Fixes:**
1. ✅ Creates `snv_deriv` when SNV + derivative selected
2. ✅ Creates `deriv_snv` for BOTH 1st and 2nd derivatives
3. ✅ All combinations respect window size selections

---

## New Debug Output

The analysis now shows a breakdown of all preprocessing configs being tested:

```
Running regression search with 5-fold CV...
Models: ['PLS']
Preprocessing configs: 7

Preprocessing breakdown:
  - snv
  - deriv (deriv=1, window=17)
  - snv_deriv (deriv=1, window=17)
  - deriv_snv (deriv=1, window=17)
  - deriv (deriv=2, window=17)
  - snv_deriv (deriv=2, window=17)
  - deriv_snv (deriv=2, window=17)

Enable variable subsets: True
Variable counts: [10, 20, 50]
Enable region subsets: True
```

This makes it **immediately clear** what preprocessing combinations will be tested.

---

## Expected Behavior Examples

### Example 1: SNV + 2nd Derivative
**GUI Selections:**
- ✅ SNV
- ✅ SG2
- ✅ Window: 17

**Result:** 3 configs
```
- snv
- deriv (deriv=2, window=17)
- snv_deriv (deriv=2, window=17)
```

### Example 2: All Derivatives
**GUI Selections:**
- ✅ SG1
- ✅ SG2
- ✅ Window: 7, 17

**Result:** 4 configs
```
- deriv (deriv=1, window=7)
- deriv (deriv=1, window=17)
- deriv (deriv=2, window=7)
- deriv (deriv=2, window=17)
```

### Example 3: Full Combination
**GUI Selections:**
- ✅ SNV
- ✅ SG1
- ✅ deriv_snv
- ✅ Window: 17

**Result:** 3 configs
```
- snv
- deriv (deriv=1, window=17)
- snv_deriv (deriv=1, window=17)
- deriv_snv (deriv=1, window=17)
```

### Example 4: Kitchen Sink
**GUI Selections:**
- ✅ Raw
- ✅ SNV
- ✅ SG1
- ✅ SG2
- ✅ deriv_snv
- ✅ Window: 7, 11, 17, 19

**Result:** 18 configs
```
- raw
- snv
- deriv (deriv=1, window=7)
- deriv (deriv=1, window=11)
- deriv (deriv=1, window=17)
- deriv (deriv=1, window=19)
- snv_deriv (deriv=1, window=7)
- snv_deriv (deriv=1, window=11)
- snv_deriv (deriv=1, window=17)
- snv_deriv (deriv=1, window=19)
- deriv_snv (deriv=1, window=7)
- deriv_snv (deriv=1, window=11)
- deriv_snv (deriv=1, window=17)
- deriv_snv (deriv=1, window=19)
- deriv (deriv=2, window=7)
- deriv (deriv=2, window=11)
- deriv (deriv=2, window=17)
- deriv (deriv=2, window=19)
- snv_deriv (deriv=2, window=7)
- snv_deriv (deriv=2, window=11)
- snv_deriv (deriv=2, window=17)
- snv_deriv (deriv=2, window=19)
- deriv_snv (deriv=2, window=7)
- deriv_snv (deriv=2, window=11)
- deriv_snv (deriv=2, window=17)
- deriv_snv (deriv=2, window=19)
```

---

## Important Notes

### 1. deriv_snv is Optional
The `deriv_snv` checkbox is for the **reverse order** (derivative then SNV). You don't need to check it to get `snv_deriv` (SNV then derivative) - that's created automatically when you select both SNV and a derivative.

### 2. Each Window Size Creates Separate Configs
If you select multiple window sizes, each combination is created for each window:
- SNV + SG1 + windows [7, 17] = 4 configs (2 deriv + 2 snv_deriv)

### 3. Combinations Multiply Configs
Be aware that selecting many options creates many configs:
- 2 derivatives × 4 windows × 3 variations (deriv, snv_deriv, deriv_snv) = 24 configs
- Each model tests each config, so runtime grows quickly

### 4. Performance Impact
More preprocessing configs = longer runtime:
- 5 configs × 4 models × 8 PLS components = 160 model fits
- 13 configs × 4 models × 8 PLS components = 416 model fits

Choose combinations wisely based on your research needs.

---

## Verification

To verify the fix is working, run an analysis and look for the new output:

```
Preprocessing breakdown:
  - snv
  - deriv (deriv=2, window=17)
  - snv_deriv (deriv=2, window=17)   ← Should see this now!
```

If you see `snv_deriv` or `deriv_snv` in the breakdown when you've selected the appropriate checkboxes, the fix is working!

---

## Summary

**Fixed:** Preprocessing combinations now respect user selections
**Key Change:** Auto-creates `snv_deriv` when SNV + derivatives selected
**Bonus:** deriv_snv now works for both 1st and 2nd derivatives
**New Feature:** Detailed preprocessing breakdown in output

**Status:** ✅ Fixed and tested
**File Modified:** `src/spectral_predict/search.py` lines 123-169, 182-195

---

**Next:** Run your analysis again and you should see all the preprocessing combinations you expect!
