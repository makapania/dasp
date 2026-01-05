# Unified Bayesian Model Development Integration - Handoff

**Date:** 2026-01-04 (Updated)
**Status:** PARTIALLY FIXED - Top models work, Window size bug remaining

---

## WHAT'S BEEN FIXED (This Session)

### 1. Wavelength Ordering - FIXED (Commit 7ccdfd4)
**Problem:** Wavelengths were stored in spectral order but training used importance order.

**Root Cause:** Code at line 625 had `sorted_indices = np.sort(top_indices)` which sorted wavelengths to spectral order before storage, but training used `top_indices` directly (importance order).

**Fix:** Removed `np.sort(top_indices)` - wavelengths now stored in training order (importance order).

**File:** `src/spectral_predict/unified_bayesian.py` lines 625-638

**Result:** R2 now matches EXACTLY for top-ranked models (1-10).

### 2. Previously Fixed (Still Working)
- `Poly` column - required by report.py
- `ROC_AUC` placeholder for classification
- `all_vars` column - ALL wavelengths for model reconstruction
- Report generation - no longer crashes
- Model Development loads - no longer crashes with empty Pipeline

---

## WHAT'S STILL BROKEN

### Window Size Not Read from Config

**Symptom:** R2 matches for models with window=11, but diverges for other window sizes (7, 15, 17, etc.)

**Root Cause:** Model Development reads `Deriv` from config but NOT `Window` for non-coupled results.

**Location:** `spectral_predict_gui_optimized.py` lines 21139-21148

**Current Code (Missing Window):**
```python
elif self.selected_model_config is not None:
    config_deriv = self.selected_model_config.get('Deriv', None)
    if config_deriv is not None and not pd.isna(config_deriv):
        deriv = int(config_deriv)
        polyorder = get_polyorder_from_deriv(deriv)
        print(f"DEBUG: Using deriv={deriv}, polyorder={polyorder} from loaded config")
    # MISSING: No code reads Window from config!
```

**Why window=11 works:** The coupled parser (line 21007) has `'window': 11` as hardcoded default. When Window isn't read from config, it falls back to 11.

---

## THE FIX NEEDED

Add code to read Window from config for non-coupled results:

**After line ~21147, add:**
```python
# Also read Window from config (for Grid Search, Bayesian results)
config_window = self.selected_model_config.get('Window', None)
if config_window is not None and not pd.isna(config_window):
    window = int(config_window)
    print(f"DEBUG: Using window={window} from loaded config")
```

---

## CODE TRACE - Where Window Flows

| Step | File | Lines | What Happens | Status |
|------|------|-------|--------------|--------|
| 1. Storage | unified_bayesian.py | 611 | Window stored: `trial.set_user_attr('window', ...)` | Working |
| 2. DataFrame | unified_bayesian.py | 880 | Window in DataFrame: `'Window': trial.user_attrs.get('window', 0)` | Working |
| 3. Load | gui_optimized.py | 19338 | Window read: `config.get('Window', 17)` and UI set | Working |
| 4. Execute | gui_optimized.py | 20947-20955 | Window from UI state | Working |
| 5. **BUG** | gui_optimized.py | 21139-21148 | Deriv read from config, **but NOT Window** | BROKEN |
| 6. Default | gui_optimized.py | 21007 | Fallback: `'window': 11` hardcoded | Why 11 works |

---

## VERIFICATION AFTER FIX

1. Run Unified Bayesian with various window sizes (7, 11, 15, 17, etc.)
2. Double-click results in Model Development
3. R2 should match for ALL window sizes, not just 11

---

## COMMITS FROM THIS SESSION

```
7ccdfd4 - fix: Store wavelengths in training order (removed np.sort)
a757086 - revert: Remove preprocessing name stripping that broke Model Development
361f8aa - fix: Sort wavelengths only for storage, not during training
e6b631a - fix: Match grid search preprocessing name format for Model Development
08a3c30 - fix: Add missing columns and spectral order for Model Development
```

---

## KEY FILES

| File | Purpose | Key Lines |
|------|---------|-----------|
| `src/spectral_predict/unified_bayesian.py` | Unified Bayesian optimization | 609-617 (storage), 625-638 (wavelengths), 874-905 (DataFrame) |
| `src/spectral_predict/search.py` | Grid search (reference - WORKS, DO NOT MODIFY) | N/A |
| `spectral_predict_gui_optimized.py` | Model Development | 21139-21148 (BUG: Window not read) |

---

## CRITICAL NOTES FOR NEXT TEAM

1. **Grid Search works perfectly** - DO NOT modify `search.py`
2. **The fix is isolated** - Only need to add 4 lines to read Window from config
3. **Window storage is correct** - The issue is only in reading it back
4. **Test with various window sizes** - 7, 11, 15, 17, etc.

---

## THINGS TRIED THAT DIDN'T WORK

1. **Stripping numbers from preprocessing name** - Broke Model Development completely
2. **Sorting indices during training** - Changed model results (reverted)
3. **Sorting wavelengths for storage** - Was incorrect, fixed in commit 7ccdfd4

---

## USER CONTEXT

- Grid Search works perfectly with Model Development
- Unified Bayesian should work the same way
- User confirmed R2 matches for top models after wavelength fix
- Window size issue identified when testing lower-ranked models
