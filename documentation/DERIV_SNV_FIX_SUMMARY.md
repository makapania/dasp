# deriv_snv Preprocessing Fix Summary

## Problem

You correctly identified that results models and Model Development weren't aligning for `deriv_snv` preprocessing, specifically when 2nd derivative was used.

## Root Cause

**Backend (search.py):**
- When `deriv_snv` is selected, it creates configurations for **BOTH** 1st and 2nd derivatives:
  - `deriv_snv` with `deriv=1, window=7/19, polyorder=2`
  - `deriv_snv` with `deriv=2, window=7/19, polyorder=3`

**Frontend (spectral_predict_gui_optimized.py):**
- The Model Development tab had **hardcoded** deriv and polyorder maps:
  ```python
  deriv_map = {
      'deriv_snv': 1  # Always 1st derivative
  }
  polyorder_map = {
      'deriv_snv': 2  # Always 2nd order (for 1st deriv)
  }
  ```
- When you loaded a 2nd derivative `deriv_snv` result, it was incorrectly rebuilt as 1st derivative

## Solution

Modified `spectral_predict_gui_optimized.py` (lines 2710-2735) to:

1. **Check loaded config first**: Use the actual `Deriv` value from `selected_model_config` when available
2. **Determine polyorder dynamically**: Based on actual derivative order (2 for 1st deriv, 3 for 2nd deriv)
3. **Fall back to maps**: Only use hardcoded maps when creating custom models from scratch

## Code Changes

**File**: `spectral_predict_gui_optimized.py`

**Before** (lines 2710-2712):
```python
preprocess_name = preprocess_name_map.get(preprocess, 'raw')
deriv = deriv_map.get(preprocess, 0)
polyorder = polyorder_map.get(preprocess, 2)
```

**After** (lines 2710-2735):
```python
preprocess_name = preprocess_name_map.get(preprocess, 'raw')

# Use actual derivative order from loaded config if available
if self.selected_model_config is not None:
    config_deriv = self.selected_model_config.get('Deriv', None)
    if config_deriv is not None and not pd.isna(config_deriv):
        deriv = int(config_deriv)
        # Determine polyorder based on actual derivative order
        if deriv == 0:
            polyorder = 2
        elif deriv == 1:
            polyorder = 2
        elif deriv == 2:
            polyorder = 3
        else:
            polyorder = 2  # Fallback
        print(f"DEBUG: Using deriv={deriv}, polyorder={polyorder} from loaded config")
    else:
        # No valid deriv in config, use map
        deriv = deriv_map.get(preprocess, 0)
        polyorder = polyorder_map.get(preprocess, 2)
else:
    # No config loaded, use map (custom model creation)
    deriv = deriv_map.get(preprocess, 0)
    polyorder = polyorder_map.get(preprocess, 2)
```

## Testing

Created and ran test cases verifying:

1. ✓ `deriv_snv` with `deriv=1` → Uses `deriv=1, polyorder=2`
2. ✓ `deriv_snv` with `deriv=2` → Uses `deriv=2, polyorder=3`
3. ✓ No config (custom model) → Falls back to `deriv=1, polyorder=2`

All tests passed successfully.

## Impact

**Before Fix:**
- Loading a `deriv_snv_2nd_w19` result → Incorrectly rebuilt as 1st derivative
- Model performance would not match results table

**After Fix:**
- Loading a `deriv_snv_2nd_w19` result → Correctly rebuilt as 2nd derivative
- Model performance exactly matches results table
- Custom models still work with default 1st derivative

## Commit

```
fix: Resolve deriv_snv preprocessing mismatch between results and model development
Commit: cadc53e
Branch: todays-changes-20251104
```

---

**Status**: ✓ Fixed and tested
**Files Changed**: 1 (spectral_predict_gui_optimized.py)
**Lines Changed**: +25, -2
