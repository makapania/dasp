# R² Discrepancy Investigation - Deep Dive

**Date:** 2025-11-03
**Status:** 🔍 UNDER INVESTIGATION
**Priority:** CRITICAL

---

## The Mystery

When a model from the Results tab is loaded into the Custom Model Development tab and re-run, there is a **significant R² discrepancy**:

### Observed Discrepancy

```
Main Analysis (Results Tab):
  R² = 0.9443
  Model: PLS
  Preprocessing: snv_sg2 (SNV → 2nd derivative)
  Window: 17
  n_components: 2
  Variables: 50 (subset from variable selection)

Model Development (Re-run):
  R² = 0.8266
  Same configuration (supposedly)

Difference: -0.1176 (11.8% absolute drop!)
```

**User Assessment:** The 0.9443 value is realistic for their data. The 0.8266 value is NOT realistic.

---

## Initial (Incorrect) Hypothesis: Data Leakage

**My initial theory:** Preprocessing before CV = data leakage → inflated R²

**Why I thought this:**
- Main analysis preprocesses data once before CV for derivatives
- Then runs CV with `skip_preprocessing=True`
- This seemed like classic data leakage

**Why this is WRONG:**

Savitzky-Golay derivatives and SNV are **per-spectrum transformations**:
- SG derivative uses a sliding window across **wavelengths** (features), not samples
- Each spectrum is processed **independently**
- SNV normalizes each spectrum using only that spectrum's mean/std
- Neither transformation uses information from other samples

**Proof:** I ran a test comparing both approaches:
```python
Method 1 (preprocess before CV): R² = -0.549490
Method 2 (preprocess inside CV):  R² = -0.549490
Difference: 0.000000  # IDENTICAL!
```

For per-spectrum transformations, preprocessing before vs inside CV gives **identical results**. There is NO data leakage.

**Conclusion:** This is NOT a data leakage issue. The main analysis R² of 0.9443 is legitimate.

---

## Possible Root Causes

### Hypothesis 1: Wavelength Mapping Issue

**Theory:** The wavelengths stored in `all_vars` don't correctly map to the selected variables.

**How variable selection works in main analysis:**
```python
# Step 1: Preprocess full data
X_preprocessed = preprocess(X_raw)  # e.g., SNV → SG2 derivative

# Step 2: Select top variables based on importance in preprocessed space
importances = get_importances(model, X_preprocessed, y)
top_indices = argsort(importances)[-50:]  # Top 50 indices

# Step 3: Save wavelengths corresponding to those indices
all_vars = wavelengths[top_indices]  # e.g., [1525.0, 1526.0, ..., 1574.0]

# Step 4: Run CV on preprocessed data subset
X_subset = X_preprocessed[:, top_indices]
cv_score = cross_validate(model, X_subset, y)  # R² = 0.9443
```

**How model development loads and runs:**
```python
# Step 1: Parse wavelengths from all_vars
selected_wavelengths = parse_all_vars()  # [1525.0, 1526.0, ..., 1574.0]

# Step 2: Select those wavelengths from RAW data
X_subset_raw = X_raw[:, selected_wavelengths]

# Step 3: Preprocess INSIDE CV
pipe = Pipeline([('snv', SNV()), ('sg2', SG2()), ('pls', PLS())])
cv_score = cross_validate(pipe, X_subset_raw, y)  # R² = 0.8266
```

**Potential issue:** If preprocessing changes the feature count or order, the wavelength indices might not align correctly.

**Check:** Do SG derivatives change feature count?
- Answer: NO. `savgol_filter` returns the same shape as input.
- Window size 17 doesn't truncate features.

**Status:** ❓ UNLIKELY but needs verification

---

### Hypothesis 2: Different Subset Selection

**Theory:** The indices used in main analysis don't correspond to the wavelengths saved in `all_vars`.

**Code to check:** `src/spectral_predict/search.py` lines 720-732
```python
if subset_tag != "full" and subset_indices is not None:
    # Save ALL wavelengths used in the subset model
    all_indices = np.arange(len(importances))
    if subset_indices is not None:
        original_wavelengths_all = wavelengths[subset_indices]
        all_wavelengths = original_wavelengths_all[all_indices]
    else:
        all_wavelengths = wavelengths[all_indices]

    all_vars_str = ','.join([f"{w:.1f}" for w in all_wavelengths])
    result['all_vars'] = all_vars_str
```

**Question:** What are `subset_indices` vs `all_indices` here?
- `subset_indices`: The selected variable indices in **preprocessed space**
- `all_indices`: All indices in the subset (0 to len(subset)-1)
- `wavelengths[subset_indices]`: Maps back to original wavelengths

**Potential issue:** Double indexing confusion?

**Status:** ❓ NEEDS INVESTIGATION

---

### Hypothesis 3: Preprocessing Parameter Mismatch

**Theory:** The preprocessing isn't exactly the same between main analysis and model development.

**From debug output:**
```
Loaded Preprocessing: snv_deriv
Loaded Deriv: 2
Loaded Window: 17
```

**Model development code** (lines 2652-2680):
```python
preprocess_name_map = {
    'snv_sg2': 'snv_deriv',  # ← Converts GUI name to search.py name
}
```

**Question:** Is the conversion correct?
- GUI: `snv_sg2`
- Search.py: `snv_deriv` with `deriv=2`
- Pipeline: SNV → SavgolDerivative(deriv=2, window=17, polyorder=3)

**Check polyorder:**
- Line 2672-2679: `polyorder_map = {'snv_sg2': 3}`
- Line 2684: `polyorder = polyorder_map.get(preprocess, 2)` → 3 ✓

**Status:** ✅ UNLIKELY - parameters match

---

### Hypothesis 4: CV Fold Randomness

**Theory:** Different random seeds or different CV splits.

**Main analysis:** `KFold(n_splits=5, shuffle=True, random_state=42)`
**Model development:** `KFold(n_splits=5, shuffle=True, random_state=42)`

**Expected variation from randomness:** ~0.001-0.01 (not 0.12!)

**Status:** ✅ NOT THE CAUSE - difference too large

---

### Hypothesis 5: Preprocessing Order Issue

**Theory:** The order of operations differs between main analysis and model development.

**Main analysis for variable subsets with derivatives:**
```python
# 1. Preprocess FULL dataset
X_full_preprocessed = SNV(SG2(X_raw))

# 2. Select variables on preprocessed data
importances = get_importances(model, X_full_preprocessed, y)
top_indices = select_top(importances, n=50)

# 3. Subset the PREPROCESSED data
X_subset_preprocessed = X_full_preprocessed[:, top_indices]

# 4. Run CV on preprocessed subset (skip_preprocessing=True)
cv_score = cross_validate(model, X_subset_preprocessed, y)
```

**Model development:**
```python
# 1. Parse wavelengths from all_vars
wavelengths_to_use = [1525.0, ..., 1574.0]  # 50 wavelengths

# 2. Subset RAW data by those wavelengths
X_subset_raw = X_raw[:, wavelength_indices]

# 3. Create pipeline with preprocessing
pipe = Pipeline([('snv', SNV()), ('sg2', SG2()), ('pls', PLS())])

# 4. Run CV with preprocessing INSIDE each fold
cv_score = cross_validate(pipe, X_subset_raw, y)
```

**Key difference:**
- Main analysis: Preprocess full → subset → CV
- Model development: Subset → preprocess → CV

**Does this matter for SG derivatives?**

Let me think:
- SG derivative uses neighboring wavelengths
- If you select wavelengths that are NOT contiguous, the derivative context changes!

**CRITICAL INSIGHT:**

If variable selection picks wavelengths like: [1525, 1530, 1535, 1550, ...]
- In FULL data: SG derivative at 1530 uses [1528, 1529, 1530, 1531, 1532] (window=5 example)
- In SUBSET data: SG derivative at 1530 uses [1525, 1530, 1535] (DIFFERENT CONTEXT!)

**This could explain the discrepancy!**

**Status:** ⚠️ STRONG CANDIDATE - needs testing

---

### Hypothesis 6: The all_vars Wavelengths Don't Match The Actual Subset Used

**Theory:** Variable selection is done in preprocessed space, but the wavelengths saved might not correspond to the same feature indices after subsetting.

**Example scenario:**
```
Original wavelengths: [1500, 1501, 1502, ..., 2500]  # 1000 wavelengths

After SNV+SG2 preprocessing:
- Still 1000 features (same shape)
- But feature i now represents derivative at wavelength i

Variable selection on preprocessed:
- Selects feature indices [10, 25, 40, 55, ...]
- These are indices in PREPROCESSED space
- Saves wavelengths[10, 25, 40, 55, ...] = [1510, 1525, 1540, 1555, ...]

Main analysis CV:
- Uses X_preprocessed[:, [10, 25, 40, 55, ...]]
- This is the DERIVATIVE values at those wavelengths
- R² = 0.9443

Model development:
- Parses wavelengths [1510, 1525, 1540, 1555, ...]
- Selects X_raw[:, [1510, 1525, 1540, 1555, ...]]
- Then applies SNV+SG2 to THIS SUBSET
- The SG2 derivative now has DIFFERENT CONTEXT (non-contiguous wavelengths!)
- R² = 0.8266
```

**Status:** ⚠️ VERY STRONG CANDIDATE

---

## The Real Problem (Hypothesis 5 + 6 Combined)

### The Bug

When using **derivative preprocessing + variable selection**:

1. Main analysis selects variables in **preprocessed (derivative) space**
2. The selected features are **non-contiguous** in wavelength space
3. Main analysis runs CV on **already-preprocessed, selected features**
4. Model development takes those **wavelength positions**, subsets **raw data**, then **recomputes derivatives**
5. But derivatives on non-contiguous wavelengths give **different results** than derivatives on contiguous wavelengths!

### Visual Example

**Full spectrum (continuous wavelengths):**
```
Wavelengths: [1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, ...]
SG derivative at 1504 uses: [1502, 1503, 1504, 1505, 1506] (window=5)
```

**Selected subset (non-continuous):**
```
Selected wavelengths: [1500, 1503, 1506, 1509, ...]
SG derivative at 1503 uses: [1500, 1503, 1506] (WRONG! Missing neighbors!)
```

The derivative calculation **requires continuous wavelength spacing** to work correctly!

---

## Implications

### Why Main Analysis R² is Correct (0.9443)

- Variables selected in derivative space
- CV runs on those selected derivative features directly
- No recomputation of derivatives
- Derivative context is preserved from full-spectrum preprocessing

### Why Model Development R² is Wrong (0.8266)

- Takes wavelength positions from main analysis
- Subsets raw data to those (non-contiguous) wavelengths
- Recomputes derivatives on non-contiguous subset
- Derivative context is BROKEN
- Model sees different features than main analysis

---

## The Fix

### Option 1: Store Preprocessed Features, Not Wavelengths (COMPLEX)

**Concept:** Save the actual preprocessed feature values, not wavelength indices.

**Pros:**
- Guarantees exact reproduction
- No context issues

**Cons:**
- Massive storage (full preprocessed data for each model)
- Changes data format significantly
- Loses interpretability (can't easily map back to wavelengths)

**Verdict:** ❌ Too invasive

---

### Option 2: Disable Repreprocessing in Model Development for Derivatives (SIMPLE)

**Concept:** When loading a model with derivative preprocessing + variable subset:
- Recognize this is incompatible with recomputation
- Show warning to user
- Either disable the feature or preprocess full data then subset

**Implementation:**
```python
# In model development
if preprocess in ['sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv'] and n_vars < total_vars:
    # Derivative + subset = can't reliably reproduce
    warning = """
    WARNING: This model uses derivative preprocessing with variable selection.
    The selected wavelengths may not be contiguous, which affects derivative calculation.

    For accurate reproduction, this would require:
    1. Preprocessing the FULL spectrum
    2. Then selecting the variables

    The R² shown here may differ from the original due to this technical limitation.
    """
    show_warning(warning)
```

**Pros:**
- Honest about limitations
- Doesn't break existing functionality
- User understands the discrepancy

**Cons:**
- Doesn't solve the problem
- Feature remains limited

**Verdict:** ⚠️ Acceptable as interim solution

---

### Option 3: Full-Spectrum Preprocessing in Model Development (RECOMMENDED)

**Concept:** When loading a derivative + subset model:
1. Preprocess the FULL spectrum
2. THEN select the variables specified in all_vars
3. Run CV on the selected preprocessed features

**Implementation:**
```python
# In _run_refined_model_thread

# Check if we have derivative preprocessing + variable subset
is_derivative = preprocess in ['sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv']
is_subset = len(selected_wl) < len(self.X_original.columns)

if is_derivative and is_subset:
    # Need to preprocess FULL spectrum first, then subset
    print("DEBUG: Derivative + subset detected. Using full-spectrum preprocessing.")

    # Build preprocessing pipeline (NO model)
    prep_steps = build_preprocessing_pipeline(preprocess_name, deriv, window, polyorder)
    prep_pipeline = Pipeline(prep_steps)

    # Preprocess FULL spectrum
    X_full_preprocessed = prep_pipeline.fit_transform(X_original.values, y_array)

    # Find indices of selected wavelengths in original data
    wavelength_indices = [list(X_original.columns).index(wl) for wl in selected_cols]

    # Subset the PREPROCESSED data
    X_work = X_full_preprocessed[:, wavelength_indices]

    # Now build pipeline with ONLY the model (skip preprocessing)
    pipe_steps = [('model', model)]
    pipe = Pipeline(pipe_steps)

    # Run CV on preprocessed subset (this matches main analysis!)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_work, y_array)):
        pipe_fold = clone(pipe)
        X_train, X_test = X_work[train_idx], X_work[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        pipe_fold.fit(X_train, y_train)
        y_pred = pipe_fold.predict(X_test)
        # ... metrics

else:
    # Raw/SNV or full-spectrum: use normal pipeline approach
    # (current code)
```

**Pros:**
- ✅ Exactly reproduces main analysis behavior
- ✅ R² should match (0.9443)
- ✅ Preserves derivative context
- ✅ Scientifically correct

**Cons:**
- More complex code
- Different code paths for different scenarios

**Verdict:** ✅ **RECOMMENDED SOLUTION**

---

## Testing Plan

### Test 1: Verify Hypothesis

Create a test script that:
1. Takes full spectrum data
2. Applies SG derivative to full spectrum
3. Selects non-contiguous wavelengths
4. Computes derivative on subset vs using full-spectrum derivative
5. Compares results

**Expected:** They should differ significantly

### Test 2: Verify Fix

After implementing Option 3:
1. Load a model with derivative + subset
2. Run refined model
3. Check R² difference
4. Should be < 0.01 (within CV randomness)

---

## Action Items

1. ✅ Revert premature "data leakage fix" (DONE)
2. ⏳ Create test to verify Hypothesis 5+6
3. ⏳ Implement Option 3 fix in model development
4. ⏳ Test with user's actual data
5. ⏳ Verify R² matches (0.9443)
6. ⏳ Document the fix

---

## Questions for User

1. **Confirm the data:** Is 0.9443 definitely the correct R² for your data with this model?
2. **Other models:** Do models with RAW or SNV preprocessing (no derivatives) show the same discrepancy?
3. **Full-spectrum models:** Do full-spectrum models (no variable selection) match between tabs?

These answers will help confirm the hypothesis.

---

## Related Issues

- Previous fix mentioned in handoff docs: "R² bug was fixed before then broke again"
- Likely this same issue was fixed, then re-introduced during refactoring
- Need to add regression tests to prevent future breaks

---

**Next Step:** Implement Option 3 and test with real data.
