# R² Reproducibility: Technical Handoff for Julia Backend Redesign

**Date**: 2025-01-18
**Purpose**: Document all model-related fixes required for Results R² and Model Development R² to match exactly
**Target Audience**: Developers implementing Julia backend
**Critical Requirement**: R² must match within ±0.001 for deterministic models (PLS, Ridge, Lasso, ElasticNet)

---

## Executive Summary

During extensive debugging of the Python implementation, we discovered **four critical issues** that prevented Results tab R² from matching Model Development tab R². These issues are **subtle but devastating** - they cause 1-3% R² discrepancies for deterministic models that should reproduce exactly.

**Any Julia reimplementation MUST preserve the exact logic we discovered, or R² will not match.**

### Success Criteria
- ✅ Derivative-only models: R² difference < 0.001
- ✅ Derivative+SNV models: R² difference < 0.001
- ✅ With wavelength restriction: R² difference < 0.001
- ✅ Without wavelength restriction: R² difference < 0.001
- ✅ With validation samples: R² difference < 0.001

### Issues Fixed (Must Preserve in Julia)
1. **SNV/Derivative Preprocessing Order** - Wavelength filtering AFTER preprocessing
2. **Validation Sample Restoration** - State restoration BEFORE validation check
3. **Training Configuration Transfer** - Metadata must flow with models
4. **Wavelength Ordering Preservation** - Never sort wavelengths

---

## Our Debugging Journey: Mistakes and Lessons Learned

This section documents **our stumbling points** during debugging - the false leads, wrong hypotheses, and hard lessons we learned. Reading this will help you avoid the same pitfalls.

### Stumbling Point #1: "It Must Be the Wavelength Compression"

**Initial Hypothesis**: We thought the wavelength specification was being compressed/parsed incorrectly, losing information in a round-trip.

**What We Tried**:
- Added diagnostics to compare original vs reconstructed wavelengths
- Tested wavelength parsing extensively
- Verified no wavelengths were lost

**Reality**: Wavelengths parsed perfectly! ✓

**Lesson**: This was a red herring. The issue wasn't wavelength loss, but wavelength **order**. We were sorting them!

**Time Lost**: ~2 hours

---

### Stumbling Point #2: "SNV Must Have a Bug"

**Initial Hypothesis**: Since derivative-only models matched but derivative+SNV didn't, SNV implementation must be buggy.

**What We Tried**:
- Traced SNV implementation line-by-line
- Verified it was stateless (no fit() operation)
- Confirmed transform() was deterministic
- Tested SNV in isolation

**Reality**: SNV implementation was perfect! ✓

**Lesson**: The bug wasn't IN SNV, it was in WHEN we applied it (before vs after wavelength filtering). SNV is a **global normalization** - it needs consistent wavelength range.

**Time Lost**: ~3 hours

---

### Stumbling Point #3: "Must Be a Random Seed Issue"

**Initial Hypothesis**: R² differences of 1-3% suggested randomness in model training or CV splits.

**What We Tried**:
- Set all random seeds explicitly
- Used deterministic CV splitting
- Tested with random_state fixed
- Verified fold indices matched

**Reality**: Random seeds were already set correctly! ✓

**Lesson**: The discrepancy was **deterministic**, not random. It happened because we were using different data (wrong wavelength range for SNV). If you see consistent R² differences (not varying between runs), it's NOT a random seed issue.

**Time Lost**: ~1 hour

---

### Stumbling Point #4: "Wavelength Ordering Can't Matter for Linear Models"

**Initial Hypothesis**: Ridge and PLS are linear models, so feature order shouldn't affect R².

**What We Tried**:
- Assumed sorting wavelengths was "clean" and "organized"
- Kept wavelength sorting for "better readability"
- Tested with different sorting orders

**Reality**: Feature order DOES matter! Even for linear models! ✗

**Why We Were Wrong**:
- Numerical precision accumulates differently with different orders
- Regularization paths depend on feature order
- Cross-validation fold assignment can be affected
- Small differences compound across CV folds

**Breakthrough**: When we removed sorting and preserved original order, R² matched exactly.

**Lesson**: **NEVER assume feature order doesn't matter**. Even tiny numerical differences can accumulate to 1-3% R² discrepancy.

**Time Lost**: ~4 hours (because we kept the sorting for too long)

---

### Stumbling Point #5: "Validation Check Should Run First"

**Initial Hypothesis**: It's logical to validate configuration before doing any work, so validation check should run before state restoration.

**What We Tried**:
- Kept validation check at line 12195 (early in function)
- Kept state restoration at line 12446 (late in function)
- Added more diagnostics to validation check

**Reality**: Validation check was CORRECT, but seeing WRONG state! ✗

**Why We Were Wrong**:
- Validation check was reading `self.validation_indices`
- But validation_indices was still empty (not restored yet)
- So check calculated wrong sample counts
- Then warned about mismatch (correctly detecting the wrong counts)

**Breakthrough**: Move restoration BEFORE validation check. Check needs to see restored state!

**Lesson**: **Order matters more than logic**. Sometimes you need to set up state before you can validate it, even if that feels "backwards".

**Time Lost**: ~5 hours (this one was subtle!)

---

### Stumbling Point #6: "The Fix Should Be in Model Dev, Not Results Tab"

**Initial Hypothesis**: Since the discrepancy shows up in Model Dev, the fix must be in Model Dev code.

**What We Tried**:
- Added diagnostics to Model Dev preprocessing
- Modified Model Dev data filtering
- Tried different PATH A implementations in Model Dev

**Reality**: The fix was needed in BOTH Results and Model Dev, but MAINLY in Results! ✗

**Why We Were Wrong**:
- Results tab was doing wavelength filtering BEFORE preprocessing
- Model Dev was trying to match, but couldn't because it didn't know the original wavelength range
- The fundamental design pattern was wrong at training time

**Breakthrough**: Fix the training workflow (Results tab) to preprocess first, then both Results and Model Dev work correctly.

**Lesson**: **Fix the root cause, not the symptom**. If Model Dev can't reproduce Results, it's usually because Results did something wrong during training.

**Time Lost**: ~6 hours

---

### Stumbling Point #7: "We Just Need to Pass More Parameters"

**Initial Hypothesis**: If we pass enough metadata (wavelength range, preprocessing config, etc.) to Model Dev, it can figure out what to do.

**What We Tried**:
- Added analysis_wl_min and analysis_wl_max parameters
- Passed them through multiple function layers
- Tried to reconstruct the original data state in Model Dev

**Reality**: Metadata helped, but wasn't enough! Need to change the ALGORITHM. ✗

**Why We Were Wrong**:
- Even with wavelength range metadata, if you filter BEFORE preprocessing, you get wrong results
- Model Dev can't "undo" the filtering that happened during training
- Need to change WHEN filtering happens, not just pass more info

**Breakthrough**: Change preprocessing order in Results tab. Then Model Dev naturally follows the same pattern.

**Lesson**: **Can't always fix design issues with more parameters**. Sometimes you need to change the algorithm itself.

**Time Lost**: ~3 hours

---

### Stumbling Point #8: "Whole Spectrum Works, So Logic Is Correct"

**Initial Hypothesis**: Since whole-spectrum models (no wavelength restriction) matched exactly, our logic must be correct.

**What We Tried**:
- Assumed restricted-spectrum failures were due to restriction parsing
- Focused debugging on wavelength parsing code
- Tested different restriction ranges

**Reality**: Whole spectrum ACCIDENTALLY worked! ✗

**Why We Were Wrong**:
- Whole spectrum worked because filtering before preprocessing doesn't matter if you don't filter
- Bug was hidden when no filtering was applied
- Bug only manifested with wavelength restriction + SNV
- This masked the fundamental design flaw

**Breakthrough**: Test with restricted wavelengths + SNV → bug appeared → found root cause.

**Lesson**: **A test passing doesn't mean the implementation is correct**. It might just mean you're not exercising the bug path. Need negative tests too!

**Time Lost**: ~2 hours (delayed finding the real issue)

---

### Stumbling Point #9: "Small R² Differences Don't Matter"

**Initial Hypothesis**: User reported R² differences of "less than 0.01" (1%) - maybe that's acceptable noise?

**What We Tried**:
- Initially dismissed small differences as numerical precision
- Focused on larger discrepancies first
- Assumed 1% was "close enough"

**Reality**: For deterministic models, even 0.001 (0.1%) is TOO MUCH! ✗

**Why We Were Wrong**:
- Deterministic models (PLS, Ridge) should reproduce exactly
- Any difference indicates a real bug, not noise
- Small differences compound and indicate systematic problems
- User was right to report it!

**Breakthrough**: Set success criteria to ±0.001, found all the bugs.

**Lesson**: **For deterministic models, perfect reproducibility is required**. Don't dismiss small differences - they're telling you something is wrong.

**Time Lost**: ~1 hour (in wrong direction)

---

### Key Patterns in Our Mistakes

Looking back, our stumbling points fell into categories:

1. **Confusing correlation with causation**: "SNV is involved, so SNV must be buggy"
2. **Assuming conventional wisdom**: "Feature order doesn't matter for linear models"
3. **Fixing symptoms, not root causes**: "Add more diagnostics to Model Dev"
4. **Dismissing small discrepancies**: "1% difference is close enough"
5. **Hidden assumptions**: "Of course we should sort wavelengths"
6. **Order-of-operations blind spots**: "Validate first, restore later"

**The biggest lesson**: When debugging R² reproducibility, **question everything**. Small differences matter. Order matters. Assumptions that "should be true" might not be.

---

## Issue #1: SNV/Derivative Preprocessing Order

### The Problem

**Original (WRONG) Implementation:**
```
1. Import full spectrum (e.g., 2151 wavelengths)
2. Apply wavelength restriction (e.g., filter to 1500-2500nm = 1500 wavelengths)
3. Apply SNV + Savitzky-Golay derivative to restricted range
4. Run variable selection on preprocessed data
```

**Why This Breaks:**
- **SNV (Standard Normal Variate)**: Global normalization that computes mean/std across **ALL** wavelengths
  - `mean([1000-2000nm])` ≠ `mean([1000-2500nm])`
  - Normalizing with different ranges gives DIFFERENT normalized values
  - Results tab trained on 1500 wavelengths, Model Dev tested on 2151 → **MISMATCH**

- **Savitzky-Golay Derivative**: Polynomial fitting requiring neighboring wavelengths
  - Edge effects differ when calculated on restricted vs full range
  - Derivative at 1500nm uses different neighbors depending on range

**Impact:**
- Derivative-only models: R² matched exactly (0.0000 difference) ✓
- Derivative+SNV with wavelength restriction: R² differed by 1-3% ✗
- Derivative+SNV on whole spectrum: R² matched exactly ✓

### The Fix

**Correct Implementation:**
```
1. Import full spectrum (e.g., 2151 wavelengths)
2. Apply SNV + Savitzky-Golay derivative to FULL SPECTRUM
3. Apply wavelength restriction to preprocessed data (for variable selection only)
4. Run variable selection on the restricted preprocessed data
```

**Critical Code Pattern:**
```python
# WRONG - Do NOT do this
X_filtered = X[:, wavelength_mask]  # Filter first
X_preprocessed = apply_snv_derivative(X_filtered)  # Preprocess after

# CORRECT - Must do this
X_preprocessed = apply_snv_derivative(X)  # Preprocess on full spectrum
X_filtered = X_preprocessed[:, wavelength_mask]  # Then filter
```

**Python Implementation (src/spectral_predict/search.py, lines 529-557):**
```python
# Apply wavelength restriction for variable selection (if specified)
# This happens AFTER preprocessing, so derivatives/SNV used full spectrum
# Create LOCAL COPIES - do NOT mutate the original arrays
if analysis_wl_min is not None or analysis_wl_max is not None:
    wavelengths_float = wavelengths.astype(float)
    wl_mask = np.ones(len(wavelengths), dtype=bool)

    if analysis_wl_min is not None:
        wl_mask &= (wavelengths_float >= analysis_wl_min)
    if analysis_wl_max is not None:
        wl_mask &= (wavelengths_float <= analysis_wl_max)

    # Create filtered COPIES for variable selection (don't mutate originals!)
    wavelengths_varsel = wavelengths[wl_mask]
    X_transformed_varsel = X_transformed[:, wl_mask]
    n_features_varsel = X_transformed_varsel.shape[1]

    print(f"Filtered to: {analysis_wl_min or 'min'} - {analysis_wl_max or 'max'} nm")
    print(f"Remaining wavelengths for variable selection: {n_features_varsel}")
    print(f"Note: Preprocessing was applied to FULL spectrum first")
else:
    # No filtering - use original arrays
    wavelengths_varsel = wavelengths
    X_transformed_varsel = X_transformed
    n_features_varsel = n_features
```

**Julia Requirements:**
1. **MUST** preprocess on full imported spectrum
2. **THEN** apply wavelength filtering for variable selection only
3. **NEVER** filter before preprocessing for SNV or derivatives
4. Create local copies when filtering (prevent array mutation in loops)

**Testing:**
- Train model with NIR restriction (1000-2500nm) + derivative+SNV
- Test in Model Dev with same config
- R² must match within 0.001

---

## Issue #2: Validation Sample Restoration

### The Problem

**Original (WRONG) Flow:**
```
1. Load model config from Results tab
2. Run validation check (lines 12228-12229)
   - Reads self.validation_indices
   - But validation_indices is empty! (not restored yet)
   - Calculates: current_validation = 0 (should be 9)
   - Calculates: current_calibration = 49 (should be 40)
   - Triggers mismatch warning!
3. Restore validation_indices from config (line 12446)
4. Test model (lines 13542-13543)
   - NOW validation samples are excluded
   - But damage done - warning already shown
```

**Why This Breaks:**
- Validation check sees empty `validation_indices` (not restored yet)
- Calculates wrong sample counts: all 49 samples instead of 40 calibration
- Model Dev tests on wrong data → R² is higher (more data) → mismatch

**Timeline of the Bug:**
- Results tab: Trains on 40 calibration samples (49 total - 9 validation)
- Model Dev: Loads model, validation_indices empty during check
- Check calculates: "Current has 49 calibration samples" (WRONG - should be 40)
- Shows warning: "trained with 40, current has 49" (CORRECT warning, based on wrong calculation)
- Then restores validation_indices
- Then excludes validation samples (now using 40) - but too late!

### The Fix

**Correct Implementation:**
```
1. Load model config from Results tab
2. Restore validation_indices BEFORE validation check (lines 12216-12226)
   - Sets self.validation_indices from config
   - Sets self.validation_enabled
3. Run validation check (lines 12228-12229)
   - Reads self.validation_indices (now correctly populated)
   - Calculates: current_validation = 9 ✓
   - Calculates: current_calibration = 40 ✓
   - No mismatch!
4. Test model (lines 13542-13543)
   - Validation samples already excluded
   - Uses same 40 samples as Results tab
   - R² matches!
```

**Python Implementation (spectral_predict_gui_optimized.py, lines 12216-12226):**
```python
# CRITICAL: Restore validation indices BEFORE validation check
# This ensures the check sees the correct validation state
if 'validation_indices' in config and config.get('validation_set_enabled'):
    self.validation_indices = set(config.get('validation_indices', []))
    self.validation_enabled.set(True)
    print(f"✓ Restored {len(self.validation_indices)} validation indices from model config")
else:
    # Clear validation if not used in original model
    self.validation_indices = set()
    self.validation_enabled.set(False)
    print("✓ No validation indices to restore (model was trained on all data)")

# CRITICAL: Also restore excluded spectra if available
if 'excluded_spectra' in config:
    self.excluded_spectra = set(config.get('excluded_spectra', []))
    if len(self.excluded_spectra) > 0:
        print(f"✓ Restored {len(self.excluded_spectra)} excluded samples from model config")

# NOW run validation check with correct state
if 'training_config' in config:
    self._validate_training_configuration(config['training_config'])
```

**Julia Requirements:**
1. **State restoration MUST happen before validation checks**
2. Restore in this exact order:
   - validation_indices
   - validation_enabled flag
   - excluded_spectra
3. Then run validation checks
4. Then filter data for model testing

**Critical Timing:**
```julia
# WRONG - validation check before restoration
validate_config(config)  # Uses empty validation_indices
restore_state!(config)   # Too late!

# CORRECT - restoration before validation check
restore_state!(config)   # Sets validation_indices
validate_config(config)  # Uses restored validation_indices
```

---

## Issue #3: Training Configuration Transfer

### The Problem

**Original (WRONG) Flow:**
```
1. Results tab runs analysis
   - Passes excluded_count and validation_count to run_search()
   - run_search() uses these for logging
   - Returns results_df
   - excluded_count and validation_count are NOT stored in results_df
2. User double-clicks result to transfer to Model Dev
   - model_config = results_df.loc[row_idx].to_dict()
   - model_config has: Model, Params, Preprocess, R2, etc.
   - model_config does NOT have: training_config
3. Model Dev loads model
   - Checks if 'training_config' in config (line 12228)
   - Not found! Skips validation check
   - No way to verify data configuration matches
```

**Why This Breaks:**
- Can't verify that Model Dev is using same data as Results tab
- No way to detect if user changed excluded/validation samples
- False sense of confidence when R² doesn't match

### The Fix

**Correct Implementation:**
```
1. Results tab runs analysis
   - After run_search() completes
   - Store training_config with all metadata (lines 11143-11156)
   - Cache: X_filtered, y_filtered, task_type, label_encoder, etc.
2. User double-clicks result to transfer to Model Dev
   - Get row from results_df
   - Attach training_config from cache (lines 11187-11207)
   - Attach validation_indices and excluded_spectra
   - model_config now has complete configuration
3. Model Dev loads model
   - training_config is present
   - Runs validation check
   - Detects any configuration mismatches
   - Warns user if data state changed
```

**Python Implementation (spectral_predict_gui_optimized.py):**

**Cache creation (lines 11143-11156):**
```python
# === CACHE TRAINING DATA FOR MANUAL ENSEMBLE RETRAINING ===
# Store filtered data and configuration so user can retrain ensembles
# with different model selections after initial analysis
self.training_data_cache = {
    'X_filtered': X_filtered.copy(),  # Deep copy to prevent modifications
    'y_filtered': y_filtered.copy(),
    'task_type': task_type,
    'folds': self.folds.get(),
    'label_encoder': label_encoder,
    'analysis_wl_min': analysis_wl_min_value,
    'analysis_wl_max': analysis_wl_max_value,
    'timestamp': datetime.now().isoformat()
}
self._log_progress(f"\n✓ Cached training data for ensemble retraining")
```

**Config attachment (lines 11187-11207):**
```python
# Attach training configuration if available (for validation checking)
# This ensures Model Dev can verify it's using the same data configuration as Results tab
if hasattr(self, 'last_training_config') and self.last_training_config is not None:
    model_config['training_config'] = self.last_training_config
    print(f"✓ Attached training configuration to model transfer")
    print(f"  Expected calibration samples: {self.last_training_config['n_samples_used']}")

    # CRITICAL: Also attach validation configuration
    # Without this, validation_indices gets cleared and causes false mismatch warnings
    if self.validation_enabled.get() and self.validation_indices:
        model_config['validation_indices'] = list(self.validation_indices)
        model_config['validation_set_enabled'] = True
        print(f"  Validation samples: {len(self.validation_indices)}")
    else:
        model_config['validation_set_enabled'] = False
        print(f"  Validation: not used")

    # Also attach excluded spectra info for consistency
    if self.excluded_spectra:
        model_config['excluded_spectra'] = list(self.excluded_spectra)
        print(f"  Excluded samples: {len(self.excluded_spectra)}")
```

**Julia Requirements:**
1. **Cache training configuration** after analysis completes:
   - Filtered data (X_filtered, y_filtered)
   - Task type (regression/classification)
   - CV folds
   - Label encoder (for classification)
   - Sample counts (total, excluded, validation, calibration)
   - Wavelength restriction info
   - Timestamp

2. **Transfer configuration** when moving model to Model Dev:
   - Attach training_config
   - Attach validation_indices
   - Attach excluded_spectra
   - Attach all metadata needed for validation

3. **Validate configuration** when loading model:
   - Check sample counts match
   - Check CV folds match
   - Check data state hasn't changed
   - Warn user if mismatch detected

---

## Issue #4: Wavelength Ordering Preservation

### The Problem

**Original (WRONG) Implementation:**
```python
# Parse wavelength specification string
selected = parse_wavelengths(spec_string)
# WRONG: Sort the wavelengths
selected = sorted(list(set(selected)))
```

**Why This Breaks:**
- Wavelengths were sorted after selection
- Changed feature order in the data matrix
- Even linear models (Ridge, PLS) can be affected by feature order due to:
  - Numerical precision differences
  - Regularization path order
  - Cross-validation split ordering
- Result: R² differed by 1-3% even for simple models

**Example:**
```
Original order: [1500.0, 1200.0, 1800.0, 1000.0]
After sorting:  [1000.0, 1200.0, 1500.0, 1800.0]

Training: Used original order [1500, 1200, 1800, 1000]
Testing:  Used sorted order [1000, 1200, 1500, 1800]
Result:   Features misaligned → R² doesn't match!
```

### The Fix

**Correct Implementation (spectral_predict_gui_optimized.py, lines 14347-14352):**
```python
# Remove duplicates while preserving order from available_wavelengths
# DO NOT sort - sorting changes feature order and breaks R² reproducibility!
selected_set = set(selected)
selected = [wl for wl in available_wavelengths if wl in selected_set]
```

**Julia Requirements:**
1. **NEVER sort wavelengths** after selection
2. Preserve original order from DataFrame columns
3. Use list comprehension pattern to remove duplicates while preserving order:
   ```julia
   # Create set for O(1) lookup
   selected_set = Set(selected)
   # Filter in original order
   selected_ordered = [wl for wl in available_wavelengths if wl in selected_set]
   ```

4. When parsing wavelength ranges, maintain order from source data

**Testing:**
- Train model with specific wavelength selection
- Test in Model Dev
- Verify wavelengths appear in same order
- R² must match within 0.001

---

## Critical Design Patterns

### PATH A vs PATH B Preprocessing

The codebase uses two different preprocessing paths depending on whether derivatives are used:

**PATH A: Derivative + Subset** (lines 502-524 in search.py)
```python
if preprocess_cfg["deriv"] is not None:
    # Apply derivative preprocessing ONCE to full spectrum
    X_transformed = apply_derivative(X_np, wavelengths, deriv, window, polyorder)

    # Variable selection on preprocessed data
    selected_indices = variable_selection(X_transformed, y_np)

    # Subset to selected features
    X_subset = X_transformed[:, selected_indices]
    wavelengths_subset = wavelengths[selected_indices]

    # Train model (no preprocessing in pipeline)
    model.fit(X_subset, y_np)
```

**PATH B: Subset + Other Preprocessing** (non-derivative methods)
```python
else:
    # Variable selection on raw/SNV data
    selected_indices = variable_selection(X_np, y_np)

    # Subset FIRST
    X_subset = X_np[:, selected_indices]
    wavelengths_subset = wavelengths[selected_indices]

    # Build pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocess', get_preprocessor(preprocess_cfg)),
        ('model', model)
    ])

    # Train with preprocessing in pipeline
    pipeline.fit(X_subset, y_np)
```

**Why Two Paths?**
- Derivatives need full spectral context (neighboring wavelengths)
- Other methods (raw, SNV) can work on subsets
- PATH A: Preprocess → Select → Train
- PATH B: Select → Preprocess → Train

**Julia Requirements:**
- Implement both paths exactly as described
- Derivatives ALWAYS use PATH A
- Other methods ALWAYS use PATH B
- Never mix the two patterns

### Array Mutation Prevention

**Critical Pattern in Nested Loops:**
```python
# WRONG - Mutates shared array
for model_name in models:
    for preprocess in preprocessing_methods:
        X_filtered = X[:, wavelength_mask]  # Same array reference!
        # Next iteration corrupts data
```

**CORRECT - Local copies:**
```python
# Create local copies for each iteration
for model_name in models:
    for preprocess in preprocessing_methods:
        X_local = X[:, wavelength_mask].copy()  # New array
        # Safe - no corruption
```

**Julia Requirements:**
- Always use local copies in nested loops
- Never mutate shared arrays
- Use `copy()` or create new arrays
- Prevents data corruption across iterations

---

## Testing Requirements

### Test Suite for R² Reproducibility

**Test 1: Derivative-Only Models**
```
1. Train Ridge model with sg1 (1st derivative)
2. Record Results tab R²
3. Transfer to Model Dev
4. Calculate Model Dev R²
5. Assert: |Results R² - Model Dev R²| < 0.001
```

**Test 2: Derivative+SNV Models**
```
1. Train PLS model with deriv_snv (derivative + SNV)
2. Record Results tab R²
3. Transfer to Model Dev
4. Calculate Model Dev R²
5. Assert: |Results R² - Model Dev R²| < 0.001
```

**Test 3: Wavelength Restriction**
```
1. Set analysis wavelength range (e.g., 1000-2500nm)
2. Train Ridge model with deriv_snv
3. Record Results tab R²
4. Transfer to Model Dev
5. Calculate Model Dev R²
6. Assert: |Results R² - Model Dev R²| < 0.001
```

**Test 4: Validation Samples**
```
1. Mark 9 samples as validation set
2. Train PLS model on remaining 40 samples
3. Record Results tab R² (on 40 samples)
4. Transfer to Model Dev
5. Verify Model Dev uses 40 samples (not 49)
6. Calculate Model Dev R²
7. Assert: |Results R² - Model Dev R²| < 0.001
```

**Test 5: Excluded Samples**
```
1. Exclude 5 samples
2. Train Ridge model
3. Record Results tab R²
4. Transfer to Model Dev
5. Verify Model Dev excludes same 5 samples
6. Calculate Model Dev R²
7. Assert: |Results R² - Model Dev R²| < 0.001
```

### Validation Checklist

Before deploying Julia backend:
- [ ] All 5 tests pass with R² < 0.001 difference
- [ ] Preprocessing order correct (full spectrum → preprocess → filter)
- [ ] State restoration order correct (restore → validate → test)
- [ ] Training config transfers with all metadata
- [ ] Wavelength order preserved (never sorted)
- [ ] PATH A used for derivatives
- [ ] PATH B used for other preprocessing
- [ ] Array mutation prevented in loops
- [ ] Deep copies used where needed
- [ ] Edge cases tested (whole spectrum, restricted, validation, excluded)

---

## Code References

### Key Commits

**Commit: 8a48ec2** - "fix: Improve R² reproducibility and preprocessing consistency"
- Fixed SNV/derivative preprocessing order
- Added post-preprocessing wavelength filtering
- Prevented array mutation in loops

**Commit: 277333e** - "feat: Add R² consistency validation and transfer models integration"
- Added training configuration transfer
- Enhanced validation checks

**Commit: af7da75** - "fix: Eliminate false configuration mismatch warnings in Model Development"
- Fixed validation sample restoration order
- Added excluded spectra restoration
- Fixed training config attachment

**Commit: a033439** - "feat: Add Train Ensemble button for custom model selection"
- Extracted reusable methods
- Added training data cache
- Enabled manual ensemble retraining

### Key Files and Line Numbers

**spectral_predict_gui_optimized.py:**
- Lines 10244-10304: Wavelength restriction parsing (DO NOT FILTER)
- Lines 11143-11156: Training data cache storage
- Lines 11187-11207: Training config attachment for model transfer
- Lines 12216-12252: Validation/excluded indices restoration (BEFORE check)
- Lines 13537-13558: Model Dev data exclusion (validation + excluded)
- Lines 14347-14352: Wavelength ordering preservation

**src/spectral_predict/search.py:**
- Lines 529-557: Post-preprocessing wavelength filtering
- Lines 502-524: PATH A preprocessing (derivative + subset)
- Lines 565-608: Variable selection with filtered data
- Lines 648-689: Model training with PATH A vs PATH B

**src/spectral_predict/preprocess.py:**
- Lines 8-40: SNV implementation (stateless, deterministic)
- Lines 145-147: deriv_snv pipeline (derivative → SNV order)

---

## Critical Implementation Notes for Julia

### Memory Management
- **Deep copies required** for cached training data
- Use `copy()` or equivalent for X_filtered, y_filtered
- Prevents mutations affecting cached data

### Data Structures
- **validation_indices**: Set of integers (sample indices)
- **excluded_spectra**: Set of integers (sample indices)
- **training_config**: Dictionary/struct with all training metadata

### State Management
- **Restoration order is critical**: Restore state → Validate → Test
- **Never restore after validation check**
- Clear cache on new analysis (prevent stale data)

### Threading/Concurrency
- If Julia backend uses threads, protect training_data_cache with locks
- Ensure state restoration is atomic
- Use thread-safe data structures

### Numerical Precision
- Use same floating point types as Python (Float64)
- Ensure SNV calculations match exactly
- Test edge cases (very small std dev, etc.)

---

## Conclusion

These fixes represent **months of debugging** to identify subtle but critical issues. The logic described in this document **MUST be preserved exactly** in any Julia reimplementation.

**The most important lessons:**
1. **Preprocessing order matters** - Always preprocess on full spectrum first
2. **State restoration order matters** - Always restore before validation checks
3. **Metadata must flow** - Training configuration must transfer with models
4. **Feature order matters** - Never sort wavelengths after selection
5. **Test ruthlessly** - R² must match within 0.001 for deterministic models

Any deviation from these patterns will break R² reproducibility.

**For questions or clarifications**, refer to:
- This handoff document
- Commit messages in git history
- Code comments in key sections
- Test suite validation results

Good luck with the Julia implementation! 🚀
