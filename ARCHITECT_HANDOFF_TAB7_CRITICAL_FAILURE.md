# ARCHITECT HANDOFF: Tab 7 Model Development - Critical Failure

**Date:** 2025-11-07
**Priority:** 🔴 CRITICAL - PRODUCTION BLOCKING
**Status:** BROKEN - Requires Master Architect + Team Review
**Assigned To:** Master Software Architect + Best Coding Team

---

## EXECUTIVE SUMMARY

Tab 7 (Model Development) has a **CRITICAL FAILURE** that makes it unusable:

- **Expected Behavior:** Load model from Results tab (R²=0.97) → Run in Tab 7 → Get same R²=0.97
- **Actual Behavior:** Load model from Results tab (R²=0.97) → Run in Tab 7 → Get R²=-0.07 (COMPLETELY WRONG)
- **Impact:** Users cannot reproduce results from Results tab, making the entire Model Development feature worthless
- **Root Cause:** Unknown - likely configuration loading/storage issue between Results tab and Tab 7

---

## THE PROBLEM IN DETAIL

### What the User Reported

1. User runs analysis in Tab 3 (Analysis Configuration)
2. Analysis completes → Results tab shows **Lasso model with R² = 0.97** (excellent performance)
3. User double-clicks result → Loads into Tab 7 (Model Development)
4. User clicks "Run Model" in Tab 7
5. **Tab 7 produces R² = -0.07** (catastrophic failure - worse than predicting the mean!)

### Expected vs Actual

| Metric | Results Tab (Correct) | Tab 7 (Broken) | Discrepancy |
|--------|----------------------|----------------|-------------|
| R² | 0.97 | -0.07 | **-104 percentage points!** |
| Model | Lasso | Lasso | Same |
| Alpha | ? (optimized) | 0.01 | **Unknown if correct** |
| Wavelengths | 50 selected | 50 selected | **Unknown if same 50** |
| Preprocessing | snv_sg2 | snv_sg2 | Same |
| Window | 17 | 17 | Same |
| CV Folds | 5 | 5 | Same |

### Console Output (Latest Test)

```
TAB 7: MODEL DEVELOPMENT - EXECUTION START
[STEP 1] Parsing parameters...
  Model Type: Lasso
  Task Type: regression
  Preprocessing: snv_sg2
  Window Size: 17
  CV Folds: 5
  Model-specific hyperparameters for Lasso: {'alpha': 0.01}
  ✓ No cross-contamination from other model types

[STEP 3] Building preprocessing pipeline...
  PATH A: Derivative + Subset detected
  Will preprocess FULL spectrum first, then subset
  Preprocessing full spectrum (2151 wavelengths)...
  Subsetted to 50 wavelengths after preprocessing

[STEP 4] Running 5-fold cross-validation...
  Using KFold (shuffle=False) for deterministic splits
  Fold 1/5: R²=-0.0252, RMSE=7.9009, MAE=7.2000
  Fold 2/5: R²=-0.1677, RMSE=7.1495, MAE=5.7859
  Fold 3/5: R²=-0.0983, RMSE=6.1867, MAE=5.4405
  Fold 4/5: R²=-0.0729, RMSE=6.5219, MAE=5.8723
  Fold 5/5: R²=-0.0042, RMSE=6.6842, MAE=5.8503

[STEP 6] Computing performance metrics...
  R²:   -0.0737 ± 0.0577  ❌ COMPLETELY WRONG (should be 0.97)
```

---

## WHAT WE'VE TRIED (AND FAILED)

### Fix #1: Removed Hyperparameter Cross-Contamination ✅ (Partial Success)

**File:** `spectral_predict_gui_optimized.py`, lines 2158-2182

**Problem:** All model types were receiving hyperparameters from ALL other models (Lasso was getting `n_components`, `n_estimators`, etc.)

**Fix:** Changed to model-specific defaults only

**Result:**
- ✅ Hyperparameters are now clean: `{'alpha': 0.01}` (no contamination)
- ❌ R² is STILL completely wrong (-0.07 instead of 0.97)

### Fix #2: Model-Specific get_model() Calls ✅ (Partial Success)

**File:** `spectral_predict_gui_optimized.py`, lines 2291-2304, 2344-2358

**Problem:** All models were receiving `n_components` parameter (even non-PLS models)

**Fix:** Only PLS models receive `n_components`

**Result:**
- ✅ Models initialized correctly with proper parameters
- ❌ R² is STILL completely wrong (-0.07 instead of 0.97)

### What This Tells Us

The hyperparameter extraction/application code is now CORRECT, but the **underlying configuration data is WRONG**.

The problem is likely:
1. **Wrong wavelengths loaded from Results tab** - We're using 50 wavelengths, but not the CORRECT 50
2. **Wrong alpha value** - alpha=0.01 might not be the optimized value from Results tab
3. **Configuration storage/retrieval bug** - Results tab might not be storing the correct information
4. **Wavelength subsetting bug** - The 50 wavelengths might be getting lost or corrupted during loading

---

## CRITICAL QUESTIONS THAT NEED ANSWERS

### 1. What is ACTUALLY stored in Results tab?

**File to investigate:** `spectral_predict_gui_optimized.py` (Results tab storage)

**Questions:**
- When analysis completes, what exactly gets stored in `self.results_list`?
- Are the optimized hyperparameters (e.g., best alpha for Lasso) stored?
- Are the exact wavelength indices/values stored?
- Is there a `model_config` dict? What's in it?

**Action needed:** Print/log the FULL `model_config` dict when saving to Results tab

### 2. What is ACTUALLY loaded into Tab 7?

**File to investigate:** `spectral_predict_gui_optimized.py`, lines 3047-3316 (`_load_model_to_NEW_tab7()`)

**Questions:**
- What does the `config` parameter contain when called?
- Are wavelengths being extracted correctly from the config?
- Is the alpha value being extracted correctly from the config?
- Are there field name mismatches (e.g., `Alpha` vs `alpha`, `all_vars` vs `wavelengths`)?

**Action needed:** Add comprehensive debug logging to `_load_model_to_NEW_tab7()` to print:
- Full config dict structure
- Extracted wavelengths (exact values)
- Extracted hyperparameters (all of them)
- Field names used for extraction

### 3. How does Tab 6 (Custom Model Development) do it CORRECTLY?

**File to investigate:** `spectral_predict_gui_optimized.py`, lines 6000-6600 (Tab 6 code)

**Known facts:**
- Tab 6 works correctly (produces matching R² values)
- Tab 6 uses similar loading/execution logic
- Tab 7 was modeled after Tab 6 but is broken

**Questions:**
- How does Tab 6 extract wavelengths from config?
- How does Tab 6 extract hyperparameters from config?
- How does Tab 6 apply wavelength subsetting?
- What is DIFFERENT between Tab 6's approach and Tab 7's approach?

**Action needed:** Line-by-line comparison of Tab 6 vs Tab 7:
- Config loading methods
- Wavelength extraction logic
- Hyperparameter extraction logic
- Preprocessing pipeline construction
- CV execution

### 4. Are the wavelengths being loaded correctly?

**Current code:** Lines 3186-3227 in `_load_model_to_NEW_tab7()`

**Issue:** Code extracts wavelengths from `config.get('all_vars')` and parses string representation

**Questions:**
- Is `all_vars` actually present in the config?
- Is it in the correct format (string of wavelength values)?
- Are the wavelengths being parsed correctly?
- When we "subset to 50 wavelengths", are we using the SAME 50 as the original analysis?

**Action needed:** Add debug logging to print:
- Raw `all_vars` field from config
- Parsed wavelength values (first 10 and last 10)
- Wavelength indices used for subsetting (first 10 and last 10)
- Comparison with original wavelengths from Tab 3 analysis

### 5. Is the alpha value correct?

**Current behavior:** Console shows `alpha: 0.01`

**Questions:**
- Is 0.01 the ACTUAL optimized alpha from the Results tab?
- Or is it a default value?
- Where is the optimized alpha stored in the Results tab?
- Is there a field name mismatch (e.g., `Alpha` vs `alpha`, `best_alpha` vs `alpha`)?

**Action needed:** Trace back to Tab 3 analysis execution:
- Find where Lasso hyperparameters are optimized
- Find where optimized alpha is stored in results
- Verify field name used for storage
- Verify Tab 7 is using the SAME field name for extraction

---

## ARCHITECTURAL ISSUES

### Issue #1: Lack of Configuration Validation

**Problem:** No validation that loaded config matches expected structure

**Impact:** Silent failures when field names don't match or data is missing

**Recommendation:** Implement strict config validation with FAIL LOUD errors:
```python
def validate_model_config(config):
    required_fields = ['Model', 'Preprocessing', 'Window', 'Task']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"CRITICAL: Config missing required field '{field}'")

    # Validate wavelengths for subset models
    if config.get('subset') and 'all_vars' not in config:
        raise ValueError(f"CRITICAL: Subset model missing 'all_vars' field")

    # Validate hyperparameters
    model_name = config['Model']
    if model_name in ['Ridge', 'Lasso'] and 'Alpha' not in config and 'alpha' not in config:
        raise ValueError(f"CRITICAL: {model_name} model missing alpha parameter")
```

### Issue #2: No Round-Trip Testing

**Problem:** No test that verifies Results tab → Tab 7 → same R²

**Impact:** Regressions go undetected until users report them

**Recommendation:** Create integration test:
```python
def test_results_to_tab7_roundtrip():
    # 1. Run analysis in Tab 3
    # 2. Get R² from Results tab
    # 3. Load into Tab 7
    # 4. Run in Tab 7
    # 5. Assert Tab 7 R² == Results tab R² (within tolerance)
```

### Issue #3: Tab 6 vs Tab 7 Code Duplication

**Problem:** Similar functionality implemented twice with different bugs

**Impact:** Fixes to Tab 6 don't automatically apply to Tab 7

**Recommendation:** Refactor shared logic into common methods:
```python
class SpectralPredictGUI:
    def _load_model_config(self, config):
        """Common method for loading model config (used by Tab 6 AND Tab 7)"""
        pass

    def _extract_wavelengths(self, config):
        """Common method for extracting wavelengths (used by Tab 6 AND Tab 7)"""
        pass

    def _extract_hyperparameters(self, config, model_name):
        """Common method for extracting hyperparameters (used by Tab 6 AND Tab 7)"""
        pass
```

### Issue #4: Insufficient Logging

**Problem:** No visibility into what's happening during config loading/execution

**Impact:** Impossible to debug without adding print statements

**Recommendation:** Add structured logging throughout:
```python
import logging
logger = logging.getLogger('SpectralPredict.Tab7')

logger.info(f"Loading model config: {config.keys()}")
logger.debug(f"Extracted wavelengths: {wavelengths[:10]}...{wavelengths[-10:]}")
logger.debug(f"Extracted hyperparameters: {hyperparams}")
logger.warning(f"Field 'all_vars' not found in config, falling back to full spectrum")
```

---

## RECOMMENDED INVESTIGATION PLAN

### Phase 1: Diagnosis (Est: 2-3 hours)

**Goal:** Understand EXACTLY what's being stored and loaded

**Tasks:**
1. Add debug logging to Results tab storage (when analysis completes)
   - Print full config dict being stored
   - Print wavelengths being stored
   - Print hyperparameters being stored

2. Add debug logging to Tab 7 loading (`_load_model_to_NEW_tab7()`)
   - Print full config dict received
   - Print extracted wavelengths (compare with stored)
   - Print extracted hyperparameters (compare with stored)

3. Run test with example data:
   - BoneCollagen.csv
   - Lasso model, snv_sg2, window=17
   - Capture all debug output
   - Compare stored vs loaded values

4. Compare with Tab 6 execution:
   - Load same model into Tab 6
   - Capture debug output
   - Verify Tab 6 produces correct R²
   - Compare Tab 6 vs Tab 7 loaded values

**Deliverable:** Diagnosis report identifying exact discrepancy

### Phase 2: Root Cause Fix (Est: 2-4 hours)

**Goal:** Fix the configuration loading/storage bug

**Approach depends on Phase 1 findings:**

**Scenario A:** Wavelengths not stored/loaded correctly
- Fix Results tab to store wavelengths in correct format
- Fix Tab 7 to load wavelengths from correct field
- Add validation

**Scenario B:** Alpha value not stored/loaded correctly
- Fix Results tab to store optimized alpha
- Fix Tab 7 to load alpha from correct field
- Add validation

**Scenario C:** Preprocessing mismatch
- Verify preprocessing path (derivative+subset vs standard)
- Ensure Tab 7 uses same path as original analysis
- Add validation

**Scenario D:** Field name mismatches
- Standardize field names across codebase
- Add mapping for legacy field names
- Add validation

**Deliverable:** Working fix with test validation

### Phase 3: Validation (Est: 1-2 hours)

**Goal:** Verify fix works for all model types

**Tasks:**
1. Test with all 6 model types:
   - PLS
   - Ridge
   - Lasso
   - RandomForest
   - MLP
   - NeuralBoosted

2. Test with different preprocessing methods:
   - raw
   - snv
   - snv_sg2 (derivative+subset)
   - msc

3. Test with different wavelength selections:
   - Full spectrum
   - Subset (top 10, top 20, top 50)
   - Manual wavelength specification

4. Verify R² matches Results tab for all tests (tolerance: ±0.001)

**Deliverable:** Test report with all scenarios passing

### Phase 4: Refactoring (Est: 3-4 hours)

**Goal:** Prevent this from happening again

**Tasks:**
1. Extract common config loading logic
2. Add strict validation
3. Add comprehensive logging
4. Create integration tests
5. Update documentation

**Deliverable:** Refactored, tested, documented code

---

## CURRENT STATE OF CODEBASE

### Files Modified Recently

1. **`spectral_predict_gui_optimized.py`**
   - Tab 7 integration (2,100+ lines added)
   - Hyperparameter contamination fixes (lines 2158-2182, 2291-2304, 2344-2358)
   - Status: Launches successfully, but produces wrong results

2. **`test_tab7_hyperparams.py`** (NEW)
   - Automated test for hyperparameter extraction
   - Status: PASSES (hyperparameters are clean)

3. **`TAB7_CRITICAL_BUG_FIX.md`** (NEW)
   - Documentation of hyperparameter fixes
   - Status: Incomplete (fixes didn't solve the problem)

### Known Working Code

- **Tab 6 (Custom Model Development):** Works correctly, produces matching R² values
- **Tab 3 (Analysis Configuration):** Works correctly, produces correct R² in Results tab
- **Tab 5 (Results):** Works correctly, stores and displays results

### Known Broken Code

- **Tab 7 (Model Development):** Broken, produces wrong R² values
  - Hyperparameter extraction: ✅ FIXED
  - Configuration loading: ❌ BROKEN (suspected)
  - Wavelength subsetting: ❌ BROKEN (suspected)
  - Alpha extraction: ❌ BROKEN (suspected)

---

## USER REQUIREMENTS

### Immediate Requirement

When user double-clicks a result in Results tab:
1. Model should load into Tab 7 automatically ✅ (works)
2. Model should RUN automatically (not just load) ❌ (NOT IMPLEMENTED)
3. Results should match Results tab EXACTLY ❌ (COMPLETELY BROKEN)
4. User can then tweak hyperparameters and re-run

### Current Gap

- Models load but don't run automatically
- When user manually runs model, R² is completely wrong
- No way to verify configuration was loaded correctly

### Proposed Solution

```python
def _load_model_to_NEW_tab7(self, config):
    """Load model configuration from Results tab into NEW Tab 7."""

    # 1. Validate config
    self._validate_model_config(config)

    # 2. Load configuration (with debug logging)
    self._populate_tab7_from_config(config)

    # 3. Verify loaded config matches original (CRITICAL!)
    if not self._verify_loaded_config(config):
        messagebox.showerror("Config Load Error",
            "Loaded configuration doesn't match original.\n\n"
            "This is a bug - please report with debug logs.")
        return

    # 4. Auto-run model to verify R² matches
    self.root.after(500, self._tab7_run_model)  # Auto-run after 500ms
```

---

## TEST DATA

**File:** `C:\Users\sponheim\git\dasp\example\BoneCollagen.csv`

**Specs:**
- 49 samples
- 2,151 wavelengths
- Regression task

**Known Good Result:**
- Model: Lasso
- Preprocessing: snv_sg2
- Window: 17
- Wavelengths: 50 selected
- R²: **0.97** (from Results tab)

**Current Broken Result:**
- Same configuration
- R²: **-0.07** (from Tab 7)

**Use this for all testing and validation.**

---

## CRITICAL FILES TO INVESTIGATE

### 1. Results Tab Storage

**Location:** Search for where `self.results_list` is populated

**Look for:**
- Dictionary creation with model configuration
- Field names used for storage
- Wavelength storage format
- Hyperparameter storage format

**Key questions:**
- Is `all_vars` being stored?
- Is optimized alpha being stored (and with what field name)?
- Is there a `model_config` or similar dict?

### 2. Tab 7 Configuration Loading

**File:** `spectral_predict_gui_optimized.py`
**Method:** `_load_model_to_NEW_tab7()` (lines 3047-3316)

**Focus on:**
- Lines 3186-3227: Wavelength loading
- Lines 3243-3296: Hyperparameter loading
- Field name extraction (`.get('field_name')` calls)

**Key questions:**
- Are field names matching what Results tab stores?
- Is wavelength parsing working correctly?
- Is alpha extraction working correctly?

### 3. Tab 6 (Working Reference)

**File:** `spectral_predict_gui_optimized.py`
**Method:** `_load_model_to_tab7()` (old Tab 6 method, lines 5287-5750 approx)

**Focus on:**
- How it extracts wavelengths
- How it extracts hyperparameters
- How it builds preprocessing pipeline
- Why it WORKS when Tab 7 doesn't

### 4. Tab 3 Analysis Execution

**File:** `spectral_predict_gui_optimized.py`
**Method:** Search for where analysis results are saved to `self.results_list`

**Focus on:**
- What gets stored after successful analysis
- How wavelengths are determined/stored
- How optimized hyperparameters are stored
- Field naming conventions

---

## DEBUGGING COMMANDS

### 1. Add Debug Logging to Results Tab Storage

```python
# Find where self.results_list is appended to
# Add before append:
print("="*80)
print("DEBUG: Storing result to Results tab")
print(f"Model config keys: {model_config.keys()}")
print(f"Model: {model_config.get('Model')}")
print(f"Alpha field names: Alpha={model_config.get('Alpha')}, alpha={model_config.get('alpha')}")
print(f"Wavelengths field names: all_vars={model_config.get('all_vars')}, wavelengths={model_config.get('wavelengths')}")
if 'all_vars' in model_config:
    print(f"all_vars content (first 200 chars): {str(model_config['all_vars'])[:200]}")
print("="*80)
```

### 2. Add Debug Logging to Tab 7 Loading

```python
# In _load_model_to_NEW_tab7(), add at start:
print("="*80)
print("DEBUG: Loading config into Tab 7")
print(f"Config keys received: {config.keys()}")
print(f"Model: {config.get('Model')}")
print(f"Alpha values: Alpha={config.get('Alpha')}, alpha={config.get('alpha')}")
print(f"Wavelength fields: all_vars={config.get('all_vars')}, wavelengths={config.get('wavelengths')}")
if 'all_vars' in config:
    print(f"all_vars content: {config['all_vars']}")
print("="*80)
```

### 3. Compare Tab 6 vs Tab 7

```python
# Load same model into Tab 6 and Tab 7
# Add logging to both
# Compare outputs side-by-side
```

---

## SUCCESS CRITERIA

### Minimum Viable Fix

1. ✅ Load model from Results tab (R²=0.97)
2. ✅ Auto-run in Tab 7
3. ✅ Tab 7 produces R²=0.97 (±0.001 tolerance)
4. ✅ Verified for all 6 model types
5. ✅ Integration test added to prevent regression

### Full Solution

1. ✅ Minimum Viable Fix (above)
2. ✅ Refactored common code between Tab 6 and Tab 7
3. ✅ Strict configuration validation
4. ✅ Comprehensive logging
5. ✅ Full test suite (unit + integration)
6. ✅ Updated documentation

---

## COMMIT HISTORY

This handoff is part of commit with:
- Hyperparameter contamination fixes (partial success)
- Test scripts
- Documentation
- This handoff document

**Previous relevant commits:**
- `220b3d7` - fix(critical): Model Development not loading Window/alpha parameters
- `371b425` - fix(critical): Preprocessing iteration bug
- `ab0d138` - fix(critical): Resolve GUI crash + PLS R² errors

---

## CONTACT / ESCALATION

**Original Issue Reporter:** User (via chat)

**Current Status:**
- Hyperparameter contamination fixed ✅
- Core issue (wrong R² values) remains ❌
- **BLOCKING PRODUCTION USE**

**Priority:** 🔴 CRITICAL

**Timeline:** Needs immediate attention from senior architect + team

---

## APPENDIX A: Full Console Output (Latest Test)

```
TAB 7: MODEL DEVELOPMENT - EXECUTION START
================================================================================

[STEP 1] Parsing parameters...
  Model Type: Lasso
  Task Type: regression
  Preprocessing: snv_sg2
  Window Size: 17
  CV Folds: 5
  Max Iterations: 100
  Wavelengths: 50 selected (1524.0 to 2299.0 nm)
  Preprocessing Config: snv_deriv (deriv=2, polyorder=3)
  Model-specific hyperparameters for Lasso: {'alpha': 0.01}
  ✓ No cross-contamination from other model types

[STEP 2] Filtering data...
  Initial shape: X=(49, 2151), y=(49,)
  Reset index after exclusions
  Final shape: X=(49, 2151), y=(49,)
  This ensures CV folds match Results tab (sequential row indexing)

[STEP 3] Building preprocessing pipeline...
  PATH A: Derivative + Subset detected
  Will preprocess FULL spectrum first, then subset
  Preprocessing full spectrum (2151 wavelengths)...
  Subsetted to 50 wavelengths after preprocessing
  Applied hyperparameters: {'alpha': 0.01}
  Pipeline: ['model'] (preprocessing already applied)

[STEP 4] Running 5-fold cross-validation...
  Using KFold (shuffle=False) for deterministic splits
  Fold 1/5: R²=-0.0252, RMSE=7.9009, MAE=7.2000
  Fold 2/5: R²=-0.1677, RMSE=7.1495, MAE=5.7859
  Fold 3/5: R²=-0.0983, RMSE=6.1867, MAE=5.4405
  Fold 4/5: R²=-0.0729, RMSE=6.5219, MAE=5.8723
  Fold 5/5: R²=-0.0042, RMSE=6.6842, MAE=5.8503

[STEP 5] Training final model on full dataset...
  Using full-spectrum preprocessor (already fitted)
  Model trained on 49 samples × 50 features

[STEP 6] Computing performance metrics...
  RMSE: 6.8886 ± 0.5936
  R²:   -0.0737 ± 0.0577
  MAE:  6.0298 ± 0.6055
  Bias: 0.0066 ± 1.7539

[STEP 7] Updating GUI...

================================================================================
TAB 7: MODEL DEVELOPMENT - EXECUTION COMPLETE
================================================================================
```

**KEY OBSERVATION:** Hyperparameters are clean, but R² is still wrong (-0.07 vs 0.97)

---

## APPENDIX B: Key Code Sections

### Tab 7 Hyperparameter Extraction (Lines 2145-2182)

```python
# Extract hyperparameters from hyperparam_widgets
hyperparams = {}
if hasattr(self, 'tab7_hyperparam_widgets') and self.tab7_hyperparam_widgets:
    for param_name, widget in self.tab7_hyperparam_widgets.items():
        try:
            value = widget.get()
            if param_name in ['n_components', 'n_estimators', 'max_depth']:
                hyperparams[param_name] = int(value)
            else:
                hyperparams[param_name] = float(value)
        except Exception as e:
            print(f"  Warning: Could not extract {param_name}: {e}")

# Set model-specific defaults ONLY (prevent cross-contamination)
if model_name == 'PLS':
    if 'n_components' not in hyperparams:
        hyperparams['n_components'] = 10
elif model_name in ['Ridge', 'Lasso']:
    if 'alpha' not in hyperparams:
        hyperparams['alpha'] = 1.0  # ← DEFAULT VALUE! Is this the optimized value?
# ... etc
```

### Tab 7 Configuration Loading (Lines 3186-3227)

```python
# CRITICAL: Load wavelengths with FAIL LOUD validation
selected_wl = None
if subset == 'subset':  # Subset model
    # Try to get wavelengths from all_vars column
    all_vars_field = config.get('all_vars')

    if pd.isna(all_vars_field) or all_vars_field is None:
        messagebox.showerror("Cannot Load Model",
            f"FAIL LOUD: Subset model loaded but 'all_vars' field is missing/empty.\n\n"
            f"Model: {model_name}\n"
            f"Subset: {subset}\n\n"
            f"This model REQUIRES wavelength specification in 'all_vars' column.\n"
            f"Cannot proceed with loading.")
        self.notebook.select(4)  # Stay on Results tab
        return

    # Parse wavelength string
    try:
        wl_str = str(all_vars_field).strip()
        # ... parsing logic ...
```

**QUESTION:** Is `all_vars` actually present and correct in the config?

---

## FINAL NOTES

This is a **CRITICAL PRODUCTION-BLOCKING BUG** that requires immediate attention from a senior architect and experienced team.

The hyperparameter extraction is now working correctly, but the underlying configuration loading/storage is broken. The R² discrepancy (0.97 vs -0.07) is too large to be explained by floating-point errors or minor implementation differences.

**Most likely root causes (in order of probability):**
1. **Wrong wavelengths loaded** - Not using the correct 50 wavelengths from original analysis
2. **Wrong alpha value** - alpha=0.01 is default, not the optimized value from Results tab
3. **Field name mismatches** - Results tab stores with one field name, Tab 7 loads with different field name
4. **Wavelength subsetting bug** - Wavelengths getting corrupted during derivative+subset preprocessing

**Recommended first step:** Add comprehensive debug logging to both Results tab storage and Tab 7 loading, then run side-by-side comparison with Tab 6 (which works correctly).

---

**End of Handoff Document**
