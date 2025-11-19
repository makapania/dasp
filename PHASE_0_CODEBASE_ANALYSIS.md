# Phase 0: Codebase Analysis & Assessment

**Date**: 2025-11-18
**Branch**: `claude/julia-backend-setup-01LPirVmjEYpWsDwn5ScAW7s`
**Status**: In Progress

---

## Executive Summary

This document provides a comprehensive analysis of the Python codebase to inform the Julia backend migration strategy. Based on the R² Reproducibility Handoff document and code inspection, we've identified critical components, dependencies, and potential challenges.

---

## I. Repository Structure

```
dasp/
├── src/spectral_predict/          # Core library (33 modules)
│   ├── preprocess.py              # SNV, Savitzky-Golay derivatives
│   ├── search.py                  # Model training, grid search
│   ├── models.py                  # Model registry
│   ├── ensemble.py                # Ensemble methods
│   ├── variable_selection.py      # SPA, UVE, UVE-SPA
│   ├── calibration_transfer.py   # CT methods
│   └── readers/                   # Vendor format parsers
│
├── spectral_predict_gui_optimized.py  # Main GUI (23,962 lines!)
├── tests/                         # Test suite
├── example/                       # Usage examples
└── documentation/                 # Docs

Key Stats:
- Main GUI: 23,962 lines (monolithic)
- Core modules: 33 Python files
- Dependencies: numpy, pandas, scikit-learn, scipy, xgboost, lightgbm, catboost
```

---

## II. Critical Components from Handoff Document

### A. Preprocessing Pipeline (src/spectral_predict/preprocess.py)

**Location**: Lines 8-150

**Key Classes**:
1. **SNV** (Lines 8-40)
   - Stateless transformer (no fit operation)
   - Deterministic (mean/std normalization per spectrum)
   - **Critical**: Uses all features - sensitive to wavelength range (Issue #1)

2. **SavgolDerivative** (Lines 43-106)
   - Scipy's savgol_filter wrapper
   - Window, polyorder, deriv parameters
   - **Critical**: Edge effects depend on neighbors - sensitive to range (Issue #1)

3. **build_preprocessing_pipeline** (Lines 109-150)
   - Builds sklearn Pipeline
   - **deriv_snv** order: derivative → SNV (line 145-147)
   - **snv_deriv** order: SNV → derivative (line 141-143)

**Julia Requirements**:
- Exact numerical match for SNV (mean/std per row)
- Exact match for Savitzky-Golay (use DSP.jl or SciPy.jl)
- Pipeline ordering must be preserved

---

### B. Model Training & Variable Selection (src/spectral_predict/search.py)

**Location**: Lines 500-600 (approximate)

**Critical Section: Post-Preprocessing Wavelength Filtering** (Lines 529-557)
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
```

**This is Issue #1 Fix**: Filter AFTER preprocessing, not before!

**Julia Requirements**:
- Array copying (not views) in loops
- Wavelength filtering happens AFTER all preprocessing
- Preserve order (no sorting)

---

### C. State Management & Validation (spectral_predict_gui_optimized.py)

**Location**: Lines 12741+ (found via grep)

**Critical Section: Validation Indices Restoration**
```python
# CRITICAL: Restore validation indices BEFORE validation check
# This ensures the check sees the correct validation state
if 'validation_indices' in config and config.get('validation_set_enabled'):
    self.validation_indices = set(config.get('validation_indices', []))
    self.validation_enabled.set(True)
    print(f"✓ Restored {len(self.validation_indices)} validation indices from model config")
```

**This is Issue #2 Fix**: Restore BEFORE validate, not after!

**Julia Requirements**:
- State restoration order is critical
- Use Set data structure for indices
- Deep copy for cached data

---

### D. Wavelength Order Preservation

**Expected Location**: Lines 14347-14352 (from handoff, not yet verified)

**Critical Pattern**:
```python
# Remove duplicates while preserving order from available_wavelengths
# DO NOT sort - sorting changes feature order and breaks R² reproducibility!
selected_set = set(selected)
selected = [wl for wl in available_wavelengths if wl in selected_set]
```

**This is Issue #4 Fix**: Never sort wavelengths!

**Julia Requirements**:
- Preserve order from DataFrame columns
- Use Set for O(1) lookup, but filter in original order
- NEVER call sort() on wavelengths

---

## III. Python Dependencies & Julia Equivalents

| Python Package | Purpose | Julia Equivalent | Status |
|----------------|---------|------------------|--------|
| numpy | Array operations | Base Julia Arrays | ✅ Built-in |
| pandas | DataFrames | DataFrames.jl | ✅ Mature |
| scikit-learn | ML models, pipelines | MLJ.jl, GLM.jl | ⚠️ Need to verify |
| scipy.signal | Savitzky-Golay | DSP.jl | ⚠️ Need to test |
| xgboost | Gradient boosting | XGBoost.jl | ✅ Available |
| lightgbm | Gradient boosting | LightGBM.jl | ✅ Available |
| catboost | Gradient boosting | ? | ❌ May not exist |
| matplotlib | Plotting | Plots.jl, Makie.jl | ✅ Available |

**Critical Gaps to Investigate**:
1. Does MLJ.jl have Ridge/Lasso/ElasticNet with identical regularization?
2. Does MultivariateStats.jl PLS match scikit-learn exactly?
3. Does DSP.jl savgol_filter match scipy exactly?
4. Is CatBoost available in Julia?

---

## IV. Supported Models (from pyproject.toml and code inspection)

### Deterministic Models (Critical for Phase 1-3)
- **Ridge** - L2 regularization (scikit-learn)
- **Lasso** - L1 regularization (scikit-learn)
- **ElasticNet** - L1+L2 regularization (scikit-learn)
- **PLS** - Partial Least Squares (likely scikit-learn or custom)

### Ensemble Models (Phase 6+)
- **XGBoost** - Gradient boosting
- **LightGBM** - Gradient boosting
- **CatBoost** - Gradient boosting

### Neural Models (Future)
- **neural_boosted.py** - Custom neural models

**Julia Migration Priority**:
1. **Phase 1**: Ridge only (simplest)
2. **Phase 2**: PLS (most common in spectroscopy)
3. **Phase 3**: Lasso, ElasticNet
4. **Phase 4**: Ensemble models
5. **Phase 5**: Neural models (if needed)

---

## V. Preprocessing Methods

From `preprocess.py` line 109-150:

1. **raw** - No preprocessing
2. **snv** - Standard Normal Variate only
3. **deriv** - Savitzky-Golay derivative only
4. **snv_deriv** - SNV → derivative
5. **deriv_snv** - derivative → SNV

**Critical for Testing**:
- **deriv** alone worked (handoff Stumbling Point #2)
- **deriv_snv** failed before fix (Issue #1)
- Order matters: derivative → SNV ≠ SNV → derivative

---

## VI. Variable Selection Methods

From code inspection (search.py):

1. **importance** - Model-based feature importance
2. **spa** - Successive Projections Algorithm
3. **uve** - Uninformative Variable Elimination
4. **uve_spa** - Hybrid UVE → SPA

**Julia Requirements**:
- Implement all 4 methods
- Ensure identical selection order
- Maintain reproducibility (same random seeds)

---

## VII. Architecture Observations

### Monolithic GUI (23,962 lines!)
- **Challenge**: Hard to test, hard to profile
- **Recommendation**: Extract core logic for Julia backend
- **Strategy**: Julia backend doesn't need GUI - just computation engine

### Two-Path Preprocessing (PATH A vs PATH B)
From handoff document:

**PATH A: Derivative methods**
```
Full spectrum → Derivative → Variable selection → Subset → Train (no pipeline)
```

**PATH B: Non-derivative methods**
```
Variable selection → Subset → Build pipeline with preprocessing → Train
```

**Why?** Derivatives need full spectral context (neighbors)

**Julia Requirements**:
- Implement both paths
- Never mix them
- Clear separation in code

---

## VIII. Identified Bottlenecks (Hypothesis - Need Profiling)

Based on codebase structure, likely bottlenecks:

### 1. Grid Search (search.py)
- Nested loops: models × preprocessing × variable selection
- Cross-validation for each combination
- **Potential speedup**: Multi-threading (Julia native)

### 2. Savitzky-Golay Filtering
- Applied to full spectrum repeatedly
- scipy implementation (C backend)
- **Potential speedup**: Julia SIMD, or already fast enough

### 3. Variable Selection (UVE, SPA)
- PLS on random data (UVE)
- Iterative projection (SPA)
- **Potential speedup**: Parallelization

### 4. Cross-Validation
- K-fold splitting, model training
- Repeated for each grid point
- **Potential speedup**: Parallel folds

### 5. Model Training (PLS, Ridge, etc.)
- Likely already optimized (BLAS/LAPACK)
- **Potential speedup**: Minimal, unless Julia has better BLAS

**Action Required**: Profile Python code to confirm!

---

## IX. Test Suite Analysis

From `/tests` directory:

**Existing Tests**:
- `test_r2_consistency_fix.py` - R² reproducibility tests (critical!)
- `test_lightgbm_fix.py` - LightGBM specific tests
- `test_model_*.py` - Model integration tests
- `test_io_*.py` - Data I/O tests

**Critical for Julia Migration**:
- `test_r2_consistency_fix.py` likely implements the 5 tests from handoff
- Need to extract golden reference outputs
- Create differential testing framework (Python vs Julia)

---

## X. Risk Assessment

### High Risk Areas
1. **Numerical Precision**
   - Python uses Float64 (numpy default)
   - Julia must match exactly
   - Test: SNV mean/std, Savitzky-Golay output

2. **Indexing**
   - Python: 0-based
   - Julia: 1-based
   - **High risk for off-by-one errors**

3. **Array Mutation**
   - Python: mutable by default
   - Julia: copy-on-write semantics differ
   - **Risk**: Subtle bugs in loops

4. **Package Maturity**
   - MLJ.jl is younger than scikit-learn
   - **Risk**: Missing features, numerical differences

5. **Regularization Paths**
   - Ridge/Lasso convergence criteria may differ
   - **Risk**: Coefficients don't match exactly

### Medium Risk Areas
1. **Random Number Generation**
   - Different RNG implementations
   - Mitigation: Use same seeds, verify fold indices

2. **Pipeline Semantics**
   - scikit-learn Pipeline vs MLJ.jl
   - Mitigation: Test thoroughly

3. **Broadcasting**
   - Numpy broadcasting vs Julia broadcasting
   - Mitigation: Explicit dimensions

### Low Risk Areas
1. **I/O** - Not performance critical
2. **Plotting** - Can use Python backend if needed
3. **GUI** - Keep Python GUI, call Julia backend

---

## XI. Recommended Phase 0 Experiments

### Experiment 1: SNV Numerical Precision
**Goal**: Verify Julia can match Python SNV exactly

**Method**:
1. Create test data (100 samples, 1000 wavelengths)
2. Implement SNV in Julia
3. Compare outputs element-wise
4. Check: max absolute difference < 1e-15

**Success Criteria**: Bit-exact match (Float64 precision)

---

### Experiment 2: Savitzky-Golay Match
**Goal**: Verify DSP.jl matches scipy

**Method**:
1. Use same test data
2. Apply savgol_filter with window=7, polyorder=2, deriv=1
3. Compare outputs
4. Test edge effects

**Success Criteria**: Max difference < 1e-12 (account for numerical order)

---

### Experiment 3: Ridge Regression Match
**Goal**: Verify MLJ.jl Ridge matches scikit-learn

**Method**:
1. Simple dataset (no preprocessing)
2. Train Ridge with alpha=1.0
3. Compare: coefficients, predictions, R²
4. Test cross-validation

**Success Criteria**: R² difference < 0.001

---

### Experiment 4: PLS Match
**Goal**: Verify MultivariateStats.jl matches scikit-learn

**Method**:
1. Same dataset
2. Train PLS with n_components=5
3. Compare: loadings, scores, predictions, R²

**Success Criteria**: R² difference < 0.001

**Critical Note**: This is the HIGHEST RISK experiment. PLS implementations vary!

---

## XII. Profiling Plan (Next Step)

### Step 1: Instrument Python Code
```python
import cProfile
import pstats

# Profile run_search() function
profiler = cProfile.Profile()
profiler.enable()

# Run typical analysis
results = run_search(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Step 2: Identify Bottlenecks
- List top 10 slowest functions
- Measure: % time, cumulative time, call count
- Prioritize: high time × high call count

### Step 3: Estimate Speedup Potential
- Which functions are vectorizable?
- Which can be parallelized?
- Which are already optimized (BLAS)?

### Step 4: Set Baseline
- Record current runtime for standard analysis
- This is our target to beat

---

## XIII. Julia Ecosystem Research (Next Step)

### Packages to Evaluate

**Core ML**:
- `MLJ.jl` - Machine learning framework (like scikit-learn)
- `GLM.jl` - Generalized linear models (Ridge, Lasso)
- `MultivariateStats.jl` - PLS, PCA
- `ScikitLearn.jl` - Python bridge (fallback if needed)

**Signal Processing**:
- `DSP.jl` - Digital signal processing (Savitzky-Golay)
- `SciPy.jl` - Python scipy bridge (fallback)

**Data**:
- `DataFrames.jl` - Like pandas
- `CSV.jl` - CSV I/O
- `XLSX.jl` - Excel I/O

**Gradient Boosting**:
- `XGBoost.jl` - XGBoost wrapper
- `LightGBM.jl` - LightGBM wrapper

**Testing**:
- `Test.jl` - Unit testing (built-in)
- `BenchmarkTools.jl` - Performance benchmarking

### Evaluation Criteria

For each package:
1. **Maturity**: Last commit, stars, issues
2. **Numerical accuracy**: Test against Python
3. **Performance**: Faster than Python?
4. **API compatibility**: Easy to translate?
5. **Dependencies**: Stable ecosystem?

---

## XIV. GO/NO-GO Decision Framework

### Proceed to Phase 1 if:
✅ SNV matches Python (Experiment 1)
✅ Savitzky-Golay matches Python (Experiment 2)
✅ Ridge matches Python (Experiment 3)
✅ Julia packages are stable/maintained
✅ Profiling shows significant speedup potential (>2x)

### Pivot to Alternatives if:
⚠️ PLS doesn't match (Experiment 4) → Use ScikitLearn.jl bridge
⚠️ Julia packages unstable → Use hybrid approach (Python + Julia hotspots)
⚠️ Speedup < 2x → Consider Numba/Cython instead

### Abort if:
❌ Cannot achieve numerical match after 2 weeks
❌ Julia ecosystem missing critical features
❌ Performance worse than Python

---

## XV. Next Actions

1. **Commit current work** ✓
   - JULIA_BACKEND_PLAN.md
   - PHASE_0_CODEBASE_ANALYSIS.md

2. **Profile Python implementation**
   - Run cProfile on typical analysis
   - Identify top 10 bottlenecks
   - Document baseline performance

3. **Run Julia experiments**
   - Experiment 1: SNV
   - Experiment 2: Savitzky-Golay
   - Experiment 3: Ridge
   - Experiment 4: PLS

4. **Research Julia packages**
   - Evaluate maturity
   - Test numerical accuracy
   - Benchmark performance

5. **Create Phase 0 Report**
   - Summarize findings
   - Make GO/NO-GO recommendation
   - Present to stakeholders

---

## XVI. Open Questions

1. **What is the typical runtime for a full analysis?**
   - How many models × preprocessing × variable selection?
   - How long does it take now?
   - What would be acceptable?

2. **What is the dataset scale?**
   - Typical number of samples?
   - Typical number of wavelengths?
   - Memory constraints?

3. **Is there existing profiling data?**
   - Previous performance analysis?
   - Known bottlenecks?

4. **What is the deployment environment?**
   - OS (Linux, macOS, Windows)?
   - Hardware (CPU, RAM, GPU)?
   - Python version?

5. **Are there existing Julia developers on the team?**
   - Who will maintain Julia code?
   - What's the learning curve?

---

## XVII. References

- **Handoff Document**: R2_REPRODUCIBILITY_HANDOFF_FOR_JULIA.md
- **Main Plan**: JULIA_BACKEND_PLAN.md
- **Key Files**:
  - src/spectral_predict/preprocess.py (SNV, derivatives)
  - src/spectral_predict/search.py (model training)
  - spectral_predict_gui_optimized.py (validation logic)
- **Dependencies**: pyproject.toml

---

**Status**: Phase 0 in progress - Codebase analysis complete
**Next**: Profiling and Julia experiments
**Timeline**: Week 1 of 12-week plan
