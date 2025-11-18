# Julia Ecosystem Research for Spectral Prediction Backend

**Date**: 2025-11-18
**Status**: Initial Assessment

---

## Package Evaluation Checklist

For Julia backend migration, we need to evaluate these packages:

---

## I. Core Machine Learning

### MLJ.jl - Machine Learning Framework
**URL**: https://github.com/alan-turing-institute/MLJ.jl

**Purpose**: scikit-learn equivalent for Julia

**Evaluation Needed**:
- [ ] Last commit date (active development?)
- [ ] Number of stars/contributors
- [ ] Ridge regression support
- [ ] Lasso regression support
- [ ] ElasticNet regression support
- [ ] Cross-validation API
- [ ] Pipeline support
- [ ] Hyperparameter tuning
- [ ] Numerical precision vs scikit-learn

**Test**: Train Ridge on synthetic data, compare coefficients to scikit-learn

---

### GLM.jl - Generalized Linear Models
**URL**: https://github.com/JuliaStats/GLM.jl

**Purpose**: Linear models (Ridge, Lasso)

**Evaluation Needed**:
- [ ] Ridge regression implementation
- [ ] Regularization parameter handling
- [ ] Convergence criteria
- [ ] Numerical stability
- [ ] Performance vs scikit-learn

**Test**: Ridge regression with L2 penalty, verify R² matches Python

---

### MultivariateStats.jl - PLS and Multivariate Analysis
**URL**: https://github.com/JuliaStats/MultivariateStats.jl

**Purpose**: PLS regression (critical for spectroscopy!)

**Evaluation Needed**:
- [ ] PLS implementation (NIPALS algorithm?)
- [ ] n_components parameter
- [ ] Loadings/scores calculation
- [ ] Prediction accuracy
- [ ] **CRITICAL**: Does it match scikit-learn PLS numerically?

**Test**:
1. Train PLS with n_components=5
2. Compare loadings, scores, predictions
3. Verify R² matches within 0.001
4. Test edge cases (n_components > n_features)

**Risk Level**: **HIGH** - PLS implementations vary significantly!

---

## II. Signal Processing

### DSP.jl - Digital Signal Processing
**URL**: https://github.com/JuliaDSP/DSP.jl

**Purpose**: Savitzky-Golay filtering (derivatives)

**Evaluation Needed**:
- [ ] `savitzky_golay` function exists?
- [ ] Parameters: window_length, polyorder, deriv
- [ ] Edge effects handling (same as scipy?)
- [ ] Numerical precision vs scipy.signal.savgol_filter

**Test**:
1. Create test signal (100 samples, 1000 features)
2. Apply savgol with window=7, polyorder=2, deriv=1
3. Compare output to Python scipy
4. Check: max absolute difference < 1e-12

**Fallback**: Use SciPy.jl (PyCall bridge) if DSP.jl doesn't match

---

### SciPy.jl - Python SciPy Bridge
**URL**: https://github.com/JuliaPy/SciPy.jl

**Purpose**: Call Python scipy from Julia (fallback option)

**Evaluation Needed**:
- [ ] Can call scipy.signal.savgol_filter?
- [ ] Performance overhead (Python call)
- [ ] Data conversion overhead (Julia ↔ Python)

**Use Case**: If DSP.jl doesn't match scipy exactly

---

## III. Data Handling

### DataFrames.jl
**URL**: https://github.com/JuliaData/DataFrames.jl

**Purpose**: pandas equivalent

**Evaluation Needed**:
- [ ] Maturity (well-established)
- [ ] API similarity to pandas
- [ ] Performance
- [ ] Column selection/indexing

**Test**: Load CSV, filter rows, select columns

---

### CSV.jl
**URL**: https://github.com/JuliaData/CSV.jl

**Purpose**: Fast CSV reading

**Evaluation Needed**:
- [ ] Read CSV to DataFrame
- [ ] Handle missing values
- [ ] Performance vs pandas.read_csv

---

### XLSX.jl
**URL**: https://github.com/felipenoris/XLSX.jl

**Purpose**: Excel file handling

**Evaluation Needed**:
- [ ] Read .xlsx files
- [ ] Write .xlsx files
- [ ] Compatibility with openpyxl/xlsxwriter

---

## IV. Gradient Boosting

### XGBoost.jl
**URL**: https://github.com/dmlc/XGBoost.jl

**Purpose**: XGBoost wrapper

**Evaluation Needed**:
- [ ] Wrapper for C++ XGBoost
- [ ] API parity with Python xgboost
- [ ] Same hyperparameters?
- [ ] Numerical reproducibility

**Priority**: Medium (Phase 4+)

---

### LightGBM.jl
**URL**: https://github.com/IQVIA-ML/LightGBM.jl

**Purpose**: LightGBM wrapper

**Evaluation Needed**:
- [ ] Active maintenance?
- [ ] API parity with Python lightgbm
- [ ] Performance

**Priority**: Medium (Phase 4+)

---

### CatBoost.jl
**URL**: Search GitHub/Julia package registry

**Purpose**: CatBoost wrapper

**Evaluation Needed**:
- [ ] Does this package exist?
- [ ] If not, can we call Python catboost via PyCall?

**Priority**: Low (may not be critical)

---

## V. Testing & Benchmarking

### Test.jl
**Purpose**: Unit testing (built-in to Julia)

**Evaluation Needed**:
- [ ] @testset macro
- [ ] @test macro for assertions
- [ ] Test running framework

---

### BenchmarkTools.jl
**URL**: https://github.com/JuliaCI/BenchmarkTools.jl

**Purpose**: Performance benchmarking

**Evaluation Needed**:
- [ ] @benchmark macro
- [ ] Statistical timing analysis
- [ ] Memory allocation tracking

---

## VI. Python Interop (Fallback Strategy)

### PyCall.jl / PythonCall.jl
**URL**:
- PyCall: https://github.com/JuliaPy/PyCall.jl
- PythonCall: https://github.com/cjdoris/PythonCall.jl

**Purpose**: Call Python from Julia (hybrid approach)

**Evaluation Needed**:
- [ ] Which is preferred? (PythonCall is newer)
- [ ] Can call scikit-learn directly?
- [ ] Performance overhead
- [ ] Data conversion cost (Array ↔ ndarray)

**Use Cases**:
1. **Fallback**: If Julia package doesn't match Python numerically
2. **Hybrid**: Keep Python for some models, use Julia for hotspots
3. **Gradual migration**: Start with Python calls, replace incrementally

---

## VII. Evaluation Experiments

### Experiment 1: SNV Implementation
**File**: `julia_experiments/test_snv.jl`

```julia
using Statistics

function snv(X::Matrix{Float64})
    """Standard Normal Variate transformation"""
    means = mean(X, dims=2)
    stds = std(X, dims=2)

    # Avoid division by zero
    stds[stds .== 0] .= 1.0

    return (X .- means) ./ stds
end

# Test
X_test = randn(100, 1000)  # 100 samples, 1000 wavelengths
X_snv = snv(X_test)

# Export to CSV for comparison with Python
using CSV, DataFrames
CSV.write("julia_snv_output.csv", DataFrame(X_snv, :auto))

println("SNV output saved. Compare with Python implementation.")
```

**Python comparison**:
```python
import numpy as np
from spectral_predict.preprocess import SNV

X_test = np.random.randn(100, 1000)
snv = SNV()
X_snv_python = snv.transform(X_test)

# Load Julia output
import pandas as pd
X_snv_julia = pd.read_csv("julia_snv_output.csv").values

# Compare
diff = np.abs(X_snv_python - X_snv_julia)
print(f"Max difference: {diff.max()}")
print(f"Mean difference: {diff.mean()}")

# Should be < 1e-15 for Float64 precision
assert diff.max() < 1e-14, "SNV outputs don't match!"
```

---

### Experiment 2: Savitzky-Golay
**File**: `julia_experiments/test_savgol.jl`

```julia
using DSP  # Or SciPy.jl if DSP doesn't work

# Test data
X_test = randn(100, 1000)

# Apply Savitzky-Golay filter
# TODO: Find correct function in DSP.jl
# window = 7, polyorder = 2, deriv = 1

# Export for comparison
```

---

### Experiment 3: Ridge Regression
**File**: `julia_experiments/test_ridge.jl`

```julia
using GLM  # or MLJ

# Simple test data
X = randn(100, 50)
y = randn(100)

# Train Ridge with alpha=1.0
# TODO: Find correct API

# Compare coefficients and predictions
```

---

### Experiment 4: PLS Regression
**File**: `julia_experiments/test_pls.jl`

```julia
using MultivariateStats

# Test data
X = randn(100, 50)
y = randn(100)

# Train PLS with n_components=5
# TODO: Verify API

# Extract loadings, scores, predictions
# Compare with sklearn.cross_decomposition.PLSRegression
```

---

## VIII. Decision Matrix

| Package | Priority | Risk | Fallback Available? |
|---------|----------|------|---------------------|
| DataFrames.jl | **High** | Low | No (but mature) |
| DSP.jl | **High** | Medium | Yes (SciPy.jl) |
| GLM.jl | **High** | Medium | Yes (ScikitLearn.jl) |
| MultivariateStats.jl | **CRITICAL** | **HIGH** | Yes (ScikitLearn.jl) |
| MLJ.jl | High | Medium | Yes (ScikitLearn.jl) |
| XGBoost.jl | Medium | Low | No (C++ wrapper) |
| LightGBM.jl | Medium | Medium | Yes (PyCall) |
| CatBoost.jl | Low | High | Yes (PyCall) |

**Critical Path**: MultivariateStats.jl PLS **must** match Python!

---

## IX. Research Action Plan

### Phase 0A: Package Discovery (2 hours)
1. Browse Julia package registry
2. Check GitHub stars, last commit, issues
3. Read documentation for each package
4. Identify API differences vs Python

### Phase 0B: Numerical Validation (1 day)
1. Run Experiment 1 (SNV) - should be easy ✓
2. Run Experiment 2 (Savitzky-Golay) - medium difficulty
3. Run Experiment 3 (Ridge) - medium difficulty
4. Run Experiment 4 (PLS) - **CRITICAL, HIGH RISK**

### Phase 0C: Performance Benchmarking (1 day)
1. Time SNV (Julia vs Python)
2. Time Savitzky-Golay (Julia vs Python)
3. Time Ridge training (Julia vs Python)
4. Time PLS training (Julia vs Python)

### Phase 0D: Report Findings (half day)
1. Document package maturity
2. Document numerical accuracy
3. Document performance gains
4. Make GO/NO-GO recommendation

---

## X. GO/NO-GO Criteria

### ✅ PROCEED to Phase 1 if:
- SNV matches Python (< 1e-14 difference)
- Savitzky-Golay matches Python (< 1e-12 difference)
- Ridge matches Python (R² < 0.001 difference)
- Packages are actively maintained (commits in last 6 months)
- Performance is ≥ 1.5x Python (even without optimization)

### ⚠️ HYBRID APPROACH if:
- PLS doesn't match Python → Use ScikitLearn.jl for PLS only
- Some packages missing → Use PyCall for those specific functions
- Performance is 1.5x-2x → Migrate only hotspots

### ❌ ABORT Julia migration if:
- Cannot achieve numerical match after 1 week of debugging
- No active Julia packages for critical functionality
- Performance is worse than Python
- Ecosystem appears unstable

**Fallback**: Numba/Cython for Python optimization instead

---

## XI. Web Resources

**Julia Packages**:
- https://juliahub.com (package registry)
- https://github.com/JuliaML (ML ecosystem)
- https://github.com/JuliaStats (statistics packages)
- https://github.com/JuliaDSP (signal processing)

**Tutorials**:
- https://alan-turing-institute.github.io/MLJ.jl/dev/ (MLJ tutorial)
- https://juliadsp.github.io/DSP.jl/stable/ (DSP tutorial)

**Community**:
- https://discourse.julialang.org (Julia forum)
- https://julialang.slack.com (Slack community)

---

## XII. Next Steps

1. **Install Julia** (if not already installed)
   ```bash
   # Download from https://julialang.org/downloads/
   # Or use package manager (apt, brew, etc.)
   ```

2. **Create experiments directory**
   ```bash
   mkdir julia_experiments
   cd julia_experiments
   ```

3. **Install packages**
   ```julia
   using Pkg
   Pkg.add("DataFrames")
   Pkg.add("CSV")
   Pkg.add("DSP")
   Pkg.add("GLM")
   Pkg.add("MultivariateStats")
   Pkg.add("MLJ")
   Pkg.add("BenchmarkTools")
   ```

4. **Run experiments**
   ```bash
   julia julia_experiments/test_snv.jl
   python compare_snv.py
   ```

5. **Document findings** in Phase 0 report

---

**Status**: Research plan defined, ready to execute
**Timeline**: 2-3 days for complete evaluation
**Output**: GO/NO-GO decision with evidence
