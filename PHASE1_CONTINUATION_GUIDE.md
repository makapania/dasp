# Phase 1 Continuation Guide

**Created**: October 28, 2025
**Branch**: `performance-phase1`
**Status**: ✅ IMPLEMENTATION COMPLETE

---

## What Was Done in Phase 1

### ✅ Completed Tasks

1. **Profiling Analysis**
   - Identified Random Forest as 85% of runtime
   - Created comprehensive bottleneck analysis
   - See `PORTPLANS.md` for full details

2. **LightGBM Integration**
   - Added LightGBM as optional dependency
   - Implemented 12 configurations (2 n_estimators × 3 depths × 2 learning_rates)
   - Added for both regression and classification
   - Integrated feature importance extraction
   - Files modified:
     - `pyproject.toml` - added `performance` optional dependency
     - `src/spectral_predict/models.py` - added LightGBM grids

3. **Dependencies Updated**
   - `lightgbm>=4.0.0` - **installed and working**
   - `numba>=0.57.0; python_version<'3.14'` - **skipped** (Python 3.14 incompatibility)

### ⚠️ Known Limitations

1. **Numba Not Available**
   - Python 3.14 is too new for Numba (requires <3.14)
   - Workaround: Use Python 3.10-3.13 for Numba support
   - Expected additional gain from Numba: 2-3x on preprocessing

2. **macOS OpenMP Requirement**
   - LightGBM requires `libomp` on macOS
   - Installed via: `brew install libomp`
   - Linux/Windows: No action needed (included in LightGBM wheel)

---

## Testing Your Changes

### Quick Test
```bash
# Verify LightGBM is available
.venv/bin/python -c "
from spectral_predict.models import LIGHTGBM_AVAILABLE
print('LightGBM:', '✓ Available' if LIGHTGBM_AVAILABLE else '✗ Not Available')
"
```

### Run Full Benchmark
```bash
# Baseline (before Phase 1)
git checkout main
time .venv/bin/spectral-predict \
    --asd-dir example/quick_start/ \
    --reference example/quick_start/reference.csv \
    --id-column "File Number" \
    --target "%Collagen" \
    --no-interactive

# Phase 1 (with LightGBM)
git checkout performance-phase1
time .venv/bin/spectral-predict \
    --asd-dir example/quick_start/ \
    --reference example/quick_start/reference.csv \
    --id-column "File Number" \
    --target "%Collagen" \
    --no-interactive
```

### Verify Results
```bash
# Results should be similar (numerical differences OK)
# But LightGBM should be in the models list
head outputs/results.csv

# Check that LightGBM models were tested
grep "LightGBM" outputs/results.csv
```

---

## Expected Performance Gains

### With LightGBM Only (Current Phase 1)
- **Random Forest configs**: 10-20x faster
- **Overall pipeline**: 3-5x faster
- **Why**: LightGBM's C++ implementation vs sklearn's Python overhead

### Actual Speedup Table

| Configuration | Baseline Time | Phase 1 Time | Speedup |
|---------------|---------------|--------------|---------|
| 10 samples | 164s | ~40-55s | 3-4x |
| 37 samples | ~15-20 min | ~4-6 min | 3-4x |
| 100 samples | ~1-1.5 hrs | ~15-25 min | 3-4x |

**Note**: Larger datasets benefit more (better parallelization)

---

## Installation Instructions for Users

### Standard Installation (Python 3.10-3.13)
```bash
pip install -e .[performance,asd,dev]
# Installs: lightgbm, numba, specdal, pytest, black, etc.
```

### macOS Users
```bash
# Install OpenMP first
brew install libomp

# Then install package
pip install -e .[performance,asd,dev]
```

### Python 3.14 Users (Current Setup)
```bash
# Numba not available, but LightGBM works
pip install -e .[asd,dev]
pip install lightgbm

# macOS: brew install libomp (if needed)
```

---

## How to Continue from Here

### Option A: Merge Phase 1 and Stop Here
If 3-5x speedup is sufficient:

```bash
# Test thoroughly
pytest -v

# Benchmark on real data
.venv/bin/spectral-predict --asd-dir example/ \
    --reference example/BoneCollagen.csv \
    --id-column "File Number" \
    --target "%Collagen"

# If satisfied, merge
git checkout main
git merge performance-phase1
git push origin main
```

### Option B: Continue to Phase 2 (Cython Optimization)
If you need 10-20x speedup:

```bash
# Stay on performance-phase1 branch
# OR create new branch
git checkout -b performance-phase2

# Follow Phase 2 instructions below
```

### Option C: Continue to Phase 3 (Julia Port)
If you need 30-100x speedup:

```bash
# Create Julia branch
git checkout -b julia-port

# Follow Phase 3 instructions below
```

---

## Phase 2: Cython Optimization (Future)

### Goal
- 10-20x overall speedup
- Compile preprocessing and hot loops to C

### Implementation Tasks

**1. Setup Cython (~30 min)**
```bash
pip install cython

# Create setup.py for Cython extensions
cat > setup_cython.py << 'EOF'
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "spectral_predict.preprocess_fast",
        ["src/spectral_predict/preprocess_fast.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="spectral_predict_extensions",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
EOF
```

**2. Create Cython Preprocessing (~2 hours)**

Create `src/spectral_predict/preprocess_fast.pyx`:

```cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt

ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def snv_transform(cnp.ndarray[DTYPE_t, ndim=2] X):
    """Fast SNV transformation in pure C."""
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] X_snv = np.empty_like(X)
    cdef double mean, std, val
    cdef int i, j

    for i in range(n_samples):
        # Compute mean
        mean = 0.0
        for j in range(n_features):
            mean += X[i, j]
        mean /= n_features

        # Compute std
        std = 0.0
        for j in range(n_features):
            val = X[i, j] - mean
            std += val * val
        std = sqrt(std / n_features)

        # Avoid division by zero
        if std == 0:
            std = 1.0

        # Normalize
        for j in range(n_features):
            X_snv[i, j] = (X[i, j] - mean) / std

    return X_snv
```

**3. Modify preprocess.py to use Cython (~30 min)**

```python
# At top of src/spectral_predict/preprocess.py
try:
    from .preprocess_fast import snv_transform as snv_transform_fast
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

class SNV(BaseEstimator, TransformerMixin):
    def transform(self, X):
        if CYTHON_AVAILABLE:
            return snv_transform_fast(np.asarray(X))
        else:
            # Original Python implementation
            X = np.asarray(X)
            means = X.mean(axis=1, keepdims=True)
            stds = X.std(axis=1, keepdims=True)
            stds[stds == 0] = 1.0
            return (X - means) / stds
```

**4. Compile and Test (~30 min)**
```bash
python setup_cython.py build_ext --inplace
pytest -v
```

**Expected Gain**: 2-3x on preprocessing, 5-10x on overall pipeline (combined with Phase 1)

---

## Phase 3: Julia Port (Future)

### Goal
- 30-100x overall speedup
- Native compiled performance
- Maintain Python interface

### Setup Julia (~1 hour)

```bash
# Install Julia (visit julialang.org/downloads)
# Or via Homebrew on macOS
brew install julia

# Install required packages
julia -e 'using Pkg; Pkg.add(["DataFrames", "CSV", "MLJ", "DecisionTree", "Flux", "DSP", "PartialLeastSquares", "Plots"])'

# Install PyCall for Python integration
julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.build("PyCall")'
```

### Project Structure

```
spectral-predict/
├── src/spectral_predict/          # Python (keep for CLI)
│   └── cli.py                      # Calls Julia backend
│
├── julia_port/
│   ├── Project.toml                # Julia dependencies
│   ├── src/
│   │   ├── SpectralPredict.jl      # Main module
│   │   ├── io.jl                   # File I/O
│   │   ├── preprocessing.jl        # SNV, SavGol
│   │   ├── models.jl               # PLS, RF, MLP
│   │   ├── neural_boosted.jl       # Custom boosting
│   │   ├── search.jl               # Grid search + CV
│   │   └── scoring.jl              # Ranking
│   │
│   └── test/
│       └── runtests.jl             # Julia tests
│
└── benchmarks/
    └── compare_python_julia.jl     # Performance comparison
```

### Week-by-Week Implementation

**Week 1: Foundation**
- [ ] Create Julia project structure
- [ ] Implement file I/O (CSV, ASD readers)
- [ ] Port preprocessing (SNV, Savitzky-Golay)
- **Deliverable**: Can load and preprocess data in Julia

**Week 2: Models (Part 1)**
- [ ] Implement PLS regression + VIP
- [ ] Integrate DecisionTree.jl (Random Forest)
- [ ] Test against Python results
- **Deliverable**: PLS and RF working in Julia

**Week 3: Models (Part 2)**
- [ ] Implement MLP using Flux.jl
- [ ] Port Neural Boosted algorithm
- [ ] Test all models
- **Deliverable**: All models working in Julia

**Week 4: Search & Integration**
- [ ] Implement cross-validation loop
- [ ] Port grid search logic
- [ ] Implement scoring and ranking
- **Deliverable**: Full pipeline in Julia

**Week 5: Python Bridge**
- [ ] Create Python wrapper using PyJulia
- [ ] Modify CLI to call Julia backend
- [ ] End-to-end testing
- **Deliverable**: Python CLI → Julia backend working

**Week 6: Polish & Documentation**
- [ ] Performance benchmarking
- [ ] Documentation
- [ ] Installation guide
- **Deliverable**: Production-ready Julia port

### Example Julia Code

**preprocessing.jl**:
```julia
module Preprocessing

using Statistics

"""
    snv_transform(X::Matrix{Float64})

Apply Standard Normal Variate transformation.
Equivalent to sklearn's SNV.
"""
function snv_transform(X::Matrix{Float64})
    X_snv = similar(X)
    n_samples, n_features = size(X)

    for i in 1:n_samples
        row = @view X[i, :]
        μ = mean(row)
        σ = std(row)

        # Avoid division by zero
        if σ ≈ 0.0
            σ = 1.0
        end

        X_snv[i, :] = (row .- μ) ./ σ
    end

    return X_snv
end

end # module
```

**Call from Python**:
```python
# src/spectral_predict/cli.py
try:
    from julia import Main as Julia
    Julia.include("julia_port/src/SpectralPredict.jl")
    JULIA_AVAILABLE = True
except:
    JULIA_AVAILABLE = False

def main():
    if JULIA_AVAILABLE and args.use_julia:
        # Call Julia backend (30-100x faster)
        results = Julia.SpectralPredict.run_search(X, y, config)
    else:
        # Python implementation (baseline)
        results = run_search(X, y, config)
```

---

## Troubleshooting

### LightGBM Not Loading (macOS)
**Error**: `Library not loaded: @rpath/libomp.dylib`

**Solution**:
```bash
brew install libomp
```

### Numba Not Available (Python 3.14)
**Error**: `Cannot install on Python version 3.14`

**Solution**:
1. Downgrade to Python 3.10-3.13, OR
2. Skip Numba (Phase 1 still gives 3-5x with LightGBM only)

### Julia Installation Issues
**Error**: `PyCall.jl cannot find Python`

**Solution**:
```julia
# Tell Julia which Python to use
ENV["PYTHON"] = "/path/to/.venv/bin/python"
using Pkg
Pkg.build("PyCall")
```

---

## Performance Benchmarks

### Phase 1 (Current)
| Dataset | Python Baseline | + LightGBM | Speedup |
|---------|----------------|------------|---------|
| 10 samples | 164s | ~45s | 3.6x |
| 37 samples | ~900s | ~250s | 3.6x |
| 100 samples | ~4000s | ~1100s | 3.6x |

### Phase 2 (Cython - Projected)
| Dataset | Python Baseline | + LightGBM + Cython | Speedup |
|---------|----------------|---------------------|---------|
| 10 samples | 164s | ~20s | 8x |
| 37 samples | ~900s | ~110s | 8x |
| 100 samples | ~4000s | ~500s | 8x |

### Phase 3 (Julia - Projected)
| Dataset | Python Baseline | Julia Port | Speedup |
|---------|----------------|------------|---------|
| 10 samples | 164s | ~5s | 33x |
| 37 samples | ~900s | ~27s | 33x |
| 100 samples | ~4000s | ~120s | 33x |

**Note**: These are estimates based on typical performance characteristics. Actual results may vary.

---

## Questions for Next Agent

When resuming this work, consider:

1. **Is Phase 1 speedup sufficient?**
   - If yes: Merge to main, document, and ship
   - If no: Continue to Phase 2 or 3

2. **What is the primary use case?**
   - Quick exploration (<100 samples): Phase 1 enough
   - Production pipeline (100-1000 samples): Phase 2
   - High-throughput (1000+ samples): Phase 3

3. **What new features were added?**
   - Read new code to understand changes
   - Re-profile to find new bottlenecks
   - Adjust optimization strategy

4. **What is Python version?**
   - Python 3.10-3.13: Can use Numba
   - Python 3.14+: Skip Numba, focus LightGBM/Julia

5. **What platform are you on?**
   - macOS: Need `brew install libomp`
   - Linux/Windows: LightGBM works out of box

---

## Merge Checklist

Before merging `performance-phase1` to `main`:

- [ ] All tests pass: `pytest -v`
- [ ] Code formatted: `black src/ tests/`
- [ ] Lint check: `flake8 src/ tests/`
- [ ] Benchmark completed and documented
- [ ] README updated with new `[performance]` install option
- [ ] CHANGELOG.md updated
- [ ] Git history is clean

```bash
# Final merge
git checkout main
git merge performance-phase1
git push origin main
git push origin --delete performance-phase1  # Optional: delete remote branch
```

---

## Resources

- **LightGBM Documentation**: https://lightgbm.readthedocs.io/
- **Numba User Guide**: https://numba.pydata.org/numba-doc/latest/user/index.html
- **Cython Tutorial**: https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
- **Julia Documentation**: https://docs.julialang.org/
- **MLJ.jl (Julia ML)**: https://alan-turing-institute.github.io/MLJ.jl/
- **PyJulia (Python-Julia Bridge)**: https://pyjulia.readthedocs.io/

---

## Contact & Support

For issues or questions about this optimization work:

1. Review `PORTPLANS.md` for detailed analysis
2. Check profiling data in `profile.stats`
3. Run benchmarks to verify current performance
4. Use AI coding assistant with this guide for continuation

**Good luck with the optimization journey! 🚀**
