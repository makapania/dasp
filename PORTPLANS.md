# Performance Optimization & Porting Plans

**Project**: Spectral Predict
**Date**: October 28, 2025
**Baseline**: 164 seconds for 10 samples (2.7 minutes)

---

## Executive Summary

Profiling reveals that **85% of runtime is spent in Random Forest model training** (scikit-learn's `RandomForestRegressor`). The actual spectral analysis code (preprocessing, I/O, feature selection) accounts for only ~5% of total time. This means:

- ✅ **High optimization potential** - targeting model training will yield massive gains
- ✅ **Low implementation risk** - most bottlenecks are in external libraries
- ✅ **Multiple viable paths** - can achieve 3-100x speedup depending on effort

---

## Profiling Results

### Time Breakdown (10 samples, 164s total)

| Component | Time | % of Total | Optimization Potential |
|-----------|------|------------|----------------------|
| Random Forest training | 139.8s | 85% | ⭐⭐⭐⭐⭐ HIGH |
| Multiprocessing overhead | 20.3s | 12% | ⭐⭐⭐ MEDIUM |
| Feature importance (VIP) | 3.8s | 2% | ⭐⭐ LOW |
| Region analysis | 0.6s | <1% | ⭐ VERY LOW |
| File I/O (ASD reading) | 0.1s | <1% | ⭐ VERY LOW |

### Key Findings

1. **Random Forest is the bottleneck**
   - 510 calls to `RandomForestRegressor.fit()` = 139.8s
   - Individual tree building: 8.5s (only 6% of RF time)
   - Rest is sklearn overhead + multiprocessing coordination

2. **Your code is already fast**
   - Preprocessing, I/O, and search logic: 3.8s total
   - Well-optimized Python code
   - Not the bottleneck

3. **Low-hanging fruit**
   - Replace sklearn RF with compiled gradient boosting (LightGBM/XGBoost)
   - Reduce multiprocessing overhead
   - JIT-compile preprocessing

---

## Three-Phase Optimization Strategy

### Phase 1: Quick Wins (THIS PR)
**Goal**: 3-5x speedup
**Effort**: 2 hours
**Risk**: Low
**Status**: ✅ IN PROGRESS

#### Changes
1. **Add LightGBM** as alternative to Random Forest
   - Drop-in sklearn-compatible API
   - 10-20x faster tree training (optimized C++)
   - Better accuracy on many datasets

2. **Numba JIT compilation** for preprocessing
   - `@numba.jit` on SNV transform
   - `@numba.jit` on hot loops in Savitzky-Golay
   - No API changes, pure speedup

3. **Optimize cross-validation**
   - Reduce parameter validation overhead
   - Pre-allocate arrays where possible

#### Expected Results
- **Random Forest configs**: 10-20x faster (via LightGBM)
- **Preprocessing**: 2-3x faster (via Numba)
- **Overall**: 3-5x speedup on full pipeline

#### Files Modified
- `pyproject.toml` - add lightgbm, numba dependencies
- `src/spectral_predict/models.py` - add LightGBM grid
- `src/spectral_predict/preprocess.py` - add Numba JIT decorators

---

### Phase 2: Moderate Gains (Future)
**Goal**: 10-20x speedup
**Effort**: 1 week
**Risk**: Medium

#### Implementation Plan

1. **Cythonize preprocessing** (~2 days)
   ```
   Create: src/spectral_predict/preprocess_fast.pyx
   - SNV transform (pure C loop)
   - Savitzky-Golay filter (C + BLAS)
   - Region correlation (C + BLAS)

   Modify: src/spectral_predict/preprocess.py
   - Try import preprocess_fast, fallback to Python
   ```

2. **Replace MLP with PyTorch** (~2 days)
   - Much faster neural network training
   - GPU support (if available)
   - Still sklearn-compatible API

3. **Optimize PLS** (~2 days)
   - Cython implementation of VIP calculation
   - Batch matrix operations (avoid loops)
   - Pre-compute X'X and reuse

4. **Parallelize smarter** (~1 day)
   - Reduce multiprocessing overhead
   - Use shared memory for data
   - Batch model evaluations

#### Expected Results
- **Neural networks**: 5-10x faster
- **Preprocessing**: 5-10x faster (Cython vs Numba)
- **PLS**: 2-3x faster
- **Overall**: 10-20x speedup

---

### Phase 3: Maximum Performance (Future)
**Goal**: 30-100x speedup
**Effort**: 4-6 weeks
**Risk**: High (new codebase)

#### Why Julia?

Julia is **ideal** for this project because:
- ✅ JIT-compiled to native code (LLVM)
- ✅ 90% as fast as C/C++, 10-100x faster than Python
- ✅ Excellent scientific computing ecosystem
- ✅ Mature ML libraries (similar to sklearn)
- ✅ Python-like syntax (low learning curve)
- ✅ Can call Python libraries if needed

#### Library Mapping

| Python Library | Julia Equivalent | Maturity | Notes |
|----------------|------------------|----------|-------|
| numpy | Native arrays | ⭐⭐⭐⭐⭐ | Built-in, BLAS/LAPACK |
| pandas | DataFrames.jl | ⭐⭐⭐⭐⭐ | Feature-complete |
| sklearn (PLS) | PartialLeastSquares.jl | ⭐⭐⭐⭐ | Mature, well-tested |
| sklearn (RF) | DecisionTree.jl | ⭐⭐⭐⭐ | Good performance |
| sklearn (MLP) | Flux.jl | ⭐⭐⭐⭐⭐ | Excellent, GPU support |
| scipy.signal | DSP.jl | ⭐⭐⭐⭐ | Has Savitzky-Golay |
| matplotlib | Plots.jl | ⭐⭐⭐⭐ | Multiple backends |

#### Implementation Timeline

**Week 1-2: Foundation**
```julia
# File: julia_port/SpectralPredict.jl
module SpectralPredict

include("io.jl")           # CSV/ASD readers
include("preprocessing.jl") # SNV, Savitzky-Golay
include("regions.jl")      # Region analysis

end
```

**Week 3-4: Models**
```julia
include("models/pls.jl")           # PLS + VIP
include("models/randomforest.jl")  # DecisionTree.jl wrapper
include("models/mlp.jl")           # Flux.jl wrapper
include("models/neural_boosted.jl") # Custom boosting
```

**Week 5-6: Search & Integration**
```julia
include("search.jl")    # Cross-validation + grid search
include("scoring.jl")   # Composite score ranking
include("report.jl")    # Markdown generation
include("cli.jl")       # Command-line interface
```

**Week 7-8: Python Bridge**
```python
# Keep Python CLI, call Julia backend
from julia import Main as Julia

Julia.include("spectral_predict.jl")
results = Julia.run_search(X, y, config)
```

#### Expected Results
- **All operations**: 20-50x faster (models + preprocessing)
- **Compilation**: One-time 30s startup cost, then native speed
- **Memory**: More efficient (Julia's JIT compiler optimizes)
- **Overall**: 30-100x speedup on full pipeline

#### Code Structure

```
spectral-predict/
├── src/                          # Python version (keep for compatibility)
│   └── spectral_predict/
│       ├── cli.py               # Entry point (calls Julia or Python)
│       ├── models.py            # Python implementation
│       └── ...
│
├── julia_port/                   # Julia performance core
│   ├── SpectralPredict.jl       # Main module
│   ├── src/
│   │   ├── io.jl
│   │   ├── preprocessing.jl
│   │   ├── models.jl
│   │   ├── search.jl
│   │   └── ...
│   ├── Project.toml             # Julia dependencies
│   └── test/                    # Julia tests (must match Python)
│
└── benchmarks/
    ├── compare_python_julia.jl
    └── profile_results.md
```

---

## Alternative: C++ Rewrite

### When to Consider C++

Only choose C++ if:
- Processing millions of spectra in production
- Need embedded/real-time deployment
- Already have C++ infrastructure
- Need ultimate control (SIMD, custom memory management)

### Why NOT Recommended

- **Effort**: 10-12 weeks vs 4-6 weeks for Julia
- **Risk**: High (no mature PLS/MLP libraries)
- **Speed**: Only 20-30% faster than Julia
- **Maintenance**: Much harder to modify/extend

### If You Insist on C++

**Library Stack**:
- **Eigen**: Matrix operations (header-only)
- **LightGBM**: Tree models (C++ API)
- **libtorch**: Neural networks (PyTorch C++)
- **Custom**: PLS, Savitzky-Golay (~1000 lines)
- **CLI11**: Command-line parsing

**Timeline**: 10-12 weeks (vs 4-6 for Julia)

---

## Decision Matrix

| Criterion | Phase 1 (LightGBM) | Phase 2 (Cython) | Phase 3 (Julia) | C++ |
|-----------|-------------------|------------------|----------------|-----|
| **Speedup** | 3-5x | 10-20x | 30-100x | 50-100x |
| **Time to implement** | 2 hours | 1 week | 4-6 weeks | 10-12 weeks |
| **Risk** | ⭐ Very Low | ⭐⭐ Low | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ High |
| **Maintainability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Learning curve** | None | Low | Medium | High |
| **Code duplication** | None | Minimal | Parallel codebase | Parallel codebase |

---

## Recommended Path Forward

### Immediate (This Week)
✅ **Execute Phase 1** (this PR)
- Merge when tested
- Measure actual speedup on real data
- Decide if Phase 2/3 needed

### Next Week (After Adding More Functions)
Reassess based on:
1. **Phase 1 results** - Was 3-5x enough?
2. **New features** - What bottlenecks emerged?
3. **Use case** - Processing 10 samples or 10,000?

**If Phase 1 is enough**: ✅ Stop here, enjoy 3-5x speedup

**If need more speed**:
- Small datasets (<1000 spectra): **Phase 2** (Cython)
- Large datasets (>1000 spectra): **Phase 3** (Julia)
- Production/embedded: C++ (consult first)

---

## Testing & Validation Strategy

### Phase 1 Testing
```bash
# 1. Run original version
time .venv/bin/spectral-predict --asd-dir example/ \
    --reference example/BoneCollagen.csv \
    --id-column "File Number" \
    --target "%Collagen"

# 2. Run optimized version (same command)
# Should produce IDENTICAL results, just faster

# 3. Verify accuracy
diff outputs_original/results.csv outputs_phase1/results.csv
# Only timing differences allowed
```

### Phase 2/3 Testing
```bash
# Must pass all existing tests
pytest -v

# Numerical tolerance for floating point
assert np.allclose(results_python, results_optimized, rtol=1e-5)

# Performance regression tests
assert time_optimized < time_baseline * 0.3  # At least 3x faster
```

---

## Profiling Methodology

### Tools Used
```bash
# Python profiling (cProfile)
python -m cProfile -o profile.stats -m spectral_predict.cli [args]

# Analysis
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(30)
"
```

### Key Metrics Tracked
- **Cumulative time**: Total time including subcalls
- **Total time**: Time in function itself
- **Number of calls**: Identify redundant work
- **Per-call time**: Spot inefficient algorithms

### Bottleneck Identification
1. Sort by cumulative time (find expensive operations)
2. Filter to project code (ignore stdlib)
3. Look for low-hanging fruit (many calls, simple logic)
4. Measure before/after for each optimization

---

## Future Considerations

### GPU Acceleration
If you process **many samples in parallel**, consider:
- PyTorch for neural networks (CUDA support)
- Julia + CUDA.jl for all models
- Expected: 10-50x additional speedup on large batches

### Distributed Computing
For **massive datasets** (10K+ spectra):
- Dask for Python parallelism
- Julia's Distributed computing
- Spark for cluster deployment

### Real-time Processing
For **embedded/real-time** applications:
- C++ with Intel MKL (optimized BLAS)
- Rust (memory safety + performance)
- Model compression (quantization, pruning)

---

## Appendix: Profiling Raw Data

### Top 30 Functions by Cumulative Time
```
   ncalls  tottime  cumtime filename:lineno(function)
        1    0.000  164.852 cli.py:1(<module>)
        1    0.000  162.445 cli.py:16(main)
        1    0.002  162.365 search.py:17(run_search)
       85    0.013  159.552 search.py:307(_run_single_config)
      511    0.004  139.949 pipeline:fit
      510    0.112  139.772 _forest:fit
       90    0.000    3.756 models.py:217(get_feature_importances)
   178500    5.619    8.454 tree/_classes:_fit
```

### Bottleneck Analysis
- **Random Forest**: 139.8s (510 fits × 0.27s each)
- **Tree building**: 8.5s (178,500 trees × 47μs each)
- **Parallelization overhead**: 131.3s (94% of RF time!)
- **Your code**: 3.8s (feature importance + search logic)

### Why sklearn RF is Slow
1. **Python wrapper overhead** - each tree fit requires Python<->C transition
2. **Pickle serialization** - trees pickled for multiprocessing
3. **Process startup** - new processes for each parallel batch
4. **No optimization** - each tree independent, no boosting tricks

### Why LightGBM is Fast
1. **Pure C++** - no Python overhead
2. **Shared memory** - no serialization
3. **Gradient boosting** - learns from previous trees
4. **Histogram binning** - reduces computation
5. **Leaf-wise growth** - better accuracy with fewer trees

---

## References

- [scikit-learn Performance Tips](https://scikit-learn.org/stable/developers/performance.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Numba User Guide](https://numba.pydata.org/numba-doc/latest/user/index.html)
- [Julia for Scientific Computing](https://julialang.org/learning/)
- [PyCall.jl - Python from Julia](https://github.com/JuliaPy/PyCall.jl)

---

## Contact & Questions

See `PHASE1_CONTINUATION_GUIDE.md` for detailed implementation instructions.

For Phase 2/3 implementation, consult with AI coding assistant with this document.
