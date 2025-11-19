# Phase 0: Performance Insights & Revised Strategy

**Date**: 2025-11-18
**Status**: 🔥 **CRITICAL UPDATE** - Strategy Revised

---

## 🎯 Key Insight from User

> "The analyses that are slower tend to be the machine learning and boosting ones. PLS is not bad at all, but even that might be better in julia with larger datasets."

**This is GREAT news!** This fundamentally changes our migration strategy.

---

## 📊 What This Tells Us

### Slow (High Priority to Optimize):
1. **Machine learning models** - Likely the grid search process
2. **Boosting models** (XGBoost, LightGBM, CatBoost)
3. **Cross-validation** (K-fold repetition)
4. **Variable selection** (SPA, UVE - these call PLS repeatedly)

### Fast (Low Priority):
1. **PLS** - "not bad at all"
2. **Preprocessing** (SNV, Savitzky-Golay) - likely already fast

---

## 🔄 Revised Migration Strategy: HYBRID APPROACH

Instead of full Python → Julia rewrite, use **targeted optimization**:

### Phase 1 (HIGHEST IMPACT): Parallelize Grid Search
**Target**: The nested loops in `search.py`

**Current (Slow)**:
```python
for model in models:                    # Sequential
    for preprocess in preprocessing:    # Sequential
        for varsel in variable_selection:  # Sequential
            # Train model with cross-validation
```

**Optimized (Fast)**:
```julia
# Parallel grid search in Julia
using Distributed

results = @distributed (vcat) for combo in grid_combinations
    # Each combination runs on separate CPU core
    train_evaluate(combo)
end
```

**Expected Speedup**: 4-8x on 8-core CPU (near-linear scaling!)

**Risk**: LOW - Just parallelizing, not changing algorithms

---

### Phase 2: Optimize Variable Selection (SPA, UVE)

From `search.py:560-600`, variable selection methods call PLS repeatedly:

**Current**:
- UVE: Builds PLS models on random noise (many iterations)
- SPA: Iterative projection algorithm
- UVE-SPA: Hybrid (slowest)

**Why It's Slow**:
- Python loops calling scikit-learn PLS repeatedly
- Can't parallelize within Python easily

**Julia Optimization**:
```julia
# Parallelize UVE random iterations
using Distributed

importances = @distributed (+) for i in 1:n_iterations
    # Build PLS on noise, get importance
    uve_iteration(X, y, i)
end
```

**Expected Speedup**: 5-10x (parallel + Julia efficiency)

**Risk**: MEDIUM - Need to preserve exact algorithms

---

### Phase 3: Boosting Models (XGBoost, LightGBM)

**Critical Question**: Are boosting models slow due to:
- **A) Python overhead** (data conversion, etc.)
- **B) Training time** (algorithm itself)
- **C) Hyperparameter grid search** (trying many configs)

**Investigation Needed**:
```python
# Profile to see where time is spent
import time

# Case A: Python overhead
t0 = time.time()
xgb_model = xgb.XGBRegressor(**params)  # Instantiation
print(f"Instantiation: {time.time() - t0}s")

t0 = time.time()
xgb_model.fit(X, y)  # Training
print(f"Training: {time.time() - t0}s")

# If instantiation is slow → Python overhead
# If training is slow → Algorithm itself
```

**Julia Options**:
1. **XGBoost.jl** - Wrapper for same C++ library
   - Should be same speed as Python (both call C++)
   - Benefit: Eliminate Python overhead
   - Expected speedup: 1.1-1.5x (minimal but easy)

2. **LightGBM.jl** - Wrapper for same C library
   - Same story as XGBoost

3. **CatBoost** - May not have Julia wrapper
   - Keep in Python, call via PyCall
   - Or skip if not critical

**Risk**: LOW - Wrappers for same C/C++ libraries

---

### Phase 4 (LOWEST PRIORITY): PLS Migration

**User said**: "PLS is not bad at all"

**Strategy**: **DON'T migrate PLS initially!**

**Options**:
1. **Keep Python PLS** (via PyCall/PythonCall)
   - Zero risk (already works)
   - Known to be correct
   - Performance is acceptable

2. **Try Julia PLS later** (Phase 6+)
   - Only if profiling shows it's worth it
   - Only for larger datasets
   - Can validate against Python

**Benefit**: Eliminates highest risk item (PLS numerical matching)!

---

## 🎯 Revised Phase Priorities

### NEW Phase 1: Parallel Grid Search (Week 2)
**Goal**: Parallelize model × preprocessing × variable selection loop

**Approach**:
```julia
# Julia orchestrates the grid search
using Distributed, PyCall

@everywhere begin
    py"""
    from spectral_predict.search import train_single_model
    """
    train_model = py"train_single_model"
end

# Create all combinations
grid = [(m, p, v) for m in models, p in preprocess, v in varsel]

# Distribute across cores
results = pmap(grid) do (model, preprocess, varsel)
    # Call Python function for single model
    train_model(model, preprocess, varsel)
end
```

**Advantages**:
- ✅ Minimal code changes (mostly orchestration)
- ✅ Low risk (calling existing Python code)
- ✅ High impact (4-8x speedup expected)
- ✅ Easy to validate (same results, just parallel)

**Disadvantages**:
- ⚠️ Still calling Python (some overhead)
- ⚠️ Need to serialize data between processes

---

### NEW Phase 2: Julia Variable Selection (Week 3-4)
**Goal**: Rewrite SPA, UVE, UVE-SPA in Julia

**Approach**:
1. Extract variable selection algorithms to pure functions
2. Rewrite in Julia (parallel loops)
3. Call Python PLS via PyCall (keep what works!)
4. Validate outputs match exactly

**Expected Speedup**: 5-10x (parallelization + Julia efficiency)

**Risk**: MEDIUM (need exact algorithm match)

---

### NEW Phase 3: Boosting Wrappers (Week 5)
**Goal**: Use XGBoost.jl, LightGBM.jl

**Approach**:
```julia
using XGBoost

# Same API as Python
bst = xgboost(X, y, num_round=100,
              max_depth=6, eta=0.3)
```

**Expected Speedup**: 1.2-2x (eliminate Python overhead)

**Risk**: LOW (wrappers for same C++ code)

---

### NEW Phase 4: Full Integration (Week 6-7)
**Goal**: End-to-end Julia pipeline (except PLS)

**Architecture**:
```julia
# Julia orchestration
using PyCall

# Import Python PLS (keep what works!)
@pyimport sklearn.cross_decomposition as skl_pls

function run_analysis(X, y, config)
    # Parallel grid search (Julia)
    results = parallel_grid_search(X, y, config)

    # Variable selection (Julia)
    importances = variable_selection(X, y, method)

    # Model training
    if model == "PLS"
        # Call Python (known to work)
        model = skl_pls.PLSRegression(n_components=5)
        model.fit(X, y)
    elseif model == "XGBoost"
        # Use Julia wrapper
        model = xgboost(X, y, params...)
    end

    return results
end
```

---

## 📈 Expected Performance Gains

| Component | Current (Python) | Optimized (Julia) | Speedup | Priority |
|-----------|------------------|-------------------|---------|----------|
| **Grid Search** | Sequential | Parallel (8 cores) | **8x** | 🔥 **HIGHEST** |
| **Variable Selection** | Python loops | Julia parallel | **5-10x** | 🔥 High |
| **XGBoost/LightGBM** | Python wrapper | Julia wrapper | **1.2-2x** | 🟡 Medium |
| **PLS** | Fast enough | Keep Python | **1x** | 🟢 Low |
| **Preprocessing** | Fast enough | Keep Python | **1x** | 🟢 Low |

**Overall Expected Speedup**: **10-20x** (from parallelization alone!)

**With aggressive optimization**: **20-50x** possible

---

## 🎯 Revised Success Criteria

### Must-Have (Phase 1-2)
- ✅ Grid search parallelization works
- ✅ R² matches Python within 0.001
- ✅ Speedup ≥ 4x on 8-core CPU
- ✅ No regressions (same results, just faster)

### Should-Have (Phase 3-4)
- ✅ Variable selection in Julia
- ✅ Boosting models using Julia wrappers
- ✅ Overall speedup ≥ 10x

### Nice-to-Have (Phase 5+)
- ✅ PLS in Julia (if profiling shows benefit)
- ✅ GPU acceleration for larger datasets
- ✅ Overall speedup ≥ 20x

---

## 🚀 Quick Win Strategy

### Week 1: Proof of Concept
**Goal**: Prove parallel grid search works

**Steps**:
1. Extract single model training to pure function
2. Create Julia script that calls it in parallel
3. Compare results (should match exactly)
4. Measure speedup

**Code Sketch**:
```julia
# test_parallel_grid.jl
using Distributed
addprocs(8)  # Use 8 cores

@everywhere using PyCall
@everywhere py"""
from spectral_predict.search import train_single_model
import numpy as np
"""

# Test data
X = rand(100, 1000)
y = rand(100)

# Sequential (baseline)
@time results_seq = [train_model(m, p) for m in models, p in preprocess]

# Parallel (optimized)
@time results_par = @distributed (vcat) for (m, p) in combinations
    train_model(m, p)
end

# Verify match
@assert results_seq == results_par
```

**If this works**: Proceed with hybrid strategy

**If this fails**: Fall back to full plan (or Numba/Cython)

---

## 🎓 Why This Strategy is Better

### Compared to Full Migration:

**Original Plan**:
- ❌ Rewrite everything in Julia (high risk)
- ❌ PLS numerical matching (highest risk item)
- ❌ 12 weeks of work
- ❌ All-or-nothing (can't deploy partially)

**Hybrid Plan**:
- ✅ Keep what works (PLS, preprocessing)
- ✅ Optimize bottlenecks only (grid search, variable selection)
- ✅ Incremental deployment (each phase adds value)
- ✅ Lower risk (calling proven Python code)
- ✅ Faster to production (useful results in 2-4 weeks)

---

## 🔬 Profiling Priorities (Updated)

When you run profiling, focus on:

### 1. Grid Search Orchestration
**Questions**:
- How much time in outer loops vs inner training?
- How many combinations are tested?
- What's the average time per model?

**Why**: Determines parallel speedup potential

### 2. Variable Selection Methods
**Questions**:
- How long does SPA take?
- How long does UVE take?
- How many PLS calls within each?

**Why**: Determines optimization priority

### 3. Boosting Model Training
**Questions**:
- XGBoost: instantiation vs training time?
- LightGBM: same question?
- What % of total time?

**Why**: Determines if Julia wrappers help

### 4. PLS Training
**Questions**:
- What % of total time?
- How many times is it called?

**Why**: Confirms it's not the bottleneck (user already said this)

---

## 📊 Profiling Command (Updated)

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run your typical slow analysis
# E.g., full grid search with boosting models
results = run_search(
    X=data_X,
    y=data_y,
    models=['XGBoost', 'LightGBM', 'Ridge', 'PLS'],
    preprocessing=['raw', 'snv', 'deriv', 'deriv_snv'],
    variable_selection=['importance', 'spa', 'uve'],
    # ... other params
)

profiler.disable()

# Analyze
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')

# Focus on these patterns:
print("\n=== GRID SEARCH OVERHEAD ===")
stats.print_stats('search.py')  # Grid search orchestration

print("\n=== VARIABLE SELECTION ===")
stats.print_stats('variable_selection')  # SPA, UVE methods

print("\n=== BOOSTING MODELS ===")
stats.print_stats('xgboost')
stats.print_stats('lightgbm')

print("\n=== PLS ===")
stats.print_stats('PLS')  # Should be small % of time
```

---

## 🎯 Next Actions (Revised)

### Immediate (You or Me):
1. **Run profiling** with focus on grid search, variable selection, boosting
2. **Measure** typical analysis runtime (baseline)
3. **Count** how many model combinations tested
4. **Identify** if boosting is Python overhead or algorithm

### Week 1 Experiment:
1. **Prototype parallel grid search** in Julia
2. **Call existing Python code** via PyCall
3. **Measure speedup** (expect 4-8x)
4. **Validate** results match exactly

### If Week 1 Succeeds:
- ✅ GO: Continue with hybrid strategy
- ✅ Deploy parallel grid search to production (immediate value!)
- ✅ Move to Phase 2 (variable selection)

### If Week 1 Fails:
- ⚠️ Reassess: Maybe PyCall overhead is too high?
- ⚠️ Try: Numba for parallelization instead?
- ⚠️ Consider: Multiprocessing in Python first (easier)

---

## 💡 Alternative: Python Multiprocessing (Even Faster to Deploy)

If Julia seems like overkill, consider **Python's built-in parallelization**:

```python
from multiprocessing import Pool
from functools import partial

def train_single_combination(combo, X, y):
    model, preprocess, varsel = combo
    # Train and evaluate
    return result

# Parallel grid search
with Pool(8) as pool:  # 8 cores
    combinations = [(m, p, v) for m in models
                    for p in preprocess
                    for v in varsel]

    train_func = partial(train_single_combination, X=X, y=y)
    results = pool.map(train_func, combinations)
```

**Advantages**:
- ✅ Pure Python (no Julia needed)
- ✅ 30 minutes to implement
- ✅ Same speedup as Julia (4-8x)
- ✅ Zero risk (same code, just parallel)

**Disadvantages**:
- ⚠️ GIL limitations (but not issue for scikit-learn - releases GIL)
- ⚠️ Memory overhead (separate processes)

**Recommendation**: **Try this first!** If it works, you get 4-8x speedup in 30 minutes!

---

## 🎯 Final Recommendation

### Path A: Python Multiprocessing (FASTEST to value)
**Timeline**: 1 day
**Effort**: Minimal (30 min coding, 30 min testing)
**Risk**: Very low
**Speedup**: 4-8x (likely good enough!)

### Path B: Julia Hybrid (if multiprocessing not enough)
**Timeline**: 2-4 weeks
**Effort**: Medium (learning Julia, PyCall setup)
**Risk**: Low-medium
**Speedup**: 10-20x

### Path C: Full Julia Migration (if hybrid not enough)
**Timeline**: 12 weeks
**Effort**: High
**Risk**: High (PLS numerical matching)
**Speedup**: 20-50x (with GPU)

---

## ❓ Questions for You

1. **How many CPU cores** does your analysis machine have?
   - Determines parallel speedup potential
   - If 4 cores → 4x max, if 16 cores → 16x potential

2. **Typical grid search size**?
   - How many models × preprocessing × variable selection combinations?
   - E.g., 5 models × 4 preprocess × 3 varsel = 60 combinations
   - More combinations → more benefit from parallelization

3. **Would 4-8x speedup be sufficient**?
   - If YES → Try Python multiprocessing first!
   - If NO → Need Julia or GPU

4. **Can you run a profiling session**?
   - Would give us exact numbers
   - Can prioritize based on actual data

---

**My Recommendation**: Start with Python `multiprocessing` - it's a 30-minute experiment that could solve your problem entirely!

Should I create a quick prototype for parallel grid search in Python?
