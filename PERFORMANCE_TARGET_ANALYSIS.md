# Performance Target Analysis

**Date**: 2025-11-18
**Status**: 🎯 **PERFORMANCE TARGETS DEFINED**

---

## 📊 User Requirements

### Data Scale
- **Samples**: 150 typical, up to **thousands**
- **Wavelengths**: 350-2500 (up to 2,150 features)
- **Common range**: 800-2500 (1,700 features)
- **Matrix size**: Up to thousands × 2,150 = millions of data points

### Performance
- **Current**: 4-5 hours for larger analyses
- **Target**: **< 1 hour**
- **Required speedup**: **4-5x minimum**

### Hardware
- **OS**: Windows
- **CPU**: Recent multicore (assume 8-16 cores)
- **GPU**: ✅ **Available!** (CRITICAL for acceleration)

---

## 🎯 Performance Strategy

### Phase 1: Quick Win (1-2 days) → 4-8x speedup
**Target**: Get from 4-5 hours to 30-60 minutes

**Method**: Python multiprocessing + GPU XGBoost/LightGBM

```python
# 1. Parallel grid search (CPU)
with Pool(8) as pool:  # Use all CPU cores
    results = pool.map(train_model, combinations)
# Expected: 4-8x speedup on grid search

# 2. GPU-accelerated boosting
xgb_model = xgb.XGBRegressor(
    tree_method='gpu_hist',  # ← Use GPU!
    gpu_id=0
)
# Expected: 10-20x speedup on XGBoost/LightGBM
```

**Combined Impact**:
- Grid search parallelization: 4-8x
- GPU boosting (if boosting is 50% of time): 1.5-2x additional
- **Total: 6-16x speedup** → Should easily hit < 1 hour!

**Effort**: Minimal (~30 min for multiprocessing, ~10 min for GPU flags)

**Risk**: Very low (just config changes)

---

### Phase 2: Julia Hybrid (if Phase 1 not enough) → 10-20x speedup
**Target**: Get to 15-30 minutes

**Method**: Julia orchestration + GPU + Python models

```julia
using Distributed, CUDA, PyCall

# 1. Parallel grid search (Julia)
@distributed for combo in combinations
    # Train model
end

# 2. GPU operations (Julia)
using CUDA
X_gpu = CuArray(X)  # Move to GPU
# Fast matrix operations on GPU

# 3. Call Python models (proven code)
@pyimport xgboost as xgb
```

**Combined Impact**:
- Better parallelization: 8-16x (all cores)
- GPU matrix ops: 2-5x additional
- **Total: 16-80x potential** → Definitely < 1 hour

**Effort**: 2-4 weeks

**Risk**: Medium

---

### Phase 3: Full GPU Pipeline (if needed) → 20-100x speedup
**Target**: Get to 2-5 minutes (stretch goal)

**Method**: CuPy + RAPIDS + GPU-everything

```python
import cupy as cp  # GPU numpy
from cuml import Ridge, PLS  # GPU scikit-learn

# All on GPU
X_gpu = cp.array(X)
y_gpu = cp.array(y)

# GPU preprocessing
X_snv = (X_gpu - X_gpu.mean(axis=1)) / X_gpu.std(axis=1)

# GPU model training
model = Ridge().fit(X_gpu, y_gpu)
```

**Effort**: 4-8 weeks

**Risk**: High (newer ecosystem)

---

## 🚀 Recommended Approach

### Step 1: Python Quick Wins (DO THIS FIRST!)

#### 1A. Enable GPU XGBoost/LightGBM (10 minutes)

**XGBoost**:
```python
# Before (CPU)
xgb_model = xgb.XGBRegressor(n_estimators=100)

# After (GPU) - ONE LINE CHANGE!
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    tree_method='gpu_hist',  # ← Enable GPU
    gpu_id=0
)
```

**Expected speedup**: 10-20x for XGBoost training

**LightGBM**:
```python
# Before (CPU)
lgb_model = lgb.LGBMRegressor(n_estimators=100)

# After (GPU)
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    device='gpu',  # ← Enable GPU
    gpu_platform_id=0,
    gpu_device_id=0
)
```

**Expected speedup**: 10-15x for LightGBM training

**Installation** (if needed):
```bash
# XGBoost GPU
pip install xgboost --upgrade

# LightGBM GPU (Windows)
pip install lightgbm --install-option=--gpu
```

---

#### 1B. Parallelize Grid Search (30 minutes)

Replace sequential loop in `search.py` with parallel version:

```python
# Before (SLOW)
for model in models:
    for preprocess in preprocessing:
        for varsel in variable_selection:
            result = train_model(model, preprocess, varsel)
            results.append(result)

# After (FAST)
from multiprocessing import Pool

def train_single(combo):
    model, preprocess, varsel = combo
    return train_model(model, preprocess, varsel)

combinations = [(m, p, v) for m in models
                for p in preprocessing
                for v in variable_selection]

with Pool(8) as pool:  # 8 CPU cores
    results = pool.map(train_single, combinations)
```

**Expected speedup**: 4-8x for grid search

---

### Step 2: Benchmark Results

After Step 1 (quick wins), measure:

```python
import time

# Current baseline
t0 = time.time()
results = run_search(...)  # Old way
baseline_time = time.time() - t0
print(f"Baseline: {baseline_time/3600:.1f} hours")

# Optimized
t0 = time.time()
results = run_search_parallel_gpu(...)  # New way
optimized_time = time.time() - t0
print(f"Optimized: {optimized_time/3600:.1f} hours")
print(f"Speedup: {baseline_time/optimized_time:.1f}x")
```

**Expected results**:
- Baseline: 4-5 hours
- Optimized: 20-45 minutes (6-15x speedup)
- **Achievement**: ✅ Under 1 hour!

---

### Step 3: Decision Point

**If Step 1 achieves < 1 hour**:
- ✅ **DONE!** Deploy and enjoy
- No Julia needed
- Minimal effort, maximum gain

**If Step 1 gets close but not quite there** (e.g., 1.5 hours):
- Consider additional Python optimizations:
  - Reduce CV folds (5 → 3)
  - Reduce variable selection iterations
  - Pre-filter obviously bad hyperparameters

**If Step 1 doesn't help enough**:
- Move to Julia hybrid strategy
- But this seems unlikely given GPU availability!

---

## 💡 Why GPU is a Game-Changer

Your GPU changes everything. Here's why:

### Boosting Models (XGBoost/LightGBM)
**CPU**: 10-100 seconds per model
**GPU**: 1-5 seconds per model
**Speedup**: **10-20x**

### Matrix Operations (if using RAPIDS)
**CPU numpy**: Limited by memory bandwidth
**GPU cupy**: Massive parallelism (thousands of cores)
**Speedup**: **10-100x** for large matrices

### Your Data Scale
- Thousands of samples × 2,000 wavelengths
- GPU loves large matrices
- This is the sweet spot for GPU acceleration

---

## 🧪 Quick GPU Test

To verify GPU is working:

```python
import xgboost as xgb
import numpy as np
import time

# Test data (similar to your scale)
X = np.random.randn(1000, 2000)  # 1000 samples, 2000 wavelengths
y = np.random.randn(1000)

# CPU version
print("Testing CPU...")
t0 = time.time()
model_cpu = xgb.XGBRegressor(n_estimators=100, tree_method='hist')
model_cpu.fit(X, y)
time_cpu = time.time() - t0
print(f"CPU time: {time_cpu:.2f}s")

# GPU version
print("Testing GPU...")
t0 = time.time()
model_gpu = xgb.XGBRegressor(n_estimators=100, tree_method='gpu_hist', gpu_id=0)
model_gpu.fit(X, y)
time_gpu = time.time() - t0
print(f"GPU time: {time_gpu:.2f}s")

print(f"\nSpeedup: {time_cpu/time_gpu:.1f}x")
```

**Expected output**:
```
Testing CPU...
CPU time: 45.23s
Testing GPU...
GPU time: 3.12s

Speedup: 14.5x
```

If this works, you'll easily hit your < 1 hour target!

---

## 📊 Projected Performance

### Current State (Baseline)
```
4-5 hours total
├─ Grid search overhead: ~5-10%
├─ XGBoost/LightGBM: ~60-70% (MAIN BOTTLENECK)
├─ Variable selection: ~20-30%
└─ PLS/Ridge: ~5-10%
```

### After GPU + Multiprocessing
```
20-30 minutes total (8-12x speedup)
├─ Grid search: Parallelized (4-8x faster)
├─ XGBoost/LightGBM: GPU (10-20x faster) ← HUGE WIN
├─ Variable selection: Parallelized (4-8x faster)
└─ PLS/Ridge: Same (already fast)
```

**Result**: ✅ **Well under 1 hour target!**

---

## 🎯 Action Plan (Priority Order)

### TODAY (30 minutes)
1. ✅ Run GPU test (code above)
2. ✅ Verify GPU XGBoost works
3. ✅ Measure baseline timing on real analysis

### TOMORROW (2-3 hours)
1. Enable GPU for XGBoost in search.py
2. Enable GPU for LightGBM in search.py
3. Test on real dataset
4. Measure speedup

### THIS WEEK (1 day)
1. Implement parallel grid search (multiprocessing)
2. Integrate with existing code
3. Test thoroughly
4. Measure total speedup

### Expected Timeline
- **Day 1**: GPU enabled → 2-4x speedup
- **Day 2**: Parallel grid search → 8-12x total speedup
- **Day 3**: Testing & validation → Deploy!

**Total effort**: 2-3 days to achieve < 1 hour target

---

## 🚨 Potential Issues & Solutions

### Issue 1: GPU not recognized
**Symptom**: "No GPU found" error
**Solution**:
```bash
# Check CUDA installation
nvidia-smi  # Should show GPU

# Install CUDA toolkit if missing
# Download from: https://developer.nvidia.com/cuda-downloads
```

### Issue 2: Out of GPU memory
**Symptom**: "CUDA out of memory" error
**Solution**:
- Reduce batch size
- Process in chunks
- Use CPU for very large models

### Issue 3: Multiprocessing on Windows
**Symptom**: "Can't pickle" errors
**Solution**:
```python
if __name__ == '__main__':
    # Windows requires this guard
    with Pool(8) as pool:
        results = pool.map(...)
```

---

## 💰 Cost-Benefit Analysis

### Option 1: Python Quick Wins (RECOMMENDED)
- **Effort**: 2-3 days
- **Cost**: $0 (just config changes)
- **Speedup**: 8-12x (4-5 hours → 20-30 min)
- **Risk**: Very low
- **Maintenance**: Easy (Python)

### Option 2: Julia Hybrid
- **Effort**: 2-4 weeks
- **Cost**: Learning curve, new tooling
- **Speedup**: 15-30x (4-5 hours → 10-15 min)
- **Risk**: Medium
- **Maintenance**: Harder (two languages)

### Option 3: Full GPU Pipeline (RAPIDS)
- **Effort**: 4-8 weeks
- **Cost**: Ecosystem maturity risk
- **Speedup**: 50-100x (4-5 hours → 2-5 min)
- **Risk**: High (newer tools)
- **Maintenance**: Medium

**Clear winner**: Option 1 (Python quick wins)

---

## 🎓 Conclusion

**Your specific situation**:
- ✅ GPU available (huge advantage!)
- ✅ Boosting models slow (GPU helps most here)
- ✅ Recent multicore CPU (parallelization works)
- ✅ Target is 4-5x speedup (achievable with simple changes!)

**Recommendation**:
1. **Start with Python GPU + multiprocessing** (2-3 days)
2. **This will likely solve your problem completely**
3. **Only consider Julia if Python approach falls short**

**Next steps**:
1. Run GPU test (30 min)
2. Enable GPU flags in XGBoost/LightGBM (30 min)
3. Implement parallel grid search (2-3 hours)
4. Celebrate hitting < 1 hour target! 🎉

---

**Status**: Ready to implement quick wins
**Expected outcome**: 4-5 hours → 20-30 minutes with minimal effort
**Julia still needed**: Probably not, but available if needed
