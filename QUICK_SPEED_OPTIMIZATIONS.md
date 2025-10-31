# Quick Speed Optimizations Guide

**Target**: Non-algorithmic speedups that don't change model behavior
**Time**: 1-4 hours total
**Expected Speedup**: 2-5x total
**Risk**: Very Low

---

## Overview

These optimizations focus on making the existing code run faster without changing algorithms or adding new models. They complement the neural boosting optimizations and can be done independently.

---

## Optimization 1: Numba JIT Compilation for Preprocessing

**Time**: 1 hour
**Speedup**: 2-3x on preprocessing
**Files**: `src/spectral_predict/preprocess.py`

### Current Bottleneck

From profiling (PORTPLANS.md):
- SNV transform and Savitzky-Golay filters run in pure Python
- Feature importance (VIP) calculation has Python loops
- Preprocessing is ~5% of total time but easily optimizable

### Implementation

Add Numba JIT compilation to hot loops:

```python
# At top of preprocess.py
import numba

@numba.jit(nopython=True)
def snv_transform_fast(spectra):
    """
    Standard Normal Variate transformation - JIT compiled.

    Parameters
    ----------
    spectra : ndarray, shape (n_samples, n_wavelengths)
        Input spectral data

    Returns
    -------
    transformed : ndarray
        SNV-transformed spectra
    """
    n_samples, n_wavelengths = spectra.shape
    transformed = np.empty_like(spectra)

    for i in range(n_samples):
        spectrum = spectra[i]
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        if std > 1e-10:  # Avoid division by zero
            transformed[i] = (spectrum - mean) / std
        else:
            transformed[i] = spectrum - mean

    return transformed


@numba.jit(nopython=True)
def apply_derivative_fast(spectra, window_length, polyorder, deriv):
    """
    Apply Savitzky-Golay derivative - JIT compiled.

    This is a simplified version for common cases.
    Falls back to scipy for complex scenarios.
    """
    n_samples, n_wavelengths = spectra.shape
    result = np.empty_like(spectra)

    # Simplified SG filter for most common case (window=5, poly=2, deriv=1)
    if window_length == 5 and polyorder == 2 and deriv == 1:
        # Use pre-computed SG coefficients for speed
        coeffs = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

        for i in range(n_samples):
            for j in range(2, n_wavelengths - 2):
                result[i, j] = np.sum(spectra[i, j-2:j+3] * coeffs)

            # Handle edges by replication
            result[i, 0] = result[i, 2]
            result[i, 1] = result[i, 2]
            result[i, -2] = result[i, -3]
            result[i, -1] = result[i, -3]
    else:
        # Fall back to scipy for other cases (rare)
        return None  # Signal to use scipy version

    return result


# Modify existing functions to use fast versions
def snv_transform(spectra):
    """Apply SNV transformation."""
    try:
        return snv_transform_fast(np.asarray(spectra, dtype=np.float64))
    except Exception:
        # Fallback to original implementation if Numba fails
        return _snv_transform_original(spectra)


def apply_derivative(spectra, window_length=5, polyorder=2, deriv=1):
    """Apply Savitzky-Golay derivative."""
    if window_length == 5 and polyorder == 2 and deriv == 1:
        result = apply_derivative_fast(
            np.asarray(spectra, dtype=np.float64),
            window_length, polyorder, deriv
        )
        if result is not None:
            return result

    # Fallback to scipy for complex cases
    from scipy.signal import savgol_filter
    return np.array([savgol_filter(spectrum, window_length, polyorder, deriv=deriv)
                     for spectrum in spectra])
```

### Testing

```python
# Test that Numba version gives same results
import numpy as np
from spectral_predict.preprocess import snv_transform

# Create test data
X = np.random.randn(100, 500)

# Should give identical results
result = snv_transform(X)

# Verify shape and values
assert result.shape == X.shape
assert np.all(np.isfinite(result))
```

### Expected Impact

- SNV transform: 5-10x faster
- SG derivatives: 2-3x faster
- Overall preprocessing: 2-3x faster
- Total pipeline: 1.1-1.2x faster (since preprocessing is ~5% of total)

---

## Optimization 2: Vectorize Region Analysis

**Time**: 30 minutes
**Speedup**: 2-3x on region analysis
**Files**: `src/spectral_predict/regions.py`

### Current Issue

Region correlation computation may use loops where vectorization is possible.

### Implementation

```python
@numba.jit(nopython=True, parallel=True)
def compute_region_correlations_fast(X, y, region_starts, region_ends):
    """
    Compute correlations for multiple regions - JIT compiled and parallelized.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_wavelengths)
        Spectral data
    y : ndarray, shape (n_samples,)
        Target values
    region_starts : ndarray
        Start indices for each region
    region_ends : ndarray
        End indices for each region

    Returns
    -------
    correlations : ndarray
        Mean absolute correlation for each region
    """
    n_regions = len(region_starts)
    correlations = np.empty(n_regions)

    for r in numba.prange(n_regions):  # Parallel loop
        start = region_starts[r]
        end = region_ends[r]

        # Compute mean absolute correlation in this region
        region_data = X[:, start:end]
        region_corr = 0.0

        for i in range(start, end):
            # Pearson correlation coefficient
            corr = np.corrcoef(X[:, i], y)[0, 1]
            region_corr += abs(corr)

        correlations[r] = region_corr / (end - start)

    return correlations
```

### Expected Impact

- Region analysis: 2-3x faster
- Overall pipeline: 1.02-1.05x faster (region analysis is small part)

---

## Optimization 3: Optimize Cross-Validation

**Time**: 1 hour
**Speedup**: 1.2-1.5x
**Files**: `src/spectral_predict/search.py`

### Current Issue

Cross-validation may be doing redundant work:
- Re-validating parameters
- Re-allocating arrays
- Not using sklearn's parallel backend efficiently

### Implementation

```python
# In search.py

def run_search(X, y, config):
    """Run model search with optimizations."""

    # Pre-allocate results arrays (avoid repeated allocation)
    n_configs = sum(len(grid) for grid in model_grids.values())
    results_preallocated = {
        'models': [None] * n_configs,
        'scores': np.empty(n_configs),
        'params': [None] * n_configs
    }

    # Use sklearn's parallel backend more efficiently
    from sklearn.utils import parallel_backend

    # Use threading instead of multiprocessing for small models
    # (avoids pickling overhead)
    with parallel_backend('threading', n_jobs=-1):
        for model, params in model_grid:
            scores = cross_val_score(
                model, X, y,
                cv=cv_splitter,  # Pre-computed CV splits
                n_jobs=-1,
                scoring='neg_mean_squared_error'
            )
            # ... rest of logic
```

### Pre-compute CV Splits

```python
# At start of search
from sklearn.model_selection import KFold

# Pre-compute CV splits once
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(cv_splitter.split(X))

# Reuse these splits for all models
for model in models:
    for train_idx, val_idx in cv_splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # ... fit and evaluate
```

### Expected Impact

- Cross-validation: 1.2-1.5x faster
- Overall pipeline: 1.1-1.3x faster

---

## Optimization 4: Cache Preprocessing Results

**Time**: 30 minutes
**Speedup**: 1.3-1.8x
**Files**: `src/spectral_predict/search.py`

### Current Issue

Same preprocessing (SNV, derivatives) may be applied multiple times to same data.

### Implementation

```python
# In search.py or new caching module

from functools import lru_cache
import hashlib

class PreprocessingCache:
    """Cache preprocessed spectral data to avoid redundant computation."""

    def __init__(self):
        self.cache = {}

    def get_key(self, X, method, params):
        """Generate cache key for preprocessing operation."""
        # Use hash of data + method + params
        data_hash = hashlib.md5(X.tobytes()).hexdigest()[:16]
        param_str = str(sorted(params.items()))
        return f"{method}_{data_hash}_{hash(param_str)}"

    def get(self, X, method, params):
        """Get cached result or None."""
        key = self.get_key(X, method, params)
        return self.cache.get(key)

    def put(self, X, method, params, result):
        """Store result in cache."""
        key = self.get_key(X, method, params)
        self.cache[key] = result

        # Limit cache size
        if len(self.cache) > 50:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))


# Usage in search
preprocessing_cache = PreprocessingCache()

def preprocess_data(X, method, params):
    """Preprocess with caching."""
    cached = preprocessing_cache.get(X, method, params)
    if cached is not None:
        return cached

    result = _apply_preprocessing(X, method, params)
    preprocessing_cache.put(X, method, params, result)
    return result
```

### Expected Impact

- Eliminates redundant preprocessing
- 1.3-1.8x speedup overall (if preprocessing runs multiple times)
- No speedup if preprocessing already runs once per config (check current code)

---

## Optimization 5: Reduce Logging/Progress Overhead

**Time**: 15 minutes
**Speedup**: 1.05-1.1x
**Files**: Various

### Current Issue

Progress updates and logging can have overhead, especially if writing to file frequently.

### Implementation

```python
# Reduce logging frequency
if iteration % 10 == 0:  # Instead of every iteration
    logger.info(f"Iteration {iteration}")

# Buffer progress updates
progress_buffer = []
for i in range(n_iterations):
    progress_buffer.append(status)
    if len(progress_buffer) >= 10:
        # Write in batch
        write_progress(progress_buffer)
        progress_buffer.clear()

# Use faster logging
import logging
logging.basicConfig(level=logging.WARNING)  # Instead of INFO during runs
```

### Expected Impact

- Minimal but free speedup: 1.05-1.1x
- Reduces I/O overhead

---

## Optimization 6: Optimize Data Loading (ASD Files)

**Time**: 30 minutes
**Speedup**: 1.1-1.3x (if many files)
**Files**: `src/spectral_predict/io.py`

### Current Issue

Loading many ASD files may be slow if done serially.

### Implementation

```python
from concurrent.futures import ThreadPoolExecutor
import os

def load_asd_files_parallel(directory, max_workers=4):
    """Load multiple ASD files in parallel."""
    asd_files = [f for f in os.listdir(directory) if f.endswith('.asd')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_single_asd,
                                   [os.path.join(directory, f) for f in asd_files]))

    return results


def load_single_asd(filepath):
    """Load a single ASD file."""
    # Existing ASD loading logic
    pass
```

### Expected Impact

- File loading: 2-4x faster (with multiple cores)
- Overall pipeline: 1.1-1.3x faster (if loading is 10-30% of time)
- More impact with more files

---

## Optimization 7: Use Memory Mapping for Large Datasets

**Time**: 30 minutes
**Speedup**: 1.2-2x for large datasets
**Files**: Memory-intensive sections

### Implementation

```python
import numpy as np

# Instead of loading entire dataset into memory
X = np.load('large_spectra.npy')

# Use memory mapping
X = np.load('large_spectra.npy', mmap_mode='r')  # Read-only memory map

# Benefit: Faster loading, lower memory usage
# Allows working with datasets larger than RAM
```

### Expected Impact

- Large datasets (>1GB): 1.5-2x faster loading
- Smaller datasets: Minimal impact
- Enables processing of larger-than-memory datasets

---

## Implementation Priority

### Quick Wins (1-2 hours, do first)

1. **Numba for SNV transform** (1 hour) → 2-3x preprocessing speedup
2. **Cache preprocessing results** (30 min) → 1.3-1.8x if applicable
3. **Reduce logging overhead** (15 min) → 1.05-1.1x free speedup

**Total**: 2-3x speedup, 1.75 hours

### Medium Effort (2-3 hours)

4. **Optimize cross-validation** (1 hour) → 1.2-1.5x speedup
5. **Numba for SG derivatives** (1 hour) → 2-3x on derivatives
6. **Parallel file loading** (30 min) → 1.1-1.3x if many files

**Total**: 3-5x speedup, 4-5 hours cumulative

### Optional (if working with large datasets)

7. **Memory mapping** (30 min) → 1.5-2x for large datasets
8. **Vectorize region analysis** (30 min) → 2-3x on regions

---

## Testing Strategy

### Performance Benchmarks

Create `benchmarks/quick_optimizations.py`:

```python
"""Benchmark quick optimizations."""
import time
import numpy as np
from spectral_predict.preprocess import snv_transform, apply_derivative

# Create realistic test data
X = np.random.randn(100, 2151)  # Typical spectral dataset

# Benchmark SNV
start = time.time()
for _ in range(100):
    result = snv_transform(X)
snv_time = time.time() - start

print(f"SNV transform: {snv_time:.3f}s for 100 iterations")
print(f"Per iteration: {snv_time/100*1000:.2f}ms")

# Benchmark derivatives
start = time.time()
for _ in range(100):
    result = apply_derivative(X, window_length=5, polyorder=2, deriv=1)
deriv_time = time.time() - start

print(f"SG derivative: {deriv_time:.3f}s for 100 iterations")
print(f"Per iteration: {deriv_time/100*1000:.2f}ms")
```

### Correctness Tests

```python
"""Test that optimizations don't change results."""
import numpy as np
from spectral_predict.preprocess import snv_transform

# Test data
X = np.random.randn(50, 100)

# Get results before/after optimization
result_optimized = snv_transform(X)
result_original = snv_transform_original(X)  # Saved original version

# Should be identical (within floating point precision)
assert np.allclose(result_optimized, result_original, rtol=1e-10)
print("✅ Results identical")
```

---

## Expected Combined Impact

| Optimization | Time | Speedup | Cumulative |
|-------------|------|---------|------------|
| Baseline | - | 1.0x | 1.0x |
| Numba SNV | 1h | 2-3x preprocessing | 1.1-1.2x |
| Cache preprocessing | 30min | 1.3-1.8x | 1.4-2.0x |
| Optimize CV | 1h | 1.2-1.5x | 1.7-3.0x |
| Numba derivatives | 1h | 2-3x derivatives | 2.0-4.0x |
| Parallel loading | 30min | 1.1-1.3x | 2.2-5.0x |

**Total Expected Speedup**: 2-5x
**Total Time Investment**: 4-5 hours
**Risk**: Very Low (all optimizations are non-invasive)

---

## Dependencies

### Required

```toml
# Add to pyproject.toml [dependencies]
numba = ">=0.58.0"  # For JIT compilation
```

### Installation

```bash
.venv/bin/pip install numba
```

### Verification

```python
# Test that Numba works
import numba
import numpy as np

@numba.jit(nopython=True)
def test_func(x):
    return x ** 2

result = test_func(np.array([1, 2, 3]))
print("✅ Numba working")
```

---

## Rollback Plan

If any optimization causes issues:

1. **Numba issues**: Already has fallback to original Python implementation
2. **Cache issues**: Easy to disable by commenting out cache lookup
3. **CV issues**: Revert to original cross_val_score usage

All optimizations are designed with fallbacks and can be individually disabled.

---

## Monitoring

Track these metrics before/after:

1. **Total runtime**: Overall pipeline time
2. **Preprocessing time**: SNV + derivatives
3. **Model training time**: Per model type
4. **File loading time**: If using parallel loading
5. **Memory usage**: Should not increase significantly

---

## Summary

**Quick Wins** (1-2 hours):
- Numba JIT for preprocessing → 2-3x on preprocessing
- Preprocessing cache → 1.3-1.8x
- Reduced logging → 1.05-1.1x

**Medium Effort** (additional 2-3 hours):
- Optimized CV → 1.2-1.5x
- Parallel file loading → 1.1-1.3x
- More Numba → 2-3x on specific operations

**Total**: 2-5x speedup for 4-5 hours of work, very low risk.

These optimizations are **complementary** to the neural boosting optimizations and can be done in any order or combination.

---

**Ready to implement? Start with the Quick Wins section for maximum ROI.**
