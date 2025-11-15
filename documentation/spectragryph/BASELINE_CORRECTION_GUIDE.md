# Baseline Correction Guide

## Overview

Baseline correction is one of the most important preprocessing steps in spectroscopy. It removes baseline drift, background signals, and other systematic variations that can interfere with quantitative analysis.

The `spectral_predict.baseline` module provides two industry-standard baseline correction methods:

1. **Asymmetric Least Squares (ALS)** - The gold standard for baseline correction
2. **Polynomial Baseline** - Simpler alternative for gentle baselines

## Installation

The baseline correction module is included in DASP. No additional dependencies required (uses existing scipy).

## Usage

### Method 1: Asymmetric Least Squares (ALS) - Recommended

```python
from spectral_predict.baseline import BaselineALS
import pandas as pd

# Load your spectral data (DataFrame with samples as rows, wavelengths as columns)
X = pd.read_csv('spectra.csv', index_col=0)

# Create baseline corrector
baseline = BaselineALS(
    lambda_=1e5,  # Smoothness (higher = smoother)
    p=0.001,      # Asymmetry (lower = stays under peaks)
    niter=10      # Number of iterations
)

# Apply correction
X_corrected = baseline.fit_transform(X)
```

### Method 2: Polynomial Baseline - Simpler Alternative

```python
from spectral_predict.baseline import BaselinePolynomial

# Create polynomial baseline corrector
baseline = BaselinePolynomial(degree=3)  # Cubic polynomial

# Apply correction
X_corrected = baseline.fit_transform(X)
```

## Parameter Tuning

### ALS Parameters

**`lambda_` (smoothness):**
- `1e2 - 1e4`: Less smooth, follows signal closely
- `1e5 - 1e6`: **Recommended for most spectra**
- `1e7 - 1e9`: Very smooth, good for broad baselines

**`p` (asymmetry):**
- `0.001 - 0.01`: **Strong asymmetry (recommended)** - baseline stays under peaks
- `0.1`: Less asymmetry - baseline can rise above signal

**`niter` (iterations):**
- `10`: **Standard** - good for most cases
- `20`: More iterations for complex baselines
- `5`: Faster for simple baselines

### Polynomial Parameters

**`degree`:**
- `1`: Linear baseline (straight line)
- `2-3`: **Gentle curved baseline (most common)**
- `4-5`: Complex curved baseline (use sparingly)

## Visualization

```python
import matplotlib.pyplot as plt

# Before and after comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Original spectrum
X.iloc[0].plot(ax=ax1, title='Before Baseline Correction', color='blue')
ax1.set_xlabel('Wavelength')
ax1.set_ylabel('Intensity')

# Corrected spectrum
X_corrected.iloc[0].plot(ax=ax2, title='After Baseline Correction (ALS)', color='green')
ax2.set_xlabel('Wavelength')
ax2.set_ylabel('Intensity')

plt.tight_layout()
plt.show()
```

## Integration with DASP Workflow

### Option 1: Standalone Preprocessing (Current)

```python
from spectral_predict.io import read_csv_spectra
from spectral_predict.baseline import BaselineALS
from spectral_predict.search import run_search

# Load data
X, metadata = read_csv_spectra('spectra.csv')
y = metadata['target']

# Apply baseline correction FIRST
baseline = BaselineALS(lambda_=1e5, p=0.001)
X_corrected = baseline.fit_transform(X)

# Then run DASP analysis on corrected data
results = run_search(
    X_corrected, y,
    task_type='regression',
    preprocessing_methods={'raw': True, 'snv': True},
    tier='standard'
)
```

### Option 2: As Part of Sklearn Pipeline (Advanced)

```python
from sklearn.pipeline import Pipeline
from spectral_predict.baseline import BaselineALS
from spectral_predict.preprocess import SNV
from sklearn.cross_decomposition import PLSRegression

# Create custom pipeline
pipe = Pipeline([
    ('baseline', BaselineALS(lambda_=1e5, p=0.001)),
    ('snv', SNV()),
    ('pls', PLSRegression(n_components=10))
])

# Fit and predict
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

## When to Use Baseline Correction

### Use ALS When:
- ✅ Spectra have sloping baselines
- ✅ Background fluorescence is present
- ✅ Instrument drift affects measurements
- ✅ Broad absorption features need removal
- ✅ You want industry-standard method

### Use Polynomial When:
- ✅ Baseline is gently curved
- ✅ You need fast computation
- ✅ Baseline shape is simple

### Don't Use Baseline Correction When:
- ❌ Absolute intensity values matter (it removes DC offset)
- ❌ Baseline contains informative features
- ❌ Data already baseline-corrected by instrument

## Comparison: ALS vs. Polynomial

| Feature | ALS | Polynomial |
|---------|-----|------------|
| Accuracy | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| Flexibility | ⭐⭐⭐⭐⭐ Very flexible | ⭐⭐ Limited |
| Speed | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Fast |
| Complexity | ⭐⭐⭐⭐ More parameters | ⭐⭐ Simple |
| Industry Standard | ✅ Yes | ✅ Yes (simpler cases) |

## Troubleshooting

**Problem:** Baseline still visible after correction

**Solution:** Increase `lambda_` (try 1e6, 1e7) or increase `degree` for polynomial

**Problem:** Peaks are being removed

**Solution:** Decrease `lambda_` (try 1e4, 1e3) or decrease `p` for more asymmetry

**Problem:** Processing is slow

**Solution:** Reduce `niter` (try 5), or use polynomial method instead

## References

1. Eilers, P. H. C., & Boelens, H. F. M. (2005). Baseline correction with asymmetric least squares smoothing. *Leiden University Medical Centre Report*, 1(1), 5.

2. Zhang, Z. M., Chen, S., & Liang, Y. Z. (2010). Baseline correction using adaptive iteratively reweighted penalized least squares. *Analyst*, 135(5), 1138-1146.

## Next Steps

For fuller integration into DASP's automated search pipeline, see the roadmap in `documentation/spectragryph/comparison.md`.

Current status: **Baseline correction is available as a standalone preprocessing step.**

Future plans: **Integration into GUI and automatic search pipeline.**
