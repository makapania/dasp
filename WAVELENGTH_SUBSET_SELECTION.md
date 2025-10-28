# Wavelength Subset Selection Methodology

## Overview

The spectral analysis pipeline includes sophisticated wavelength subset selection to identify the most informative spectral features. This document explains exactly how different subsets of wavelengths are chosen and tested.

---

## Types of Wavelength Subsets

The system evaluates **three types** of wavelength subsets:

1. **Full Spectrum** - All available wavelengths
2. **Feature Importance-Based Subsets** - Top N most important variables
3. **Spectral Region-Based Subsets** - Wavelength ranges with high correlation to target

---

## 1. Full Spectrum Analysis

### Description
Uses **all available wavelengths** from the input data without any selection.

### When Applied
- Every model configuration is first tested with the full spectrum
- Serves as the baseline for comparison

### Subset Tag
`full`

### Example
If your data has wavelengths from 350-2500nm with 2151 channels, the full spectrum uses all 2151 wavelengths.

---

## 2. Feature Importance-Based Subsets

### Overview
For models that can compute feature importances (PLS, PLS-DA, RandomForest, MLP), the system:
1. Fits the model on the full spectrum
2. Extracts feature importance scores
3. Selects the top N most important wavelengths
4. Re-tests the model using only those wavelengths

### Models Supporting Importance Extraction

#### PLS / PLS-DA
**Method:** Variable Importance in Projection (VIP) scores

VIP scores measure how much each wavelength contributes to the PLS model's predictive power:

```
VIP_i = sqrt(p × Σ(w_i,j² × SSY_j) / SSY_total)

Where:
- p = number of wavelengths
- w_i,j = weight of variable i in component j
- SSY_j = sum of squares explained by component j
- SSY_total = total explained variance
```

**Interpretation:**
- VIP > 1.0: Very important variable
- VIP > 0.8: Important variable
- VIP < 0.5: Less important variable

**File Location:** `src/spectral_predict/models.py:129-168` (function `compute_vip`)

---

#### RandomForest
**Method:** Gini importance (built-in sklearn feature importances)

Measures how much each wavelength decreases impurity (Gini index) across all trees:
- Higher values = more important for making accurate splits
- Normalized so all importances sum to 1.0

**File Location:** `src/spectral_predict/models.py:195-197`

---

#### MLP (Neural Network)
**Method:** Average absolute weight of first hidden layer

```
importance_i = mean(|W_i,h|) for all hidden units h

Where:
- W_i,h = weight from input i to hidden unit h
```

**Interpretation:**
- Higher average weight magnitude = input is more influential
- Simple heuristic but effective for basic importance ranking

**File Location:** `src/spectral_predict/models.py:199-203`

---

### Variable Selection Grid

The system tests **seven different subset sizes** using a logarithmic spacing:

| Subset Size | Subset Tag | Description |
|-------------|-----------|-------------|
| 10 | `top10` | Ultra-sparse: Only 10 most important wavelengths |
| 20 | `top20` | Very sparse: 20 most important wavelengths |
| 50 | `top50` | Sparse: 50 most important wavelengths |
| 100 | `top100` | Moderately sparse: 100 most important wavelengths |
| 250 | `top250` | Moderately dense: 250 most important wavelengths |
| 500 | `top500` | Dense: 500 most important wavelengths |
| 1000 | `top1000` | Very dense: 1000 most important wavelengths |

**Note:** Only subset sizes smaller than the total number of features are tested.

**File Location:** `src/spectral_predict/search.py:224-228`

---

### Selection Algorithm

```python
# Step 1: Fit model on full spectrum
pipe.fit(X_full, y)

# Step 2: Extract feature importances
importances = get_feature_importances(fitted_model, model_name, X, y)

# Step 3: For each subset size (10, 20, 50, 100, 250, 500, 1000)
for n_top in [10, 20, 50, 100, 250, 500, 1000]:
    # Get indices of top N most important features
    top_indices = np.argsort(importances)[-n_top:][::-1]

    # Create subset of X using only those wavelengths
    X_subset = X[:, top_indices]

    # Refit and evaluate model on subset
    evaluate_model(X_subset, y)
```

**File Location:** `src/spectral_predict/search.py:189-248`

---

### Why This Matters

**Benefits of Testing Multiple Subset Sizes:**

1. **Reduced Noise:** Fewer wavelengths can sometimes improve model performance by removing noisy features
2. **Faster Predictions:** Models with fewer features are faster to evaluate
3. **Simpler Models:** Easier to interpret and deploy
4. **Overfitting Prevention:** Limiting features can reduce overfitting, especially with small datasets

**Example Results:**
- Full spectrum (2151 vars): R² = 0.92, RMSE = 0.085
- Top 250 vars: R² = 0.94, RMSE = 0.078 ← **Better performance with fewer features!**
- Top 50 vars: R² = 0.88, RMSE = 0.095 ← Too sparse

---

## 3. Spectral Region-Based Subsets

### Overview

Instead of selecting individual wavelengths, this approach identifies **spectral regions** (contiguous wavelength ranges) that are highly correlated with the target variable.

**Key Insight:** Important spectral features often occur in clusters (e.g., C-H stretch region ~2800-3000nm, O-H stretch ~1400nm).

---

### Region Discovery Algorithm

#### Step 1: Divide Spectrum into Overlapping Windows

```python
region_size = 50 nm      # Size of each window
overlap = 25 nm          # Overlap between adjacent windows

# Example for 350-2500nm range:
# Region 1: 350-400nm
# Region 2: 375-425nm (overlaps with Region 1 by 25nm)
# Region 3: 400-450nm
# ... and so on
```

**File Location:** `src/spectral_predict/regions.py:8-76` (function `compute_region_correlations`)

---

#### Step 2: Compute Correlation for Each Region

For each wavelength window:
1. Extract all wavelengths in that range
2. Compute Pearson correlation between each wavelength and the target variable
3. Calculate region statistics:
   - **mean_corr**: Average absolute correlation in this region
   - **max_corr**: Maximum absolute correlation in this region
   - **n_features**: Number of wavelengths in this region

```python
for each region:
    correlations = []
    for wavelength in region:
        corr = pearson_correlation(X[:, wavelength], y)
        correlations.append(abs(corr))

    mean_corr = np.mean(correlations)
    max_corr = np.max(correlations)
```

**File Location:** `src/spectral_predict/regions.py:42-76`

---

#### Step 3: Rank Regions by Correlation

Regions are sorted by `mean_corr` (average absolute correlation) in descending order:

```
Rank 1: 1450-1500nm (mean_corr = 0.82, max_corr = 0.91)
Rank 2: 2200-2250nm (mean_corr = 0.78, max_corr = 0.85)
Rank 3: 950-1000nm  (mean_corr = 0.71, max_corr = 0.77)
...
```

**File Location:** `src/spectral_predict/regions.py:79-98` (function `get_top_regions`)

---

#### Step 4: Create Region-Based Subsets

The system creates **multiple subset configurations** from the top-ranked regions:

| Subset Type | Description | Subset Tag | Typical Size |
|-------------|-------------|-----------|--------------|
| Individual Top Regions | Each of the top 3 regions separately | `region1`, `region2`, `region3` | ~50-100 vars each |
| Top 2 Regions Combined | Union of top 2 regions | `top2regions` | ~100-200 vars |
| Top 3 Regions Combined | Union of top 3 regions | `top3regions` | ~150-300 vars |
| Top 5 Regions Combined | Union of top 5 regions | `top5regions` | ~250-500 vars |

**File Location:** `src/spectral_predict/regions.py:129-207` (function `create_region_subsets`)

---

### When Region-Based Selection is Applied

**Applied ONLY for:**
- Non-derivative preprocessing (`raw` or `snv`)
- Models that support importance extraction (PLS, PLS-DA, RandomForest)

**NOT applied for:**
- Derivative preprocessing (1st or 2nd derivative)
  - Reason: Derivatives already emphasize spectral features, making region analysis redundant

**File Location:** `src/spectral_predict/search.py:250-268`

---

### Example Region Analysis

```
======================================================================
Top Spectral Regions (by correlation with target)
======================================================================

Rank   Region (nm)          Mean |r|     Max |r|      N vars
----------------------------------------------------------------------
1      1450-1500            0.8234       0.9102       25
2      2200-2250            0.7812       0.8543       24
3      950-1000             0.7123       0.7734       23
4      1350-1400            0.6891       0.7421       26
5      2100-2150            0.6543       0.7201       25

Note: Regions with high correlations may indicate important
spectral features related to the target variable.
======================================================================
```

This output is generated by `format_region_report()` in `regions.py:210-250`.

---

## Complete Workflow: How Subsets are Chosen

### For Each Model Configuration:

```
1. Test Full Spectrum
   ↓

2. IF model supports importances (PLS, PLS-DA, RF, MLP):
   2a. Fit model on full spectrum
   2b. Extract feature importances
   2c. Test 7 importance-based subsets (top10, top20, top50, top100, top250, top500, top1000)

3. IF preprocessing is non-derivative (raw/snv):
   3a. Compute region correlations
   3b. Identify top 5 regions
   3c. Test 3 individual regions (region1, region2, region3)
   3d. Test combined regions (top2regions, top3regions, top5regions)
```

**Result:** Each model configuration can generate up to **18 result rows**:
- 1 full spectrum
- Up to 7 importance-based subsets
- Up to 10 region-based subsets (if applicable)

---

## Code Organization

| File | Function | Purpose |
|------|----------|---------|
| `search.py:189-268` | Main subset loop | Orchestrates all subset selection |
| `models.py:129-168` | `compute_vip()` | Computes VIP scores for PLS |
| `models.py:171-206` | `get_feature_importances()` | Extracts importances from any model |
| `regions.py:8-76` | `compute_region_correlations()` | Divides spectrum into windows |
| `regions.py:79-98` | `get_top_regions()` | Ranks regions by correlation |
| `regions.py:129-207` | `create_region_subsets()` | Creates region-based subsets |

---

## Output in Results CSV

Each row in the results CSV represents one tested configuration. The `SubsetTag` column indicates which subset was used:

```csv
Model,Preprocess,n_vars,full_vars,SubsetTag,RMSE,R2,top_vars
PLS,snv,2151,2151,full,0.085,0.92,"1450.0,1455.0,2250.0,..."
PLS,snv,250,2151,top250,0.078,0.94,"1450.0,1455.0,2250.0,..."
PLS,snv,50,2151,top50,0.095,0.88,"1450.0,2250.0,1455.0,..."
PLS,snv,89,2151,region1,0.082,0.93,"1450.0,1455.0,1460.0,..."
PLS,snv,167,2151,top2regions,0.080,0.935,"1450.0,2200.0,..."
```

**New Column: `top_vars`**
- Shows the top 30 most important wavelengths for each model (in order of importance)
- Format: Comma-separated list of wavelength values (e.g., "1450.0,2250.0,1455.0,...")
- Allows you to see which specific wavelengths are driving each model's predictions

---

## Performance vs. Complexity Trade-off

The ranking algorithm balances:
- **Performance**: Lower RMSE / Higher R² is better
- **Complexity**: Fewer variables is better (simpler, faster models)

**Composite Score Formula:**
```
composite_score = performance_score + λ × complexity_penalty

Where:
- performance_score = 0.5×z(RMSE) - 0.5×z(R²)
- complexity_penalty = LV_penalty + vars_penalty + sparsity_penalty
- λ = complexity penalty weight (default: 0.15)
```

This means a model with:
- R² = 0.94, 250 variables might rank **higher** than
- R² = 0.92, 2151 variables

Because the performance gain (R² +0.02) outweighs the simpler model benefit.

**File Location:** `src/spectral_predict/scoring.py:7-112`

---

## Summary

### Quick Reference: What Gets Tested

| Condition | Subsets Tested |
|-----------|---------------|
| **All models** | Full spectrum |
| **PLS/PLS-DA/RF/MLP only** | + Top 10, 20, 50, 100, 250, 500, 1000 variables |
| **Non-derivative preprocessing** | + Individual regions 1-3, Top 2/3/5 regions combined |

### Key Parameters You Can Adjust

In the GUI or CLI:

1. **Complexity Penalty (λ)**: Default 0.15
   - Higher → Prefer simpler models
   - Lower → Prefer better performance

2. **Max PLS Components**: Default 24
   - Controls maximum number of PLS latent variables

3. **CV Folds**: Default 5
   - Number of cross-validation splits

---

## References

**VIP Scores:**
- Wold, S. et al. (2001). "PLS-regression: a basic tool of chemometrics." *Chemometrics and Intelligent Laboratory Systems* 58(2): 109-130.

**Feature Selection in Spectroscopy:**
- Mehmood, T. et al. (2012). "A review of variable selection methods in Partial Least Squares Regression." *Chemometrics and Intelligent Laboratory Systems* 118: 62-69.

---

*Last Updated: October 27, 2025*
*Version: 2.0*
