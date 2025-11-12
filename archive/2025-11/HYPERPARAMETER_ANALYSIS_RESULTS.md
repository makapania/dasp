# Hyperparameter Analysis Results for NIR Spectroscopy
**Empirical testing on bone collagen dataset (49 samples, 2151 wavelengths)**

Date: 2025-11-09

---

## Executive Summary

Comprehensive empirical testing reveals **significant opportunities for optimization**:

### Key Findings

1. **Random Forest n_estimators=100 is optimal** - Going to 200 or 500 provides **no benefit** but increases time 2-5x
2. **Random Forest max_depth doesn't matter** - All depths (15, 30, 50, None) give **identical R²**
3. **NeuralBoosted learning_rate=0.3 is best** - **MISSING from current defaults** [0.1, 0.2]!
4. **Ridge alpha > 10 can be skipped** - Performance degrades significantly
5. **Lasso alpha ≥ 1.0 completely fails** - Can skip 4 of 6 current test values

---

## Detailed Results

### 1. Random Forest

**Dataset**: 49 samples, 2151 wavelengths, 5-fold CV

| n_estimators | max_depth | R² Score | Time (s) | Recommendation |
|--------------|-----------|----------|----------|----------------|
| 50           | any       | 0.6743   | 1.3      | Too few trees  |
| **100**      | **any**   | **0.6859**   | **2.1**      | **OPTIMAL**    |
| 200          | any       | 0.6840   | 3.7      | No benefit     |
| 500          | any       | 0.6784   | 10.3     | No benefit     |
| 1000         | any       | 0.6823   | 19.8     | No benefit     |

**Key Insight**: max_depth has **zero impact** on performance (all values give R²=0.686). Use the simplest (15 or 30).

**RECOMMENDATION**:
- **n_estimators**: Keep [100] only, remove [200, 500]
- **max_depth**: Keep [15] or [30] only (performance identical, so choose fastest)
- **Speedup**: 75% faster grid search (4 configs → 1 config)

---

### 2. NeuralBoosted

**Dataset**: 49 samples, 2151 wavelengths, 5-fold CV

| n_estimators | learning_rate | R² Score | Time (s) | Recommendation |
|--------------|---------------|----------|----------|----------------|
| 25           | 0.3           | **0.8405**   | 94       | Best combo     |
| 50           | 0.3           | 0.8395   | 78       | Nearly as good |
| 100          | 0.3           | 0.8395   | 63       | Optimal speed  |
| 200-300      | 0.3           | 0.8395   | 54-60    | No benefit     |
| 25-300       | 0.1           | 0.8093   | 45-166   | Suboptimal     |
| 25-300       | 0.05-0.2      | 0.76-0.84| varies   | Mixed results  |

**CRITICAL FINDING**: **learning_rate=0.3 performs best (R²=0.84 vs 0.68 for Random Forest)**
- Current GUI defaults [0.1, 0.2] **miss the optimal value**!
- n_estimators beyond 50 provides minimal benefit

**RECOMMENDATION**:
- **learning_rate**: **ADD 0.3** to defaults, keep [0.1, 0.2, 0.3]
- **n_estimators**: Keep [50, 100], remove higher values
- **Impact**: NeuralBoosted now achieves R²=0.84 (23% better than Random Forest!)

---

### 3. Ridge Regression

**Dataset**: 49 samples, 2151 wavelengths, 5-fold CV

| Alpha    | R² Score | Time (s) | Recommendation |
|----------|----------|----------|----------------|
| **0.001**    | **0.8244**   | 0.15     | **Best**           |
| 0.01     | 0.8191   | 0.15     | Good           |
| 0.1      | 0.7470   | 0.14     | OK             |
| 1.0      | 0.7182   | 0.14     | Suboptimal     |
| 10.0     | 0.6820   | 0.14     | Poor           |
| 100.0    | 0.6059   | 0.17     | **Skip**       |
| 1000.0   | 0.3040   | 0.15     | **Skip**       |

**Key Insight**: Performance degrades sharply above alpha=10. High regularization hurts NIR data.

**RECOMMENDATION**:
- **Keep**: [0.001, 0.01, 0.1, 1.0, 10.0]
- **Remove**: [100.0, 1000.0]
- **Speedup**: 29% faster (7 → 5 values)

---

### 4. Lasso Regression

**Dataset**: 49 samples, 2151 wavelengths, 5-fold CV

| Alpha    | R² Score | Time (s) | Recommendation |
|----------|----------|----------|----------------|
| **0.001**    | **0.8167**   | 4.2      | **Best**           |
| 0.01     | 0.7194   | 3.1      | OK             |
| 0.1      | 0.6933   | 3.0      | Suboptimal     |
| 1.0      | -0.1126  | 0.16     | **FAILS**      |
| 10.0     | -0.1126  | 0.14     | **FAILS**      |
| 100.0    | -0.1126  | 0.16     | **FAILS**      |

**CRITICAL**: Lasso **completely fails** at alpha ≥ 1.0 (negative R²)!

**RECOMMENDATION**:
- **Keep**: [0.001, 0.01, 0.1]
- **Remove**: [1.0, 10.0, 100.0]
- **Speedup**: 50% faster (6 → 3 values)

---

## Recommended Configuration for Tab 3

### Quick Analysis Mode (Fast, Good Performance)

**Random Forest**:
- n_estimators: [100] only
- max_depth: [30] only

**NeuralBoosted**:
- n_estimators: [50, 100]
- learning_rate: [**0.3**] only (add this!)

**Ridge**:
- alpha: [0.001, 0.01, 0.1]

**Lasso**:
- alpha: [0.001, 0.01, 0.1]

**Estimated Time Savings**: 60-70% faster than current defaults

---

### Comprehensive Mode (Maximum Coverage)

**Random Forest**:
- n_estimators: [100, 200]
- max_depth: [30] (more doesn't help)

**NeuralBoosted**:
- n_estimators: [50, 100, 200]
- learning_rate: [0.1, 0.2, **0.3**]

**Ridge**:
- alpha: [0.001, 0.01, 0.1, 1.0, 10.0]

**Lasso**:
- alpha: [0.001, 0.01, 0.1]

**Estimated Time Savings**: 40-50% faster than current defaults

---

## Performance Summary

**Best performing models on this NIR dataset**:

1. **NeuralBoosted** (lr=0.3, n=25): R²=0.8405 [94s]  ← **BEST**
2. **Ridge** (alpha=0.001): R²=0.8244 [0.15s]        ← **FASTEST**
3. **Lasso** (alpha=0.001): R²=0.8167 [4.2s]
4. **Random Forest** (n=100): R²=0.6859 [2.1s]

**Key Takeaway**: Linear models (Ridge/Lasso/NeuralBoosted) **outperform** Random Forest for this NIR composition prediction task.

---

## Action Items for GUI Updates

### CRITICAL (Impacting Performance):

1. **ADD learning_rate=0.3 to NeuralBoosted defaults** (currently missing optimal value)
   - File: `spectral_predict_gui_optimized.py` (Tab 3 NeuralBoosted section)
   - Change: Add checkbox for lr=0.3 and check it by default

### HIGH PRIORITY (Speed Improvements):

2. **Reduce Random Forest defaults**
   - n_estimators: Change default from [200, 500] to [100, 200]
   - max_depth: Keep only [30] or [None, 30]

3. **Reduce Lasso alpha range**
   - Remove: [1.0, 10.0, 100.0] (these fail completely)
   - Keep: [0.001, 0.01, 0.1]

### MEDIUM PRIORITY (Marginal Gains):

4. **Reduce Ridge alpha range**
   - Remove: [100.0, 1000.0]
   - Keep: [0.001, 0.01, 0.1, 1.0, 10.0]

---

## Appendix: Analysis Details

### Dataset Characteristics
- **Source**: Bone collagen NIR spectroscopy
- **Samples**: 49
- **Wavelengths**: 2151 (350-2500 nm)
- **Target**: %Collagen (0.9% - 22.1%)
- **Task**: Regression (continuous prediction)

### Methodology
- **Cross-validation**: 5-fold KFold
- **Metrics**: R² score (primary), RMSE (secondary)
- **Random seed**: 42 (reproducible)
- **Hardware**: Windows 11, Python 3.14

### Result Files Generated
- `results_random_forest.csv` (20 configurations tested)
- `results_neuralboosted.csv` (25 configurations tested)
- `results_ridge_lasso.csv` (13 configurations tested)
- `results_preprocessing.csv` (5 methods tested)
- `results_window_sizes.csv` (8 configurations tested)
- `results_variable_selection.csv` (7 variable counts tested)

### Scripts
- `analyze_hyperparameters.py` - Main analysis script
- Dataset: `example/` folder (49 ASD files + BoneCollagen.csv)

---

## Next Steps

1. Update GUI to add learning_rate=0.3 for NeuralBoosted
2. Consider adjusting default checkboxes based on recommendations
3. Run validation on other NIR datasets to confirm findings
4. Consider adding user presets: "Quick", "Balanced", "Comprehensive"

---

**Generated**: 2025-11-09
**Analysis Time**: ~10 minutes (full parameter sweep)
**Key Insight**: Current defaults miss optimal NeuralBoosted learning_rate and test many unnecessary Random Forest configurations

