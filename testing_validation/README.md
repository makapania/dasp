# DASP Validation Testing Infrastructure

This directory contains a comprehensive testing framework for validating DASP against R and Python implementations of standard machine learning algorithms for spectral analysis.

## Directory Structure

```
testing_validation/
├── README.md                          # This file
├── prepare_datasets.py                # Prepare train/test splits
├── export_spectra_for_r.py           # Export spectral data to CSV for R
│
├── data/                             # Train/test split metadata
│   ├── regression_train.csv          # 36 samples
│   ├── regression_test.csv           # 13 samples
│   ├── binary_train.csv              # Binary classification (High/Low collagen)
│   ├── binary_test.csv
│   ├── 4class_train.csv              # 4-class (A, F, G, H)
│   ├── 4class_test.csv
│   ├── 7class_train.csv              # 7-class (A, C, F, G, H, I, J)
│   ├── 7class_test.csv
│   └── metadata.json                 # Dataset statistics
│
├── r_data/                           # Spectral data in R-compatible format
│   ├── export_metadata.json
│   ├── regression/
│   │   ├── X_train.csv               # 36 × 2151 spectral matrix
│   │   ├── X_test.csv                # 13 × 2151 spectral matrix
│   │   ├── y_train.csv               # %Collagen values
│   │   ├── y_test.csv
│   │   └── wavelengths.csv           # 2151 wavelengths (350-2500 nm)
│   ├── binary/                       # Same structure
│   ├── 4class/                       # Same structure
│   └── 7class/                       # Same structure
│
├── r_scripts/                        # R validation scripts
│   ├── install_packages.R            # Install required R packages
│   ├── regression_models.R           # R regression testing
│   └── classification_models.R       # R classification testing
│
├── results/                          # Output directory
│   ├── dataset_preparation_summary.txt
│   ├── r_regression/                 # R regression results (JSON)
│   └── r_classification/             # R classification results (JSON)
│       ├── binary/
│       └── 4class/
│
└── reports/                          # Final reports (to be generated)
```

## Quick Start

### Step 1: Prepare Datasets (COMPLETED)

The datasets have been prepared with stratified train/test splits (75/25) for all task types.

```bash
python prepare_datasets.py
```

**Output:**
- 4 task types: regression, binary, 4class, 7class
- Train/test splits with fixed random seed (42)
- Metadata with class distributions

### Step 2: Export Spectral Data for R (COMPLETED)

Spectral data has been exported from ASD files to CSV format for R compatibility.

```bash
python export_spectra_for_r.py
```

**Output:**
- Spectral matrices (samples × wavelengths) for each task
- Reference values (y_train.csv, y_test.csv)
- Wavelength values (350-2500 nm, 2151 wavelengths)

### Step 3: Install R Packages

Install all required R packages for model fitting and comparison.

```bash
cd r_scripts
Rscript install_packages.R
```

**Required packages:**
- `pls` - PLS regression
- `glmnet` - Ridge, Lasso, ElasticNet
- `randomForest` - Random Forest
- `xgboost` - XGBoost
- `lightgbm` - LightGBM (optional)
- `prospectr` - Spectral preprocessing (SNV, Savitzky-Golay)
- `caret` - ML utilities
- `jsonlite` - JSON export
- `pROC` - ROC-AUC calculation

### Step 4: Run R Regression Testing

Test all regression models with parameters equivalent to DASP defaults.

```bash
cd r_scripts
Rscript regression_models.R
```

**Models tested:**
- PLS: 12 configurations (n_components: 2-50)
- Ridge: 5 configurations (lambda: 0.001-10.0)
- Lasso: 4 configurations (lambda: 0.001-1.0)
- ElasticNet: 20 configurations (lambda × l1_ratio grid)
- Random Forest: 6 configurations (ntree × max_depth)
- XGBoost: 8 configurations (nrounds × eta × max_depth)

**Output:**
- JSON files with predictions and metrics (RMSE, R², MAE)
- `all_models_summary.json` with all results

### Step 5: Run R Classification Testing

Test classification models on binary and multi-class tasks.

```bash
cd r_scripts
Rscript classification_models.R
```

**Models tested:**
- PLS-DA: 8 configurations
- Random Forest: 6 configurations
- XGBoost: 8 configurations

**Tasks:**
- Binary: High (>10%) vs. Low (≤10%) collagen
- 4-class: Categories A, F, G, H

**Output:**
- JSON files with predictions and metrics (Accuracy, AUC, F1)
- Separate results for each task

### Step 6: Run DASP Comparison (NEXT STEP)

Create Python scripts to run identical tests with DASP and compare results.

**TODO:**
- Create `dasp_regression.py` - Run DASP regression with same parameters
- Create `dasp_classification.py` - Run DASP classification
- Create `compare_results.py` - Compare DASP vs. R predictions

### Step 7: Generate Comparison Reports (NEXT STEP)

Create comprehensive reports comparing DASP and R implementations.

**TODO:**
- Scatter plots of predictions (DASP vs. R)
- Metric comparison tables
- Statistical tests (paired t-tests)
- Parameter equivalence documentation

## Dataset Details

### Dataset 1: Bone Collagen

- **Total samples:** 49 (37 used in testing)
- **Spectral range:** 350-2500 nm (VIS-NIR)
- **Wavelengths:** 2,151 channels
- **Target:** %Collagen (continuous: 0.9% - 22.1%)
- **Categories:** 7 sample groups (A, C, F, G, H, I, J)
- **Use:** Regression + classification testing

### Dataset 2: Enamel d13C (NEW!)

- **Total samples:** 140 (matched spectra + reference)
- **Spectral range:** 350-2500 nm (NIR)
- **Wavelengths:** 2,151 channels
- **Target:** d13C (carbon isotope ratio: -26.4‰ to -14.3‰)
- **Use:** Robust regression testing (larger sample size)
- **Expected R²:** 0.75-0.90 (previous DASP run: 0.845)

**See `DATASETS_SUMMARY.md` for detailed information on all datasets.**

### Train/Test Splits

| Task | Dataset | Train | Test | Classes/Range |
|------|---------|-------|------|---------------|
| **Regression** | Bone Collagen | 36 | 13 | 0.9% - 22.1% collagen |
| **Regression** | **Enamel d13C** | **105** | **35** | **-26.4‰ to -14.3‰** |
| Binary | Bone Collagen | 36 | 13 | Low (≤10%): 31, High (>10%): 18 |
| 4-class | Bone Collagen | 32 | 11 | A:13, F:13, G:6, H:11 |
| 7-class | Bone Collagen | 36 | 13 | A:13, C:1, F:13, G:6, H:11, I:3, J:2 |

**Note:**
- 7-class split is not stratified due to classes with only 1-2 samples
- **d13C dataset provides more robust regression testing** with n=105 train, n=35 test

## Parameter Equivalence: DASP ↔ R

### PLS Regression

| DASP | R (`pls::plsr`) | Notes |
|------|-----------------|-------|
| `n_components` | `ncomp` | Number of PLS components |
| `max_iter=500` | N/A | Algorithm difference |
| `tol=1e-6` | N/A | Algorithm difference |
| `scale=False` | `scale=FALSE` | No scaling, only centering |

### Ridge/Lasso/ElasticNet

| DASP (scikit-learn) | R (`glmnet`) | Notes |
|---------------------|--------------|-------|
| `alpha` (regularization) | `lambda` | **Confusing naming!** |
| `l1_ratio` (L1/L2 mix) | `alpha` | 0=Ridge, 1=Lasso, 0.5=ElasticNet |

**Example:**
- DASP: `Ridge(alpha=1.0)` → R: `glmnet(alpha=0, lambda=1.0)`
- DASP: `Lasso(alpha=0.1)` → R: `glmnet(alpha=1, lambda=0.1)`
- DASP: `ElasticNet(alpha=1.0, l1_ratio=0.5)` → R: `glmnet(alpha=0.5, lambda=1.0)`

### Random Forest

| DASP (scikit-learn) | R (`randomForest`) | Notes |
|---------------------|-------------------|-------|
| `n_estimators` | `ntree` | Number of trees |
| `max_depth` | `maxnodes` | R uses max nodes (≈ 2^depth) |
| `min_samples_leaf` | `nodesize` | Min samples per leaf node |
| `max_features='sqrt'` | `mtry=sqrt(p)` | Features per split |

### XGBoost

| DASP (xgboost) | R (`xgboost`) | Notes |
|----------------|---------------|-------|
| `n_estimators` | `nrounds` | Number of boosting rounds |
| `learning_rate` | `eta` | Learning rate |
| `max_depth` | `max_depth` | Same |
| `subsample` | `subsample` | Same |
| `colsample_bytree` | `colsample_bytree` | Same |

## Expected Outcomes

### Regression Performance (Bone Collagen)

**Expected R² range:** 0.70 - 0.85
**Expected RMSE range:** 2.5 - 4.5% collagen

**Best models:**
- PLS with SNV preprocessing (n_components: 10-20)
- Random Forest with default parameters
- XGBoost with moderate depth (3-6)

### Classification Performance

**Binary (High/Low collagen):**
- Expected accuracy: 75-85%
- Expected AUC: 0.80-0.90

**4-class (A, F, G, H):**
- Expected accuracy: 60-75%
- Expected F1: 0.55-0.70

## Validation Criteria

### Model Equivalence
- **R² match:** Within ±2% for PLS/Ridge/Lasso
- **R² match:** Within ±3-5% for RF/XGBoost (due to randomness)
- **RMSE match:** Within ±5%
- **Accuracy match:** Within ±3%

### Reproducibility
- Identical results with fixed random seed (within machine precision)
- Consistent performance across multiple runs (CV std < 0.05)

### Hyperparameter Quality
- DASP defaults should perform within top 20% of tested configurations
- Grid search should include optimal parameter ranges

## Next Steps

1. **Create DASP comparison scripts** - Run identical tests with DASP
2. **Compare predictions** - Scatter plots, correlation analysis
3. **Statistical testing** - Paired t-tests for significance
4. **Preprocessing validation** - Compare SNV/SG outputs directly
5. **Variable selection validation** - Compare VIP, SPA, UVE, iPLS
6. **Performance benchmarking** - Time and memory usage
7. **Generate final report** - Comprehensive validation document

## Troubleshooting

### R Package Installation Issues

**LightGBM:**
```R
# May require manual compilation
install.packages("lightgbm", repos="https://cran.r-project.org")
```

**mdatools (for SPA, iPLS):**
```R
# Install from GitHub if not on CRAN
devtools::install_github("svkucheryavski/mdatools")
```

### Data Loading Issues

If CSV files have encoding issues, try:
```R
X_train <- as.matrix(read.csv("X_train.csv", fileEncoding="UTF-8"))
```

### Memory Issues

For large datasets, use:
```R
options(java.parameters = "-Xmx8g")  # Increase heap size
gc()  # Force garbage collection
```

## References

### DASP
- Model implementations: `src/spectral_predict/models.py`
- Hyperparameter configs: `src/spectral_predict/model_config.py`
- Preprocessing: `src/spectral_predict/preprocess.py`

### R Packages
- PLS: https://cran.r-project.org/package=pls
- glmnet: https://cran.r-project.org/package=glmnet
- randomForest: https://cran.r-project.org/package=randomForest
- xgboost: https://xgboost.readthedocs.io/
- prospectr: https://cran.r-project.org/package=prospectr

## Contact

For issues or questions about this validation framework, please refer to the main DASP repository documentation.

---

**Last updated:** 2025-11-14
**DASP version:** Current development branch
**Random seed:** 42 (for all splits and models)
