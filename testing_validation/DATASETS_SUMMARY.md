# Testing Validation Datasets Summary

This document provides detailed information about all datasets included in the comprehensive testing framework.

---

## Overview

The testing framework includes **2 primary datasets** covering different spectral analysis scenarios:

1. **Bone Collagen** - Small dataset for both regression and classification
2. **Enamel d13C** - Larger dataset for robust regression testing

---

## Dataset 1: Bone Collagen

### Description
Bone samples analyzed for collagen content using VIS-NIR spectroscopy. This dataset serves as the primary test for both regression (predicting %Collagen) and multi-task classification (categorical sample groups).

### Source
- **Location:** `example/BoneCollagen.csv` + `example/Spectrum*.asd`
- **Original purpose:** DASP example dataset
- **Publication:** (if applicable - add reference)

### Dataset Characteristics

| Property | Value |
|----------|-------|
| **Total Samples** | 49 (37 used in testing) |
| **Spectral Range** | 350-2500 nm (VIS-NIR) |
| **Wavelengths** | 2,151 channels |
| **Spectral Type** | Reflectance |
| **Format** | Binary ASD files |

### Target Variables

#### Regression: %Collagen
- **Type:** Continuous
- **Range:** 0.9% - 22.1%
- **Mean:** 8.8% ± 6.2%
- **Distribution:** Right-skewed (more low-collagen samples)

#### Classification: Sample Categories
- **Type:** Categorical (7 classes)
- **Classes:** A, C, F, G, H, I, J
- **Distribution:**
  - A: 13 samples (26.5%)
  - F: 13 samples (26.5%)
  - H: 11 samples (22.4%)
  - G: 6 samples (12.2%)
  - I: 3 samples (6.1%)
  - J: 2 samples (4.1%)
  - C: 1 sample (2.0%)

### Train/Test Splits

#### 1. Regression Task
- **Train:** 36 samples (75%)
- **Test:** 13 samples (25%)
- **Stratification:** By collagen quartiles
- **Files:** `data/regression_train.csv`, `data/regression_test.csv`

#### 2. Binary Classification
- **Task:** High (>10%) vs. Low (≤10%) collagen
- **Train:** 36 samples (Low=23, High=13)
- **Test:** 13 samples (Low=8, High=5)
- **Balance ratio:** 0.58 (acceptable)
- **Files:** `data/binary_train.csv`, `data/binary_test.csv`

#### 3. 4-Class Classification
- **Task:** Predict category (A, F, G, H only)
- **Train:** 32 samples
- **Test:** 11 samples
- **Distribution:** Balanced among major classes
- **Files:** `data/4class_train.csv`, `data/4class_test.csv`

#### 4. 7-Class Classification
- **Task:** Predict all categories
- **Train:** 36 samples
- **Test:** 13 samples
- **Note:** Non-stratified due to very small classes (C=1, J=2, I=3)
- **Files:** `data/7class_train.csv`, `data/7class_test.csv`

### Expected Performance

**Regression:**
- **R²:** 0.70 - 0.85
- **RMSE:** 2.5 - 4.5% collagen
- **Best models:** PLS (10-20 components), Random Forest, XGBoost

**Binary Classification:**
- **Accuracy:** 75-85%
- **ROC-AUC:** 0.80-0.90

**4-Class Classification:**
- **Accuracy:** 60-75%
- **Macro-F1:** 0.55-0.70

### Challenges
- **Small sample size** (n=49) - Risk of overfitting
- **Class imbalance** for 7-class task
- **Limited dynamic range** for collagen (only 21.2% span)

### Scientific Context
- **Application:** Forensic anthropology, archaeological bone analysis
- **Importance:** Collagen content indicates bone preservation quality
- **Spectral features:** Strong absorption bands in NIR related to protein content

---

## Dataset 2: Enamel d13C

### Description
Enamel samples analyzed for carbon isotope ratio (d13C) using NIR spectroscopy. This dataset provides a larger sample size for robust regression testing and validation of DASP's performance on isotopic analysis.

### Source
- **Location:** `C:\Users\sponheim\Desktop\ellie\Ellie_NIR_Data.csv` + ASD files
- **Copied to:** `testing_validation/data_sources/d13c/`
- **Original purpose:** Enamel isotope research
- **Publication:** (if applicable - add reference)

### Dataset Characteristics

| Property | Value |
|----------|-------|
| **Total Samples** | 140 (matched spectra + reference) |
| **Spectral Range** | 350-2500 nm (NIR) |
| **Wavelengths** | 2,151 channels |
| **Spectral Type** | Absorbance (53.8% confidence) |
| **Format** | Binary ASD files |

### Target Variable: d13C

- **Type:** Continuous (carbon isotope ratio)
- **Range:** -26.40‰ to -14.30‰
- **Mean:** -21.53‰ ± 3.33‰
- **Distribution:** Approximately normal
- **Units:** Per mil (‰) relative to VPDB standard

### Train/Test Split

- **Train:** 105 samples (75%)
  - d13C range: -26.40‰ to -14.40‰
  - d13C mean: -21.50‰
- **Test:** 35 samples (25%)
  - d13C range: -26.00‰ to -14.30‰
  - d13C mean: -21.63‰
- **Stratification:** By d13C quartiles
- **Files:** `data/d13c_train.csv`, `data/d13c_test.csv`

### Data Quality Notes

- **Missing values:** 5 samples removed (no d13C value)
- **Mismatched samples:** 7 reference samples without spectra, 6 spectra without reference
  - Likely due to naming inconsistencies (e.g., "04-TSV-391" vs "04-TSV-391-Relabel")
- **Final matched:** 140 of 152 original samples (92.1% match rate)

### Expected Performance

**Regression:**
- **R²:** 0.75 - 0.90 (based on previous DASP runs: R²=0.845)
- **RMSE:** 1.0 - 1.5‰
- **Best models:** LightGBM, XGBoost, Random Forest with derivative preprocessing
- **Best preprocessing:** 2nd derivative (Savitzky-Golay, window=17)

### Advantages Over Bone Collagen

- **Larger sample size** (140 vs. 49) → More robust validation
- **Better distribution** (approximately normal vs. right-skewed)
- **Wider dynamic range** (12.1‰ span vs. 21.2% collagen, but relative to mean: 0.56 vs. 2.4)
- **Real-world application** - Successful DASP analysis (R²=0.845)

### Challenges

- **Low confidence in data type detection** (absorbance vs. reflectance)
- **Mismatched sample names** requiring careful ID matching
- **Potential sample heterogeneity** (collected over time: Dec 2002 - Nov 2003)

### Scientific Context

- **Application:** Dietary reconstruction, paleoclimate, ecology
- **Importance:** d13C indicates C3 vs. C4 plant consumption
- **Spectral features:** NIR absorption related to organic matter content, which correlates with isotopic composition
- **Challenge:** Indirect spectral relationship (NIR → organic content → isotopes)

---

## Comparison: Bone Collagen vs. Enamel d13C

| Aspect | Bone Collagen | Enamel d13C |
|--------|---------------|-------------|
| **Primary Use** | Multi-task (regression + classification) | Regression only |
| **Sample Size** | 49 (small) | 140 (moderate) |
| **Test Set** | 13 samples | 35 samples |
| **Target Range** | 0.9-22.1% (21.2% span) | -26.4 to -14.3‰ (12.1‰ span) |
| **Distribution** | Right-skewed | Normal |
| **Expected R²** | 0.70-0.85 | 0.75-0.90 |
| **Validation Strength** | Weak (n=13 test) | Strong (n=35 test) |
| **Scientific Domain** | Forensic/archaeological | Isotope ecology |
| **DASP Familiarity** | Example dataset | Real-world use case |

---

## Dataset Files Organization

```
testing_validation/
├── data/                           # Train/test splits (CSV)
│   ├── regression_train.csv        # Bone collagen regression (36)
│   ├── regression_test.csv         # (13)
│   ├── binary_train.csv            # Bone collagen binary classification
│   ├── binary_test.csv
│   ├── 4class_train.csv            # Bone collagen 4-class
│   ├── 4class_test.csv
│   ├── 7class_train.csv            # Bone collagen 7-class
│   ├── 7class_test.csv
│   ├── d13c_train.csv              # Enamel d13C regression (105)
│   ├── d13c_test.csv               # (35)
│   ├── metadata.json               # Bone collagen metadata
│   └── d13c_metadata.json          # Enamel d13C metadata
│
├── r_data/                         # Spectral matrices for R (CSV)
│   ├── regression/                 # Bone collagen
│   │   ├── X_train.csv             # 36 × 2151
│   │   ├── X_test.csv              # 13 × 2151
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   └── wavelengths.csv
│   ├── binary/                     # Same structure
│   ├── 4class/
│   ├── 7class/
│   └── d13c/                       # Enamel d13C
│       ├── X_train.csv             # 105 × 2151
│       ├── X_test.csv              # 35 × 2151
│       ├── y_train.csv
│       ├── y_test.csv
│       └── wavelengths.csv
│
└── data_sources/                   # Original source data (copied)
    └── d13c/
        ├── Ellie_NIR_Data.csv
        └── spectra/
            └── *.asd               # 146 ASD files
```

---

## Testing Strategy by Dataset

### Bone Collagen (Primary: Multi-task Validation)

**Focus:**
1. **Parameter equivalence** - Small dataset makes differences more visible
2. **Multi-task capability** - Test both regression and classification
3. **Small sample robustness** - Validate warnings, error handling
4. **Classification metrics** - Binary and multi-class

**Tests:**
- All model types (PLS, Ridge, Lasso, RF, XGBoost, LightGBM)
- All preprocessing methods
- Variable selection (critical for n=49)
- Edge cases (n_components > n_samples warnings)

### Enamel d13C (Primary: Robust Regression Validation)

**Focus:**
1. **Statistical power** - Larger n=140 for robust comparison
2. **Regression performance** - Single clear target
3. **Real-world validation** - Known good performance (R²=0.845)
4. **Preprocessing effects** - Derivatives critical for this data

**Tests:**
- Full regression model comparison (DASP vs. R)
- Preprocessing validation (SNV, derivatives)
- Variable selection (50-200 variables from 2151)
- Hyperparameter sensitivity (larger n allows finer grid)
- Performance benchmarking (moderate size, realistic runtime)

---

## Usage in Testing Framework

### For DASP vs. R Comparison
- **Use both datasets** for regression
- **Use bone collagen** for classification
- **Use d13C** as primary regression benchmark (more robust)

### For Hyperparameter Tuning
- **Use d13C** primarily (better statistical power)
- **Use bone collagen** to verify consistency across scales

### For Edge Case Testing
- **Use bone collagen** (small n triggers warnings)
- **Create synthetic subsets** of d13C for scalability testing

### For Variable Selection
- **Use d13C** (clearer signal, derivative features)
- **Compare to bone collagen** (different spectral characteristics)

---

## Known Issues & Limitations

### Bone Collagen
1. **Small test set (n=13)** - Limited statistical power
2. **Class imbalance** - C=1, J=2, I=3 samples
3. **Non-stratified 7-class split** - Due to very small classes

### Enamel d13C
1. **Sample name mismatches** - 7 reference, 6 spectra unmatched
2. **Low confidence data type** - Absorbance detection only 53.8%
3. **No classification targets** - Regression only
4. **Temporal variation** - Samples collected over 1 year (potential confound)

---

## Future Dataset Additions

**Desirable characteristics:**
1. **Medium size** (n=500-1000) for scalability testing
2. **Multi-class classification** (5-10 balanced classes)
3. **Public benchmark dataset** for reproducibility
4. **Different spectral range** (e.g., Raman, mid-IR)

**Candidates:**
- NIR grain datasets (protein, moisture) - publically available
- Raman bacteria classification - multi-class, published
- Hyperspectral imaging data - high dimensionality (>5000 wavelengths)

---

## Citations & References

### Bone Collagen
- (Add publication if available)
- DASP example dataset

### Enamel d13C
- (Add publication if available)
- Source: Ellie research project

---

**Last Updated:** 2025-11-14
**Framework Version:** 1.0
**Datasets:** 2 (Bone Collagen, Enamel d13C)
**Total Samples:** 189 (49 + 140)
**Total Tasks:** 5 (1 d13C regression, 1 collagen regression, 3 collagen classification)
