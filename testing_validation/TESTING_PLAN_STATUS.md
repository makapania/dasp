# DASP Comprehensive Testing Plan - Current Status

**Date:** 2025-11-14
**Purpose:** Rigorous scientific validation of DASP against R and Python standard implementations
**Critical Goal:** Ensure accuracy and reproducibility for scientific research applications

---

## Executive Summary

A comprehensive testing infrastructure has been established to validate DASP's machine learning implementations against industry-standard R and Python packages. The goal is to ensure that DASP produces equivalent results to established implementations, with proper hyperparameter defaults, and maintains scientific reproducibility standards.

### Current Status: Phase 2 Complete ✓

- ✅ **Phase 1:** Dataset preparation complete (4 task types, train/test splits)
- ✅ **Phase 2:** R comparison infrastructure complete (installation + testing scripts)
- ⏳ **Phase 3-10:** Ready to execute testing and analysis

---

## What Has Been Built

### 1. Dataset Preparation Infrastructure ✓

**Created Files:**
- `testing_validation/prepare_datasets.py` - Automated dataset preparation
- `testing_validation/export_spectra_for_r.py` - Spectral data export for R

**Output:**
- **Regression task:** Predict %Collagen (continuous 0.9% - 22.1%)
  - Train: 36 samples, Test: 13 samples
- **Binary classification:** High (>10%) vs. Low (≤10%) collagen
  - Train: 36 samples, Test: 13 samples
  - Class balance: 18 High, 31 Low (ratio: 0.58)
- **4-class classification:** Categories A, F, G, H (balanced major classes)
  - Train: 32 samples, Test: 11 samples
  - Classes: A=13, F=13, G=6, H=11
- **7-class classification:** All categories A-J (includes small classes)
  - Train: 36 samples, Test: 13 samples
  - Classes: A=13, C=1, F=13, G=6, H=11, I=3, J=2

**Key Features:**
- Stratified splits (75/25 train/test) with fixed random seed (42)
- Spectral data: 2,151 wavelengths (350-2500 nm VIS-NIR)
- Exported to CSV format for R compatibility
- Complete metadata tracking (class distributions, wavelength ranges)

### 2. R Comparison Scripts ✓

**Created Files:**
- `testing_validation/r_scripts/install_packages.R` - R package installation
- `testing_validation/r_scripts/regression_models.R` - Comprehensive regression testing
- `testing_validation/r_scripts/classification_models.R` - Comprehensive classification testing

**Regression Models Tested (with parameter equivalence to DASP):**
1. **PLS Regression** - 12 configurations (n_components: 2-50)
2. **Ridge Regression** - 5 configurations (lambda: 0.001-10.0)
3. **Lasso Regression** - 4 configurations (lambda: 0.001-1.0)
4. **ElasticNet** - 20 configurations (lambda × l1_ratio grid)
5. **Random Forest** - 6 configurations (ntree × max_depth)
6. **XGBoost** - 8 configurations (nrounds × eta × max_depth)
7. **LightGBM** - (optional, if available)

**Classification Models Tested:**
1. **PLS-DA** - 8 configurations
2. **Random Forest** - 6 configurations
3. **XGBoost** - 8 configurations

**Metrics Computed:**
- **Regression:** RMSE, R², MAE
- **Classification:** Accuracy, ROC-AUC (binary), Precision, Recall, F1, Macro-F1 (multiclass)

### 3. Documentation ✓

**Created Files:**
- `testing_validation/README.md` - Complete infrastructure documentation
- `testing_validation/TESTING_PLAN_STATUS.md` - This file
- `testing_validation/data/metadata.json` - Dataset statistics
- `testing_validation/results/dataset_preparation_summary.txt` - Dataset summary

**Documentation Includes:**
- Directory structure and file organization
- Quick start guide for running tests
- Parameter equivalence tables (DASP ↔ R)
- Expected performance benchmarks
- Troubleshooting guide

---

## Critical Parameter Equivalencies (DASP ↔ R)

### ⚠️ IMPORTANT: Naming Confusion

**Ridge/Lasso/ElasticNet:**
- DASP `alpha` (regularization strength) = R `lambda`
- DASP `l1_ratio` (L1/L2 mix) = R `alpha`

**Example:**
```python
# DASP
Ridge(alpha=1.0)
ElasticNet(alpha=1.0, l1_ratio=0.5)
```

```R
# R equivalent
glmnet(alpha=0, lambda=1.0)      # Ridge
glmnet(alpha=0.5, lambda=1.0)    # ElasticNet
```

### Other Models

| DASP | R | Parameter |
|------|---|-----------|
| `n_estimators` | `ntree` | Random Forest trees |
| `n_estimators` | `nrounds` | XGBoost rounds |
| `learning_rate` | `eta` | XGBoost learning rate |
| `n_components` | `ncomp` | PLS components |

---

## Next Steps: Execution Plan

### Phase 3: Run Regression Testing (Week 1)

**Immediate Actions:**

1. **Install R packages:**
   ```bash
   cd testing_validation/r_scripts
   Rscript install_packages.R
   ```

2. **Run R regression models:**
   ```bash
   Rscript regression_models.R
   ```
   - Output: `testing_validation/results/r_regression/*.json`
   - Expected time: 10-20 minutes
   - Models: ~55 configurations total

3. **Create DASP comparison script:**
   - Build `dasp_regression.py` to run identical tests
   - Load same train/test splits
   - Apply same preprocessing
   - Use same hyperparameters
   - Export predictions to JSON

4. **Compare results:**
   - Create `compare_regression_results.py`
   - Load DASP and R predictions
   - Calculate prediction correlation (should be >0.99)
   - Generate scatter plots
   - Compute metric differences (R² diff, RMSE diff)
   - Statistical tests (paired t-tests)

**Success Criteria:**
- R² matches within ±2% for PLS/Ridge/Lasso
- R² matches within ±3-5% for RF/XGBoost (randomness)
- Prediction correlation >0.99 for deterministic models
- No crashes or errors for any configuration

### Phase 4: Run Classification Testing (Week 1-2)

**Actions:**

1. **Run R classification models:**
   ```bash
   Rscript classification_models.R
   ```
   - Output: `testing_validation/results/r_classification/{binary,4class}/*.json`
   - Expected time: 10-15 minutes

2. **Create DASP classification script:**
   - Build `dasp_classification.py`
   - Test binary and 4-class tasks
   - Export predictions and probabilities

3. **Compare results:**
   - Accuracy match within ±3%
   - AUC match within ±0.05
   - Confusion matrix comparison

### Phase 5: Variable Selection Validation (Week 2)

**Models to Test:**
1. **VIP (PLS)** - Compare to R `pls::VIP()`
2. **Random Forest Importance** - Compare to R `importance()`
3. **SPA** - Compare to R `mdatools::spa()` (if available)
4. **iPLS** - Compare to R `mdatools::ipls()` (if available)

**Tests:**
- Select top 20 variables with each method
- Calculate Jaccard similarity (DASP vs. R selections)
- Compare model performance with selected variables
- Target: >70% overlap for deterministic methods

### Phase 6: Hyperparameter Optimization (Week 2-3)

**Tests:**

1. **Default hyperparameters:**
   - Run DASP with defaults (no tuning)
   - Run R with defaults
   - Compare performance
   - Target: DASP defaults should be in top 30% of grid

2. **Grid search coverage:**
   - Verify DASP grids include optimal regions
   - Test if "Comprehensive" tier finds best configs
   - Plot performance landscapes (heatmaps)

3. **Model tiers:**
   - Test Quick, Standard, Comprehensive tiers
   - Measure time vs. performance tradeoffs
   - Validate: Standard tier is good for routine use (<15 min)

### Phase 7: Edge Cases & Robustness (Week 3)

**Tests:**

1. **Small sample sizes:**
   - Test with 10, 15, 20 samples (quick_start subset)
   - Expect warnings for n_components > samples
   - Verify no crashes

2. **Outliers:**
   - Add synthetic outliers (±3σ) to 5 random samples
   - Compare model robustness (RF, XGBoost, PLS)
   - Test NeuralBoosted Huber loss

3. **Missing data:**
   - Remove random wavelength ranges
   - Test error handling
   - Verify graceful degradation

### Phase 8: Performance Benchmarking (Week 3)

**Tests:**

1. **Speed:**
   - Time each model tier on full dataset
   - Quick: expect <5 min
   - Standard: expect <15 min
   - Comprehensive: expect <30 min
   - Document hardware specs

2. **Scalability:**
   - Duplicate dataset to create 50, 100, 200 samples
   - Measure training time vs. sample size
   - Check for linear or worse scaling

3. **Memory:**
   - Monitor peak RAM during analysis
   - Test large synthetic dataset (1000 × 5000)
   - Target: <4GB for typical datasets

### Phase 9: Reproducibility (Week 3-4)

**Tests:**

1. **Seed consistency:**
   - Run same analysis 3 times with fixed seed
   - Verify exact reproduction (within machine precision)

2. **Platform independence:**
   - Test on Windows (current) and Linux (if possible)
   - Verify identical results

3. **Version stability:**
   - Document all package versions
   - Test with different scikit-learn versions if possible

### Phase 10: Final Documentation (Week 4)

**Deliverables:**

1. **Model Equivalence Report:**
   - Tables: DASP vs. R metrics
   - Scatter plots: Predictions correlation
   - Statistical tests: Paired t-tests
   - Parameter mapping guide

2. **Hyperparameter Analysis Report:**
   - Sensitivity plots (R² vs. n_components, etc.)
   - Optimal parameter recommendations
   - Performance vs. computation time tradeoffs

3. **Variable Selection Report:**
   - Overlap matrices (Jaccard similarity)
   - Performance with reduced variables
   - Method comparison (VIP vs. SPA vs. UVE vs. iPLS)

4. **Benchmark Report:**
   - Speed benchmarks (all models, all tiers)
   - Memory usage profiles
   - Scalability curves

5. **Validation Methodology Document:**
   - Test protocol
   - Acceptance criteria
   - Quality control procedures
   - Reproducibility guidelines

6. **User Guide:**
   - Parameter equivalence guide (DASP ↔ R ↔ Python)
   - Best practices document
   - Common pitfalls and solutions

---

## Expected Outcomes

### Regression Performance (Bone Collagen Dataset)

**Based on literature and preliminary tests:**
- **R² range:** 0.70 - 0.85
- **RMSE range:** 2.5 - 4.5% collagen
- **Best models:** PLS (n_comp=10-20) with SNV, Random Forest, XGBoost

### Classification Performance

**Binary (High/Low collagen):**
- **Accuracy:** 75-85%
- **AUC:** 0.80-0.90

**4-class (A, F, G, H):**
- **Accuracy:** 60-75%
- **Macro-F1:** 0.55-0.70

### Model Equivalence

**Acceptance criteria:**
- R² within ±2% for PLS/Ridge/Lasso (deterministic)
- R² within ±3-5% for RF/XGBoost (stochastic)
- Prediction correlation >0.99 for same random seed
- No systematic bias (mean error ≈ 0)

---

## Risk Assessment

### Potential Issues

1. **R package installation failures:**
   - **Risk:** LightGBM, mdatools may not install easily
   - **Mitigation:** Mark as optional, provide alternative sources

2. **Parameter equivalence complexity:**
   - **Risk:** Confusion between DASP `alpha` and R `lambda`
   - **Mitigation:** Clear documentation, conversion functions

3. **Random seed differences:**
   - **Risk:** R and Python RNGs may differ
   - **Mitigation:** Accept small differences for stochastic models, focus on deterministic first

4. **Performance discrepancies:**
   - **Risk:** Implementation differences may cause metric gaps
   - **Mitigation:** Document differences, investigate if >10% gap

5. **Time constraints:**
   - **Risk:** Full testing plan takes 3-4 weeks
   - **Mitigation:** Prioritize critical tests (Phases 3-6), defer edge cases if needed

---

## Success Metrics

### Minimum Viable Validation (MVP)

1. ✅ Dataset preparation complete
2. ✅ R infrastructure complete
3. ⏳ PLS regression equivalence demonstrated (R² within ±2%)
4. ⏳ Random Forest regression equivalence demonstrated (R² within ±5%)
5. ⏳ Binary classification equivalence demonstrated (Accuracy within ±3%)
6. ⏳ Hyperparameter defaults validated (top 30% performance)
7. ⏳ Reproducibility confirmed (fixed seed → exact results)

### Full Validation

1. All 6-7 regression models validated
2. All 3 classification models validated
3. Variable selection methods validated
4. Preprocessing equivalence confirmed
5. Performance benchmarks documented
6. Edge cases tested
7. Comprehensive report generated

---

## Resource Requirements

### Software

- **Python 3.14** (current)
- **R 4.x** (latest recommended)
- **Required disk space:** ~500 MB (data + results)
- **Required RAM:** 8 GB recommended, 4 GB minimum

### Time Estimates

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| 1-2 | Dataset prep + R infrastructure | ✅ Complete |
| 3 | Regression testing + comparison | 3-4 days |
| 4 | Classification testing | 2-3 days |
| 5 | Variable selection | 2-3 days |
| 6 | Hyperparameter validation | 2-3 days |
| 7 | Edge cases | 2 days |
| 8 | Performance benchmarking | 2 days |
| 9 | Reproducibility | 1-2 days |
| 10 | Documentation + reports | 2-3 days |
| **Total** | **Full validation** | **3-4 weeks** |

**Fast track (MVP only):** 1-2 weeks (Phases 1-2 done, 3-4 + key tests from 5-6)

---

## Next Immediate Actions

### Today (2025-11-14)

1. ✅ Review this status document
2. ⏳ Install R packages: `Rscript install_packages.R`
3. ⏳ Run R regression testing: `Rscript regression_models.R`
4. ⏳ Create `dasp_regression.py` script

### This Week

1. Complete regression comparison (DASP vs. R)
2. Run classification comparison
3. Generate initial comparison plots
4. Document any discrepancies

### Next Week

1. Variable selection validation
2. Hyperparameter optimization testing
3. Begin performance benchmarking

---

## Questions for Consideration

1. **Priority:** Which tests are most critical for your research needs?
   - Regression vs. classification?
   - Specific models (PLS, RF, XGBoost)?
   - Variable selection methods?

2. **Threshold:** What level of agreement is acceptable?
   - Current plan: ±2-5% for metrics
   - More stringent? Less stringent?

3. **Scope:** Full validation (3-4 weeks) or MVP (1-2 weeks)?
   - MVP: Focus on core models (PLS, RF, XGBoost) + basic validation
   - Full: All models + edge cases + comprehensive reports

4. **R availability:** Do you have R installed and configured?
   - If not, we can focus on Python-only comparisons (DASP vs. scikit-learn defaults)

---

## Conclusion

A robust testing infrastructure is now in place to comprehensively validate DASP against industry-standard implementations. The framework is designed to ensure scientific rigor, reproducibility, and accuracy for research applications.

**Current Status:** Ready to execute testing phases
**Next Critical Step:** Run R regression testing and create DASP comparison scripts
**Expected Timeline:** 3-4 weeks for full validation, 1-2 weeks for MVP

The infrastructure is modular and well-documented, allowing for incremental testing and validation as needed.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-14
**Contact:** See main DASP repository for issues/questions
