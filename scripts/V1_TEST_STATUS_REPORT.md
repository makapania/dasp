# V1 Comprehensive Testing - Final Report

**Date**: 2025-12-21
**Test Script**: `scripts/v1_comprehensive_test.py`
**Data**: `C:\Users\sponheim\Desktop\bone` (49 samples)
**Status**: COMPLETED

---

## TEST CONFIGURATION

- **Data**: BoneCollagen.csv + 49 ASD spectral files
- **Split**: 41 train / 8 test using SPXY
- **Targets**: %Collagen (regression, 0.9-22.1%), CollagenCat (classification)
- **Run Command**: `PYTHONIOENCODING=utf-8 python scripts/v1_comprehensive_test.py --no-prompt`
- **Output File**: `test_output_5.txt` (175K+ lines)

---

## COMPLETED RESULTS

### Stage 1: Data Loading ✅
- 49 spectra loaded from ASD files
- Wavelength range: 350-2500nm, 2151 wavelengths
- SPXY split successful (41/8)
- Test set covers full target range (0.9-21.7)

### Stage 2: Grid Search Baseline ✅

**Regression (%Collagen) - Best Results:**
| Model | Preprocessing | RMSE | R² |
|-------|--------------|------|-----|
| PLS | deriv (2nd) | 1.722 | 0.871 |
| Ridge | snv | 1.972 | 0.797 |
| LightGBM | deriv (2nd) | 1.733 | 0.878 |

**Classification (CollagenCat) - Best Results:**
- AUC: 0.969 (PLS-DA)
- Accuracy: 0.881 (LightGBM)

### Stage 3: Variable Selection ✅
- Tested 6 methods: importance, spa, uve, uve_spa, ipls, cars
- 154 configs per method (924+ total combinations)
- Variable counts tested: 10, 20, 50, 100, 250
- Best iPLS result: R²=0.933, RMSE=1.524 (better than baseline!)

### Stage 4: Bayesian Optimization ✅
- Ran with PLS and Ridge models
- Quick-tier comparison baseline established
- Tested across 14 preprocessing configurations

### Stage 5: NSGA2 Multi-Objective ✅
- Completed Pareto front generation
- Multi-objective optimization finished

### Stage 6: Model Save/Load ✅ (with bug found)
- **BUG FOUND**: Test script had wrong argument order for `save_model()`
- **Fixed**: Changed from `save_model(model, save_path, metadata)` to `save_model(model, None, metadata, save_path)`
- All 7 models failed due to test script bug (not model_io.py bug)

### Stage 7: Holdout Validation ✅

**Regression Holdout Results:**
| Model | CV RMSE | Holdout RMSE | Holdout R² |
|-------|---------|--------------|------------|
| PLS | 1.722 | 4.978 | 0.520 |
| Ridge | 1.972 | 4.416 | 0.622 |
| LightGBM | 1.733 | 4.318 | 0.639 |

⚠️ **WARNING**: Holdout RMSE much worse than CV RMSE for all models

**Classification Holdout Results:**
| Model | CV Accuracy | Holdout Accuracy |
|-------|-------------|------------------|
| PLS-DA | 0.761 | 0.000 |
| LightGBM | 0.881 | 0.875 |

---

## ISSUES FOUND & FIXES

### Issue 1: Unicode Encoding on Windows ✅ FIXED
- **Problem**: `search.py` line 1621 has emoji (⚠️) that crashes Windows cp1252
- **Fix**: Run with `PYTHONIOENCODING=utf-8`
- **Long-term**: Remove emojis from search.py or add encoding handling

### Issue 2: Column Name Mismatches ✅ FIXED
- **Problem**: Test script used wrong column names
- **Fixed**: `'CV RMSE'` → `'RMSE'`, `'Preprocessing'` → `'Preprocess'`, `'Variable Subset'` → `'SubsetTag'`

### Issue 3: save_model() Argument Order ✅ FIXED
- **Problem**: Test script called `save_model(model, save_path, metadata)`
- **Correct**: `save_model(model, preprocessor, metadata, filepath)`
- **Fixed**: Updated both regression and classification calls in test script

### Issue 4: Holdout vs CV Discrepancy ⚠️ NEEDS INVESTIGATION
- **Problem**: All regression models have much worse holdout RMSE (4.3-5.0) vs CV RMSE (1.7-1.9)
- **Possible causes**:
  1. 8-sample test set too small to be representative
  2. SPXY placed unusual samples in test set
  3. Some overfitting during CV
- **Recommendation**: Run permutation test or try different random splits

### Issue 5: PLS-DA Holdout Failure ⚠️ NEEDS INVESTIGATION
- **Problem**: PLS-DA got 0% accuracy on holdout (0/8 correct)
- **Note**: LightGBM got 87.5% on same holdout data
- **Possible cause**: Label encoding issue or model fit problem

---

## KEY CODE LOCATIONS

| Component | File | Key Function |
|-----------|------|--------------|
| Grid Search | `src/spectral_predict/search.py:159` | `run_search()` |
| Bayesian Search | `src/spectral_predict/search.py:1554` | `run_bayesian_search()` |
| NSGA2 Search | `src/spectral_predict/nsga2_search.py:871` | `run_nsga2_search()` |
| Variable Selection | `src/spectral_predict/variable_selection.py` | spa, uve, ipls, cars |
| Model Save/Load | `src/spectral_predict/model_io.py` | `save_model()`, `load_model()` |
| SPXY Split | `src/spectral_predict/sample_selection.py:243` | `spxy()` |

---

## BASELINE NUMBERS

**Regression baseline** (grid search best):
- Best RMSE: 1.722 (PLS with 2nd derivative)
- Best R²: 0.878 (LightGBM with 2nd derivative)

**Variable Selection Improvement**:
- Best iPLS result: RMSE 1.524, R² 0.933 (beat baseline!)

**Classification baseline**:
- Best AUC: 0.969
- Best Accuracy: 0.881

---

## NEXT STEPS

1. Re-run Stage 6 (Save/Load) with fixed test script
2. Investigate holdout vs CV discrepancy
3. Investigate PLS-DA classification failure
4. Consider using larger holdout or repeated random splits
5. Remove emojis from search.py for Windows compatibility
