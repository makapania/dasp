# Recovery TODO - December 16-20, 2025 Work

> Tracking 4 days of work to be preserved/restored.
> Created: 2025-12-20
> Status: **BACKED UP** to `backup_2025-12-20/`

---

## USER PRIORITY FEATURES (Implement First)

### UI/Layout Fixes
- [x] **1. Move Data Management to Advanced section** - Currently in Data section, should be at bottom in Advanced
- [x] **2. Remove extra space in Data Management** - Space between icon and text
- [x] **3. Data Management opens on Import & Preview** - Should always open on Import & Preview tab

### Imbalance Handling
- [x] **4. Add SMOTE for regression** - RegressionResampler with 'smogn' method + GUI
- [x] **5. Add SMOTE-Tomek for regression** - RegressionResampler with 'smotetomek' method + GUI
- [x] **6. Add artificial sampling for regression imbalances** - RegressionResampler with 'oversample' method + GUI

### Variable Selection
- [x] **7. Add CARS** - Already exists in variable_selection.py and GUI
- [x] **8. Add VCPA-IRIV** - Already exists in wavelength_selection.py and GUI

### Genetic Algorithm Features
- [x] **9. Add GA Variable Selection** - Working (slow but functional)
- [x] **10. Add GA Preprocessing** - Already fully implemented in GUI + backend

### Ensemble/Buttons
- [ ] **11. Redo all buttons after Ensemble Models** - Text overlapping graphics, half don't work - FROM SCRATCH
- [ ] **12. Check Ensembles for classification** - If not supported, disable; if should work, implement

### Optimization
- [ ] **13. Add NSGA-II** - As addition to Bayesian, works beyond PLS/Ridge

### Model Config Cleanup
- [ ] **14. Remove 'Modern' from Gradient Boosting** - Make header blue like others
- [ ] **15. Remove all "new" logos** - Prerelease = everything is new
- [ ] **16. Rename 'Advanced Models' to 'Other Models'**

### Preprocessing
- [ ] **17. Verify SG3 and SG4 integration** - Were not correctly integrated at one point

---

## DECEMBER 16-20 FEATURES (From Backup)

*Note: Some overlap with user features above - code exists in `backup_2025-12-20/`*

### Already Have Code For (overlaps with above)
- [ ] **NSGA-II** → overlaps with #11 - code in `backup_2025-12-20/src/spectral_predict/nsga2_search.py`
- [ ] **GA-PLS** → overlaps with #7 - code in `backup_2025-12-20/src/spectral_predict/ga_pls.py`
- [ ] **GA-LightGBM** → extends #7 for tree models - code in `backup_2025-12-20/src/spectral_predict/ga_lightgbm.py`

### NEW Features (integrate from backup)
- [ ] **18. Spectral Library Search** - Persistent local library, duplicate detection, wavelength alignment
  - Code: `backup_2025-12-20/src/spectral_predict/library_search.py`
- [ ] **19. Similarity Metrics** - HQI, SAM, Euclidean, derivative correlation
  - Code: `backup_2025-12-20/src/spectral_predict/similarity_metrics.py`
- [ ] **20. Search Controller** - Thread-safe pause/resume/end for long searches
  - Code: `backup_2025-12-20/src/spectral_predict/search_controller.py`
- [ ] **21. Test Suite** - Comprehensive pytest fixtures and test infrastructure
  - Code: `backup_2025-12-20/tests/`

---

## Backup Location

All files have been backed up to: `C:\Users\sponheim\git\dasp\backup_2025-12-20\`

Contents:
- `src/spectral_predict/` - 6 new modules
- `tests/` - Full test suite (50+ files)
- `scripts/` - Utility scripts

---

## December 16-20 New Modules (After User Features)

### NSGA-II Multi-Objective Optimization (`src/spectral_predict/nsga2_search.py`)
- [ ] Pareto optimization for multiple conflicting objectives
- [ ] Minimize prediction error (RMSE/1-Accuracy)
- [ ] Minimize wavelengths (parsimony)
- [ ] Minimize model complexity (LVs, model type)
- [ ] Knee point detection for "best compromise"
- [ ] Uses pymoo library

### Spectral Library Search (`src/spectral_predict/library_search.py`)
- [ ] Persistent local library storage across sessions
- [ ] Automatic duplicate detection
- [ ] Multiple similarity metrics
- [ ] Wavelength alignment for different instruments
- [ ] Uses platformdirs for cross-platform storage

### Similarity Metrics (`src/spectral_predict/similarity_metrics.py`)
- [ ] Hit Quality Index (HQI) - industry standard
- [ ] Spectral Angle Mapper (SAM)
- [ ] Euclidean distance
- [ ] Derivative correlation
- [ ] Batch similarity computation

### GA-PLS Wavelength Selection (`src/spectral_predict/ga_pls.py`)
- [ ] Genetic Algorithm for variable selection (Leardi 1998)
- [ ] Binary chromosome (wavelength selected/excluded)
- [ ] Multiple GA runs aggregation
- [ ] Fitness caching for performance
- [ ] Early stopping

### GA-LightGBM (`src/spectral_predict/ga_lightgbm.py`)
- [ ] Genetic Algorithm with LightGBM model
- [ ] Wavelength selection optimization

### Search Controller (`src/spectral_predict/search_controller.py`)
- [ ] Thread-safe pause/resume/end
- [ ] Uses threading.Event for cross-thread signaling
- [ ] Check-and-wait pattern for search loops

---

## High Priority - Test Infrastructure

### Test Framework (`tests/`)
- [ ] `conftest.py` - Shared pytest fixtures
- [ ] `fixtures/synthetic_data.py` - Data generators
- [ ] `gold_standards/` - Reference outputs for regression testing
- [ ] 50+ test files covering all modules

### Key Test Files
- [ ] `test_nsga2_search.py` - NSGA-II tests
- [ ] `test_library_search.py` - Library search tests
- [ ] `test_ga_pls.py` - GA-PLS tests
- [ ] `test_ga_lightgbm.py` - GA-LightGBM tests
- [ ] `test_search_comprehensive.py` - Search integration
- [ ] `test_end_to_end_workflows.py` - Full workflow tests

### Fixtures
- [ ] Synthetic spectra (small/medium/large)
- [ ] Classification data (balanced/imbalanced)
- [ ] Outlier data with known outliers
- [ ] Pre-trained model fixtures

---

## Medium Priority - Scripts

### Utility Scripts (`scripts/`)
- [ ] `apply_nsga2_fixes.py` - NSGA-II fixes
- [ ] `apply_nsga2_phase4.py` - Phase 4 updates
- [ ] `check_coverage.py` - Coverage analysis
- [ ] `generate_gold_standards.py` - Generate reference outputs
- [ ] `run_tests.py` - Test runner

---

## Recent Commit (Already in Git)

- `ca74c19` fix: UVE-SPA and VCPA-IRIV results not appearing in Results tab

---

## Files Location Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/spectral_predict/nsga2_search.py` | Backed up | NSGA-II optimization |
| `src/spectral_predict/library_search.py` | Backed up | Library search |
| `src/spectral_predict/similarity_metrics.py` | Backed up | Similarity metrics |
| `src/spectral_predict/ga_pls.py` | Backed up | GA-PLS |
| `src/spectral_predict/ga_lightgbm.py` | Backed up | GA-LightGBM |
| `src/spectral_predict/search_controller.py` | Backed up | Search controller |
| `tests/` | Backed up | Full test suite |
| `scripts/` | Backed up | Utility scripts |
| `recovery_blobs/` | Not backed up | Old November blobs (not needed) |

---

## Completed

- [x] Create backup directory
- [x] Backup all new modules
- [x] Backup test suite
- [x] Backup scripts
- [x] Create this TODO file

---

## Next Steps

1. Review backed up code quality
2. Decide what to restore after reset
3. Test restored functionality
4. Commit working code
