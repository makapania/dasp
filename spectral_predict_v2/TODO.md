# Spectral Predict v2 - TODO

**Status: Work in Progress - Many issues need fixing**

Last updated: 2025-12-02

---

## Critical Issues (Must Fix)

### File Loading
- [ ] File/folder upload not working properly
- [ ] Column config dialog may not handle all file formats
- [ ] Need to test with various Excel and CSV formats
- [ ] Error handling needs improvement - show user-friendly messages

### Analysis Execution
- [ ] Analysis fails to run - needs debugging
- [ ] Check model name mappings between UI and search module
- [ ] Verify task type (regression/classification) is passed correctly
- [ ] Test with both regression and classification datasets

### UI/UX Issues
- [ ] UI is unattractive - needs styling
- [ ] Add proper dark theme or modern styling
- [ ] Improve layout and spacing
- [ ] Add loading indicators/spinners
- [ ] Better error dialogs

---

## Features to Complete

### Explore Mode
- [ ] Fix file drop widget
- [ ] Test preset loading and application
- [ ] Verify results table populates correctly
- [ ] Test model pinning and comparison
- [ ] Export results to CSV

### Build Mode
- [ ] Test model selection from Explore results
- [ ] Verify training works
- [ ] Test diagnostic plots (pred vs ref, residuals, loadings)
- [ ] Test model saving to .dasp format
- [ ] Test report export

### Predict Mode
- [ ] Test model loading from .dasp files
- [ ] Test prediction on new data
- [ ] Verify uncertainty estimation
- [ ] Test applicability domain flagging
- [ ] Test batch folder processing
- [ ] Test CSV/report export

### Tools
- [ ] Test Calibration Transfer tool
- [ ] Test Data Quality tool (outlier detection)
- [ ] Test Interference Removal tool
- [ ] Test Preset Manager

---

## Code Quality

### Testing
- [ ] Run existing test suite: `pytest spectral_predict_v2/tests/ -v`
- [ ] Add more unit tests for engine API
- [ ] Add integration tests for full workflows
- [ ] Test with real spectral data files

### Documentation
- [x] README.md created
- [x] MIGRATION.md created
- [x] PERFORMANCE.md created
- [ ] Add inline code comments where needed
- [ ] Document API for programmatic use

---

## Known Dependencies

Using Python 3.12 venv (`.venv312`):
- PySide6 (GUI framework)
- numpy, pandas, scipy
- scikit-learn
- matplotlib
- optuna (Bayesian optimization)
- lightgbm, xgboost, catboost
- joblib

Run with: `run_spectral_predict_v2.bat`

---

## Architecture Reference

```
spectral_predict_v2/
├── main.py                    # Entry point
├── engine/
│   └── api.py                 # Engine API - wraps src/spectral_predict
├── orchestration/
│   ├── state_store.py         # Central state management
│   ├── job_runner.py          # Background task execution
│   └── config_manager.py      # Presets and configuration
├── ui/
│   ├── app.py                 # Main application window
│   ├── modes/
│   │   ├── explore.py         # Explore mode (auto analysis)
│   │   ├── build.py           # Build mode (train models)
│   │   └── predict.py         # Predict mode (apply models)
│   ├── tools/                 # Tool panels
│   └── widgets/               # Reusable UI components
└── tests/                     # Test suite
```

---

## Quick Debug Commands

```bash
# Run the app
run_spectral_predict_v2.bat

# Test imports
.venv312\Scripts\python.exe -c "import sys; sys.path.insert(0, 'spectral_predict_v2'); from engine.api import EngineAPI; print('OK')"

# Run tests
.venv312\Scripts\python.exe -m pytest spectral_predict_v2/tests/ -v

# Check syntax of a file
.venv312\Scripts\python.exe -m py_compile spectral_predict_v2/ui/modes/explore.py
```

---

## Notes

- The v1 GUI (`spectral_predict_gui_optimized.py`) still works and can be used while v2 is being fixed
- v2 wraps the same analysis engine as v1 - core analysis code is unchanged
- Main issues are in the UI layer and integration, not the analysis engine
