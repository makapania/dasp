# Spectral Predict - Agent Guide

---

## Project Overview

Spectral Predict automates spectral modeling: load spectra, test preprocessing x model combinations with cross-validation, get ranked results.

## Entry Point

| Version | Entry Point | Framework | Location |
|---------|-------------|-----------|----------|
| **V1** | `python spectral_predict_gui_optimized.py` | Tkinter | `src/spectral_predict/` |

## Directory Structure

```
dasp/
├── spectral_predict_gui_optimized.py   # V1 GUI (Tkinter, 7 tabs)
├── src/spectral_predict/               # V1 backend
│   ├── search.py                       # Core automation
│   ├── models.py                       # Model implementations
│   ├── preprocess.py                   # Preprocessing pipeline
│   ├── variable_selection.py           # Variable selection
│   ├── calibration_transfer.py         # DS, PDS transfer
│   └── io.py                           # File format readers
├── tests/                              # Test suite
├── docs/                               # User guide and reference docs
├── installer/                          # Inno Setup build infrastructure
└── example/                            # Example data
```

## Key Features

- **Models**: PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, SVM, MLP
- **Preprocessing**: SNV, Savitzky-Golay derivatives (1st, 2nd), baseline correction
- **Variable Selection**: UVE, SPA, iPLS, CARS, GA-PLS
- **Calibration Transfer**: Direct Standardization (DS), Piecewise DS (PDS)
- **File Formats**: CSV, Excel, ASD, OPUS, SPC, JCAMP-DX, PerkinElmer

## Development Notes

- Tier (Quick/Standard/Comprehensive) only affects which models are tested
- All hyperparameters are exposed and user-editable
