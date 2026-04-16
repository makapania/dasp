# Spectral Predict - Agent Guide

---

## 🔴 TOP PRIORITY (added 2026-04-16)

**Fix shared-model-state bug before next PyInstaller bundle.** The bundled app runs on Python 3.11 + sklearn 1.5.2; under that sklearn the grid-search refit block fails with `"X has N features, but LGBMRegressor is expecting 2135"` when variable subsets + region subsets are enabled, and LightGBM calibration metrics come back NaN. Root cause + two-site fix (both `clone(model)`) are documented in full in `docs/PROJECT_STATUS.md` under the "🔴 PRIORITY FOR NEXT SESSION" heading. Do this on a fresh branch off `main`, not on `cv-strategy-overhaul`. Do not ship the bundled app until this is fixed.

---

## Session Protocol (MANDATORY — READ THIS FIRST)

> **CRITICAL:** This project is worked on across multiple computers. All project knowledge MUST be stored in the repo, not in local memory. Failure to maintain these docs wastes the user's time re-explaining context.

### Session Start
1. **MUST** read `docs/PROJECT_STATUS.md` before doing any work. This tells you what's working, what's broken, and what to do next.
2. Check `docs/SESSION_LOG.md` if you need historical context on past bugs or architecture decisions.

### During Session
- **MUST** append to `docs/SESSION_LOG.md` immediately when you discover a non-obvious bug root cause, architecture gotcha, or failed approach. Do NOT wait until end of session — you may forget or the session may end unexpectedly.
- **MUST** update `docs/PROJECT_STATUS.md` after any commit that changes what works or what's broken.

### Session End
- **MUST** update `docs/PROJECT_STATUS.md` with current state (what works, known issues, next steps).
- **MUST** commit and push all doc changes so other machines see them.
- This is not optional. The user should not have to ask for this.

### Housekeeping
- `docs/SESSION_LOG.md` is reference-only — grep it for specific topics, don't read the whole thing.
- When `SESSION_LOG.md` exceeds ~200 lines, move older entries to `docs/SESSION_LOG_ARCHIVE.md` and keep only the last ~2 months.

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
│   ├── search.py                       # Core automation + run_one_class_search()
│   ├── models.py                       # Model implementations
│   ├── contamination.py               # One-class models, run_one_class_cv(), compute_one_class_importances()
│   ├── unified_bayesian.py            # Bayesian optimization (handles one_class task_type)
│   ├── scoring.py                     # Scoring/ranking with one-class metrics
│   ├── model_io.py                    # Save/load with scaler/PCA persistence
│   ├── preprocessing_discovery.py     # Smart preprocessing (one-class path at line ~680)
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
- **One-Class Models**: OneClassSVM, IsolationForest, LOF, EllipticEnvelope, PCA-SIMCA (contamination detection / membership screening)
- **Preprocessing**: SNV, Savitzky-Golay derivatives (1st, 2nd), baseline correction
- **Variable Selection**: UVE, SPA, iPLS, CARS, GA-PLS (note: only 'importance' works for one-class)
- **Calibration Transfer**: Direct Standardization (DS), Piecewise DS (PDS)
- **File Formats**: CSV, Excel, ASD, OPUS, SPC, JCAMP-DX, PerkinElmer

## Development Notes

- Tier (Quick/Standard/Comprehensive) only affects which models are tested
- All hyperparameters are exposed and user-editable
