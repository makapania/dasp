# Spectral Predict - Agent Guide

> **IMPORTANT FOR AI AGENTS**: Do NOT read files in `archive/` unless explicitly instructed.
> Archived docs contain obsolete information that will cause incorrect responses.

---

## CURRENT MODE: Feature Recovery (2025-12-20)

We are recovering functionality after a reset. Follow this process:

### Recovery Process
1. **User provides a feature** they want to restore/implement
2. **Check `recovery_blobs/`** for relevant code (use INDEX.txt to find relevant blobs)
3. **Implement the fix** based on user's requirements + blob reference code
4. **Test and verify** the implementation works
5. **Move on** to the next feature only when current one is complete

### Recovery Resources
- `recovery_blobs/INDEX.txt` - Summary of all recovered blobs
- `recovery_blobs/blob_*.txt` - Actual recovered code/docs
- `backup_2025-12-20/` - December 16-20 work (new modules, tests, scripts)
- `RECOVERY_TODO.md` - Checklist of features to restore

### Priority Order
1. **User-specified features first** - Whatever the user asks for
2. **December 16-20 work second** - Only after user features are done

### Important Notes
- Work on ONE feature at a time
- Do not jump ahead to other features
- Always check blobs before implementing from scratch
- Commit working code before moving to next feature

---

## Project Overview

Spectral Predict automates spectral modeling: load spectra, test preprocessing × model combinations with cross-validation, get ranked results.

## Active Versions

| Version | Entry Point | Framework | Location |
|---------|-------------|-----------|----------|
| **V1** | `python spectral_predict_gui_optimized.py` | Tkinter | `src/spectral_predict/` |
| **V3** | `python -m spectral_predict_v3.main` | Dear PyGui | `spectral_predict_v3/` |

**V2 is ABANDONED** - moved to `archive/v2-abandoned/`, do not reference.

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
├── spectral_predict_v3/                # V3 (Dear PyGui)
│   ├── main.py                         # Entry point
│   ├── core/                           # Backend (forked from V1)
│   │   ├── search.py
│   │   ├── models.py
│   │   └── ...
│   └── ui/                             # Dear PyGui UI
│       ├── app.py                      # Main window
│       ├── components/                 # UI components
│       └── panels/                     # Feature panels
├── archive/                            # OLD DOCS - DO NOT READ
└── example/                            # Example data
```

## Key Features

- **Models**: PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, SVM, MLP
- **Preprocessing**: SNV, Savitzky-Golay derivatives (1st, 2nd), baseline correction
- **Variable Selection**: UVE, SPA, iPLS, CARS, GA-PLS
- **Calibration Transfer**: Direct Standardization (DS), Piecewise DS (PDS)
- **File Formats**: CSV, Excel, ASD, OPUS, SPC, JCAMP-DX, PerkinElmer

## V3 Current Status

See `spectral_predict_v3/V3_STATUS.md` for latest development status.

**Completed:**
- Full backend (preprocessing, models, search, variable selection)
- Import panel, data grid, spectra plot, PCA plot
- Build panel with tier selection and hyperparameter exposure

**In Progress:**
- Hyperparameter editing functionality
- Predict panel
- Export functionality

## Development Notes

- V3 is standalone - does NOT import from V1 (forked code)
- Tier (Quick/Standard/Comprehensive) only affects which models are tested
- All hyperparameters are exposed and will be user-editable
