# Migration Guide: v1 to v2

This guide helps users transition from Spectral Predict v1 (the original GUI) to v2 (the redesigned application).

## Overview of Changes

### Architecture

| Aspect | v1 | v2 |
|--------|----|----|
| GUI Framework | Tkinter/Custom | PySide6 (Qt) |
| Code Structure | Single large file | Modular (UI/Orchestration/Engine) |
| State Management | Scattered variables | Centralized StateStore |
| Background Tasks | Blocking | Non-blocking threads |
| Configuration | Hardcoded | Preset-based YAML |

### User Interface

| v1 | v2 |
|----|----|
| Tab-based interface | Mode-based interface |
| Manual configuration | Preset-driven workflows |
| Basic results table | Advanced filtering/sorting |
| Limited diagnostics | Full diagnostic plots |

## Workflow Mapping

### Loading Data

**v1:**
1. Click "Load Data" button
2. Select file in dialog
3. Manually configure column mappings
4. Click "Process"

**v2:**
1. Drag and drop file (or click Browse)
2. Auto-detection of format and columns
3. Select target column from dropdown
4. Data loads automatically

### Running Analysis

**v1:**
1. Select preprocessing methods (checkboxes)
2. Select model types (checkboxes)
3. Configure each model's parameters
4. Set cross-validation settings
5. Click "Run Grid Search"
6. Wait (blocking UI)

**v2:**
1. Select a preset (e.g., "NIR Protein")
2. Adjust thoroughness slider (optional)
3. Click "Run Auto Analysis"
4. UI remains responsive during analysis
5. Progress bar shows status

### Viewing Results

**v1:**
- Single table with all results
- Manual scrolling to find best models
- Limited sorting options

**v2:**
- Filterable/sortable results table
- "Why best" badges explain rankings
- Pin models for comparison
- Side-by-side diagnostic plots

### Saving Models

**v1:**
- Custom pickle format
- Manual file naming
- Limited metadata

**v2:**
- Standardized `.dasp` format
- Includes wavelengths, training stats, timestamp
- Version tracking for compatibility

## Feature Mapping

### Preprocessing

All v1 preprocessing methods are available in v2:

| v1 Name | v2 Name | Notes |
|---------|---------|-------|
| Raw | raw | No change |
| SNV | snv | No change |
| 1st Derivative | sg1 | Savitzky-Golay 1st derivative |
| 2nd Derivative | sg2 | Savitzky-Golay 2nd derivative |
| MSC | msc | No change |
| Baseline | baseline | No change |
| Normalize | normalize | No change |

### Models

| v1 Name | v2 Name | Notes |
|---------|---------|-------|
| PLS | pls | No change |
| Ridge | ridge | No change |
| Lasso | lasso | No change |
| ElasticNet | elasticnet | No change |
| Random Forest | rf | Abbreviated |
| SVR | svr | No change |
| LDA | lda | Classification only |
| QDA | qda | Classification only |
| SVM | svm | Classification only |

### Variable Selection

| v1 Name | v2 Name | Notes |
|---------|---------|-------|
| UVE | uve | Unchanged |
| SPA | spa | Unchanged |
| iPLS | ipls | Unchanged |
| VIP | vip | New in v2 |

## Configuration Migration

### v1 Configuration (Python dict)

```python
config = {
    "preprocessing": ["snv", "sg1"],
    "models": ["pls", "ridge"],
    "cv_folds": 10,
    "max_components": 15,
    "bayesian": True
}
```

### v2 Configuration (Preset YAML)

```yaml
name: My Analysis Preset
description: Custom configuration
preprocessing:
  methods: [snv, sg1]
models:
  model_types: [pls, ridge]
  tier: standard
  use_bayesian: true
validation:
  cv_folds: 10
  holdout_fraction: 0.0
```

### Converting v1 Config to v2 Preset

1. Open v2 application
2. Go to Tools > Preset Manager
3. Click "New Preset"
4. Fill in equivalent settings
5. Save with descriptive name

## Model File Migration

### v1 Model Files (.pkl)

v1 saved models as simple pickle files containing the sklearn model object.

### v2 Model Files (.dasp)

v2 uses a richer format including:
- Model object
- Preprocessing configuration
- Wavelengths
- Training statistics
- Metadata (timestamp, version)

### Converting v1 Models

v1 models cannot be directly loaded in v2. To migrate:

1. Load original training data in v2
2. Recreate the model configuration
3. Train in Build mode
4. Save as `.dasp`

If you have many v1 models, contact the development team for a batch conversion script.

## API Changes

### v1 API (Direct Function Calls)

```python
from src.spectral_predict.search import run_grid_search
from src.spectral_predict.preprocess import apply_snv

# Preprocess
X_preprocessed = apply_snv(X)

# Run search
results = run_grid_search(X_preprocessed, y, config)
```

### v2 API (Engine API)

```python
from spectral_predict_v2.engine.api import EngineAPI

engine = EngineAPI()

# Preprocess
X_preprocessed = engine.apply_preprocessing(X, "snv")

# Train single model
trained = engine.train_model(X, y, model_type="pls", preprocessing="snv")

# Or run full analysis
results = engine.run_analysis(X, y, preset="nir_protein")
```

## Frequently Asked Questions

### Can I use my v1 model files in v2?

No, the file formats are incompatible. You'll need to retrain models in v2.

### Do I need to change my data files?

No, v2 supports the same input formats as v1 (CSV, Excel, JCAMP-DX, etc.).

### Are the analysis results the same?

Yes, the underlying analysis engine is unchanged. Results should be identical for the same configuration.

### What happened to the "Batch Mode" tab?

Batch processing is now integrated into Predict mode. Load a model, then use "Process Folder" to batch-process multiple files.

### Where are my custom configurations?

v1 configurations were stored in Python code or ad-hoc files. In v2, create presets via Tools > Preset Manager. They're saved to `~/.spectral_predict/presets/`.

### Can I still run analyses from the command line?

Yes, use the Engine API programmatically:

```python
from spectral_predict_v2.engine.api import EngineAPI

engine = EngineAPI()
# ... your analysis code
```

### How do I report issues?

Create an issue in the repository or contact the development team.

## Getting Help

- **README**: See README.md for general usage
- **Presets**: Built-in presets cover most common workflows
- **Tools**: Access specialized tools via the Tools menu
- **Support**: Contact the development team for assistance
