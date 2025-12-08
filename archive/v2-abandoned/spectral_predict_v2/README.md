# Spectral Predict v2

A modern, automated spectral analysis platform for chemometric modeling and prediction.

## Overview

Spectral Predict v2 is a complete redesign of the spectral analysis workflow, featuring:

- **Mode-based interface**: Explore, Build, and Predict modes for intuitive workflows
- **Automated model discovery**: One-click analysis with preset-driven configurations
- **Professional diagnostics**: Pred vs Ref plots, residuals, loadings visualization
- **Advanced tools**: Calibration transfer, interference removal, outlier detection

## Installation

### Requirements

- Python 3.10+
- PySide6
- NumPy, Pandas, SciPy
- scikit-learn
- Matplotlib

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python -m spectral_predict_v2.main
```

Or use the provided batch file:

```bash
./run_gui.bat  # Windows
./run_gui.sh   # Linux/Mac
```

## Quick Start

### 1. Load Data (Explore Mode)

1. Launch the application - starts in **Explore** mode
2. Drag and drop a CSV/Excel file or click "Browse"
3. Select the target column (property to predict)
4. Choose a preset (e.g., "NIR Protein" for grain analysis)
5. Click **Run Auto Analysis**

### 2. Review Results

After analysis completes:
- Results table shows all model combinations ranked by composite score
- Filter by model type, preprocessing, or minimum score
- Pin up to 4 models for side-by-side comparison
- Click "Compare Pinned" for detailed visual comparison

### 3. Build Production Model (Build Mode)

1. Select a model from Explore results (or start fresh)
2. Switch to **Build** mode
3. Adjust configuration if needed
4. Click **Train Model**
5. Review diagnostics (Pred vs Ref, Residuals, Loadings)
6. Save model as `.dasp` file

### 4. Make Predictions (Predict Mode)

1. Switch to **Predict** mode
2. Load a saved model (`.dasp` file)
3. Load new spectral data
4. Click **Predict**
5. Export results to CSV or Markdown report

## Modes

### Explore Mode

Automated model discovery and comparison.

**Features:**
- Preset-driven analysis (NIR Protein, NIR Moisture, Mid-IR Bone, etc.)
- Thoroughness slider (Quick ~2min to Comprehensive ~45min)
- Grid search + Bayesian optimization
- Composite scoring with explainability badges
- Advanced options for power users

### Build Mode

Refine and train production models.

**Features:**
- Import models from Explore results
- Fine-tune preprocessing and hyperparameters
- Cross-validation diagnostics
- Interactive diagnostic plots
- Export to `.dasp` format

### Predict Mode

Apply saved models to new data.

**Features:**
- Load `.dasp` model files
- Wavelength compatibility checking
- Uncertainty estimation
- Applicability domain flagging
- Batch folder processing
- CSV and report export

## Tools

Access via **Tools** menu:

### Data Quality Assessment

- PCA-based outlier detection
- Hotelling T² statistics
- Q-residual (SPE) analysis
- Mahalanobis distance
- Flag or remove outliers

### Calibration Transfer

- Direct Standardization (DS)
- Piecewise Direct Standardization (PDS)
- Transfer model save/load
- Apply to new instruments

### Interference Removal

- MSC (Multiplicative Scatter Correction)
- OSC (Orthogonal Signal Correction)
- DOSC (Direct OSC)
- EPO (External Parameter Orthogonalization)

### Preset Manager

- View built-in presets
- Create custom presets
- Edit/delete user presets
- Share configurations

## Presets

Built-in presets for common applications:

| Preset | Use Case | Preprocessing | Models |
|--------|----------|---------------|--------|
| NIR Protein (Grain) | Wheat, corn, soy protein | SNV, SG1 | PLS, Ridge |
| NIR Moisture | Moisture in grains/feeds | MSC, SNV | PLS |
| NIR Fat/Oil | Oil content in seeds | SNV, SG2 | PLS, ElasticNet |
| NIR Bone | Bone mineral, collagen | SNV, SG1, SG2 | PLS, Ridge |
| Mid-IR Bone | FTIR bone composition | Baseline, SNV | PLS |
| Mid-IR Enamel | Enamel composition | Baseline, MSC | PLS, Ridge |
| Classification | Discrimination tasks | SNV | LDA, RF, SVM |
| Comprehensive | Full exploration | All | All |

## File Formats

### Input Data

- **CSV**: Comma-separated values
- **Excel**: .xlsx, .xls files
- **JCAMP-DX**: Spectroscopy standard format
- **SPC**: Thermo Scientific format
- **OPUS**: Bruker format

### Model Files

- **.dasp**: Spectral Predict model format (joblib-based)
  - Contains: model, preprocessing, wavelengths, training stats
  - Portable across systems with same Python version

### Export Formats

- **CSV**: Raw predictions with uncertainty
- **Markdown**: Formatted report with statistics

## Architecture

```
spectral_predict_v2/
├── main.py              # Entry point
├── ui/                  # GUI Layer (PySide6)
│   ├── app.py           # Main window
│   ├── modes/           # Explore, Build, Predict views
│   ├── tools/           # Tool panels
│   └── widgets/         # Reusable components
├── orchestration/       # Middle Layer
│   ├── state_store.py   # Central state management
│   ├── job_runner.py    # Background execution
│   └── config_manager.py # Presets
└── engine/              # Analysis Engine
    └── api.py           # Clean API to analysis code
```

## Configuration

### User Presets

Saved to: `~/.spectral_predict/presets/`

### Application Settings

Settings are stored in the application state and include:
- Last used directory
- Window geometry
- Recent files
- Theme preference

## API Usage

For programmatic access:

```python
from spectral_predict_v2.engine.api import EngineAPI

# Initialize
engine = EngineAPI()

# Load data
data = engine.load_file("spectra.csv", target_column="protein")

# Train model
trained = engine.train_model(
    X=data.X,
    y=data.y,
    model_type="pls",
    preprocessing="snv",
    n_components=5
)

# Make predictions
predictions, uncertainty, ad_flags = engine.predict(
    X_new, trained, return_uncertainty=True
)

# Save model
engine.save_model(trained, "model.dasp", wavelengths=data.wavelengths)
```

## Testing

Run the test suite:

```bash
# All tests
pytest spectral_predict_v2/tests/ -v

# Specific test file
pytest spectral_predict_v2/tests/test_engine/test_api.py -v

# Integration tests
pytest spectral_predict_v2/tests/test_integration.py -v
```

## Troubleshooting

### Common Issues

**"No module named 'spectral_predict_v2'"**
- Ensure you're running from the project root directory
- Add the project to your PYTHONPATH

**"Wavelength mismatch" error in Predict mode**
- Model was trained on different wavelength range
- Check that new data has same spectral range as training data

**Slow analysis**
- Use "Quick" thoroughness for initial exploration
- Reduce number of preprocessing methods in Advanced settings
- Use fewer cross-validation folds

**Out of memory**
- Reduce dataset size or number of wavelengths
- Use subset selection before full analysis

## License

Proprietary - Internal use only.

## Support

For issues and feature requests, contact the development team or create an issue in the repository.
