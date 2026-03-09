# GUI Test Suite for Spectral Predict V1

Automated tests that run features through the GUI to catch integration issues.

## Quick Start

```bash
# Run all GUI tests
pytest tests/gui/ -v -m gui

# Run comprehensive feature tests
pytest tests/gui/test_comprehensive.py -v -s -m comprehensive

# Run with visible window (for debugging)
pytest tests/gui/ -v --visible
```

## Test Files

| File | Purpose |
|------|---------|
| `test_workflows.py` | Basic workflow tests (data loading, regression, classification) |
| `test_comprehensive.py` | Full feature comparison against baseline |
| `harness.py` | GUITestHarness class for test automation |
| `conftest.py` | Pytest fixtures and configuration |
| `run_gui_tests.py` | Standalone runner with interactive mode |

## Test Categories

### Basic Workflows (`test_workflows.py`)
```bash
pytest tests/gui/test_workflows.py -v -s
```
- Data loading (regression and classification)
- Quick PLS analysis
- Results validation
- Smoke tests

### Comprehensive Tests (`test_comprehensive.py`)

All tests compare against baseline: PLS, Ridge, ElasticNet with SPXY 8-sample holdout.

#### All Model Types
```bash
pytest tests/gui/test_comprehensive.py::TestAllModelsViaGUI -v -s
```
- RandomForest, XGBoost, LightGBM, CatBoost, SVR, MLP

#### Variable Selection Methods
```bash
pytest tests/gui/test_comprehensive.py::TestVariableSelectionViaGUI -v -s
```
- UVE, SPA, iPLS, CARS

#### Calibration Transfer (requires tablet data)
```bash
pytest tests/gui/test_comprehensive.py::TestCalibrationTransfer -v -s
```
- DS, PDS, TSR, CTAI, NS-PFCE, JYPLS-inv
- Uses: `C:\Users\sponheim\Desktop\LS Tablet` (master) and `FS Tablet` (slave)

#### Interference Removal
```bash
pytest tests/gui/test_comprehensive.py::TestInterferenceRemoval -v -s
```
- MSC, OSC, DOSC, GLSW

#### Model Save/Load
```bash
pytest tests/gui/test_comprehensive.py::TestModelSaveLoad -v -s
```
- Save/load PLS model
- Save/load with preprocessing
- Multiple model types (PLS, Ridge, RandomForest)

#### Prediction Workflow
```bash
pytest tests/gui/test_comprehensive.py::TestPredictionWorkflow -v -s
```
- Predict new samples
- Predict with preprocessing
- Batch prediction
- Confidence estimation
- Full workflow: train -> save -> load -> predict

## Command Options

| Option | Description |
|--------|-------------|
| `--visible` | Show GUI window during tests |
| `--data-path PATH` | Use custom data folder (default: `example/`) |
| `-v` | Verbose output |
| `-s` | Show print statements |
| `-k "keyword"` | Run tests matching keyword |

## Examples

```bash
# Run only regression tests
pytest tests/gui/test_workflows.py::TestRegressionAnalysis -v -s

# Run tests matching "pls"
pytest tests/gui/test_comprehensive.py -v -s -k "pls"

# Run with custom data
pytest tests/gui/ -v --data-path="C:/mydata/spectra"

# Skip slow tests
pytest tests/gui/ -v -m "gui and not slow"

# Interactive mode (step-by-step)
python tests/gui/run_gui_tests.py --interactive

# Quick smoke test
python tests/gui/run_gui_tests.py --smoke
```

## Output Legend

```
[+] OUTPERFORMS baseline (R2 > baseline + 0.01)
[=] MATCHES baseline (within 0.01)
[-] UNDERPERFORMS baseline (R2 < baseline - 0.01)
```

## Test Data

Default: `example/BoneCollagen.csv` with 49 samples
- `%Collagen` - regression target (0.9-22.1%)
- `CollagenCat` - classification target (Low/Medium/High)
