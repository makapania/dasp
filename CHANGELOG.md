# Changelog

All notable changes to Spectral Predict will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-27

### Added

#### Core Features
- **CSV Input Support**
  - Wide format: first column = ID, remaining columns = wavelengths
  - Long format: automatic detection and pivoting for single-spectrum files
  - Validation for minimum 100 wavelengths and monotonic ordering

- **ASD File Support**
  - ASCII .sig file reader with robust numeric data detection
  - ASCII .asd file reader
  - Binary .asd detection with clear error messages
  - Support for multi-column formats (automatically selects last column as reflectance)
  - Header line skipping for files with metadata

- **Preprocessing Pipeline**
  - Standard Normal Variate (SNV) transformer
  - Savitzky-Golay derivative (1st and 2nd order)
  - Configurable window sizes (7, 19) and polynomial orders
  - Multiple preprocessing combinations: raw, snv, deriv, snv→deriv, deriv→snv

- **Model Ensemble**
  - **Regression**: PLS Regression, Random Forest, MLP
  - **Classification**: PLS-DA, Random Forest, MLP
  - Grid search over hyperparameters:
    - PLS: n_components [2, 4, 6, 8, 10, 12, 16, 20, 24]
    - Random Forest: n_estimators [200, 500], max_depth [None, 15, 30]
    - MLP: hidden layers [(64,), (128, 64)], alpha [1e-4, 1e-3], learning_rate [1e-3, 1e-2]

- **Feature Selection**
  - Variable Importance in Projection (VIP) for PLS models
  - Feature importances for Random Forest
  - Weight-based importances for MLP
  - Automated subset selection: top-20, top-5, top-3 variables

- **Cross-Validation & Metrics**
  - 5-fold CV (configurable)
  - Stratified K-fold for classification
  - Regression metrics: RMSE, R²
  - Classification metrics: Accuracy, ROC-AUC (binary and multiclass)

- **Intelligent Ranking**
  - Composite scoring with simplicity penalty
  - Configurable lambda penalty (default: 0.15)
  - Formula: z(metric) + λ × (LVs/25 + n_vars/full_vars)
  - Lower scores = better models

- **Output & Reporting**
  - CSV results table with all model runs
  - Markdown reports with top-5 models
  - Detailed configuration and performance metrics

#### CLI
- `spectral-predict` command-line interface
- `--spectra` mode for CSV input
- `--asd-dir` mode for ASD directory input
- `--reference` for target variable mapping
- `--target` for single-target prediction
- `--folds` for CV configuration
- `--lambda-penalty` for complexity penalty tuning
- `--outdir` for output directory configuration
- `--asd-reader` flag (auto/python/rs-prospectr/rs-asdreader)

#### Infrastructure
- Complete test suite (30 tests)
- CI/CD with GitHub Actions
  - Linux and Windows testing
  - Python 3.10, 3.11, 3.12 support
  - Black code formatting checks
  - Flake8 linting
  - Package build validation
- Development dependencies: pytest, black, flake8, build, twine
- Optional dependencies: specdal for binary ASD support

#### Documentation
- Comprehensive README with installation and usage examples
- Inline documentation for all functions
- Type hints for better IDE support
- Example commands for common use cases

### Planned (Future Releases)

#### Binary ASD Readers
- **Native Python reader** (stub in `readers/asd_native.py`)
  - Pure-Python binary ASD parser
  - No external dependencies

- **R Bridge** (stub in `readers/asd_r_bridge.py`)
  - Integration with R's asdreader package
  - Integration with R's prospectr package
  - Requires rpy2 and R installation

#### Future Enhancements
- Interactive mode for target selection
- CSV directory batch processing
- Model persistence and reloading
- Feature selection optimization
- Additional preprocessing methods
- Support for additional file formats

## [Unreleased]

### To Be Added
- SpecDAL integration for binary ASD files
- Native Python binary ASD reader
- R bridge implementation
- Interactive CLI mode
- Model export/import functionality
- Additional spectral file formats (SPC, OPUS, etc.)

---

## Version History

- **0.1.0** (2025-01-27) - Initial release with CSV and ASCII ASD support

[0.1.0]: https://github.com/yourusername/deepspec/releases/tag/v0.1.0
