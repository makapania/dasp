# Changelog

All notable changes to Spectral Predict will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Note: the entries between `0.1.0` (early 2025) and `0.5.0b1` (April 2026)
> were not maintained in this file. The git history under
> `git log v0.1.0..0.5.0b1` (or commit `bbf7766` for the b1 cut) is the
> authoritative record for that period.

## [0.5.0b2] - 2026-05-03

Second beta of the 0.5.0 cycle. Bug-fix-and-observability batch on top of
`0.5.0b1`, plus one user-visible behavior change (T-19 Auto mode).

### Added

- **T-19** — model-native imbalance handling exposed through the Search tab,
  including an `Auto` mode that resolves to a sensible per-model default at
  runtime (instead of forcing the user to pick one). Boosting paths thread
  `sample_weight` correctly across resamplers.

### Changed

- **T-47** — Bayesian persistence default flipped from `"never"` to `"auto"`.
  Searches are now resumable out of the box; users get the recovery path
  without having to opt in.
- **T-14 / T-14b** — every version-displaying surface (report footer,
  exported-script header, exported-notebook metadata, GUI title bar,
  in-canvas version label, PyInstaller `version_info.txt`, Inno Setup
  `MyAppVersion`, build script `VERSION`) now derives from the canonical
  `spectral_predict.__version__`. Bumping the version in one place updates
  every artefact in lockstep. Regression tests pin the contract.

### Fixed

- **T-06 / T-06b** — canonical Araújo-2001 SPA enumeration; parallelised seed
  loop via joblib threading.
- **T-21** — hides x-unit Convert button in cases that produced a non-uniform
  wavelength grid for Savitzky–Golay derivatives.
- **T-11** — pause/resume hardening, Optuna SQLite storage, on-disk run logs,
  study-name fingerprint completeness, narrowed import catches.
- **T-29** — replaced bare `except:` in scoring with `except Exception` and
  warning emission, so silent metric failures surface in the run log.
- **T-30** — removed leftover `[DEBUG]` and `[PLS-DA DEBUG]` `print()` calls
  from `search.py` (`calibration_transfer.py` and `nsga2_search.py` triage
  follow as T-30b).
- **T-32** — corrected `y_train_for_model` threading through resampler +
  `sample_weight` path (boosting models on imbalanced classification).
- **T-38** — deleted dead preprocessing modules and a dead GUI flag.
- **T-42 / T-43 / T-44** — sidecar metadata correctness: write-path plumbing,
  resume restore validation indices, n_trials variable typo fix,
  task_type sibling phantom hasattr.
- **T-45** — wired file handler so module `logger.warning` lands on disk;
  CLI bypass + reload dedup follow-ups closed.
- **T-46** — surfaced `_apply_wal_pragmas` return value at both call sites.
- **T-47** — fix-of-fixes for the `auto` default flip (DeepSeek MEDIUM + 2
  LOWs).
- **T-49** — persisted validation indices on resume (correctness blocker).
- **T-50** — auto-cleanup of stale Optuna SQLite trial archives at app
  startup; configurable retention is queued as T-50b.

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
