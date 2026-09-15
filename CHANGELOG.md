# Changelog

All notable changes to Spectral Predict will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Note: the entries between `0.1.0` (early 2025) and `0.5.0b1` (April 2026)
> were not maintained in this file. The git history under
> `git log v0.1.0..0.5.0b1` (or commit `bbf7766` for the b1 cut) is the
> authoritative record for that period.

## [0.5.0b3] - Unreleased

### Added

- Results-row rebuild helpers on the declared composition surface
  (`docs/AGENT_COMPOSITION.md` §8b): `models.parse_row_params`,
  `models.estimator_params_from_row`, `models.plsda_head_kwargs` and
  `preprocess.preprocessing_config_from_row`.

- **T-51 PR A** — opt-in extra hyperparameter axes for the unified Bayesian search.
  `run_unified_bayesian` gains `enabled_extra_axes`, `search_space` and
  `n_startup_trials`. The new `spectral_predict.search_spaces` module provides
  `AxisSpec`, `BundleSpec`, `ExtraAxesConfigError` and the curated `BUNDLES` registry,
  which is empty until PR B/C. Bundles may open only hyperparameters the default space
  pins to a single value. Collisions and malformed bundles raise before any study is
  created. With nothing enabled, the search, its TPE trajectory and its study names are
  unchanged. `n_startup_trials` now also survives the T-41 in-memory→SQLite
  auto-migration. See `docs/AGENT_COMPOSITION.md` §7b.
- **T-51 PR B** — curated supervised bundles in `search_spaces.BUNDLES`, all off by
  default: `rf_features`, `xgb_regularization`, `xgb_child`, `xgb_sampling`,
  `lgbm_regularization`, `lgbm_sampling`, `lgbm_child` (including `min_split_gain`),
  `catboost_sampling` (also sets `bootstrap_type='Bernoulli'`, which `subsample` needs
  for multiclass), `svm_gamma` (written only on RBF-kernel trials), `mlp_activation` and
  `plsda_head` (the PLS-DA logistic head's `C`, stored as `lr__C`). Enable them from
  Python with `run_unified_bayesian(..., enabled_extra_axes=(...))`; there is no GUI
  control yet (PR D). Enabled bundles give the run its own study name. Sampled values are
  stored in `Params` as estimator parameters and survive rebuild, Tab 7 refit,
  save/load and export. With no bundle enabled, searches and study names are unchanged.
  `apply_extra_axes` now also rejects two axes that share an Optuna name or key when
  `resolve_bundles` is bypassed.

### Fixed

- **Ensembles trained from Bayesian results now use the tuned hyperparameters.**
  Ensemble model reconstruction discarded every `model__*` key in a row's `Params`, and
  Bayesian rows store all estimator params under that prefix, so each base model trained
  with defaults (e.g. RandomForest `n_estimators`/`max_features`, SVM `C`/`gamma`, MLP
  `activation`, Ridge `alpha`, boosting learning rates and regularisation). The prefix is
  now stripped through the new shared `models.estimator_params_from_row`, which the
  validation rebuild also uses. Grid-search rows (bare keys) are unaffected, except
  that PLS `n_components` above 10 is no longer clipped to 10 in ensembles.
  **Ensemble results built from Bayesian rows change.**
- **Ensemble models are rebuilt with the row's full preprocessing.** Only `snv`,
  `snv_deriv` and `deriv_snv` rows got preprocessing. `deriv` rows (no derivative),
  every `+`-affixed name (`raw+autoscale`, `als+snv`) and the `Autoscale`, baseline and
  smoothing columns were ignored, and the per-model scaler was added even when the
  search had autoscaled. Wavelength subsets of grid/Bayesian rows were taken *before*
  preprocessing. The ensemble rebuild now shares `preprocess.preprocessing_config_from_row`
  with the validation rebuild, skips the per-model scaler for autoscaled rows, and takes
  subsets after preprocessing, as the search does. `raw`/`snv` rows are now plain
  Pipelines rather than GUI preprocessing wrappers, so ensembles of them refit base
  models per CV fold (the ensemble default). Legacy `sg1`/`sg2`, `deriv1`-style and GA
  preprocessing names keep their old path. Validation rebuild: a NaN `smoothing` cell
  (mixed results tables) no longer turns smoothing on, and a NaN `PreprocessBase` falls
  back to `Preprocess`. **Ensemble results change.**
- **NSGA-II rows rebuild with their hyperparameters.** They store `Params` as a dict,
  and validation rebuild and ensemble reconstruction only parsed strings, so both used
  defaults. New shared `models.parse_row_params` accepts either.
- `plsda_head_kwargs` coerces `lr__random_state=42.0` to `42` and `'None'` to `None`,
  and raises `ValueError` on values `LogisticRegression` would reject.
- **PLS-DA heads rebuilt from a row keep the search's seed and class weighting.**
  Validation rebuild and ensemble reconstruction forced `random_state=42`; ensemble
  reconstruction also dropped `class_weight`. Both now restore the row's
  `lr__random_state` and `lr__class_weight` (new `models.plsda_head_kwargs`), so a
  search run with another seed and a stochastic solver (`saga`) refits the same head.
  Rows without a recorded seed keep 42.
- **Ensemble refits of CatBoost models saved before the `catboost_info/` fix** no longer
  write that directory. Per-fold clones get `allow_writing_files=False`, found by walking
  params, attributes and step lists (the GUI wrappers' `get_params(deep=True)` is
  shallow), so CatBoost nested in a Pipeline, a GUI wrapper or a VotingRegressor is
  covered. Previously the refit failed in an
  unwritable cwd and the model silently got NaN out-of-fold predictions.
- **GUI NameErrors.** The GUI module had no `logger`, so Model Development refit crashed
  when the task radio disagreed with the saved result's Task (and in two other warning
  branches); it now logs to `spectral_predict.gui`, which reaches `dasp.log`. The
  learning-curve error callback referenced the except-bound `e` after the block ended
  and raised instead of showing the error.

- **CatBoost no longer writes `catboost_info/`.** Every CatBoost fit wrote a
  training-log directory into the current working directory, so fits failed with
  `Can't create train working dir: catboost_info` when the cwd was unwritable (an
  install under Program Files), concurrent fits could race on it (seen in CI), and it
  littered wherever the app ran. Every CatBoost construction (`get_model`,
  `build_model`, the grid, NSGA-II, preprocessing discovery, diagnostics validation
  curves) now passes `allow_writing_files=False` via `models.CATBOOST_RUNTIME_PARAMS`.
  It is a runtime kwarg, not model identity: result-row `Params`, Bayesian trial
  `model_params`, fit fingerprints and study names are unchanged. Exported Python
  scripts now include `'allow_writing_files': False` in the CatBoost `model_params`.
  Models saved before this fix still carry the default if refit after loading.

- **T-51 PR B0** — PLS-DA models rebuilt from a results row now keep the tuned
  LogisticRegression head (`C`, `solver`, `max_iter`) and the PLS transformer settings.
  Validation rebuild used `C=1.0` for every current PLS-DA row. Model Development
  refit and exported scripts lost the head for rows that spell it `lr_C` / `lr_solver` /
  `lr_max_iter`, and ensemble training ignored it for every row. Refits of grid searches
  whose `plsda_lr_C_list` differs from 1.0 therefore change, and now match the search's
  head params. Ensemble training re-applying `class_weight` and the head seed is covered
  by the PLS-DA head entry above.
  `build_model('PLS-DA', params)` no longer raises on `lr_*` or `pls__*` keys.
  Search-time scores, the default Bayesian search and study names are unchanged; no
  version bump.

- **T-51 step 1** — classification SVM is now StandardScaler-wrapped in grid search,
  Bayesian search, validation rebuild and Model Development refit. The scale-sensitive
  sets listed `'SVC'`, which no model name matches; the registered family is `'SVM'`, so
  every classification SVM had been fit on unscaled spectra. Exported scripts already
  scaled it. **This changes SVM classification results.** `__version__` is part of
  every Optuna study name, so persisted Bayesian studies for **all** models start fresh
  after upgrading. The old studies stay on disk.

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

[0.1.0]: https://github.com/makapania/dasp/releases/tag/v0.1.0
