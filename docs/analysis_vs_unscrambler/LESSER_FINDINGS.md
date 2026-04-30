# Spectral Predict: Lesser Findings — Complete Audit

> **Date**: 2026-04-29  
> **Method**: 3 parallel specialized agents (code hygiene, UX/QA, correctness)  
> **Scope**: Everything NOT in the Top 12 Blockers but still worth fixing  
> **Companion to**: `MASTER_ANALYSIS.md`, `VALIDATION_SUPPLEMENT.md`

---

## Executive Summary

Beyond the 12 commercial blockers, the codebase contains **~200+ lesser issues** that collectively create a "death by a thousand cuts" impression. These are organized into five categories:

| Category | Count | Theme |
|----------|-------|-------|
| **Silent Correctness Bugs** | 11 | Wrong results that users trust |
| **UX Polish & Inconsistency** | ~150+ | Terminology, feedback, dialogs, buttons |
| **Code Hygiene** | 36 | Dead code, missing docstrings, imports |
| **Performance** | 8 | Inefficient pandas, redundant copies |
| **Robustness / Edge Cases** | 14 | Encoding, division by zero, validation gaps |

**Fixing the P1/P2 items in this list (~40 hours of work) would raise perceived quality from "student project" to "professional beta."**

---

## Part 1: Silent Correctness Bugs

These are the most dangerous issues — the app doesn't crash, but produces **wrong results** that users act upon.

### 1.1 Binary Specificity Computes Recall of Negative Class, Not TNR
- **File**: `src/spectral_predict/scoring.py:344-348`
- **Bug**: `fp = cm[0, 1]` is false negatives for the positive class. True specificity (TNR) requires `fp = cm[1, 0]`.
- **Impact**: Binary classification models appear to have better specificity than they do. In imbalanced spectral classification, a poor model can look excellent.
- **Fix**: One line — change `cm[0, 1]` to `cm[1, 0]`.
- **Severity**: **HIGH**

### 1.2 VIP Formula Uses Total Y Variance Instead of Per-Component Explained Variance
- **File**: `src/spectral_predict/models.py:1738`
- **Bug**: `ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)` uses global `var(y)` instead of per-component `y_loadings_`.
- **Impact**: VIP rankings are distorted. Wavelength selection based on VIP retains wrong variables.
- **Fix**: Replace with canonical Wold (2001) formula using `y_loadings_`.
- **Severity**: **MEDIUM-HIGH**

### 1.3 SPA is Completely Deterministic Despite `n_random_starts` Parameter
- **File**: `src/spectral_predict/variable_selection.py:400-408`
- **Bug**: `n_random_starts=10` loops but always picks the same first variable via `np.argmax(initial_corrs)`. `random_state` is unused.
- **Impact**: Users believe they're getting robust multi-start optimization. They're not.
- **Fix**: Use `rng.choice()` for random starts, respecting `random_state`.
- **Severity**: **MEDIUM**

### 1.4 Ensemble OOF Predictions Use Preprocessor Fit on Full Dataset
- **File**: `src/spectral_predict/ensemble.py:315-316, 571-572, 818-819`
- **Bug**: Out-of-fold predictions for ensemble weight computation apply a preprocessor that was fitted on the FULL training data, including validation fold information.
- **Impact**: Ensemble weights are computed on optimistically biased OOF predictions. Can inflate R² by 0.01–0.03.
- **Fix**: Refit preprocessor per fold or document limitation.
- **Severity**: **MEDIUM**

### 1.5 Preprocessing Discovery Computes Importances on Full Data Before CV
- **File**: `src/spectral_predict/preprocessing_discovery.py:217, 235, 297`
- **Bug**: Feature importances used to rank preprocessing configs are computed by fitting on the **full dataset**.
- **Impact**: Preprocessing selection is optimistically biased. The "discovered" preprocessing performs worse on truly held-out data.
- **Fix**: Compute importances within CV folds or use nested CV.
- **Severity**: **MEDIUM**

### 1.6 PDS Window Arithmetic Fails for Even Window Sizes
- **File**: `src/spectral_predict/calibration_transfer.py:193-221`
- **Bug**: `half_window = window // 2` for even `window` (e.g., 10) gives `half_window=5`. Actual window becomes 11 columns, but `B` is allocated for 10. Assignment overflows.
- **Impact**: PDS transfer matrix corruption or crash for even window sizes.
- **Fix**: Enforce odd window or adjust arithmetic.
- **Severity**: **MEDIUM**

### 1.7 PLS Component Grid Can Exceed CV Fold Training Samples
- **File**: `src/spectral_predict/models.py:842-843`
- **Bug**: Grid includes `n_components` up to `min(n_features, max_n_components)` but does NOT clamp by `n_samples`. A 20-sample dataset with 5-fold CV has 16-sample training folds — components 17-20 are invalid.
- **Impact**: Silent PLS failures during grid search, missing results.
- **Fix**: Clamp by `min(n_features, n_samples_train, max_n_components)`.
- **Severity**: **MEDIUM**

### 1.8 One-Class UVE Prefilter Trained on Binary Labels Including Outliers
- **File**: `src/spectral_predict/search.py:5649-5655`
- **Bug**: UVE (a regression method) is called with binary `y_oc` (+1/-1) that includes ALL samples (inliers + outliers). UVE selects variables that distinguish outliers, not variables stable within the inlier class.
- **Impact**: One-class variable selection contradicts the one-class philosophy. PCA-SIMCA / OCSVM performance degraded.
- **Fix**: Use inlier-only y or skip UVE for one-class.
- **Severity**: **MEDIUM**

### 1.9 CARS Tree-Mode Weight Update Biases Toward Unselected Variables
- **File**: `src/spectral_predict/variable_selection.py:1519-1522, 1549`
- **Bug**: Selected variables get tiny normalized weights (~0.01 each), but unselected variables keep weight ~1.0. After normalization, unselected variables have much higher sampling probability.
- **Impact**: CARS oscillates between noise and signal instead of converging. Unstable selected variable sets.
- **Fix**: Use softmax/exponential weighting consistent with standard CARS.
- **Severity**: **MEDIUM**

### 1.10 SNV Exact Equality Guard Misses Near-Zero Std
- **File**: `src/spectral_predict/preprocess.py:43`
- **Bug**: `stds[stds == 0] = 1.0` uses exact equality. Near-constant spectra with tiny non-zero std are divided by near-zero, producing extreme values.
- **Impact**: Numerical artifacts propagate through derivatives and models.
- **Fix**: Use a tolerance threshold (e.g., `stds < 1e-12`).
- **Severity**: **LOW**

### 1.11 Scoring Metrics Silently Return 0.0 on Failure
- **File**: `src/spectral_predict/scoring.py:509, 517, 535`
- **Bug**: Bare `except:` catches metric computation failures and returns 0.0/None. Single-class CV folds produce `f1=0.0` instead of an informative error.
- **Impact**: Models appear terrible when the issue is fold stratification, not model quality.
- **Fix**: Catch specific exceptions, return NaN with warning.
- **Severity**: **MEDIUM**

---

## Part 2: UX Polish & Inconsistency

These don't crash the app but make it feel unprofessional. There are **~150+ distinct issues** in this category.

### 2.1 Terminology Inconsistency (CRITICAL for credibility)

The same concept has 4+ names across the UI:

| Concept | Names Used | Locations |
|---------|-----------|-----------|
| A single data row | "Specimen", "Sample", "Spectrum" | 209+ "specimen" matches, thousands for "sample/spectrum" |
| The Y variable | "Target", "Y variable", "Response", "Reference" | Import tab, Explore tab, Preprocessing, Data Management |
| Data loading tabs | "Import & Preview" vs "Data Management" | Two top-level tabs with overlapping purpose |
| Model config | "Analysis Configuration" containing "Model Config" | Parent/child name collision |

**Fix**: Standardize on ONE term per concept globally.

### 2.2 "Check Console for Details" in a Windowed App (CRITICAL)

**9 instances** tell users to check the console. In the PyInstaller `--windowed` bundle, stdout is invisible.

- `spectral_predict_gui_optimized.py:17148, 17224, 17398, 17562, 32615, 35203, 36682, 38019, 42643`
- Same issue in `src/spectral_predict/report.py:39`

**Fix**: Log to a file and offer a "View Details" button.

### 2.3 Generic "Error" Title on All Message Boxes (CRITICAL)

**161** `messagebox.showerror("Error", ...)` calls. Every error dialog says just "Error."

**Fix**: Use contextual titles: "Import Failed", "Validation Error", "Save Error".

### 2.4 Progress Indicators (CRITICAL)

- **Zero progress bars** on the Analysis Progress tab. Only ONE `ttk.Progressbar` exists in the entire 57K-line GUI (Prediction tab).
- No busy cursor during file loading (only 3 places set `cursor='wait'`).
- "Run Outlier Detection" button stays enabled during computation — user can click multiple times.

### 2.5 Dialog Window Management (HIGH)

- **15+ dialogs** open at OS-default position (often top-left). Only ONE dialog properly centers.
- **No window size/position persistence** — app always starts maximized.
- Dialogs lack minimum size constraints — users can resize to 0x0.
- Some dialogs are not modal when they should be.

### 2.6 Error Messages Blame the User (HIGH)

**184** "Please ..." messages. UX anti-pattern:
- "Please load data first" (30+ variants)
- "Please check installation" (no explanation of WHAT to install)
- "Please check console for details" (invisible in windowed build)

**Fix**: Rewrite to be actionable: "Go to Import & Preview tab to load data" instead of "Please load data first."

### 2.7 Inconsistent Button Styling (HIGH)

- **3 different arrow styles** for Run buttons: `\u25b6`, `▶`, `▶️`
- **Emoji inconsistency**: `➕ Add Source`, `💾 Save Config`, `📂 Load Config` — emoji may not render on corporate Windows builds
- **Mix of `tk.Button` and `ttk.Button`** in same toolbar (raised 3D vs flat themed)
- Inconsistent verb usage: "Export" vs "Save" vs "Export to" vs "Save as"

### 2.8 Inconsistent Export Formats (HIGH)

| Issue | Count | Detail |
|-------|-------|--------|
| CSV `index=` parameter | 6 True, 6 False | Some exports include unnamed index column, others don't |
| CSV `sep=` parameter | Never set | On European Windows, locale may use `;` |
| Excel `engine=` | Inconsistent | Some specify `xlsxwriter`, others don't |
| File dialog filters | Mixed | `.dasp;*.pkl` vs `*.dasp` vs no filter |

### 2.9 Missing Keyboard Shortcuts & Accessibility (CRITICAL)

- **Zero Alt-key accelerators** (`&File`, `&Load Data`) — screen readers and keyboard users blocked
- **Tab order never set** — jumps unpredictably across frames
- **Emoji in tab names break screen readers** — "🗃️ Data Management" announced as "card file box Data Management"
- **No focus indicator** on custom-styled buttons

### 2.10 Magic Numbers Without Explanation (HIGH)

| Value | Location | Why This Number? |
|-------|----------|-----------------|
| `validation_top_n = 700` | `gui:2986` | No explanation. Spinbox goes to 9999. |
| `smart_preprocess_n_top = 7` | `gui:3194` | Not explained in tooltip. |
| `explore_top_n = 250` | `gui:6991` | Why 250 wavelengths? |
| `min_wavelengths = 100` | `io.py:140, 1864` | Rejects valid datasets (e.g., 80-wavelength ROI) |

### 2.11 No Help System (HIGH)

- No Help menu, no F1 binding, no "?" button, no link to user guide.
- For a 14-tab professional application, this is unacceptable.

### 2.12 Backend Prints Invisible to GUI Users (HIGH)

- `src/spectral_predict/io.py` prints 100+ warnings to stdout (e.g., "⚠️ WARNING: Found duplicate sample IDs")
- `src/spectral_predict/search.py` prints 268+ progress messages
- All invisible in PyInstaller windowed build

**Fix**: Return structured warnings so GUI can display them; use progress callbacks.

---

## Part 3: Code Hygiene

### 3.1 Dead / Commented-Out Code

| Location | Lines | Detail |
|----------|-------|--------|
| `search.py` | ~2130-2132 | Commented-out `import traceback` block |
| `gui.py` | 41282-41491, 42061+ | Large blocks of commented-out Instrument Lab code |
| `interference.py` | 564-600 | Duplicate EPO stub class |

### 3.2 Missing Docstrings

**`search.py`** (5,516 lines, public API functions with no docstrings):
- `_get_edge_zone_size` (~173)
- `_apply_edge_mask_to_data` (~200)
- `_supports_sample_weight` (~244)
- `_needs_resampling_pipeline` (~257)
- `_rebuild_model_from_row` (~305)
- `_oc_extract_defaults` (~4931)
- `_build_ocsvm_custom_grid`, `_build_if_custom_grid`, etc. (~4939-5024)

**`io.py`**:
- `_is_likely_reference_csv` (~174)
- `_rename_duplicate_ids` (~350)
- `detect_combined_format` (~1092)
- `identify_wavelength_columns` (~1132)

**`ensemble.py`**:
- `extract_preprocessor_config` (~1265)

**`models.py`**:
- `build_model` omits `one_class` task_type documentation
- `get_model_grids` omits parameter details for classification/one-class

**Templates** (`src/spectral_predict/templates/*.py`):
- None have module docstrings explaining they are templates (not runnable code).

### 3.3 Unused Imports

| File | Import | Line |
|------|--------|------|
| `diagnostics.py` | `sys` | 17 (never referenced directly) |
| `models.py` | `logging` | 3 (imported but no logger instantiated) |

### 3.4 Missing `__all__` Exports

No `__all__` defined in: `models.py`, `search.py`, `ensemble.py`, `scoring.py`, `baseline.py`, `preprocess.py`, `variable_selection.py`.

### 3.5 `open()` Without Explicit Encoding

Dozens of `open()` and `pd.read_csv()` calls omit `encoding`, relying on platform defaults (cp1252 on Windows, UTF-8 on Linux/Mac). International instrument files will be corrupted.

Affected files: `report.py`, `calibration_transfer.py`, `library_search.py`, `data_management.py`, `instrument_profiles.py`, `io.py`.

### 3.6 Thin Module Docstrings

`report.py`: single-line docstring (`"""Report generation functions."""`) doesn't mention Markdown-only output, no figures, stale version string.

---

## Part 4: Performance

### 4.1 Redundant DataFrame Copies in `search.py`
- **Location**: `search.py:5404, 5420, 5587-5588`
- **Issue**: Triple `.copy()` of preprocessed data. For 50k x 2k datasets, this wastes significant memory.
- **Fix**: Remove redundant copies.
- **Time**: 20 min

### 4.2 `DataFrame.iterrows()` in `ensemble.py`
- **Location**: `ensemble.py:1446, 1522, 1543`
- **Issue**: `iterrows()` scans result DataFrames. For 10,000+ rows, much slower than `itertuples()` or vectorized ops.
- **Fix**: Refactor to vectorized pandas.
- **Time**: 30 min

### 4.3 `DataFrame.iterrows()` in `io.py` ASCII Reader
- **Location**: `io.py:1765`
- **Issue**: Python-level loop over thousands of ASCII files.
- **Fix**: Vectorized read.
- **Time**: 20 min

### 4.4 `pd.to_numeric` via `DataFrame.apply()` in `io.py`
- **Location**: `io.py:1490, 2936`
- **Issue**: `X.apply(pd.to_numeric)` instead of vectorized `pd.to_numeric(X)`.
- **Fix**: Remove apply wrapper.
- **Time**: 10 min

### 4.5 Per-Sample Interpolation in `calibration_transfer.py`
- **Location**: `calibration_transfer.py:68-71`
- **Issue**: Creates separate `interp1d` object per sample in loop.
- **Fix**: Vectorized `np.interp` or batched `interp1d`.
- **Time**: 20 min

### 4.6 Per-Sample Baseline Correction Loops
- **Location**: `baseline.py:91-93, 169-191`
- **Issue**: `BaselineRubberBand`, `BaselineALS`, `BaselinePolynomial` all loop over samples in Python.
- **Fix**: Research vectorized ALS (sparse matrix batch ops).
- **Time**: 1-2 hours

### 4.7 `pd.Series.apply()` in `_normalize_mixed_type_labels`
- **Location**: `search.py:46`, `gui.py:93`
- **Issue**: `labels.apply(_norm)` for large label sets.
- **Fix**: Vectorized numpy string ops.
- **Time**: 15 min

### 4.8 Lambda Formatting in `report.py`
- **Location**: `report.py:130, 132`
- **Issue**: `apply(lambda x: f"{x:.4f}")` on small table (negligible impact but poor example).
- **Fix**: Use `.map()` or vectorized formatting.
- **Time**: 10 min

---

## Part 5: Robustness / Edge Cases

### 5.1 `pd.read_csv()` Without Encoding Detection
- **Location**: `io.py:77, 197, 417, 1397, 3304`
- **Issue**: Will raise `UnicodeDecodeError` on non-UTF-8 files (Latin-1, cp1252, shift_jis).
- **Fix**: Add encoding detection with `chardet` or try/except fallback.
- **Severity**: P1

### 5.2 Division by Zero in `compute_vip` When `y` is Constant
- **Location**: `models.py:1738-1749`
- **Issue**: `ssy_total = 0` for constant y → `vip_scores = inf`.
- **Fix**: Guard with warning.
- **Severity**: P1

### 5.3 Silent Data Modification in SNV for Constant Spectra
- **Location**: `preprocess.py:42-43`
- **Issue**: `stds[stds == 0] = 1.0` silently returns original spectrum for constant rows. No warning that some spectra were skipped.
- **Fix**: Log warning when constant spectra detected.
- **Severity**: P2

### 5.4 Tiny Epsilon Guards Instead of Proper Zero-Checks
- **Locations**: `diagnostics.py:54`, `variable_selection.py:382`, `ensemble.py:180`
- **Issue**: Hardcoded `1e-10` epsilon. If true value is smaller, produces wildly scaled outputs.
- **Fix**: Explicit zero-checks with warnings.
- **Severity**: P2

### 5.5 Empty DataFrame Handling Gap in `read_reference_csv`
- **Location**: `io.py:413-436`
- **Issue**: No guard against empty file or header-only file.
- **Fix**: Add pre-check.
- **Severity**: P3

### 5.6 `simpledialog` Inputs Not Validated
- **Locations**: `gui.py:10259, 10351, 17706, 22021, 22027, 22041, 30758, 30887, 46194`
- **Issue**: Accept empty strings, path separators, extremely long names.
- **Fix**: Add validation wrappers.
- **Severity**: P2

### 5.7 Hardcoded Minimum Wavelength Count of 100
- **Location**: `io.py:140, 1864`
- **Issue**: Rejects valid datasets (80-wavelength ROI, 64-band sensor).
- **Fix**: Lower to 10 or make configurable.
- **Severity**: P2

### 5.8 `open()` Without Encoding in Report and Model I/O
- **Locations**: `report.py:41, 143`, `calibration_transfer.py:307, 326, 351`, `library_search.py:522, 539`, `data_management.py:737, 744`, `instrument_profiles.py:360, 373`
- **Issue**: On Windows with cp1252, non-ASCII characters in model names are corrupted.
- **Fix**: Add `encoding='utf-8'` everywhere.
- **Severity**: P2

### 5.9 No Input Validation on User-Provided Numbers
- **Location**: GUI spinboxes and entries throughout
- **Issue**: Negative fold counts, zero components, empty strings accepted.
- **Severity**: P2

### 5.10 File Encoding Assumptions for CSV
- **Location**: `io.py` throughout
- **Issue**: Always assumes UTF-8. International instruments may use Latin-1.
- **Severity**: P2

### 5.11 No Spectral Artifact Detection
- **Location**: `io.py`
- **Issue**: Cosmic rays, saturation, detector overflow pass silently.
- **Severity**: P3 (feature gap)

### 5.12 No Wavelength Spacing Uniformity Check
- **Location**: `preprocess.py`
- **Issue**: SG derivatives require uniform spacing but no validation exists.
- **Severity**: P2

### 5.13 `build_model` Docstring Omits `one_class`
- **Location**: `models.py:353-400`
- **Issue**: Documents regression/classification but not one_class path.
- **Severity**: P2

### 5.14 Sidebar Navigation Disables Scrolling
- **Location**: `gui.py:1577-1606`
- **Issue**: Sidebar items are cut off when window is small. No scroll capability.
- **Severity**: P3

---

## Prioritized Fix Roadmap (Lesser Findings Only)

### Week 1: Silent Correctness (Highest User Impact)
1. Fix binary specificity formula (`scoring.py:346`) — **1 line, 5 min**
2. Fix VIP formula (`models.py:1738`) — **15 min**
3. Fix SPA determinism (`variable_selection.py:407`) — **20 min**
4. Fix PDS even-window bug (`calibration_transfer.py:193`) — **10 min**
5. Fix CARS weight bias (`variable_selection.py:1522`) — **20 min**

### Week 2: Encoding & Robustness (Prevents Real User Crashes)
6. Add `encoding='utf-8'` to all `open()` and `pd.read_csv()` calls — **30 min**
7. Add division-by-zero guard in `compute_vip` — **10 min**
8. Add SNV constant-spectrum warning — **10 min**
9. Add `simpledialog` validation wrappers — **45 min**
10. Lower `min_wavelengths` from 100 to 10 — **5 min**

### Week 3: UX Polish (Biggest Perceived Quality Gain)
11. Replace all "Check console" messages with file-logged details — **2 hr**
12. Standardize terminology (specimen→sample, pick one for target) — **3 hr**
13. Add contextual error titles instead of generic "Error" — **2 hr**
14. Center all modal dialogs on parent — **1 hr**
15. Add progress bar to Analysis Progress tab — **1 day**
16. Remove emoji from buttons, standardize arrow styles — **1 hr**

### Week 4: Performance & Hygiene
17. Remove redundant `.copy()` calls in `search.py` — **20 min**
18. Replace `iterrows()` with `itertuples()` in `ensemble.py` — **30 min**
19. Fix per-sample interpolation in `calibration_transfer.py` — **20 min**
20. Remove dead/commented-out code blocks — **1 hr**
21. Add missing docstrings to public API functions — **3 hr**
22. Add `__all__` exports to core modules — **1 hr**

---

## Appendix: Quick Reference Counts

| Issue Type | Count | Primary Files |
|------------|-------|---------------|
| `messagebox.showerror("Error", ...)` | 161 | `gui.py` |
| "Please ..." messages | 184 | `gui.py` |
| "Specimen" usage | 209 | `gui.py` |
| `print()` in `search.py` | 268+ | `search.py` |
| `print()` in `io.py` | 100+ | `io.py` |
| `print()` in `calibration_transfer.py` | ~66 | `calibration_transfer.py` |
| `print()` in `nsga2_search.py` | 42 | `nsga2_search.py` |
| `print()` in `unified_bayesian.py` | 47 | `unified_bayesian.py` |
| Bare `except:` blocks | 509 | GUI + backend |
| Dialogs without centering | ~15 | `gui.py` |
| Inconsistent CSV `index=` | 6 True, 6 False | `gui.py` |
| Commented-out code blocks | 5+ | `gui.py`, `search.py`, `interference.py` |
| Missing docstrings | 20+ | `search.py`, `io.py`, `ensemble.py`, `models.py` |
| Missing `__all__` | 7 modules | Core modules |

---

## Conclusion

The Top 12 Blockers are the structural problems that prevent commercial release. These **lesser findings** are the polish issues that determine whether early adopters say "this is promising" or "this is amateur."

**Fixing the 22 items in the 4-week roadmap above (~40 hours) would:**
- Eliminate silent wrong-result bugs (user trust)
- Prevent crashes on international data (global usability)
- Remove the "student project" impression (professional credibility)
- Reduce memory waste and improve speed (perceived performance)

**Do this alongside Phase 1 work.** Many items are 5-30 minute fixes that can be batched in a single PR per week.
