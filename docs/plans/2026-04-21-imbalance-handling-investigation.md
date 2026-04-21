# Imbalance Handling Investigation

**Date:** 2026-04-21  
**Trigger:** User reports ADASYN silently falls back to SMOGN for some task types without notification.  
**Status:** Read-only investigation — no code modified.

---

## Section 1: Current Behavior

### 1.1 Where imbalance handling lives

| File | Lines | Role |
|------|-------|------|
| `src/spectral_predict/imbalance.py` | 1–1302 (entire file) | Core module: detection, transformers, factory, validation |
| `src/spectral_predict/preprocess.py` | 280–283 (signature), 551–575 (imbalance step) | Integrates imbalance into preprocessing pipeline |
| `src/spectral_predict/search.py` | 226–269 (`_needs_resampling_pipeline`), 852 (param), 993–1005 (classification validation), 4066–4067 (params), 4119–4130 (transformer build), 4146–4148 (pipeline), 4645–4651 (result storage) | Grid search: passes imbalance through every CV evaluation |
| `src/spectral_predict/unified_bayesian.py` | 1704–1716 (fallback), 1718–1725 (validation) | Bayesian optimization: same, plus silent regression fallback |
| `src/spectral_predict/nsga2_search.py` | 125–151 (`_needs_resampling_pipeline`), 1381–1394 (pipeline), 1711–1718 (fallback), 1810–1818 (substitute) | NSGA-II multi-objective: same, plus silent regression fallback |
| `spectral_predict_gui_optimized.py` | 10878–10943 (UI widgets), 1789–1805 (filename suffix), 20514 (detection trigger), 22140–22270 (`_detect_and_display_imbalance`), 22271–22286 (toggle), 22287–22350 (description update), 22352–22372 (`_get_imbalance_params`), 26292 (passed to search), 26768–26769 (passed to Bayesian), 27243–27244 (passed to grid search) | GUI: presentation, detection, parameter collection, dispatch |

### 1.2 Where the silent fallback happens

There are **three distinct fallback sites** — one in the GUI and two in backend modules. None are surfaced to the user during a search run.

#### Fallback A: GUI dropdown silent reset

**File:** `spectral_predict_gui_optimized.py:22174–22189`

```python
# Update method dropdown based on task type
if task_type == 'classification':
    classification_methods = [
        'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
        'tomek_links', 'smote_tomek', 'smote_enn', 'class_weight'
    ]
    self.imbalance_method_combo['values'] = classification_methods
    # Set default if current selection is not valid
    if self.imbalance_method.get() not in classification_methods:
        self.imbalance_method.set('smote')       # <-- SILENT RESET
else:  # regression
    regression_methods = ['smogn', 'oversample', 'smotetomek', 'undersample', 'binning', 'rare_boost', 'balanced']
    self.imbalance_method_combo['values'] = regression_methods
    # Set default if current selection is not valid
    if self.imbalance_method.get() not in regression_methods:
        self.imbalance_method.set('smogn')       # <-- SILENT RESET
```

**Trigger:** `_detect_and_display_imbalance()` (line 22140) is called from the outlier detection completion path (line 20514).  
**Problem:** The method is reset without any visible notification. The `imbalance_warning_label` and `imbalance_recommendation_label` are updated with imbalance *detection* results but **not** with a "your selected method was changed" message.  
**Compounded by:** `_on_task_type_changed()` (line 16242) does **NOT** call `_detect_and_display_imbalance()` or update the imbalance dropdown. If a user changes the task type after outlier detection has already run, the dropdown keeps its stale values.

#### Fallback B: Bayesian optimization backend substitution

**File:** `src/spectral_predict/unified_bayesian.py:1708–1716`

```python
# Substitute regression sample weighting methods that don't work with cross_val_score
# These methods (binning, rare_boost, balanced) require manual sample_weight extraction
# which is only implemented in Grid Search. Substitute with smogn (SMOTE-style for regression).
UNSUPPORTED_REGRESSION_METHODS = {'binning', 'rare_boost', 'balanced'}
if task_type == 'regression' and imbalance_method in UNSUPPORTED_REGRESSION_METHODS:
    original_method = imbalance_method
    imbalance_method = 'smogn'  # SMOTE-style synthetic oversampling for regression
    if verbose:
        print(f"Note: '{original_method}' requires Grid Search. Using 'smogn' instead for Bayesian optimization.")
```

**Logging:** `print()` only — not captured by the GUI progress callback, not written to any logger. The file imports `logging` (line 34) but never calls `getLogger()`.  
**Scope:** Affects 'binning', 'rare_boost', 'balanced' when used with Bayesian search on regression tasks.

#### Fallback C: NSGA-II backend substitution

**File:** `src/spectral_predict/nsga2_search.py:1810–1818`

```python
# Substitute regression sample weighting methods that don't work with cross_val_score
# These methods (binning, rare_boost, balanced) require manual sample_weight extraction
# which is only implemented in Grid Search. Substitute with smogn (SMOTE-style for regression).
UNSUPPORTED_REGRESSION_METHODS = {'binning', 'rare_boost', 'balanced'}
if task_type == 'regression' and imbalance_method in UNSUPPORTED_REGRESSION_METHODS:
    original_method = imbalance_method
    imbalance_method = 'smogn'  # SMOTE-style synthetic oversampling for regression
    if verbose >= 1:
        print(f"  Note: '{original_method}' requires Grid Search. Using 'smogn' instead for NSGA-II.")
```

**Logging:** Same `print()`-only pattern. The file *does* have a `logger` (line 74) but does not use it here.

### 1.3 How the GUI presents imbalance options

The initial dropdown values are set at **`spectral_predict_gui_optimized.py:10893–10896`**:

```python
self.imbalance_method_combo['values'] = [
    'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
    'tomek_links', 'smote_tomek', 'smote_enn', 'class_weight',
    'binning', 'rare_boost', 'balanced'
]
```

This is a **single combined list** of both classification and regression methods. It does not include `'smogn'`, `'oversample'`, `'smotetomek'`, or `'undersample'` (regression resampling methods) in the initial list. The list is only later filtered by `_detect_and_display_imbalance()` (line 22180 or 22185).

The imbalance section is always visible (line 10878: `"Imbalance Handling Settings"` LabelFrame). There is no task-type-aware visibility toggle. For **one-class** tasks, the imbalance section remains visible and interactive even though one-class search has no `imbalance_method` parameter (confirmed: `run_one_class_search` at `search.py:4985` has no imbalance parameter).

---

## Section 2: Library-Level Truth

### 2.1 imbalanced-learn (imblearn) methods — classification only

All of these require discrete class labels and cannot operate on continuous targets:

| Method | imblearn module | Classification | Regression | One-Class |
|--------|----------------|----------------|------------|-----------|
| **SMOTE** | `imblearn.over_sampling.SMOTE` | ✅ | ❌ | ❌ |
| **ADASYN** | `imblearn.over_sampling.ADASYN` | ✅ | ❌ | ❌ |
| **BorderlineSMOTE** | `imblearn.over_sampling.BorderlineSMOTE` | ✅ | ❌ | ❌ |
| **RandomUnderSampler** | `imblearn.under_sampling.RandomUnderSampler` | ✅ | ❌ | ❌ |
| **TomekLinks** | `imblearn.under_sampling.TomekLinks` | ✅ | ❌ | ❌ |
| **SMOTETomek** | `imblearn.combine.SMOTETomek` | ✅ | ❌ | ❌ |
| **SMOTEENN** | `imblearn.combine.SMOTEENN` | ✅ | ❌ | ❌ |

**Source:** imbalanced-learn documentation (https://imbalanced-learn.org/stable/references/over_sampling.html) — all over/under-sampling classes inherit from `BaseOverSampler` or `BaseUnderSampler`, which require `y` to be discrete class labels. The `fit_resample(X, y)` method expects classification targets.

**class_weight** is not an imblearn method — it's a model parameter (`class_weight='balanced'`) supported by many sklearn classifiers (e.g., `RandomForestClassifier`, `SVC`, `LogisticRegression`). It is classification-only by definition.

### 2.2 Regression methods — in-project implementation

These are implemented in `src/spectral_predict/imbalance.py`, not from external librar
