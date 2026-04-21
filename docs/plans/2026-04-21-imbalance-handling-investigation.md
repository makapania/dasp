# Imbalance Handling Investigation

**Date:** 2026-04-21
**Trigger:** User reports ADASYN silently falls back to SMOGN for some task types without notification.
**Status:** Read-only investigation (no code modified).

---

## Section 1: Current Behavior

### 1.1 Where imbalance handling lives

| File | Lines | Role |
|------|-------|------|
| src/spectral_predict/imbalance.py | 1-1302 (entire file) | Core module: detection, transformers, factory, validation |
| src/spectral_predict/preprocess.py | 280-283 (signature), 551-575 (imbalance step) | Integrates imbalance into preprocessing pipeline |
| src/spectral_predict/search.py | 226-269, 852, 993-1005, 4066-4067, 4119-4130, 4146-4148, 4645-4651 | Grid search: passes imbalance through every CV evaluation |
| src/spectral_predict/unified_bayesian.py | 1704-1716 (fallback), 1718-1725 (validation) | Bayesian optimization: same, plus silent regression fallback |
| src/spectral_predict/nsga2_search.py | 125-151, 1381-1394, 1711-1718, 1810-1818 | NSGA-II multi-objective: same, plus silent regression fallback |
| spectral_predict_gui_optimized.py | 10878-10943, 1789-1805, 20514, 22172-22286, 22287-22350, 22352-22372, 26292, 26768-26769, 27243-27244 | GUI: presentation, detection, parameter collection, dispatch |

### 1.2 Where the silent fallback happens

There are three distinct fallback sites (one in the GUI and two in backend modules). None are surfaced to the user during a search run.

#### Fallback A: GUI dropdown silent reset

File: spectral_predict_gui_optimized.py:22206-22221

```python
# Update method dropdown based on task type
if task_type == 'classification':
    classification_methods = [
        'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
        'tomek_links', 'smote_tomek', 'smote_enn', 'class_weight'
    ]
    self.imbalance_method_combo['values'] = classification_methods
    if self.imbalance_method.get() not in classification_methods:
        self.imbalance_method.set('smote')       # SILENT RESET
else:  # regression
    regression_methods = ['smogn', 'oversample', 'smotetomek', 'undersample', 'binning', 'rare_boost', 'balanced']
    self.imbalance_method_combo['values'] = regression_methods
    if self.imbalance_method.get() not in regression_methods:
        self.imbalance_method.set('smogn')       # SILENT RESET
```

Trigger: _detect_and_display_imbalance() (line 22172) is called from the outlier detection completion path (line 20514).

Problem: The method is reset without any visible notification. The imbalance_warning_label and imbalance_recommendation_label are updated with imbalance detection results but NOT with a message indicating the selected method was changed.

Compounded by: _on_task_type_changed() (line 16242) does NOT call _detect_and_display_imbalance() or update the imbalance dropdown. If a user changes the task type after outlier detection has already run, the dropdown keeps its stale values.

#### Fallback B: Bayesian optimization backend substitution

File: src/spectral_predict/unified_bayesian.py:1708-1716

```python
# Substitute regression sample weighting methods that do not work with cross_val_score
# These methods (binning, rare_boost, balanced) require manual sample_weight extraction
# which is only implemented in Grid Search. Substitute with smogn (SMOTE-style for regression).
UNSUPPORTE"_REGRESSION_METHO"S = {'binning', 'rare_boost', 'balanced'}
if task_type == 'regression' and imbalance_method in UNSUPPORTE"_REGRESSION_METHO"S:
    original_method = imbalance_method
    imbalance_method = 'smogn'  # SMOTE-style synthetic oversampling for regression
    if verbose:
        print(f"Note: '{original_method}' requires Grid Search. Using 'smogn' instead for Bayesian optimization.")
```

Logging: print() only (not captured by the GUI progress callback, not written to any logger). The file imports logging (line 34) but never calls getLogger().

Scope: Affects 'binning', 'rare_boost', 'balanced' when used with Bayesian search on regression tasks.

#### Fallback C: NSGA-II backend substitution

File: src/spectral_predict/nsga2_search.py:1810-1818

```python
# Substitute regression sample weighting methods that do not work with cross_val_score
# These methods (binning, rare_boost, balanced) require manual sample_weight extraction
# which is only implemented in Grid Search. Substitute with smogn (SMOTE-style for regression).
UNSUPPORTE"_REGRESSION_METHO"S = {'binning', 'rare_boost', 'balanced'}
if task_type == 'regression' and imbalance_method in UNSUPPORTE"_REGRESSION_METHO"S:
    original_method = imbalance_method
    imbalance_method = 'smogn'  # SMOTE-style synthetic oversampling for regression
    if verbose >= 1:
        print(f"  Note: '{original_method}' requires Grid Search. Using 'smogn' instead for NSGA-II.")
```

Logging: Same print()-only pattern. The file does have a logger (line 74) but does not use it here.

### 1.3 How the GUI presents imbalance options

The initial dropdown values are set at spectral_predict_gui_optimized.py:10893-10898:

```python
self.imbalance_method_combo['values'] = [
    'smote', 'adasyn', 'borderline_smote', 'random_undersampler',
    'tomek_links', 'smote_tomek', 'smote_enn', 'class_weight',
    'binning', 'rare_boost', 'balanced'
]
```

This is a single combined list of classification-only imblearn methods plus regression sample-weighting methods. It does not include the regression resampling methods ('smogn', 'oversample', 'smotetomek', 'undersample') in the initial list. The list is only later filtered by _detect_and_display_imbalance() (line 22206 or 22217) which only runs after outlier detection.

The imbalance section is always visible (line 10878: "Imbalance Handling Settings" LabelFrame). There is no task-type-aware visibility toggle. For one-class tasks, the imbalance section remains visible and interactive even though one-class search has no imbalance_method parameter (confirmed: run_one_class_search at search.py:4996 has no imbalance_method parameter; its signature runs to line 5021 with 25 parameters, none of which relate to imbalance).

---

## Section 2: Library-Level Truth

### 2.1 imbalanced-learn (imblearn) methods (classification only)

All of these require discrete class labels and cannot operate on continuous targets:

| Method | imblearn module | Classification | Regression | One-Class |
|--------|----------------|----------------|------------|-----------|
| SMOTE | imblearn.over_sampling.SMOTE | Yes | No | No |
| A"ASYN | imblearn.over_sampling.A"ASYN | Yes | No | No |
| BorderlineSMOTE | imblearn.over_sampling.BorderlineSMOTE | Yes | No | No |
| RandomUnderSampler | imblearn.under_sampling.RandomUnderSampler | Yes | No | No |
| TomekLinks | imblearn.under_sampling.TomekLinks | Yes | No | No |
| SMOTETomek | imblearn.combine.SMOTETomek | Yes | No | No |
| SMOTEENN | imblearn.combine.SMOTEENN | Yes | No | No |

Source: imbalanced-learn documentation. All over/under-sampling classes inherit from BaseOverSampler or BaseUnderSampler, which require y to be discrete class labels. See https://imbalanced-learn.org/stable/references/over_sampling.html and https://imbalanced-learn.org/stable/references/under_sampling.html. The fit_resample(X, y) method expects classification targets.

class_weight is not an imblearn method (it is a model parameter, class_weight='balanced', supported by many sklearn classifiers such as RandomForestClassifier, SVC, LogisticRegression). It is classification-only by definition. See https://scikit-learn.org/stable/glossary.html#term-class_weight.

### 2.2 Regression methods (in-project implementation)

These are implemented in src/spectral_predict/imbalance.py, not from external libraries:

| Method | Implementation class | Classification | Regression | One-Class |
|--------|---------------------|----------------|------------|-----------|
| smogn | RegressionResampler(method='smogn') | No | Yes | No |
| oversample | RegressionResampler(method='oversample') | No | Yes | No |
| smotetomek | RegressionResampler(method='smotetomek') | No | Yes | No |
| undersample | RegressionUndersampler | No | Yes | No |
| binning | RegressionSampleWeighter(strategy='binning') | No | Yes (grid search only) | No |
| rare_boost | RegressionSampleWeighter(strategy='rare_boost') | No | Yes (grid search only) | No |
| balanced | RegressionSampleWeighter(strategy='balanced') | No | Yes (grid search only) | No |

Key distinction: The first four (smogn, oversample, smotetomek, undersample) are resampling methods (they change X and y via fit_resample()). The last three (binning, rare_boost, balanced) are sample weighting methods (they compute sample_weight_ during fit() and pass X through unchanged during transform()).

SMOGN is inspired by the SMOGN algorithm (Branco et al., 2017, "SMOGN: A Pre-Processing Approach for Imbalanced Regression", Proc. Machine Learning Research). The in-project implementation (imbalance.py:607-713) is a simplified SMOTE-like k-NN interpolation for continuous targets, not the published SMOGN package from PyPI.

Why binning/rare_boost/balanced fail in Bayesian and NSGA-II: These sample-weighting methods require manual extraction of sample_weight_ from the transformer after fit() and passing it to the model .fit(X, y, sample_weight=...) call. Only the grid search path (search.py:4146-4148) implements this extraction. Bayesian (unified_bayesian.py) and NSGA-II (nsga2_search.py) use cross_val_score() / cross_val_predict(), which have no mechanism for per-sample weights (hence the silent substitution to smogn).

### 2.3 One-class tasks (no applicable methods)

One-class models (OneClassSVM, IsolationForest, LOF, EllipticEnvelope, PCA-SIMCA) are trained on a single class of normal samples and evaluated on their ability to detect outliers. Imbalance handling is not applicable because:

1. No class imbalance concept: There is only one class (inliers). There are no minority/majority class groups to rebalance.
2. No continuous target distribution: One-class models operate on unsupervised feature-space density, not target values.
3. Backend confirms: run_one_class_search() (search.py:4996-5021) accepts no imbalance_method parameter whatsoever. The one-class search path never touches imbalance.py.

---

## Section 3: Gap Analysis

### 3.1 Scenario table

| Task type | Selected method | Search type | What actually happens | User aware? |
|-----------|-----------------|-------------|----------------------|-------------|
| Classification | smote / adasyn / etc. | Any | Works correctly via imblearn | Yes |
| Classification | class_weight | Grid | Works (model parameter) | Yes |
| Classification | class_weight | Bayesian / NSGA-II | Works (model parameter, not a resampler) | Yes |
| Regression | smogn / oversample / smotetomek / undersample | Any | Works correctly via RegressionResampler / RegressionUndersampler | Yes (if dropdown was updated) |
| Regression | binning / rare_boost / balanced | Grid | Works correctly (sample_weight extraction implemented) | Yes |
| Regression | binning / rare_boost / balanced | Bayesian | Silently substituted to smogn | NO (print only) |
| Regression | binning / rare_boost / balanced | NSGA-II | Silently substituted to smogn | NO (print only) |
| Regression | smote / adasyn / any classification method | Any | Silently reset to smogn in GUI dropdown | NO |
| One-class | any method | Any | Ignored (parameter never passed to backend) | NO (section still visible) |

### 3.2 The core user-facing problem

A user selects ADASYN for a regression task (or a regression method for a classification task), starts a search, and gets results computed with a completely different method. There is no warning dialog, no status message, no log entry visible in the GUI. The user believes they ran ADASYN but actually ran SMOGN.

This is a scientific integrity risk: published results may claim a methodology that was not actually used.

### 3.3 The initial dropdown is incomplete

At launch, the dropdown shows: smote, adasyn, borderline_smote, random_undersampler, tomek_links, smote_tomek, smote_enn, class_weight, binning, rare_boost, balanced.

Missing from the initial list: smogn, oversample, smotetomek, undersample (four of the seven regression methods). These only appear after _detect_and_display_imbalance() runs (post-outlier-detection). If a user selects regression + grid search before running outlier detection, they cannot even select the most appropriate regression methods.

---

## Section 4: Recommended Fix

### 4.1 Task-type-aware dropdown filtering (HIGH priority)

Location: spectral_predict_gui_optimized.py:10893-10898 (initial setup) and 22206-22221 (update in _detect_and_display_imbalance).

Change: Set the dropdown values based on task type from the start, and update immediately when task type changes.

- On GUI init: use classification list (default task type is usually classification).
- In _on_task_type_changed(): call the same dropdown update logic that _detect_and_display_imbalance() uses.
- Remove the initial combined list entirely.

Proposed method split for each task type:

| Task Type | Methods shown |
|-----------|---------------|
| Classification | smote, adasyn, borderline_smote, random_undersampler, tomek_links, smote_tomek, smote_enn, class_weight |
| Regression | smogn, oversample, smotetomek, undersample, binning, rare_boost, balanced |
| One-Class | Imbalance section hidden or disabled entirely |

### 4.2 Hide imbalance section for one-class (HIGH priority)

Location: spectral_predict_gui_optimized.py:10878 (LabelFrame creation) plus _update_one_class_controls_visibility() at line 16308.

Change: When task type is one_class, hide or disable the entire Imbalance Handling Settings LabelFrame. The backend (run_one_class_search) ignores it anyway (showing it is misleading).

### 4.3 GUI notification on method substitution (MEDIUM priority)

Location: spectral_predict_gui_optimized.py:22206-22221 (Fallback A).

Change: When the dropdown resets the method, update imbalance_warning_label or show a temporary info banner along the lines of:

    Method changed: {old_method} -> {new_method} (not applicable for {task_type})

This should also fire from _on_task_type_changed() if the current selection is invalid for the new task type.

### 4.4 Backend logging (MEDIUM priority)

Locations:
- src/spectral_predict/unified_bayesian.py:1708-1716
- src/spectral_predict/nsga2_search.py:1810-1818

Change: Replace print() with logger.warning() in both files. Both files already import logging. For nsga2_search.py, use the existing logger at line 74. For unified_bayesian.py, add logger = logging.getLogger(__name__) near the top.

Additionally, surface the substitution through the progress_callback so the GUI can display it in the progress text area during the search.

### 4.5 Fix initial dropdown values (LOW priority; superseded by 4.1)

If 4.1 is implemented (task-type-aware dropdown), this becomes moot. If 4.1 is deferred, the initial list should at minimum include all regression resampling methods.

### 4.6 Grid search crash guard (LOW priority; defense in depth)

Location: src/spectral_predict/search.py near line 993 (existing validation).

Problem: If a classification-only method (e.g., adasyn) is passed to build_imbalance_transformer with task_type='regression', it raises ValueError at imbalance.py:952. In grid search (search.py:4124), this would crash the entire run if the GUI filter is bypassed.

Fix: Add a validation guard at the top of run_search() that checks method-task compatibility for regression too (currently only classification is validated).

### 4.7 Implementation order

| Step | Item | Priority | Effort |
|------|------|----------|--------|
| 1 | 4.1: Task-type-aware dropdown in _on_task_type_changed() + initial setup | HIGH | Small |
| 2 | 4.2: Hide imbalance section for one-class | HIGH | Small |
| 3 | 4.3: GUI notification on method change | MEDIUM | Small |
| 4 | 4.4: Backend logger.warning() instead of print() | MEDIUM | Small |
| 5 | 4.5: Fix initial dropdown (only if 4.1 is deferred) | LOW | Trivial |
| 6 | 4.6: Grid search crash guard | LOW | Trivial |

Steps 1 and 2 should be done together (they both touch the task-type-change path and the imbalance UI section). Items 4.1 + 4.2 together address the core user complaint. Items 4.3-4.6 are hardening.
