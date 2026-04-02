# One-Class Detection — Production-Ready Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the one-class detection feature fully production-ready — GUI parity with regression/classification, Bayesian optimization, Model Development integration, proper visualization, and end-to-end save/predict workflow.

**Architecture:** The one-class pipeline (`run_one_class_search`) must become a first-class citizen alongside regression and classification. This means wiring it into every GUI panel, the Bayesian optimizer, Model Development CV/plotting, model save/load, and the prediction tab. Variable selection methods (UVE, SPA, iPLS, CARS, GA) are PLS-specific and incompatible with one-class models — this is documented but not hacked around.

**Tech Stack:** Python, Tkinter, sklearn, scipy, matplotlib, numpy, pandas

**Naming:** The feature is called "One-Class" (not "Contamination"). It supports any one-class membership problem, not just contamination screening. All UI labels, docstrings, and log messages must use "One-Class" terminology.

---

## Task 1: Rename "Contamination" to "One-Class" Everywhere

**Files:**
- Modify: `spectral_predict_gui_optimized.py` (multiple locations)
- Modify: `src/spectral_predict/contamination.py` (module docstring, class docstrings, print/log messages)
- Modify: `src/spectral_predict/search.py` (log messages in `run_one_class_search`)
- Modify: `src/spectral_predict/model_config.py` (tier descriptions)
- Modify: `tests/test_contamination_detection.py` (test docstrings)

**Step 1: Find all "contamination" and "Contamination" references**

Run: `grep -rn -i "contamination" src/spectral_predict/contamination.py src/spectral_predict/search.py src/spectral_predict/model_config.py spectral_predict_gui_optimized.py tests/test_contamination_detection.py | grep -v "contamination=" | grep -v "IsolationForest"`

**Step 2: Replace UI-facing strings**

- `spectral_predict_gui_optimized.py`: Change `"One-Class (Contamination)"` radio button text to `"One-Class"`
- `spectral_predict_gui_optimized.py`: Change all `"Contamination Screening"` log messages to `"One-Class Screening"`
- `search.py`: Change `"ONE-CLASS CONTAMINATION SCREENING"` banner to `"ONE-CLASS SCREENING"`
- `search.py`: Change all `"contamination"` in log messages to `"one-class"`
- `contamination.py`: Update module docstring — remove contamination-specific language, describe as general one-class membership
- `model_config.py`: Change `'recommended_for': 'Quick contamination screening, daily QC'` to `'recommended_for': 'Quick one-class screening'` etc.

**Step 3: Run tests**

Run: `pytest tests/test_contamination_detection.py -v`
Expected: All pass (no functional changes)

**Step 4: Commit**

```bash
git add -u
git commit -m "refactor: Rename contamination to one-class throughout UI and docs"
```

---

## Task 2: Add One-Class Model Checkboxes to Analysis Config

**Files:**
- Modify: `spectral_predict_gui_optimized.py`
  - `_create_tab4c_model_configuration` (~line 11355): Add one-class checkbox panel
  - `_on_task_type_changed` (~line 15315): Show/hide one-class vs standard model panels

**Step 1: Write the one-class model checkbox panel**

After the existing Gradient Boosting section (line ~11580), add a new section:

```python
# One-Class Models section (hidden by default, shown when task_type == "one_class")
self.oc_models_frame = ttk.LabelFrame(models_frame, text="One-Class Models", padding=5)
# Don't grid it yet — _on_task_type_changed will show/hide it

self.ocsvm_checkbox = ttk.Checkbutton(self.oc_models_frame, text="OneClassSVM",
    variable=self.use_ocsvm)
self.ocsvm_checkbox.grid(row=0, column=0, sticky=tk.W, padx=5)

self.iforest_checkbox = ttk.Checkbutton(self.oc_models_frame, text="IsolationForest",
    variable=self.use_isolation_forest)
self.iforest_checkbox.grid(row=0, column=1, sticky=tk.W, padx=5)

self.ee_checkbox = ttk.Checkbutton(self.oc_models_frame, text="EllipticEnvelope",
    variable=self.use_elliptic_envelope)
self.ee_checkbox.grid(row=1, column=0, sticky=tk.W, padx=5)

self.lof_checkbox = ttk.Checkbutton(self.oc_models_frame, text="LOF",
    variable=self.use_lof)
self.lof_checkbox.grid(row=1, column=1, sticky=tk.W, padx=5)

self.pcasimca_checkbox = ttk.Checkbutton(self.oc_models_frame, text="PCA-SIMCA (DD-SIMCA)",
    variable=self.use_pca_simca)
self.pcasimca_checkbox.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
```

Store references in a dict for show/hide:
```python
self.oc_model_checkbox_widgets = {
    'OneClassSVM': self.ocsvm_checkbox,
    'IsolationForest': self.iforest_checkbox,
    'EllipticEnvelope': self.ee_checkbox,
    'LOF': self.lof_checkbox,
    'PCA-SIMCA': self.pcasimca_checkbox,
}
```

**Step 2: Update `_on_task_type_changed` to show/hide panels**

In `_on_task_type_changed` (~line 15315), after the existing model filter logic:

```python
if actual_task == "one_class":
    # Hide standard model checkboxes
    for widget in self.model_checkbox_widgets.values():
        widget.grid_remove()
    # Show one-class model checkboxes
    self.oc_models_frame.grid(row=next_row, column=0, columnspan=4, sticky='ew', pady=5)
    # Disable variable selection (incompatible with one-class)
    # ... set enable_variable_subsets to False, disable the checkbox
else:
    # Show standard model checkboxes
    for widget in self.model_checkbox_widgets.values():
        widget.grid()
    # Hide one-class model checkboxes
    self.oc_models_frame.grid_remove()
```

**Step 3: Test manually**

Run: `python spectral_predict_gui_optimized.py`
- Load data with class labels
- Select "One-Class" radio button
- Verify: standard model checkboxes disappear, one-class checkboxes appear
- Select "Regression" — verify standard checkboxes return

**Step 4: Commit**

```bash
git commit -m "feat: Add one-class model checkbox panel to Analysis Config"
```

---

## Task 3: Add One-Class to Model Development Tab

**Files:**
- Modify: `spectral_predict_gui_optimized.py`
  - Task type radio buttons (~line 13825): Add "One-Class" option
  - `_on_refine_task_type_changed` (~line 15543): Handle one_class model list
  - `_load_model_for_refinement` (~line 30290): Detect one_class from result config
  - `_run_refined_model_thread` (~line 32640): Add one-class CV path
  - Add new plot function: `_plot_one_class_results`

**Step 1: Add One-Class radio button to Model Development**

At line ~13830, add:
```python
ttk.Radiobutton(task_frame, text="One-Class", variable=self.refine_task_type, value='one_class').pack(side='left', padx=5)
```

**Step 2: Update `_on_refine_task_type_changed`**

At line ~15543, add one_class branch:
```python
if task_type == 'one_class':
    from spectral_predict.model_registry import ONE_CLASS_MODELS
    supported_models = ONE_CLASS_MODELS
    self.refine_model_type.set('PCA-SIMCA')
```

**Step 3: Update `_load_model_for_refinement` task type detection**

At line ~30290, add detection for one_class results:
```python
# Check if result was from one-class search
if config.get('Task') == 'one_class' or config.get('Model') in ['OneClassSVM', 'IsolationForest', 'EllipticEnvelope', 'LOF', 'PCA-SIMCA']:
    detected_task_type = 'one_class'
    self.refine_task_type.set('one_class')
elif self.y is not None:
    # existing detection logic...
```

**Step 4: Add one-class CV path in `_run_refined_model_thread`**

After the existing CV logic (~line 33596), add:
```python
if task_type == "one_class":
    # One-class CV: train on inliers only, test on both
    inlier_label = self.inlier_class_label.get().strip()
    if not inlier_label:
        inlier_label = y_array[np.argmax(np.unique(y_array, return_counts=True)[1])]
    
    y_oc = np.where(y_array == inlier_label, 1, -1)
    inlier_indices = np.where(y_oc == 1)[0]
    outlier_indices = np.where(y_oc == -1)[0]
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_preds = np.zeros(len(y_oc))
    all_scores = np.zeros(len(y_oc))
    
    for train_inlier_idx, test_inlier_idx in kf.split(inlier_indices):
        train_idx = inlier_indices[train_inlier_idx]
        test_inlier = inlier_indices[test_inlier_idx]
        test_idx = np.concatenate([test_inlier, outlier_indices])
        
        X_train_fold = X_preprocessed[train_idx]
        X_test_fold = X_preprocessed[test_idx]
        
        # Scale if needed
        if model_name in ('OneClassSVM', 'EllipticEnvelope', 'LOF'):
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_test_fold = scaler.transform(X_test_fold)
        
        model_clone = clone(model)
        model_clone.fit(X_train_fold)
        
        preds = model_clone.predict(X_test_fold)
        all_preds[test_idx] = preds
        
        if hasattr(model_clone, 'decision_function'):
            all_scores[test_idx] = model_clone.decision_function(X_test_fold)
    
    self.refined_y_true = y_oc
    self.refined_y_pred = all_preds
    self.refined_y_scores = all_scores
    self.refined_inlier_label = inlier_label
```

**Step 5: Add one-class plot function `_plot_one_class_results`**

New function that displays:
1. **Anomaly score histogram** — separate distributions for inlier vs outlier predictions
2. **Summary metrics** — Sensitivity, Specificity, Balanced Accuracy, AUC
3. **Per-class breakdown** — how each original class label was classified

```python
def _plot_one_class_results(self):
    """Plot one-class model results: score distributions and metrics."""
    from sklearn.metrics import balanced_accuracy_score
    from spectral_predict.contamination import one_class_metrics
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Score distribution
    ax1 = axes[0]
    inlier_mask = self.refined_y_true == 1
    outlier_mask = self.refined_y_true == -1
    
    if hasattr(self, 'refined_y_scores'):
        scores = self.refined_y_scores
        ax1.hist(scores[inlier_mask], bins=30, alpha=0.7, label='Inlier', color='steelblue')
        if outlier_mask.any():
            ax1.hist(scores[outlier_mask], bins=30, alpha=0.7, label='Outlier', color='salmon')
        ax1.axvline(x=0, color='red', linestyle='--', label='Decision boundary')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Score Distribution')
        ax1.legend()
    
    # Right: Metrics summary
    ax2 = axes[1]
    metrics = one_class_metrics(self.refined_y_true, self.refined_y_pred, 
                                 scores=getattr(self, 'refined_y_scores', None))
    
    metric_names = ['Sensitivity', 'Specificity', 'Precision', 'F1', 'Balanced\nAccuracy', 'AUC']
    metric_values = [
        metrics.get('sensitivity', 0), metrics.get('specificity', 0),
        metrics.get('precision', 0), metrics.get('f1', 0),
        metrics.get('balanced_accuracy', 0), metrics.get('auc', 0)
    ]
    
    colors = ['#2ecc71' if v >= 0.8 else '#f39c12' if v >= 0.6 else '#e74c3c' 
              for v in metric_values]
    bars = ax2.bar(metric_names, metric_values, color=colors, edgecolor='white')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Score')
    ax2.set_title('One-Class Metrics (CV)')
    for bar, val in zip(bars, metric_values):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    fig.tight_layout()
    # Display in refine_plot_frame...
```

**Step 6: Wire the plot into the display logic**

In the results display section of `_run_refined_model_thread` (after CV completes):
```python
if task_type == "one_class":
    self.root.after(0, self._plot_one_class_results)
elif task_type == "classification":
    # existing confusion matrix path
```

**Step 7: Test manually**

- Run analysis with One-Class mode
- Double-click a result in Results table
- Verify Model Development shows One-Class radio selected
- Click Run Model
- Verify: score histogram + metrics bar chart appear (not confusion matrix)

**Step 8: Commit**

```bash
git commit -m "feat: Add one-class support to Model Development tab"
```

---

## Task 4: Add One-Class to Results Table Display

**Files:**
- Modify: `spectral_predict_gui_optimized.py`
  - `_populate_results_table_inner` (~line 25986): Add one_class detection and column formatting

**Step 1: Add one_class detection in results table**

At line ~26025, after the classification check, add:
```python
# Detect one-class results
is_one_class = 'BalancedAcccv' in results_df.columns and 'Sensitivitycv' in results_df.columns and 'Task' in results_df.columns
if is_one_class:
    is_one_class = (results_df['Task'] == 'one_class').any()
```

**Step 2: Add one-class column display logic**

Define the priority columns for one-class:
```python
if is_one_class:
    display_cols = ['Rank', 'Model', 'Preprocess', 'BalancedAcccv', 'Sensitivitycv', 
                    'Specificitycv', 'AUCcv', 'BalancedAcc', 'Sensitivity', 'Specificity',
                    'n_vars', 'Params']
```

**Step 3: Test**

Run analysis in one-class mode, verify results table shows appropriate columns.

**Step 4: Commit**

```bash
git commit -m "feat: Add one-class column display to Results table"
```

---

## Task 5: Integrate One-Class with Bayesian Optimization

**Files:**
- Modify: `src/spectral_predict/search.py`
  - `run_one_class_search`: Add `optimization_method` parameter
  - Remove the `return` that prevents reaching the Bayesian dispatch
- Modify: `src/spectral_predict/unified_bayesian.py` (or equivalent): Add one_class objective
- Modify: `spectral_predict_gui_optimized.py`: Don't short-circuit before Bayesian dispatch

**Step 1: Check the Bayesian optimization architecture**

Read `src/spectral_predict/unified_bayesian.py` to understand how it handles task_type for regression/classification. The one-class objective should optimize `BalancedAcccv` (or a combination of sensitivity and specificity).

**Step 2: Add one-class objective to Bayesian optimizer**

The objective function needs to:
- Use one-class CV (train on inliers only)
- Optimize balanced accuracy
- Include one-class model hyperparameter spaces (nu, contamination, alpha, n_components)

**Step 3: Remove early return in GUI dispatch**

In `spectral_predict_gui_optimized.py` line ~24862, the `return` after `run_one_class_search` prevents reaching the Bayesian dispatch. Change this so when `optimization_method == "unified"`, it calls the Bayesian optimizer instead of the grid search.

**Step 4: Test**

- Select One-Class mode
- Select Bayesian optimization
- Run analysis
- Verify it completes with valid results

**Step 5: Commit**

```bash
git commit -m "feat: Integrate one-class models with Bayesian optimization"
```

---

## Task 6: Model Save/Load End-to-End for One-Class

**Files:**
- Modify: `spectral_predict_gui_optimized.py`
  - `_save_refined_model` (~line 34771): Wire scaler/PCA reducer from refined model
  - `_run_refined_model_thread`: Store fitted scaler/PCA as `self.refined_oc_scaler` etc.
- Modify: `src/spectral_predict/model_io.py`
  - `predict_with_uncertainty`: Add one_class branch

**Step 1: Store fitted auxiliary objects during refined model training**

In the one-class CV path added in Task 3, after fitting the final calibration model:
```python
# Store scaler and PCA reducer for persistence
if model_name in ('OneClassSVM', 'EllipticEnvelope', 'LOF'):
    self.refined_oc_scaler = scaler  # Fitted on all inliers
else:
    self.refined_oc_scaler = None
self.refined_oc_pca_reducer = None  # Set if EllipticEnvelope needed PCA
```

**Step 2: Update predict_with_uncertainty for one_class**

In `model_io.py`, the `predict_with_uncertainty` function wraps `predict_with_model`. Ensure one-class predictions return in a format compatible with the prediction tab.

**Step 3: Test save/load roundtrip through GUI**

- Train a one-class model in Model Development
- Save as .dasp file
- Load in Prediction tab
- Predict on new data
- Verify predictions are +1/-1

**Step 4: Commit**

```bash
git commit -m "feat: End-to-end one-class model save/load/predict"
```

---

## Task 7: Prediction Tab One-Class Support

**Files:**
- Modify: `spectral_predict_gui_optimized.py`
  - `_run_predictions` (~line 36767): Handle one-class prediction output
  - `_display_prediction_results`: Format +1/-1 as "Inlier"/"Outlier" labels
  - Prediction plots: Show anomaly scores instead of regression residuals

**Step 1: Handle one-class prediction output**

In `_run_predictions`, after getting predictions:
```python
task_type = model_dict['metadata'].get('task_type', 'regression')
if task_type == 'one_class':
    # Map +1/-1 to readable labels
    label_map = {1: 'Inlier', -1: 'Outlier'}
    predictions_display = np.array([label_map.get(p, str(p)) for p in predictions])
    # Get decision scores for display
    if hasattr(model_dict['model'], 'decision_function'):
        scores = model_dict['model'].decision_function(X_processed)
```

**Step 2: Display one-class results in prediction table**

Add columns: Sample, Prediction (Inlier/Outlier), Anomaly Score

**Step 3: Commit**

```bash
git commit -m "feat: Add one-class support to Prediction tab"
```

---

## Task 8: Variable Selection for One-Class Models

**Files:**
- Modify: `spectral_predict_gui_optimized.py`
  - `_on_task_type_changed`: Limit variable selection to compatible methods
- Modify: `src/spectral_predict/search.py`
  - `run_one_class_search`: Add variable selection support using CARS-Tree

**Context:** PLS-specific methods (UVE, SPA, iPLS) require PLS regression internally. However, multiple approaches work with one-class:
- **CARS-Tree** — uses LGBMClassifier with +1/-1 binary labels. Already in the codebase.
- **IsolationForest** — exposes `feature_importances_` natively (sklearn API)
- **PCA-SIMCA loadings** — `pca_.components_` gives wavelength contribution weights
- **Permutation importance** — model-agnostic, works with any sklearn estimator
- **GA (Genetic Algorithm)** — can use any fitness function, not PLS-locked

These should be wired into the existing `supports_feature_importance` registry and the variable selection pipeline.

**Step 1: Enable CARS-Tree for one-class**

In `_on_task_type_changed`, when one_class is selected:
- Disable PLS-specific methods (UVE, SPA, iPLS, GA-PLS)
- Keep CARS-Tree enabled — it uses LGBMClassifier with +1/-1 labels
- Keep region subsets enabled (model-agnostic)

```python
if actual_task == "one_class":
    # Disable PLS-specific variable selection methods
    for method in ['uve', 'spa', 'ipls', 'ga']:
        checkbox = getattr(self, f'{method}_checkbox', None)
        if checkbox:
            checkbox.configure(state='disabled')
    # CARS-Tree works with binary classification (inlier/outlier)
    if hasattr(self, 'cars_checkbox'):
        self.cars_checkbox.configure(state='!disabled')
```

**Step 2: Update `model_registry.py` — register feature importance support**

In `supports_feature_importance()`, add one-class models that expose importances:
```python
# One-class models with native feature importance
'IsolationForest',   # sklearn feature_importances_
'PCA-SIMCA',         # pca_.components_ loadings
```

**Step 3: Wire CARS-Tree into `run_one_class_search`**

Call `cars_selection(X, y_oc, task_type='classification', model_type='LightGBM')` where `y_oc` is the +1/-1 labels. Use the resulting importance scores for wavelength subset selection, following the same pattern as the classification search path.

**Step 4: Extract importance from one-class models**

In `search.py`, after fitting calibration model, extract importances:
```python
if model_name == 'IsolationForest':
    importances = cal_model.feature_importances_
elif model_name == 'PCA-SIMCA':
    importances = np.sum(np.abs(cal_model.pca_.components_), axis=0)
```

**Step 5: Commit**

```bash
git commit -m "feat: Enable variable selection for one-class (CARS-Tree, native importances)"
```

---

## Task 9: Add Complexity Scores for One-Class Models

**Files:**
- Modify: `src/spectral_predict/scoring.py`
  - `_compute_unified_complexity` (~line 209): Add one-class model complexity scores

**Step 1: Add one-class model complexity scores**

```python
# One-class model complexity (from simple to complex)
'PCA-SIMCA': 20,        # Linear PCA model, very interpretable
'IsolationForest': 35,   # Ensemble of random trees
'OneClassSVM': 55,       # Kernel method, moderate complexity
'LOF': 45,               # Density-based, moderate
'EllipticEnvelope': 30,  # Gaussian assumption, simple
```

**Step 2: Commit**

```bash
git commit -m "feat: Add complexity scores for one-class models"
```

---

## Task 10: Comprehensive Test Suite

**Files:**
- Modify: `tests/test_contamination_detection.py`

**Step 1: Add GUI integration test for one-class results display**

Test that `create_results_dataframe('one_class')` produces the right columns and that `compute_composite_score` ranks them correctly with the new complexity scores.

**Step 2: Add Model Development one-class CV test**

Test that the one-class CV path in `_run_refined_model_thread` correctly trains on inliers only and evaluates on both.

**Step 3: Add prediction tab format test**

Test that `predict_with_model` returns numpy array for one-class and that +1/-1 values are correct.

**Step 4: Run full test suite**

Run: `pytest tests/test_contamination_detection.py tests/test_scoring.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git commit -m "test: Comprehensive one-class integration tests"
```

---

## Verification

After all tasks:

1. **Full test suite:** `pytest tests/ -x --timeout=120`
2. **Manual GUI test:**
   - Load spectral data with class labels
   - Select One-Class → verify checkbox panel switches, variable selection disabled
   - Run Grid Search → verify results table shows correct columns
   - Run Bayesian Optimization → verify it works
   - Double-click result → verify Model Development shows One-Class with correct plots
   - Save model → verify .dasp file saves correctly
   - Load in Prediction tab → verify predictions are Inlier/Outlier
3. **Code review:** Launch code-reviewer agent
4. **Codex review:** Run `codex exec --full-auto` for final check

---

## Task Dependency Graph

```
Task 1 (rename) ──> Task 2 (checkboxes) ──> Task 8 (disable incompatible)
                                        ──> Task 3 (Model Dev) ──> Task 6 (save/load)
                                                               ──> Task 7 (prediction)
                    Task 4 (results table) ─────────────────────>
                    Task 5 (Bayesian) ──────────────────────────>
                    Task 9 (complexity) ────────────────────────>
                                                                    Task 10 (tests)
```

**Parallelizable:** Tasks 2, 4, 5, 9 are independent after Task 1. Task 3 depends on Task 2. Tasks 6, 7 depend on Task 3. Task 10 is last.
