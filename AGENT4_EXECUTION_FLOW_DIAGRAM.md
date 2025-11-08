# Tab 6 Execution Flow Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER CLICKS "▶ Run Model"                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  _run_refined_model() - Main Thread (GUI)                           │
│                                                                      │
│  ✓ Validate data loaded                                            │
│  ✓ Validate parameters (_validate_refinement_parameters)           │
│  ✓ Disable Run button                                              │
│  ✓ Update status: "Running refined model..."                       │
│  ✓ Launch background thread →                                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  _run_refined_model_thread() - Background Thread (~683 lines)       │
│                                                                      │
│  This is where ALL the heavy computation happens!                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Execution Flow

### Phase 1: Data Preparation (Lines 4057-4138)

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: Parse Parameters                                           │
├─────────────────────────────────────────────────────────────────────┤
│  • Parse wavelength specification from Text widget                  │
│  • Get model type (PLS, Ridge, etc.)                               │
│  • Get preprocessing method (raw, sg1, sg2, etc.)                  │
│  • Get window size, CV folds, max_iter                             │
│  • Extract hyperparameters from loaded config (if available)       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: Filter & Prepare Data                                      │
├─────────────────────────────────────────────────────────────────────┤
│  • Start with X_original (full dataset)                            │
│  • Filter to wavelength range (min/max)                            │
│  • Apply excluded spectra (from outlier detection)                 │
│  • Apply validation set filter (CRITICAL!)                         │
│  • Reset DataFrame index (CRITICAL for CV!)                        │
│                                                                      │
│  Why Reset Index?                                                   │
│  After exclusions, index might be [0,1,2,5,7,9,...]               │
│  CV splitters use indices, so gaps cause fold mismatches!          │
│  Reset to [0,1,2,3,4,5,...] ensures deterministic folds           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
```

### Phase 2: Preprocessing Path Decision (Lines 4142-4443)

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: Determine Preprocessing Path                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  is_derivative = preprocess in ['sg1', 'sg2', 'snv_sg1', ...]      │
│  is_subset = len(selected_wl) < len(all_wavelengths)               │
│                                                                      │
│  use_full_spectrum_preprocessing = is_derivative AND is_subset      │
└────────────────────┬────────────────────────┬───────────────────────┘
                     │                        │
          ┌──────────▼──────────┐  ┌─────────▼──────────┐
          │  PATH A: TRUE       │  │  PATH B: FALSE     │
          │  (Derivative +      │  │  (Raw/SNV or       │
          │   Subset)           │  │   Full Spectrum)   │
          └──────────┬──────────┘  └─────────┬──────────┘
                     │                        │
                     ▼                        ▼
```

#### PATH A: Derivative + Subset (Lines 4392-4425)

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATH A: Derivative + Subset (Full-Spectrum Preprocessing)          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Build preprocessing pipeline (WITHOUT model)                    │
│     prep_pipeline = Pipeline([('snv', SNV()), ('deriv', SG())])    │
│                                                                      │
│  2. Preprocess FULL spectrum (all wavelengths)                      │
│     X_full_preprocessed = prep_pipeline.fit_transform(X_full)      │
│     [n_samples, all_wavelengths] → [n_samples, trimmed_wls]       │
│                                                                      │
│  3. Find indices of selected wavelengths in ORIGINAL data           │
│     wavelength_indices = [idx for wl in selected_wl                │
│                          where wl matches all_wavelengths[idx]]    │
│                                                                      │
│  4. Subset the PREPROCESSED data (not raw!)                         │
│     X_work = X_full_preprocessed[:, wavelength_indices]            │
│     [n_samples, trimmed_wls] → [n_samples, subset_wls]            │
│                                                                      │
│  5. Build pipeline with ONLY model (preprocessing already done)     │
│     pipe = Pipeline([('model', model)])                            │
│                                                                      │
│  Why This Approach?                                                 │
│  • Derivatives need context from neighboring wavelengths            │
│  • Computing derivative on subset loses this context                │
│  • This matches search.py behavior and fixes R² discrepancies      │
└─────────────────────────────────────────────────────────────────────┘
```

#### PATH B: Raw/SNV or Full Spectrum (Lines 4427-4442)

```
┌─────────────────────────────────────────────────────────────────────┐
│  PATH B: Raw/SNV or Full Spectrum (Standard Preprocessing)          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Subset raw data first                                           │
│     X_work = X_base_df[selected_wavelengths].values                │
│                                                                      │
│  2. Build full pipeline (preprocessing + model)                     │
│     pipe = Pipeline([('snv', SNV()), ('model', model)])            │
│                                                                      │
│  3. Preprocessing happens INSIDE CV loop                            │
│     Each fold: preprocess training data, fit model, predict         │
│                                                                      │
│  Why This Approach?                                                 │
│  • Raw/SNV don't need full-spectrum context                        │
│  • More efficient to preprocess inside CV                          │
│  • Standard scikit-learn pipeline pattern                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Model Creation & Hyperparameter Loading (Lines 4237-4376)

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: Create Model & Load Hyperparameters                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Get base model from spectral_predict.models                     │
│     model = get_model(model_name, task_type, n_components, ...)    │
│                                                                      │
│  2. Extract hyperparameters from loaded config (if available)       │
│                                                                      │
│     For each model type:                                            │
│     • PLS: n_components (from 'LVs' column)                        │
│     • Ridge/Lasso: alpha (from 'Alpha' or 'alpha')                 │
│     • RandomForest: n_estimators, max_features, max_depth          │
│     • MLP: learning_rate_init, hidden_layer_sizes                  │
│     • NeuralBoosted: n_estimators, learning_rate, hidden_size      │
│                                                                      │
│  3. Robust parameter extraction:                                    │
│     • Try multiple column name variants ('alpha' vs 'Alpha')       │
│     • Parse 'Params' column as JSON fallback                       │
│     • Use sensible defaults if not found                           │
│     • Log all parameter loading for debugging                      │
│                                                                      │
│  4. Apply parameters to model                                       │
│     model.set_params(**params_from_search)                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Cross-Validation (Lines 4381-4495)

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: Run Cross-Validation                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Create CV splitter (CRITICAL: shuffle=False!)                   │
│     cv = KFold(n_splits=n_folds, shuffle=False)                    │
│     # OR                                                            │
│     cv = StratifiedKFold(n_splits=n_folds, shuffle=False)          │
│                                                                      │
│     Why shuffle=False?                                              │
│     • Python and Julia use different RNG algorithms                 │
│     • Even with same seed, shuffled splits differ                   │
│     • shuffle=False ensures deterministic, order-based folds        │
│     • Results now match Julia backend exactly!                      │
│                                                                      │
│  2. Loop over folds                                                 │
│     ┌──────────────────────────────────────────┐                   │
│     │  For each fold:                          │                   │
│     │                                          │                   │
│     │  • Clone ENTIRE pipeline (not just model)│                   │
│     │    pipe_fold = clone(pipe)              │                   │
│     │                                          │                   │
│     │  • Split data                            │                   │
│     │    X_train, X_test = X[train_idx], ...  │                   │
│     │                                          │                   │
│     │  • Fit pipeline on training data         │                   │
│     │    pipe_fold.fit(X_train, y_train)      │                   │
│     │                                          │                   │
│     │  • Predict on test data                  │                   │
│     │    y_pred = pipe_fold.predict(X_test)   │                   │
│     │                                          │                   │
│     │  • Compute metrics                       │                   │
│     │    RMSE, R², MAE (regression)           │                   │
│     │    Acc, Prec, Rec, F1 (classification)  │                   │
│     │                                          │                   │
│     │  • Store predictions for plotting        │                   │
│     └──────────────────────────────────────────┘                   │
│                                                                      │
│  3. Aggregate metrics across folds                                  │
│     • Compute mean and std for each metric                         │
│     • Store all y_true and y_pred for overall plot                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 5: Prediction Intervals (Optional, PLS only) (Lines 4497-4552)

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: Compute Prediction Intervals (PLS only, n < 300)           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  if model_name == 'PLS' and len(X) < 300:                          │
│                                                                      │
│    1. Re-iterate through CV folds                                   │
│                                                                      │
│    2. For each fold:                                                │
│       • Fit pipeline on training data                               │
│       • Compute jackknife intervals on test data                    │
│         (leave-one-out within training set)                         │
│       • Store lower, upper bounds, and std_err                      │
│                                                                      │
│    3. Compute average standard error                                │
│       avg_std_err = mean(all_std_errors)                           │
│                                                                      │
│    Why Jackknife?                                                   │
│    • Provides realistic uncertainty estimates                       │
│    • No distributional assumptions needed                           │
│    • ±1 SE more interpretable than 95% CI                          │
│                                                                      │
│    Why n < 300?                                                     │
│    • Jackknife is O(n²) - expensive for large n                    │
│    • 1-2 minutes for n=200, impractical for n>300                  │
│                                                                      │
│  else:                                                              │
│    Skip interval computation (not applicable or too slow)           │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 6: Final Model Training (Lines 4628-4719)

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 7: Fit Final Model on All Data (for saving)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Clone pipeline                                                  │
│     final_pipe = clone(pipe)                                        │
│                                                                      │
│  2. Fit on ALL data (not just CV training sets)                     │
│     final_pipe.fit(X_work, y_array)                                │
│                                                                      │
│  3. Extract components for saving                                   │
│     final_model = final_pipe.named_steps['model']                  │
│     final_preprocessor = Pipeline(steps[:-1]) or prep_pipeline     │
│                                                                      │
│  4. Handle wavelength trimming (derivatives)                        │
│     • Derivatives remove edge wavelengths                           │
│     • Calculate how many wavelengths were trimmed                   │
│     • Store ACTUAL wavelengths model expects                        │
│                                                                      │
│     Example:                                                        │
│     Selected: [1400, 1410, 1420, ..., 1990, 2000]  (61 wls)       │
│     After SG deriv (window=11): trim 5 per side                    │
│     Final: [1450, 1460, ..., 1940, 1950]  (51 wls)                │
│                                                                      │
│  5. Store everything for persistence                                │
│     self.refined_model = final_model                               │
│     self.refined_preprocessor = final_preprocessor                 │
│     self.refined_wavelengths = wavelengths_after_trimming          │
│     self.refined_performance = results                             │
│     self.refined_config = {...}                                    │
│                                                                      │
│  6. Enable Save Model button                                        │
│     self.refine_save_button.config(state='normal')                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 7: UI Update (Lines 4721-4753)

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 8: Update UI (Thread-Safe)                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Format results text                                             │
│     ┌─────────────────────────────────────────────────┐            │
│     │  Refined Model Results:                         │            │
│     │                                                  │            │
│     │  Cross-Validation Performance (5 folds):        │            │
│     │    RMSE: 3.62 ± 0.45                           │            │
│     │    R²: 0.876 ± 0.023                           │            │
│     │    MAE: 2.87 ± 0.34                            │            │
│     │                                                  │            │
│     │  COMPARISON TO LOADED MODEL:                    │            │
│     │    Original R² (from Results tab): 0.875        │            │
│     │    Refined R² (just computed):     0.876        │            │
│     │    Difference:                     +0.001       │            │
│     │                                                  │            │
│     │  Configuration:                                 │            │
│     │    Model: PLS                                   │            │
│     │    Preprocessing: sg2                           │            │
│     │    Window Size: 17                              │            │
│     │    Wavelengths: 50 wavelengths (1500-1724 nm)  │            │
│     │    Features: 50                                 │            │
│     │    Samples: 180                                 │            │
│     │    CV Folds: 5                                  │            │
│     │    n_components: 10                             │            │
│     │                                                  │            │
│     │  DEBUG INFO:                                    │            │
│     │    Loaded n_components from config: 10          │            │
│     │    Processing Path: Full-spectrum preprocessing │            │
│     │    Fold splits: deterministic (shuffle=False)   │            │
│     └─────────────────────────────────────────────────┘            │
│                                                                      │
│  2. Update results display (via self.root.after for thread safety)  │
│     self.root.after(0, lambda: self._update_refined_results(...))  │
│                                                                      │
│  3. _update_refined_results() [main thread]:                        │
│     • Update text widget                                            │
│     • Re-enable Run button                                          │
│     • Enable Save button (on success)                               │
│     • Update status label                                           │
│     • Generate plots:                                               │
│       - Prediction plot (with error bars)                           │
│       - Residual diagnostics (3 panels)                             │
│       - Leverage analysis (hat values)                              │
│     • Show success messagebox                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Exception Handling in _run_refined_model_thread()                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  try:                                                               │
│      [All the heavy computation above]                              │
│                                                                      │
│  except Exception as e:                                             │
│      1. Capture full traceback                                      │
│         error_msg = traceback.format_exc()                         │
│                                                                      │
│      2. Format error message                                        │
│         error_text = f"Error: {e}\n\n{error_msg}"                  │
│                                                                      │
│      3. Update UI (thread-safe)                                     │
│         self.root.after(0, lambda: self._update_refined_results(   │
│             error_text, is_error=True))                            │
│                                                                      │
│      4. _update_refined_results() with is_error=True:              │
│         • Display error in results text widget                      │
│         • Re-enable Run button                                      │
│         • Disable Save button                                       │
│         • Update status: "Error running refined model"              │
│         • Show error messagebox                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Thread Safety

### Why Threading?

```
┌─────────────────────────────────────────────────────────────────────┐
│  Problem: Heavy Computation Blocks GUI                              │
│                                                                      │
│  Without threading:                                                 │
│    User clicks Run → GUI freezes → Computation done → GUI unfreezes │
│    ❌ Poor user experience                                          │
│    ❌ Can't cancel                                                  │
│    ❌ Looks like app crashed                                        │
│                                                                      │
│  With threading:                                                    │
│    User clicks Run → GUI responsive → Computation in background     │
│    ✓ User can interact with other tabs                             │
│    ✓ Console shows progress                                         │
│    ✓ Can implement cancel button (future)                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Thread-Safe UI Updates

```
┌─────────────────────────────────────────────────────────────────────┐
│  Rule: NEVER modify GUI from background thread!                     │
│                                                                      │
│  ❌ WRONG (causes crashes):                                         │
│    def _run_refined_model_thread(self):                            │
│        results = compute_results()                                  │
│        self.refine_results_text.insert('1.0', results)  # CRASH!   │
│                                                                      │
│  ✓ CORRECT (use self.root.after):                                  │
│    def _run_refined_model_thread(self):                            │
│        results = compute_results()                                  │
│        self.root.after(0, lambda: self._update_ui(results))        │
│                                                                      │
│    def _update_ui(self, results):  # Runs on main thread           │
│        self.refine_results_text.insert('1.0', results)  # Safe!    │
│                                                                      │
│  self.root.after(0, callback) schedules callback to run on          │
│  main thread's next event loop iteration.                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Complete Execution Pipeline

```
User Clicks Run
      ↓
Validate Inputs
      ↓
Disable Run Button
      ↓
Launch Background Thread
      ↓
┌─────────────────────────────────────┐
│  BACKGROUND THREAD                  │
│                                     │
│  1. Parse Parameters               │
│  2. Filter Data                    │
│  3. Build Preprocessing Pipeline   │
│  4. Create Model                   │
│  5. Run Cross-Validation           │
│  6. Compute Prediction Intervals   │
│  7. Fit Final Model                │
│  8. Schedule UI Update              │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  MAIN THREAD (UI UPDATE)            │
│                                     │
│  • Update results text              │
│  • Generate plots                   │
│  • Re-enable buttons                │
│  • Show success message             │
└─────────────────────────────────────┘
      ↓
User Reviews Results
      ↓
User Clicks Save Model
      ↓
Model Saved to .dasp File
      ↓
Done!
```

---

## Key Design Principles

1. **Thread Safety:** All UI updates via `self.root.after()`
2. **Reproducibility:** `shuffle=False` for deterministic CV
3. **Data Consistency:** Reset index after exclusions
4. **Context Preservation:** Full-spectrum preprocessing for derivatives
5. **Error Handling:** Comprehensive try/except with user-friendly messages
6. **Performance:** Skip expensive operations (jackknife) when n > 300
7. **Debugging:** Console logging at every major step
8. **Metadata:** Store all config for reproducibility and persistence

---

**End of Flow Diagram**
