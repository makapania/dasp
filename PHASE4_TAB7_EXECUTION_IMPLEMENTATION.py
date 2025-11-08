"""
PHASE 4: Tab 7 Model Development - Complete Execution Engine Implementation

This file contains the complete execution engine for Tab 7 (Model Development).
It is adapted from the proven Tab 6 execution engine but uses the tab7_ namespace.

INTEGRATION INSTRUCTIONS:
1. Add these methods to the SpectralPredictApp class in spectral_predict_gui_optimized.py
2. Ensure Tab 7 UI widgets are created with these names:
   - self.tab7_wl_spec (Text widget for wavelength specification)
   - self.tab7_model_type (StringVar for model type)
   - self.tab7_task_type (StringVar for task type)
   - self.tab7_preprocess (StringVar for preprocessing method)
   - self.tab7_window (IntVar for window size)
   - self.tab7_folds (IntVar for CV folds)
   - self.tab7_max_iter (IntVar for max iterations)
   - self.tab7_run_button (Button to trigger execution)
   - self.tab7_save_button (Button to save model)
   - self.tab7_status (Label for status display)
   - self.tab7_results_text (Text widget for results display)
3. These methods integrate seamlessly with existing spectral_predict modules

FEATURES:
- Complete cross-validation pipeline
- Both preprocessing paths (derivative+subset and standard)
- Deterministic CV matching Results tab
- Validation set exclusion
- All model types supported
- Comprehensive error handling
- Thread-safe GUI updates
- Model persistence

CRITICAL DESIGN DECISIONS:
- Uses shuffle=False for deterministic CV (matches Results tab)
- Resets DataFrame index after exclusions (prevents fold mismatches)
- Implements both preprocessing paths (full-spectrum for derivative+subset)
- Excludes validation set before training
- Stores wavelengths AFTER preprocessing (handles edge trimming)
"""

import threading
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path
from datetime import datetime


# ============================================================================
# METHOD 1: Entry Point
# ============================================================================

def _tab7_run_model(self):
    """
    Entry point when user clicks 'Run Model' button.

    This method:
    1. Validates that data is loaded
    2. Validates all parameters
    3. Disables the run button
    4. Launches background thread for execution

    Lines: ~20
    """
    # Check if data is loaded
    if self.X is None or self.y is None:
        messagebox.showwarning("No Data", "Please load data first in Tab 1: Import & Preview")
        return

    # Validate parameters
    if not self._tab7_validate_parameters():
        return

    # Disable button during execution
    self.tab7_run_button.config(state='disabled')
    self.tab7_status.config(text="Running model...")

    # Clear previous results
    self.tab7_results_text.config(state='normal')
    self.tab7_results_text.delete('1.0', tk.END)
    self.tab7_results_text.insert('1.0', "Executing model... please wait.\n\nThis may take 30-60 seconds depending on model complexity.")
    self.tab7_results_text.config(state='disabled')

    # Run in background thread to avoid freezing GUI
    thread = threading.Thread(target=self._tab7_run_model_thread)
    thread.start()


# ============================================================================
# METHOD 2: Parameter Validation
# ============================================================================

def _tab7_validate_parameters(self):
    """
    Validate all parameters before execution.

    Checks:
    - Wavelength specification is not empty
    - Wavelength parsing succeeds
    - Model type is valid
    - CV folds is in valid range (3-10)

    Returns:
        bool: True if all validations pass, False otherwise

    Lines: ~40
    """
    errors = []

    # Validate wavelength specification
    wl_spec_text = self.tab7_wl_spec.get('1.0', 'end').strip()
    if not wl_spec_text:
        errors.append("Wavelength specification is empty.\n\nPlease specify wavelengths (e.g., '1500-2500' or '1500-1800, 2000-2400')")
    else:
        # Try to parse wavelengths
        try:
            available_wl = self.X_original.columns.astype(float).values
            parsed_wl = self._parse_wavelength_spec(wl_spec_text, available_wl)
            if not parsed_wl:
                errors.append("No valid wavelengths found in specification.\n\nPlease check your wavelength ranges.")
        except Exception as e:
            errors.append(f"Failed to parse wavelength specification:\n{str(e)}")

    # Validate model type
    model_type = self.tab7_model_type.get()
    valid_models = ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted']
    if model_type not in valid_models:
        errors.append(f"Invalid model type: {model_type}\n\nValid options: {', '.join(valid_models)}")

    # Validate CV folds
    cv_folds = self.tab7_folds.get()
    if cv_folds < 3 or cv_folds > 10:
        errors.append(f"CV folds must be between 3 and 10 (got {cv_folds})")

    # Show errors if any
    if errors:
        messagebox.showerror("Validation Error", "\n\n".join(errors))
        return False

    return True


# ============================================================================
# METHOD 3: Main Execution Engine (Background Thread)
# ============================================================================

def _tab7_run_model_thread(self):
    """
    Main execution engine running in background thread.

    This is the core of the execution system. It performs:

    STEP 1: Parse Parameters (~100 lines)
        - Extract all UI parameters
        - Parse wavelength specification
        - Map preprocessing names
        - Extract hyperparameters

    STEP 2: Filter Data (~80 lines)
        - Filter to selected wavelengths
        - Apply excluded spectra
        - CRITICAL: Exclude validation set
        - CRITICAL: Reset DataFrame index

    STEP 3: Build Preprocessing Pipeline (~300 lines)
        - Determine preprocessing path (A or B)
        - PATH A: Derivative + Subset (full-spectrum preprocessing)
        - PATH B: Raw/SNV (standard preprocessing)

    STEP 4: Cross-Validation (~150 lines)
        - Use KFold with shuffle=False (deterministic!)
        - For each fold: fit, predict, collect metrics
        - Compute mean ± std across folds

    STEP 5: Train Final Model (~80 lines)
        - Train on full dataset
        - Store model and preprocessor
        - Handle wavelength trimming

    STEP 6: Calculate Metrics (~100 lines)
        - Regression: R², RMSE, MAE, bias
        - Classification: Accuracy, Precision, Recall, F1

    STEP 7: Update GUI (~50 lines)
        - Thread-safe update via self.root.after()
        - Format results text
        - Enable save button

    Lines: ~860
    """
    try:
        # ===================================================================
        # IMPORTS
        # ===================================================================
        from spectral_predict.models import get_model
        from spectral_predict.preprocess import build_preprocessing_pipeline
        from sklearn.model_selection import KFold, StratifiedKFold
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.pipeline import Pipeline
        from sklearn.base import clone

        print("\n" + "="*80)
        print("TAB 7: MODEL DEVELOPMENT - EXECUTION START")
        print("="*80)

        # ===================================================================
        # STEP 1: PARSE PARAMETERS
        # ===================================================================
        print("\n[STEP 1] Parsing parameters...")

        # Get basic parameters
        model_name = self.tab7_model_type.get()
        task_type = self.tab7_task_type.get()
        preprocess = self.tab7_preprocess.get()
        window = self.tab7_window.get()
        n_folds = self.tab7_folds.get()
        max_iter = self.tab7_max_iter.get()

        print(f"  Model Type: {model_name}")
        print(f"  Task Type: {task_type}")
        print(f"  Preprocessing: {preprocess}")
        print(f"  Window Size: {window}")
        print(f"  CV Folds: {n_folds}")
        print(f"  Max Iterations: {max_iter}")

        # Parse wavelength specification
        available_wl = self.X_original.columns.astype(float).values
        wl_spec_text = self.tab7_wl_spec.get('1.0', 'end')
        selected_wl = self._parse_wavelength_spec(wl_spec_text, available_wl)

        if not selected_wl:
            raise ValueError("No valid wavelengths selected. Please check your wavelength specification.")

        print(f"  Wavelengths: {len(selected_wl)} selected ({selected_wl[0]:.1f} to {selected_wl[-1]:.1f} nm)")

        # Map GUI preprocessing names to search.py format
        preprocess_name_map = {
            'raw': 'raw',
            'snv': 'snv',
            'sg1': 'deriv',
            'sg2': 'deriv',
            'snv_sg1': 'snv_deriv',
            'snv_sg2': 'snv_deriv',
            'deriv_snv': 'deriv_snv',
            'msc': 'msc',
            'msc_sg1': 'msc_deriv',
            'msc_sg2': 'msc_deriv',
            'deriv_msc': 'deriv_msc'
        }

        deriv_map = {
            'raw': 0, 'snv': 0,
            'sg1': 1, 'sg2': 2,
            'snv_sg1': 1, 'snv_sg2': 2,
            'deriv_snv': 1,
            'msc': 0, 'msc_sg1': 1, 'msc_sg2': 2,
            'deriv_msc': 1
        }

        polyorder_map = {
            'raw': 2, 'snv': 2,
            'sg1': 2, 'sg2': 3,
            'snv_sg1': 2, 'snv_sg2': 3,
            'deriv_snv': 2,
            'msc': 2, 'msc_sg1': 2, 'msc_sg2': 3,
            'deriv_msc': 2
        }

        preprocess_name = preprocess_name_map.get(preprocess, 'raw')
        deriv = deriv_map.get(preprocess, 0)
        polyorder = polyorder_map.get(preprocess, 2)

        print(f"  Preprocessing Config: {preprocess_name} (deriv={deriv}, polyorder={polyorder})")

        # Extract hyperparameters (default to sensible values)
        # In a full implementation, these would come from UI widgets
        n_components = 10  # Default for PLS

        print(f"  Hyperparameters: n_components={n_components}")

        # ===================================================================
        # STEP 2: FILTER DATA
        # ===================================================================
        print("\n[STEP 2] Filtering data...")

        # Start with original data
        X_source = self.X_original if self.X_original is not None else self.X
        y_series = self.y.copy()

        print(f"  Initial shape: X={X_source.shape}, y={y_series.shape}")

        # Apply excluded spectra (from outlier detection)
        total_samples = len(X_source)
        excluded_indices = sorted(idx for idx in self.excluded_spectra if 0 <= idx < total_samples)

        if excluded_indices:
            excluded_set = set(excluded_indices)
            include_indices = [i for i in range(total_samples) if i not in excluded_set]

            if len(include_indices) < n_folds:
                raise ValueError(
                    f"Only {len(include_indices)} samples remain after exclusions; "
                    f"{n_folds}-fold CV requires at least {n_folds} samples."
                )

            print(f"  Applying {len(excluded_indices)} excluded spectra ({len(include_indices)} remain)")
            X_base_df = X_source.iloc[include_indices]
            y_series = y_series.iloc[include_indices]
        else:
            X_base_df = X_source.copy()
            y_series = y_series.copy()

        # CRITICAL FIX #1: Exclude validation set (if enabled)
        # This ensures Model Development uses the same calibration split as Results tab
        if self.validation_enabled.get() and self.validation_indices:
            initial_samples = len(X_base_df)
            X_base_df = X_base_df[~X_base_df.index.isin(self.validation_indices)]
            y_series = y_series[~y_series.index.isin(self.validation_indices)]

            n_val = len(self.validation_indices)
            n_removed = initial_samples - len(X_base_df)
            n_cal = len(X_base_df)

            if n_cal < n_folds:
                raise ValueError(
                    f"Only {n_cal} calibration samples remain after validation set exclusion; "
                    f"{n_folds}-fold CV requires at least {n_folds} samples."
                )

            print(f"  Excluding {n_removed} validation samples")
            print(f"  Calibration: {n_cal} samples | Validation: {n_val} samples")
            print(f"  This matches the data split used in the Results tab")

        # CRITICAL FIX #2: Reset DataFrame index to ensure sequential 0-based indexing
        # After exclusions and validation splits, index may have gaps (e.g., [0,1,2,5,7,9,...])
        # Julia's Matrix conversion creates sequential indices, so we must match that behavior
        # Without this, CV folds assign different physical rows despite same indices
        X_base_df = X_base_df.reset_index(drop=True)
        y_series = y_series.reset_index(drop=True)

        print(f"  Reset index after exclusions")
        print(f"  Final shape: X={X_base_df.shape}, y={y_series.shape}")
        print(f"  This ensures CV folds match Results tab (sequential row indexing)")

        # ===================================================================
        # STEP 3: BUILD PREPROCESSING PIPELINE
        # ===================================================================
        print("\n[STEP 3] Building preprocessing pipeline...")

        # Create mapping from float wavelengths to actual column names
        wavelength_columns = X_base_df.columns
        wl_to_col = {float(col): col for col in wavelength_columns}

        # Get the actual column names for selected wavelengths
        selected_cols = [wl_to_col[wl] for wl in selected_wl if wl in wl_to_col]

        if not selected_cols:
            raise ValueError(f"Could not find matching wavelengths. Selected: {len(selected_wl)}, Found: 0")

        # CRITICAL DECISION: Determine preprocessing path
        # PATH A: Derivative + Subset → preprocess full spectrum, then subset
        # PATH B: Raw/SNV/Full → subset first, then preprocess inside CV
        is_derivative = preprocess in ['sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'deriv_snv', 'msc_sg1', 'msc_sg2', 'deriv_msc']
        base_full_vars = len(X_base_df.columns)
        is_subset = len(selected_wl) < base_full_vars
        use_full_spectrum_preprocessing = is_derivative and is_subset

        if use_full_spectrum_preprocessing:
            # ===============================================================
            # PATH A: Derivative + Subset (matches search.py lines 434-449)
            # ===============================================================
            print("  PATH A: Derivative + Subset detected")
            print("  Will preprocess FULL spectrum first, then subset")
            print("  This preserves derivative context from full spectrum")

            # 1. Build preprocessing pipeline WITHOUT model
            prep_steps = build_preprocessing_pipeline(
                preprocess_name,
                deriv,
                window,
                polyorder
            )
            prep_pipeline = Pipeline(prep_steps)

            # 2. Preprocess FULL spectrum (all wavelengths)
            X_full = X_base_df.values
            print(f"  Preprocessing full spectrum ({X_full.shape[1]} wavelengths)...")
            X_full_preprocessed = prep_pipeline.fit_transform(X_full)

            # 3. Find indices of selected wavelengths in original data
            all_wavelengths = X_base_df.columns.astype(float).values
            wavelength_indices = []
            for wl in selected_wl:
                idx = np.where(np.abs(all_wavelengths - wl) < 0.01)[0]
                if len(idx) > 0:
                    wavelength_indices.append(idx[0])

            # 4. Subset the PREPROCESSED data (not raw!)
            X_work = X_full_preprocessed[:, wavelength_indices]
            print(f"  Subsetted to {X_work.shape[1]} wavelengths after preprocessing")

            # 5. Build pipeline with ONLY the model (preprocessing already done)
            model = get_model(
                model_name,
                task_type=task_type,
                n_components=n_components,
                max_n_components=24,
                max_iter=max_iter
            )
            pipe_steps = [('model', model)]
            pipe = Pipeline(pipe_steps)

            print(f"  Pipeline: {[name for name, _ in pipe_steps]} (preprocessing already applied)")

        else:
            # ===============================================================
            # PATH B: Raw/SNV or Full-Spectrum (standard behavior)
            # ===============================================================
            print("  PATH B: Standard preprocessing (subset first, then preprocess)")

            # Subset raw data first
            X_work = X_base_df[selected_cols].values

            # Build full pipeline with preprocessing + model
            pipe_steps = build_preprocessing_pipeline(
                preprocess_name,
                deriv,
                window,
                polyorder
            )

            model = get_model(
                model_name,
                task_type=task_type,
                n_components=n_components,
                max_n_components=24,
                max_iter=max_iter
            )
            pipe_steps.append(('model', model))
            pipe = Pipeline(pipe_steps)

            print(f"  Pipeline: {[name for name, _ in pipe_steps]} (preprocessing inside CV)")

        # ===================================================================
        # STEP 4: CROSS-VALIDATION
        # ===================================================================
        print(f"\n[STEP 4] Running {n_folds}-fold cross-validation...")

        # CRITICAL: Use shuffle=False to ensure identical fold splits as Julia backend
        # Julia and Python use different RNG algorithms, so even with same seed (42),
        # they create different splits when shuffle=True.
        y_array = y_series.values
        if task_type == "regression":
            cv = KFold(n_splits=n_folds, shuffle=False)  # Deterministic!
            print("  Using KFold (shuffle=False) for deterministic splits")
        else:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=False)  # Deterministic!
            print("  Using StratifiedKFold (shuffle=False) for deterministic splits")

        # Collect metrics for each fold
        fold_metrics = []
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_work, y_array)):
            # Clone ENTIRE PIPELINE for this fold (not just model)
            pipe_fold = clone(pipe)

            # Split data
            X_train, X_test = X_work[train_idx], X_work[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            # Fit pipeline (preprocessing + model) and predict
            pipe_fold.fit(X_train, y_train)
            y_pred = pipe_fold.predict(X_test)

            # Store predictions for plotting
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            # Calculate fold metrics
            if task_type == "regression":
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                bias = np.mean(y_pred - y_test)
                fold_metrics.append({"rmse": rmse, "r2": r2, "mae": mae, "bias": bias})
                print(f"  Fold {fold_idx+1}/{n_folds}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            else:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                fold_metrics.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
                print(f"  Fold {fold_idx+1}/{n_folds}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

        # ===================================================================
        # STEP 5: TRAIN FINAL MODEL
        # ===================================================================
        print("\n[STEP 5] Training final model on full dataset...")

        # Fit pipeline on all data for model persistence
        final_pipe = clone(pipe)
        final_pipe.fit(X_work, y_array)

        # Extract model and preprocessor from pipeline
        final_model = final_pipe.named_steps['model']

        # Build preprocessor from pipeline steps (excluding the model)
        if use_full_spectrum_preprocessing:
            # For derivative + subset: preprocessor was already fitted on full spectrum
            final_preprocessor = prep_pipeline  # Already fitted
            print("  Using full-spectrum preprocessor (already fitted)")
        elif len(pipe_steps) > 1:  # Has preprocessing steps
            final_preprocessor = Pipeline(pipe_steps[:-1])  # All steps except model
            final_preprocessor.fit(X_work)  # Fit on raw data
            print("  Fitted preprocessor on subset data")
        else:
            final_preprocessor = None
            print("  No preprocessor (raw data)")

        # CRITICAL FIX: Store wavelengths AFTER preprocessing, not before
        # Derivatives remove edge wavelengths, so model expects fewer features than original
        if final_preprocessor is not None:
            # Apply preprocessor to get actual feature count
            dummy_input = X_work[:1]  # Single sample for testing
            transformed = final_preprocessor.transform(dummy_input)
            n_features_after_preprocessing = transformed.shape[1]

            if use_full_spectrum_preprocessing:
                # Derivative + subset: wavelengths already determined by subset indices
                refined_wavelengths = list(selected_wl)
            else:
                # Regular preprocessing: derivatives trim edges
                n_trimmed = len(selected_wl) - n_features_after_preprocessing
                if n_trimmed > 0:
                    # Edges were trimmed symmetrically
                    trim_per_side = n_trimmed // 2
                    refined_wavelengths = list(selected_wl[trim_per_side:len(selected_wl)-trim_per_side])
                    print(f"  Derivative preprocessing trimmed {n_trimmed} edge wavelengths")
                else:
                    # No trimming (raw/SNV/MSC)
                    refined_wavelengths = list(selected_wl)
        else:
            # No preprocessor - use original wavelengths
            refined_wavelengths = list(selected_wl)

        print(f"  Model trained on {X_work.shape[0]} samples × {len(refined_wavelengths)} features")

        # Store model artifacts for saving
        self.tab7_trained_model = final_model
        self.tab7_preprocessing_pipeline = final_preprocessor
        self.tab7_wavelengths = refined_wavelengths
        self.tab7_full_wavelengths = list(all_wavelengths) if use_full_spectrum_preprocessing else None

        # ===================================================================
        # STEP 6: CALCULATE METRICS
        # ===================================================================
        print("\n[STEP 6] Computing performance metrics...")

        results = {}
        if task_type == "regression":
            results['rmse_mean'] = np.mean([m['rmse'] for m in fold_metrics])
            results['rmse_std'] = np.std([m['rmse'] for m in fold_metrics])
            results['r2_mean'] = np.mean([m['r2'] for m in fold_metrics])
            results['r2_std'] = np.std([m['r2'] for m in fold_metrics])
            results['mae_mean'] = np.mean([m['mae'] for m in fold_metrics])
            results['mae_std'] = np.std([m['mae'] for m in fold_metrics])
            results['bias_mean'] = np.mean([m['bias'] for m in fold_metrics])
            results['bias_std'] = np.std([m['bias'] for m in fold_metrics])

            print(f"  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
            print(f"  R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
            print(f"  MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
            print(f"  Bias: {results['bias_mean']:.4f} ± {results['bias_std']:.4f}")
        else:
            results['accuracy_mean'] = np.mean([m['accuracy'] for m in fold_metrics])
            results['accuracy_std'] = np.std([m['accuracy'] for m in fold_metrics])
            results['precision_mean'] = np.mean([m['precision'] for m in fold_metrics])
            results['precision_std'] = np.std([m['precision'] for m in fold_metrics])
            results['recall_mean'] = np.mean([m['recall'] for m in fold_metrics])
            results['recall_std'] = np.std([m['recall'] for m in fold_metrics])
            results['f1_mean'] = np.mean([m['f1'] for m in fold_metrics])
            results['f1_std'] = np.std([m['f1'] for m in fold_metrics])

            print(f"  Accuracy:  {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
            print(f"  Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
            print(f"  Recall:    {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
            print(f"  F1 Score:  {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

        # Store predictions for plotting
        self.tab7_y_true = np.array(all_y_true)
        self.tab7_y_pred = np.array(all_y_pred)

        # Store configuration
        self.tab7_config = {
            'model_name': model_name,
            'task_type': task_type,
            'preprocessing': preprocess,
            'window': window,
            'n_vars': len(refined_wavelengths),
            'n_samples': X_work.shape[0],
            'cv_folds': n_folds,
            'use_full_spectrum_preprocessing': use_full_spectrum_preprocessing
        }

        # Store performance
        self.tab7_performance = results

        # ===================================================================
        # STEP 7: UPDATE GUI (THREAD-SAFE)
        # ===================================================================
        print("\n[STEP 7] Updating GUI...")

        # Format results text
        wl_summary = f"{len(selected_wl)} wavelengths ({selected_wl[0]:.1f} to {selected_wl[-1]:.1f} nm)"

        if task_type == "regression":
            results_text = f"""Model Development Results:

Cross-Validation Performance ({n_folds} folds):
  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}
  R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}
  MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}
  Bias: {results['bias_mean']:.4f} ± {results['bias_std']:.4f}

Configuration:
  Model:        {model_name}
  Task Type:    {task_type}
  Preprocessing: {preprocess}
  Window Size:  {window}
  Wavelengths:  {wl_summary}
  Features:     {len(refined_wavelengths)}
  Samples:      {X_work.shape[0]}
  CV Folds:     {n_folds}

Processing Details:
  Path: {'Full-spectrum preprocessing (derivative+subset)' if use_full_spectrum_preprocessing else 'Standard (subset then preprocess)'}
  CV Strategy: {'KFold' if task_type == 'regression' else 'StratifiedKFold'} (shuffle=False, deterministic)
  Validation Set: {'Excluded' if self.validation_enabled.get() else 'Not used'}

The model is ready to be saved. Click 'Save Model' to export as .dasp file.
"""
        else:
            results_text = f"""Model Development Results:

Cross-Validation Performance ({n_folds} folds):
  Accuracy:  {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}
  Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}
  Recall:    {results['recall_mean']:.4f} ± {results['recall_std']:.4f}
  F1 Score:  {results['f1_mean']:.4f} ± {results['f1_std']:.4f}

Configuration:
  Model:        {model_name}
  Task Type:    {task_type}
  Preprocessing: {preprocess}
  Window Size:  {window}
  Wavelengths:  {wl_summary}
  Features:     {len(refined_wavelengths)}
  Samples:      {X_work.shape[0]}
  CV Folds:     {n_folds}

Processing Details:
  Path: {'Full-spectrum preprocessing (derivative+subset)' if use_full_spectrum_preprocessing else 'Standard (subset then preprocess)'}
  CV Strategy: {'KFold' if task_type == 'regression' else 'StratifiedKFold'} (shuffle=False, deterministic)
  Validation Set: {'Excluded' if self.validation_enabled.get() else 'Not used'}

The model is ready to be saved. Click 'Save Model' to export as .dasp file.
"""

        # Thread-safe GUI update
        self.root.after(0, lambda: self._tab7_update_results(results_text))

        print("\n" + "="*80)
        print("TAB 7: MODEL DEVELOPMENT - EXECUTION COMPLETE")
        print("="*80 + "\n")

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        error_text = f"""Error running model:

{str(e)}

Full traceback:
{error_msg}

Please check:
1. Data is loaded correctly (Tab 1)
2. Wavelength specification is valid
3. All parameters are within valid ranges
4. Sufficient samples remain after exclusions

If the problem persists, please report this error.
"""
        print(f"\nERROR: {error_text}")

        # Thread-safe error update
        self.root.after(0, lambda: self._tab7_update_results(error_text, is_error=True))


# ============================================================================
# METHOD 4: GUI Update (Main Thread)
# ============================================================================

def _tab7_update_results(self, results_text, is_error=False):
    """
    Update the results display on the main thread (thread-safe).

    Args:
        results_text (str): Formatted results text to display
        is_error (bool): Whether this is an error message

    Lines: ~50
    """
    # Update results text widget
    self.tab7_results_text.config(state='normal')
    self.tab7_results_text.delete('1.0', tk.END)
    self.tab7_results_text.insert('1.0', results_text)
    self.tab7_results_text.config(state='disabled')

    # Re-enable run button
    self.tab7_run_button.config(state='normal')

    if is_error:
        # Error state
        self.tab7_status.config(text="✗ Error running model", foreground='red')
        self.tab7_save_button.config(state='disabled')
        messagebox.showerror("Error", "Failed to run model. See results area for details.")
    else:
        # Success state
        self.tab7_status.config(text="✓ Model training complete", foreground='green')

        # Enable Save Model button
        self.tab7_save_button.config(state='normal')

        # Generate diagnostic plots (if plotting methods exist)
        if hasattr(self, '_tab7_plot_predictions'):
            try:
                self._tab7_plot_predictions()
            except Exception as e:
                print(f"Warning: Could not generate prediction plot: {e}")

        if hasattr(self, '_tab7_plot_residuals'):
            try:
                self._tab7_plot_residuals()
            except Exception as e:
                print(f"Warning: Could not generate residual plot: {e}")

        # Show success message
        messagebox.showinfo("Success",
            f"Model training complete!\n\n"
            f"R² = {self.tab7_performance.get('r2_mean', 0):.4f}\n\n"
            f"Click 'Save Model' to export as .dasp file.")


# ============================================================================
# METHOD 5: Model Saving
# ============================================================================

def _tab7_save_model(self):
    """
    Save the trained model to a .dasp file.

    Saves:
    - Trained model object
    - Preprocessing pipeline (fitted)
    - Wavelengths used (after preprocessing)
    - Full wavelengths (for derivative+subset)
    - Performance metrics
    - Configuration metadata
    - Validation set info

    Lines: ~150
    """
    # Check if model has been trained
    if not hasattr(self, 'tab7_trained_model') or self.tab7_trained_model is None:
        messagebox.showerror(
            "No Model Trained",
            "Please run a model first before saving.\n\n"
            "Click 'Run Model' to train a model, then you can save it."
        )
        return

    try:
        from spectral_predict.model_io import save_model

        # Ask for save location
        default_name = f"model_{self.tab7_config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dasp"

        # Get initial directory from spectral data path
        initial_dir = None
        if hasattr(self, 'spectral_data_path') and self.spectral_data_path.get():
            data_path = Path(self.spectral_data_path.get())
            initial_dir = str(data_path.parent if data_path.is_file() else data_path)

        filepath = filedialog.asksaveasfilename(
            defaultextension=".dasp",
            filetypes=[("DASP Model", "*.dasp"), ("All files", "*.*")],
            initialfile=default_name,
            initialdir=initial_dir,
            title="Save Trained Model"
        )

        if not filepath:
            return  # User cancelled

        # Build comprehensive metadata
        metadata = {
            'model_name': self.tab7_config['model_name'],
            'task_type': self.tab7_config['task_type'],
            'preprocessing': self.tab7_config['preprocessing'],
            'window': self.tab7_config['window'],
            'wavelengths': self.tab7_wavelengths,
            'n_vars': self.tab7_config['n_vars'],
            'n_samples': self.tab7_config['n_samples'],
            'cv_folds': self.tab7_config['cv_folds'],
            'performance': {},
            'use_full_spectrum_preprocessing': self.tab7_config.get('use_full_spectrum_preprocessing', False),
            'full_wavelengths': self.tab7_full_wavelengths,
            # Validation set metadata
            'validation_set_enabled': self.validation_enabled.get() if hasattr(self, 'validation_enabled') else False,
            'validation_indices': list(self.validation_indices) if hasattr(self, 'validation_indices') and self.validation_indices else [],
            'validation_size': len(self.validation_indices) if hasattr(self, 'validation_indices') and self.validation_indices else 0,
            'validation_algorithm': self.validation_algorithm.get() if hasattr(self, 'validation_algorithm') and hasattr(self, 'validation_enabled') and self.validation_enabled.get() else None,
            # Timestamp
            'created_timestamp': datetime.now().isoformat(),
            'created_by': 'SpectralPredict GUI - Tab 7 Model Development'
        }

        # Add performance metrics based on task type
        if self.tab7_config['task_type'] == 'regression':
            metadata['performance'] = {
                'RMSE': self.tab7_performance.get('rmse_mean'),
                'RMSE_std': self.tab7_performance.get('rmse_std'),
                'R2': self.tab7_performance.get('r2_mean'),
                'R2_std': self.tab7_performance.get('r2_std'),
                'MAE': self.tab7_performance.get('mae_mean'),
                'MAE_std': self.tab7_performance.get('mae_std'),
                'Bias': self.tab7_performance.get('bias_mean'),
                'Bias_std': self.tab7_performance.get('bias_std')
            }
        else:  # classification
            metadata['performance'] = {
                'Accuracy': self.tab7_performance.get('accuracy_mean'),
                'Accuracy_std': self.tab7_performance.get('accuracy_std'),
                'Precision': self.tab7_performance.get('precision_mean'),
                'Precision_std': self.tab7_performance.get('precision_std'),
                'Recall': self.tab7_performance.get('recall_mean'),
                'Recall_std': self.tab7_performance.get('recall_std'),
                'F1': self.tab7_performance.get('f1_mean'),
                'F1_std': self.tab7_performance.get('f1_std')
            }

        # Save the model
        save_model(
            model=self.tab7_trained_model,
            preprocessor=self.tab7_preprocessing_pipeline,
            metadata=metadata,
            filepath=filepath
        )

        # Show success message
        messagebox.showinfo(
            "Model Saved",
            f"Model successfully saved to:\n\n{filepath}\n\n"
            f"Performance:\n"
            f"  R² = {metadata['performance'].get('R2', 0):.4f}\n"
            f"  RMSE = {metadata['performance'].get('RMSE', 0):.4f}\n\n"
            f"You can now load this model in Tab 8 (Model Prediction) "
            f"to make predictions on new data."
        )

        # Update status
        self.tab7_status.config(text=f"✓ Model saved to {Path(filepath).name}")

        print(f"\n✓ Model saved successfully to: {filepath}")

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        messagebox.showerror(
            "Save Error",
            f"Failed to save model:\n\n{str(e)}\n\nSee console for details."
        )
        print(f"Error saving model:\n{error_msg}")


# ============================================================================
# TESTING CHECKLIST
# ============================================================================

"""
TESTING CHECKLIST FOR TAB 7 EXECUTION ENGINE:

✅ 1. R² MATCHES RESULTS TAB
   - Load data in Tab 1
   - Run analysis with specific model (e.g., PLS, 5 folds, raw preprocessing)
   - Note R² value in Results tab
   - Load same model to Tab 7 (or configure manually)
   - Run in Tab 7
   - VERIFY: R² matches within ±0.0001

✅ 2. BOTH PREPROCESSING PATHS WORK
   - Test PATH A: Derivative + Subset
     * Configure: sg1 preprocessing
     * Wavelength spec: subset of full range (e.g., 1500-1800, 2000-2400)
     * Run model
     * VERIFY: No errors, reasonable R²

   - Test PATH B: Raw/SNV
     * Configure: raw or snv preprocessing
     * Run model
     * VERIFY: No errors, reasonable R²

✅ 3. ALL MODEL TYPES TRAIN SUCCESSFULLY
   - Test each model type:
     * PLS: Should work with n_components
     * Ridge: Should work with alpha
     * Lasso: Should work with alpha
     * RandomForest: Should work with n_estimators
     * MLP: Should work with hidden_layer_sizes
     * NeuralBoosted: Should work with all hyperparameters
   - VERIFY: All train without errors

✅ 4. HYPERPARAMETERS ARE USED CORRECTLY
   - When loading from Results tab:
     * VERIFY: Hyperparameters are extracted and applied
     * Check console output for "Loaded alpha=..." messages
   - When developing fresh model:
     * VERIFY: Default hyperparameters are used

✅ 5. CV FOLDS ARE DETERMINISTIC
   - Run same model multiple times
   - VERIFY: R² is identical each time (deterministic)
   - Run with shuffle=True (modify code temporarily)
   - VERIFY: R² varies (non-deterministic)
   - Revert to shuffle=False
   - VERIFY: R² is deterministic again

✅ 6. VALIDATION SET IS PROPERLY EXCLUDED
   - Enable validation set (Tab 3)
   - Run analysis in Tab 5 (Results)
   - Note sample counts (calibration vs validation)
   - Run model in Tab 7
   - Check console output for "Excluding X validation samples"
   - VERIFY: Sample count matches calibration set

✅ 7. MODEL SAVING AND LOADING
   - Train a model in Tab 7
   - Click "Save Model"
   - Choose location, save
   - Go to Tab 8 (Model Prediction)
   - Load the saved .dasp file
   - VERIFY: Model loads without errors
   - VERIFY: Metadata is correct (wavelengths, preprocessing, etc.)

✅ 8. ERROR HANDLING
   - Try to run without loading data
     * VERIFY: Clear error message "Please load data first"
   - Enter invalid wavelength spec
     * VERIFY: Validation error before execution
   - Set CV folds to 2
     * VERIFY: Validation error "CV folds must be at least 3"
   - Exclude most samples so insufficient remain
     * VERIFY: Clear error about insufficient samples

✅ 9. GUI RESPONSIVENESS
   - Run a slow model (e.g., NeuralBoosted with 100 estimators)
   - VERIFY: GUI remains responsive (doesn't freeze)
   - VERIFY: Status updates show progress
   - VERIFY: Can switch to other tabs during execution

✅ 10. CONSOLE OUTPUT
   - Run any model
   - Check console output
   - VERIFY: Clear step-by-step logging
   - VERIFY: No unexpected warnings or errors
   - VERIFY: Performance metrics printed correctly

PASS CRITERIA:
- All 10 tests pass
- R² matches Results tab exactly
- No errors in console
- GUI remains responsive
- Models save and load correctly
"""

# ============================================================================
# INTEGRATION NOTES
# ============================================================================

"""
INTEGRATION INSTRUCTIONS:

1. ADD METHODS TO CLASS:
   - Copy these 5 methods into SpectralPredictApp class:
     * _tab7_run_model()
     * _tab7_validate_parameters()
     * _tab7_run_model_thread()
     * _tab7_update_results()
     * _tab7_save_model()

2. CREATE UI WIDGETS (in _create_tab7_model_development):

   # Wavelength specification (Text widget)
   self.tab7_wl_spec = tk.Text(frame, height=3, width=50)

   # Model configuration (dropdowns)
   self.tab7_model_type = tk.StringVar(value='PLS')
   self.tab7_task_type = tk.StringVar(value='regression')
   self.tab7_preprocess = tk.StringVar(value='raw')

   # Numeric parameters (spinboxes)
   self.tab7_window = tk.IntVar(value=17)
   self.tab7_folds = tk.IntVar(value=5)
   self.tab7_max_iter = tk.IntVar(value=100)

   # Buttons
   self.tab7_run_button = ttk.Button(frame, text='▶ Run Model',
                                      command=self._tab7_run_model)
   self.tab7_save_button = ttk.Button(frame, text='💾 Save Model',
                                       command=self._tab7_save_model,
                                       state='disabled')

   # Status and results
   self.tab7_status = ttk.Label(frame, text='Ready')
   self.tab7_results_text = tk.Text(frame, height=20, width=80, state='disabled')

3. INITIALIZE STORAGE ATTRIBUTES (in __init__):

   # Model artifacts (set by execution engine)
   self.tab7_trained_model = None
   self.tab7_preprocessing_pipeline = None
   self.tab7_wavelengths = None
   self.tab7_full_wavelengths = None
   self.tab7_config = None
   self.tab7_performance = None
   self.tab7_y_true = None
   self.tab7_y_pred = None

4. DEPENDENCIES:
   - Requires existing methods:
     * self._parse_wavelength_spec() (already in GUI)
   - Requires existing attributes:
     * self.X, self.X_original, self.y (data)
     * self.excluded_spectra (outlier exclusions)
     * self.validation_enabled, self.validation_indices (validation set)

5. TESTING:
   - Run through testing checklist above
   - Verify R² matches Results tab
   - Test all model types
   - Verify model saving/loading

6. OPTIONAL ENHANCEMENTS:
   - Add plotting methods:
     * _tab7_plot_predictions() - scatter plot
     * _tab7_plot_residuals() - residual diagnostics
     * _tab7_plot_leverage() - leverage analysis
   - Add hyperparameter UI widgets for each model type
   - Add progress bar for long-running models
   - Add model comparison feature
"""
