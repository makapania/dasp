#!/usr/bin/env python3
"""
Test script to programmatically replicate Tab 7 model recreation.
This script loads a model from Results tab and recreates it exactly as Tab 7 does,
comparing results to find the bug causing R² = -0.3668.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from spectral_predict.preprocess import build_preprocessing_pipeline
from spectral_predict.models import get_model

def load_results_from_analysis():
    """Load the most recent analysis results."""
    # Look for results files
    results_files = list(Path('.').glob('*_results.csv'))
    if not results_files:
        print("ERROR: No results files found. Please run an analysis first.")
        return None

    # Get most recent
    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_results}")

    df = pd.read_csv(latest_results)
    print(f"  Found {len(df)} model configurations")
    return df

def load_dataset(data_folder):
    """Load the dataset used in the analysis."""
    data_folder = Path(data_folder)

    # Try to find X and y files
    X_file = data_folder / 'X.csv'
    y_file = data_folder / 'y.csv'

    if not X_file.exists() or not y_file.exists():
        print(f"ERROR: Could not find X.csv and y.csv in {data_folder}")
        return None, None, None

    X_df = pd.read_csv(X_file, index_col=0)
    y_df = pd.read_csv(y_file, index_col=0)

    # Get wavelengths (column names as floats)
    wavelengths = X_df.columns.astype(float).values

    print(f"  Loaded dataset:")
    print(f"    X shape: {X_df.shape}")
    print(f"    y shape: {y_df.shape}")
    print(f"    Wavelengths: {wavelengths[0]:.1f} to {wavelengths[-1]:.1f} nm ({len(wavelengths)} total)")

    return X_df, y_df.iloc[:, 0], wavelengths

def parse_preprocessing_config(row):
    """Parse preprocessing configuration from results row."""
    preprocess = row.get('Preprocess', 'raw')
    deriv = row.get('Deriv', None)
    window = row.get('Window', 17)

    # Convert search.py internal names to GUI names
    if preprocess == 'snv_deriv':
        if deriv == 1:
            gui_preprocess = 'snv_sg1'
        elif deriv == 2:
            gui_preprocess = 'snv_sg2'
        else:
            gui_preprocess = 'snv'
    elif preprocess == 'deriv':
        if deriv == 1:
            gui_preprocess = 'sg1'
        elif deriv == 2:
            gui_preprocess = 'sg2'
        else:
            gui_preprocess = 'raw'
    else:
        gui_preprocess = preprocess

    return gui_preprocess, deriv, window

def recreate_model_RESULTS_TAB_WAY(X_df, y_series, row):
    """
    Recreate model using Results tab approach (search.py).
    This is the REFERENCE - what the Results tab did.
    """
    print("\n" + "="*80)
    print("RECREATING MODEL: RESULTS TAB WAY (search.py)")
    print("="*80)

    model_name = row['Model']
    task_type = row.get('Task', 'regression')

    # Parse preprocessing
    gui_preprocess, deriv, window = parse_preprocessing_config(row)
    print(f"  Model: {model_name}")
    print(f"  Preprocessing: {gui_preprocess} (deriv={deriv}, window={window})")

    # Parse wavelength subset
    all_vars_str = row.get('top_vars', row.get('all_vars', None))
    if all_vars_str and all_vars_str != 'N/A' and str(all_vars_str).strip():
        selected_wavelengths = [float(w.strip()) for w in str(all_vars_str).split(',')]
        is_subset = True
        print(f"  Wavelength subset: {len(selected_wavelengths)} wavelengths")
        print(f"    Range: {selected_wavelengths[0]:.1f} to {selected_wavelengths[-1]:.1f} nm")
    else:
        selected_wavelengths = X_df.columns.astype(float).values
        is_subset = False
        print(f"  Using full spectrum: {len(selected_wavelengths)} wavelengths")

    # Get hyperparameters
    hyperparams = {}
    if model_name in ['Ridge', 'Lasso']:
        alpha = row.get('Alpha', None)
        if alpha is not None:
            hyperparams['alpha'] = float(alpha)
            print(f"  Alpha: {hyperparams['alpha']}")
        else:
            print(f"  WARNING: Alpha not found in config!")
    elif model_name == 'PLS':
        n_components = row.get('LVs', row.get('Components', None))
        if n_components is not None and n_components != 'N/A':
            hyperparams['n_components'] = int(n_components)
            print(f"  Components: {hyperparams['n_components']}")

    # Build preprocessing pipeline
    is_derivative = deriv is not None and deriv > 0

    if is_derivative and is_subset:
        print(f"\n  PATH: Derivative + Subset (preprocess full spectrum, then subset)")

        # Preprocess FULL spectrum
        prep_steps = build_preprocessing_pipeline(gui_preprocess, deriv, window, polyorder=3)
        prep_pipeline = Pipeline(prep_steps)

        X_full = X_df.values
        print(f"    Step 1: Preprocess full spectrum {X_full.shape}")
        X_full_preprocessed = prep_pipeline.fit_transform(X_full)
        print(f"    Result: {X_full_preprocessed.shape}")

        # Find wavelength indices
        all_wavelengths = X_df.columns.astype(float).values
        wavelength_indices = []
        for wl in selected_wavelengths:
            idx = np.where(np.abs(all_wavelengths - wl) < 0.01)[0]
            if len(idx) > 0:
                wavelength_indices.append(idx[0])

        print(f"    Step 2: Find wavelength indices")
        print(f"    Selected {len(wavelength_indices)} indices: {wavelength_indices[:5]}...{wavelength_indices[-5:]}")

        # Subset preprocessed data
        X_work = X_full_preprocessed[:, wavelength_indices]
        print(f"    Step 3: Subset preprocessed data to {X_work.shape}")

        # Pipeline with model only (no preprocessing)
        model = get_model(model_name, task_type, **hyperparams)
        pipe = Pipeline([('model', model)])

    else:
        print(f"\n  PATH: Standard (subset first, then preprocess)")
        # This path not relevant for the bug, but included for completeness
        X_work = X_df.values
        prep_steps = build_preprocessing_pipeline(gui_preprocess, deriv, window, polyorder=3)
        model = get_model(model_name, task_type, **hyperparams)
        pipe = Pipeline(prep_steps + [('model', model)])

    # Cross-validation with SHUFFLE=TRUE (Results tab way)
    n_folds = int(row.get('Folds', 5))
    print(f"\n  Cross-Validation: {n_folds} folds")
    print(f"    Shuffle: TRUE (random_state=42)")

    if task_type == "regression":
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Run CV
    y_array = y_series.values
    fold_r2 = []
    fold_rmse = []
    fold_mae = []

    print(f"\n  Running CV...")
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_work, y_array)):
        from sklearn.base import clone
        pipe_fold = clone(pipe)

        X_train, X_test = X_work[train_idx], X_work[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        pipe_fold.fit(X_train, y_train)
        y_pred = pipe_fold.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        fold_r2.append(r2)
        fold_rmse.append(rmse)
        fold_mae.append(mae)

        print(f"    Fold {fold_idx+1}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Summary
    mean_r2 = np.mean(fold_r2)
    std_r2 = np.std(fold_r2)
    mean_rmse = np.mean(fold_rmse)
    std_rmse = np.std(fold_rmse)
    mean_mae = np.mean(fold_mae)
    std_mae = np.std(fold_mae)

    print(f"\n  RESULTS (shuffle=True, random_state=42):")
    print(f"    R²:   {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"    RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"    MAE:  {mean_mae:.4f} ± {std_mae:.4f}")

    return {
        'r2_mean': mean_r2,
        'r2_std': std_r2,
        'rmse_mean': mean_rmse,
        'rmse_std': std_rmse,
        'mae_mean': mean_mae,
        'mae_std': std_mae,
        'fold_r2': fold_r2,
        'fold_rmse': fold_rmse,
        'fold_mae': fold_mae
    }

def recreate_model_TAB7_WAY(X_df, y_series, row):
    """
    Recreate model using Tab 7 approach (spectral_predict_gui_optimized.py).
    This is what Tab 7 ACTUALLY does - we're testing if it matches Results tab.
    """
    print("\n" + "="*80)
    print("RECREATING MODEL: TAB 7 WAY (spectral_predict_gui_optimized.py)")
    print("="*80)

    model_name = row['Model']
    task_type = row.get('Task', 'regression')

    # Parse preprocessing
    gui_preprocess, deriv, window = parse_preprocessing_config(row)
    print(f"  Model: {model_name}")
    print(f"  Preprocessing: {gui_preprocess} (deriv={deriv}, window={window})")

    # Parse wavelength subset
    all_vars_str = row.get('top_vars', row.get('all_vars', None))
    if all_vars_str and all_vars_str != 'N/A' and str(all_vars_str).strip():
        selected_wavelengths = [float(w.strip()) for w in str(all_vars_str).split(',')]
        is_subset = True
        print(f"  Wavelength subset: {len(selected_wavelengths)} wavelengths")
        print(f"    Range: {selected_wavelengths[0]:.1f} to {selected_wavelengths[-1]:.1f} nm")
    else:
        selected_wavelengths = X_df.columns.astype(float).values
        is_subset = False
        print(f"  Using full spectrum: {len(selected_wavelengths)} wavelengths")

    # Get hyperparameters
    hyperparams = {}
    if model_name in ['Ridge', 'Lasso']:
        alpha = row.get('Alpha', None)
        if alpha is not None:
            hyperparams['alpha'] = float(alpha)
            print(f"  Alpha: {hyperparams['alpha']}")
        else:
            print(f"  WARNING: Alpha not found in config! Using default 1.0")
            hyperparams['alpha'] = 1.0  # Tab 7 default
    elif model_name == 'PLS':
        n_components = row.get('LVs', row.get('Components', None))
        if n_components is not None and n_components != 'N/A':
            hyperparams['n_components'] = int(n_components)
            print(f"  Components: {hyperparams['n_components']}")

    # Build preprocessing pipeline (EXACTLY as Tab 7 does)
    is_derivative = deriv is not None and deriv > 0

    if is_derivative and is_subset:
        print(f"\n  PATH: Derivative + Subset (preprocess full spectrum, then subset)")

        # Preprocess FULL spectrum
        prep_steps = build_preprocessing_pipeline(gui_preprocess, deriv, window, polyorder=3)
        prep_pipeline = Pipeline(prep_steps)

        X_full = X_df.values
        print(f"    Step 1: Preprocess full spectrum {X_full.shape}")
        X_full_preprocessed = prep_pipeline.fit_transform(X_full)
        print(f"    Result: {X_full_preprocessed.shape}")

        # Find wavelength indices
        all_wavelengths = X_df.columns.astype(float).values
        wavelength_indices = []
        for wl in selected_wavelengths:
            idx = np.where(np.abs(all_wavelengths - wl) < 0.01)[0]
            if len(idx) > 0:
                wavelength_indices.append(idx[0])

        print(f"    Step 2: Find wavelength indices")
        print(f"    Selected {len(wavelength_indices)} indices: {wavelength_indices[:5]}...{wavelength_indices[-5:]}")

        # Subset preprocessed data
        X_work = X_full_preprocessed[:, wavelength_indices]
        print(f"    Step 3: Subset preprocessed data to {X_work.shape}")

        # Pipeline with model only (no preprocessing)
        model = get_model(model_name, task_type, **hyperparams)
        pipe = Pipeline([('model', model)])

    else:
        print(f"\n  PATH: Standard (subset first, then preprocess)")
        X_work = X_df.values
        prep_steps = build_preprocessing_pipeline(gui_preprocess, deriv, window, polyorder=3)
        model = get_model(model_name, task_type, **hyperparams)
        pipe = Pipeline(prep_steps + [('model', model)])

    # Cross-validation with SHUFFLE=FALSE (Tab 7 way - THIS IS THE BUG!)
    n_folds = int(row.get('Folds', 5))
    print(f"\n  Cross-Validation: {n_folds} folds")
    print(f"    Shuffle: FALSE (deterministic splits)")

    if task_type == "regression":
        cv = KFold(n_splits=n_folds, shuffle=False)  # Tab 7 uses shuffle=False
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=False)

    # Run CV
    y_array = y_series.values
    fold_r2 = []
    fold_rmse = []
    fold_mae = []

    print(f"\n  Running CV...")
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_work, y_array)):
        from sklearn.base import clone
        pipe_fold = clone(pipe)

        X_train, X_test = X_work[train_idx], X_work[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        pipe_fold.fit(X_train, y_train)
        y_pred = pipe_fold.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        fold_r2.append(r2)
        fold_rmse.append(rmse)
        fold_mae.append(mae)

        print(f"    Fold {fold_idx+1}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Summary
    mean_r2 = np.mean(fold_r2)
    std_r2 = np.std(fold_r2)
    mean_rmse = np.mean(fold_rmse)
    std_rmse = np.std(fold_rmse)
    mean_mae = np.mean(fold_mae)
    std_mae = np.std(fold_mae)

    print(f"\n  RESULTS (shuffle=False):")
    print(f"    R²:   {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"    RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"    MAE:  {mean_mae:.4f} ± {std_mae:.4f}")

    return {
        'r2_mean': mean_r2,
        'r2_std': std_r2,
        'rmse_mean': mean_rmse,
        'rmse_std': std_rmse,
        'mae_mean': mean_mae,
        'mae_std': std_mae,
        'fold_r2': fold_r2,
        'fold_rmse': fold_rmse,
        'fold_mae': fold_mae
    }

def compare_results(results_tab_metrics, tab7_metrics, expected_r2):
    """Compare Results tab vs Tab 7 metrics."""
    print("\n" + "="*80)
    print("COMPARISON: Results Tab vs Tab 7")
    print("="*80)

    print(f"\nExpected R² (from Results tab display): {expected_r2:.4f}")
    print(f"\nResults Tab Recreation (shuffle=True, random_state=42):")
    print(f"  R²:   {results_tab_metrics['r2_mean']:.4f} ± {results_tab_metrics['r2_std']:.4f}")
    print(f"  RMSE: {results_tab_metrics['rmse_mean']:.4f} ± {results_tab_metrics['rmse_std']:.4f}")
    print(f"  MAE:  {results_tab_metrics['mae_mean']:.4f} ± {results_tab_metrics['mae_std']:.4f}")

    print(f"\nTab 7 Recreation (shuffle=False):")
    print(f"  R²:   {tab7_metrics['r2_mean']:.4f} ± {tab7_metrics['r2_std']:.4f}")
    print(f"  RMSE: {tab7_metrics['rmse_mean']:.4f} ± {tab7_metrics['rmse_std']:.4f}")
    print(f"  MAE:  {tab7_metrics['mae_mean']:.4f} ± {tab7_metrics['mae_std']:.4f}")

    # Deltas
    r2_delta = tab7_metrics['r2_mean'] - results_tab_metrics['r2_mean']
    rmse_delta = tab7_metrics['rmse_mean'] - results_tab_metrics['rmse_mean']
    mae_delta = tab7_metrics['mae_mean'] - results_tab_metrics['mae_mean']

    print(f"\nDelta (Tab 7 - Results Tab):")
    print(f"  ΔR²:   {r2_delta:+.4f}")
    print(f"  ΔRMSE: {rmse_delta:+.4f}")
    print(f"  ΔMAE:  {mae_delta:+.4f}")

    # Verdict
    print(f"\n" + "="*80)
    if abs(r2_delta) < 0.01:
        print("✅ VERDICT: Results MATCH - No significant difference")
        print("   The shuffle parameter is NOT causing the bug.")
    else:
        print("❌ VERDICT: Results MISMATCH - Significant difference detected!")
        print(f"   R² differs by {abs(r2_delta):.4f} ({abs(r2_delta/expected_r2)*100:.1f}% relative error)")
        print("   The shuffle parameter IS likely the root cause.")
    print("="*80)

    return r2_delta

def main():
    """Main test function."""
    print("="*80)
    print("MODEL RECREATION TEST")
    print("Testing if Tab 7 accurately recreates models from Results tab")
    print("="*80)

    # Step 1: Load results
    print("\n[STEP 1] Loading results...")
    results_df = load_results_from_analysis()
    if results_df is None:
        return

    # Step 2: Select a model to test (use top-ranked model)
    print("\n[STEP 2] Selecting model to test...")
    # Filter for derivative+subset models (most likely to have bugs)
    # Check if top_vars or all_vars column exists
    vars_col = 'top_vars' if 'top_vars' in results_df.columns else 'all_vars' if 'all_vars' in results_df.columns else None

    if vars_col:
        subset_models = results_df[results_df[vars_col].notna() & (results_df[vars_col] != 'N/A')]
        if len(subset_models) > 0:
            test_row = subset_models.iloc[0]
        else:
            print("  No subset models found. Using top model from full results.")
            test_row = results_df.iloc[0]
    else:
        print("  No wavelength subset column found. Using top model from full results.")
        test_row = results_df.iloc[0]

    print(f"  Selected Model:")
    print(f"    Rank: {test_row.get('Rank', 1)}")
    print(f"    Model: {test_row['Model']}")
    print(f"    R²: {test_row['R2']:.4f}")
    print(f"    Preprocessing: {test_row.get('Preprocess', 'N/A')}")
    print(f"    Subset: {test_row.get('Subset', 'N/A')}")

    expected_r2 = test_row['R2']

    # Step 3: Load dataset
    print("\n[STEP 3] Loading dataset...")
    # Try to infer data folder from results file name
    data_folder = Path('data/BoneCollagen')  # Default
    print(f"  Using data folder: {data_folder}")

    X_df, y_series, wavelengths = load_dataset(data_folder)
    if X_df is None:
        return

    # Step 4: Recreate model using Results tab approach
    print("\n[STEP 4] Recreating model: Results Tab approach...")
    results_tab_metrics = recreate_model_RESULTS_TAB_WAY(X_df, y_series, test_row)

    # Step 5: Recreate model using Tab 7 approach
    print("\n[STEP 5] Recreating model: Tab 7 approach...")
    tab7_metrics = recreate_model_TAB7_WAY(X_df, y_series, test_row)

    # Step 6: Compare results
    print("\n[STEP 6] Comparing results...")
    r2_delta = compare_results(results_tab_metrics, tab7_metrics, expected_r2)

    # Step 7: Conclusion
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"\nIf R² delta > 0.01: The shuffle parameter is causing the bug")
    print(f"If R² delta < 0.01: The bug is elsewhere (hyperparameters, wavelengths, etc.)")
    print(f"\nActual R² delta: {abs(r2_delta):.4f}")

    if abs(r2_delta) > 0.01:
        print("\n🎯 FIX: Change Tab 7 line 2406 from:")
        print("    cv = KFold(n_splits=n_folds, shuffle=False)")
        print("  To:")
        print("    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)")
    else:
        print("\n⚠️  The shuffle parameter is not the issue.")
        print("   Need to investigate hyperparameters or wavelength selection.")

if __name__ == '__main__':
    main()
