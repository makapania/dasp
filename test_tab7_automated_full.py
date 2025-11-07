#!/usr/bin/env python3
"""
Comprehensive Automated Test Suite: Tab 7 Model Development

This script runs automated tests for Tab 7 (Model Development) to ensure:
1. Models load correctly from Results tab
2. Hyperparameters are extracted correctly
3. R² values match between Results tab and Tab 7 execution
4. All model types work correctly
5. All preprocessing methods work correctly

The script runs WITHOUT user interaction - fully automated.

Usage:
    python test_tab7_automated_full.py [--quick | --full | --exhaustive]

Options:
    --quick       Test 2 models with 1 preprocessing (fastest, ~1-2 min)
    --full        Test all 6 models with 3 preprocessing methods (~5-10 min)
    --exhaustive  Test all combinations of models × preprocessing (~30-60 min)
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from spectral_predict.io import read_asd_dir, read_reference_csv, align_xy
from spectral_predict.search import run_search
from spectral_predict.preprocess import build_preprocessing_pipeline
from spectral_predict.models import get_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

TEST_CONFIGS = {
    'quick': {
        'models': ['PLS', 'Lasso'],
        'preprocessing': ['snv_sg2'],
        'variable_counts': [50],
        'description': 'Quick test (PLS + Lasso, snv_sg2, 50 vars)'
    },
    'full': {
        'models': ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted'],
        'preprocessing': ['raw', 'snv', 'snv_sg2'],
        'variable_counts': [50],
        'description': 'Full test (6 models, 3 preprocessing methods)'
    },
    'exhaustive': {
        'models': ['PLS', 'Ridge', 'Lasso', 'RandomForest', 'MLP', 'NeuralBoosted'],
        'preprocessing': ['raw', 'snv', 'msc', 'sg1', 'sg2', 'snv_sg1', 'snv_sg2'],
        'variable_counts': [10, 20, 50],
        'description': 'Exhaustive test (6 models, 7 preprocessing, 3 subset sizes)'
    }
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_bone_collagen_data():
    """Load BoneCollagen dataset from example directory."""
    example_dir = Path(__file__).parent / "example"

    if not example_dir.exists():
        raise FileNotFoundError(f"Example directory not found: {example_dir}")

    print(f"Loading data from: {example_dir}")

    # Load ASD files
    X = read_asd_dir(str(example_dir))
    print(f"  ✓ Loaded {len(X)} spectra with {len(X.columns)} wavelengths")

    # Load reference CSV
    ref = read_reference_csv(str(example_dir / "BoneCollagen.csv"), "File Number")
    print(f"  ✓ Loaded reference data with {len(ref)} samples")

    # Align X and y
    X_aligned, y = align_xy(X, ref, "File Number", "%Collagen")
    print(f"  ✓ Aligned data: {len(X_aligned)} samples matched")
    print(f"    X shape: {X_aligned.shape}")
    print(f"    y range: {y.min():.2f} to {y.max():.2f}")

    return X_aligned, y

# ============================================================================
# TAB 7 SIMULATION (Programmatic Execution)
# ============================================================================

def simulate_tab7_execution(X_data, y_data, config):
    """
    Simulate Tab 7 execution programmatically (without GUI).

    This replicates the logic in _tab7_run_model_thread().

    Args:
        X_data: Full spectral data (DataFrame)
        y_data: Target values (Series)
        config: Model configuration dict from Results tab

    Returns:
        dict: Performance metrics (r2_mean, rmse_mean, etc.)
    """
    # Extract configuration
    model_name = config['Model']
    task_type = config.get('Task', 'regression')
    preprocess = config.get('Preprocess', 'raw')
    deriv = config.get('Deriv', None)
    window = config.get('Window', 17)
    polyorder = 3 if deriv == 2 else 2
    subset_tag = config.get('SubsetTag', 'full')
    n_vars = config.get('n_vars', len(X_data.columns))

    print(f"\n  Simulating Tab 7 execution:")
    print(f"    Model: {model_name}")
    print(f"    Preprocessing: {preprocess}")
    print(f"    Subset: {subset_tag}")
    print(f"    n_vars: {n_vars}")

    # Extract wavelengths
    all_wavelengths = X_data.columns.astype(float).values
    is_subset_model = (subset_tag not in ['full', 'N/A'])

    if is_subset_model:
        # Parse wavelengths from all_vars field
        all_vars_str = str(config['all_vars']).strip()
        wavelength_strings = [w.strip() for w in all_vars_str.split(',')]
        selected_wl = [float(w) for w in wavelength_strings if w]
        print(f"    Wavelengths: {len(selected_wl)} selected")
    else:
        selected_wl = list(all_wavelengths)
        print(f"    Wavelengths: all {len(selected_wl)}")

    # Extract hyperparameters
    hyperparams = {}
    if model_name == 'PLS':
        hyperparams['n_components'] = int(config.get('LVs', 10))
    elif model_name in ['Ridge', 'Lasso']:
        alpha_val = config.get('Alpha', None)
        if alpha_val is None:
            raise ValueError(f"CRITICAL: Alpha not found in config for {model_name}!")
        hyperparams['alpha'] = float(alpha_val)
        print(f"    ⚡ Alpha extracted: {hyperparams['alpha']}")
    elif model_name == 'RandomForest':
        hyperparams['n_estimators'] = int(config.get('n_estimators', 100))
        max_depth = config.get('max_depth', None)
        hyperparams['max_depth'] = None if pd.isna(max_depth) or max_depth == 'None' else int(max_depth)
    elif model_name == 'MLP':
        hyperparams['learning_rate_init'] = float(config.get('LR_init', 0.001))
    elif model_name == 'NeuralBoosted':
        hyperparams['n_estimators'] = int(config.get('n_estimators', 100))
        hyperparams['learning_rate'] = float(config.get('LearningRate', 0.1))

    # Prepare data
    X_work_df = X_data.copy()
    y_work = y_data.copy().reset_index(drop=True)

    # Check if derivative preprocessing + subset (PATH A logic)
    is_derivative = preprocess in ['sg1', 'sg2', 'snv_sg1', 'snv_sg2', 'msc_sg1', 'msc_sg2']
    use_full_spectrum_preprocessing = is_derivative and is_subset_model

    if use_full_spectrum_preprocessing:
        print(f"    PATH A: Full-spectrum preprocessing + subset")
        # Preprocess full spectrum first
        from spectral_predict.preprocess import build_preprocessing_pipeline

        # Map preprocessing names
        if preprocess in ['sg1', 'snv_sg1', 'msc_sg1']:
            preprocess_name = 'deriv' if preprocess == 'sg1' else 'snv_deriv' if preprocess == 'snv_sg1' else 'msc_deriv'
            deriv_val = 1
        elif preprocess in ['sg2', 'snv_sg2', 'msc_sg2']:
            preprocess_name = 'deriv' if preprocess == 'sg2' else 'snv_deriv' if preprocess == 'snv_sg2' else 'msc_deriv'
            deriv_val = 2
        else:
            preprocess_name = preprocess
            deriv_val = deriv

        prep_steps = build_preprocessing_pipeline(preprocess_name, deriv_val, window, polyorder)
        prep_pipeline = Pipeline(prep_steps)

        # Preprocess full spectrum
        X_full_preprocessed = prep_pipeline.fit_transform(X_work_df.values)

        # Subset to selected wavelengths
        wavelength_indices = [np.argmin(np.abs(all_wavelengths - wl)) for wl in selected_wl]
        X_work = X_full_preprocessed[:, wavelength_indices]

        # Pipeline with only model
        # Use higher max_iter for Lasso to ensure convergence parity with GUI
        effective_max_iter = 2000 if model_name == 'Lasso' else 100
        n_components = hyperparams.get('n_components', 10)
        model = get_model(model_name, task_type, n_components, min(20, len(X_work)//5), effective_max_iter)

        # Apply hyperparameters
        if model_name in ['Ridge', 'Lasso']:
            model.set_params(alpha=hyperparams['alpha'])

        pipe = Pipeline([('model', model)])
    else:
        print(f"    PATH B: Standard (subset then preprocess)")
        # Subset first, then preprocess
        selected_cols = [col for col in X_work_df.columns if float(col) in selected_wl]
        X_subset_df = X_work_df[selected_cols]
        X_work = X_subset_df.values

        # Build full pipeline
        from spectral_predict.preprocess import build_preprocessing_pipeline

        # Map preprocessing names
        if preprocess == 'raw':
            preprocess_name = 'raw'
            deriv_val = None
        elif preprocess in ['snv', 'msc']:
            preprocess_name = preprocess
            deriv_val = None
        else:
            preprocess_name = preprocess
            deriv_val = deriv

        pipe_steps = build_preprocessing_pipeline(preprocess_name, deriv_val, window, polyorder) if preprocess_name != 'raw' else []
        effective_max_iter = 2000 if model_name == 'Lasso' else 100
        n_components = hyperparams.get('n_components', 10)
        model = get_model(model_name, task_type, n_components, min(20, len(X_work)//5), effective_max_iter)
        pipe_steps.append(('model', model))
        pipe = Pipeline(pipe_steps)

        # Apply hyperparameters
        if model_name in ['Ridge', 'Lasso']:
            pipe.set_params(model__alpha=hyperparams['alpha'])

    # Run cross-validation (same as Tab 7)
    n_folds = 5
    # Match Tab 7 CV settings: shuffle=True, deterministic seed
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_work)):
        X_train, X_test = X_work[train_idx], X_work[test_idx]
        y_train, y_test = y_work.iloc[train_idx], y_work.iloc[test_idx]

        # Train and predict
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        bias = np.mean(y_pred - y_test)

        fold_metrics.append({'r2': r2, 'rmse': rmse, 'mae': mae, 'bias': bias})

    # Aggregate metrics
    results = {
        'r2_mean': np.mean([m['r2'] for m in fold_metrics]),
        'r2_std': np.std([m['r2'] for m in fold_metrics]),
        'rmse_mean': np.mean([m['rmse'] for m in fold_metrics]),
        'rmse_std': np.std([m['rmse'] for m in fold_metrics]),
        'mae_mean': np.mean([m['mae'] for m in fold_metrics]),
        'mae_std': np.std([m['mae'] for m in fold_metrics]),
        'bias_mean': np.mean([m['bias'] for m in fold_metrics]),
        'bias_std': np.std([m['bias'] for m in fold_metrics])
    }

    print(f"    Tab 7 R²: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")

    return results

# ============================================================================
# TEST EXECUTION
# ============================================================================

def run_test_suite(test_mode='quick'):
    """Run the automated test suite."""
    config = TEST_CONFIGS[test_mode]

    print("="*80)
    print(f"TAB 7 AUTOMATED TEST SUITE: {test_mode.upper()} MODE")
    print("="*80)
    print(f"Description: {config['description']}")
    print(f"Models: {', '.join(config['models'])}")
    print(f"Preprocessing: {', '.join(config['preprocessing'])}")
    print(f"Variable counts: {config['variable_counts']}")
    print("="*80)
    print()

    # Load data
    print("[STEP 1] Loading BoneCollagen dataset...")
    X, y = load_bone_collagen_data()
    print()

    # Run tests
    print("[STEP 2] Running automated tests...")
    print()

    results_list = []
    test_count = 0
    pass_count = 0
    fail_count = 0

    for model in config['models']:
        for preproc in config['preprocessing']:
            for var_count in config['variable_counts']:
                test_count += 1
                test_name = f"{model}_{preproc}_var{var_count}"

                print(f"\n{'='*80}")
                print(f"TEST {test_count}: {test_name}")
                print(f"{'='*80}")

                try:
                    # Build preprocessing method dict
                    preproc_methods = {preproc: True}

                    # Run analysis (Results tab simulation)
                    print(f"  Running analysis (Results tab simulation)...")
                    results_df = run_search(
                        X, y,
                        task_type='regression',
                        folds=5,
                        models_to_test=[model],
                        preprocessing_methods=preproc_methods,
                        window_sizes=[17],
                        enable_variable_subsets=(var_count < len(X.columns)),
                        variable_counts=[var_count] if var_count < len(X.columns) else [],
                        variable_selection_methods=['importance'] if var_count < len(X.columns) else [],
                        lambda_penalty=0.15,
                        max_n_components=20,
                        max_iter=100
                    )

                    if len(results_df) == 0:
                        print(f"  ⚠️  No results returned - SKIP")
                        continue

                    # Get top model
                    top_model = results_df.iloc[0]
                    results_r2 = top_model['R2']
                    print(f"  Results tab R²: {results_r2:.4f}")

                    # Simulate Tab 7 execution
                    tab7_metrics = simulate_tab7_execution(X, y, top_model.to_dict())
                    tab7_r2 = tab7_metrics['r2_mean']

                    # Compare R² values
                    r2_diff = abs(results_r2 - tab7_r2)
                    tolerance = 0.01

                    print(f"\n  {'='*40}")
                    print(f"  COMPARISON:")
                    print(f"    Results tab R²: {results_r2:.4f}")
                    print(f"    Tab 7 R²:       {tab7_r2:.4f}")
                    print(f"    Difference:     {r2_diff:.4f} ({r2_diff*100:.2f} percentage points)")
                    print(f"    Tolerance:      {tolerance}")

                    if r2_diff < tolerance:
                        print(f"    Status: ✅ PASS")
                        pass_count += 1
                        status = 'PASS'
                    else:
                        print(f"    Status: ❌ FAIL - R² mismatch exceeds tolerance!")
                        fail_count += 1
                        status = 'FAIL'

                    print(f"  {'='*40}")

                    results_list.append({
                        'test': test_name,
                        'model': model,
                        'preprocessing': preproc,
                        'var_count': var_count,
                        'results_r2': results_r2,
                        'tab7_r2': tab7_r2,
                        'r2_diff': r2_diff,
                        'status': status
                    })

                except Exception as e:
                    print(f"  ❌ ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    fail_count += 1
                    results_list.append({
                        'test': test_name,
                        'model': model,
                        'preprocessing': preproc,
                        'var_count': var_count,
                        'results_r2': None,
                        'tab7_r2': None,
                        'r2_diff': None,
                        'status': 'ERROR'
                    })

    # Summary report
    print(f"\n\n{'='*80}")
    print(f"TEST SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {test_count}")
    print(f"Passed:      {pass_count} ✅")
    print(f"Failed:      {fail_count} ❌")
    print(f"Success rate: {pass_count/test_count*100:.1f}%")
    print(f"{'='*80}")

    # Detailed results table
    if results_list:
        results_summary = pd.DataFrame(results_list)
        print(f"\nDetailed Results:")
        print(results_summary.to_string(index=False))

    # Return exit code (0 = all pass, 1 = any fail)
    return 0 if fail_count == 0 else 1

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Tab 7 Automated Test Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick test (fastest)')
    parser.add_argument('--full', action='store_true', help='Run full test (all models, 3 preprocessing)')
    parser.add_argument('--exhaustive', action='store_true', help='Run exhaustive test (all combinations)')

    args = parser.parse_args()

    # Determine test mode
    if args.exhaustive:
        test_mode = 'exhaustive'
    elif args.full:
        test_mode = 'full'
    else:
        test_mode = 'quick'

    # Run tests
    exit_code = run_test_suite(test_mode)
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
