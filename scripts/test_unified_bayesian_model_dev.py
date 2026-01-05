"""Test Unified Bayesian R² Mismatch in Model Development.

This script:
1. Loads example data (BoneCollagen.csv)
2. Runs Unified Bayesian optimization with 10-20 trials for PLS
3. Takes a result where window ≠ 11
4. Simulates exactly what Model Development does
5. Compares R² from result vs reproduced R²
6. Shows where the window value gets lost

The test should reveal that Model Development uses window=11 (hardcoded)
instead of the window from the Unified Bayesian result.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, KFold

# Import DASP components
from spectral_predict.unified_bayesian import run_unified_bayesian
from spectral_predict.models import build_model
from spectral_predict.preprocess import build_preprocessing_pipeline


def load_example_data():
    """Load ASD files from example directory and collagen values."""
    from spectral_predict.io import read_asd_dir

    # Load ASD files
    asd_dir = repo_root / "example"
    df_spectra, metadata = read_asd_dir(str(asd_dir), reader_mode="auto")

    # Load collagen values from CSV
    csv_path = repo_root / "example" / "BoneCollagen.csv"
    df_meta = pd.read_csv(csv_path)

    # Create mapping from file number to collagen %
    file_to_collagen = {}
    for _, row in df_meta.iterrows():
        file_num = row['File Number'].replace('Spectrum ', '').strip()
        collagen = row['%Collagen']
        file_to_collagen[file_num] = collagen

    # Match spectra to collagen values
    y_list = []
    valid_indices = []
    for i, sample_name in enumerate(df_spectra.index):
        # Extract file number from sample name (e.g., 'Spectrum00001' -> '00001')
        file_num = sample_name.replace('Spectrum', '').strip()
        if file_num in file_to_collagen:
            y_list.append(file_to_collagen[file_num])
            valid_indices.append(i)

    # Filter to only samples with collagen values
    df_filtered = df_spectra.iloc[valid_indices]
    y = np.array(y_list)
    wavelengths = np.array([float(c) for c in df_filtered.columns])
    X = df_filtered.values

    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} wavelengths")
    print(f"Target: Collagen %")
    print(f"Wavelength range: {wavelengths[0]:.1f} to {wavelengths[-1]:.1f} nm")

    return X, y, wavelengths, "Collagen %"


def simulate_model_development(config, X, y, wavelengths):
    """Simulate Model Development tab's exact code path.

    This replicates the logic from _run_refined_model_thread() in the GUI.
    Specifically lines 21000-21150 that handle Unified Bayesian results.
    """
    print("\n" + "="*80)
    print("SIMULATING MODEL DEVELOPMENT TAB")
    print("="*80)

    # Extract config values
    model_name = config.get('Model', 'PLS')
    preprocess_str = config.get('Preprocess', 'raw')
    params_str = config.get('Params', '{}')

    print(f"\nLoaded config:")
    print(f"  Model: {model_name}")
    print(f"  Preprocess: {preprocess_str}")
    print(f"  Deriv: {config.get('Deriv', 'N/A')}")
    print(f"  Window: {config.get('Window', 'N/A')}")
    print(f"  Params: {params_str}")

    # === THIS IS THE CRITICAL SECTION FROM GUI LINE 21000-21150 ===

    # Internal function from GUI (line 20994-21054)
    def _parse_coupled_preprocessing(preprocess_str, params_str='', window_config=None):
        """Parse coupled preprocessing string.

        Returns dict with: baseline, snv, deriv, snv_first, window, polyorder
        """
        # Use window from config if provided (Unified Bayesian stores it in Window column)
        # Otherwise default to 11 (for old results or coupled optimization)
        default_window = window_config if window_config is not None else 11

        result = {
            'baseline': None,
            'snv': False,
            'deriv': None,
            'snv_first': True,
            'window': default_window,
            'polyorder': 2  # Default
        }

        if not preprocess_str or preprocess_str == 'raw':
            return result

        # Split by '+' for baseline+preprocessing
        parts = preprocess_str.split('+')

        # Check for baseline methods
        baseline_methods = ['airpls', 'als', 'polynomial', 'modpoly', 'imodpoly']
        for part in parts[:]:
            if part.lower() in baseline_methods:
                result['baseline'] = part.lower()
                parts.remove(part)
                break

        # Parse remaining preprocessing (snv, deriv, combinations)
        if parts:
            preproc = parts[0].lower()

            # Check for SNV
            if 'snv' in preproc:
                result['snv'] = True

            # Check for derivative and order
            import re
            deriv_match = re.search(r'deriv(\d)', preproc)
            if deriv_match:
                result['deriv'] = int(deriv_match.group(1))
                result['polyorder'] = result['deriv'] + 1

            # Check SNV position (snv_deriv vs deriv_snv)
            if 'deriv' in preproc and 'snv' in preproc:
                result['snv_first'] = preproc.index('snv') < preproc.index('deriv')

        # Try to extract window from Params string
        if params_str:
            try:
                import ast
                params = ast.literal_eval(params_str)
                if 'savgol_window' in params:
                    result['window'] = params['savgol_window']
                    print(f"    DEBUG: Found savgol_window={result['window']} in Params")
            except:
                pass

        return result

    # Detect if this is a coupled optimization result (line 21073-21092)
    is_coupled_result = False
    if config is not None:
        # Primary detection: explicit flag
        is_coupled_result = config.get('is_coupled', False)

        # Fallback detection: check for coupled Preprocess format patterns
        # Unified Bayesian uses formats like 'deriv1', 'snv_deriv2', 'deriv1_snv'
        if not is_coupled_result:
            import re
            has_deriv_number = bool(re.search(r'deriv\d', preprocess_str))
            has_baseline_plus = '+' in preprocess_str
            is_coupled_result = has_deriv_number or has_baseline_plus

    print(f"\n  Is coupled result? {is_coupled_result}")
    print(f"    (detected via deriv\\d pattern: {bool(re.search(r'deriv\\d', preprocess_str))})")

    # Parse preprocessing (line 21099-21138)
    if is_coupled_result:
        # Parse coupled preprocessing string
        # CRITICAL: Pass Window from config for Unified Bayesian results
        window_from_config = config.get('Window', None)
        coupled_config = _parse_coupled_preprocessing(preprocess_str, params_str, window_config=window_from_config)

        deriv = coupled_config['deriv']
        window = coupled_config['window']
        polyorder = coupled_config['polyorder']
        use_snv = coupled_config['snv']
        baseline_method = coupled_config['baseline']

        print(f"\n{'='*70}")
        print(f"COUPLED OPTIMIZATION RESULT DETECTED")
        print(f"{'='*70}")
        print(f"  Preprocessing string: {preprocess_str}")
        print(f"  Parsed configuration:")
        print(f"    Derivative order: {deriv}")
        print(f"    Savitzky-Golay window: {window}")  # THIS IS THE BUG - uses hardcoded 11!
        print(f"    Polynomial order: {polyorder}")
        print(f"    SNV enabled: {use_snv}")
        print(f"    Baseline method: {baseline_method}")
        print(f"{'='*70}\n")

        # Map to preprocessing name
        if use_snv and deriv:
            if coupled_config.get('snv_first', True):
                preprocess_name = 'snv_deriv'
            else:
                preprocess_name = 'deriv_snv'
        elif use_snv:
            preprocess_name = 'snv'
        elif deriv:
            preprocess_name = 'deriv'
        else:
            preprocess_name = 'raw'
    else:
        # Not a coupled result - use config Deriv
        deriv = config.get('Deriv', None)
        window = config.get('Window', 11)
        polyorder = deriv + 1 if deriv else 2
        use_snv = 'snv' in preprocess_str.lower()
        preprocess_name = 'raw'

    # Build preprocessing pipeline (like GUI does)
    print(f"\nBuilding preprocessing pipeline:")
    print(f"  preprocess_name: {preprocess_name}")
    print(f"  deriv: {deriv}")
    print(f"  window: {window}")
    print(f"  polyorder: {polyorder}")

    # Get wavelengths to use
    if 'all_vars' in config and config['all_vars'] != 'N/A':
        # Subset model - use selected wavelengths
        selected_wl = np.array([float(w) for w in config['all_vars'].split(',')])
        # Find indices in original wavelengths
        wl_indices = [np.argmin(np.abs(wavelengths - w)) for w in selected_wl]
        X_subset = X[:, wl_indices]
        print(f"  Using {len(selected_wl)} selected wavelengths")
    else:
        # Full spectrum
        X_subset = X
        selected_wl = wavelengths
        print(f"  Using all {len(wavelengths)} wavelengths")

    # Apply preprocessing
    from sklearn.pipeline import Pipeline
    prep_steps = build_preprocessing_pipeline(
        preprocess_name,
        deriv=deriv,
        window=window,
        polyorder=polyorder
    )
    preprocessor = Pipeline(prep_steps)

    X_processed = preprocessor.fit_transform(X_subset)
    print(f"  Processed shape: {X_processed.shape}")

    # Build model with params from config
    import ast
    model_params = ast.literal_eval(params_str) if params_str else {}
    model = build_model(model_name, model_params, task_type='regression')

    # Run CV (same as GUI)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_processed, y,
        cv=cv,
        scoring={'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'},
        n_jobs=1,
        error_score='raise'
    )

    rmse_model_dev = -cv_results['test_rmse'].mean()
    r2_model_dev = cv_results['test_r2'].mean()

    print(f"\nModel Development Results:")
    print(f"  RMSE: {rmse_model_dev:.6f}")
    print(f"  R²: {r2_model_dev:.6f}")

    return rmse_model_dev, r2_model_dev, window


def main():
    """Run the test."""
    print("="*80)
    print("Testing Unified Bayesian R² Mismatch in Model Development")
    print("="*80)

    # 1. Load data
    X, y, wavelengths, target_col = load_example_data()

    # 2. Run Unified Bayesian (small number of trials for testing)
    print("\n" + "="*80)
    print("Running Unified Bayesian Optimization")
    print("="*80)
    print("Testing PLS model with 15 trials")

    results_df, study = run_unified_bayesian(
        X, y, wavelengths,
        model_name='PLS',
        n_trials=15,
        cv_folds=5,
        n_top_regions=10,
        random_state=42,
        verbose=True
    )

    # 3. Find a result where window ≠ 11
    results_with_deriv = results_df[
        (results_df['Window'] > 0) & (results_df['Window'] != 11)
    ]

    if len(results_with_deriv) == 0:
        print("\nWARNING: No results with window ≠ 11 found.")
        print("Using any result with derivatives...")
        results_with_deriv = results_df[results_df['Window'] > 0]

    if len(results_with_deriv) == 0:
        print("\nERROR: No results with derivatives found!")
        print("Try running with more trials or check the data.")
        return

    # Pick the best result with non-11 window
    test_config = results_with_deriv.iloc[0].to_dict()

    print("\n" + "="*80)
    print("SELECTED TEST RESULT")
    print("="*80)
    print(f"  Model: {test_config['Model']}")
    print(f"  Preprocess: {test_config['Preprocess']}")
    print(f"  Deriv: {test_config['Deriv']}")
    print(f"  Window: {test_config['Window']}")
    print(f"  RMSE (from Unified Bayesian): {test_config['RMSE']:.6f}")
    print(f"  R² (from Unified Bayesian): {test_config['R2']:.6f}")

    # 4. Simulate Model Development
    rmse_model_dev, r2_model_dev, window_used = simulate_model_development(
        test_config, X, y, wavelengths
    )

    # 5. Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Original Unified Bayesian result:")
    print(f"  Window: {test_config['Window']}")
    print(f"  RMSE: {test_config['RMSE']:.6f}")
    print(f"  R²: {test_config['R2']:.6f}")
    print()
    print(f"Model Development reproduction:")
    print(f"  Window used: {window_used}")
    print(f"  RMSE: {rmse_model_dev:.6f}")
    print(f"  R²: {r2_model_dev:.6f}")
    print()

    rmse_diff = abs(rmse_model_dev - test_config['RMSE'])
    r2_diff = abs(r2_model_dev - test_config['R2'])

    print(f"Differences:")
    print(f"  RMSE diff: {rmse_diff:.6f}")
    print(f"  R² diff: {r2_diff:.6f}")

    # Check for mismatch
    TOLERANCE = 1e-4
    if window_used != test_config['Window']:
        print(f"\n{'!'*80}")
        print(f"WINDOW MISMATCH DETECTED!")
        print(f"{'!'*80}")
        print(f"  Original window: {test_config['Window']}")
        print(f"  Model Development used: {window_used}")
        print(f"  This is the ROOT CAUSE of R² mismatch!")
        print(f"{'!'*80}")

    if r2_diff > TOLERANCE:
        print(f"\n{'!'*80}")
        print(f"R² MISMATCH CONFIRMED!")
        print(f"{'!'*80}")
        print(f"  Expected R²: {test_config['R2']:.6f}")
        print(f"  Got R²: {r2_model_dev:.6f}")
        print(f"  Difference: {r2_diff:.6f} (tolerance: {TOLERANCE})")
        print(f"  This proves Model Development is NOT reproducing the original result!")
        print(f"{'!'*80}")
        return False
    else:
        print(f"\n{'='*80}")
        print(f"SUCCESS: R² matches within tolerance!")
        print(f"{'='*80}")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
