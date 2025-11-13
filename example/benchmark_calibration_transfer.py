"""
Benchmark Calibration Transfer Methods

Compares performance of all available calibration transfer methods:
- DS (Direct Standardization)
- PDS (Piecewise Direct Standardization)
- TSR (Transfer Sample Regression / Shenk-Westerhaus)
- CTAI (Calibration Transfer based on Affine Invariance)
- NS-PFCE (Non-supervised Parameter-Free Calibration Enhancement)
- JYPLS-inv (Joint-Y PLS with Inversion)

This script generates synthetic data with known transformations and
evaluates how well each method recovers the original spectra.
"""

import numpy as np
import time
from pathlib import Path

# Import calibration transfer methods
from spectral_predict.calibration_transfer import (
    estimate_ds, apply_ds,
    estimate_pds, apply_pds,
    estimate_tsr, apply_tsr,
    estimate_ctai, apply_ctai
)

# Import sample selection
from spectral_predict.sample_selection import kennard_stone


def generate_synthetic_data(
    n_master: int = 100,
    n_slave: int = 100,
    n_wavelengths: int = 200,
    transformation_type: str = 'affine',
    noise_level: float = 0.01,
    random_seed: int = 42
):
    """
    Generate synthetic master and slave spectra with known transformation.

    Parameters
    ----------
    n_master : int
        Number of master instrument samples.
    n_slave : int
        Number of slave instrument samples.
    n_wavelengths : int
        Number of wavelengths.
    transformation_type : str
        Type of transformation: 'affine', 'wavelength_dependent', 'nonlinear'
    noise_level : float
        Standard deviation of Gaussian noise to add.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_master : np.ndarray
        Master spectra (n_master, n_wavelengths)
    X_slave : np.ndarray
        Slave spectra (n_slave, n_wavelengths)
    transform_params : dict
        True transformation parameters
    """
    np.random.seed(random_seed)

    # Generate master spectra
    X_master = np.random.randn(n_master, n_wavelengths)

    # Generate slave base spectra (different samples for realistic scenario)
    X_slave_base = np.random.randn(n_slave, n_wavelengths)

    # Apply transformation
    if transformation_type == 'affine':
        # Simple affine: X_slave = slope * X_base + bias
        slope = 0.92
        bias = 0.08
        X_slave = slope * X_slave_base + bias
        transform_params = {'slope': slope, 'bias': bias}

    elif transformation_type == 'wavelength_dependent':
        # Different slope/bias per wavelength
        slopes = np.linspace(0.85, 1.05, n_wavelengths)
        biases = np.linspace(-0.1, 0.1, n_wavelengths)
        X_slave = X_slave_base * slopes + biases
        transform_params = {'slopes': slopes, 'biases': biases}

    elif transformation_type == 'nonlinear':
        # Slight nonlinearity (harder for methods to handle)
        slope = 0.92
        bias = 0.08
        X_slave = slope * X_slave_base + bias + 0.02 * X_slave_base**2
        transform_params = {'slope': slope, 'bias': bias, 'nonlinear_coeff': 0.02}

    else:
        raise ValueError(f"Unknown transformation type: {transformation_type}")

    # Add noise
    X_master += noise_level * np.random.randn(*X_master.shape)
    X_slave += noise_level * np.random.randn(*X_slave.shape)

    return X_master, X_slave, transform_params


def evaluate_transfer(X_master, X_slave, X_transferred, metric='rmse'):
    """
    Evaluate transfer quality.

    Parameters
    ----------
    X_master : np.ndarray
        True master spectra.
    X_slave : np.ndarray
        Original slave spectra (before transfer).
    X_transferred : np.ndarray
        Slave spectra after calibration transfer.
    metric : str
        Metric to use: 'rmse', 'mae', 'r2'

    Returns
    -------
    score : float
        Evaluation score.
    """
    # Use overlapping samples for comparison
    n_compare = min(X_master.shape[0], X_transferred.shape[0])
    X_master_sub = X_master[:n_compare]
    X_transferred_sub = X_transferred[:n_compare]

    if metric == 'rmse':
        return np.sqrt(np.mean((X_transferred_sub - X_master_sub) ** 2))

    elif metric == 'mae':
        return np.mean(np.abs(X_transferred_sub - X_master_sub))

    elif metric == 'r2':
        ss_res = np.sum((X_transferred_sub - X_master_sub) ** 2)
        ss_tot = np.sum((X_master_sub - np.mean(X_master_sub)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    else:
        raise ValueError(f"Unknown metric: {metric}")


def benchmark_all_methods(
    X_master,
    X_slave,
    n_transfer_samples=12,
    ds_lambda=0.001,
    pds_window=11
):
    """
    Benchmark all calibration transfer methods.

    Returns
    -------
    results : dict
        Dictionary with method names as keys and results as values.
    """
    results = {}

    print("\n" + "="*80)
    print("BENCHMARKING CALIBRATION TRANSFER METHODS")
    print("="*80)

    # ----- METHOD 1: DS (Direct Standardization) -----
    print("\n1. Testing DS (Direct Standardization)...")
    start_time = time.time()
    try:
        A = estimate_ds(X_master, X_slave, lam=ds_lambda)
        X_ds = apply_ds(X_slave, A)
        elapsed = time.time() - start_time

        results['DS'] = {
            'method': 'Direct Standardization',
            'rmse': evaluate_transfer(X_master, X_slave, X_ds, 'rmse'),
            'mae': evaluate_transfer(X_master, X_slave, X_ds, 'mae'),
            'r2': evaluate_transfer(X_master, X_slave, X_ds, 'r2'),
            'time_seconds': elapsed,
            'transfer_samples': X_master.shape[0],  # Uses all samples
            'success': True
        }
        print(f"   ✓ Completed in {elapsed:.4f}s")
        print(f"   RMSE: {results['DS']['rmse']:.6f}")

    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        results['DS'] = {'success': False, 'error': str(e)}

    # ----- METHOD 2: PDS (Piecewise Direct Standardization) -----
    print("\n2. Testing PDS (Piecewise Direct Standardization)...")
    start_time = time.time()
    try:
        B = estimate_pds(X_master, X_slave, window=pds_window)
        X_pds = apply_pds(X_slave, B, window=pds_window)
        elapsed = time.time() - start_time

        results['PDS'] = {
            'method': 'Piecewise Direct Standardization',
            'rmse': evaluate_transfer(X_master, X_slave, X_pds, 'rmse'),
            'mae': evaluate_transfer(X_master, X_slave, X_pds, 'mae'),
            'r2': evaluate_transfer(X_master, X_slave, X_pds, 'r2'),
            'time_seconds': elapsed,
            'transfer_samples': X_master.shape[0],
            'window': pds_window,
            'success': True
        }
        print(f"   ✓ Completed in {elapsed:.4f}s")
        print(f"   RMSE: {results['PDS']['rmse']:.6f}")

    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        results['PDS'] = {'success': False, 'error': str(e)}

    # ----- METHOD 3: TSR (Transfer Sample Regression) -----
    print(f"\n3. Testing TSR (Transfer Sample Regression) with {n_transfer_samples} samples...")
    start_time = time.time()
    try:
        # Select transfer samples using Kennard-Stone
        transfer_indices = kennard_stone(X_master, n_samples=n_transfer_samples)

        tsr_params = estimate_tsr(X_master, X_slave, transfer_indices)
        X_tsr = apply_tsr(X_slave, tsr_params)
        elapsed = time.time() - start_time

        results['TSR'] = {
            'method': 'Transfer Sample Regression (Shenk-Westerhaus)',
            'rmse': evaluate_transfer(X_master, X_slave, X_tsr, 'rmse'),
            'mae': evaluate_transfer(X_master, X_slave, X_tsr, 'mae'),
            'r2': evaluate_transfer(X_master, X_slave, X_tsr, 'r2'),
            'time_seconds': elapsed,
            'transfer_samples': n_transfer_samples,
            'mean_r_squared': tsr_params['mean_r_squared'],
            'success': True
        }
        print(f"   ✓ Completed in {elapsed:.4f}s")
        print(f"   RMSE: {results['TSR']['rmse']:.6f}")
        print(f"   Mean R²: {tsr_params['mean_r_squared']:.4f}")

    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        results['TSR'] = {'success': False, 'error': str(e)}

    # ----- METHOD 4: CTAI (Affine Invariance) -----
    print("\n4. Testing CTAI (Calibration Transfer based on Affine Invariance)...")
    start_time = time.time()
    try:
        ctai_params = estimate_ctai(X_master, X_slave)
        X_ctai = apply_ctai(X_slave, ctai_params)
        elapsed = time.time() - start_time

        results['CTAI'] = {
            'method': 'Calibration Transfer based on Affine Invariance',
            'rmse': evaluate_transfer(X_master, X_slave, X_ctai, 'rmse'),
            'mae': evaluate_transfer(X_master, X_slave, X_ctai, 'mae'),
            'r2': evaluate_transfer(X_master, X_slave, X_ctai, 'r2'),
            'time_seconds': elapsed,
            'transfer_samples': 0,  # NO samples needed!
            'n_components': ctai_params['n_components'],
            'explained_variance': ctai_params['explained_variance'],
            'success': True
        }
        print(f"   ✓ Completed in {elapsed:.4f}s")
        print(f"   RMSE: {results['CTAI']['rmse']:.6f}")
        print(f"   Components: {ctai_params['n_components']}")
        print(f"   Explained Variance: {ctai_params['explained_variance']:.4f}")

    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        results['CTAI'] = {'success': False, 'error': str(e)}

    # ----- METHOD 5: NS-PFCE (Non-supervised Parameter-Free Calibration Enhancement) -----
    print("\n5. Testing NS-PFCE (Non-supervised Parameter-Free Calibration Enhancement)...")
    start_time = time.time()
    try:
        from spectral_predict.calibration_transfer import estimate_nspfce, apply_nspfce

        # Generate wavelengths for NS-PFCE (required parameter)
        n_wavelengths = X_master.shape[1]
        wavelengths = np.linspace(1000, 2500, n_wavelengths)

        # Test with wavelength selection (VCPA-IRIV)
        nspfce_params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=True,
            wavelength_selector='vcpa-iriv',
            max_iterations=100
        )
        X_nspfce = apply_nspfce(X_slave, nspfce_params)
        elapsed = time.time() - start_time

        results['NS-PFCE'] = {
            'method': 'Non-supervised Parameter-Free Calibration Enhancement',
            'rmse': evaluate_transfer(X_master, X_slave, X_nspfce, 'rmse'),
            'mae': evaluate_transfer(X_master, X_slave, X_nspfce, 'mae'),
            'r2': evaluate_transfer(X_master, X_slave, X_nspfce, 'r2'),
            'time_seconds': elapsed,
            'transfer_samples': 0,  # NO samples needed!
            'iterations': nspfce_params['n_iterations'],
            'converged': nspfce_params['converged'],
            'wavelength_selection': True,
            'n_selected_wavelengths': len(nspfce_params.get('selected_wavelength_indices', [])),
            'success': True
        }
        print(f"   ✓ Completed in {elapsed:.4f}s")
        print(f"   RMSE: {results['NS-PFCE']['rmse']:.6f}")
        print(f"   Iterations: {nspfce_params['n_iterations']}")
        print(f"   Converged: {nspfce_params['converged']}")
        if nspfce_params.get('selected_wavelength_indices') is not None:
            n_selected = len(nspfce_params['selected_wavelength_indices'])
            print(f"   Selected Wavelengths: {n_selected} / {n_wavelengths}")

    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        results['NS-PFCE'] = {'success': False, 'error': str(e)}

    # ----- METHOD 6: JYPLS-inv (Joint-Y PLS with Inversion) -----
    print("\n6. Testing JYPLS-inv (Joint-Y PLS with Inversion)...")
    start_time = time.time()
    try:
        from spectral_predict.calibration_transfer import estimate_jypls_inv, apply_jypls_inv
        from spectral_predict.sample_selection import kennard_stone

        # Select transfer samples (12-13 optimal)
        n_transfer = 12
        transfer_idx = kennard_stone(X_master, n_samples=n_transfer)

        # Generate pseudo-Y values for transfer samples (spectral mean)
        y_transfer = X_master[transfer_idx].mean(axis=1)

        # Estimate JYPLS-inv with auto component selection
        jypls_params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=None  # Auto-select via CV
        )
        X_jypls = apply_jypls_inv(X_slave, jypls_params)
        elapsed = time.time() - start_time

        results['JYPLS-inv'] = {
            'method': 'Joint-Y PLS with Inversion',
            'rmse': evaluate_transfer(X_master, X_slave, X_jypls, 'rmse'),
            'mae': evaluate_transfer(X_master, X_slave, X_jypls, 'mae'),
            'r2': evaluate_transfer(X_master, X_slave, X_jypls, 'r2'),
            'time_seconds': elapsed,
            'transfer_samples': n_transfer,
            'n_components': jypls_params['n_components'],
            'cv_rmse': jypls_params['cv_rmse'],
            'explained_variance': jypls_params['explained_variance_ratio'],
            'success': True
        }
        print(f"   ✓ Completed in {elapsed:.4f}s")
        print(f"   RMSE: {results['JYPLS-inv']['rmse']:.6f}")
        print(f"   PLS Components: {jypls_params['n_components']} (auto-selected)")
        print(f"   CV RMSE: {jypls_params['cv_rmse']:.6f}")
        print(f"   Explained Variance: {jypls_params['explained_variance_ratio']:.4f}")

    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        results['JYPLS-inv'] = {'success': False, 'error': str(e)}

    return results


def print_summary_table(results):
    """Print formatted summary table of results."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Header
    print(f"{'Method':<10} {'RMSE':>12} {'MAE':>12} {'R²':>12} {'Time (s)':>12} {'Samples':>10}")
    print("-"*80)

    # Sort by RMSE (lower is better)
    sorted_methods = sorted(
        [(k, v) for k, v in results.items() if v.get('success', False)],
        key=lambda x: x[1].get('rmse', float('inf'))
    )

    for method_name, result in sorted_methods:
        rmse = result.get('rmse', float('nan'))
        mae = result.get('mae', float('nan'))
        r2 = result.get('r2', float('nan'))
        time_s = result.get('time_seconds', float('nan'))
        n_samples = result.get('transfer_samples', '?')

        # Highlight best method with ★
        prefix = "★" if sorted_methods[0][0] == method_name else " "

        print(f"{prefix}{method_name:<9} {rmse:12.6f} {mae:12.6f} {r2:12.4f} {time_s:12.4f} {n_samples:>10}")

    print("-"*80)
    print("★ = Best performance (lowest RMSE)")
    print("\nKey findings:")
    print(f"  • Best method: {sorted_methods[0][0]} (RMSE={sorted_methods[0][1]['rmse']:.6f})")
    print(f"  • Fastest method: {min(sorted_methods, key=lambda x: x[1]['time_seconds'])[0]}")
    print(f"  • No samples needed: CTAI, NS-PFCE")
    print(f"  • Fewest samples (12-13): TSR, JYPLS-inv")
    print(f"  • Advanced wavelength selection: NS-PFCE")
    print(f"  • PLS-based transfer: JYPLS-inv")


def main():
    """Run complete benchmark."""
    print("Calibration Transfer Methods Benchmark")
    print("="*80)

    # Test different scenarios
    scenarios = [
        {
            'name': 'Simple Affine Transformation',
            'transformation_type': 'affine',
            'n_master': 100,
            'n_slave': 100,
            'n_wavelengths': 200,
            'noise_level': 0.01
        },
        {
            'name': 'Wavelength-Dependent Transformation',
            'transformation_type': 'wavelength_dependent',
            'n_master': 80,
            'n_slave': 120,
            'n_wavelengths': 150,
            'noise_level': 0.02
        },
        {
            'name': 'High Noise',
            'transformation_type': 'affine',
            'n_master': 100,
            'n_slave': 100,
            'n_wavelengths': 200,
            'noise_level': 0.05
        }
    ]

    all_results = {}

    for i, scenario in enumerate(scenarios):
        print(f"\n\n{'='*80}")
        print(f"SCENARIO {i+1}: {scenario['name']}")
        print(f"{'='*80}")
        print(f"  Master samples: {scenario['n_master']}")
        print(f"  Slave samples: {scenario['n_slave']}")
        print(f"  Wavelengths: {scenario['n_wavelengths']}")
        print(f"  Noise level: {scenario['noise_level']}")

        # Generate data
        X_master, X_slave, transform_params = generate_synthetic_data(
            n_master=scenario['n_master'],
            n_slave=scenario['n_slave'],
            n_wavelengths=scenario['n_wavelengths'],
            transformation_type=scenario['transformation_type'],
            noise_level=scenario['noise_level']
        )

        # Run benchmark
        results = benchmark_all_methods(X_master, X_slave)

        # Print results
        print_summary_table(results)

        all_results[scenario['name']] = results

    # Overall conclusions
    print("\n\n" + "="*80)
    print("OVERALL CONCLUSIONS")
    print("="*80)

    for scenario_name, results in all_results.items():
        successful = [k for k, v in results.items() if v.get('success', False)]
        if successful:
            best = min(successful, key=lambda k: results[k]['rmse'])
            print(f"\n{scenario_name}:")
            print(f"  Best: {best} (RMSE={results[best]['rmse']:.6f})")

    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)


if __name__ == "__main__":
    main()
