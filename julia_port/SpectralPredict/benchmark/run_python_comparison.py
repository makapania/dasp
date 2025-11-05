"""
run_python_comparison.py

Helper script to benchmark Python implementations for comparison with Julia.

This script runs equivalent tests to the Julia benchmarks, allowing direct
performance comparison.

Usage:
    python benchmark/run_python_comparison.py

Output:
    - Console output with timing results
    - python_benchmark_results.json for comparison
"""

import time
import json
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from spectral_predict.variable_selection import (
        uve_selection, spa_selection, ipls_selection, uve_spa_selection
    )
    from spectral_predict.diagnostics import (
        compute_residuals, compute_leverage, jackknife_prediction_intervals
    )
    from spectral_predict.neural_boosted import NeuralBoostedRegressor
    from spectral_predict.preprocess import apply_msc
    IMPORTS_OK = True
except ImportError as e:
    print(f"Warning: Could not import SpectralPredict modules: {e}")
    print("Make sure you're running from the correct directory.")
    IMPORTS_OK = False


def generate_synthetic_spectral_data(n_samples, n_wavelengths, n_informative, seed=42):
    """Generate synthetic spectral data matching Julia version."""
    np.random.seed(seed)

    # Create informative wavelengths
    true_indices = sorted(np.random.choice(n_wavelengths, n_informative, replace=False))

    # Generate spectra
    X = np.zeros((n_samples, n_wavelengths))

    for i in range(n_samples):
        # Baseline
        baseline = 1.0 + 0.1 * np.random.randn()

        # Smooth curve
        wavelengths = np.linspace(0, 2*np.pi, n_wavelengths)
        smooth_curve = np.sin(wavelengths + np.random.randn() * 0.5) * 0.3

        # Informative peaks
        informative_signal = np.zeros(n_wavelengths)
        for idx in true_indices:
            width = 10.0
            for j in range(n_wavelengths):
                dist = (j - idx)**2
                informative_signal[j] += np.exp(-dist / (2 * width**2)) * np.random.randn()

        # Noise
        noise = np.random.randn(n_wavelengths) * 0.05

        # Combine
        X[i, :] = baseline + smooth_curve + informative_signal + noise

    # Generate target
    beta = np.random.randn(n_informative)
    y = X[:, true_indices] @ beta + np.random.randn(n_samples) * 0.1

    return X, y, true_indices


def generate_regression_data(n_samples, n_features, noise_level=0.1, seed=42):
    """Generate regression data for diagnostics/neural boosted tests."""
    np.random.seed(seed)

    X = np.random.randn(n_samples, n_features)

    # Nonlinear target
    y = np.zeros(n_samples)
    for i in range(n_samples):
        y[i] = np.sin(X[i, 0]) + X[i, 1]**2 + np.tanh(X[i, min(2, n_features-1)])
        if n_features >= 5:
            y[i] += X[i, 3] * X[i, 4]

    y += np.random.randn(n_samples) * noise_level

    return X, y


def time_function(func, *args, n_runs=5, warmup=2):
    """Time a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = func(*args)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def benchmark_variable_selection():
    """Benchmark variable selection methods."""
    print("\n" + "="*80)
    print("PYTHON: Variable Selection Benchmarks")
    print("="*80)

    test_scales = [
        ("Small (100 × 500)", 100, 500, 20),
        ("Medium (300 × 1500)", 300, 1500, 50),
        ("Large (1000 × 2151)", 1000, 2151, 100)
    ]

    results = {}

    for scale_name, n_samples, n_wavelengths, n_informative in test_scales:
        print(f"\n{scale_name}")
        print("-"*60)

        X, y, true_indices = generate_synthetic_spectral_data(
            n_samples, n_wavelengths, n_informative
        )

        scale_results = {}

        # UVE
        print("  UVE selection...")
        stats = time_function(uve_selection, X, y, n_components=10, cv_folds=5,
                            noise_factor=1.0, n_runs=5)
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} s")
        scale_results['uve'] = stats

        # SPA
        print("  SPA selection...")
        n_vars = min(50, n_wavelengths // 10)
        stats = time_function(spa_selection, X, y, n_vars=n_vars, n_runs=5)
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} s")
        scale_results['spa'] = stats

        # iPLS
        print("  iPLS selection...")
        stats = time_function(ipls_selection, X, y, n_intervals=10,
                            n_components=5, cv_folds=5, n_runs=3)
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} s")
        scale_results['ipls'] = stats

        # UVE-SPA
        print("  UVE-SPA selection...")
        n_vars = min(30, n_wavelengths // 20)
        stats = time_function(uve_spa_selection, X, y, n_vars=n_vars,
                            n_components=10, cv_folds=5, noise_factor=1.0, n_runs=5)
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} s")
        scale_results['uve_spa'] = stats

        results[scale_name] = scale_results

    return results


def benchmark_diagnostics():
    """Benchmark diagnostics methods."""
    print("\n" + "="*80)
    print("PYTHON: Diagnostics Benchmarks")
    print("="*80)

    test_scales = [
        ("Small (100 × 50)", 100, 50),
        ("Medium (300 × 150)", 300, 150),
        ("Large (1000 × 300)", 1000, 300)
    ]

    results = {}

    for scale_name, n_samples, n_features in test_scales:
        print(f"\n{scale_name}")
        print("-"*60)

        X, y = generate_regression_data(n_samples, n_features)

        scale_results = {}

        # Residuals
        print("  Residual analysis...")
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        stats = time_function(compute_residuals, y, y_pred, n_runs=10)
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} s")
        scale_results['residuals'] = stats

        # Leverage
        print("  Leverage computation...")
        stats = time_function(compute_leverage, X, n_runs=5)
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} s")
        scale_results['leverage'] = stats

        # Jackknife (only for small/medium)
        if n_samples <= 300:
            print("  Jackknife intervals...")
            n_train = int(n_samples * 0.7)
            X_train, y_train = X[:n_train], y[:n_train]
            X_test = X[n_train:]

            def model_fn(X_fit, y_fit):
                return np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]

            stats = time_function(jackknife_prediction_intervals,
                                model_fn, X_train, y_train, X_test,
                                confidence=0.95, n_runs=3)
            print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} s")
            scale_results['jackknife'] = stats

        results[scale_name] = scale_results

    return results


def benchmark_neural_boosted():
    """Benchmark neural boosted regressor."""
    print("\n" + "="*80)
    print("PYTHON: Neural Boosted Benchmarks")
    print("="*80)

    test_scales = [
        ("Small (100 × 50)", 100, 50),
        ("Medium (300 × 150)", 300, 150),
        ("Large (1000 × 300)", 1000, 300)
    ]

    results = {}

    for scale_name, n_samples, n_features in test_scales:
        print(f"\n{scale_name}")
        print("-"*60)

        X, y = generate_regression_data(n_samples, n_features)
        n_train = int(n_samples * 0.7)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test = X[n_train:]

        scale_results = {}

        # Training
        print("  Training...")
        def train_model():
            model = NeuralBoostedRegressor(
                n_estimators=100,
                learning_rate=0.1,
                hidden_layer_sizes=(3,),
                activation='tanh',
                early_stopping=True,
                validation_fraction=0.15,
                verbose=0,
                random_state=42
            )
            model.fit(X_train, y_train)
            return model

        stats = time_function(train_model, n_runs=3)
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} s")
        scale_results['training'] = stats

        # Train one model for prediction test
        model = train_model()

        # Prediction
        print("  Prediction...")
        stats = time_function(model.predict, X_test, n_runs=10)
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} s")
        scale_results['prediction'] = stats

        results[scale_name] = scale_results

    return results


def benchmark_msc():
    """Benchmark MSC preprocessing."""
    print("\n" + "="*80)
    print("PYTHON: MSC Preprocessing Benchmarks")
    print("="*80)

    test_scales = [
        ("Small (100 × 500)", 100, 500),
        ("Medium (300 × 1500)", 300, 1500),
        ("Large (1000 × 2151)", 1000, 2151),
        ("Extra Large (5000 × 2151)", 5000, 2151)
    ]

    results = {}

    for scale_name, n_samples, n_wavelengths in test_scales:
        print(f"\n{scale_name}")
        print("-"*60)

        # Generate spectral data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_wavelengths) * 0.1 + 1.0

        print("  MSC computation...")
        stats = time_function(apply_msc, X, n_runs=5)
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} s")

        results[scale_name] = {'msc': stats}

    return results


def main():
    """Run all Python benchmarks."""
    if not IMPORTS_OK:
        print("ERROR: Cannot run benchmarks - imports failed")
        return

    print("="*80)
    print("Python Performance Benchmarks")
    print("SpectralPredict - Python Implementation")
    print("="*80)
    print(f"Start Time: {datetime.now()}")
    print(f"Python Version: {sys.version}")
    print(f"NumPy Version: {np.__version__}")
    print()

    all_results = {}

    # Run benchmarks
    all_results['variable_selection'] = benchmark_variable_selection()
    all_results['diagnostics'] = benchmark_diagnostics()
    all_results['neural_boosted'] = benchmark_neural_boosted()
    all_results['msc'] = benchmark_msc()

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'results': all_results
    }

    output_path = os.path.join(os.path.dirname(__file__), 'python_benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*80)
    print("Python benchmarks complete!")
    print(f"Results saved to: {output_path}")
    print()
    print("Next steps:")
    print("  1. Run Julia benchmarks: julia --threads=auto benchmark/bench_comprehensive.jl")
    print("  2. Compare results to calculate speedups")
    print("="*80)


if __name__ == '__main__':
    main()
