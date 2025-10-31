"""
Baseline benchmark for optimization testing.

This script creates a simple test dataset and times the current implementation
to establish a baseline for performance improvements.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.spectral_predict.search import run_search


def create_test_data(n_samples=100, n_features=500, random_state=42):
    """Create synthetic spectral data for testing."""
    np.random.seed(random_state)

    # Generate random spectral data
    X = np.random.randn(n_samples, n_features) + 100

    # Add some structure (peaks at certain wavelengths)
    peaks = [100, 200, 300, 400]
    for peak in peaks:
        X[:, peak-10:peak+10] += np.random.randn(n_samples, 1) * 20

    # Generate target variable correlated with peak intensities
    y = (X[:, peaks].mean(axis=1) +
         np.random.randn(n_samples) * 5 +
         np.random.uniform(20, 80, n_samples))

    # Create wavelengths
    wavelengths = np.linspace(400, 2500, n_features)

    return X, y, wavelengths


def run_baseline_benchmark(save_results=True):
    """Run baseline benchmark with minimal but representative configuration."""
    print("=" * 70)
    print("BASELINE BENCHMARK")
    print("=" * 70)

    # Create test data
    print("\n1. Creating test dataset...")
    X, y, wavelengths = create_test_data(n_samples=100, n_features=500)

    # Convert to DataFrame
    col_names = [f"wl_{w:.1f}" for w in wavelengths]
    X_df = pd.DataFrame(X, columns=col_names)
    y_series = pd.Series(y, name="target")

    print(f"   Dataset shape: {X_df.shape}")
    print(f"   Target range: {y.min():.2f} - {y.max():.2f}")

    # Configure search - using minimal set for faster baseline
    print("\n2. Configuration:")
    models_to_test = ["PLS", "NeuralBoosted"]
    preprocess_methods = ["raw", "snv", "deriv"]

    print(f"   Models: {models_to_test}")
    print(f"   Preprocessing: {preprocess_methods}")
    print(f"   CV folds: 3")

    # Run benchmark
    print("\n3. Running search...")
    print("   (This establishes the baseline timing)")

    start_time = time.time()

    try:
        # Configure preprocessing methods
        preprocessing_config = {
            'raw': True,
            'snv': True,
            'sg1': True,  # First derivative
            'sg2': False,
            'snv_deriv': False,
            'deriv_snv': False
        }

        results_df = run_search(
            X_df,
            y_series,
            task_type="regression",
            folds=3,
            models_to_test=models_to_test,
            preprocessing_methods=preprocessing_config,
            window_sizes=[7],
            variable_counts=[50, 100],  # Small subset analysis
            lambda_penalty=0.1,
            enable_variable_subsets=True,
            enable_region_subsets=False,  # Disable for faster baseline
        )

        elapsed_time = time.time() - start_time

        print(f"\n4. Results:")
        print(f"   - Completed successfully")
        print(f"   - Total configurations tested: {len(results_df)}")
        print(f"   - Elapsed time: {elapsed_time:.2f} seconds")
        print(f"   - Time per config: {elapsed_time/len(results_df):.3f} seconds")

        # Save baseline results
        if save_results:
            results_df.to_csv("baseline_results.csv", index=False)
            print(f"   - Baseline results saved to: baseline_results.csv")

            # Save timing info
            with open("baseline_timing.txt", "w") as f:
                f.write(f"Baseline Timing\n")
                f.write(f"===============\n")
                f.write(f"Total time: {elapsed_time:.2f} seconds\n")
                f.write(f"Configurations: {len(results_df)}\n")
                f.write(f"Time per config: {elapsed_time/len(results_df):.3f} seconds\n")
            print(f"   - Baseline timing saved to: baseline_timing.txt")

        # Show top results
        print(f"\n5. Top 5 configurations by R²:")
        top_results = results_df.nlargest(5, 'R2_CV')[['Rank', 'Model', 'Preprocess', 'R2_CV', 'RMSE_CV', 'subset']]
        print(top_results.to_string(index=False))

        print("\n" + "=" * 70)
        print("BASELINE ESTABLISHED")
        print("=" * 70)

        return elapsed_time, results_df

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_with_baseline(new_time, new_results_df):
    """Compare new results with saved baseline."""
    try:
        # Load baseline
        with open("baseline_timing.txt", "r") as f:
            lines = f.readlines()
            baseline_time = float(lines[2].split(":")[1].strip().split()[0])

        baseline_df = pd.read_csv("baseline_results.csv")

        # Calculate speedup
        speedup = baseline_time / new_time

        print("\n" + "=" * 70)
        print("COMPARISON WITH BASELINE")
        print("=" * 70)
        print(f"Baseline time: {baseline_time:.2f} seconds")
        print(f"New time:      {new_time:.2f} seconds")
        print(f"Speedup:       {speedup:.2f}x")

        # Check accuracy preservation
        if new_results_df is not None:
            # Compare R2 values for matching configurations
            merged = baseline_df.merge(
                new_results_df,
                on=['Model', 'Preprocess', 'subset'],
                suffixes=('_baseline', '_new')
            )

            if len(merged) > 0:
                r2_diff = np.abs(merged['R2_CV_baseline'] - merged['R2_CV_new'])
                rmse_diff = np.abs(merged['RMSE_CV_baseline'] - merged['RMSE_CV_new'])

                print(f"\nAccuracy preservation:")
                print(f"  Mean R² difference:   {r2_diff.mean():.6f}")
                print(f"  Max R² difference:    {r2_diff.max():.6f}")
                print(f"  Mean RMSE difference: {rmse_diff.mean():.6f}")
                print(f"  Max RMSE difference:  {rmse_diff.max():.6f}")

                if r2_diff.max() < 0.001 and rmse_diff.max() < 0.1:
                    print(f"  [OK] Accuracy preserved (within tolerance)")
                else:
                    print(f"  [WARNING] Accuracy differences detected")

        print("=" * 70)

        return speedup

    except FileNotFoundError:
        print("No baseline found. Run with save_results=True first.")
        return None


if __name__ == "__main__":
    print("\nThis benchmark uses a small test dataset to measure performance.")
    print("It establishes a baseline for optimization improvements.\n")

    baseline_time, baseline_df = run_baseline_benchmark(save_results=True)

    if baseline_time:
        print(f"\n[OK] Baseline benchmark complete!")
        print(f"  Use this baseline to compare optimization improvements.")
        print(f"  To compare: compare_with_baseline(new_time, new_results_df)")
