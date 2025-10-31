"""Test optimized version and compare with baseline."""
import time
import sys

# Import baseline benchmark functions
from benchmark_baseline import create_test_data, compare_with_baseline

# Import the optimized search function
from src.spectral_predict.search import run_search
import pandas as pd


def test_optimized():
    """Test the optimized version with parallel CV."""
    print("\n" + "=" * 70)
    print("TESTING OPTIMIZED VERSION (Parallel CV)")
    print("=" * 70)

    # Create same test data
    print("\n1. Creating test dataset...")
    X, y, wavelengths = create_test_data(n_samples=100, n_features=500)

    # Convert to DataFrame
    col_names = [f"wl_{w:.1f}" for w in wavelengths]
    X_df = pd.DataFrame(X, columns=col_names)
    y_series = pd.Series(y, name="target")

    print(f"   Dataset shape: {X_df.shape}")

    # Configure search - same as baseline
    print("\n2. Configuration (same as baseline)")
    models_to_test = ["PLS", "NeuralBoosted"]

    preprocessing_config = {
        'raw': True,
        'snv': True,
        'sg1': True,  # First derivative
        'sg2': False,
        'snv_deriv': False,
        'deriv_snv': False
    }

    # Run optimized version
    print("\n3. Running optimized search with parallel CV...")
    print("   (This uses joblib to parallelize CV folds)")

    start_time = time.time()

    try:
        results_df = run_search(
            X_df,
            y_series,
            task_type="regression",
            folds=3,
            models_to_test=models_to_test,
            preprocessing_methods=preprocessing_config,
            window_sizes=[7],
            variable_counts=[50, 100],
            lambda_penalty=0.1,
            enable_variable_subsets=True,
            enable_region_subsets=False,
        )

        elapsed_time = time.time() - start_time

        print(f"\n4. Results:")
        print(f"   - Completed successfully")
        print(f"   - Total configurations: {len(results_df)}")
        print(f"   - Elapsed time: {elapsed_time:.2f} seconds")
        print(f"   - Time per config: {elapsed_time/len(results_df):.3f} seconds")

        # Compare with baseline
        print("\n5. Comparing with baseline...")
        speedup = compare_with_baseline(elapsed_time, results_df)

        print("\n" + "=" * 70)
        if speedup and speedup >= 2.0:
            print(f"SUCCESS! Achieved {speedup:.2f}x speedup with parallel CV")
        elif speedup and speedup >= 1.5:
            print(f"GOOD! Achieved {speedup:.2f}x speedup (expected 3-4x on typical CPUs)")
        elif speedup:
            print(f"Speedup: {speedup:.2f}x (less than expected, may vary by system)")
        print("=" * 70)

        return elapsed_time, results_df, speedup

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    print("Testing optimized version with parallel CV...")
    print("This should be 3-4x faster on a typical multi-core CPU.\n")

    elapsed_time, results_df, speedup = test_optimized()

    if speedup:
        print(f"\nFinal speedup: {speedup:.2f}x")
    else:
        print("\nFailed to complete optimization test.")
        sys.exit(1)
