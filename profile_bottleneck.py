"""
Profile DASP to identify actual bottlenecks.

Run this with your typical dataset to see where time is spent.
"""

import time
import numpy as np
import pandas as pd
from src.spectral_predict.search import run_search
from src.spectral_predict.models import get_model_grids

def profile_models():
    """Profile each model type to see which are slow."""

    # Create synthetic data similar to your real data
    n_samples = 50
    n_features = 1700

    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"{1000 + i*0.5:.1f}" for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples) * 10 + 50)

    # Test each model type individually
    model_times = {}

    models_to_test = [
        'PLS',
        'Ridge',
        'Lasso',
        'ElasticNet',
        'RandomForest',
        'XGBoost',
        'LightGBM',
        'MLP',
        'NeuralBoosted'
    ]

    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}...")
        print(f"{'='*60}")

        start = time.time()

        try:
            results = run_search(
                X, y,
                task_type='regression',
                folds=5,
                models_to_test=[model_name],
                preprocessing_methods=['raw', 'snv'],  # Just 2 preprocessing methods
                enable_variable_subsets=False,  # Disable subsets for this test
                enable_region_subsets=False,
                tier='quick'  # Use quick tier to minimize configs
            )

            elapsed = time.time() - start
            model_times[model_name] = elapsed

            n_configs = len(results)
            time_per_config = elapsed / n_configs if n_configs > 0 else 0

            print(f"\n✓ {model_name}:")
            print(f"  Total time: {elapsed:.1f}s")
            print(f"  Configs tested: {n_configs}")
            print(f"  Time per config: {time_per_config:.2f}s")

        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            model_times[model_name] = 0

    # Summary
    print(f"\n{'='*60}")
    print("BOTTLENECK ANALYSIS")
    print(f"{'='*60}")

    sorted_times = sorted(model_times.items(), key=lambda x: x[1], reverse=True)

    print("\nModel types ranked by total time (slowest first):\n")
    for model_name, total_time in sorted_times:
        if total_time > 0:
            percentage = (total_time / sum(model_times.values())) * 100
            print(f"  {model_name:20s}: {total_time:6.1f}s  ({percentage:5.1f}%)")

    print("\nCONCLUSION:")
    print(f"  Slowest model: {sorted_times[0][0]} ({sorted_times[0][1]:.1f}s)")
    print(f"  2nd slowest: {sorted_times[1][0]} ({sorted_times[1][1]:.1f}s)")
    print(f"  3rd slowest: {sorted_times[2][0]} ({sorted_times[2][1]:.1f}s)")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    top_3_time = sum([t[1] for t in sorted_times[:3]])
    total_time = sum(model_times.values())

    print(f"  Top 3 models account for {(top_3_time/total_time)*100:.1f}% of time")
    print(f"  → Focus optimization on: {sorted_times[0][0]}, {sorted_times[1][0]}, {sorted_times[2][0]}")

    if 'NeuralBoosted' in [t[0] for t in sorted_times[:3]]:
        print(f"\n  NeuralBoosted is a bottleneck!")
        print(f"  → Try reducing n_estimators from 100 to 30-50")
        print(f"  → Or migrate to Julia/Flux for 10-20x speedup")

    if 'MLP' in [t[0] for t in sorted_times[:3]]:
        print(f"\n  MLP is a bottleneck!")
        print(f"  → Try reducing max_iter or using early stopping")
        print(f"  → Or migrate to Julia/Flux")

if __name__ == "__main__":
    print("DASP Bottleneck Profiler")
    print("=" * 60)
    print("This will test each model type to identify bottlenecks.")
    print("Using synthetic data (50 samples × 1700 wavelengths)")
    print("=" * 60)

    profile_models()
