"""
Test script to check why some rows have null RMSE values.
"""

import numpy as np
import pandas as pd
from spectral_predict_julia_bridge import run_search_julia

# Create simple test data
np.random.seed(42)
n_samples = 30
n_features = 50

X = pd.DataFrame(
    np.random.rand(n_samples, n_features),
    columns=[str(w) for w in range(400, 400 + n_features)]
)
y = pd.Series(np.random.rand(n_samples))

print("Testing Julia Backend - Checking NULL values")
print("="*70)

results = run_search_julia(
    X, y,
    task_type='regression',
    models_to_test=['PLS', 'Ridge'],
    preprocessing_methods={'raw': True, 'snv': True},
    enable_variable_subsets=True,
    variable_counts=[10, 20],
    variable_selection_methods=['importance'],
    enable_region_subsets=False,
    folds=3,
    progress_callback=None  # No progress output
)

print()
print("Rows with NULL RMSE:")
null_rmse = results[results['RMSE'].isnull()]
print(null_rmse[['Model', 'Preprocess', 'Subset', 'n_vars', 'LVs', 'RMSE', 'R2']])
print()

print("Total rows with NULL RMSE:", len(null_rmse))
print()

# Now test with NeuralBoosted
print("="*70)
print("Testing with NeuralBoosted model...")
print("="*70)
print()

try:
    results2 = run_search_julia(
        X, y,
        task_type='regression',
        models_to_test=['PLS', 'NeuralBoosted'],
        preprocessing_methods={'raw': True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        folds=3,
        progress_callback=None
    )

    print()
    print(f"SUCCESS: NeuralBoosted test returned {len(results2)} results")
    print()
    print("Results by model:")
    print(results2.groupby('Model').size())
    print()
    print("Check for NULL RMSE:")
    null_rmse2 = results2[results2['RMSE'].isnull()]
    print(f"Rows with NULL RMSE: {len(null_rmse2)}")
    if len(null_rmse2) > 0:
        print(null_rmse2[['Model', 'Preprocess', 'RMSE', 'R2']])

except Exception as e:
    print(f"ERROR testing NeuralBoosted: {e}")
    import traceback
    traceback.print_exc()
