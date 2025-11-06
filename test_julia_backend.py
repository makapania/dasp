"""
Test script to directly call Julia backend and check what it returns.
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

print("="*70)
print("Testing Julia Backend")
print("="*70)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print()

try:
    print("Calling run_search_julia...")
    print()

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
        progress_callback=lambda info: print(f"  Progress: {info.get('message', '')}")
    )

    print()
    print("="*70)
    print("Julia Backend Test Results")
    print("="*70)
    print()

    if results is None:
        print("ERROR: run_search_julia returned None!")
    elif len(results) == 0:
        print("ERROR: run_search_julia returned empty DataFrame!")
    else:
        print(f"SUCCESS: Results returned: {len(results)} configurations")
        print()
        print("Columns:")
        print(list(results.columns))
        print()
        print("First few rows:")
        print(results.head())
        print()
        print("Data types:")
        print(results.dtypes)
        print()
        print("Check for empty/null values:")
        print(results.isnull().sum())
        print()
        print("SUCCESS: Julia backend is working!")

except Exception as e:
    print()
    print("="*70)
    print("ERROR: Julia backend test failed")
    print("="*70)
    print()
    import traceback
    traceback.print_exc()
    print()
    print(f"Error: {e}")
