"""
Test Julia backend with mixed model types (PLS + Ridge) to verify
the DataFrame key normalization fix works.
"""

import numpy as np
import pandas as pd
from spectral_predict_julia_bridge import run_search_julia

# Create test data
np.random.seed(42)
n_samples = 50
n_features = 30

X = pd.DataFrame(
    np.random.rand(n_samples, n_features),
    columns=[str(w) for w in range(400, 400 + n_features)]
)
y = pd.Series(np.random.rand(n_samples))

print("="*70)
print("Testing Julia Backend with Mixed Models")
print("="*70)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print()
print("Testing: PLS (no alpha) + Ridge (has alpha)")
print("This previously caused KeyError: 'alpha'")
print()

try:
    results = run_search_julia(
        X, y,
        task_type='regression',
        models_to_test=['PLS', 'Ridge'],
        preprocessing_methods={'raw': True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        folds=3,
        progress_callback=None
    )

    print()
    print("="*70)
    print("SUCCESS: Mixed model test passed!")
    print("="*70)
    print()
    print(f"Total results: {len(results)}")
    print()
    print("Results by model:")
    print(results.groupby('Model').size())
    print()
    print("Column names:")
    print(list(results.columns))
    print()
    print("First few rows:")
    print(results.head())

except Exception as e:
    print()
    print("="*70)
    print("FAILED: Mixed model test failed")
    print("="*70)
    print()
    import traceback
    traceback.print_exc()
    print()
    print(f"Error: {e}")
