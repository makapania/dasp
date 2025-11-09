#!/usr/bin/env python3
"""Benchmark full model search workflow: Original vs Optimized."""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

print("\n" + "="*70)
print("FULL WORKFLOW BENCHMARK: Original vs Optimized")
print("="*70)

# Create realistic test data
np.random.seed(42)
n_samples = 100
n_features = 500
n_classes = 3

X = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f"{400 + i*2}" for i in range(n_features)]
)
y_regression = pd.Series(np.random.randn(n_samples))
y_classification = pd.Series(np.random.randint(0, n_classes, n_samples))

# Test configuration - smaller for faster benchmark
test_config = {
    'folds': 3,  # Reduced from 5
    'variable_penalty': 3,
    'complexity_penalty': 5,
    'max_n_components': 10,  # Reduced from 24
    'max_iter': 100,  # Reduced from 500
    'models_to_test': ['PLS', 'Ridge'],  # Just 2 models for speed
    'preprocessing_methods': {
        'raw': True,
        'snv': True,
        'sg1': False,  # Disable derivatives for speed
        'sg2': False,
        'deriv_snv': False
    },
    'enable_variable_subsets': False,  # Disable for initial test
    'enable_region_subsets': False
}

print(f"\nTest data: {n_samples} samples × {n_features} features")
print(f"Models: {test_config['models_to_test']}")
print(f"Preprocessing: raw, snv only")
print(f"CV folds: {test_config['folds']}")
print()

# Test ORIGINAL version
print("Testing ORIGINAL version...")
sys.path.insert(0, '/home/user/dasp/src')
from spectral_predict.search import run_search as run_search_original

start = time.perf_counter()
try:
    df_original = run_search_original(
        X, y_regression,
        task_type='regression',
        **test_config
    )
    time_original = time.perf_counter() - start
    success_original = True
    print(f"✓ Original completed: {time_original:.2f}s")
    print(f"  Models tested: {len(df_original)}")
except Exception as e:
    time_original = None
    success_original = False
    print(f"✗ Original failed: {e}")

# Clear imports
if 'spectral_predict' in sys.modules:
    del sys.modules['spectral_predict']
for key in list(sys.modules.keys()):
    if key.startswith('spectral_predict'):
        del sys.modules[key]

# Test OPTIMIZED version
print("\nTesting OPTIMIZED version...")
sys.path.insert(0, '/home/user/dasp-performance-opt/src')
from spectral_predict.search import run_search as run_search_optimized

start = time.perf_counter()
try:
    df_optimized = run_search_optimized(
        X, y_regression,
        task_type='regression',
        **test_config
    )
    time_optimized = time.perf_counter() - start
    success_optimized = True
    print(f"✓ Optimized completed: {time_optimized:.2f}s")
    print(f"  Models tested: {len(df_optimized)}")
except Exception as e:
    time_optimized = None
    success_optimized = False
    print(f"✗ Optimized failed: {e}")

# Compare results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

if success_original and success_optimized:
    print(f"Original time:   {time_original:.2f}s")
    print(f"Optimized time:  {time_optimized:.2f}s")

    if time_optimized < time_original:
        speedup = time_original / time_optimized
        improvement = ((time_original - time_optimized) / time_original) * 100
        print(f"\n✓ FASTER: {speedup:.2f}x speedup ({improvement:.1f}% faster)")
    else:
        slowdown = time_optimized / time_original
        regression = ((time_optimized - time_original) / time_original) * 100
        print(f"\n✗ SLOWER: {slowdown:.2f}x slowdown ({regression:.1f}% slower)")

    # Verify results match
    print("\nVerifying results match...")
    try:
        # Compare top models
        top_orig = df_original.head(5)[['Model', 'Preprocess', 'RMSE', 'R2']].round(4)
        top_opt = df_optimized.head(5)[['Model', 'Preprocess', 'RMSE', 'R2']].round(4)

        if top_orig.equals(top_opt):
            print("✓ Top 5 models match exactly")
        else:
            print("⚠ Top 5 models differ:")
            print("\nOriginal:")
            print(top_orig)
            print("\nOptimized:")
            print(top_opt)
    except Exception as e:
        print(f"⚠ Could not compare results: {e}")

elif success_original:
    print(f"Original completed in {time_original:.2f}s, but optimized failed")
elif success_optimized:
    print(f"Optimized completed in {time_optimized:.2f}s, but original failed")
else:
    print("Both versions failed")

print("\n" + "="*70)
