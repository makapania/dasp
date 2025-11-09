#!/usr/bin/env python3
"""Benchmark original vs optimized versions of performance-critical functions."""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

# Add both versions to path
sys.path.insert(0, '/home/user/dasp/src')
sys.path.insert(1, '/home/user/dasp-performance-opt/src')

def benchmark_vip_scores():
    """Benchmark VIP score calculation."""
    print("\n" + "="*70)
    print("BENCHMARK 1: VIP Score Calculation")
    print("="*70)

    # Create test data
    n_samples = 100
    n_features = 1000
    n_components = 10

    np.random.seed(42)
    W = np.random.randn(n_features, n_components)
    T = np.random.randn(n_samples, n_components)
    y = np.random.randn(n_samples)

    # Original version (loop-based)
    def vip_original(W, T, y):
        y = np.asarray(y).reshape(-1, 1)
        ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)
        ssy_total = np.sum(ssy_comp)

        n_features = W.shape[0]
        vip_scores = np.zeros(n_features)
        for i in range(n_features):
            weight = np.sum((W[i, :] ** 2) * ssy_comp)
            vip_scores[i] = np.sqrt(n_features * weight / ssy_total)
        return vip_scores

    # Optimized version (vectorized)
    def vip_optimized(W, T, y):
        y = np.asarray(y).reshape(-1, 1)
        ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)
        ssy_total = np.sum(ssy_comp)

        n_features = W.shape[0]
        weight = np.sum((W ** 2) * ssy_comp, axis=1)
        vip_scores = np.sqrt(n_features * weight / ssy_total)
        return vip_scores

    # Warm-up
    vip_original(W, T, y)
    vip_optimized(W, T, y)

    # Benchmark original
    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_orig = vip_original(W, T, y)
    time_orig = time.perf_counter() - start

    # Benchmark optimized
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_opt = vip_optimized(W, T, y)
    time_opt = time.perf_counter() - start

    # Verify results are identical
    np.testing.assert_allclose(result_orig, result_opt, rtol=1e-10)

    print(f"Data: {n_features} features, {n_components} components")
    print(f"Iterations: {n_iterations}")
    print(f"Original (loop):     {time_orig:.4f}s ({time_orig/n_iterations*1000:.2f}ms per call)")
    print(f"Optimized (vector):  {time_opt:.4f}s ({time_opt/n_iterations*1000:.2f}ms per call)")
    print(f"Speedup: {time_orig/time_opt:.2f}x")
    print(f"Results identical: ✓")

    return time_orig / time_opt


def benchmark_region_correlations():
    """Benchmark region correlation calculation."""
    print("\n" + "="*70)
    print("BENCHMARK 2: Region Correlation Calculation")
    print("="*70)

    # Create test data
    n_samples = 100
    n_features = 50  # Typical region size

    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Original version (loop with pearsonr)
    def corr_original(X, y):
        from scipy.stats import pearsonr
        correlations = []
        for idx in range(X.shape[1]):
            try:
                corr, _ = pearsonr(X[:, idx], y)
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            except:
                pass
        return correlations

    # Optimized version (vectorized)
    def corr_optimized(X, y):
        try:
            combined = np.column_stack([X, y.ravel()])
            corr_matrix = np.corrcoef(combined, rowvar=False)
            feature_y_corrs = corr_matrix[:-1, -1]
            correlations = np.abs(feature_y_corrs)
            correlations = correlations[~np.isnan(correlations)].tolist()
        except:
            correlations = []
        return correlations

    # Warm-up
    corr_original(X, y)
    corr_optimized(X, y)

    # Benchmark original
    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_orig = corr_original(X, y)
    time_orig = time.perf_counter() - start

    # Benchmark optimized
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_opt = corr_optimized(X, y)
    time_opt = time.perf_counter() - start

    # Verify results are close (correlation can have slight numerical differences)
    np.testing.assert_allclose(result_orig, result_opt, rtol=1e-6)

    print(f"Data: {n_samples} samples, {n_features} features")
    print(f"Iterations: {n_iterations}")
    print(f"Original (loop):     {time_orig:.4f}s ({time_orig/n_iterations*1000:.2f}ms per call)")
    print(f"Optimized (vector):  {time_opt:.4f}s ({time_opt/n_iterations*1000:.2f}ms per call)")
    print(f"Speedup: {time_orig/time_opt:.2f}x")
    print(f"Results identical: ✓")

    return time_orig / time_opt


def benchmark_iterrows():
    """Benchmark iterrows vs itertuples."""
    print("\n" + "="*70)
    print("BENCHMARK 3: DataFrame Iteration (iterrows vs itertuples)")
    print("="*70)

    # Create test data
    df = pd.DataFrame({
        'Rank': range(1, 6),
        'Model': ['PLS', 'Ridge', 'Lasso', 'RF', 'MLP'],
        'RMSE': np.random.rand(5),
        'R2': np.random.rand(5),
        'CompositeScore': np.random.rand(5)
    })

    # Original version (iterrows)
    def iterate_rows(df):
        results = []
        for idx, row in df.iterrows():
            results.append(f"Rank {row['Rank']}: {row['Model']} - RMSE: {row['RMSE']:.4f}")
        return results

    # Optimized version (itertuples)
    def iterate_tuples(df):
        results = []
        for row in df.itertuples(index=False):
            results.append(f"Rank {row.Rank}: {row.Model} - RMSE: {row.RMSE:.4f}")
        return results

    # Warm-up
    iterate_rows(df)
    iterate_tuples(df)

    # Benchmark original
    n_iterations = 10000
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_orig = iterate_rows(df)
    time_orig = time.perf_counter() - start

    # Benchmark optimized
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_opt = iterate_tuples(df)
    time_opt = time.perf_counter() - start

    # Verify results are identical
    assert result_orig == result_opt

    print(f"Data: {len(df)} rows")
    print(f"Iterations: {n_iterations}")
    print(f"Original (iterrows):  {time_orig:.4f}s ({time_orig/n_iterations*1000:.3f}ms per call)")
    print(f"Optimized (itertuples): {time_opt:.4f}s ({time_opt/n_iterations*1000:.3f}ms per call)")
    print(f"Speedup: {time_orig/time_opt:.2f}x")
    print(f"Results identical: ✓")

    return time_orig / time_opt


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK: Original vs Optimized Code")
    print("="*70)

    speedups = []

    try:
        speedup = benchmark_vip_scores()
        speedups.append(('VIP Scores', speedup))
    except Exception as e:
        print(f"VIP benchmark failed: {e}")

    try:
        speedup = benchmark_region_correlations()
        speedups.append(('Region Correlations', speedup))
    except Exception as e:
        print(f"Region correlation benchmark failed: {e}")

    try:
        speedup = benchmark_iterrows()
        speedups.append(('DataFrame Iteration', speedup))
    except Exception as e:
        print(f"DataFrame iteration benchmark failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, speedup in speedups:
        status = "✓ FASTER" if speedup > 1 else "✗ SLOWER"
        print(f"{name:30s}: {speedup:6.2f}x {status}")

    if speedups:
        avg_speedup = sum(s for _, s in speedups) / len(speedups)
        print(f"\nAverage speedup: {avg_speedup:.2f}x")

    print("\nNote: This only tests individual functions in isolation.")
    print("Real-world speedup depends on overall workflow and data size.")
