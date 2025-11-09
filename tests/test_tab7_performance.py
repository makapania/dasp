"""
Performance Benchmarks for Tab 7 Model Development.

This test suite benchmarks performance metrics for Tab 7:
1. Model loading speed
2. Wavelength parsing speed
3. Execution time with different dataset sizes
4. Memory usage
5. GUI responsiveness

Test Categories:
- TestTab7LoadingPerformance: Model and data loading benchmarks
- TestTab7ExecutionPerformance: Model execution benchmarks
- TestTab7ParsingPerformance: Wavelength parsing benchmarks
- TestTab7MemoryUsage: Memory usage monitoring
"""

import numpy as np
import pandas as pd
import pytest
import time
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spectral_predict.search import run_search
from spectral_predict.models import get_model

# Import test utilities
from tab7_test_utils import (
    create_minimal_synthetic_data,
    run_minimal_analysis,
    get_top_result
)


class TestTab7LoadingPerformance:
    """Benchmark model and data loading performance."""

    def test_model_loading_speed_from_results(self):
        """Test speed of loading a model configuration from Results tab."""
        # Create data and run analysis
        X, y = create_minimal_synthetic_data(n_samples=30, n_wavelengths=100)
        results = run_minimal_analysis(X, y, n_folds=3, verbose=False)

        # Get top result
        top_result = get_top_result(results)

        # Benchmark loading configuration
        start_time = time.time()

        # Simulate loading into Tab 7 (configuration parsing)
        model_name = top_result['Model']
        n_vars = int(top_result['n_vars'])
        preprocess = top_result['Preprocess']
        r2 = float(top_result['R2'])

        # Extract hyperparameters
        if model_name == 'PLS':
            n_components = int(top_result['LVs'])
        elif model_name == 'Ridge':
            alpha = float(top_result['Alpha'])

        elapsed = time.time() - start_time

        print(f"\nModel loading time: {elapsed*1000:.2f} ms")
        assert elapsed < 0.1, f"Model loading should be < 100ms, got {elapsed*1000:.0f}ms"

    def test_wavelength_parsing_speed_small_list(self):
        """Test wavelength parsing speed with small list (~50 wavelengths)."""
        from spectral_predict_gui_optimized import SpectralPredictApp
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        try:
            app = SpectralPredictApp(root)

            # Create wavelength list
            available_wl = np.linspace(1500, 2500, 500)
            wl_spec = ", ".join([f"{w:.1f}" for w in available_wl[:50]])

            # Benchmark parsing
            start_time = time.time()
            parsed = app._parse_wavelength_spec(wl_spec, available_wl)
            elapsed = time.time() - start_time

            print(f"\nSmall list parsing time: {elapsed*1000:.2f} ms ({len(parsed)} wavelengths)")
            assert elapsed < 0.1, f"Parsing should be < 100ms, got {elapsed*1000:.0f}ms"
        finally:
            root.destroy()

    def test_wavelength_parsing_speed_large_list(self):
        """Test wavelength parsing speed with large list (~500 wavelengths)."""
        from spectral_predict_gui_optimized import SpectralPredictApp
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        try:
            app = SpectralPredictApp(root)

            # Create large wavelength list
            available_wl = np.linspace(1500, 2500, 1000)
            wl_spec = ", ".join([f"{w:.1f}" for w in available_wl[:500]])

            # Benchmark parsing
            start_time = time.time()
            parsed = app._parse_wavelength_spec(wl_spec, available_wl)
            elapsed = time.time() - start_time

            print(f"\nLarge list parsing time: {elapsed*1000:.2f} ms ({len(parsed)} wavelengths)")
            assert elapsed < 0.5, f"Parsing should be < 500ms, got {elapsed*1000:.0f}ms"
        finally:
            root.destroy()

    def test_wavelength_range_parsing_speed(self):
        """Test wavelength range parsing speed (e.g., '1500-2500')."""
        from spectral_predict_gui_optimized import SpectralPredictApp
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        try:
            app = SpectralPredictApp(root)

            available_wl = np.linspace(1500, 2500, 1000)
            wl_spec = "1500-2500"

            # Benchmark parsing
            start_time = time.time()
            parsed = app._parse_wavelength_spec(wl_spec, available_wl)
            elapsed = time.time() - start_time

            print(f"\nRange parsing time: {elapsed*1000:.2f} ms ({len(parsed)} wavelengths)")
            assert elapsed < 0.2, f"Range parsing should be < 200ms, got {elapsed*1000:.0f}ms"
        finally:
            root.destroy()


class TestTab7ExecutionPerformance:
    """Benchmark model execution performance with different dataset sizes."""

    def test_execution_time_small_dataset(self):
        """Test execution time with small dataset (10 samples, 100 wavelengths)."""
        X, y = create_minimal_synthetic_data(n_samples=10, n_wavelengths=100, seed=42)

        # Train PLS model
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold, cross_val_score

        model = PLSRegression(n_components=5)
        cv = KFold(n_splits=3, shuffle=False)

        # Benchmark execution
        start_time = time.time()
        scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='r2')
        elapsed = time.time() - start_time

        print(f"\nSmall dataset execution time: {elapsed:.2f} s")
        print(f"  Mean R²: {scores.mean():.4f}")
        assert elapsed < 5.0, f"Small dataset should run in < 5s, got {elapsed:.1f}s"

    def test_execution_time_medium_dataset(self):
        """Test execution time with medium dataset (30 samples, 200 wavelengths)."""
        X, y = create_minimal_synthetic_data(n_samples=30, n_wavelengths=200, seed=42)

        # Train PLS model
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold, cross_val_score

        model = PLSRegression(n_components=10)
        cv = KFold(n_splits=5, shuffle=False)

        # Benchmark execution
        start_time = time.time()
        scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='r2')
        elapsed = time.time() - start_time

        print(f"\nMedium dataset execution time: {elapsed:.2f} s")
        print(f"  Mean R²: {scores.mean():.4f}")
        assert elapsed < 10.0, f"Medium dataset should run in < 10s, got {elapsed:.1f}s"

    def test_execution_time_large_dataset(self):
        """Test execution time with large dataset (50 samples, 500 wavelengths)."""
        X, y = create_minimal_synthetic_data(n_samples=50, n_wavelengths=500, seed=42)

        # Train PLS model
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold, cross_val_score

        model = PLSRegression(n_components=15)
        cv = KFold(n_splits=5, shuffle=False)

        # Benchmark execution
        start_time = time.time()
        scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='r2')
        elapsed = time.time() - start_time

        print(f"\nLarge dataset execution time: {elapsed:.2f} s")
        print(f"  Mean R²: {scores.mean():.4f}")
        assert elapsed < 30.0, f"Large dataset should run in < 30s, got {elapsed:.1f}s"

    def test_execution_time_with_preprocessing(self):
        """Test execution time with derivative preprocessing."""
        X, y = create_minimal_synthetic_data(n_samples=30, n_wavelengths=200, seed=42)

        # Build preprocessing pipeline
        from spectral_predict.preprocess import build_preprocessing_pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold, cross_val_score

        prep_steps = build_preprocessing_pipeline('deriv', deriv=1, window=11, polyorder=2)
        model = PLSRegression(n_components=10)
        prep_steps.append(('model', model))
        pipeline = Pipeline(prep_steps)

        cv = KFold(n_splits=5, shuffle=False)

        # Benchmark execution with preprocessing
        start_time = time.time()
        scores = cross_val_score(pipeline, X.values, y.values, cv=cv, scoring='r2')
        elapsed = time.time() - start_time

        print(f"\nExecution time with preprocessing: {elapsed:.2f} s")
        print(f"  Mean R²: {scores.mean():.4f}")
        assert elapsed < 15.0, f"With preprocessing should run in < 15s, got {elapsed:.1f}s"


class TestTab7ModelComparisonPerformance:
    """Benchmark performance across different model types."""

    def test_pls_performance(self):
        """Benchmark PLS model performance."""
        X, y = create_minimal_synthetic_data(n_samples=30, n_wavelengths=150)

        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_score, KFold

        model = PLSRegression(n_components=10)
        cv = KFold(n_splits=5, shuffle=False)

        start_time = time.time()
        scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='r2')
        elapsed = time.time() - start_time

        print(f"\nPLS performance: {elapsed:.2f} s (R²={scores.mean():.4f})")
        return elapsed

    def test_ridge_performance(self):
        """Benchmark Ridge model performance."""
        X, y = create_minimal_synthetic_data(n_samples=30, n_wavelengths=150)

        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score, KFold

        model = Ridge(alpha=1.0)
        cv = KFold(n_splits=5, shuffle=False)

        start_time = time.time()
        scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='r2')
        elapsed = time.time() - start_time

        print(f"\nRidge performance: {elapsed:.2f} s (R²={scores.mean():.4f})")
        return elapsed

    def test_randomforest_performance(self):
        """Benchmark RandomForest model performance."""
        X, y = create_minimal_synthetic_data(n_samples=30, n_wavelengths=150)

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score, KFold

        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        cv = KFold(n_splits=5, shuffle=False)

        start_time = time.time()
        scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='r2')
        elapsed = time.time() - start_time

        print(f"\nRandomForest performance: {elapsed:.2f} s (R²={scores.mean():.4f})")
        return elapsed

    def test_model_performance_comparison(self):
        """Compare performance across all models."""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)

        pls_time = self.test_pls_performance()
        ridge_time = self.test_ridge_performance()
        rf_time = self.test_randomforest_performance()

        print("\n" + "="*60)
        print("SUMMARY:")
        print(f"  PLS:          {pls_time:.2f} s")
        print(f"  Ridge:        {ridge_time:.2f} s")
        print(f"  RandomForest: {rf_time:.2f} s")
        print("="*60)


class TestTab7MemoryUsage:
    """Monitor memory usage during Tab 7 operations."""

    def test_memory_usage_small_dataset(self):
        """Monitor memory usage with small dataset."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create and process data
        X, y = create_minimal_synthetic_data(n_samples=30, n_wavelengths=200)

        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=10)
        model.fit(X.values, y.values)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        print(f"\nMemory usage (small dataset):")
        print(f"  Before: {mem_before:.1f} MB")
        print(f"  After:  {mem_after:.1f} MB")
        print(f"  Used:   {mem_used:.1f} MB")

        assert mem_used < 100, f"Memory usage should be < 100MB, got {mem_used:.0f}MB"

    def test_memory_usage_large_dataset(self):
        """Monitor memory usage with large dataset."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create large dataset
        X, y = create_minimal_synthetic_data(n_samples=100, n_wavelengths=1000)

        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=20)
        model.fit(X.values, y.values)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        print(f"\nMemory usage (large dataset):")
        print(f"  Before: {mem_before:.1f} MB")
        print(f"  After:  {mem_after:.1f} MB")
        print(f"  Used:   {mem_used:.1f} MB")

        assert mem_used < 500, f"Memory usage should be < 500MB, got {mem_used:.0f}MB"


class TestTab7PreprocessingPerformance:
    """Benchmark preprocessing performance."""

    def test_raw_preprocessing_performance(self):
        """Benchmark raw (no preprocessing) performance."""
        X, y = create_minimal_synthetic_data(n_samples=50, n_wavelengths=200)

        start_time = time.time()
        X_preprocessed = X.values.copy()  # No preprocessing
        elapsed = time.time() - start_time

        print(f"\nRaw preprocessing: {elapsed*1000:.2f} ms")
        assert elapsed < 0.1, "Raw preprocessing should be instant"

    def test_derivative_preprocessing_performance(self):
        """Benchmark derivative preprocessing performance."""
        X, y = create_minimal_synthetic_data(n_samples=50, n_wavelengths=200)

        from spectral_predict.preprocess import build_preprocessing_pipeline
        from sklearn.pipeline import Pipeline

        prep_steps = build_preprocessing_pipeline('deriv', deriv=1, window=11, polyorder=2)
        pipeline = Pipeline(prep_steps)

        start_time = time.time()
        X_preprocessed = pipeline.fit_transform(X.values)
        elapsed = time.time() - start_time

        print(f"\nDerivative preprocessing: {elapsed*1000:.2f} ms")
        assert elapsed < 1.0, f"Derivative preprocessing should be < 1s, got {elapsed:.2f}s"

    def test_snv_preprocessing_performance(self):
        """Benchmark SNV preprocessing performance."""
        X, y = create_minimal_synthetic_data(n_samples=50, n_wavelengths=200)

        from spectral_predict.preprocess import build_preprocessing_pipeline
        from sklearn.pipeline import Pipeline

        prep_steps = build_preprocessing_pipeline('snv', deriv=0, window=11, polyorder=2)
        pipeline = Pipeline(prep_steps)

        start_time = time.time()
        X_preprocessed = pipeline.fit_transform(X.values)
        elapsed = time.time() - start_time

        print(f"\nSNV preprocessing: {elapsed*1000:.2f} ms")
        assert elapsed < 0.5, f"SNV preprocessing should be < 500ms, got {elapsed:.2f}s"

    def test_combined_preprocessing_performance(self):
        """Benchmark combined preprocessing (SNV + derivative) performance."""
        X, y = create_minimal_synthetic_data(n_samples=50, n_wavelengths=200)

        from spectral_predict.preprocess import build_preprocessing_pipeline
        from sklearn.pipeline import Pipeline

        prep_steps = build_preprocessing_pipeline('snv_deriv', deriv=1, window=11, polyorder=2)
        pipeline = Pipeline(prep_steps)

        start_time = time.time()
        X_preprocessed = pipeline.fit_transform(X.values)
        elapsed = time.time() - start_time

        print(f"\nCombined preprocessing: {elapsed*1000:.2f} ms")
        assert elapsed < 1.0, f"Combined preprocessing should be < 1s, got {elapsed:.2f}s"


class TestTab7ScalabilityBenchmarks:
    """Test scalability with increasing data sizes."""

    def test_scalability_sample_count(self):
        """Test how execution time scales with sample count."""
        print("\n" + "="*60)
        print("SCALABILITY: SAMPLE COUNT")
        print("="*60)

        sample_counts = [10, 20, 30, 50, 100]
        times = []

        for n_samples in sample_counts:
            X, y = create_minimal_synthetic_data(n_samples=n_samples, n_wavelengths=150)

            from sklearn.cross_decomposition import PLSRegression
            from sklearn.model_selection import cross_val_score, KFold

            model = PLSRegression(n_components=10)
            cv = KFold(n_splits=min(5, n_samples), shuffle=False)

            start_time = time.time()
            scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='r2')
            elapsed = time.time() - start_time

            times.append(elapsed)
            print(f"  {n_samples:3d} samples: {elapsed:.2f} s")

        print("="*60)

    def test_scalability_wavelength_count(self):
        """Test how execution time scales with wavelength count."""
        print("\n" + "="*60)
        print("SCALABILITY: WAVELENGTH COUNT")
        print("="*60)

        wavelength_counts = [50, 100, 200, 500, 1000]
        times = []

        for n_wavelengths in wavelength_counts:
            X, y = create_minimal_synthetic_data(n_samples=30, n_wavelengths=n_wavelengths)

            from sklearn.cross_decomposition import PLSRegression
            from sklearn.model_selection import cross_val_score, KFold

            model = PLSRegression(n_components=10)
            cv = KFold(n_splits=5, shuffle=False)

            start_time = time.time()
            scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='r2')
            elapsed = time.time() - start_time

            times.append(elapsed)
            print(f"  {n_wavelengths:4d} wavelengths: {elapsed:.2f} s")

        print("="*60)


if __name__ == '__main__':
    # Run performance tests
    pytest.main([__file__, '-v', '--tb=short', '-s'])
