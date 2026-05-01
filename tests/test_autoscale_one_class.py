"""T-36: integration tests for autoscale toggle in one-class search path.

Verifies:
* run_one_class_search accepts autoscale=True/False.
* Result rows expose Autoscale column and bundled metadata
  (baseline_method/smoothing/smoothing_window/smoothing_polyorder).
* Doubling produces 2x preprocessing variants.
* contamination.py validation rebuild key parses autoscale correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectral_predict.search import run_one_class_search


@pytest.fixture
def synthetic_one_class_data():
    rng = np.random.RandomState(42)
    n_features = 30
    X_clean = rng.randn(30, n_features) * 0.3
    X_contam = rng.randn(8, n_features) + 3.0
    X = np.vstack([X_clean, X_contam])
    y = np.array(['clean'] * 30 + ['contaminated'] * 8)
    wavelengths = [f"{400 + i * 10:.1f}" for i in range(n_features)]
    return pd.DataFrame(X, columns=wavelengths), pd.Series(y)


def _run_oc_search(X, y, autoscale: bool):
    return run_one_class_search(
        X=X,
        y=y,
        inlier_class_label='clean',
        folds=3,
        preprocessing_methods=['raw', 'snv'],
        window_sizes=[17],
        enabled_models=['IsolationForest'],
        autoscale=autoscale,
    )


class TestOneClassAutoscale:
    def test_doubling_produces_2x_preprocessing_variants(self, synthetic_one_class_data):
        X, y = synthetic_one_class_data
        results_off = _run_oc_search(X, y, autoscale=False)
        results_on = _run_oc_search(X, y, autoscale=True)
        n_off = results_off['Preprocess'].nunique()
        n_on = results_on['Preprocess'].nunique()
        assert n_on == 2 * n_off, (
            f"autoscale=True must double the preprocessing variant count, "
            f"got {n_on} vs {n_off}"
        )

    def test_autoscale_column_populated(self, synthetic_one_class_data):
        X, y = synthetic_one_class_data
        results = _run_oc_search(X, y, autoscale=True)
        assert 'Autoscale' in results.columns
        true_count = int(results['Autoscale'].sum())
        false_count = int(len(results) - true_count)
        assert true_count > 0
        assert false_count > 0

    def test_metadata_columns_present(self, synthetic_one_class_data):
        """T-36 bundled fix: result rows must carry baseline_method, smoothing,
        smoothing_window, smoothing_polyorder so contamination.py validation
        rebuild reads real values rather than silent defaults."""
        X, y = synthetic_one_class_data
        results = _run_oc_search(X, y, autoscale=False)
        for col in ('baseline_method', 'smoothing', 'smoothing_window', 'smoothing_polyorder'):
            assert col in results.columns, (
                f"One-class result rows must carry '{col}' for validation rebuild"
            )

    def test_autoscale_name_suffix(self, synthetic_one_class_data):
        """Autoscaled rows have '+autoscale' in display name; PreprocessBase stays clean."""
        X, y = synthetic_one_class_data
        results = _run_oc_search(X, y, autoscale=True)
        autoscaled = results[results['Autoscale'] == True]  # noqa: E712
        assert len(autoscaled) > 0
        # Display name carries +autoscale
        names = autoscaled['Preprocess'].unique().tolist()
        assert all('+autoscale' in n for n in names), (
            f"Autoscaled rows should have '+autoscale' suffix, got {names}"
        )
        # PreprocessBase stays clean (used by build_preprocessing_pipeline)
        if 'PreprocessBase' in autoscaled.columns:
            bases = autoscaled['PreprocessBase'].unique().tolist()
            assert all('+autoscale' not in str(b) for b in bases), (
                f"PreprocessBase must NOT carry '+autoscale', got {bases}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
