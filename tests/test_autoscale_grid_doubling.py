"""T-36: integration tests for autoscale toggle in run_search grid path.

Verifies that enabling `autoscale=True` doubles the result-row count and
populates the `Autoscale` column correctly. These run on synthetic data so
they stay fast (under ~10s) but exercise the full grid path including the
preprocess-config doubling block, the per-model scaler skip, and the
result-row write.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectral_predict.search import run_search


@pytest.fixture
def synthetic_regression_data():
    """Tiny synthetic spectral regression dataset (N=24, J=40)."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 24, 40
    wavelengths = np.linspace(900, 1700, n_features)
    # Latent variable target — gives PLS something to fit.
    latent = rng.normal(size=n_samples)
    base = np.sin(np.linspace(0, 4 * np.pi, n_features))
    X = np.outer(latent, base) + rng.normal(scale=0.05, size=(n_samples, n_features))
    y = 2.0 * latent + rng.normal(scale=0.1, size=n_samples)
    X_df = pd.DataFrame(X, columns=[f"{w:.1f}" for w in wavelengths])
    y_s = pd.Series(y, name="target")
    return X_df, y_s


def _run_minimal_search(X, y, autoscale: bool) -> pd.DataFrame:
    """Run a tiny grid search with two preprocessing methods and a single model."""
    return run_search(
        X,
        y,
        task_type="regression",
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True, "snv": True},
        window_sizes=[],
        folds=3,
        cv_strategy="kfold",
        max_n_components=3,
        enable_variable_subsets=False,
        enable_region_subsets=False,
        autoscale=autoscale,
        progress_callback=None,
    )[0]


class TestAutoscaleGridDoubling:
    def test_doubling_produces_2x_rows(self, synthetic_regression_data):
        """autoscale=True should produce 2x the result rows of autoscale=False
        for the spectral preprocessing dimension."""
        X, y = synthetic_regression_data
        results_off = _run_minimal_search(X, y, autoscale=False)
        results_on = _run_minimal_search(X, y, autoscale=True)
        # Same model + same hyperparams, only preprocessing dimension doubles.
        # Sanity: each per-config row is one PLS hyperparam combo, identical between
        # autoscale=on/off; total ratio should be 2x.
        n_off_preps = results_off['Preprocess'].nunique()
        n_on_preps = results_on['Preprocess'].nunique()
        assert n_on_preps == 2 * n_off_preps, (
            f"Expected 2x preprocessing variants, got {n_on_preps} vs {n_off_preps}"
        )

    def test_autoscale_column_populated(self, synthetic_regression_data):
        """`Autoscale` column must exist and split 50/50 when enabled."""
        X, y = synthetic_regression_data
        results = _run_minimal_search(X, y, autoscale=True)
        assert 'Autoscale' in results.columns
        true_count = int(results['Autoscale'].sum())
        false_count = int(len(results) - true_count)
        assert true_count > 0 and false_count > 0
        assert true_count == false_count, (
            f"Autoscale=True/False rows should be balanced, got {true_count}/{false_count}"
        )

    def test_autoscale_disabled_all_false(self, synthetic_regression_data):
        """When autoscale=False, every row must have Autoscale=False."""
        X, y = synthetic_regression_data
        results = _run_minimal_search(X, y, autoscale=False)
        # Either column missing entirely (older write) or every value is False.
        if 'Autoscale' in results.columns:
            assert not results['Autoscale'].any(), (
                "No row should be Autoscale=True when autoscale=False"
            )

    def test_autoscale_name_suffix(self, synthetic_regression_data):
        """Autoscaled configs should carry the '+autoscale' display-name suffix
        and a clean PreprocessBase for validation rebuild."""
        X, y = synthetic_regression_data
        results = _run_minimal_search(X, y, autoscale=True)
        autoscaled = results[results['Autoscale'] == True]  # noqa: E712 — pandas idiom
        assert len(autoscaled) > 0
        names = autoscaled['Preprocess'].unique().tolist()
        assert all('+autoscale' in n for n in names), (
            f"All autoscaled rows should have '+autoscale' suffix, got {names}"
        )
        bases = autoscaled['PreprocessBase'].unique().tolist()
        assert all('+autoscale' not in str(b) for b in bases), (
            f"PreprocessBase should NOT carry '+autoscale' (used by build_preprocessing_pipeline), got {bases}"
        )

    def test_autoscale_with_smoothing_doubling(self, synthetic_regression_data):
        """Smoothing + autoscale doubling stack: each preprocessing variant should
        appear in 4 forms (smoothing × autoscale = 2 × 2). PreprocessBase must
        remain clean (no sg0/autoscale prefix or suffix) so the validation rebuild
        path can reconstruct the pipeline.
        """
        X, y = synthetic_regression_data
        results = run_search(
            X,
            y,
            task_type="regression",
            models_to_test=["PLS"],
            preprocessing_methods={"raw": True, "snv": True},
            window_sizes=[],
            folds=3,
            cv_strategy="kfold",
            max_n_components=3,
            enable_variable_subsets=False,
            enable_region_subsets=False,
            smoothing=True,
            smoothing_window=11,
            smoothing_polyorder=2,
            autoscale=True,
            progress_callback=None,
        )[0]
        # 2 base preps × 2 smoothing × 2 autoscale = 8 unique preprocessing display names.
        n_unique_preps = results['Preprocess'].nunique()
        assert n_unique_preps == 8, (
            f"smoothing × autoscale doubling should yield 8 preprocessing variants, "
            f"got {n_unique_preps}: {sorted(results['Preprocess'].unique().tolist())}"
        )

        autoscaled = results[results['Autoscale'] == True]  # noqa: E712
        # All autoscaled rows should have a clean PreprocessBase that
        # build_preprocessing_pipeline can consume — no '+autoscale', no 'sg0+', no '+sg0+'.
        for base in autoscaled['PreprocessBase'].unique():
            base_s = str(base)
            assert '+autoscale' not in base_s and 'sg0' not in base_s, (
                f"PreprocessBase '{base_s}' contains a display-only suffix"
            )

        # And both smoothing prefix and autoscale suffix should appear together
        # on at least one config (the sg0+snv+autoscale form).
        names = set(autoscaled['Preprocess'].unique().tolist())
        assert any('sg0' in n and '+autoscale' in n for n in names), (
            f"Expected at least one sg0+...+autoscale variant, got {sorted(names)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
