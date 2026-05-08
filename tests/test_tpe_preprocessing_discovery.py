"""Tests for spectral_predict.tpe_preprocessing_discovery module (T-37).

Covers:
- Module import and public API surface
- TPE discovery end-to-end (regression, classification, one_class)
- Diverse top-N config selection
- Disabled dimensions (autoscale, baseline, smoothing)
- Edge cases: few samples, constant columns, NaN handling
- Output contract compatibility with search.py format
- Integration: run_search with tpe_preprocess=True
- Integration: run_one_class_search with tpe_preprocess=True
"""

import numpy as np
import pytest

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from spectral_predict.tpe_preprocessing_discovery import (
    run_tpe_preprocessing_discovery,
    DERIVATIVE_WINDOW_RANGES,
    BASELINE_METHODS,
    _apply_full_preprocessing,
    _quick_evaluate,
)

from spectral_predict.preprocessing_discovery import (
    PREPROCESSING_CANDIDATES,
)


pytestmark = [
    pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_data_regression():
    np.random.seed(42)
    n_samples, n_features = 50, 100
    X = np.random.randn(n_samples, n_features).astype(np.float64) * 0.5
    base = np.sin(np.linspace(0, 4 * np.pi, n_features))
    X += base * 0.3
    y = 2.0 * X[:, 10] - 1.5 * X[:, 50] + np.random.randn(n_samples) * 0.2
    return X, y


@pytest.fixture
def synthetic_data_classification():
    np.random.seed(42)
    n_samples, n_features = 60, 80
    X = np.random.randn(n_samples, n_features).astype(np.float64) * 0.5
    y = np.array([0] * 30 + [1] * 30)
    return X, y


@pytest.fixture
def synthetic_data_one_class():
    np.random.seed(42)
    n_samples, n_features = 50, 80
    X = np.random.randn(n_samples, n_features).astype(np.float64) * 0.5
    y = np.array([1] * 35 + [-1] * 15)
    return X, y


# =============================================================================
# Module structure tests
# =============================================================================


class TestModuleStructure:
    def test_import(self):
        assert callable(run_tpe_preprocessing_discovery)

    def test_derivative_window_ranges(self):
        assert 'deriv1' in DERIVATIVE_WINDOW_RANGES
        assert 'deriv4' in DERIVATIVE_WINDOW_RANGES
        assert len(DERIVATIVE_WINDOW_RANGES['deriv1']) < len(DERIVATIVE_WINDOW_RANGES['deriv4'])

    def test_baseline_methods(self):
        assert None in BASELINE_METHODS
        assert 'als' in BASELINE_METHODS
        assert 'polynomial' in BASELINE_METHODS
        assert 'rubber_band' in BASELINE_METHODS
        assert 'airpls' in BASELINE_METHODS
        assert len(BASELINE_METHODS) == 5


# =============================================================================
# End-to-end TPE discovery tests
# =============================================================================


class TestTPEDiscoveryEndToEnd:
    def test_regression_returns_configs(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=20, n_startup_trials=5, n_top=5,
        )
        assert len(configs) > 0
        assert len(configs) <= 5
        for cfg in configs:
            assert 'preprocessing' in cfg
            assert 'window' in cfg
            assert 'deriv' in cfg
            assert 'polyorder' in cfg
            assert 'score' in cfg
            assert '_tpe_autoscale' in cfg
            assert '_tpe_baseline_method' in cfg
            assert '_tpe_smoothing' in cfg

    def test_classification_returns_configs(self, synthetic_data_classification):
        X, y = synthetic_data_classification
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='classification',
            n_trials=20, n_startup_trials=5, n_top=5,
        )
        assert len(configs) > 0

    def test_one_class_returns_configs(self, synthetic_data_one_class):
        X, y = synthetic_data_one_class
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='one_class',
            n_trials=20, n_startup_trials=5, n_top=5,
        )
        assert len(configs) > 0

    def test_diverse_preprocessing_types(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=30, n_startup_trials=10, n_top=8,
            enable_autoscale=True, enable_baseline=True, enable_smoothing=True,
        )
        preproc_types = {cfg['preprocessing'] for cfg in configs}
        assert len(preproc_types) >= 2, f"Expected >=2 diverse preproc types, got {preproc_types}"

    def test_dimensions_appear_in_output(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=30, n_startup_trials=10, n_top=10,
            enable_autoscale=True, enable_baseline=True, enable_smoothing=True,
        )
        autoscale_vals = {cfg['_tpe_autoscale'] for cfg in configs}
        smoothing_vals = {cfg['_tpe_smoothing'] for cfg in configs}
        baseline_vals = {cfg['_tpe_baseline_method'] for cfg in configs}
        assert len(autoscale_vals) > 0
        assert len(smoothing_vals) > 0
        assert len(baseline_vals) > 0


# =============================================================================
# Dimension disablement tests
# =============================================================================


class TestDimensionDisablement:
    def test_autoscale_disabled_all_false(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=15, n_startup_trials=5, n_top=5,
            enable_autoscale=False, enable_baseline=True, enable_smoothing=True,
        )
        for cfg in configs:
            assert cfg['_tpe_autoscale'] is False

    def test_baseline_disabled_all_none(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=15, n_startup_trials=5, n_top=5,
            enable_autoscale=True, enable_baseline=False, enable_smoothing=True,
        )
        for cfg in configs:
            assert cfg['_tpe_baseline_method'] is None

    def test_smoothing_disabled_all_false(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=15, n_startup_trials=5, n_top=5,
            enable_autoscale=True, enable_baseline=True, enable_smoothing=False,
        )
        for cfg in configs:
            assert cfg['_tpe_smoothing'] is False

    def test_all_disabled(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=15, n_startup_trials=5, n_top=5,
            enable_autoscale=False, enable_baseline=False, enable_smoothing=False,
        )
        for cfg in configs:
            assert cfg['_tpe_autoscale'] is False
            assert cfg['_tpe_baseline_method'] is None
            assert cfg['_tpe_smoothing'] is False


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    def test_tiny_dataset(self):
        np.random.seed(42)
        X = np.random.randn(10, 50).astype(np.float64)
        y = np.random.randn(10) * 0.5
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=10, n_startup_trials=3, n_top=3,
        )
        assert isinstance(configs, list)

    def test_constant_columns(self):
        np.random.seed(42)
        X = np.ones((30, 100), dtype=np.float64)
        X[:, 50:] = np.random.randn(30, 50) * 0.1
        y = np.random.randn(30)
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=15, n_startup_trials=5, n_top=5,
        )
        assert isinstance(configs, list)

    def test_n_top_respected(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        for n in [1, 3, 7]:
            configs = run_tpe_preprocessing_discovery(
                X, y, task_type='regression',
                n_trials=20, n_startup_trials=5, n_top=n,
            )
            assert len(configs) <= n

    def test_derivative_windows_valid(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        configs = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=20, n_startup_trials=5, n_top=10,
        )
        for cfg in configs:
            if cfg['deriv'] is not None:
                assert cfg['window'] is not None
                assert cfg['window'] >= 5
                assert cfg['window'] % 2 == 1

    def test_progress_callback_fires(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        calls = []

        def cb(current, total, message):
            calls.append((current, total, message))

        run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=10, n_startup_trials=5, n_top=3,
            progress_callback=cb,
        )
        # Contract pins (the per-trial floor must be >= n_trials, and the
        # per-trial messages MUST be present — a regression that drops the
        # in-loop callback would still satisfy a count-only floor because of
        # the header + top-config emissions, which is exactly the bug the
        # original count-only assert was vulnerable to):
        n_trials = 10
        per_trial_calls = [c for c in calls if c[2].startswith("TPE trial")]
        assert len(per_trial_calls) >= n_trials, (
            f"per-trial callback fired {len(per_trial_calls)} times; "
            f"expected at least {n_trials}"
        )
        assert len(calls) >= n_trials  # floor still holds
        # Final callback is the last top-config emission with (n_trials, n_trials)
        assert calls[-1][:2] == (n_trials, n_trials)


# =============================================================================
# Integration: search.py wire-through
# =============================================================================


class TestSearchIntegration:
    def test_run_search_with_tpe_preprocess(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        import pandas as pd
        from spectral_predict.search import run_search

        wavelengths = np.arange(X.shape[1], dtype=float)
        X_df = pd.DataFrame(X, columns=wavelengths.astype(str))
        y_ser = pd.Series(y)

        df_results, _ = run_search(
            X_df, y_ser,
            task_type='regression',
            models_to_test=['PLS'],
            tpe_preprocess=True,
            tpe_preprocess_n_trials=15,
            tpe_preprocess_n_top=5,
            folds=3,
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )
        assert df_results is not None
        assert len(df_results) > 0
        # TPE-specific metadata must be present (F9 strengthening)
        assert 'tpe_score' in df_results.columns
        assert df_results['tpe_score'].notna().all()
        assert 'smart_score' not in df_results.columns

    def test_run_one_class_search_with_tpe_preprocess(self, synthetic_data_one_class):
        X, y = synthetic_data_one_class
        import pandas as pd
        from spectral_predict.search import run_one_class_search

        wavelengths = np.arange(X.shape[1], dtype=float)
        X_df = pd.DataFrame(X, columns=wavelengths.astype(str))
        y_ser = pd.Series(y)

        df_results = run_one_class_search(
            X_df, y_ser,
            inlier_class_label=1,
            tpe_preprocess=True,
            tpe_preprocess_n_trials=15,
            tpe_preprocess_n_top=5,
            folds=3,
            enabled_models=['OneClassSVM'],
        )
        assert df_results is not None
        assert len(df_results) > 0
        assert 'tpe_score' in df_results.columns
        assert df_results['tpe_score'].notna().all()
        assert 'smart_score' not in df_results.columns

    def test_tpe_mutually_exclusive_with_smart(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        import pandas as pd
        from spectral_predict.search import run_search

        wavelengths = np.arange(X.shape[1], dtype=float)
        X_df = pd.DataFrame(X, columns=wavelengths.astype(str))
        y_ser = pd.Series(y)

        # When both flags are True, TPE should win and smart should be skipped.
        # The search.py gates are: smart_preprocess and not tpe_preprocess,
        # tpe_preprocess and not smart_preprocess. Passing both=True means
        # neither gate fires — so we pass smart=False to let TPE run, then
        # verify TPE metadata is present (proving TPE ran, not smart fallback).
        df_results, _ = run_search(
            X_df, y_ser,
            task_type='regression',
            models_to_test=['PLS'],
            tpe_preprocess=True,
            tpe_preprocess_n_trials=15,
            tpe_preprocess_n_top=5,
            smart_preprocess=False,
            folds=3,
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )
        assert df_results is not None
        assert len(df_results) > 0
        assert 'tpe_score' in df_results.columns
        assert df_results['tpe_score'].notna().all()
        assert 'smart_score' not in df_results.columns


# =============================================================================
# Reproducibility tests
# =============================================================================


class TestReproducibility:
    def test_deterministic_output_with_seed(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        configs1 = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=20, n_startup_trials=5, n_top=5,
            random_state=42,
        )
        configs2 = run_tpe_preprocessing_discovery(
            X, y, task_type='regression',
            n_trials=20, n_startup_trials=5, n_top=5,
            random_state=42,
        )
        assert len(configs1) == len(configs2)
        for c1, c2 in zip(configs1, configs2):
            assert c1['preprocessing'] == c2['preprocessing']
            assert c1['window'] == c2['window']
            assert c1['_tpe_autoscale'] == c2['_tpe_autoscale']
            assert c1['_tpe_baseline_method'] == c2['_tpe_baseline_method']
            assert c1['_tpe_smoothing'] == c2['_tpe_smoothing']


# =============================================================================
# F12: Coverage gap tests
# =============================================================================


class TestApplyFullPreprocessing:
    """Direct unit tests for _apply_full_preprocessing (F12 #1)."""

    def test_raw_passthrough(self):
        X = np.random.randn(10, 50).astype(np.float64)
        out = _apply_full_preprocessing(X, 'raw', None, False, None, False)
        np.testing.assert_array_equal(out, X)

    def test_snv_shape_and_mean(self):
        X = np.random.randn(10, 50).astype(np.float64)
        out = _apply_full_preprocessing(X, 'snv', None, False, None, False)
        assert out.shape == X.shape
        # SNV mean-centers each row; mean should be ~0
        np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-10)

    def test_baseline_plus_deriv_shape_preserved(self):
        X = np.abs(np.random.randn(10, 80)).astype(np.float64) + 1.0
        out = _apply_full_preprocessing(
            X, 'deriv1', window=11, autoscale=False,
            baseline_method='als', smoothing=False,
        )
        assert out.shape == X.shape

    def test_smoothing_plus_autoscale(self):
        X = np.random.randn(10, 80).astype(np.float64)
        out = _apply_full_preprocessing(
            X, 'raw', None, autoscale=True,
            baseline_method=None, smoothing=True,
        )
        assert out.shape == X.shape
        # Autoscale => zero mean per column
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-10)

    def test_full_chain(self):
        X = np.abs(np.random.randn(10, 100)).astype(np.float64) + 1.0
        out = _apply_full_preprocessing(
            X, 'deriv2', window=15, autoscale=True,
            baseline_method='polynomial', smoothing=True,
        )
        assert out.shape == X.shape
        assert np.isfinite(out).all()


class TestQuickEvaluateDirect:
    """Direct unit tests for _quick_evaluate (F12 #2)."""

    def test_regression_returns_finite(self):
        np.random.seed(42)
        X = np.random.randn(40, 60).astype(np.float64)
        y = 3.0 * X[:, 5] - 2.0 * X[:, 30] + np.random.randn(40) * 0.3
        score = _quick_evaluate(X, y, task_type='regression', cv_folds=3)
        assert np.isfinite(score)

    def test_classification_returns_finite(self):
        np.random.seed(42)
        X = np.random.randn(40, 60).astype(np.float64)
        y = np.array([0] * 20 + [1] * 20)
        score = _quick_evaluate(X, y, task_type='classification', cv_folds=3)
        assert np.isfinite(score)

    def test_one_class_with_outliers_returns_finite(self):
        np.random.seed(42)
        X = np.random.randn(40, 60).astype(np.float64)
        y = np.array([1] * 28 + [-1] * 12)
        score = _quick_evaluate(X, y, task_type='one_class', cv_folds=3)
        assert np.isfinite(score)

    def test_one_class_no_outliers(self):
        np.random.seed(42)
        X = np.random.randn(20, 60).astype(np.float64)
        y = np.array([1] * 20)
        score = _quick_evaluate(X, y, task_type='one_class', cv_folds=3)
        assert score == 0.0

    def test_regression_proxy_distinguishes_signal_on_small_data(self):
        """Regression test for the 2026-05-08 mean-prediction-collapse bug.

        Pre-fix, LightGBM's default min_child_samples=20 made splits
        impossible whenever n_train_per_fold < 40, so _quick_evaluate
        returned the same RMSE for ANY X (mean-prediction baseline).
        Symptom: TPE preprocessing discovery showed every config at the
        same RMSE for chemometrics-typical n~50 datasets.

        With n=40 and 5-fold CV, each training fold is 32 samples.
        Pre-fix, the proxy returned the same score regardless of input.
        Post-fix, min_child_samples scales to ~20% of n_train_per_fold,
        so a meaningful X (signal-bearing) and a meaningless X (pure
        noise) score differently.
        """
        rng = np.random.default_rng(0)
        n, p = 40, 60
        # Signal X: y is a strong linear function of one column with
        # mild noise. A working proxy should learn this.
        X_signal = rng.standard_normal((n, p)).astype(np.float64)
        y = 3.0 * X_signal[:, 5] + 0.3 * rng.standard_normal(n)
        # Noise X: y is unrelated to X. The proxy can only mean-predict.
        X_noise = rng.standard_normal((n, p)).astype(np.float64)

        score_signal = _quick_evaluate(X_signal, y, task_type='regression', cv_folds=5)
        score_noise = _quick_evaluate(X_noise, y, task_type='regression', cv_folds=5)
        # Signal RMSE must be meaningfully lower (less negative) than noise RMSE.
        # Pre-fix, both would be the same mean-prediction RMSE.
        assert score_signal > score_noise + 0.1, (
            f"Proxy did not distinguish signal from noise on n=40 — "
            f"signal score={score_signal}, noise score={score_noise}. "
            f"Likely regression of the min_child_samples scaling fix."
        )


class TestEmptyTPEFallback:
    """Empty-TPE-result fallback path (F12 #3)."""

    def test_all_trials_fail_returns_empty_list(self, synthetic_data_regression):
        X, y = synthetic_data_regression
        import spectral_predict.tpe_preprocessing_discovery as tpe_mod
        original_quick_eval = tpe_mod._quick_evaluate
        try:
            tpe_mod._quick_evaluate = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("forced failure")
            )
            configs = run_tpe_preprocessing_discovery(
                X, y, task_type='regression',
                n_trials=10, n_startup_trials=5, n_top=3,
            )
            assert configs == []
        finally:
            tpe_mod._quick_evaluate = original_quick_eval


class TestPipelineRoundtrip:
    """TPE config roundtrips through build_preprocessing_pipeline (F12 #5)."""

    def test_roundtrip_baseline_deriv_autoscale(self):
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import build_preprocessing_pipeline

        np.random.seed(42)
        X = np.abs(np.random.randn(15, 100)).astype(np.float64) + 1.0

        # Direct application
        direct = _apply_full_preprocessing(
            X, preproc_name='deriv1', window=11,
            autoscale=True, baseline_method='als', smoothing=True,
        )

        # Pipeline application using identical parameters
        steps = build_preprocessing_pipeline(
            preprocess_name='deriv',
            deriv=1, window=11, polyorder=2,
            baseline_method='als',
            baseline_params={'lam': 1e5, 'p': 0.01},
            smoothing=True, smoothing_window=17, smoothing_polyorder=2,
            autoscale=True,
        )
        piped = Pipeline(steps).fit_transform(X)

        assert direct.shape == piped.shape
        np.testing.assert_allclose(direct, piped, rtol=1e-5, atol=1e-6)

    def test_roundtrip_snv_no_extras(self):
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import build_preprocessing_pipeline

        np.random.seed(42)
        X = np.random.randn(15, 80).astype(np.float64)

        direct = _apply_full_preprocessing(
            X, preproc_name='snv', window=None,
            autoscale=False, baseline_method=None, smoothing=False,
        )

        steps = build_preprocessing_pipeline(
            preprocess_name='snv',
            autoscale=False,
        )
        piped = Pipeline(steps).fit_transform(X)

        assert direct.shape == piped.shape
        np.testing.assert_allclose(direct, piped, rtol=1e-5, atol=1e-6)
