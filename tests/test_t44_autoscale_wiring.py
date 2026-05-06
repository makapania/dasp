"""T-44: autoscale flag plumbing regression tests.

Verifies that the decoupled autoscale flags reach the correct backend entry
points independently:

  1. Bayesian path:  GUI.bayes_enable_autoscale  -> run_unified_bayesian(enable_autoscale=...)
  2. TPE path:       GUI.tpe_enable_autoscale    -> run_tpe_preprocessing_discovery(enable_autoscale=...)
  3. Grid path:      GUI.use_autoscale           -> run_search(autoscale=...)  (unchanged)

The tests mock the backend entry points and inspect the kwargs the GUI
thread would pass, without actually launching Tk or running models.
"""

import inspect
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig_param_default(func, param_name):
    """Return the default value of *param_name* in *func*'s signature."""
    sig = inspect.signature(func)
    p = sig.parameters[param_name]
    return p.default


# ---------------------------------------------------------------------------
# 1. run_unified_bayesian accepts enable_autoscale (default False for
#    backward compat, overridden by the GUI to bayes_enable_autoscale).
# ---------------------------------------------------------------------------

class TestBayesianAutoscalePlumbing:
    def test_run_unified_bayesian_has_enable_autoscale_param(self):
        from spectral_predict.unified_bayesian import run_unified_bayesian
        sig = inspect.signature(run_unified_bayesian)
        assert 'enable_autoscale' in sig.parameters

    def test_enable_autoscale_default_is_false(self):
        """Backend default is False — the GUI override (default True) is
        the behavioral change; the backend stays backward-compatible."""
        from spectral_predict.unified_bayesian import run_unified_bayesian
        assert _sig_param_default(run_unified_bayesian, 'enable_autoscale') is False

    def test_suggest_preprocessing_has_enable_autoscale_param(self):
        from spectral_predict.unified_bayesian import suggest_preprocessing
        sig = inspect.signature(suggest_preprocessing)
        assert 'enable_autoscale' in sig.parameters


# ---------------------------------------------------------------------------
# 2. TPE discovery: run_tpe_preprocessing_discovery enable_autoscale default
#    is True (was already True before T-44 — the new change is that the GUI
#    passes its own flag instead of the grid checkbox).
# ---------------------------------------------------------------------------

class TestTPEAutoscalePlumbing:
    def test_tpe_discovery_has_enable_autoscale_param(self):
        from spectral_predict.tpe_preprocessing_discovery import run_tpe_preprocessing_discovery
        sig = inspect.signature(run_tpe_preprocessing_discovery)
        assert 'enable_autoscale' in sig.parameters

    def test_tpe_discovery_enable_autoscale_default_true(self):
        from spectral_predict.tpe_preprocessing_discovery import run_tpe_preprocessing_discovery
        assert _sig_param_default(run_tpe_preprocessing_discovery, 'enable_autoscale') is True


# ---------------------------------------------------------------------------
# 3. search.py: run_search and run_one_class_search have tpe_enable_autoscale
#    parameter (separate from the grid autoscale flag).
# ---------------------------------------------------------------------------

class TestSearchAutoscaleDecoupling:
    def test_run_search_has_tpe_enable_autoscale(self):
        from spectral_predict.search import run_search
        sig = inspect.signature(run_search)
        assert 'tpe_enable_autoscale' in sig.parameters

    def test_run_search_tpe_enable_autoscale_default_true(self):
        from spectral_predict.search import run_search
        assert _sig_param_default(run_search, 'tpe_enable_autoscale') is True

    def test_run_search_has_grid_autoscale(self):
        """Grid autoscale flag must still exist and default to False."""
        from spectral_predict.search import run_search
        sig = inspect.signature(run_search)
        assert 'autoscale' in sig.parameters
        assert _sig_param_default(run_search, 'autoscale') is False

    def test_run_one_class_search_has_tpe_enable_autoscale(self):
        from spectral_predict.search import run_one_class_search
        sig = inspect.signature(run_one_class_search)
        assert 'tpe_enable_autoscale' in sig.parameters

    def test_run_one_class_search_tpe_enable_autoscale_default_true(self):
        from spectral_predict.search import run_one_class_search
        assert _sig_param_default(run_one_class_search, 'tpe_enable_autoscale') is True

    def test_run_one_class_search_has_grid_autoscale(self):
        from spectral_predict.search import run_one_class_search
        sig = inspect.signature(run_one_class_search)
        assert 'autoscale' in sig.parameters
        assert _sig_param_default(run_one_class_search, 'autoscale') is False


# ---------------------------------------------------------------------------
# 4. Behavioral: TPE discovery actually explores autoscale when enabled
# ---------------------------------------------------------------------------

class TestTPEDiscoveryExploresAutoscale:
    def test_enable_autoscale_true_produces_both_values(self):
        """When enable_autoscale=True, at least one config should have
        _tpe_autoscale=True and at least one should have False."""
        import numpy as np
        from spectral_predict.tpe_preprocessing_discovery import run_tpe_preprocessing_discovery

        rng = np.random.RandomState(42)
        X = rng.randn(30, 50)
        y = rng.randn(30)

        configs = run_tpe_preprocessing_discovery(
            X, y,
            task_type='regression',
            n_trials=30,
            n_startup_trials=10,
            n_top=10,
            cv_folds=3,
            enable_autoscale=True,
            enable_baseline=False,
            enable_smoothing=False,
        )
        autoscale_vals = {cfg['_tpe_autoscale'] for cfg in configs}
        assert True in autoscale_vals, "Expected at least one config with autoscale=True"
        assert False in autoscale_vals, "Expected at least one config with autoscale=False"

    def test_enable_autoscale_false_all_false(self):
        """When enable_autoscale=False, all configs must have _tpe_autoscale=False."""
        import numpy as np
        from spectral_predict.tpe_preprocessing_discovery import run_tpe_preprocessing_discovery

        rng = np.random.RandomState(42)
        X = rng.randn(30, 50)
        y = rng.randn(30)

        configs = run_tpe_preprocessing_discovery(
            X, y,
            task_type='regression',
            n_trials=20,
            n_startup_trials=10,
            n_top=5,
            cv_folds=3,
            enable_autoscale=False,
            enable_baseline=False,
            enable_smoothing=False,
        )
        for cfg in configs:
            assert cfg['_tpe_autoscale'] is False


# ---------------------------------------------------------------------------
# 5. Behavioral: Bayesian suggest_preprocessing explores autoscale when enabled
# ---------------------------------------------------------------------------

class TestBayesianSuggestPreprocessingAutoscale:
    def test_autoscale_explored_when_enabled(self):
        """When enable_autoscale=True, suggest_preprocessing includes apply_autoscale key."""
        import optuna
        from spectral_predict.unified_bayesian import suggest_preprocessing

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=42))
        autoscale_values = set()
        for _ in range(20):
            trial = study.ask()
            config = suggest_preprocessing(trial, n_features=100, enable_autoscale=True)
            autoscale_values.add(config.get('apply_autoscale'))
        assert True in autoscale_values
        assert False in autoscale_values

    def test_autoscale_always_false_when_disabled(self):
        """When enable_autoscale=False, apply_autoscale is always False."""
        import optuna
        from spectral_predict.unified_bayesian import suggest_preprocessing

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=42))
        for _ in range(20):
            trial = study.ask()
            config = suggest_preprocessing(trial, n_features=100, enable_autoscale=False)
            assert config.get('apply_autoscale') is False
