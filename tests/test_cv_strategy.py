"""Unit tests for the CV strategy factory and LOO early-stopping guard."""
import pytest
import warnings
import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut,
)

from spectral_predict.cv_utils import build_cv_splitter, cross_validate_with_early_stopping


class TestBuildCvSplitter:
    """Tests for build_cv_splitter factory."""

    def test_kfold_regression(self):
        splitter = build_cv_splitter('kfold', 5, 'regression')
        assert isinstance(splitter, KFold)
        assert not isinstance(splitter, StratifiedKFold)
        assert splitter.n_splits == 5
        assert splitter.shuffle is True

    def test_kfold_classification(self):
        splitter = build_cv_splitter('kfold', 5, 'classification')
        assert isinstance(splitter, StratifiedKFold)
        assert splitter.n_splits == 5

    def test_kfold_one_class(self):
        # one-class should use plain KFold (not stratified)
        splitter = build_cv_splitter('kfold', 5, 'one_class')
        assert isinstance(splitter, KFold)
        assert not isinstance(splitter, StratifiedKFold)

    def test_repeated_kfold_regression(self):
        splitter = build_cv_splitter('repeated_kfold', 5, 'regression', n_repeats=3)
        assert isinstance(splitter, RepeatedKFold)
        # RepeatedKFold produces n_splits * n_repeats total splits
        X = np.zeros((20, 3))
        assert splitter.get_n_splits(X) == 5 * 3

    def test_repeated_kfold_classification(self):
        splitter = build_cv_splitter('repeated_kfold', 5, 'classification', n_repeats=3)
        assert isinstance(splitter, RepeatedStratifiedKFold)

    def test_repeated_kfold_one_class(self):
        splitter = build_cv_splitter('repeated_kfold', 5, 'one_class', n_repeats=2)
        # one-class uses plain RepeatedKFold (not stratified)
        assert isinstance(splitter, RepeatedKFold)
        assert not isinstance(splitter, RepeatedStratifiedKFold)

    def test_loo_regression(self):
        splitter = build_cv_splitter('loo', 5, 'regression')
        assert isinstance(splitter, LeaveOneOut)

    def test_loo_classification(self):
        # LOO ignores task_type — same instance type either way
        splitter = build_cv_splitter('loo', 5, 'classification')
        assert isinstance(splitter, LeaveOneOut)

    def test_loo_one_class(self):
        splitter = build_cv_splitter('loo', 5, 'one_class')
        assert isinstance(splitter, LeaveOneOut)

    def test_loo_ignores_n_folds(self):
        # Pass a bogus n_folds — LOO should not care
        splitter = build_cv_splitter('loo', 0, 'regression')
        assert isinstance(splitter, LeaveOneOut)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown CV strategy"):
            build_cv_splitter('bogus', 5, 'regression')

    def test_random_state_propagated(self):
        # Both splitter calls with same random state should produce identical splits
        X = np.arange(20).reshape(-1, 1)
        y = np.arange(20)
        s1 = build_cv_splitter('kfold', 5, 'regression', random_state=42)
        s2 = build_cv_splitter('kfold', 5, 'regression', random_state=42)
        splits_1 = [(list(tr), list(te)) for tr, te in s1.split(X, y)]
        splits_2 = [(list(tr), list(te)) for tr, te in s2.split(X, y)]
        assert splits_1 == splits_2


class TestLooEarlyStoppingGuard:
    """Tests for the LeaveOneOut early-stopping disable guard in cross_validate_with_early_stopping."""

    def test_loo_disables_early_stopping_for_boosting_model(self):
        """XGBoost under LOO should train without early stopping and emit a warning."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            pytest.skip("xgboost not available")

        rng = np.random.default_rng(42)
        X = rng.normal(size=(12, 4))
        y = rng.normal(size=12)

        model = XGBRegressor(n_estimators=10, verbosity=0)
        cv = build_cv_splitter('loo', 5, 'regression')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = cross_validate_with_early_stopping(
                model, X, y, cv=cv,
                scoring='neg_root_mean_squared_error',
                early_stopping_rounds=10,
            )
            # Check that the warning fired
            matching = [wi for wi in w if "Early stopping disabled" in str(wi.message)]
            assert len(matching) >= 1, f"Expected LOO early-stopping warning, got: {[str(wi.message) for wi in w]}"

        # Check that CV actually ran and produced per-fold scores (n_splits == n_samples == 12)
        assert 'test_score' in results
        assert len(results['test_score']) == 12

    def test_kfold_does_not_disable_early_stopping(self):
        """Regular KFold should leave early stopping enabled (no guard fires)."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            pytest.skip("xgboost not available")

        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 4))
        y = rng.normal(size=30)

        model = XGBRegressor(n_estimators=10, verbosity=0)
        cv = build_cv_splitter('kfold', 5, 'regression')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = cross_validate_with_early_stopping(
                model, X, y, cv=cv,
                scoring='neg_root_mean_squared_error',
                early_stopping_rounds=5,
            )
            # No LOO guard warning
            matching = [wi for wi in w if "Early stopping disabled under LeaveOneOut" in str(wi.message)]
            assert len(matching) == 0
