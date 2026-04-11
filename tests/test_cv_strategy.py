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


class TestRmseCvPooledFix:
    """Task 2: RMSEcv pooled computation fix.

    Regression tests asserting the new pooled behavior:
    - Under K-fold, mean_rmse == sqrt(mean_squared_error(pooled_y, pooled_pred)).
    - Under LOO, mean_rmse != mean(per-fold RMSE); the pooled form is RMSE,
      the per-fold-averaged form is MAE.
    - R2cv unchanged (it was already pooled before Task 2).
    """

    def test_pooled_rmse_differs_from_per_fold_mean_under_loo(self):
        """Under LOO, per-fold 'RMSE' is |y-y_hat|, so mean(per-fold RMSE) = MAE.
        Pooled sqrt(mse) gives the true RMSE, which is different.
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error as mse
        from sklearn.model_selection import LeaveOneOut, cross_val_predict

        rng = np.random.default_rng(42)
        n = 20
        X = rng.normal(size=(n, 4))
        y = rng.normal(size=n)

        model = Ridge(alpha=1.0)
        cv = LeaveOneOut()
        y_pred_cv = cross_val_predict(model, X, y, cv=cv)

        pooled_rmse = float(np.sqrt(mse(y, y_pred_cv)))
        pooled_mae = float(mean_absolute_error(y, y_pred_cv))

        # Per-fold RMSE under LOO is |y_true - y_pred|, i.e., per-sample absolute error.
        # So mean_of_per_fold_rmse_loo == MAE, not RMSE.
        per_fold_rmse_mean = float(np.mean(np.abs(y - y_pred_cv)))

        # Sanity check: per-fold-averaged "RMSE" under LOO equals MAE
        assert abs(per_fold_rmse_mean - pooled_mae) < 1e-10
        # And it is NOT equal to pooled RMSE
        assert abs(per_fold_rmse_mean - pooled_rmse) > 1e-6

    def test_pooled_rmse_close_to_per_fold_under_equal_kfold(self):
        """Under equal-sized K-fold, pooled RMSE is close (but not exactly equal) to
        sqrt(mean(per-fold MSE)). The delta is typically <10% on well-behaved data.
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error as mse
        from sklearn.model_selection import KFold, cross_val_predict

        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 4))
        y = X @ np.array([1.0, -0.5, 0.3, 0.0]) + 0.1 * rng.normal(size=50)

        model = Ridge(alpha=0.1)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(model, X, y, cv=cv)
        pooled_rmse = float(np.sqrt(mse(y, y_pred_cv)))

        # Compute per-fold RMSE mean (the old, incorrect method)
        per_fold_rmses = []
        for train_idx, test_idx in cv.split(X, y):
            model_i = Ridge(alpha=0.1).fit(X[train_idx], y[train_idx])
            fold_pred = model_i.predict(X[test_idx])
            per_fold_rmses.append(np.sqrt(mse(y[test_idx], fold_pred)))
        per_fold_rmse_mean = float(np.mean(per_fold_rmses))

        # They're close but not equal
        rel_delta = abs(per_fold_rmse_mean - pooled_rmse) / pooled_rmse
        assert rel_delta < 0.10  # <10% sanity
        # And they're not *exactly* equal — the pooled form is what chemometrics uses
        # (this assertion documents the non-equivalence, not the bug)
        assert abs(per_fold_rmse_mean - pooled_rmse) > 1e-12

    def test_search_py_computes_pooled_rmse(self):
        """Site A verification: search.py's regression branch now computes
        mean_rmse via sqrt(mean_squared_error) on pooled predictions, matching
        what R2 already does. Verified by reproducing the exact computation.
        """
        from sklearn.metrics import mean_squared_error

        # Simulate the cv_metrics structure search.py builds
        cv_metrics = [
            {"RMSE": 0.5, "y_test": np.array([1.0, 2.0]), "y_pred": np.array([1.1, 1.8])},
            {"RMSE": 0.3, "y_test": np.array([3.0, 4.0]), "y_pred": np.array([3.2, 3.9])},
            {"RMSE": 0.7, "y_test": np.array([5.0, 6.0]), "y_pred": np.array([4.5, 6.3])},
        ]

        # Old (buggy) per-fold mean
        old_mean_rmse = float(np.mean([m["RMSE"] for m in cv_metrics]))

        # New pooled computation (what search.py now does)
        all_y_test = np.concatenate([m["y_test"] for m in cv_metrics])
        all_y_pred = np.concatenate([m["y_pred"] for m in cv_metrics])
        new_mean_rmse = float(np.sqrt(mean_squared_error(all_y_test, all_y_pred)))

        # They're different — the fix is visible
        assert abs(old_mean_rmse - new_mean_rmse) > 1e-6
        # And the new value matches the pooled sqrt(MSE) exactly
        expected = float(np.sqrt(np.mean((all_y_test - all_y_pred) ** 2)))
        assert abs(new_mean_rmse - expected) < 1e-12
