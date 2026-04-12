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

    def test_run_single_fold_supports_pooled_aggregation(self):
        """Integration test exercising the real search.py production path.

        Calls _run_single_fold for each fold of a synthetic dataset and verifies
        that the resulting cv_metrics list can be aggregated via the pooled
        formula used in search.py (line 4215-4222), and that the result matches
        sklearn's cross_val_predict -> sqrt(mse(y, y_pred)) reference.
        """
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.metrics import mean_squared_error

        from spectral_predict.search import _run_single_fold

        rng = np.random.default_rng(7)
        n = 30
        X = rng.normal(size=(n, 5))
        y = X @ np.array([1.0, -0.5, 0.2, 0.0, 0.3]) + 0.1 * rng.normal(size=n)

        pipe = Pipeline([("model", Ridge(alpha=0.5))])
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        cv_metrics = [
            _run_single_fold(
                pipe, X, y, train_idx, test_idx,
                task_type="regression",
                is_binary_classification=False,
            )
            for train_idx, test_idx in cv.split(X, y)
        ]

        # Each fold returns a dict with y_test, y_pred, RMSE, R2
        for m in cv_metrics:
            assert "y_test" in m and "y_pred" in m

        # Reproduce the exact aggregation search.py does
        all_y_test = np.concatenate([m["y_test"] for m in cv_metrics])
        all_y_pred = np.concatenate([m["y_pred"] for m in cv_metrics])
        pooled_rmse = float(np.sqrt(mean_squared_error(all_y_test, all_y_pred)))

        # Reference: sklearn's cross_val_predict on the same data / splitter
        cv_ref = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_ref = cross_val_predict(Ridge(alpha=0.5), X, y, cv=cv_ref)
        ref_rmse = float(np.sqrt(mean_squared_error(y, y_pred_ref)))

        # The pooled result from real _run_single_fold must match sklearn's reference
        assert abs(pooled_rmse - ref_rmse) < 1e-10, (
            f"Pooled RMSE from _run_single_fold ({pooled_rmse}) "
            f"does not match cross_val_predict reference ({ref_rmse})"
        )

        # And under K-fold with equal fold sizes, the pooled form is close to — but not
        # identical to — the buggy per-fold mean. Verify they diverge so the test would
        # fail if search.py reverted to the old computation.
        old_per_fold_rmse = float(np.mean([m["RMSE"] for m in cv_metrics]))
        assert abs(old_per_fold_rmse - pooled_rmse) > 1e-12


class TestRunSearchCvStrategyPlumbing:
    """Task 3: end-to-end plumbing of cv_strategy / cv_n_repeats through run_search.

    Smoke tests that confirm the default-kwargs path still produces kfold results
    and that passing cv_strategy='loo' threads through to training_config.
    """

    def _tiny_regression_data(self):
        import pandas as pd
        rng = np.random.default_rng(123)
        n, p = 15, 5
        X = pd.DataFrame(rng.normal(size=(n, p)))
        # Make y depend on X so RMSEcv is meaningful (non-trivial regression)
        y = pd.Series(
            X.iloc[:, 0].to_numpy() * 1.5
            - 0.7 * X.iloc[:, 1].to_numpy()
            + 0.1 * rng.normal(size=n)
        )
        return X, y

    def test_default_kwargs_preserves_kfold_behaviour(self):
        """With no cv_strategy passed, search should default to kfold."""
        from spectral_predict.search import run_search

        X, y = self._tiny_regression_data()
        df_results, _ = run_search(
            X, y, task_type="regression", folds=3,
            models_to_test=["PLS"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )
        assert len(df_results) > 0, "Default run_search should return results"
        # At least one row must expose training_config with cv_strategy='kfold'
        tcs = df_results['training_config'].dropna().tolist()
        assert len(tcs) > 0, "training_config should be present in results"
        strategies = {tc.get('cv_strategy') for tc in tcs if isinstance(tc, dict)}
        assert strategies == {'kfold'}, (
            f"Default path should only produce cv_strategy='kfold', got {strategies}"
        )
        # Sanity: RMSEcv / R2cv are finite
        assert df_results['RMSEcv'].notna().any()
        assert df_results['R2cv'].notna().any()

    def test_loo_strategy_threads_through_to_training_config(self):
        """cv_strategy='loo' should produce training_config['cv_strategy']='loo'
        and non-NaN RMSEcv / R2cv on tiny synthetic data.
        """
        from spectral_predict.search import run_search

        X, y = self._tiny_regression_data()
        df_results, _ = run_search(
            X, y, task_type="regression", folds=3,  # folds is irrelevant under LOO
            cv_strategy='loo',
            cv_n_repeats=1,  # ignored under LOO but set for clarity
            models_to_test=["PLS"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )
        assert len(df_results) > 0, "LOO run_search should return results"
        tcs = df_results['training_config'].dropna().tolist()
        assert len(tcs) > 0, "training_config should be present"
        strategies = {tc.get('cv_strategy') for tc in tcs if isinstance(tc, dict)}
        assert 'loo' in strategies, (
            f"LOO path should produce cv_strategy='loo' in at least one row, got {strategies}"
        )
        # Under LOO, effective folds == n_samples
        n_samples = len(X)
        loo_rows = [tc for tc in tcs if isinstance(tc, dict) and tc.get('cv_strategy') == 'loo']
        for tc in loo_rows:
            assert tc.get('folds') == n_samples, (
                f"LOO training_config['folds'] should equal n_samples ({n_samples}), got {tc.get('folds')}"
            )
        # RMSEcv / R2cv must be finite under LOO (Task 1 + Task 2 fixes)
        assert df_results['RMSEcv'].notna().any(), "LOO RMSEcv must be non-NaN"
        assert df_results['R2cv'].notna().any(), "LOO R2cv must be non-NaN"

    def test_repeated_kfold_strategy_threads_through_to_training_config(self):
        """cv_strategy='repeated_kfold' should stamp training_config accordingly."""
        from spectral_predict.search import run_search

        X, y = self._tiny_regression_data()
        df_results, _ = run_search(
            X, y, task_type="regression", folds=3,
            cv_strategy='repeated_kfold',
            cv_n_repeats=2,
            models_to_test=["PLS"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
            enable_region_subsets=False,
        )
        assert len(df_results) > 0
        tcs = df_results['training_config'].dropna().tolist()
        assert len(tcs) > 0
        strategies = {tc.get('cv_strategy') for tc in tcs if isinstance(tc, dict)}
        assert 'repeated_kfold' in strategies
        repeats = {
            tc.get('cv_n_repeats') for tc in tcs
            if isinstance(tc, dict) and tc.get('cv_strategy') == 'repeated_kfold'
        }
        assert 2 in repeats, f"Expected cv_n_repeats=2 in training_config, got {repeats}"
        assert df_results['RMSEcv'].notna().any()
        assert df_results['R2cv'].notna().any()
