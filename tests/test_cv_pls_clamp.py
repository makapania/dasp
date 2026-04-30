"""Tests for PLS-component clamping by training-fold size (T-10).

Covers:
- compute_min_train_fold_size for kfold / repeated_kfold / loo
- Edge cases (n_samples=2, n_folds=2, n_folds=n_samples)
- Group-strategy NotImplementedError
- Unknown strategy ValueError
- n_folds > n_samples invalid geometry (Codex suggestion #1)
"""
from __future__ import annotations

import pytest

from spectral_predict.cv_utils import compute_min_train_fold_size


class TestComputeMinTrainFoldSize:
    """Pure-function tests for the new helper."""

    def test_kfold_n10_k5_returns_8(self):
        assert compute_min_train_fold_size('kfold', 10, 5) == 8

    def test_kfold_n20_k5_returns_16(self):
        assert compute_min_train_fold_size('kfold', 20, 5) == 16

    def test_kfold_n100_k5_returns_80(self):
        assert compute_min_train_fold_size('kfold', 100, 5) == 80

    def test_kfold_n9_k5_returns_7(self):
        # 9 * 4 // 5 = 7 (exact for sklearn KFold when n not divisible by k;
        # smallest train fold is n - ceil(n/k) = 9 - 2 = 7)
        assert compute_min_train_fold_size('kfold', 9, 5) == 7

    def test_kfold_n7_k3_returns_4(self):
        # 7 * 2 // 3 = 4.  Exact: n - ceil(n/k) = 7 - 3 = 4.
        assert compute_min_train_fold_size('kfold', 7, 3) == 4

    def test_repeated_kfold_matches_kfold(self):
        assert (
            compute_min_train_fold_size('repeated_kfold', 20, 5)
            == compute_min_train_fold_size('kfold', 20, 5)
        )

    def test_loo_n20_returns_19(self):
        assert compute_min_train_fold_size('loo', 20, 5) == 19

    def test_loo_n10_returns_9(self):
        assert compute_min_train_fold_size('loo', 10, 5) == 9

    def test_loo_ignores_n_folds(self):
        assert compute_min_train_fold_size('loo', 50, 99) == 49
        assert compute_min_train_fold_size('loo', 50, 0) == 49

    def test_kfold_minimum_n2_k2(self):
        assert compute_min_train_fold_size('kfold', 2, 2) == 1

    def test_kfold_n_samples_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_samples >= 2"):
            compute_min_train_fold_size('kfold', 1, 5)

    def test_kfold_n_folds_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_folds >= 2"):
            compute_min_train_fold_size('kfold', 20, 1)

    def test_invalid_geometry_n_folds_greater_than_n_samples_raises(self):
        # Codex suggestion #1: n_folds > n_samples is invalid for KFold.
        with pytest.raises(ValueError, match="Cannot have more folds"):
            compute_min_train_fold_size('kfold', 5, 10)

    def test_invalid_geometry_repeated_kfold_n_folds_greater_than_n_samples_raises(self):
        with pytest.raises(ValueError, match="Cannot have more folds"):
            compute_min_train_fold_size('repeated_kfold', 5, 10)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown cv_strategy"):
            compute_min_train_fold_size('bogus', 20, 5)

    def test_group_strategies_not_implemented(self):
        with pytest.raises(NotImplementedError, match="T-15"):
            compute_min_train_fold_size('group_kfold', 20, 5)
        with pytest.raises(NotImplementedError, match="T-15"):
            compute_min_train_fold_size('leave_one_group_out', 20, 5)


class TestRunSearchPLSGridClamping:
    """Black-box: run_search must clamp the PLS grid for small datasets.

    These tests call run_search with a single PLS model and inspect the
    result DataFrame to confirm that no grid row used n_components >
    min_train_fold_size.  Uses synthetic regression data sized so the bug
    would show up if the clamp is missing.  Keeps run_search invocations
    small (3 folds, no variable subsets) so each test runs in <30 s.
    """

    @pytest.fixture
    def tiny_regression_data(self):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 50))
        y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.05 * rng.standard_normal(10)
        return pd.DataFrame(X), pd.Series(y)

    @pytest.fixture
    def normal_regression_data(self):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 50))
        y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.05 * rng.standard_normal(80)
        return pd.DataFrame(X), pd.Series(y)

    def test_n10_kfold_clamps_to_8_components(self, tiny_regression_data):
        """N=10, k=5 -> max grid n_components must be 8, NOT 20."""
        from spectral_predict.search import run_search
        X, y = tiny_regression_data
        df, _ = run_search(
            X, y,
            task_type='regression',
            folds=5,
            cv_strategy='kfold',
            max_n_components=20,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'sg3': False, 'sg4': False, 'deriv_snv': False},
            window_sizes=[7],
            enable_variable_subsets=False,
            enable_region_subsets=False,
            variable_selection_methods=['none'],
        )
        n_components_seen = _extract_n_components_seen(df)
        assert n_components_seen, f"No PLS rows produced; df cols={list(df.columns)}"
        assert max(n_components_seen) <= 8, (
            f"PLS grid for N=10 k=5 produced n_components={max(n_components_seen)}, "
            f"expected max 8. Clamp is broken. Seen: {sorted(n_components_seen)}"
        )
        assert 1 in n_components_seen

    def test_n10_loo_clamps_to_9_components(self, tiny_regression_data):
        """N=10, LOO -> max grid n_components must be 9 (n-1), NOT 8."""
        from spectral_predict.search import run_search
        X, y = tiny_regression_data
        df, _ = run_search(
            X, y,
            task_type='regression',
            folds=5,
            cv_strategy='loo',
            max_n_components=20,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'sg3': False, 'sg4': False, 'deriv_snv': False},
            window_sizes=[7],
            enable_variable_subsets=False,
            enable_region_subsets=False,
            variable_selection_methods=['none'],
        )
        n_components_seen = _extract_n_components_seen(df)
        assert n_components_seen, f"No PLS rows produced; df cols={list(df.columns)}"
        assert max(n_components_seen) <= 9, (
            f"PLS grid for N=10 LOO produced n_components={max(n_components_seen)}, "
            f"expected max 9 (n-1). Seen: {sorted(n_components_seen)}"
        )
        assert max(n_components_seen) == 9, (
            f"LOO clamp expected n_components_max == 9, got {max(n_components_seen)}. "
            "If this is 8, the fix is using the K-fold formula instead of n-1."
        )

    def test_n80_kfold_uses_full_grid_default_max(self, normal_regression_data):
        """N=80, k=5, max_n_components=10 -> all 10 components present.

        Confirms the clamp does NOT artificially shrink grids on larger
        datasets: train fold = 64 >> 10, so the bind is max_n_components.
        """
        from spectral_predict.search import run_search
        X, y = normal_regression_data
        df, _ = run_search(
            X, y,
            task_type='regression',
            folds=5,
            cv_strategy='kfold',
            max_n_components=10,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'sg3': False, 'sg4': False, 'deriv_snv': False},
            window_sizes=[7],
            enable_variable_subsets=False,
            enable_region_subsets=False,
            variable_selection_methods=['none'],
        )
        n_components_seen = _extract_n_components_seen(df)
        assert n_components_seen, f"No PLS rows produced; df cols={list(df.columns)}"
        assert max(n_components_seen) == 10, (
            f"PLS grid for N=80 k=5 with max_n_components=10 should reach 10, "
            f"got {max(n_components_seen)}. Clamp may be over-aggressive."
        )

    def test_n10_repeated_kfold_matches_kfold(self, tiny_regression_data):
        """RepeatedKFold should produce the same n_components ceiling as KFold."""
        from spectral_predict.search import run_search
        X, y = tiny_regression_data
        df, _ = run_search(
            X, y,
            task_type='regression',
            folds=5,
            cv_strategy='repeated_kfold',
            cv_n_repeats=2,
            max_n_components=20,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'sg3': False, 'sg4': False, 'deriv_snv': False},
            window_sizes=[7],
            enable_variable_subsets=False,
            enable_region_subsets=False,
            variable_selection_methods=['none'],
        )
        n_components_seen = _extract_n_components_seen(df)
        assert n_components_seen, f"No PLS rows produced; df cols={list(df.columns)}"
        assert max(n_components_seen) <= 8, (
            f"RepeatedKFold N=10 k=5 produced max n_components={max(n_components_seen)}, "
            f"expected 8 (same as kfold). Seen: {sorted(n_components_seen)}"
        )


def _extract_n_components_seen(df) -> set[int]:
    """Pull the unique set of n_components values from a result DataFrame.

    Codex suggestion #4: prefer the 'LVs' column as canonical.  If absent,
    fall back to 'Params' using ast.literal_eval (not json.loads) because
    this codebase stores Params as Python repr with single quotes, which
    json.loads cannot parse.
    """
    import ast

    seen: set[int] = set()
    if df is None or len(df) == 0:
        return seen

    for col in ('LVs', 'n_components', 'NumComponents'):
        if col in df.columns:
            for v in df[col].dropna().tolist():
                try:
                    seen.add(int(v))
                except (TypeError, ValueError):
                    continue
            if seen:
                return seen

    for col in ('Params', 'Model_Params'):
        if col in df.columns:
            for v in df[col].dropna().tolist():
                if isinstance(v, dict):
                    if 'n_components' in v:
                        seen.add(int(v['n_components']))
                elif isinstance(v, str):
                    try:
                        parsed = ast.literal_eval(v)
                    except (ValueError, SyntaxError):
                        continue
                    if isinstance(parsed, dict) and 'n_components' in parsed:
                        seen.add(int(parsed['n_components']))
            if seen:
                return seen

    return seen


class TestRunBayesianSearchPLSGridClamping:
    """Black-box: run_bayesian_search must clamp the PLS LV upper bound.

    Bayesian search uses an Optuna IntDistribution for n_components but
    the upper bound flows through the same min_train_samples clamp.
    We confirm by inspecting the LVs column.
    """

    @pytest.fixture
    def tiny_regression_data(self):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 50))
        y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.05 * rng.standard_normal(10)
        return pd.DataFrame(X), pd.Series(y)

    def test_n10_loo_bayesian_caps_at_9(self, tiny_regression_data):
        from spectral_predict.search import run_bayesian_search
        X, y = tiny_regression_data
        df, _ = run_bayesian_search(
            X, y,
            task_type='regression',
            folds=5,
            cv_strategy='loo',
            n_trials=8,
            max_n_components=20,
            models_to_test=['PLS'],
            preprocessing_methods={'raw': True, 'snv': False, 'sg1': False, 'sg2': False, 'sg3': False, 'sg4': False, 'deriv_snv': False},
            enable_variable_subsets=False,
            enable_region_subsets=False,
            variable_selection_methods=['none'],
        )
        n_components_seen = _extract_n_components_seen(df)
        assert n_components_seen, f"No PLS rows produced; df cols={list(df.columns)}"
        assert max(n_components_seen) <= 9, (
            f"Bayesian PLS for N=10 LOO produced n_components={max(n_components_seen)}, "
            f"expected max 9 (n-1). Clamp is broken in run_bayesian_search."
        )
