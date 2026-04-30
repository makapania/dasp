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
