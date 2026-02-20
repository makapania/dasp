"""
Verification test for LightGBM fix.
Tests the problematic scenario with the new min_child_samples=5 setting.
"""

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


pytestmark = pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")


class TestLightGBMFix:
    """Verify that min_child_samples=5 fix resolves negative R2 issue."""

    def test_old_config_small_samples(self):
        """Test old configuration (min_child_samples=20) on small dataset."""
        X, y = make_regression(n_samples=50, n_features=2000, random_state=42, noise=10)

        model_old = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbosity=-1
        )
        scores_old = cross_val_score(model_old, X, y, cv=3, scoring='r2')
        # Just verify it runs without error
        assert len(scores_old) == 3

    def test_new_config_improves_r2(self):
        """Test that new configuration (min_child_samples=5) produces better R2."""
        X, y = make_regression(n_samples=50, n_features=2000, random_state=42, noise=10)

        model_old = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbosity=-1
        )
        scores_old = cross_val_score(model_old, X, y, cv=3, scoring='r2')

        model_new = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1
        )
        scores_new = cross_val_score(model_new, X, y, cv=3, scoring='r2')

        mean_r2_old = np.mean(scores_old)
        mean_r2_new = np.mean(scores_new)

        assert mean_r2_new >= mean_r2_old, (
            f"New config R2 ({mean_r2_new:.4f}) should be >= old config R2 ({mean_r2_old:.4f})"
        )
