"""Comprehensive unit tests for scoring and ranking system."""

import numpy as np
import pandas as pd
import pytest

from spectral_predict.scoring import compute_composite_score


class TestCompositeScoring:
    """Test the compute_composite_score function."""

    def create_test_data_regression(self, n_models=10):
        """Create test regression data with known characteristics."""
        np.random.seed(42)

        # Create models with varying characteristics
        data = {
            "Model": ["PLS"] * n_models,
            "RMSE": np.random.uniform(0.1, 0.5, n_models),
            "R2": np.random.uniform(0.6, 0.95, n_models),
            "n_vars": np.random.randint(10, 2000, n_models),
            "full_vars": [2151] * n_models,
            "LVs": np.random.randint(5, 20, n_models),
            "Params": ["{}"] * n_models,
            "Preprocess": ["raw"] * n_models,
            "Deriv": [0] * n_models,
            "Window": [0] * n_models,
            "Poly": [0] * n_models,
            "SubsetTag": ["full"] * n_models,
            "top_vars": [None] * n_models,
        }
        return pd.DataFrame(data)

    def test_penalty_zero_ranks_by_performance_only(self):
        """At penalty=0, ranking should be based purely on R² performance."""
        df = self.create_test_data_regression(100)

        # Add a clearly best model (highest R², lowest RMSE)
        df.loc[50, "R2"] = 0.99
        df.loc[50, "RMSE"] = 0.05
        df.loc[50, "n_vars"] = 2000  # Use ALL variables - shouldn't matter at penalty=0

        result = compute_composite_score(df, "regression", variable_penalty=0, complexity_penalty=0)

        # Best performing model should be rank 1
        best_model_rank = result.loc[50, "Rank"]
        assert best_model_rank == 1, f"Best model ranked #{best_model_rank}, expected #1"

    def test_penalty_two_favors_performance_over_simplicity(self):
        """At penalty=2 (low), high-performance models should rank well even with many variables."""
        df = self.create_test_data_regression(100)

        # Model A: Excellent performance, many variables
        df.loc[10, "R2"] = 0.95
        df.loc[10, "RMSE"] = 0.08
        df.loc[10, "n_vars"] = 2000

        # Model B: Good performance, few variables
        df.loc[20, "R2"] = 0.85
        df.loc[20, "RMSE"] = 0.15
        df.loc[20, "n_vars"] = 50

        result = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=2)

        # Model A (better performance) should rank higher than Model B
        rank_a = result.loc[10, "Rank"]
        rank_b = result.loc[20, "Rank"]
        assert rank_a < rank_b, f"High-performance model A ranked #{rank_a}, should beat simpler model B ranked #{rank_b}"

    def test_penalty_ten_favors_simplicity_strongly(self):
        """At penalty=10 (high), simple models should be strongly favored."""
        df = self.create_test_data_regression(100)

        # Model A: Excellent performance, many variables
        df.loc[10, "R2"] = 0.92
        df.loc[10, "RMSE"] = 0.10
        df.loc[10, "n_vars"] = 2000

        # Model B: Slightly worse performance, very few variables
        df.loc[20, "R2"] = 0.88
        df.loc[20, "RMSE"] = 0.12
        df.loc[20, "n_vars"] = 20

        result = compute_composite_score(df, "regression", variable_penalty=10, complexity_penalty=10)

        # Model B (simpler) should rank higher than Model A
        rank_a = result.loc[10, "Rank"]
        rank_b = result.loc[20, "Rank"]
        assert rank_b < rank_a, f"Simple model B ranked #{rank_b}, should beat complex model A ranked #{rank_a} at penalty=10"

    def test_quadratic_penalty_scaling(self):
        """Verify that penalty scaling is quadratic, not linear."""
        df = self.create_test_data_regression(50)

        # Add two models with same performance, different variable counts
        df.loc[10, "R2"] = 0.90
        df.loc[10, "RMSE"] = 0.10
        df.loc[10, "n_vars"] = 100

        df.loc[20, "R2"] = 0.90
        df.loc[20, "RMSE"] = 0.10
        df.loc[20, "n_vars"] = 2000

        # At penalty=2, impact should be small (quadratic scaling)
        result_p2 = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=0)
        score_diff_p2 = abs(result_p2.loc[20, "CompositeScore"] - result_p2.loc[10, "CompositeScore"])

        # At penalty=10, impact should be much larger
        result_p10 = compute_composite_score(df, "regression", variable_penalty=10, complexity_penalty=0)
        score_diff_p10 = abs(result_p10.loc[20, "CompositeScore"] - result_p10.loc[10, "CompositeScore"])

        # Ratio should be approximately (10/2)^2 = 25
        ratio = score_diff_p10 / score_diff_p2
        assert 20 < ratio < 30, f"Penalty scaling ratio {ratio:.1f} should be ~25 (quadratic)"

    def test_regression_user_bug_scenario(self):
        """Reproduce the user's bug: R²=0.943 model ranked #876."""
        # Simulate 876 models like user's dataset
        df = self.create_test_data_regression(876)

        # Add the user's best model (by R²)
        df.loc[500, "R2"] = 0.943
        df.loc[500, "RMSE"] = 0.10
        df.loc[500, "n_vars"] = 2000  # Using many variables

        # Add hundreds of slightly worse models with fewer variables
        for i in range(50, 150):
            df.loc[i, "R2"] = np.random.uniform(0.88, 0.92)
            df.loc[i, "RMSE"] = np.random.uniform(0.11, 0.15)
            df.loc[i, "n_vars"] = np.random.randint(20, 200)

        # With the FIX and penalty=2, best R² model should rank well
        result = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=2)

        best_r2_rank = result.loc[500, "Rank"]

        # With the fix, this should rank in top 50 (ideally top 10)
        assert best_r2_rank <= 50, (
            f"Model with R²=0.943 ranked #{best_r2_rank}, "
            f"should be in top 50 at penalty=2"
        )

    def test_ranking_is_stable(self):
        """Verify ranking is deterministic and stable."""
        df = self.create_test_data_regression(50)

        result1 = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=2)
        result2 = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=2)

        # Rankings should be identical
        assert result1["Rank"].equals(result2["Rank"]), "Ranking should be deterministic"

    def test_no_rank_ties_unless_identical_scores(self):
        """Verify rank() uses method='min' correctly."""
        df = self.create_test_data_regression(10)

        # Create two models with identical scores
        df.loc[0, "R2"] = 0.90
        df.loc[0, "RMSE"] = 0.10
        df.loc[0, "n_vars"] = 100
        df.loc[0, "LVs"] = 10

        df.loc[1, "R2"] = 0.90
        df.loc[1, "RMSE"] = 0.10
        df.loc[1, "n_vars"] = 100
        df.loc[1, "LVs"] = 10

        result = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=2)

        # These two should have same rank (method='min')
        assert result.loc[0, "Rank"] == result.loc[1, "Rank"], "Identical models should have same rank"

    def test_classification_scoring(self):
        """Test that classification scoring works correctly."""
        df = pd.DataFrame({
            "Model": ["RandomForest"] * 10,
            "Accuracy": np.random.uniform(0.7, 0.95, 10),
            "ROC_AUC": np.random.uniform(0.75, 0.98, 10),
            "n_vars": np.random.randint(10, 100, 10),
            "full_vars": [2151] * 10,
            "LVs": [0] * 10,
            "Params": ["{}"] * 10,
            "Preprocess": ["raw"] * 10,
            "Deriv": [0] * 10,
            "Window": [0] * 10,
            "Poly": [0] * 10,
            "SubsetTag": ["full"] * 10,
            "top_vars": [None] * 10,
        })

        # Best model: highest AUC and accuracy
        df.loc[5, "ROC_AUC"] = 0.99
        df.loc[5, "Accuracy"] = 0.96

        result = compute_composite_score(df, "classification", variable_penalty=0, complexity_penalty=0)

        # Should rank #1
        assert result.loc[5, "Rank"] == 1, "Best classification model should rank #1"

    def test_complexity_penalty_affects_lvs(self):
        """Test that complexity penalty affects models with many LVs."""
        df = self.create_test_data_regression(50)

        # Two models with same performance, different LVs
        df.loc[10, "R2"] = 0.90
        df.loc[10, "RMSE"] = 0.10
        df.loc[10, "LVs"] = 5

        df.loc[20, "R2"] = 0.90
        df.loc[20, "RMSE"] = 0.10
        df.loc[20, "LVs"] = 20

        # At complexity_penalty=10, model with fewer LVs should rank better
        result = compute_composite_score(df, "regression", variable_penalty=0, complexity_penalty=10)

        rank_low_lv = result.loc[10, "Rank"]
        rank_high_lv = result.loc[20, "Rank"]

        assert rank_low_lv < rank_high_lv, "Model with fewer LVs should rank better at high complexity penalty"

    def test_nan_handling(self):
        """Test that NaN values in metrics are handled correctly."""
        df = self.create_test_data_regression(10)

        # Add a model with NaN R²
        df.loc[5, "R2"] = np.nan
        df.loc[5, "RMSE"] = 0.15

        # Should not crash
        result = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=2)

        # Model with NaN should still get a rank
        assert pd.notna(result.loc[5, "Rank"]), "Models with NaN metrics should still be ranked"

    def test_column_order(self):
        """Test that output has Rank as first column."""
        df = self.create_test_data_regression(10)
        result = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=2)

        assert result.columns[0] == "Rank", "Rank should be first column"
        assert result.columns[-1] == "top_vars", "top_vars should be last column"

    def test_complexity_score_added(self):
        """Test that ComplexityScore column is added."""
        df = self.create_test_data_regression(10)
        result = compute_composite_score(df, "regression", variable_penalty=2, complexity_penalty=2)

        assert "ComplexityScore" in result.columns, "ComplexityScore should be added"
        assert result["ComplexityScore"].notna().all(), "ComplexityScore should have values"


class TestPenaltyBehavior:
    """Test penalty scaling behavior in detail."""

    def test_penalty_zero_no_impact(self):
        """At penalty=0, penalties should have zero impact."""
        # Create two models: identical performance, different complexity
        df = pd.DataFrame({
            "Model": ["PLS", "PLS"],
            "RMSE": [0.10, 0.10],
            "R2": [0.90, 0.90],
            "n_vars": [10, 2000],
            "full_vars": [2151, 2151],
            "LVs": [5, 20],
            "Params": ["{}", "{}"],
            "Preprocess": ["raw", "raw"],
            "Deriv": [0, 0],
            "Window": [0, 0],
            "Poly": [0, 0],
            "SubsetTag": ["full", "full"],
            "top_vars": [None, None],
        })

        result = compute_composite_score(df, "regression", variable_penalty=0, complexity_penalty=0)

        # Scores should be identical
        assert abs(result.loc[0, "CompositeScore"] - result.loc[1, "CompositeScore"]) < 1e-10, (
            "At penalty=0, models with same performance should have identical scores"
        )

    def test_penalty_scaling_smoothness(self):
        """Verify that penalty scaling is smooth from 0 to 10."""
        df = pd.DataFrame({
            "Model": ["PLS", "PLS"],
            "RMSE": [0.10, 0.10],
            "R2": [0.90, 0.90],
            "n_vars": [100, 2000],
            "full_vars": [2151, 2151],
            "LVs": [5, 5],
            "Params": ["{}", "{}"],
            "Preprocess": ["raw", "raw"],
            "Deriv": [0, 0],
            "Window": [0, 0],
            "Poly": [0, 0],
            "SubsetTag": ["full", "full"],
            "top_vars": [None, None],
        })

        penalty_impacts = []
        for penalty in range(0, 11):
            result = compute_composite_score(df, "regression", variable_penalty=penalty, complexity_penalty=0)
            impact = abs(result.loc[1, "CompositeScore"] - result.loc[0, "CompositeScore"])
            penalty_impacts.append(impact)

        # Impacts should increase monotonically
        for i in range(1, len(penalty_impacts)):
            assert penalty_impacts[i] >= penalty_impacts[i-1], (
                f"Penalty impact should increase monotonically: "
                f"penalty={i-1} impact={penalty_impacts[i-1]:.4f}, "
                f"penalty={i} impact={penalty_impacts[i]:.4f}"
            )

        # Impact at penalty=0 should be essentially zero
        assert penalty_impacts[0] < 1e-10, "Impact at penalty=0 should be zero"

        # Impact at penalty=10 should be significant
        assert penalty_impacts[10] > 0.5, "Impact at penalty=10 should be significant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
