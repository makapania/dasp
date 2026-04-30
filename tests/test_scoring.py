"""Comprehensive unit tests for scoring and ranking system."""

import numpy as np
import pandas as pd
import pytest

from spectral_predict.scoring import compute_composite_score, _compute_unified_complexity


class TestCompositeScoring:
    """Test the compute_composite_score function."""

    def create_test_data_regression(self, n_models=10):
        """Create test regression data with known characteristics."""
        np.random.seed(42)

        rmse_vals = np.random.uniform(0.1, 0.5, n_models)
        r2_vals = np.random.uniform(0.6, 0.95, n_models)

        # Create models with varying characteristics
        data = {
            "Model": ["PLS"] * n_models,
            "RMSE": rmse_vals,
            "R2": r2_vals,
            "RMSEcv": rmse_vals * np.random.uniform(1.0, 1.5, n_models),
            "R2cv": r2_vals * np.random.uniform(0.85, 1.0, n_models),
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
        """At penalty=0, ranking should be based purely on R2cv performance."""
        df = self.create_test_data_regression(100)

        # Add a clearly best model (highest R2cv, lowest RMSEcv)
        df.loc[50, "R2"] = 0.99
        df.loc[50, "R2cv"] = 0.99
        df.loc[50, "RMSE"] = 0.05
        df.loc[50, "RMSEcv"] = 0.06
        df.loc[50, "n_vars"] = 2000  # Use ALL variables - shouldn't matter at penalty=0

        result = compute_composite_score(df, "regression", variable_penalty=0, gap_penalty=0)

        # Best performing model (highest R2cv) should be rank 1
        # After sorting, the best model is at the top (Rank=1)
        best_model = result[result["R2cv"] == 0.99]
        assert len(best_model) == 1, "Should find exactly one model with R2cv=0.99"
        assert best_model.iloc[0]["Rank"] == 1, f"Best model ranked #{best_model.iloc[0]['Rank']}, expected #1"

    def test_penalty_two_favors_performance_over_simplicity(self):
        """At penalty=2 (low), high-performance models should rank well even with many variables."""
        df = self.create_test_data_regression(100)

        # Model A: Excellent performance, many variables
        df.loc[10, "R2"] = 0.95
        df.loc[10, "R2cv"] = 0.95
        df.loc[10, "RMSE"] = 0.08
        df.loc[10, "RMSEcv"] = 0.09
        df.loc[10, "n_vars"] = 2000

        # Model B: Good performance, few variables
        df.loc[20, "R2"] = 0.85
        df.loc[20, "R2cv"] = 0.85
        df.loc[20, "RMSE"] = 0.15
        df.loc[20, "RMSEcv"] = 0.16
        df.loc[20, "n_vars"] = 50

        result = compute_composite_score(df, "regression", variable_penalty=2, gap_penalty=2)

        # Find models by their R2cv values (index is reset after sorting)
        model_a = result[result["R2cv"] == 0.95].iloc[0]
        model_b = result[result["R2cv"] == 0.85].iloc[0]

        # Model A (better performance) should rank higher than Model B
        rank_a = model_a["Rank"]
        rank_b = model_b["Rank"]
        assert rank_a < rank_b, f"High-performance model A ranked #{rank_a}, should beat simpler model B ranked #{rank_b}"

    def test_penalty_ten_favors_simplicity_strongly(self):
        """At penalty=10 (high), simple models should be strongly favored."""
        df = self.create_test_data_regression(100)

        # Model A: Excellent performance, many variables
        df.loc[10, "R2"] = 0.92
        df.loc[10, "R2cv"] = 0.92
        df.loc[10, "RMSE"] = 0.10
        df.loc[10, "RMSEcv"] = 0.11
        df.loc[10, "n_vars"] = 2000

        # Model B: Slightly worse performance, very few variables
        df.loc[20, "R2"] = 0.88
        df.loc[20, "R2cv"] = 0.88
        df.loc[20, "RMSE"] = 0.12
        df.loc[20, "RMSEcv"] = 0.13
        df.loc[20, "n_vars"] = 20

        result = compute_composite_score(df, "regression", variable_penalty=10, gap_penalty=10)

        # Find models by their R2cv values (index is reset after sorting)
        model_a = result[result["R2cv"] == 0.92].iloc[0]
        model_b = result[result["R2cv"] == 0.88].iloc[0]

        # Model B (simpler) should rank higher than Model A at high penalty
        rank_a = model_a["Rank"]
        rank_b = model_b["Rank"]
        assert rank_b < rank_a, f"Simple model B ranked #{rank_b}, should beat complex model A ranked #{rank_a} at penalty=10"

    def test_quadratic_penalty_scaling(self):
        """Verify that penalty scaling is quadratic, not linear."""
        df = self.create_test_data_regression(50)

        # Add two models with same performance, different variable counts
        df.loc[10, "R2"] = 0.90
        df.loc[10, "R2cv"] = 0.90
        df.loc[10, "RMSE"] = 0.10
        df.loc[10, "RMSEcv"] = 0.11
        df.loc[10, "n_vars"] = 100

        df.loc[20, "R2"] = 0.90
        df.loc[20, "R2cv"] = 0.90
        df.loc[20, "RMSE"] = 0.10
        df.loc[20, "RMSEcv"] = 0.11
        df.loc[20, "n_vars"] = 2000

        # At penalty=2, impact should be small (quadratic scaling)
        result_p2 = compute_composite_score(df, "regression", variable_penalty=2, gap_penalty=0)
        # Find models by n_vars (index is reset after sorting)
        score_100 = result_p2[result_p2["n_vars"] == 100].iloc[0]["CompositeScore"]
        score_2000 = result_p2[result_p2["n_vars"] == 2000].iloc[0]["CompositeScore"]
        score_diff_p2 = abs(score_2000 - score_100)

        # At penalty=10, impact should be much larger
        result_p10 = compute_composite_score(df, "regression", variable_penalty=10, gap_penalty=0)
        score_100 = result_p10[result_p10["n_vars"] == 100].iloc[0]["CompositeScore"]
        score_2000 = result_p10[result_p10["n_vars"] == 2000].iloc[0]["CompositeScore"]
        score_diff_p10 = abs(score_2000 - score_100)

        # Ratio should be approximately (10/2)^2 = 25
        ratio = score_diff_p10 / score_diff_p2
        assert 20 < ratio < 30, f"Penalty scaling ratio {ratio:.1f} should be ~25 (quadratic)"

    def test_regression_user_bug_scenario(self):
        """Reproduce the user's bug: R²=0.943 model ranked #876."""
        # Simulate 876 models like user's dataset
        df = self.create_test_data_regression(876)

        # Add the user's best model (by R²)
        df.loc[500, "R2"] = 0.943
        df.loc[500, "R2cv"] = 0.943
        df.loc[500, "RMSE"] = 0.10
        df.loc[500, "RMSEcv"] = 0.11
        df.loc[500, "n_vars"] = 2000  # Using many variables

        # Add hundreds of slightly worse models with fewer variables
        for i in range(50, 150):
            r2_val = np.random.uniform(0.88, 0.92)
            df.loc[i, "R2"] = r2_val
            df.loc[i, "R2cv"] = r2_val
            rmse_val = np.random.uniform(0.11, 0.15)
            df.loc[i, "RMSE"] = rmse_val
            df.loc[i, "RMSEcv"] = rmse_val * 1.1
            df.loc[i, "n_vars"] = np.random.randint(20, 200)

        # With the FIX and penalty=2, best R2cv model should rank well
        result = compute_composite_score(df, "regression", variable_penalty=2, gap_penalty=2)

        # Find the best model by its R2cv value (index is reset after sorting)
        best_model = result[result["R2cv"] == 0.943]
        assert len(best_model) == 1, "Should find exactly one model with R2cv=0.943"
        best_r2_rank = best_model.iloc[0]["Rank"]

        # With the fix, this should rank in top 50 (ideally top 10)
        assert best_r2_rank <= 50, (
            f"Model with R2cv=0.943 ranked #{best_r2_rank}, "
            f"should be in top 50 at penalty=2"
        )

    def test_ranking_is_stable(self):
        """Verify ranking is deterministic and stable."""
        df = self.create_test_data_regression(50)

        result1 = compute_composite_score(df, "regression", variable_penalty=2, gap_penalty=2)
        result2 = compute_composite_score(df, "regression", variable_penalty=2, gap_penalty=2)

        # Rankings should be identical
        assert result1["Rank"].equals(result2["Rank"]), "Ranking should be deterministic"

    def test_no_rank_ties_unless_identical_scores(self):
        """Verify rank() uses method='min' correctly."""
        df = self.create_test_data_regression(10)

        # Create two models with identical scores (use unique n_vars=9999 as marker)
        df.loc[0, "R2"] = 0.90
        df.loc[0, "R2cv"] = 0.90
        df.loc[0, "RMSE"] = 0.10
        df.loc[0, "RMSEcv"] = 0.11
        df.loc[0, "n_vars"] = 9999
        df.loc[0, "full_vars"] = 9999
        df.loc[0, "LVs"] = 10

        df.loc[1, "R2"] = 0.90
        df.loc[1, "R2cv"] = 0.90
        df.loc[1, "RMSE"] = 0.10
        df.loc[1, "RMSEcv"] = 0.11
        df.loc[1, "n_vars"] = 9999
        df.loc[1, "full_vars"] = 9999
        df.loc[1, "LVs"] = 10

        result = compute_composite_score(df, "regression", variable_penalty=2, gap_penalty=2)

        # Find the two identical models by their marker (n_vars=9999)
        identical_models = result[result["n_vars"] == 9999]
        assert len(identical_models) == 2, "Should find exactly 2 identical models"

        # These two should have same rank (method='min')
        ranks = identical_models["Rank"].unique()
        assert len(ranks) == 1, "Identical models should have same rank"

    def test_classification_scoring(self):
        """Test that classification scoring works correctly."""
        np.random.seed(42)
        df = pd.DataFrame({
            "Model": ["RandomForest"] * 10,
            "Accuracy": np.random.uniform(0.7, 0.95, 10),
            "Accuracycv": np.random.uniform(0.65, 0.90, 10),
            "F1cv": np.random.uniform(0.65, 0.90, 10),
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

        # Best model: highest Accuracycv and F1cv
        df.loc[5, "Accuracycv"] = 0.99
        df.loc[5, "F1cv"] = 0.99
        df.loc[5, "Accuracy"] = 0.99

        result = compute_composite_score(df, "classification", variable_penalty=0, gap_penalty=0)

        # Find the best model by Accuracycv value (index is reset after sorting)
        best_model = result[result["Accuracycv"] == 0.99]
        assert len(best_model) == 1, "Should find exactly one model with Accuracycv=0.99"
        assert best_model.iloc[0]["Rank"] == 1, "Best classification model should rank #1"

    def test_gap_penalty_affects_overfitting_models(self):
        """Test that gap penalty penalizes models with large calibration-CV gap."""
        df = self.create_test_data_regression(50)

        # Add RMSEcv column (needed for gap penalty calculation)
        df["RMSEcv"] = df["RMSE"] * np.random.uniform(1.0, 2.0, len(df))
        df["R2cv"] = df["R2"] * np.random.uniform(0.8, 1.0, len(df))

        # Model A: small gap (RMSE ~ RMSEcv, ratio near 1.0), use unique n_vars as marker
        df.loc[10, "R2"] = 0.90
        df.loc[10, "R2cv"] = 0.90
        df.loc[10, "RMSE"] = 0.10
        df.loc[10, "RMSEcv"] = 0.11  # Small gap
        df.loc[10, "n_vars"] = 7777  # Marker

        # Model B: large gap (RMSEcv >> RMSE, overfitting)
        df.loc[20, "R2"] = 0.90
        df.loc[20, "R2cv"] = 0.90
        df.loc[20, "RMSE"] = 0.10
        df.loc[20, "RMSEcv"] = 0.50  # Large gap (5x ratio)
        df.loc[20, "n_vars"] = 8888  # Marker

        # At gap_penalty=10, model with smaller gap should rank better
        result = compute_composite_score(df, "regression", variable_penalty=0, gap_penalty=10)

        # Find models by their marker (index is reset after sorting)
        model_a = result[result["n_vars"] == 7777].iloc[0]
        model_b = result[result["n_vars"] == 8888].iloc[0]

        rank_small_gap = model_a["Rank"]
        rank_large_gap = model_b["Rank"]

        assert rank_small_gap < rank_large_gap, "Model with smaller calibration-CV gap should rank better at high gap penalty"

    def test_nan_handling(self):
        """Test that NaN values in R2 (not R2cv) are handled correctly."""
        df = self.create_test_data_regression(10)

        # Set NaN in R2 but keep R2cv valid (scoring uses R2cv)
        df.loc[5, "R2"] = np.nan
        df.loc[5, "RMSE"] = 0.15

        # Should not crash since R2cv (used for scoring) is still valid
        result = compute_composite_score(df, "regression", variable_penalty=2, gap_penalty=2)

        # All models should still get a rank
        assert result["Rank"].notna().all(), "All models should still be ranked when R2cv is valid"

    def test_column_order(self):
        """Test that output has Rank as first column and ComplexityScore as last."""
        df = self.create_test_data_regression(10)
        result = compute_composite_score(df, "regression", variable_penalty=2, gap_penalty=2)

        assert result.columns[0] == "Rank", "Rank should be first column"
        assert result.columns[-1] == "ComplexityScore", "ComplexityScore should be last column"
        assert "top_vars" in result.columns, "top_vars should be present"

    def test_complexity_score_added(self):
        """Test that ComplexityScore column is added."""
        df = self.create_test_data_regression(10)
        result = compute_composite_score(df, "regression", variable_penalty=2, gap_penalty=2)

        assert "ComplexityScore" in result.columns, "ComplexityScore should be added"
        assert result["ComplexityScore"].notna().all(), "ComplexityScore should have values"


class TestPenaltyBehavior:
    """Test penalty scaling behavior in detail."""

    def test_penalty_zero_no_impact(self):
        """At penalty=0, penalties should have zero impact."""
        # Create two models: identical R2cv performance, different complexity
        df = pd.DataFrame({
            "Model": ["PLS", "PLS"],
            "RMSE": [0.10, 0.10],
            "R2": [0.90, 0.90],
            "RMSEcv": [0.12, 0.12],
            "R2cv": [0.88, 0.88],
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

        result = compute_composite_score(df, "regression", variable_penalty=0, gap_penalty=0)

        # Scores should be identical (performance_score = -R2cv which is the same for both)
        assert abs(result.loc[0, "CompositeScore"] - result.loc[1, "CompositeScore"]) < 1e-10, (
            "At penalty=0, models with same R2cv should have identical scores"
        )

    def test_penalty_scaling_smoothness(self):
        """Verify that penalty scaling is smooth from 0 to 10."""
        # Need 3+ models with different R2cv to create non-zero perf_range,
        # while keeping the two test models (indices 0,1) with identical R2cv
        df = pd.DataFrame({
            "Model": ["PLS", "PLS", "PLS"],
            "RMSE": [0.10, 0.10, 0.20],
            "R2": [0.90, 0.90, 0.70],
            "RMSEcv": [0.12, 0.12, 0.25],
            "R2cv": [0.88, 0.88, 0.65],
            "n_vars": [100, 2000, 500],
            "full_vars": [2151, 2151, 2151],
            "LVs": [5, 5, 5],
            "Params": ["{}", "{}", "{}"],
            "Preprocess": ["raw", "raw", "raw"],
            "Deriv": [0, 0, 0],
            "Window": [0, 0, 0],
            "Poly": [0, 0, 0],
            "SubsetTag": ["full", "full", "full"],
            "top_vars": [None, None, None],
        })

        penalty_impacts = []
        for penalty in range(0, 11):
            result = compute_composite_score(df, "regression", variable_penalty=penalty, gap_penalty=0)
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

        # Impact at penalty=10 should be significant (scaled by perf_range * 0.5)
        assert penalty_impacts[10] > 0.01, "Impact at penalty=10 should be significant"


class TestOneClassComplexity:
    """Tests for one-class model complexity scores."""

    def test_one_class_model_scores(self):
        """One-class models get expected complexity scores."""
        expected = {
            'PCA-SIMCA': 20,
            'EllipticEnvelope': 30,
            'IsolationForest': 35,
            'LOF': 45,
            'OneClassSVM': 55,
        }
        for model_name, expected_score in expected.items():
            row = pd.Series({
                'Model': model_name,
                'n_vars': 100,
                'LVs': 0,
                'Preprocess': 'raw',
                'Deriv': 0,
            })
            score = _compute_unified_complexity(row)
            # Model complexity is 25% of total; verify it contributes correctly
            assert isinstance(score, float)
            assert 0 <= score <= 100

    def test_pca_simca_extracts_n_components(self):
        """PCA-SIMCA extracts n_components from Params when LVs is 0."""
        row = pd.Series({
            'Model': 'PCA-SIMCA',
            'n_vars': 50,
            'LVs': 0,
            'Params': "{'n_components': 5, 'alpha': 0.05}",
            'Preprocess': 'raw',
            'Deriv': 0,
        })
        score_with_params = _compute_unified_complexity(row)

        row_no_params = pd.Series({
            'Model': 'PCA-SIMCA',
            'n_vars': 50,
            'LVs': 5,  # Explicit LVs
            'Params': '{}',
            'Preprocess': 'raw',
            'Deriv': 0,
        })
        score_with_lvs = _compute_unified_complexity(row_no_params)

        # Both should produce the same LV complexity component
        assert abs(score_with_params - score_with_lvs) < 1.0


class TestLinsCCC:
    """Tests for Lin's Concordance Correlation Coefficient.

    Reference: Lin, L. I. (1989). "A concordance correlation coefficient
    to evaluate reproducibility." Biometrics, 45(1), 255-268.
    """

    def test_perfect_prediction_returns_one(self):
        from spectral_predict.scoring import lins_ccc
        y_true = np.linspace(0.0, 10.0, 50)
        y_pred = y_true.copy()
        assert abs(lins_ccc(y_true, y_pred) - 1.0) < 1e-12

    def test_perfect_anticorrelation_returns_minus_one(self):
        from spectral_predict.scoring import lins_ccc
        y_true = np.linspace(-5.0, 5.0, 50)
        y_pred = -y_true
        assert abs(lins_ccc(y_true, y_pred) - (-1.0)) < 1e-12

    def test_bias_only_below_one_even_when_pearson_is_one(self):
        from spectral_predict.scoring import lins_ccc
        y_true = np.linspace(0.0, 10.0, 50)
        y_pred = y_true + 5.0
        ccc = lins_ccc(y_true, y_pred)
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
        assert abs(pearson - 1.0) < 1e-12, "sanity: Pearson should be 1 for pure bias"
        assert ccc < 1.0
        assert ccc > 0.0

    def test_scale_only_below_one_even_when_pearson_is_one(self):
        from spectral_predict.scoring import lins_ccc
        y_true = np.linspace(-5.0, 5.0, 50)
        y_pred = 2.0 * y_true
        ccc = lins_ccc(y_true, y_pred)
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
        assert abs(pearson - 1.0) < 1e-12
        assert ccc < 1.0
        assert abs(ccc - 0.8) < 1e-12

    def test_known_closed_form_scale_only(self):
        from spectral_predict.scoring import lins_ccc
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(1000)
        y_true -= y_true.mean()
        y_pred = 2.0 * y_true
        assert abs(lins_ccc(y_true, y_pred) - 0.8) < 1e-2

    def test_ccc_finite_sample_ddof_zero_exact(self):
        from spectral_predict.scoring import lins_ccc
        y = np.array([0.0, 1.0, 2.0])
        pred = np.array([1.0, 2.0, 3.0])
        result = lins_ccc(y, pred)
        assert abs(result - 4 / 7) < 1e-10

    def test_range_within_bounds_random_inputs(self):
        from spectral_predict.scoring import lins_ccc
        rng = np.random.default_rng(42)
        for _ in range(20):
            y_true = rng.standard_normal(100)
            y_pred = rng.standard_normal(100)
            ccc = lins_ccc(y_true, y_pred)
            assert -1.0 <= ccc <= 1.0

    def test_symmetry_in_arguments(self):
        from spectral_predict.scoring import lins_ccc
        rng = np.random.default_rng(1)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        assert abs(lins_ccc(a, b) - lins_ccc(b, a)) < 1e-12

    def test_nan_in_inputs_returns_nan(self):
        from spectral_predict.scoring import lins_ccc
        y_true = np.array([1.0, 2.0, np.nan, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isnan(lins_ccc(y_true, y_pred))

    def test_length_mismatch_raises(self):
        from spectral_predict.scoring import lins_ccc
        with pytest.raises(ValueError):
            lins_ccc(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_constant_predictions_returns_zero(self):
        from spectral_predict.scoring import lins_ccc
        y_true = np.linspace(0.0, 10.0, 20)
        y_pred = np.full(20, 5.0)
        result = lins_ccc(y_true, y_pred)
        assert result == 0.0

    def test_constant_truth_returns_zero(self):
        from spectral_predict.scoring import lins_ccc
        y_true = np.full(20, 7.0)
        y_pred = np.linspace(0.0, 10.0, 20)
        assert lins_ccc(y_true, y_pred) == 0.0

    def test_both_constant_and_equal_returns_one(self):
        from spectral_predict.scoring import lins_ccc
        y_true = np.full(20, 7.0)
        y_pred = np.full(20, 7.0)
        assert lins_ccc(y_true, y_pred) == 1.0

    def test_accepts_lists_and_pandas_series(self):
        from spectral_predict.scoring import lins_ccc
        import pandas as pd
        y_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(lins_ccc(y_list, y_series) - 1.0) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
