"""
Test to verify ranking behavior with CompositeScore.

Validates that pandas .rank() with default ascending=True correctly assigns
Rank 1 to the lowest CompositeScore (since lower is better).
"""

import pandas as pd
import numpy as np
import pytest


class TestRankingBehavior:
    """Test ranking behavior for CompositeScore (lower is better)."""

    def test_best_model_gets_rank_1(self):
        """Verify that the model with lowest CompositeScore gets Rank 1."""
        data = {
            'Model': ['ModelA', 'ModelB', 'ModelC', 'ModelD'],
            'R2': [0.95, 0.85, 0.75, 0.65],
            'RMSE': [0.10, 0.20, 0.30, 0.40],
        }
        df = pd.DataFrame(data)

        # Compute z-scores
        z_rmse = (df["RMSE"] - df["RMSE"].mean()) / df["RMSE"].std()
        z_r2 = (df["R2"] - df["R2"].mean()) / df["R2"].std()

        # Performance score (lower is better)
        df["CompositeScore"] = 0.5 * z_rmse - 0.5 * z_r2

        # Default ascending=True assigns Rank 1 to lowest value
        df["Rank"] = df["CompositeScore"].rank(method="min")

        best_model_rank = df.loc[df['Model'] == 'ModelA', 'Rank'].values[0]
        assert best_model_rank == 1, (
            f"Best model (ModelA) should have Rank 1, got Rank {best_model_rank}"
        )

    def test_ascending_false_gives_wrong_ranking(self):
        """Verify that ascending=False would give WRONG ranking for lower-is-better."""
        data = {
            'Model': ['ModelA', 'ModelB', 'ModelC', 'ModelD'],
            'R2': [0.95, 0.85, 0.75, 0.65],
            'RMSE': [0.10, 0.20, 0.30, 0.40],
        }
        df = pd.DataFrame(data)

        z_rmse = (df["RMSE"] - df["RMSE"].mean()) / df["RMSE"].std()
        z_r2 = (df["R2"] - df["R2"].mean()) / df["R2"].std()
        df["CompositeScore"] = 0.5 * z_rmse - 0.5 * z_r2

        # ascending=False assigns Rank 1 to HIGHEST value (wrong for lower-is-better)
        df["Rank_desc"] = df["CompositeScore"].rank(method="min", ascending=False)

        best_model_rank = df.loc[df['Model'] == 'ModelA', 'Rank_desc'].values[0]
        assert best_model_rank != 1, (
            "ascending=False should NOT give Rank 1 to the best model"
        )

    def test_ranking_preserves_order(self):
        """Verify that ranking preserves the ordering of CompositeScore."""
        data = {
            'Model': ['ModelA', 'ModelB', 'ModelC', 'ModelD'],
            'R2': [0.95, 0.85, 0.75, 0.65],
            'RMSE': [0.10, 0.20, 0.30, 0.40],
        }
        df = pd.DataFrame(data)

        z_rmse = (df["RMSE"] - df["RMSE"].mean()) / df["RMSE"].std()
        z_r2 = (df["R2"] - df["R2"].mean()) / df["R2"].std()
        df["CompositeScore"] = 0.5 * z_rmse - 0.5 * z_r2

        df["Rank"] = df["CompositeScore"].rank(method="min")

        # Ranks should follow the ordering: A < B < C < D
        ranks = df.set_index('Model')['Rank']
        assert ranks['ModelA'] < ranks['ModelB'] < ranks['ModelC'] < ranks['ModelD']
