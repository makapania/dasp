"""Tests for the T-31 multi-class SIMCA search + task-type plumbing (Phase C).

C1: `multiclass_simca` is threaded through the results/scoring branch sites and
never falls through to the classification/regression path (spec §7 / §9.12).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectral_predict.scoring import (
    compute_composite_score,
    create_results_dataframe,
)


class TestTaskTypePlumbingC1:
    def test_multiclass_results_schema_is_distinct(self):
        df = create_results_dataframe("multiclass_simca")
        cols = set(df.columns)
        # dedicated class-modeling metric + the per-row tags (spec decision #3)
        assert "NoveltyAUC" in cols
        assert "engine_family" in cols
        assert "varsel_path" in cols
        # NOT silently the classification or one-class schema
        assert cols != set(create_results_dataframe("classification").columns)
        assert cols != set(create_results_dataframe("one_class").columns)
        # and it must NOT carry the classification-only Accuracycv column
        assert "Accuracycv" not in cols

    def test_composite_score_ranks_by_novelty_auc(self):
        df = create_results_dataframe("multiclass_simca")
        rows = [
            {"Model": "pca-simca", "NoveltyAUC": 0.60, "MinClassN": 30,
             "n_vars": 20, "full_vars": 20},
            {"Model": "pca-simca", "NoveltyAUC": 0.90, "MinClassN": 30,
             "n_vars": 20, "full_vars": 20},
            {"Model": "ocsvm", "NoveltyAUC": 0.75, "MinClassN": 30,
             "n_vars": 20, "full_vars": 20},
        ]
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        scored = compute_composite_score(df, "multiclass_simca")
        best = scored.sort_values("Rank").iloc[0]
        assert best["NoveltyAUC"] == pytest.approx(0.90)
        assert best["Rank"] == 1

    def test_composite_score_does_not_fall_through_to_classification(self):
        # The multiclass schema has NO Accuracycv column; if scoring fell through
        # to the classification branch it would KeyError. Success proves an
        # explicit multiclass_simca branch (spec §9.12 fall-through guard).
        df = create_results_dataframe("multiclass_simca")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [{"Model": "pca-simca", "NoveltyAUC": 0.8, "MinClassN": 25,
                      "n_vars": 15, "full_vars": 15}]
                ),
            ],
            ignore_index=True,
        )
        scored = compute_composite_score(df, "multiclass_simca")  # must not raise
        assert "CompositeScore" in scored.columns
        assert np.isfinite(scored["CompositeScore"]).all()

    def test_supported_models_returns_multiclass_engines(self):
        from spectral_predict.model_registry import (
            MULTICLASS_ENGINES,
            get_supported_models,
        )

        engines = get_supported_models("multiclass_simca")
        assert engines == MULTICLASS_ENGINES
        assert "pca-simca" in engines
        # distinct from the one-class registry names (PCA-SIMCA vs pca-simca)
        assert engines != get_supported_models("one_class")

    def test_composite_score_tiebreak_prefers_larger_min_class_n(self):
        # Equal NoveltyAUC -> the row with the larger smallest-class n ranks first
        # (spec §7 tie-break: min per-class n).
        df = create_results_dataframe("multiclass_simca")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {"Model": "a", "NoveltyAUC": 0.8, "MinClassN": 12,
                         "n_vars": 20, "full_vars": 20},
                        {"Model": "b", "NoveltyAUC": 0.8, "MinClassN": 60,
                         "n_vars": 20, "full_vars": 20},
                    ]
                ),
            ],
            ignore_index=True,
        )
        scored = compute_composite_score(df, "multiclass_simca")
        assert scored.sort_values("Rank").iloc[0]["Model"] == "b"
