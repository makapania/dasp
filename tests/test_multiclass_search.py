"""Tests for the T-31 multi-class SIMCA search + task-type plumbing (Phase C).

C1: `multiclass_simca` is threaded through the results/scoring branch sites and
never falls through to the classification/regression path (spec §7 / §9.12).
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from spectral_predict.scoring import (
    compute_composite_score,
    create_results_dataframe,
)


def _graded(sizes, n_features=24, n_relevant=6, seed=0, sep=14.0):
    """K classes with graded between-class separation + shared within-class
    latent structure (copied from tests/test_simca.py). Gives a genuine,
    recoverable SIMCA separation so LOCO NoveltyAUC is meaningful."""
    rng = np.random.RandomState(seed)
    strength = np.geomspace(sep, sep / 8.0, n_features)
    loading = rng.normal(size=n_features)
    Xs, ys = [], []
    for k, n in enumerate(sizes):
        X = rng.normal(scale=0.3, size=(n, n_features))
        t = rng.normal(size=(n, 1))
        X += t * loading
        X += (k + 1) * strength
        Xs.append(X)
        ys.append(np.full(n, k))
    relevant = np.argsort(strength)[::-1][:n_relevant]
    return np.vstack(Xs), np.concatenate(ys), relevant


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


class TestMulticlassSearchC2:
    """C2: run_multiclass_simca_search — grid = preprocessing × engines ×
    varsel_paths (NO G^K product); per-class n_components auto-tuned inside each
    row; ranked by the LOCO NoveltyAUC; unmodelable classes flagged not dropped
    (spec §7 / §8)."""

    def _run(self, **kw):
        from spectral_predict.search import run_multiclass_simca_search

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_multiclass_simca_search(**kw)

    def test_grid_row_count_no_gk_blowup(self):
        X, y, _ = _graded([60, 60, 60], seed=0)
        df = self._run(
            X=X, y=y,
            engines=["pca-simca", "ocsvm"],
            varsel_paths=["none", "wold_balanced"],
            preprocessing_methods=["raw", "snv"],
            min_class_samples=10,
            cv_splits=4,
        )
        # 2 preproc × 2 engines × 2 varsel = 8 — NOT a G^K per-class blowup
        assert len(df) == 8
        for col in [
            "Task", "Model", "NoveltyAUC", "Efficiency", "MinClassN",
            "n_classes", "Alpha", "engine_family", "varsel_path", "Rank",
        ]:
            assert col in df.columns
        assert (df["Task"] == "multiclass_simca").all()
        # Rank present on every row; min==1 (best). Ties keep method="min"
        # duplicate ranks, so ranks are not necessarily a 1..n permutation.
        assert df["Rank"].notna().all()
        assert int(df["Rank"].min()) == 1
        assert int(df["Rank"].max()) <= 8

    def test_engine_family_and_varsel_path_on_every_row(self):
        X, y, _ = _graded([60, 60, 60], seed=1)
        engines = ["pca-simca", "ocsvm"]
        varsel = ["none", "wold_modeling"]
        df = self._run(
            X=X, y=y, engines=engines, varsel_paths=varsel,
            preprocessing_methods=["raw", "snv"], cv_splits=4,
        )
        assert set(df["engine_family"]) <= set(engines)
        assert set(df["varsel_path"]) <= set(varsel)
        # exact product: each (engine, varsel_path) pair appears n_preproc(=2) times
        combos = Counter(zip(df["engine_family"], df["varsel_path"]))
        assert len(combos) == 4
        assert all(v == 2 for v in combos.values())
        assert (df["engine_family"] == df["Model"]).all()

    def test_per_class_ncomponents_tuned_inside_row(self):
        X, y, _ = _graded([60, 60, 60], seed=2)
        df = self._run(
            X=X, y=y, engines=["pca-simca"], varsel_paths=["none"],
            preprocessing_methods=["raw"], cv_splits=4,
        )
        # 1 preproc × 1 engine × 1 varsel: n_components is NOT a grid axis
        assert len(df) == 1
        lvs = str(df.iloc[0]["LVs"])
        # a per-class {class: n_components} dict (A5 tuning), not one global value
        assert "{" in lvs and "}" in lvs

    def test_ranked_by_novelty_auc_nan_last(self):
        # pca-simca models fine; elliptic-envelope classes are all n<20 ->
        # every class unmodelable -> whole config fails -> NaN NoveltyAUC row.
        # It must SORT LAST (not crash, not rank first).
        X, y, _ = _graded([15, 15, 15], seed=3)
        df = self._run(
            X=X, y=y,
            engines=["pca-simca", "elliptic-envelope"],
            varsel_paths=["none"],
            preprocessing_methods=["raw"],
            min_class_samples=10, cv_splits=3,
        )
        assert len(df) == 2
        finite = df[df["NoveltyAUC"].notna()]
        nan_rows = df[df["NoveltyAUC"].isna()]
        assert len(finite) >= 1 and len(nan_rows) >= 1
        # every finite-AUC row ranks strictly before every NaN-AUC row
        assert finite["Rank"].max() < nan_rows["Rank"].min()
        best = df.sort_values("Rank").iloc[0]
        assert best["NoveltyAUC"] == pytest.approx(finite["NoveltyAUC"].max())

    def test_unmodelable_class_flagged_not_dropped(self):
        # class 2 has n=8 < min_class_samples=10 -> unmodelable, but the config
        # row is still emitted, all 3 classes preserved, the class flagged.
        X, y, _ = _graded([40, 40, 8], seed=4)
        df = self._run(
            X=X, y=y, engines=["pca-simca"], varsel_paths=["none"],
            preprocessing_methods=["raw"], min_class_samples=10, cv_splits=3,
        )
        assert len(df) == 1
        row = df.iloc[0]
        assert int(row["n_classes"]) == 3  # none dropped
        # smallest MODELED class n is 40 (class 2 excluded from the modeled set)
        assert int(row["MinClassN"]) == 40
        # the unmodelable class is recorded on the row (flagged, not dropped)
        assert str(row["unmodelable_classes"]) not in ("", "nan", "[]", "None")
