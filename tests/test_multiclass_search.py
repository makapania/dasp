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


class _FakeLOCOModel:
    """Minimal stand-in for MultiClassClassModel used to drive
    ``_multiclass_loco_novelty_auc`` with fully-controlled own-class OOF
    p-values (via ``cross_validate``) and foreign p-values (via
    ``fit``+``decision_matrix``). ``foreign_fn(X_held)`` returns the (n, K)
    p-value matrix for a held-out class; the true class of each held row is
    encoded in ``X_held[:, 0]`` so the fn can vary novelty per held class."""

    def __init__(self, classes, own_p_fn, foreign_fn):
        self.classes = list(classes)
        self.own_p_fn = own_p_fn
        self.foreign_fn = foreign_fn

    def cross_validate(self, X, y, n_splits=5):
        cl = list(self.classes)
        col = {c: k for k, c in enumerate(cl)}
        P = np.full((len(y), len(cl)), 0.5, dtype=np.float64)
        own = self.own_p_fn(y)
        for i in range(len(y)):
            P[i, col[y[i]]] = own[i]
        return {"decision_matrix": (P, P >= 0.05), "classes": cl}

    def fit(self, X, y):
        return self

    def decision_matrix(self, X):
        P = self.foreign_fn(np.asarray(X, dtype=np.float64), self.classes)
        return P, P >= 0.05


class TestLOCONoveltyAUCGateFixes:
    """Phase-C gate fixes to ``_multiclass_loco_novelty_auc`` (B1/B4/B5)."""

    def _auc(self, own_p_fn, foreign_fn, y, X=None):
        from spectral_predict.search import _multiclass_loco_novelty_auc

        classes = list(np.unique(y))
        if X is None:
            X = y.reshape(-1, 1).astype(np.float64)  # col0 encodes class
        return _multiclass_loco_novelty_auc(
            lambda: _FakeLOCOModel(classes, own_p_fn, foreign_fn),
            X, y, cv_splits=3,
        )

    def test_b1_perfect_separator_not_zero(self):
        # own-p all 1.0 (never falsely rejected) + foreign-p all 0.0 (always
        # novel) is a PERFECT separator -> AUC must be high, not the 0.0 the
        # endpoint-collapse bug produced.
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)

        def foreign(X, classes):
            return np.zeros((X.shape[0], len(classes)), dtype=np.float64)

        auc = self._auc(lambda yy: np.ones(len(yy)), foreign, y)
        assert auc >= 0.9

    def test_b4_all_nan_foreign_rows_do_not_inflate_novelty(self):
        # Real held rows are NOT novel (foreign p = 1.0); half the held rows are
        # all-NaN. The all-NaN rows must be EXCLUDED (not counted -inf/novel), so
        # novelty stays low and AUC stays low.
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        own = lambda yy: np.linspace(0.01, 0.99, len(yy))

        def foreign(X, classes):
            n, K = X.shape[0], len(classes)
            P = np.ones((n, K), dtype=np.float64)  # finite rows: NOT novel
            P[::2, :] = np.nan                       # every other row all-NaN
            return P

        auc = self._auc(own, foreign, y)
        assert auc < 0.2

    def test_b5_novelty_rate_is_class_balanced(self):
        # Imbalanced held-out classes {10, 100, 100}. Flipping ONLY the small
        # class between novel and not-novel must move the (class-balanced) AUC
        # substantially (~1/3), which a sample-weighted mean (~10/210) could not.
        y = np.array([0] * 10 + [1] * 100 + [2] * 100)
        own = lambda yy: np.linspace(0.01, 0.99, len(yy))

        def make_foreign(small_novel):
            def foreign(X, classes):
                K = len(classes)
                out = np.empty((X.shape[0], K), dtype=np.float64)
                for i in range(X.shape[0]):
                    c = int(round(X[i, 0]))
                    if c == 0:
                        out[i, :] = 0.0 if small_novel else 1.0
                    else:
                        out[i, :] = 0.0  # big classes always novel
                return out
            return foreign

        auc_novel = self._auc(own, make_foreign(True), y)
        auc_not = self._auc(own, make_foreign(False), y)
        assert (auc_novel - auc_not) >= 0.25


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


class TestLexicographicRankingB2:
    """B2: multiclass ranking is lexicographic (NaN last -> NoveltyAUC desc ->
    MinClassN desc), NOT the old additive ``-1e-9*MinClassN`` tiebreak that a
    large MinClassN or a sub-1e-9 AUC gap could corrupt."""

    def _score(self, rows):
        df = create_results_dataframe("multiclass_simca")
        for r in rows:
            r.setdefault("n_vars", 20)
            r.setdefault("full_vars", 20)
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        return compute_composite_score(df, "multiclass_simca")

    def test_nan_auc_ranks_last_even_with_huge_min_class_n(self):
        rows = [
            {"Model": "nan_big", "NoveltyAUC": np.nan, "MinClassN": 999},
            {"Model": "zero", "NoveltyAUC": 0.0, "MinClassN": 1},
            {"Model": "tiny", "NoveltyAUC": 1e-12, "MinClassN": 1},
        ]
        scored = self._score(rows)
        by_model = scored.set_index("Model")
        assert by_model.loc["nan_big", "Rank"] == scored["Rank"].max()
        # the NaN row is strictly worse than both finite rows
        assert by_model.loc["nan_big", "Rank"] > by_model.loc["zero", "Rank"]
        assert by_model.loc["nan_big", "Rank"] > by_model.loc["tiny", "Rank"]
        # among finite rows the larger AUC wins
        assert by_model.loc["tiny", "Rank"] < by_model.loc["zero", "Rank"]

    def test_subepsilon_auc_gap_not_flipped_by_min_class_n(self):
        rows = [
            {"Model": "hi_auc", "NoveltyAUC": 0.80000001, "MinClassN": 1},
            {"Model": "lo_auc", "NoveltyAUC": 0.80, "MinClassN": 10000},
        ]
        scored = self._score(rows).set_index("Model")
        # higher AUC wins despite the other row's enormous MinClassN
        assert scored.loc["hi_auc", "Rank"] < scored.loc["lo_auc", "Rank"]


class TestSchemaAndSearchGuardsB3B6B7:
    def _run(self, **kw):
        from spectral_predict.search import run_multiclass_simca_search

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_multiclass_simca_search(**kw)

    def test_b6_search_columns_are_declared_in_schema(self):
        X, y, _ = _graded([40, 40, 40], seed=0)
        df = self._run(
            X=X, y=y, engines=["pca-simca"], varsel_paths=["none"],
            preprocessing_methods=["raw"], cv_splits=3,
        )
        schema = set(create_results_dataframe("multiclass_simca").columns)
        added_by_scoring = {
            "CompositeScore", "Rank", "PerformanceScore",
            "VarPenalty", "GapPenalty", "ComplexityScore",
        }
        assert set(df.columns) - schema <= added_by_scoring

    def test_b3_all_nan_leaderboard_warns(self, monkeypatch, caplog):
        import logging

        import spectral_predict.simca as simca_mod

        def _boom(self, *a, **k):
            raise RuntimeError("forced fit failure")

        monkeypatch.setattr(simca_mod.MultiClassClassModel, "fit", _boom)
        X, y, _ = _graded([40, 40, 40], seed=1)
        with caplog.at_level(logging.WARNING):
            df = self._run(
                X=X, y=y, engines=["pca-simca"], varsel_paths=["none"],
                preprocessing_methods=["raw"], cv_splits=3,
            )
        assert df["NoveltyAUC"].isna().all()
        assert any(
            "no configuration" in rec.getMessage().lower()
            and "noveltyauc" in rec.getMessage().lower()
            for rec in caplog.records
        )

    def test_b7_malformed_preprocess_config_does_not_abort(self):
        X, y, _ = _graded([40, 40, 40], seed=2)
        good = {
            "method": "raw", "name": "raw", "deriv": None, "window": None,
            "polyorder": None,
        }
        bad = {  # a deriv config with an impossible (even, tiny) window
            "method": "savgol", "name": "bad_deriv", "deriv": 1, "window": 2,
            "polyorder": 9,
        }
        df = self._run(
            X=X, y=y, engines=["pca-simca"], varsel_paths=["none"],
            preprocess_configs=[good, bad], cv_splits=3,
        )
        assert len(df) == 2  # no raise; both configs recorded
        by_prep = df.set_index("Preprocess")
        assert np.isfinite(by_prep.loc["raw", "NoveltyAUC"])
        assert pd.isna(by_prep.loc["bad_deriv", "NoveltyAUC"])
        assert "preprocessing_failed" in str(by_prep.loc["bad_deriv", "reason"])


class TestNoveltyOrientedNComponentsD2:
    """D2: the search default n_components is a novelty-oriented variance
    fraction (float), not the discrimination-oriented ``per_class_cv``."""

    def _run(self, **kw):
        from spectral_predict.search import run_multiclass_simca_search

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_multiclass_simca_search(**kw)

    def test_default_uses_float_variance_fraction(self):
        import inspect

        from spectral_predict.search import run_multiclass_simca_search

        sig = inspect.signature(run_multiclass_simca_search)
        default = sig.parameters["n_components"].default
        assert isinstance(default, float) and 0.0 < default < 1.0

    def test_lvs_reflects_per_class_resolved_ints(self):
        X, y, _ = _graded([60, 60, 60], seed=3)
        df = self._run(
            X=X, y=y, engines=["pca-simca"], varsel_paths=["none"],
            preprocessing_methods=["raw"], cv_splits=4,
        )
        lvs = str(df.iloc[0]["LVs"])
        assert "{" in lvs and "}" in lvs  # per-class {class: resolved int}
        # resolved ints, not the raw fraction 0.99
        assert "0.99" not in lvs


def test_multiclass_schema_has_ncomponents_column():
    df = create_results_dataframe(task_type="multiclass_simca")
    assert "NComponents" in df.columns
    # Ordered right after Alpha for readability
    cols = list(df.columns)
    assert cols.index("NComponents") == cols.index("Alpha") + 1


def _toy():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 40)
    y = np.array(["A", "B", "C"] * 20)
    return X, y


def test_scalar_alpha_ncomp_matches_single_row_group():
    from spectral_predict.search import run_multiclass_simca_search

    X, y = _toy()
    df = run_multiclass_simca_search(
        X, y, engines=["pca-simca"], preprocessing_methods=["raw"],
        alpha=0.05, n_components=0.99, varsel_paths=["none"],
    )
    assert (df["Alpha"] == 0.05).all()
    assert (df["NComponents"].astype(str) == "0.99").all()


def test_list_alpha_expands_grid():
    from spectral_predict.search import run_multiclass_simca_search

    X, y = _toy()
    df1 = run_multiclass_simca_search(
        X, y, engines=["pca-simca"], preprocessing_methods=["raw"],
        alpha=0.05, n_components=0.99, varsel_paths=["none"],
    )
    df2 = run_multiclass_simca_search(
        X, y, engines=["pca-simca"], preprocessing_methods=["raw"],
        alpha=[0.01, 0.05], n_components=[0.95, 0.99], varsel_paths=["none"],
    )
    # 2 alphas x 2 n_components = 4x the single-value row count
    assert len(df2) == 4 * len(df1)
    assert set(df2["Alpha"].unique()) == {0.01, 0.05}
    assert set(df2["NComponents"].astype(str).unique()) == {"0.95", "0.99"}
