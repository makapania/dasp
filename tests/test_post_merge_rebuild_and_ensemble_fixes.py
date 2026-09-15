"""Post-merge review fixes: row-params helpers, PLS-DA head seed/class_weight, CatBoost refit.

* ``models.estimator_params_from_row`` turns pipeline-captured ``model__*`` params
  (Bayesian rows) and bare grid params into estimator params. The GUI ensemble
  reconstruction used to filter ``model__*`` out, training Bayesian rows with defaults
  (GUI parity tests: ``tests/gui/test_post_merge_gui_fixes.py``).
* ``models.plsda_head_kwargs`` restores the row's ``lr__random_state`` and
  ``lr__class_weight``. The validation rebuild forced ``random_state=42``, so a search
  run with another seed and a stochastic solver refit a different head.
* Ensemble per-fold refits of a CatBoost model saved before ``allow_writing_files=False``
  wrote ``catboost_info/`` into the cwd, and failed (NaN OOF predictions) when the cwd
  was unwritable.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_predict.models import (
    PLSDA_HEAD_DEFAULTS,
    PLSTransformer,
    estimator_params_from_row,
    plsda_head_kwargs,
)
from spectral_predict.search import _rebuild_model_from_row

# --- estimator_params_from_row --------------------------------------------------------


def test_estimator_params_strip_model_prefix_and_drop_other_steps() -> None:
    row = {
        "memory": None,
        "verbose": False,
        "steps": "ignored",
        "scaler__with_mean": True,
        "imbalance__k_neighbors": 3,
        "lr__C": 0.1,
        "model__max_features": 0.3,
        "model__n_estimators": 50,
    }
    assert estimator_params_from_row(row) == {"max_features": 0.3, "n_estimators": 50}


def test_estimator_params_keep_bare_grid_keys_and_prefixed_wins() -> None:
    assert estimator_params_from_row({"alpha": 0.5, "n_components": 7}) == {
        "alpha": 0.5,
        "n_components": 7,
    }
    assert estimator_params_from_row({"alpha": 1.0, "model__alpha": 0.2}) == {"alpha": 0.2}
    assert estimator_params_from_row({"pls__n_components": 4}) == {"n_components": 4}
    assert estimator_params_from_row(None) == {}


# --- PLS-DA head: seed and class_weight -----------------------------------------------


def test_plsda_head_kwargs_restores_recorded_seed_and_class_weight() -> None:
    row = {
        "pls__n_components": 3,
        "lr__C": 0.5,
        "lr__random_state": 7,
        "lr__class_weight": "balanced",
    }
    assert plsda_head_kwargs(row) == {
        **PLSDA_HEAD_DEFAULTS,
        "C": 0.5,
        "random_state": 7,
        "class_weight": "balanced",
    }
    # Rows without a recorded seed keep today's 42 and add no class_weight.
    assert plsda_head_kwargs({"lr_C": 0.3}) == {**PLSDA_HEAD_DEFAULTS, "C": 0.3, "random_state": 42}


def _plsda_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 25))
    signal = X[:, 0] - 0.8 * X[:, 4] + 0.6 * rng.standard_normal(80)
    y = (signal > np.quantile(signal, 0.75)).astype(int)  # imbalanced 3:1
    return X, y, rng.standard_normal((20, 25))


HEAD = {"C": 0.5, "solver": "saga", "max_iter": 30}


def _plsda_reference(random_state: int, class_weight) -> Pipeline:
    return Pipeline(
        [
            ("pls", PLSTransformer(n_components=3, scale=False)),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(**HEAD, random_state=random_state, class_weight=class_weight),
            ),
        ]
    )


def test_validation_rebuild_plsda_uses_recorded_seed_and_class_weight() -> None:
    X, y, X_test = _plsda_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # saga with few iterations does not converge
        reference = _plsda_reference(7, "balanced").fit(X, y)
        seed_42 = _plsda_reference(42, "balanced").fit(X, y)
        unweighted = _plsda_reference(7, None).fit(X, y)
    ref_proba = reference.predict_proba(X_test)
    # Preconditions: both the seed and the weighting change this head's predictions.
    assert not np.allclose(ref_proba, seed_42.predict_proba(X_test), rtol=1e-6, atol=1e-8)
    assert not np.allclose(ref_proba, unweighted.predict_proba(X_test), rtol=1e-6, atol=1e-8)

    # Params as the search captures them from the fitted pipeline.
    row_params = {
        "pls__n_components": 3,
        "pls__scale": False,
        "lr__C": HEAD["C"],
        "lr__solver": HEAD["solver"],
        "lr__max_iter": HEAD["max_iter"],
        "lr__random_state": 7,
        "lr__class_weight": "balanced",
    }
    row = pd.Series({"Model": "PLS-DA", "Params": str(row_params), "LVs": 3})
    rebuilt = _rebuild_model_from_row(row, "classification")
    lr = rebuilt.named_steps["lr"]
    assert lr.random_state == 7
    assert lr.class_weight == "balanced"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rebuilt.fit(X, y)
    np.testing.assert_allclose(rebuilt.predict_proba(X_test), ref_proba, rtol=1e-6, atol=1e-8)


def test_validation_rebuild_generic_row_unchanged_for_model_prefix() -> None:
    """The rebuild's own model__ normalisation now goes through the shared helper."""
    row = pd.Series(
        {"Model": "Ridge", "Params": str({"model__alpha": 0.37, "scaler__with_mean": True})}
    )
    rebuilt = _rebuild_model_from_row(row, "regression")
    assert rebuilt.named_steps["model"].alpha == 0.37


# --- CatBoost legacy model refit in ensembles -----------------------------------------

catboost = pytest.importorskip("catboost")


def _legacy_catboost_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool):
    """Fit and pickle a CatBoost built the pre-#73 way (no allow_writing_files)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 12))
    y = 2.0 * X[:, 0] - X[:, 2] + 0.1 * rng.standard_normal(40)
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    monkeypatch.chdir(fit_dir)  # the legacy fit itself writes catboost_info/ here
    legacy = catboost.CatBoostRegressor(iterations=10, depth=2, random_state=0, verbose=False)
    assert "allow_writing_files" not in legacy.get_params()
    model = Pipeline([("scaler", StandardScaler()), ("model", legacy)]) if nested else legacy
    model.fit(X, y)
    blob = pickle.dumps(model)
    return X, y, blob


@pytest.mark.parametrize("nested", [False, True], ids=["bare", "pipeline"])
@pytest.mark.parametrize(
    "ensemble_cls",
    ["RegionAwareWeightedEnsemble", "MixtureOfExpertsEnsemble", "StackingEnsemble"],
)
def test_ensemble_refit_of_legacy_catboost_writes_no_train_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool, ensemble_cls: str
) -> None:
    from spectral_predict import ensemble as ens

    X, y, blob = _legacy_catboost_pickle(tmp_path, monkeypatch, nested)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "catboost_info").write_text("not a directory", encoding="utf-8")
    monkeypatch.chdir(blocked)

    loaded = pickle.loads(blob)
    ensemble = getattr(ens, ensemble_cls)(
        models=[loaded, Ridge(alpha=1.0).fit(X, y)],
        model_names=["CatBoost", "Ridge"],
        n_regions=2,
        cv=3,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*failed during OOF prediction.*")
        ensemble.fit(X, y)

    assert np.all(np.isfinite(ensemble.predict(X)))
    assert (blocked / "catboost_info").is_file()
    # The loaded (fitted) original is untouched; only the per-fold clones changed.
    member = loaded.named_steps["model"] if nested else loaded
    assert "allow_writing_files" not in member.get_params()


class _ShallowWrapper:
    """Mimics the GUI wrappers: get_params(deep=True) returns a shallow dict."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def get_params(self, deep=True):
        return {"pipeline": self.pipeline}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


def test_refit_clone_reaches_catboost_behind_shallow_get_params() -> None:
    from sklearn.ensemble import VotingRegressor

    from spectral_predict.ensemble import _clone_for_refit

    def legacy():
        return catboost.CatBoostRegressor(iterations=5, verbose=False)

    wrapped = _ShallowWrapper(Pipeline([("scaler", StandardScaler()), ("model", legacy())]))
    voting = VotingRegressor([("cb", legacy()), ("ridge", Ridge())])
    for model, reach in [
        (wrapped, lambda m: [m.pipeline.named_steps["model"]]),
        (voting, lambda m: [m.estimators[0][1]]),
    ]:
        clone_ = _clone_for_refit(model)
        for member in reach(clone_):
            assert member.get_params()["allow_writing_files"] is False
        for member in reach(model):
            assert "allow_writing_files" not in member.get_params()


# --- Row parsing helpers --------------------------------------------------------------


def test_plsda_head_kwargs_coerces_serialised_seed_and_class_weight() -> None:
    kwargs = plsda_head_kwargs({"lr__random_state": 42.0, "lr__class_weight": "None"})
    assert kwargs["random_state"] == 42 and isinstance(kwargs["random_state"], int)
    assert kwargs["class_weight"] is None
    assert plsda_head_kwargs({"lr__random_state": "None"})["random_state"] is None
    assert plsda_head_kwargs({"lr__random_state": np.int64(9)})["random_state"] == 9
    assert plsda_head_kwargs({"lr__class_weight": {0: 1.0, 1: 3.0}})["class_weight"] == {
        0: 1.0,
        1: 3.0,
    }
    LogisticRegression(**plsda_head_kwargs({"lr__random_state": 7.0}))  # accepted by sklearn


@pytest.mark.parametrize(
    "params",
    [
        {"lr__random_state": 4.5},
        {"lr__random_state": "7"},
        {"lr__random_state": True},
        {"lr__class_weight": "balanced_subsample"},
        {"lr__class_weight": 3},
    ],
)
def test_plsda_head_kwargs_rejects_unusable_values(params) -> None:
    with pytest.raises(ValueError, match="lr__"):
        plsda_head_kwargs(params)


def test_parse_row_params_accepts_dict_and_string() -> None:
    from spectral_predict.models import parse_row_params

    stored = {"n_estimators": 25, "max_features": 0.3}
    assert parse_row_params(stored) == stored
    assert parse_row_params(stored) is not stored
    assert parse_row_params(str(stored)) == stored
    for junk in (None, float("nan"), "", "{not a dict", "[1, 2]"):
        assert parse_row_params(junk) == {}


def test_validation_rebuild_accepts_dict_params_cell() -> None:
    """In-memory result rows can hold Params as a dict; the rebuild used to ignore it."""
    row = pd.Series({"Model": "RandomForest", "Params": None}, dtype=object)
    row["Params"] = {"n_estimators": 25, "max_features": 0.3, "max_depth": 4}
    rebuilt = _rebuild_model_from_row(row, "regression")
    assert (rebuilt.n_estimators, rebuilt.max_features, rebuilt.max_depth) == (25, 0.3, 4)


def test_preprocessing_config_from_row_reads_columns_and_display_name() -> None:
    from spectral_predict.preprocess import preprocessing_config_from_row

    bayes = {
        "Preprocess": "polynomial+sg0+snv_deriv+autoscale",
        "PreprocessBase": "snv_deriv",
        "Deriv": 1,
        "Window": 11,
        "Poly": 2,
        "Autoscale": True,
        "baseline_method": "polynomial",
        "baseline_params": "{'degree': 3}",
        "smoothing": True,
        "smoothing_window": 13,
        "smoothing_polyorder": 3,
    }
    assert preprocessing_config_from_row(bayes) == {
        "preprocess_name": "snv_deriv",
        "deriv": 1,
        "window": 11,
        "polyorder": 2,
        "baseline_method": "polynomial",
        "baseline_params": {"degree": 3},
        "smoothing": True,
        "smoothing_window": 13,
        "smoothing_polyorder": 3,
        "autoscale": True,
    }
    # Old row: only the display name, with NaN cells from a mixed results table.
    nan = float("nan")
    old = {
        "PreprocessBase": nan,
        "Preprocess": "als+raw+autoscale",
        "Deriv": nan,
        "Window": nan,
        "Poly": nan,
        "Autoscale": nan,
        "smoothing": nan,
        "smoothing_window": nan,
    }
    config = preprocessing_config_from_row(old)
    assert config["preprocess_name"] == "raw"
    assert config["baseline_method"] == "als"
    assert config["autoscale"] is True
    assert config["smoothing"] is False  # NaN is not "smoothing on"
    assert (config["deriv"], config["window"], config["polyorder"]) == (None, None, None)
    assert config["smoothing_window"] == 17
    assert preprocessing_config_from_row({"Autoscale": "False"})["autoscale"] is False


@pytest.mark.parametrize(
    ("cell", "expected"),
    [("False", False), ("false", False), ("0", False), ("True", True), ("1", True), (0.0, False)],
)
def test_preprocessing_config_from_row_parses_smoothing_strings(cell, expected) -> None:
    from spectral_predict.preprocess import preprocessing_config_from_row

    assert preprocessing_config_from_row({"smoothing": cell})["smoothing"] is expected


@pytest.mark.parametrize("name", ["deriv", "snv_deriv", "deriv_snv"])
def test_preprocessing_config_from_row_defaults_missing_derivative_settings(name) -> None:
    from spectral_predict.preprocess import preprocessing_config_from_row

    nan = float("nan")
    config = preprocessing_config_from_row(
        {"Preprocess": name, "Deriv": nan, "Window": nan, "Poly": nan}
    )
    assert (config["deriv"], config["window"], config["polyorder"]) == (1, 15, None)
    # An explicit order is kept; only the window is defaulted.
    config = preprocessing_config_from_row({"Preprocess": name, "Deriv": 2, "Window": None})
    assert (config["deriv"], config["window"]) == (2, 15)


def test_validation_rebuild_scores_row_with_missing_derivative_window() -> None:
    """On main this row raised in SavgolDerivative.transform and got RMSEP=NaN."""
    from spectral_predict.preprocess import SNV, SavgolDerivative
    from spectral_predict.search import compute_validation_metrics_for_top_models

    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 30)).cumsum(axis=1)
    y = X[:, 3] - X[:, 10]
    X_val = rng.standard_normal((15, 30)).cumsum(axis=1)
    y_val = X_val[:, 3] - X_val[:, 10]
    df = pd.DataFrame(
        [
            {
                "CompositeScore": 0.0,
                "Task": "regression",
                "Model": "Ridge",
                "Params": str({"alpha": 2.0}),
                "Preprocess": "snv_deriv",
                "Deriv": 1,
                "Window": np.nan,
                "Poly": 2,
            }
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compute_validation_metrics_for_top_models(
            df, X, y, X_val, y_val, "regression", np.arange(1000.0, 1060.0, 2.0), top_n=1
        )

    reference = Pipeline(
        [
            ("snv", SNV()),
            ("savgol", SavgolDerivative(deriv=1, window=15, polyorder=2)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=2.0, random_state=42)),
        ]
    ).fit(X, y)
    expected = np.sqrt(np.mean((reference.predict(X_val) - y_val) ** 2))
    np.testing.assert_allclose(out.loc[0, "RMSEP"], expected, rtol=1e-9)


# --- Preprocessing chromosomes --------------------------------------------------------


def test_chromosome_from_row_shapes() -> None:
    from spectral_predict.ga_preprocessing import chromosome_from_row

    assert list(chromosome_from_row({"preprocess_chromosome": "[6, 3, 0]"})) == [6, 3, 0]
    assert list(chromosome_from_row({"preprocess_chromosome": [2, 4]})) == [2, 4]
    assert list(chromosome_from_row({"ga_genes": "[1, 0]"})) == [1, 0]  # pre-rename CSVs
    for row in ({}, {"preprocess_chromosome": ""}, {"preprocess_chromosome": float("nan")}):
        assert chromosome_from_row(row) is None
    with pytest.raises(ValueError, match="preprocess_chromosome"):
        chromosome_from_row({"preprocess_chromosome": "[6, 3"})


def test_chromosome_to_steps_matches_search_transform() -> None:
    from sklearn.base import clone

    from spectral_predict.ga_preprocessing import (
        PREPROC_TYPES,
        chromosome_to_steps,
        chromosome_to_transform,
    )

    rng = np.random.default_rng(1)
    X = rng.standard_normal((12, 80)).cumsum(axis=1)
    for p in range(len(PREPROC_TYPES)):
        genes = [p, 6]  # window 17 is legal for every derivative order
        _, transform = chromosome_to_transform(np.array(genes))
        steps = chromosome_to_steps(genes)
        expected = X if transform is None else transform(X)
        got = Pipeline([(n, clone(s)) for n, s in steps]).fit_transform(X) if steps else X
        np.testing.assert_array_equal(got, expected)
    assert chromosome_to_steps([0, 6]) == []
    assert [n for n, _ in chromosome_to_steps([0, 6, 1])] == ["autoscale"]
    assert [n for n, _ in chromosome_to_steps([1, 6], autoscale=True)] == ["snv", "autoscale"]
