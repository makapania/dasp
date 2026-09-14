"""T-51 PR B0: PLS-DA logistic-head params survive every rebuild consumer.

A PLS-DA results row stores the head's ``C`` / ``solver`` / ``max_iter`` either as
canonical ``lr__*`` keys (captured from the fitted Pipeline, which is what grid and
Bayesian rows carry today) or as legacy ``lr_*`` keys (grid config dicts, older rows).
Before B0 each consumer lost them in a different way:

* ``build_model('PLS-DA', params)`` raised ``TypeError`` on any head or ``pls__`` key.
* ``search._rebuild_model_from_row`` skipped ``lr__*`` and then read ``lr_C``, so a
  canonical row was rebuilt with ``C=1.0``. Legacy keys were also pushed through
  ``PLSTransformer.set_params``, which never validates keys, leaving junk ``lr_*``
  attributes on the transformer.
* The exporter routed legacy ``lr_C`` into the LogisticRegression kwargs under that
  literal name, so the exported head used C=1.0.
* Tab 7 refit and ensemble reconstruction (covered in
  ``tests/gui/test_t51_b0_tab7_plsda_refit.py``) never read legacy ``lr_*`` keys, and
  ensemble reconstruction ignored canonical ones too.

The matrix here is {canonical row, legacy row} x {default, non-default transformer
settings} with a non-default head (C, solver, max_iter). Every consumer must yield the
exact head/transformer params and ``predict_proba`` equal to the search-time pipeline.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_predict.code_generator import CodeGenerator, ExportOptions
from spectral_predict.models import (
    PLSDA_HEAD_DEFAULTS,
    PLSTransformer,
    build_model,
    split_plsda_params,
)
from spectral_predict.search import _rebuild_model_from_row, run_search

# Non-default head: every value differs from PLSDA_HEAD_DEFAULTS and moves the probabilities.
HEAD = {"C": 0.05, "solver": "newton-cg", "max_iter": 250}

# (n_components, pls tol): PLSTransformer defaults vs non-default transformer settings.
TRANSFORMER_SETTINGS = {
    "default_transformer": {"n_components": 2, "max_iter": 500, "tol": 1e-6},
    "nondefault_transformer": {"n_components": 4, "max_iter": 500, "tol": 1e-5},
}

N_FEATURES = 40


def _make_data(seed: int = 7) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((60, N_FEATURES))
    y = (X[:, 0] - 0.7 * X[:, 3] + 0.6 * rng.standard_normal(60) > 0).astype(int)
    X_test = rng.standard_normal((20, N_FEATURES))
    cols = [f"{1000 + 2 * i}" for i in range(N_FEATURES)]
    return pd.DataFrame(X, columns=cols), pd.Series(y), X_test


X_TRAIN, Y_TRAIN, X_TEST = _make_data()


def _search_time_pipeline(transformer: dict[str, Any], head: dict[str, Any]) -> Pipeline:
    """The PLS-DA pipeline exactly as search.py builds it at search time."""
    return Pipeline(
        [
            (
                "pls",
                PLSTransformer(
                    n_components=transformer["n_components"],
                    max_iter=transformer["max_iter"],
                    tol=transformer["tol"],
                    scale=False,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=head["C"], solver=head["solver"], max_iter=head["max_iter"], random_state=42
                ),
            ),
        ]
    )


@pytest.fixture(scope="module")
def canonical_grid_rows() -> dict[tuple[int, float], dict[str, Any]]:
    """Real canonical ``Params`` rows from a PLS-DA grid search with a non-default head."""
    df, _ = run_search(
        X_TRAIN,
        Y_TRAIN,
        "classification",
        folds=3,
        max_n_components=5,
        models_to_test=["PLS-DA"],
        preprocessing_methods={"raw": True},
        pls_tol_list=[1e-6, 1e-5],
        plsda_lr_C_list=[HEAD["C"]],
        plsda_lr_solver_list=[HEAD["solver"]],
        plsda_lr_max_iter_list=[HEAD["max_iter"]],
        enable_variable_subsets=False,
        enable_region_subsets=False,
    )
    rows: dict[tuple[int, float], dict[str, Any]] = {}
    for params_str in df["Params"]:
        params = ast.literal_eval(params_str)
        rows[(params["pls__n_components"], params["pls__tol"])] = params
    return rows


def _row_params(style: str, transformer: dict[str, Any], canonical_rows) -> dict[str, Any]:
    if style == "canonical":
        return dict(canonical_rows[(transformer["n_components"], transformer["tol"])])
    # Legacy spelling: the grid config dict before full-param capture.
    return {
        "n_components": transformer["n_components"],
        "max_iter": transformer["max_iter"],
        "tol": transformer["tol"],
        "lr_C": HEAD["C"],
        "lr_solver": HEAD["solver"],
        "lr_max_iter": HEAD["max_iter"],
    }


MATRIX = [
    pytest.param(style, name, id=f"{style}-{name}")
    for style in ("canonical", "legacy")
    for name in TRANSFORMER_SETTINGS
]


def _assert_pipeline_matches(model: Pipeline, transformer: dict[str, Any]) -> None:
    params = model.get_params()
    for key, value in transformer.items():
        assert params[f"pls__{key}"] == value, key
    assert params["pls__scale"] is False
    for key, value in HEAD.items():
        assert params[f"lr__{key}"] == value, key
    # PLSTransformer.set_params setattr's any key, so a leaked head key shows up here.
    leaked = [k for k in vars(model.named_steps["pls"]) if k.startswith("lr")]
    assert leaked == [], f"head keys leaked onto the transformer: {leaked}"


def _reference_proba(transformer: dict[str, Any]) -> np.ndarray:
    ref = _search_time_pipeline(transformer, HEAD).fit(X_TRAIN.values, Y_TRAIN.values)
    return ref.predict_proba(X_TEST)


# ---------------------------------------------------------------------------
# The shared helper
# ---------------------------------------------------------------------------


def test_canonical_grid_row_is_the_search_time_pipeline(canonical_grid_rows):
    """Guard for the fixture: the row's head really is the non-default head."""
    for (n_components, tol), params in canonical_grid_rows.items():
        assert params["lr__C"] == HEAD["C"]
        assert params["lr__solver"] == HEAD["solver"]
        assert params["lr__max_iter"] == HEAD["max_iter"]
    assert (2, 1e-6) in canonical_grid_rows and (4, 1e-5) in canonical_grid_rows


@pytest.mark.parametrize("style,setting", MATRIX)
def test_split_plsda_params_matrix(style, setting, canonical_grid_rows):
    transformer = TRANSFORMER_SETTINGS[setting]
    params = _row_params(style, transformer, canonical_grid_rows)
    transformer_params, head_params = split_plsda_params(params)

    assert head_params == HEAD
    for key, value in transformer.items():
        assert transformer_params[key] == value
    assert not any(k.startswith("lr") or "__" in k for k in transformer_params)


def test_split_plsda_params_canonical_wins_over_legacy():
    params = {
        "lr_C": 9.0,
        "lr__C": 0.05,
        "lr_solver": "liblinear",
        "lr__max_iter": 250,
        "lr_max_iter": 7,
        "n_components": 3,
        "pls__n_components": 4,
    }
    transformer_params, head_params = split_plsda_params(params)
    assert head_params == {"C": 0.05, "solver": "liblinear", "max_iter": 250}
    assert transformer_params == {"n_components": 4}


def test_split_plsda_params_drops_non_head_and_wrapper_keys():
    params = {
        "pls__tol": 1e-5,
        "scaler__with_mean": True,
        "imbalance__k_neighbors": 3,
        "lr__class_weight": None,
        "lr__random_state": 42,
        "lr_class_weight": "balanced",
        "memory": None,
        "verbose": False,
    }
    transformer_params, head_params = split_plsda_params(params)
    assert transformer_params == {"tol": 1e-5}
    assert head_params == {}


def test_split_plsda_params_empty_inputs():
    assert split_plsda_params(None) == ({}, {})
    assert split_plsda_params({}) == ({}, {})
    assert PLSDA_HEAD_DEFAULTS == {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}


# ---------------------------------------------------------------------------
# (a) build_model
# ---------------------------------------------------------------------------


def test_build_model_plsda_bare_params_unchanged():
    """T-B0a: the no-op case for current callers (Bayesian objective)."""
    built = build_model("PLS-DA", {"n_components": 5}, task_type="classification")
    assert isinstance(built, PLSTransformer)
    assert built.get_params() == PLSTransformer(n_components=5, scale=False).get_params()


@pytest.mark.parametrize("style,setting", MATRIX)
def test_build_model_matrix(style, setting, canonical_grid_rows):
    transformer = TRANSFORMER_SETTINGS[setting]
    params = _row_params(style, transformer, canonical_grid_rows)

    built = build_model("PLS-DA", params, task_type="classification")

    assert isinstance(built, PLSTransformer)
    for key, value in transformer.items():
        assert getattr(built, key) == value
    assert built.scale is False

    _, head_params = split_plsda_params(params)
    pipe = Pipeline(
        [
            ("pls", built),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**{**PLSDA_HEAD_DEFAULTS, **head_params}, random_state=42)),
        ]
    )
    _assert_pipeline_matches(pipe, transformer)
    pipe.fit(X_TRAIN.values, Y_TRAIN.values)
    assert np.allclose(pipe.predict_proba(X_TEST), _reference_proba(transformer))


# ---------------------------------------------------------------------------
# (b) validation rebuild
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style,setting", MATRIX)
def test_rebuild_model_from_row_matrix(style, setting, canonical_grid_rows):
    transformer = TRANSFORMER_SETTINGS[setting]
    params = _row_params(style, transformer, canonical_grid_rows)
    row = pd.Series({"Model": "PLS-DA", "Params": str(params), "LVs": transformer["n_components"]})

    model = _rebuild_model_from_row(row, "classification")

    _assert_pipeline_matches(model, transformer)
    model.fit(X_TRAIN.values, Y_TRAIN.values)
    assert np.allclose(model.predict_proba(X_TEST), _reference_proba(transformer))


def test_rebuild_legacy_row_with_head_key_first_keeps_transformer_params():
    """Key order must not matter, and a stale LVs must not beat the row's n_components."""
    transformer = TRANSFORMER_SETTINGS["nondefault_transformer"]
    params = {"lr_C": HEAD["C"], "lr_solver": HEAD["solver"], "lr_max_iter": HEAD["max_iter"]}
    params.update(transformer)
    row = pd.Series({"Model": "PLS-DA", "Params": str(params), "LVs": 9})

    model = _rebuild_model_from_row(row, "classification")

    _assert_pipeline_matches(model, transformer)


# ---------------------------------------------------------------------------
# (d) exporter
# ---------------------------------------------------------------------------


def _export_config(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": "PLS-DA",
        "preprocessing": "raw",
        "task_type": "classification",
        "target_name": "target",
        "params": params,
        "metrics": {"Accuracy": 0.0},
        "variable_indices": None,
        "wavelengths": list(range(N_FEATURES)),
        "cv_folds": 3,
        "imbalance_method": None,
    }


@pytest.mark.parametrize("style,setting", MATRIX)
def test_export_split_matrix(style, setting, canonical_grid_rows):
    transformer = TRANSFORMER_SETTINGS[setting]
    params = _row_params(style, transformer, canonical_grid_rows)
    generator = CodeGenerator(_export_config(params), ExportOptions(format="script"))

    pls_params, lr_params = generator._split_pls_da_params(params)

    for key, value in transformer.items():
        assert pls_params[key] == value
    for key, value in HEAD.items():
        assert lr_params[key] == value
    assert not any(k.startswith("lr_") for k in lr_params)


@pytest.mark.parametrize("style,setting", MATRIX)
def test_export_script_matches_search_time_pipeline(style, setting, canonical_grid_rows, tmp_path):
    """The exported script's predict_proba equals the search-time pipeline's."""
    from tests.test_t20_saved_model_export_parity import _run_parity

    transformer = TRANSFORMER_SETTINGS[setting]
    params = _row_params(style, transformer, canonical_grid_rows)
    _run_parity(
        model=_search_time_pipeline(transformer, HEAD),
        model_name="PLS-DA",
        task_type="classification",
        X_train=X_TRAIN.values,
        y_train=Y_TRAIN.values,
        X_test=X_TEST,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )
