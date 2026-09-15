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
