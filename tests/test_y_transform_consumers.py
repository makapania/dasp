"""Consumers of a Y-transformed (TransformedTargetRegressor) model outside Tab 7.

Review round 1 on fix/ytransform-save: RF tree-variance uncertainty, ensembles with a
transformed base model, code export guards and transform validation edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

from spectral_predict.code_generator import CodeGenerator, ExportOptions
from spectral_predict.ensemble import create_ensemble
from spectral_predict.model_io import predict_with_uncertainty
from spectral_predict.y_transform import YTransformWrapper


def _data(n: int = 40, p: int = 6):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n, p))
    y = np.exp(0.5 * X[:, 0] + 0.2 * X[:, 1]) + 0.05 * rng.random(n) + 0.1
    return X, y


def _model_dict(model, task_type="regression"):
    return {
        "model": model,
        "preprocessor": None,
        "metadata": {
            "task_type": task_type,
            "wavelengths": list(range(6)),
            "model_class": type(model).__name__,
            "performance": {"RMSE": 0.1},
        },
    }


def test_ttr_random_forest_uncertainty_in_original_units():
    X, y = _data()
    ttr = YTransformWrapper.wrap(RandomForestRegressor(n_estimators=25, random_state=0), "Log").fit(
        X, y
    )

    out = predict_with_uncertainty(_model_dict(ttr), X)

    assert "tree_variance" in out["uncertainty"]
    trees = np.array([np.exp(t.predict(X)) for t in ttr.regressor_.estimators_])
    np.testing.assert_allclose(out["uncertainty"]["tree_variance"], trees.std(axis=0))
    # Original units: the per-tree spread is on the y scale, not the log scale.
    log_spread = np.array([t.predict(X) for t in ttr.regressor_.estimators_]).std(axis=0)
    assert not np.allclose(out["uncertainty"]["tree_variance"], log_spread)


def test_plain_random_forest_uncertainty_unchanged():
    X, y = _data()
    rf = RandomForestRegressor(n_estimators=25, random_state=0).fit(X, y)
    out = predict_with_uncertainty(_model_dict(rf), X)
    trees = np.array([t.predict(X) for t in rf.estimators_])
    np.testing.assert_allclose(out["uncertainty"]["tree_variance"], trees.std(axis=0))


@pytest.mark.parametrize("ensemble_type", ["simple_average", "stacking"])
def test_ensemble_with_ttr_base_model(ensemble_type):
    X, y = _data()
    ttr = YTransformWrapper.wrap(PLSRegression(n_components=2), "Log").fit(X, y)
    plain = PLSRegression(n_components=2).fit(X, y)

    ens = create_ensemble([ttr, plain], ["pls_log", "pls"], X, y, ensemble_type=ensemble_type)
    pred = np.asarray(ens.predict(X)).ravel()

    assert pred.shape == (len(y),) and np.all(np.isfinite(pred))
    if ensemble_type == "simple_average":
        expected = (ttr.predict(X).ravel() + plain.predict(X).ravel()) / 2
        np.testing.assert_allclose(pred, expected)
    # A clone of the transformed base model refits with its transform.
    refit = clone(ttr).fit(X, y)
    np.testing.assert_allclose(refit.predict(X), ttr.predict(X))


def _export_config(**overrides):
    cfg = {
        "model_name": "PLS",
        "preprocessing": "raw",
        "task_type": "regression",
        "target_name": "target",
        "params": {"n_components": 2},
        "metrics": {},
        "cv_folds": 3,
        "cv_strategy": "kfold",
        "cv_n_repeats": 5,
        "imbalance_method": None,
        "imbalance_params": {},
        "autoscale": False,
        "variable_indices": None,
        "variable_selection_method": None,
        "trim_derivative_edges": False,
        "inlier_class_label": "",
        "wavelengths": list(range(6)),
        "early_stopping_rounds": None,
        "y_transform": "boxcox",
    }
    cfg.update(overrides)
    return cfg


def test_export_rejects_y_transform_with_imbalance():
    with pytest.raises(ValueError, match="Y-transform"):
        CodeGenerator(_export_config(imbalance_method="smogn"), ExportOptions())


def test_export_without_transform_is_unchanged():
    for value in ("none", "None", None):
        script = CodeGenerator(_export_config(y_transform=value), ExportOptions()).generate_script()
        assert "YTransformRegressor" not in script
    # Classification never carries a target transform.
    clf = _export_config(task_type="classification", model_name="PLS-DA", y_transform="log")
    assert CodeGenerator(clf, ExportOptions()).y_transform == "none"


def test_export_script_trains_on_transformed_target():
    X, y = _data()
    opts = ExportOptions(
        format="notebook", include_data=True, data_X=X, data_y=y, include_visualization=False
    )
    ns: dict = {}
    for cell in CodeGenerator(_export_config(), opts).generate_notebook()["cells"]:
        code = "".join(cell["source"])
        if cell["cell_type"] == "code" and "subprocess.check_call" not in code:
            exec(code, ns)
    # Same inner estimator as exported (exports add their default PLS params).
    inner = clone(ns["model"].regressor)
    assert isinstance(inner, PLSRegression) and inner.n_components == 2
    expected = YTransformWrapper.wrap(inner, "Box-Cox").fit(X, y)
    np.testing.assert_allclose(ns["model"].predict(X), expected.predict(X), rtol=1e-8)


def test_log1p_rejects_minus_one():
    assert YTransformWrapper.validate(np.array([-1.0, 0.5]), "Log1p") is not None
    assert YTransformWrapper.validate(np.array([-0.5, 0.5]), "Log1p") is None
