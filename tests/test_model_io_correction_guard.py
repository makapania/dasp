"""R010: model_io never stores or applies a bias/nonlinear correction for non-regression.

Files written before the save-side guard can hold a stale regression correction in a
classification model; ``predict_with_model`` must ignore it rather than shift labels.
Also pins the Y-transform name normalisation (R048: 'Box-Cox' was rejected).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression

from spectral_predict.model_io import load_model, predict_with_model, save_model
from spectral_predict.y_transform import (
    YTransformWrapper,
    normalize_y_transform_method,
    replace_fitted_regressor,
)

LINEAR = {"method": "linear", "bias": 10.0, "slope": 2.0}


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8))
    y_cls = (X[:, 0] > 0).astype(int)
    y_reg = 3.0 + X[:, 0] + 0.1 * rng.normal(size=30)
    return X, y_cls, y_reg


def _meta(task_type):
    meta = {"model_name": "m", "wavelengths": list(range(8)), "n_vars": 8}
    if task_type is not None:
        meta["task_type"] = task_type
    return meta


def test_classifier_correction_ignored_at_predict():
    X, y_cls, _ = _data()
    clf = LogisticRegression().fit(X, y_cls)
    model_dict = {
        "model": clf,
        "preprocessor": None,
        "metadata": _meta("classification"),
        "bias_correction": LINEAR,  # legacy file with a stale regression correction
    }
    np.testing.assert_array_equal(predict_with_model(model_dict, X), clf.predict(X))


@pytest.mark.parametrize("task_type", ["regression", None])
def test_regression_correction_still_applied(task_type):
    X, _, y_reg = _data()
    pls = PLSRegression(n_components=2).fit(X, y_reg)
    model_dict = {
        "model": pls,
        "preprocessor": None,
        "metadata": _meta(task_type),
        "bias_correction": LINEAR,
    }
    raw = np.asarray(pls.predict(X)).ravel()
    out = np.asarray(predict_with_model(model_dict, X)).ravel()
    assert not np.allclose(out, raw)


def test_save_drops_correction_for_classifier(tmp_path):
    X, y_cls, _ = _data()
    clf = LogisticRegression().fit(X, y_cls)
    path = tmp_path / "c.dasp"
    save_model(clf, None, _meta("classification"), path, bias_correction=LINEAR)
    loaded = load_model(path)
    assert loaded["bias_correction"] is None
    assert loaded["metadata"]["has_bias_correction"] is False


def test_save_keeps_correction_for_regression(tmp_path):
    X, _, y_reg = _data()
    pls = PLSRegression(n_components=2).fit(X, y_reg)
    path = tmp_path / "r.dasp"
    save_model(pls, None, _meta("regression"), path, bias_correction=LINEAR)
    assert load_model(path)["bias_correction"] == LINEAR


@pytest.mark.parametrize(
    "spelling,canonical",
    [
        ("Box-Cox", "boxcox"),
        ("box-cox", "boxcox"),
        ("boxcox", "boxcox"),
        ("Yeo-Johnson", "yeo-johnson"),
        ("Log", "log"),
        ("Log1p", "log1p"),
        ("Sqrt", "sqrt"),
        ("None", "none"),
        (None, "none"),
    ],
)
def test_normalize_y_transform_method(spelling, canonical):
    assert normalize_y_transform_method(spelling) == canonical


def test_unknown_transform_rejected():
    with pytest.raises(ValueError):
        normalize_y_transform_method("cube")
    assert YTransformWrapper.validate(np.ones(3), "cube") is not None


def test_boxcox_gui_spelling_wraps_and_validates():
    assert isinstance(
        YTransformWrapper.wrap(PLSRegression(), "Box-Cox"), TransformedTargetRegressor
    )
    assert YTransformWrapper.validate(np.array([1.0, 0.0]), "Box-Cox") is not None


def test_replace_fitted_regressor_keeps_transform():
    X, _, y_reg = _data()
    ttr = YTransformWrapper.wrap(PLSRegression(n_components=2), "Log").fit(X, y_reg)
    swapped = replace_fitted_regressor(ttr, ttr.regressor_)
    np.testing.assert_allclose(swapped.predict(X), ttr.predict(X))
    assert swapped is not ttr
