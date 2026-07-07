"""T-31 Phase D deferred fold-ins:
- predict_with_uncertainty multiclass branch
- LOCO oof_cv reuse equivalence (perf, behavior-preserving)
- _cross_fit_null surfaces EE covariance-fold failure instead of silent collapse
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectral_predict.simca import MultiClassClassModel


def _synthetic(K=3, n=40, p=30, seed=5):
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for k in range(K):
        blocks.append(rng.normal(k * 4.0, 1.0, size=(n, p)))
        labels += [f"C{k}"] * n
    return np.vstack(blocks).astype(np.float64), np.array(labels)


def test_predict_with_uncertainty_multiclass_roundtrip(tmp_path):
    from spectral_predict.model_io import (
        load_model,
        predict_with_uncertainty,
        save_model,
    )

    X, y = _synthetic()
    model = MultiClassClassModel(
        engine="pca-simca", alpha=0.05, n_components=0.99, scaling="per_class",
    ).fit(X, y)

    path = tmp_path / "mc.dasp"
    save_model(
        model, None,
        {"model_name": "MultiClassSIMCA", "task_type": "multiclass_simca",
         "wavelengths": list(range(X.shape[1])), "n_vars": X.shape[1],
         "class_names": [str(c) for c in model.classes_],
         "engine_family": "pca-simca", "alpha": 0.05, "scaling": "per_class"},
        path,
    )
    loaded = load_model(path)
    res = predict_with_uncertainty(loaded, X, validate_wavelengths=False)

    assert res["has_uncertainty"] is True
    unc = res["uncertainty"]
    assert unc["p_values"].shape == (len(X), len(model.classes_))
    assert unc["decision_matrix"].shape == (len(X), len(model.classes_))
    assert len(unc["accepted_classes"]) == len(X)
    assert len(res["predictions"]) == len(X)


def test_loco_oof_cv_reuse_is_equivalent():
    from spectral_predict.search import _multiclass_loco_novelty_auc

    X, y = _synthetic()

    def _build():
        return MultiClassClassModel(
            engine="pca-simca", alpha=0.05, n_components=0.99, scaling="per_class",
        )

    auc_recompute = _multiclass_loco_novelty_auc(_build, X, y, cv_splits=4)
    cv = _build().cross_validate(X, y, n_splits=4)
    auc_reuse = _multiclass_loco_novelty_auc(_build, X, y, cv_splits=4, oof_cv=cv)

    assert np.isclose(auc_recompute, auc_reuse, equal_nan=True)


def test_cross_fit_null_warns_when_all_folds_fail(monkeypatch):
    """When every null-calibration fold fit raises (e.g. EE covariance on wide
    spectra), _cross_fit_null must WARN and return an empty null, not silently
    swallow the failure."""
    import spectral_predict.simca as simca_mod

    def _raising_builder(*_a, **_k):
        class _Boom:
            def fit(self, _X):
                raise ValueError("n_samples < n_features (singular covariance)")
        return _Boom()

    monkeypatch.setattr(simca_mod, "get_one_class_model", _raising_builder)

    m = MultiClassClassModel(engine="elliptic-envelope", alpha=0.05, scaling="none")
    X_raw = np.random.default_rng(0).normal(size=(20, 200))
    with pytest.warns(UserWarning, match="ALL .* null-calibration folds failed"):
        null = m._cross_fit_null(
            X_raw, builder_name="elliptic-envelope",
            score_method="decision_function", scaling="none",
        )
    assert null.size == 0


def test_cross_fit_null_partial_failure_warns_but_survives(monkeypatch):
    """A subset of failed folds warns but still returns the surviving scores."""
    import spectral_predict.simca as simca_mod

    calls = {"n": 0}

    def _sometimes_raising(*_a, **_k):
        calls["n"] += 1

        class _Eng:
            def __init__(self, boom):
                self._boom = boom

            def fit(self, _X):
                if self._boom:
                    raise ValueError("boom")
                return self

            def decision_function(self, X):
                return np.zeros(len(X))
        # Fail the first fold only.
        return _Eng(boom=(calls["n"] == 1))

    monkeypatch.setattr(simca_mod, "get_one_class_model", _sometimes_raising)

    m = MultiClassClassModel(engine="ocsvm", alpha=0.05, scaling="none")
    X_raw = np.random.default_rng(1).normal(size=(25, 10))
    with pytest.warns(UserWarning, match="null-calibration folds\\s+failed"):
        null = m._cross_fit_null(
            X_raw, builder_name="ocsvm", score_method="decision_function",
            scaling="none",
        )
    assert null.size > 0


def test_empty_null_marks_class_unmodelable(monkeypatch):
    """A class whose null calibration comes back EMPTY must be marked
    unmodelable (dropped from models_), not left as a live column that silently
    rejects every sample (all-NaN p >= alpha is False)."""
    import spectral_predict.simca as simca_mod

    # C0's null comes back empty (first call); C1/C2 calibrate normally.
    orig = simca_mod.MultiClassClassModel._cross_fit_null
    state = {"n": 0}

    def _wrapped(self, X_raw, builder_name, score_method, scaling, reuse_scaler=None):
        state["n"] += 1
        if state["n"] == 1:
            return np.asarray([], dtype=np.float64)
        return orig(self, X_raw, builder_name, score_method, scaling, reuse_scaler)

    monkeypatch.setattr(simca_mod.MultiClassClassModel, "_cross_fit_null", _wrapped)

    X, y = _synthetic(K=3, n=30, p=20, seed=9)  # n>=20 so all classes modelable
    m = MultiClassClassModel(engine="ocsvm", alpha=0.05, scaling="none",
                             min_class_samples=10).fit(X, y)
    classes = sorted(set(y))
    # The class whose null was empty is unmodelable and absent from models_.
    assert classes[0] in m.unmodelable_
    assert classes[0] not in m.models_
    # Its decision-matrix column is all-NaN (dropped), not silently all-reject.
    P, A = m.decision_matrix(X)
    k0 = list(m.classes_).index(classes[0])
    assert np.isnan(P[:, k0]).all()
    assert not A[:, k0].any()
