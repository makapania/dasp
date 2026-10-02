"""Persistent ensemble wrappers (spectral_predict.model_wrappers) and their loading.

* Saving an ensemble with a GA / Combined wrapper raised PicklingError (cached local
  closure). ``__getstate__`` now drops the cache from a COPY of the state: editing the
  live ``__dict__`` cleared the cache of an object mid-prediction.
* The wrappers used to be defined in the GUI script, so pickles name
  ``__main__.<Class>`` or ``spectral_predict_gui_optimized.<Class>``. ``model_io``
  resolves those names to this module while loading.
* Ensembles saved with ``task_type='auto'`` (the GUI's task radio) load as regression.
"""

from __future__ import annotations

import pickle
import sys
import types

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_predict import model_wrappers as mw
from spectral_predict.ensemble import RegionAwareWeightedEnsemble
from spectral_predict.model_io import (
    load_ensemble,
    load_model,
    predict_with_model,
    save_ensemble,
    save_model,
)

N_WL = 40
COLS = [str(1000.0 + 2 * i) for i in range(N_WL)]
CONFIG = {"type": "snv_deriv1", "window": 11, "baseline": None, "smooth": None}


def _data(classification: bool = False):
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.standard_normal((40, N_WL)), axis=1) + 50.0
    y = 0.5 * X[:, 10] - 0.3 * X[:, 30] + 0.1 * rng.standard_normal(40)
    if classification:
        y = (y > np.median(y)).astype(int)
    return X, y


def _wrappers(classification: bool):
    head = LogisticRegression(max_iter=500) if classification else Ridge(alpha=1.0)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", head)])
    subset = COLS[3:35:2]
    if classification:
        return {
            "WavelengthSubset": mw.WavelengthSubsetClassifierWrapper(
                clone(pipe), subset, all_columns=COLS
            ),
            "GAPreprocess": mw.GAPreprocessClassifierWrapper(clone(pipe), CONFIG),
            "CombinedPreprocess": mw.CombinedPreprocessClassifierWrapper(
                clone(pipe), CONFIG, subset, COLS
            ),
        }
    return {
        "WavelengthSubset": mw.WavelengthSubsetWrapper(clone(pipe), subset, all_columns=COLS),
        "GAPreprocess": mw.GAPreprocessWrapper(clone(pipe), CONFIG),
        "CombinedPreprocess": mw.CombinedPreprocessWrapper(clone(pipe), CONFIG, subset, COLS),
    }


ALL_CASES = [(kind, cls) for cls in (False, True) for kind in _wrappers(cls)]


@pytest.mark.parametrize(
    "kind,classification", ALL_CASES, ids=[f"{k}-{'clf' if c else 'reg'}" for k, c in ALL_CASES]
)
def test_wrapper_pickle_round_trip_is_identical(kind, classification):
    X, y = _data(classification)
    wrapper = _wrappers(classification)[kind].fit(X, y)
    expected = wrapper.predict(X)

    for restored in (pickle.loads(pickle.dumps(wrapper)), _joblib_round_trip(wrapper)):
        assert type(restored) is type(wrapper)
        np.testing.assert_array_equal(restored.predict(X), expected)
        if classification:
            np.testing.assert_allclose(restored.predict_proba(X), wrapper.predict_proba(X))


def _joblib_round_trip(obj):
    import io

    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    return joblib.load(buf)


@pytest.mark.parametrize("classification", [False, True], ids=["reg", "clf"])
@pytest.mark.parametrize("kind", ["GAPreprocess", "CombinedPreprocess"])
def test_serialising_leaves_the_live_transform_cache_intact(kind, classification):
    X, y = _data(classification)
    wrapper = _wrappers(classification)[kind].fit(X, y)
    cached = wrapper.transform  # builds the cache
    assert wrapper._transform is cached

    pickle.dumps(wrapper)
    _joblib_round_trip(wrapper)

    assert wrapper._transform is cached, "pickling must not clear the live object's cache"
    wrapper.predict(X)


def test_gui_module_reexports_the_backend_classes():
    gui = pytest.importorskip("spectral_predict_gui_optimized")
    for name in mw.LEGACY_PICKLE_NAMES:
        assert getattr(gui, name) is getattr(mw, name)


# --- Pre-move pickles ----------------------------------------------------------------------


@pytest.mark.parametrize("legacy_module", ["__main__", "spectral_predict_gui_optimized"])
@pytest.mark.parametrize("name", ["GAPreprocessWrapper", "WavelengthSubsetWrapper"])
def test_model_saved_with_gui_defined_wrapper_loads_without_gui(tmp_path, legacy_module, name):
    X, y = _data()
    base = getattr(mw, name)
    # A class pickled exactly as the GUI-defined one was: <legacy_module>.<name>.
    legacy_cls = type(name, (base,), {"__module__": legacy_module, "__qualname__": name})
    module = sys.modules.get(legacy_module)
    stub = None
    if module is None or legacy_module != "__main__":
        stub = types.ModuleType(legacy_module)
    saved_module = sys.modules.get(legacy_module)
    target = stub if stub is not None else module
    if stub is not None:
        sys.modules[legacy_module] = stub
    setattr(target, name, legacy_cls)
    try:
        model = _wrappers(False)[name.replace("Wrapper", "")]
        model.__class__ = legacy_cls
        model.fit(X, y)
        expected = model.predict(X)
        path = tmp_path / "legacy.dasp"
        meta = {"model_name": "PLS", "task_type": "regression", "wavelengths": COLS, "n_vars": N_WL}
        save_model(model, None, meta, str(path))
    finally:
        # The loading process has no GUI globals.
        delattr(target, name)
        if stub is not None:
            if saved_module is None:
                del sys.modules[legacy_module]
            else:
                sys.modules[legacy_module] = saved_module

    loaded = load_model(str(path))["model"]

    assert type(loaded).__module__ == "spectral_predict.model_wrappers"
    np.testing.assert_allclose(loaded.predict(X), expected)
    # The loader removes everything it added.
    assert not hasattr(sys.modules["__main__"], name) or legacy_module != "__main__"
    if saved_module is None and legacy_module != "__main__":
        assert legacy_module not in sys.modules


def test_ensemble_saved_with_auto_task_type_predicts_as_regression(tmp_path):
    X, y = _data()
    models = [Ridge(alpha=1.0).fit(X, y), PLSRegression(n_components=3).fit(X, y)]
    ens = RegionAwareWeightedEnsemble(models, ["Ridge", "PLS"], n_regions=2).fit(X, y)
    path = tmp_path / "auto.dasp"
    wl = [float(c) for c in COLS]
    save_ensemble(
        ens,
        str(path),
        {
            "ensemble_type": "region_weighted",
            "task_type": "auto",
            "wavelengths": wl,
            "n_vars": N_WL,
        },
    )

    loaded = load_ensemble(str(path))
    assert loaded["metadata"]["task_type"] == "regression"
    assert all(md["metadata"]["task_type"] == "regression" for md in loaded["base_model_dicts"])
    model_dict = {"model": loaded["ensemble"], "metadata": loaded["metadata"], "preprocessor": None}
    np.testing.assert_allclose(np.ravel(predict_with_model(model_dict, X)), ens.predict(X))
