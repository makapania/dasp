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

import contextlib
import importlib
import io
import pickle
import sys
import threading
import types
from pathlib import Path

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

GUI_MODULE = "spectral_predict_gui_optimized"


@contextlib.contextmanager
def _gui_era_class(legacy_module: str, name: str):
    """Make ``<legacy_module>.<name>`` exist only while the test pickles an object.

    Yields a subclass that pickles exactly as the GUI-defined class did. Everything is
    removed on exit, so the LOADING side runs with no GUI globals.
    """
    legacy_cls = type(
        name, (getattr(mw, name),), {"__module__": legacy_module, "__qualname__": name}
    )
    saved = sys.modules.get(legacy_module)
    target = saved if legacy_module == "__main__" else types.ModuleType(legacy_module)
    sys.modules[legacy_module] = target
    setattr(target, name, legacy_cls)
    try:
        yield legacy_cls
    finally:
        delattr(target, name)
        if saved is None:
            del sys.modules[legacy_module]
        else:
            sys.modules[legacy_module] = saved


def _fitted_legacy(legacy_cls, name):
    X, y = _data()
    model = _wrappers(False)[name.replace("Wrapper", "")]
    model.__class__ = legacy_cls
    return model.fit(X, y)


def _global_state():
    """Everything a global-mutation shim could leave behind."""
    main = sys.modules["__main__"]
    return (
        tuple(n for n in mw.LEGACY_PICKLE_NAMES if hasattr(main, n)),
        sys.modules.get(GUI_MODULE),
    )


@pytest.mark.parametrize("legacy_module", ["__main__", GUI_MODULE])
@pytest.mark.parametrize("name", ["GAPreprocessWrapper", "WavelengthSubsetWrapper"])
def test_model_saved_with_gui_defined_wrapper_loads_without_gui(tmp_path, legacy_module, name):
    X, _ = _data()
    path = tmp_path / "legacy.dasp"
    with _gui_era_class(legacy_module, name) as legacy_cls:
        model = _fitted_legacy(legacy_cls, name)
        expected = model.predict(X)
        meta = {"model_name": "PLS", "task_type": "regression", "wavelengths": COLS, "n_vars": N_WL}
        save_model(model, None, meta, str(path))

    before = _global_state()
    loaded = load_model(str(path))["model"]

    assert type(loaded) is getattr(mw, name)
    np.testing.assert_allclose(loaded.predict(X), expected)
    assert _global_state() == before, "loading must not touch sys.modules or __main__"


def test_plain_pickle_with_gui_defined_wrapper_loads_via_legacy_unpickler():
    """The GUI's raw .pkl prediction-model path."""
    X, _ = _data()
    with _gui_era_class("__main__", "GAPreprocessWrapper") as legacy_cls:
        model = _fitted_legacy(legacy_cls, "GAPreprocessWrapper")
        blob = pickle.dumps(model)
    loaded = mw.LegacyWrapperUnpickler(io.BytesIO(blob)).load()
    assert type(loaded) is mw.GAPreprocessWrapper
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


# Pause points reached from INSIDE an unpickling (via __reduce__), to force overlap.
_PAUSE: dict[str, threading.Barrier] = {}


def _pause_inside_load():
    _PAUSE["arrived"].wait(timeout=60)
    _PAUSE["resume"].wait(timeout=60)
    return "paused"


def _interrupt_inside_load():
    raise KeyboardInterrupt


class _PauseHere:
    def __reduce__(self):
        return (_pause_inside_load, ())


class _InterruptHere:
    def __reduce__(self):
        return (_interrupt_inside_load, ())


def _legacy_payload(tmp_path, tag: str, extra) -> tuple[Path, np.ndarray]:
    X, _ = _data()
    path = tmp_path / f"{tag}.pkl"
    with _gui_era_class("__main__", "GAPreprocessWrapper") as legacy_cls:
        model = _fitted_legacy(legacy_cls, "GAPreprocessWrapper")
        joblib.dump((extra, model), path)
    return path, model.predict(X)


def test_concurrent_legacy_loads_and_gui_import_do_not_interfere(tmp_path):
    from spectral_predict.model_io import _joblib_load

    paths = [_legacy_payload(tmp_path, f"p{i}", _PauseHere()) for i in range(2)]
    _PAUSE["arrived"] = threading.Barrier(3)
    _PAUSE["resume"] = threading.Barrier(3)
    results: dict[int, object] = {}
    errors: list[BaseException] = []

    def load(i):
        try:
            results[i] = _joblib_load(paths[i][0])
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=load, args=(i,)) for i in range(2)]
    before = _global_state()
    for t in threads:
        t.start()
    try:
        _PAUSE["arrived"].wait(timeout=60)
        # Both loads are now paused mid-unpickle.
        assert _global_state() == before
        gui = importlib.import_module(GUI_MODULE)  # the real module, not a stub
        assert hasattr(gui, "SpectralPredictApp")
        assert gui.GAPreprocessWrapper is mw.GAPreprocessWrapper
    finally:
        _PAUSE["resume"].wait(timeout=60)
        for t in threads:
            t.join(timeout=60)

    assert not errors, errors
    X, _ = _data()
    for i, (_, expected) in enumerate(paths):
        marker, model = results[i]
        assert marker == "paused" and type(model) is mw.GAPreprocessWrapper
        np.testing.assert_allclose(model.predict(X), expected)


def test_interrupted_legacy_load_leaves_no_residue(tmp_path):
    from spectral_predict.model_io import _joblib_load

    path, _ = _legacy_payload(tmp_path, "interrupt", _InterruptHere())
    before = _global_state()
    with pytest.raises(KeyboardInterrupt):
        _joblib_load(path)
    assert _global_state() == before


def test_standalone_auto_regressor_loads_as_regression_classifier_untouched(tmp_path):
    X, y = _data()
    meta = {"model_name": "Ridge", "task_type": "auto", "wavelengths": COLS, "n_vars": N_WL}
    save_model(_wrappers(False)["GAPreprocess"].fit(X, y), None, meta, str(tmp_path / "r.dasp"))
    Xc, yc = _data(classification=True)
    save_model(_wrappers(True)["GAPreprocess"].fit(Xc, yc), None, meta, str(tmp_path / "c.dasp"))

    assert load_model(str(tmp_path / "r.dasp"))["metadata"]["task_type"] == "regression"
    assert load_model(str(tmp_path / "c.dasp"))["metadata"]["task_type"] == "auto"


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


def test_ensemble_with_classifier_member_keeps_auto_task_type(tmp_path):
    from spectral_predict.ensemble import SimpleAverageEnsemble

    X, y = _data(classification=True)
    members = [LogisticRegression(max_iter=500).fit(X, y), Ridge(alpha=1.0).fit(X, y)]
    ens = SimpleAverageEnsemble(members, ["LogReg", "Ridge"]).fit(X, y)
    path = tmp_path / "auto_clf.dasp"
    wl = [float(c) for c in COLS]
    save_ensemble(
        ens,
        str(path),
        {"ensemble_type": "simple_average", "task_type": "auto", "wavelengths": wl, "n_vars": N_WL},
    )
    assert load_ensemble(str(path))["metadata"]["task_type"] == "auto"


# --- _joblib_load parity with joblib.load ----------------------------------------------------


@pytest.mark.parametrize("compress", [0, 3], ids=["uncompressed", "compress3"])
def test_joblib_load_matches_joblib_on_ordinary_payloads(tmp_path, compress):
    from spectral_predict.model_io import _joblib_load

    X, y = _data()
    pipe = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression(n_components=3))]).fit(
        X, y
    )
    big = np.random.default_rng(1).standard_normal((400, 400))  # ~1.3 MB
    path = tmp_path / f"payload_{compress}.pkl"
    joblib.dump({"pipe": pipe, "big": big, "meta": {"a": 1}}, path, compress=compress)

    ours, theirs = _joblib_load(path), joblib.load(path)

    np.testing.assert_array_equal(ours["big"], theirs["big"])
    np.testing.assert_array_equal(ours["big"], big)
    np.testing.assert_array_equal(ours["pipe"].predict(X), theirs["pipe"].predict(X))
    assert ours["meta"] == theirs["meta"]


def test_joblib_load_translates_unicode_errors_like_joblib(tmp_path, monkeypatch):
    from spectral_predict import model_io

    path = tmp_path / "x.pkl"
    joblib.dump({"a": 1}, path)

    def broken_load(self):
        raise UnicodeDecodeError("ascii", b"\xff", 0, 1, "bad byte")

    monkeypatch.setattr(model_io._LegacyAwareNumpyUnpickler, "load", broken_load)
    with pytest.raises(ValueError, match="python 2"):
        model_io._joblib_load(path)
