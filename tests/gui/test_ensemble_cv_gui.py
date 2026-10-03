"""GUI ensemble training: honest CV, label-indexed targets, refittable wrappers.

* R002: ``_train_ensembles`` reported R2CV from ensembles whose base models had been
  fitted on every calibration row, including the scored fold.
* R018: it indexed the specimen-ID-indexed ``y_filtered`` Series with positional fold
  indices (KeyError for string IDs under pandas 3, wrong rows for permuted int IDs).
* R021: any GA / Combined / WavelengthSubset wrapper switched the whole ensemble to
  ``refit_base_models=False``, so weights and the stacking meta-model were learned from
  in-sample predictions. The wrappers are refittable; they must be refitted.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pytestmark = pytest.mark.gui

N_WL = 60
WAVELENGTHS = [1000.0 + 2 * i for i in range(N_WL)]
METHOD_VARS = (
    "ensemble_simple_average",
    "ensemble_region_weighted",
    "ensemble_mixture_experts",
    "ensemble_stacking",
    "ensemble_stacking_region",
)


def _spectra(n: int = 40, seed: int = 0, noise_target: bool = False):
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.standard_normal((n, N_WL)), axis=1) + 50.0
    if noise_target:
        y = rng.standard_normal(n) * 3 + 10
    else:
        y = 0.5 * X[:, 10] - 0.3 * X[:, 40] + 0.2 * rng.standard_normal(n)
    return pd.DataFrame(X, columns=WAVELENGTHS), y


def _rows() -> pd.DataFrame:
    subset = ",".join(str(w) for w in WAVELENGTHS[5:45:2])
    rows = [
        # unwrapped, strongly overfitting member
        dict(
            Model="RandomForest",
            Params=str({"n_estimators": 10, "random_state": 0}),
            Preprocess="raw",
            Deriv=0,
            Window=17,
            Poly=2,
        ),
        # legacy CARS-style row -> WavelengthSubsetWrapper(RandomForest)
        dict(
            Model="RandomForest",
            Params=str({"n_estimators": 10, "random_state": 1}),
            Preprocess="sg1",
            Deriv=1,
            Window=11,
            Poly=2,
            all_vars=subset,
        ),
        # GA preprocessing row -> GAPreprocessWrapper(PLS)
        dict(
            Model="PLS",
            Params=str({"n_components": 3}),
            Preprocess="deriv1_w11_pls",
            Deriv=1,
            Window=11,
            Poly=2,
        ),
    ]
    df = pd.DataFrame(rows)
    # Real search rows carry SubsetTag/n_vars; without all_vars the ensemble rebuild
    # uses every column only when they say so (fix/wavelength-mapping).
    df["SubsetTag"] = ["full", "top20", "full"]
    df["n_vars"] = [N_WL, 20, N_WL]
    df["CompositeScore"] = np.arange(len(df), dtype=float)
    df["R2cv"] = np.nan
    df["RMSE"] = np.nan
    df["Select"] = True
    return df


@contextlib.contextmanager
def _ensemble_settings(app, methods=METHOD_VARS, n_regions=3):
    saved = {name: getattr(app, name).get() for name in METHOD_VARS}
    saved_regions = app.ensemble_n_regions.get()
    saved_val = app.validation_enabled.get()
    try:
        for name in METHOD_VARS:
            getattr(app, name).set(name in methods)
        app.ensemble_n_regions.set(n_regions)
        app.validation_enabled.set(False)
        yield
    finally:
        for name, value in saved.items():
            getattr(app, name).set(value)
        app.ensemble_n_regions.set(saved_regions)
        app.validation_enabled.set(saved_val)


def _train(app, X, y, monkeypatch):
    logs: list[str] = []
    monkeypatch.setattr(app, "_log_progress", lambda msg: logs.append(str(msg)))
    with contextlib.redirect_stdout(io.StringIO()):
        results, trained = app._train_ensembles(_rows(), X, y, "regression", is_manual_retrain=True)
    return results, trained, logs


# --- R018 + R002 through the real GUI path ----------------------------------------------

_REFERENCE: dict[str, list] = {}  # RangeIndex run, shared by the parametrized cases


@pytest.mark.parametrize("variant", ["string_ids", "gapped_int", "permuted_int"])
def test_train_ensembles_accepts_label_indexed_targets(gui_app, monkeypatch, variant):
    X, y = _spectra()
    n = len(y)
    index = {
        "string_ids": pd.Index([f"Spectrum {i:05d}" for i in range(n)]),
        "gapped_int": pd.Index(np.arange(n) * 2 + 5),
        "permuted_int": pd.Index(np.random.default_rng(2).permutation(n)),
    }[variant]
    X_lab = X.set_axis(index, axis=0)
    y_lab = pd.Series(y, index=index)

    with _ensemble_settings(gui_app):
        res_lab, _, logs = _train(gui_app, X_lab, y_lab, monkeypatch)
        if "positional" not in _REFERENCE:
            _REFERENCE["positional"] = _train(gui_app, X, y, monkeypatch)[0]

    assert res_lab is not None, "\n".join(logs[-40:])
    assert len(res_lab) == len(METHOD_VARS), "\n".join(line for line in logs if "[X]" in line)
    by_method = {r["method"]: r["r2"] for r in _REFERENCE["positional"]}
    for r in res_lab:
        assert np.isfinite(r["r2"])
        assert r["r2"] == pytest.approx(by_method[r["method"]])


def test_train_ensembles_refuses_classification_runs(gui_app, monkeypatch):
    """Manual 'Train Ensemble' after a classification run must not fit class labels."""
    X, y = _spectra()
    labels = pd.Series(np.where(y > np.median(y), 1, 0), index=X.index)
    logs: list[str] = []
    monkeypatch.setattr(gui_app, "_log_progress", lambda msg: logs.append(str(msg)))
    with _ensemble_settings(gui_app):
        out = gui_app._train_ensembles(_rows(), X, labels, "classification", is_manual_retrain=True)
    assert out == (None, None)
    assert any("regression only" in line for line in logs)


def test_train_ensembles_cv_is_honest_for_memorising_members_on_noise(gui_app, monkeypatch):
    X, y = _spectra(noise_target=True)
    X.index = [f"S{i}" for i in range(len(y))]
    y = pd.Series(y, index=X.index)

    with _ensemble_settings(gui_app):
        results, _, logs = _train(gui_app, X, y, monkeypatch)

    assert results is not None and len(results) == len(METHOD_VARS), "\n".join(logs[-40:])
    for r in results:
        # The old loop scored these with members fitted on the scored rows: R2CV ~0.8.
        assert r["r2"] < 0.3, (r["method"], r["r2"], r["r2_cal"])


# --- R021: wrapped members are refitted -------------------------------------------------


def test_gui_wrappers_refit_like_their_full_data_fit(gui_app):
    X, y = _spectra()
    Xs = X.copy()
    Xs.columns = [str(c) for c in Xs.columns]
    with contextlib.redirect_stdout(io.StringIO()):
        recon = gui_app._reconstruct_models_from_results(_rows(), Xs, y, "regression")
    kinds = {type(m).__name__ for m, _, _ in recon}
    assert {"WavelengthSubsetWrapper", "GAPreprocessWrapper"} <= kinds

    for model, _, _ in recon:
        refit = clone(model).fit(Xs, y)
        np.testing.assert_allclose(np.ravel(refit.predict(Xs)), np.ravel(model.predict(Xs)))
        # Full-width arrays (validation sets, numpy callers) select the same columns.
        np.testing.assert_allclose(
            np.ravel(model.predict(Xs.to_numpy())), np.ravel(model.predict(Xs))
        )


def test_combined_wrapper_refits_like_its_full_data_fit():
    from spectral_predict_gui_optimized import CombinedPreprocessWrapper

    X, y = _spectra()
    cols = [str(c) for c in X.columns]
    wrapper = CombinedPreprocessWrapper(
        pipeline=Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        preprocess_config={"type": "snv_deriv1", "window": 11, "baseline": None, "smooth": None},
        wavelength_cols=cols[3:50:3],
        all_columns=cols,
    ).fit(X.to_numpy(), y)
    refit = clone(wrapper).fit(X.to_numpy(), y)
    np.testing.assert_allclose(refit.predict(X.to_numpy()), wrapper.predict(X.to_numpy()))


def test_wavelength_subset_wrapper_rejects_unmappable_array():
    from spectral_predict_gui_optimized import WavelengthSubsetWrapper

    X, y = _spectra()
    cols = [str(c) for c in X.columns]
    Xs = X.set_axis(cols, axis=1)
    wrapper = WavelengthSubsetWrapper(PLSRegression(2), cols[:10], all_columns=cols).fit(Xs, y)
    with pytest.raises(ValueError, match="cannot be mapped"):
        wrapper.predict(X.to_numpy()[:, :30])


def test_train_ensembles_stacking_meta_model_sees_held_out_wrapped_predictions(
    gui_app, monkeypatch
):
    """Mixed wrapped + unwrapped members: the deployed meta-model's inputs are OOF."""
    from spectral_predict import ensemble as ens_mod

    seen: list[np.ndarray] = []

    class RecordingRidge(Ridge):
        def fit(self, X, y, sample_weight=None):
            seen.append(np.array(X, copy=True))
            return super().fit(X, y, sample_weight=sample_weight)

    monkeypatch.setattr(ens_mod, "Ridge", RecordingRidge)
    X, y = _spectra()
    X.index = [f"S{i}" for i in range(len(y))]
    y_ser = pd.Series(y, index=X.index)

    with _ensemble_settings(gui_app, methods=("ensemble_stacking",)):
        results, trained, logs = _train(gui_app, X, y_ser, monkeypatch)
    assert results is not None, "\n".join(logs[-40:])

    deployed = trained["stacking"]
    meta_X = [s for s in seen if len(s) == len(y)][-1]  # the full-data (deployed) fit
    assert meta_X.shape[1] == len(deployed.models) == 3

    kf = KFold(5, shuffle=True, random_state=42)
    for j, model in enumerate(deployed.models):
        expected = np.zeros(len(y))
        for train_idx, val_idx in kf.split(X):
            fold = clone(model).fit(X.iloc[train_idx], y[train_idx])
            expected[val_idx] = np.ravel(fold.predict(X.iloc[val_idx]))
        np.testing.assert_allclose(meta_X[:, j], expected, rtol=1e-6, atol=1e-8)
        in_sample = np.ravel(model.predict(X))
        assert not np.allclose(meta_X[:, j], in_sample), type(model).__name__


# --- Persistence (review round 1) ---------------------------------------------------------


def test_saved_ensemble_stores_out_of_fold_predictions_for_uncertainty(
    gui_app, monkeypatch, tmp_path
):
    """The .dasp CV data must be the honest OOF predictions, not in-sample re-predictions."""
    import spectral_predict_gui_optimized as gui_mod
    from spectral_predict.model_io import load_ensemble

    X, y = _spectra()
    X.index = [f"S{i}" for i in range(len(y))]
    y_ser = pd.Series(y, index=X.index)
    with _ensemble_settings(gui_app, methods=("ensemble_region_weighted",)):
        results, _, logs = _train(gui_app, X, y_ser, monkeypatch)
    assert results is not None, "\n".join(logs[-40:])
    result = results[0]
    assert len(result["cv_predictions"]) == len(y)

    path = tmp_path / "ens.dasp"
    monkeypatch.setattr(gui_mod.filedialog, "asksaveasfilename", lambda **kw: str(path))
    monkeypatch.setattr(gui_app, "_get_selected_ensemble", lambda: result)
    monkeypatch.setattr(gui_app, "ensemble_results", results, raising=False)
    monkeypatch.setattr(gui_app, "ensemble_X", X, raising=False)
    monkeypatch.setattr(gui_app, "ensemble_y", y_ser, raising=False)
    gui_app._save_selected_ensemble()

    assert path.exists(), "\n".join(logs[-20:])
    cv_data = load_ensemble(str(path))["base_model_dicts"][0]["cv_data"]
    np.testing.assert_allclose(cv_data["cv_predictions"], result["cv_predictions"])
    np.testing.assert_allclose(cv_data["cv_residuals"], result["cv_predictions"] - y)
    in_sample = result["ensemble"].predict(X)
    assert not np.allclose(cv_data["cv_predictions"], in_sample)


def test_saved_ensemble_with_subset_wrapper_predicts_full_width_numpy(gui_app, tmp_path):
    """predict_with_model hands the loaded ensemble a full-width array; subset members cope."""
    from spectral_predict.ensemble import RegionAwareWeightedEnsemble
    from spectral_predict.model_io import load_ensemble, predict_with_model, save_ensemble

    X, y = _spectra()
    Xs = X.set_axis([str(c) for c in X.columns], axis=1)
    with contextlib.redirect_stdout(io.StringIO()):
        recon = gui_app._reconstruct_models_from_results(_rows(), Xs, y, "regression")
    models = [m for m, _, _ in recon]
    assert "WavelengthSubsetWrapper" in {type(m).__name__ for m in models}
    ens = RegionAwareWeightedEnsemble(models, [n for _, n, _ in recon], n_regions=3).fit(Xs, y)

    path = tmp_path / "subset_ens.dasp"
    save_ensemble(
        ens,
        str(path),
        {
            "ensemble_type": "region_weighted",
            "task_type": "regression",
            "wavelengths": WAVELENGTHS,
            "n_vars": len(WAVELENGTHS),
        },
    )
    loaded = load_ensemble(str(path))
    model_dict = {"model": loaded["ensemble"], "metadata": loaded["metadata"], "preprocessor": None}

    preds = predict_with_model(model_dict, X.to_numpy())
    np.testing.assert_allclose(np.ravel(preds), ens.predict(Xs), rtol=1e-6)
