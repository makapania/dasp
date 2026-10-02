"""One wavelength-to-column contract for training, prediction and validation.

Covers the 2026-09-28 review findings:

- R009/R026/R112: Tab 7 mapped wavelengths to columns by first hit within +/-0.5 while
  prediction used +/-0.01, so on grids finer than 0.5 the model trained on the
  neighbouring channel. Both sides now use ``match_wavelengths``.
- R031/R078: ``all_vars`` was written with ``%g`` and looked up by exact float equality,
  so validation silently scored subset rows on the full spectrum (or skipped them).
- R016: a stale search label encoder was saved next to a raw-numeric-label model
  (np.int64 keys crashed json; float keys decoded to the wrong classes).
"""

from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from spectral_predict.model_io import load_model, predict_with_model, save_model
from spectral_predict.preprocess import build_preprocessing_pipeline
from spectral_predict.search import compute_validation_metrics_for_top_models
from spectral_predict.wavelength_matching import (
    WavelengthMatchError,
    format_wavelength_list,
    match_wavelengths,
    resolve_wavelength_list,
)


def _g(values) -> str:
    """The pre-fix ``%g`` writer, for building legacy rows."""
    return ",".join(f"{float(w):g}" for w in values)


def _old_first_hit(requested, axis) -> list[int]:
    """The pre-fix Tab 7 rule: first column within 0.5."""
    axis = np.asarray(axis, dtype=float)
    return [int(np.flatnonzero(np.abs(axis - w) < 0.5)[0]) for w in requested]


def _axis(spacing: float, order: str, n: int = 120, start: float = 1000.0) -> np.ndarray:
    axis = start + spacing * np.arange(n)
    return axis if order == "asc" else axis[::-1].copy()


def _spectra(n_samples: int, axis: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, axis.size)
    base = 0.5 + 0.2 * np.sin(6 * x)
    peaks = rng.normal(size=(n_samples, 3))
    centres = (0.2, 0.5, 0.8)
    X = base + sum(peaks[:, [k]] * np.exp(-((x - c) ** 2) / 0.002) for k, c in enumerate(centres))
    return X + rng.normal(scale=0.01, size=X.shape)


# --------------------------------------------------------------------------------------
# The helper itself
# --------------------------------------------------------------------------------------


class TestMatchWavelengths:
    @pytest.mark.parametrize("spacing", [0.3, 0.482])
    @pytest.mark.parametrize("order", ["asc", "desc"])
    def test_exact_values_map_to_their_own_column_on_fine_grids(self, spacing, order):
        axis = _axis(spacing, order)
        wanted = [40, 41, 7, 100, 99]  # adjacent pairs and out of order
        assert match_wavelengths(axis[wanted], axis).tolist() == wanted

    @pytest.mark.parametrize("spacing", [0.3, 0.482])
    def test_old_first_hit_rule_took_the_neighbour(self, spacing):
        """Pins the R009 mechanism: the replaced rule shifts every interior channel."""
        axis = _axis(spacing, "asc")
        wanted = [40, 41, 100]
        assert _old_first_hit(axis[wanted], axis) != wanted
        assert match_wavelengths(axis[wanted], axis).tolist() == wanted

    def test_header_noise_within_tolerance_resolves(self):
        axis = np.arange(1000.0, 1010.0)
        assert match_wavelengths([1003.004, 1001.0], axis).tolist() == [3, 1]

    def test_missing_raises_with_the_values(self):
        axis = np.arange(1000.0, 1010.0)
        with pytest.raises(WavelengthMatchError) as exc:
            match_wavelengths([1001.0, 1500.0], axis)
        assert exc.value.missing == [1500.0]

    def test_window_holding_two_channels_is_ambiguous_not_first_hit(self):
        axis = np.array([1000.0, 1000.3, 1000.6, 1000.9])
        with pytest.raises(WavelengthMatchError) as exc:
            match_wavelengths([1000.45], axis, tolerance=0.5)
        assert exc.value.ambiguous == [1000.45]

    def test_exact_value_wins_even_inside_a_wide_window(self):
        axis = np.array([1000.0, 1000.3, 1000.6, 1000.9])
        assert match_wavelengths([1000.6], axis, tolerance=0.5).tolist() == [2]

    def test_duplicated_axis_value_is_ambiguous(self):
        with pytest.raises(WavelengthMatchError):
            match_wavelengths([1001.0], [1000.0, 1001.0, 1001.0])

    def test_two_values_on_one_column_raise(self):
        axis = np.arange(1000.0, 1010.0)
        with pytest.raises(WavelengthMatchError, match="both map"):
            match_wavelengths([1001.0, 1001.004], axis)

    def test_nan_axis_raises(self):
        with pytest.raises(WavelengthMatchError):
            match_wavelengths([1.0], [1.0, np.nan])


class TestStoredWavelengthLists:
    def test_writer_round_trips_exactly(self):
        axis = 1e7 / np.arange(1000, 1100)
        text = format_wavelength_list(axis[[5, 2, 80]])
        assert [float(t) for t in text.split(",")] == axis[[5, 2, 80]].tolist()
        assert resolve_wavelength_list(text, axis).tolist() == [5, 2, 80]

    @pytest.mark.parametrize(
        "axis",
        [
            1e7 / np.arange(1000, 1100),  # nm -> cm-1 conversion, >6 significant digits
            3999.6419 - 1.9285 * np.arange(100),  # OPUS-style wavenumbers
            10000.25 + 4.0 * np.arange(100),  # above 9999.99: %g keeps one decimal
        ],
        ids=["1e7_over_x", "opus", "above_1e4"],
    )
    def test_legacy_g_rows_still_resolve(self, axis):
        wanted = [3, 50, 51, 7]
        text = _g(axis[wanted])
        assert {float(t) for t in text.split(",")} != set(axis[wanted])  # really lossy
        assert resolve_wavelength_list(text, axis).tolist() == wanted

    def test_legacy_g_row_on_a_grid_it_cannot_distinguish_raises(self):
        axis = 12345.6 + 0.01 * np.arange(20)  # %g keeps 12345.7 for 12345.67 and 12345.70
        with pytest.raises(WavelengthMatchError, match="more than one"):
            resolve_wavelength_list(_g(axis[[7]]), axis)

    def test_integer_axis_g_text_is_exact(self):
        axis = np.arange(1100.0, 1150.0)
        assert resolve_wavelength_list(_g(axis[[0, 49]]), axis).tolist() == [0, 49]

    @pytest.mark.parametrize("text", ["", " , ", "1000.0,abc", "nan"])
    def test_corrupt_text_raises(self, text):
        with pytest.raises(WavelengthMatchError):
            resolve_wavelength_list(text, np.arange(1000.0, 1010.0))

    def test_partial_match_is_a_failure(self):
        axis = np.arange(1000.0, 1010.0)
        with pytest.raises(WavelengthMatchError) as exc:
            resolve_wavelength_list("1001.0,1002.0,9999.0", axis)
        assert exc.value.missing == [9999.0]


# --------------------------------------------------------------------------------------
# Save -> load -> predict identity (the Tab 7 training mapping and model_io agree)
# --------------------------------------------------------------------------------------


def _fit_like_tab7(X: np.ndarray, axis: np.ndarray, selected: np.ndarray):
    """Path A as the Tab 7 refit runs it: preprocess full, map, subset, fit."""
    prep = Pipeline(build_preprocessing_pipeline("snv_deriv", deriv=1, window=11, polyorder=2))
    X_pre = prep.fit_transform(X)
    idx = match_wavelengths(selected, axis)
    y = X_pre[:, idx[: min(3, idx.size)]].sum(axis=1)
    model = PLSRegression(n_components=3).fit(X_pre[:, idx], y)
    metadata = {
        "model_name": "PLS",
        "task_type": "regression",
        "wavelengths": [float(axis[i]) for i in idx],
        "full_wavelengths": [float(w) for w in axis],
        "use_full_spectrum_preprocessing": True,
        "n_vars": int(idx.size),
        "preprocessing": "snv_deriv",
    }
    return model, prep, metadata, X_pre[:, idx]


@pytest.mark.parametrize("spacing", [0.3, 0.482])
@pytest.mark.parametrize("order", ["asc", "desc"])
@pytest.mark.parametrize("selection", ["subset", "full"])
def test_fine_grid_save_load_predict_identity(tmp_path, spacing, order, selection):
    axis = _axis(spacing, order)
    X = _spectra(40, axis)
    sel = [30, 31, 32, 60, 61, 95] if selection == "subset" else list(range(axis.size))
    model, prep, metadata, X_train = _fit_like_tab7(X, axis, axis[sel])

    path = tmp_path / "fine.dasp"
    save_model(model, prep, metadata, path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a fixed-era model must load without a warning
        loaded = load_model(path)
    got = predict_with_model(loaded, pd.DataFrame(X, columns=axis))

    # Float-ulp tolerance only (transform vs fit_transform); a one-channel shift on
    # SG-derivative features is orders of magnitude larger.
    np.testing.assert_allclose(np.ravel(got), np.ravel(model.predict(X_train)), rtol=0, atol=1e-12)
    assert loaded["metadata"]["wavelengths"] == [float(axis[i]) for i in sel]


def _strip_matching_stamp(path: Path) -> None:
    """Rewrite a .dasp as a pre-fix save would have left it."""
    with zipfile.ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    meta = json.loads(members["metadata.json"])
    meta.pop("wavelength_matching")
    members["metadata.json"] = json.dumps(meta).encode("utf-8")
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


class TestLegacySavedModels:
    def test_fine_grid_subset_model_warns_on_load(self, tmp_path):
        axis = _axis(0.3, "asc")
        X = _spectra(30, axis)
        model, prep, metadata, _ = _fit_like_tab7(X, axis, axis[[30, 31, 60]])
        path = tmp_path / "old.dasp"
        save_model(model, prep, metadata, path)
        _strip_matching_stamp(path)

        with pytest.warns(UserWarning, match="neighbouring channel"):
            loaded = load_model(path)
        assert "3 of 3" in loaded["wavelength_mapping_warning"]

    def test_one_nm_grid_model_loads_silently(self, tmp_path):
        axis = _axis(1.0, "asc")
        X = _spectra(30, axis)
        model, prep, metadata, _ = _fit_like_tab7(X, axis, axis[[30, 31, 60]])
        path = tmp_path / "old_1nm.dasp"
        save_model(model, prep, metadata, path)
        _strip_matching_stamp(path)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = load_model(path)
        assert loaded["wavelength_mapping_warning"] is None

    def test_old_g_rounded_metadata_wavelengths_still_predict(self, tmp_path):
        """Pre-fix Tab 7 saved the %g-parsed all_vars values as metadata wavelengths."""
        axis = 3999.6419 - 1.9285 * np.arange(120)
        X = _spectra(30, axis)
        model, prep, metadata, X_train = _fit_like_tab7(X, axis, axis[[10, 20, 30]])
        metadata["wavelengths"] = [float(f"{w:g}") for w in metadata["wavelengths"]]
        path = tmp_path / "g.dasp"
        save_model(model, prep, metadata, path)
        got = predict_with_model(load_model(path), pd.DataFrame(X, columns=axis))
        np.testing.assert_allclose(
            np.ravel(got), np.ravel(model.predict(X_train)), rtol=0, atol=1e-12
        )


def test_predict_dataframe_missing_wavelength_still_says_missing(tmp_path):
    axis = np.arange(1000.0, 1050.0)
    X = _spectra(20, axis)
    model = PLSRegression(n_components=2).fit(X, X[:, 0])
    metadata = {
        "model_name": "PLS",
        "task_type": "regression",
        "wavelengths": axis.tolist(),
        "n_vars": axis.size,
    }
    save_model(model, None, metadata, tmp_path / "m.dasp")
    loaded = load_model(tmp_path / "m.dasp")
    with pytest.raises(ValueError, match="Missing 1 required wavelengths"):
        predict_with_model(loaded, pd.DataFrame(X[:, :-1], columns=axis[:-1]))


# --------------------------------------------------------------------------------------
# Validation rebuild of a results row (R031)
# --------------------------------------------------------------------------------------


def _regression_row(all_vars: str, subset_tag: str = "top10", lvs: int = 2) -> dict:
    return {
        "Model": "PLS",
        "Task": "regression",
        "PreprocessBase": "raw",
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 0,
        "Poly": 0,
        "LVs": lvs,
        "Params": "{}",
        "SubsetTag": subset_tag,
        "CompositeScore": 1.0,
        "R2cv": 0.5,
        "top_vars": "N/A",
        "all_vars": all_vars,
    }


@pytest.fixture
def wavenumber_split():
    axis = 1e7 / np.arange(1000, 1060)  # descending cm-1 with 15+ significant digits
    rng = np.random.default_rng(3)
    X = rng.normal(size=(70, axis.size))
    y = 2.0 * X[:, 5] - X[:, 12] + 0.5 * X[:, 40] + rng.normal(scale=0.3, size=70)
    return axis, X[:50], y[:50], X[50:], y[50:]


def _validate(rows, split):
    axis, X_tr, y_tr, X_va, y_va = split
    df = pd.DataFrame(rows)
    df["CompositeScore"] = np.arange(len(df), dtype=float)
    return compute_validation_metrics_for_top_models(
        df, X_tr, y_tr, X_va, y_va, "regression", axis, top_n=len(df)
    )


def test_validation_scores_the_subset_not_the_full_spectrum(wavenumber_split):
    axis, X_tr, y_tr, X_va, y_va = wavenumber_split
    sel = [5, 12, 40, 3]
    out = _validate(
        [
            _regression_row(format_wavelength_list(axis[sel])),
            _regression_row(_g(axis[sel])),  # legacy %g row, same subset
            _regression_row(format_wavelength_list(axis), subset_tag="full"),
        ],
        wavenumber_split,
    )
    # Reference: the same rebuild on an integer axis, where the subset was never at risk.
    int_axis = np.arange(axis.size, dtype=float)
    ref = compute_validation_metrics_for_top_models(
        pd.DataFrame([_regression_row(_g(int_axis[sel]))]),
        X_tr,
        y_tr,
        X_va,
        y_va,
        "regression",
        int_axis,
        top_n=1,
    )
    rmsep = float(ref.loc[0, "RMSEP"])

    assert np.isfinite(rmsep)
    assert out.loc[0, "RMSEP"] == pytest.approx(rmsep, rel=1e-12)
    assert out.loc[1, "RMSEP"] == pytest.approx(rmsep, rel=1e-12)
    assert out.loc[0, "R2pred"] != pytest.approx(out.loc[2, "R2pred"])
    assert out.attrs["validation_failures"] == {}


@pytest.mark.parametrize("all_vars", ["9999.0,9998.0", None], ids=["zero_match", "partial_match"])
def test_unmappable_all_vars_fails_the_row_visibly(wavenumber_split, all_vars):
    axis = wavenumber_split[0]
    if all_vars is None:
        all_vars = format_wavelength_list(axis[[5, 12]]) + ",9999.0"
    out = _validate([_regression_row(all_vars)], wavenumber_split)
    assert np.isnan(out.loc[0, "RMSEP"]) and np.isnan(out.loc[0, "R2pred"])
    assert "does not match the spectral axis" in out.attrs["validation_failures"][0]


# --------------------------------------------------------------------------------------
# Label encoder ownership (R016)
# --------------------------------------------------------------------------------------


def _clf_data(labels):
    rng = np.random.default_rng(1)
    y = np.repeat(np.asarray(labels), 15)
    X = rng.normal(size=(y.size, 20)) + np.searchsorted(np.unique(y), y)[:, None]
    return X, y


def _clf_metadata(n_vars: int) -> dict:
    return {
        "model_name": "RandomForest",
        "task_type": "classification",
        "wavelengths": [1000.0 + i for i in range(n_vars)],
        "n_vars": n_vars,
    }


def _save_load(tmp_path, model, encoder, n_vars):
    path = tmp_path / "clf.dasp"
    save_model(model, None, _clf_metadata(n_vars), path, label_encoder=encoder)
    return load_model(path)


@pytest.mark.parametrize("labels", [["a", "b", "c"], [1, 2, 3], [1.0, 2.0, 3.0]])
def test_encoder_the_model_was_trained_with_round_trips(tmp_path, labels):
    X, y = _clf_data(labels)
    encoder = LabelEncoder().fit(y)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, encoder.transform(y))

    loaded = _save_load(tmp_path, model, encoder, X.shape[1])

    assert loaded["label_encoder"] is not None
    assert set(loaded["metadata"]["label_mapping"]) == {str(c) for c in encoder.classes_}
    got = predict_with_model(loaded, X, validate_wavelengths=False)
    np.testing.assert_array_equal(got, encoder.inverse_transform(model.predict(X)))


@pytest.mark.parametrize("labels", [[1, 2, 3], [1.0, 2.0, 3.0]], ids=["int", "float"])
def test_stale_encoder_is_not_saved_with_a_raw_label_model(tmp_path, labels):
    X, y = _clf_data(labels)
    stale = LabelEncoder().fit(y)  # the Bayesian/NSGA search's encoder
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)

    with pytest.warns(UserWarning, match="does not match the model"):
        loaded = _save_load(tmp_path, model, stale, X.shape[1])

    assert loaded["label_encoder"] is None
    assert loaded["metadata"]["has_label_encoder"] is False
    np.testing.assert_array_equal(
        predict_with_model(loaded, X, validate_wavelengths=False), model.predict(X)
    )


def test_legacy_artifact_with_stale_float_encoder_predicts_raw_labels(tmp_path):
    X, y = _clf_data([1.0, 2.0, 3.0])
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    path = tmp_path / "legacy.dasp"
    save_model(model, None, _clf_metadata(X.shape[1]), path)
    # A pre-fix save could hold the search encoder next to this raw-label model.
    enc_file = tmp_path / "label_encoder.pkl"
    joblib.dump(LabelEncoder().fit(y), enc_file)
    with zipfile.ZipFile(path, "a") as zf:
        zf.write(enc_file, "label_encoder.pkl")

    loaded = load_model(path)
    assert loaded["label_encoder"] is not None
    with pytest.warns(UserWarning, match="R016"):
        got = predict_with_model(loaded, X, validate_wavelengths=False)
    np.testing.assert_array_equal(got, model.predict(X))
