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

    # --- review round 1 (Codex) -------------------------------------------------------

    @pytest.mark.parametrize(
        "token,axis",
        [("10000", [9999.97]), ("-10000", [-9999.97]), ("1000", [999.996, 1001.0])],
        ids=["below_power_of_ten", "negative", "below_1000"],
    )
    def test_legacy_token_never_matches_a_column_g_prints_differently(self, token, axis):
        """%g prints 9999.97 as '9999.97', so '10000' cannot stand for it."""
        assert all(f"{w:g}" != token for w in axis)
        with pytest.raises(WavelengthMatchError, match="not on the axis"):
            resolve_wavelength_list(token, axis)

    def test_legacy_token_matches_the_column_g_rounds_to_it(self):
        axis = [9999.996, 10001.0]  # %g(9999.996) == '10000'
        assert resolve_wavelength_list("10000", axis).tolist() == [0]

    def test_new_rows_are_never_read_as_legacy(self):
        axis = [10000.1, 10000.12, 10000.2, 10000.22]
        text = format_wavelength_list([10000.1, 10000.2])
        assert text == "10000.10,10000.20"
        assert resolve_wavelength_list(text, axis).tolist() == [0, 2]
        # The same values as old %g text really are ambiguous on this axis.
        with pytest.raises(WavelengthMatchError, match="more than one"):
            resolve_wavelength_list("10000.1", axis)

    def test_new_scientific_notation_tokens_are_exact(self):
        axis = [1e-05, 1.000001e-05, 2e-05]
        text = format_wavelength_list([1e-05, 2e-05])
        assert text == "1.0e-05,2.0e-05"
        assert resolve_wavelength_list(text, axis).tolist() == [0, 2]
        with pytest.raises(WavelengthMatchError):
            resolve_wavelength_list("1e-05", axis)  # legacy %g text: two columns print it

    @pytest.mark.parametrize(
        "token,axis", [("1e+00", [1.0]), ("1e+03", [1000.0]), ("1e+006", [1000000.0])]
    )
    def test_non_g_exponent_spellings_match_exactly(self, token, axis):
        """Codex round 3: these used to be routed to %g matching and rejected."""
        assert resolve_wavelength_list(token, axis).tolist() == [0]

    def test_legacy_token_is_classified_by_its_text_not_by_reformatting(self):
        """Codex round 2: float('1e-318') prints as '9.99999e-319', yet '1e-318' is
        %g text for the subnormal that does print that way."""
        v = float("1e-318")
        x = 1.000004e-318
        assert f"{v:g}" != "1e-318" and f"{x:g}" == "1e-318"
        assert resolve_wavelength_list("1e-318", [v, x]).tolist() == [1]
        with pytest.raises(WavelengthMatchError, match="not on the axis"):
            resolve_wavelength_list("1e-318", [v])

    @pytest.mark.parametrize(
        "token,is_g",
        [
            ("1500", True),
            ("7407.41", True),
            ("1e+07", True),
            ("0.000123457", True),
            ("1500.0", False),
            ("3999.6419", False),
            ("1000000", False),  # %g writes 1e+06
            ("0.0000123", False),  # %g writes 1.23e-05
            ("10000.10", False),
            # Exponent spellings: only what %g itself writes (review round 3).
            ("1e+06", True),
            ("1.5e-05", True),
            ("1e-318", True),
            ("1e+100", True),
            ("1e+00", False),  # %g writes '1'
            ("1e+03", False),  # %g writes '1000'
            ("1e+006", False),  # %g pads to two digits
            ("1.5e+05", False),  # %g writes '150000'
            ("1E+06", False),
            ("+1500", False),
        ],
    )
    def test_g_token_syntax(self, token, is_g):
        from spectral_predict.wavelength_matching import _looks_like_g_token

        assert _looks_like_g_token(token) is is_g

    @pytest.mark.parametrize(
        "values", [[1500.0, 1500.5], [7407.407407407408, 10000.1, -3.25, 1e-05, 1e7]]
    )
    def test_every_new_token_round_trips_and_is_not_g_text(self, values):
        tokens = format_wavelength_list(values).split(",")
        assert [float(t) for t in tokens] == values
        assert all(f"{float(t):g}" != t for t in tokens)


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

    def test_legacy_0p012_grid_with_g_metadata_warns_and_predicts_named_channels(self, tmp_path):
        """GLM H-1: on a 0.012 cm-1 grid ~45% of %g-rounded wavelengths have both
        neighbours inside +/-0.01. The legacy path must warn on load and map them by
        %g text, not raise 'more than one axis column' at predict."""
        axis = 4000.0 + 0.012 * np.arange(120)
        X = _spectra(30, axis)
        sel = list(range(20, 100, 7))
        model, prep, metadata, X_train = _fit_like_tab7(X, axis, axis[sel])
        metadata["wavelengths"] = [float(f"{w:g}") for w in metadata["wavelengths"]]
        with pytest.raises(WavelengthMatchError):  # the fixed 0.01 rule alone fails
            match_wavelengths(metadata["wavelengths"], axis)
        path = tmp_path / "old_0p012.dasp"
        save_model(model, prep, metadata, path)
        _strip_matching_stamp(path)

        with pytest.warns(UserWarning, match="Retrain"):
            loaded = load_model(path)
        got = predict_with_model(loaded, pd.DataFrame(X, columns=axis))
        np.testing.assert_allclose(
            np.ravel(got), np.ravel(model.predict(X_train)), rtol=0, atol=1e-12
        )

    def test_legacy_grid_too_fine_for_g_text_warns_and_refuses_to_predict(self, tmp_path):
        axis = 4000.0 + 0.004 * np.arange(120)  # several channels share each %g text
        X = _spectra(30, axis)
        model, prep, metadata, _ = _fit_like_tab7(X, axis, axis[[30, 31, 60]])
        metadata["wavelengths"] = [float(f"{w:g}") for w in metadata["wavelengths"]]
        path = tmp_path / "old_0p004.dasp"
        save_model(model, prep, metadata, path)
        _strip_matching_stamp(path)

        with pytest.warns(UserWarning, match="do not each name a single channel"):
            loaded = load_model(path)
        with pytest.raises(WavelengthMatchError, match="Retrain"):
            predict_with_model(loaded, pd.DataFrame(X, columns=axis))

    def test_exact_ensemble_on_a_grid_g_cannot_tell_apart_predicts(self, tmp_path):
        """Codex round 2: every value of [10000.0, 10000.02, ...] prints as %g
        "10000". Legacy %g matching must apply only to pre-fix Tab 7 models, and
        new ensembles carry the stamp."""
        from spectral_predict.ensemble import SimpleAverageEnsemble
        from spectral_predict.model_io import load_ensemble, save_ensemble

        axis = 10000.0 + 0.02 * np.arange(40)
        X = _spectra(30, axis)
        y = X[:, 3] - X[:, 20]
        models = [PLSRegression(n_components=k).fit(X, y) for k in (2, 3)]
        ensemble = SimpleAverageEnsemble(models, model_names=["PLS2", "PLS3"])
        metadata = {
            "ensemble_type": "simple_average",
            "ensemble_name": "avg",
            "task_type": "regression",
            "wavelengths": axis.tolist(),
            "full_wavelengths": axis.tolist(),
            "use_full_spectrum_preprocessing": True,
            "n_vars": axis.size,
            "preprocessing": "raw",
        }
        path = tmp_path / "ens.dasp"
        save_ensemble(ensemble, str(path), metadata)
        loaded = load_ensemble(str(path))
        assert loaded["metadata"]["wavelength_matching"] == 1

        expected = np.mean([np.ravel(m.predict(X)) for m in models], axis=0)
        for meta in (loaded["metadata"], {**loaded["metadata"], "wavelength_matching": None}):
            meta = {k: v for k, v in meta.items() if v is not None}
            model_dict = {"model": loaded["ensemble"], "preprocessor": None, "metadata": meta}
            got = predict_with_model(model_dict, pd.DataFrame(X, columns=axis))
            np.testing.assert_allclose(np.ravel(got), expected, rtol=1e-12)

    def test_ensemble_member_on_fine_grid_is_not_flagged(self, tmp_path):
        """Ensemble members store the whole exact axis; they never went through the
        Tab 7 first-hit rule, so replaying it would be a false 'retrain' warning."""
        axis = _axis(0.3, "asc")
        X = _spectra(30, axis)
        model, prep, metadata, _ = _fit_like_tab7(X, axis, axis)
        metadata["ensemble_parent"] = True
        path = tmp_path / "member.dasp"
        save_model(model, prep, metadata, path)
        _strip_matching_stamp(path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert load_model(path)["wavelength_mapping_warning"] is None


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


@pytest.mark.parametrize("all_vars", [None, np.nan, "", "N/A"], ids=["None", "NaN", "empty", "NA"])
def test_subset_row_without_all_vars_is_not_validated_on_the_full_spectrum(
    wavenumber_split, all_vars
):
    """Codex/GLM round 1: a top-N row with no usable all_vars used to be refit on
    every column and scored like the full-spectrum model."""
    row = dict(_regression_row(all_vars), n_vars=1, SubsetTag="top1")
    out = _validate([row], wavenumber_split)
    assert np.isnan(out.loc[0, "RMSEP"])
    assert "wavelength-subset row" in out.attrs["validation_failures"][0]


def test_full_row_without_all_vars_validates_only_if_n_vars_covers_the_axis(wavenumber_split):
    axis = wavenumber_split[0]
    rows = [
        dict(_regression_row("N/A", subset_tag="full"), n_vars=axis.size),
        dict(_regression_row("N/A", subset_tag="full"), n_vars=axis.size - 10),
        dict(_regression_row("N/A", subset_tag="full")),  # no n_vars at all
    ]
    out = _validate(rows, wavenumber_split)
    assert np.isfinite(out.loc[0, "RMSEP"])
    assert set(out.attrs["validation_failures"]) == {1, 2}
    assert out.attrs["validation_attempted"] == [0, 1, 2]
    assert out.attrs["validation_succeeded"] == [0]


def test_failed_multiclass_holdout_row_keeps_no_stale_metrics():
    """GLM round 3: a failing multi-class row must not keep an earlier run's val_*.
    (The helper re-initialises these columns for every row on entry.)"""
    rng = np.random.default_rng(2)
    X_tr, X_va = rng.normal(size=(40, 30)), rng.normal(size=(12, 30))
    y_tr = np.array(["a", "b"] * 20, dtype=object)
    y_va = np.array(["a", "b"] * 6, dtype=object)
    row = {
        "CompositeScore": 0.0,
        "engine_family": "no-such-engine",
        "val_MeanSensitivity": 0.99,
        "val_ExactSetRate": 0.99,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compute_validation_metrics_for_top_models(
            pd.DataFrame([row]),
            X_tr,
            y_tr,
            X_va,
            y_va,
            "multiclass_simca",
            np.arange(1000.0, 1030.0),
            top_n=1,
        )
    assert np.isnan(out.loc[0, "val_MeanSensitivity"])
    assert np.isnan(out.loc[0, "val_ExactSetRate"])
    assert 0 in out.attrs["validation_failures"]
    assert out.attrs["validation_succeeded"] == []


@pytest.mark.parametrize(
    "tags",
    [
        {"SubsetTag": None, "Subset": "top10"},  # null SubsetTag must not hide Subset
        {"SubsetTag": None},
        {"SubsetTag": ""},
        {"SubsetTag": "N/A"},
        {},
    ],
    ids=["null_tag_subset_top10", "null_tag", "empty_tag", "NA_tag", "no_tag"],
)
def test_full_spectrum_fallback_needs_an_affirmative_full_tag(wavenumber_split, tags):
    axis = wavenumber_split[0]
    row = {k: v for k, v in _regression_row("N/A").items() if k != "SubsetTag"}
    row.update(tags, n_vars=axis.size)
    out = _validate([row], wavenumber_split)
    assert np.isnan(out.loc[0, "RMSEP"])
    assert 0 in out.attrs["validation_failures"]


class TestEnsembleWavelengthPaths:
    """Ensemble rebuild paths use the shared resolver (Codex round 2, items 6-7)."""

    def test_preprocessor_config_maps_exact_columns_and_raises_on_misses(self):
        from spectral_predict.preprocessing_wrapper import PreprocessorConfig

        axis = [1000.0, 1000.01, 1000.02, 1001.0]
        cfg = PreprocessorConfig("raw", wavelengths=[1000.02, 1000.0], all_wavelengths=axis)
        assert cfg.wavelength_indices_.tolist() == [2, 0]
        with pytest.raises(WavelengthMatchError):
            PreprocessorConfig("raw", wavelengths=[1000.02, 1003.0], all_wavelengths=axis)

    def test_extract_preprocessor_config_resolves_text_and_refuses_silent_full(self):
        from spectral_predict.ensemble import extract_preprocessor_config

        axis = [1000.0, 1000.01, 1000.02, 1001.0]
        cfg = extract_preprocessor_config({"all_vars": "1000.020,1001.0"}, axis)
        assert cfg.wavelengths == [1000.02, 1001.0]
        with pytest.raises(ValueError):
            extract_preprocessor_config({"all_vars": "1000.020,9999.0"}, axis)
        with pytest.raises(ValueError, match="subset"):
            extract_preprocessor_config({"all_vars": "N/A", "SubsetTag": "top2"}, axis)
        full = extract_preprocessor_config(
            {"all_vars": "N/A", "SubsetTag": "full", "n_vars": 4}, axis
        )
        assert full.wavelengths is None


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


def _legacy_with_encoder(tmp_path, model, encoder, n_vars):
    """A pre-ownership-stamp .dasp: encoder pickle present, no label_encoder_owned."""
    path = tmp_path / "legacy_enc.dasp"
    save_model(model, None, _clf_metadata(n_vars), path)
    enc_file = tmp_path / "label_encoder.pkl"
    joblib.dump(encoder, enc_file)
    with zipfile.ZipFile(path, "a") as zf:
        zf.write(enc_file, "label_encoder.pkl")
    loaded = load_model(path)
    assert loaded["label_encoder"] is not None
    assert "label_encoder_owned" not in loaded["metadata"]
    return loaded


@pytest.mark.parametrize(
    "labels,stale",
    [([1, 2, 3], ["a", "b", "c", "d"]), ([0, 1], ["x", "y", "z"])],
    ids=["codex_123_vs_abcd", "glm_01_vs_xyz"],
)
def test_legacy_stale_encoder_with_valid_looking_codes_is_not_used(tmp_path, labels, stale):
    """Review round 3: raw labels that happen to be valid codes for a bigger stale
    encoder used to decode silently (1,2,3 -> b,c,d)."""
    X, y = _clf_data(labels)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    loaded = _legacy_with_encoder(tmp_path, model, LabelEncoder().fit(stale), X.shape[1])
    with pytest.warns(UserWarning, match="cannot be shown to belong"):
        got = predict_with_model(loaded, X, validate_wavelengths=False)
    np.testing.assert_array_equal(got, model.predict(X))


def test_legacy_superset_encoder_cannot_be_proven_and_is_not_used(tmp_path):
    X, y = _clf_data(["a", "b"])
    encoder = LabelEncoder().fit(["a", "b", "c"])
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, encoder.transform(y))
    loaded = _legacy_with_encoder(tmp_path, model, encoder, X.shape[1])
    with pytest.warns(UserWarning, match="cannot be shown to belong"):
        got = predict_with_model(loaded, X, validate_wavelengths=False)
    np.testing.assert_array_equal(got, model.predict(X))


def test_legacy_encoder_that_provably_fits_still_decodes(tmp_path):
    X, y = _clf_data(["a", "b", "c"])
    encoder = LabelEncoder().fit(y)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, encoder.transform(y))
    loaded = _legacy_with_encoder(tmp_path, model, encoder, X.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = predict_with_model(loaded, X, validate_wavelengths=False)
    np.testing.assert_array_equal(got, encoder.inverse_transform(model.predict(X)))


def test_bool_label_model_is_never_decoded_through_a_text_encoder(tmp_path):
    X, y = _clf_data([False, True])
    y = y.astype(bool)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    stale = LabelEncoder().fit(["neg", "pos"])
    with pytest.warns(UserWarning, match="does not match the model"):
        saved = _save_load(tmp_path, model, stale, X.shape[1])
    assert saved["label_encoder"] is None
    legacy = _legacy_with_encoder(tmp_path, model, stale, X.shape[1])
    with pytest.warns(UserWarning):
        got = predict_with_model(legacy, X, validate_wavelengths=False)
    np.testing.assert_array_equal(got, model.predict(X))


def test_new_files_record_encoder_ownership(tmp_path):
    X, y = _clf_data(["a", "b", "c"])
    encoder = LabelEncoder().fit(y)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, encoder.transform(y))
    loaded = _save_load(tmp_path, model, encoder, X.shape[1])
    assert loaded["metadata"]["label_encoder_owned"] is True


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


def test_encoder_knowing_more_classes_than_the_model_saw_is_kept(tmp_path):
    """Codex round 2: an encoder fit on a, b, c and a model trained on its codes for
    a and b only is a valid pair; decoding must still give a/b, not 0/1."""
    X, y = _clf_data(["a", "b"])
    encoder = LabelEncoder().fit(["a", "b", "c"])
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, encoder.transform(y))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = _save_load(tmp_path, model, encoder, X.shape[1])
        got = predict_with_model(loaded, X, validate_wavelengths=False)
    assert loaded["label_encoder"] is not None
    assert set(got) <= {"a", "b"}
    np.testing.assert_array_equal(got, encoder.inverse_transform(model.predict(X)))


# --------------------------------------------------------------------------------------
# GUI validation summary counts only this run's successes (review round 1, item 5)
# --------------------------------------------------------------------------------------


def test_validation_summary_ignores_stale_metrics_on_failed_rows():
    from spectral_predict_gui_optimized import _validation_summary_lines

    df = pd.DataFrame({"Model": ["PLS", "PLS", "PLS"], "R2pred": [0.9, 0.8, 0.7]})
    df.attrs["validation_failures"] = {1: "all_vars does not match the spectral axis"}
    df.attrs["validation_attempted"] = [0, 1]
    lines = _validation_summary_lines(df, top_n=2, metric_col="R2pred")
    assert "only 1 of the top 2" in lines[0]
    assert "row 1" in lines[1]

    df.attrs["validation_failures"] = {}
    assert "[OK]" in _validation_summary_lines(df, top_n=2, metric_col="R2pred")[0]


def test_validation_summary_trusts_recorded_successes_over_a_nan_metric():
    """Codex round 2: one-class val_BalancedAcc is NaN by design on an inlier-only
    validation set; a completed row is still a success."""
    from spectral_predict_gui_optimized import _validation_summary_lines

    df = pd.DataFrame({"Model": ["IsolationForest"] * 2, "val_BalancedAcc": [np.nan, np.nan]})
    df.attrs.update(
        validation_failures={}, validation_attempted=[0, 1], validation_succeeded=[0, 1]
    )
    assert "[OK]" in _validation_summary_lines(df, 2, "val_BalancedAcc")[0]
    df.attrs["validation_succeeded"] = [0]
    assert "only 1 of the top 2" in _validation_summary_lines(df, 2, "val_BalancedAcc")[0]


def test_rerank_moves_validation_attrs_with_their_rows():
    """Codex round 2: compute_composite_score resets the index after sorting; the
    failure reason of a PLS row ended up on a Ridge row."""
    from spectral_predict.scoring import compute_composite_score

    df = pd.DataFrame(
        {
            "Model": ["PLS", "Ridge"],
            "RMSE": [2.0, 1.0],
            "R2": [0.5, 0.9],
            "RMSEcv": [2.0, 1.0],
            "R2cv": [0.5, 0.9],
            "n_vars": [10, 10],
            "full_vars": [10, 10],
            "LVs": [2, np.nan],
            "SubsetTag": ["full", "full"],
            "top_vars": ["N/A", "N/A"],
        }
    )
    df.attrs.update(
        validation_failures={0: "PLS failed"},
        validation_attempted=[0, 1],
        validation_succeeded=[1],
    )
    out = compute_composite_score(df, "regression")
    pls_pos = int(np.flatnonzero(out["Model"].to_numpy() == "PLS")[0])
    ridge_pos = int(np.flatnonzero(out["Model"].to_numpy() == "Ridge")[0])
    assert pls_pos != 0  # the re-rank really moved the rows
    assert out.attrs["validation_failures"] == {pls_pos: "PLS failed"}
    assert out.attrs["validation_succeeded"] == [ridge_pos]
    assert sorted(out.attrs["validation_attempted"]) == [0, 1]
