"""Tab 7 trains on, saves and predicts with the same wavelength columns (R009/R112, R016).

Drives ``_run_refined_model_thread`` and ``_save_refined_model`` on the session Tk app.
On a grid finer than 0.5 units the refit used to take the first column within 0.5
(the lower neighbour) while ``predict_with_model`` read the named column; and the save
fell back to the search's label encoder for a model trained on raw numeric labels.
"""

from __future__ import annotations

import contextlib
import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from spectral_predict.model_io import load_model, predict_with_model
from spectral_predict.wavelength_matching import format_wavelength_list

pytestmark = pytest.mark.gui


def _spectra(n: int, axis: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, axis.size)
    peaks = rng.normal(size=(n, 2))
    X = 0.5 + peaks[:, [0]] * np.exp(-((x - 0.3) ** 2) / 0.003)
    X = X + peaks[:, [1]] * np.exp(-((x - 0.7) ** 2) / 0.003)
    return X + rng.normal(scale=0.01, size=X.shape)


def _refit(app, X_df: pd.DataFrame, y: pd.Series, row: dict, order: list[float] | None):
    app.X_original = X_df
    app.X = X_df
    app.y = y
    app.active_indices = None
    app.excluded_spectra = set()
    app.validation_enabled.set(False)
    app.validation_indices = []
    app.use_autoscale.set(False)
    app.selected_model_config = dict(row)
    app.loaded_model_config = None
    app._original_wavelength_order = order
    app.refine_task_type.set(row["Task"])
    app.refine_model_type.set(row["Model"])
    app.refine_preprocess.set("raw")
    app.refine_folds.set(3)
    app.refine_cv_strategy.set("kfold")
    app.model_loaded_from_results = True
    app.refine_hyperparams_modified = False
    app.refined_model = None

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        app._run_refined_model_thread()
    app.root.update()
    assert app.refined_model is not None, buf.getvalue()[-4000:]


def _save(app, tmp_path) -> dict:
    path = tmp_path / "refined.dasp"
    errors = []
    with (
        patch(
            "spectral_predict_gui_optimized.filedialog.asksaveasfilename", return_value=str(path)
        ),
        patch(
            "spectral_predict_gui_optimized.messagebox.showerror",
            side_effect=lambda *a, **k: errors.append(a),
        ),
    ):
        app._save_refined_model()
    assert not errors, errors
    return load_model(path)


@pytest.mark.parametrize("spacing", [0.3, 0.482])
@pytest.mark.parametrize("order", ["asc", "desc"])
def test_fine_grid_subset_refit_trains_and_saves_the_named_channels(
    gui_app, tmp_path, spacing, order
):
    axis = 1000.0 + spacing * np.arange(90)
    if order == "desc":
        axis = axis[::-1].copy()
    X_df = pd.DataFrame(_spectra(36, axis), columns=[float(w) for w in axis])
    X_df.index = [f"s{i}" for i in range(len(X_df))]
    sel = [20, 21, 22, 60, 61]
    y = pd.Series(X_df.values[:, sel].sum(axis=1), index=X_df.index)
    row = {
        "Model": "PLS",
        "Task": "regression",
        "Params": "{}",
        "LVs": 2,
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
        "SubsetTag": "top5",
        "all_vars": format_wavelength_list(axis[sel]),
    }

    _refit(gui_app, X_df, y, row, [float(w) for w in axis[sel]])

    np.testing.assert_array_equal(gui_app.refined_X_train, X_df.values[:, sel])
    assert gui_app.refined_wavelengths == [float(w) for w in axis[sel]]

    loaded = _save(gui_app, tmp_path)
    got = predict_with_model(loaded, X_df)
    expected = gui_app.refined_model.predict(gui_app.refined_X_train)
    np.testing.assert_allclose(np.ravel(got), np.ravel(expected), rtol=0, atol=1e-12)


def test_refit_with_unmappable_wavelengths_fails_instead_of_dropping(gui_app):
    axis = 1000.0 + 0.3 * np.arange(60)
    X_df = pd.DataFrame(_spectra(30, axis), columns=[float(w) for w in axis])
    X_df.index = [f"s{i}" for i in range(len(X_df))]
    y = pd.Series(X_df.values[:, 5], index=X_df.index)
    row = {
        "Model": "PLS",
        "Task": "regression",
        "Params": "{}",
        "LVs": 2,
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
        "SubsetTag": "top3",
    }
    gui_app.X_original = X_df
    gui_app.X = X_df
    gui_app.y = y
    gui_app.active_indices = None
    gui_app.excluded_spectra = set()
    gui_app.validation_enabled.set(False)
    gui_app.selected_model_config = row
    gui_app.loaded_model_config = None
    # 1000.45 sits between two channels of a 0.3 grid: the old rule took 1000.3.
    gui_app._original_wavelength_order = [float(axis[2]), 1000.45]
    gui_app.refine_task_type.set("regression")
    gui_app.refine_model_type.set("PLS")
    gui_app.refine_preprocess.set("raw")
    gui_app.refined_model = None
    with contextlib.redirect_stdout(io.StringIO()):
        gui_app._run_refined_model_thread()
    gui_app.root.update()
    assert gui_app.refined_model is None


def test_load_resolves_legacy_g_all_vars_to_exact_axis_values(gui_app):
    axis = 1e7 / np.arange(1000.0, 1100.0)
    X_df = pd.DataFrame(_spectra(12, axis), columns=[float(w) for w in axis])
    X_df.index = [f"s{i}" for i in range(len(X_df))]
    gui_app.X_original = X_df
    gui_app.X = X_df
    gui_app.y = pd.Series(np.arange(12.0), index=X_df.index)
    sel = [4, 9, 50]
    config = {
        "Model": "PLS",
        "Task": "regression",
        "Params": "{}",
        "LVs": 2,
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
        "SubsetTag": "top3",
        "n_vars": 3,
        "full_vars": 100,
        "top_vars": "N/A",
        "all_vars": ",".join(f"{w:g}" for w in axis[sel]),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        gui_app._load_model_for_refinement(config)
    assert gui_app._original_wavelength_order == [float(w) for w in axis[sel]]


def _load_box(gui_app, config: dict) -> str:
    axis = np.array([1500.0, 1501.0, 1502.0, 1503.0] + list(1504.0 + np.arange(16)))
    X_df = pd.DataFrame(_spectra(12, axis), columns=[float(w) for w in axis])
    X_df.index = [f"s{i}" for i in range(len(X_df))]
    gui_app.X_original = X_df
    gui_app.X = X_df
    gui_app.y = pd.Series(np.arange(12.0), index=X_df.index)
    base = {
        "Model": "PLS",
        "Task": "regression",
        "Params": "{}",
        "LVs": 2,
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
        "full_vars": axis.size,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        gui_app._load_model_for_refinement({**base, **config})
    return gui_app.refine_wl_spec.get("1.0", "end")


@pytest.mark.parametrize(
    "config",
    [
        {"SubsetTag": "top2", "n_vars": 2, "all_vars": None, "top_vars": "N/A"},
        {"SubsetTag": "full", "n_vars": 2, "all_vars": "N/A", "top_vars": "N/A"},
        # top_vars holds only part of the trained subset (it keeps the top 30).
        {"SubsetTag": "top3", "n_vars": 3, "all_vars": None, "top_vars": "1500,1501"},
    ],
    ids=["subset_no_lists", "full_tag_but_n_vars_short", "incomplete_top_vars"],
)
def test_load_refuses_to_expand_a_row_without_its_wavelengths(gui_app, config):
    """Codex round 3: Model Development loading turned these into full-spectrum refits."""
    box = _load_box(gui_app, config)
    assert box.lstrip().startswith("# ERROR"), box
    assert gui_app._original_wavelength_order is None


def test_load_accepts_complete_top_vars_and_true_full_rows(gui_app):
    _load_box(
        gui_app, {"SubsetTag": "top2", "n_vars": 2, "all_vars": None, "top_vars": "1502,1500"}
    )
    assert gui_app._original_wavelength_order == [1502.0, 1500.0]
    box = _load_box(
        gui_app, {"SubsetTag": "full", "n_vars": 20, "all_vars": "N/A", "top_vars": "N/A"}
    )
    assert not box.lstrip().startswith("# ERROR"), box
    assert gui_app._original_wavelength_order is None


def _classification_data(labels):
    rng = np.random.default_rng(4)
    axis = np.arange(1000.0, 1040.0)
    y = np.repeat(np.asarray(labels, dtype=object), 12)
    shift = pd.Series(y).map({lab: k for k, lab in enumerate(labels)}).to_numpy(float)
    X = rng.normal(size=(y.size, axis.size)) + shift[:, None]
    X_df = pd.DataFrame(X, columns=[float(w) for w in axis])
    X_df.index = [f"c{i}" for i in range(len(X_df))]
    return X_df, pd.Series(list(y), index=X_df.index).infer_objects()


def _rf_row():
    return {
        "Model": "RandomForest",
        "Task": "classification",
        "Params": "{'n_estimators': 20, 'random_state': 0}",
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
        "SubsetTag": "full",
    }


@pytest.mark.parametrize("labels", [[1, 2, 3], [1.0, 2.0, 3.0]], ids=["int", "float"])
def test_save_ignores_the_search_encoder_for_raw_numeric_labels(gui_app, tmp_path, labels):
    X_df, y = _classification_data(labels)
    gui_app.label_encoder = LabelEncoder().fit(y)  # left by a Bayesian/NSGA search

    _refit(gui_app, X_df, y, _rf_row(), list(X_df.columns))
    assert gui_app.refined_label_encoder is None

    loaded = _save(gui_app, tmp_path)
    assert loaded["label_encoder"] is None
    got = predict_with_model(loaded, X_df)
    np.testing.assert_array_equal(got, gui_app.refined_model.predict(X_df.values))
    assert set(np.unique(got)) <= set(labels)


def test_task_switch_from_text_to_numeric_labels_drops_the_encoder(gui_app, tmp_path):
    X_txt, y_txt = _classification_data(["low", "mid", "high"])
    _refit(gui_app, X_txt, y_txt, _rf_row(), list(X_txt.columns))
    assert gui_app.refined_label_encoder is not None
    loaded = _save(gui_app, tmp_path)
    assert set(predict_with_model(loaded, X_txt)) <= {"low", "mid", "high"}

    X_num, y_num = _classification_data([1, 2, 3])
    _refit(gui_app, X_num, y_num, _rf_row(), list(X_num.columns))
    assert gui_app.refined_label_encoder is None
    loaded = _save(gui_app, tmp_path)
    assert loaded["label_encoder"] is None
    np.testing.assert_array_equal(
        predict_with_model(loaded, X_num), gui_app.refined_model.predict(X_num.values)
    )


def test_ensemble_reconstruction_uses_the_named_columns_and_excludes_bad_rows(gui_app, monkeypatch):
    """Codex round 2: the ensemble rebuild rounded to 1 decimal and took the first
    collision ("1000.020" -> 1000.01), and a missing all_vars meant full spectrum."""
    axis = np.array([1000.0, 1000.01, 1000.02, 1001.0] + list(1002.0 + np.arange(20)))
    X = pd.DataFrame(_spectra(30, axis), columns=list(axis))
    y = X.iloc[:, 2].to_numpy() + X.iloc[:, 3].to_numpy()
    base = {
        "Model": "Ridge",
        "Params": str({"alpha": 1.0}),
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
        "Poly": 2,
        "CompositeScore": 0.0,
    }
    rows = [
        {**base, "all_vars": "1000.020,1001.0", "SubsetTag": "top2", "n_vars": 2},
        {**base, "all_vars": np.nan, "SubsetTag": "top2", "n_vars": 2},
        {**base, "all_vars": "1000.020,9999.0", "SubsetTag": "top2", "n_vars": 2},
    ]
    logs: list[str] = []
    monkeypatch.setattr(gui_app, "_log_progress", logs.append)
    with contextlib.redirect_stdout(io.StringIO()):
        rebuilt = gui_app._reconstruct_models_from_results(pd.DataFrame(rows), X, y, "regression")

    assert len(rebuilt) == 1, logs
    assert rebuilt[0][2]["wavelengths"] == [1000.02, 1001.0]
    assert sum("Failed to reconstruct" in line for line in logs) == 2


def test_tab8_uncertainty_display_aligns_models_with_different_classes(gui_app):
    """Codex round 5: headers came from the first model, so a b/c model's P(b) was
    shown under P(a). Headers are now the union; absent classes are blank."""
    names = ["s0", "s1"]
    gui_app.predictions_df = pd.DataFrame({"Sample": names, "AB": ["a", "b"], "BC": ["c", "b"]})
    gui_app.predictions_uncertainty = {
        "AB": {
            "probabilities": np.array([[0.9, 0.1], [0.2, 0.8]]),
            "confidence": np.array([0.9, 0.8]),
            "class_names": ["a", "b"],
        },
        "BC": {
            "probabilities": np.array([[0.3, 0.7], [0.6, 0.4]]),
            "confidence": np.array([0.7, 0.6]),
            "class_names": ["b", "c"],
        },
    }
    gui_app._display_uncertainty()

    tree = gui_app.uncertainty_tree
    assert list(tree["columns"])[4:] == ["P(a)", "P(b)", "P(c)"]
    rows = [tuple(str(v) for v in tree.item(i, "values")) for i in tree.get_children()]
    by_key = {(r[0], r[1]): r[4:] for r in rows}
    assert by_key[("s0", "AB")] == ("90.00", "10.00", "")
    assert by_key[("s0", "BC")] == ("", "30.00", "70.00")
    assert by_key[("s1", "BC")] == ("", "60.00", "40.00")


def test_tab8_uncertainty_display_with_superset_encoder(gui_app, tmp_path):
    """Codex round 4: class names came from every encoder class (3) while the model
    gave 2 probability columns, so _display_uncertainty raised IndexError."""
    import warnings

    from sklearn.ensemble import RandomForestClassifier

    from spectral_predict.model_io import predict_with_uncertainty, save_model

    rng = np.random.default_rng(6)
    y = np.repeat(np.array(["a", "b"]), 10)
    X = rng.normal(size=(20, 12)) + (y == "b")[:, None]
    encoder = LabelEncoder().fit(["a", "b", "c"])
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, encoder.transform(y))
    metadata = {
        "model_name": "RF",
        "task_type": "classification",
        "wavelengths": [1000.0 + i for i in range(12)],
        "n_vars": 12,
    }
    save_model(model, None, metadata, tmp_path / "rf.dasp", label_encoder=encoder)
    loaded = load_model(tmp_path / "rf.dasp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = predict_with_uncertainty(loaded, X, validate_wavelengths=False)

    names = [f"s{i}" for i in range(len(X))]
    gui_app.predictions_df = pd.DataFrame({"Sample": names, "RF": result["predictions"]})
    gui_app.predictions_uncertainty = {"RF": result["uncertainty"]}
    gui_app._display_uncertainty()

    rows = gui_app.uncertainty_tree.get_children()
    assert len(rows) == len(X)
    assert list(gui_app.uncertainty_tree["columns"])[-2:] == ["P(a)", "P(b)"]
