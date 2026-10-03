"""Dataset state follows the loaded data (adversarial review R004-R007, R037, R038).

- R004/R037: every loader (Import, Data Management Use/Merge/filter/trim,
  calibration-transfer replace) installs data through ``_install_dataset``.
  Replacing the dataset clears the old exclusions, validation split and Quality
  Check report; appending keeps them.
- R005: the validation spectra a run scores are taken from the run's own data
  (current wavelengths, current spectra, minus exclusions), never from a
  snapshot frozen when the split was made.
- R006: clicking a spectrum excludes its real label, also for numeric-string IDs.
- R007: Quality Check marks the real sample label, not the displayed row number.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import spectral_predict_gui_optimized as gui_module
from spectral_predict.data_management import DataSource, DataSourceManager
from spectral_predict.run_state import LABEL_NORMALIZATION

from tests.gui.test_resume_data_mismatch_keeps_run import _FakeThread  # noqa: F401
from tests.gui.test_resume_round9 import (  # noqa: F401 -- fixtures
    _recording_bayesian,
    _select_models,
    fake_thread,
)
from tests.gui.test_resume_run_completion import worker_env  # noqa: F401 -- fixture

N_WL = 120


def _spectra(ids, seed: int, offset: float = 0.0, step: float = 2.0):
    """Spectra whose values encode the target, so a mispairing is visible."""
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.uniform(0, 10, len(ids)) + offset, index=pd.Index(ids))
    X = pd.DataFrame(
        np.outer(y.to_numpy(), np.linspace(0.5, 1.5, N_WL)) + rng.normal(0, 0.2, (len(ids), N_WL)),
        index=y.index,
        columns=[f"{1000 + step * i:g}" for i in range(N_WL)],
    )
    return X, y


def _write_combined(path, ids, seed, offset=0.0, step=2.0, extra=None):
    X, y = _spectra(ids, seed, offset, step)
    df = X.copy()
    df.insert(0, "protein", y.to_numpy())
    for name, values in (extra or {}).items():
        df.insert(0, name, values)
    df.insert(0, "sample_id", list(ids))
    df.to_csv(path, index=False)
    return X, y


_STATE_ATTRS = (
    "X",
    "X_original",
    "y",
    "ref",
    "combined_metadata_df",
    "excluded_spectra",
    "validation_indices",
    "validation_X",
    "validation_y",
    "outlier_report",
    "active_group_filter",
    "active_indices",
    "_pending_validation_indices",
    "_pending_analysis_settings",
    "data_sources",
    "source_group_names",
    "use_custom_group_names",
    "_exact_wavelength_axis",
    "_last_run_validation",
    "training_data_cache",
    "prediction_data",
    "prediction_actuals",
    "data_value_scale",
    "combined_file_path",
)
_STATE_VARS = (
    "wavelength_min",
    "wavelength_max",
    "validation_enabled",
    "show_validation_metrics",
    "append_mode",
    "combined_data_file",
    "spectral_data_path",
    "target_column",
    "current_data_type",
    "original_data_type",
    "current_x_unit",
    "original_x_unit",
    "use_absorbance",
    "pred_data_source",
)


@pytest.fixture
def clean_state(gui_app):
    """Start from an empty dataset and put the session app back afterwards."""
    saved = {a: getattr(gui_app, a, None) for a in _STATE_ATTRS}
    saved_vars = {v: getattr(gui_app, v).get() for v in _STATE_VARS}
    saved_detected = getattr(gui_app, "detected_type", None)
    gui_app.excluded_spectra = set()
    gui_app.validation_indices = set()
    gui_app.validation_X = gui_app.validation_y = None
    gui_app.outlier_report = None
    gui_app.active_group_filter = None
    gui_app.active_indices = None
    gui_app._pending_validation_indices = None
    gui_app._pending_analysis_settings = None
    gui_app.combined_metadata_df = None
    gui_app.ref = None
    gui_app._exact_wavelength_axis = False
    gui_app.wavelength_min.set("")
    gui_app.wavelength_max.set("")
    gui_app.validation_enabled.set(False)
    gui_app.show_validation_metrics.set(True)
    gui_app.append_mode.set(False)
    yield gui_app
    for a, v in saved.items():
        setattr(gui_app, a, v)
    for v, value in saved_vars.items():
        getattr(gui_app, v).set(value)
    gui_app.detected_type = saved_detected


def _load_combined(app, path, append=False):
    app.combined_data_file.set(str(path))
    app.spectral_data_path.set("")
    app.detected_type = "combined"
    app.target_column.set("protein")
    app.append_mode.set(append)
    app._load_and_plot_data()


def _split(app, labels):
    app.validation_indices = set(labels)
    app.validation_enabled.set(True)
    app._refresh_validation_snapshot()


def _recording_validation(monkeypatch):
    """Record what the worker passes to compute_validation_metrics_for_top_models."""
    seen: list[dict] = []

    def fake(
        df_results,
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        task_type=None,
        wavelengths=None,
        *args,
        **kwargs,
    ):
        seen.append(
            {
                "X_train": np.array(X_train, copy=True),
                "X_val": np.array(X_val, copy=True),
                "y_val": np.array(y_val, copy=True),
                "wavelengths": np.array(wavelengths, copy=True),
            }
        )
        return df_results

    monkeypatch.setattr("spectral_predict.search.compute_validation_metrics_for_top_models", fake)
    return seen


def _run(app, monkeypatch):
    """Click Run Analysis (Bayesian, PLS) and run the worker inline."""
    _select_models(app, ["PLS"])
    fake, calls = _recording_bayesian()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)
    seen = _recording_validation(monkeypatch)
    with patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
        app._run_analysis()
        worker = _FakeThread.created[-1]
        worker.target(*worker.args, **worker.kwargs)
    return calls, seen


# ---------------------------------------------------------------------------
# R004: replacing the dataset clears the old one's exclusions and split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ids_b",
    [[str(i) for i in range(1, 31)], [f"B{i}" for i in range(1, 31)]],
    ids=["overlapping-ids", "disjoint-ids"],
)
def test_import_replace_clears_exclusions_and_split(
    clean_state, worker_env, fake_thread, tmp_path, monkeypatch, ids_b
):
    app = clean_state
    ids_a = [str(i) for i in range(1, 31)]
    _write_combined(tmp_path / "a.csv", ids_a, seed=1)
    _load_combined(app, tmp_path / "a.csv")
    a_index = list(app.X.index)
    _split(app, a_index[:6])
    app.excluded_spectra = {a_index[10], a_index[11]}

    _, y_b = _write_combined(tmp_path / "b.csv", ids_b, seed=2, offset=100.0)
    _load_combined(app, tmp_path / "b.csv")

    assert app.excluded_spectra == set()
    assert app.validation_indices == set()
    assert app.validation_X is None and app.validation_y is None
    assert app.validation_enabled.get() is False
    np.testing.assert_allclose(app.y.to_numpy(), y_b.to_numpy())

    calls, seen = _run(app, monkeypatch)
    assert calls[0][2] == 30, "every row of B trains; none removed by A's labels"
    assert seen == [], "no validation metrics from A's holdout"


def test_import_append_keeps_exclusions_and_split(clean_state, tmp_path, monkeypatch):
    app = clean_state
    _write_combined(tmp_path / "a.csv", [f"A{i}" for i in range(1, 31)], seed=1)
    _load_combined(app, tmp_path / "a.csv")
    a_index = list(app.X.index)
    _split(app, a_index[:6])
    app.excluded_spectra = {a_index[10]}

    monkeypatch.setattr(app, "_prompt_for_group_names", lambda: ("A", "B"))
    _write_combined(tmp_path / "b.csv", [f"B{i}" for i in range(1, 11)], seed=2)
    _load_combined(app, tmp_path / "b.csv", append=True)

    assert len(app.X) == 40
    assert app.excluded_spectra == {a_index[10]}
    assert app.validation_indices == set(a_index[:6])
    pd.testing.assert_frame_equal(app.validation_X, app.X.loc[a_index[:6]])


def test_sub_integer_load_keeps_previous_dataset_whole(clean_state, tmp_path):
    """R038: a rejected load leaves the old X, y and metadata together."""
    app = clean_state
    _write_combined(tmp_path / "a.csv", [f"A{i}" for i in range(1, 31)], seed=1)
    _load_combined(app, tmp_path / "a.csv")
    X_a, X_orig_a, y_a = app.X, app.X_original, app.y
    _split(app, list(app.X.index[:6]))

    _write_combined(
        tmp_path / "b.csv", [f"A{i}" for i in range(1, 31)], seed=2, offset=100.0, step=0.3
    )
    with patch("tkinter.messagebox.showerror") as err:
        _load_combined(app, tmp_path / "b.csv")

    assert err.called
    assert app.X is X_a and app.X_original is X_orig_a and app.y is y_a
    assert app.validation_indices == set(X_a.index[:6])


def test_replace_during_pending_resume_keeps_validation_checkbox(clean_state, tmp_path):
    """A resumed run's split is restored by the launch gate, not carried by the load."""
    app = clean_state
    app._pending_validation_indices = ["A1", "A2"]
    app.validation_enabled.set(True)
    _write_combined(tmp_path / "a.csv", [f"A{i}" for i in range(1, 31)], seed=1)
    _load_combined(app, tmp_path / "a.csv")

    assert app.validation_enabled.get() is True
    assert app._pending_validation_indices == ["A1", "A2"]
    assert app.validation_indices == set()


# ---------------------------------------------------------------------------
# R037: Data Management installs coherently
# ---------------------------------------------------------------------------


def test_data_management_use_then_wavelength_update_keeps_b(clean_state, monkeypatch):
    app = clean_state
    X_a, y_a = _spectra([f"Sample_{i}" for i in range(1, 31)], seed=1)
    meta_a = pd.DataFrame({"protein": y_a, "site": "north"}, index=X_a.index)
    assert app._install_dataset(X_a, y_a, None, meta_a, replot=False)
    app.excluded_spectra = {"Sample_3"}
    _split(app, ["Sample_1", "Sample_2", "Sample_4"])

    # B collides with A's generated IDs and has a different target.
    X_b, y_b = _spectra([f"Sample_{i}" for i in range(1, 31)], seed=2, offset=100.0)
    ref_b = pd.DataFrame({"moisture": y_b * 2}, index=X_b.index)
    source = DataSource(
        source_id="b", name="B", path="b.csv", format_type="csv", X=X_b, y=y_b, ref=ref_b
    )
    monkeypatch.setattr(
        app,
        "data_source_manager",
        SimpleNamespace(get_source=lambda sid: source, merged_dataset=None),
    )
    monkeypatch.setattr(
        app,
        "data_sources_tree",
        SimpleNamespace(selection=lambda: ("item",), item=lambda item: {"text": "b"}),
    )
    monkeypatch.setattr(app, "_update_ensemble_controls_state", lambda: None)
    app._use_for_analysis()

    np.testing.assert_allclose(app.X.to_numpy(), X_b.to_numpy())
    assert app.combined_metadata_df is None
    assert app.excluded_spectra == set() and app.validation_indices == set()

    app.wavelength_min.set("1020")
    app.wavelength_max.set("1100")
    app._update_wavelengths()
    wl = X_b.columns.astype(float)
    expected = X_b.loc[:, (wl >= 1020) & (wl <= 1100)]  # Data Management keeps its axis
    pd.testing.assert_frame_equal(app.X, expected)
    np.testing.assert_allclose(app.y.to_numpy(), y_b.to_numpy())

    # Switching target reads B's metadata, not A's leftover combined metadata.
    app.target_column.set("moisture")
    app._on_target_column_changed()
    np.testing.assert_allclose(app.y.to_numpy(), (y_b * 2).to_numpy())


def test_data_management_filter_and_trim_follow_x_original(clean_state, monkeypatch):
    app = clean_state
    X, y = _spectra([f"S{i}" for i in range(1, 21)], seed=4)
    assert app._install_dataset(X, y, None, None, replot=False)
    _split(app, ["S1", "S2", "S3"])
    app.excluded_spectra = {"S4"}
    monkeypatch.setattr(app, "data_source_manager", DataSourceManager())

    app.filter_type_var.set("list")
    monkeypatch.setattr(app.filter_value_var, "get", lambda: ["S1", "S4", "S5", "S6"])
    app.filter_column_var.set("")
    app._apply_sample_filter()
    assert list(app.X_original.index) == ["S1", "S4", "S5", "S6"]
    assert app.excluded_spectra == {"S4"}
    assert app.validation_indices == {"S1"}
    assert list(app.validation_X.index) == ["S1"]

    monkeypatch.setattr(app.min_wavelength_var, "get", lambda: 1100.0)
    monkeypatch.setattr(app.max_wavelength_var, "get", lambda: 1200.0)
    app._trim_wavelengths()
    assert app.X_original.columns.min() >= 1100 and app.X_original.columns.max() <= 1200
    assert list(app.validation_X.columns) == list(app.X.columns)
    app._update_wavelengths()  # a later Update keeps the trim
    assert app.X.columns.max() <= 1200


# ---------------------------------------------------------------------------
# R005: validation spectra come from the run's own data
# ---------------------------------------------------------------------------


def _install_regression(app, n=30, seed=3):
    X, y = _spectra([f"S{i}" for i in range(1, n + 1)], seed=seed)
    assert app._install_dataset(X, y, None, None, replot=False)
    return X, y


def test_wavelength_narrowing_after_split_scores_current_wavelengths(
    clean_state, worker_env, fake_thread, monkeypatch
):
    app = clean_state
    _install_regression(app)
    _split(app, ["S1", "S2", "S3", "S4", "S5"])
    app.wavelength_min.set("1100")
    app.wavelength_max.set("1200")
    app._update_wavelengths()
    assert list(app.validation_X.columns) == list(app.X.columns)

    calls, seen = _run(app, monkeypatch)
    assert calls[0][2] == 25
    assert seen[0]["X_val"].shape == (5, app.X.shape[1])
    np.testing.assert_allclose(seen[0]["X_val"], app.X.loc[["S1", "S2", "S3", "S4", "S5"]])
    assert seen[0]["X_train"].shape[1] == seen[0]["X_val"].shape[1]


def test_equal_width_axis_change_is_caught():
    """Same width, different wavelengths: identity check, not a width check."""
    with pytest.raises(ValueError):
        gui_module._check_validation_axis([1000, 1002, 1004], [1100, 1102, 1104])
    gui_module._check_validation_axis([1000, 1002], [1000, 1002])


def test_baseline_replace_after_split_corrects_validation_too(
    clean_state, worker_env, fake_thread, monkeypatch
):
    app = clean_state
    _install_regression(app)
    _split(app, ["S1", "S2", "S3"])
    monkeypatch.setattr(app, "_compute_corrected_spectra", lambda method: app.X.to_numpy() - 5.0)
    monkeypatch.setattr(app, "_generate_explore_plots", lambda: None)
    app._replace_working_data("als")
    pd.testing.assert_frame_equal(app.validation_X, app.X.loc[["S1", "S2", "S3"]])

    _, seen = _run(app, monkeypatch)
    np.testing.assert_allclose(seen[0]["X_val"], app.X.loc[["S1", "S2", "S3"]])


def test_exclusion_after_split_is_not_scored(clean_state, worker_env, fake_thread, monkeypatch):
    app = clean_state
    _, y = _install_regression(app)
    _split(app, ["S1", "S2", "S3", "S4"])
    app.excluded_spectra = {"S2"}

    _, seen = _run(app, monkeypatch)
    np.testing.assert_allclose(seen[0]["y_val"], y.loc[["S1", "S3", "S4"]].to_numpy())


def test_backend_rejects_train_val_width_mismatch():
    from spectral_predict.contamination import compute_validation_metrics_for_top_one_class_models
    from spectral_predict.search import compute_validation_metrics_for_top_models

    df = pd.DataFrame([{"Model": "PLS", "Params": "{}", "CompositeScore": 1.0}])
    X_tr, X_val = np.zeros((10, 20)), np.zeros((4, 30))
    with pytest.raises(ValueError, match="same wavelength axis"):
        compute_validation_metrics_for_top_models(
            df, X_tr, np.zeros(10), X_val, np.zeros(4), "regression", np.arange(20)
        )
    with pytest.raises(ValueError, match="same wavelength axis"):
        compute_validation_metrics_for_top_one_class_models(
            df, X_tr, np.array(["a"] * 10), X_val, np.array(["a"] * 4), "a", np.arange(20)
        )


# ---------------------------------------------------------------------------
# R006: spectrum-click exclusion keeps the real label
# ---------------------------------------------------------------------------

_LABEL_CASES = {
    "numeric-strings": [str(i) for i in range(1, 9)],
    "leading-zeros": ["007", "07", "7", "0", "00", "8", "9", "10"],
    "negative-looking": ["-3", "3", "-1", "1", "2", "-2", "4", "5"],
    "gapped-integers": [10, 3, 42, 7, 0, 1, 99, 5],
}


@pytest.fixture(params=list(_LABEL_CASES), ids=list(_LABEL_CASES))
def labelled(clean_state, request):
    app = clean_state
    X, y = _spectra(_LABEL_CASES[request.param], seed=5)
    assert app._install_dataset(X, y, None, None, replot=False)
    return app, X, y


def _line_for(app, label):
    fig = Figure()
    ax = fig.add_subplot(111)
    pos = app.X.index.get_loc(label)
    (line,) = ax.plot(app.X.columns, app.X.iloc[pos])
    app._tag_sample_artist(line, label, pos)
    return line, ax


@pytest.mark.parametrize("click", [1, 3])
def test_import_plot_click_excludes_real_label(labelled, monkeypatch, click):
    app, X, y = labelled
    shown = []
    monkeypatch.setattr(
        app, "_create_or_update_annotation", lambda ax, x, yy, text, canvas: shown.append(text)
    )
    target = X.index[3]  # a label that also looks like (or is) a row number
    line, _ = _line_for(app, target)
    event = SimpleNamespace(
        artist=line,
        canvas=SimpleNamespace(draw=lambda: None),
        mouseevent=SimpleNamespace(button=click),
    )
    app._on_spectrum_click(event)

    assert app.excluded_spectra == {target}
    assert app.X.index.isin(app.excluded_spectra).sum() == 1
    assert f"Specimen: {target}" in shown[0]
    assert f"{y.loc[target]:.4f}" in shown[0]

    app._on_spectrum_click(event)
    assert app.excluded_spectra == set()


def test_gid_only_artist_is_matched_by_label_text(labelled):
    app, X, _ = labelled
    target = X.index[2]
    line, _ = _line_for(app, target)
    del line._dasp_sample_label, line._dasp_sample_pos
    assert app._sample_from_artist(line) == (target, 2)


def test_explore_click_excludes_real_label(labelled, monkeypatch):
    app, X, y = labelled
    target = X.index[5]
    line, ax = _line_for(app, target)
    frame = object()
    stub = SimpleNamespace(config=lambda **k: None, draw_idle=lambda: None)
    state = {
        "info_label": stub,
        "toggle_btn": stub,
        "ax": ax,
        "canvas": stub,
        "frame": frame,
        "excl_count_label": stub,
        "restore_all_btn": None,
        "alpha": 0.3,
        "selected_sample": None,
    }
    monkeypatch.setattr(app, "_explore_plot_state", {id(frame): state})
    monkeypatch.setattr(app, "_create_or_update_annotation", lambda *a, **k: None)
    monkeypatch.setattr(app, "_peak_calc_dialog", None)
    app._set_assign_mode.set(False)

    event = SimpleNamespace(artist=line, mouseevent=SimpleNamespace(button=3))
    app._on_explore_spectrum_pick(event, id(frame))

    assert state["selected_sample"] == target
    assert app.excluded_spectra == {target}
    assert app.X.index.isin(app.excluded_spectra).sum() == 1


# ---------------------------------------------------------------------------
# R007: Quality Check marks the real sample label
# ---------------------------------------------------------------------------

_QC_LABELS = {
    "strings": [f"S{i:03d}" for i in range(1, 41)],
    "shuffled-integers": list(np.random.default_rng(0).permutation(40)),
    "gapped-integers": [3 * i + 7 for i in range(40)],
}


@pytest.fixture(params=list(_QC_LABELS), ids=list(_QC_LABELS))
def qc_app(clean_state, request):
    app = clean_state
    labels = _QC_LABELS[request.param]
    X, y = _spectra(labels, seed=6)
    X.iloc[4] = X.iloc[4] * 8 + 50  # a gross outlier at row position 4
    X.iloc[17] = X.iloc[17] * -6  # and one at position 17
    assert app._install_dataset(X, y, None, None, replot=False)
    for name in (
        "_plot_pca_scores",
        "_plot_hotelling_t2",
        "_plot_q_residuals",
        "_plot_mahalanobis",
        "_plot_y_distribution",
        "_generate_plots",
        "_generate_explore_plots",
        "_detect_and_display_imbalance",
    ):
        setattr(app, name, lambda *a, **k: None)
    app.y_min_bound.set("")
    app.y_max_bound.set("")
    yield app
    for name in (
        "_plot_pca_scores",
        "_plot_hotelling_t2",
        "_plot_q_residuals",
        "_plot_mahalanobis",
        "_plot_y_distribution",
        "_generate_plots",
        "_generate_explore_plots",
        "_detect_and_display_imbalance",
    ):
        app.__dict__.pop(name, None)


def _flagged_labels(app, minimum_flags=1):
    summary = app.outlier_report["outlier_summary"]
    rows = summary[summary["Total_Flags"] >= minimum_flags]
    return {app.X.index[int(p)] for p in rows["Sample_Index"]}


def test_quality_check_marks_and_unmarks_real_labels(qc_app):
    app = qc_app
    app._run_outlier_detection()
    expected = _flagged_labels(app)
    assert app.X.index[4] in expected and app.X.index[17] in expected

    app.select_all_flagged.set(True)
    app._auto_select_flagged()
    app._mark_selected_for_exclusion()
    assert app.excluded_spectra == expected
    shown = {app.outlier_tree.item(i, "values")[0] for i in app.outlier_tree.get_children()}
    assert {str(label) for label in app.X.index} == shown

    app._unmark_selected_from_exclusion()
    assert app.excluded_spectra == set()


def test_quality_check_high_and_moderate_select_real_labels(qc_app):
    app = qc_app
    app._run_outlier_detection()
    for var, select, flags in (
        (app.select_high_conf, app._auto_select_high_confidence, lambda n: n >= 3),
        (app.select_moderate_conf, app._auto_select_moderate_confidence, lambda n: n == 2),
    ):
        summary = app.outlier_report["outlier_summary"]
        want = {
            app.X.index[int(r.Sample_Index)] for r in summary.itertuples() if flags(r.Total_Flags)
        }
        var.set(True)
        select()
        got = {app._get_sample_index_from_tree_item(i) for i in app.outlier_tree.selection()}
        var.set(False)
        assert got == want


def test_stale_quality_check_report_is_refused(qc_app):
    app = qc_app
    app._run_outlier_detection()
    app.select_all_flagged.set(True)
    app._auto_select_flagged()
    app.X = app.X.iloc[1:]  # rows removed outside the install path
    with patch("tkinter.messagebox.showwarning") as warn:
        app._mark_selected_for_exclusion()
    assert warn.called and app.excluded_spectra == set()


def test_dataset_replacement_drops_quality_check_report(qc_app):
    app = qc_app
    app._run_outlier_detection()
    X_b, y_b = _spectra([f"B{i}" for i in range(10)], seed=9)
    assert app._install_dataset(X_b, y_b, None, None, replot=False)
    assert app.outlier_report is None
    assert app.outlier_tree.get_children() == ()


# ---------------------------------------------------------------------------
# Review round 1
# ---------------------------------------------------------------------------


def _ftir_spectra(ids, seed: int):
    """FTIR-like data on a 0.48 cm^-1 axis (collapses if rounded to integers)."""
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.uniform(0, 10, len(ids)), index=pd.Index(ids))
    columns = [round(4000.0 - 0.48 * i, 2) for i in range(N_WL)][::-1]
    X = pd.DataFrame(
        np.outer(y.to_numpy(), np.linspace(0.5, 1.5, N_WL)) + rng.normal(0, 0.2, (len(ids), N_WL)),
        index=y.index,
        columns=columns,
    )
    return X, y


def _use_dm_source(app, monkeypatch, X, y, ref=None, name="B"):
    source = DataSource(
        source_id="b", name=name, path="b.csv", format_type="csv", X=X, y=y, ref=ref
    )
    monkeypatch.setattr(
        app,
        "data_source_manager",
        SimpleNamespace(get_source=lambda sid: source, merged_dataset=None),
    )
    monkeypatch.setattr(
        app,
        "data_sources_tree",
        SimpleNamespace(selection=lambda: ("item",), item=lambda item: {"text": "b"}),
    )
    monkeypatch.setattr(app, "_update_ensemble_controls_state", lambda: None)
    app._use_for_analysis()


def _gui_record(app) -> dict:
    """What the GUI records about the calibration rows at a launch."""
    return {
        "calibration_rows": app._calibration_rows_for_record(),
        "label_normalization": LABEL_NORMALIZATION,
        "calibration_identity": app._calibration_identity_now(),
    }


def _start_and_resume(rs, X, y, **kwargs):
    """A saved Bayesian run, claimed for resume as the startup 'Resume' answer does."""
    from pathlib import Path

    meta = rs.start_run(
        label="t",
        dataset_fingerprint=rs.fingerprint_dataset(X, y),
        bayesian_persistence_mode="always",
        **kwargs,
    )
    Path(meta.storage_path).write_bytes(b"SQLite format 3\x00")
    rs._reset_for_tests()
    assert rs.resume_run(meta.run_id) is not None
    return meta


# --- Item 1: resume keeps the run's calibration rows ------------------------


@pytest.fixture
def resumed_from_file(clean_state, worker_env, fake_thread, tmp_path):
    app, rs = clean_state, worker_env
    path = tmp_path / "a.csv"
    _write_combined(path, [f"A{i}" for i in range(1, 31)], seed=1)
    _load_combined(app, path)
    excluded = app.X.index[2]
    app.excluded_spectra = {excluded}
    meta = _start_and_resume(rs, app.X, app.y, **_gui_record(app))
    _load_combined(app, path)  # reload the identical file: replace clears exclusions
    assert app.excluded_spectra == set()
    yield app, rs, meta, excluded
    rs._reset_for_tests()


def test_resume_after_reloading_same_data_preserves_excluded_calibration_rows(
    resumed_from_file, monkeypatch
):
    app, rs, meta, excluded = resumed_from_file
    with patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask:
        calls, _ = _run(app, monkeypatch)
    assert ask.call_args[0][0] == "Excluded samples differ from the interrupted run"
    assert app.excluded_spectra == {excluded}
    assert calls and calls[0][2] == 29, "resumed on the run's own calibration rows"


def test_resume_exclusion_mismatch_cancel_keeps_record(resumed_from_file):
    app, rs, meta, _ = resumed_from_file
    with patch("tkinter.messagebox.askyesnocancel", return_value=None):
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert rs.is_resuming() and rs.find_incomplete_run().run_id == meta.run_id
    assert app.excluded_spectra == set()


def test_resume_exclusions_match_needs_no_question(resumed_from_file):
    app, rs, meta, excluded = resumed_from_file
    app.excluded_spectra = {excluded}
    with patch("tkinter.messagebox.askyesnocancel") as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert not ask.called


def test_resume_of_record_without_calibration_rows_still_resumes(clean_state, worker_env):
    app, rs = clean_state, worker_env
    X, y = _spectra([f"S{i}" for i in range(1, 21)], seed=2)
    assert app._install_dataset(X, y, None, None, replot=False)
    _start_and_resume(rs, app.X, app.y)
    assert app._confirm_resume_before_launch(["PLS"], "quick") is True


# --- Item 2: Data Management keeps its exact axis ---------------------------


def test_data_management_use_preserves_subunit_wavelength_axis(clean_state, monkeypatch):
    app = clean_state
    X_b, y_b = _ftir_spectra([f"F{i}" for i in range(1, 21)], seed=3)
    with patch("tkinter.messagebox.showerror") as err:
        _use_dm_source(app, monkeypatch, X_b, y_b)
    assert not err.called
    pd.testing.assert_frame_equal(app.X, X_b)

    app.wavelength_min.set("3990")
    app.wavelength_max.set("3995")
    with patch("tkinter.messagebox.showerror") as err:
        app._update_wavelengths()
    assert not err.called
    wl = X_b.columns.astype(float)
    pd.testing.assert_frame_equal(app.X, X_b.loc[:, (wl >= 3990) & (wl <= 3995)])


def test_resume_data_management_preserves_fractional_wavelength_axis(
    clean_state, worker_env, monkeypatch
):
    app, rs = clean_state, worker_env
    X_b, y_b = _ftir_spectra([f"F{i}" for i in range(1, 21)], seed=3)
    # The run was recorded on Data Management's exact axis, as before this branch.
    _start_and_resume(rs, X_b, y_b)
    _use_dm_source(app, monkeypatch, X_b, y_b)
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert not ask.called, "no 'different data' question for the same data"


# --- Items 3 and 4: a rejected load restores units and metadata -------------


def test_rejected_import_restores_units_and_conversion_state(clean_state, tmp_path):
    app = clean_state
    _write_combined(tmp_path / "a.csv", [f"A{i}" for i in range(1, 31)], seed=1)
    _load_combined(app, tmp_path / "a.csv")
    app.current_data_type.set("absorbance")
    app.original_data_type.set("absorbance")
    app.current_x_unit.set("cm-1")
    app.original_x_unit.set("cm-1")
    app.data_value_scale = 100.0
    app.data_has_been_converted = True
    meta_a = app.combined_metadata_df
    before = {v: getattr(app, v).get() for v in app._DATASET_STATE_VARS}

    _write_combined(tmp_path / "b.csv", [f"B{i}" for i in range(1, 31)], seed=2, step=0.3)
    with patch("tkinter.messagebox.showerror"):
        _load_combined(app, tmp_path / "b.csv")

    assert {v: getattr(app, v).get() for v in app._DATASET_STATE_VARS} == before
    assert app.data_value_scale == 100.0 and app.data_has_been_converted is True
    assert app.combined_metadata_df is meta_a


def test_browsing_a_file_does_not_touch_loaded_metadata(clean_state, tmp_path):
    app = clean_state
    _write_combined(tmp_path / "a.csv", [f"A{i}" for i in range(1, 31)], seed=1)
    _load_combined(app, tmp_path / "a.csv")
    meta_a = app.combined_metadata_df
    folder = tmp_path / "b"
    folder.mkdir()
    _write_combined(
        folder / "b.csv",
        [f"B{i}" for i in range(1, 31)],
        seed=2,
        step=0.3,
        extra={"site": ["north"] * 30},
    )
    with patch("tkinter.filedialog.askdirectory", return_value=str(folder)):
        app._browse_spectral_data()
    assert app.combined_metadata_df is meta_a
    with patch("tkinter.messagebox.showerror"):
        app._load_and_plot_data()  # sub-integer axis: rejected
    assert app.combined_metadata_df is meta_a
    assert "site" not in app._get_available_target_columns()


# --- Restore paths that stop before the install ------------------------------


def _loaded_a(app, tmp_path):
    _write_combined(tmp_path / "a.csv", [f"A{i}" for i in range(1, 31)], seed=1)
    _load_combined(app, tmp_path / "a.csv")
    _write_combined(tmp_path / "b.csv", [f"B{i}" for i in range(1, 21)], seed=2, offset=100.0)
    return app.X, app.X_original, app.y, app.combined_metadata_df


def _raise(exc):
    def _fail(*args, **kwargs):
        raise exc

    return _fail


def test_append_failure_restores_previous_dataset(clean_state, tmp_path, monkeypatch):
    app = clean_state
    X_a, X_orig_a, y_a, meta_a = _loaded_a(app, tmp_path)
    monkeypatch.setattr(app, "_prompt_for_group_names", lambda: ("A", "B"))
    monkeypatch.setattr(app, "_merge_spectral_data", _raise(ValueError("boom")))
    with patch("tkinter.messagebox.showerror"):
        _load_combined(app, tmp_path / "b.csv", append=True)
    assert app.X is X_a and app.X_original is X_orig_a and app.y is y_a
    assert app.combined_metadata_df is meta_a


def test_x_y_misalignment_restores_previous_dataset(clean_state, tmp_path, monkeypatch):
    import spectral_predict.io as io_module

    app = clean_state
    X_a, X_orig_a, y_a, meta_a = _loaded_a(app, tmp_path)
    real = io_module.read_combined_csv

    def misaligned(*args, **kwargs):
        X, y, meta_df, meta = real(*args, **kwargs)
        return X, y.set_axis([f"other{i}" for i in range(len(y))]), meta_df, meta

    monkeypatch.setattr(io_module, "read_combined_csv", misaligned)
    with patch("tkinter.messagebox.showerror") as err:
        _load_combined(app, tmp_path / "b.csv")
    assert err.call_args[0][0] == "Data Alignment Error"
    assert app.X is X_a and app.X_original is X_orig_a and app.y is y_a
    assert app.combined_metadata_df is meta_a


def test_exception_before_install_restores_previous_dataset(clean_state, tmp_path, monkeypatch):
    app = clean_state
    X_a, X_orig_a, y_a, meta_a = _loaded_a(app, tmp_path)
    monkeypatch.setattr(app, "_install_dataset", _raise(RuntimeError("boom")))
    with patch("tkinter.messagebox.showerror") as err:
        _load_combined(app, tmp_path / "b.csv")
    assert err.called
    assert app.X is X_a and app.X_original is X_orig_a and app.y is y_a
    assert app.combined_metadata_df is meta_a


def test_calibration_transfer_replace_installs_new_dataset(clean_state, monkeypatch):
    app = clean_state
    X_a, y_a = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    assert app._install_dataset(X_a, y_a, None, None, replot=False)
    app.excluded_spectra = {"A3"}
    _split(app, ["A1", "A2", "A4"])
    X_b, y_b = _ftir_spectra([f"T{i}" for i in range(1, 11)], seed=4)
    saved = (
        getattr(app, "transformed_spectra", None),
        getattr(app, "export_metadata_context", None),
    )
    app.transformed_spectra = (np.asarray(X_b.columns, dtype=float), X_b.to_numpy())
    app.export_metadata_context = {
        "source_format": "smart_csv",
        "specimen_ids": list(X_b.index),
        "metadata_df": pd.DataFrame({"protein": y_b.to_numpy(), "site": "n"}),
    }
    monkeypatch.setattr(app, "_ct_select_y_column_dialog", lambda cols: "protein")
    try:
        app._ct_use_as_working_data()
    finally:
        app.transformed_spectra, app.export_metadata_context = saved

    np.testing.assert_allclose(app.X.to_numpy(), X_b.to_numpy())
    assert list(app.X.columns) == [float(c) for c in X_b.columns], "exact CT axis"
    np.testing.assert_allclose(app.y.to_numpy(), y_b.to_numpy())
    assert app.excluded_spectra == set() and app.validation_indices == set()
    assert app.validation_X is None
    assert list(app.combined_metadata_df.columns) == ["site"]


# --- Item 5: targets merged by sample across metadata stores ---------------


def test_data_management_append_target_switch_preserves_all_targets_and_subset_columns(
    clean_state, tmp_path, monkeypatch
):
    app = clean_state
    X_b, y_b = _spectra([f"B{i}" for i in range(1, 11)], seed=2)
    ref_b = pd.DataFrame({"protein": y_b, "site": "north"}, index=X_b.index)
    _use_dm_source(app, monkeypatch, X_b, y_b, ref=ref_b)
    assert app.data_sources[0][0].startswith("Data Management")

    monkeypatch.setattr(app, "_prompt_for_group_names", lambda: ("B", "C"))
    _write_combined(tmp_path / "c.csv", [f"C{i}" for i in range(1, 11)], seed=3)
    _load_combined(app, tmp_path / "c.csv", append=True)
    assert len(app.X) == 20

    app.target_column.set("protein")
    app._on_target_column_changed()
    assert not app.y.isna().any()
    np.testing.assert_allclose(app.y.loc[X_b.index].to_numpy(), y_b.to_numpy())
    site = app._get_column_series("site")
    assert (site.loc[X_b.index] == "north").all()
    assert {"protein", "site"} <= set(app._get_available_target_columns())


# --- Item 6: prediction actuals travel with the prediction spectra ----------


def test_validation_prediction_data_stays_aligned_after_exclusion_and_search(clean_state):
    app = clean_state
    _install_regression(app)
    _split(app, ["S1", "S2", "S3", "S4", "S5"])
    app.pred_data_source.set("validation")
    app._load_prediction_data()
    assert list(app.prediction_actuals.index) == list(app.prediction_data.index)

    app.excluded_spectra = {"S2"}
    app._refresh_validation_snapshot()  # what a launch does
    assert "S2" not in app.validation_y.index
    actual = app.prediction_actuals.loc[app.prediction_data.index]  # no KeyError
    assert len(actual) == 5 and app._prediction_has_actuals()


# --- Item 7: consumers between runs use matching validation data -----------


def test_explore_exclusion_refreshes_validation_before_refinement(clean_state):
    import contextlib
    import io

    app = clean_state
    _install_regression(app)
    _split(app, ["S1", "S2", "S3", "S4", "S5"])
    app.excluded_spectra = {"S2"}  # after the split; self.validation_X still has S2
    assert "S2" in app.validation_X.index
    app.use_autoscale.set(False)
    app.selected_model_config = {
        "Model": "PLS",
        "Task": "regression",
        "Params": "{'n_components': 2}",
        "LVs": 2,
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
    }
    app._original_wavelength_order = [float(c) for c in app.X.columns]
    app.refine_task_type.set("regression")
    app.refine_model_type.set("PLS")
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
    out = buf.getvalue()
    assert app.refined_model is not None, out[-3000:]
    assert "Validation prediction successful (n=4)" in out, out[-3000:]


# --- Items 8, 9, 11: other paths that change the data -----------------------


def test_data_type_conversion_keeps_exclusions_out_of_validation(clean_state):
    app = clean_state
    _install_regression(app)
    _split(app, ["S1", "S2", "S3", "S4"])
    app.excluded_spectra = {"S2"}
    app.type_confidence = 0.0
    app._convert_and_replot()
    assert list(app.validation_X.index) == ["S1", "S3", "S4"]
    pd.testing.assert_frame_equal(app.validation_X, app.X.loc[["S1", "S3", "S4"]])


class _StubSheet:
    def __init__(self, headers, rows):
        self._headers, self._rows = headers, rows

    def get_sheet_data(self):
        return self._rows

    def headers(self):
        return self._headers


def test_data_viewer_edit_prunes_split_and_revert_restores_it(clean_state, monkeypatch):
    app = clean_state
    X, y = _install_regression(app, n=12)
    _split(app, ["S1", "S2", "S3"])
    app.excluded_spectra = {"S5"}
    app.target_column.set("protein")
    monkeypatch.setattr(app, "_populate_data_viewer", lambda *a, **k: None)
    app._snapshot_data_viewer_state()
    app.outlier_report = {"outlier_summary": pd.DataFrame()}

    keep = [s for s in X.index if s not in ("S2", "S5")]  # rows deleted in the sheet
    headers = ["Sample ID", "protein"] + [str(c) for c in app.X.columns]
    rows = [[s, y[s]] + list(app.X.loc[s]) for s in keep]
    monkeypatch.setattr(app, "data_viewer_sheet", _StubSheet(headers, rows))
    app._apply_data_viewer_edits()

    assert list(app.X.index) == keep
    assert app.validation_indices == {"S1", "S3"} and app.excluded_spectra == set()
    assert list(app.validation_X.index) == ["S1", "S3"]
    assert app.outlier_report is None

    app._revert_data_viewer()
    assert app.validation_indices == {"S1", "S2", "S3"}
    assert list(app.validation_X.index) == ["S1", "S2", "S3"]


def test_quality_check_report_refused_after_baseline_replace(qc_app, monkeypatch):
    app = qc_app
    app._run_outlier_detection()
    app.select_all_flagged.set(True)
    app._auto_select_flagged()
    monkeypatch.setattr(app, "_compute_corrected_spectra", lambda method: app.X.to_numpy() - 1)
    app._replace_working_data("als")
    with patch("tkinter.messagebox.showwarning") as warn:
        app._mark_selected_for_exclusion()
    assert warn.called and app.excluded_spectra == set()


# --- Item 10: unique internal sample IDs ------------------------------------


def test_duplicate_suffix_collision_keeps_qc_and_plot_sample_identity(clean_state, monkeypatch):
    from spectral_predict.io import rename_duplicate_ids

    new, n, _ = rename_duplicate_ids(pd.Index(["A", "A.1", "A"]))
    assert list(new) == ["A", "A.1", "A.2"] and n == 1

    app = clean_state
    X, y = _spectra(["A", "A.1", "A.1", "B", "C", "D"], seed=7)
    with patch("tkinter.messagebox.showwarning") as warn:
        assert app._install_dataset(X, y, None, None, replot=False)
    assert warn.called and app.X.index.is_unique
    np.testing.assert_allclose(app.y.to_numpy(), y.to_numpy())

    monkeypatch.setattr(app, "_create_or_update_annotation", lambda *a, **k: None)
    line, _ = _line_for(app, app.X.index[2])
    app._on_spectrum_click(SimpleNamespace(artist=line, canvas=SimpleNamespace(draw=lambda: None)))
    assert app.X.index.isin(app.excluded_spectra).sum() == 1


# ---------------------------------------------------------------------------
# Review round 2
# ---------------------------------------------------------------------------


@pytest.fixture
def resumable(clean_state, worker_env, fake_thread):
    """Save a run on the loaded data with given exclusions/subset, then claim it."""
    app, rs = clean_state, worker_env

    def _save(X, y, excluded=(), active=None, **kwargs):
        assert app._install_dataset(X, y, None, None, replot=False)
        app.excluded_spectra = set(excluded)
        app.active_indices = None if active is None else set(active)
        for key, value in _gui_record(app).items():
            kwargs.setdefault(key, value)
        return _start_and_resume(rs, app.X, app.y, **kwargs)

    yield app, rs, _save
    rs._reset_for_tests()


def test_resume_refuses_missing_excluded_label(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    meta = save(X, y, excluded=["A3"])
    renamed = X.rename(index={"A3": "A3-renamed"})
    assert app._install_dataset(renamed, y.set_axis(renamed.index), None, None, replot=False)
    with patch("tkinter.messagebox.showerror") as err:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert err.call_args[0][0] == "Can't restore the calibration samples"
    assert rs.find_incomplete_run().run_id == meta.run_id
    assert app.excluded_spectra == set()


def test_resume_refuses_missing_active_label(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    meta = save(X, y, active=[f"A{i}" for i in range(1, 11)])
    renamed = X.rename(index={"A5": "A5-renamed"})
    assert app._install_dataset(renamed, y.set_axis(renamed.index), None, None, replot=False)
    app.active_indices = {s for s in renamed.index[:10]}
    with patch("tkinter.messagebox.showerror") as err:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert err.called and rs.find_incomplete_run().run_id == meta.run_id


def test_float_labels_are_recorded_and_restored_on_resume(resumable):
    app, rs, save = resumable
    X, y = _spectra([float(i) for i in range(1, 21)], seed=1)
    save(X, y, excluded=[3.0])
    assert app._install_dataset(X, y, None, None, replot=False)  # reload clears exclusions
    with patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert ask.call_args[0][0] == "Excluded samples differ from the interrupted run"
    assert app.excluded_spectra == {3.0}


def test_new_record_without_rows_asks_instead_of_resuming_silently(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    meta = save(X, y, calibration_rows=None)
    with patch("tkinter.messagebox.askyesnocancel", return_value=None) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert ask.call_args[0][0] == "Can't verify the calibration samples"
    assert rs.find_incomplete_run().run_id == meta.run_id


def test_legacy_resume_duplicate_suffix_preserves_holdout_membership(
    clean_state, worker_env, fake_thread, tmp_path
):
    """A run saved under the old suffix scheme is not resumed on relabelled rows.

    The file repeats "A" next to an existing "A.1". The old reader named both
    repeats "A.1", so a holdout of "A.1" held out two spectra; the reader now
    gives "A.1" and "A.2", and "A.1" would hold out one.
    """
    app, rs = clean_state, worker_env
    ids = ["A", "A.1", "A"] + [f"S{i}" for i in range(4, 31)]
    path = tmp_path / "dup.csv"
    _write_combined(path, ids, seed=1)
    _load_combined(app, path)
    assert list(app.X.index[:3]) == ["A", "A.1", "A.2"]
    meta = _start_and_resume(
        rs,
        app.X,
        app.y,
        validation_indices=["A.1", "S4", "S5"],
        label_normalization=None,  # a record written before the new scheme
    )
    app.validation_enabled.set(True)
    with patch("tkinter.messagebox.askyesnocancel", return_value=None) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert ask.call_args[0][0] == "Can't verify the calibration samples"
    assert rs.find_incomplete_run().run_id == meta.run_id
    assert app.validation_indices == set(), "the old holdout was not applied to new rows"


def test_legacy_resume_without_repeated_ids_still_resumes(
    clean_state, worker_env, fake_thread, tmp_path
):
    app, rs = clean_state, worker_env
    path = tmp_path / "a.csv"
    _write_combined(path, [f"A{i}" for i in range(1, 31)], seed=1)
    _load_combined(app, path)
    _start_and_resume(rs, app.X, app.y, label_normalization=None)
    assert app._confirm_resume_before_launch(["PLS"], "quick") is True


def test_dm_append_subset_dialog_includes_ref_only_columns(clean_state, tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import ttk

    app = clean_state
    X_b, y_b = _spectra([f"B{i}" for i in range(1, 11)], seed=2)
    ref_b = pd.DataFrame({"protein": y_b, "site": "north"}, index=X_b.index)
    _use_dm_source(app, monkeypatch, X_b, y_b, ref=ref_b)
    monkeypatch.setattr(app, "_prompt_for_group_names", lambda: ("B", "C"))
    _write_combined(
        tmp_path / "c.csv", [f"C{i}" for i in range(1, 11)], seed=3, extra={"batch": ["x"] * 10}
    )
    _load_combined(app, tmp_path / "c.csv", append=True)

    seen = {}

    def _combobox_values(widget):
        for child in widget.winfo_children():
            if isinstance(child, ttk.Combobox):
                return list(child.cget("values"))
            found = _combobox_values(child)
            if found is not None:
                return found
        return None

    def _close(dialog):
        seen["columns"] = _combobox_values(dialog)
        dialog.destroy()

    monkeypatch.setattr(tk.Toplevel, "wait_window", _close)
    app._show_active_group_dialog()
    assert {"site", "batch", "protein"} <= set(seen["columns"])


def test_comparison_validation_load_refreshes_excluded_rows(clean_state):
    app = clean_state
    _install_regression(app)
    _split(app, ["S1", "S2", "S3"])
    app.excluded_spectra = {"S2"}  # after the split; cached validation still has S2
    saved = (getattr(app, "comparison_data", None), app.comparison_data_source.get())
    try:
        app.comparison_data_source.set("validation")
        app._load_comparison_data()
        assert list(app.comparison_data.index) == ["S1", "S3"]
    finally:
        app.comparison_data = saved[0]
        app.comparison_data_source.set(saved[1])


def test_quality_check_report_refused_after_target_change(qc_app):
    app = qc_app
    app._run_outlier_detection()
    app.select_all_flagged.set(True)
    app._auto_select_flagged()
    app.combined_metadata_df = pd.DataFrame(
        {"protein": app.y, "moisture": app.y[::-1].to_numpy()}, index=app.X.index
    )
    app.target_column.set("moisture")
    app._on_target_column_changed()
    with patch("tkinter.messagebox.showwarning") as warn:
        app._mark_selected_for_exclusion()
    assert warn.called and app.excluded_spectra == set()


def test_manual_ensemble_retrain_receives_the_cached_holdout(clean_state, monkeypatch):
    app = clean_state
    X, y = _install_regression(app)
    pair = (X.loc[["S1", "S2"]], y.loc[["S1", "S2"]])
    app.training_data_cache = {
        "X_filtered": X.drop(index=["S1", "S2"]),
        "y_filtered": y.drop(index=["S1", "S2"]),
        "task_type": "regression",
        "label_encoder": None,
        "validation": pair,
    }
    got = {}

    def fake_train(*args, **kwargs):
        got["validation_data"] = kwargs.get("validation_data")
        raise RuntimeError("stop after the call")

    monkeypatch.setattr(app, "_train_ensembles", fake_train)
    monkeypatch.setattr(app.root, "after", lambda *a, **k: None)
    app._train_ensemble_thread()
    assert got["validation_data"] is pair


# ---------------------------------------------------------------------------
# Review round 3: one identity check decides
# ---------------------------------------------------------------------------

_DIFFERS = "The data or sample selection differs from the interrupted run"


def _reinstall(app, X, y):
    assert app._install_dataset(X, y, None, None, replot=False)


def test_matching_data_resumes_after_identity_check(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    save(X, y, excluded=["A3"])
    _reinstall(app, X, y)  # reload: exclusions cleared, restored on Yes
    with (
        patch("tkinter.messagebox.askyesnocancel", return_value=True),
        patch("tkinter.messagebox.askyesno") as ask,
    ):
        assert app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert not ask.called and app.excluded_spectra == {"A3"}


def test_label_swap_with_identical_values_is_refused(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    meta = save(X, y)
    swapped = X.rename(index={"A1": "A2", "A2": "A1"})
    _reinstall(app, swapped, y.set_axis(swapped.index))
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert ask.call_args[0][0] == _DIFFERS
    assert rs.find_incomplete_run().run_id == meta.run_id


def test_row_order_change_is_refused(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    meta = save(X, y)
    order = list(X.index)
    order[3], order[4] = order[4], order[3]  # away from the coarse fingerprint's cells
    _reinstall(app, X.loc[order], y.loc[order])
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert ask.call_args[0][0] == _DIFFERS
    assert rs.find_incomplete_run().run_id == meta.run_id


def test_edited_spectrum_is_refused_even_when_labels_match(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    save(X, y)
    edited = X.copy()
    edited.iloc[5, 7] += 1e-3
    _reinstall(app, edited, y)
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert ask.call_args[0][0] == _DIFFERS


def test_float_holdout_labels_survive_and_are_restored(resumable):
    app, rs, save = resumable
    X, y = _spectra([float(i) for i in range(1, 21)], seed=1)
    _reinstall(app, X, y)
    _split(app, [1.0, 2.0, 3.0])
    save_rows = _gui_record(app)
    meta = _start_and_resume(rs, app.X, app.y, **save_rows)
    _reinstall(app, X, y)  # reload clears the holdout
    app.validation_enabled.set(True)
    assert app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert app.validation_indices == {1.0, 2.0, 3.0}
    assert meta.calibration_identity["n_holdout"] == 3


def test_unsupported_label_type_asks(resumable):
    import datetime as dt

    app, rs, save = resumable
    labels = [dt.date(2026, 1, d) for d in range(1, 21)]
    X, y = _spectra(labels, seed=1)
    meta = save(X, y)
    assert meta.calibration_identity is None  # the dates can't be fingerprinted
    with patch("tkinter.messagebox.askyesnocancel", return_value=None) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert ask.call_args[0][0] == "Can't verify the calibration samples"
    assert rs.find_incomplete_run().run_id == meta.run_id


def test_unknown_record_version_asks(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    save(X, y, label_normalization=99)
    with patch("tkinter.messagebox.askyesnocancel", return_value=None) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert ask.call_args[0][0] == "Can't verify the calibration samples"


def test_legacy_resume_refuses_dm_normalized_duplicate_ids(clean_state, worker_env, monkeypatch):
    """Data Management data whose labels were already renamed by its reader."""
    app, rs = clean_state, worker_env
    ids = ["A", "A.1", "A.2"] + [f"S{i}" for i in range(4, 21)]
    X, y = _spectra(ids, seed=2)
    _use_dm_source(app, monkeypatch, X, y)
    meta = _start_and_resume(rs, app.X, app.y, validation_indices=["A.1"])  # legacy record
    with patch("tkinter.messagebox.askyesnocancel", return_value=None) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert ask.call_args[0][0] == "Can't verify the calibration samples"
    assert rs.find_incomplete_run().run_id == meta.run_id


def test_viewer_revert_restores_legacy_label_check(clean_state, monkeypatch):
    app = clean_state
    X, y = _install_regression(app, n=8)
    app.target_column.set("protein")
    monkeypatch.setattr(app, "_populate_data_viewer", lambda *a, **k: None)
    app._snapshot_data_viewer_state()
    assert app._labels_look_renamed() is False

    ids = list(X.index)
    ids[1] = ids[0]  # an edit gives two rows the same ID; install renames one
    headers = ["Sample ID", "protein"] + [str(c) for c in app.X.columns]
    rows = [[ids[i], y.iloc[i]] + list(app.X.iloc[i]) for i in range(len(ids))]
    monkeypatch.setattr(app, "data_viewer_sheet", _StubSheet(headers, rows))
    with patch("tkinter.messagebox.showwarning"):
        app._apply_data_viewer_edits()
    assert app._labels_look_renamed() is True

    app._revert_data_viewer()
    assert app._labels_look_renamed() is False


def test_rename_duplicate_ids_accepts_duplicate_multiindex():
    from spectral_predict.io import rename_duplicate_ids

    index = pd.MultiIndex.from_tuples([("a", 1), ("a", 1), ("b", 2)])
    new, n, _ = rename_duplicate_ids(index)
    assert n == 1 and pd.Index(new).is_unique


# ---------------------------------------------------------------------------
# Review round 4: transactional reconciliation, worker parity, record checks
# ---------------------------------------------------------------------------


def test_refused_resume_leaves_selection_as_before_the_click(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    _reinstall(app, X, y)
    _split(app, ["A1", "A2"])
    save(X, y, excluded=["A3"])
    edited = X.copy()
    edited.iloc[7, 3] += 1e-3  # a cell the coarse fingerprint doesn't sample
    _reinstall(app, edited, y)  # reload: exclusions and holdout cleared
    app.validation_enabled.set(True)
    with (
        patch("tkinter.messagebox.askyesnocancel", return_value=True) as approve,
        patch("tkinter.messagebox.askyesno", return_value=False) as keep,
    ):
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert approve.call_args[0][0] == "Excluded samples differ from the interrupted run"
    assert keep.call_args[0][0] == _DIFFERS
    assert app.excluded_spectra == set() and app.validation_indices == set()
    assert app.validation_X is None


def test_rare_class_rows_the_worker_drops_are_not_in_the_identity(resumable):
    app, rs, save = resumable
    ids = [f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)] + ["C0"]
    X, _ = _spectra(ids, seed=3)
    y = pd.Series(["a"] * 5 + ["b"] * 5 + ["c"], index=X.index)
    app.task_type.set("classification")
    app.folds.set(5)
    save(X, y)
    renamed = X.rename(index={"C0": "C0-renamed"})
    _reinstall(app, renamed, y.set_axis(renamed.index))
    with patch("tkinter.messagebox.askyesno", return_value=False) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert not ask.called


def _capturing_bayesian():
    calls = []

    def fake(X, y, wavelengths, model_name, **kwargs):
        calls.append((np.array(X, copy=True), np.array(y, copy=True)))
        from tests.gui.test_resume_run_completion import _successful_results_row

        return _successful_results_row(), None

    return fake, calls


def _labelled(X_run, values):
    """The rows of ``X_run`` the worker received, found by their exact values."""
    labels = []
    for row in values:
        match = [lab for lab, ref in zip(X_run.index, X_run.to_numpy()) if np.array_equal(ref, row)]
        assert len(match) == 1
        labels.append(match[0])
    return X_run.loc[labels]


@pytest.mark.parametrize("call_site", ["launch_gate", "worker_fallback"])
@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"excluded": ["S3", "S9"]},
        {"active": [f"S{i}" for i in range(1, 25)], "excluded": ["S2"], "holdout": ["S5", "S6"]},
    ],
    ids=["none", "excluded", "subset-excluded-holdout"],
)
def test_recorded_identity_matches_what_the_worker_trains_on(
    clean_state, worker_env, fake_thread, monkeypatch, call_site, selection
):
    from spectral_predict.run_state import calibration_identity

    app, rs = clean_state, worker_env
    X, y = _spectra([f"S{i}" for i in range(1, 31)], seed=4)
    _reinstall(app, X, y)
    app.excluded_spectra = set(selection.get("excluded", ()))
    app.active_indices = set(selection["active"]) if "active" in selection else None
    if "holdout" in selection:
        _split(app, selection["holdout"])
    recorded = []
    real_start_run = rs.start_run

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("calibration_identity"))
        return real_start_run(*args, **kwargs)

    monkeypatch.setattr(rs, "start_run", spy)
    fake, calls = _capturing_bayesian()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)
    seen = _recording_validation(monkeypatch)
    _select_models(app, ["PLS"])
    with patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
        if call_site == "launch_gate":
            app._run_analysis()
            worker = _FakeThread.created[-1]
            worker.target(*worker.args, **worker.kwargs)
        else:
            app._pending_analysis_settings = None
            app._run_analysis_thread(["PLS"], "quick")

    X_cal = _labelled(app.X, calls[0][0])
    y_cal = pd.Series(calls[0][1], index=X_cal.index)
    if "holdout" in selection:
        X_val = _labelled(app.X, seen[0]["X_val"])
        y_val = pd.Series(seen[0]["y_val"], index=X_val.index)
    else:
        X_val = y_val = None
    expected = calibration_identity(X_cal, y_cal, X_val, y_val)
    assert recorded and recorded[0]["calibration"] == expected["calibration"]
    assert recorded[0]["holdout"] == expected["holdout"]


def test_mixed_type_classification_identity_matches_worker(
    clean_state, worker_env, fake_thread, monkeypatch
):
    from spectral_predict.run_state import calibration_identity

    app, rs = clean_state, worker_env
    X, _ = _spectra([f"S{i}" for i in range(1, 25)], seed=5)
    y = pd.Series(["a"] * 8 + [1] * 8 + ["1.0"] * 8, index=X.index, dtype=object)
    _reinstall(app, X, y)
    app.task_type.set("classification")
    recorded = []
    real_start_run = rs.start_run
    monkeypatch.setattr(
        rs,
        "start_run",
        lambda *a, **k: recorded.append(k.get("calibration_identity")) or real_start_run(*a, **k),
    )
    fake, calls = _capturing_bayesian()
    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    monkeypatch.setattr("spectral_predict.report.write_markdown_report", lambda *a, **k: None)
    _select_models(app, ["PLS"])
    with patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
        app._run_analysis()
        worker = _FakeThread.created[-1]
        worker.target(*worker.args, **worker.kwargs)
    assert calls, "the worker ran"
    X_cal = _labelled(app.X, calls[0][0])
    y_cal = pd.Series(calls[0][1], index=X_cal.index)
    assert (
        recorded[0]["calibration"] == calibration_identity(X_cal, y_cal, None, None)["calibration"]
    )


def test_record_missing_active_key_asks_without_crashing(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    save(X, y)
    meta = rs.get_resumed_run()
    # A record read back with "active" missing (start_run itself would not write it).
    meta.calibration_rows = {"version": 1, "excluded": [], "holdout": []}
    with patch("tkinter.messagebox.askyesnocancel", return_value=None) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert ask.call_args[0][0] == "Can't verify the calibration samples"
    assert rs.find_incomplete_run().run_id == meta.run_id


def test_malformed_holdout_resume_anyway_uses_current_selection(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    save(X, y)
    meta = rs.get_resumed_run()
    meta.calibration_rows = {
        "version": 1,
        "excluded": [],
        "active": None,
        "holdout": [["A1"]],
    }
    app.excluded_spectra = {"A4"}
    with patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert ask.call_args[0][0] == "Can't verify the calibration samples"
    assert app.excluded_spectra == {"A4"} and app.validation_indices == set()


def test_unknown_identity_version_asks(resumable):
    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    identity = dict(app._calibration_identity_now() or {})
    _reinstall(app, X, y)
    identity = dict(app._calibration_identity_now(), version=1)
    save(X, y, calibration_identity=identity)
    with patch("tkinter.messagebox.askyesnocancel", return_value=None) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert ask.call_args[0][0] == "Can't verify the calibration samples"


def test_legacy_resume_with_dotted_ids_is_unverifiable_not_refused(
    clean_state, worker_env, fake_thread
):
    app, rs = clean_state, worker_env
    X, y = _spectra(["S1", "S1.2"] + [f"T{i}" for i in range(18)], seed=6)
    _reinstall(app, X, y)
    _start_and_resume(rs, app.X, app.y)  # legacy record
    with patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask:
        assert app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert ask.call_args[0][0] == "Can't verify the calibration samples"


def test_rename_duplicate_ids_with_missing_tuple_parts_is_unique():
    from spectral_predict.io import rename_duplicate_ids

    index = pd.MultiIndex.from_tuples([("a", np.nan), ("a", np.nan), ("b", 2)])
    new, n, _ = rename_duplicate_ids(index)
    assert n == 1 and len(new) == 3
    assert len({repr(v) for v in new}) == 3


# ---------------------------------------------------------------------------
# Review round 5
# ---------------------------------------------------------------------------


def test_exception_during_reconciliation_restores_selection(resumable, monkeypatch):
    import spectral_predict.run_state as run_state

    app, rs, save = resumable
    X, y = _spectra([f"A{i}" for i in range(1, 21)], seed=1)
    _reinstall(app, X, y)
    _split(app, ["A1", "A2"])
    save(X, y, excluded=["A3"])
    _reinstall(app, X, y)  # reload: exclusions and holdout cleared
    app.validation_enabled.set(True)
    status_before = app.validation_status_label.cget("text")

    def broken(*args, **kwargs):
        raise RuntimeError("digest failed")

    monkeypatch.setattr(run_state, "calibration_identity", broken)
    # _run_analysis reports such an error and launches nothing.
    with (
        patch("tkinter.messagebox.askyesnocancel", return_value=True),
        patch("tkinter.messagebox.askyesno", return_value=False),
        pytest.raises(RuntimeError),
    ):
        app._confirm_resume_before_launch(["PLS"], "quick")
    assert app.excluded_spectra == set() and app.validation_indices == set()
    assert app.validation_X is None
    assert app.validation_status_label.cget("text") == status_before


def test_one_class_identity_matches_what_the_worker_trains_on(clean_state, worker_env, monkeypatch):
    from spectral_predict.run_state import calibration_identity

    class _Stop(BaseException):
        pass

    app, rs = clean_state, worker_env
    X, _ = _spectra([f"S{i}" for i in range(1, 31)], seed=7)
    y = pd.Series(["a"] * 22 + ["b"] * 8, index=X.index)
    _reinstall(app, X, y)
    app.excluded_spectra = {"S4", "S25"}
    app.task_type.set("one_class")
    recorded, calls = [], []
    real_start_run = rs.start_run
    monkeypatch.setattr(
        rs,
        "start_run",
        lambda *a, **k: recorded.append(k.get("calibration_identity")) or real_start_run(*a, **k),
    )

    def fake(X, y, wavelengths, model_name, **kwargs):
        calls.append((np.array(X, copy=True), np.array(y, copy=True)))
        raise _Stop

    monkeypatch.setattr(gui_module, "run_unified_bayesian", fake)
    app._pending_analysis_settings = None
    with patch("tkinter.messagebox.showerror"), patch("tkinter.messagebox.showwarning"):
        try:
            app._run_analysis_thread(["PCA-SIMCA"], "quick", resolved_inlier_label="a")
        except _Stop:
            pass
    assert calls, "the one-class search was reached"
    X_cal = _labelled(app.X, calls[0][0])
    assert "S4" not in X_cal.index and len(X_cal) == 28
    y_cal = pd.Series(calls[0][1], index=X_cal.index)
    assert (
        recorded[0]["calibration"] == calibration_identity(X_cal, y_cal, None, None)["calibration"]
    )


def test_prepare_calibration_aligns_x_and_y_before_hashing():
    X, y = _spectra([f"S{i}" for i in range(1, 7)], seed=8)
    shifted = y.set_axis([f"S{i}" for i in range(2, 8)])  # same length, labels shifted
    prepared = gui_module._prepare_calibration(
        X,
        shifted,
        active=None,
        excluded=(),
        holdout=(),
        validation_on=False,
        task_type="regression",
        cv_strategy="kfold",
        folds=3,
    )
    assert list(prepared.X.index) == list(prepared.y.index) == [f"S{i}" for i in range(2, 7)]
    assert prepared.n_realigned == 5  # the worker reports this


def test_rename_duplicate_ids_treats_nan_and_na_as_repeats():
    from spectral_predict.io import rename_duplicate_ids

    new, n, _ = rename_duplicate_ids(pd.Index([np.nan, pd.NA], dtype=object))
    assert n == 2 and list(new) == ["nan", "nan.1"]
