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


def _write_combined(path, ids, seed, offset=0.0, step=2.0):
    X, y = _spectra(ids, seed, offset, step)
    df = X.copy()
    df.insert(0, "protein", y.to_numpy())
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
    "data_sources",
    "source_group_names",
    "use_custom_group_names",
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
    gui_app.combined_metadata_df = None
    gui_app.ref = None
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
    expected = X_b.set_axis(X_b.columns.astype(float).astype(int), axis=1)
    expected = expected.loc[:, (expected.columns >= 1020) & (expected.columns <= 1100)]
    np.testing.assert_allclose(app.X.to_numpy(), expected.to_numpy())
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
