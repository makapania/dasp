"""GUI handling of reader metadata (fix/readers review round 1).

Checks, without building the Tk window, that:
- an OPUS log-reflectance spectrum (already -log10 R) is not logged a second time
  by the GUI's absorbance conversion, because the reader labels it absorbance;
- reader ``import_warnings`` reach a dialog (warnings.warn/print never do);
- a single-channel OPUS import is flagged in the data-type status note.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectral_predict import io as sp_io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
gui = pytest.importorskip("spectral_predict_gui_optimized")


class _Var:
    def __init__(self, value=None):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _bare_app():
    app = gui.SpectralPredictApp.__new__(gui.SpectralPredictApp)
    app.original_data_type = _Var()
    app.current_data_type = _Var()
    app.use_absorbance = _Var(False)
    return app


class _FakeData:
    def __init__(self, y):
        self.x = np.linspace(4000.0, 3410.0, y.size)
        self.y = y
        self.label = "fake"


class _FakeOpus:
    def __init__(self, blocks):
        self.is_opus = True
        self.data_keys = list(blocks)
        for key, data in blocks.items():
            setattr(self, key, data)

    def __getattr__(self, name):
        return None


def test_log_reflectance_is_not_logged_again(tmp_path, monkeypatch):
    logr = np.linspace(0.40, 0.50, 60)  # OPUS logr block: already -log10(R)
    path = tmp_path / "s.0"
    path.write_bytes(b"placeholder")
    opus = _FakeOpus({"logr": _FakeData(logr), "sm": _FakeData(np.full(60, 111.0))})
    module = types.ModuleType("brukeropus")
    module.read_opus = lambda p: opus
    monkeypatch.setitem(sys.modules, "brukeropus", module)

    X, metadata = sp_io.read_opus_file(path)
    app = _bare_app()
    app._apply_data_type_metadata(metadata)

    assert app.original_data_type.get() == "absorbance"
    converted = app._convert_data_type(X.to_numpy(), app.current_data_type.get(), "absorbance")
    # No second log: values are exactly the stored -log10(R)
    np.testing.assert_array_equal(np.asarray(converted)[0], logr[::-1])


def test_import_warnings_are_shown_in_a_dialog(monkeypatch):
    shown = []
    monkeypatch.setattr(
        gui.messagebox, "showwarning", lambda title, msg: shown.append((title, msg))
    )
    app = _bare_app()

    app._show_import_warnings({"import_warnings": ["first problem", "second problem"]}, "OPUS")

    assert len(shown) == 1
    assert shown[0][0] == "OPUS import warnings"
    assert "first problem" in shown[0][1] and "second problem" in shown[0][1]


def test_no_dialog_without_import_warnings(monkeypatch):
    shown = []
    monkeypatch.setattr(gui.messagebox, "showwarning", lambda *a: shown.append(a))
    app = _bare_app()

    app._show_import_warnings({"import_warnings": []}, "OPUS")
    app._show_import_warnings({}, "ASCII")
    app._show_import_warnings(None, "ASCII")

    assert shown == []


@pytest.mark.parametrize("label", ["OPUS", "ASCII", "PerkinElmer"])
def test_import_branches_call_the_dialog(label):
    """The main import paths for the three folder readers surface import_warnings."""
    source = Path(gui.__file__).read_text(encoding="utf-8")
    assert f'self._show_import_warnings(metadata, "{label}")' in source


class _Widget:
    def __init__(self):
        self.options: dict = {}

    def config(self, **kwargs):
        self.options.update(kwargs)


def _status_app(metadata):
    app = _bare_app()
    app._apply_data_type_metadata(metadata)
    app.X = np.zeros((1, 3))
    app.colors = {"success": "green", "warning": "orange", "text_light": "gray"}
    for name in (
        "data_type_status_label",
        "convert_data_button",
        "reflectance_radio",
        "absorbance_radio",
        "absorbance_checkbox",
    ):
        setattr(app, name, _Widget())
    app._update_data_type_status_ui()
    return app


def test_single_channel_source_is_noted_in_status():
    app = _status_app(
        {"data_type": "absorbance", "type_confidence": 60.0, "source_data_type": "sample"}
    )

    assert "OPUS single-channel: raw intensities" in app.data_type_status_label.options["text"]


def test_log_reflectance_status_offers_no_log_conversion():
    app = _status_app(
        {
            "data_type": "absorbance",
            "type_confidence": 95.0,
            "source_data_type": "log_reflectance",
        }
    )

    # The legacy "use absorbance" (log) checkbox is disabled for absorbance-like data
    assert app.absorbance_checkbox.options["state"] == "disabled"
    assert app.convert_data_button.options["text"] == "Convert to Reflectance"


# ---------------------------------------------------------------------------
# Review round 2: 'other' data types and the prediction / calibration-transfer paths
# ---------------------------------------------------------------------------


class _AnyWidget(_Widget):
    """Fake Tk widget: records config(); Text-style delete/insert keep the text."""

    def __init__(self):
        super().__init__()
        self.text = ""

    configure = _Widget.config

    def delete(self, *args):
        self.text = ""

    def insert(self, index, text="", **kwargs):
        self.text += str(text)

    def grid(self, *args, **kwargs):
        pass

    def grid_remove(self, *args, **kwargs):
        pass

    def get_children(self):
        return []


@pytest.fixture
def dialogs(monkeypatch):
    """Record every messagebox call instead of showing it."""
    calls: list[tuple[str, str, str]] = []
    for name in ("showwarning", "showerror", "showinfo"):
        monkeypatch.setattr(
            gui.messagebox,
            name,
            lambda title, msg="", _n=name, **kw: calls.append((_n, title, msg)),
        )
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *a, **k: True)
    return calls


def _install_opus(monkeypatch, files: dict[Path, dict[str, np.ndarray]]):
    registry = {}
    for path, blocks in files.items():
        path.write_bytes(b"placeholder")
        registry[str(path)] = _FakeOpus({k: _FakeData(v) for k, v in blocks.items()})
    module = types.ModuleType("brukeropus")
    module.read_opus = lambda p: registry[str(Path(p))]
    monkeypatch.setitem(sys.modules, "brukeropus", module)


def _ct_app(path):
    app = _bare_app()
    app.colors = {"success": "green", "warning": "orange", "text_light": "gray", "accent": "blue"}
    app.source_data_type = None
    app.data_value_scale = 1.0
    app.current_transfer_model = None
    app.ct_transfer_model = None
    app.ct_wavelengths_common = None
    app.play_sound = lambda *a, **k: None
    # Mode A (prediction)
    app.ct_pred_satellite_path_var = _Var(str(path))
    app.ct_pred_data_type = _Var()
    app.ct_pred_data_converted = False
    for name in (
        "ct_pred_data_type_status",
        "ct_pred_refl_radio",
        "ct_pred_abs_radio",
        "ct_pred_convert_btn",
        "ct_pred_satellite_info_text",
        "ct_pred_conversion_status",
    ):
        setattr(app, name, _AnyWidget())
    return app


LOGR = np.linspace(0.40, 0.50, 60)  # OPUS logr block: already -log10(R)


def test_ct_predict_keeps_reader_type_for_log_reflectance(tmp_path, monkeypatch, dialogs):
    """By value, 0.4-0.5 looks like reflectance; the reader knows it is -log10 R."""
    _install_opus(monkeypatch, {tmp_path / f"s{i}.0": {"logr": LOGR} for i in range(2)})
    app = _ct_app(tmp_path)

    app._load_new_satellite_data_predict()

    assert not [c for c in dialogs if c[0] == "showerror"], dialogs
    assert app.ct_pred_data_type.get() == "absorbance"
    assert app.ct_pred_source_data_type == "log_reflectance"
    assert app.ct_pred_convert_btn.options["text"] == "Convert to Reflectance"
    # The stored spectra are the file's values, not logged again
    _, X = app.new_satellite_data_predict
    np.testing.assert_array_equal(X[0], LOGR[::-1])


def test_ct_predict_shows_mixed_opus_warning(tmp_path, monkeypatch, dialogs):
    _install_opus(
        monkeypatch,
        {
            tmp_path / "s0.0": {"a": np.full(60, 0.5)},
            tmp_path / "s1.0": {"sm": np.full(60, 111.0)},
        },
    )
    app = _ct_app(tmp_path)

    app._load_new_satellite_data_predict()

    shown = [c for c in dialogs if c[1] == "Satellite spectra import warnings"]
    assert len(shown) == 1
    assert "different data types" in shown[0][2]
    assert "single-channel" in shown[0][2]


def test_ct_predict_other_type_offers_no_conversion(tmp_path, monkeypatch, dialogs):
    _install_opus(monkeypatch, {tmp_path / f"s{i}.0": {"km": np.full(60, 0.3)} for i in range(2)})
    app = _ct_app(tmp_path)

    app._load_new_satellite_data_predict()

    assert app.ct_pred_data_type.get() == "other"
    assert app.ct_pred_convert_btn.options["state"] == "disabled"
    assert "Kubelka-Munk" in app.ct_pred_satellite_info_text.text
    _, X_before = app.new_satellite_data_predict
    app._ct_pred_convert_data_type()  # refused, data unchanged
    _, X_after = app.new_satellite_data_predict
    np.testing.assert_array_equal(X_after, X_before)
    assert any(c[1] == "No Conversion" for c in dialogs)


def test_ct_export_keeps_reader_type(tmp_path, monkeypatch, dialogs):
    _install_opus(monkeypatch, {tmp_path / f"s{i}.0": {"logr": LOGR} for i in range(2)})
    app = _ct_app(tmp_path)
    app.current_transfer_model = object()  # Mode B refuses to load without one
    app.ct_export_satellite_path_var = _Var(str(tmp_path))
    app.ct_export_data_type = _Var()
    app.export_metadata_context = None
    app._detect_data_format = lambda p: "folder"
    app._update_ct_use_as_working_btn_state = lambda: None
    app._plot_export_spectra_preview = lambda *a: None
    for name in (
        "ct_export_data_info_text",
        "ct_export_detected_type_label",
        "ct_export_refl_radio",
        "ct_export_abs_radio",
        "ct_convert_to_abs_btn",
        "ct_convert_to_refl_btn",
        "ct_conversion_status_label",
    ):
        setattr(app, name, _AnyWidget())

    app._load_new_satellite_data_export()

    assert not [c for c in dialogs if c[0] == "showerror"], dialogs
    assert app.ct_export_data_type.get() == "absorbance"
    assert app.ct_export_source_data_type == "log_reflectance"
    # Only "to reflectance" is offered; no log10(1/x) of -log10 R values
    assert app.ct_convert_to_abs_btn.options["state"] == "disabled"
    assert app.ct_convert_to_refl_btn.options["state"] == "normal"


def _pred_app(path):
    app = _bare_app()
    app.colors = {"success": "green", "warning": "orange", "text_light": "gray", "text": "black"}
    app.source_data_type = None
    app.data_value_scale = 1.0
    app.loaded_models = []
    app.pred_data_source = _Var("directory")
    app.pred_data_path = _Var(str(path))
    app.root = types.SimpleNamespace(update=lambda: None)
    for name in (
        "pred_status",
        "pred_data_status",
        "pred_type_status_label",
        "pred_model_expects_label",
        "pred_convert_btn",
        "pred_type_match_label",
    ):
        setattr(app, name, _AnyWidget())
    return app


def test_prediction_load_uses_reader_metadata_and_warnings(tmp_path, monkeypatch, dialogs):
    (tmp_path / "s0.dpt").write_text("1000,0.3\n1001,0.31\n")
    df = pd.DataFrame([[0.3, 0.31]], index=["s0"], columns=[1000.0, 1001.0])
    metadata = {
        "data_type": "other",
        "type_confidence": 95.0,
        "source_data_type": "kubelka_munk",
        "import_warnings": ["s9.dpt: skipped 3 line(s)"],
    }
    monkeypatch.setattr(sp_io, "read_ascii_spectra", lambda p, **k: (df, metadata))
    app = _pred_app(tmp_path)

    app._load_prediction_data()

    assert not [c for c in dialogs if c[0] == "showerror"], dialogs
    assert app.prediction_data_type == "other"
    assert app.pred_source_data_type == "kubelka_munk"
    assert "KUBELKA-MUNK" in app.pred_type_status_label.options["text"]
    assert app.pred_convert_btn.options["state"] == "disabled"
    assert any(c[1] == "Prediction data import warnings" for c in dialogs)


def test_prediction_load_real_ascii_folder_shows_parse_warning(tmp_path, dialogs):
    for i in range(2):
        rows = "\n".join(f"{1000 + j},{0.3 + j / 1000}" for j in range(50))
        (tmp_path / f"s{i}.dpt").write_text(rows + "\nERROR\n")
    app = _pred_app(tmp_path)

    app._load_prediction_data()

    assert app.prediction_data.shape == (2, 50)
    shown = [c for c in dialogs if c[1] == "Prediction data import warnings"]
    assert shown and "skipped non-numeric lines" in shown[0][2]
    assert "s0.dpt" in shown[0][2] and "s1.dpt" in shown[0][2]


def test_main_tab_other_type_conversion_refused(dialogs):
    app = _status_app({"data_type": "other", "type_confidence": 95.0, "source_data_type": "raman"})
    app.X_original = app.X

    assert app.convert_data_button.options["state"] == "disabled"
    assert "Raman intensity" in app.data_type_status_label.options["text"]
    assert app._get_spectral_ylabel() == "Raman intensity"
    app._convert_and_replot()
    assert any(c[1] == "No Conversion" for c in dialogs)


def test_data_type_helpers():
    assert gui._is_convertible_data_type("absorbance")
    assert not gui._is_convertible_data_type("other")
    assert gui._data_type_label("other", "kubelka_munk") == "Kubelka-Munk"
    assert gui._data_type_label("absorbance", "log_reflectance") == "Absorbance"


def test_model_file_keeps_source_data_type(tmp_path):
    """The GUI now saves source_data_type and conversion history with the model."""
    from sklearn.linear_model import Ridge

    from spectral_predict.model_io import load_model, save_model

    X = np.random.default_rng(0).normal(size=(10, 5))
    model = Ridge().fit(X, X[:, 0])
    metadata = {
        "model_name": "Ridge",
        "task_type": "regression",
        "wavelengths": [1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
        "n_vars": 5,
        "data_type": "other",
        "source_data_type": "kubelka_munk",
        "data_type_converted_from": None,
    }
    path = tmp_path / "m.dasp"
    save_model(model, None, metadata, path)

    loaded = load_model(path)["metadata"]

    assert loaded["data_type"] == "other"
    assert loaded["source_data_type"] == "kubelka_munk"
    assert loaded["data_type_converted_from"] is None
    source = Path(gui.__file__).read_text(encoding="utf-8")
    assert "'source_data_type': getattr(self, 'source_data_type', None)," in source


def test_resolve_loaded_data_type_prefers_reader():
    X = np.full((1, 5), 0.45)  # looks like reflectance by value
    meta = {
        "data_type": "absorbance",
        "type_confidence": 95.0,
        "source_data_type": "log_reflectance",
    }

    assert gui._resolve_loaded_data_type(meta, X) == ("absorbance", 95.0, "log_reflectance")
    data_type, _, source = gui._resolve_loaded_data_type(None, X)
    assert source is None and data_type in ("absorbance", "reflectance")


# ---------------------------------------------------------------------------
# Review round 3: contamination, comparison, CT scale, compatibility, ensembles
# ---------------------------------------------------------------------------


def _contam_app(monkeypatch, path):
    app = _bare_app()
    app.colors = {"success": "green", "warning": "orange", "text": "black"}
    app.source_data_type = None
    app.data_value_scale = 1.0
    app.contam_clean_path = _Var()
    app.contam_original_data_type = _Var()
    app.contam_current_data_type = _Var()
    app.contam_groups = {}
    app.contam_data_converted = False
    app.contam_type_confidence = 0.0
    for name in ("contam_clean_info_label", "contam_dtype_status_label", "contam_convert_btn"):
        setattr(app, name, _AnyWidget())
    app._contam_update_summary = lambda: None
    app._contam_auto_populate_spectra_plot = lambda: None
    app._contam_plot_group_spectra = lambda **k: None
    monkeypatch.setattr(gui.messagebox, "askquestion", lambda *a, **k: "yes")
    monkeypatch.setattr(gui.filedialog, "askdirectory", lambda *a, **k: str(path))
    return app


def test_contamination_keeps_reader_type_for_log_reflectance(tmp_path, monkeypatch, dialogs):
    _install_opus(monkeypatch, {tmp_path / f"s{i}.0": {"logr": LOGR} for i in range(2)})
    app = _contam_app(monkeypatch, tmp_path)

    app._contam_load_clean_data()

    assert not [c for c in dialogs if c[0] == "showerror"], dialogs
    assert app.contam_current_data_type.get() == "absorbance"
    assert app.contam_source_data_type == "log_reflectance"
    # The offer is absorbance -> reflectance, never another log10(1/x)
    assert app.contam_convert_btn.options["text"] == "Convert to Reflectance"
    np.testing.assert_array_equal(app.contam_clean_data[0], LOGR[::-1])


def test_contamination_other_type_refuses_conversion(tmp_path, monkeypatch, dialogs):
    _install_opus(monkeypatch, {tmp_path / f"s{i}.0": {"km": np.full(60, 0.3)} for i in range(2)})
    app = _contam_app(monkeypatch, tmp_path)

    app._contam_load_clean_data()

    assert app.contam_current_data_type.get() == "other"
    assert app.contam_convert_btn.options["state"] == "disabled"
    before = app.contam_clean_data.copy()
    app._contam_convert_data_type()
    np.testing.assert_array_equal(app.contam_clean_data, before)
    assert any(c[1] == "No Conversion" for c in dialogs)


def _with_dataset_state(app, X=None, y=None, holdout=()):
    """The loaded-dataset state the validation and dataset-install paths read.

    The Comparison tab rebuilds the validation set from the loaded data, and
    calibration-transfer replace installs through ``_install_dataset``, which
    clears the previous dataset's exclusions and validation split.
    """
    app.X = X
    app.X_original = X
    app.y = y
    app.ref = None
    app.combined_metadata_df = None
    app.excluded_spectra = set()
    app.validation_indices = set(holdout)
    app.validation_X = app.validation_y = None
    app.validation_enabled = _Var(bool(holdout))
    app.wavelength_min = _Var("")
    app.wavelength_max = _Var("")
    app._pending_validation_indices = None
    app.active_indices = None
    app.outlier_report = None
    app.data_sources = []
    app.source_group_names = [""]
    app.use_custom_group_names = False
    return app


def _comparison_app(source, path=""):
    app = _bare_app()
    app.colors = {"success": "green", "warning": "orange", "text_light": "gray", "accent": "blue"}
    app.source_data_type = None
    app.data_value_scale = 1.0
    app.data_has_been_converted = False
    app.comparison_data_source = _Var(source)
    app.comparison_data_path = _Var(str(path))
    app.comparison_data_type = _Var()
    app.comparison_primary_model = None
    app.comparison_auxiliary_models = []
    for name in (
        "comparison_conversion_status",
        "comparison_data_type_frame",
        "comparison_data_status",
        "comparison_status",
        "comparison_refl_radio",
        "comparison_abs_radio",
        "comparison_convert_btn",
        "comparison_type_status",
        "comparison_model_expects_label",
        "comparison_type_match_label",
    ):
        setattr(app, name, _AnyWidget())
    return app


def test_comparison_validation_set_keeps_main_tab_type(dialogs):
    """A validation matrix of -log10(R) values must stay absorbance in the comparison tab."""
    app = _comparison_app("validation")
    X = pd.DataFrame([LOGR], index=["v0"], columns=np.linspace(3410.0, 4000.0, 60))
    _with_dataset_state(app, X, pd.Series([1.0], index=X.index), holdout=["v0"])
    app.current_data_type.set("absorbance")
    app.type_confidence = 95.0
    app.source_data_type = "log_reflectance"

    app._load_comparison_data()

    assert not [c for c in dialogs if c[0] == "showerror"], dialogs
    assert app.comparison_data_type.get() == "absorbance"
    assert app.comparison_source_data_type == "log_reflectance"
    assert app.comparison_convert_btn.options["text"] == "Convert to Reflectance"


def test_comparison_directory_uses_reader_metadata(tmp_path, monkeypatch, dialogs):
    (tmp_path / "s0.dpt").write_text("1000,0.3\n1001,0.31\n")
    df = pd.DataFrame([[0.3, 0.31]], index=["s0"], columns=[1000.0, 1001.0])
    meta = {"data_type": "other", "type_confidence": 95.0, "source_data_type": "kubelka_munk"}
    monkeypatch.setattr(sp_io, "read_ascii_spectra", lambda p, **k: (df, meta))
    app = _comparison_app("directory", tmp_path)

    app._load_comparison_data()

    assert app.comparison_data_type.get() == "other"
    assert app.comparison_convert_btn.options["state"] == "disabled"
    before = app.comparison_data.copy()
    app._comparison_convert_data_type()
    pd.testing.assert_frame_equal(app.comparison_data, before)
    assert any(c[1] == "No Conversion" for c in dialogs)


def _export_app(X, data_type="reflectance", scale=100.0):
    app = _bare_app()
    app.colors = {"success": "green", "warning": "orange", "accent": "blue"}
    app.source_data_type = "transmittance"  # main tab state must not leak in
    app.data_value_scale = 1.0
    app.new_satellite_data_export = (np.linspace(1000.0, 1100.0, X.shape[1]), X)
    app.ct_export_data_type = _Var(data_type)
    app.ct_export_value_scale = scale
    app.ct_export_data_converted = False
    app.ct_export_source_data_type = None
    app._plot_export_spectra_preview = lambda *a: None
    for name in (
        "ct_export_detected_type_label",
        "ct_convert_to_abs_btn",
        "ct_convert_to_refl_btn",
        "ct_conversion_status_label",
    ):
        setattr(app, name, _AnyWidget())
    return app


def test_ct_export_percent_reflectance_round_trip():
    X = np.full((2, 5), 50.0)  # percent reflectance
    app = _export_app(X)

    app._ct_convert_to_absorbance()
    _, A = app.new_satellite_data_export
    np.testing.assert_allclose(A, np.log10(1 / 0.5))
    app._ct_convert_to_reflectance()
    _, R = app.new_satellite_data_export

    np.testing.assert_allclose(R, 50.0)
    assert app.data_value_scale == 1.0 and app.source_data_type == "transmittance"


def test_ct_export_load_keeps_percent_scale(tmp_path, monkeypatch, dialogs):
    for i in range(2):
        rows = "\n".join(f"{1000 + j},{40 + j / 10}" for j in range(60))
        (tmp_path / f"s{i}.dpt").write_text(rows + "\n")
    app = _ct_app(tmp_path)
    app.current_transfer_model = object()
    app.ct_export_satellite_path_var = _Var(str(tmp_path))
    app.ct_export_data_type = _Var()
    app.export_metadata_context = None
    app._detect_data_format = lambda p: "folder"
    app._update_ct_use_as_working_btn_state = lambda: None
    app._plot_export_spectra_preview = lambda *a: None
    for name in (
        "ct_export_data_info_text",
        "ct_export_detected_type_label",
        "ct_export_refl_radio",
        "ct_export_abs_radio",
        "ct_convert_to_abs_btn",
        "ct_convert_to_refl_btn",
        "ct_conversion_status_label",
    ):
        setattr(app, name, _AnyWidget())

    app._load_new_satellite_data_export()
    assert app.ct_export_data_type.get() == "reflectance"
    assert app.ct_export_value_scale == 100.0
    _, X0 = app.new_satellite_data_export
    app._ct_convert_to_absorbance()
    app._ct_convert_to_reflectance()
    _, X1 = app.new_satellite_data_export

    np.testing.assert_allclose(X1, X0)


def test_ct_working_data_handoff_keeps_converted_type():
    """After Mode B data was converted to absorbance, the handoff must not re-detect."""
    app = _bare_app()
    app.ct_export_data_type = _Var("absorbance")
    app.ct_export_type_confidence = 90.0
    app.ct_export_data_converted = True
    app.ct_export_source_data_type = "transmittance"
    # Simulate the transform step's bookkeeping, then the handoff's type resolution
    carried = {
        "data_type": app.ct_export_data_type.get(),
        "type_confidence": app.ct_export_type_confidence,
        "source_data_type": None,
        "value_scale": 1.0,
    }
    X = np.full((3, 5), 0.45)  # would look like reflectance by value
    assert gui._resolve_loaded_data_type(carried, X)[0] == "absorbance"
    source = Path(gui.__file__).read_text(encoding="utf-8")
    assert "carried = getattr(self, 'transformed_spectra_type', None)" in source


def test_ct_prediction_checks_data_type(tmp_path, monkeypatch, dialogs):
    from spectral_predict import model_io

    monkeypatch.setattr(model_io, "predict_with_model", lambda md, X, **k: np.zeros(len(X)))
    app = _bare_app()
    app.current_transfer_model = object()
    app.ct_transfer_model = None
    app.current_prediction_model = object()
    app.current_prediction_model_dict = {
        "metadata": {"data_type": "other", "source_data_type": "raman", "wavelengths": None}
    }
    app.new_satellite_data_predict = (np.linspace(1000.0, 1100.0, 5), np.full((2, 5), 0.3))
    app.ct_pred_data_type = _Var("other")
    app.ct_pred_data_converted = False
    app.ct_pred_source_data_type = "kubelka_munk"
    app.ct_pred_loaded_sample_ids = None
    app._apply_transfer_with_roi = lambda X, model: X
    app._plot_ct_prediction_results = lambda *a: None
    app.play_sound = lambda *a, **k: None
    app.ct_predictions_tree = _AnyWidget()
    app.ct_export_predictions_button = _AnyWidget()

    app._run_prediction_workflow()

    mismatch = [c for c in dialogs if c[1] == "Data Type Mismatch"]
    assert mismatch and "RAMAN" in mismatch[0][2] and "KUBELKA MUNK" in mismatch[0][2]
    assert not [c for c in dialogs if c[0] == "showerror"], dialogs


def test_check_data_type_compatibility_rules():
    from spectral_predict.model_io import check_data_type_compatibility as check

    raman = {"data_type": "other", "source_data_type": "raman"}
    assert "KUBELKA MUNK" in check(raman, "other", "kubelka_munk")
    assert check(raman, "other", "raman") is None
    assert check(raman, "other", None) is None  # unknown prediction source
    assert check({"data_type": "other"}, "other", "kubelka_munk") is None  # legacy model
    assert "ABSORBANCE" in check(raman, "absorbance", None)
    converted = {
        "data_type": "absorbance",
        "source_data_type": "transmittance",
        "data_type_converted_from": "reflectance",
    }
    assert check(converted, "absorbance", "absorbance") is None
    assert check({}, "absorbance", "absorbance") is None


def test_predict_with_uncertainty_warns_on_other_subtype(tmp_path):
    from sklearn.linear_model import Ridge

    from spectral_predict.model_io import load_model, predict_with_uncertainty, save_model

    X = np.random.default_rng(1).normal(size=(12, 5))
    wl = [1000.0, 1001.0, 1002.0, 1003.0, 1004.0]
    meta = {
        "model_name": "Ridge",
        "task_type": "regression",
        "wavelengths": wl,
        "n_vars": 5,
        "data_type": "other",
        "source_data_type": "raman",
    }
    save_model(Ridge().fit(X, X[:, 0]), None, meta, tmp_path / "m.dasp")
    model_dict = load_model(tmp_path / "m.dasp")
    X_new = pd.DataFrame(X[:3], columns=wl)

    result = predict_with_uncertainty(
        model_dict, X_new, prediction_data_type="other", prediction_source_data_type="kubelka_munk"
    )

    assert "RAMAN" in result["data_type_warning"]


def test_ensemble_save_carries_ordinate_metadata(tmp_path):
    import json
    import zipfile

    from sklearn.linear_model import Ridge

    from spectral_predict.model_io import load_model, save_ensemble

    X = np.random.default_rng(2).normal(size=(12, 5))
    ensemble = types.SimpleNamespace(
        models=[Ridge().fit(X, X[:, 0]), Ridge(alpha=2).fit(X, X[:, 0])],
        model_names=["r1", "r2"],
    )
    meta = {
        "task_type": "regression",
        "wavelengths": [1.0, 2.0, 3.0, 4.0, 5.0],
        "n_vars": 5,
        "data_type": "other",
        "source_data_type": "kubelka_munk",
        "data_type_converted_from": None,
    }
    path = tmp_path / "e.dasp"
    save_ensemble(ensemble, str(path), meta)

    with zipfile.ZipFile(path) as zf:
        config = json.loads(zf.read("ensemble_config.json"))
        zf.extract("base_model_0.dasp", tmp_path)
    assert config["metadata"]["source_data_type"] == "kubelka_munk"
    base = load_model(tmp_path / "base_model_0.dasp")["metadata"]
    assert base["data_type"] == "other"
    assert base["source_data_type"] == "kubelka_munk"
    assert "data_type_converted_from" in base
    source = Path(gui.__file__).read_text(encoding="utf-8")
    assert "# Ordinate type of the training data (save_ensemble copies these to" in source


@pytest.mark.parametrize(
    "data_type, source, suffix",
    [
        ("absorbance", "log_reflectance", "_abs"),
        ("reflectance", "transmittance", "_ref"),
        ("other", "kubelka_munk", "_km"),
        ("other", "raman", "_raman"),
        ("other", None, "_other"),
    ],
)
def test_data_type_suffix(data_type, source, suffix):
    assert gui._data_type_suffix(data_type, source) == suffix


# ---------------------------------------------------------------------------
# Review round 4: contaminant groups, carried scale, canonical source labels
# ---------------------------------------------------------------------------


def test_contaminant_group_with_other_type_is_refused(tmp_path, monkeypatch, dialogs):
    clean_dir = tmp_path / "clean"
    km_dir = tmp_path / "km"
    clean_dir.mkdir()
    km_dir.mkdir()
    R = np.linspace(0.40, 0.50, 60)
    files = {clean_dir / f"c{i}.0": {"r": R} for i in range(2)}
    files.update({km_dir / f"k{i}.0": {"km": np.full(60, 0.45)} for i in range(2)})
    _install_opus(monkeypatch, files)
    app = _contam_app(monkeypatch, clean_dir)
    app.contam_group_paths = {}
    app.contam_groups_listbox = _AnyWidget()
    app.contam_wavelengths = None
    app._contam_load_clean_data()
    assert app.contam_current_data_type.get() == "reflectance"

    added = app._contam_add_single_group("KM", str(km_dir))

    assert added is False
    assert "KM" not in app.contam_groups
    refusal = [c for c in dialogs if c[1] == "Data Type Mismatch"]
    assert refusal and "Kubelka-Munk" in refusal[0][2]


def test_contaminant_group_with_log_reflectance_is_refused(tmp_path, monkeypatch, dialogs):
    clean_dir = tmp_path / "clean"
    logr_dir = tmp_path / "logr"
    clean_dir.mkdir()
    logr_dir.mkdir()
    files = {clean_dir / f"c{i}.0": {"r": np.linspace(0.4, 0.5, 60)} for i in range(2)}
    files.update({logr_dir / f"g{i}.0": {"logr": LOGR} for i in range(2)})
    _install_opus(monkeypatch, files)
    app = _contam_app(monkeypatch, clean_dir)
    app.contam_group_paths = {}
    app.contam_groups_listbox = _AnyWidget()
    app.contam_wavelengths = None
    app._contam_load_clean_data()

    assert app._contam_add_single_group("logR", str(logr_dir)) is False
    assert "logR" not in app.contam_groups


def test_matching_contaminant_groups_convert_with_clean_data(tmp_path, monkeypatch, dialogs):
    clean_dir = tmp_path / "clean"
    grp_dir = tmp_path / "grp"
    clean_dir.mkdir()
    grp_dir.mkdir()
    files = {clean_dir / f"c{i}.0": {"r": np.full(60, 0.5)} for i in range(2)}
    files.update({grp_dir / f"g{i}.0": {"r": np.full(60, 0.25)} for i in range(2)})
    _install_opus(monkeypatch, files)
    app = _contam_app(monkeypatch, clean_dir)
    app.contam_group_paths = {}
    app.contam_groups_listbox = _AnyWidget()
    app.contam_wavelengths = None
    app._contam_load_clean_data()
    assert app._contam_add_single_group("G", str(grp_dir)) is True

    app._contam_convert_data_type()

    np.testing.assert_allclose(app.contam_clean_data, np.log10(1 / 0.5))
    np.testing.assert_allclose(app.contam_groups["G"], np.log10(1 / 0.25))
    assert app.contam_group_types["G"]["data_type"] == "absorbance"


def test_contaminant_conversion_refuses_mismatched_groups(tmp_path, monkeypatch, dialogs):
    """A group whose type differs (e.g. loaded before the clean data) blocks conversion."""
    app = _contam_app(monkeypatch, tmp_path)
    app.contam_clean_data = np.full((2, 5), 0.5)
    app.contam_current_data_type.set("reflectance")
    app.contam_original_data_type.set("reflectance")
    app.contam_data_value_scale = 1.0
    app.contam_groups = {"KM": np.full((2, 5), 0.45)}
    app.contam_group_types = {
        "KM": {"data_type": "other", "source_data_type": "kubelka_munk", "value_scale": 1.0}
    }

    app._contam_convert_data_type()

    np.testing.assert_array_equal(app.contam_groups["KM"], np.full((2, 5), 0.45))
    np.testing.assert_array_equal(app.contam_clean_data, np.full((2, 5), 0.5))
    assert any(c[1] == "Data Type Mismatch" for c in dialogs)


def test_comparison_keeps_carried_percent_scale_for_absorbance(dialogs):
    """50 % reflectance converted to absorbance on the main tab converts back to 50."""
    app = _comparison_app("validation")
    A = np.full((1, 5), np.log10(1 / 0.5))
    X = pd.DataFrame(A, index=["v0"], columns=[1000.0, 1001.0, 1002.0, 1003.0, 1004.0])
    _with_dataset_state(app, X, pd.Series([1.0], index=X.index), holdout=["v0"])
    app.current_data_type.set("absorbance")
    app.original_data_type.set("reflectance")
    app.data_has_been_converted = True
    app.data_value_scale = 100.0
    app.type_confidence = 95.0

    app._load_comparison_data()
    assert app.comparison_value_scale == 100.0
    app._comparison_convert_data_type()

    np.testing.assert_allclose(app.comparison_data.to_numpy(), 50.0)


def test_ct_handoff_keeps_type_and_percent_scale(monkeypatch, dialogs):
    X = np.full((2, 5), 50.0)  # percent reflectance in Mode B
    app = _export_app(X)
    app._ct_convert_to_absorbance()
    wl, X_abs = app.new_satellite_data_export
    app.transformed_spectra = (wl, X_abs)
    app._record_transformed_spectra_type()
    app.export_metadata_context = {
        "source_format": "smart_csv",
        "specimen_ids": ["a", "b"],
        "metadata_df": None,
    }
    _with_dataset_state(app)
    app.tab1_status = _AnyWidget()
    app.notebook = types.SimpleNamespace(select=lambda i: None)
    for name in (
        "_refresh_active_group_indices",
        "_generate_plots",
        "_generate_explore_plots",
        "_update_data_type_status_ui",
        "_update_x_unit_status_ui",
        "_update_x_unit_labels",
        "_update_dm_data_type_label",
        "_populate_data_viewer",
        "_update_task_type_label",
    ):
        setattr(app, name, lambda *a, **k: None)

    app._ct_use_as_working_data()

    assert not [c for c in dialogs if c[0] == "showerror"], dialogs
    # Absorbance values 0.30103 would read as reflectance by value; the carried type wins
    assert app.current_data_type.get() == "absorbance"
    assert app.data_value_scale == 100.0
    back = app._convert_data_type(app.X.to_numpy(), "absorbance", "reflectance")
    np.testing.assert_allclose(back, 50.0)


def test_ct_with_y_load_uses_reader_metadata(monkeypatch, dialogs):
    X = pd.DataFrame([LOGR, LOGR], index=["s1", "s2"], columns=np.linspace(3410, 4000, 60))
    meta = {
        "data_type": "absorbance",
        "type_confidence": 95.0,
        "source_data_type": "Log(1/R)",
        "value_scale": 1.0,
    }
    monkeypatch.setattr(sp_io, "read_omnic_dir", lambda p: (X, meta), raising=False)
    monkeypatch.setattr(sp_io, "read_reference_csv", lambda p, c: pd.DataFrame({"y": [1, 2]}))
    monkeypatch.setattr(
        sp_io,
        "align_xy",
        lambda X_, ref, c, t, return_alignment_info=True: (
            X_,
            pd.Series([1.0, 2.0], index=X_.index),
            {
                "unmatched_spectra": [],
                "n_nan_dropped": 0,
                "matched_ids": list(X_.index),
                "unmatched_reference": [],
                "used_fuzzy_matching": False,
            },
        ),
    )
    app = _bare_app()
    app.ct_primary_spectra_path_var = _Var("folder")
    app.ct_primary_reference_path_var = _Var("ref.csv")
    app.ct_primary_spectral_file_col_var = _Var("file")
    app.ct_primary_target_col_var = _Var("y")
    app.ct_primary_detected_type = "omnic"
    app.ct_primary_data_type = _Var()
    app._update_data_info = lambda: None
    app.play_sound = lambda *a, **k: None

    app._load_primary_data_with_y()

    assert not [c for c in dialogs if c[0] == "showerror"], dialogs
    assert app.ct_primary_data_type.get() == "absorbance"
    assert app.ct_primary_source_data_type == "Log(1/R)"
    assert gui._data_type_suffix("other", "Kubelka-Munk") == "_km"


def test_canonical_source_labels():
    from spectral_predict.io import canonical_source_data_type as canon
    from spectral_predict.model_io import check_data_type_compatibility as check

    assert canon("Log(1/R)") == canon("log_reflectance") == "log_reflectance"
    assert canon("Kubelka-Munk") == canon("kubelka_munk") == "kubelka_munk"
    assert canon("unknown") is None and canon(None) is None and canon("") is None
    assert canon("Some Thing") == "some_thing"
    model = {"data_type": "absorbance", "source_data_type": "log_reflectance"}
    assert check(model, "absorbance", "Log(1/R)") is None
    assert check(model, "absorbance", "absorbance") is not None
    assert gui._data_type_label("other", "Kubelka-Munk") == "Kubelka-Munk"


def test_metadata_captures_and_prediction_source_stay_wired():
    """Tripwires for plumbing the behavioural tests cannot reach cheaply."""
    source = Path(gui.__file__).read_text(encoding="utf-8")
    # Directory loaders leave reader metadata for their callers
    assert source.count("self._last_dir_load_metadata = metadata") >= 3
    assert source.count("df, self._last_dir_load_metadata = ") >= 4
    assert source.count("self._contam_last_metadata = metadata") == 6
    # Main-tab prediction passes the prediction data's source type
    assert (
        "prediction_source_data_type=(\n                            None if self.pred_data_has_been_converted"
        in source
    )


# ---------------------------------------------------------------------------
# Review round 5: contaminant compatibility, empty groups, transactional conversion
# ---------------------------------------------------------------------------


class _Listbox:
    def __init__(self):
        self.items = []

    def insert(self, index, text):
        self.items.append(text)

    def delete(self, first, last=None):
        if last is not None:
            self.items = []
        else:
            del self.items[first]

    def size(self):
        return len(self.items)

    def get(self, idx):
        return self.items[idx]


def _contam_folder_app(monkeypatch, tmp_path, folders):
    """Contamination app with OPUS folders: {name: {block: values}} (2 files each)."""
    files = {}
    for name, blocks in folders.items():
        d = tmp_path / name
        d.mkdir()
        files.update({d / f"{name}{i}.0": blocks for i in range(2)})
    _install_opus(monkeypatch, files)
    app = _contam_app(monkeypatch, tmp_path / next(iter(folders)))
    app.contam_group_paths = {}
    app.contam_groups_listbox = _Listbox()
    app.contam_wavelengths = None
    app.contam_clean_data = None
    app.contam_group_types = {}
    return app


def test_other_types_with_different_sources_are_refused(tmp_path, monkeypatch, dialogs):
    app = _contam_folder_app(
        monkeypatch, tmp_path, {"km": {"km": np.full(60, 0.3)}, "ra": {"ra": np.full(60, 500.0)}}
    )
    app._contam_load_clean_data()
    assert app.contam_current_data_type.get() == "other"

    assert app._contam_add_single_group("Raman", str(tmp_path / "ra")) is False
    refusal = [c for c in dialogs if c[1] == "Data Type Mismatch"]
    assert refusal and "Raman intensity" in refusal[0][2] and "Kubelka-Munk" in refusal[0][2]


def test_other_types_with_the_same_source_are_accepted(tmp_path, monkeypatch, dialogs):
    app = _contam_folder_app(
        monkeypatch, tmp_path, {"km": {"km": np.full(60, 0.3)}, "km2": {"km": np.full(60, 0.4)}}
    )
    app._contam_load_clean_data()

    assert app._contam_add_single_group("KM2", str(tmp_path / "km2")) is True


def test_group_added_before_clean_data_is_revalidated(tmp_path, monkeypatch, dialogs):
    app = _contam_folder_app(
        monkeypatch, tmp_path, {"clean": {"r": np.full(60, 0.5)}, "km": {"km": np.full(60, 0.45)}}
    )
    assert app._contam_add_single_group("KM", str(tmp_path / "km")) is True  # no clean yet
    asked = []
    monkeypatch.setattr(
        gui.messagebox, "askyesno", lambda title, msg, **k: asked.append(msg) or True
    )

    app._contam_load_clean_data()

    assert asked and "KM" in asked[0] and "Kubelka-Munk" in asked[0]
    assert "KM" not in app.contam_groups and app.contam_groups_listbox.items == []


def test_analysis_runs_are_blocked_while_a_group_mismatches(tmp_path, monkeypatch, dialogs):
    app = _contam_folder_app(
        monkeypatch, tmp_path, {"clean": {"r": np.full(60, 0.5)}, "km": {"km": np.full(60, 0.45)}}
    )
    app._contam_add_single_group("KM", str(tmp_path / "km"))
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *a, **k: False)  # keep the group
    app._contam_load_clean_data()
    assert "KM" in app.contam_groups
    preprocessed = []
    app._contam_preprocess_data = lambda X: preprocessed.append(X) or X
    app.contam_preprocessing = _Var("None (Raw)")

    app._contam_run_difference_analysis()
    app._contam_run_automated_detection()

    blocked = [c for c in dialogs if c[0] == "showerror" and c[1] == "Data Type Mismatch"]
    assert len(blocked) == 2 and "KM" in blocked[0][2]
    assert preprocessed == []  # neither analysis started


def test_empty_group_is_rejected(tmp_path, monkeypatch, dialogs):
    path = tmp_path / "empty.npy"
    np.save(path, np.empty((0, 3)))
    app = _contam_app(monkeypatch, tmp_path)
    app.contam_group_paths = {}
    app.contam_groups_listbox = _Listbox()
    app.contam_wavelengths = None
    app.contam_clean_data = np.full((2, 3), 0.5)
    app.contam_current_data_type.set("reflectance")

    assert app._contam_add_single_group("E", str(path)) is False
    assert "E" not in app.contam_groups
    assert any("no spectra" in c[2] for c in dialogs)


def test_contaminant_conversion_is_transactional(monkeypatch, tmp_path, dialogs):
    app = _contam_app(monkeypatch, tmp_path)
    app.contam_clean_data = np.full((2, 3), 0.5)
    app.contam_current_data_type.set("reflectance")
    app.contam_original_data_type.set("reflectance")
    app.contam_data_value_scale = 1.0
    app.contam_groups = {"A": np.full((2, 3), 0.25), "E": np.empty((0, 3))}
    app.contam_group_types = {}

    app._contam_convert_data_type()

    # The empty group fails the conversion; nothing was committed
    np.testing.assert_array_equal(app.contam_clean_data, np.full((2, 3), 0.5))
    np.testing.assert_array_equal(app.contam_groups["A"], np.full((2, 3), 0.25))
    assert app.contam_current_data_type.get() == "reflectance"
    assert any(c[1] == "Error" and "no spectra" in c[2] for c in dialogs)


def test_combined_file_groups_get_their_own_records(monkeypatch, tmp_path, dialogs):
    app = _contam_app(monkeypatch, tmp_path)
    app.contam_group_paths = {}
    app.contam_groups_listbox = _Listbox()
    app.contam_group_types = {}
    wl = [str(1000 + i) for i in range(5)]
    rows = [["clean"] + [50.0] * 5, ["clean"] + [52.0] * 5, ["sand"] + [30.0] * 5]
    app._contam_combined_df = pd.DataFrame(rows, columns=["group"] + wl)
    app._contam_combined_wl_cols = wl
    app.contam_combined_group_col = _Var("group")
    app.contam_combined_clean_value = _Var("clean")
    app.contam_combined_file_path = _Var("combined.csv")
    app._contam_combined_status = _AnyWidget()

    app._contam_process_combined_file()

    assert app.contam_group_types["sand"]["data_type"] == "reflectance"
    assert app.contam_group_types["sand"]["value_scale"] == 100.0
    app._contam_convert_data_type()
    app._contam_convert_data_type()
    np.testing.assert_allclose(app.contam_groups["sand"], 30.0)
    np.testing.assert_allclose(app.contam_clean_data[0], 50.0)
