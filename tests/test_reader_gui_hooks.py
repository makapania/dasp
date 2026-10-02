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

    def insert(self, index, text):
        self.text += text


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
    assert shown and "skipped 1 line" in shown[0][2]


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
