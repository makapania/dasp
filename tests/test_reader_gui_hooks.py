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
