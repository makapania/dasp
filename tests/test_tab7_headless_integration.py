import os
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure Matplotlib writes its cache somewhere writable when the GUI module imports it.
_mpl_dir = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "mplconfig"))
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

from spectral_predict_gui_optimized import SpectralPredictApp, Tab7ModelState


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _DummyText:
    def __init__(self, text):
        self._text = text

    def get(self, start, end):
        return self._text


class _DummyLabel:
    def __init__(self, text):
        self._text = text

    def cget(self, key):
        if key == "text":
            return self._text
        raise KeyError(key)


class _DummyRoot:
    def after(self, _delay_ms, callback):
        callback()


def _make_headless_app(state_wavelengths):
    """Build a SpectralPredictApp instance without Tk widgets for testing."""
    numeric_wls = list(state_wavelengths)
    if not numeric_wls:
        raise ValueError("state_wavelengths must contain at least one wavelength.")

    app = object.__new__(SpectralPredictApp)
    app.root = _DummyRoot()
    app.tab7_model_type = _DummyVar("Lasso")
    app.tab7_task_type = _DummyVar("regression")
    app.tab7_preprocess = _DummyVar("raw")
    app.tab7_window = _DummyVar(17)
    app.tab7_folds = _DummyVar(3)
    app.tab7_max_iter = _DummyVar(500)
    app.tab7_mode_label = _DummyLabel("Mode: Loaded from Results (Rank 1)")

    sorted_spec = ", ".join(f"{wl:.1f}" for wl in sorted(numeric_wls))
    app.tab7_wl_spec = _DummyText(sorted_spec)

    rng = np.random.default_rng(123)
    columns = sorted({f"{wl:.1f}" for wl in numeric_wls})
    X = rng.normal(size=(24, len(columns)))
    coeffs = np.linspace(0.1, 0.3, len(columns))
    y = X @ coeffs + rng.normal(scale=0.01, size=X.shape[0])

    app.X_original = pd.DataFrame(X, columns=columns)
    app.X = app.X_original.copy()
    app.y = pd.Series(y, name="target")

    app.excluded_spectra = set()
    app.validation_enabled = _DummyVar(False)
    app.validation_indices = set()
    app.tab7_hyperparam_widgets = {}
    app.tab7_model_state = Tab7ModelState(
        raw_config={"Model": "Lasso", "Rank": 1},
        model_wavelengths=list(numeric_wls),
        preprocess="raw",
        n_folds=3,
    )
    app.tab7_expected_r2 = None
    app.tab7_config = None
    app.tab7_performance = None
    app.tab7_preprocessing_pipeline = None
    app.tab7_trained_model = None
    app.tab7_full_wavelengths = None
    app.tab7_wavelengths = None
    app.tab7_y_true = None
    app.tab7_y_pred = None
    app._tab7_updates = []

    def _headless_update(self, text, is_error=False):
        self._tab7_updates.append({"text": text, "is_error": is_error})

    def _fake_parser(self, text, _available, preserve_order=False):
        tokens = [float(tok.strip()) for tok in text.split(",") if tok.strip()]
        if preserve_order:
            return tokens
        return sorted(tokens)

    app._tab7_update_results = types.MethodType(_headless_update, app)
    app._parse_wavelength_spec = types.MethodType(_fake_parser, app)
    return app


def test_tab7_headless_preserves_wavelength_order_from_results_state():
    """Ensure the real Tab 7 runner honors preserved wavelengths when state exists."""
    state_wls = [1300.0, 1100.0, 1250.0, 1175.0]
    app = _make_headless_app(state_wls)

    app._tab7_run_model_thread()

    assert app.tab7_wavelengths == state_wls
    assert app._tab7_updates, "Expected Tab 7 results to be recorded"
    assert app._tab7_updates[-1]["is_error"] is False


def test_tab7_headless_reports_error_when_state_missing_in_loaded_mode():
    """If the GUI claims 'Loaded from Results' but state vanished, Tab 7 should fail loudly."""
    app = _make_headless_app([1300.0, 1100.0, 1250.0])
    app.tab7_model_state = None  # Simulate state loss between Results and Tab 7

    app._tab7_run_model_thread()

    assert app._tab7_updates, "Expected error message to be recorded"
    last_update = app._tab7_updates[-1]
    assert last_update["is_error"] is True
    assert "expected a loaded model configuration" in last_update["text"]
