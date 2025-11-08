import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_predict_gui_optimized import Tab7ModelState, resolve_tab7_wavelengths


def test_resolve_tab7_wavelengths_prefers_saved_state():
    state = Tab7ModelState(raw_config={}, model_wavelengths=[1200.0, 1100.0])
    parser_called = False

    def fake_parser(_text, _available):
        nonlocal parser_called
        parser_called = True
        return [999.0]

    result = resolve_tab7_wavelengths(
        state,
        "Mode: Loaded from Results (Rank 1)",
        "",
        np.array([1100.0, 1200.0]),
        fake_parser,
    )

    assert result == [1200.0, 1100.0]
    assert parser_called is False


def test_resolve_tab7_wavelengths_falls_back_to_parser_when_manual_mode():
    captured = {}

    def fake_parser(text, available):
        captured["text"] = text
        captured["available"] = available
        return [available[0]]

    available = np.array([1000.0, 1010.0])
    result = resolve_tab7_wavelengths(
        None,
        "Mode: Fresh Development",
        "1000",
        available,
        fake_parser,
    )

    assert result == [1000.0]
    assert captured["text"] == "1000"
    assert np.array_equal(captured["available"], available)


def test_resolve_tab7_wavelengths_errors_when_state_missing_but_required():
    with pytest.raises(RuntimeError, match="expected a loaded model configuration"):
        resolve_tab7_wavelengths(
            None,
            "Mode: Loaded from Results (Rank 2)",
            "1000",
            np.array([1000.0]),
            lambda text, available: [1000.0],
        )
